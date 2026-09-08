// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Contract tests for MIMOCoherentCompensationOp (HSDK 5 Graph harness).
// TDM: per-Doppler-bin inter-TX phase correction, waveform configurable from
// Params. DDM: standard linear-phase Doppler-band demux into an explicit TX
// axis (offsets + linear-phase codes, guard bands); non-linear
// conjugate-correlation path. Opaque leading-batch rejection and
// cross-cutting input validation.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"
#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_stage.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::DDMCodeDescriptor;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::FMCWWaveformSpec;
using holoscan::radar::TXSchedule;
using holoscan::radar::kPi;
using holoscan::radar::kTwoPi;
using holoscan::radar::MIMOCoherentCompensationOp;
using holoscan::radar::MIMOCoherentCompensationParams;
using holoscan::radar::MultiplexingScheme;
using holoscan::radar::complex_dtype;
using holoscan::radar::dldatatype_equals;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kNumTx = 2;
constexpr int64_t kNumRx = 2;
constexpr int64_t kDoppler = 8;
constexpr int64_t kRange = 4;
constexpr double kPri = 1.0e-3;

FMCWWaveformDescriptor make_test_waveform(double pri = kPri) {
  FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 1.0e6;
  spec.chirp_duration = 50.0e-6;
  spec.sample_rate = 2.0e6;
  spec.chirp_repetition_interval = pri;
  return FMCWWaveformDescriptor(spec);
}

int64_t tensor_count(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  return count;
}

template <typename T>
std::vector<std::complex<T>> build_misaligned_rd_host(
    const FMCWWaveformDescriptor& waveform, int64_t num_tx, int64_t num_rx,
    int64_t num_doppler, int64_t num_range, int64_t batch = 1,
    const std::vector<int64_t>& slot_by_tx = {}) {
  const auto table = holoscan::radar::detail::build_tdm_correction_table(
      waveform, num_tx, num_doppler, slot_by_tx);
  const int64_t slice = num_tx * num_rx * num_doppler * num_range;
  std::vector<std::complex<T>> host(static_cast<size_t>(batch * slice));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t d = 0; d < num_doppler; ++d) {
          for (int64_t r = 0; r < num_range; ++r) {
            const size_t idx = static_cast<size_t>(
                b * slice +
                (((tx * num_rx + rx) * num_doppler + d) * num_range) + r);
            const auto corr =
                table[static_cast<size_t>(tx * num_doppler + d)];
            const double mag =
                1.0 + 0.01 * static_cast<double>(rx) +
                0.001 * static_cast<double>(r);
            const std::complex<double> misaligned =
                std::conj(corr) * mag;
            host[idx] = std::complex<T>(
                static_cast<T>(misaligned.real()),
                static_cast<T>(misaligned.imag()));
          }
        }
      }
    }
  }
  return host;
}

template <typename T>
void expect_tx_lanes_aligned(const HostTensor& out, int64_t num_tx,
                             int64_t num_rx, int64_t num_doppler,
                             int64_t num_range, int64_t batch = 1,
                             T tol = static_cast<T>(1e-4)) {
  const auto* data = out.as<std::complex<T>>();
  const int64_t slice = num_tx * num_rx * num_doppler * num_range;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      for (int64_t d = 0; d < num_doppler; ++d) {
        for (int64_t r = 0; r < num_range; ++r) {
          const size_t ref = static_cast<size_t>(
              b * slice + ((0 * num_rx + rx) * num_doppler + d) * num_range +
              r);
          for (int64_t tx = 1; tx < num_tx; ++tx) {
            const size_t other = static_cast<size_t>(
                b * slice +
                ((tx * num_rx + rx) * num_doppler + d) * num_range + r);
            EXPECT_NEAR(data[other].real(), data[ref].real(), tol)
                << "batch=" << b << " tx=" << tx << " rx=" << rx
                << " d=" << d << " r=" << r;
            EXPECT_NEAR(data[other].imag(), data[ref].imag(), tol)
                << "batch=" << b << " tx=" << tx << " rx=" << rx
                << " d=" << d << " r=" << r;
          }
        }
      }
    }
  }
}

// Graph wiring lambda for 1-in / 1-out MIMOCoherentCompensationOp.
auto compensation_wire() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->rd_maps, q);
    graph.add_flow(op->rd_maps_out, sink->in, q);
  };
}

MIMOCoherentCompensationParams tdm_params(
    const FMCWWaveformDescriptor& waveform) {
  return MIMOCoherentCompensationParams{
      .multiplexing_scheme = MultiplexingScheme::kTdm,
      .waveform = waveform,
  };
}

MIMOCoherentCompensationParams ddm_params(
    const FMCWWaveformDescriptor& waveform,
    const DDMCodeDescriptor& code) {
  return MIMOCoherentCompensationParams{
      .multiplexing_scheme = MultiplexingScheme::kDdm,
      .waveform = waveform,
      .ddm_code = code,
  };
}

template <typename T>
GraphRunResult run_tdm_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs,
    MIMOCoherentCompensationParams params) {
  return run_single_io_operator_graph_result<
      MIMOCoherentCompensationOp<T>>(
      std::move(inputs), std::move(params), "rd_maps_out",
      compensation_wire());
}

template <typename T>
HostTensor run_tdm_once(
    std::shared_ptr<holoscan::Tensor> input,
    MIMOCoherentCompensationParams params) {
  GraphRunResult result =
      run_tdm_result<T>({std::move(input)}, std::move(params));
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames.size() != 1U) {
    throw std::runtime_error(
        "expected 1 frame, got " +
        std::to_string(result.frames.size()));
  }
  return std::move(result.frames.front());
}

template <typename T>
HostTensor run_compensation(const std::vector<std::complex<T>>& host,
                            const std::vector<int64_t>& shape,
                            const FMCWWaveformDescriptor& waveform) {
  return run_tdm_once<T>(
      to_device_tensor(host, shape, complex_dtype<T>()),
      tdm_params(waveform));
}

struct LiveWaveformTick {
  std::shared_ptr<holoscan::Tensor> rd_maps;
  std::optional<FMCWWaveformDescriptor> waveform;
  int64_t waveform_skew_ns{0};
  bool invalid_waveform{false};
};

struct LiveWaveformBank {
  static void reset() {
    ticks.clear();
    index = 0;
  }
  static inline std::vector<LiveWaveformTick> ticks;
  static inline std::size_t index{0};
};

holoscan::expected<void, holoscan::Error> emit_wire_waveform(
    holoscan::Output<holoscan::radar::wire::FMCWWaveformT>& output,
    holoscan::radar::wire::FMCWWaveformT wire,
    std::span<const std::complex<double>> samples,
    holoscan::TimePoint capture_time) {
  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(samples.size())};
  constexpr DLDataType kDtype{kDLComplex, 128U, 1U};
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = kDtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit(
      [samples](const holoscan::TensorOutputWriteGuard& writer) {
        std::copy(samples.begin(), samples.end(),
                  writer.data_as<std::complex<double>>());
      });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return output.emit_tensor(
      std::move(*loan), wire,
      holoscan::EmitOptions{.capture_time = capture_time});
}

class LiveWaveformSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    constexpr holoscan::TensorRepresentation kWaveform{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLComplex, 128U, 1U},
        .rank = 1U,
    };
    spec.output(rd_maps, "out")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 4U, 256ULL * 1024ULL * 1024ULL));
    spec.output(waveform, "waveform")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kWaveform, 1U, 1024U * sizeof(std::complex<double>)));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (LiveWaveformBank::index >= LiveWaveformBank::ticks.size()) {
      return {};
    }
    const LiveWaveformTick& tick =
        LiveWaveformBank::ticks[LiveWaveformBank::index++];
    const holoscan::TimePoint rd_time = context.activation_time();
    if (tick.waveform.has_value()) {
      auto wire = holoscan::radar::to_wire(*tick.waveform);
      if (tick.invalid_waveform) {
        wire.chirp_duration = 0.0;
      }
      holoscan::TimePoint waveform_time = rd_time;
      waveform_time.timestamp_ns += tick.waveform_skew_ns;
      auto emitted = emit_wire_waveform(
          waveform, std::move(wire),
          std::span<const std::complex<double>>{
              tick.waveform->reference_samples()},
          waveform_time);
      if (!emitted) {
        return emitted;
      }
    }
    return holoscan::radar::testing::detail::emit_bank_frame(
        rd_maps, context, tick.rd_maps, "LiveWaveformSourceOp");
  }

  holoscan::Output<holoscan::Tensor> rd_maps{"out"};
  holoscan::Output<holoscan::radar::wire::FMCWWaveformT>
      waveform{"waveform"};
};

GraphRunResult run_live_waveform_graph(
    std::vector<LiveWaveformTick> ticks,
    MIMOCoherentCompensationParams params) {
  GraphRunResult result;
  holoscan::radar::testing::detail::FrameBank::reset();
  LiveWaveformBank::reset();
  LiveWaveformBank::ticks = std::move(ticks);
  const std::size_t expected = LiveWaveformBank::ticks.size();

  holoscan::Graph graph{"mimo-live-waveform"};
  const auto source = graph.op<LiveWaveformSourceOp>("source");
  auto op = holoscan::radar::testing::emplace_harness_op<
      MIMOCoherentCompensationOp<float>>(
      graph, holoscan::radar::testing::default_test_extents());
  op.set_params(std::move(params));
  const auto sink =
      graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("sink");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->waveform, op->waveform_in, q);
  graph.add_flow(source->rd_maps, op->rd_maps, q);
  graph.add_flow(op->rd_maps_out, sink->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  holoscan::radar::testing::detail::bind_single_io_device_pools(
      options, "rd_maps_out");
  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }
    holoscan::RunSession session = holoscan::run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() +
                          holoscan::radar::testing::detail::kHarnessDeadline;
    while (session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      std::size_t count = 0;
      {
        const std::lock_guard lock(
            holoscan::radar::testing::detail::FrameBank::mutex);
        count =
            holoscan::radar::testing::detail::FrameBank::captured.size();
      }
      if (count >= expected) {
        break;
      }
      std::this_thread::sleep_for(
          holoscan::radar::testing::detail::kHarnessPoll);
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& diagnostic : session.diagnostics()) {
      result.diagnostics.emplace_back(diagnostic.token);
    }
  } catch (const holoscan::RuntimeError& error) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + error.what();
  }
  {
    const std::lock_guard lock(
        holoscan::radar::testing::detail::FrameBank::mutex);
    result.frames = std::move(
        holoscan::radar::testing::detail::FrameBank::captured);
  }
  return result;
}

// ---- DDM helpers ----------------------------------------------------------

constexpr int64_t kDdmTx = 4;
constexpr int64_t kDdmRx = 2;
constexpr int64_t kDdmDoppler = 16;
constexpr int64_t kDdmRange = 3;

int64_t fft_center(int64_t n) { return (n + 1) / 2; }

std::vector<double> uniform_offsets(int64_t num_tx) {
  std::vector<double> offsets(static_cast<size_t>(num_tx));
  for (int64_t m = 0; m < num_tx; ++m) {
    offsets[static_cast<size_t>(m)] =
        static_cast<double>(m) / static_cast<double>(num_tx);
  }
  return offsets;
}

DDMCodeDescriptor make_offset_code(int64_t num_tx, int64_t guard = 0) {
  DDMCodeDescriptor code;
  code.num_tx = num_tx;
  code.doppler_offsets = uniform_offsets(num_tx);
  code.guard_bands = guard;
  return code;
}

DDMCodeDescriptor make_offset_code_from(const std::vector<double>& offsets,
                                        int64_t guard = 0) {
  DDMCodeDescriptor code;
  code.num_tx = static_cast<int64_t>(offsets.size());
  code.doppler_offsets = offsets;
  code.guard_bands = guard;
  return code;
}

template <typename T>
std::complex<T> ddm_cell_value(int64_t tx, int64_t rx, int64_t r) {
  return std::complex<T>(
      static_cast<T>(1.0 + 0.1 * static_cast<double>(tx) +
                     0.01 * static_cast<double>(rx)),
      static_cast<T>(0.05 * static_cast<double>(r) -
                     0.02 * static_cast<double>(tx)));
}

template <typename T>
std::vector<std::complex<T>> build_ddm_input_host(
    const std::vector<double>& offsets, int64_t num_tx, int64_t num_rx,
    int64_t num_doppler, int64_t num_range, int64_t p0_reduced,
    int64_t batch = 1) {
  const int64_t D = num_doppler;
  const int64_t c = fft_center(D);
  const int64_t slice = num_rx * D * num_range;
  std::vector<std::complex<T>> host(
      static_cast<size_t>(batch * slice), std::complex<T>(0, 0));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t m = 0; m < num_tx; ++m) {
      const int64_t start =
          ((std::llround(offsets[static_cast<size_t>(m)] *
                         static_cast<double>(D)) %
            D) +
           D) %
          D;
      const int64_t j = (start + p0_reduced) % D;
      const int64_t in_cz = ((j - c) % D + D) % D;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t r = 0; r < num_range; ++r) {
          const size_t idx = static_cast<size_t>(
              b * slice + (rx * D + in_cz) * num_range + r);
          host[idx] = ddm_cell_value<T>(m, rx, r);
        }
      }
    }
  }
  return host;
}

template <typename T>
GraphRunResult run_ddm_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs,
    MIMOCoherentCompensationParams params) {
  return run_single_io_operator_graph_result<
      MIMOCoherentCompensationOp<T>>(
      std::move(inputs), std::move(params), "rd_maps_out",
      compensation_wire());
}

template <typename T>
HostTensor run_ddm_once(
    std::shared_ptr<holoscan::Tensor> input,
    MIMOCoherentCompensationParams params) {
  GraphRunResult result =
      run_ddm_result<T>({std::move(input)}, std::move(params));
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames.size() != 1U) {
    throw std::runtime_error(
        "expected 1 DDM frame, got " +
        std::to_string(result.frames.size()));
  }
  return std::move(result.frames.front());
}

template <typename T>
HostTensor run_ddm_from_host(const std::vector<std::complex<T>>& host,
                             const std::vector<int64_t>& shape,
                             const DDMCodeDescriptor& code,
                             const FMCWWaveformDescriptor& waveform) {
  return run_ddm_once<T>(
      to_device_tensor(host, shape, complex_dtype<T>()),
      ddm_params(waveform, code));
}

template <typename T>
void expect_ddm_recovered(const HostTensor& out, int64_t num_tx,
                          int64_t num_rx, int64_t band_width,
                          int64_t num_range, int64_t p0_reduced,
                          int64_t batch = 1,
                          T tol = static_cast<T>(1e-4)) {
  const int64_t W = band_width;
  const int64_t cW = fft_center(W);
  const int64_t o0 = ((p0_reduced - cW) % W + W) % W;
  const auto* data = out.as<std::complex<T>>();
  const int64_t slice = num_tx * num_rx * W * num_range;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t m = 0; m < num_tx; ++m) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t o = 0; o < W; ++o) {
          for (int64_t r = 0; r < num_range; ++r) {
            const size_t idx = static_cast<size_t>(
                b * slice + ((m * num_rx + rx) * W + o) * num_range + r);
            const std::complex<T> expected =
                (o == o0) ? ddm_cell_value<T>(m, rx, r)
                          : std::complex<T>(0, 0);
            EXPECT_NEAR(data[idx].real(), expected.real(), tol)
                << "batch=" << b << " tx=" << m << " rx=" << rx
                << " o=" << o << " r=" << r;
            EXPECT_NEAR(data[idx].imag(), expected.imag(), tol)
                << "batch=" << b << " tx=" << m << " rx=" << rx
                << " o=" << o << " r=" << r;
          }
        }
      }
    }
  }
}

// ---- TDM tests ------------------------------------------------------------

TEST(MIMOCoherentCompensationOp, TdmRemovesInterTxPhaseAcrossDopplerBins) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto out = run_compensation<float>(
      build_misaligned_rd_host<float>(waveform, kNumTx, kNumRx, kDoppler,
                                      kRange),
      shape, waveform);
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, FourTxLanesAlignAfterCompensation) {
  constexpr int64_t kTx = 4;
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kTx, kNumRx, kDoppler, kRange};
  const auto out = run_compensation<float>(
      build_misaligned_rd_host<float>(waveform, kTx, kNumRx, kDoppler,
                                      kRange),
      shape, waveform);
  expect_tx_lanes_aligned<float>(out, kTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, TdmCorrectionTableIsUnitMagnitude) {
  const auto table = holoscan::radar::detail::build_tdm_correction_table(
      make_test_waveform(), kNumTx, kDoppler);
  for (const auto& factor : table) {
    EXPECT_NEAR(std::abs(factor), 1.0, 1e-12);
  }
}

TEST(MIMOCoherentCompensationOp, TdmCorrectionTableTxZeroIsIdentity) {
  const auto table = holoscan::radar::detail::build_tdm_correction_table(
      make_test_waveform(), kNumTx, kDoppler);
  for (int64_t d = 0; d < kDoppler; ++d) {
    const auto& factor = table[static_cast<size_t>(d)];
    EXPECT_NEAR(factor.real(), 1.0, 1e-12) << "doppler bin " << d;
    EXPECT_NEAR(factor.imag(), 0.0, 1e-12) << "doppler bin " << d;
  }
}

TEST(MIMOCoherentCompensationOp, TdmCorrectionTableUsesFiringSlot) {
  // tx_order [1, 0]: TX 1 fires first (identity), TX 0 is slot 1.
  const std::vector<int64_t> slot_by_tx{1, 0};
  const auto waveform = make_test_waveform();
  const auto table = holoscan::radar::detail::build_tdm_correction_table(
      waveform, kNumTx, kDoppler, slot_by_tx);
  const auto identity = holoscan::radar::detail::build_tdm_correction_table(
      waveform, kNumTx, kDoppler);
  for (int64_t d = 0; d < kDoppler; ++d) {
    const auto& first = table[static_cast<size_t>(1 * kDoppler + d)];
    EXPECT_NEAR(first.real(), 1.0, 1e-12) << "first-firing TX at d=" << d;
    EXPECT_NEAR(first.imag(), 0.0, 1e-12) << "first-firing TX at d=" << d;
    const auto& swapped = table[static_cast<size_t>(d)];
    const auto& expect = identity[static_cast<size_t>(1 * kDoppler + d)];
    EXPECT_NEAR(swapped.real(), expect.real(), 1e-12) << "d=" << d;
    EXPECT_NEAR(swapped.imag(), expect.imag(), 1e-12) << "d=" << d;
  }
}

TEST(MIMOCoherentCompensationOp, PermutedTxOrderAlignsAfterCompensation) {
  const auto waveform = make_test_waveform();
  TXSchedule schedule;
  schedule.num_tx = kNumTx;
  schedule.tx_order = {1, 0};
  const auto slots = holoscan::radar::detail::tdm_firing_slots(
      schedule, kNumTx);
  ASSERT_EQ(slots, (std::vector<int64_t>{1, 0}));
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  MIMOCoherentCompensationParams params = tdm_params(waveform);
  params.tx_schedule = schedule;
  const auto out = run_tdm_once<float>(
      to_device_tensor(
          build_misaligned_rd_host<float>(waveform, kNumTx, kNumRx,
                                          kDoppler, kRange, 1, slots),
          shape, complex_dtype<float>()),
      params);
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, RejectsTxScheduleNumTxMismatch) {
  const auto waveform = make_test_waveform();
  TXSchedule schedule;
  schedule.num_tx = 3;
  schedule.tx_order = {0, 1, 2};
  MIMOCoherentCompensationParams params = tdm_params(waveform);
  params.tx_schedule = schedule;
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const GraphRunResult result = run_tdm_result<float>(
      {to_device_tensor(host, shape, complex_dtype<float>())}, params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, ComplexFloatDtypePreserved) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto out = run_compensation<float>(host, shape, waveform);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<float>()));
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, ComplexDoublePreservesValues) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<double>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto out = run_compensation<double>(host, shape, waveform);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<double>()));
  expect_tx_lanes_aligned<double>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, DeterministicAcrossRuns) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto first = run_compensation<float>(host, shape, waveform);
  const auto second = run_compensation<float>(host, shape, waveform);
  ASSERT_EQ(first.shape, second.shape);
  const auto* a = first.as<std::complex<float>>();
  const auto* b = second.as<std::complex<float>>();
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), 1e-6f);
    EXPECT_NEAR(a[i].imag(), b[i].imag(), 1e-6f);
  }
}

TEST(MIMOCoherentCompensationOp, SingleTxIsIdentity) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{1, kNumRx, kDoppler, kRange};
  const auto host =
      build_misaligned_rd_host<float>(waveform, 1, kNumRx, kDoppler, kRange);
  const auto out = run_compensation<float>(host, shape, waveform);
  const auto* got = out.as<std::complex<float>>();
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_NEAR(got[i].real(), host[i].real(), 1e-5f);
    EXPECT_NEAR(got[i].imag(), host[i].imag(), 1e-5f);
  }
}

TEST(MIMOCoherentCompensationOp, RejectsOpaqueLeadingBatch) {
  // TDM contract is (M_tx, N_rx, D, R); an extra leading dim is opaque batch.
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{2, kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange, 2);
  const GraphRunResult result = run_tdm_result<float>(
      {to_device_tensor(host, shape, complex_dtype<float>())},
      tdm_params(waveform));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, WaveformConfigurableFromParams) {
  const auto waveform_a = make_test_waveform(kPri);
  const auto waveform_b = make_test_waveform(kPri * 1.1);
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform_a, kNumTx, kNumRx, kDoppler, kRange);
  const auto out_a = run_compensation<float>(host, shape, waveform_a);
  const auto out_b = run_compensation<float>(host, shape, waveform_b);
  expect_tx_lanes_aligned<float>(out_a, kNumTx, kNumRx, kDoppler, kRange);
  expect_tx_lanes_aligned<float>(out_b, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, LiveWaveformOverrideAppliesPerTick) {
  const auto waveform_a = make_test_waveform(kPri);
  const auto waveform_b = make_test_waveform(kPri * 1.3);
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto frame_a = to_device_tensor(
      build_misaligned_rd_host<float>(
          waveform_a, kNumTx, kNumRx, kDoppler, kRange),
      shape, complex_dtype<float>());
  const auto frame_b = to_device_tensor(
      build_misaligned_rd_host<float>(
          waveform_b, kNumTx, kNumRx, kDoppler, kRange),
      shape, complex_dtype<float>());
  MIMOCoherentCompensationParams params = tdm_params(waveform_a);

  // Tick 0 and tick 2 publish nothing, so both compensate with the Params
  // waveform_a and are paired with the frame built from it; tick 1 matches
  // Exact on waveform_b. An override that lasted longer would misalign
  // tick 2's lanes.
  const GraphRunResult out = run_live_waveform_graph(
      {{frame_a, std::nullopt}, {frame_b, waveform_b},
       {frame_a, std::nullopt}},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 3U);
  for (const HostTensor& frame : out.frames) {
    expect_tx_lanes_aligned<float>(
        frame, kNumTx, kNumRx, kDoppler, kRange);
  }
}

TEST(MIMOCoherentCompensationOp, LiveWaveformUsesExactCaptureTime) {
  const auto waveform_a = make_test_waveform(kPri);
  const auto waveform_b = make_test_waveform(kPri * 1.3);
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto frame = to_device_tensor(
      build_misaligned_rd_host<float>(
          waveform_a, kNumTx, kNumRx, kDoppler, kRange),
      shape, complex_dtype<float>());
  MIMOCoherentCompensationParams params = tdm_params(waveform_a);

  const GraphRunResult out = run_live_waveform_graph(
      {{frame, waveform_b, 1'000'000}}, std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 1U);
  expect_tx_lanes_aligned<float>(
      out.frames.front(), kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, RejectsInvalidLiveWaveform) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto frame = to_device_tensor(
      build_misaligned_rd_host<float>(
          waveform, kNumTx, kNumRx, kDoppler, kRange),
      shape, complex_dtype<float>());
  MIMOCoherentCompensationParams params = tdm_params(waveform);

  const GraphRunResult out = run_live_waveform_graph(
      {{frame, waveform, 0, true}}, std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_TRUE(out.frames.empty());
  EXPECT_FALSE(out.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, FloatDoubleAgreement) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host_f = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto host_d = build_misaligned_rd_host<double>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto out_f = run_compensation<float>(host_f, shape, waveform);
  const auto out_d = run_compensation<double>(host_d, shape, waveform);
  const auto* f = out_f.as<std::complex<float>>();
  const auto* d = out_d.as<std::complex<double>>();
  for (size_t i = 0; i < host_f.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(f[i].real()), d[i].real(), 1e-4);
    EXPECT_NEAR(static_cast<double>(f[i].imag()), d[i].imag(), 1e-4);
  }
}

TEST(MIMOCoherentCompensationOp, RejectsUnspecifiedMultiplexingScheme) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  MIMOCoherentCompensationParams params{
      .multiplexing_scheme = MultiplexingScheme::kUnspecified,
      .waveform = waveform,
  };
  const GraphRunResult result =
      run_tdm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          std::move(params));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, RejectsInvalidWaveform) {
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  const GraphRunResult result =
      run_tdm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          tdm_params(FMCWWaveformDescriptor{}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, RejectsRankTooSmall) {
  const auto waveform = make_test_waveform();
  const std::vector<std::complex<float>> host(8, {1.0f, 0.0f});
  const GraphRunResult result =
      run_tdm_result<float>(
          {to_device_tensor(host, {2, 2, 2}, complex_dtype<float>())},
          tdm_params(waveform));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, AcceptsPinnedHostInput) {
  // [EA HSDK5 ONLY] Graph harness only delivers device Tensors.
  GTEST_SKIP() << "HSDK 5 Graph: pinned host input not deliverable "
                  "through device-bound tensor loans";
}

TEST(MIMOCoherentCompensationOp, AcceptsManagedInput) {
  // [EA HSDK5 ONLY] Graph harness only delivers device Tensors.
  GTEST_SKIP() << "HSDK 5 Graph: managed input not deliverable "
                  "through device-bound tensor loans";
}

TEST(MIMOCoherentCompensationOp, OutputEmittedOnDevice) {
  // [EA HSDK5 ONLY] emit_device_tensor always produces kDLCUDA output;
  // the Graph harness returns HostTensors via to_host(). Device origin
  // is guaranteed by the produces_tensor(kDevice) declaration.
  GTEST_SKIP() << "HSDK 5 Graph: device output guaranteed by "
                  "produces_tensor(kDevice); no in-flight capture";
}

TEST(MIMOCoherentCompensationOp, AcceptsSingleEntryWithArbitraryKey) {
  // [EA HSDK5 ONLY] HSDK 5 uses typed Input<Tensor> — no TensorMap,
  // no map key semantics. Port name is the identity.
  GTEST_SKIP() << "HSDK 5: typed Input<Tensor> replaces TensorMap; "
                  "no arbitrary-key semantics";
}

TEST(MIMOCoherentCompensationOp, RejectsMultiEntryTensorMap) {
  // [EA HSDK5 ONLY] HSDK 5 uses typed Input<Tensor> — no TensorMap.
  GTEST_SKIP() << "HSDK 5: typed Input<Tensor> replaces TensorMap; "
                  "multi-entry not applicable";
}

TEST(MIMOCoherentCompensationOp, RejectsNonContiguousInput) {
  // [EA HSDK5 ONLY] Graph harness produces contiguous device tensors.
  GTEST_SKIP() << "HSDK 5 Graph: non-contiguous input cannot reach "
                  "the operator through device-bound tensor loans";
}

// ---- DDM tests ------------------------------------------------------------

TEST(MIMOCoherentCompensationOp, DdmDemuxRecoversEachTxIntoBand) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], kDdmTx);
  EXPECT_EQ(out.shape[1], kDdmRx);
  EXPECT_EQ(out.shape[2], band);
  EXPECT_EQ(out.shape[3], kDdmRange);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmSingleTxIsIdentity) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(1);
  const int64_t p0 = 2;
  const auto offsets = uniform_offsets(1);
  const auto host = build_ddm_input_host<float>(offsets, 1, kDdmRx,
                                                kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], 1);
  EXPECT_EQ(out.shape[2], kDdmDoppler);
  const auto* got = out.as<std::complex<float>>();
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_NEAR(got[i].real(), host[i].real(), 1e-5f) << "i=" << i;
    EXPECT_NEAR(got[i].imag(), host[i].imag(), 1e-5f) << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, DdmRejectsOpaqueLeadingBatch) {
  // DDM input contract is (N_rx, D, R); an extra leading dim is opaque batch.
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t p0 = 0;
  const int64_t batch = 2;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0, batch);
  const std::vector<int64_t> shape{batch, kDdmRx, kDdmDoppler, kDdmRange};
  const GraphRunResult result = run_ddm_result<float>(
      {to_device_tensor(host, shape, complex_dtype<float>())},
      ddm_params(waveform, code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, DdmLinearPhaseCodesMatchOffsets) {
  const auto waveform = make_test_waveform();
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};

  DDMCodeDescriptor code;
  code.num_tx = kDdmTx;
  code.guard_bands = 0;
  const int64_t num_chirps = kDdmDoppler;
  code.phase_codes.resize(static_cast<size_t>(kDdmTx * num_chirps));
  for (int64_t m = 0; m < kDdmTx; ++m) {
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double phase = kTwoPi *
                           (static_cast<double>(m) /
                            static_cast<double>(kDdmTx)) *
                           static_cast<double>(k);
      code.phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }

  const int64_t band = kDdmDoppler / kDdmTx;
  const auto out_codes =
      run_ddm_from_host<float>(host, shape, code, waveform);
  const auto out_offsets = run_ddm_from_host<float>(
      host, shape, make_offset_code(kDdmTx), waveform);
  ASSERT_EQ(out_codes.shape.size(), out_offsets.shape.size());
  const auto* a = out_codes.as<std::complex<float>>();
  const auto* b = out_offsets.as<std::complex<float>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), 1e-5f) << "i=" << i;
    EXPECT_NEAR(a[i].imag(), b[i].imag(), 1e-5f) << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp,
     DdmNonLinearPhaseCodesCorrelateRecovers) {
  const auto waveform = make_test_waveform();
  constexpr int64_t kNlTx = 1;
  constexpr int64_t kNlRx = 2;
  constexpr int64_t kNlDoppler = 16;
  constexpr int64_t kNlRange = 3;
  const int64_t num_chirps = kNlDoppler;
  constexpr int64_t d0 = 3;
  const int64_t D = kNlDoppler;
  const int64_t c = fft_center(D);
  const int64_t d0_cz = ((d0 - c) % D + D) % D;
  const std::complex<double> val(1.5, -0.3);

  DDMCodeDescriptor code;
  code.num_tx = kNlTx;
  code.guard_bands = 0;
  code.phase_codes.resize(static_cast<size_t>(kNlTx * num_chirps));
  for (int64_t k = 0; k < num_chirps; ++k) {
    const double phase = std::sin(0.7 * static_cast<double>(k)) * 2.0;
    code.phase_codes[static_cast<size_t>(k)] =
        std::complex<double>(std::cos(phase), std::sin(phase));
  }

  std::vector<std::complex<double>> spectrum(static_cast<size_t>(D));
  const auto* c_m = code.phase_codes.data();
  for (int64_t k = 0; k < D; ++k) {
    std::complex<double> sum(0.0, 0.0);
    for (int64_t n = 0; n < num_chirps; ++n) {
      const double angle = -kTwoPi * static_cast<double>(n) *
                           static_cast<double>(k) / static_cast<double>(D);
      sum += c_m[n] * std::complex<double>(std::cos(angle),
                                           std::sin(angle));
    }
    spectrum[static_cast<size_t>(k)] = sum;
  }

  std::vector<std::complex<float>> host(
      static_cast<size_t>(kNlRx * D * kNlRange), {0.0f, 0.0f});
  for (int64_t rx = 0; rx < kNlRx; ++rx) {
    for (int64_t r = 0; r < kNlRange; ++r) {
      for (int64_t k = 0; k < D; ++k) {
        const int64_t idx = ((k - d0) % D + D) % D;
        const auto y = val * spectrum[static_cast<size_t>(idx)];
        const int64_t d_cz = ((k - c) % D + D) % D;
        host[static_cast<size_t>((rx * D + d_cz) * kNlRange + r)] =
            std::complex<float>(static_cast<float>(y.real()),
                                static_cast<float>(y.imag()));
      }
    }
  }

  const std::vector<int64_t> shape{kNlRx, kNlDoppler, kNlRange};
  const auto out =
      run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], kNlTx);
  EXPECT_EQ(out.shape[1], kNlRx);
  EXPECT_EQ(out.shape[2], kNlDoppler);
  EXPECT_EQ(out.shape[3], kNlRange);

  const auto* data = out.as<std::complex<float>>();
  const std::complex<float> expected(
      static_cast<float>(val.real() * static_cast<double>(D)),
      static_cast<float>(val.imag() * static_cast<double>(D)));
  const float tol = 0.05f * std::abs(expected);
  for (int64_t rx = 0; rx < kNlRx; ++rx) {
    for (int64_t r = 0; r < kNlRange; ++r) {
      float peak = 0.0f;
      float other = 0.0f;
      std::complex<float> at_d0{0.0f, 0.0f};
      for (int64_t d = 0; d < D; ++d) {
        const size_t idx_out = static_cast<size_t>(
            ((0 * kNlRx + rx) * D + d) * kNlRange + r);
        const auto z = data[idx_out];
        const float mag = std::abs(z);
        if (d == d0_cz) {
          peak = mag;
          at_d0 = z;
        } else {
          other = std::max(other, mag);
        }
      }
      EXPECT_NEAR(at_d0.real(), expected.real(), tol) << "rx=" << rx;
      EXPECT_NEAR(at_d0.imag(), expected.imag(), tol) << "rx=" << rx;
      EXPECT_GT(peak, 5.0f * other) << "rx=" << rx;
    }
  }
}

TEST(MIMOCoherentCompensationOp, DdmNonLinearMultiTxSeparatesPeaks) {
  const auto waveform = make_test_waveform();
  constexpr int64_t kNlTx = 2;
  constexpr int64_t kNlRx = 1;
  constexpr int64_t kNlDoppler = 15;
  constexpr int64_t kNlRange = 2;
  const int64_t num_chirps = kNlDoppler;
  constexpr int64_t d0 = 2;
  constexpr int64_t d1 = 5;
  const int64_t D = kNlDoppler;
  const int64_t c = fft_center(D);
  const int64_t d0_cz = ((d0 - c) % D + D) % D;
  const int64_t d1_cz = ((d1 - c) % D + D) % D;
  const std::complex<double> val0(1.2, -0.1);
  const std::complex<double> val1(-0.8, 0.4);

  std::vector<int64_t> roots;
  for (int64_t u = 1;
       u < D && static_cast<int64_t>(roots.size()) < kNlTx; ++u) {
    if (std::gcd(u, D) == 1) {
      roots.push_back(u);
    }
  }
  ASSERT_EQ(static_cast<int64_t>(roots.size()), kNlTx);

  DDMCodeDescriptor code;
  code.num_tx = kNlTx;
  code.guard_bands = 0;
  code.phase_codes.resize(static_cast<size_t>(kNlTx * num_chirps));
  const bool odd_n = (num_chirps % 2) != 0;
  for (int64_t m = 0; m < kNlTx; ++m) {
    const double u = static_cast<double>(roots[static_cast<size_t>(m)]);
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double kd = static_cast<double>(k);
      const double arg =
          odd_n
              ? (-kPi * u * kd * (kd + 1.0) /
                 static_cast<double>(num_chirps))
              : (-kPi * u * kd * kd /
                 static_cast<double>(num_chirps));
      code.phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(arg), std::sin(arg));
    }
  }

  auto code_spectrum = [&](int64_t m) {
    std::vector<std::complex<double>> spectrum(static_cast<size_t>(D));
    const auto* c_m =
        &code.phase_codes[static_cast<size_t>(m * num_chirps)];
    for (int64_t k = 0; k < D; ++k) {
      std::complex<double> sum(0.0, 0.0);
      for (int64_t n = 0; n < num_chirps; ++n) {
        const double angle = -kTwoPi * static_cast<double>(n) *
                             static_cast<double>(k) /
                             static_cast<double>(D);
        sum +=
            c_m[n] *
            std::complex<double>(std::cos(angle), std::sin(angle));
      }
      spectrum[static_cast<size_t>(k)] = sum;
    }
    return spectrum;
  };
  const auto spec0 = code_spectrum(0);
  const auto spec1 = code_spectrum(1);

  std::vector<std::complex<float>> host(
      static_cast<size_t>(kNlRx * D * kNlRange), {0.0f, 0.0f});
  for (int64_t rx = 0; rx < kNlRx; ++rx) {
    for (int64_t r = 0; r < kNlRange; ++r) {
      for (int64_t k = 0; k < D; ++k) {
        const int64_t idx0 = ((k - d0) % D + D) % D;
        const int64_t idx1 = ((k - d1) % D + D) % D;
        const auto y = val0 * spec0[static_cast<size_t>(idx0)] +
                       val1 * spec1[static_cast<size_t>(idx1)];
        const int64_t d_cz = ((k - c) % D + D) % D;
        host[static_cast<size_t>((rx * D + d_cz) * kNlRange + r)] =
            std::complex<float>(static_cast<float>(y.real()),
                                static_cast<float>(y.imag()));
      }
    }
  }

  const std::vector<int64_t> shape{kNlRx, kNlDoppler, kNlRange};
  const auto out =
      run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], kNlTx);
  EXPECT_EQ(out.shape[1], kNlRx);
  EXPECT_EQ(out.shape[2], kNlDoppler);
  EXPECT_EQ(out.shape[3], kNlRange);

  const auto* data = out.as<std::complex<float>>();
  for (int64_t rx = 0; rx < kNlRx; ++rx) {
    for (int64_t r = 0; r < kNlRange; ++r) {
      for (int64_t m = 0; m < kNlTx; ++m) {
        const int64_t expect_cz = (m == 0) ? d0_cz : d1_cz;
        float peak = 0.0f;
        float other_max = 0.0f;
        int64_t peak_bin = -1;
        for (int64_t d = 0; d < D; ++d) {
          const auto z = data[static_cast<size_t>(
              ((m * kNlRx + rx) * D + d) * kNlRange + r)];
          const float mag = std::abs(z);
          if (mag > peak) {
            peak = mag;
            peak_bin = d;
          }
          if (d != expect_cz) {
            other_max = std::max(other_max, mag);
          }
        }
        EXPECT_EQ(peak_bin, expect_cz) << "m=" << m << " rx=" << rx;
        EXPECT_GT(peak, other_max) << "m=" << m << " rx=" << rx;
      }
    }
  }
}

TEST(MIMOCoherentCompensationOp,
     DdmNonLinearRejectsNonZeroGuardBands) {
  const auto waveform = make_test_waveform();
  constexpr int64_t kNlTx = 2;
  constexpr int64_t kNlDoppler = 16;
  constexpr int64_t kNlRx = 2;
  constexpr int64_t kNlRange = 3;
  const int64_t num_chirps = kNlDoppler;

  DDMCodeDescriptor code;
  code.num_tx = kNlTx;
  code.guard_bands = 1;
  code.phase_codes.resize(static_cast<size_t>(kNlTx * num_chirps));
  for (int64_t m = 0; m < kNlTx; ++m) {
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double phase =
          std::sin(0.7 * static_cast<double>(k) + m) * 2.0;
      code.phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }

  const std::vector<int64_t> shape{kNlRx, kNlDoppler, kNlRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kNlRx * kNlDoppler * kNlRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(waveform, code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp,
     DdmNonLinearRejectsChirpDopplerMismatch) {
  const auto waveform = make_test_waveform();
  constexpr int64_t kNlTx = 2;
  constexpr int64_t kNlDoppler = 16;
  constexpr int64_t kNlRx = 2;
  constexpr int64_t kNlRange = 3;
  const int64_t num_chirps = 8;

  DDMCodeDescriptor code;
  code.num_tx = kNlTx;
  code.guard_bands = 0;
  code.phase_codes.resize(static_cast<size_t>(kNlTx * num_chirps));
  for (int64_t m = 0; m < kNlTx; ++m) {
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double phase =
          std::sin(0.7 * static_cast<double>(k) + m) * 2.0;
      code.phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }

  const std::vector<int64_t> shape{kNlRx, kNlDoppler, kNlRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kNlRx * kNlDoppler * kNlRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(waveform, code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, DdmGuardBandsReduceWidth) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx, /*guard=*/1);
  const int64_t band = kDdmDoppler / kDdmTx - 1;
  const int64_t p0 = 0;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[2], band);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmGuardBandDiscardsEdgeBin) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx, /*guard=*/1);
  const int64_t band = kDdmDoppler / kDdmTx - 1;
  const int64_t edge = kDdmDoppler / kDdmTx - 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, edge);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  const auto* data = out.as<std::complex<float>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(data[i].real(), 0.0f, 1e-5f) << "i=" << i;
    EXPECT_NEAR(data[i].imag(), 0.0f, 1e-5f) << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, DdmComplexDoublePreserved) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 2;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<double>(host, shape, code, waveform);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<double>()));
  expect_ddm_recovered<double>(
      out, kDdmTx, kDdmRx, band, kDdmRange, p0, 1, 1e-9);
}

TEST(MIMOCoherentCompensationOp, DdmComplexFloatDtypePreserved) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<float>()));
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmFloatDoubleAgreement) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto host_f = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const auto host_d = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const auto out_f =
      run_ddm_from_host<float>(host_f, shape, code, waveform);
  const auto out_d =
      run_ddm_from_host<double>(host_d, shape, code, waveform);
  const auto* f = out_f.as<std::complex<float>>();
  const auto* d = out_d.as<std::complex<double>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(static_cast<double>(f[i].real()), d[i].real(), 1e-4)
        << "i=" << i;
    EXPECT_NEAR(static_cast<double>(f[i].imag()), d[i].imag(), 1e-4)
        << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, DdmNonUniformOffsetPermutation) {
  const auto waveform = make_test_waveform();
  const std::vector<double> permuted{0.5, 0.0, 0.75, 0.25};
  const auto code = make_offset_code_from(permuted);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 0;
  const auto host = build_ddm_input_host<float>(
      permuted, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmDeterministicAcrossRuns) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto first = run_ddm_from_host<float>(host, shape, code, waveform);
  const auto second =
      run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(first.shape, second.shape);
  const auto* a = first.as<std::complex<float>>();
  const auto* b = second.as<std::complex<float>>();
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), 1e-6f) << "i=" << i;
    EXPECT_NEAR(a[i].imag(), b[i].imag(), 1e-6f) << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, RejectsEmptyDdmCode) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(waveform, DDMCodeDescriptor{}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, DdmRejectsNonDivisibleDoppler) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(3);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(waveform, code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, DdmRejectsGuardTooLarge) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx, /*guard=*/4);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(waveform, code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, DdmRejectsRankTooSmall) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(2);
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, {kDdmDoppler, kDdmRange},
                            complex_dtype<float>())},
          ddm_params(waveform, code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, DdmRejectsBothOffsetForms) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  DDMCodeDescriptor code;
  code.num_tx = 2;
  code.doppler_offsets = {0.0, 0.5};
  code.phase_codes = {{1.0, 0.0}, {0.0, 1.0}};
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(waveform, code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, DdmOutputEmittedOnDevice) {
  // [EA HSDK5 ONLY] Device output guaranteed by produces_tensor(kDevice).
  GTEST_SKIP() << "HSDK 5 Graph: device output guaranteed by "
                  "produces_tensor(kDevice); no in-flight capture";
}

TEST(MIMOCoherentCompensationOp, DdmAcceptsSingleEntryWithArbitraryKey) {
  // [EA HSDK5 ONLY] No TensorMap — typed Input<Tensor>.
  GTEST_SKIP() << "HSDK 5: typed Input<Tensor> replaces TensorMap; "
                  "no arbitrary-key semantics";
}

TEST(MIMOCoherentCompensationOp, DdmAcceptsPinnedHostInput) {
  GTEST_SKIP() << "HSDK 5 Graph: pinned host input not deliverable "
                  "through device-bound tensor loans";
}

TEST(MIMOCoherentCompensationOp, DdmAcceptsManagedInput) {
  GTEST_SKIP() << "HSDK 5 Graph: managed input not deliverable "
                  "through device-bound tensor loans";
}

TEST(MIMOCoherentCompensationOp, DdmWaveformOverrideAcceptedButInert) {
  // [EA HSDK5 ONLY] waveform is Params-only on EA (§9); mid-stream
  // override testing blocked (§1). The DDM waveform-inert property is
  // still validated by running two distinct waveforms through Params.
  const auto waveform_a = make_test_waveform(kPri);
  const auto waveform_b = make_test_waveform(kPri * 1.3);
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out_a = run_ddm_from_host<float>(host, shape, code, waveform_a);
  const auto out_b = run_ddm_from_host<float>(host, shape, code, waveform_b);
  ASSERT_EQ(out_a.shape, out_b.shape);
  const auto* a = out_a.as<std::complex<float>>();
  const auto* b = out_b.as<std::complex<float>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), 1e-6f) << "i=" << i;
    EXPECT_NEAR(a[i].imag(), b[i].imag(), 1e-6f) << "i=" << i;
  }
  expect_ddm_recovered<float>(out_b, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmRejectsMismatchedPrecision) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, 0);
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<double>())},
          ddm_params(waveform, code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, RejectsMismatchedPrecision) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<double>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const GraphRunResult result =
      run_tdm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<double>())},
          tdm_params(waveform));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, RejectsRealInput) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const std::vector<float> host(
      static_cast<size_t>(tensor_count(shape)), 1.0f);
  const GraphRunResult result =
      run_tdm_result<float>(
          {to_device_tensor(host, shape, DLDataType{kDLFloat, 32, 1})},
          tdm_params(waveform));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MIMOCoherentCompensationOp, RejectsZeroTrailingAxis) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{0, kNumRx, kDoppler, kRange};
  const std::vector<std::complex<float>> host;
  const GraphRunResult result =
      run_tdm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          tdm_params(waveform));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

}  // namespace
