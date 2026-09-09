// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Contract tests for DDMDenseDemultiplexOp (HSDK 5 Graph harness).
// Covers linear Doppler-band gather and nonlinear conjugate correlation,
// including code forms, guard bands, dtype, shape, and reject paths.

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

#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::DDMCodeDescriptor;
using holoscan::radar::kPi;
using holoscan::radar::kTwoPi;
using holoscan::radar::DDMDenseDemultiplexOp;
using holoscan::radar::DDMDenseDemultiplexParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::dldatatype_equals;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

// Graph wiring lambda for 1-in / 1-out DDMDenseDemultiplexOp.
auto compensation_wire() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->rd_maps, q);
    graph.add_flow(op->rd_maps_out, sink->in, q);
  };
}


DDMDenseDemultiplexParams ddm_params(const DDMCodeDescriptor& code) {
  return DDMDenseDemultiplexParams{.ddm_code = code};
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
    DDMDenseDemultiplexParams params,
    const holoscan::radar::MaxTensorExtents& extents =
        holoscan::radar::testing::default_test_extents()) {
  return run_single_io_operator_graph_result<
      DDMDenseDemultiplexOp<T>>(
      std::move(inputs), std::move(params), "rd_maps_out",
      compensation_wire(), {}, extents);
}

template <typename T>
HostTensor run_ddm_once(
    std::shared_ptr<holoscan::Tensor> input,
    DDMDenseDemultiplexParams params) {
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
                             const DDMCodeDescriptor& code) {
  return run_ddm_once<T>(
      to_device_tensor(host, shape, complex_dtype<T>()),
      ddm_params(code));
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
TEST(DDMDenseDemultiplexOp, DdmDemuxRecoversEachTxIntoBand) {
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], kDdmTx);
  EXPECT_EQ(out.shape[1], kDdmRx);
  EXPECT_EQ(out.shape[2], band);
  EXPECT_EQ(out.shape[3], kDdmRange);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(DDMDenseDemultiplexOp, DdmSingleTxIsIdentity) {
  const auto code = make_offset_code(1);
  const int64_t p0 = 2;
  const auto offsets = uniform_offsets(1);
  const auto host = build_ddm_input_host<float>(offsets, 1, kDdmRx,
                                                kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], 1);
  EXPECT_EQ(out.shape[2], kDdmDoppler);
  const auto* got = out.as<std::complex<float>>();
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_NEAR(got[i].real(), host[i].real(), 1e-5f) << "i=" << i;
    EXPECT_NEAR(got[i].imag(), host[i].imag(), 1e-5f) << "i=" << i;
  }
}

TEST(DDMDenseDemultiplexOp, DdmRejectsOpaqueLeadingBatch) {
  // DDM input contract is (N_rx, D, R); an extra leading dim is opaque batch.
  const auto code = make_offset_code(kDdmTx);
  const int64_t p0 = 0;
  const int64_t batch = 2;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0, batch);
  const std::vector<int64_t> shape{batch, kDdmRx, kDdmDoppler, kDdmRange};
  const GraphRunResult result = run_ddm_result<float>(
      {to_device_tensor(host, shape, complex_dtype<float>())},
      ddm_params(code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp, RejectsInputBeyondCapturedExtents) {
  const auto code = make_offset_code(kDdmTx);
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, 0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_rx = kDdmRx - 1;
  const GraphRunResult result = run_ddm_result<float>(
      {to_device_tensor(host, shape, complex_dtype<float>())},
      ddm_params(code), extents);
  EXPECT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp, DdmLinearPhaseCodesMatchOffsets) {
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
      run_ddm_from_host<float>(host, shape, code);
  const auto out_offsets = run_ddm_from_host<float>(
      host, shape, make_offset_code(kDdmTx));
  ASSERT_EQ(out_codes.shape.size(), out_offsets.shape.size());
  const auto* a = out_codes.as<std::complex<float>>();
  const auto* b = out_offsets.as<std::complex<float>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), 1e-5f) << "i=" << i;
    EXPECT_NEAR(a[i].imag(), b[i].imag(), 1e-5f) << "i=" << i;
  }
}

TEST(DDMDenseDemultiplexOp, DdmLinearCodesRemoveNonzeroInitialPhase) {
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  auto host = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  DDMCodeDescriptor code;
  code.num_tx = kDdmTx;
  const std::array<std::complex<double>, kDdmTx> initial{
      std::complex<double>{0.0, 1.0}, {-1.0, 0.0},
      {0.0, -1.0}, {std::sqrt(0.5), std::sqrt(0.5)}};
  for (int64_t tx = 0; tx < kDdmTx; ++tx) {
    for (int64_t chirp = 0; chirp < kDdmDoppler; ++chirp) {
      const double phase = kTwoPi * offsets[static_cast<size_t>(tx)] *
                           static_cast<double>(chirp);
      code.phase_codes.push_back(
          initial[static_cast<size_t>(tx)] *
          std::complex<double>(std::cos(phase), std::sin(phase)));
    }
  }
  const int64_t width = kDdmDoppler / kDdmTx;
  const auto table = holoscan::radar::detail::build_ddm_demux_table(
      offsets, kDdmTx, kDdmDoppler, width);
  for (int64_t tx = 0; tx < kDdmTx; ++tx) {
    for (int64_t out_d = 0; out_d < width; ++out_d) {
      const int64_t source_d =
          table[static_cast<size_t>(tx * width + out_d)];
      for (int64_t rx = 0; rx < kDdmRx; ++rx) {
        for (int64_t range = 0; range < kDdmRange; ++range) {
          host[static_cast<size_t>(
              (rx * kDdmDoppler + source_d) * kDdmRange + range)] *=
              initial[static_cast<size_t>(tx)];
        }
      }
    }
  }
  const auto output = run_ddm_from_host<double>(host, shape, code);
  expect_ddm_recovered<double>(
      output, kDdmTx, kDdmRx, width, kDdmRange, p0, 1, 1e-12);
}

TEST(DDMDenseDemultiplexOp,
     DdmNonLinearPhaseCodesCorrelateRecovers) {
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
      run_ddm_from_host<float>(host, shape, code);
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

TEST(DDMDenseDemultiplexOp, DdmNonLinearMultiTxSeparatesPeaks) {
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
      run_ddm_from_host<float>(host, shape, code);
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

TEST(DDMDenseDemultiplexOp,
     DdmNonLinearRejectsNonZeroGuardBands) {
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
          ddm_params(code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp,
     DdmNonLinearRejectsChirpDopplerMismatch) {
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
          ddm_params(code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp, DdmGuardBandsReduceWidth) {
  const auto code = make_offset_code(kDdmTx, /*guard=*/1);
  const int64_t band = kDdmDoppler / kDdmTx - 1;
  const int64_t p0 = 0;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[2], band);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(DDMDenseDemultiplexOp, DdmGuardBandDiscardsEdgeBin) {
  const auto code = make_offset_code(kDdmTx, /*guard=*/1);
  const int64_t band = kDdmDoppler / kDdmTx - 1;
  const int64_t edge = kDdmDoppler / kDdmTx - 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, edge);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code);
  const auto* data = out.as<std::complex<float>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(data[i].real(), 0.0f, 1e-5f) << "i=" << i;
    EXPECT_NEAR(data[i].imag(), 0.0f, 1e-5f) << "i=" << i;
  }
}

TEST(DDMDenseDemultiplexOp, DdmComplexDoublePreserved) {
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 2;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<double>(host, shape, code);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<double>()));
  expect_ddm_recovered<double>(
      out, kDdmTx, kDdmRx, band, kDdmRange, p0, 1, 1e-9);
}

TEST(DDMDenseDemultiplexOp, DdmComplexFloatDtypePreserved) {
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<float>()));
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(DDMDenseDemultiplexOp, DdmFloatDoubleAgreement) {
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
      run_ddm_from_host<float>(host_f, shape, code);
  const auto out_d =
      run_ddm_from_host<double>(host_d, shape, code);
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

TEST(DDMDenseDemultiplexOp, DdmNonUniformOffsetPermutation) {
  const std::vector<double> permuted{0.5, 0.0, 0.75, 0.25};
  const auto code = make_offset_code_from(permuted);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 0;
  const auto host = build_ddm_input_host<float>(
      permuted, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(DDMDenseDemultiplexOp, DdmDeterministicAcrossRuns) {
  const auto code = make_offset_code(kDdmTx);
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto first = run_ddm_from_host<float>(host, shape, code);
  const auto second =
      run_ddm_from_host<float>(host, shape, code);
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

TEST(DDMDenseDemultiplexOp, RejectsEmptyDdmCode) {
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(DDMCodeDescriptor{}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp, DdmRejectsNonDivisibleDoppler) {
  const auto code = make_offset_code(3);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp, DdmRejectsGuardTooLarge) {
  const auto code = make_offset_code(kDdmTx, /*guard=*/4);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<float>())},
          ddm_params(code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp, DdmRejectsRankTooSmall) {
  const auto code = make_offset_code(2);
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, {kDdmDoppler, kDdmRange},
                            complex_dtype<float>())},
          ddm_params(code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp, DdmRejectsBothOffsetForms) {
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
          ddm_params(code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMDenseDemultiplexOp, DdmOutputEmittedOnDevice) {
  // [EA HSDK5 ONLY] Device output guaranteed by produces_tensor(kDevice).
  GTEST_SKIP() << "HSDK 5 Graph: device output guaranteed by "
                  "produces_tensor(kDevice); no in-flight capture";
}

TEST(DDMDenseDemultiplexOp, DdmAcceptsSingleEntryWithArbitraryKey) {
  // [EA HSDK5 ONLY] No TensorMap — typed Input<Tensor>.
  GTEST_SKIP() << "HSDK 5: typed Input<Tensor> replaces TensorMap; "
                  "no arbitrary-key semantics";
}

TEST(DDMDenseDemultiplexOp, DdmAcceptsPinnedHostInput) {
  GTEST_SKIP() << "HSDK 5 Graph: pinned host input not deliverable "
                  "through device-bound tensor loans";
}

TEST(DDMDenseDemultiplexOp, DdmAcceptsManagedInput) {
  GTEST_SKIP() << "HSDK 5 Graph: managed input not deliverable "
                  "through device-bound tensor loans";
}

TEST(DDMDenseDemultiplexOp, DdmRejectsMismatchedPrecision) {
  const auto code = make_offset_code(kDdmTx);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, 0);
  const GraphRunResult result =
      run_ddm_result<float>(
          {to_device_tensor(host, shape, complex_dtype<double>())},
          ddm_params(code));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

}  // namespace
