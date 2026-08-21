// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for FMCWRangeProcessorOp (HSDK 5 Graph harness):
// analytic point-target peak localization and range-axis validation,
// real-IF path, no-waveform (range_bins suppressed), software dechirp
// parity, window sidelobe suppression, nfft oversampling, batch axes,
// error paths, and float/double agreement.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
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

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/bin_model.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/sim_reference.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::complex_dtype;
using holoscan::radar::FftNorm;
using holoscan::radar::FMCWRangeProcessorOp;
using holoscan::radar::FMCWRangeProcessorParams;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::FMCWWaveformSpec;
using holoscan::radar::real_dtype;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::testing::complex_cube_to_device;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::expected_range_axis;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::range_of_bin;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;
namespace sim = holoscan::radar::sim;

constexpr int64_t kChirps = 32;
constexpr int64_t kSamples = 128;

double bin_width(int64_t nfft) {
  const sim::SimWaveform waveform{};
  return range_of_bin(1, waveform.slope(), waveform.sample_rate, nfft,
                      sim::kSimSpeedOfLight);
}

double range_at_bin(double bin) {
  return bin * bin_width(kSamples);
}

FMCWWaveformDescriptor to_descriptor(const sim::SimWaveform& waveform) {
  FMCWWaveformSpec spec;
  spec.carrier_frequency = waveform.carrier_frequency;
  spec.bandwidth = waveform.bandwidth;
  spec.chirp_duration = waveform.chirp_duration;
  spec.sample_rate = waveform.sample_rate;
  spec.chirp_repetition_interval =
      waveform.chirp_repetition_interval;
  return FMCWWaveformDescriptor(spec);
}

sim::SimScene make_scene(double range_m, bool dechirped = true) {
  sim::SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kSamples;
  scene.dechirped = dechirped;
  sim::PointTarget target;
  target.position = sim::Vec3{range_m, 0.0, 0.0};
  scene.targets.push_back(target);
  return scene;
}

template <typename T>
std::vector<double> slice_abs(const HostTensor& tensor, int64_t batch,
                              int64_t chirp) {
  const int64_t bins = tensor.shape.back();
  const int64_t chirps = tensor.shape[tensor.shape.size() - 2];
  const auto* data = tensor.as<std::complex<T>>();
  std::vector<double> magnitude(static_cast<size_t>(bins));
  const int64_t offset = (batch * chirps + chirp) * bins;
  for (int64_t i = 0; i < bins; ++i) {
    magnitude[static_cast<size_t>(i)] =
        std::abs(std::complex<double>(data[offset + i]));
  }
  return magnitude;
}

int64_t argmax(const std::vector<double>& values) {
  return static_cast<int64_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

// Wire function for the 1-in / 2-out Graph harness.
auto wire_range_op() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink0, const auto& sink1) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->adc_cube, q);
    graph.add_flow(op->range_profiles, sink0->in, q);
    graph.add_flow(op->range_bins, sink1->in, q);
  };
}

// Wire function for 1-in / 1-out (profiles only, no bins sink).
auto wire_range_op_profiles_only() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->adc_cube, q);
    graph.add_flow(op->range_profiles, sink->in, q);
  };
}

template <typename T>
DualIOGraphRunResult run_complex_frame(
    const sim::SimResult& result, FMCWRangeProcessorParams params) {
  auto input = complex_cube_to_device<T>(result.cube, result.shape);
  return run_single_in_dual_out_operator_graph_result<
      FMCWRangeProcessorOp<T>>(
      {std::move(input)}, std::move(params),
      "range_profiles", "range_bins", wire_range_op());
}

// Single-IO graph for profiles-only tests. Wires only range_profiles but
// binds range_bins on device (IMPLICIT_DEVICE_ZERO_REJECTED requires all
// produces_tensor outputs to be explicitly bound).
template <typename T>
GraphRunResult run_profiles_only_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs,
    FMCWRangeProcessorParams params) {
  return run_single_io_operator_graph_result<FMCWRangeProcessorOp<T>>(
      std::move(inputs), std::move(params), "range_profiles",
      wire_range_op_profiles_only(),
      {"range_bins"});
}

TEST(FMCWRangeProcessorOpTest, SingleTargetComplexIfPeakAndRangeAxis) {
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);
  const auto result = sim::simulate(make_scene(target_range));

  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult out =
      run_complex_frame<float>(result, std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames0.size(), 1u);
  ASSERT_EQ(out.frames1.size(), 1u);

  const HostTensor& profiles = out.frames0[0];
  const HostTensor& bins = out.frames1[0];
  EXPECT_EQ(profiles.shape, (std::vector<int64_t>{kChirps, kSamples}));
  ASSERT_EQ(bins.shape, (std::vector<int64_t>{kSamples}));
  const float* axis = bins.as<float>();

  const auto magnitude = slice_abs<float>(profiles, 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(axis[peak], target_range, bin_width(kSamples));

  const auto expected =
      expected_range_axis(kSamples, waveform.slope(), waveform.sample_rate,
                          kSamples, sim::kSimSpeedOfLight);
  for (int64_t i = 0; i < kSamples; ++i) {
    EXPECT_NEAR(axis[i], expected[static_cast<size_t>(i)],
                1e-3 + 1e-5 * std::abs(expected[static_cast<size_t>(i)]))
        << "bin " << i;
  }
}

TEST(FMCWRangeProcessorOpTest, RealIfPathPeakAtRange) {
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);
  const auto result = sim::simulate(make_scene(target_range));
  const auto real_cube = sim::to_real_if(result);

  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  auto input = real_cube_to_device<float>(real_cube, result.shape);
  const DualIOGraphRunResult out =
      run_single_in_dual_out_operator_graph_result<
          FMCWRangeProcessorOp<float>>(
          {std::move(input)}, std::move(params),
          "range_profiles", "range_bins", wire_range_op());

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames0.size(), 1u);
  ASSERT_EQ(out.frames1.size(), 1u);

  const HostTensor& profiles = out.frames0[0];
  const HostTensor& bins = out.frames1[0];
  EXPECT_EQ(profiles.shape,
            (std::vector<int64_t>{kChirps, kSamples / 2 + 1}));
  ASSERT_EQ(bins.shape, (std::vector<int64_t>{kSamples / 2 + 1}));

  const auto magnitude = slice_abs<float>(profiles, 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(bins.as<float>()[peak], target_range,
              bin_width(kSamples));
}

TEST(FMCWRangeProcessorOpTest, NoRangeBinsWithoutWaveform) {
  const auto result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto input = complex_cube_to_device<float>(result.cube, result.shape);

  // No waveform → default-constructed is_valid() == false → range_bins
  // not emitted. Use single-IO harness (profiles only); range_bins port
  // is declared but unwired.
  FMCWRangeProcessorParams params;
  const GraphRunResult out =
      run_profiles_only_graph_result<float>({std::move(input)},
                                            std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 1u);
  EXPECT_EQ(out.frames[0].shape,
            (std::vector<int64_t>{kChirps, kSamples}));
}

TEST(FMCWRangeProcessorOpTest, ApplyDechirpMatchesHardwareDechirp) {
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.5);

  const auto hw_sim = sim::simulate(make_scene(target_range, true));
  const auto pre_mixer_sim =
      sim::simulate(make_scene(target_range, false));

  FMCWRangeProcessorParams hw_params;
  hw_params.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult hw =
      run_complex_frame<float>(hw_sim, std::move(hw_params));

  FMCWRangeProcessorParams sw_params;
  sw_params.waveform = to_descriptor(waveform);
  sw_params.apply_dechirp = true;
  const DualIOGraphRunResult sw =
      run_complex_frame<float>(pre_mixer_sim, std::move(sw_params));

  ASSERT_TRUE(hw.plan_ok);
  ASSERT_TRUE(sw.plan_ok);
  ASSERT_EQ(hw.frames0.size(), 1u);
  ASSERT_EQ(sw.frames0.size(), 1u);
  ASSERT_EQ(hw.frames0[0].shape, sw.frames0[0].shape);

  const auto* a = hw.frames0[0].as<std::complex<float>>();
  const auto* b = sw.frames0[0].as<std::complex<float>>();
  double peak_mag = 0.0;
  for (int64_t i = 0; i < hw.frames0[0].size(); ++i) {
    peak_mag = std::max(peak_mag,
                        std::abs(std::complex<double>(a[i])));
  }
  ASSERT_GT(peak_mag, 0.0);
  for (int64_t i = 0; i < hw.frames0[0].size(); ++i) {
    const double diff =
        std::abs(std::complex<double>(a[i]) - std::complex<double>(b[i]));
    ASSERT_LE(diff, 2e-3 * peak_mag) << "element " << i;
  }
}

TEST(FMCWRangeProcessorOpTest, ApplyDechirpWithoutWaveformErrors) {
  const auto result =
      sim::simulate(make_scene(range_at_bin(30.0), false));
  auto input =
      complex_cube_to_device<float>(result.cube, result.shape);

  FMCWRangeProcessorParams params;
  params.apply_dechirp = true;
  // No waveform set → should error.
  const GraphRunResult out =
      run_profiles_only_graph_result<float>({std::move(input)},
                                            std::move(params));

  EXPECT_TRUE(out.plan_ok);
  EXPECT_TRUE(out.frames.empty());
  EXPECT_FALSE(out.diagnostics.empty());
}

TEST(FMCWRangeProcessorOpTest, ApplyDechirpWithRealInputErrors) {
  const sim::SimWaveform waveform{};
  const auto result = sim::simulate(make_scene(range_at_bin(30.0)));
  const auto real_cube = sim::to_real_if(result);
  auto input = real_cube_to_device<float>(real_cube, result.shape);

  FMCWRangeProcessorParams params;
  params.apply_dechirp = true;
  params.waveform = to_descriptor(waveform);
  const GraphRunResult out =
      run_profiles_only_graph_result<float>({std::move(input)},
                                            std::move(params));

  EXPECT_TRUE(out.plan_ok);
  EXPECT_TRUE(out.frames.empty());
  EXPECT_FALSE(out.diagnostics.empty());
}

TEST(FMCWRangeProcessorOpTest, HannWindowLowersSidelobesKeepsPeak) {
  const sim::SimWaveform waveform{};
  WindowSpec hann;
  hann.type = WindowType::kHann;

  // Off-bin target: rectangular sidelobes are fully exposed.
  const auto off_bin_sim = sim::simulate(make_scene(range_at_bin(30.5)));

  FMCWRangeProcessorParams rect_params;
  rect_params.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult rect =
      run_complex_frame<float>(off_bin_sim, std::move(rect_params));

  FMCWRangeProcessorParams hann_params;
  hann_params.waveform = to_descriptor(waveform);
  hann_params.range_window = hann;
  const DualIOGraphRunResult hann_run =
      run_complex_frame<float>(off_bin_sim, std::move(hann_params));

  ASSERT_TRUE(rect.plan_ok);
  ASSERT_TRUE(hann_run.plan_ok);
  ASSERT_EQ(rect.frames0.size(), 1u);
  ASSERT_EQ(hann_run.frames0.size(), 1u);

  const auto rect_mag = slice_abs<float>(rect.frames0[0], 0, 0);
  const auto hann_mag = slice_abs<float>(hann_run.frames0[0], 0, 0);

  const int64_t peak = argmax(rect_mag);
  EXPECT_NEAR(static_cast<double>(argmax(hann_mag)),
              static_cast<double>(peak), 1.0);

  double rect_sidelobe = 0.0;
  double hann_sidelobe = 0.0;
  for (int64_t i = peak + 4; i < std::min<int64_t>(peak + 16, kSamples);
       ++i) {
    rect_sidelobe = std::max(rect_sidelobe,
                             rect_mag[static_cast<size_t>(i)]);
    hann_sidelobe = std::max(hann_sidelobe,
                             hann_mag[static_cast<size_t>(i)]);
  }
  EXPECT_LT(hann_sidelobe, 0.5 * rect_sidelobe);

  // On-bin target: coherent-gain normalization preserves the peak
  // magnitude (within a few %) under fft_norm = backward.
  const auto on_bin_sim = sim::simulate(make_scene(range_at_bin(30.0)));

  FMCWRangeProcessorParams rect_on_params;
  rect_on_params.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult rect_on =
      run_complex_frame<float>(on_bin_sim, std::move(rect_on_params));

  FMCWRangeProcessorParams hann_on_params;
  hann_on_params.waveform = to_descriptor(waveform);
  hann_on_params.range_window = hann;
  const DualIOGraphRunResult hann_on =
      run_complex_frame<float>(on_bin_sim, std::move(hann_on_params));

  ASSERT_TRUE(rect_on.plan_ok);
  ASSERT_TRUE(hann_on.plan_ok);
  const auto rect_on_mag = slice_abs<float>(rect_on.frames0[0], 0, 0);
  const auto hann_on_mag = slice_abs<float>(hann_on.frames0[0], 0, 0);
  const double rect_peak = rect_on_mag[static_cast<size_t>(30)];
  const double hann_peak = hann_on_mag[static_cast<size_t>(30)];
  EXPECT_NEAR(hann_peak, rect_peak, 0.03 * rect_peak);
}

TEST(FMCWRangeProcessorOpTest, WindowConfigurableFromParams) {
  // Verify Params-authored WindowSpec is equivalent to the 4.x Arg path.
  const sim::SimWaveform waveform{};
  WindowSpec hann;
  hann.type = WindowType::kHann;

  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.5)));
  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  params.range_window = hann;
  const DualIOGraphRunResult out =
      run_complex_frame<float>(sim_result, std::move(params));
  ASSERT_TRUE(out.plan_ok);
  ASSERT_EQ(out.frames0.size(), 1u);
}

TEST(FMCWRangeProcessorOpTest, WaveformConfigurableFromParams) {
  const sim::SimWaveform waveform{};
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult out =
      run_complex_frame<float>(sim_result, std::move(params));
  ASSERT_TRUE(out.plan_ok);
  ASSERT_EQ(out.frames0.size(), 1u);
  ASSERT_EQ(out.frames1.size(), 1u);
}

TEST(FMCWRangeProcessorOpTest, NfftOversamplingInterpolatesGrid) {
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);
  const auto sim_result = sim::simulate(make_scene(target_range));
  const int64_t nfft = 2 * kSamples;

  FMCWRangeProcessorParams base_params;
  base_params.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult base =
      run_complex_frame<float>(sim_result, std::move(base_params));

  FMCWRangeProcessorParams over_params;
  over_params.waveform = to_descriptor(waveform);
  over_params.nfft = nfft;
  const DualIOGraphRunResult oversampled =
      run_complex_frame<float>(sim_result, std::move(over_params));

  ASSERT_TRUE(base.plan_ok);
  ASSERT_TRUE(oversampled.plan_ok);
  ASSERT_EQ(oversampled.frames0.size(), 1u);
  ASSERT_EQ(oversampled.frames1.size(), 1u);

  EXPECT_EQ(oversampled.frames0[0].shape,
            (std::vector<int64_t>{kChirps, nfft}));
  ASSERT_EQ(oversampled.frames1[0].shape, (std::vector<int64_t>{nfft}));
  const float* axis = oversampled.frames1[0].as<float>();

  const auto magnitude = slice_abs<float>(oversampled.frames0[0], 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(axis[peak], target_range, bin_width(kSamples));

  ASSERT_EQ(base.frames1.size(), 1u);
  const float* base_axis = base.frames1[0].as<float>();
  EXPECT_NEAR(axis[1], 0.5 * base_axis[1], 1e-4 * base_axis[1]);
}

TEST(FMCWRangeProcessorOpTest, NamedRxAxisRoundTrip) {
  // Rank-3 leading dim is the named N_rx axis (SIMO), not opaque batch.
  // default_test_extents() sets max_rx so the leading axis is admitted.
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);

  sim::SimScene scene = make_scene(target_range);
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.0, 0.0019, 0.0}};
  const auto sim_result = sim::simulate(scene);
  ASSERT_EQ(sim_result.shape,
            (std::vector<int64_t>{2, kChirps, kSamples}));

  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult out =
      run_complex_frame<float>(sim_result, std::move(params));

  ASSERT_TRUE(out.plan_ok);
  ASSERT_EQ(out.frames0.size(), 1u);
  EXPECT_EQ(out.frames0[0].shape,
            (std::vector<int64_t>{2, kChirps, kSamples}));

  for (int64_t chirp = 0; chirp < kChirps; ++chirp) {
    const auto rx0 = slice_abs<float>(out.frames0[0], 0, chirp);
    const auto rx1 = slice_abs<float>(out.frames0[0], 1, chirp);
    for (int64_t i = 0; i < kSamples; ++i) {
      EXPECT_NEAR(rx0[static_cast<size_t>(i)],
                  rx1[static_cast<size_t>(i)], 1e-3)
          << "chirp " << chirp << " bin " << i;
    }
  }
}

struct LiveNfftTick {
  std::shared_ptr<holoscan::Tensor> cube;
  std::optional<int64_t> nfft;
  int64_t nfft_skew_ns{0};
};

struct LiveNfftBank {
  static void reset() {
    ticks.clear();
    index = 0;
  }
  static inline std::vector<LiveNfftTick> ticks;
  static inline std::size_t index{0};
};

holoscan::expected<void, holoscan::Error> emit_cube_at(
    holoscan::Output<holoscan::Tensor>& out,
    holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame,
    holoscan::TimePoint capture_time) {
  const auto shape = frame->shape();
  const DLDataType dtype = frame->dtype();
  auto loan = out.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (writer->byte_size() > 0 &&
      cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(),
                      cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        "LiveNfftSourceOp: cudaMemcpyAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(std::move(*loan),
                  holoscan::EmitOptions{.capture_time = capture_time});
}

class LiveNfftSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(cube, "out")
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 8U, .max_byte_span = 256ULL * 1024ULL * 1024ULL});
    spec.output(nfft, "nfft");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    LiveNfftTick tick;
    {
      if (LiveNfftBank::index >= LiveNfftBank::ticks.size()) {
        return {};
      }
      tick = LiveNfftBank::ticks[LiveNfftBank::index++];
    }
    // Publish the override before the cube so Exact can see it.
    const holoscan::TimePoint cube_time = context.activation_time();
    if (tick.nfft.has_value()) {
      holoscan::TimePoint nfft_time = cube_time;
      nfft_time.timestamp_ns += tick.nfft_skew_ns;
      auto emitted = nfft.emit(
          *tick.nfft, holoscan::EmitOptions{.capture_time = nfft_time});
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    return emit_cube_at(cube, context, tick.cube, cube_time);
  }

  holoscan::Output<holoscan::Tensor> cube{"out"};
  holoscan::Output<int64_t> nfft{"nfft"};
};

GraphRunResult run_nfft_override_graph(
    std::vector<LiveNfftTick> ticks, FMCWRangeProcessorParams params) {
  GraphRunResult result;
  holoscan::radar::testing::detail::FrameBank::reset();
  LiveNfftBank::reset();
  LiveNfftBank::ticks = std::move(ticks);
  const std::size_t expected = LiveNfftBank::ticks.size();

  holoscan::Graph graph{"nfft-live-override"};
  const auto source = graph.op<LiveNfftSourceOp>("source");
  auto op = holoscan::radar::testing::emplace_harness_op<
      FMCWRangeProcessorOp<float>>(
      graph, holoscan::radar::testing::default_test_extents());
  op.set_params(std::move(params));
  const auto sink =
      graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("sink");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->cube, op->adc_cube, q);
  graph.add_flow(source->nfft, op->nfft_in, q);
  graph.add_flow(op->range_profiles, sink->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  holoscan::radar::testing::detail::bind_single_io_device_pools(
      options, "range_profiles");
  holoscan::radar::testing::detail::bind_op_device_output(
      options, "range_bins");

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
    {
      std::unique_lock lock(
          holoscan::radar::testing::detail::FrameBank::mutex);
      while (holoscan::radar::testing::detail::FrameBank::captured.size() <
                 expected &&
             session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        holoscan::radar::testing::detail::FrameBank::ready.wait_for(
            lock, holoscan::radar::testing::detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(
        holoscan::radar::testing::detail::FrameBank::mutex);
    result.frames =
        std::move(holoscan::radar::testing::detail::FrameBank::captured);
  }
  return result;
}

std::vector<LiveNfftTick> two_cube_ticks(
    const std::shared_ptr<holoscan::Tensor>& cube,
    std::optional<int64_t> nfft0, std::optional<int64_t> nfft1,
    int64_t nfft0_skew_ns = 0, int64_t nfft1_skew_ns = 0) {
  return {{cube, nfft0, nfft0_skew_ns}, {cube, nfft1, nfft1_skew_ns}};
}

TEST(FMCWRangeProcessorOpTest, LiveNfftExactMatchChangesBinCount) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeProcessorParams params;
  params.expect_nfft = true;
  params.sticky_nfft = false;
  const GraphRunResult out = run_nfft_override_graph(
      two_cube_ticks(cube, int64_t{256}, int64_t{256}), std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2u);
  EXPECT_EQ(out.frames[0].shape.back(), 256);
  EXPECT_EQ(out.frames[1].shape.back(), 256);
}

TEST(FMCWRangeProcessorOpTest, LiveNfftStickyKeepsLastMatch) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeProcessorParams params;
  params.expect_nfft = true;
  params.sticky_nfft = true;
  const GraphRunResult out = run_nfft_override_graph(
      two_cube_ticks(cube, int64_t{256}, std::nullopt), std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2u);
  EXPECT_EQ(out.frames[0].shape.back(), 256);
  EXPECT_EQ(out.frames[1].shape.back(), 256);
}

TEST(FMCWRangeProcessorOpTest, LiveNfftEphemeralRevertsToParams) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeProcessorParams params;
  params.expect_nfft = true;
  params.sticky_nfft = false;
  const GraphRunResult out = run_nfft_override_graph(
      two_cube_ticks(cube, int64_t{256}, std::nullopt), std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2u);
  EXPECT_EQ(out.frames[0].shape.back(), 256);
  EXPECT_EQ(out.frames[1].shape.back(), kSamples);
}

TEST(FMCWRangeProcessorOpTest, LiveNfftSkewedCaptureTimeMissesExact) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
  const int64_t skew_ns = 1'000'000;

  FMCWRangeProcessorParams params;
  params.expect_nfft = true;
  params.sticky_nfft = false;
  const GraphRunResult out = run_nfft_override_graph(
      two_cube_ticks(cube, int64_t{256}, int64_t{256}, skew_ns, skew_ns),
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2u);
  EXPECT_EQ(out.frames[0].shape.back(), kSamples);
  EXPECT_EQ(out.frames[1].shape.back(), kSamples);
}

TEST(FMCWRangeProcessorOpTest, LiveWindowOverrideAppliesPerTick) {
  // [EA HSDK5 ONLY] See docs/planning/HSDK5_PORT.md §1/§9 — Graph
  // mid-stream sticky override harness fails on EA device/partition
  // binding; WindowSpec is Params-only (owns vectors). Params authoring
  // covers window selection; restore once §1 + §9 are closed.
  GTEST_SKIP() << "HSDK 5 EA: live window override harness deferred "
                  "(HSDK5_PORT.md §1/§9)";
}

TEST(FMCWRangeProcessorOpTest, FloatAndDoubleInstantiationsAgree) {
  const sim::SimWaveform waveform{};
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));

  FMCWRangeProcessorParams params_f;
  params_f.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult run_f =
      run_complex_frame<float>(sim_result, std::move(params_f));

  FMCWRangeProcessorParams params_d;
  params_d.waveform = to_descriptor(waveform);
  const DualIOGraphRunResult run_d =
      run_complex_frame<double>(sim_result, std::move(params_d));

  ASSERT_TRUE(run_f.plan_ok);
  ASSERT_TRUE(run_d.plan_ok);
  ASSERT_EQ(run_f.frames0.size(), 1u);
  ASSERT_EQ(run_d.frames0.size(), 1u);

  const auto mag_f = slice_abs<float>(run_f.frames0[0], 0, 0);
  const auto mag_d = slice_abs<double>(run_d.frames0[0], 0, 0);

  const double peak = mag_d[static_cast<size_t>(argmax(mag_d))];
  EXPECT_EQ(argmax(mag_f), argmax(mag_d));
  for (int64_t i = 0; i < kSamples; ++i) {
    EXPECT_NEAR(mag_f[static_cast<size_t>(i)],
                mag_d[static_cast<size_t>(i)], 1e-4 * peak)
        << "bin " << i;
  }
}

}  // namespace
