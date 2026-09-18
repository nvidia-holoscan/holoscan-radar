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
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
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
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
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
  auto scene = make_scene(target_range);
  scene.analog_if.sampling_mode = sim::IfSamplingMode::kReal;
  const auto result = sim::simulate(scene);
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

// Supplying a waveform makes range_bins computable, but only wiring makes it
// wanted, and the operator cannot tell the two apart — nothing exposes an
// output's wiring to contract() or compute(). So it emits, and the first
// failure warns and latches the port off. Propagating that failure instead
// would fail the activation and take range_profiles down with it, which is
// what this pins: an undrained range_bins costs one warning, not the run.
// NoRangeBinsWithoutWaveform is the companion where no waveform means nothing
// is emitted at all. See OPERATOR_API.md, "Optional output convention".
TEST(FMCWRangeProcessorOpTest, UndrainedRangeBinsDoesNotFailTheRun) {
  const sim::SimWaveform waveform{};
  const auto result = sim::simulate(make_scene(range_at_bin(30.0)));

  // Many ticks, not one: the failed emit still takes a pool loan, so a slot
  // the runtime did not recycle would starve the operator some frames in
  // rather than on the first. It would surface here as a diagnostic, not a
  // silent drop — prepare_device_tensor's kResourceExhausted is propagated.
  // Any count above the pool depth suffices (that pool holds one loan); this
  // is far above it and still sub-second. Verified separately at 2000.
  constexpr std::size_t kFrames = 128;
  std::vector<std::shared_ptr<holoscan::Tensor>> inputs;
  inputs.reserve(kFrames);
  for (std::size_t i = 0; i < kFrames; ++i) {
    inputs.push_back(complex_cube_to_device<float>(result.cube, result.shape));
  }

  // Waveform present → the operator tries to emit range_bins, which the
  // profiles-only harness declares but leaves unwired.
  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  const GraphRunResult out =
      run_profiles_only_graph_result<float>(std::move(inputs),
                                            std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_TRUE(out.diagnostics.empty());
  // Every tick's sibling output is delivered, start to finish.
  ASSERT_EQ(out.frames.size(), kFrames);
  for (const HostTensor& frame : out.frames) {
    EXPECT_EQ(frame.shape, (std::vector<int64_t>{kChirps, kSamples}));
  }
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
  auto scene = make_scene(range_at_bin(30.0));
  scene.analog_if.sampling_mode = sim::IfSamplingMode::kReal;
  const auto result = sim::simulate(scene);
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
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(cube, "out")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, 256ULL * 1024ULL * 1024ULL));
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
  const GraphRunResult out = run_nfft_override_graph(
      two_cube_ticks(cube, int64_t{256}, int64_t{256}), std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2u);
  EXPECT_EQ(out.frames[0].shape.back(), 256);
  EXPECT_EQ(out.frames[1].shape.back(), 256);
}

TEST(FMCWRangeProcessorOpTest, LiveNfftRevertsToParamsAfterMiss) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeProcessorParams params;
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
  const GraphRunResult out = run_nfft_override_graph(
      two_cube_ticks(cube, int64_t{256}, int64_t{256}, skew_ns, skew_ns),
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2u);
  EXPECT_EQ(out.frames[0].shape.back(), kSamples);
  EXPECT_EQ(out.frames[1].shape.back(), kSamples);
}

struct LiveRangeDescriptorTick {
  std::shared_ptr<holoscan::Tensor> cube;
  std::optional<WindowSpec> range_window;
  std::optional<FMCWWaveformDescriptor> waveform;
  bool invalid_waveform{false};
};

struct LiveRangeDescriptorBank {
  static void reset() {
    ticks.clear();
    index = 0;
  }
  static inline std::vector<LiveRangeDescriptorTick> ticks;
  static inline std::size_t index{0};
};

template <typename Wire, typename Element>
holoscan::expected<void, holoscan::Error> emit_descriptor_tensor(
    holoscan::Output<Wire>& output, Wire wire,
    std::span<const Element> values, holoscan::TimePoint capture_time) {
  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(values.size())};
  constexpr DLDataType dtype{
      std::is_same_v<Element, double> ? kDLFloat : kDLComplex,
      static_cast<uint8_t>(sizeof(Element) * 8), 1U};
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit(
      [values](const holoscan::TensorOutputWriteGuard& writer) {
        std::copy(values.begin(), values.end(),
                  writer.template data_as<Element>());
      });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return output.emit_tensor(
      std::move(*loan), wire,
      holoscan::EmitOptions{.capture_time = capture_time});
}

class LiveRangeDescriptorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    constexpr holoscan::TensorRepresentation kWindow{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 1U,
    };
    constexpr holoscan::TensorRepresentation kWaveform{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLComplex, 128U, 1U},
        .rank = 1U,
    };
    spec.output(cube, "cube")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, 256ULL * 1024ULL * 1024ULL));
    spec.output(range_window, "range_window")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kWindow, 1U, kSamples * sizeof(double)));
    spec.output(waveform, "waveform")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kWaveform, 1U, kSamples * sizeof(std::complex<double>)));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (LiveRangeDescriptorBank::index >=
        LiveRangeDescriptorBank::ticks.size()) {
      return {};
    }
    const LiveRangeDescriptorTick& tick =
        LiveRangeDescriptorBank::ticks[LiveRangeDescriptorBank::index++];
    const holoscan::TimePoint capture_time = context.activation_time();
    if (tick.range_window.has_value()) {
      auto wire = holoscan::radar::to_wire(*tick.range_window);
      auto emitted = emit_descriptor_tensor(
          range_window, std::move(wire),
          std::span<const double>{tick.range_window->coefficients},
          capture_time);
      if (!emitted) {
        return emitted;
      }
    }
    if (tick.waveform.has_value()) {
      auto wire = holoscan::radar::to_wire(*tick.waveform);
      if (tick.invalid_waveform) {
        wire.chirp_duration = 0.0;
      }
      auto emitted = emit_descriptor_tensor(
          waveform, std::move(wire),
          std::span<const std::complex<double>>{
              tick.waveform->reference_samples()},
          capture_time);
      if (!emitted) {
        return emitted;
      }
    }
    return emit_cube_at(cube, context, tick.cube, capture_time);
  }

  holoscan::Output<holoscan::Tensor> cube{"cube"};
  holoscan::Output<holoscan::radar::wire::WindowSpecT>
      range_window{"range_window"};
  holoscan::Output<holoscan::radar::wire::FMCWWaveformT>
      waveform{"waveform"};
};

DualIOGraphRunResult run_range_descriptor_override_graph(
    std::vector<LiveRangeDescriptorTick> ticks,
    FMCWRangeProcessorParams params) {
  DualIOGraphRunResult result;
  holoscan::radar::testing::detail::FrameBank::reset();
  holoscan::radar::testing::detail::FrameBankB::reset();
  LiveRangeDescriptorBank::reset();
  LiveRangeDescriptorBank::ticks = std::move(ticks);
  const std::size_t expected = LiveRangeDescriptorBank::ticks.size();

  holoscan::Graph graph{"range-descriptor-overrides"};
  const auto source = graph.op<LiveRangeDescriptorSourceOp>("source");
  auto op = holoscan::radar::testing::emplace_harness_op<
      FMCWRangeProcessorOp<float>>(
      graph, holoscan::radar::testing::default_test_extents());
  op.set_params(std::move(params));
  const auto sink0 =
      graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("sink0");
  const auto sink1 =
      graph.op<holoscan::radar::testing::detail::TensorCaptureBOp>("sink1");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->range_window, op->range_window_in, q);
  graph.add_flow(source->waveform, op->waveform_in, q);
  graph.add_flow(source->cube, op->adc_cube, q);
  graph.add_flow(op->range_profiles, sink0->in, q);
  graph.add_flow(op->range_bins, sink1->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "cube",
          .device = holoscan::DeviceId{0}});
  holoscan::radar::testing::detail::bind_op_device_output(
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
    while (session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      std::size_t count0 = 0;
      std::size_t count1 = 0;
      {
        const std::lock_guard lock(
            holoscan::radar::testing::detail::FrameBank::mutex);
        count0 =
            holoscan::radar::testing::detail::FrameBank::captured.size();
      }
      {
        const std::lock_guard lock(
            holoscan::radar::testing::detail::FrameBankB::mutex);
        count1 =
            holoscan::radar::testing::detail::FrameBankB::captured.size();
      }
      if (count0 >= expected && count1 >= expected) {
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
    result.frames0 = std::move(
        holoscan::radar::testing::detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(
        holoscan::radar::testing::detail::FrameBankB::mutex);
    result.frames1 = std::move(
        holoscan::radar::testing::detail::FrameBankB::captured);
  }
  return result;
}

TEST(FMCWRangeProcessorOpTest, LiveWindowOverrideAppliesPerTick) {
  const sim::SimWaveform base_waveform{};
  sim::SimWaveform narrow_waveform = base_waveform;
  narrow_waveform.bandwidth *= 0.5;
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.5)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  WindowSpec hann;
  hann.type = WindowType::kHann;
  LiveRangeDescriptorTick base{cube, std::nullopt, std::nullopt};
  LiveRangeDescriptorTick changed{
      cube, hann, to_descriptor(narrow_waveform)};

  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(base_waveform);
  const DualIOGraphRunResult out = run_range_descriptor_override_graph(
      {base, changed, base}, std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames0.size(), 3u);
  ASSERT_EQ(out.frames1.size(), 3u);

  const auto rect = slice_abs<float>(out.frames0[0], 0, 0);
  const auto windowed = slice_abs<float>(out.frames0[1], 0, 0);
  const auto reverted = slice_abs<float>(out.frames0[2], 0, 0);
  const int64_t peak = argmax(rect);
  double rect_sidelobe = 0.0;
  double windowed_sidelobe = 0.0;
  for (int64_t i = peak + 4; i < std::min<int64_t>(peak + 16, kSamples);
       ++i) {
    rect_sidelobe =
        std::max(rect_sidelobe, rect[static_cast<size_t>(i)]);
    windowed_sidelobe =
        std::max(windowed_sidelobe, windowed[static_cast<size_t>(i)]);
  }
  EXPECT_LT(windowed_sidelobe, 0.5 * rect_sidelobe);
  // Tick 2 publishes no window, so the Params (rectangular) window is back.
  EXPECT_EQ(rect, reverted);

  const float base_spacing = out.frames1[0].as<float>()[1];
  const float changed_spacing = out.frames1[1].as<float>()[1];
  EXPECT_NEAR(changed_spacing, 2.0F * base_spacing,
              1e-4F * changed_spacing);
  // Likewise for the waveform: the range axis returns to the Params spacing.
  EXPECT_EQ(out.frames1[0].as<float>()[1],
            out.frames1[2].as<float>()[1]);
}

TEST(FMCWRangeProcessorOpTest, InvalidLiveWaveformRejectedBeforeDsp) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
  LiveRangeDescriptorTick invalid{
      cube, std::nullopt, to_descriptor(sim::SimWaveform{})};
  invalid.invalid_waveform = true;

  FMCWRangeProcessorParams params;
  const DualIOGraphRunResult out =
      run_range_descriptor_override_graph({invalid}, std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_TRUE(out.frames0.empty());
  EXPECT_TRUE(out.frames1.empty());
  EXPECT_FALSE(out.diagnostics.empty());
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
