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
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
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
#include "utils/sim_reference.hpp"
#include "utils/test_extents.hpp"
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
using holoscan::radar::testing::expected_range_axis;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::range_of_bin;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::to_device_tensor;
namespace sim = holoscan::radar::sim;

constexpr int64_t kChirps = 32;
constexpr int64_t kSamples = 128;

double bin_width(int64_t nfft) {
  const sim::SimWaveform waveform{};
  return range_of_bin(1, waveform.slope(), waveform.sample_rate, nfft, sim::kSimSpeedOfLight);
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
  spec.chirp_repetition_interval = waveform.chirp_repetition_interval;
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
std::vector<double> slice_abs(const HostTensor& tensor, int64_t batch, int64_t chirp) {
  const int64_t bins = tensor.shape.back();
  const int64_t chirps = tensor.shape[tensor.shape.size() - 2];
  const auto* data = tensor.as<std::complex<T>>();
  std::vector<double> magnitude(static_cast<size_t>(bins));
  const int64_t offset = (batch * chirps + chirp) * bins;
  for (int64_t i = 0; i < bins; ++i) {
    magnitude[static_cast<size_t>(i)] = std::abs(std::complex<double>(data[offset + i]));
  }
  return magnitude;
}

int64_t argmax(const std::vector<double>& values) {
  return static_cast<int64_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

template <typename T>
holoscan::testing::TestResult run_range_frames(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs, FMCWRangeProcessorParams params,
    bool capture_bins = true) {
  using Op = FMCWRangeProcessorOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  for (auto& input : inputs) {
    fixture.push(&Op::adc_cube, *input);
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("range_profiles", inputs.size(),
                                                      kMaximumBytes);
  if (capture_bins) {
    fixture.template capture_snapshot<holoscan::Tensor>("range_bins", inputs.size(), kMaximumBytes);
  }
  fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
  fixture.output_device("range_profiles", holoscan::DeviceId{0});
  fixture.output_device("range_bins", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "range_profiles", inputs.size());
  if (capture_bins) {
    (void)completion.wait_for_output(running, "range_bins", inputs.size());
  }
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

template <typename T>
holoscan::testing::TestResult run_complex_frame(const sim::SimResult& result,
                                                FMCWRangeProcessorParams params) {
  auto input = complex_cube_to_device<T>(result.cube, result.shape);
  return run_range_frames<T>({std::move(input)}, std::move(params));
}

// Single-IO graph for profiles-only tests. Wires only range_profiles but
// binds range_bins on device (IMPLICIT_DEVICE_ZERO_REJECTED requires all
// produces_tensor outputs to be explicitly bound).
template <typename T>
holoscan::testing::TestResult run_profiles_only_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs, FMCWRangeProcessorParams params) {
  return run_range_frames<T>(std::move(inputs), std::move(params), false);
}

std::vector<HostTensor> range_frames(const holoscan::testing::TestResult& result,
                                     const char* port) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.template snapshots<holoscan::Tensor>(port)) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

TEST(FMCWRangeProcessorOpTest, SingleTargetComplexIfPeakAndRangeAxis) {
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);
  const auto result = sim::simulate(make_scene(target_range));

  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  const auto out = run_complex_frame<float>(result, std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 1u);
  ASSERT_EQ(range_frames(out, "range_bins").size(), 1u);

  const HostTensor profiles = range_frames(out, "range_profiles")[0];
  const HostTensor bins = range_frames(out, "range_bins")[0];
  EXPECT_EQ(profiles.shape, (std::vector<int64_t>{kChirps, kSamples}));
  ASSERT_EQ(bins.shape, (std::vector<int64_t>{kSamples}));
  const float* axis = bins.as<float>();

  const auto magnitude = slice_abs<float>(profiles, 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(axis[peak], target_range, bin_width(kSamples));

  const auto expected = expected_range_axis(kSamples, waveform.slope(), waveform.sample_rate,
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
  const auto out = run_range_frames<float>({std::move(input)}, std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 1u);
  ASSERT_EQ(range_frames(out, "range_bins").size(), 1u);

  const HostTensor profiles = range_frames(out, "range_profiles")[0];
  const HostTensor bins = range_frames(out, "range_bins")[0];
  EXPECT_EQ(profiles.shape, (std::vector<int64_t>{kChirps, kSamples / 2 + 1}));
  ASSERT_EQ(bins.shape, (std::vector<int64_t>{kSamples / 2 + 1}));

  const auto magnitude = slice_abs<float>(profiles, 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(bins.as<float>()[peak], target_range, bin_width(kSamples));
}

TEST(FMCWRangeProcessorOpTest, NoRangeBinsWithoutWaveform) {
  const auto result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto input = complex_cube_to_device<float>(result.cube, result.shape);

  // No waveform → default-constructed is_valid() == false → range_bins
  // not emitted. Use single-IO harness (profiles only); range_bins port
  // is declared but unwired.
  FMCWRangeProcessorParams params;
  const auto out = run_profiles_only_graph_result<float>({std::move(input)}, std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 1u);
  EXPECT_EQ(range_frames(out, "range_profiles")[0].shape,
            (std::vector<int64_t>{kChirps, kSamples}));
}

// Supplying a waveform makes range_bins computable, but only wiring makes it
// wanted, and the operator cannot tell the two apart — nothing exposes an
// output's wiring to contract() or compute(). So it emits, and the first
// failure warns and latches the port off. Propagating that failure instead
// would fail the activation and take range_profiles down with it, which is
// what this pins: an undrained range_bins costs one warning, not the run.
// NoRangeBinsWithoutWaveform is the companion where no waveform means nothing
// is emitted at all. See the operator README, "Optional output convention".
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
  const auto out = run_profiles_only_graph_result<float>(std::move(inputs), std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  // Every tick's sibling output is delivered, start to finish.
  ASSERT_EQ(range_frames(out, "range_profiles").size(), kFrames);
  for (const HostTensor& frame : range_frames(out, "range_profiles")) {
    EXPECT_EQ(frame.shape, (std::vector<int64_t>{kChirps, kSamples}));
  }
}

TEST(FMCWRangeProcessorOpTest, ApplyDechirpMatchesHardwareDechirp) {
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.5);

  const auto hw_sim = sim::simulate(make_scene(target_range, true));
  const auto pre_mixer_sim = sim::simulate(make_scene(target_range, false));

  FMCWRangeProcessorParams hw_params;
  hw_params.waveform = to_descriptor(waveform);
  const auto hw = run_complex_frame<float>(hw_sim, std::move(hw_params));

  FMCWRangeProcessorParams sw_params;
  sw_params.waveform = to_descriptor(waveform);
  sw_params.apply_dechirp = true;
  const auto sw = run_complex_frame<float>(pre_mixer_sim, std::move(sw_params));

  ASSERT_TRUE(hw.ok());
  ASSERT_TRUE(sw.ok());
  ASSERT_EQ(range_frames(hw, "range_profiles").size(), 1u);
  ASSERT_EQ(range_frames(sw, "range_profiles").size(), 1u);
  ASSERT_EQ(range_frames(hw, "range_profiles")[0].shape,
            range_frames(sw, "range_profiles")[0].shape);

  const auto hw_profiles = range_frames(hw, "range_profiles");
  const auto sw_profiles = range_frames(sw, "range_profiles");
  const auto* a = hw_profiles[0].as<std::complex<float>>();
  const auto* b = sw_profiles[0].as<std::complex<float>>();
  double peak_mag = 0.0;
  for (int64_t i = 0; i < range_frames(hw, "range_profiles")[0].size(); ++i) {
    peak_mag = std::max(peak_mag, std::abs(std::complex<double>(a[i])));
  }
  ASSERT_GT(peak_mag, 0.0);
  for (int64_t i = 0; i < range_frames(hw, "range_profiles")[0].size(); ++i) {
    const double diff = std::abs(std::complex<double>(a[i]) - std::complex<double>(b[i]));
    ASSERT_LE(diff, 2e-3 * peak_mag) << "element " << i;
  }
}

TEST(FMCWRangeProcessorOpTest, ApplyDechirpWithoutWaveformErrors) {
  const auto result = sim::simulate(make_scene(range_at_bin(30.0), false));
  auto input = complex_cube_to_device<float>(result.cube, result.shape);

  FMCWRangeProcessorParams params;
  params.apply_dechirp = true;
  // No waveform set → should error.
  const auto out = run_profiles_only_graph_result<float>({std::move(input)}, std::move(params));

  EXPECT_FALSE(out.ok());
  EXPECT_TRUE(range_frames(out, "range_profiles").empty());
  EXPECT_FALSE(out.diagnostics().empty());
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
  const auto out = run_profiles_only_graph_result<float>({std::move(input)}, std::move(params));

  EXPECT_FALSE(out.ok());
  EXPECT_TRUE(range_frames(out, "range_profiles").empty());
  EXPECT_FALSE(out.diagnostics().empty());
}

TEST(FMCWRangeProcessorOpTest, HannWindowLowersSidelobesKeepsPeak) {
  const sim::SimWaveform waveform{};
  WindowSpec hann;
  hann.type = WindowType::kHann;

  // Off-bin target: rectangular sidelobes are fully exposed.
  const auto off_bin_sim = sim::simulate(make_scene(range_at_bin(30.5)));

  FMCWRangeProcessorParams rect_params;
  rect_params.waveform = to_descriptor(waveform);
  const auto rect = run_complex_frame<float>(off_bin_sim, std::move(rect_params));

  FMCWRangeProcessorParams hann_params;
  hann_params.waveform = to_descriptor(waveform);
  hann_params.range_window = hann;
  const auto hann_run = run_complex_frame<float>(off_bin_sim, std::move(hann_params));

  ASSERT_TRUE(rect.ok());
  ASSERT_TRUE(hann_run.ok());
  ASSERT_EQ(range_frames(rect, "range_profiles").size(), 1u);
  ASSERT_EQ(range_frames(hann_run, "range_profiles").size(), 1u);

  const auto rect_mag = slice_abs<float>(range_frames(rect, "range_profiles")[0], 0, 0);
  const auto hann_mag = slice_abs<float>(range_frames(hann_run, "range_profiles")[0], 0, 0);

  const int64_t peak = argmax(rect_mag);
  EXPECT_NEAR(static_cast<double>(argmax(hann_mag)), static_cast<double>(peak), 1.0);

  double rect_sidelobe = 0.0;
  double hann_sidelobe = 0.0;
  for (int64_t i = peak + 4; i < std::min<int64_t>(peak + 16, kSamples); ++i) {
    rect_sidelobe = std::max(rect_sidelobe, rect_mag[static_cast<size_t>(i)]);
    hann_sidelobe = std::max(hann_sidelobe, hann_mag[static_cast<size_t>(i)]);
  }
  EXPECT_LT(hann_sidelobe, 0.5 * rect_sidelobe);

  // On-bin target: coherent-gain normalization preserves the peak
  // magnitude (within a few %) under fft_norm = backward.
  const auto on_bin_sim = sim::simulate(make_scene(range_at_bin(30.0)));

  FMCWRangeProcessorParams rect_on_params;
  rect_on_params.waveform = to_descriptor(waveform);
  const auto rect_on = run_complex_frame<float>(on_bin_sim, std::move(rect_on_params));

  FMCWRangeProcessorParams hann_on_params;
  hann_on_params.waveform = to_descriptor(waveform);
  hann_on_params.range_window = hann;
  const auto hann_on = run_complex_frame<float>(on_bin_sim, std::move(hann_on_params));

  ASSERT_TRUE(rect_on.ok());
  ASSERT_TRUE(hann_on.ok());
  const auto rect_on_mag = slice_abs<float>(range_frames(rect_on, "range_profiles")[0], 0, 0);
  const auto hann_on_mag = slice_abs<float>(range_frames(hann_on, "range_profiles")[0], 0, 0);
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
  const auto out = run_complex_frame<float>(sim_result, std::move(params));
  ASSERT_TRUE(out.ok());
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 1u);
}

TEST(FMCWRangeProcessorOpTest, WaveformConfigurableFromParams) {
  const sim::SimWaveform waveform{};
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  const auto out = run_complex_frame<float>(sim_result, std::move(params));
  ASSERT_TRUE(out.ok());
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 1u);
  ASSERT_EQ(range_frames(out, "range_bins").size(), 1u);
}

TEST(FMCWRangeProcessorOpTest, NfftOversamplingInterpolatesGrid) {
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);
  const auto sim_result = sim::simulate(make_scene(target_range));
  const int64_t nfft = 2 * kSamples;

  FMCWRangeProcessorParams base_params;
  base_params.waveform = to_descriptor(waveform);
  const auto base = run_complex_frame<float>(sim_result, std::move(base_params));

  FMCWRangeProcessorParams over_params;
  over_params.waveform = to_descriptor(waveform);
  over_params.nfft = nfft;
  const auto oversampled = run_complex_frame<float>(sim_result, std::move(over_params));

  ASSERT_TRUE(base.ok());
  ASSERT_TRUE(oversampled.ok());
  ASSERT_EQ(range_frames(oversampled, "range_profiles").size(), 1u);
  ASSERT_EQ(range_frames(oversampled, "range_bins").size(), 1u);

  EXPECT_EQ(range_frames(oversampled, "range_profiles")[0].shape,
            (std::vector<int64_t>{kChirps, nfft}));
  ASSERT_EQ(range_frames(oversampled, "range_bins")[0].shape, (std::vector<int64_t>{nfft}));
  const auto oversampled_bins = range_frames(oversampled, "range_bins");
  const float* axis = oversampled_bins[0].as<float>();

  const auto magnitude = slice_abs<float>(range_frames(oversampled, "range_profiles")[0], 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(axis[peak], target_range, bin_width(kSamples));

  ASSERT_EQ(range_frames(base, "range_bins").size(), 1u);
  const auto base_bins = range_frames(base, "range_bins");
  const float* base_axis = base_bins[0].as<float>();
  EXPECT_NEAR(axis[1], 0.5 * base_axis[1], 1e-4 * base_axis[1]);
}

TEST(FMCWRangeProcessorOpTest, NamedRxAxisRoundTrip) {
  // Rank-3 leading dim is the named N_rx axis (SIMO), not opaque batch.
  // default_test_extents() sets max_rx so the leading axis is admitted.
  const sim::SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);

  sim::SimScene scene = make_scene(target_range);
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0}, sim::Vec3{0.0, 0.0019, 0.0}};
  const auto sim_result = sim::simulate(scene);
  ASSERT_EQ(sim_result.shape, (std::vector<int64_t>{2, kChirps, kSamples}));

  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(waveform);
  const auto out = run_complex_frame<float>(sim_result, std::move(params));

  ASSERT_TRUE(out.ok());
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 1u);
  EXPECT_EQ(range_frames(out, "range_profiles")[0].shape,
            (std::vector<int64_t>{2, kChirps, kSamples}));

  for (int64_t chirp = 0; chirp < kChirps; ++chirp) {
    const auto rx0 = slice_abs<float>(range_frames(out, "range_profiles")[0], 0, chirp);
    const auto rx1 = slice_abs<float>(range_frames(out, "range_profiles")[0], 1, chirp);
    for (int64_t i = 0; i < kSamples; ++i) {
      EXPECT_NEAR(rx0[static_cast<size_t>(i)], rx1[static_cast<size_t>(i)], 1e-3)
          << "chirp " << chirp << " bin " << i;
    }
  }
}

struct LiveNfftTick {
  std::shared_ptr<holoscan::Tensor> cube;
  std::optional<int64_t> nfft;
  int64_t nfft_skew_ns{0};
};

holoscan::testing::TestResult run_nfft_override_graph(std::vector<LiveNfftTick> ticks,
                                                      FMCWRangeProcessorParams params) {
  using Op = FMCWRangeProcessorOp<float>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto& tick = ticks[index];
    if (tick.nfft_skew_ns == 0) {
      if (tick.nfft.has_value()) {
        fixture.push_frame(
            holoscan::testing::frame_at(capture_time, holoscan::testing::input("nfft", *tick.nfft),
                                        holoscan::testing::input("adc_cube", *tick.cube)));
      } else {
        fixture.push_frame(holoscan::testing::frame_at(
            capture_time, holoscan::testing::input("adc_cube", *tick.cube)));
      }
    } else {
      if (tick.nfft.has_value()) {
        fixture.push_at("nfft", *tick.nfft,
                        capture_time + std::chrono::nanoseconds{tick.nfft_skew_ns});
      }
      fixture.push_at("adc_cube", *tick.cube, capture_time);
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("range_profiles", ticks.size(), kMaximumBytes);
  fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
  fixture.output_device("range_profiles", holoscan::DeviceId{0});
  fixture.output_device("range_bins", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "range_profiles", ticks.size());
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}
std::vector<LiveNfftTick> two_cube_ticks(const std::shared_ptr<holoscan::Tensor>& cube,
                                         std::optional<int64_t> nfft0, std::optional<int64_t> nfft1,
                                         int64_t nfft0_skew_ns = 0, int64_t nfft1_skew_ns = 0) {
  return {{cube, nfft0, nfft0_skew_ns}, {cube, nfft1, nfft1_skew_ns}};
}

TEST(FMCWRangeProcessorOpTest, LiveNfftExactMatchChangesBinCount) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeProcessorParams params;
  const auto out =
      run_nfft_override_graph(two_cube_ticks(cube, int64_t{256}, int64_t{256}), std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 2u);
  EXPECT_EQ(range_frames(out, "range_profiles")[0].shape.back(), 256);
  EXPECT_EQ(range_frames(out, "range_profiles")[1].shape.back(), 256);
}

TEST(FMCWRangeProcessorOpTest, LiveNfftRevertsToParamsAfterMiss) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeProcessorParams params;
  const auto out =
      run_nfft_override_graph(two_cube_ticks(cube, int64_t{256}, std::nullopt), std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 2u);
  EXPECT_EQ(range_frames(out, "range_profiles")[0].shape.back(), 256);
  EXPECT_EQ(range_frames(out, "range_profiles")[1].shape.back(), kSamples);
}

TEST(FMCWRangeProcessorOpTest, LiveNfftSkewedCaptureTimeMissesExact) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
  const int64_t skew_ns = 1'000'000;

  FMCWRangeProcessorParams params;
  const auto out = run_nfft_override_graph(
      two_cube_ticks(cube, int64_t{256}, int64_t{256}, skew_ns, skew_ns), std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 2u);
  EXPECT_EQ(range_frames(out, "range_profiles")[0].shape.back(), kSamples);
  EXPECT_EQ(range_frames(out, "range_profiles")[1].shape.back(), kSamples);
}

struct LiveRangeDescriptorTick {
  std::shared_ptr<holoscan::Tensor> cube;
  std::optional<WindowSpec> range_window;
  std::optional<FMCWWaveformDescriptor> waveform;
  bool invalid_waveform{false};
};

holoscan::testing::TestResult run_range_descriptor_override_graph(
    std::vector<LiveRangeDescriptorTick> ticks, FMCWRangeProcessorParams params) {
  using Op = FMCWRangeProcessorOp<float>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto& tick = ticks[index];
    const auto cube = holoscan::testing::input("adc_cube", *tick.cube);
    if (tick.range_window.has_value() && tick.waveform.has_value()) {
      auto window = holoscan::radar::to_wire(*tick.range_window);
      auto waveform = holoscan::radar::to_wire(*tick.waveform);
      if (tick.invalid_waveform) {
        waveform.chirp_duration = 0.0;
      }
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("range_window", std::move(window)),
          holoscan::testing::input("waveform", std::move(waveform)), cube));
    } else if (tick.range_window.has_value()) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time,
          holoscan::testing::input("range_window", holoscan::radar::to_wire(*tick.range_window)),
          cube));
    } else if (tick.waveform.has_value()) {
      auto waveform = holoscan::radar::to_wire(*tick.waveform);
      if (tick.invalid_waveform) {
        waveform.chirp_duration = 0.0;
      }
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("waveform", std::move(waveform)), cube));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(capture_time, cube));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("range_profiles", ticks.size(), kMaximumBytes);
  fixture.capture_snapshot<holoscan::Tensor>("range_bins", ticks.size(), kMaximumBytes);
  fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
  fixture.output_device("range_profiles", holoscan::DeviceId{0});
  fixture.output_device("range_bins", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "range_profiles", ticks.size());
  (void)completion.wait_for_output(running, "range_bins", ticks.size());
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
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
  LiveRangeDescriptorTick changed{cube, hann, to_descriptor(narrow_waveform)};

  FMCWRangeProcessorParams params;
  params.waveform = to_descriptor(base_waveform);
  const auto out = run_range_descriptor_override_graph({base, changed, base}, std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(range_frames(out, "range_profiles").size(), 3u);
  ASSERT_EQ(range_frames(out, "range_bins").size(), 3u);

  const auto rect = slice_abs<float>(range_frames(out, "range_profiles")[0], 0, 0);
  const auto windowed = slice_abs<float>(range_frames(out, "range_profiles")[1], 0, 0);
  const auto reverted = slice_abs<float>(range_frames(out, "range_profiles")[2], 0, 0);
  const int64_t peak = argmax(rect);
  double rect_sidelobe = 0.0;
  double windowed_sidelobe = 0.0;
  for (int64_t i = peak + 4; i < std::min<int64_t>(peak + 16, kSamples); ++i) {
    rect_sidelobe = std::max(rect_sidelobe, rect[static_cast<size_t>(i)]);
    windowed_sidelobe = std::max(windowed_sidelobe, windowed[static_cast<size_t>(i)]);
  }
  EXPECT_LT(windowed_sidelobe, 0.5 * rect_sidelobe);
  // Tick 2 publishes no window, so the Params (rectangular) window is back.
  EXPECT_EQ(rect, reverted);

  const float base_spacing = range_frames(out, "range_bins")[0].as<float>()[1];
  const float changed_spacing = range_frames(out, "range_bins")[1].as<float>()[1];
  EXPECT_NEAR(changed_spacing, 2.0F * base_spacing, 1e-4F * changed_spacing);
  // Likewise for the waveform: the range axis returns to the Params spacing.
  EXPECT_EQ(range_frames(out, "range_bins")[0].as<float>()[1],
            range_frames(out, "range_bins")[2].as<float>()[1]);
}

TEST(FMCWRangeProcessorOpTest, InvalidLiveWaveformRejectedBeforeDsp) {
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));
  auto cube = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
  LiveRangeDescriptorTick invalid{cube, std::nullopt, to_descriptor(sim::SimWaveform{})};
  invalid.invalid_waveform = true;

  FMCWRangeProcessorParams params;
  const auto out = run_range_descriptor_override_graph({invalid}, std::move(params));

  ASSERT_FALSE(out.ok());
  EXPECT_TRUE(range_frames(out, "range_profiles").empty());
  EXPECT_TRUE(range_frames(out, "range_bins").empty());
  EXPECT_FALSE(out.diagnostics().empty());
}

TEST(FMCWRangeProcessorOpTest, FloatAndDoubleInstantiationsAgree) {
  const sim::SimWaveform waveform{};
  const auto sim_result = sim::simulate(make_scene(range_at_bin(30.0)));

  FMCWRangeProcessorParams params_f;
  params_f.waveform = to_descriptor(waveform);
  const auto run_f = run_complex_frame<float>(sim_result, std::move(params_f));

  FMCWRangeProcessorParams params_d;
  params_d.waveform = to_descriptor(waveform);
  const auto run_d = run_complex_frame<double>(sim_result, std::move(params_d));

  ASSERT_TRUE(run_f.ok());
  ASSERT_TRUE(run_d.ok());
  ASSERT_EQ(range_frames(run_f, "range_profiles").size(), 1u);
  ASSERT_EQ(range_frames(run_d, "range_profiles").size(), 1u);

  const auto mag_f = slice_abs<float>(range_frames(run_f, "range_profiles")[0], 0, 0);
  const auto mag_d = slice_abs<double>(range_frames(run_d, "range_profiles")[0], 0, 0);

  const double peak = mag_d[static_cast<size_t>(argmax(mag_d))];
  EXPECT_EQ(argmax(mag_f), argmax(mag_d));
  for (int64_t i = 0; i < kSamples; ++i) {
    EXPECT_NEAR(mag_f[static_cast<size_t>(i)], mag_d[static_cast<size_t>(i)], 1e-4 * peak)
        << "bin " << i;
  }
}

}  // namespace
