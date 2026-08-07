// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for FMCWRangeProcessorOp.
//
// The analytic point-target simulator
// (radar_common/sim/scene.hpp) provides the input cubes and the
// forward bin model (utils/bin_model.hpp) the
// independent ground truth; assertions are made in the physical domain via
// the operator's own range_bins output wherever possible. Driven through
// the SDK-native holoscan::test::OperatorTestHarness (utils/sdk_harness.hpp).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "utils/bin_model.hpp"
#include "utils/sim_reference.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::FMCWRangeProcessorOp;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::testing::complex_cube_to_device;
using holoscan::radar::testing::expected_range_axis;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::sim::kSimSpeedOfLight;
using holoscan::radar::sim::PointTarget;
using holoscan::radar::testing::range_of_bin;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::sim::simulate;
using holoscan::radar::sim::SimResult;
using holoscan::radar::sim::SimScene;
using holoscan::radar::sim::SimWaveform;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::sim::to_real_if;
using holoscan::radar::sim::Vec3;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

constexpr int64_t kChirps = 32;
constexpr int64_t kSamples = 128;

// Default SimWaveform: 77 GHz, 150 MHz sweep over 20 us, fs = 12.8 MHz.
// One range bin = c * fs / (2 * slope * nfft) ~ 2.0 m at nfft = 128.
double bin_width(int64_t nfft) {
  const SimWaveform waveform{};
  return range_of_bin(1, waveform.slope(), waveform.sample_rate, nfft,
                      kSimSpeedOfLight);
}

// Range that falls exactly on `bin` (+ optional fractional offset) of the
// default nfft = kSamples grid.
double range_at_bin(double bin) {
  return bin * bin_width(kSamples);
}

FMCWWaveformDescriptor to_descriptor(const SimWaveform& waveform) {
  holoscan::radar::FMCWWaveformSpec spec;
  spec.carrier_frequency = waveform.carrier_frequency;
  spec.bandwidth = waveform.bandwidth;
  spec.chirp_duration = waveform.chirp_duration;
  spec.sample_rate = waveform.sample_rate;
  spec.chirp_repetition_interval =
      waveform.chirp_repetition_interval;
  return FMCWWaveformDescriptor(spec);
}

SimScene make_scene(double range_m, bool dechirped = true) {
  SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kSamples;
  scene.dechirped = dechirped;
  PointTarget target;
  target.position = Vec3{range_m, 0.0, 0.0};
  scene.targets.push_back(target);
  return scene;
}

// |profile| of one (batch, chirp) slice of a captured range_profiles
// tensor with shape (..., num_chirps, num_bins).
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

// The host copies of one tick's range_profiles and range_bins outputs.
struct RunOutputs {
  HostTensor profiles;
  HostTensor bins;
};

// Run the operator once over a prepared frame through the SDK harness and
// return the captured host outputs. The waveform must be set on the
// harness's operator args (range_bins is emitted every tick).
template <typename HarnessT>
RunOutputs run_harness_single(const std::shared_ptr<HarnessT>& harness,
                              holoscan::TensorMap frame) {
  TensorMapCapture profiles;
  TensorMapCapture bins;
  harness->template add_input_port<holoscan::TensorMap>("adc_cube",
                                                        {frame});
  harness->template add_output_port<holoscan::TensorMap>(
      "range_profiles", profiles.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "range_bins", bins.validators());
  harness->run_test();
  if (profiles.frames.size() != 1 || bins.frames.size() != 1) {
    throw std::runtime_error("expected exactly one captured output frame");
  }
  return RunOutputs{std::move(profiles.frames.front()),
                    std::move(bins.frames.front())};
}

// Runs one complex-IF frame through the operator (waveform attached via
// `args`) and returns the captured host outputs.
template <typename T, typename... ArgsT>
RunOutputs run_complex_frame(const SimResult& sim, ArgsT&&... args) {
  auto frame =
      single_map(complex_cube_to_device<T>(sim.cube, sim.shape), "adc_cube");
  return run_harness_single(
      create_operator_test<FMCWRangeProcessorOp<T>>(
          std::forward<ArgsT>(args)...),
      std::move(frame));
}

TEST(FMCWRangeProcessorOpTest, SingleTargetComplexIfPeakAndRangeAxis) {
  const SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);
  const auto sim = simulate(make_scene(target_range));

  const RunOutputs out = run_complex_frame<float>(
      sim, holoscan::Arg("waveform") = to_descriptor(waveform));

  EXPECT_EQ(out.profiles.shape, (std::vector<int64_t>{kChirps, kSamples}));
  ASSERT_EQ(out.bins.shape, (std::vector<int64_t>{kSamples}));
  const float* axis = out.bins.as<float>();

  // Peak maps to the target range within one bin via the op's own axis.
  const auto magnitude = slice_abs<float>(out.profiles, 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(axis[peak], target_range, bin_width(kSamples));

  // range_bins matches the independent bin model.
  const auto expected =
      expected_range_axis(kSamples, waveform.slope(), waveform.sample_rate,
                          kSamples, kSimSpeedOfLight);
  for (int64_t i = 0; i < kSamples; ++i) {
    EXPECT_NEAR(axis[i], expected[static_cast<size_t>(i)],
                1e-3 + 1e-5 * std::abs(expected[static_cast<size_t>(i)]))
        << "bin " << i;
  }
}

TEST(FMCWRangeProcessorOpTest, RealIfPathPeakAtRange) {
  const SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);
  const auto sim = simulate(make_scene(target_range));
  const auto real_cube = to_real_if(sim);

  auto frame = single_map(real_cube_to_device<float>(real_cube, sim.shape),
                          "adc_cube");
  const RunOutputs out = run_harness_single(
      create_operator_test<FMCWRangeProcessorOp<float>>(
          holoscan::Arg("waveform") = to_descriptor(waveform)),
      std::move(frame));

  EXPECT_EQ(out.profiles.shape,
            (std::vector<int64_t>{kChirps, kSamples / 2 + 1}));
  ASSERT_EQ(out.bins.shape, (std::vector<int64_t>{kSamples / 2 + 1}));

  const auto magnitude = slice_abs<float>(out.profiles, 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(out.bins.as<float>()[peak], target_range,
              bin_width(kSamples));
}

TEST(FMCWRangeProcessorOpTest, NoRangeBinsWithoutWaveform) {
  const auto sim = simulate(make_scene(range_at_bin(30.0)));

  // No waveform descriptor: range_profiles still flows, but the optional
  // range_bins output must never be emitted. Its sink has no validators
  // (so message-less ticks are tolerated) and is checked by count.
  auto harness = create_operator_test<FMCWRangeProcessorOp<float>>();
  TensorMapCapture profiles;
  auto frame =
      single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                 "adc_cube");
  harness->add_input_port<holoscan::TensorMap>("adc_cube", {frame});
  harness->add_output_port<holoscan::TensorMap>("range_profiles",
                                                profiles.validators());
  harness->add_output_port<holoscan::TensorMap>("range_bins");
  harness->run_test();

  ASSERT_EQ(profiles.frames.size(), 1u);
  EXPECT_EQ(profiles.frames[0].shape,
            (std::vector<int64_t>{kChirps, kSamples}));
  EXPECT_EQ(
      harness->get_sink<holoscan::TensorMap>("range_bins")
          ->get_received_count(),
      0u);
}

TEST(FMCWRangeProcessorOpTest, ApplyDechirpMatchesHardwareDechirp) {
  const SimWaveform waveform{};
  const double target_range = range_at_bin(30.5);

  const auto hw_sim = simulate(make_scene(target_range, true));
  const auto pre_mixer_sim = simulate(make_scene(target_range, false));

  const RunOutputs hw = run_complex_frame<float>(
      hw_sim, holoscan::Arg("waveform") = to_descriptor(waveform));
  const RunOutputs sw = run_complex_frame<float>(
      pre_mixer_sim, holoscan::Arg("waveform") = to_descriptor(waveform),
      holoscan::Arg("apply_dechirp") = true);

  ASSERT_EQ(hw.profiles.shape, sw.profiles.shape);

  const auto* a = hw.profiles.as<std::complex<float>>();
  const auto* b = sw.profiles.as<std::complex<float>>();
  double peak_mag = 0.0;
  for (int64_t i = 0; i < hw.profiles.size(); ++i) {
    peak_mag = std::max(peak_mag,
                        std::abs(std::complex<double>(a[i])));
  }
  ASSERT_GT(peak_mag, 0.0);
  for (int64_t i = 0; i < hw.profiles.size(); ++i) {
    const double diff =
        std::abs(std::complex<double>(a[i]) - std::complex<double>(b[i]));
    ASSERT_LE(diff, 2e-3 * peak_mag) << "element " << i;
  }
}

TEST(FMCWRangeProcessorOpTest, ApplyDechirpWithoutWaveformErrors) {
  const auto sim = simulate(make_scene(range_at_bin(30.0), false));

  auto harness = create_operator_test<FMCWRangeProcessorOp<float>>(
      holoscan::Arg("apply_dechirp") = true);
  auto frame =
      single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                 "adc_cube");
  harness->add_input_port<holoscan::TensorMap>("adc_cube", {frame});
  harness->add_output_port<holoscan::TensorMap>("range_profiles");
  EXPECT_ANY_THROW(harness->run_test());
}

TEST(FMCWRangeProcessorOpTest, ApplyDechirpWithRealInputErrors) {
  const SimWaveform waveform{};
  const auto sim = simulate(make_scene(range_at_bin(30.0)));
  const auto real_cube = to_real_if(sim);

  auto harness = create_operator_test<FMCWRangeProcessorOp<float>>(
      holoscan::Arg("apply_dechirp") = true,
      holoscan::Arg("waveform") = to_descriptor(waveform));
  auto frame = single_map(real_cube_to_device<float>(real_cube, sim.shape),
                          "adc_cube");
  harness->add_input_port<holoscan::TensorMap>("adc_cube", {frame});
  harness->add_output_port<holoscan::TensorMap>("range_profiles");
  EXPECT_ANY_THROW(harness->run_test());
}

TEST(FMCWRangeProcessorOpTest, HannWindowLowersSidelobesKeepsPeak) {
  const SimWaveform waveform{};
  WindowSpec hann;
  hann.type = WindowType::kHann;

  // Off-bin target: rectangular sidelobes are fully exposed.
  const auto off_bin_sim = simulate(make_scene(range_at_bin(30.5)));
  const RunOutputs rect = run_complex_frame<float>(
      off_bin_sim, holoscan::Arg("waveform") = to_descriptor(waveform));
  const RunOutputs hann_run = run_complex_frame<float>(
      off_bin_sim, holoscan::Arg("waveform") = to_descriptor(waveform),
      holoscan::Arg("range_window") = hann);

  const auto rect_mag = slice_abs<float>(rect.profiles, 0, 0);
  const auto hann_mag = slice_abs<float>(hann_run.profiles, 0, 0);

  // The half-bin-offset target sits between two bins, so the argmax can
  // legitimately land on either neighbor depending on the window.
  const int64_t peak = argmax(rect_mag);
  EXPECT_NEAR(static_cast<double>(argmax(hann_mag)),
              static_cast<double>(peak), 1.0);

  // Peak sidelobe in a window near (but outside) the main lobe.
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
  const auto on_bin_sim = simulate(make_scene(range_at_bin(30.0)));
  const RunOutputs rect_on = run_complex_frame<float>(
      on_bin_sim, holoscan::Arg("waveform") = to_descriptor(waveform));
  const RunOutputs hann_on = run_complex_frame<float>(
      on_bin_sim, holoscan::Arg("waveform") = to_descriptor(waveform),
      holoscan::Arg("range_window") = hann);
  const auto rect_on_mag = slice_abs<float>(rect_on.profiles, 0, 0);
  const auto hann_on_mag = slice_abs<float>(hann_on.profiles, 0, 0);
  const double rect_peak = rect_on_mag[static_cast<size_t>(30)];
  const double hann_peak = hann_on_mag[static_cast<size_t>(30)];
  EXPECT_NEAR(hann_peak, rect_peak, 0.03 * rect_peak);
}

TEST(FMCWRangeProcessorOpTest, NfftOversamplingInterpolatesGrid) {
  const SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);
  const auto sim = simulate(make_scene(target_range));
  const int64_t nfft = 2 * kSamples;

  const RunOutputs base = run_complex_frame<float>(
      sim, holoscan::Arg("waveform") = to_descriptor(waveform));
  const RunOutputs oversampled = run_complex_frame<float>(
      sim, holoscan::Arg("waveform") = to_descriptor(waveform),
      holoscan::Arg("nfft") = nfft);

  EXPECT_EQ(oversampled.profiles.shape,
            (std::vector<int64_t>{kChirps, nfft}));
  ASSERT_EQ(oversampled.bins.shape, (std::vector<int64_t>{nfft}));
  const float* axis = oversampled.bins.as<float>();

  // Peak location in meters is unchanged (within the original resolution).
  const auto magnitude = slice_abs<float>(oversampled.profiles, 0, 0);
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(axis[peak], target_range, bin_width(kSamples));

  // Bin spacing halves relative to the exact-N axis.
  const float* base_axis = base.bins.as<float>();
  EXPECT_NEAR(axis[1], 0.5 * base_axis[1], 1e-4 * base_axis[1]);
}

TEST(FMCWRangeProcessorOpTest, LeadingBatchAxesRoundTrip) {
  const SimWaveform waveform{};
  const double target_range = range_at_bin(30.0);

  // SIMO, broadside target: rx elements offset along y, target along x,
  // so the aperture projection is zero and both channels are identical.
  SimScene scene = make_scene(target_range);
  scene.rx_positions = {Vec3{0.0, 0.0, 0.0}, Vec3{0.0, 0.0019, 0.0}};
  const auto sim = simulate(scene);
  ASSERT_EQ(sim.shape, (std::vector<int64_t>{2, kChirps, kSamples}));

  const RunOutputs out = run_complex_frame<float>(
      sim, holoscan::Arg("waveform") = to_descriptor(waveform));

  EXPECT_EQ(out.profiles.shape,
            (std::vector<int64_t>{2, kChirps, kSamples}));

  for (int64_t chirp = 0; chirp < kChirps; ++chirp) {
    const auto rx0 = slice_abs<float>(out.profiles, 0, chirp);
    const auto rx1 = slice_abs<float>(out.profiles, 1, chirp);
    for (int64_t i = 0; i < kSamples; ++i) {
      EXPECT_NEAR(rx0[static_cast<size_t>(i)], rx1[static_cast<size_t>(i)],
                  1e-3)
          << "chirp " << chirp << " bin " << i;
    }
  }
}

// Override values occupy the LEADING ticks (wire_leading_override), and the
// data ticks outnumber them, so the trailing tick exercises the
// sticky-override rule (an override remains effective
// until the next override).
class LiveOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  std::vector<WindowSpec> overrides;
  FMCWWaveformDescriptor waveform;
  TensorMapCapture* profiles = nullptr;
  TensorMapCapture* bins = nullptr;

  void compose() override {
    auto op = make_operator<FMCWRangeProcessorOp<float>>(
        "op_under_test", holoscan::Arg("waveform") = waveform);
    wire_data_source(this, op, "adc_cube", frames);
    wire_leading_override<WindowSpec>(this, op, "range_window", overrides);
    wire_capture_sink(this, op, "range_profiles", profiles);
    wire_capture_sink(this, op, "range_bins", bins);
  }
};

TEST(FMCWRangeProcessorOpTest, LiveWindowOverrideAppliesPerTick) {
  const SimWaveform waveform{};
  // Off-bin target so rectangular and Hann profiles differ materially.
  const auto sim = simulate(make_scene(range_at_bin(30.5)));
  WindowSpec hann;
  hann.type = WindowType::kHann;
  auto frame = [&]() {
    return single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                      "adc_cube");
  };

  // Parameter-default (rectangular) reference profile from a plain run.
  const RunOutputs reference = run_complex_frame<float>(
      sim, holoscan::Arg("waveform") = to_descriptor(waveform));
  const auto rect_ref = slice_abs<float>(reference.profiles, 0, 0);

  // Sticky override semantics (see operators/common/README.md): tick 0
  // Hann override; tick 1 explicit rectangular
  // override — back to the default output via the cached rectangular
  // configuration; tick 2 no message — the rectangular override REMAINS
  // effective.
  TensorMapCapture profiles;
  TensorMapCapture bins;
  auto app = holoscan::make_application<LiveOverrideApp>();
  app->frames = {frame(), frame(), frame()};
  app->overrides = {hann, WindowSpec{}};
  app->waveform = to_descriptor(waveform);
  app->profiles = &profiles;
  app->bins = &bins;
  app->run();

  ASSERT_EQ(profiles.frames.size(), 3u);
  EXPECT_EQ(bins.frames.size(), 3u);  // waveform set → bins every tick
  const auto tick0 = slice_abs<float>(profiles.frames[0], 0, 0);
  const auto tick1 = slice_abs<float>(profiles.frames[1], 0, 0);
  const auto tick2 = slice_abs<float>(profiles.frames[2], 0, 0);

  const double peak = rect_ref[static_cast<size_t>(argmax(rect_ref))];
  double diff_ref0 = 0.0;
  double diff_ref1 = 0.0;
  double diff12 = 0.0;
  for (int64_t i = 0; i < kSamples; ++i) {
    diff_ref0 = std::max(diff_ref0,
                         std::abs(rect_ref[static_cast<size_t>(i)] -
                                  tick0[static_cast<size_t>(i)]));
    diff_ref1 = std::max(diff_ref1,
                         std::abs(rect_ref[static_cast<size_t>(i)] -
                                  tick1[static_cast<size_t>(i)]));
    diff12 = std::max(diff12, std::abs(tick1[static_cast<size_t>(i)] -
                                       tick2[static_cast<size_t>(i)]));
  }
  // The Hann tick differs from the default; the explicit rectangular
  // override reproduces the default output (same cached rectangular
  // configuration); the message-less tick keeps the rectangular override.
  EXPECT_GT(diff_ref0, 0.01 * peak);
  EXPECT_LE(diff_ref1, 1e-5 * peak);
  EXPECT_LE(diff12, 1e-5 * peak);
}

TEST(FMCWRangeProcessorOpTest, FloatAndDoubleInstantiationsAgree) {
  const SimWaveform waveform{};
  // On-bin target: a half-bin offset (30.5) straddles two equal-magnitude
  // neighbors, so float vs double rounding can tip argmax either way without
  // indicating a real disagreement (see HannWindowLowersSidelobesKeepsPeak).
  const auto sim = simulate(make_scene(range_at_bin(30.0)));

  const RunOutputs run_f = run_complex_frame<float>(
      sim, holoscan::Arg("waveform") = to_descriptor(waveform));
  const RunOutputs run_d = run_complex_frame<double>(
      sim, holoscan::Arg("waveform") = to_descriptor(waveform));

  const auto mag_f = slice_abs<float>(run_f.profiles, 0, 0);
  const auto mag_d = slice_abs<double>(run_d.profiles, 0, 0);

  const double peak = mag_d[static_cast<size_t>(argmax(mag_d))];
  EXPECT_EQ(argmax(mag_f), argmax(mag_d));
  for (int64_t i = 0; i < kSamples; ++i) {
    EXPECT_NEAR(mag_f[static_cast<size_t>(i)],
                mag_d[static_cast<size_t>(i)], 1e-4 * peak)
        << "bin " << i;
  }
}

}  // namespace
