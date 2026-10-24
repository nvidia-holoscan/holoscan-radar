// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for FMCWRangeDopplerProcessorOp (HSDK 5 Graph harness).
//
// The core requirement is PARITY: the fused operator must produce rd_maps
// identical (within tolerance) to the split FMCWRangeProcessorOp ->
// [ClutterFilterOp] -> FMCWDopplerProcessorOp chain for matching parameters,
// with exactly matching range_bins / velocity_bins. The split chain is run
// as separate Graph harness invocations, host-capturing and re-uploading
// between stages.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/additives_reference.hpp"
#include "utils/bin_model.hpp"
#include "utils/sim_reference.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::CFARDetectorOp;
using holoscan::radar::CFARParams;
using holoscan::radar::CfarVariant;
using holoscan::radar::ClutterFilterMethod;
using holoscan::radar::ClutterFilterOp;
using holoscan::radar::ClutterFilterParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::FftNorm;
using holoscan::radar::FMCWDopplerProcessorOp;
using holoscan::radar::FMCWDopplerProcessorParams;
using holoscan::radar::FMCWRangeDopplerProcessorOp;
using holoscan::radar::FMCWRangeDopplerProcessorParams;
using holoscan::radar::FMCWRangeProcessorOp;
using holoscan::radar::FMCWRangeProcessorParams;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::FMCWWaveformSpec;
using holoscan::radar::real_dtype;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::testing::complex_cube_to_device;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::to_device_tensor;
namespace sim = holoscan::radar::sim;

constexpr int64_t kChirps = 32;
constexpr int64_t kSamples = 128;

double range_bin_width(int64_t nfft) {
  const sim::SimWaveform waveform{};
  return holoscan::radar::testing::range_of_bin(1, waveform.slope(), waveform.sample_rate, nfft,
                                                sim::kSimSpeedOfLight);
}

double range_at_bin(double bin) {
  return bin * range_bin_width(kSamples);
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

sim::SimScene parity_scene(bool dechirped = true) {
  sim::SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kSamples;
  scene.dechirped = dechirped;
  sim::PointTarget still;
  still.position = sim::Vec3{range_at_bin(20.0), 0.0, 0.0};
  scene.targets.push_back(still);
  sim::PointTarget mover;
  mover.position = sim::Vec3{range_at_bin(40.0), 0.0, 0.0};
  mover.velocity = sim::Vec3{5.0, 0.0, 0.0};
  scene.targets.push_back(mover);
  return scene;
}

const sim::SimResult& parity_sim() {
  static const sim::SimResult result = sim::simulate(parity_scene());
  return result;
}

const sim::SimResult& parity_sim_real() {
  static const sim::SimResult result = [] {
    auto scene = parity_scene();
    scene.analog_if.sampling_mode = sim::IfSamplingMode::kReal;
    return sim::simulate(scene);
  }();
  return result;
}

const sim::SimResult& parity_sim_pre_mixer() {
  static const sim::SimResult result = sim::simulate(parity_scene(/*dechirped=*/false));
  return result;
}

struct ParityConfig {
  ClutterFilterMethod clutter{ClutterFilterMethod::kNone};
  std::optional<int64_t> nfft_range{};
  std::optional<int64_t> nfft_doppler{};
  WindowSpec range_window{};
  WindowSpec doppler_window{};
  bool match_clutter_window{true};
  bool apply_dechirp{false};
};

struct ChainOutputs {
  HostTensor rd;
  std::optional<HostTensor> range_bins;
  std::optional<HostTensor> velocity_bins;
};

std::vector<HostTensor> tensor_frames(const holoscan::testing::TestResult& result,
                                      const char* port) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.template snapshots<holoscan::Tensor>(port)) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> reupload(const HostTensor& host) {
  const auto* data = host.as<std::complex<T>>();
  std::vector<std::complex<T>> copy(data, data + host.size());
  return to_device_tensor(copy, host.shape, complex_dtype<T>());
}

template <typename T>
ChainOutputs run_split(const std::shared_ptr<holoscan::Tensor>& input_tensor,
                       const ParityConfig& cfg, const FMCWWaveformDescriptor& waveform) {
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  ChainOutputs out;
  HostTensor profiles_host;
  {
    using Op = FMCWRangeProcessorOp<T>;
    FMCWRangeProcessorParams params;
    params.range_window = cfg.range_window;
    params.apply_dechirp = cfg.apply_dechirp;
    params.waveform = waveform;
    if (cfg.nfft_range) {
      params.nfft = *cfg.nfft_range;
    }
    auto fixture = holoscan::testing::create_operator_test<Op>(
        holoscan::radar::testing::default_test_extents());
    fixture.set_params(std::move(params));
    fixture.push(&Op::adc_cube, *input_tensor);
    for (const char* output : {"range_profiles", "range_bins"}) {
      fixture.template capture_snapshot<holoscan::Tensor>(output, 1U, kMaximumBytes);
      fixture.output_device(output, holoscan::DeviceId{0});
    }
    fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
    holoscan::radar::testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    (void)completion.wait_for_output(running, "range_profiles", 1U);
    (void)completion.wait_for_output(running, "range_bins", 1U);
    running.request_stop();
    const auto result = running.finish();
    completion.check(result);
    EXPECT_TRUE(result.ok()) << result.describe_failure();
    profiles_host = tensor_frames(result, "range_profiles").front();
    out.range_bins = tensor_frames(result, "range_bins").front();
  }
  if (cfg.clutter != ClutterFilterMethod::kNone) {
    using Op = ClutterFilterOp<T>;
    ClutterFilterParams params;
    params.method = cfg.clutter;
    if (cfg.clutter == ClutterFilterMethod::kMeanSubtraction && cfg.match_clutter_window) {
      params.doppler_window = cfg.doppler_window;
    }
    auto input = reupload<T>(profiles_host);
    auto fixture = holoscan::testing::create_operator_test<Op>(
        holoscan::radar::testing::default_test_extents());
    fixture.set_params(std::move(params));
    fixture.push(&Op::range_profiles, *input);
    fixture.template capture_snapshot<holoscan::Tensor>("range_profiles_out", 1U, kMaximumBytes);
    fixture.input_source_device("range_profiles", holoscan::DeviceId{0});
    fixture.output_device("range_profiles_out", holoscan::DeviceId{0});
    holoscan::radar::testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    (void)completion.wait_for_output(running, "range_profiles_out", 1U);
    running.request_stop();
    const auto result = running.finish();
    completion.check(result);
    EXPECT_TRUE(result.ok()) << result.describe_failure();
    profiles_host = tensor_frames(result, "range_profiles_out").front();
  }
  {
    using Op = FMCWDopplerProcessorOp<T>;
    FMCWDopplerProcessorParams params;
    params.doppler_window = cfg.doppler_window;
    params.waveform = waveform;
    if (cfg.nfft_doppler) {
      params.nfft = *cfg.nfft_doppler;
    }
    auto input = reupload<T>(profiles_host);
    auto fixture = holoscan::testing::create_operator_test<Op>(
        holoscan::radar::testing::default_test_extents());
    fixture.set_params(std::move(params));
    fixture.push(&Op::range_profiles, *input);
    for (const char* output : {"rd_maps", "velocity_bins"}) {
      fixture.template capture_snapshot<holoscan::Tensor>(output, 1U, kMaximumBytes);
      fixture.output_device(output, holoscan::DeviceId{0});
    }
    fixture.input_source_device("range_profiles", holoscan::DeviceId{0});
    holoscan::radar::testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    (void)completion.wait_for_output(running, "rd_maps", 1U);
    (void)completion.wait_for_output(running, "velocity_bins", 1U);
    running.request_stop();
    const auto result = running.finish();
    completion.check(result);
    EXPECT_TRUE(result.ok()) << result.describe_failure();
    out.rd = tensor_frames(result, "rd_maps").front();
    out.velocity_bins = tensor_frames(result, "velocity_bins").front();
  }
  return out;
}

template <typename T>
holoscan::testing::TestResult run_fused_result(
    const std::shared_ptr<holoscan::Tensor>& input, FMCWRangeDopplerProcessorParams params,
    bool capture_axes = true,
    const holoscan::radar::MaxTensorExtents& extents =
        holoscan::radar::testing::default_test_extents()) {
  using Op = FMCWRangeDopplerProcessorOp<T>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  fixture.push(&Op::adc_cube, *input);
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("rd_maps", 1U, kMaximumBytes);
  if (capture_axes) {
    fixture.template capture_snapshot<holoscan::Tensor>("range_bins", 1U, kMaximumBytes);
    fixture.template capture_snapshot<holoscan::Tensor>("velocity_bins", 1U, kMaximumBytes);
  }
  fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
  for (const char* output : {"rd_maps", "range_bins", "velocity_bins"}) {
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "rd_maps", 1U);
  if (capture_axes) {
    (void)completion.wait_for_output(running, "range_bins", 1U);
    (void)completion.wait_for_output(running, "velocity_bins", 1U);
  }
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

template <typename T>
ChainOutputs run_fused(const std::shared_ptr<holoscan::Tensor>& input_tensor,
                       const ParityConfig& cfg, const FMCWWaveformDescriptor& waveform) {
  FMCWRangeDopplerProcessorParams params;
  params.range_window = cfg.range_window;
  params.doppler_window = cfg.doppler_window;
  params.apply_dechirp = cfg.apply_dechirp;
  params.clutter_filter = cfg.clutter;
  params.waveform = waveform;
  if (cfg.nfft_range) {
    params.nfft_range = *cfg.nfft_range;
  }
  if (cfg.nfft_doppler) {
    params.nfft_doppler = *cfg.nfft_doppler;
  }
  const auto result = run_fused_result<T>(input_tensor, std::move(params));
  EXPECT_TRUE(result.ok()) << result.describe_failure();
  return {tensor_frames(result, "rd_maps").front(), tensor_frames(result, "range_bins").front(),
          tensor_frames(result, "velocity_bins").front()};
}

// ---- Comparison helpers ----

template <typename T>
double max_abs(const HostTensor& tensor) {
  const auto* data = tensor.as<std::complex<T>>();
  double peak = 0.0;
  for (int64_t i = 0; i < tensor.size(); ++i) {
    peak = std::max(peak, std::abs(std::complex<double>(data[i])));
  }
  return peak;
}

template <typename T>
double max_abs_diff(const HostTensor& a, const HostTensor& b) {
  const auto* pa = a.as<std::complex<T>>();
  const auto* pb = b.as<std::complex<T>>();
  double worst = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(std::complex<double>(pa[i]) - std::complex<double>(pb[i])));
  }
  return worst;
}

template <typename T>
void expect_axis_exact(const std::optional<HostTensor>& fused,
                       const std::optional<HostTensor>& split, const std::string& label) {
  ASSERT_TRUE(fused.has_value()) << label;
  ASSERT_TRUE(split.has_value()) << label;
  ASSERT_EQ(fused->shape, split->shape) << label;
  const T* fa = fused->template as<T>();
  const T* sa = split->template as<T>();
  for (int64_t i = 0; i < fused->size(); ++i) {
    ASSERT_EQ(fa[i], sa[i]) << label << " bin " << i;
  }
}

template <typename T>
void expect_parity(const ChainOutputs& fused, const ChainOutputs& split, double rel_tol,
                   const std::string& label) {
  ASSERT_EQ(fused.rd.shape, split.rd.shape) << label;
  const double reference = max_abs<T>(split.rd);
  ASSERT_GT(reference, 0.0) << label;
  const double delta = max_abs_diff<T>(fused.rd, split.rd) / reference;
  EXPECT_LT(delta, rel_tol) << label;
  std::printf("[ parity ] %-40s rel delta = %.3e\n", label.c_str(), delta);
  expect_axis_exact<T>(fused.range_bins, split.range_bins, label + " range_bins");
  expect_axis_exact<T>(fused.velocity_bins, split.velocity_bins, label + " velocity_bins");
}

constexpr double kParityUlps = 8.0;

template <typename T>
void expect_parity_ulp(const ChainOutputs& fused, const ChainOutputs& split,
                       const std::string& label) {
  ASSERT_EQ(fused.rd.shape, split.rd.shape) << label;
  const double reference = max_abs<T>(split.rd);
  ASSERT_GT(reference, 0.0) << label;
  const double delta = max_abs_diff<T>(fused.rd, split.rd) / reference;
  const double tol = kParityUlps * std::numeric_limits<T>::epsilon();
  EXPECT_LT(delta, tol) << label << ": rel delta = " << delta << ", tolerance = " << tol;
  expect_axis_exact<T>(fused.range_bins, split.range_bins, label + " range_bins");
  expect_axis_exact<T>(fused.velocity_bins, split.velocity_bins, label + " velocity_bins");
}

// ---- Parametrized test helpers ----

template <typename T>
void ulp_parity_all_clutter_filters() {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  auto input = complex_cube_to_device<T>(parity_sim().cube, parity_sim().shape);
  const ClutterFilterMethod methods[] = {
      ClutterFilterMethod::kNone, ClutterFilterMethod::kMeanSubtraction,
      ClutterFilterMethod::kSingleDelay, ClutterFilterMethod::kDoubleDelay};
  for (ClutterFilterMethod method : methods) {
    for (bool windowed : {false, true}) {
      ParityConfig cfg;
      cfg.clutter = method;
      if (windowed) {
        cfg.doppler_window.type = WindowType::kHamming;
      }
      auto make = [&] {
        return complex_cube_to_device<T>(parity_sim().cube, parity_sim().shape);
      };
      const auto fused = run_fused<T>(make(), cfg, waveform);
      const auto split = run_split<T>(make(), cfg, waveform);
      expect_parity_ulp<T>(fused, split,
                           std::string("clutter=") + std::string(to_string(method)) +
                               ",dopwin=" + (windowed ? "hamming" : "none"));
    }
  }
}

template <typename T>
void parity_all_clutter_filters(double rel_tol) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  const ClutterFilterMethod methods[] = {
      ClutterFilterMethod::kNone, ClutterFilterMethod::kMeanSubtraction,
      ClutterFilterMethod::kSingleDelay, ClutterFilterMethod::kDoubleDelay};
  for (ClutterFilterMethod method : methods) {
    ParityConfig cfg;
    cfg.clutter = method;
    auto make = [&] {
      return complex_cube_to_device<T>(parity_sim().cube, parity_sim().shape);
    };
    const auto fused = run_fused<T>(make(), cfg, waveform);
    const auto split = run_split<T>(make(), cfg, waveform);
    expect_parity<T>(fused, split, rel_tol,
                     std::string("clutter=") + std::string(to_string(method)));
  }
}

// ---- Tests ----

TEST(FMCWRangeDopplerProcessorOpTest, ParityAllClutterFiltersFloat) {
  parity_all_clutter_filters<float>(1e-5);
}

TEST(FMCWRangeDopplerProcessorOpTest, UlpParityAllClutterFiltersFloat) {
  ulp_parity_all_clutter_filters<float>();
}

TEST(FMCWRangeDopplerProcessorOpTest, UlpParityAllClutterFiltersDouble) {
  ulp_parity_all_clutter_filters<double>();
}

TEST(FMCWRangeDopplerProcessorOpTest, UlpParityNfftDoppler) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  cfg.nfft_doppler = 2 * kChirps;
  auto make = [] {
    return complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape);
  };
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity_ulp<float>(fused, split, "nfft_doppler=2x");
}

TEST(FMCWRangeDopplerProcessorOpTest, NfftPoolUsesMaxDopplerBins) {
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_chirps = kChirps;
  extents.max_doppler_bins = 2 * kChirps;
  extents.max_range_bins = kSamples;
  FMCWRangeDopplerProcessorParams params;
  params.nfft_doppler = 2 * kChirps;
  const auto result = run_fused_result<float>(
      complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape), params, false, extents);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(tensor_frames(result, "rd_maps").front().shape,
            (std::vector<int64_t>{2 * kChirps, kSamples}));
}

TEST(FMCWRangeDopplerProcessorOpTest, UlpParityRealIf) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  const sim::SimResult& sim_result = parity_sim_real();
  const std::vector<double> real_cube = sim::to_real_if(sim_result);
  auto make = [&] {
    return real_cube_to_device<float>(real_cube, sim_result.shape);
  };
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity_ulp<float>(fused, split, "real-IF input");
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityAllClutterFiltersDouble) {
  parity_all_clutter_filters<double>(1e-12);
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityNfftOverrides) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  cfg.nfft_range = 2 * kSamples;
  cfg.nfft_doppler = 2 * kChirps;
  auto make = [] {
    return complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape);
  };
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "nfft_range/doppler=2x");
  EXPECT_EQ(fused.rd.shape, (std::vector<int64_t>{2 * kChirps, 2 * kSamples}));
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityWindows) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  ParityConfig cfg;
  cfg.range_window.type = WindowType::kHann;
  cfg.doppler_window.type = WindowType::kHamming;
  auto make = [] {
    return complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape);
  };
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "hann range + hamming doppler");
}

TEST(FMCWRangeDopplerProcessorOpTest, MatchingWindowRemovesCenterBinForOffBinTone) {
  std::vector<std::complex<float>> cube(static_cast<size_t>(kChirps * kSamples));
  constexpr double kCyclesPerChirp = 0.137;
  for (int64_t chirp = 0; chirp < kChirps; ++chirp) {
    const auto tone =
        std::polar(1.0, 2.0 * holoscan::radar::kPi * kCyclesPerChirp * static_cast<double>(chirp));
    for (int64_t sample = 0; sample < kSamples; ++sample) {
      cube[static_cast<size_t>(chirp * kSamples + sample)] = std::complex<float>(tone);
    }
  }
  const auto waveform = to_descriptor(sim::SimWaveform{});
  ParityConfig matched;
  matched.clutter = ClutterFilterMethod::kMeanSubtraction;
  matched.doppler_window.type = WindowType::kHann;
  const ChainOutputs weighted = run_fused<float>(
      to_device_tensor(cube, {kChirps, kSamples}, complex_dtype<float>()), matched, waveform);

  ParityConfig uniform = matched;
  uniform.match_clutter_window = false;
  const ChainOutputs legacy = run_split<float>(
      to_device_tensor(cube, {kChirps, kSamples}, complex_dtype<float>()), uniform, waveform);
  const int64_t center = kChirps / 2;
  const double weighted_center = std::abs(weighted.rd.as<std::complex<float>>()[center * kSamples]);
  const double uniform_center = std::abs(legacy.rd.as<std::complex<float>>()[center * kSamples]);
  EXPECT_LT(weighted_center, 1e-3);
  EXPECT_GT(uniform_center, 1e-2);
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityRealIfInput) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  const sim::SimResult& sim_result = parity_sim_real();
  const std::vector<double> real_cube = sim::to_real_if(sim_result);
  auto make = [&] {
    return real_cube_to_device<float>(real_cube, sim_result.shape);
  };
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "real-IF input");
  EXPECT_EQ(fused.rd.shape, (std::vector<int64_t>{kChirps, kSamples / 2 + 1}));
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityApplyDechirp) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  auto make = [] {
    return complex_cube_to_device<float>(parity_sim_pre_mixer().cube, parity_sim_pre_mixer().shape);
  };
  ParityConfig cfg;
  cfg.apply_dechirp = true;
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "apply_dechirp (pre-mixer)");
}

TEST(FMCWRangeDopplerProcessorOpTest, InlineClutterReducesDopplerLength) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  const struct {
    ClutterFilterMethod method;
    int64_t expected_doppler;
  } cases[] = {
      {ClutterFilterMethod::kMeanSubtraction, kChirps},
      {ClutterFilterMethod::kSingleDelay, kChirps - 1},
      {ClutterFilterMethod::kDoubleDelay, kChirps - 2},
  };
  for (const auto& c : cases) {
    ParityConfig cfg;
    cfg.clutter = c.method;
    auto make = [] {
      return complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape);
    };
    const auto fused = run_fused<float>(make(), cfg, waveform);
    EXPECT_EQ(fused.rd.shape, (std::vector<int64_t>{c.expected_doppler, kSamples}))
        << to_string(c.method);
    ASSERT_TRUE(fused.velocity_bins.has_value());
    EXPECT_EQ(fused.velocity_bins->shape, (std::vector<int64_t>{c.expected_doppler}))
        << to_string(c.method);
  }
}

TEST(FMCWRangeDopplerProcessorOpTest, PeakMapsToTruthRangeAndVelocity) {
  sim::SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kSamples;
  sim::PointTarget mover;
  mover.position = sim::Vec3{range_at_bin(40.0), 0.0, 0.0};
  mover.velocity = sim::Vec3{5.0, 0.0, 0.0};
  scene.targets.push_back(mover);
  const sim::SimResult sim_result = sim::simulate(scene);
  ASSERT_EQ(sim_result.truth.size(), 1u);

  const auto waveform = to_descriptor(scene.waveform);
  auto input = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
  const auto fused = run_fused<float>(input, ParityConfig{}, waveform);
  ASSERT_EQ(fused.rd.shape, (std::vector<int64_t>{kChirps, kSamples}));

  const auto* rd = fused.rd.as<std::complex<float>>();
  int64_t peak_doppler = 0;
  int64_t peak_range = 0;
  double peak_mag = -1.0;
  for (int64_t d = 0; d < kChirps; ++d) {
    for (int64_t r = 0; r < kSamples; ++r) {
      const double mag = std::abs(std::complex<double>(rd[d * kSamples + r]));
      if (mag > peak_mag) {
        peak_mag = mag;
        peak_doppler = d;
        peak_range = r;
      }
    }
  }

  using holoscan::radar::testing::expected_doppler_bin;
  using holoscan::radar::testing::expected_range_bin;

  EXPECT_EQ(peak_range,
            expected_range_bin(sim_result.truth[0].range, scene.waveform.slope(),
                               scene.waveform.sample_rate, kSamples, sim::kSimSpeedOfLight));
  EXPECT_EQ(peak_doppler,
            expected_doppler_bin(sim_result.truth[0].radial_velocity, scene.waveform.wavelength(),
                                 scene.waveform.chirp_repetition_interval, kChirps,
                                 /*center_zero_doppler=*/true));

  ASSERT_TRUE(fused.range_bins.has_value());
  ASSERT_TRUE(fused.velocity_bins.has_value());
  const float* range_axis = fused.range_bins->as<float>();
  const float* velocity_axis = fused.velocity_bins->as<float>();
  EXPECT_NEAR(range_axis[peak_range], sim_result.truth[0].range, range_bin_width(kSamples) + 1e-9);
  const double velocity_bin_width = scene.waveform.wavelength() / 2.0 /
                                    scene.waveform.chirp_repetition_interval /
                                    static_cast<double>(kChirps);
  EXPECT_NEAR(velocity_axis[peak_doppler], sim_result.truth[0].radial_velocity,
              velocity_bin_width + 1e-9);
}

TEST(FMCWRangeDopplerProcessorOpTest, CriFoldedPeakPowerRetainsPhysicalRangeLoss) {
  auto make_scene = [](double range) {
    sim::SimScene scene;
    scene.num_chirps = kChirps;
    scene.samples_per_chirp = kSamples;
    scene.range_loss = {sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
    scene.targets = {{sim::Vec3{range, 0.0, 0.0}, sim::Vec3{}, {1.0, 0.0}}};
    return scene;
  };

  const auto near_scene = make_scene(40.0);
  const double ambiguity_range =
      0.5 * near_scene.propagation_speed * near_scene.waveform.chirp_repetition_interval;
  const auto far_scene = make_scene(ambiguity_range + 40.0);
  const auto near_sim = sim::simulate(near_scene);
  const auto far_sim = sim::simulate(far_scene);
  const auto waveform = to_descriptor(near_scene.waveform);

  const auto near = run_fused<float>(complex_cube_to_device<float>(near_sim.cube, near_sim.shape),
                                     ParityConfig{}, waveform);
  const auto far = run_fused<float>(complex_cube_to_device<float>(far_sim.cube, far_sim.shape),
                                    ParityConfig{}, waveform);
  const double near_peak = max_abs<float>(near.rd);
  const double far_peak = max_abs<float>(far.rd);
  const double expected_voltage = 40.0 / (ambiguity_range + 40.0) * 40.0 / (ambiguity_range + 40.0);
  const double expected_power = expected_voltage * expected_voltage;

  ASSERT_GT(near_peak, 0.0);
  EXPECT_NEAR((far_peak * far_peak) / (near_peak * near_peak), expected_power,
              1e-4 * expected_power);
}

TEST(FMCWRangeDopplerProcessorOpTest, RangeLossMakesCriFoldedTargetFallBelowCfar) {
  auto make_scene = [](double range) {
    sim::SimScene scene;
    scene.num_chirps = kChirps;
    scene.samples_per_chirp = kSamples;
    scene.range_loss = {sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
    scene.snr_db = 20.0;
    scene.targets = {{sim::Vec3{range, 0.0, 0.0}, sim::Vec3{}, {1.0, 0.0}}};
    return scene;
  };
  auto near_scene = make_scene(40.0);
  const double ambiguity_range =
      0.5 * near_scene.propagation_speed * near_scene.waveform.chirp_repetition_interval;
  auto far_scene = make_scene(ambiguity_range + 40.0);
  auto near_sim = sim::simulate(near_scene);
  auto far_sim = sim::simulate(far_scene);
  sim::add_thermal_noise(near_sim.cube, near_scene.snr_db, near_scene.seed);
  sim::add_thermal_noise(far_sim.cube, far_scene.snr_db, far_scene.seed);

  ParityConfig processing;
  processing.range_window.type = WindowType::kHann;
  processing.doppler_window.type = WindowType::kHann;
  const auto waveform = to_descriptor(near_scene.waveform);
  const auto near = run_fused<float>(complex_cube_to_device<float>(near_sim.cube, near_sim.shape),
                                     processing, waveform);
  const auto far = run_fused<float>(complex_cube_to_device<float>(far_sim.cube, far_sim.shape),
                                    processing, waveform);

  const auto detection_count = [](const HostTensor& rd) {
    CFARParams params;
    params.cfar_variant = CfarVariant::kCA;
    params.guard_cells = {3, 3};
    params.training_cells = {3, 3};
    params.pfa = 1e-9;
    using Op = CFARDetectorOp<float>;
    auto input = reupload<float>(rd);
    auto fixture = holoscan::testing::create_operator_test<Op>(
        holoscan::radar::testing::default_test_extents());
    fixture.set_params(std::move(params));
    fixture.push(&Op::rd_maps, *input);
    constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
    fixture.capture_snapshot<holoscan::Tensor>("detection_idx", 1U, kMaximumBytes);
    fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
    for (const char* output :
         {"detection_idx", "detection_mask", "detection_score", "noise_estimate", "cfar_power"}) {
      fixture.output_device(output, holoscan::DeviceId{0});
    }
    holoscan::radar::testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    (void)completion.wait_for_output(running, "detection_idx", 1U);
    running.request_stop();
    const auto result = running.finish();
    completion.check(result);
    EXPECT_TRUE(result.ok()) << result.describe_failure();
    const auto frames = tensor_frames(result, "detection_idx");
    EXPECT_EQ(frames.size(), 1U);
    return frames.empty() ? int64_t{-1} : frames.front().shape.front();
  };

  EXPECT_GT(detection_count(near.rd), 0);
  EXPECT_EQ(detection_count(far.rd), 0);
}

TEST(FMCWRangeDopplerProcessorOpTest, OptionalOutputsOnlyWithWaveform) {
  const sim::SimResult& sim_result = parity_sim();
  auto input = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeDopplerProcessorParams params;
  const auto result = run_fused_result<float>(input, std::move(params), false);

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(tensor_frames(result, "rd_maps").size(), 1u);
}

TEST(FMCWRangeDopplerProcessorOpTest, ApplyDechirpWithoutWaveformErrors) {
  const sim::SimResult& sim_result = parity_sim_pre_mixer();
  auto input = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeDopplerProcessorParams params;
  params.apply_dechirp = true;
  const auto result = run_fused_result<float>(input, std::move(params), false);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(tensor_frames(result, "rd_maps").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(FMCWRangeDopplerProcessorOpTest, ApplyDechirpWithRealInputErrors) {
  const sim::SimResult& sim_result = parity_sim_real();
  const auto real_cube = sim::to_real_if(sim_result);
  auto input = real_cube_to_device<float>(real_cube, sim_result.shape);

  FMCWRangeDopplerProcessorParams params;
  params.apply_dechirp = true;
  params.waveform = to_descriptor(sim::SimWaveform{});
  const auto result = run_fused_result<float>(input, std::move(params), false);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(tensor_frames(result, "rd_maps").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(FMCWRangeDopplerProcessorOpTest, NfftRangeTooSmallErrors) {
  const sim::SimResult& sim_result = parity_sim();
  auto input = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeDopplerProcessorParams params;
  params.nfft_range = kSamples / 2;
  const auto result = run_fused_result<float>(input, std::move(params), false);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(tensor_frames(result, "rd_maps").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(FMCWRangeDopplerProcessorOpTest, NfftDopplerTooSmallErrors) {
  const sim::SimResult& sim_result = parity_sim();

  // nfft_doppler below the raw chirp count.
  {
    auto input = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
    FMCWRangeDopplerProcessorParams params;
    params.nfft_doppler = kChirps / 2;
    const auto result = run_fused_result<float>(input, std::move(params), false);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(tensor_frames(result, "rd_maps").empty());
    EXPECT_FALSE(result.diagnostics().empty());
  }

  // nfft_doppler below the post-clutter chirp count.
  {
    auto input = complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
    FMCWRangeDopplerProcessorParams params;
    params.clutter_filter = ClutterFilterMethod::kSingleDelay;
    params.nfft_doppler = kChirps - 2;
    const auto result = run_fused_result<float>(input, std::move(params), false);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(tensor_frames(result, "rd_maps").empty());
    EXPECT_FALSE(result.diagnostics().empty());
  }
}

struct LiveFusedDescriptorTick {
  std::shared_ptr<holoscan::Tensor> cube;
  std::optional<WindowSpec> range_window;
  std::optional<WindowSpec> doppler_window;
  std::optional<FMCWWaveformDescriptor> waveform;
};

holoscan::testing::TestResult run_fused_descriptor_override_graph(
    std::vector<LiveFusedDescriptorTick> ticks, FMCWRangeDopplerProcessorParams params) {
  using Op = FMCWRangeDopplerProcessorOp<float>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto& tick = ticks[index];
    const auto cube = holoscan::testing::input("adc_cube", *tick.cube);
    if (tick.range_window && tick.doppler_window && tick.waveform) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time,
          holoscan::testing::input("range_window", holoscan::radar::to_wire(*tick.range_window)),
          holoscan::testing::input("doppler_window",
                                   holoscan::radar::to_wire(*tick.doppler_window)),
          holoscan::testing::input("waveform", holoscan::radar::to_wire(*tick.waveform)), cube));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(capture_time, cube));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  for (const char* output : {"rd_maps", "range_bins", "velocity_bins"}) {
    fixture.capture_snapshot<holoscan::Tensor>(output, ticks.size(), kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  for (const char* output : {"rd_maps", "range_bins", "velocity_bins"}) {
    (void)completion.wait_for_output(running, output, ticks.size());
  }
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}
TEST(FMCWRangeDopplerProcessorOpTest, LiveWindowAndWaveformOverride) {
  const auto base_waveform = to_descriptor(sim::SimWaveform{});
  sim::SimWaveform changed_sim_waveform{};
  changed_sim_waveform.bandwidth *= 0.5;
  const auto changed_waveform = to_descriptor(changed_sim_waveform);
  WindowSpec range_window;
  range_window.type = WindowType::kHann;
  WindowSpec doppler_window;
  doppler_window.type = WindowType::kHamming;
  auto cube = complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape);
  LiveFusedDescriptorTick base{cube, std::nullopt, std::nullopt, std::nullopt};
  LiveFusedDescriptorTick changed{cube, range_window, doppler_window, changed_waveform};
  FMCWRangeDopplerProcessorParams params;
  params.waveform = base_waveform;
  const auto out = run_fused_descriptor_override_graph({base, changed, base}, std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto rd_maps = tensor_frames(out, "rd_maps");
  const auto range_bins = tensor_frames(out, "range_bins");
  const auto velocity_bins = tensor_frames(out, "velocity_bins");
  ASSERT_EQ(rd_maps.size(), 3u);
  ASSERT_EQ(range_bins.size(), 3u);
  ASSERT_EQ(velocity_bins.size(), 3u);

  ParityConfig expected_cfg;
  expected_cfg.range_window = range_window;
  expected_cfg.doppler_window = doppler_window;
  const ChainOutputs expected =
      run_fused<float>(complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape),
                       expected_cfg, changed_waveform);
  const double reference = max_abs<float>(expected.rd);
  EXPECT_LT(max_abs_diff<float>(rd_maps[1], expected.rd), 1e-5 * reference);
  EXPECT_GT(max_abs_diff<float>(rd_maps[0], rd_maps[1]), 1e-2 * reference);
  // Tick 2 publishes no override, so it reproduces tick 0's Params result.
  EXPECT_LT(max_abs_diff<float>(rd_maps[2], rd_maps[0]), 1e-6 * reference);
  EXPECT_NEAR(range_bins[1].as<float>()[1], 2.0F * range_bins[0].as<float>()[1],
              1e-4F * range_bins[1].as<float>()[1]);
  EXPECT_EQ(range_bins[2].as<float>()[1], range_bins[0].as<float>()[1]);
  EXPECT_EQ(velocity_bins[2].as<float>()[1], velocity_bins[0].as<float>()[1]);
}

TEST(FMCWRangeDopplerProcessorOpTest, LiveClutterFilterAppliesOneTickThenReverts) {
  const auto input = [] {
    return complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape);
  };
  FMCWRangeDopplerProcessorParams params;
  params.clutter_filter = ClutterFilterMethod::kNone;
  using Op = FMCWRangeDopplerProcessorOp<float>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < 3U; ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto cube = input();
    if (index == 1U) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time,
          holoscan::testing::input("clutter_filter", ClutterFilterMethod::kDoubleDelay),
          holoscan::testing::input("adc_cube", *cube)));
    } else {
      fixture.push_frame(
          holoscan::testing::frame_at(capture_time, holoscan::testing::input("adc_cube", *cube)));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("rd_maps", 3U, kMaximumBytes);
  fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
  for (const char* output : {"rd_maps", "range_bins", "velocity_bins"}) {
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "rd_maps", 3U);
  running.request_stop();
  const auto out = running.finish();
  completion.check(out);
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto frames = tensor_frames(out, "rd_maps");
  ASSERT_EQ(frames.size(), 3U);
  EXPECT_EQ(frames[0].shape, (std::vector<int64_t>{kChirps, kSamples}));
  EXPECT_EQ(frames[1].shape, (std::vector<int64_t>{kChirps - 2, kSamples}));
  EXPECT_EQ(frames[2].shape, frames[0].shape);

  ParityConfig expected_cfg;
  expected_cfg.clutter = ClutterFilterMethod::kDoubleDelay;
  const ChainOutputs expected =
      run_fused<float>(input(), expected_cfg, to_descriptor(sim::SimWaveform{}));
  const double reference = std::max(max_abs<float>(expected.rd), 1.0);
  EXPECT_LT(max_abs_diff<float>(frames[1], expected.rd), 1e-5 * reference);
  EXPECT_LT(max_abs_diff<float>(frames[2], frames[0]), 1e-6 * reference);
}

TEST(FMCWRangeDopplerProcessorOpTest, FloatAndDoubleAgree) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  auto make_f = [] {
    return complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape);
  };
  auto make_d = [] {
    return complex_cube_to_device<double>(parity_sim().cube, parity_sim().shape);
  };
  const auto fused_f = run_fused<float>(make_f(), cfg, waveform);
  const auto fused_d = run_fused<double>(make_d(), cfg, waveform);

  ASSERT_EQ(fused_f.rd.shape, fused_d.rd.shape);
  const auto* pf = fused_f.rd.as<std::complex<float>>();
  const auto* pd = fused_d.rd.as<std::complex<double>>();
  const double peak = max_abs<double>(fused_d.rd);
  ASSERT_GT(peak, 0.0);
  for (int64_t i = 0; i < fused_f.rd.size(); ++i) {
    const double mag_f = std::abs(std::complex<double>(pf[i]));
    const double mag_d = std::abs(pd[i]);
    ASSERT_NEAR(mag_f, mag_d, 1e-4 * peak) << "element " << i;
  }
}

}  // namespace
