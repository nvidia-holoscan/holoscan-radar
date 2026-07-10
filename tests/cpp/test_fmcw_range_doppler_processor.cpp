// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for FMCWRangeDopplerProcessorOp.
//
// The core requirement is PARITY: the fused operator must produce rd_maps
// identical (within tolerance) to the split FMCWRangeProcessorOp →
// [ClutterFilterOp] → FMCWDopplerProcessorOp chain for matching parameters,
// with exactly matching range_bins / velocity_bins. The split chain is run
// as three separate SDK operator-test harness runs
// (holoscan::test::create_operator_test via utils/sdk_harness.hpp),
// host-capturing and re-uploading between stages.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/bin_model.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ClutterFilterMethod;
using holoscan::radar::ClutterFilterOp;
using holoscan::radar::FMCWDopplerProcessorOp;
using holoscan::radar::FMCWRangeDopplerProcessorOp;
using holoscan::radar::FMCWRangeProcessorOp;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::testing::complex_cube_to_device;
using holoscan::radar::testing::expected_doppler_bin;
using holoscan::radar::testing::expected_range_bin;
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
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::sim::to_real_if;
using holoscan::radar::sim::Vec3;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

constexpr int64_t kChirps = 32;
constexpr int64_t kSamples = 128;

// Default SimWaveform: 77 GHz, 150 MHz over 20 us, fs = 12.8 MHz — one
// range bin ~2.0 m at nfft = 128.
double range_bin_width(int64_t nfft) {
  const SimWaveform waveform{};
  return range_of_bin(1, waveform.slope(), waveform.sample_rate, nfft,
                      kSimSpeedOfLight);
}

double range_at_bin(double bin) { return bin * range_bin_width(kSamples); }

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

// Parity scene: a static target plus a receding one (the parity
// requirement asks the
// clutter methods to have something to remove and something to keep).
SimScene parity_scene(bool dechirped = true) {
  SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kSamples;
  scene.dechirped = dechirped;
  PointTarget still;
  still.position = Vec3{range_at_bin(20.0), 0.0, 0.0};
  scene.targets.push_back(still);
  PointTarget mover;
  mover.position = Vec3{range_at_bin(40.0), 0.0, 0.0};
  mover.velocity = Vec3{5.0, 0.0, 0.0};  // 5 m/s receding
  scene.targets.push_back(mover);
  return scene;
}

const SimResult& parity_sim() {
  static const SimResult sim = simulate(parity_scene());
  return sim;
}

const SimResult& parity_sim_pre_mixer() {
  static const SimResult sim = simulate(parity_scene(/*dechirped=*/false));
  return sim;
}

using TensorFactory = std::function<std::shared_ptr<holoscan::Tensor>()>;

template <typename T>
TensorFactory complex_factory(const SimResult& sim) {
  return [&sim] { return complex_cube_to_device<T>(sim.cube, sim.shape); };
}

// Re-upload a captured complex host tensor for the next split stage.
template <typename T>
std::shared_ptr<holoscan::Tensor> reupload_complex(const HostTensor& host) {
  const auto* data = host.as<std::complex<T>>();
  std::vector<std::complex<T>> copy(data, data + host.size());
  return to_device_tensor(copy, host.shape,
                          holoscan::radar::complex_dtype<T>());
}

// One parameterization shared verbatim by the fused op and the split chain.
struct ParityConfig {
  ClutterFilterMethod clutter{ClutterFilterMethod::kNone};
  std::optional<int64_t> nfft_range{};
  std::optional<int64_t> nfft_doppler{};
  WindowSpec range_window{};
  WindowSpec doppler_window{};
  bool apply_dechirp{false};
};

struct ChainOutputs {
  HostTensor rd;
  std::optional<HostTensor> range_bins;
  std::optional<HostTensor> velocity_bins;
};

// FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp as
// three SDK harness runs, host round-tripping between stages. The optional
// nfft parameters ride their override ports (equivalent to setting
// the parameter, and avoids per-config argument-list branching).
template <typename T>
ChainOutputs run_split(const TensorFactory& make_input,
                       const ParityConfig& cfg,
                       const FMCWWaveformDescriptor& waveform) {
  ChainOutputs out;
  HostTensor profiles_host;
  {
    TensorMapCapture profiles;
    TensorMapCapture bins;
    auto harness = create_operator_test<FMCWRangeProcessorOp<T>>(
        holoscan::Arg("range_window") = cfg.range_window,
        holoscan::Arg("apply_dechirp") = cfg.apply_dechirp,
        holoscan::Arg("waveform") = waveform);
    harness->template add_input_port<holoscan::TensorMap>(
        "adc_cube", {single_map(make_input(), "adc_cube")});
    if (cfg.nfft_range.has_value()) {
      harness->template add_input_port<int64_t>("nfft", {*cfg.nfft_range});
    }
    harness->template add_output_port<holoscan::TensorMap>(
        "range_profiles", profiles.validators());
    harness->template add_output_port<holoscan::TensorMap>(
        "range_bins", bins.validators());
    harness->run_test();
    profiles_host = std::move(profiles.frames.at(0));
    out.range_bins = std::move(bins.frames.at(0));
  }
  if (cfg.clutter != ClutterFilterMethod::kNone) {
    TensorMapCapture filtered;
    auto harness = create_operator_test<ClutterFilterOp<T>>(
        holoscan::Arg("method") = cfg.clutter);
    harness->template add_input_port<holoscan::TensorMap>(
        "range_profiles",
        {single_map(reupload_complex<T>(profiles_host),
                    "range_profiles")});
    harness->template add_output_port<holoscan::TensorMap>(
        "range_profiles", filtered.validators());
    harness->run_test();
    profiles_host = std::move(filtered.frames.at(0));
  }
  {
    TensorMapCapture rd;
    TensorMapCapture velocity;
    auto harness = create_operator_test<FMCWDopplerProcessorOp<T>>(
        holoscan::Arg("doppler_window") = cfg.doppler_window,
        holoscan::Arg("waveform") = waveform);
    harness->template add_input_port<holoscan::TensorMap>(
        "range_profiles",
        {single_map(reupload_complex<T>(profiles_host),
                    "range_profiles")});
    if (cfg.nfft_doppler.has_value()) {
      harness->template add_input_port<int64_t>("nfft",
                                                {*cfg.nfft_doppler});
    }
    harness->template add_output_port<holoscan::TensorMap>(
        "rd_maps", rd.validators());
    harness->template add_output_port<holoscan::TensorMap>(
        "velocity_bins", velocity.validators());
    harness->run_test();
    out.rd = std::move(rd.frames.at(0));
    out.velocity_bins = std::move(velocity.frames.at(0));
  }
  return out;
}

template <typename T>
ChainOutputs run_fused(const TensorFactory& make_input,
                       const ParityConfig& cfg,
                       const FMCWWaveformDescriptor& waveform) {
  TensorMapCapture rd;
  TensorMapCapture range_bins;
  TensorMapCapture velocity;
  auto harness = create_operator_test<FMCWRangeDopplerProcessorOp<T>>(
      holoscan::Arg("range_window") = cfg.range_window,
      holoscan::Arg("doppler_window") = cfg.doppler_window,
      holoscan::Arg("apply_dechirp") = cfg.apply_dechirp,
      holoscan::Arg("clutter_filter") = cfg.clutter,
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>(
      "adc_cube", {single_map(make_input(), "adc_cube")});
  if (cfg.nfft_range.has_value()) {
    harness->template add_input_port<int64_t>("nfft_range",
                                              {*cfg.nfft_range});
  }
  if (cfg.nfft_doppler.has_value()) {
    harness->template add_input_port<int64_t>("nfft_doppler",
                                              {*cfg.nfft_doppler});
  }
  harness->template add_output_port<holoscan::TensorMap>(
      "rd_maps", rd.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "range_bins", range_bins.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "velocity_bins", velocity.validators());
  harness->run_test();
  ChainOutputs out;
  out.rd = std::move(rd.frames.at(0));
  out.range_bins = std::move(range_bins.frames.at(0));
  out.velocity_bins = std::move(velocity.frames.at(0));
  return out;
}

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
    worst = std::max(worst, std::abs(std::complex<double>(pa[i]) -
                                     std::complex<double>(pb[i])));
  }
  return worst;
}

// Axis outputs must match the split chain exactly (both sides compute the
// axis on the host in double and cast to T — bit-identical).
template <typename T>
void expect_axis_exact(const std::optional<HostTensor>& fused,
                       const std::optional<HostTensor>& split,
                       const std::string& label) {
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
void expect_parity(const ChainOutputs& fused, const ChainOutputs& split,
                   double rel_tol, const std::string& label) {
  ASSERT_EQ(fused.rd.shape, split.rd.shape) << label;
  const double reference = max_abs<T>(split.rd);
  ASSERT_GT(reference, 0.0) << label;
  const double delta = max_abs_diff<T>(fused.rd, split.rd) / reference;
  EXPECT_LT(delta, rel_tol) << label;
  std::printf("[ parity ] %-40s rel delta = %.3e\n", label.c_str(), delta);
  expect_axis_exact<T>(fused.range_bins, split.range_bins,
                       label + " range_bins");
  expect_axis_exact<T>(fused.velocity_bins, split.velocity_bins,
                       label + " velocity_bins");
}

template <typename T>
void parity_all_clutter_filters(double rel_tol) {
  const auto waveform = to_descriptor(SimWaveform{});
  const auto factory = complex_factory<T>(parity_sim());
  const ClutterFilterMethod methods[] = {
      ClutterFilterMethod::kNone, ClutterFilterMethod::kMeanSubtraction,
      ClutterFilterMethod::kSingleDelay, ClutterFilterMethod::kDoubleDelay};
  for (ClutterFilterMethod method : methods) {
    ParityConfig cfg;
    cfg.clutter = method;
    const auto fused = run_fused<T>(factory, cfg, waveform);
    const auto split = run_split<T>(factory, cfg, waveform);
    expect_parity<T>(fused, split, rel_tol,
                     std::string("clutter=") +
                         std::string(to_string(method)));
  }
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityAllClutterFiltersFloat) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  parity_all_clutter_filters<float>(1e-5);
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityAllClutterFiltersDouble) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  parity_all_clutter_filters<double>(1e-12);
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityNfftOverrides) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto waveform = to_descriptor(SimWaveform{});
  const auto factory = complex_factory<float>(parity_sim());
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  cfg.nfft_range = 2 * kSamples;
  cfg.nfft_doppler = 2 * kChirps;
  const auto fused = run_fused<float>(factory, cfg, waveform);
  const auto split = run_split<float>(factory, cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "nfft_range/doppler=2x");
  EXPECT_EQ(fused.rd.shape,
            (std::vector<int64_t>{2 * kChirps, 2 * kSamples}));
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityWindows) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto waveform = to_descriptor(SimWaveform{});
  const auto factory = complex_factory<float>(parity_sim());
  ParityConfig cfg;
  cfg.range_window.type = WindowType::kHann;
  cfg.doppler_window.type = WindowType::kHamming;
  const auto fused = run_fused<float>(factory, cfg, waveform);
  const auto split = run_split<float>(factory, cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "hann range + hamming doppler");
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityRealIfInput) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto waveform = to_descriptor(SimWaveform{});
  const SimResult& sim = parity_sim();
  const std::vector<double> real_cube = to_real_if(sim);
  const TensorFactory factory = [&] {
    return real_cube_to_device<float>(real_cube, sim.shape);
  };
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  const auto fused = run_fused<float>(factory, cfg, waveform);
  const auto split = run_split<float>(factory, cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "real-IF input");
  // Real IF halves the range axis: samples/2 + 1 bins.
  EXPECT_EQ(fused.rd.shape,
            (std::vector<int64_t>{kChirps, kSamples / 2 + 1}));
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityApplyDechirp) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto waveform = to_descriptor(SimWaveform{});
  const auto factory = complex_factory<float>(parity_sim_pre_mixer());
  ParityConfig cfg;
  cfg.apply_dechirp = true;
  const auto fused = run_fused<float>(factory, cfg, waveform);
  const auto split = run_split<float>(factory, cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "apply_dechirp (pre-mixer)");
}

// Inline clutter reduces num_doppler exactly as the standalone op's
// num_chirps_out contract prescribes (nfft_doppler unset).
TEST(FMCWRangeDopplerProcessorOpTest, InlineClutterReducesDopplerLength) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto waveform = to_descriptor(SimWaveform{});
  const auto factory = complex_factory<float>(parity_sim());
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
    const auto fused = run_fused<float>(factory, cfg, waveform);
    EXPECT_EQ(fused.rd.shape,
              (std::vector<int64_t>{c.expected_doppler, kSamples}))
        << to_string(c.method);
    ASSERT_TRUE(fused.velocity_bins.has_value());
    EXPECT_EQ(fused.velocity_bins->shape,
              (std::vector<int64_t>{c.expected_doppler}))
        << to_string(c.method);
  }
}

// Physical-domain check: the fused rd_maps peak maps through its OWN
// range_bins / velocity_bins to the target's range and radial velocity
// within one bin each.
TEST(FMCWRangeDopplerProcessorOpTest, PeakMapsToTruthRangeAndVelocity) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kSamples;
  PointTarget mover;
  mover.position = Vec3{range_at_bin(40.0), 0.0, 0.0};
  mover.velocity = Vec3{5.0, 0.0, 0.0};
  scene.targets.push_back(mover);
  const SimResult sim = simulate(scene);
  ASSERT_EQ(sim.truth.size(), 1u);

  const auto waveform = to_descriptor(scene.waveform);
  const auto fused = run_fused<float>(complex_factory<float>(sim),
                                      ParityConfig{}, waveform);
  ASSERT_EQ(fused.rd.shape, (std::vector<int64_t>{kChirps, kSamples}));

  // Global argmax over the (doppler, range) map.
  const auto* rd = fused.rd.as<std::complex<float>>();
  int64_t peak_doppler = 0;
  int64_t peak_range = 0;
  double peak_mag = -1.0;
  for (int64_t d = 0; d < kChirps; ++d) {
    for (int64_t r = 0; r < kSamples; ++r) {
      const double mag =
          std::abs(std::complex<double>(rd[d * kSamples + r]));
      if (mag > peak_mag) {
        peak_mag = mag;
        peak_doppler = d;
        peak_range = r;
      }
    }
  }

  // Cross-check the peak cell against the independent bin model.
  EXPECT_EQ(peak_range,
            expected_range_bin(sim.truth[0].range, scene.waveform.slope(),
                               scene.waveform.sample_rate, kSamples,
                               kSimSpeedOfLight));
  EXPECT_EQ(peak_doppler,
            expected_doppler_bin(sim.truth[0].radial_velocity,
                                 scene.waveform.wavelength(),
                                 scene.waveform.chirp_repetition_interval,
                                 kChirps, /*center_zero_doppler=*/true));

  // Read the peak through the operator's own axes: within one bin of the
  // simulator ground truth.
  ASSERT_TRUE(fused.range_bins.has_value());
  ASSERT_TRUE(fused.velocity_bins.has_value());
  const float* range_axis = fused.range_bins->as<float>();
  const float* velocity_axis = fused.velocity_bins->as<float>();
  EXPECT_NEAR(range_axis[peak_range], sim.truth[0].range,
              range_bin_width(kSamples) + 1e-9);
  const double velocity_bin_width = scene.waveform.wavelength() / 2.0 /
                                    scene.waveform.chirp_repetition_interval /
                                    static_cast<double>(kChirps);
  EXPECT_NEAR(velocity_axis[peak_doppler], sim.truth[0].radial_velocity,
              velocity_bin_width + 1e-9);
}

TEST(FMCWRangeDopplerProcessorOpTest, OptionalOutputsOnlyWithWaveform) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const SimResult& sim = parity_sim();
  TensorMapCapture rd;
  auto rd_validators = rd.validators();
  rd_validators.push_back([](const holoscan::TensorMap& map) {
    EXPECT_EQ(map.count("rd_maps"), 1u);
  });
  auto harness = create_operator_test<FMCWRangeDopplerProcessorOp<float>>();
  harness->add_input_port<holoscan::TensorMap>(
      "adc_cube",
      {single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                  "adc_cube")});
  harness->add_output_port<holoscan::TensorMap>("rd_maps", rd_validators);
  // No validators on the axis ports: the SDK sink then tolerates absence,
  // and get_received_count() asserts nothing arrived without a waveform.
  harness->add_output_port<holoscan::TensorMap>("range_bins");
  harness->add_output_port<holoscan::TensorMap>("velocity_bins");
  harness->run_test();

  ASSERT_EQ(rd.frames.size(), 1u);
  EXPECT_EQ(harness->get_sink<holoscan::TensorMap>("range_bins")
                ->get_received_count(),
            0u);
  EXPECT_EQ(harness->get_sink<holoscan::TensorMap>("velocity_bins")
                ->get_received_count(),
            0u);
}

TEST(FMCWRangeDopplerProcessorOpTest, ApplyDechirpWithoutWaveformErrors) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const SimResult& sim = parity_sim_pre_mixer();
  auto harness = create_operator_test<FMCWRangeDopplerProcessorOp<float>>(
      holoscan::Arg("apply_dechirp") = true);
  harness->add_input_port<holoscan::TensorMap>(
      "adc_cube",
      {single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                  "adc_cube")});
  harness->add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_ANY_THROW(harness->run_test());
}

TEST(FMCWRangeDopplerProcessorOpTest, ApplyDechirpWithRealInputErrors) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const SimResult& sim = parity_sim();
  const auto real_cube = to_real_if(sim);
  auto harness = create_operator_test<FMCWRangeDopplerProcessorOp<float>>(
      holoscan::Arg("apply_dechirp") = true,
      holoscan::Arg("waveform") = to_descriptor(SimWaveform{}));
  harness->add_input_port<holoscan::TensorMap>(
      "adc_cube",
      {single_map(real_cube_to_device<float>(real_cube, sim.shape),
                  "adc_cube")});
  harness->add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_ANY_THROW(harness->run_test());
}

TEST(FMCWRangeDopplerProcessorOpTest, NfftRangeTooSmallErrors) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const SimResult& sim = parity_sim();
  auto harness = create_operator_test<FMCWRangeDopplerProcessorOp<float>>(
      holoscan::Arg("nfft_range") = kSamples / 2);
  harness->add_input_port<holoscan::TensorMap>(
      "adc_cube",
      {single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                  "adc_cube")});
  harness->add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_ANY_THROW(harness->run_test());
}

TEST(FMCWRangeDopplerProcessorOpTest, NfftDopplerTooSmallErrors) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const SimResult& sim = parity_sim();
  // nfft_doppler below the raw chirp count.
  {
    auto harness = create_operator_test<FMCWRangeDopplerProcessorOp<float>>(
        holoscan::Arg("nfft_doppler") = kChirps / 2);
    harness->add_input_port<holoscan::TensorMap>(
        "adc_cube",
        {single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                    "adc_cube")});
    harness->add_output_port<holoscan::TensorMap>("rd_maps");
    EXPECT_ANY_THROW(harness->run_test());
  }
  // nfft_doppler >= the raw chirp count but below the POST-CLUTTER chirp
  // count minus the delay-canceller loss... i.e. below chirps_out.
  {
    auto harness = create_operator_test<FMCWRangeDopplerProcessorOp<float>>(
        holoscan::Arg("clutter_filter") = ClutterFilterMethod::kSingleDelay,
        holoscan::Arg("nfft_doppler") = kChirps - 2);
    harness->add_input_port<holoscan::TensorMap>(
        "adc_cube",
        {single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                    "adc_cube")});
    harness->add_output_port<holoscan::TensorMap>("rd_maps");
    EXPECT_ANY_THROW(harness->run_test());
  }
}

// A clutter_filter override mid-stream changes the output Doppler
// length (downstream re-infers shapes each tick). Overrides ride the
// LEADING ticks (wire_leading_override), and the last one stays effective
// for the trailing message-less tick (sticky-override rule).
// range_bins is left unconnected: its port condition is kNone, so
// emission to it never gates scheduling.
class LiveClutterOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  std::vector<ClutterFilterMethod> overrides;
  FMCWWaveformDescriptor waveform;
  TensorMapCapture* rd = nullptr;
  TensorMapCapture* velocity = nullptr;

  void compose() override {
    auto op = make_operator<FMCWRangeDopplerProcessorOp<float>>(
        "op_under_test", holoscan::Arg("waveform") = waveform);
    wire_data_source(this, op, "adc_cube", frames);
    wire_leading_override<ClutterFilterMethod>(this, op, "clutter_filter",
                                               overrides);
    wire_capture_sink(this, op, "rd_maps", rd);
    wire_capture_sink(this, op, "velocity_bins", velocity);
  }
};

TEST(FMCWRangeDopplerProcessorOpTest, LiveClutterOverrideChangesShape) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const SimResult& sim = parity_sim();
  auto frame = [&] {
    return single_map(complex_cube_to_device<float>(sim.cube, sim.shape),
                      "adc_cube");
  };

  TensorMapCapture rd;
  TensorMapCapture velocity;
  auto app = holoscan::make_application<LiveClutterOverrideApp>();
  app->frames = {frame(), frame(), frame()};
  // Tick 0: explicit none (the parameter default). Tick 1: override to
  // single_delay mid-stream. Tick 2: no message — the tick-1 override is
  // sticky.
  app->overrides = {ClutterFilterMethod::kNone,
                    ClutterFilterMethod::kSingleDelay};
  app->waveform = to_descriptor(SimWaveform{});
  app->rd = &rd;
  app->velocity = &velocity;
  app->run();

  ASSERT_EQ(rd.frames.size(), 3u);
  EXPECT_EQ(rd.frames[0].shape,
            (std::vector<int64_t>{kChirps, kSamples}));
  EXPECT_EQ(rd.frames[1].shape,
            (std::vector<int64_t>{kChirps - 1, kSamples}));
  EXPECT_EQ(rd.frames[2].shape,
            (std::vector<int64_t>{kChirps - 1, kSamples}));
  ASSERT_EQ(velocity.frames.size(), 3u);
  EXPECT_EQ(velocity.frames[0].shape, (std::vector<int64_t>{kChirps}));
  EXPECT_EQ(velocity.frames[1].shape,
            (std::vector<int64_t>{kChirps - 1}));
  EXPECT_EQ(velocity.frames[2].shape,
            (std::vector<int64_t>{kChirps - 1}));
}

TEST(FMCWRangeDopplerProcessorOpTest, FloatAndDoubleAgree) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto waveform = to_descriptor(SimWaveform{});
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  const auto fused_f = run_fused<float>(complex_factory<float>(parity_sim()),
                                        cfg, waveform);
  const auto fused_d =
      run_fused<double>(complex_factory<double>(parity_sim()), cfg,
                        waveform);

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
