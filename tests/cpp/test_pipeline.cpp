// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cross-cutting end-to-end tests: the full SISO split chain driven by
// the analytic simulator, and
// determinism of the chain under a fixed seed. The chain is fed and
// captured with the SDK test-harness source/sink operators
// (utils/sdk_harness.hpp).

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "utils/bin_model.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ClutterFilterMethod;
using holoscan::radar::ClutterFilterOp;
using holoscan::radar::FMCWDopplerProcessorOp;
using holoscan::radar::FMCWRangeProcessorOp;
using holoscan::radar::FMCWWaveformDescriptor;
namespace rt = holoscan::radar::testing;
namespace psim = holoscan::radar::sim;

FMCWWaveformDescriptor chain_waveform() {
  FMCWWaveformDescriptor waveform;
  waveform.carrier_frequency = 77e9;
  waveform.bandwidth = 150e6;
  waveform.chirp_duration = 20e-6;
  waveform.sample_rate = 12.8e6;
  waveform.chirp_repetition_interval = 25e-6;
  return waveform;
}

psim::SimScene chain_scene() {
  psim::SimScene scene;
  scene.waveform.carrier_frequency = 77e9;
  scene.waveform.bandwidth = 150e6;
  scene.waveform.chirp_duration = 20e-6;
  scene.waveform.sample_rate = 12.8e6;
  scene.waveform.chirp_repetition_interval = 25e-6;
  scene.num_chirps = 64;
  scene.samples_per_chirp = 128;
  scene.snr_db = 30.0;
  scene.seed = 0xC0FFEE;
  // Static clutter (stronger) + moving target.
  scene.targets.push_back(
      {psim::Vec3{0.0, 60.0, 0.0}, psim::Vec3{0.0, 0.0, 0.0}, {2.0, 0.0}});
  scene.targets.push_back(
      {psim::Vec3{0.0, 42.0, 0.0}, psim::Vec3{0.0, 5.5, 0.0}, {1.0, 0.0}});
  return scene;
}

// The full split chain in one application: SDK test-harness source ->
// range -> clutter -> doppler -> capture sinks (rd_maps + both axis
// outputs). The range -> clutter -> doppler add_flow wiring is the real
// chain under test.
class SplitChainApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  rt::TensorMapCapture* rd_maps = nullptr;
  rt::TensorMapCapture* range_bins = nullptr;
  rt::TensorMapCapture* velocity_bins = nullptr;

  void compose() override {
    using holoscan::Arg;
    const FMCWWaveformDescriptor waveform = chain_waveform();

    auto range = make_operator<FMCWRangeProcessorOp<float>>(
        "range", Arg("waveform") = waveform);
    auto clutter = make_operator<ClutterFilterOp<float>>(
        "clutter", Arg("method") = ClutterFilterMethod::kMeanSubtraction);
    auto doppler = make_operator<FMCWDopplerProcessorOp<float>>(
        "doppler", Arg("waveform") = waveform);

    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});

    rt::wire_data_source(this, range, "adc_cube", frames);
    rt::wire_capture_sink(this, doppler, "rd_maps", rd_maps);
    rt::wire_capture_sink(this, range, "range_bins", range_bins);
    rt::wire_capture_sink(this, doppler, "velocity_bins", velocity_bins);
  }
};

struct ChainResult {
  rt::HostTensor rd_map;
  rt::HostTensor range_axis;
  rt::HostTensor velocity_axis;
};

ChainResult run_split_chain(const psim::SimScene& scene) {
  const auto sim = psim::simulate(scene);
  auto cube = rt::complex_cube_to_device<float>(sim.cube, sim.shape);

  rt::TensorMapCapture rd_maps;
  rt::TensorMapCapture range_bins;
  rt::TensorMapCapture velocity_bins;
  auto app = holoscan::make_application<SplitChainApp>();
  app->frames = {rt::single_map(cube, "adc_cube")};
  app->rd_maps = &rd_maps;
  app->range_bins = &range_bins;
  app->velocity_bins = &velocity_bins;
  app->run();

  ChainResult result;
  result.rd_map = std::move(rd_maps.frames.at(0));
  result.range_axis = std::move(range_bins.frames.at(0));
  result.velocity_axis = std::move(velocity_bins.frames.at(0));
  return result;
}

TEST(EndToEndPipeline, SisoChainDetectsMovingTargetAtTruth) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const psim::SimScene scene = chain_scene();
  const ChainResult result = run_split_chain(scene);

  ASSERT_EQ(result.rd_map.shape.size(), 2u);
  const int64_t num_doppler = result.rd_map.shape[0];
  const int64_t num_range = result.rd_map.shape[1];
  EXPECT_EQ(num_doppler, scene.num_chirps);
  EXPECT_EQ(num_range, scene.samples_per_chirp);

  const auto* rd = result.rd_map.as<std::complex<float>>();
  int64_t peak = 0;
  float peak_mag = 0.0F;
  for (int64_t i = 0; i < num_doppler * num_range; ++i) {
    const float mag = std::abs(rd[i]);
    if (mag > peak_mag) {
      peak_mag = mag;
      peak = i;
    }
  }
  const int64_t doppler_bin = peak / num_range;
  const int64_t range_bin = peak % num_range;

  // Physical-domain assertions via the operators' own axis outputs
  // against the simulator's ground truth for the MOVING
  // target — the stronger static return is suppressed by the clutter
  // filter. "Within one bin" uses the axes' own spacing.
  const auto* ranges = result.range_axis.as<float>();
  const auto* velocities = result.velocity_axis.as<float>();
  const double range_spacing = ranges[1] - ranges[0];
  const double velocity_spacing = std::abs(velocities[1] - velocities[0]);
  EXPECT_NEAR(ranges[range_bin], 42.0, range_spacing);
  EXPECT_NEAR(velocities[doppler_bin], 5.5, velocity_spacing);

  // The static return's cell (its range bin at the zero-Doppler column) is
  // well below the moving target's peak.
  const int64_t clutter_range_bin = rt::expected_range_bin(
      60.0, scene.waveform.slope(), scene.waveform.sample_rate,
      scene.samples_per_chirp, scene.propagation_speed);
  const int64_t zero_doppler_bin = rt::expected_doppler_bin(
      0.0, scene.waveform.wavelength(),
      scene.waveform.chirp_repetition_interval, num_doppler,
      /*center_zero_doppler=*/true);
  const float clutter_mag =
      std::abs(rd[zero_doppler_bin * num_range + clutter_range_bin]);
  EXPECT_LT(clutter_mag, 0.05F * peak_mag);
}

TEST(EndToEndPipeline, ChainIsDeterministicForFixedSeed) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const psim::SimScene scene = chain_scene();
  const ChainResult a = run_split_chain(scene);
  const ChainResult b = run_split_chain(scene);
  ASSERT_EQ(a.rd_map.data.size(), b.rd_map.data.size());
  EXPECT_EQ(a.rd_map.data, b.rd_map.data);
  EXPECT_EQ(a.range_axis.data, b.range_axis.data);
  EXPECT_EQ(a.velocity_axis.data, b.velocity_axis.data);
}

}  // namespace
