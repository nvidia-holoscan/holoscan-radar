// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Simulator-backed dense-DDM integration coverage for DopplerNotchOp.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sim/scene.hpp"
#include "utils/bin_model.hpp"
#include "utils/mimo_reference.hpp"
#include "utils/integration_graph_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace hr = holoscan::radar;
namespace rt = holoscan::radar::testing;
namespace sim = holoscan::radar::sim;

constexpr int64_t kNumTx = 2;
constexpr int64_t kNumRx = 2;
constexpr int64_t kNumChirps = 64;
constexpr int64_t kNumSamples = 128;

struct PipelineResult {
  bool plan_ok{false};
  std::string plan_json;
  std::vector<std::string> diagnostics;
  rt::HostTensor maps;
  rt::HostTensor velocity;
};

hr::FMCWWaveformDescriptor waveform_descriptor() {
  hr::FMCWWaveformSpec waveform;
  waveform.carrier_frequency = 77.0e9;
  waveform.bandwidth = 150.0e6;
  waveform.chirp_duration = 20.0e-6;
  waveform.sample_rate = 12.8e6;
  waveform.chirp_repetition_interval = 25.0e-6;
  return hr::FMCWWaveformDescriptor(waveform);
}

sim::SimScene make_scene(double clutter_velocity = 0.0) {
  sim::SimScene scene;
  scene.waveform.carrier_frequency = 77.0e9;
  scene.waveform.bandwidth = 150.0e6;
  scene.waveform.chirp_duration = 20.0e-6;
  scene.waveform.sample_rate = 12.8e6;
  scene.waveform.chirp_repetition_interval = 25.0e-6;
  scene.num_chirps = kNumChirps;
  scene.samples_per_chirp = kNumSamples;
  scene.tx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.5 * scene.waveform.wavelength(), 0.0, 0.0}};
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.0, 0.5 * scene.waveform.wavelength(), 0.0}};
  scene.targets = {{sim::Vec3{0.0, 30.0, 0.0}, sim::Vec3{0.0, clutter_velocity, 0.0}, {4.0, 0.0}},
                   {sim::Vec3{0.0, 45.0, 0.0}, sim::Vec3{0.0, 5.0, 0.0}, {1.0, 0.0}}};
  return scene;
}

std::vector<std::complex<double>> zadoff_chu_codes() {
  std::vector<std::complex<double>> codes(static_cast<size_t>(kNumTx * kNumChirps));
  const std::vector<int64_t> roots{1, 3};
  for (int64_t tx = 0; tx < kNumTx; ++tx) {
    for (int64_t chirp = 0; chirp < kNumChirps; ++chirp) {
      const double k = static_cast<double>(chirp);
      const double phase = -hr::kPi * static_cast<double>(roots[static_cast<size_t>(tx)]) * k * k /
                           static_cast<double>(kNumChirps);
      codes[static_cast<size_t>(tx * kNumChirps + chirp)] = std::polar(1.0, phase);
    }
  }
  return codes;
}

void bind_output(holoscan::CompileOptions* options, const std::string& op, const char* port) {
  options->deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
}

PipelineResult run_pipeline(const sim::SimResult& simulated, const hr::DDMCodeDescriptor& code,
                            bool apply_notch) {
  rt::integration::FrameBank::reset();
  rt::integration::FrameBankB::reset();
  rt::integration::FrameBank::input_frames = {
      rt::complex_cube_to_device<float>(simulated.cube, simulated.shape)};

  const hr::FMCWWaveformDescriptor waveform = waveform_descriptor();
  hr::MaxTensorExtents range_extents;
  range_extents.max_rx = kNumRx;
  range_extents.max_chirps = kNumChirps;
  range_extents.max_samples_per_chirp = kNumSamples;
  range_extents.max_range_bins = kNumSamples;
  hr::MaxTensorExtents doppler_extents = range_extents;
  doppler_extents.max_doppler_bins = kNumChirps;
  hr::MaxTensorExtents demux_extents = doppler_extents;
  demux_extents.max_tx = kNumTx;

  static uint64_t graph_run = 0;
  const std::string run_suffix =
      "-" + std::to_string(static_cast<int64_t>(getpid())) + "-" + std::to_string(graph_run++);
  const std::string graph_name = "ddm-notch-pipeline" + run_suffix;
  const std::string source_name = "source" + run_suffix;
  const std::string range_name = "range" + run_suffix;
  const std::string doppler_name = "doppler" + run_suffix;
  const std::string demux_name = "demux" + run_suffix;
  const std::string notch_name = "notch" + run_suffix;
  const std::string map_sink_name = "map_sink" + run_suffix;
  const std::string axis_sink_name = "axis_sink" + run_suffix;
  holoscan::Graph graph{graph_name};
  const auto source = graph.op<rt::integration::TensorSourceOp>(source_name);
  auto range = graph.op<hr::FMCWRangeProcessorOp<float>>(range_name, range_extents);
  hr::FMCWRangeProcessorParams range_params;
  range_params.waveform = waveform;
  range.set_params(std::move(range_params));

  auto doppler = graph.op<hr::FMCWDopplerProcessorOp<float>>(doppler_name, doppler_extents);
  hr::FMCWDopplerProcessorParams doppler_params;
  doppler_params.waveform = waveform;
  doppler_params.doppler_window.type = hr::WindowType::kHann;
  doppler.set_params(std::move(doppler_params));

  auto demux = graph.op<hr::DDMDenseDemultiplexOp<float>>(demux_name, demux_extents);
  hr::DDMDenseDemultiplexParams demux_params;
  demux_params.ddm_code = code;
  demux.set_params(std::move(demux_params));

  const auto map_sink = graph.op<rt::integration::TensorCaptureOp>(map_sink_name);
  const auto axis_sink = graph.op<rt::integration::TensorCaptureBOp>(axis_sink_name);

  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->out, range->adc_cube, queue);
  graph.add_flow(range->range_profiles, doppler->range_profiles, queue);
  graph.add_flow(doppler->velocity_bins, demux->velocity_bins, queue);
  graph.add_flow(doppler->rd_maps, demux->rd_maps, queue);
  if (apply_notch) {
    auto notch = graph.op<hr::DopplerNotchOp<float>>(notch_name, demux_extents);
    notch.set_params(hr::DopplerNotchParams{});
    graph.add_flow(demux->velocity_bins_out, notch->velocity_bins, queue);
    graph.add_flow(demux->rd_maps_out, notch->rd_maps, queue);
    graph.add_flow(notch->rd_maps_out, map_sink->in, queue);
  } else {
    graph.add_flow(demux->rd_maps_out, map_sink->in, queue);
  }
  graph.add_flow(demux->velocity_bins_out, axis_sink->in, queue);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  bind_output(&options, source_name, "out");
  bind_output(&options, range_name, "range_profiles");
  bind_output(&options, range_name, "range_bins");
  bind_output(&options, doppler_name, "velocity_bins");
  bind_output(&options, doppler_name, "rd_maps");
  bind_output(&options, demux_name, "velocity_bins_out");
  bind_output(&options, demux_name, "rd_maps_out");
  if (apply_notch)
    bind_output(&options, notch_name, "rd_maps_out");

  PipelineResult result;
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  result.plan_ok = plan.ok();
  result.plan_json = plan.json();
  if (!plan.ok())
    return result;

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + rt::integration::kHarnessDeadline;
  while (session.diagnostics().empty() && std::chrono::steady_clock::now() < deadline) {
    std::size_t maps = 0;
    std::size_t axes = 0;
    {
      const std::lock_guard lock(rt::integration::FrameBank::mutex);
      maps = rt::integration::FrameBank::captured.size();
    }
    {
      const std::lock_guard lock(rt::integration::FrameBankB::mutex);
      axes = rt::integration::FrameBankB::captured.size();
    }
    if (maps >= 1U && axes >= 1U)
      break;
    std::this_thread::sleep_for(rt::integration::kHarnessPoll);
  }
  session.request_stop();
  session.wait();
  for (const auto& diagnostic : session.diagnostics()) {
    result.diagnostics.push_back(std::string(diagnostic.token));
  }
  {
    const std::lock_guard lock(rt::integration::FrameBank::mutex);
    if (!rt::integration::FrameBank::captured.empty()) {
      result.maps = std::move(rt::integration::FrameBank::captured.front());
    }
  }
  {
    const std::lock_guard lock(rt::integration::FrameBankB::mutex);
    if (!rt::integration::FrameBankB::captured.empty()) {
      result.velocity = std::move(rt::integration::FrameBankB::captured.front());
    }
  }
  return result;
}

void expect_notch_behavior(const PipelineResult& baseline, const PipelineResult& notched,
                           const sim::SimScene& scene) {
  ASSERT_TRUE(baseline.plan_ok) << baseline.plan_json;
  ASSERT_TRUE(notched.plan_ok) << notched.plan_json;
  ASSERT_TRUE(baseline.diagnostics.empty());
  ASSERT_TRUE(notched.diagnostics.empty());
  ASSERT_FALSE(notched.maps.data.empty());
  ASSERT_EQ(baseline.maps.shape, notched.maps.shape);
  ASSERT_EQ(baseline.velocity.shape, notched.velocity.shape);

  const int64_t num_doppler = notched.maps.shape[2];
  const int64_t num_range = notched.maps.shape[3];
  const auto* axis = notched.velocity.as<float>();
  int64_t zero = 0;
  for (int64_t index = 1; index < num_doppler; ++index) {
    if (std::abs(axis[index]) < std::abs(axis[zero]))
      zero = index;
  }
  const int64_t clutter_range =
      rt::expected_range_bin(30.0, scene.waveform.slope(), scene.waveform.sample_rate,
                             scene.samples_per_chirp, scene.propagation_speed);
  const int64_t moving_range =
      rt::expected_range_bin(45.0, scene.waveform.slope(), scene.waveform.sample_rate,
                             scene.samples_per_chirp, scene.propagation_speed);
  int64_t moving_doppler = 0;
  for (int64_t index = 1; index < num_doppler; ++index) {
    if (std::abs(axis[index] - 5.0F) < std::abs(axis[moving_doppler] - 5.0F)) {
      moving_doppler = index;
    }
  }
  EXPECT_NEAR(axis[zero], 0.0F, 1e-6F);
  EXPECT_NEAR(axis[moving_doppler], 5.0F, std::abs(axis[1] - axis[0]));

  const auto* before = baseline.maps.as<std::complex<float>>();
  const auto* after = notched.maps.as<std::complex<float>>();
  const int64_t cells_per_channel = num_doppler * num_range;
  for (int64_t channel = 0; channel < kNumTx * kNumRx; ++channel) {
    const int64_t clutter_index = channel * cells_per_channel + zero * num_range + clutter_range;
    EXPECT_NEAR(std::abs(after[clutter_index]), 0.0F, 1e-5F);
    const int64_t moving_index =
        channel * cells_per_channel + moving_doppler * num_range + moving_range;
    const float tolerance = 1e-4F * std::abs(before[moving_index]) + 1e-4F;
    EXPECT_NEAR(after[moving_index].real(), before[moving_index].real(), tolerance);
    EXPECT_NEAR(after[moving_index].imag(), before[moving_index].imag(), tolerance);
  }
  for (std::size_t index = 0; index < baseline.maps.data.size() / sizeof(std::complex<float>);
       ++index) {
    EXPECT_LE(std::abs(after[index]), std::abs(before[index]) + 1e-3F);
    const int64_t bin = static_cast<int64_t>(index) / num_range % num_doppler;
    // Default hard notch removes only the nearest zero bin, preserving every other cell.
    const auto expected = bin == zero ? std::complex<float>{} : before[index];
    const float tolerance = 1e-4F * std::abs(before[index]) + 1e-4F;
    EXPECT_NEAR(after[index].real(), expected.real(), tolerance) << "Doppler bin " << bin;
    EXPECT_NEAR(after[index].imag(), expected.imag(), tolerance) << "Doppler bin " << bin;
  }
}

TEST(DopplerNotchPipeline, LinearDdmSuppressesStaticPhysicalBin) {
  const sim::SimScene scene = make_scene();
  hr::DDMCodeDescriptor code;
  code.num_tx = kNumTx;
  code.doppler_offsets = sim::uniform_ddm_offsets(kNumTx);
  const sim::SimResult simulated = sim::simulate_mimo_ddm_coded(scene, code.doppler_offsets);
  expect_notch_behavior(run_pipeline(simulated, code, false), run_pipeline(simulated, code, true),
                        scene);
}

TEST(DopplerNotchPipeline, LinearDdmPreservesNearStaticOutsideCenter) {
  const sim::SimScene scene = make_scene(1.25);
  hr::DDMCodeDescriptor code;
  code.num_tx = kNumTx;
  code.doppler_offsets = sim::uniform_ddm_offsets(kNumTx);
  const sim::SimResult simulated = sim::simulate_mimo_ddm_coded(scene, code.doppler_offsets);
  const PipelineResult baseline = run_pipeline(simulated, code, false);
  const PipelineResult notched = run_pipeline(simulated, code, true);
  ASSERT_TRUE(baseline.plan_ok) << baseline.plan_json;
  ASSERT_TRUE(notched.plan_ok) << notched.plan_json;
  ASSERT_TRUE(notched.diagnostics.empty());
  const auto* before = baseline.maps.as<std::complex<float>>();
  const auto* after = notched.maps.as<std::complex<float>>();
  float before_peak = 0.0F;
  float after_peak = 0.0F;
  std::size_t before_peak_index = 0;
  for (std::size_t index = 0; index < baseline.maps.data.size() / sizeof(std::complex<float>);
       ++index) {
    if (std::abs(before[index]) > before_peak) {
      before_peak = std::abs(before[index]);
      before_peak_index = index;
    }
    after_peak = std::max(after_peak, std::abs(after[index]));
  }
  const int64_t num_doppler = baseline.maps.shape[2];
  const int64_t num_range = baseline.maps.shape[3];
  const int64_t doppler_index =
      static_cast<int64_t>(before_peak_index % static_cast<std::size_t>(num_doppler * num_range)) /
      num_range;
  const auto* axis = baseline.velocity.as<float>();
  EXPECT_NEAR(axis[doppler_index], 1.25F, std::abs(axis[1] - axis[0]));
  EXPECT_GT(after_peak, 0.9F * before_peak);
}

TEST(DopplerNotchPipeline, NonlinearCodesKeepPhysicalZeroBin) {
  const sim::SimScene scene = make_scene();
  const auto codes = zadoff_chu_codes();
  hr::DDMCodeDescriptor code;
  code.num_tx = kNumTx;
  code.phase_codes = codes;
  const sim::SimResult simulated = sim::simulate_mimo_ddm_phase_coded(scene, codes);
  expect_notch_behavior(run_pipeline(simulated, code, false), run_pipeline(simulated, code, true),
                        scene);
}

}  // namespace
