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
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/bin_model.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/sim_reference.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

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
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::run_operator_graph;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;
using holoscan::radar::testing::run_single_in_three_out_operator_graph_result;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::ThreeOutGraphRunResult;
using holoscan::radar::testing::to_device_tensor;
namespace sim = holoscan::radar::sim;

constexpr int64_t kChirps = 32;
constexpr int64_t kSamples = 128;

double range_bin_width(int64_t nfft) {
  const sim::SimWaveform waveform{};
  return holoscan::radar::testing::range_of_bin(
      1, waveform.slope(), waveform.sample_rate, nfft,
      sim::kSimSpeedOfLight);
}

double range_at_bin(double bin) { return bin * range_bin_width(kSamples); }

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

const sim::SimResult& parity_sim_pre_mixer() {
  static const sim::SimResult result =
      sim::simulate(parity_scene(/*dechirped=*/false));
  return result;
}

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

// ---- Wiring lambdas for Graph harness ----

auto wire_range_dual() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink0, const auto& sink1) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->adc_cube, q);
    graph.add_flow(op->range_profiles, sink0->in, q);
    graph.add_flow(op->range_bins, sink1->in, q);
  };
}

auto wire_clutter_single() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->range_profiles, q);
    graph.add_flow(op->range_profiles_out, sink->in, q);
  };
}

auto wire_doppler_dual() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink0, const auto& sink1) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->range_profiles, q);
    graph.add_flow(op->rd_maps, sink0->in, q);
    graph.add_flow(op->velocity_bins, sink1->in, q);
  };
}

auto wire_fused_triple() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink0, const auto& sink1, const auto& sink2) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->adc_cube, q);
    graph.add_flow(op->rd_maps, sink0->in, q);
    graph.add_flow(op->range_bins, sink1->in, q);
    graph.add_flow(op->velocity_bins, sink2->in, q);
  };
}

auto wire_fused_rd_only() {
  return [](holoscan::Graph& graph, const auto& source, auto& op,
            const auto& sink) {
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(source->out, op->adc_cube, q);
    graph.add_flow(op->rd_maps, sink->in, q);
  };
}

// ---- Split chain via ported Graph harness ----

template <typename T>
std::shared_ptr<holoscan::Tensor> reupload(const HostTensor& host) {
  const auto* data = host.as<std::complex<T>>();
  std::vector<std::complex<T>> copy(data, data + host.size());
  return to_device_tensor(copy, host.shape, complex_dtype<T>());
}

template <typename T>
ChainOutputs run_split(
    const std::shared_ptr<holoscan::Tensor>& input_tensor,
    const ParityConfig& cfg,
    const FMCWWaveformDescriptor& waveform) {
  ChainOutputs out;

  // 1. Range stage (dual-out: profiles + range_bins).
  HostTensor profiles_host;
  {
    FMCWRangeProcessorParams rp;
    rp.range_window = cfg.range_window;
    rp.apply_dechirp = cfg.apply_dechirp;
    rp.waveform = waveform;
    if (cfg.nfft_range.has_value()) {
      rp.nfft = *cfg.nfft_range;
    }
    DualIOGraphRunResult result =
        run_single_in_dual_out_operator_graph_result<
            FMCWRangeProcessorOp<T>>(
            {input_tensor}, std::move(rp),
            "range_profiles", "range_bins", wire_range_dual());
    EXPECT_TRUE(result.plan_ok) << result.plan_json;
    EXPECT_TRUE(result.diagnostics.empty());
    EXPECT_EQ(result.frames0.size(), 1u);
    EXPECT_EQ(result.frames1.size(), 1u);
    profiles_host = std::move(result.frames0[0]);
    out.range_bins = std::move(result.frames1[0]);
  }

  // 2. Optional clutter filter.
  if (cfg.clutter != ClutterFilterMethod::kNone) {
    ClutterFilterParams cp;
    cp.method = cfg.clutter;
    auto frames = run_operator_graph<ClutterFilterOp<T>>(
        {reupload<T>(profiles_host)}, std::move(cp));
    EXPECT_EQ(frames.size(), 1u);
    profiles_host = std::move(frames[0]);
  }

  // 3. Doppler stage (dual-out: rd_maps + velocity_bins).
  {
    FMCWDopplerProcessorParams dp;
    dp.doppler_window = cfg.doppler_window;
    dp.waveform = waveform;
    if (cfg.nfft_doppler.has_value()) {
      dp.nfft = *cfg.nfft_doppler;
    }
    DualIOGraphRunResult result =
        run_single_in_dual_out_operator_graph_result<
            FMCWDopplerProcessorOp<T>>(
            {reupload<T>(profiles_host)}, std::move(dp),
            "rd_maps", "velocity_bins", wire_doppler_dual());
    EXPECT_TRUE(result.plan_ok) << result.plan_json;
    EXPECT_TRUE(result.diagnostics.empty());
    EXPECT_EQ(result.frames0.size(), 1u);
    EXPECT_EQ(result.frames1.size(), 1u);
    out.rd = std::move(result.frames0[0]);
    out.velocity_bins = std::move(result.frames1[0]);
  }
  return out;
}

// ---- Fused op via Graph harness ----

template <typename T>
ChainOutputs run_fused(
    const std::shared_ptr<holoscan::Tensor>& input_tensor,
    const ParityConfig& cfg,
    const FMCWWaveformDescriptor& waveform) {
  FMCWRangeDopplerProcessorParams params;
  params.range_window = cfg.range_window;
  params.doppler_window = cfg.doppler_window;
  params.apply_dechirp = cfg.apply_dechirp;
  params.clutter_filter = cfg.clutter;
  params.waveform = waveform;
  if (cfg.nfft_range.has_value()) {
    params.nfft_range = *cfg.nfft_range;
  }
  if (cfg.nfft_doppler.has_value()) {
    params.nfft_doppler = *cfg.nfft_doppler;
  }

  ThreeOutGraphRunResult result =
      run_single_in_three_out_operator_graph_result<
          FMCWRangeDopplerProcessorOp<T>>(
          {input_tensor}, std::move(params),
          "rd_maps", "range_bins", "velocity_bins",
          wire_fused_triple());

  EXPECT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.diagnostics.empty());
  EXPECT_EQ(result.frames0.size(), 1u);
  EXPECT_EQ(result.frames1.size(), 1u);
  EXPECT_EQ(result.frames2.size(), 1u);

  ChainOutputs out;
  out.rd = std::move(result.frames0[0]);
  out.range_bins = std::move(result.frames1[0]);
  out.velocity_bins = std::move(result.frames2[0]);
  return out;
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
    worst = std::max(worst, std::abs(std::complex<double>(pa[i]) -
                                     std::complex<double>(pb[i])));
  }
  return worst;
}

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

constexpr double kParityUlps = 8.0;

template <typename T>
void expect_parity_ulp(const ChainOutputs& fused, const ChainOutputs& split,
                       const std::string& label) {
  ASSERT_EQ(fused.rd.shape, split.rd.shape) << label;
  const double reference = max_abs<T>(split.rd);
  ASSERT_GT(reference, 0.0) << label;
  const double delta = max_abs_diff<T>(fused.rd, split.rd) / reference;
  const double tol = kParityUlps * std::numeric_limits<T>::epsilon();
  EXPECT_LT(delta, tol)
      << label << ": rel delta = " << delta << ", tolerance = " << tol;
  expect_axis_exact<T>(fused.range_bins, split.range_bins,
                       label + " range_bins");
  expect_axis_exact<T>(fused.velocity_bins, split.velocity_bins,
                       label + " velocity_bins");
}

// ---- Parametrized test helpers ----

template <typename T>
void ulp_parity_all_clutter_filters() {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  auto input = complex_cube_to_device<T>(parity_sim().cube,
                                          parity_sim().shape);
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
        return complex_cube_to_device<T>(parity_sim().cube,
                                          parity_sim().shape);
      };
      const auto fused = run_fused<T>(make(), cfg, waveform);
      const auto split = run_split<T>(make(), cfg, waveform);
      expect_parity_ulp<T>(
          fused, split,
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
      return complex_cube_to_device<T>(parity_sim().cube,
                                        parity_sim().shape);
    };
    const auto fused = run_fused<T>(make(), cfg, waveform);
    const auto split = run_split<T>(make(), cfg, waveform);
    expect_parity<T>(fused, split, rel_tol,
                     std::string("clutter=") +
                         std::string(to_string(method)));
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
    return complex_cube_to_device<float>(parity_sim().cube,
                                          parity_sim().shape);
  };
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity_ulp<float>(fused, split, "nfft_doppler=2x");
}

TEST(FMCWRangeDopplerProcessorOpTest, UlpParityRealIf) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  const sim::SimResult& sim_result = parity_sim();
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
    return complex_cube_to_device<float>(parity_sim().cube,
                                          parity_sim().shape);
  };
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "nfft_range/doppler=2x");
  EXPECT_EQ(fused.rd.shape,
            (std::vector<int64_t>{2 * kChirps, 2 * kSamples}));
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityWindows) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  ParityConfig cfg;
  cfg.range_window.type = WindowType::kHann;
  cfg.doppler_window.type = WindowType::kHamming;
  auto make = [] {
    return complex_cube_to_device<float>(parity_sim().cube,
                                          parity_sim().shape);
  };
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "hann range + hamming doppler");
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityRealIfInput) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  const sim::SimResult& sim_result = parity_sim();
  const std::vector<double> real_cube = sim::to_real_if(sim_result);
  auto make = [&] {
    return real_cube_to_device<float>(real_cube, sim_result.shape);
  };
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  const auto fused = run_fused<float>(make(), cfg, waveform);
  const auto split = run_split<float>(make(), cfg, waveform);
  expect_parity<float>(fused, split, 1e-5, "real-IF input");
  EXPECT_EQ(fused.rd.shape,
            (std::vector<int64_t>{kChirps, kSamples / 2 + 1}));
}

TEST(FMCWRangeDopplerProcessorOpTest, ParityApplyDechirp) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  auto make = [] {
    return complex_cube_to_device<float>(parity_sim_pre_mixer().cube,
                                          parity_sim_pre_mixer().shape);
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
      return complex_cube_to_device<float>(parity_sim().cube,
                                            parity_sim().shape);
    };
    const auto fused = run_fused<float>(make(), cfg, waveform);
    EXPECT_EQ(fused.rd.shape,
              (std::vector<int64_t>{c.expected_doppler, kSamples}))
        << to_string(c.method);
    ASSERT_TRUE(fused.velocity_bins.has_value());
    EXPECT_EQ(fused.velocity_bins->shape,
              (std::vector<int64_t>{c.expected_doppler}))
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
  auto input =
      complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
  const auto fused = run_fused<float>(input, ParityConfig{}, waveform);
  ASSERT_EQ(fused.rd.shape, (std::vector<int64_t>{kChirps, kSamples}));

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

  using holoscan::radar::testing::expected_doppler_bin;
  using holoscan::radar::testing::expected_range_bin;

  EXPECT_EQ(peak_range,
            expected_range_bin(sim_result.truth[0].range,
                               scene.waveform.slope(),
                               scene.waveform.sample_rate, kSamples,
                               sim::kSimSpeedOfLight));
  EXPECT_EQ(peak_doppler,
            expected_doppler_bin(sim_result.truth[0].radial_velocity,
                                 scene.waveform.wavelength(),
                                 scene.waveform.chirp_repetition_interval,
                                 kChirps, /*center_zero_doppler=*/true));

  ASSERT_TRUE(fused.range_bins.has_value());
  ASSERT_TRUE(fused.velocity_bins.has_value());
  const float* range_axis = fused.range_bins->as<float>();
  const float* velocity_axis = fused.velocity_bins->as<float>();
  EXPECT_NEAR(range_axis[peak_range], sim_result.truth[0].range,
              range_bin_width(kSamples) + 1e-9);
  const double velocity_bin_width = scene.waveform.wavelength() / 2.0 /
                                    scene.waveform.chirp_repetition_interval /
                                    static_cast<double>(kChirps);
  EXPECT_NEAR(velocity_axis[peak_doppler],
              sim_result.truth[0].radial_velocity,
              velocity_bin_width + 1e-9);
}

TEST(FMCWRangeDopplerProcessorOpTest, OptionalOutputsOnlyWithWaveform) {
  const sim::SimResult& sim_result = parity_sim();
  auto input =
      complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeDopplerProcessorParams params;
  GraphRunResult result =
      run_single_io_operator_graph_result<
          FMCWRangeDopplerProcessorOp<float>>(
          {std::move(input)}, std::move(params), "rd_maps",
          wire_fused_rd_only(),
          {"range_bins", "velocity_bins"});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1u);
}

TEST(FMCWRangeDopplerProcessorOpTest, ApplyDechirpWithoutWaveformErrors) {
  const sim::SimResult& sim_result = parity_sim_pre_mixer();
  auto input =
      complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeDopplerProcessorParams params;
  params.apply_dechirp = true;
  GraphRunResult result =
      run_single_io_operator_graph_result<
          FMCWRangeDopplerProcessorOp<float>>(
          {std::move(input)}, std::move(params), "rd_maps",
          wire_fused_rd_only(),
          {"range_bins", "velocity_bins"});

  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(FMCWRangeDopplerProcessorOpTest, ApplyDechirpWithRealInputErrors) {
  const sim::SimResult& sim_result = parity_sim();
  const auto real_cube = sim::to_real_if(sim_result);
  auto input = real_cube_to_device<float>(real_cube, sim_result.shape);

  FMCWRangeDopplerProcessorParams params;
  params.apply_dechirp = true;
  params.waveform = to_descriptor(sim::SimWaveform{});
  GraphRunResult result =
      run_single_io_operator_graph_result<
          FMCWRangeDopplerProcessorOp<float>>(
          {std::move(input)}, std::move(params), "rd_maps",
          wire_fused_rd_only(),
          {"range_bins", "velocity_bins"});

  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(FMCWRangeDopplerProcessorOpTest, NfftRangeTooSmallErrors) {
  const sim::SimResult& sim_result = parity_sim();
  auto input =
      complex_cube_to_device<float>(sim_result.cube, sim_result.shape);

  FMCWRangeDopplerProcessorParams params;
  params.nfft_range = kSamples / 2;
  GraphRunResult result =
      run_single_io_operator_graph_result<
          FMCWRangeDopplerProcessorOp<float>>(
          {std::move(input)}, std::move(params), "rd_maps",
          wire_fused_rd_only(),
          {"range_bins", "velocity_bins"});

  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(FMCWRangeDopplerProcessorOpTest, NfftDopplerTooSmallErrors) {
  const sim::SimResult& sim_result = parity_sim();

  // nfft_doppler below the raw chirp count.
  {
    auto input =
        complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
    FMCWRangeDopplerProcessorParams params;
    params.nfft_doppler = kChirps / 2;
    GraphRunResult result =
        run_single_io_operator_graph_result<
            FMCWRangeDopplerProcessorOp<float>>(
            {std::move(input)}, std::move(params), "rd_maps",
            wire_fused_rd_only(),
            {"range_bins", "velocity_bins"});

    EXPECT_TRUE(result.plan_ok);
    EXPECT_TRUE(result.frames.empty());
    EXPECT_FALSE(result.diagnostics.empty());
  }

  // nfft_doppler below the post-clutter chirp count.
  {
    auto input =
        complex_cube_to_device<float>(sim_result.cube, sim_result.shape);
    FMCWRangeDopplerProcessorParams params;
    params.clutter_filter = ClutterFilterMethod::kSingleDelay;
    params.nfft_doppler = kChirps - 2;
    GraphRunResult result =
        run_single_io_operator_graph_result<
            FMCWRangeDopplerProcessorOp<float>>(
            {std::move(input)}, std::move(params), "rd_maps",
            wire_fused_rd_only(),
            {"range_bins", "velocity_bins"});

    EXPECT_TRUE(result.plan_ok);
    EXPECT_TRUE(result.frames.empty());
    EXPECT_FALSE(result.diagnostics.empty());
  }
}

struct LiveFusedDescriptorTick {
  std::shared_ptr<holoscan::Tensor> cube;
  std::optional<WindowSpec> range_window;
  std::optional<WindowSpec> doppler_window;
  std::optional<FMCWWaveformDescriptor> waveform;
};

struct LiveFusedDescriptorBank {
  static void reset() {
    ticks.clear();
    index = 0;
  }
  static inline std::vector<LiveFusedDescriptorTick> ticks;
  static inline std::size_t index{0};
};

template <typename Wire, typename Element>
holoscan::expected<void, holoscan::Error> emit_fused_descriptor_tensor(
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

class LiveFusedDescriptorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice};
    constexpr holoscan::TensorRepresentation kWindow{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 1U};
    constexpr holoscan::TensorRepresentation kWaveform{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLComplex, 128U, 1U},
        .rank = 1U};
    spec.output(cube, "cube")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, 256ULL * 1024ULL * 1024ULL));
    for (auto* output : {&range_window, &doppler_window}) {
      spec.output(*output)
          .produces_tensor(holoscan::radar::pooled_tensor_output(
              kWindow, 1U, 4096U * sizeof(double)));
    }
    spec.output(waveform, "waveform")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kWaveform, 1U, 4096U * sizeof(std::complex<double>)));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (LiveFusedDescriptorBank::index >=
        LiveFusedDescriptorBank::ticks.size()) {
      return {};
    }
    const auto& tick = LiveFusedDescriptorBank::ticks[
        LiveFusedDescriptorBank::index++];
    const holoscan::TimePoint capture_time = context.activation_time();
    auto publish_window =
        [&](holoscan::Output<holoscan::radar::wire::WindowSpecT>& output,
            const std::optional<WindowSpec>& window)
        -> holoscan::expected<void, holoscan::Error> {
      if (!window.has_value()) {
        return {};
      }
      return emit_fused_descriptor_tensor(
          output, holoscan::radar::to_wire(*window),
          std::span<const double>{window->coefficients}, capture_time);
    };
    auto emitted = publish_window(range_window, tick.range_window);
    if (!emitted) {
      return emitted;
    }
    emitted = publish_window(doppler_window, tick.doppler_window);
    if (!emitted) {
      return emitted;
    }
    if (tick.waveform.has_value()) {
      auto waveform_emitted = emit_fused_descriptor_tensor(
          waveform, holoscan::radar::to_wire(*tick.waveform),
          std::span<const std::complex<double>>{
              tick.waveform->reference_samples()},
          capture_time);
      if (!waveform_emitted) {
        return waveform_emitted;
      }
    }
    return holoscan::radar::testing::detail::emit_bank_frame(
        cube, context, tick.cube, "LiveFusedDescriptorSourceOp");
  }

  holoscan::Output<holoscan::Tensor> cube{"cube"};
  holoscan::Output<holoscan::radar::wire::WindowSpecT>
      range_window{"range_window"};
  holoscan::Output<holoscan::radar::wire::WindowSpecT>
      doppler_window{"doppler_window"};
  holoscan::Output<holoscan::radar::wire::FMCWWaveformT>
      waveform{"waveform"};
};

ThreeOutGraphRunResult run_fused_descriptor_override_graph(
    std::vector<LiveFusedDescriptorTick> ticks,
    FMCWRangeDopplerProcessorParams params) {
  ThreeOutGraphRunResult result;
  holoscan::radar::testing::detail::FrameBank::reset();
  holoscan::radar::testing::detail::FrameBankB::reset();
  holoscan::radar::testing::detail::FrameBankC::reset();
  LiveFusedDescriptorBank::reset();
  LiveFusedDescriptorBank::ticks = std::move(ticks);
  const std::size_t expected = LiveFusedDescriptorBank::ticks.size();
  holoscan::Graph graph{"fused-descriptor-overrides"};
  const auto source = graph.op<LiveFusedDescriptorSourceOp>("source");
  auto op = holoscan::radar::testing::emplace_harness_op<
      FMCWRangeDopplerProcessorOp<float>>(
      graph, holoscan::radar::testing::default_test_extents());
  op.set_params(std::move(params));
  const auto sink0 =
      graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("sink0");
  const auto sink1 =
      graph.op<holoscan::radar::testing::detail::TensorCaptureBOp>("sink1");
  const auto sink2 =
      graph.op<holoscan::radar::testing::detail::TensorCaptureCOp>("sink2");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->range_window, op->range_window_in, q);
  graph.add_flow(source->doppler_window, op->doppler_window_in, q);
  graph.add_flow(source->waveform, op->waveform_in, q);
  graph.add_flow(source->cube, op->adc_cube, q);
  graph.add_flow(op->rd_maps, sink0->in, q);
  graph.add_flow(op->range_bins, sink1->in, q);
  graph.add_flow(op->velocity_bins, sink2->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "cube",
          .device = holoscan::DeviceId{0}});
  for (const char* port : {"rd_maps", "range_bins", "velocity_bins"}) {
    holoscan::radar::testing::detail::bind_op_device_output(options, port);
  }
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
      std::size_t count2 = 0;
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
      {
        const std::lock_guard lock(
            holoscan::radar::testing::detail::FrameBankC::mutex);
        count2 =
            holoscan::radar::testing::detail::FrameBankC::captured.size();
      }
      if (count0 >= expected && count1 >= expected && count2 >= expected) {
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
  {
    const std::lock_guard lock(
        holoscan::radar::testing::detail::FrameBankC::mutex);
    result.frames2 = std::move(
        holoscan::radar::testing::detail::FrameBankC::captured);
  }
  return result;
}

TEST(FMCWRangeDopplerProcessorOpTest,
     LiveClutterOverrideChangesShape) {
  // [EA HSDK5 ONLY] Graph mid-stream sticky override harness deferred —
  // HSDK5_PORT.md §1 (EA device/partition binding issues).
  GTEST_SKIP() << "HSDK 5 EA: live clutter_filter override harness "
                  "deferred (HSDK5_PORT.md §1)";
}

TEST(FMCWRangeDopplerProcessorOpTest,
     LiveWindowAndWaveformOverride) {
  const auto base_waveform = to_descriptor(sim::SimWaveform{});
  sim::SimWaveform changed_sim_waveform{};
  changed_sim_waveform.bandwidth *= 0.5;
  const auto changed_waveform = to_descriptor(changed_sim_waveform);
  WindowSpec range_window;
  range_window.type = WindowType::kHann;
  WindowSpec doppler_window;
  doppler_window.type = WindowType::kHamming;
  auto cube = complex_cube_to_device<float>(
      parity_sim().cube, parity_sim().shape);
  LiveFusedDescriptorTick base{
      cube, std::nullopt, std::nullopt, std::nullopt};
  LiveFusedDescriptorTick changed{
      cube, range_window, doppler_window, changed_waveform};
  FMCWRangeDopplerProcessorParams params;
  params.waveform = base_waveform;
  const ThreeOutGraphRunResult out =
      run_fused_descriptor_override_graph(
          {base, changed, base}, std::move(params));
  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames0.size(), 3u);
  ASSERT_EQ(out.frames1.size(), 3u);
  ASSERT_EQ(out.frames2.size(), 3u);

  ParityConfig expected_cfg;
  expected_cfg.range_window = range_window;
  expected_cfg.doppler_window = doppler_window;
  const ChainOutputs expected = run_fused<float>(
      complex_cube_to_device<float>(parity_sim().cube, parity_sim().shape),
      expected_cfg, changed_waveform);
  const double reference = max_abs<float>(expected.rd);
  EXPECT_LT(max_abs_diff<float>(out.frames0[1], expected.rd),
            1e-5 * reference);
  EXPECT_GT(max_abs_diff<float>(out.frames0[0], out.frames0[1]),
            1e-2 * reference);
  EXPECT_LT(max_abs_diff<float>(out.frames0[1], out.frames0[2]),
            1e-6 * reference);
  EXPECT_NEAR(out.frames1[1].as<float>()[1],
              2.0F * out.frames1[0].as<float>()[1],
              1e-4F * out.frames1[1].as<float>()[1]);
  EXPECT_EQ(out.frames1[1].as<float>()[1],
            out.frames1[2].as<float>()[1]);
  EXPECT_EQ(out.frames2[1].as<float>()[1],
            out.frames2[2].as<float>()[1]);
}

TEST(FMCWRangeDopplerProcessorOpTest, FloatAndDoubleAgree) {
  const auto waveform = to_descriptor(sim::SimWaveform{});
  ParityConfig cfg;
  cfg.clutter = ClutterFilterMethod::kMeanSubtraction;
  auto make_f = [] {
    return complex_cube_to_device<float>(parity_sim().cube,
                                          parity_sim().shape);
  };
  auto make_d = [] {
    return complex_cube_to_device<double>(parity_sim().cube,
                                           parity_sim().shape);
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
