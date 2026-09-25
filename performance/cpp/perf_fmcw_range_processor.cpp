// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FMCWRangeProcessorOp — windowed FFT along
// fast-time. Sweeps the (chirps x samples) size ladder, the range window
// (rectangular / Hann), and a multi-channel batch.
//
// This driver measures Params-selected range_window configurations; it does not
// report mid-session live-override latency. Measurement tool only: no timing
// assertions.
// See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <set>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // v1 characterizes single precision (complex64).

struct SizeSpec {
  std::string id;
  int64_t num_chirps;
  int64_t samples_per_chirp;  // fast-time FFT length (natural nfft)
  std::string role;
};

struct RangeScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t samples_per_chirp{0};
  int64_t channels{1};  // leading batch axis (e.g. rx elements)
  hr::WindowType window{hr::WindowType::kNone};
  bool apply_dechirp{false};
  hr::FftNorm fft_norm{hr::FftNorm::kBackward};
  int64_t nfft{0};
  double pri{25.0e-6};
  std::string live_port;
  std::string override_value;
  bool cached_override{false};
};

// FFT-friendly powers of two so cuFFT takes its fast path; the ladder spans a
// smoke size up to a moderate production cube.
const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 16, 128, "small cube; fast smoke"},
      {"sm", 32, 256, "compact sensor"},
      {"md", 64, 512, "common dev size"},
      {"lg", 128, 1024, "moderate production-scale"},
      {"xl", 256, 2048, "range-heavy moderate (v1 upper bound)"}};
  return ladder;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& s : size_ladder()) {
    if (s.id == id) {
      return &s;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* s = find_size(id);
  if (s == nullptr) {
    std::cerr << "internal error: unknown size id '" << id << "'\n";
    std::abort();
  }
  return *s;
}

std::string window_tag(hr::WindowType w) {
  return w == hr::WindowType::kHann ? "hann" : "rect";
}

hr::WindowSpec window_of(hr::WindowType type) {
  hr::WindowSpec w;
  w.type = type;
  return w;
}

// A fixed, valid waveform descriptor so range_bins is emitted every tick. The
// exact values do not affect timing; they only need to pass descriptor
// validation (all finite and > 0, bandwidth defined).
hr::FMCWWaveformDescriptor make_waveform(double pri = 25.0e-6) {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = pri;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<int64_t> cube_shape(const RangeScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.samples_per_chirp};
  }
  return {scen.num_chirps, scen.samples_per_chirp};
}

// ---------------------------------------------------------------------------
// Graph: adc_cube source -> range op -> terminal (profiles) + null (bins).
// range_bins is always emitted here because we set a valid waveform.
// ---------------------------------------------------------------------------
hr::FftNorm norm_of(const std::string& value) {
  if (value == "backward")
    return hr::FftNorm::kBackward;
  if (value == "ortho")
    return hr::FftNorm::kOrtho;
  if (value == "forward")
    return hr::FftNorm::kForward;
  throw std::invalid_argument("unknown FFT normalization: " + value);
}

hr::WindowType window_type_of(const std::string& value) {
  if (value == "none")
    return hr::WindowType::kNone;
  if (value == "hann")
    return hr::WindowType::kHann;
  if (value == "hamming")
    return hr::WindowType::kHamming;
  throw std::invalid_argument("unknown range window: " + value);
}

perf::TimingCollector run_range_graph(const RangeScenario& scen, const perf::RunPlan& plan) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen));
  const auto configure = [&](holoscan::Graph& graph, const auto& src,
                             holoscan::CompileOptions& options, auto connect) {
    hr::MaxTensorExtents extents;
    extents.max_chirps = scen.num_chirps;
    extents.max_range_bins = std::max<int64_t>(
        scen.live_port == "nfft" ? scen.samples_per_chirp * 2 : scen.samples_per_chirp, scen.nfft);
    if (scen.channels > 1)
      extents.max_rx = scen.channels;
    auto op = graph.op<hr::FMCWRangeProcessorOp<PerfPrecision>>("range", extents);
    hr::FMCWRangeProcessorParams params;
    params.waveform = make_waveform(scen.pri);
    params.range_window = window_of(scen.window);
    params.apply_dechirp = scen.apply_dechirp;
    params.fft_norm = scen.fft_norm;
    params.nfft = scen.nfft;
    op.set_params(std::move(params));
    const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
    const auto ns = graph.op<perf::PerfNullSinkOp>("bins_sink");
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(src->out, op->adc_cube, q);
    connect(graph, src, op, q);
    graph.add_flow(op->range_profiles, sink->in, q);
    graph.add_flow(op->range_bins, ns->in, q);
    perf::bind_tensor_out(options, "range", "range_profiles");
    perf::bind_tensor_out(options, "range", "range_bins");
  };
  if (scen.live_port == "range_window") {
    return perf::run_perf_graph_with_window_override(
        std::move(cube), plan.total_frames, plan.override_at, window_of(scen.window),
        window_of(window_type_of(scen.override_value)), "range-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->range_window_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "apply_dechirp") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.apply_dechirp,
        scen.override_value == "true", "range-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->apply_dechirp_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "waveform") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, hr::to_wire(make_waveform(scen.pri)),
        hr::to_wire(make_waveform(std::stod(scen.override_value))), "range-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->waveform_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "fft_norm") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.fft_norm,
        norm_of(scen.override_value), "range-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->fft_norm_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "nfft") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.nfft,
        int64_t{2 * scen.samples_per_chirp}, "range-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->nfft_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true, scen.cached_override);
  }
  return perf::run_perf_graph(std::move(cube), plan.total_frames, "range-perf",
                              [&](auto& graph, const auto& src, auto& options) {
                                configure(graph, src, options,
                                          [](auto&, const auto&, const auto&, const auto&) {});
                              });
}

void fill_throughput(perf::ResultRow* row, const RangeScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double chirps = static_cast<double>(scen.num_chirps);
  const double n = static_cast<double>(scen.samples_per_chirp);
  row->cells_per_second = batch * chirps * n / sec;
  row->slices_per_second = batch * chirps / sec;

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double in_bytes = batch * chirps * n * csz;
  const double out_bytes = batch * chirps * n * csz + n * sizeof(PerfPrecision);
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const RangeScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "FMCWRangeProcessorOp";
  row.variant = window_tag(scen.window);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_chirps, scen.samples_per_chirp};
  row.input_shape = cube_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.samples_per_chirp;
  row.live_port = scen.live_port;
  return row;
}

RangeScenario make_scenario(hr::WindowType window, const SizeSpec& s, int64_t channels = 1) {
  RangeScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.samples_per_chirp = s.samples_per_chirp;
  sc.channels = channels;
  sc.window = window;
  std::ostringstream id;
  id << "range_" << window_tag(window) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

// ---------------------------------------------------------------------------
// Traits: the operator-specific surface the shared driver plugs into.
// ---------------------------------------------------------------------------
struct RangeTraits : perf::PerfDriverDefaults {
  using Scenario = RangeScenario;

  static const char* tool_name() { return "perf_fmcw_range_processor"; }
  static const char* tool_description() {
    return "FMCWRangeProcessorOp performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Rectangular-window size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Rectangular-window size sweep across an explicit "
           "subset";
  }
  static const char* full_help() {
    return "Full matrix (windows x sizes x channel batch); "
           "live-override timing is not measured";
  }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "fmcw_range_processor.yaml",
        "fmcw_range_processor");
  }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 2)
        throw std::invalid_argument("FMCWRange profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape, shape[0] * shape[1],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }
  static std::vector<RangeScenario> build_profile_matrix(const std::string& scope) {
    const auto profiles = suite();
    std::set<std::string> selected_ids;
    for (const auto& c : profiles.select(scope))
      selected_ids.insert(c.id);
    std::vector<RangeScenario> result;
    for (const auto& c : profiles.cases) {
      size_t i = 0;
      for (; i < perf::kProfiles.size(); ++i) {
        if (c.profile == perf::kProfiles[i])
          break;
      }
      const auto& shape = profiles.shapes.at(i);
      if (shape.size() != 2 || shape[1] < 2)
        throw std::invalid_argument("FMCWRange invalid profile shape");
      perf::require_keys(c.parameters,
                         {"range_window", "apply_dechirp", "fft_norm", "nfft", "pri", "channels"},
                         "parameters");
      if (!c.parameters["range_window"] || !c.parameters["fft_norm"])
        throw std::invalid_argument("FMCWRange needs window and FFT norm");
      const auto window = window_type_of(c.parameters["range_window"].as<std::string>());
      const SizeSpec size{c.profile, shape[0], shape[1], c.profile};
      auto scenario = make_scenario(window, size);
      scenario.id = c.id;
      scenario.size_id = c.profile;
      scenario.fft_norm = norm_of(c.parameters["fft_norm"].as<std::string>());
      if (c.parameters["apply_dechirp"])
        scenario.apply_dechirp = c.parameters["apply_dechirp"].as<bool>();
      if (c.parameters["channels"])
        scenario.channels = c.parameters["channels"].as<int64_t>();
      if (c.parameters["nfft"])
        scenario.nfft = c.parameters["nfft"].as<int64_t>();
      if (c.parameters["pri"])
        scenario.pri = c.parameters["pri"].as<double>();
      if (scenario.channels <= 0 || (scenario.nfft != 0 && scenario.nfft < shape[1]) ||
          scenario.nfft > 2 * shape[1] || scenario.pri <= 20.0e-6)
        throw std::invalid_argument("FMCWRange incompatible parameters");
      if (c.override_value) {
        scenario.live_port = (*c.override_value)["port"].as<std::string>();
        scenario.override_value = (*c.override_value)["value"].as<std::string>();
        if ((*c.override_value)["cache"]) {
          const auto cache = (*c.override_value)["cache"].as<std::string>();
          if (scenario.live_port != "nfft")
            throw std::invalid_argument("cached transition only applies to nfft");
          scenario.cached_override = cache == "cached";
        }
        if (scenario.live_port == "range_window") {
          (void)window_type_of(scenario.override_value);
        } else if (scenario.live_port == "apply_dechirp") {
          if (scenario.override_value != "true" && scenario.override_value != "false")
            throw std::invalid_argument("invalid apply_dechirp override");
        } else if (scenario.live_port == "waveform") {
          if (std::stod(scenario.override_value) <= 20.0e-6)
            throw std::invalid_argument("invalid waveform PRI override");
        } else if (scenario.live_port == "fft_norm") {
          (void)norm_of(scenario.override_value);
        } else if (scenario.live_port == "nfft") {
          if (scenario.override_value != "double")
            throw std::invalid_argument("nfft override must be double");
        } else {
          throw std::invalid_argument("unknown FMCWRange live port");
        }
      }
      if (selected_ids.contains(c.id))
        result.push_back(std::move(scenario));
    }
    return result;
  }

  static void run_scenario(const RangeScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    if (scen.cached_override && rs.warmup_frames < 1)
      throw std::invalid_argument("cached transition requires at least one warmup frame");
    const perf::RunPlan plan = perf::make_run_plan(rs, !scen.live_port.empty());
    const auto samples =
        perf::collect_scenario_samples(rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_range_graph(scen, p);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency including initial "
        "window/FFT plan build; median/p90 are post-warmup "
        "steady state; live range_window override timing is not "
        "measured; "
        "cells/s is complex range bins/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(build_profile_matrix("minimal").front(), warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<RangeTraits>(argc, argv);
}
