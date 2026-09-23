// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FMCWRangeDopplerProcessorOp — fused
// range + optional inline clutter + Doppler in one graph node.  Sweeps
// the (chirps x samples) size ladder, the inline clutter stage (off /
// mean subtraction), and a multi-channel batch.
//
// This driver measures Params-selected clutter-filter configurations; it does
// not report mid-session live-override latency. Measurement tool only: no
// timing assertions.
//
// This tool times the fused node alone; for an end-to-end split vs
// fused comparison (range → [clutter] → Doppler vs this op), use
// perf_fmcw_split_vs_fused — do not sum the three per-operator tools.
// See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
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
  int64_t samples_per_chirp;
  std::string role;
};

struct FusedScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t samples_per_chirp{0};
  int64_t channels{1};
  hr::ClutterFilterMethod clutter{hr::ClutterFilterMethod::kNone};
  hr::WindowType doppler_window{hr::WindowType::kNone};
};

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

std::string clutter_tag(hr::ClutterFilterMethod m) {
  switch (m) {
    case hr::ClutterFilterMethod::kNone:
      return "noclt";
    case hr::ClutterFilterMethod::kMeanSubtraction:
      return "msub";
    case hr::ClutterFilterMethod::kSingleDelay:
      return "sdel";
    case hr::ClutterFilterMethod::kDoubleDelay:
      return "ddel";
  }
  return "unk";
}

hr::FMCWWaveformDescriptor make_waveform() {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25.0e-6;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<int64_t> cube_shape(const FusedScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.samples_per_chirp};
  }
  return {scen.num_chirps, scen.samples_per_chirp};
}

// -------------------------------------------------------------------
// Graph: adc_cube source → fused op → terminal (rd_maps) + null
// (range_bins, velocity_bins).  Both axis outputs are always emitted
// here because a valid waveform is set.
// -------------------------------------------------------------------
perf::TimingCollector run_fused_graph(const FusedScenario& scen, int total_frames) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen));
  return perf::run_perf_graph(
      std::move(cube), total_frames, "rdproc-perf",
      [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_chirps = scen.num_chirps;
        extents.max_range_bins = scen.samples_per_chirp;
        if (scen.channels > 1) {
          extents.max_rx = scen.channels;
        }
        auto op = graph.op<hr::FMCWRangeDopplerProcessorOp<PerfPrecision>>("fused", extents);
        hr::FMCWRangeDopplerProcessorParams params;
        params.waveform = make_waveform();
        params.clutter_filter = scen.clutter;
        params.doppler_window.type = scen.doppler_window;
        op.set_params(std::move(params));
        const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
        const auto rb = graph.op<perf::PerfNullSinkOp>("rb_sink");
        const auto vb = graph.op<perf::PerfNullSinkOp>("vb_sink");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(src->out, op->adc_cube, q);
        graph.add_flow(op->rd_maps, sink->in, q);
        graph.add_flow(op->range_bins, rb->in, q);
        graph.add_flow(op->velocity_bins, vb->in, q);
        perf::bind_tensor_out(options, "fused", "rd_maps");
        perf::bind_tensor_out(options, "fused", "range_bins");
        perf::bind_tensor_out(options, "fused", "velocity_bins");
      });
}

void fill_throughput(perf::ResultRow* row, const FusedScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double chirps = static_cast<double>(scen.num_chirps);
  const double n = static_cast<double>(scen.samples_per_chirp);
  row->cells_per_second = batch * chirps * n / sec;  // complex rd-map cells/s
  row->slices_per_second = batch / sec;              // full cubes/s

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double in_bytes = batch * chirps * n * csz;
  const double out_bytes = batch * chirps * n * csz + (chirps + n) * sizeof(PerfPrecision);
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const FusedScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "FMCWRangeDopplerProcessorOp";
  row.variant = clutter_tag(scen.clutter) + "_" + std::string(hr::to_string(scen.doppler_window));
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_chirps, scen.samples_per_chirp};
  row.input_shape = cube_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.samples_per_chirp;
  row.notes = "live_override=not_measured";
  return row;
}

FusedScenario make_scenario(hr::ClutterFilterMethod clutter, const SizeSpec& s,
                            int64_t channels = 1,
                            hr::WindowType doppler_window = hr::WindowType::kNone) {
  FusedScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.samples_per_chirp = s.samples_per_chirp;
  sc.channels = channels;
  sc.clutter = clutter;
  sc.doppler_window = doppler_window;
  std::ostringstream id;
  id << "rdproc_" << clutter_tag(clutter) << "_" << hr::to_string(doppler_window) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

// -------------------------------------------------------------------
// Traits: the operator-specific surface used by the shared driver.
// -------------------------------------------------------------------
struct FusedTraits : perf::PerfDriverDefaults {
  using Scenario = FusedScenario;

  static const char* tool_name() { return "perf_fmcw_range_doppler_processor"; }
  static const char* tool_description() {
    return "FMCWRangeDopplerProcessorOp (fused) "
           "performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "No-clutter size sweep across the full "
           "ladder";
  }
  static const char* sizes_subset_help() {
    return "No-clutter size sweep across an explicit "
           "subset";
  }
  static const char* full_help() {
    return "Full matrix (clutter modes x sizes x "
           "channel batch); live-override timing "
           "live-override timing is not measured";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_chirps, s.samples_per_chirp};
      def.cells = s.num_chirps * s.samples_per_chirp;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<FusedScenario> build_default_matrix() {
    std::vector<FusedScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(hr::ClutterFilterMethod::kNone, must_find_size(sid)));
    }
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, md));
    out.push_back(
        make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, md, 1, hr::WindowType::kHann));
    out.push_back(make_scenario(hr::ClutterFilterMethod::kNone, md, 4));
    return out;
  }

  static std::vector<FusedScenario> build_size_sweep(const std::vector<std::string>& sizes) {
    std::vector<FusedScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(hr::ClutterFilterMethod::kNone, *s));
    }
    return out;
  }

  static std::vector<FusedScenario> build_full_matrix() {
    std::vector<FusedScenario> out;
    for (const auto& s : size_ladder()) {
      for (auto c : {hr::ClutterFilterMethod::kNone, hr::ClutterFilterMethod::kMeanSubtraction,
                     hr::ClutterFilterMethod::kSingleDelay}) {
        out.push_back(make_scenario(c, s));
      }
      out.push_back(
          make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, s, 1, hr::WindowType::kHann));
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      for (int64_t ch : {4, 16}) {
        out.push_back(make_scenario(hr::ClutterFilterMethod::kNone, s, ch));
      }
    }
    return out;
  }

  static void run_scenario(const FusedScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples =
        perf::collect_scenario_samples(rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_fused_graph(scen, p.total_frames);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency including "
        "both FFT plan builds; median/p90 are "
        "post-warmup steady state; live "
        "clutter_filter override timing is not measured; "
        "cells/s is complex rd-map cells/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(hr::ClutterFilterMethod::kNone, must_find_size("xs")), warm,
                   &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<FusedTraits>(argc, argv);
}
