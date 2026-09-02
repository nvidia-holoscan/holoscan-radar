// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ClutterFilterOp — slow-time static-clutter
// suppression (mean subtraction / 1st- / 2nd-order delay-line MTI). Sweeps the
// (chirps x range_bins) size ladder, the suppression method, and a
// multi-channel batch. Every compute run inlines a mid-session [live] method
// override so the method-switch cost (and output chirp-count change) is
// measured.
//
// [EA HSDK5 ONLY] Graph-based rewrite of the HSDK 4.x Application harness.
// Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.

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

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
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
  int64_t num_range_bins;
  std::string role;
};

struct ClutterScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t num_range_bins{0};
  int64_t channels{1};
  hr::ClutterFilterMethod method{hr::ClutterFilterMethod::kMeanSubtraction};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 16, 128, "small map; fast smoke"},
      {"sm", 32, 256, "compact sensor / few chirps"},
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

std::string method_tag(hr::ClutterFilterMethod m) {
  switch (m) {
    case hr::ClutterFilterMethod::kMeanSubtraction: return "msub";
    case hr::ClutterFilterMethod::kSingleDelay: return "sdel";
    case hr::ClutterFilterMethod::kDoubleDelay: return "ddel";
    case hr::ClutterFilterMethod::kNone: return "none";
  }
  return "unk";
}

// Flip to a different valid method so a live override always changes the
// kernel path (and the output chirp count). kNone is invalid here.
hr::ClutterFilterMethod flip_method(hr::ClutterFilterMethod m) {
  if (m == hr::ClutterFilterMethod::kMeanSubtraction) {
    return hr::ClutterFilterMethod::kSingleDelay;
  }
  return hr::ClutterFilterMethod::kMeanSubtraction;
}

std::vector<int64_t> profiles_shape(const ClutterScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.num_range_bins};
  }
  return {scen.num_chirps, scen.num_range_bins};
}

perf::TimingCollector run_clutter_graph(const ClutterScenario& scen,
                                        const perf::RunPlan& plan) {
  auto cube =
      perf::make_iq_cube_device<PerfPrecision>(profiles_shape(scen));
  hr::MaxTensorExtents extents;
  extents.max_chirps = scen.num_chirps;
  extents.max_range_bins = scen.num_range_bins;
  if (scen.channels > 1) {
    extents.max_rx = scen.channels;
  }
  using BaseOp = hr::ClutterFilterOp<PerfPrecision>;
  // Production Exact{} + per-tick co-emit. Do not wrap PerfLiveContractOp
  // (restacking LatestWithin is TEMPORAL_CONTRACT_INVALID).
  const hr::ClutterFilterMethod override_method = flip_method(scen.method);
  return perf::run_perf_graph_with_override(
      std::move(cube), plan.total_frames, plan.override_at, scen.method,
      override_method, "clutter-perf",
      [&](holoscan::Graph& graph, const auto& src,
          holoscan::CompileOptions& options) {
        auto op = graph.op<BaseOp>("clutter", extents);
        hr::ClutterFilterParams params;
        params.method = scen.method;
        op.set_params(std::move(params));
        const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(src->out, op->range_profiles, q);
        graph.add_flow(src->override_out, op->method, q);
        graph.add_flow(op->range_profiles_out, sink->in, q);
        graph.partition("perf").add(src).add(op).add(sink);
        perf::bind_tensor_out(options, "clutter", "range_profiles_out");
      });
}

void fill_throughput(perf::ResultRow* row, const ClutterScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double chirps = static_cast<double>(scen.num_chirps);
  const double bins = static_cast<double>(scen.num_range_bins);
  row->cells_per_second = batch * chirps * bins / sec;
  row->slices_per_second = batch * bins / sec;

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double bytes = 2.0 * batch * chirps * bins * csz;
  row->estimated_effective_gbps = bytes / sec / 1e9;
}

perf::ResultRow base_row(const ClutterScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "ClutterFilterOp";
  row.variant = method_tag(scen.method);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_chirps, scen.num_range_bins};
  row.input_shape = profiles_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.num_chirps;
  row.notes = "live_override=method";
  return row;
}

ClutterScenario make_scenario(hr::ClutterFilterMethod method, const SizeSpec& s,
                              int64_t channels = 1) {
  ClutterScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.num_range_bins = s.num_range_bins;
  sc.channels = channels;
  sc.method = method;
  std::ostringstream id;
  id << "clutter_" << method_tag(method) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

struct ClutterTraits : perf::PerfDriverDefaults {
  using Scenario = ClutterScenario;

  static const char* tool_name() { return "perf_clutter_filter"; }
  static const char* tool_description() {
    return "ClutterFilterOp performance characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Mean-subtraction size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Mean-subtraction size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (methods x sizes x channel batch) with live "
           "method-override timing";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_chirps, s.num_range_bins};
      def.cells = s.num_chirps * s.num_range_bins;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<ClutterScenario> build_default_matrix() {
    std::vector<ClutterScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(hr::ClutterFilterMethod::kMeanSubtraction,
                                  must_find_size(sid)));
    }
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(hr::ClutterFilterMethod::kSingleDelay, md));
    out.push_back(make_scenario(hr::ClutterFilterMethod::kDoubleDelay, md));
    out.push_back(make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, md,
                                4));
    return out;
  }

  static std::vector<ClutterScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<ClutterScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(
          make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, *s));
    }
    return out;
  }

  static std::vector<ClutterScenario> build_full_matrix() {
    std::vector<ClutterScenario> out;
    const auto methods = {hr::ClutterFilterMethod::kMeanSubtraction,
                          hr::ClutterFilterMethod::kSingleDelay,
                          hr::ClutterFilterMethod::kDoubleDelay};
    for (const auto& s : size_ladder()) {
      for (auto m : methods) {
        out.push_back(make_scenario(m, s));
      }
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      for (int64_t ch : {4, 16}) {
        out.push_back(
            make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, s, ch));
      }
    }
    return out;
  }

  static void run_scenario(const ClutterScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_clutter_graph(scen, p);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are pre-override "
        "steady state; override_ms is peak latency over the 3-frame live "
        "method switch (changes the kernel path and output chirp count); "
        "cells/s is complex cells/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(hr::ClutterFilterMethod::kMeanSubtraction,
                                 must_find_size("xs")),
                   warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<ClutterTraits>(argc, argv);
}
