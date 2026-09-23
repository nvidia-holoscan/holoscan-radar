// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for MIMODeinterleaveOp — TDM slow-time gather
// into an explicit TX axis. Sweeps (M_tx, N_rx, L, samples) and a leading
// batch axis. No [live] parameters. Measurement tool only.
//
//  Graph rewrite; drains adc_cube_out (§2 unique out name).

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

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
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

using PerfPrecision = float;

struct SizeSpec {
  std::string id;
  int64_t num_tx;
  int64_t num_rx;
  int64_t chirps_per_tx;
  int64_t samples;
  std::string role;
};

struct DeinterleaveScenario {
  std::string id;
  std::string size_id;
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t chirps_per_tx{0};
  int64_t samples{0};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 2, 4, 8, 64, "small TDM; fast smoke"},
      {"sm", 4, 8, 16, 128, "compact MIMO"},
      {"md", 4, 16, 32, 256, "common dev size"},
      {"lg", 8, 32, 64, 512, "moderate MIMO"},
      {"xl", 8, 64, 128, 1024, "upper characterization bound"}};
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

// The contract input is exactly (N_rx, slow_time, samples). There is no
// leading batch rung: an extra leading dim is opaque batch, which the operator
// rejects (see MIMODeinterleaveOp.RejectsOpaqueLeadingBatch).
std::vector<int64_t> interleaved_shape(const DeinterleaveScenario& scen) {
  const int64_t slow = scen.num_tx * scen.chirps_per_tx;
  return {scen.num_rx, slow, scen.samples};
}

perf::TimingCollector run_deinterleave_graph(const DeinterleaveScenario& scen, int total_frames) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(interleaved_shape(scen));
  return perf::run_perf_graph(
      std::move(cube), total_frames, "deinterleave-perf",
      [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_tx = scen.num_tx;
        extents.max_rx = scen.num_rx;
        extents.max_chirps = scen.chirps_per_tx;
        extents.max_samples_per_chirp = scen.samples;
        auto op = graph.op<hr::MIMODeinterleaveOp<PerfPrecision>>("deinterleave", extents);
        hr::MIMODeinterleaveParams params;
        params.tx_schedule.num_tx = scen.num_tx;
        op.set_params(std::move(params));
        const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(src->out, op->adc_cube, q);
        graph.add_flow(op->adc_cube_out, sink->in, q);
        perf::bind_tensor_out(options, "deinterleave", "adc_cube_out");
      });
}

void fill_throughput(perf::ResultRow* row, const DeinterleaveScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double elems = static_cast<double>(scen.num_tx) * static_cast<double>(scen.num_rx) *
                       static_cast<double>(scen.chirps_per_tx) * static_cast<double>(scen.samples);
  row->cells_per_second = elems / sec;
  const double csz = sizeof(std::complex<PerfPrecision>);
  const double slow = static_cast<double>(scen.num_tx * scen.chirps_per_tx);
  const double in_bytes =
      static_cast<double>(scen.num_rx) * slow * static_cast<double>(scen.samples) * csz;
  const double out_bytes = elems * csz;
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const DeinterleaveScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "MIMODeinterleaveOp";
  row.variant = "rr";
  row.size_id = scen.size_id;
  row.input_shape = interleaved_shape(scen);
  row.k = scen.chirps_per_tx;
  return row;
}

DeinterleaveScenario make_scenario(const SizeSpec& s) {
  DeinterleaveScenario sc;
  sc.size_id = s.id;
  sc.num_tx = s.num_tx;
  sc.num_rx = s.num_rx;
  sc.chirps_per_tx = s.chirps_per_tx;
  sc.samples = s.samples;
  std::ostringstream id;
  id << "deinterleave_rr_" << s.id;
  sc.id = id.str();
  return sc;
}

struct DeinterleaveTraits : perf::PerfDriverDefaults {
  using Scenario = DeinterleaveScenario;

  static const char* tool_name() { return "perf_mimo_deinterleave"; }
  static const char* tool_description() {
    return "MIMODeinterleaveOp performance characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() { return "Round-robin TDM gather across the full ladder"; }
  static const char* sizes_subset_help() {
    return "Round-robin TDM gather across an explicit subset";
  }
  static const char* full_help() { return "Round-robin TDM gather across the full ladder"; }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      const int64_t slow = s.num_tx * s.chirps_per_tx;
      def.shape = {s.num_rx, slow, s.samples};
      def.cells = s.num_rx * slow * s.samples;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<DeinterleaveScenario> build_default_matrix() {
    std::vector<DeinterleaveScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    return out;
  }

  static std::vector<DeinterleaveScenario> build_size_sweep(const std::vector<std::string>& sizes) {
    std::vector<DeinterleaveScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(*s));
    }
    return out;
  }

  static std::vector<DeinterleaveScenario> build_full_matrix() {
    std::vector<DeinterleaveScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    return out;
  }

  static void run_scenario(const DeinterleaveScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples =
        perf::collect_scenario_samples(rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_deinterleave_graph(scen, p.total_frames);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "pure gather marshaling; cold_ms is first-frame latency; "
        "median/p90 are steady state; cells/s counts complex samples moved";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs")), warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<DeinterleaveTraits>(argc, argv);
}
