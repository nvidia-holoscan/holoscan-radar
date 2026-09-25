// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for MIMODeinterleaveOp — TDM slow-time gather
// into an explicit TX axis. YAML profiles vary named TX, RX, chirp, and
// sample extents. No [live] parameters. Measurement tool only.

#include <complex>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <memory>
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
#include "utils/perf_profiles.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

struct DeinterleaveScenario {
  std::string id;
  std::string size_id;
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t chirps_per_tx{0};
  int64_t samples{0};
};

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
    const auto suite = perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "mimo_deinterleave.yaml",
        "mimo_deinterleave");
    std::vector<perf::SizeDef> out;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = suite.shapes[i];
      if (shape.size() != 4)
        throw std::invalid_argument("profile shape rank mismatch");
      const int64_t slow = shape[0] * shape[2];
      out.push_back({std::string(perf::kProfiles[i]),
                     {shape[1], slow, shape[3]},
                     shape[1] * slow * shape[3],
                     std::string(perf::kProfiles[i])});
    }
    return out;
  }

  static std::vector<DeinterleaveScenario> build_profile_matrix(const std::string& scope) {
    const auto suite = perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "mimo_deinterleave.yaml",
        "mimo_deinterleave");
    std::vector<DeinterleaveScenario> result;
    for (const auto& c : suite.cases) {
      perf::require_keys(c.parameters, {"tx_schedule"}, "parameters");
      if (c.parameters["tx_schedule"].as<std::string>() != "round_robin" ||
          c.override_value.has_value()) {
        throw std::invalid_argument("MIMODeinterleave supports round-robin schedule only");
      }
    }
    for (const auto& c : suite.select(scope)) {
      size_t index = 0;
      for (; index < perf::kProfiles.size(); ++index) {
        if (c.profile == perf::kProfiles[index])
          break;
      }
      const auto& shape = suite.shapes.at(index);
      if (shape.size() != 4) {
        throw std::invalid_argument("MIMODeinterleave profile must have TX,RX,chirps,samples");
      }
      result.push_back({c.id, c.profile, shape[0], shape[1], shape[2], shape[3]});
    }
    return result;
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
      run_scenario(build_profile_matrix("minimal").front(), warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<DeinterleaveTraits>(argc, argv);
}
