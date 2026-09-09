// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/ddm_power_fold_op/ddm_power_fold_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;
namespace {

struct SizeSpec {
  std::string id;
  int64_t tx;
  int64_t rx;
  int64_t doppler;
  int64_t range;
  std::string role;
};
struct Scenario {
  std::string id;
  std::string size_id;
  SizeSpec size;
};

const std::vector<SizeSpec>& sizes() {
  static const std::vector<SizeSpec> result{
      {"xs", 2, 4, 64, 256, "smoke"},
      {"sm", 4, 4, 128, 512, "sparse automotive"},
      {"md", 4, 8, 256, 512, "common MIMO"},
      {"lg", 4, 16, 512, 1024, "large array"},
      {"xl", 8, 16, 512, 2048, "upper bound"}};
  return result;
}

hr::DDMCodeDescriptor code(int64_t tx) {
  hr::DDMCodeDescriptor result;
  result.num_tx = tx;
  for (int64_t index = 0; index < tx; ++index) {
    result.doppler_offsets.push_back(
        static_cast<double>(index) / static_cast<double>(tx));
  }
  return result;
}

perf::TimingCollector run_graph(
    const Scenario& scenario, const perf::RunPlan& plan) {
  const auto& size = scenario.size;
  auto input = perf::make_iq_cube_device<float>(
      {size.rx, size.doppler, size.range});
  return perf::run_perf_graph(
      std::move(input), plan.total_frames, "ddm-power-fold-perf",
      [&](holoscan::Graph& graph, const auto& source,
          holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_rx = size.rx;
        extents.max_doppler_bins = size.doppler;
        extents.max_range_bins = size.range;
        auto op = graph.op<hr::DDMPowerFoldOp<float>>("fold", extents);
        op.set_params(hr::DDMPowerFoldParams{.ddm_code = code(size.tx)});
        const auto terminal =
            graph.op<perf::PerfTerminalSinkOp>("terminal");
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        graph.add_flow(source->out, op->rd_maps, queue);
        graph.add_flow(op->power_map, terminal->in, queue);
        graph.partition("perf").add(source).add(op).add(terminal);
        perf::bind_tensor_out(options, "fold", "power_map");
      });
}

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "DDMPowerFoldOp";
  row.variant = "linear_exact";
  row.size_id = scenario.size_id;
  row.map_shape = {scenario.size.doppler, scenario.size.range};
  row.input_shape = {
      scenario.size.rx, scenario.size.doppler, scenario.size.range};
  row.k = scenario.size.tx * scenario.size.rx;
  row.notes = "reads N_rx*D*R; emits honest folded (D/M)*R extent";
  return row;
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_ddm_power_fold"; }
  static const char* tool_description() {
    return "Sparse DDM power-fold characterization";
  }
  static const char* sizes_all_help() { return "Run full size ladder"; }
  static const char* sizes_subset_help() { return "Run selected sizes"; }
  static const char* full_help() { return "Run all DDM fold sizes"; }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    for (const auto& size : sizes()) {
      result.push_back({size.id, {size.doppler, size.range},
                        size.doppler * size.range, size.role});
    }
    return result;
  }
  static std::vector<Scenario> build_size_sweep(
      const std::vector<std::string>& ids) {
    std::vector<Scenario> result;
    for (const auto& size : sizes()) {
      if (std::find(ids.begin(), ids.end(), size.id) != ids.end()) {
        result.push_back({"ddm_fold_" + size.id, size.id, size});
      }
    }
    return result;
  }
  static std::vector<Scenario> build_default_matrix() {
    return build_size_sweep({"xs", "sm", "md"});
  }
  static std::vector<Scenario> build_full_matrix() {
    return build_size_sweep({"xs", "sm", "md", "lg", "xl"});
  }
  static void run_scenario(
      const Scenario& scenario, const perf::RunSettings& settings,
      perf::PerfReport* report) {
    const auto plan = perf::make_run_plan(settings, false);
    const auto samples = perf::collect_scenario_samples(
        settings, plan, [&](int, const perf::RunPlan& current) {
          return run_graph(scenario, current);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    auto row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    if (row.latency.median_ms > 0.0) {
      row.cells_per_second =
          static_cast<double>(scenario.size.doppler * scenario.size.range) /
          (row.latency.median_ms / 1000.0);
    }
    report->results.push_back(std::move(row));
  }
  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([&] {
      perf::PerfReport report;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(build_size_sweep({"xs"}).front(), settings, &report);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
