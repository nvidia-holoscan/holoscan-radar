// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/ddm_power_fold_op/ddm_power_fold_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_profiles.hpp"
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

hr::DDMCodeDescriptor code(int64_t tx) {
  hr::DDMCodeDescriptor result;
  result.num_tx = tx;
  for (int64_t index = 0; index < tx; ++index) {
    result.doppler_offsets.push_back(static_cast<double>(index) / static_cast<double>(tx));
  }
  return result;
}

perf::TimingCollector run_graph(const Scenario& scenario, const perf::RunPlan& plan) {
  const auto& size = scenario.size;
  auto input = perf::make_iq_cube_device<float>({size.rx, size.doppler, size.range});
  return perf::run_perf_graph(
      std::move(input), plan.total_frames, "ddm-power-fold-perf",
      [&](holoscan::Graph& graph, const auto& source, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_rx = size.rx;
        extents.max_doppler_bins = size.doppler;
        extents.max_range_bins = size.range;
        auto op = graph.op<hr::DDMPowerFoldOp<float>>("fold", extents);
        op.set_params(hr::DDMPowerFoldParams{.ddm_code = code(size.tx)});
        const auto terminal = graph.op<perf::PerfTerminalSinkOp>("terminal");
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
  row.input_shape = {scenario.size.rx, scenario.size.doppler, scenario.size.range};
  row.k = scenario.size.tx * scenario.size.rx;
  row.notes = "reads N_rx*D*R; emits honest folded (D/M)*R extent";
  return row;
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_ddm_power_fold"; }
  static const char* tool_description() { return "Sparse DDM power-fold characterization"; }
  static const char* sizes_all_help() { return "Run full size ladder"; }
  static const char* sizes_subset_help() { return "Run selected sizes"; }
  static const char* full_help() { return "Run all DDM fold sizes"; }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    const auto suite = perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "ddm_power_fold.yaml",
        "ddm_power_fold");
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = suite.shapes[i];
      if (shape.size() != 4)
        throw std::invalid_argument("profile shape rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]),
                        {shape[2], shape[3]},
                        shape[2] * shape[3],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }
  static std::vector<Scenario> build_profile_matrix(const std::string& scope) {
    const auto suite = perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "ddm_power_fold.yaml",
        "ddm_power_fold");
    std::vector<Scenario> result;
    for (const auto& c : suite.cases) {
      perf::require_keys(c.parameters, {"ddm_code"}, "parameters");
      if (c.parameters["ddm_code"].as<std::string>() != "linear" || c.override_value.has_value()) {
        throw std::invalid_argument("DDMPowerFold supports only linear code and no live override");
      }
    }
    for (const auto& c : suite.select(scope)) {
      size_t index = 0;
      for (; index < perf::kProfiles.size(); ++index) {
        if (c.profile == perf::kProfiles[index])
          break;
      }
      const auto& shape = suite.shapes.at(index);
      if (shape.size() != 4)
        throw std::invalid_argument("DDMPowerFold profile must be rank 4");
      if (shape[2] % shape[0] != 0) {
        throw std::invalid_argument("DDMPowerFold Doppler size must be divisible by TX count");
      }
      result.push_back(
          {c.id, c.profile, {c.profile, shape[0], shape[1], shape[2], shape[3], c.profile}});
    }
    return result;
  }
  static void run_scenario(const Scenario& scenario, const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const auto plan = perf::make_run_plan(settings, false);
    const auto samples =
        perf::collect_scenario_samples(settings, plan, [&](int, const perf::RunPlan& current) {
          return run_graph(scenario, current);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    auto row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    if (row.latency.median_ms > 0.0) {
      row.cells_per_second = static_cast<double>(scenario.size.doppler * scenario.size.range) /
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
      run_scenario(build_profile_matrix("minimal").front(), settings, &report);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
