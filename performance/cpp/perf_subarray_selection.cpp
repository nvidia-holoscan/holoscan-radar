// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for SubarraySelectionOp. Sweeps item, channel,
// and snapshot counts on a holey ULA that requires a channel gather.
// Measurement tool only: no timing assertions.

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

#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
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
  int64_t items;
  int64_t channels;
  int64_t snapshots;
  std::string role;
};

struct SubarrayScenario {
  std::string id;
  std::string size_id;
  int64_t items{0};
  int64_t channels{0};
  int64_t selected_channels{0};
  int64_t snapshots{0};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 64, 4, 2, "small array; fast smoke"},
      {"sm", 256, 8, 4, "compact SIMO array"},
      {"md", 1024, 16, 8, "common virtual array"},
      {"lg", 4096, 32, 16, "dense virtual array"},
      {"xl", 16384, 64, 32, "upper characterization bound"}};
  return ladder;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& size : size_ladder()) {
    if (size.id == id) {
      return &size;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* size = find_size(id);
  if (size == nullptr) {
    std::cerr << "internal error: unknown size id '" << id << "'\n";
    std::abort();
  }
  return *size;
}

SubarrayScenario make_scenario(const SizeSpec& size) {
  SubarrayScenario scenario;
  scenario.size_id = size.id;
  scenario.items = size.items;
  scenario.channels = size.channels;
  scenario.selected_channels = size.channels - size.channels / 4;
  scenario.snapshots = size.snapshots;
  std::ostringstream id;
  id << "subarray_" << size.id << "_n" << size.items << "_c"
     << size.channels << "_l" << scenario.selected_channels << "_k"
     << size.snapshots;
  scenario.id = id.str();
  return scenario;
}

hr::ArrayGeometry make_holey_ula(const SubarrayScenario& scenario) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = hr::ArrayGeometryType::kUla;
  geometry.element_positions.reserve(
      static_cast<size_t>(scenario.channels * 3));
  for (int64_t channel = 0; channel < scenario.channels; ++channel) {
    const int64_t lattice_index =
        channel < scenario.selected_channels ? channel : channel + 1;
    geometry.element_positions.insert(
        geometry.element_positions.end(),
        {0.0, 0.5 * static_cast<double>(lattice_index), 0.0});
  }
  return geometry;
}

perf::TimingCollector run_subarray_graph(
    const SubarrayScenario& scenario, int total_frames) {
  auto snapshot = perf::make_iq_cube_device<PerfPrecision>(
      {scenario.items, scenario.channels, scenario.snapshots});
  return perf::run_perf_graph(
      std::move(snapshot), total_frames, "subarray-selection-perf",
      [&](holoscan::Graph& graph, const auto& source,
          holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_items = scenario.items;
        extents.max_subarray_channels = scenario.channels;
        extents.max_snapshots = scenario.snapshots;
        auto op = graph.op<hr::SubarraySelectionOp<PerfPrecision>>(
            "subarray_selection", extents);
        hr::SubarraySelectionParams params;
        params.array_geometry = make_holey_ula(scenario);
        op.set_params(std::move(params));

        const auto snapshot_sink =
            graph.op<perf::PerfTerminalSinkOp>("snapshot_sink");
        const auto geometry_drain =
            graph.op<perf::PerfPayloadSinkOp<hr::WireArrayGeometry>>(
                "geometry_drain");
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        graph.add_flow(source->out, op->snapshot, queue);
        graph.add_flow(op->snapshot_out, snapshot_sink->in, queue);
        graph.add_flow(op->array_geometry, geometry_drain->in, queue);
        perf::bind_tensor_out(
            options, "subarray_selection", "snapshot_out");
        // Declared optional device outputs still require placement on EA.
        perf::bind_tensor_out(
            options, "subarray_selection", "is_valid_out");
      });
}

void fill_throughput(perf::ResultRow* row,
                     const SubarrayScenario& scenario,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double seconds = median_ms / 1000.0;
  const double items = static_cast<double>(scenario.items);
  const double values =
      items * static_cast<double>(scenario.selected_channels) *
      static_cast<double>(scenario.snapshots);
  row->cells_per_second = items / seconds;
  row->slices_per_second = values / seconds;
  const double complex_bytes =
      2.0 * static_cast<double>(sizeof(PerfPrecision));
  row->estimated_effective_gbps =
      (2.0 * values * complex_bytes) / seconds / 1e9;
}

perf::ResultRow base_row(const SubarrayScenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "SubarraySelectionOp";
  row.variant = "holey_ula";
  row.size_id = scenario.size_id;
  row.input_shape = {
      scenario.items, scenario.channels, scenario.snapshots};
  row.out_batch = scenario.items;
  row.k = scenario.selected_channels;
  return row;
}

struct SubarrayTraits : perf::PerfDriverDefaults {
  using Scenario = SubarrayScenario;

  static const char* tool_name() {
    return "perf_subarray_selection";
  }
  static const char* tool_description() {
    return "SubarraySelectionOp performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Default holey-ULA size sweep";
  }
  static const char* sizes_subset_help() {
    return "Holey-ULA sweep across an explicit size subset";
  }
  static const char* full_help() {
    return "Full item/channel/snapshot size ladder; schema and host-Tensor "
           "live override timing deferred on HSDK 5 EA";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> output;
    for (const auto& size : size_ladder()) {
      perf::SizeDef definition;
      definition.id = size.id;
      definition.shape = {size.items, size.channels, size.snapshots};
      definition.cells = size.items * size.channels * size.snapshots;
      definition.role = size.role;
      output.push_back(definition);
    }
    return output;
  }

  static std::vector<SubarrayScenario> build_default_matrix() {
    std::vector<SubarrayScenario> output;
    for (const char* id : {"xs", "sm", "md"}) {
      output.push_back(make_scenario(must_find_size(id)));
    }
    return output;
  }

  static std::vector<SubarrayScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<SubarrayScenario> output;
    for (const auto& id : sizes) {
      const SizeSpec* size = find_size(id);
      if (size == nullptr) {
        std::cerr << "warning: unknown size id '" << id << "', skipping\n";
        continue;
      }
      output.push_back(make_scenario(*size));
    }
    return output;
  }

  static std::vector<SubarrayScenario> build_full_matrix() {
    std::vector<SubarrayScenario> output;
    for (const auto& size : size_ladder()) {
      output.push_back(make_scenario(size));
    }
    return output;
  }

  static void run_scenario(const SubarrayScenario& scenario,
                           const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(settings, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        settings, plan, [&](int /*repeat*/, const perf::RunPlan& current) {
          return run_subarray_graph(scenario, current.total_frames);
        });
    perf::append_setup_row(base_row(scenario), samples, report);

    perf::ResultRow row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scenario, row.latency.median_ms);
    row.notes =
        "cold_ms includes first selection plan/table upload; median/p90 are "
        "steady gather + geometry publication; cells/s is items/s; live "
        "schema/host-Tensor override timing deferred";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(
          make_scenario(must_find_size("xs")), settings, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<SubarrayTraits>(argc, argv);
}
