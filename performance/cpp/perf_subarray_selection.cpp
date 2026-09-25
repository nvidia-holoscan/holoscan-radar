// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for SubarraySelectionOp. Sweeps item, channel,
// and snapshot counts on a holey ULA that requires a channel gather.
// Measurement tool only: no timing assertions.

#include <cstdint>
#include <cstdlib>
#include <set>
#include <type_traits>
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
  int64_t min_subarray_size{2};
  std::string live_port;
};

SubarrayScenario make_scenario(const SizeSpec& size) {
  SubarrayScenario scenario;
  scenario.size_id = size.id;
  scenario.items = size.items;
  scenario.channels = size.channels;
  scenario.selected_channels = size.channels - size.channels / 4;
  scenario.snapshots = size.snapshots;
  std::ostringstream id;
  id << "subarray_" << size.id << "_n" << size.items << "_c" << size.channels << "_l"
     << scenario.selected_channels << "_k" << size.snapshots;
  scenario.id = id.str();
  return scenario;
}

hr::ArrayGeometry make_holey_ula(const SubarrayScenario& scenario) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = hr::ArrayGeometryType::kUla;
  geometry.element_positions.reserve(static_cast<size_t>(scenario.channels * 3));
  for (int64_t channel = 0; channel < scenario.channels; ++channel) {
    const int64_t lattice_index = channel < scenario.selected_channels ? channel : channel + 1;
    geometry.element_positions.insert(geometry.element_positions.end(),
                                      {0.0, 0.5 * static_cast<double>(lattice_index), 0.0});
  }
  return geometry;
}

perf::TimingCollector run_graph(const SubarrayScenario& scenario, const perf::RunPlan& plan) {
  auto snapshot = perf::make_iq_cube_device<PerfPrecision>(
      {scenario.items, scenario.channels, scenario.snapshots});
  const auto build = [&](holoscan::Graph& graph, const auto& source,
                         holoscan::CompileOptions& options) {
    hr::MaxTensorExtents extents;
    extents.max_items = scenario.items;
    extents.max_subarray_channels = scenario.channels;
    extents.max_snapshots = scenario.snapshots;
    auto op = graph.op<hr::SubarraySelectionOp<PerfPrecision>>("subarray_selection", extents);
    hr::SubarraySelectionParams params;
    params.array_geometry = make_holey_ula(scenario);
    params.min_subarray_size = scenario.min_subarray_size;
    op.set_params(std::move(params));

    const auto snapshot_sink = graph.op<perf::PerfTerminalSinkOp>("snapshot_sink");
    const auto geometry_drain =
        graph.op<perf::PerfPayloadSinkOp<hr::WireArrayGeometry>>("geometry_drain");
    const holoscan::ConnectionOptions queue{.queue_depth = 4U};
    graph.add_flow(source->out, op->snapshot, queue);
    if constexpr (requires { source->override_out; }) {
      if constexpr (std::is_same_v<std::remove_reference_t<decltype(source->override_out)>,
                                   holoscan::Output<hr::WireArrayGeometry>>)
        graph.add_flow(source->override_out, op->array_geometry_in, queue);
      else
        graph.add_flow(source->override_out, op->subarray, queue);
    }
    graph.add_flow(op->snapshot_out, snapshot_sink->in, queue);
    graph.add_flow(op->array_geometry, geometry_drain->in, queue);
    perf::bind_tensor_out(options, "subarray_selection", "snapshot_out");
    // Declared optional device outputs still require placement on EA.
    perf::bind_tensor_out(options, "subarray_selection", "is_valid_out");
  };
  if (scenario.live_port == "array_geometry") {
    auto changed = make_holey_ula(scenario);
    changed.wavelength = 0.9;
    return perf::run_perf_graph_with_override(
        std::move(snapshot), plan.total_frames, plan.override_at,
        hr::to_wire(make_holey_ula(scenario)), hr::to_wire(changed), "subarray-selection-perf",
        build, perf::PerfPublishMode::kCopy, true);
  }
  if (scenario.live_port == "subarray") {
    std::vector<int64_t> indices;
    for (int64_t i = 0; i < scenario.selected_channels; ++i)
      indices.push_back(i);
    return perf::run_perf_graph_with_vector_override<int64_t>(
        std::move(snapshot), plan.total_frames, plan.override_at, {}, indices,
        "subarray-selection-perf", build, perf::PerfPublishMode::kCopy, true);
  }
  return perf::run_perf_graph(std::move(snapshot), plan.total_frames, "subarray-selection-perf",
                              build);
}

void fill_throughput(perf::ResultRow* row, const SubarrayScenario& scenario, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double seconds = median_ms / 1000.0;
  const double items = static_cast<double>(scenario.items);
  const double values = items * static_cast<double>(scenario.selected_channels) *
                        static_cast<double>(scenario.snapshots);
  row->cells_per_second = items / seconds;
  row->slices_per_second = values / seconds;
  const double complex_bytes = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  row->estimated_effective_gbps = (2.0 * values * complex_bytes) / seconds / 1e9;
}

perf::ResultRow base_row(const SubarrayScenario& scenario) {
  perf::ResultRow row;
  row.op_name = "SubarraySelectionOp";
  row.variant = "holey_ula";
  row.input_shape = {scenario.items, scenario.channels, scenario.snapshots};
  row.out_batch = scenario.items;
  row.k = scenario.selected_channels;
  row.live_port = scenario.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const SubarrayScenario& scenario) {
  fill_throughput(&row, scenario, row.latency.median_ms);
  row.notes = "cold_ms includes first selection plan; override_ms is one changed frame for " +
              scenario.live_port + "; cells/s is items/s";
  if (row.live_port.empty())
    row.notes = "default parameters; no live override";
}

struct SubarrayTraits : perf::PerfDriverDefaults {
  using Scenario = SubarrayScenario;

  static const char* tool_name() { return "perf_subarray_selection"; }
  static const char* tool_description() {
    return "SubarraySelectionOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 3)
      throw std::invalid_argument("SubarraySelection profile rank mismatch");
    return {id, shape, shape[0] * shape[1] * shape[2], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 3 || shape[0] < 1 || shape[1] < 4 || shape[2] < 1)
      throw std::invalid_argument("SubarraySelection invalid shape");
    perf::require_keys(c.parameters, {"min_subarray_size"}, "parameters");
    const auto min_size = c.parameters["min_subarray_size"].as<int64_t>();
    if (min_size < 2 || min_size > shape[1] - shape[1] / 4)
      throw std::invalid_argument("SubarraySelection invalid minimum size");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], c.profile};
    auto scenario = make_scenario(size);
    scenario.id = c.id;
    scenario.min_subarray_size = min_size;
    if (c.override_value) {
      scenario.live_port = (*c.override_value)["port"].as<std::string>();
      const auto value = (*c.override_value)["value"].as<std::string>();
      if ((scenario.live_port == "array_geometry" && value != "wavelength_09") ||
          (scenario.live_port == "subarray" && value != "contiguous") ||
          (scenario.live_port != "array_geometry" && scenario.live_port != "subarray"))
        throw std::invalid_argument("SubarraySelection invalid override");
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<SubarrayTraits>(argc, argv);
}
