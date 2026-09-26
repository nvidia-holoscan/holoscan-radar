// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MultiSourceExpansionOp performance characterization. Sweeps original rows,
// selected fraction, confirmed source count, and detection-index width.
// Measurement only: no timing thresholds.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "detection/multi_source_expansion_op/multi_source_expansion_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_runner.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;
constexpr DLDataType kInt64Dtype{kDLInt, 64U, 1U};
constexpr DLDataType kInt32Dtype{kDLInt, 32U, 1U};

struct SizeSpec {
  std::string id;
  int64_t rows;
  std::string role;
};

struct ExpansionScenario {
  std::string id;
  std::string size_id;
  int64_t rows{0};
  double selected_fraction{0.25};
  int64_t sources{2};
  int64_t index_columns{2};
};

template <typename Element>
std::shared_ptr<holoscan::Tensor> upload(const std::vector<Element>& host,
                                         const std::vector<int64_t>& shape, DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto writable = tensor->mutable_data();
    if (!writable || cudaMemcpy(writable.value(), host.data(), host.size() * sizeof(Element),
                                cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("perf_multi_source_expansion: input upload failed");
    }
  }
  return tensor;
}

int64_t selected_rows(const ExpansionScenario& scenario) {
  return std::max<int64_t>(
      1, static_cast<int64_t>(
             std::llround(scenario.selected_fraction * static_cast<double>(scenario.rows))));
}

struct ExpansionInputs {
  std::shared_ptr<holoscan::Tensor> detection_idx;
  std::shared_ptr<holoscan::Tensor> baseline;
  std::shared_ptr<holoscan::Tensor> selected;
  std::shared_ptr<holoscan::Tensor> resolved;
  std::shared_ptr<holoscan::Tensor> multi;
  std::shared_ptr<holoscan::Tensor> count;
};

ExpansionInputs make_inputs(const ExpansionScenario& scenario) {
  const int64_t selected_count = selected_rows(scenario);
  std::vector<int64_t> idx(static_cast<size_t>(scenario.rows * scenario.index_columns));
  std::vector<PerfPrecision> baseline(static_cast<size_t>(scenario.rows * 2));
  for (int64_t row = 0; row < scenario.rows; ++row) {
    for (int64_t column = 0; column < scenario.index_columns; ++column) {
      idx[static_cast<size_t>(row * scenario.index_columns + column)] = row * 11 + column;
    }
    baseline[static_cast<size_t>(row * 2)] = static_cast<PerfPrecision>((row % 101) * 0.001);
  }

  std::vector<int64_t> selected(static_cast<size_t>(selected_count));
  std::vector<PerfPrecision> resolved(static_cast<size_t>(selected_count * 2));
  std::vector<PerfPrecision> multi(static_cast<size_t>(selected_count * scenario.sources * 2));
  std::vector<int32_t> count(static_cast<size_t>(selected_count),
                             static_cast<int32_t>(scenario.sources));
  for (int64_t row = 0; row < selected_count; ++row) {
    selected[static_cast<size_t>(row)] = row;
    resolved[static_cast<size_t>(row * 2)] = -0.1F;
    for (int64_t source = 0; source < scenario.sources; ++source) {
      multi[static_cast<size_t>((row * scenario.sources + source) * 2)] =
          -0.1F + 0.05F * static_cast<PerfPrecision>(source);
    }
  }

  return ExpansionInputs{
      .detection_idx = upload(idx, {scenario.rows, scenario.index_columns}, kInt64Dtype),
      .baseline = upload(baseline, {scenario.rows, 2}, hr::real_dtype<PerfPrecision>()),
      .selected = upload(selected, {selected_count}, kInt64Dtype),
      .resolved = upload(resolved, {selected_count, 2}, hr::real_dtype<PerfPrecision>()),
      .multi =
          upload(multi, {selected_count, scenario.sources, 2}, hr::real_dtype<PerfPrecision>()),
      .count = upload(count, {selected_count}, kInt32Dtype)};
}

perf::TimingCollector run_graph(const ExpansionScenario& scenario, const perf::RunPlan& plan) {
  auto inputs = make_inputs(scenario);
  return perf::run_data_recipe(
      std::array{std::move(inputs.detection_idx), std::move(inputs.baseline),
                 std::move(inputs.selected), std::move(inputs.resolved), std::move(inputs.multi),
                 std::move(inputs.count)},
      plan, "multi-source-expansion-perf",
      [&](holoscan::Graph& graph, const auto& source, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_items = scenario.rows;
        extents.max_sources = scenario.sources;
        extents.max_points = scenario.rows + selected_rows(scenario) * (scenario.sources - 1);
        const auto expansion =
            graph.op<hr::MultiSourceExpansionOp<PerfPrecision>>("expansion", extents);
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        graph.add_flow(source->a, expansion->detection_idx, queue);
        graph.add_flow(source->b, expansion->baseline_aoa, queue);
        graph.add_flow(source->c, expansion->selected_idx, queue);
        graph.add_flow(source->d, expansion->resolved_aoa, queue);
        graph.add_flow(source->e, expansion->doa_multi, queue);
        graph.add_flow(source->f, expansion->source_count, queue);
        perf::PerfRecipeOutputs outputs(graph, options, "expansion");
        outputs.timed(expansion->detection_idx_out, "detection_idx_out");
        outputs.drain(expansion->aoa_estimates_out, "aoa_estimates_out");
        outputs.finish(source, expansion);
      });
}

ExpansionScenario make_scenario(const SizeSpec& size, double fraction, int64_t sources,
                                int64_t columns) {
  ExpansionScenario scenario;
  scenario.size_id = size.id;
  scenario.rows = size.rows;
  scenario.selected_fraction = fraction;
  scenario.sources = sources;
  scenario.index_columns = columns;
  std::ostringstream id;
  id << "expand_" << size.id << "_n" << size.rows << "_sel" << static_cast<int>(fraction * 100.0)
     << "_s" << sources << "_w" << columns;
  scenario.id = id.str();
  return scenario;
}

perf::ResultRow base_row(const ExpansionScenario& scenario) {
  perf::ResultRow row;
  row.op_name = "MultiSourceExpansionOp";
  row.variant = "confirmed_sources";
  row.input_shape = {scenario.rows, scenario.index_columns};
  row.out_batch = scenario.rows + selected_rows(scenario) * (scenario.sources - 1);
  row.k = scenario.sources;
  row.notes = "both outputs drained";
  return row;
}

void fill_throughput(perf::ResultRow* row, const ExpansionScenario& scenario, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double seconds = median_ms / 1000.0;
  row->cells_per_second = static_cast<double>(scenario.rows) / seconds;
  row->slices_per_second = static_cast<double>(row->out_batch) / seconds;
}

void decorate_row(perf::ResultRow& row, const ExpansionScenario& scenario) {
  fill_throughput(&row, scenario, row.latency.median_ms);
  row.notes =
      "cold_ms is first frame; median/p90 are steady state; both outputs "
      "are drained; cells/s is original rows/s; slices/s is emitted rows/s";
}

struct ExpansionTraits : perf::PerfDriverDefaults {
  using Scenario = ExpansionScenario;

  static const char* tool_name() { return "perf_multi_source_expansion"; }
  static const char* tool_description() {
    return "MultiSourceExpansionOp HSDK 5 Graph characterization";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 1)
      throw std::invalid_argument("multi_source_expansion profile rank mismatch");
    return {id, shape, shape[0], id};
  }

  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 1)
      throw std::invalid_argument("multi_source_expansion profile rank mismatch");
    perf::require_keys(c.parameters, {"selected_fraction", "sources", "index_columns"},
                       "parameters");
    if (c.override_value.has_value())
      throw std::invalid_argument("MultiSourceExpansion has no live input");
    const double fraction = c.parameters["selected_fraction"].as<double>();
    const int64_t sources = c.parameters["sources"].as<int64_t>();
    const int64_t columns = c.parameters["index_columns"].as<int64_t>();
    if (!(fraction > 0.0 && fraction <= 1.0) || sources < 1 || sources > 4 || columns < 2 ||
        columns > 10)
      throw std::invalid_argument("MultiSourceExpansion parameters out of range");
    const SizeSpec size{c.profile, shape[0], c.profile};
    auto scenario = make_scenario(size, fraction, sources, columns);
    scenario.id = c.id;
    scenario.size_id = c.profile;
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<ExpansionTraits>(argc, argv);
}
