// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
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
#include "utils/perf_recipe.hpp"
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
  return perf::run_data_recipe(
      std::array{std::move(input)}, plan, "ddm-power-fold-perf",
      [&](holoscan::Graph& graph, const auto& source, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_rx = size.rx;
        extents.max_doppler_bins = size.doppler;
        extents.max_range_bins = size.range;
        auto op = graph.op<hr::DDMPowerFoldOp<float>>("fold", extents);
        op.set_params(hr::DDMPowerFoldParams{.ddm_code = code(size.tx)});
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        graph.add_flow(source->out, op->rd_maps, queue);
        perf::PerfRecipeOutputs outputs(graph, options, "fold");
        outputs.timed(op->power_map, "power_map");
        outputs.finish(source, op);
      });
}

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.op_name = "DDMPowerFoldOp";
  row.variant = "linear_exact";
  perf::set_input_and_trailing_map_shape(
      row, {scenario.size.rx, scenario.size.doppler, scenario.size.range});
  row.k = scenario.size.tx * scenario.size.rx;
  row.notes = "reads N_rx*D*R; emits honest folded (D/M)*R extent";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scenario) {
  if (row.latency.median_ms > 0.0) {
    row.cells_per_second = static_cast<double>(scenario.size.doppler * scenario.size.range) /
                           (row.latency.median_ms / 1000.0);
  }
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_ddm_power_fold"; }
  static const char* tool_description() { return "Sparse DDM power-fold characterization"; }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 4)
      throw std::invalid_argument("profile shape rank mismatch");
    return {id, {shape[2], shape[3]}, shape[2] * shape[3], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    perf::require_keys(c.parameters, {"ddm_code"}, "parameters");
    if (c.parameters["ddm_code"].as<std::string>() != "linear" || c.override_value.has_value()) {
      throw std::invalid_argument("DDMPowerFold supports only linear code and no live override");
    }

    if (shape.size() != 4)
      throw std::invalid_argument("DDMPowerFold profile must be rank 4");
    if (shape[2] % shape[0] != 0) {
      throw std::invalid_argument("DDMPowerFold Doppler size must be divisible by TX count");
    }
    return {c.id, c.profile, {c.profile, shape[0], shape[1], shape[2], shape[3], c.profile}};
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
