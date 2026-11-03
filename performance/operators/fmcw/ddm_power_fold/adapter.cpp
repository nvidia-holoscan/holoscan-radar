// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
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
#include "utils/perf_registered_recipe.hpp"
#include "utils/perf_yaml_recipe.hpp"
#include "utils/perf_yaml_pilot.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;
namespace perf_ddm_power_fold_adapter {

struct Scenario : perf::RecipeScenario {};

hr::DDMCodeDescriptor code(int64_t tx) {
  hr::DDMCodeDescriptor result;
  result.num_tx = tx;
  for (int64_t index = 0; index < tx; ++index) {
    result.doppler_offsets.push_back(static_cast<double>(index) / static_cast<double>(tx));
  }
  return result;
}

perf::TimingCollector run_graph(const Scenario& scenario, const perf::RunPlan& plan) {
  const auto extents = perf::direct_recipe_extents(scenario.recipe, scenario.values);
  const hr::DDMPowerFoldParams params{.ddm_code = code(scenario.axis("tx"))};
  return perf::run_recipe_case<hr::DDMPowerFoldOp<perf::Scalar>, 1>(scenario, plan, extents,
                                                                    params);
}

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.op_name = "DDMPowerFoldOp";
  row.variant = "linear_exact";
  perf::set_input_and_trailing_map_shape(
      row, {scenario.axis("rx"), scenario.axis("doppler"), scenario.axis("range")});
  row.k = scenario.axis("tx") * scenario.axis("rx");
  row.notes = "reads N_rx*D*R; emits honest folded (D/M)*R extent";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scenario) {
  perf::set_recipe_throughput(
      row, static_cast<double>(scenario.axis("doppler") * scenario.axis("range")));
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = perf_ddm_power_fold_adapter::Scenario;
  static const char* tool_name() { return "perf_ddm_power_fold"; }
  static const char* tool_description() { return "Sparse DDM power-fold characterization"; }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 4)
      throw std::invalid_argument("profile shape rank mismatch");
    return {id, {shape[2], shape[3]}, shape[2] * shape[3], id};
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<hr::DDMPowerFoldOp<perf::Scalar>>(
        suite, "radar.DDMPowerFoldOp.float", "",
        {{"rd_maps", {"rx", "doppler", "range"}, "complex64", "device", "iq_cube", {}}},
        {"power_map", {}, {}, 4U}, {});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scenario = perf::pilot_scenario<Scenario>(suite, c, shape, 4);
    if (shape[2] % shape[0] != 0 || c.override_value)
      throw std::invalid_argument("DDM power fold needs divisible rank-4 dimensions and no live");
    const auto& recipe = scenario.recipe;
    const auto bounds = perf::direct_recipe_extents(recipe, scenario.values);
    if (!bounds.max_rx || *bounds.max_rx < shape[1] || !bounds.max_doppler_bins ||
        *bounds.max_doppler_bins < shape[2] || !bounds.max_range_bins ||
        *bounds.max_range_bins < shape[3])
      throw std::invalid_argument("DDM power fold constructor extents do not cover input");
    perf::require_keys(recipe.params["ddm_code"], {"factory"}, "ddm_code");
    if (recipe.params["ddm_code"]["factory"].as<std::string>() != "linear_exact_ddm_code")
      throw std::invalid_argument("DDM power fold needs a linear exact-bin code");
    if (!c.parameters.IsMap() || c.parameters.size() != 0)
      throw std::invalid_argument("DDM power fold has no per-case Params changes");
    return scenario;
  }
};

}  // namespace perf_ddm_power_fold_adapter

int main(int argc, char** argv) {
  return perf::perf_main<perf_ddm_power_fold_adapter::Traits>(argc, argv);
}
