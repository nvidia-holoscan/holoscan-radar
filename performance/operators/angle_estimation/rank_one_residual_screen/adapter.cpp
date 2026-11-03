// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::RankOneResidualScreenOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

hr::RankOneResidualScreenParams parameters(const Scenario& scen) {
  hr::RankOneResidualScreenParams params;
  params.array_geometry =
      perf::recipe_ula_geometry(scen.params["array_geometry"], scen.axis("channels"));
  params.residual_threshold = scen.params["residual_threshold"].as<double>();
  return params;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  return perf::run_recipe_selected_case<Op, 2>(
      scen, plan, perf::direct_recipe_extents(scen.recipe, scen.values), parameters(scen));
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "RankOneResidualScreenOp";
  row.variant = "normalized_projection";
  row.input_shape = perf::recipe_shape(scen.recipe.inputs.at(0), scen.values);
  row.out_batch = static_cast<int64_t>(std::llround(
      static_cast<double>(scen.axis("selection_percent") * scen.axis("items")) / 100.0));
  row.k = scen.axis("channels");
  row.live_port = scen.live_port;
  row.notes = "all outputs drained";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const auto values = static_cast<double>(scen.axis("items")) *
                      static_cast<double>(scen.axis("channels")) *
                      static_cast<double>(scen.axis("looks"));
  perf::set_recipe_throughput(row, values, static_cast<double>(scen.axis("items")));
  row.notes = "end-to-end Graph latency; cells/s is complex snapshot values/s";
  if (!scen.live_port.empty())
    row.notes += "; one-frame geometry change";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_rank_one_residual_screen"; }
  static const char* tool_description() {
    return "RankOneResidualScreenOp HSDK 5 Graph characterization";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 3, 3);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.RankOneResidualScreenOp.float", "",
        {{"snapshot", {"items", "channels", "looks"}, "complex64", "device", "rank_one_mix", {}},
         {"baseline_aoa", {"items", "two"}, "float32", "device", "zero_float", {}}},
        {"snapshot_out",
         {"selected_idx", "is_valid_out", "residual_score", "selection_mask"},
         {},
         4U},
        {{"array_geometry", "array_geometry"}}, {{"array_geometry", "ula_geometry_patch"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 3);
    if (scen.axis("items") < 1 || scen.axis("channels") < 2 || scen.axis("looks") < 1 ||
        scen.axis("selection_percent") > 100)
      throw std::invalid_argument("RankOneResidualScreen invalid profile or fixture");
    perf::require_recipe_params(scen.params, {"array_geometry", "residual_threshold"});
    (void)parameters(scen);
    if (!scen.live_port.empty()) {
      auto patch = YAML::Clone(scen.params["array_geometry"]);
      patch["spacing_wavelengths"] = scen.live_value["spacing_wavelengths"];
      (void)perf::recipe_ula_geometry(patch, scen.axis("channels"));
    }
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
