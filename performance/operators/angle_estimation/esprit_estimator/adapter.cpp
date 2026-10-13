// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "utils/perf_angle_recipe.hpp"
#include "utils/perf_driver.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::ESPRITEstimatorOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, perf::direct_recipe_extents(scen.recipe, scen.values),
      perf::angle_recipe_params<hr::ESPRITParams>(scen));
}

perf::ResultRow base_row(const Scenario& scen) {
  auto row = perf::angle_recipe_row(scen, "ESPRITEstimatorOp", "esprit_ls_live_" + scen.live_port);
  row.notes = "live_override=" + scen.live_port + "; subarray_spec is static";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  perf::set_recipe_throughput(row, static_cast<double>(scen.axis("detections")));
  row.notes = scen.live_port.empty()
                  ? "default parameters; no live override"
                  : "override_ms isolates inline " + scen.live_port +
                        "; steady num_sources is fixed at one except the source_count_pfa "
                        "scenario, which uses auto; subarray_spec is static";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_esprit_estimator"; }
  static const char* tool_description() {
    return "ESPRITEstimatorOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 3, 3);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.ESPRITEstimatorOp.float", "",
        {{"snapshot",
          {"detections", "channels", "snapshots"},
          "complex64",
          "device",
          "iq_cube",
          {}}},
        {"aoa_estimates", {"doa_multi", "source_count", "eigenvalues"}, {}, 8U},
        {{"array_geometry", "array_geometry"},
         {"spatial_smoothing", "spatial_smoothing"},
         {"num_sources", "num_sources"},
         {"source_count_pfa", "source_count_pfa"},
         {"max_sources", "max_sources"}},
        {{"array_geometry", "ula_geometry_patch"},
         {"spatial_smoothing", "spatial_smoothing"},
         {"num_sources", "int64"},
         {"source_count_pfa", "double"},
         {"max_sources", "int64"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 3);
    if (scen.axis("detections") < 4 || scen.axis("snapshots") < 2 ||
        scen.axis("max_sources_bound") != 4)
      throw std::invalid_argument("ESPRIT invalid profile or bounds");
    perf::require_recipe_params(
        scen.params, {"array_geometry", "angle_convention", "num_sources", "source_count_pfa",
                      "max_sources", "spatial_smoothing"});
    perf::validate_angle_recipe<hr::ESPRITParams>(scen);
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
