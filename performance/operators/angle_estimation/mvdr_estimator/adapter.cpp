// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "utils/perf_angle_recipe.hpp"
#include "utils/perf_driver.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::MVDREstimatorOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, perf::direct_recipe_extents(scen.recipe, scen.values),
      perf::angle_recipe_params<hr::MVDRParams>(scen));
}

perf::ResultRow base_row(const Scenario& scen) {
  auto row = perf::angle_recipe_row(scen, "MVDREstimatorOp", "capon_live_" + scen.live_port);
  row.notes = "live_override=" + scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double detections = static_cast<double>(scen.axis("detections"));
  const double beams = static_cast<double>(scen.axis("az")) * static_cast<double>(scen.axis("el"));
  perf::set_recipe_throughput(row, detections, detections * beams);
  row.notes = scen.live_port.empty()
                  ? "default parameters; no live override"
                  : "override_ms isolates inline " + scen.live_port +
                        "; cells/s is detections/s; Capon scan dominates per tick";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_mvdr_estimator"; }
  static const char* tool_description() {
    return "MVDREstimatorOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 5, 3);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.MVDREstimatorOp.float", "",
        {{"snapshot",
          {"detections", "channels", "snapshots"},
          "complex64",
          "device",
          "iq_cube",
          {}}},
        {"aoa_estimates", {"angle_spectrum", "doa_multi", "source_count", "eigenvalues"}, {}, 8U},
        {{"array_geometry", "array_geometry"},
         {"az_coordinates", "az_coordinates"},
         {"el_coordinates", "el_coordinates"},
         {"spatial_smoothing", "spatial_smoothing"},
         {"diagonal_loading", "diagonal_loading"},
         {"max_sources", "max_sources"}},
        {{"array_geometry", "ula_geometry_patch"},
         {"az_coordinates", "shifted_linspace"},
         {"el_coordinates", "shifted_linspace"},
         {"spatial_smoothing", "spatial_smoothing"},
         {"diagonal_loading", "double"},
         {"max_sources", "int64"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 5);
    if (scen.axis("detections") < 4 || scen.axis("snapshots") < 2 ||
        scen.axis("max_sources_bound") != 4)
      throw std::invalid_argument("MVDREstimatorOp invalid profile or bounds");
    perf::require_recipe_params(
        scen.params, {"array_geometry", "angle_convention", "az_coordinates", "el_coordinates",
                      "diagonal_loading", "max_sources", "spatial_smoothing"});
    perf::validate_angle_recipe<hr::MVDRParams>(scen);
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
