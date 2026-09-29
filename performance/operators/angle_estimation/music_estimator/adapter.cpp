// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/music_estimator_op/music_estimator_op.hpp"
#include "utils/perf_angle_recipe.hpp"
#include "utils/perf_driver.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::MUSICEstimatorOp<float>;
struct Scenario : perf::RecipeScenario {};

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, perf::direct_recipe_extents(scen.recipe, scen.values),
      perf::angle_recipe_params<hr::MUSICParams>(scen));
}

perf::ResultRow base_row(const Scenario& scen) {
  const auto order = scen.params["num_sources"].as<int64_t>();
  const auto variant = scen.live_port.empty() && order > 0
                           ? "music_model_order_" + std::to_string(order)
                           : "music_live_" + scen.live_port;
  auto row = perf::angle_recipe_row(scen, "MUSICEstimatorOp", variant);
  row.notes = scen.live_port.empty() && order > 0 ? "fixed_model_order=" + std::to_string(order)
                                                  : "live_override=" + scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double detections = static_cast<double>(scen.axis("detections"));
  perf::set_recipe_throughput(
      row, detections,
      detections * static_cast<double>(scen.axis("az")) * static_cast<double>(scen.axis("el")));
  const auto order = scen.params["num_sources"].as<int64_t>();
  row.notes =
      scen.live_port.empty()
          ? "Params num_sources=" + (order < 0 ? std::string("auto") : std::to_string(order)) +
                "; no inline override; cells/s is detections/s"
          : "override_ms isolates inline " + scen.live_port +
                "; Params num_sources=" + std::to_string(order) + "; cells/s is detections/s";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_music_estimator"; }
  static const char* tool_description() {
    return "MUSICEstimatorOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 5, 3);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.MUSICEstimatorOp.float", "",
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
         {"num_sources", "num_sources"},
         {"source_count_pfa", "source_count_pfa"},
         {"max_sources", "max_sources"}},
        {{"array_geometry", "ula_geometry_patch"},
         {"az_coordinates", "shifted_linspace"},
         {"el_coordinates", "shifted_linspace"},
         {"spatial_smoothing", "spatial_smoothing"},
         {"num_sources", "int64"},
         {"source_count_pfa", "double"},
         {"max_sources", "int64"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 5);
    if (scen.axis("detections") < 4 || scen.axis("snapshots") < 2 ||
        scen.axis("max_sources_bound") != 4)
      throw std::invalid_argument("MUSICEstimatorOp invalid profile or bounds");
    perf::require_recipe_params(
        scen.params, {"array_geometry", "angle_convention", "az_coordinates", "el_coordinates",
                      "num_sources", "source_count_pfa", "max_sources", "spatial_smoothing"});
    perf::validate_angle_recipe<hr::MUSICParams>(scen);
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
