// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
#include "utils/perf_angle_recipe.hpp"
#include "utils/perf_driver.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::PhaseShiftBeamformerOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, perf::direct_recipe_extents(scen.recipe, scen.values),
      perf::angle_recipe_params<hr::PhaseShiftBeamformerParams>(scen));
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "PhaseShiftBeamformerOp";
  row.variant = "bartlett_dense";
  row.live_port = scen.live_port;
  row.map_shape = {scen.axis("az"), scen.axis("el")};
  row.input_shape = perf::recipe_shape(scen.recipe.inputs.at(0), scen.values);
  row.out_batch = scen.axis("doppler") * scen.axis("range");
  row.k = scen.axis("channels");
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double cells =
      static_cast<double>(scen.axis("doppler")) * static_cast<double>(scen.axis("range"));
  const double beams = static_cast<double>(scen.axis("az")) * static_cast<double>(scen.axis("el"));
  const double channels = static_cast<double>(scen.axis("channels"));
  const double bytes =
      (cells * channels + beams * channels + cells * beams) * sizeof(std::complex<perf::Scalar>);
  perf::set_recipe_throughput(row, cells, cells * beams, bytes);
  row.notes =
      scen.live_port.empty()
          ? "default parameters; no live override"
          : "cold_ms is first-frame latency; median/p90 are pre-change steady state; override_ms "
            "is the one changed frame for " +
                scen.live_port +
                " changes the steering configuration; cells/s is RD-cells/s (beams/s = slices/s)";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_phase_shift_beamformer"; }
  static const char* tool_description() {
    return "PhaseShiftBeamformerOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 5)
      throw std::invalid_argument("PhaseShiftBeamformer profile rank mismatch");
    return {id, shape, shape[0] * shape[3] * shape[4], id};
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.PhaseShiftBeamformerOp.float", "",
        {{"rd_maps", {"channels", "doppler", "range"}, "complex64", "device", "iq_cube", {}}},
        {"rda_cube", {}, {"az_grid", "el_grid"}, 8U},
        {{"array_geometry", "array_geometry"},
         {"az_coordinates", "az_coordinates"},
         {"el_coordinates", "el_coordinates"},
         {"array_taper", "array_taper"}},
        {{"array_geometry", "ula_geometry_patch"},
         {"az_coordinates", "shifted_linspace"},
         {"el_coordinates", "shifted_linspace"},
         {"array_taper", "window_spec"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 5);
    for (const auto axis : {"az", "el", "doppler", "range"})
      if (scen.axis(axis) < 1)
        throw std::invalid_argument("PhaseShiftBeamformer invalid profile shape");
    perf::require_recipe_params(scen.params, {"array_geometry", "angle_convention",
                                              "az_coordinates", "el_coordinates", "array_taper"});
    perf::validate_angle_recipe<hr::PhaseShiftBeamformerParams>(scen);
    if (scen.live_port == "array_taper")
      (void)perf::recipe_window_type(scen.live_value["type"].as<std::string>());
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
