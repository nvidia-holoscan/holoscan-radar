// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::DDMDenseDemultiplexOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

hr::DDMCodeDescriptor code(const Scenario& scen) {
  const auto tx = scen.axis("tx");
  const auto doppler = scen.axis("doppler");
  const auto factory = scen.params["ddm_code"]["factory"].as<std::string>();
  hr::DDMCodeDescriptor result;
  result.num_tx = tx;
  result.guard_bands = 0;
  if (factory == "linear_exact_ddm_code") {
    for (int64_t i = 0; i < tx; ++i)
      result.doppler_offsets.push_back(static_cast<double>(i) / static_cast<double>(tx));
  } else if (factory == "nonlinear_phase_code") {
    result.phase_codes.resize(static_cast<std::size_t>(tx * doppler));
    for (int64_t m = 0; m < tx; ++m) {
      for (int64_t k = 0; k < doppler; ++k) {
        const double phase = std::sin(0.7 * static_cast<double>(k) + static_cast<double>(m)) * 2.0;
        result.phase_codes[static_cast<std::size_t>(m * doppler + k)] =
            std::complex<double>(std::cos(phase), std::sin(phase));
      }
    }
  } else {
    throw std::invalid_argument("unknown DDM dense code factory: " + factory);
  }
  return result;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  auto active = scen;
  const bool axis = scen.axis("velocity_axis") != 0;
  if (!axis) {
    active.recipe.inputs.erase(active.recipe.inputs.begin());
    active.recipe.outputs.drain.clear();
    active.recipe.outputs.unwired = {"velocity_bins_out"};
  }
  hr::DDMDenseDemultiplexParams params;
  params.ddm_code = code(scen);
  const auto extents = perf::direct_recipe_extents(scen.recipe, scen.values);
  if (axis)
    return perf::run_recipe_case<Op, 2>(active, plan, extents, params);
  return perf::run_recipe_case<Op, 1>(active, plan, extents, params);
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "DDMDenseDemultiplexOp";
  row.variant =
      (scen.params["ddm_code"]["factory"].as<std::string>() == "nonlinear_phase_code" ? "ddm_nl"
                                                                                      : "ddm") +
      std::string(scen.axis("velocity_axis") ? "_axis" : "");
  row.input_shape = perf::recipe_shape(scen.recipe.inputs.at(1), scen.values);
  row.k = scen.axis("doppler");
  row.notes = "live_override=not_measured";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double elems = static_cast<double>(scen.axis("rx")) *
                       static_cast<double>(scen.axis("doppler")) *
                       static_cast<double>(scen.axis("range"));
  perf::set_recipe_throughput(row, elems, std::nullopt,
                              2.0 * elems * sizeof(std::complex<perf::Scalar>));
  row.notes =
      scen.params["ddm_code"]["factory"].as<std::string>() == "nonlinear_phase_code"
          ? "DDM nonlinear phase-code correlation; cold_ms includes weights and FFT setup; output "
            "keeps the full Doppler axis"
          : "DDM linear per-band gather; cold_ms includes the initial demultiplex table build";
  row.notes += scen.axis("velocity_axis") ? "; velocity_axis=wired" : "; velocity_axis=absent";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_ddm_dense_demultiplex"; }
  static const char* tool_description() {
    return "DDMDenseDemultiplexOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 4, 4);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.DDMDenseDemultiplexOp.float", "",
        {{"velocity_bins", {"doppler"}, "float32", "device", "centered_axis_ramp", {}},
         {"rd_maps", {"rx", "doppler", "range"}, "complex64", "device", "iq_cube", {}}},
        {"rd_maps_out", {"velocity_bins_out"}, {}, 4U}, {}, {});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 4);
    if (scen.axis("tx") < 2 || scen.axis("doppler") % scen.axis("tx") != 0 ||
        scen.axis("velocity_axis") > 1)
      throw std::invalid_argument("DDMDenseDemultiplex invalid profile or axis fixture");
    perf::require_recipe_params(scen.params, {"ddm_code"});
    if (!scen.recipe.extension["optional_velocity_axis"] ||
        scen.recipe.extension["optional_velocity_axis"].as<std::string>() != "velocity_axis")
      throw std::invalid_argument("DDM dense recipe needs optional velocity axis declaration");
    (void)code(scen);
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
