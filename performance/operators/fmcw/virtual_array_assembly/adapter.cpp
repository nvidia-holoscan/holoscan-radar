// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::VirtualArrayAssemblyOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {
  hr::OverlapPolicy policy{hr::OverlapPolicy::kDedup};
};

std::string policy_tag(hr::OverlapPolicy policy) {
  switch (policy) {
    case hr::OverlapPolicy::kDedup:
      return "dedup";
    case hr::OverlapPolicy::kAverage:
      return "avg";
    case hr::OverlapPolicy::kKeepAll:
      return "keep";
  }
  return "unk";
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  const auto extents = perf::direct_recipe_extents(scen.recipe, scen.values);
  hr::VirtualArrayAssemblyParams params;
  params.tx_positions = perf::recipe_linear_positions(
      scen.axis("tx"), scen.params["tx_positions"]["spacing"].as<double>(), 1);
  params.rx_positions = perf::recipe_linear_positions(
      scen.axis("rx"), scen.params["rx_positions"]["spacing"].as<double>(), 1);
  params.wavelength = scen.params["wavelength"].as<double>();
  params.overlap_policy = scen.policy;
  return perf::run_recipe_selected_case<Op, 1>(scen, plan, extents, params);
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "VirtualArrayAssemblyOp";
  row.variant = policy_tag(scen.policy);
  row.input_shape = perf::recipe_shape(scen.recipe.inputs.at(0), scen.values);
  row.k = scen.axis("doppler");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double elems = static_cast<double>(scen.axis("tx")) * static_cast<double>(scen.axis("rx")) *
                       static_cast<double>(scen.axis("doppler")) *
                       static_cast<double>(scen.axis("range"));
  perf::set_recipe_throughput(row, elems, std::nullopt,
                              2.0 * elems * sizeof(std::complex<perf::Scalar>));
  row.notes = "virtual channel flatten + geometry emit; cells/s counts input complex samples";
  if (!scen.live_port.empty())
    row.notes += "; one-frame wavelength change";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_virtual_array_assembly"; }
  static const char* tool_description() {
    return "VirtualArrayAssemblyOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 4, 4);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.VirtualArrayAssemblyOp.float", "",
        {{"rd_maps", {"tx", "rx", "doppler", "range"}, "complex64", "device", "iq_cube", {}}},
        {"rd_maps_out", {"array_geometry"}, {}, 4U, "", {{"array_geometry", "array_geometry"}}},
        {{"wavelength", "wavelength"}}, {{"wavelength", "double"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 4);
    perf::require_recipe_params(scen.params,
                                {"tx_positions", "rx_positions", "wavelength", "overlap_policy"});
    for (const auto field : {"tx_positions", "rx_positions"}) {
      perf::require_keys(scen.params[field], {"factory", "spacing"}, field);
      if (scen.params[field]["factory"].as<std::string>() != "linear_positions" ||
          scen.params[field]["spacing"].as<double>() <= 0.0)
        throw std::invalid_argument("VirtualArrayAssembly invalid positions factory");
    }
    if (scen.params["wavelength"].as<double>() <= 0.0)
      throw std::invalid_argument("VirtualArrayAssembly wavelength must be positive");
    scen.policy = perf::recipe_overlap_policy(scen.params["overlap_policy"].as<std::string>());
    if (!scen.live_port.empty() && scen.live_value.as<double>() <= 0.0)
      throw std::invalid_argument("VirtualArrayAssembly live wavelength must be positive");
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
