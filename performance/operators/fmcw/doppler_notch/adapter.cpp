// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::DopplerNotchOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  hr::MaxTensorExtents extents;
  extents.max_doppler_bins = scen.axis("doppler");
  extents.max_range_bins = scen.axis("range");
  if (scen.axis("tx") > 1) {
    extents.max_tx = scen.axis("tx");
    extents.max_rx = scen.axis("rx");
  }
  hr::DopplerNotchParams params;
  params.center_velocity = scen.params["center_velocity"].as<double>();
  params.stop_half_width = scen.params["stop_half_width"].as<double>();
  params.transition_width = scen.params["transition_width"].as<double>();
  params.cyclic_velocity_axis = scen.params["cyclic_velocity_axis"].as<bool>();
  return perf::run_recipe_selected_case<Op, 2>(scen, plan, extents, params);
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "DopplerNotchOp";
  row.variant = scen.params["transition_width"].as<double>() > 0.0 ? "cosine" : "hard";
  row.variant += scen.axis("tx") > 1 ? "_mimo" : "_siso";
  row.input_shape = perf::recipe_shape(scen.recipe.inputs.at(1), scen.values);
  row.k = scen.axis("doppler");
  row.live_port = scen.live_port;
  row.notes = scen.live_port.empty() ? "" : "one-frame live override";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  double elements = 1.0;
  for (const auto extent : row.input_shape)
    elements *= static_cast<double>(extent);
  perf::set_recipe_throughput(row, elements, std::nullopt,
                              2.0 * elements * sizeof(std::complex<perf::Scalar>));
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_doppler_notch"; }
  static const char* tool_description() { return "DopplerNotchOp performance characterization"; }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 2, 2);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.DopplerNotchOp.float", "doppler_notch_bounds",
        {{"velocity_bins", {"doppler"}, "float32", "device", "centered_axis_ramp", {}},
         {"rd_maps",
          {"tx", "rx", "doppler", "range"},
          "complex64",
          "device",
          "iq_cube",
          {"tx", "rx"}}},
        {"rd_maps_out", {}, {}, 4U},
        {{"center_velocity", "center_velocity"},
         {"stop_half_width", "stop_half_width"},
         {"transition_width", "transition_width"}},
        {{"center_velocity", "double"},
         {"stop_half_width", "double"},
         {"transition_width", "double"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 2);
    if (shape[0] < 2 || (scen.axis("tx") > 1 && scen.axis("rx") != 4) ||
        (scen.axis("tx") != 1 && scen.axis("tx") != 4))
      throw std::invalid_argument("DopplerNotch invalid profile or MIMO fixture");
    perf::require_recipe_params(scen.params, {"center_velocity", "stop_half_width",
                                              "transition_width", "cyclic_velocity_axis"});
    for (const auto name : {"center_velocity", "stop_half_width", "transition_width"})
      (void)scen.params[name].as<double>();
    (void)scen.params["cyclic_velocity_axis"].as<bool>();
    if (!scen.live_port.empty()) {
      const auto value = scen.live_value.as<double>();
      if (scen.live_port != "center_velocity" && value < 0.0)
        throw std::invalid_argument("DopplerNotch widths must be nonnegative");
    }
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
