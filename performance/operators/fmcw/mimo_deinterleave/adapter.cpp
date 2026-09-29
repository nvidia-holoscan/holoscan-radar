// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::MIMODeinterleaveOp<float>;
struct Scenario : perf::RecipeScenario {};

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  const auto extents = perf::direct_recipe_extents(scen.recipe, scen.values);
  hr::MIMODeinterleaveParams params;
  params.tx_schedule.num_tx = scen.axis("tx");
  return perf::run_recipe_case<Op, 1>(scen, plan, extents, params);
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "MIMODeinterleaveOp";
  row.variant = "rr";
  row.input_shape = perf::recipe_shape(scen.recipe.inputs.at(0), scen.values);
  row.k = scen.axis("chirps_per_tx");
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double elems = static_cast<double>(scen.axis("tx")) * static_cast<double>(scen.axis("rx")) *
                       static_cast<double>(scen.axis("chirps_per_tx")) *
                       static_cast<double>(scen.axis("samples"));
  const double bytes = 2.0 * elems * sizeof(std::complex<float>);
  perf::set_recipe_throughput(row, elems, std::nullopt, bytes);
  row.notes =
      "pure gather marshaling; cold_ms is first-frame latency; "
      "median/p90 are steady state; cells/s counts complex samples moved";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_mimo_deinterleave"; }
  static const char* tool_description() {
    return "MIMODeinterleaveOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 4)
      throw std::invalid_argument("profile shape rank mismatch");
    const auto slow = shape[0] * shape[2];
    return {id, {shape[1], slow, shape[3]}, shape[1] * slow * shape[3], id};
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.MIMODeinterleaveOp.float", "",
        {{"adc_cube", {"rx", "slow_time", "samples"}, "complex64", "device", "iq_cube", {}}},
        {"adc_cube_out", {}, {}, 4U}, {});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 4);
    perf::require_recipe_params(scen.params, {"tx_schedule"});
    perf::require_keys(scen.params["tx_schedule"], {"factory"}, "tx_schedule");
    if (scen.params["tx_schedule"]["factory"].as<std::string>() != "round_robin" ||
        !scen.live_port.empty())
      throw std::invalid_argument("MIMODeinterleave supports round-robin schedule only");
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
