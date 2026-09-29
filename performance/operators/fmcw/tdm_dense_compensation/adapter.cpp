// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmcw/tdm_dense_compensation_op/tdm_dense_compensation_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::TDMDenseCompensationOp<float>;
struct Scenario : perf::RecipeScenario {};

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  const auto extents = perf::direct_recipe_extents(scen.recipe, scen.values);
  hr::TDMDenseCompensationParams params;
  params.waveform = perf::recipe_fmcw_waveform(scen.params["waveform"]);
  return perf::run_recipe_selected_case<Op, 1>(scen, plan, extents, params);
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "TDMDenseCompensationOp";
  row.variant = "tdm";
  row.input_shape = perf::recipe_shape(scen.recipe.inputs.at(0), scen.values);
  row.k = scen.axis("doppler");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double elems = static_cast<double>(scen.axis("tx")) * static_cast<double>(scen.axis("rx")) *
                       static_cast<double>(scen.axis("doppler")) *
                       static_cast<double>(scen.axis("range"));
  perf::set_recipe_throughput(row, elems, std::nullopt, 2.0 * elems * sizeof(std::complex<float>));
  row.notes =
      "TDM per-bin complex multiply; cold_ms includes the initial "
      "correction-table build; median/p90 are steady state; cells/s "
      "counts complex samples touched";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_tdm_dense_compensation"; }
  static const char* tool_description() {
    return "TDMDenseCompensationOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 4, 4);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.TDMDenseCompensationOp.float", "",
        {{"rd_maps", {"tx", "rx", "doppler", "range"}, "complex64", "device", "iq_cube", {}}},
        {"rd_maps_out", {}, {}, 4U}, {{"waveform", "waveform"}},
        {{"waveform", "waveform_descriptor"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 4);
    if (shape[0] < 2)
      throw std::invalid_argument("TDMDenseCompensation invalid profile shape");
    perf::require_recipe_params(scen.params, {"waveform"});
    (void)perf::recipe_fmcw_waveform(scen.params["waveform"]);
    if (!scen.live_port.empty())
      (void)perf::recipe_fmcw_waveform(scen.params["waveform"],
                                       scen.live_value["pri_seconds"].as<double>());
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
