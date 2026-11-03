// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_sparse_mimo.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::DDMSparseDemultiplexOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

hr::DDMCodeDescriptor linear_code(int64_t tx) {
  hr::DDMCodeDescriptor result;
  result.num_tx = tx;
  for (int64_t i = 0; i < tx; ++i)
    result.doppler_offsets.push_back(static_cast<double>(i) / static_cast<double>(tx));
  return result;
}

hr::DDMSparseDemultiplexParams parameters(const Scenario& scen) {
  hr::DDMSparseDemultiplexParams params;
  params.ddm_code = linear_code(scen.axis("tx"));
  params.tx_positions = perf::recipe_linear_positions(
      scen.axis("tx"), scen.params["tx_positions"]["spacing"].as<double>(), 0);
  params.rx_positions = perf::recipe_linear_positions(
      scen.axis("rx"), scen.params["rx_positions"]["spacing"].as<double>(), 0);
  params.overlap_policy =
      perf::recipe_overlap_policy(scen.params["overlap_policy"].as<std::string>());
  params.wavelength = scen.params["wavelength"].as<double>();
  const auto window = perf::sparse_window(scen.params["snapshot_window"]);
  params.snapshot_window = {window[0], window[1]};
  return params;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  const auto params = parameters(scen);
  const auto extents = perf::sparse_mimo_extents(scen);
  return perf::run_recipe_selected_case<Op, 2>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec != "sparse_window")
          throw std::invalid_argument("unsupported sparse DDM live codec: " + codec);
        const auto window = perf::sparse_window(scen.live_value);
        const hr::DDMSparseWindowOverride value{.doppler = window[0], .range = window[1]};
        return perf::run_recipe_live_case<Op, 2>(scen, plan, extents, params, value, value);
      });
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "DDMSparseDemultiplexOp";
  row.variant = "sparse_ddm";
  row.live_port = scen.live_port;
  perf::set_input_and_trailing_map_shape(row,
                                         perf::recipe_shape(scen.recipe.inputs.at(0), scen.values));
  row.out_batch = scen.axis("detections");
  row.k = scen.axis("tx") * scen.axis("rx");
  row.notes = "honest pre-demux input and detection-sized snapshot output";
  return row;
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_ddm_sparse_demultiplex"; }
  static const char* tool_description() { return "Sparse DDM detection-count characterization"; }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 5)
      throw std::invalid_argument("DDMSparseDemultiplex profile rank mismatch");
    return {id, shape, shape[1] * shape[2] * shape[3], id};
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.DDMSparseDemultiplexOp.float", "sparse_mimo_bounds",
        {{"rd_maps", {"rx", "doppler", "range"}, "complex64", "device", "iq_cube", {}},
         {"detection_idx",
          {"detections", "two"},
          "int64",
          "device",
          "in_bounds_rd_indices",
          {},
          "tx"}},
        {"snapshot",
         {"is_valid", "array_geometry"},
         {},
         4U,
         "",
         {{"array_geometry", "array_geometry"}}},
        {{"wavelength", "wavelength"}, {"snapshot_window", "snapshot_window"}},
        {{"wavelength", "double"}, {"snapshot_window", "sparse_window"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 5);
    if (scen.axis("tx") < 2 || scen.axis("doppler") % scen.axis("tx") != 0 ||
        scen.axis("detections") < 1)
      throw std::invalid_argument("DDMSparseDemultiplex invalid profile shape");
    perf::require_recipe_params(scen.params, {"ddm_code", "tx_positions", "rx_positions",
                                              "overlap_policy", "wavelength", "snapshot_window"});
    (void)parameters(scen);
    (void)perf::sparse_mimo_extents(scen);
    if (scen.live_port == "wavelength" && scen.live_value.as<double>() <= 0)
      throw std::invalid_argument("DDM wavelength must be positive");
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
