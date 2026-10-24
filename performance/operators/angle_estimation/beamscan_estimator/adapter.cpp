// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for BeamscanEstimatorOp — conventional
// sparse-path Bartlett DOA on a scan grid. Sweeps detection count, channel
// count, snapshot count, and scan-grid size. Measurement tool only: no timing
// assertions.
// See docs/planning/OPERATOR_DEVELOPMENT.md.
//
//  Graph-based rewrite of the HSDK 4.x Application harness.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_payloads.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_registered_recipe.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_yaml_pilot.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace perf_beamscan_estimator_adapter {

using perf::linspace;
using perf::make_ula_geometry;
using perf::emit_host_vector;
using perf::emit_wire_doubles;

using PerfPrecision = perf::Scalar;

struct BeamScenario : perf::RecipeScenario {};

perf::TimingCollector run_graph(const BeamScenario& scen, const perf::RunPlan& plan) {
  hr::BeamscanParams initial;
  initial.array_geometry =
      perf::recipe_ula_geometry(scen.params["array_geometry"], scen.axis("channels"));
  initial.az_coordinates = perf::recipe_coordinates(scen.params["az_coordinates"], scen.values);
  initial.el_coordinates = perf::recipe_coordinates(scen.params["el_coordinates"], scen.values);
  initial.peak_interpolation =
      perf::recipe_peak_interpolation(scen.params["peak_interpolation"].as<std::string>());
  initial.array_taper = perf::recipe_window(
      perf::recipe_window_type(scen.params["array_taper"]["type"].as<std::string>()));
  const auto extents = perf::direct_recipe_extents(scen.recipe, scen.values);
  using Op = hr::BeamscanEstimatorOp<PerfPrecision>;
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, extents, initial, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec != "enum" || scen.live_port != "peak_interpolation")
          throw std::invalid_argument("unsupported Beamscan live codec: " + codec);
        const auto changed = perf::recipe_peak_interpolation(scen.live_value.as<std::string>());
        return perf::run_recipe_live_case_with_connector<Op, 1>(
            scen, plan, extents, initial, changed, changed,
            [](auto& graph, const auto& source, const auto& op, const auto& queue) {
              graph.add_flow(source->override_out, op->peak_interpolation, queue);
            });
      });
}

void fill_throughput(perf::ResultRow* row, const BeamScenario& scen) {
  const double n = static_cast<double>(scen.axis("items"));
  const double beams = static_cast<double>(scen.axis("az")) * static_cast<double>(scen.axis("el"));
  const double ch = static_cast<double>(scen.axis("channels"));

  const double csz = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double snap_bytes = n * ch * static_cast<double>(scen.axis("looks")) * csz;
  const double weight_bytes = beams * ch * csz;
  const double spec_bytes = n * beams * static_cast<double>(sizeof(PerfPrecision));
  const double aoa_bytes = n * 2.0 * static_cast<double>(sizeof(PerfPrecision));
  perf::set_recipe_throughput(*row, n, n * beams,
                              snap_bytes + weight_bytes + spec_bytes + aoa_bytes);
}

perf::ResultRow base_row(const BeamScenario& scen) {
  perf::ResultRow row;
  row.op_name = "BeamscanEstimatorOp";
  row.variant = "bartlett";
  row.map_shape = {scen.axis("az"), scen.axis("el")};
  row.input_shape = {scen.axis("items"), scen.axis("channels"), scen.axis("looks")};
  row.out_batch = scen.axis("items");
  row.k = scen.axis("channels");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const BeamScenario& scen) {
  fill_throughput(&row, scen);
  row.notes =
      "cold_ms is first-frame latency; median/p90 "
      "are post-warmup steady state; override_ms "
      "isolates " +
      scen.live_port + "; cells/s is detections/s";
  if (row.live_port.empty())
    row.notes = "default parameters; no live override";
}

struct BeamTraits : perf::PerfDriverDefaults {
  using Scenario = BeamScenario;

  static const char* tool_name() { return "perf_beamscan_estimator"; }
  static const char* tool_description() {
    return "BeamscanEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 5, 3);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<hr::BeamscanEstimatorOp<PerfPrecision>>(
        suite, "radar.BeamscanEstimatorOp.float", "",
        {{"snapshot", {"items", "channels", "looks"}, "complex64", "device", "iq_cube", {}}},
        {"aoa_estimates", {"angle_spectrum"}, {}, 8U},
        {{"array_geometry", "array_geometry"},
         {"az_coordinates", "az_coordinates"},
         {"el_coordinates", "el_coordinates"},
         {"array_taper", "array_taper"},
         {"peak_interpolation", "peak_interpolation"}},
        {{"array_geometry", "ula_geometry_patch"},
         {"az_coordinates", "shifted_linspace"},
         {"el_coordinates", "shifted_linspace"},
         {"array_taper", "window_spec"},
         {"peak_interpolation", "enum"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scenario = perf::pilot_scenario<BeamScenario>(suite, c, shape, 5);
    if (shape[0] < 2)
      throw std::invalid_argument("Beamscan invalid shape");
    const auto& params = scenario.params;
    perf::require_recipe_params(params, {"array_geometry", "az_coordinates", "el_coordinates",
                                         "array_taper", "peak_interpolation"});
    (void)perf::recipe_window_type(params["array_taper"]["type"].as<std::string>());
    (void)perf::recipe_peak_interpolation(params["peak_interpolation"].as<std::string>());
    (void)perf::recipe_ula_geometry(params["array_geometry"], scenario.axis("channels"));
    (void)perf::recipe_coordinates(params["az_coordinates"], scenario.values);
    (void)perf::recipe_coordinates(params["el_coordinates"], scenario.values);
    if (scenario.live_port == "array_geometry") {
      auto geometry = YAML::Clone(params["array_geometry"]);
      geometry["spacing_wavelengths"] = scenario.live_value["spacing_wavelengths"];
      (void)perf::recipe_ula_geometry(geometry, scenario.axis("channels"));
    } else if (scenario.live_port == "az_coordinates" || scenario.live_port == "el_coordinates") {
      const auto axis =
          scenario.live_port == "az_coordinates" ? "az_coordinates" : "el_coordinates";
      (void)perf::recipe_coordinates(params[axis], scenario.values,
                                     scenario.live_value["shift"].as<double>());
    } else if (scenario.live_port == "array_taper") {
      (void)perf::recipe_window_type(scenario.live_value["type"].as<std::string>());
    } else if (scenario.live_port == "peak_interpolation") {
      (void)perf::recipe_peak_interpolation(scenario.live_value.as<std::string>());
    } else if (!scenario.live_port.empty()) {
      throw std::invalid_argument("unknown Beamscan live port");
    }
    return scenario;
  }
};

}  // namespace perf_beamscan_estimator_adapter

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<perf_beamscan_estimator_adapter::BeamTraits>(argc, argv);
}
