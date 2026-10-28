// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/enums.hpp"
#include "sar/autofocus_op/autofocus_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::AutofocusOp<perf::Scalar>;

struct Scenario : perf::RecipeScenario {
  int64_t num_iterations{4};
  double window_fraction{0.5};
  int64_t min_scatterers{16};
  double max_correction{0.0};
};

hr::AutofocusAzimuthAxis parse_axis(const std::string& name) {
  if (name == "u")
    return hr::AutofocusAzimuthAxis::kU;
  if (name == "v")
    return hr::AutofocusAzimuthAxis::kV;
  throw std::invalid_argument("unknown autofocus azimuth_axis: " + name);
}

hr::AutofocusParams params_for(const Scenario& scen) {
  hr::AutofocusParams params;
  params.algorithm = hr::AutofocusAlgorithm::kPga;
  params.azimuth_axis = parse_axis(scen.params["azimuth_axis"].as<std::string>());
  params.num_iterations = scen.num_iterations;
  params.window_fraction = scen.window_fraction;
  params.min_scatterers = scen.min_scatterers;
  params.max_correction = scen.max_correction;
  return params;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  hr::MaxTensorExtents extents;
  extents.max_image_v = scen.axis("range_bins");
  extents.max_image_u = scen.axis("azimuth");
  const auto params = params_for(scen);
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec == "int64") {
          const auto changed = scen.live_value.as<int64_t>();
          if (scen.live_port == "num_iterations") {
            return perf::run_recipe_live_case_with_connector<Op, 1>(
                scen, plan, extents, params, changed, changed,
                [](auto& graph, const auto& source, const auto& op, const auto& queue) {
                  graph.add_flow(source->override_out, op->num_iterations, queue);
                });
          }
          if (scen.live_port == "min_scatterers") {
            return perf::run_recipe_live_case_with_connector<Op, 1>(
                scen, plan, extents, params, changed, changed,
                [](auto& graph, const auto& source, const auto& op, const auto& queue) {
                  graph.add_flow(source->override_out, op->min_scatterers, queue);
                });
          }
        }
        if (codec == "double") {
          const auto changed = scen.live_value.as<double>();
          if (scen.live_port == "window_fraction") {
            return perf::run_recipe_live_case_with_connector<Op, 1>(
                scen, plan, extents, params, changed, changed,
                [](auto& graph, const auto& source, const auto& op, const auto& queue) {
                  graph.add_flow(source->override_out, op->window_fraction, queue);
                });
          }
          if (scen.live_port == "max_correction") {
            return perf::run_recipe_live_case_with_connector<Op, 1>(
                scen, plan, extents, params, changed, changed,
                [](auto& graph, const auto& source, const auto& op, const auto& queue) {
                  graph.add_flow(source->override_out, op->max_correction, queue);
                });
          }
        }
        throw std::invalid_argument("unsupported autofocus live port/codec: " + scen.live_port +
                                    "/" + codec);
      });
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "AutofocusOp";
  // The variant is what a reader needs to tell these rows apart: the window is
  // the accuracy knob and the iteration count is the cost knob.
  row.variant = "w" + std::to_string(static_cast<int>(scen.window_fraction * 100.0 + 0.5)) + "_i" +
                std::to_string(scen.num_iterations);
  perf::set_input_and_trailing_map_shape(row,
                                         perf::recipe_shape(scen.recipe.inputs.at(0), scen.values));
  row.out_batch = 1;
  row.k = scen.num_iterations;
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double rows = static_cast<double>(scen.axis("range_bins"));
  const double pixels = rows * static_cast<double>(scen.axis("azimuth"));
  const double cell = static_cast<double>(sizeof(std::complex<perf::Scalar>));
  // Every iteration corrects and re-scores the whole image, so traffic tracks
  // the image per iteration and not the selected range bins -- measured, by
  // widening the selection 32-fold for no cost. On top of that sit the two
  // passes that always happen: the strong-bin scan and the copy to image_out.
  // This is an upper bound on the accepted iterations: the stage stops early
  // once an iteration fails to improve sharpness.
  const double bytes = (2.0 + 4.0 * static_cast<double>(scen.num_iterations)) * pixels * cell;
  perf::set_recipe_throughput(row, pixels, rows, bytes);
  row.notes = "end-to-end Graph latency; cells/s is complex image pixels/s";
  if (scen.min_scatterers > scen.axis("range_bins"))
    row.notes += "; early-out floor (too few scatterers), no iterations run";
  else if (scen.max_correction > 0.0 && scen.max_correction < 1.0)
    row.notes += "; estimate paid for then rejected over max_correction";
  if (scen.window_fraction < 0.5)
    row.notes += "; narrow window must not shorten the gradient transform";
  if (!scen.live_port.empty())
    row.notes += "; " + scen.live_port + " changes for one frame";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_autofocus"; }
  static const char* tool_description() {
    return "AutofocusOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 2, 2);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.AutofocusOp.float", "",
        {{"image", {"range_bins", "azimuth"}, "complex64", "device", "scatterer_image", {}}},
        {"image_out", {"phase_correction", "sharpness"}, {}, 4U},
        {{"num_iterations", "num_iterations"},
         {"window_fraction", "window_fraction"},
         {"min_scatterers", "min_scatterers"},
         {"max_correction", "max_correction"}},
        {{"num_iterations", "int64"},
         {"window_fraction", "double"},
         {"min_scatterers", "int64"},
         {"max_correction", "double"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 2);
    perf::require_recipe_params(scen.params,
                                {"algorithm", "azimuth_axis", "num_iterations", "window_fraction",
                                 "min_scatterers", "max_correction"});
    if (scen.params["algorithm"].as<std::string>() != "pga")
      throw std::invalid_argument("autofocus recipe: algorithm must be pga");
    (void)parse_axis(scen.params["azimuth_axis"].as<std::string>());
    scen.num_iterations = scen.params["num_iterations"].as<int64_t>();
    scen.window_fraction = scen.params["window_fraction"].as<double>();
    scen.min_scatterers = scen.params["min_scatterers"].as<int64_t>();
    scen.max_correction = scen.params["max_correction"].as<double>();
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
