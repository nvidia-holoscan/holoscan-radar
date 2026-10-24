// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::ClutterFilterOp<perf::Scalar>;

struct Scenario : perf::RecipeScenario {
  hr::ClutterFilterMethod method{hr::ClutterFilterMethod::kMeanSubtraction};
  hr::WindowSpec window{};
};

hr::ClutterFilterMethod parse_method(const std::string& name) {
  if (name == "mean_subtraction")
    return hr::ClutterFilterMethod::kMeanSubtraction;
  if (name == "single_delay")
    return hr::ClutterFilterMethod::kSingleDelay;
  if (name == "double_delay")
    return hr::ClutterFilterMethod::kDoubleDelay;
  if (name == "none")
    return hr::ClutterFilterMethod::kNone;
  throw std::invalid_argument("unknown clutter method: " + name);
}

std::string method_tag(hr::ClutterFilterMethod method) {
  switch (method) {
    case hr::ClutterFilterMethod::kMeanSubtraction:
      return "msub";
    case hr::ClutterFilterMethod::kSingleDelay:
      return "sdel";
    case hr::ClutterFilterMethod::kDoubleDelay:
      return "ddel";
    case hr::ClutterFilterMethod::kNone:
      return "none";
  }
  return "unk";
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  hr::MaxTensorExtents extents;
  extents.max_chirps = scen.axis("chirps");
  extents.max_range_bins = scen.axis("range_bins");
  if (scen.axis("channels") > 1)
    extents.max_rx = scen.axis("channels");
  hr::ClutterFilterParams params;
  params.method = scen.method;
  params.doppler_window = scen.window;
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec != "enum" || scen.live_port != "method")
          throw std::invalid_argument("unsupported clutter live codec: " + codec);
        const auto changed = parse_method(scen.live_value.as<std::string>());
        return perf::run_recipe_live_case_with_connector<Op, 1>(
            scen, plan, extents, params, changed, changed,
            [](auto& graph, const auto& source, const auto& op, const auto& queue) {
              graph.add_flow(source->override_out, op->method, queue);
            });
      });
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "ClutterFilterOp";
  row.variant =
      method_tag(scen.method) + "_" +
      (scen.window.type == hr::WindowType::kNone ? "rect"
                                                 : std::string(hr::to_string(scen.window.type)));
  perf::set_input_and_trailing_map_shape(row,
                                         perf::recipe_shape(scen.recipe.inputs.at(0), scen.values));
  row.out_batch = scen.axis("channels");
  row.k = scen.axis("chirps");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double cells = static_cast<double>(scen.axis("channels")) *
                       static_cast<double>(scen.axis("chirps")) *
                       static_cast<double>(scen.axis("range_bins"));
  const double slices = static_cast<double>(scen.axis("channels") * scen.axis("range_bins"));
  perf::set_recipe_throughput(row, cells, slices, 2.0 * cells * sizeof(std::complex<perf::Scalar>));
  row.notes = "end-to-end Graph latency; cells/s is complex cells/s";
  if (scen.live_port == "doppler_window")
    row.notes += "; unseen Hamming window on one frame";
  if (scen.live_port == "method")
    row.notes += "; method changes for one frame";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_clutter_filter"; }
  static const char* tool_description() {
    return "ClutterFilterOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 2, 2);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.ClutterFilterOp.float", "clutter_filter_bounds",
        {{"range_profiles",
          {"channels", "chirps", "range_bins"},
          "complex64",
          "device",
          "iq_cube",
          {"channels"}}},
        {"range_profiles_out", {}, {}, 4U},
        {{"method", "method"}, {"doppler_window", "doppler_window"}},
        {{"method", "enum"}, {"doppler_window", "window_spec"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 2);
    perf::require_recipe_params(scen.params, {"method", "doppler_window"});
    scen.method = parse_method(scen.params["method"].as<std::string>());
    scen.window = perf::recipe_window(
        perf::recipe_window_type(scen.params["doppler_window"]["type"].as<std::string>()));
    if (scen.live_port == "method")
      (void)parse_method(scen.live_value.as<std::string>());
    if (scen.live_port == "doppler_window")
      (void)perf::recipe_window_type(scen.live_value["type"].as<std::string>());
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
