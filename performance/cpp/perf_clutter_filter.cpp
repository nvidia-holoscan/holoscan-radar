// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ClutterFilterOp characterization from curated YAML cases: three input
// sizes, static method/window choices, and isolated one-frame Exact overrides
// for both live inputs. Measurement tool only; no timing thresholds.

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_profiles.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // v1 characterizes single precision (complex64).

struct ClutterScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t num_range_bins{0};
  int64_t channels{1};
  hr::ClutterFilterMethod method{hr::ClutterFilterMethod::kMeanSubtraction};
  hr::WindowSpec window{};
  bool time_window_override{false};
  bool time_method_override{false};
};

std::string method_tag(hr::ClutterFilterMethod m) {
  switch (m) {
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

std::string window_tag(const hr::WindowSpec& window) {
  return window.type == hr::WindowType::kNone ? "rect" : std::string(hr::to_string(window.type));
}

// Flip to a different valid method so a live override always changes the
// kernel path (and the output chirp count). kNone is invalid here.
hr::ClutterFilterMethod flip_method(hr::ClutterFilterMethod m) {
  if (m == hr::ClutterFilterMethod::kMeanSubtraction) {
    return hr::ClutterFilterMethod::kSingleDelay;
  }
  return hr::ClutterFilterMethod::kMeanSubtraction;
}

std::vector<int64_t> profiles_shape(const ClutterScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.num_range_bins};
  }
  return {scen.num_chirps, scen.num_range_bins};
}

perf::TimingCollector run_graph(const ClutterScenario& scen, const perf::RunPlan& plan) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(profiles_shape(scen));
  hr::MaxTensorExtents extents;
  extents.max_chirps = scen.num_chirps;
  extents.max_range_bins = scen.num_range_bins;
  if (scen.channels > 1) {
    extents.max_rx = scen.channels;
  }
  using BaseOp = hr::ClutterFilterOp<PerfPrecision>;
  const auto configure = [&](holoscan::Graph& graph, const auto& src,
                             holoscan::CompileOptions& options, auto connect) {
    auto op = graph.op<BaseOp>("clutter", extents);
    hr::ClutterFilterParams params;
    params.method = scen.method;
    params.doppler_window = scen.window;
    op.set_params(std::move(params));
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(src->out, op->range_profiles, q);
    connect(graph, src, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "clutter");
    outputs.timed(op->range_profiles_out, "range_profiles_out");
    outputs.finish(src, op);
  };
  if (!scen.time_window_override && !scen.time_method_override) {
    return perf::run_data_recipe(std::array{std::move(cube)}, plan, "clutter-perf",
                                 [&](auto& graph, const auto& src, auto& options) {
                                   configure(graph, src, options,
                                             [](auto&, const auto&, const auto&, const auto&) {});
                                 });
  }
  const auto cases = std::make_tuple(
      perf::live_option("doppler_window",
                        [&] {
                          hr::WindowSpec changed;
                          changed.type = hr::WindowType::kHamming;
                          return perf::live_case(
                              "doppler_window", scen.window, changed,
                              [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                graph.add_flow(src->override_out, op->doppler_window_in, q);
                              });
                        }),
      perf::live_option("method", [&] {
        return perf::live_case("method", scen.method, flip_method(scen.method),
                               [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                 graph.add_flow(src->override_out, op->method, q);
                               });
      }));
  const std::string port = scen.time_window_override ? "doppler_window" : "method";
  const char* graph_name = scen.time_window_override ? "clutter-window-perf" : "clutter-perf";
  return perf::dispatch_live_case(port, cases, [&](const auto& selected) {
    return perf::run_selected_live_recipe(std::array{std::move(cube)}, plan, graph_name, selected,
                                          configure);
  });
}

void fill_throughput(perf::ResultRow* row, const ClutterScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double chirps = static_cast<double>(scen.num_chirps);
  const double bins = static_cast<double>(scen.num_range_bins);
  row->cells_per_second = batch * chirps * bins / sec;
  row->slices_per_second = batch * bins / sec;

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double bytes = 2.0 * batch * chirps * bins * csz;
  row->estimated_effective_gbps = bytes / sec / 1e9;
}

perf::ResultRow base_row(const ClutterScenario& scen) {
  perf::ResultRow row;
  row.op_name = "ClutterFilterOp";
  row.variant = method_tag(scen.method) + "_" + window_tag(scen.window);
  perf::set_input_and_trailing_map_shape(row, profiles_shape(scen));
  row.out_batch = scen.channels;
  row.k = scen.num_chirps;
  row.live_port =
      scen.time_window_override ? "doppler_window" : (scen.time_method_override ? "method" : "");
  return row;
}

void decorate_row(perf::ResultRow& row, const ClutterScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes = "end-to-end Graph latency; cells/s is complex cells/s";
  if (scen.time_window_override)
    row.notes += "; unseen Hamming window on one frame";
  if (scen.time_method_override)
    row.notes += "; method changes for one frame";
}

struct ClutterTraits : perf::PerfDriverDefaults {
  using Scenario = ClutterScenario;

  static const char* tool_name() { return "perf_clutter_filter"; }
  static const char* tool_description() {
    return "ClutterFilterOp performance characterization (HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 2)
      throw std::invalid_argument("profile shape rank mismatch");
    return {id, shape, shape[0] * shape[1], id};
  }

  static ClutterScenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    perf::require_keys(c.parameters, {"method", "doppler_window", "channels"}, "parameters");
    auto method_of = [](const std::string& name) {
      if (name == "mean_subtraction")
        return hr::ClutterFilterMethod::kMeanSubtraction;
      if (name == "single_delay")
        return hr::ClutterFilterMethod::kSingleDelay;
      if (name == "double_delay")
        return hr::ClutterFilterMethod::kDoubleDelay;
      if (name == "none")
        return hr::ClutterFilterMethod::kNone;
      throw std::invalid_argument("unknown clutter method: " + name);
    };
    auto window_of = [](const std::string& name) {
      if (name == "none")
        return hr::WindowType::kNone;
      if (name == "hann")
        return hr::WindowType::kHann;
      if (name == "hamming")
        return hr::WindowType::kHamming;
      throw std::invalid_argument("unknown clutter window: " + name);
    };
    if (shape.size() != 2)
      throw std::invalid_argument("clutter profile shape must be rank 2");
    ClutterScenario scenario;
    scenario.id = c.id;
    scenario.size_id = c.profile;
    scenario.num_chirps = shape[0];
    scenario.num_range_bins = shape[1];
    scenario.method = method_of(c.parameters["method"].as<std::string>());
    scenario.window.type = window_of(c.parameters["doppler_window"].as<std::string>());
    if (c.parameters["channels"])
      scenario.channels = c.parameters["channels"].as<int64_t>();
    if (scenario.channels <= 0)
      throw std::invalid_argument("channels must be positive");
    if (c.override_value.has_value()) {
      const auto port = (*c.override_value)["port"].as<std::string>();
      const auto value = (*c.override_value)["value"].as<std::string>();
      if (port == "method") {
        scenario.time_method_override = true;
        if (method_of(value) != flip_method(scenario.method)) {
          throw std::invalid_argument("method override must select the one-frame flip");
        }
      } else if (port == "doppler_window") {
        scenario.time_window_override = true;
        if (window_of(value) != hr::WindowType::kHamming) {
          throw std::invalid_argument("window override must select Hamming");
        }
      }
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<ClutterTraits>(argc, argv);
}
