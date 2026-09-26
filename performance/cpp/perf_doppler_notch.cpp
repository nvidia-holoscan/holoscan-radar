// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for DopplerNotchOp. Measurement only.

#include <complex>
#include <set>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

struct NotchScenario {
  std::string id;
  std::string size_id;
  int64_t num_doppler;
  int64_t num_range;
  int64_t num_tx;
  int64_t num_rx;
  bool cosine;
  bool live_override;
  std::string live_port;
  double override_value{0.0};
  bool cyclic{true};
};

struct SizeSpec {
  const char* id;
  int64_t doppler;
  int64_t range;
  const char* role;
};

NotchScenario make_scenario(const SizeSpec& size, bool cosine, bool rank4,
                            bool live_override = false) {
  const std::string variant = std::string(cosine ? "cosine" : "hard") +
                              (rank4 ? "_mimo" : "_siso") + (live_override ? "_live" : "");
  return NotchScenario{.id = "doppler_notch_" + variant + "_" + size.id,
                       .size_id = size.id,
                       .num_doppler = size.doppler,
                       .num_range = size.range,
                       .num_tx = rank4 ? 4 : 1,
                       .num_rx = rank4 ? 4 : 1,
                       .cosine = cosine,
                       .live_override = live_override};
}

std::vector<int64_t> input_shape(const NotchScenario& scenario) {
  if (scenario.num_tx > 1) {
    return {scenario.num_tx, scenario.num_rx, scenario.num_doppler, scenario.num_range};
  }
  return {scenario.num_doppler, scenario.num_range};
}

perf::TimingCollector run_graph(const NotchScenario& scenario, const perf::RunPlan& plan) {
  auto axis = perf::make_power_map_device<PerfPrecision>({scenario.num_doppler});
  auto maps = perf::make_iq_cube_device<PerfPrecision>(input_shape(scenario));
  const auto configure = [&](holoscan::Graph& graph, const auto& source,
                             holoscan::CompileOptions& options, auto connect) {
    hr::MaxTensorExtents extents;
    extents.max_doppler_bins = scenario.num_doppler;
    extents.max_range_bins = scenario.num_range;
    if (scenario.num_tx > 1) {
      extents.max_tx = scenario.num_tx;
      extents.max_rx = scenario.num_rx;
    }
    auto op = graph.op<hr::DopplerNotchOp<PerfPrecision>>("notch", extents);
    hr::DopplerNotchParams params;
    params.stop_half_width = 0.5;
    params.transition_width = scenario.cosine ? 1.0 : 0.0;
    params.cyclic_velocity_axis = scenario.cyclic;
    op.set_params(std::move(params));
    const holoscan::ConnectionOptions queue{.queue_depth = 4U};
    graph.add_flow(source->out_a, op->velocity_bins, queue);
    graph.add_flow(source->out_b, op->rd_maps, queue);
    connect(graph, source, op, queue);
    perf::PerfRecipeOutputs outputs(graph, options, "notch");
    outputs.timed(op->rd_maps_out, "rd_maps_out");
    outputs.finish(source, op);
  };
  if (!scenario.live_override) {
    return perf::run_data_recipe(
        std::array{std::move(axis), std::move(maps)}, plan, "doppler-notch-perf",
        [&](auto& graph, const auto& source, auto& options) {
          configure(graph, source, options, [](auto&, const auto&, const auto&, const auto&) {});
        });
  }
  const auto cases = std::make_tuple(
      perf::live_option("center_velocity",
                        [&] {
                          return perf::live_case(
                              "center_velocity", 0.0, scenario.override_value,
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->center_velocity, q);
                              });
                        }),
      perf::live_option("stop_half_width",
                        [&] {
                          return perf::live_case(
                              "stop_half_width", 0.0, scenario.override_value,
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->stop_half_width, q);
                              });
                        }),
      perf::live_option("transition_width", [&] {
        return perf::live_case("transition_width", 0.0, scenario.override_value,
                               [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                 graph.add_flow(source->override_out, op->transition_width, q);
                               });
      }));
  return perf::dispatch_live_case(scenario.live_port, cases, [&](const auto& selected) {
    return perf::run_selected_live_recipe(std::array{std::move(axis), std::move(maps)}, plan,
                                          "doppler-notch-perf", selected, configure);
  });
}

perf::ResultRow base_row(const NotchScenario& scenario) {
  perf::ResultRow row;
  row.op_name = "DopplerNotchOp";
  row.variant = scenario.cosine ? "cosine" : "hard";
  row.variant += scenario.num_tx > 1 ? "_mimo" : "_siso";
  row.input_shape = input_shape(scenario);
  row.k = scenario.num_doppler;
  row.live_port = scenario.live_port;
  row.notes = scenario.live_override ? "one-frame live override" : "";
  return row;
}

void fill_throughput(perf::ResultRow* row, const NotchScenario& scenario) {
  if (row->latency.median_ms <= 0.0)
    return;
  int64_t elements = 1;
  for (const int64_t extent : input_shape(scenario))
    elements *= extent;
  const double seconds = row->latency.median_ms / 1000.0;
  row->cells_per_second = static_cast<double>(elements) / seconds;
  row->estimated_effective_gbps =
      2.0 * static_cast<double>(elements) * sizeof(std::complex<PerfPrecision>) / seconds / 1e9;
}

void decorate_row(perf::ResultRow& row, const NotchScenario& scenario) {
  fill_throughput(&row, scenario);
}

struct NotchTraits : perf::PerfDriverDefaults {
  using Scenario = NotchScenario;

  static const char* tool_name() { return "perf_doppler_notch"; }
  static const char* tool_description() { return "DopplerNotchOp performance characterization"; }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 2)
      throw std::invalid_argument("DopplerNotch profile rank mismatch");
    return {id, shape, shape[0] * shape[1], id};
  }

  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 2 || shape[0] < 2)
      throw std::invalid_argument("DopplerNotch invalid profile shape");
    perf::require_keys(c.parameters, {"transition", "mimo", "cyclic"}, "parameters");
    if (!c.parameters["transition"] || !c.parameters["mimo"])
      throw std::invalid_argument("DopplerNotch case requires transition and mimo");
    const auto transition = c.parameters["transition"].as<std::string>();
    if (transition != "hard" && transition != "cosine")
      throw std::invalid_argument("DopplerNotch transition must be hard or cosine");
    const SizeSpec size{c.profile.c_str(), shape[0], shape[1], c.profile.c_str()};
    auto scenario = make_scenario(size, transition == "cosine", c.parameters["mimo"].as<bool>());
    scenario.id = c.id;
    scenario.size_id = c.profile;
    if (c.parameters["cyclic"])
      scenario.cyclic = c.parameters["cyclic"].as<bool>();
    if (c.override_value) {
      scenario.live_port = (*c.override_value)["port"].as<std::string>();
      scenario.override_value = (*c.override_value)["value"].as<double>();
      if (scenario.live_port != "center_velocity" && scenario.live_port != "stop_half_width" &&
          scenario.live_port != "transition_width")
        throw std::invalid_argument("DopplerNotch unknown live port");
      if (scenario.live_port != "center_velocity" && scenario.override_value < 0.0)
        throw std::invalid_argument("DopplerNotch widths must be nonnegative");
      scenario.live_override = true;
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<NotchTraits>(argc, argv);
}
