// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for DopplerNotchOp. Measurement only.

#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
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
};

struct SizeSpec {
  const char* id;
  int64_t doppler;
  int64_t range;
  const char* role;
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> sizes = {
      {"xs", 32, 128, "small map; fast smoke"},
      {"md", 128, 512, "common development size"},
      {"xl", 256, 2048, "range-heavy upper characterization"}};
  return sizes;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& size : size_ladder()) {
    if (size.id == id) return &size;
  }
  return nullptr;
}

NotchScenario make_scenario(const SizeSpec& size, bool cosine,
                            bool rank4, bool live_override = false) {
  const std::string variant =
      std::string(cosine ? "cosine" : "hard") +
      (rank4 ? "_mimo" : "_siso") +
      (live_override ? "_live" : "");
  return NotchScenario{
      .id = "doppler_notch_" + variant + "_" + size.id,
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
    return {scenario.num_tx, scenario.num_rx, scenario.num_doppler,
            scenario.num_range};
  }
  return {scenario.num_doppler, scenario.num_range};
}

perf::TimingCollector run_notch_graph(const NotchScenario& scenario,
                                      const perf::RunPlan& plan) {
  auto axis =
      perf::make_power_map_device<PerfPrecision>({scenario.num_doppler});
  auto maps =
      perf::make_iq_cube_device<PerfPrecision>(input_shape(scenario));
  const auto configure = [&](holoscan::Graph& graph,
                             holoscan::CompileOptions& options) {
    hr::MaxTensorExtents extents;
    extents.max_doppler_bins = scenario.num_doppler;
    extents.max_range_bins = scenario.num_range;
    if (scenario.num_tx > 1) {
      extents.max_tx = scenario.num_tx;
      extents.max_rx = scenario.num_rx;
    }
    auto op =
        graph.op<hr::DopplerNotchOp<PerfPrecision>>("notch", extents);
    hr::DopplerNotchParams params;
    params.stop_half_width = 0.5;
    params.transition_width = scenario.cosine ? 1.0 : 0.0;
    op.set_params(std::move(params));
    const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
    perf::bind_tensor_out(options, "notch", "rd_maps_out");
    return std::make_pair(op, sink);
  };
  if (scenario.live_override) {
    return perf::run_perf_graph_dual_with_override(
        std::move(axis), std::move(maps), plan.total_frames,
        plan.override_at, 0.0, 1.0, "doppler-notch-perf",
        [&](holoscan::Graph& graph, const auto& source,
            holoscan::CompileOptions& options) {
          auto [op, sink] = configure(graph, options);
          const holoscan::ConnectionOptions queue{.queue_depth = 4U};
          graph.add_flow(source->out_a, op->velocity_bins, queue);
          graph.add_flow(source->out_b, op->rd_maps, queue);
          graph.add_flow(source->override_out, op->center_velocity, queue);
          graph.add_flow(op->rd_maps_out, sink->in, queue);
          graph.partition("perf").add(source).add(op).add(sink);
        });
  }
  return perf::run_perf_graph_dual(
      std::move(axis), std::move(maps), plan.total_frames,
      "doppler-notch-perf",
      [&](holoscan::Graph& graph, const auto& source,
          holoscan::CompileOptions& options) {
        auto [op, sink] = configure(graph, options);
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        graph.add_flow(source->out_a, op->velocity_bins, queue);
        graph.add_flow(source->out_b, op->rd_maps, queue);
        graph.add_flow(op->rd_maps_out, sink->in, queue);
        graph.partition("perf").add(source).add(op).add(sink);
      });
}

perf::ResultRow base_row(const NotchScenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "DopplerNotchOp";
  row.variant = scenario.cosine ? "cosine" : "hard";
  row.variant += scenario.num_tx > 1 ? "_mimo" : "_siso";
  row.size_id = scenario.size_id;
  row.input_shape = input_shape(scenario);
  row.k = scenario.num_doppler;
  row.notes = scenario.live_override ? "center_velocity live override" : "";
  return row;
}

void fill_throughput(perf::ResultRow* row,
                     const NotchScenario& scenario) {
  if (row->latency.median_ms <= 0.0) return;
  int64_t elements = 1;
  for (const int64_t extent : input_shape(scenario)) elements *= extent;
  const double seconds = row->latency.median_ms / 1000.0;
  row->cells_per_second = static_cast<double>(elements) / seconds;
  row->estimated_effective_gbps =
      2.0 * static_cast<double>(elements) *
      sizeof(std::complex<PerfPrecision>) / seconds / 1e9;
}

struct NotchTraits : perf::PerfDriverDefaults {
  using Scenario = NotchScenario;

  static const char* tool_name() { return "perf_doppler_notch"; }
  static const char* tool_description() {
    return "DopplerNotchOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Hard/cosine and rank-2/rank-4 sweep";
  }
  static const char* sizes_subset_help() {
    return "Notch sweep over selected sizes";
  }
  static const char* full_help() {
    return "Full notch variant and size matrix";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    for (const auto& size : size_ladder()) {
      result.push_back(perf::SizeDef{
          .id = size.id,
          .shape = {size.doppler, size.range},
          .cells = size.doppler * size.range,
          .role = size.role});
    }
    return result;
  }

  static std::vector<Scenario> build_size_sweep(
      const std::vector<std::string>& ids) {
    std::vector<Scenario> result;
    for (const auto& id : ids) {
      const SizeSpec* size = find_size(id);
      if (size == nullptr) continue;
      for (const bool cosine : {false, true}) {
        result.push_back(make_scenario(*size, cosine, false));
        result.push_back(make_scenario(*size, cosine, true));
      }
    }
    return result;
  }

  static std::vector<Scenario> build_default_matrix() {
    auto result = build_size_sweep({"xs", "md", "xl"});
    result.push_back(
        make_scenario(*find_size("md"), true, true, true));
    return result;
  }

  static std::vector<Scenario> build_full_matrix() {
    return build_size_sweep({"xs", "md", "xl"});
  }

  static void run_scenario(const Scenario& scenario,
                           const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(settings, scenario.live_override);
    const auto samples = perf::collect_scenario_samples(
        settings, plan, [&](int, const perf::RunPlan& run) {
          return run_notch_graph(scenario, run);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    perf::ResultRow row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scenario);
    report->results.push_back(std::move(row));
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(make_scenario(*find_size("xs"), false, false),
                   settings, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<NotchTraits>(argc, argv);
}
