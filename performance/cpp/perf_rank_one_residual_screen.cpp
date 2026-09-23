// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RankOneResidualScreenOp performance characterization. Sweeps item, channel,
// look, and selection counts; every run includes one Exact live geometry
// override. Measurement only: no timing thresholds.

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

struct SizeSpec {
  std::string id;
  int64_t items;
  int64_t channels;
  int64_t looks;
  std::string role;
};

struct ScreenScenario {
  std::string id;
  std::string size_id;
  int64_t items{0};
  int64_t channels{0};
  int64_t looks{0};
  double selection_fraction{0.25};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {{"xs", 32, 8, 3, "small; performance smoke"},
                                               {"sm", 256, 16, 5, "compact sparse-angle workload"},
                                               {"md", 1024, 32, 9, "common multi-look workload"},
                                               {"lg", 4096, 64, 9, "dense detection workload"}};
  return ladder;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& size : size_ladder()) {
    if (size.id == id) {
      return &size;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* size = find_size(id);
  if (size == nullptr) {
    std::cerr << "unknown size id '" << id << "'\n";
    std::abort();
  }
  return *size;
}

template <typename Element>
std::shared_ptr<holoscan::Tensor> upload(const std::vector<Element>& host,
                                         const std::vector<int64_t>& shape, DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto writable = tensor->mutable_data();
    if (!writable || cudaMemcpy(writable.value(), host.data(), host.size() * sizeof(Element),
                                cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("perf_rank_one_residual_screen: input upload failed");
    }
  }
  return tensor;
}

hr::ArrayGeometry make_geometry(int64_t channels, double spacing) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = hr::ArrayGeometryType::kUla;
  geometry.element_positions.reserve(static_cast<size_t>(channels * 3));
  for (int64_t channel = 0; channel < channels; ++channel) {
    geometry.element_positions.insert(geometry.element_positions.end(),
                                      {0.0, spacing * static_cast<double>(channel), 0.0});
  }
  return geometry;
}

std::vector<cuda::std::complex<PerfPrecision>> steering(int64_t channels, double spacing,
                                                        double azimuth) {
  std::vector<cuda::std::complex<PerfPrecision>> result(static_cast<size_t>(channels));
  for (int64_t channel = 0; channel < channels; ++channel) {
    const double phase = 2.0 * M_PI * spacing * static_cast<double>(channel) * std::sin(azimuth);
    result[static_cast<size_t>(channel)] = {static_cast<PerfPrecision>(std::cos(phase)),
                                            static_cast<PerfPrecision>(std::sin(phase))};
  }
  return result;
}

std::shared_ptr<holoscan::Tensor> make_snapshots(const ScreenScenario& scenario) {
  const auto primary = steering(scenario.channels, 0.5, 0.0);
  const auto secondary = steering(scenario.channels, 0.5, 0.25);
  const int64_t selected = static_cast<int64_t>(
      std::llround(scenario.selection_fraction * static_cast<double>(scenario.items)));
  std::vector<cuda::std::complex<PerfPrecision>> host(
      static_cast<size_t>(scenario.items * scenario.channels * scenario.looks));
  for (int64_t item = 0; item < scenario.items; ++item) {
    for (int64_t channel = 0; channel < scenario.channels; ++channel) {
      const auto value = item < selected ? primary[static_cast<size_t>(channel)] +
                                               secondary[static_cast<size_t>(channel)]
                                         : primary[static_cast<size_t>(channel)];
      for (int64_t look = 0; look < scenario.looks; ++look) {
        host[static_cast<size_t>((item * scenario.channels + channel) * scenario.looks + look)] =
            value;
      }
    }
  }
  return upload(host, {scenario.items, scenario.channels, scenario.looks},
                hr::complex_dtype<PerfPrecision>());
}

std::shared_ptr<holoscan::Tensor> make_baseline(const ScreenScenario& scenario) {
  return upload(
      std::vector<PerfPrecision>(static_cast<size_t>(scenario.items * 2), PerfPrecision{0}),
      {scenario.items, 2}, hr::real_dtype<PerfPrecision>());
}

perf::TimingCollector run_screen_graph(const ScreenScenario& scenario, const perf::RunPlan& plan) {
  auto snapshot = make_snapshots(scenario);
  auto baseline = make_baseline(scenario);
  const auto geometry = make_geometry(scenario.channels, 0.5);
  const auto changed_geometry = make_geometry(scenario.channels, 0.49);
  return perf::run_perf_graph_dual_with_override(
      std::move(snapshot), std::move(baseline), plan.total_frames, plan.override_at,
      hr::to_wire(geometry), hr::to_wire(changed_geometry), "rank-one-residual-screen-perf",
      [&](holoscan::Graph& graph, const auto& source, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_items = scenario.items;
        extents.max_channels = scenario.channels;
        extents.max_snapshots = scenario.looks;
        auto screen = graph.op<hr::RankOneResidualScreenOp<PerfPrecision>>("screen", extents);
        hr::RankOneResidualScreenParams params;
        params.array_geometry = geometry;
        params.residual_threshold = 0.03;
        screen.set_params(std::move(params));

        const auto timed = graph.op<perf::PerfTerminalSinkOp>("snapshot_sink");
        const auto idx_drain = graph.op<perf::PerfNullSinkOp>("idx_drain");
        const auto valid_drain = graph.op<perf::PerfNullSinkOp>("valid_drain");
        const auto score_drain = graph.op<perf::PerfNullSinkOp>("score_drain");
        const auto mask_drain = graph.op<perf::PerfNullSinkOp>("mask_drain");
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};

        graph.add_flow(source->out_a, screen->snapshot, queue);
        graph.add_flow(source->out_b, screen->baseline_aoa, queue);
        graph.add_flow(source->override_out, screen->array_geometry, queue);
        graph.add_flow(screen->snapshot_out, timed->in, queue);
        graph.add_flow(screen->selected_idx, idx_drain->in, queue);
        graph.add_flow(screen->is_valid_out, valid_drain->in, queue);
        graph.add_flow(screen->residual_score, score_drain->in, queue);
        graph.add_flow(screen->selection_mask, mask_drain->in, queue);

        graph.partition("perf")
            .add(source)
            .add(screen)
            .add(timed)
            .add(idx_drain)
            .add(valid_drain)
            .add(score_drain)
            .add(mask_drain);
        for (const char* port :
             {"snapshot_out", "selected_idx", "is_valid_out", "residual_score", "selection_mask"}) {
          perf::bind_tensor_out(options, "screen", port);
        }
      });
}

ScreenScenario make_scenario(const SizeSpec& size, double fraction) {
  ScreenScenario scenario;
  scenario.size_id = size.id;
  scenario.items = size.items;
  scenario.channels = size.channels;
  scenario.looks = size.looks;
  scenario.selection_fraction = fraction;
  std::ostringstream id;
  id << "rank1_" << size.id << "_n" << size.items << "_c" << size.channels << "_l" << size.looks
     << "_sel" << static_cast<int>(fraction * 100.0);
  scenario.id = id.str();
  return scenario;
}

perf::ResultRow base_row(const ScreenScenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "RankOneResidualScreenOp";
  row.variant = "normalized_projection";
  row.size_id = scenario.size_id;
  row.input_shape = {scenario.items, scenario.channels, scenario.looks};
  row.out_batch = static_cast<int64_t>(std::llround(scenario.selection_fraction * scenario.items));
  row.k = scenario.channels;
  row.notes = "all outputs drained; Exact live geometry override";
  return row;
}

void fill_throughput(perf::ResultRow* row, const ScreenScenario& scenario, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double seconds = median_ms / 1000.0;
  row->cells_per_second =
      static_cast<double>(scenario.items * scenario.channels * scenario.looks) / seconds;
  row->slices_per_second = static_cast<double>(scenario.items) / seconds;
}

struct ScreenTraits : perf::PerfDriverDefaults {
  using Scenario = ScreenScenario;

  static const char* tool_name() { return "perf_rank_one_residual_screen"; }
  static const char* tool_description() {
    return "RankOneResidualScreenOp HSDK 5 Graph characterization";
  }
  static const char* sizes_all_help() { return "Item/channel/look sweep at 25 percent selection"; }
  static const char* sizes_subset_help() { return "Selected size subset at 25 percent selection"; }
  static const char* full_help() {
    return "Sizes x selection fraction; every scenario measures geometry override";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    for (const auto& size : size_ladder()) {
      result.push_back(perf::SizeDef{.id = size.id,
                                     .shape = {size.items, size.channels, size.looks},
                                     .cells = size.items * size.channels * size.looks,
                                     .role = size.role});
    }
    return result;
  }

  static std::vector<Scenario> build_default_matrix() {
    const auto& medium = must_find_size("md");
    return {make_scenario(must_find_size("xs"), 0.25), make_scenario(medium, 0.0),
            make_scenario(medium, 0.25), make_scenario(medium, 0.75),
            make_scenario(must_find_size("lg"), 0.25)};
  }

  static std::vector<Scenario> build_size_sweep(const std::vector<std::string>& sizes) {
    std::vector<Scenario> result;
    for (const auto& id : sizes) {
      if (const auto* size = find_size(id)) {
        result.push_back(make_scenario(*size, 0.25));
      }
    }
    return result;
  }

  static std::vector<Scenario> build_full_matrix() {
    std::vector<Scenario> result;
    for (const auto& size : size_ladder()) {
      for (double fraction : {0.0, 0.10, 0.50, 1.0}) {
        result.push_back(make_scenario(size, fraction));
      }
    }
    return result;
  }

  static void run_scenario(const Scenario& scenario, const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(settings, /*has_override=*/true);
    const auto samples =
        perf::collect_scenario_samples(settings, plan, [&](int, const perf::RunPlan& run_plan) {
          return run_screen_graph(scenario, run_plan);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    auto row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scenario, row.latency.median_ms);
    row.notes =
        "cold_ms is first frame; steady metrics precede the Exact geometry "
        "override; override_ms measures a new geometry cache key; all outputs "
        "are drained; cells/s is complex snapshot values/s";
    report->results.push_back(std::move(row));
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs"), 0.25), settings, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<ScreenTraits>(argc, argv);
}
