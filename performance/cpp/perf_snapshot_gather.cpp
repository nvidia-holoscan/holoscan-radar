// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for SnapshotGatherOp — the sparse-path bridge
// that gathers a per-detection channel-vector neighborhood out of per-channel
// RD maps. Sweeps channel count, RD-map size, detection count, and snapshot
// window. Measurement tool only: no timing assertions.
//
//  Graph-based rewrite: rd_maps, detection_idx, and any
// snapshot_window override are co-emitted with the same capture time and
// consumed through the production OnAligned + Exact contract.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

struct SizeSpec {
  std::string id;
  int64_t channels;
  int64_t doppler;
  int64_t range;
  int64_t detections;
  std::string role;
};

struct Window {
  int64_t wd{0};
  int64_t wr{0};
};

struct SnapScenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t doppler{0};
  int64_t range{0};
  int64_t detections{0};
  Window window;
  bool live_window{false};
  Window override_window;
};

int64_t snapshots_for(const Window& w) {
  return (2 * w.wd + 1) * (2 * w.wr + 1);
}

std::string window_tag(const Window& w) {
  std::ostringstream os;
  os << "w" << w.wd << "x" << w.wr;
  return os.str();
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_from_host(const std::vector<T>& host,
                                                        const std::vector<int64_t>& shape,
                                                        DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto mut = tensor->mutable_data();
    if (!mut) {
      throw std::runtime_error("perf_snapshot_gather: tensor not writable");
    }
    if (cudaMemcpy(mut.value(), host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      throw std::runtime_error("perf_snapshot_gather: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_rd_maps(int64_t channels, int64_t doppler, int64_t range) {
  return perf::make_iq_cube_device<T>({channels, doppler, range});
}

std::shared_ptr<holoscan::Tensor> make_detection_idx(int64_t detections, int64_t doppler,
                                                     int64_t range) {
  std::vector<int64_t> host(static_cast<size_t>(std::max<int64_t>(detections, 0) * 2), 0);
  const int64_t cells = std::max<int64_t>(doppler * range, 1);
  for (int64_t i = 0; i < detections; ++i) {
    const int64_t cell = (i * 997) % cells;
    host[static_cast<size_t>(i * 2)] = cell / range;
    host[static_cast<size_t>(i * 2 + 1)] = cell % range;
  }
  return make_device_from_host(host, {detections, 2}, kInt64Dtype);
}

perf::TimingCollector run_graph(const SnapScenario& scen, const perf::RunPlan& plan) {
  auto rd = make_rd_maps<PerfPrecision>(scen.channels, scen.doppler, scen.range);
  auto idx = make_detection_idx(scen.detections, scen.doppler, scen.range);
  using BaseOp = hr::SnapshotGatherOp<PerfPrecision>;
  const auto configure = [&](holoscan::Graph& graph, const auto& src,
                             holoscan::CompileOptions& options, auto connect) {
    hr::MaxTensorExtents extents;
    extents.max_detections = scen.detections;
    extents.max_channels = scen.channels;
    extents.max_snapshots = std::max(snapshots_for(scen.window),
                                     scen.live_window ? snapshots_for(scen.override_window) : 1);
    auto op = graph.op<BaseOp>("gather", extents);
    hr::SnapshotGatherParams params;
    params.snapshot_window = {scen.window.wd, scen.window.wr};
    params.doppler_edge = hr::SnapshotEdgeMode::kWrap;
    params.range_edge = hr::SnapshotEdgeMode::kShift;
    op.set_params(std::move(params));
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(src->out_a, op->rd_maps, q);
    graph.add_flow(src->out_b, op->detection_idx, q);
    connect(graph, src, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "gather");
    outputs.timed(op->snapshot, "snapshot");
    outputs.drain(op->is_valid, "is_valid");
    outputs.finish(src, op);
  };
  if (scen.live_window) {
    const auto selected = perf::live_case(
        "snapshot_window",
        hr::SnapshotWindowOverride{.doppler = scen.window.wd, .range = scen.window.wr},
        hr::SnapshotWindowOverride{.doppler = scen.override_window.wd,
                                   .range = scen.override_window.wr},
        [](auto& graph, const auto& src, const auto& op, const auto& q) {
          graph.add_flow(src->override_out, op->snapshot_window, q);
        });
    return perf::run_selected_live_recipe(std::array{std::move(rd), std::move(idx)}, plan,
                                          "snapshot-perf", selected, configure);
  }
  return perf::run_data_recipe(std::array{std::move(rd), std::move(idx)}, plan, "snapshot-perf",
                               [&](auto& graph, const auto& src, auto& options) {
                                 configure(graph, src, options,
                                           [](auto&, const auto&, const auto&, const auto&) {});
                               });
}

void fill_throughput(perf::ResultRow* row, const SnapScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  const double snaps = static_cast<double>(snapshots_for(scen.window));
  const double gathered = n * static_cast<double>(scen.channels) * snaps;

  row->cells_per_second = n / sec;
  row->slices_per_second = n * snaps / sec;

  const double complex_sz = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double idx_bytes = n * 2.0 * static_cast<double>(sizeof(int64_t));
  const double valid_bytes = n * snaps;
  row->estimated_effective_gbps =
      (2.0 * gathered * complex_sz + valid_bytes + idx_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const SnapScenario& scen) {
  perf::ResultRow row;
  row.op_name = "SnapshotGatherOp";
  row.variant = window_tag(scen.window);
  row.map_shape = {scen.doppler, scen.range};
  row.input_shape = {scen.detections, 2};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.live_port = scen.live_window ? "snapshot_window" : "";
  return row;
}

SnapScenario make_scenario(const SizeSpec& s, Window w) {
  SnapScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.doppler = s.doppler;
  sc.range = s.range;
  sc.detections = s.detections;
  sc.window = w;
  std::ostringstream id;
  id << "snap_gather_" << s.id << "_c" << s.channels << "_n" << s.detections << "_"
     << window_tag(w);
  sc.id = id.str();
  return sc;
}

void decorate_row(perf::ResultRow& row, const SnapScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes = "end-to-end Graph latency; cells/s is detections/s";
  if (scen.live_window)
    row.notes += "; one-frame snapshot_window change";
}

struct SnapTraits : perf::PerfDriverDefaults {
  using Scenario = SnapScenario;

  static const char* tool_name() { return "perf_snapshot_gather"; }
  static const char* tool_description() {
    return "SnapshotGatherOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 4)
      throw std::invalid_argument("SnapshotGather profile rank mismatch");
    return {id, shape, shape[0] * shape[1] * shape[2], id};
  }

  static SnapScenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 4 || shape[3] < 1)
      throw std::invalid_argument("SnapshotGather invalid profile shape");
    perf::require_keys(c.parameters, {"window"}, "parameters");
    const auto window = c.parameters["window"];
    if (!window || !window.IsSequence() || window.size() != 2)
      throw std::invalid_argument("SnapshotGather window must have two extents");
    const Window w{window[0].as<int64_t>(), window[1].as<int64_t>()};
    if (w.wd < 0 || w.wr < 0 || w.wd > 2 || w.wr > 2)
      throw std::invalid_argument("SnapshotGather window out of range");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], c.profile};
    auto scenario = make_scenario(size, w);
    scenario.id = c.id;
    scenario.size_id = c.profile;
    if (c.override_value) {
      if ((*c.override_value)["port"].as<std::string>() != "snapshot_window")
        throw std::invalid_argument("SnapshotGather only snapshot_window is live");
      const auto changed = (*c.override_value)["value"];
      if (!changed.IsSequence() || changed.size() != 2)
        throw std::invalid_argument("SnapshotGather override window needs two extents");
      scenario.override_window = {changed[0].as<int64_t>(), changed[1].as<int64_t>()};
      if (scenario.override_window.wd < 0 || scenario.override_window.wr < 0 ||
          scenario.override_window.wd > 2 || scenario.override_window.wr > 2)
        throw std::invalid_argument("SnapshotGather override window out of range");
      scenario.live_window = true;
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<SnapTraits>(argc, argv);
}
