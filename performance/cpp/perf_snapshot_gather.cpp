// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for SnapshotGatherOp — the sparse-path bridge
// that gathers a per-detection channel-vector neighborhood out of per-channel
// RD maps. Sweeps channel count, RD-map size, detection count, and snapshot
// window. Measurement tool only: no timing assertions.
//
// [EA HSDK5 ONLY] Graph-based rewrite via run_perf_graph_dual (rd_maps +
// detection_idx co-emitted on the same OnClock tick, consumed via OnAligned).
// Mid-session [live] snapshot_window override measurement is deferred until
// docs/planning/HSDK5_PORT.md §1. Params still select the window under test.

#include <cuda_runtime_api.h>

#include <algorithm>
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

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
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
};

int64_t snapshots_for(const Window& w) {
  return (2 * w.wd + 1) * (2 * w.wr + 1);
}

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 4, 64, 256, 64, "small map; fast smoke"},
      {"sm", 8, 128, 512, 256, "compact SIMO array"},
      {"md", 16, 256, 512, 1024, "common dev size"},
      {"lg", 32, 512, 1024, 4096, "dense virtual array"},
      {"xl", 64, 512, 2048, 16384,
       "upper characterization bound"}};
  return ladder;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& s : size_ladder()) {
    if (s.id == id) {
      return &s;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* s = find_size(id);
  if (s == nullptr) {
    std::cerr << "internal error: unknown size id '" << id << "'\n";
    std::abort();
  }
  return *s;
}

std::string window_tag(const Window& w) {
  std::ostringstream os;
  os << "w" << w.wd << "x" << w.wr;
  return os.str();
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_from_host(
    const std::vector<T>& host, const std::vector<int64_t>& shape,
    DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto mut = tensor->mutable_data();
    if (!mut) {
      throw std::runtime_error(
          "perf_snapshot_gather: tensor not writable");
    }
    if (cudaMemcpy(mut.value(), host.data(), sizeof(T) * host.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "perf_snapshot_gather: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_rd_maps(int64_t channels,
                                               int64_t doppler,
                                               int64_t range) {
  return perf::make_iq_cube_device<T>({channels, doppler, range});
}

std::shared_ptr<holoscan::Tensor> make_detection_idx(
    int64_t detections, int64_t doppler, int64_t range) {
  std::vector<int64_t> host(
      static_cast<size_t>(
          std::max<int64_t>(detections, 0) * 2),
      0);
  const int64_t cells = std::max<int64_t>(doppler * range, 1);
  for (int64_t i = 0; i < detections; ++i) {
    const int64_t cell = (i * 997) % cells;
    host[static_cast<size_t>(i * 2)] = cell / range;
    host[static_cast<size_t>(i * 2 + 1)] = cell % range;
  }
  return make_device_from_host(host, {detections, 2}, kInt64Dtype);
}

perf::TimingCollector run_snapshot_graph(const SnapScenario& scen,
                                         int total_frames) {
  auto rd = make_rd_maps<PerfPrecision>(scen.channels, scen.doppler,
                                        scen.range);
  auto idx = make_detection_idx(scen.detections, scen.doppler,
                                scen.range);
  return perf::run_perf_graph_dual(
      std::move(rd), std::move(idx), total_frames, "snapshot-perf",
      [&](holoscan::Graph& graph, const auto& src,
          holoscan::CompileOptions& options) {
        auto op = graph.op<hr::SnapshotGatherOp<PerfPrecision>>(
            "gather");
        hr::SnapshotGatherParams params;
        params.snapshot_window = {scen.window.wd, scen.window.wr};
        params.doppler_edge = hr::SnapshotEdgeMode::kWrap;
        params.range_edge = hr::SnapshotEdgeMode::kShift;
        op.set_params(std::move(params));

        const auto snap_sink =
            graph.op<perf::PerfTerminalSinkOp>("snap_sink");
        const auto valid_drain =
            graph.op<perf::PerfNullSinkOp>("valid_drain");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};

        graph.add_flow(src->out_a, op->rd_maps, q);
        graph.add_flow(src->out_b, op->detection_idx, q);
        graph.add_flow(op->snapshot, snap_sink->in, q);
        graph.add_flow(op->is_valid, valid_drain->in, q);

        perf::bind_tensor_out(options, "gather", "snapshot");
        perf::bind_tensor_out(options, "gather", "is_valid");
      });
}

void fill_throughput(perf::ResultRow* row, const SnapScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  const double snaps =
      static_cast<double>(snapshots_for(scen.window));
  const double gathered =
      n * static_cast<double>(scen.channels) * snaps;

  row->cells_per_second = n / sec;
  row->slices_per_second = n * snaps / sec;

  const double complex_sz =
      2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double idx_bytes =
      n * 2.0 * static_cast<double>(sizeof(int64_t));
  const double valid_bytes = n * snaps;
  row->estimated_effective_gbps =
      (2.0 * gathered * complex_sz + valid_bytes + idx_bytes) / sec /
      1e9;
}

perf::ResultRow base_row(const SnapScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "SnapshotGatherOp";
  row.variant = window_tag(scen.window);
  row.size_id = scen.size_id;
  row.map_shape = {scen.doppler, scen.range};
  row.input_shape = {scen.detections, 2};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.notes = "live_override=deferred_ea";
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
  id << "snap_gather_" << s.id << "_c" << s.channels << "_n"
     << s.detections << "_" << window_tag(w);
  sc.id = id.str();
  return sc;
}

struct SnapTraits : perf::PerfDriverDefaults {
  using Scenario = SnapScenario;

  static const char* tool_name() { return "perf_snapshot_gather"; }
  static const char* tool_description() {
    return "SnapshotGatherOp performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Window w1x1 size sweep; live snapshot_window override "
           "deferred (HSDK5_PORT.md §1)";
  }
  static const char* sizes_subset_help() {
    return "Window w1x1 size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (sizes x windows {0x0, 1x1, 2x2}); "
           "live-override timing deferred on HSDK 5 EA";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.doppler, s.range};
      def.cells = s.doppler * s.range;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<SnapScenario> build_default_matrix() {
    std::vector<SnapScenario> out;
    const SizeSpec& md = must_find_size("md");
    for (const Window w :
         {Window{0, 0}, Window{1, 1}, Window{2, 2}}) {
      out.push_back(make_scenario(md, w));
    }
    for (const char* sid : {"xs", "sm", "md", "lg"}) {
      out.push_back(
          make_scenario(must_find_size(sid), Window{1, 1}));
    }
    return out;
  }

  static std::vector<SnapScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<SnapScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid
                  << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(*s, Window{1, 1}));
    }
    return out;
  }

  static std::vector<SnapScenario> build_full_matrix() {
    std::vector<SnapScenario> out;
    for (const auto& s : size_ladder()) {
      for (const Window w :
           {Window{0, 0}, Window{1, 1}, Window{2, 2}}) {
        out.push_back(make_scenario(s, w));
      }
    }
    return out;
  }

  static void run_scenario(const SnapScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_snapshot_graph(scen, p.total_frames);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are steady "
        "state; live snapshot_window override timing deferred "
        "(HSDK5_PORT.md §1); cells/s is detections/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([&] {
      perf::PerfReport discarded;
      perf::RunSettings warm_rs;
      warm_rs.warmup_frames = 0;
      warm_rs.measured_frames = 2;
      warm_rs.repeats = 1;
      run_scenario(
          make_scenario(must_find_size("xs"), Window{1, 1}),
          warm_rs, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<SnapTraits>(argc, argv);
}
