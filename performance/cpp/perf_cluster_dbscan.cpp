// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ClusterDBSCANOp — brute-force O(N²·F)
// DBSCAN over a (num_points, num_features) feature matrix. The workload is
// compact, well-separated blobs so the connected-components loop converges
// in a few iterations; N and F both drive cost (FeatureAssembly layouts are
// typically F∈{2,3,5}). Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.
//
// Measures a mid-session min_points override through the production Exact
// contract. The epsilon and max_clusters live ports are not timed.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

struct SizeSpec {
  std::string id;
  int64_t num_points;
  int64_t num_features;
  int64_t num_clusters;
  std::string role;
};

struct DbscanScenario {
  std::string id;
  std::string size_id;
  int64_t num_points{0};
  int64_t num_features{2};
  int64_t num_clusters{1};
  std::vector<double> epsilon{0.5};
  int64_t min_points{0};
  int64_t max_clusters{0};
};

// O(N² · F) sweeps. N spans compact→dense detection clouds; F spans the
// FeatureAssembly layouts that actually feed this op (2 = range+vel,
// 3 = +score pre-angle, 5 = +az+el post-angle). Pairwise distance cost
// grows with F, so the ladder is intentionally (N, F)-diverse, not N-only.
const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 256, 2, 8, "small; fast smoke (R+V)"},
      {"sm", 1024, 2, 32, "compact 2-feature (R+V)"},
      {"sm3", 1024, 3, 32, "compact 3-feature (R+V+score)"},
      {"md", 2048, 2, 64, "mid-scale 2-feature"},
      {"md3", 2048, 3, 64, "mid-scale 3-feature (pre-angle)"},
      {"md5", 2048, 5, 64, "mid-scale 5-feature (post-angle)"},
      {"lg", 4096, 2, 128, "dense 2-feature cloud"},
      {"lg5", 4096, 5, 128, "dense 5-feature cloud"},
      {"xl", 8192, 2, 256, "upper N bound (O(N²), 2-feature)"},
      {"xl5", 8192, 5, 256, "upper N×F bound (O(N²F))"}};
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

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_from_host(const std::vector<T>& host,
                                                        const std::vector<int64_t>& shape) {
  auto tensor = hr::make_device_tensor(shape, hr::real_dtype<T>());
  if (!host.empty()) {
    auto mut = tensor->mutable_data();
    if (!mut) {
      throw std::runtime_error("perf_cluster_dbscan: tensor not writable");
    }
    if (cudaMemcpy(mut.value(), host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      throw std::runtime_error("perf_cluster_dbscan: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

// Deterministic compact blobs: cluster c centered at (c·100, 0, …) with
// a small (±0.1) per-axis jitter so within-blob points are eps-neighbors
// and separate blobs are not. No RNG (keeps runs reproducible).
template <typename T>
std::shared_ptr<holoscan::Tensor> make_points(int64_t num_points, int64_t num_features,
                                              int64_t num_clusters) {
  const int64_t nclus = std::max<int64_t>(num_clusters, 1);
  const double spacing = 100.0;
  std::vector<T> host(
      static_cast<size_t>(std::max<int64_t>(num_points, 0) * std::max<int64_t>(num_features, 0)));
  for (int64_t i = 0; i < num_points; ++i) {
    const int64_t c = i % nclus;
    for (int64_t f = 0; f < num_features; ++f) {
      const double base = (f == 0) ? static_cast<double>(c) * spacing : 0.0;
      const double jitter = static_cast<double>(((i * 131 + f * 17) % 21) - 10) / 100.0;
      host[static_cast<size_t>(i * num_features + f)] = static_cast<T>(base + jitter);
    }
  }
  return make_device_from_host(host, {num_points, num_features});
}

perf::TimingCollector run_dbscan_graph(const DbscanScenario& scen, const perf::RunPlan& plan) {
  auto pts = make_points<PerfPrecision>(scen.num_points, scen.num_features, scen.num_clusters);
  using BaseOp = hr::ClusterDBSCANOp<PerfPrecision>;
  const int64_t override_min_points = scen.min_points + 1;
  return perf::run_perf_graph_with_override(
      std::move(pts), plan.total_frames, plan.override_at, scen.min_points, override_min_points,
      "dbscan-perf",
      [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_points = std::max<int64_t>(scen.num_points, 1);
        extents.max_clusters = std::max<int64_t>(scen.num_clusters, 1);
        auto op = graph.op<BaseOp>("dbscan", extents);
        hr::ClusterDBSCANParams params;
        params.epsilon = scen.epsilon;
        params.min_points = scen.min_points;
        params.max_clusters = scen.max_clusters;
        op.set_params(std::move(params));
        const auto sink = graph.op<perf::PerfTerminalSinkOp>("labels_sink");
        const auto null = graph.op<perf::PerfNullSinkOp>("sizes_drain");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(src->out, op->points, q);
        graph.add_flow(src->override_out, op->min_points, q);
        graph.add_flow(op->labels, sink->in, q);
        graph.add_flow(op->sizes, null->in, q);
        graph.partition("perf").add(src).add(op).add(sink).add(null);
        perf::bind_tensor_out(options, "dbscan", "labels");
        perf::bind_tensor_out(options, "dbscan", "sizes");
      });
}

void fill_throughput(perf::ResultRow* row, const DbscanScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.num_points);
  row->cells_per_second = n / sec;
  row->slices_per_second = static_cast<double>(std::max<int64_t>(scen.num_clusters, 1)) / sec;

  const double tsz = sizeof(PerfPrecision);
  const double pts_bytes = n * static_cast<double>(scen.num_features) * tsz;
  const double labels_bytes = n * static_cast<double>(sizeof(int64_t));
  row->estimated_effective_gbps = (pts_bytes + labels_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const DbscanScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "ClusterDBSCANOp";
  row.variant = "dbscan";
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_points, scen.num_features};
  row.input_shape = {scen.num_points, scen.num_features};
  row.out_batch = scen.num_clusters;
  row.k = scen.num_features;
  row.notes = "live_override=min_points; epsilon/max_clusters not timed";
  return row;
}

DbscanScenario make_scenario(const SizeSpec& s) {
  DbscanScenario sc;
  sc.size_id = s.id;
  sc.num_points = s.num_points;
  sc.num_features = s.num_features;
  sc.num_clusters = s.num_clusters;
  sc.epsilon = {0.5};
  sc.min_points = 2 * s.num_features;
  sc.max_clusters = 0;
  std::ostringstream id;
  id << "dbscan_" << s.id << "_n" << s.num_points << "_f" << s.num_features << "_compute";
  sc.id = id.str();
  return sc;
}

struct DbscanTraits : perf::PerfDriverDefaults {
  using Scenario = DbscanScenario;

  static const char* tool_name() { return "perf_cluster_dbscan"; }
  static const char* tool_description() {
    return "ClusterDBSCANOp performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() { return "Size sweep across the full ladder"; }
  static const char* sizes_subset_help() { return "Size sweep across an explicit subset"; }
  static const char* full_help() {
    return "Full size x feature ladder with min_points override timing";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_points, s.num_features};
      def.cells = s.num_points;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<DbscanScenario> build_default_matrix() {
    std::vector<DbscanScenario> out;
    for (const char* sid : {"xs", "sm", "sm3", "md", "md3", "md5", "lg", "lg5"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    return out;
  }

  static std::vector<DbscanScenario> build_size_sweep(const std::vector<std::string>& sizes) {
    std::vector<DbscanScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(*s));
    }
    return out;
  }

  static std::vector<DbscanScenario> build_full_matrix() {
    std::vector<DbscanScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    return out;
  }

  static void run_scenario(const DbscanScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples =
        perf::collect_scenario_samples(rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_dbscan_graph(scen, p);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are "
        "post-warmup steady state; override_ms is the peak after an inline "
        "min_points change; epsilon/max_clusters not timed; cells/s is "
        "points/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs")), warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<DbscanTraits>(argc, argv);
}
