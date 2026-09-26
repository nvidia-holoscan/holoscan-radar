// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ClusterDBSCANOp — brute-force O(N²·F)
// DBSCAN over a (num_points, num_features) feature matrix. The workload is
// compact, well-separated blobs so the connected-components loop converges
// in a few iterations; N and F both drive cost (FeatureAssembly layouts are
// typically F∈{2,3,5}). Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.
//
// Curated YAML cases measure each documented live input on one frame.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
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
  std::string live_port;
  double changed_epsilon{0.8};
  int64_t changed_scalar{0};
};

// O(N² · F) sweeps. N spans compact→dense detection clouds; F spans the
// FeatureAssembly layouts that actually feed this op (2 = range+vel,
// 3 = +score pre-angle, 5 = +az+el post-angle). Pairwise distance cost
// grows with F, so the ladder is intentionally (N, F)-diverse, not N-only.

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

perf::TimingCollector run_graph(const DbscanScenario& scen, const perf::RunPlan& plan) {
  auto pts = make_points<PerfPrecision>(scen.num_points, scen.num_features, scen.num_clusters);
  using BaseOp = hr::ClusterDBSCANOp<PerfPrecision>;
  const auto configure = [&](holoscan::Graph& graph, const auto& src,
                             holoscan::CompileOptions& options, auto connect) {
    hr::MaxTensorExtents extents;
    extents.max_points = std::max<int64_t>(scen.num_points, 1);
    extents.max_features = scen.num_features;
    extents.max_clusters = std::max<int64_t>(scen.num_clusters, 1);
    auto op = graph.op<BaseOp>("dbscan", extents);
    hr::ClusterDBSCANParams params;
    params.epsilon = scen.epsilon;
    params.min_points = scen.min_points;
    params.max_clusters = scen.max_clusters;
    op.set_params(std::move(params));
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(src->out, op->points, q);
    connect(graph, src, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "dbscan");
    outputs.timed(op->labels, "labels");
    outputs.drain(op->sizes, "sizes");
    outputs.finish(src, op);
  };
  if (scen.live_port.empty()) {
    return perf::run_data_recipe(std::array{std::move(pts)}, plan, "dbscan-perf",
                                 [&](auto& graph, const auto& src, auto& options) {
                                   configure(graph, src, options,
                                             [](auto&, const auto&, const auto&, const auto&) {});
                                 });
  }
  const auto cases = std::make_tuple(
      perf::live_option("epsilon",
                        [&] {
                          return perf::live_case(
                              "epsilon", scen.epsilon, std::vector<double>{scen.changed_epsilon},
                              [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                graph.add_flow(src->override_out, op->epsilon, q);
                              });
                        }),
      perf::live_option("min_points",
                        [&] {
                          return perf::live_case(
                              "min_points", scen.min_points, scen.changed_scalar,
                              [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                graph.add_flow(src->override_out, op->min_points, q);
                              });
                        }),
      perf::live_option("max_clusters", [&] {
        return perf::live_case("max_clusters", scen.max_clusters, scen.changed_scalar,
                               [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                 graph.add_flow(src->override_out, op->max_clusters, q);
                               });
      }));
  return perf::dispatch_live_case(scen.live_port, cases, [&](const auto& selected) {
    return perf::run_selected_live_recipe(std::array{std::move(pts)}, plan, "dbscan-perf", selected,
                                          configure);
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
  row.op_name = "ClusterDBSCANOp";
  row.variant = "dbscan";
  row.map_shape = {scen.num_points, scen.num_features};
  row.input_shape = {scen.num_points, scen.num_features};
  row.out_batch = scen.num_clusters;
  row.k = scen.num_features;
  row.live_port = scen.live_port;
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

void decorate_row(perf::ResultRow& row, const DbscanScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes = "end-to-end Graph latency; cells/s is points/s";
  if (!scen.live_port.empty())
    row.notes += "; one-frame " + scen.live_port + " change";
}

struct DbscanTraits : perf::PerfDriverDefaults {
  using Scenario = DbscanScenario;

  static const char* tool_name() { return "perf_cluster_dbscan"; }
  static const char* tool_description() {
    return "ClusterDBSCANOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 3)
      throw std::invalid_argument("ClusterDBSCAN profile rank mismatch");
    return {id, shape, shape[0], id};
  }
  static DbscanScenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 3 || shape[0] < 1 || shape[1] < 2 || shape[1] > 5 || shape[2] < 1)
      throw std::invalid_argument("ClusterDBSCAN invalid profile shape");
    perf::require_keys(c.parameters, {"epsilon", "min_points", "max_clusters"}, "parameters");
    if (!c.parameters["epsilon"])
      throw std::invalid_argument("ClusterDBSCAN requires epsilon");
    const double epsilon = c.parameters["epsilon"].as<double>();
    if (epsilon <= 0.0)
      throw std::invalid_argument("ClusterDBSCAN epsilon must be positive");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], c.profile};
    auto scenario = make_scenario(size);
    scenario.id = c.id;
    scenario.size_id = c.profile;
    scenario.epsilon = {epsilon};
    if (c.parameters["min_points"])
      scenario.min_points = c.parameters["min_points"].as<int64_t>();
    if (c.parameters["max_clusters"])
      scenario.max_clusters = c.parameters["max_clusters"].as<int64_t>();
    if (scenario.min_points < 0 || scenario.max_clusters < 0)
      throw std::invalid_argument("ClusterDBSCAN counts must be nonnegative");
    if (c.override_value) {
      scenario.live_port = (*c.override_value)["port"].as<std::string>();
      if (scenario.live_port == "epsilon") {
        scenario.changed_epsilon = (*c.override_value)["value"].as<double>();
        if (scenario.changed_epsilon <= 0.0)
          throw std::invalid_argument("ClusterDBSCAN override epsilon must be positive");
      } else if (scenario.live_port == "min_points" || scenario.live_port == "max_clusters") {
        scenario.changed_scalar = (*c.override_value)["value"].as<int64_t>();
        if (scenario.changed_scalar < 0)
          throw std::invalid_argument("ClusterDBSCAN override count must be nonnegative");
      } else {
        throw std::invalid_argument("unknown ClusterDBSCAN live port");
      }
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<DbscanTraits>(argc, argv);
}
