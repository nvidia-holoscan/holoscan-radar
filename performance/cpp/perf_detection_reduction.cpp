// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for DetectionReductionOp — group-by-label
// reduction over (labels, detection_idx, points). Sweeps detection counts,
// cluster counts, and representative policies. Measurement tool only: no
// timing assertions.
//
// [EA HSDK5 ONLY] Graph-based rewrite via run_perf_graph_triple (labels /
// detection_idx / points co-emitted on the same OnClock tick). Mid-session
// [live] representative / feature_weights / score_method override measurement
// is deferred until docs/planning/HSDK5_PORT.md §1. Params still select the
// policy under test.

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

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
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
  int64_t num_points;
  int64_t num_clusters;
  int64_t num_features;
  std::string role;
};

struct RedScenario {
  std::string id;
  std::string size_id;
  hr::RepresentativePolicy policy{hr::RepresentativePolicy::kPeak};
  hr::ScoreMethod score_method{hr::ScoreMethod::kWeightedSum};
  int64_t num_points{0};
  int64_t num_clusters{0};
  int64_t num_features{2};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 256, 32, 2, "small; fast smoke"},
      {"sm", 1024, 64, 2, "compact detection set"},
      {"md", 4096, 128, 2, "common mid-scale"},
      {"lg", 16384, 256, 2, "dense cluster-then-angle"},
      {"xl", 65536, 512, 2, "upper characterization bound"}};
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

std::string policy_tag(hr::RepresentativePolicy p) {
  return std::string(hr::to_string(p));
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
          "perf_detection_reduction: tensor not writable");
    }
    if (cudaMemcpy(mut.value(), host.data(), sizeof(T) * host.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "perf_detection_reduction: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

std::shared_ptr<holoscan::Tensor> make_labels(int64_t num_points,
                                              int64_t num_clusters) {
  std::vector<int64_t> host(
      static_cast<size_t>(std::max<int64_t>(num_points, 0)));
  const int64_t nclus = std::max<int64_t>(num_clusters, 1);
  for (int64_t i = 0; i < num_points; ++i) {
    host[static_cast<size_t>(i)] =
        (i % 8 == 0) ? -1 : (i % nclus);
  }
  return make_device_from_host(host, {num_points}, kInt64Dtype);
}

std::shared_ptr<holoscan::Tensor> make_detection_idx(
    int64_t num_points) {
  std::vector<int64_t> host(
      static_cast<size_t>(
          std::max<int64_t>(num_points, 0) * 2));
  for (int64_t i = 0; i < num_points; ++i) {
    host[static_cast<size_t>(i * 2)] = i % 1024;
    host[static_cast<size_t>(i * 2 + 1)] = (i * 3) % 2048;
  }
  return make_device_from_host(host, {num_points, 2}, kInt64Dtype);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_points(
    int64_t num_points, int64_t num_features) {
  std::vector<T> host(static_cast<size_t>(
      std::max<int64_t>(num_points, 0) *
      std::max<int64_t>(num_features, 0)));
  for (int64_t i = 0; i < num_points; ++i) {
    for (int64_t f = 0; f < num_features; ++f) {
      host[static_cast<size_t>(i * num_features + f)] =
          static_cast<T>((i * 17 + f * 3) % 100) *
          static_cast<T>(0.01);
    }
  }
  return make_device_from_host(
      host, {num_points, num_features}, hr::real_dtype<T>());
}

perf::TimingCollector run_reduction_graph(const RedScenario& scen,
                                          int total_frames) {
  auto labels_t = make_labels(scen.num_points, scen.num_clusters);
  auto idx_t = make_detection_idx(scen.num_points);
  auto points_t = make_points<PerfPrecision>(scen.num_points,
                                             scen.num_features);
  return perf::run_perf_graph_triple(
      std::move(labels_t), std::move(idx_t), std::move(points_t),
      total_frames, "reduction-perf",
      [&](holoscan::Graph& graph, const auto& src,
          holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_clusters = std::max<int64_t>(
            scen.num_clusters, 1);
        extents.max_points = std::max<int64_t>(
            scen.num_points, 1);
        auto op =
            graph.op<hr::DetectionReductionOp<PerfPrecision>>(
                "reduction", extents);
        hr::DetectionReductionParams params;
        params.representative = scen.policy;
        params.score_method = scen.score_method;
        std::vector<double> weights(
            static_cast<size_t>(scen.num_features), 1.0);
        params.feature_weights = std::move(weights);
        op.set_params(std::move(params));

        const auto rep_sink =
            graph.op<perf::PerfTerminalSinkOp>("rep_sink");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};

        graph.add_flow(src->out_a, op->labels, q);
        graph.add_flow(src->out_b, op->detection_idx, q);
        graph.add_flow(src->out_c, op->points, q);
        graph.add_flow(op->rep_idx, rep_sink->in, q);

        perf::bind_tensor_out(options, "reduction", "rep_idx");
      });
}

void fill_throughput(perf::ResultRow* row, const RedScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.num_points);
  row->cells_per_second = n / sec;
  row->slices_per_second =
      static_cast<double>(
          std::max<int64_t>(scen.num_clusters, 1)) /
      sec;

  const double tsz = sizeof(PerfPrecision);
  const double labels_bytes =
      n * static_cast<double>(sizeof(int64_t));
  const double idx_bytes =
      n * 2.0 * static_cast<double>(sizeof(int64_t));
  const double points_bytes =
      n * static_cast<double>(scen.num_features) * tsz;
  const double rep_bytes =
      static_cast<double>(scen.num_clusters) * 2.0 *
      static_cast<double>(sizeof(int64_t));
  row->estimated_effective_gbps =
      (labels_bytes + idx_bytes + points_bytes + rep_bytes) / sec /
      1e9;
}

perf::ResultRow base_row(const RedScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "DetectionReductionOp";
  row.variant = policy_tag(scen.policy);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_points, scen.num_features};
  row.input_shape = {scen.num_points, 2};
  row.out_batch = scen.num_clusters;
  row.k = scen.num_features;
  row.notes = "live_override=deferred_ea";
  return row;
}

RedScenario make_scenario(hr::RepresentativePolicy policy,
                          const SizeSpec& s) {
  RedScenario sc;
  sc.policy = policy;
  sc.size_id = s.id;
  sc.num_points = s.num_points;
  sc.num_clusters = s.num_clusters;
  sc.num_features = s.num_features;
  std::ostringstream id;
  id << "red_" << policy_tag(sc.policy) << "_" << s.id << "_n"
     << s.num_points << "_compute";
  sc.id = id.str();
  return sc;
}

struct RedTraits : perf::PerfDriverDefaults {
  using Scenario = RedScenario;

  static const char* tool_name() {
    return "perf_detection_reduction";
  }
  static const char* tool_description() {
    return "DetectionReductionOp performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Peak-policy size sweep";
  }
  static const char* sizes_subset_help() {
    return "Peak-policy size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (policies x sizes); live-override timing "
           "deferred on HSDK 5 EA (HSDK5_PORT.md §1)";
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

  static std::vector<RedScenario> build_default_matrix() {
    std::vector<RedScenario> out;
    const SizeSpec& md = must_find_size("md");
    for (auto p : {hr::RepresentativePolicy::kFirst,
                   hr::RepresentativePolicy::kPeak,
                   hr::RepresentativePolicy::kMedoid}) {
      out.push_back(make_scenario(p, md));
    }
    for (const char* sid : {"xs", "sm", "md", "lg"}) {
      out.push_back(make_scenario(
          hr::RepresentativePolicy::kPeak, must_find_size(sid)));
    }
    return out;
  }

  static std::vector<RedScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<RedScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid
                  << "', skipping\n";
        continue;
      }
      out.push_back(
          make_scenario(hr::RepresentativePolicy::kPeak, *s));
    }
    return out;
  }

  static std::vector<RedScenario> build_full_matrix() {
    std::vector<RedScenario> out;
    for (const auto& s : size_ladder()) {
      for (auto p : {hr::RepresentativePolicy::kFirst,
                     hr::RepresentativePolicy::kPeak,
                     hr::RepresentativePolicy::kMedoid}) {
        out.push_back(make_scenario(p, s));
      }
    }
    return out;
  }

  static void run_scenario(const RedScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_reduction_graph(scen, p.total_frames);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are steady "
        "state; live override timing deferred (HSDK5_PORT.md §1); "
        "cells/s is points/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([&] {
      perf::PerfReport discarded;
      perf::RunSettings warm_rs;
      warm_rs.warmup_frames = 0;
      warm_rs.measured_frames = 2;
      warm_rs.repeats = 1;
      const SizeSpec& xs = must_find_size("xs");
      for (auto p : {hr::RepresentativePolicy::kFirst,
                     hr::RepresentativePolicy::kPeak,
                     hr::RepresentativePolicy::kMedoid}) {
        run_scenario(make_scenario(p, xs), warm_rs, &discarded);
      }
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<RedTraits>(argc, argv);
}
