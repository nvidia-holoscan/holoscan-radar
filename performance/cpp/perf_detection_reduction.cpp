// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for DetectionReductionOp — group-by-label
// reduction over (labels, detection_idx, points). Sweeps detection counts,
// cluster counts, and representative policies. Measurement tool only: no
// timing assertions.
//
//  Graph-based rewrite via run_perf_graph_triple (labels /
// detection_idx / points co-emitted on the same OnClock tick). Mid-session
// [live] representative / feature_weights / score_method overrides share
// the same one-frame Exact source timing path as other adapters.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
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
#include <holoscan/core/tensor_output_loan.hpp>

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_payloads.hpp"
#include "utils/perf_platform.hpp"
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
  std::string live_port;
  std::vector<double> weights;
};

std::string policy_tag(hr::RepresentativePolicy p) {
  return std::string(hr::to_string(p));
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_from_host(const std::vector<T>& host,
                                                        const std::vector<int64_t>& shape,
                                                        DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto mut = tensor->mutable_data();
    if (!mut) {
      throw std::runtime_error("perf_detection_reduction: tensor not writable");
    }
    if (cudaMemcpy(mut.value(), host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      throw std::runtime_error("perf_detection_reduction: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

std::shared_ptr<holoscan::Tensor> make_labels(int64_t num_points, int64_t num_clusters) {
  std::vector<int64_t> host(static_cast<size_t>(std::max<int64_t>(num_points, 0)));
  const int64_t nclus = std::max<int64_t>(num_clusters, 1);
  for (int64_t i = 0; i < num_points; ++i) {
    host[static_cast<size_t>(i)] = (i % 8 == 0) ? -1 : (i % nclus);
  }
  return make_device_from_host(host, {num_points}, kInt64Dtype);
}

std::shared_ptr<holoscan::Tensor> make_detection_idx(int64_t num_points) {
  std::vector<int64_t> host(static_cast<size_t>(std::max<int64_t>(num_points, 0) * 2));
  for (int64_t i = 0; i < num_points; ++i) {
    host[static_cast<size_t>(i * 2)] = i % 1024;
    host[static_cast<size_t>(i * 2 + 1)] = (i * 3) % 2048;
  }
  return make_device_from_host(host, {num_points, 2}, kInt64Dtype);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_points(int64_t num_points, int64_t num_features) {
  std::vector<T> host(
      static_cast<size_t>(std::max<int64_t>(num_points, 0) * std::max<int64_t>(num_features, 0)));
  for (int64_t i = 0; i < num_points; ++i) {
    for (int64_t f = 0; f < num_features; ++f) {
      host[static_cast<size_t>(i * num_features + f)] =
          static_cast<T>((i * 17 + f * 3) % 100) * static_cast<T>(0.01);
    }
  }
  return make_device_from_host(host, {num_points, num_features}, hr::real_dtype<T>());
}

struct LivePorts {
 public:
  LivePorts(std::string port, int64_t features) : port_(std::move(port)), features_(features) {}
  void setup(holoscan::OperatorSpec& spec) {
    perf::setup_device_input(spec, out_a, perf::PerfFrameBank::span_a());
    perf::setup_device_input(spec, out_b, perf::PerfFrameBank::span_b());
    perf::setup_device_input(spec, out_c, perf::PerfFrameBank::span_c());
    perf::setup_scalar_payload(spec, representative);
    perf::setup_scalar_payload(spec, score_method);
    perf::setup_host_payload(spec, feature_weights, 1U, 1U, DLDataType{kDLFloat, 64U, 1U},
                             static_cast<size_t>(features_) * sizeof(double));
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_override(
      holoscan::ExecutionContext& context) {
    const auto time = context.activation_time();
    if (port_ == "representative") {
      return perf::emit_scalar_payload(representative, hr::RepresentativePolicy::kMedoid, time);
    }
    if (port_ == "score_method") {
      return perf::emit_scalar_payload(score_method, hr::ScoreMethod::kWeightedL2, time);
    }
    if (port_ == "feature_weights") {
      return perf::emit_host_vector_generated<double>(
          feature_weights, features_,
          [features = features_](double* data) {
            for (int64_t i = 0; i < features; ++i)
              data[i] = i == 0 ? 2.0 : 1.0;
          },
          time);
    }
    return {};
  }

  holoscan::Output<holoscan::Tensor> out_a{"out_a"};
  holoscan::Output<holoscan::Tensor> out_b{"out_b"};
  holoscan::Output<holoscan::Tensor> out_c{"out_c"};
  holoscan::Output<hr::RepresentativePolicy> representative{"representative"};
  holoscan::Output<hr::ScoreMethod> score_method{"score_method"};
  holoscan::Output<holoscan::Tensor> feature_weights{"feature_weights"};

 private:
  std::string port_;
  int64_t features_{0};
};

constexpr auto kInputA = &LivePorts::out_a;
constexpr auto kInputB = &LivePorts::out_b;
constexpr auto kInputC = &LivePorts::out_c;
using LiveSource = perf::PerfLiveSourceOp<LivePorts, kInputA, kInputB, kInputC>;

perf::TimingCollector run_graph(const RedScenario& scen, const perf::RunPlan& plan) {
  auto labels_t = make_labels(scen.num_points, scen.num_clusters);
  auto idx_t = make_detection_idx(scen.num_points);
  auto points_t = make_points<PerfPrecision>(scen.num_points, scen.num_features);
  const auto build = [&](holoscan::Graph& graph, const auto& src,
                         holoscan::CompileOptions& options) {
    hr::MaxTensorExtents extents;
    extents.max_clusters = std::max<int64_t>(scen.num_clusters, 1);
    extents.max_points = std::max<int64_t>(scen.num_points, 1);
    extents.max_features = scen.num_features;
    hr::DetectionReductionParams params;
    params.representative = scen.policy;
    params.score_method = scen.score_method;
    params.feature_weights = scen.weights;
    auto op = graph.op<hr::DetectionReductionOp<PerfPrecision>>("reduction", extents);
    op.set_params(std::move(params));

    const auto sink = graph.op<perf::PerfDualTerminalSinkOp>("sink");
    const holoscan::ConnectionOptions q{.queue_depth = 4U};

    graph.add_flow(src->out_a, op->labels, q);
    graph.add_flow(src->out_b, op->detection_idx, q);
    graph.add_flow(src->out_c, op->points, q);
    if constexpr (requires { src->representative; }) {
      if (scen.live_port == "representative")
        graph.add_flow(src->representative, op->representative, q);
      if (scen.live_port == "score_method")
        graph.add_flow(src->score_method, op->score_method, q);
      if (scen.live_port == "feature_weights")
        graph.add_flow(src->feature_weights, op->feature_weights, q);
    }
    graph.add_flow(op->rep_idx, sink->in0, q);
    graph.add_flow(op->reduced_points, sink->in1, q);

    perf::bind_tensor_out(options, "reduction", "rep_idx");
    perf::bind_tensor_out(options, "reduction", "reduced_points");
  };
  if (scen.live_port.empty())
    return perf::run_perf_graph_triple(std::move(labels_t), std::move(idx_t), std::move(points_t),
                                       plan.total_frames, "reduction-perf", build);
  return perf::run_perf_graph_triple_with_override<LiveSource>(
      std::move(labels_t), std::move(idx_t), std::move(points_t), plan.total_frames,
      plan.override_at, scen.live_port, scen.num_features, "reduction-perf", build);
}

void fill_throughput(perf::ResultRow* row, const RedScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.num_points);
  row->cells_per_second = n / sec;
  row->slices_per_second = static_cast<double>(std::max<int64_t>(scen.num_clusters, 1)) / sec;

  const double tsz = sizeof(PerfPrecision);
  const double labels_bytes = n * static_cast<double>(sizeof(int64_t));
  const double idx_bytes = n * 2.0 * static_cast<double>(sizeof(int64_t));
  const double points_bytes = n * static_cast<double>(scen.num_features) * tsz;
  const double rep_bytes =
      static_cast<double>(scen.num_clusters) * 2.0 * static_cast<double>(sizeof(int64_t));
  const double reduced_bytes = static_cast<double>(scen.num_clusters * scen.num_features) * tsz;
  row->estimated_effective_gbps =
      (labels_bytes + idx_bytes + points_bytes + rep_bytes + reduced_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const RedScenario& scen) {
  perf::ResultRow row;
  row.op_name = "DetectionReductionOp";
  row.variant = policy_tag(scen.policy);
  row.map_shape = {scen.num_points, scen.num_features};
  row.input_shape = {scen.num_points, 2};
  row.out_batch = scen.num_clusters;
  row.k = scen.num_features;
  row.live_port = scen.live_port;
  return row;
}

RedScenario make_scenario(hr::RepresentativePolicy policy, const SizeSpec& s) {
  RedScenario sc;
  sc.policy = policy;
  sc.size_id = s.id;
  sc.num_points = s.num_points;
  sc.num_clusters = s.num_clusters;
  sc.num_features = s.num_features;
  sc.weights.assign(static_cast<size_t>(s.num_features), 1.0);
  std::ostringstream id;
  id << "red_" << policy_tag(sc.policy) << "_" << s.id << "_n" << s.num_points << "_compute";
  sc.id = id.str();
  return sc;
}

void decorate_row(perf::ResultRow& row, const RedScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes = "cold_ms is first-frame latency; override_ms is one changed frame for " +
              scen.live_port + "; cells/s is points/s";
  if (row.live_port.empty())
    row.notes = "default parameters; no live override";
}

struct RedTraits : perf::PerfDriverDefaults {
  using Scenario = RedScenario;

  static const char* tool_name() { return "perf_detection_reduction"; }
  static const char* tool_description() {
    return "DetectionReductionOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 3)
      throw std::invalid_argument("DetectionReduction profile rank mismatch");
    return {id, shape, shape[0], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 3 || shape[0] < 1 || shape[1] < 1 || shape[1] > shape[0] || shape[2] < 1)
      throw std::invalid_argument("DetectionReduction invalid shape");
    perf::require_keys(c.parameters, {"representative", "score_method"}, "parameters");
    const auto policy = c.parameters["representative"].as<std::string>();
    const auto method = c.parameters["score_method"].as<std::string>();
    if (policy != "peak" && policy != "medoid" && policy != "first" && policy != "weighted_mean")
      throw std::invalid_argument("DetectionReduction invalid policy");
    if (method != "weighted_sum" && method != "weighted_l2" && method != "weighted_max")
      throw std::invalid_argument("DetectionReduction invalid score method");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], c.profile};
    const auto parsed_policy = policy == "first"           ? hr::RepresentativePolicy::kFirst
                               : policy == "medoid"        ? hr::RepresentativePolicy::kMedoid
                               : policy == "weighted_mean" ? hr::RepresentativePolicy::kWeightedMean
                                                           : hr::RepresentativePolicy::kPeak;
    auto scenario = make_scenario(parsed_policy, size);
    scenario.id = c.id;
    scenario.score_method = method == "weighted_l2"    ? hr::ScoreMethod::kWeightedL2
                            : method == "weighted_max" ? hr::ScoreMethod::kWeightedMax
                                                       : hr::ScoreMethod::kWeightedSum;
    if (c.override_value) {
      scenario.live_port = (*c.override_value)["port"].as<std::string>();
      const auto value = (*c.override_value)["value"].as<std::string>();
      if ((scenario.live_port == "representative" && value != "medoid") ||
          (scenario.live_port == "score_method" && value != "weighted_l2") ||
          (scenario.live_port == "feature_weights" && value != "first_double") ||
          (scenario.live_port != "representative" && scenario.live_port != "score_method" &&
           scenario.live_port != "feature_weights"))
        throw std::invalid_argument("DetectionReduction invalid override");
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<RedTraits>(argc, argv);
}
