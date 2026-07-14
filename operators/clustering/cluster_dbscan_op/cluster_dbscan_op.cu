// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "clustering/cluster_dbscan_op/cluster_dbscan_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

namespace cd = cluster_dbscan;

constexpr const char* kOpName = "ClusterDBSCANOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};
constexpr DLDataType kUint8Dtype{kDLUInt, 8, 1};
constexpr DLDataType kUint32Dtype{kDLUInt, 32, 1};

}  // namespace

template <typename T>
void ClusterDBSCANOp<T>::initialize() {
  // epsilon (std::vector<double>), min_points / max_clusters (int64_t) are all
  // SDK built-in parameter types, so the default radar arg setters cover them.
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void ClusterDBSCANOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("points");

  spec.output<holoscan::TensorMap>("labels");
  spec.output<holoscan::TensorMap>("sizes");

  // epsilon has no meaningful default (required); an empty vector is rejected
  // at compute time with an operator-specific message.
  spec.param(epsilon_, "epsilon", "Neighborhood radius",
             "Scalar (length 1, isotropic) or per-feature (length "
             "num_features) DBSCAN neighborhood radius; all values > 0",
             std::vector<double>{});
  spec.param(min_points_, "min_points", "Min core neighbors",
             "Minimum neighbors (incl. self) for a core point; <= 0 selects "
             "the 2*num_features default",
             int64_t{0});
  spec.param(max_clusters_, "max_clusters", "Max reported clusters",
             "Upper cap on reported clusters (surplus dropped to noise with a "
             "warning); <= 0 is uncapped",
             int64_t{0});

  add_override_port<std::vector<double>>(spec, "epsilon");
  add_override_port<int64_t>(spec, "min_points");
  add_override_port<int64_t>(spec, "max_clusters");
}

template <typename T>
void ClusterDBSCANOp<T>::compute(InputContext& op_input,
                                 OutputContext& op_output, ExecutionContext&) {
  if (!seeded_) {
    effective_epsilon_ = epsilon_.get();
    effective_min_points_ = min_points_.get();
    effective_max_clusters_ = max_clusters_.get();
    seeded_ = true;
  }
  effective_epsilon_ =
      resolve_override_param(op_input, "epsilon", effective_epsilon_, nullptr);
  effective_min_points_ = resolve_override_param(
      op_input, "min_points", effective_min_points_, nullptr);
  effective_max_clusters_ = resolve_override_param(
      op_input, "max_clusters", effective_max_clusters_, nullptr);

  auto points = receive_single(op_input, "points", kOpName);
  require_cuda_accessible(*points, "points", kOpName);
  require_contiguous(*points, "points", kOpName);
  require_dtype(*points, real_dtype<T>(), "points", kOpName);

  const auto& shape = points->shape();
  if (shape.size() != 2) {
    throw std::invalid_argument(
        "ClusterDBSCANOp: points must have shape (num_points, num_features)");
  }
  const int64_t num_points = shape[0];
  const int64_t num_features = shape[1];
  if (num_features < 1) {
    throw std::invalid_argument(
        "ClusterDBSCANOp: num_features must be >= 1");
  }

  // Validate epsilon: scalar (length 1) or per-feature (length num_features),
  // all finite and strictly positive.
  const auto& eps = effective_epsilon_;
  if (eps.empty()) {
    throw std::invalid_argument("ClusterDBSCANOp: epsilon is required");
  }
  if (static_cast<int64_t>(eps.size()) != 1 &&
      static_cast<int64_t>(eps.size()) != num_features) {
    throw std::invalid_argument(
        "ClusterDBSCANOp: epsilon must be a scalar (length 1) or match "
        "num_features");
  }
  for (double e : eps) {
    if (!std::isfinite(e) || e <= 0.0) {
      throw std::invalid_argument(
          "ClusterDBSCANOp: epsilon values must be finite and > 0");
    }
  }

  // Resolve min_points: <= 0 → the 2*num_features heuristic.
  const int64_t min_points =
      effective_min_points_ <= 0 ? 2 * num_features : effective_min_points_;
  const int64_t max_clusters = effective_max_clusters_;  // <= 0 → uncapped

  auto stream = op_input.receive_cuda_stream("points");

  // Empty input → empty outputs (a valid pipeline state).
  if (num_points == 0) {
    auto labels = make_device_tensor({0}, kInt64Dtype);
    auto sizes = make_device_tensor({0}, kInt64Dtype);
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "labels");
      op_output.set_cuda_stream(stream, "sizes");
    }
    emit_single(op_output, "labels", labels);
    emit_single(op_output, "sizes", sizes);
    retained_points_ = std::move(points);
    return;
  }

  if (num_points > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "ClusterDBSCANOp: num_points exceeds the supported range");
  }

  // Per-axis inverse squared radius (isotropic scalar broadcast to all axes).
  std::vector<T> eps_inv2(static_cast<size_t>(num_features));
  for (int64_t a = 0; a < num_features; ++a) {
    const double e = eps.size() == 1 ? eps[0] : eps[static_cast<size_t>(a)];
    eps_inv2[static_cast<size_t>(a)] = static_cast<T>(1.0 / (e * e));
  }
  auto eps_inv2_dev = make_device_tensor({num_features}, real_dtype<T>());
  if (cudaMemcpyAsync(eps_inv2_dev->data(), eps_inv2.data(),
                      sizeof(T) * eps_inv2.size(), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error("ClusterDBSCANOp: epsilon H2D copy failed");
  }

  const T* points_ptr = static_cast<const T*>(points->data());
  const T* eps_ptr = static_cast<const T*>(eps_inv2_dev->data());

  // Device scratch.
  auto is_core = make_device_tensor({num_points}, kUint8Dtype);
  auto cc_a = make_device_tensor({num_points}, kInt64Dtype);
  auto cc_b = make_device_tensor({num_points}, kInt64Dtype);
  auto changed_dev = make_device_tensor({1}, kInt32Dtype);

  cd::launch_mark_core<T>(points_ptr, eps_ptr, num_points, num_features,
                          min_points, static_cast<uint8_t*>(is_core->data()),
                          static_cast<int64_t*>(cc_a->data()), stream);
  check_cuda("mark_core", kOpName);

  // Connected components: min-label propagation (ping-pong) to convergence.
  // One-hop propagation bounds the iteration count by the graph diameter
  // (< num_points); the +64 slack is a defensive infinite-loop guard.
  int64_t* cur = static_cast<int64_t*>(cc_a->data());
  int64_t* nxt = static_cast<int64_t*>(cc_b->data());
  const int64_t max_iters = num_points + 64;
  int64_t iter = 0;
  for (;; ++iter) {
    if (iter > max_iters) {
      throw std::runtime_error(
          "ClusterDBSCANOp: connected-components did not converge");
    }
    if (cudaMemsetAsync(changed_dev->data(), 0, sizeof(int), stream) !=
        cudaSuccess) {
      throw std::runtime_error("ClusterDBSCANOp: changed-flag clear failed");
    }
    cd::launch_cc_step<T>(
        points_ptr, eps_ptr, static_cast<const uint8_t*>(is_core->data()), cur,
        nxt, num_points, num_features,
        static_cast<int*>(changed_dev->data()), stream);
    check_cuda("cc_step", kOpName);
    std::swap(cur, nxt);

    int host_changed = 0;
    if (cudaMemcpyAsync(&host_changed, changed_dev->data(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      throw std::runtime_error("ClusterDBSCANOp: changed-flag D2H failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("ClusterDBSCANOp: stream sync failed");
    }
    if (host_changed == 0) {
      break;
    }
  }

  // Border points join their lowest-root core neighbor (in place on `cur`).
  cd::launch_assign_border<T>(points_ptr, eps_ptr,
                              static_cast<const uint8_t*>(is_core->data()), cur,
                              num_points, num_features, stream);
  check_cuda("assign_border", kOpName);

  // Compact component roots into dense cluster ids (ascending-root order).
  auto root_flag = make_device_tensor({num_points}, kUint32Dtype);
  auto dense = make_device_tensor({num_points}, kUint32Dtype);
  cd::launch_root_flag(static_cast<const uint8_t*>(is_core->data()), cur,
                       static_cast<uint32_t*>(root_flag->data()), num_points,
                       stream);
  check_cuda("root_flag", kOpName);

  const size_t scan_bytes = cd::cub_scan_temp_bytes(num_points);
  const int64_t scan_alloc =
      static_cast<int64_t>(std::max<size_t>(scan_bytes, 1));
  auto scan_temp = make_device_tensor({scan_alloc}, kUint8Dtype);
  cd::launch_scan_root_ids(scan_temp->data(), scan_bytes,
                           static_cast<const uint32_t*>(root_flag->data()),
                           static_cast<uint32_t*>(dense->data()), num_points,
                           stream);
  check_cuda("scan_root_ids", kOpName);

  // num_clusters = exclusive_scan[N-1] + root_flag[N-1].
  uint32_t dense_last = 0;
  uint32_t flag_last = 0;
  const int64_t last = num_points - 1;
  if (cudaMemcpyAsync(&dense_last,
                      static_cast<const uint32_t*>(dense->data()) + last,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess ||
      cudaMemcpyAsync(&flag_last,
                      static_cast<const uint32_t*>(root_flag->data()) + last,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess) {
    throw std::runtime_error("ClusterDBSCANOp: cluster-count D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("ClusterDBSCANOp: cluster-count sync failed");
  }
  const int64_t num_clusters =
      static_cast<int64_t>(dense_last) + static_cast<int64_t>(flag_last);

  auto labels = make_device_tensor({num_points}, kInt64Dtype);
  cd::launch_finalize_labels(cur, static_cast<const uint32_t*>(dense->data()),
                             static_cast<int64_t*>(labels->data()), num_points,
                             stream);
  check_cuda("finalize_labels", kOpName);

  // Apply the max_clusters cap: drop surplus clusters (highest dense ids) to
  // noise and report only the first `max_clusters`.
  int64_t reported_clusters = num_clusters;
  if (max_clusters > 0 && num_clusters > max_clusters) {
    if (!warned_max_clusters_) {
      HOLOSCAN_LOG_WARN(
          "ClusterDBSCANOp: {} clusters exceed max_clusters={}, dropping {} "
          "surplus cluster(s) to noise",
          num_clusters, max_clusters, num_clusters - max_clusters);
      warned_max_clusters_ = true;
    }
    cd::launch_cap_labels(static_cast<int64_t*>(labels->data()), max_clusters,
                          num_points, stream);
    check_cuda("cap_labels", kOpName);
    reported_clusters = max_clusters;
  }

  auto sizes = make_device_tensor({reported_clusters}, kInt64Dtype);
  if (reported_clusters > 0) {
    if (cudaMemsetAsync(sizes->data(), 0, sizeof(int64_t) * reported_clusters,
                        stream) != cudaSuccess) {
      throw std::runtime_error("ClusterDBSCANOp: sizes clear failed");
    }
    cd::launch_histogram(static_cast<const int64_t*>(labels->data()),
                         static_cast<int64_t*>(sizes->data()), num_points,
                         stream);
    check_cuda("histogram", kOpName);
  }

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "labels");
    op_output.set_cuda_stream(stream, "sizes");
  }
  emit_single(op_output, "labels", labels);
  emit_single(op_output, "sizes", sizes);

  retained_points_ = std::move(points);
}

template class ClusterDBSCANOp<float>;
template class ClusterDBSCANOp<double>;

}  // namespace holoscan::radar
