// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/logger/logger.hpp>

#include "clustering/cluster_dbscan_op/cluster_dbscan_stage.hpp"
#include "radar_common/sdk/config_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

namespace cd = cluster_dbscan;

constexpr const char* kOpName = "ClusterDBSCANOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};
constexpr DLDataType kUint8Dtype{kDLUInt, 8, 1};
constexpr DLDataType kUint32Dtype{kDLUInt, 32, 1};
constexpr DLDataType kFloat64Dtype{kDLFloat, 64, 1};

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

expected<void*, Error> writable_ptr(holoscan::Tensor& tensor,
                                    const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": " + what + " is not writable: " +
            mut.error().message});
  }
  return *mut;
}

}  // namespace

template <typename T>
ClusterDBSCANOp<T>::ClusterDBSCANOp(
    const MaxTensorExtents& extents) {
  const char* who = "ClusterDBSCANOp";
  const std::int64_t max_points =
      require_extent(extents.max_points, "max_points", who);
  const std::int64_t max_clusters =
      require_extent(extents.max_clusters, "max_clusters", who);
  max_features_ = extents.max_features.value_or(0);
  extents_ = extents;
  // labels: int64 (max_points,)
  max_labels_byte_span_ = byte_span_elems(
      static_cast<std::size_t>(max_points),
      sizeof(std::int64_t), who);
  // sizes: int64 (max_clusters,)
  max_sizes_byte_span_ = byte_span_elems(
      static_cast<std::size_t>(max_clusters),
      sizeof(std::int64_t), who);
}

template <typename T>
void ClusterDBSCANOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kHostVector{
      .memory_kind = MemoryKind::kHost,
      .dtype = kFloat64Dtype,
      .rank = 1U,
  };
  spec.input(points).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(labels)
      .produces_tensor(pooled_tensor_output(kDevice, 1U, max_labels_byte_span_));
  spec.output(sizes)
      .produces_tensor(pooled_tensor_output(kDevice, 1U, max_sizes_byte_span_));
  // [live] override ports. Leaving them unconnected is legal; compute() seeds
  // from Params and logical sticky state owns converted vector values.
  spec.input(epsilon).queue_depth(2U).expects_tensor(tensor_input(kHostVector));
  spec.input(min_points).queue_depth(2U);
  spec.input(max_clusters).queue_depth(2U);
}

template <typename T>
Contract ClusterDBSCANOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{points});
  add_live_override_to_contract(result, epsilon);
  add_live_override_to_contract(result, min_points);
  add_live_override_to_contract(result, max_clusters);
  return result;
}

template <typename T>
expected<void, Error> ClusterDBSCANOp<T>::compute(ExecutionContext& context) {
  HOLOSCAN_TRY(auto points_in, receive_tensor(points, kOpName));

  const ClusterDBSCANParams& pinned = context.params<ClusterDBSCANParams>();
  HOLOSCAN_TRY(
      auto eps,
      resolve_live_override_via(
          epsilon, pinned.epsilon, sticky_epsilon_, pinned.sticky_epsilon,
          [this](const Tensor& tensor)
              -> expected<std::vector<double>, Error> {
            // The bound only has to exist for a caller that sends overrides,
            // so this is checked on arrival rather than at every activation.
            if (max_features_ <= 0) {
              return make_unexpected(Error{
                  ErrorCode::kInvalidArgument,
                  "ClusterDBSCANOp: MaxTensorExtents.max_features must be "
                  "positive to accept an epsilon override"});
            }
            return read_host_config_vector<double>(
                tensor, kFloat64Dtype, max_features_, "epsilon");
          }));
  HOLOSCAN_TRY(auto effective_min_points,
               resolve_live_override(
                   min_points, pinned.min_points, sticky_min_points_,
                   pinned.sticky_min_points));
  HOLOSCAN_TRY(auto effective_max_clusters,
               resolve_live_override(
                   max_clusters, pinned.max_clusters, sticky_max_clusters_,
                   pinned.sticky_max_clusters));
  try {
    require_cuda_accessible(points_in, "points", kOpName);
    require_contiguous(points_in, "points", kOpName);
    require_dtype(points_in, real_dtype<T>(), "points", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const auto& shape = points_in.shape();
  if (shape.size() != 2) {
    return fail_invalid(
        "ClusterDBSCANOp: points must have shape (num_points, num_features)");
  }
  const int64_t num_points = shape[0];
  const int64_t num_features = shape[1];
  if (num_features < 1) {
    return fail_invalid("ClusterDBSCANOp: num_features must be >= 1");
  }

  // Validate epsilon: scalar (length 1) or per-feature (length num_features),
  // all finite and strictly positive.
  if (eps.empty()) {
    return fail_invalid("ClusterDBSCANOp: epsilon is required");
  }
  if (static_cast<int64_t>(eps.size()) != 1 &&
      static_cast<int64_t>(eps.size()) != num_features) {
    return fail_invalid(
        "ClusterDBSCANOp: epsilon must be a scalar (length 1) or match "
        "num_features");
  }
  for (double e : eps) {
    if (!std::isfinite(e) || e <= 0.0) {
      return fail_invalid(
          "ClusterDBSCANOp: epsilon values must be finite and > 0");
    }
  }

  // Resolve min_points: <= 0 → the 2*num_features heuristic.
  const int64_t min_pts =
      effective_min_points <= 0 ? 2 * num_features : effective_min_points;
  const int64_t max_clust = effective_max_clusters;  // <= 0 → uncapped

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "ClusterDBSCANOp: no CUDA compute stream"});
  }

  // Empty input → empty outputs (a valid pipeline state).
  if (num_points == 0) {
    // Prepare both loans before any publish so a late allocate/commit
    // failure cannot leave a sibling port already downstream.
    const std::vector<int64_t> empty_shape = {0};
    auto noop = [](void*, std::size_t) -> expected<void, Error> {
      return {};
    };
    HOLOSCAN_TRY(auto labels_loan,
                 prepare_device_tensor(labels, context, empty_shape,
                                      kInt64Dtype, noop));
    HOLOSCAN_TRY(auto sizes_loan,
                 prepare_device_tensor(sizes, context, empty_shape,
                                      kInt64Dtype, noop));
    HOLOSCAN_TRY(
        emit_prepared_tensor(labels, context, std::move(labels_loan)));
    HOLOSCAN_TRY(
        emit_prepared_tensor(sizes, context, std::move(sizes_loan)));
    retained_points_ = std::move(points_in);
    return {};
  }

  if (num_points > std::numeric_limits<int>::max()) {
    return fail_invalid(
        "ClusterDBSCANOp: num_points exceeds the supported range");
  }

  // Per-axis inverse squared radius (isotropic scalar broadcast to all axes).
  std::vector<T> eps_inv2(static_cast<size_t>(num_features));
  for (int64_t a = 0; a < num_features; ++a) {
    const double e = eps.size() == 1 ? eps[0] : eps[static_cast<size_t>(a)];
    eps_inv2[static_cast<size_t>(a)] = static_cast<T>(1.0 / (e * e));
  }
  auto eps_inv2_dev = make_device_tensor({num_features}, real_dtype<T>());
  HOLOSCAN_TRY(auto eps_mut, writable_ptr(*eps_inv2_dev, "eps_inv2"));
  if (cudaMemcpyAsync(eps_mut, eps_inv2.data(),
                      sizeof(T) * eps_inv2.size(), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return fail_runtime("ClusterDBSCANOp: epsilon H2D copy failed");
  }

  const T* points_ptr = static_cast<const T*>(points_in.data());
  const T* eps_ptr = static_cast<const T*>(eps_inv2_dev->data());

  // Device scratch.
  auto is_core = make_device_tensor({num_points}, kUint8Dtype);
  auto cc_a = make_device_tensor({num_points}, kInt64Dtype);
  auto cc_b = make_device_tensor({num_points}, kInt64Dtype);
  auto changed_dev = make_device_tensor({1}, kInt32Dtype);
  HOLOSCAN_TRY(auto is_core_ptr, writable_ptr(*is_core, "is_core"));
  HOLOSCAN_TRY(auto cc_a_ptr, writable_ptr(*cc_a, "cc_a"));
  HOLOSCAN_TRY(auto cc_b_ptr, writable_ptr(*cc_b, "cc_b"));
  HOLOSCAN_TRY(auto changed_ptr, writable_ptr(*changed_dev, "changed"));

  try {
    cd::launch_mark_core<T>(points_ptr, eps_ptr, num_points, num_features,
                            min_pts, static_cast<uint8_t*>(is_core_ptr),
                            static_cast<int64_t*>(cc_a_ptr), stream);
    check_cuda("mark_core", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  // Connected components: min-label propagation (ping-pong) to convergence.
  // One-hop propagation bounds the iteration count by the graph diameter
  // (< num_points); the +64 slack is a defensive infinite-loop guard.
  int64_t* cur = static_cast<int64_t*>(cc_a_ptr);
  int64_t* nxt = static_cast<int64_t*>(cc_b_ptr);
  const int64_t max_iters = num_points + 64;
  int64_t iter = 0;
  for (;; ++iter) {
    if (iter > max_iters) {
      return fail_runtime(
          "ClusterDBSCANOp: connected-components did not converge");
    }
    if (cudaMemsetAsync(changed_ptr, 0, sizeof(int), stream) != cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: changed-flag clear failed");
    }
    try {
      cd::launch_cc_step<T>(
          points_ptr, eps_ptr, static_cast<const uint8_t*>(is_core_ptr), cur,
          nxt, num_points, num_features, static_cast<int*>(changed_ptr),
          stream);
      check_cuda("cc_step", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
    std::swap(cur, nxt);

    int host_changed = 0;
    if (cudaMemcpyAsync(&host_changed, changed_ptr, sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: changed-flag D2H failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: stream sync failed");
    }
    if (host_changed == 0) {
      break;
    }
  }

  // Border points join their lowest-root core neighbor (in place on `cur`).
  try {
    cd::launch_assign_border<T>(points_ptr, eps_ptr,
                                static_cast<const uint8_t*>(is_core_ptr), cur,
                                num_points, num_features, stream);
    check_cuda("assign_border", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  // Compact component roots into dense cluster ids (ascending-root order).
  auto root_flag = make_device_tensor({num_points}, kUint32Dtype);
  auto dense = make_device_tensor({num_points}, kUint32Dtype);
  HOLOSCAN_TRY(auto root_flag_ptr, writable_ptr(*root_flag, "root_flag"));
  HOLOSCAN_TRY(auto dense_ptr, writable_ptr(*dense, "dense"));
  try {
    cd::launch_root_flag(static_cast<const uint8_t*>(is_core_ptr), cur,
                         static_cast<uint32_t*>(root_flag_ptr), num_points,
                         stream);
    check_cuda("root_flag", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  const size_t scan_bytes = cd::cub_scan_temp_bytes(num_points);
  const int64_t scan_alloc =
      static_cast<int64_t>(std::max<size_t>(scan_bytes, 1));
  auto scan_temp = make_device_tensor({scan_alloc}, kUint8Dtype);
  HOLOSCAN_TRY(auto scan_temp_ptr, writable_ptr(*scan_temp, "scan_temp"));
  try {
    cd::launch_scan_root_ids(scan_temp_ptr, scan_bytes,
                             static_cast<const uint32_t*>(root_flag_ptr),
                             static_cast<uint32_t*>(dense_ptr), num_points,
                             stream);
    check_cuda("scan_root_ids", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  // num_clusters = exclusive_scan[N-1] + root_flag[N-1].
  uint32_t dense_last = 0;
  uint32_t flag_last = 0;
  const int64_t last = num_points - 1;
  if (cudaMemcpyAsync(&dense_last,
                      static_cast<const uint32_t*>(dense_ptr) + last,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess ||
      cudaMemcpyAsync(&flag_last,
                      static_cast<const uint32_t*>(root_flag_ptr) + last,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess) {
    return fail_runtime("ClusterDBSCANOp: cluster-count D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return fail_runtime("ClusterDBSCANOp: cluster-count sync failed");
  }
  const int64_t num_clusters =
      static_cast<int64_t>(dense_last) + static_cast<int64_t>(flag_last);

  // Loan both outputs: labels shape is known; sizes shape depends on the cap
  // applied below. Allocate labels now and write through remap/cap in place.
  const std::vector<int64_t> labels_shape = {num_points};
  auto labels_loan = labels.allocate_tensor(
      TensorLoanRequest{.shape = labels_shape, .dtype = kInt64Dtype});
  if (!labels_loan) {
    return make_unexpected(std::move(labels_loan).error());
  }
  auto labels_writer = labels_loan->write(context.cuda_stream());
  if (!labels_writer) {
    return make_unexpected(std::move(labels_writer).error());
  }
  int64_t* labels_ptr = static_cast<int64_t*>(labels_writer->data());

  try {
    cd::launch_finalize_labels(cur, static_cast<const uint32_t*>(dense_ptr),
                               labels_ptr, num_points, stream);
    check_cuda("finalize_labels", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  // Renumber clusters by descending point count (tie-break: ascending root id)
  // so cluster 0 is the largest. The host argsorts the per-id counts and
  // remap_labels applies the permutation on the GPU; the count array is tiny
  // (num_clusters entries) and the op already syncs per iteration, so the extra
  // D2H is negligible. This ordering is what makes the max_clusters cap below
  // (drop the highest ids) keep the *largest* clusters. Determinism holds:
  // `labels` stays a pure function of (points, epsilon, min_points).
  if (num_clusters > 1) {
    auto sizes_all = make_device_tensor({num_clusters}, kInt64Dtype);
    HOLOSCAN_TRY(auto sizes_all_ptr, writable_ptr(*sizes_all, "sizes_all"));
    if (cudaMemsetAsync(sizes_all_ptr, 0, sizeof(int64_t) * num_clusters,
                        stream) != cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: size-histogram clear failed");
    }
    try {
      cd::launch_histogram(labels_ptr, static_cast<int64_t*>(sizes_all_ptr),
                           num_points, stream);
      check_cuda("size_histogram", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }

    std::vector<int64_t> host_sizes(static_cast<size_t>(num_clusters));
    if (cudaMemcpyAsync(host_sizes.data(), sizes_all_ptr,
                        sizeof(int64_t) * num_clusters, cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: size D2H failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: size sync failed");
    }

    // order[r] = ascending-root id ranked r-th by (size desc, id asc); the
    // stable sort over the ascending iota breaks size ties by ascending id.
    std::vector<int64_t> order(static_cast<size_t>(num_clusters));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&host_sizes](int64_t a, int64_t b) {
                       return host_sizes[static_cast<size_t>(a)] >
                              host_sizes[static_cast<size_t>(b)];
                     });
    std::vector<int64_t> remap(static_cast<size_t>(num_clusters));
    for (int64_t r = 0; r < num_clusters; ++r) {
      remap[static_cast<size_t>(order[static_cast<size_t>(r)])] = r;
    }
    auto remap_dev = make_device_tensor({num_clusters}, kInt64Dtype);
    HOLOSCAN_TRY(auto remap_ptr, writable_ptr(*remap_dev, "remap"));
    if (cudaMemcpyAsync(remap_ptr, remap.data(),
                        sizeof(int64_t) * num_clusters, cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: remap H2D failed");
    }
    try {
      cd::launch_remap_labels(labels_ptr,
                              static_cast<const int64_t*>(remap_ptr),
                              num_points, stream);
      check_cuda("remap_labels", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  }

  // Apply the max_clusters cap: drop surplus clusters (highest dense ids —
  // now the smallest, after the descending-size renumber above) to noise and
  // report only the first `max_clusters` (the largest).
  int64_t reported_clusters = num_clusters;
  if (max_clust > 0 && num_clusters > max_clust) {
    if (!warned_max_clusters_) {
      HOLOSCAN_LOG_WARN(
          "ClusterDBSCANOp: {} clusters exceed max_clusters={}, dropping {} "
          "surplus cluster(s) to noise",
          num_clusters, max_clust, num_clusters - max_clust);
      warned_max_clusters_ = true;
    }
    try {
      cd::launch_cap_labels(labels_ptr, max_clust, num_points, stream);
      check_cuda("cap_labels", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
    reported_clusters = max_clust;
  }

  const std::vector<int64_t> sizes_shape = {reported_clusters};
  auto sizes_loan = sizes.allocate_tensor(
      TensorLoanRequest{.shape = sizes_shape, .dtype = kInt64Dtype});
  if (!sizes_loan) {
    return make_unexpected(std::move(sizes_loan).error());
  }
  auto sizes_writer = sizes_loan->write(context.cuda_stream());
  if (!sizes_writer) {
    return make_unexpected(std::move(sizes_writer).error());
  }
  int64_t* sizes_ptr = static_cast<int64_t*>(sizes_writer->data());

  if (reported_clusters > 0) {
    if (cudaMemsetAsync(sizes_ptr, 0, sizeof(int64_t) * reported_clusters,
                        stream) != cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: sizes clear failed");
    }
    try {
      cd::launch_histogram(labels_ptr, sizes_ptr, num_points, stream);
      check_cuda("histogram", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  }

  auto labels_committed = std::move(*labels_writer).commit();
  if (!labels_committed) {
    return make_unexpected(std::move(labels_committed).error());
  }
  auto sizes_committed = std::move(*sizes_writer).commit();
  if (!sizes_committed) {
    return make_unexpected(std::move(sizes_committed).error());
  }

  auto labels_emitted = labels.emit(
      std::move(*labels_loan),
      EmitOptions{.capture_time = context.activation_time()});
  if (!labels_emitted) {
    return make_unexpected(std::move(labels_emitted).error());
  }
  auto sizes_emitted = sizes.emit(
      std::move(*sizes_loan),
      EmitOptions{.capture_time = context.activation_time()});
  if (!sizes_emitted) {
    return make_unexpected(std::move(sizes_emitted).error());
  }

  retained_points_ = std::move(points_in);
  return {};
}

template class ClusterDBSCANOp<float>;
template class ClusterDBSCANOp<double>;

}  // namespace holoscan::radar
