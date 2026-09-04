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
#include <optional>
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

namespace detail {

struct ClusterDBSCANState {
  std::vector<double> epsilon_key;
  int64_t epsilon_num_features{-1};
  std::shared_ptr<Tensor> epsilon_inv2;
  bool warned_max_clusters{false};
};

}  // namespace detail

namespace {

namespace cd = cluster_dbscan;

constexpr const char* kOpName = "ClusterDBSCANOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};
constexpr DLDataType kUint8Dtype{kDLUInt, 8, 1};
constexpr DLDataType kUint32Dtype{kDLUInt, 32, 1};
constexpr DLDataType kFloat64Dtype{kDLFloat, 64, 1};

struct ClusterDBSCANStageConfig {
  std::vector<double> epsilon;
  std::optional<int64_t> min_points;
  std::optional<int64_t> max_clusters;
};

struct ClusterDBSCANStagePlan {
  ClusterDBSCANStageConfig stage;
  int64_t num_points;
  int64_t num_features;
  int64_t min_points;
};

struct DbscanWorkspace {
  std::shared_ptr<Tensor> is_core_tensor;
  std::shared_ptr<Tensor> cc_a_tensor;
  std::shared_ptr<Tensor> cc_b_tensor;
  std::shared_ptr<Tensor> changed_tensor;
  std::shared_ptr<Tensor> root_flag_tensor;
  std::shared_ptr<Tensor> dense_tensor;
  std::shared_ptr<Tensor> scan_temp_tensor;
  std::shared_ptr<Tensor> sizes_tensor;
  std::shared_ptr<Tensor> remap_tensor;
  uint8_t* is_core{nullptr};
  int64_t* cc_a{nullptr};
  int64_t* cc_b{nullptr};
  int* changed{nullptr};
  uint32_t* root_flag{nullptr};
  uint32_t* dense{nullptr};
  int64_t* roots{nullptr};
};

struct PreparedClusters {
  TensorOutputLoan labels;
  std::vector<int64_t> sorted_sizes;
  int64_t reported_clusters;
};

template <typename T>
struct PreparedEpsilon {
  std::vector<T> host_staging;
  const T* device{nullptr};
};

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

expected<ClusterDBSCANStageConfig, Error> resolve_params(
    Input<Tensor>& epsilon, Input<int64_t>& min_points,
    Input<int64_t>& max_clusters, const ClusterDBSCANParams& pinned,
    int64_t max_features) {
  ClusterDBSCANStageConfig stage;
  HOLOSCAN_TRY(
      stage.epsilon,
      resolve_live_override_via(
          epsilon, pinned.epsilon,
          [max_features](const Tensor& tensor)
              -> expected<std::vector<double>, Error> {
            if (max_features <= 0) {
              return make_unexpected(Error{
                  ErrorCode::kInvalidArgument,
                  std::string(kOpName) +
                      ": MaxTensorExtents.max_features must be positive "
                      "to accept an epsilon override"});
            }
            return read_host_config_vector<double>(
                tensor, kFloat64Dtype, max_features, "epsilon");
          }));

  int64_t min_points_value = 0;
  HOLOSCAN_TRY(min_points_value,
               resolve_live_override(min_points, pinned.min_points));
  if (min_points_value > 0) {
    stage.min_points = min_points_value;
  }

  int64_t max_clusters_value = 0;
  HOLOSCAN_TRY(max_clusters_value,
               resolve_live_override(max_clusters, pinned.max_clusters));
  if (max_clusters_value > 0) {
    stage.max_clusters = max_clusters_value;
  }
  return stage;
}

template <typename T>
expected<ClusterDBSCANStagePlan, Error> build_cluster_stage_plan(
    const Tensor& points, ClusterDBSCANStageConfig stage) {
  try {
    require_cuda_accessible(points, "points", kOpName);
    require_contiguous(points, "points", kOpName);
    require_dtype(points, real_dtype<T>(), "points", kOpName);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }

  const auto& shape = points.shape();
  if (shape.size() != 2) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "ClusterDBSCANOp: points must have shape "
        "(num_points, num_features)"});
  }
  const int64_t num_points = shape[0];
  const int64_t num_features = shape[1];
  if (num_features < 1) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "ClusterDBSCANOp: num_features must be >= 1"});
  }
  if (num_points > std::numeric_limits<int>::max()) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "ClusterDBSCANOp: num_points exceeds the supported range"});
  }
  if (stage.epsilon.empty()) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "ClusterDBSCANOp: epsilon is required"});
  }
  if (static_cast<int64_t>(stage.epsilon.size()) != 1 &&
      static_cast<int64_t>(stage.epsilon.size()) != num_features) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "ClusterDBSCANOp: epsilon must be a scalar (length 1) or match "
        "num_features"});
  }
  for (const double value : stage.epsilon) {
    if (!std::isfinite(value) || value <= 0.0) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "ClusterDBSCANOp: epsilon values must be finite and > 0"});
    }
  }

  const int64_t resolved_min_points =
      stage.min_points.value_or(2 * num_features);
  return ClusterDBSCANStagePlan{
      .stage = std::move(stage),
      .num_points = num_points,
      .num_features = num_features,
      .min_points = resolved_min_points};
}

template <typename T>
expected<PreparedEpsilon<T>, Error> prepare_epsilon_inverse(
    detail::ClusterDBSCANState& state,
    const ClusterDBSCANStagePlan& plan, cudaStream_t stream) {
  const bool cache_hit =
      state.epsilon_inv2 &&
      state.epsilon_num_features == plan.num_features &&
      state.epsilon_key == plan.stage.epsilon;
  PreparedEpsilon<T> prepared;
  if (!cache_hit) {
    prepared.host_staging.resize(
        static_cast<size_t>(plan.num_features));
    for (int64_t axis = 0; axis < plan.num_features; ++axis) {
      const double value =
          plan.stage.epsilon.size() == 1
              ? plan.stage.epsilon[0]
              : plan.stage.epsilon[static_cast<size_t>(axis)];
      prepared.host_staging[static_cast<size_t>(axis)] =
          static_cast<T>(1.0 / (value * value));
    }
    auto epsilon_inv2 =
        make_device_tensor({plan.num_features}, real_dtype<T>());
    HOLOSCAN_TRY(auto epsilon_mut,
                 writable_ptr(*epsilon_inv2, "epsilon_inv2"));
    if (cudaMemcpyAsync(epsilon_mut, prepared.host_staging.data(),
                        sizeof(T) * prepared.host_staging.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return make_unexpected(Error{
          ErrorCode::kFailure,
          "ClusterDBSCANOp: epsilon H2D copy failed"});
    }
    state.epsilon_key = plan.stage.epsilon;
    state.epsilon_num_features = plan.num_features;
    state.epsilon_inv2 = std::move(epsilon_inv2);
  }
  prepared.device =
      static_cast<const T*>(state.epsilon_inv2->data());
  return prepared;
}

expected<DbscanWorkspace, Error> prepare_dbscan_workspace(
    int64_t num_points) {
  DbscanWorkspace workspace;
  workspace.is_core_tensor =
      make_device_tensor({num_points}, kUint8Dtype);
  workspace.cc_a_tensor = make_device_tensor({num_points}, kInt64Dtype);
  workspace.cc_b_tensor = make_device_tensor({num_points}, kInt64Dtype);
  workspace.changed_tensor = make_device_tensor({1}, kInt32Dtype);

  HOLOSCAN_TRY(auto is_core,
               writable_ptr(*workspace.is_core_tensor, "is_core"));
  HOLOSCAN_TRY(auto cc_a, writable_ptr(*workspace.cc_a_tensor, "cc_a"));
  HOLOSCAN_TRY(auto cc_b, writable_ptr(*workspace.cc_b_tensor, "cc_b"));
  HOLOSCAN_TRY(auto changed,
               writable_ptr(*workspace.changed_tensor, "changed"));
  workspace.is_core = static_cast<uint8_t*>(is_core);
  workspace.cc_a = static_cast<int64_t*>(cc_a);
  workspace.cc_b = static_cast<int64_t*>(cc_b);
  workspace.changed = static_cast<int*>(changed);
  return workspace;
}

template <typename T>
expected<void, Error> find_component_roots(
    const T* points, const T* epsilon_inv2,
    const ClusterDBSCANStagePlan& plan, DbscanWorkspace& workspace,
    cudaStream_t stream) {
  try {
    cd::launch_mark_core<T>(
        points, epsilon_inv2, plan.num_points, plan.num_features,
        plan.min_points, workspace.is_core, workspace.cc_a, stream);
    check_cuda("mark_core", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  // Min-label propagation over core-core edges, ping-ponged to convergence.
  int64_t* current = workspace.cc_a;
  int64_t* next = workspace.cc_b;
  const int64_t max_iterations = plan.num_points + 64;
  for (int64_t iteration = 0;; ++iteration) {
    if (iteration > max_iterations) {
      return fail_runtime(
          "ClusterDBSCANOp: connected-components did not converge");
    }
    if (cudaMemsetAsync(workspace.changed, 0, sizeof(int), stream) !=
        cudaSuccess) {
      return fail_runtime("ClusterDBSCANOp: changed-flag clear failed");
    }
    try {
      cd::launch_cc_step<T>(
          points, epsilon_inv2, workspace.is_core, current, next,
          plan.num_points, plan.num_features, workspace.changed, stream);
      check_cuda("cc_step", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
    std::swap(current, next);

    int host_changed = 0;
    if (cudaMemcpyAsync(&host_changed, workspace.changed, sizeof(int),
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

  try {
    cd::launch_assign_border<T>(
        points, epsilon_inv2, workspace.is_core, current, plan.num_points,
        plan.num_features, stream);
    check_cuda("assign_border", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }
  workspace.roots = current;
  return {};
}

expected<int64_t, Error> compact_component_roots(
    int64_t num_points, DbscanWorkspace& workspace, cudaStream_t stream) {
  workspace.root_flag_tensor =
      make_device_tensor({num_points}, kUint32Dtype);
  workspace.dense_tensor = make_device_tensor({num_points}, kUint32Dtype);
  HOLOSCAN_TRY(auto root_flag,
               writable_ptr(*workspace.root_flag_tensor, "root_flag"));
  HOLOSCAN_TRY(auto dense,
               writable_ptr(*workspace.dense_tensor, "dense"));
  workspace.root_flag = static_cast<uint32_t*>(root_flag);
  workspace.dense = static_cast<uint32_t*>(dense);

  try {
    cd::launch_root_flag(workspace.is_core, workspace.roots,
                         workspace.root_flag, num_points, stream);
    check_cuda("root_flag", kOpName);
  } catch (const std::runtime_error& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }

  const size_t scan_bytes = cd::cub_scan_temp_bytes(num_points);
  const int64_t scan_allocation =
      static_cast<int64_t>(std::max<size_t>(scan_bytes, 1));
  workspace.scan_temp_tensor =
      make_device_tensor({scan_allocation}, kUint8Dtype);
  HOLOSCAN_TRY(
      auto scan_temp,
      writable_ptr(*workspace.scan_temp_tensor, "scan_temp"));
  try {
    cd::launch_scan_root_ids(scan_temp, scan_bytes, workspace.root_flag,
                             workspace.dense, num_points, stream);
    check_cuda("scan_root_ids", kOpName);
  } catch (const std::runtime_error& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }

  uint32_t dense_last = 0;
  uint32_t flag_last = 0;
  const int64_t last = num_points - 1;
  if (cudaMemcpyAsync(&dense_last, workspace.dense + last, sizeof(uint32_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&flag_last, workspace.root_flag + last,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        "ClusterDBSCANOp: cluster-count D2H failed"});
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        "ClusterDBSCANOp: cluster-count sync failed"});
  }
  return static_cast<int64_t>(dense_last) +
         static_cast<int64_t>(flag_last);
}

expected<PreparedClusters, Error> prepare_ranked_clusters(
    Output<Tensor>& labels, ExecutionContext& context,
    const ClusterDBSCANStagePlan& plan, int64_t num_clusters,
    DbscanWorkspace& workspace, detail::ClusterDBSCANState& state,
    cudaStream_t stream) {
  const std::vector<int64_t> labels_shape = {plan.num_points};
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
    cd::launch_finalize_labels(workspace.roots, workspace.dense, labels_ptr,
                               plan.num_points, stream);
    check_cuda("finalize_labels", kOpName);
  } catch (const std::runtime_error& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }

  std::vector<int64_t> host_sizes(static_cast<size_t>(num_clusters));
  if (num_clusters > 0) {
    workspace.sizes_tensor =
        make_device_tensor({num_clusters}, kInt64Dtype);
    HOLOSCAN_TRY(auto sizes,
                 writable_ptr(*workspace.sizes_tensor, "sizes_all"));
    if (cudaMemsetAsync(sizes, 0, sizeof(int64_t) * num_clusters, stream) !=
        cudaSuccess) {
      return make_unexpected(Error{
          ErrorCode::kFailure,
          "ClusterDBSCANOp: size-histogram clear failed"});
    }
    try {
      cd::launch_histogram(labels_ptr, static_cast<int64_t*>(sizes),
                           plan.num_points, stream);
      check_cuda("size_histogram", kOpName);
    } catch (const std::runtime_error& e) {
      return make_unexpected(Error{ErrorCode::kFailure, e.what()});
    }
    if (cudaMemcpyAsync(host_sizes.data(), sizes,
                        sizeof(int64_t) * num_clusters,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      return make_unexpected(Error{
          ErrorCode::kFailure, "ClusterDBSCANOp: size D2H failed"});
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return make_unexpected(Error{
          ErrorCode::kFailure, "ClusterDBSCANOp: size sync failed"});
    }
  }

  if (num_clusters > 1) {
    std::vector<int64_t> order(static_cast<size_t>(num_clusters));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&host_sizes](int64_t a, int64_t b) {
                       return host_sizes[static_cast<size_t>(a)] >
                              host_sizes[static_cast<size_t>(b)];
                     });
    std::vector<int64_t> remap(static_cast<size_t>(num_clusters));
    std::vector<int64_t> sorted_sizes(static_cast<size_t>(num_clusters));
    for (int64_t rank = 0; rank < num_clusters; ++rank) {
      const int64_t old_id = order[static_cast<size_t>(rank)];
      remap[static_cast<size_t>(old_id)] = rank;
      sorted_sizes[static_cast<size_t>(rank)] =
          host_sizes[static_cast<size_t>(old_id)];
    }
    host_sizes = std::move(sorted_sizes);

    workspace.remap_tensor =
        make_device_tensor({num_clusters}, kInt64Dtype);
    HOLOSCAN_TRY(auto remap_device,
                 writable_ptr(*workspace.remap_tensor, "remap"));
    if (cudaMemcpyAsync(remap_device, remap.data(),
                        sizeof(int64_t) * num_clusters,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return make_unexpected(Error{
          ErrorCode::kFailure, "ClusterDBSCANOp: remap H2D failed"});
    }
    try {
      cd::launch_remap_labels(
          labels_ptr, static_cast<const int64_t*>(remap_device),
          plan.num_points, stream);
      check_cuda("remap_labels", kOpName);
    } catch (const std::runtime_error& e) {
      return make_unexpected(Error{ErrorCode::kFailure, e.what()});
    }
  }

  int64_t reported_clusters = num_clusters;
  const int64_t max_clusters = plan.stage.max_clusters.value_or(0);
  if (max_clusters > 0 && num_clusters > max_clusters) {
    if (!state.warned_max_clusters) {
      HOLOSCAN_LOG_WARN(
          "ClusterDBSCANOp: {} clusters exceed max_clusters={}, dropping {} "
          "surplus cluster(s) to noise",
          num_clusters, max_clusters, num_clusters - max_clusters);
      state.warned_max_clusters = true;
    }
    try {
      cd::launch_cap_labels(labels_ptr, max_clusters, plan.num_points, stream);
      check_cuda("cap_labels", kOpName);
    } catch (const std::runtime_error& e) {
      return make_unexpected(Error{ErrorCode::kFailure, e.what()});
    }
    reported_clusters = max_clusters;
  }

  auto committed = std::move(*labels_writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return PreparedClusters{.labels = std::move(*labels_loan),
                          .sorted_sizes = std::move(host_sizes),
                          .reported_clusters = reported_clusters};
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
  // labels: int64 (max_points,)
  max_labels_byte_span_ = byte_span_elems(
      static_cast<std::size_t>(max_points),
      sizeof(std::int64_t), who);
  // sizes: int64 (max_clusters,)
  max_sizes_byte_span_ = byte_span_elems(
      static_cast<std::size_t>(max_clusters),
      sizeof(std::int64_t), who);
  state_ = std::make_shared<detail::ClusterDBSCANState>();
}

template <typename T>
void ClusterDBSCANOp<T>::setup(OperatorSpec& spec) {
  // Tensor representations and pool bounds.
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

  // Data plane.
  spec.input(points).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(labels)
      .produces_tensor(pooled_tensor_output(kDevice, 1U, max_labels_byte_span_));
  spec.output(sizes)
      .produces_tensor(pooled_tensor_output(kDevice, 1U, max_sizes_byte_span_));

  // [live] Exact overrides, Params order.
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
  // Receive the device feature matrix.
  HOLOSCAN_TRY(auto points_in, receive_tensor(points, kOpName));

  // Resolve Params plus this tick's Exact overrides.
  const ClusterDBSCANParams& pinned = context.params<ClusterDBSCANParams>();
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(epsilon, min_points, max_clusters, pinned, max_features_));

  // Validate shape/dtype and derive tick geometry.
  HOLOSCAN_TRY(
      auto plan,
      (build_cluster_stage_plan<T>(points_in, std::move(stage))));
  const int64_t num_points = plan.num_points;

  // Select the activation's CUDA stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "ClusterDBSCANOp: no CUDA compute stream"});
  }

  // Empty input: publish required labels, then the independent optional sizes.
  if (num_points == 0) {
    const std::vector<int64_t> empty_shape = {0};
    auto noop = [](void*, std::size_t) -> expected<void, Error> {
      return {};
    };
    HOLOSCAN_TRY(auto labels_loan,
                 prepare_device_tensor(labels, context, empty_shape,
                                      kInt64Dtype, noop));
    HOLOSCAN_TRY(
        emit_prepared_tensor(labels, context, std::move(labels_loan)));
    if (sizes_out_.active()) {
      HOLOSCAN_TRY(auto loan,
                   prepare_device_tensor(sizes, context, empty_shape,
                                        kInt64Dtype, noop));
      sizes_out_.emit(sizes, context, std::move(loan),
                      "ClusterDBSCANOp.sizes");
    }
    retained_points_ = std::move(points_in);
    return {};
  }

  // Resolve the cached device inverse-radius vector for this tick.
  HOLOSCAN_TRY(
      auto epsilon,
      (prepare_epsilon_inverse<T>(*state_, plan, stream)));
  const T* points_ptr = static_cast<const T*>(points_in.data());

  // Find the minimum component root for every core or border point.
  HOLOSCAN_TRY(auto workspace, prepare_dbscan_workspace(num_points));
  HOLOSCAN_TRY(find_component_roots<T>(
      points_ptr, epsilon.device, plan, workspace, stream));

  // Compact roots into ascending-root dense ids and recover their count.
  HOLOSCAN_TRY(
      auto num_clusters,
      compact_component_roots(num_points, workspace, stream));

  // Build descending-size labels and retain the sorted counts for sizes.
  HOLOSCAN_TRY(
      auto prepared,
      prepare_ranked_clusters(labels, context, plan, num_clusters, workspace,
                              *state_, stream));

  // Publish required labels before touching the independent optional output.
  HOLOSCAN_TRY(
      emit_prepared_tensor(labels, context, std::move(prepared.labels)));

  // sizes is optional/latched. Cluster counts are already needed to establish
  // deterministic label order, so latching skips only its loan and H2D copy.
  if (sizes_out_.active()) {
    const std::vector<int64_t> sizes_shape = {prepared.reported_clusters};
    HOLOSCAN_TRY(
        auto sizes_loan,
        prepare_device_tensor(
            sizes, context, sizes_shape, kInt64Dtype,
            [&](void* dst, std::size_t bytes) -> expected<void, Error> {
              if (bytes == 0) {
                return {};
              }
              if (cudaMemcpyAsync(dst, prepared.sorted_sizes.data(), bytes,
                                  cudaMemcpyHostToDevice, stream) !=
                  cudaSuccess) {
                return fail_runtime("ClusterDBSCANOp: sizes H2D copy failed");
              }
              return {};
            }));
    sizes_out_.emit(sizes, context, std::move(sizes_loan),
                    "ClusterDBSCANOp.sizes");
  }

  retained_points_ = std::move(points_in);
  return {};
}

template class ClusterDBSCANOp<float>;
template class ClusterDBSCANOp<double>;

}  // namespace holoscan::radar
