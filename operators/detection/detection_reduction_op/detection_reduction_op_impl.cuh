// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CUDA-TU implementation of DetectionReductionOp<T, ScoreFn>.
// Include from a .cu that explicitly instantiates the needed specializations.

#pragma once

#include "detection/detection_reduction_op/detection_reduction_op.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "detection/detection_reduction_op/detection_reduction_stage.hpp"
#include "radar_common/sdk/config_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace detection_reduction_op_detail {

constexpr const char* kOpName = "DetectionReductionOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
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

}  // namespace detection_reduction_op_detail

using detection_reduction_op_detail::fail_invalid;
using detection_reduction_op_detail::fail_runtime;
using detection_reduction_op_detail::kFloat64Dtype;
using detection_reduction_op_detail::kInt64Dtype;
using detection_reduction_op_detail::kOpName;
using detection_reduction_op_detail::writable_ptr;

template <typename T, typename ScoreFn>
DetectionReductionOp<T, ScoreFn>::DetectionReductionOp(
    const MaxTensorExtents& extents) {
  const char* who = "DetectionReductionOp";
  const std::int64_t max_clusters =
      require_extent(extents.max_clusters, "max_clusters", who);
  max_features_ = extents.max_features.value_or(0);
  // rep_idx shape is (num_clusters, idx_cols). idx_cols is at most 2
  // + leading-batch cols; conservatively assume max_points columns.
  const std::int64_t max_points =
      extents.max_points.value_or(max_clusters);
  const std::int64_t max_cols = std::max<std::int64_t>(
      2, 2 + max_points);
  (void)max_cols;
  // rep_idx is int64 (num_clusters, idx_cols). Size for generous
  // idx_cols = 8 (more than any realistic preserved-batch count).
  max_byte_span_ = byte_span_elems(
      product_extents({max_clusters, 8}, who),
      sizeof(std::int64_t), who);
  extents_ = extents;
}

template <typename T, typename ScoreFn>
void DetectionReductionOp<T, ScoreFn>::setup(OperatorSpec& spec) {
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
  spec.input(labels).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(detection_idx).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  // Optional points: SafeDefault / omitted from contract so unconnected
  // graphs compile (required at runtime unless representative == first).
  spec.input(points).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(rep_idx)
      .produces_tensor(pooled_tensor_output(kDevice, 2U, max_byte_span_));
  // [live] override ports. Leaving them unconnected is legal; compute()
  // seeds from Params on any tick without a match.
  spec.input(representative).queue_depth(2U);
  spec.input(score_method).queue_depth(2U);
  spec.input(feature_weights).queue_depth(2U).expects_tensor(tensor_input(kHostVector));
}

template <typename T, typename ScoreFn>
Contract DetectionReductionOp<T, ScoreFn>::contract() const {
  // Trigger when labels + detection_idx + points share an aligned event time.
  // The three [live] overrides join unconditionally; they are not alignment
  // members, so leaving one unwired is legal and reads as no override.
  //
  // [EA HSDK5 ONLY] points is Exact in the contract even though OPERATOR_API
  // marks it optional for representative=first. Unlike the overrides it is a
  // trigger port, so an unwired points is still TEMPORAL_INPUT_UNCONNECTED and
  // omitting it while wiring yields a missed sample (same family as
  // FeatureAssembly OPEN_ITEMS §1). Graph tests always wire points (dummy for
  // first); see OPEN_ITEMS.
  Contract result;
  result
      .trigger(OnAligned{.ports = ports(labels, detection_idx, points),
                         .window = std::chrono::nanoseconds{0}})
      .input(labels, Exact{})
      .input(detection_idx, Exact{})
      .input(points, Exact{});
  add_live_override_to_contract(result, representative);
  add_live_override_to_contract(result, score_method);
  add_live_override_to_contract(result, feature_weights);
  return result;
}

template <typename T, typename ScoreFn>
expected<void, Error> DetectionReductionOp<T, ScoreFn>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto labels_in, receive_tensor(labels, kOpName));
  HOLOSCAN_TRY(auto idx_in, receive_tensor(detection_idx, kOpName));
  HOLOSCAN_TRY(auto maybe_points, receive_optional_value(points));

  const DetectionReductionParams& pinned =
      context.params<DetectionReductionParams>();
  HOLOSCAN_TRY(auto effective_representative,
               resolve_live_override(representative, pinned.representative));
  HOLOSCAN_TRY(auto effective_score_method,
               resolve_live_override(score_method, pinned.score_method));
  HOLOSCAN_TRY(
      auto effective_feature_weights,
      resolve_live_override_via(
          feature_weights, pinned.feature_weights,
          [this](const Tensor& tensor)
              -> expected<std::vector<double>, Error> {
            // Only a wired-and-sampled override needs the bound; an app that
            // never overrides feature_weights may leave max_features unset.
            if (max_features_ <= 0) {
              return make_unexpected(Error{
                  ErrorCode::kInvalidArgument,
                  "DetectionReductionOp: MaxTensorExtents.max_features must "
                  "be positive to accept a feature_weights override"});
            }
            return read_host_config_vector<double>(
                tensor, kFloat64Dtype, max_features_, "feature_weights");
          }));

  if (effective_representative == RepresentativePolicy::kPeak &&
      effective_score_method == ScoreMethod::kCustom &&
      std::is_same_v<ScoreFn, NoCustomScore>) {
    return fail_invalid(
        "DetectionReductionOp: score_method=custom requires a compile-time "
        "ScoreFn template argument (DetectionReductionOp<T, ScoreFn>)");
  }

  try {
    require_cuda_accessible(labels_in, "labels", kOpName);
    require_contiguous(labels_in, "labels", kOpName);
    require_dtype(labels_in, kInt64Dtype, "labels", kOpName);
    require_cuda_accessible(idx_in, "detection_idx", kOpName);
    require_contiguous(idx_in, "detection_idx", kOpName);
    require_dtype(idx_in, kInt64Dtype, "detection_idx", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  if (labels_in.shape().size() != 1) {
    return fail_invalid(
        "DetectionReductionOp: labels must have shape (num_points,)");
  }
  const auto& idx_shape = idx_in.shape();
  if (idx_shape.size() != 2 || idx_shape[1] < 2) {
    return fail_invalid(
        "DetectionReductionOp: detection_idx must have shape "
        "(num_points, 2 [+ n_batch])");
  }
  const int64_t num_points = labels_in.shape()[0];
  if (idx_shape[0] != num_points) {
    return fail_invalid(
        "DetectionReductionOp: labels and detection_idx row counts must "
        "match");
  }
  const int64_t idx_cols = idx_shape[1];

  const bool needs_points =
      effective_representative != RepresentativePolicy::kFirst;
  const bool needs_weights =
      effective_representative == RepresentativePolicy::kMedoid ||
      (effective_representative == RepresentativePolicy::kPeak &&
       effective_score_method != ScoreMethod::kCustom);

  if (needs_points && !maybe_points.has_value()) {
    return fail_invalid(
        std::string("DetectionReductionOp: points required for "
                    "representative='") +
        std::string(to_string(effective_representative)) + "'");
  }

  int64_t num_features = 0;
  const T* points_ptr = nullptr;
  if (maybe_points.has_value()) {
    try {
      require_cuda_accessible(*maybe_points, "points", kOpName);
      require_contiguous(*maybe_points, "points", kOpName);
      require_dtype(*maybe_points, real_dtype<T>(), "points", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    if (maybe_points->shape().size() != 2 ||
        maybe_points->shape()[0] != num_points) {
      return fail_invalid(
          "DetectionReductionOp: points must have shape "
          "(num_points, num_features)");
    }
    num_features = maybe_points->shape()[1];
    points_ptr = static_cast<const T*>(maybe_points->data());
  }

  if (needs_weights) {
    if (static_cast<int64_t>(effective_feature_weights.size()) !=
        num_features) {
      return fail_invalid(
          "DetectionReductionOp: feature_weights length must equal "
          "num_features");
    }
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "DetectionReductionOp: no CUDA compute "
                                 "stream"});
  }

  auto emit_empty = [&]() -> expected<void, Error> {
    const std::vector<int64_t> empty_shape = {0, idx_cols};
    HOLOSCAN_TRY(emit_device_tensor(
        rep_idx, context, empty_shape, kInt64Dtype,
        [](void*, std::size_t) -> expected<void, Error> { return {}; }));
    retained_labels_ = std::move(labels_in);
    retained_idx_ = std::move(idx_in);
    if (maybe_points.has_value()) {
      retained_points_ = std::move(*maybe_points);
    }
    retained_weights_.reset();
    return {};
  };

  if (num_points == 0) {
    return emit_empty();
  }

  // Discover max label → slot count.
  auto max_label_t = make_device_tensor({1}, kInt64Dtype);
  auto any_cluster_t =
      make_device_tensor({1}, DLDataType{kDLInt, 32, 1});
  HOLOSCAN_TRY(auto max_label_ptr, writable_ptr(*max_label_t, "max_label"));
  HOLOSCAN_TRY(auto any_cluster_ptr,
               writable_ptr(*any_cluster_t, "any_cluster"));
  {
    const int64_t zero = 0;
    const int zero_i = 0;
    if (cudaMemcpyAsync(max_label_ptr, &zero, sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(any_cluster_ptr, &zero_i, sizeof(int),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return fail_runtime("DetectionReductionOp: max-label clear failed");
    }
  }
  try {
    detail::launch_find_max_label(
        static_cast<const int64_t*>(labels_in.data()), num_points,
        static_cast<int64_t*>(max_label_ptr),
        static_cast<int*>(any_cluster_ptr), stream);
    check_cuda("find_max_label", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  int64_t max_label = 0;
  int any_cluster = 0;
  if (cudaMemcpyAsync(&max_label, max_label_ptr, sizeof(int64_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&any_cluster, any_cluster_ptr, sizeof(int),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: max-label D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: max-label sync failed");
  }
  if (any_cluster == 0) {
    return emit_empty();
  }
  const int64_t n_slots = max_label + 1;
  constexpr int64_t kMaxSlots = 1 << 20;
  if (n_slots > kMaxSlots) {
    return fail_invalid(
        "DetectionReductionOp: max cluster label exceeds supported range "
        "(expected DBSCAN-style dense labels)");
  }

  auto best_rows = make_device_tensor({n_slots}, kInt64Dtype);
  HOLOSCAN_TRY(auto best_rows_ptr, writable_ptr(*best_rows, "best_rows"));
  try {
    detail::launch_init_i64(static_cast<int64_t*>(best_rows_ptr), n_slots,
                            detail::kEmptyRow, stream);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  std::shared_ptr<holoscan::Tensor> weights_dev;
  if (needs_weights) {
    std::vector<T> host_w(static_cast<size_t>(num_features));
    for (int64_t f = 0; f < num_features; ++f) {
      host_w[static_cast<size_t>(f)] =
          static_cast<T>(
              effective_feature_weights[static_cast<size_t>(f)]);
    }
    weights_dev = make_device_tensor({num_features}, real_dtype<T>());
    HOLOSCAN_TRY(auto weights_ptr, writable_ptr(*weights_dev, "weights"));
    if (cudaMemcpyAsync(weights_ptr, host_w.data(),
                        sizeof(T) * host_w.size(), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return fail_runtime("DetectionReductionOp: feature_weights H2D failed");
    }
  }

  try {
    if (effective_representative == RepresentativePolicy::kFirst) {
      detail::launch_select_first(
          static_cast<const int64_t*>(labels_in.data()), num_points,
          static_cast<int64_t*>(best_rows_ptr), stream);
    } else if (effective_representative == RepresentativePolicy::kPeak) {
      auto best_score =
          make_device_tensor({n_slots}, DLDataType{kDLUInt, 64, 1});
      HOLOSCAN_TRY(auto best_score_ptr,
                   writable_ptr(*best_score, "best_score"));
      detail::launch_init_u64(
          static_cast<detail::cuda_u64*>(best_score_ptr), n_slots, 0ull,
          stream);
      if (effective_score_method == ScoreMethod::kCustom) {
        if constexpr (!std::is_same_v<ScoreFn, NoCustomScore>) {
          detail::launch_peak_select_custom<T, ScoreFn>(
              static_cast<const int64_t*>(labels_in.data()), points_ptr,
              num_points, num_features, ScoreFn{},
              static_cast<detail::cuda_u64*>(best_score_ptr),
              static_cast<int64_t*>(best_rows_ptr), stream);
        }
      } else {
        detail::launch_peak_select<T>(
            static_cast<const int64_t*>(labels_in.data()), points_ptr,
            static_cast<const T*>(weights_dev->data()), num_points,
            num_features, effective_score_method,
            static_cast<detail::cuda_u64*>(best_score_ptr),
            static_cast<int64_t*>(best_rows_ptr), stream);
      }
    } else {
      // medoid
      auto counts = make_device_tensor({n_slots}, kInt64Dtype);
      auto sum_feat =
          make_device_tensor({n_slots * num_features}, real_dtype<T>());
      HOLOSCAN_TRY(auto counts_ptr, writable_ptr(*counts, "counts"));
      HOLOSCAN_TRY(auto sum_feat_ptr, writable_ptr(*sum_feat, "sum_feat"));
      if (cudaMemsetAsync(counts_ptr, 0, sizeof(int64_t) * n_slots, stream) !=
          cudaSuccess) {
        return fail_runtime("DetectionReductionOp: medoid counts clear failed");
      }
      detail::launch_init_t_zero<T>(static_cast<T*>(sum_feat_ptr),
                                    n_slots * num_features, stream);
      auto best_score =
          make_device_tensor({n_slots}, DLDataType{kDLUInt, 64, 1});
      HOLOSCAN_TRY(auto best_score_ptr,
                   writable_ptr(*best_score, "best_score"));
      detail::launch_init_u64(
          static_cast<detail::cuda_u64*>(best_score_ptr), n_slots, 0ull,
          stream);
      detail::launch_medoid_select<T>(
          static_cast<const int64_t*>(labels_in.data()), points_ptr,
          static_cast<const T*>(weights_dev->data()),
          static_cast<T*>(sum_feat_ptr), static_cast<int64_t*>(counts_ptr),
          num_points, n_slots, num_features,
          static_cast<detail::cuda_u64*>(best_score_ptr),
          static_cast<int64_t*>(best_rows_ptr), stream);
    }
    check_cuda("select_representative", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  auto cluster_count_t = make_device_tensor({1}, kInt64Dtype);
  HOLOSCAN_TRY(auto cluster_count_ptr,
               writable_ptr(*cluster_count_t, "cluster_count"));
  {
    const int64_t zero = 0;
    if (cudaMemcpyAsync(cluster_count_ptr, &zero, sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return fail_runtime("DetectionReductionOp: cluster-count clear failed");
    }
  }
  try {
    detail::launch_count_occupied(
        static_cast<const int64_t*>(best_rows_ptr), n_slots,
        static_cast<int64_t*>(cluster_count_ptr), stream);
    check_cuda("count_occupied", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }
  int64_t num_clusters = 0;
  if (cudaMemcpyAsync(&num_clusters, cluster_count_ptr, sizeof(int64_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: cluster-count D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: cluster-count sync failed");
  }

  const std::vector<int64_t> rep_shape = {num_clusters, idx_cols};
  auto rep_loan = rep_idx.allocate_tensor(
      TensorLoanRequest{.shape = rep_shape, .dtype = kInt64Dtype});
  if (!rep_loan) {
    return make_unexpected(std::move(rep_loan).error());
  }
  auto rep_writer = rep_loan->write(context.cuda_stream());
  if (!rep_writer) {
    return make_unexpected(std::move(rep_writer).error());
  }
  int64_t* rep_ptr = static_cast<int64_t*>(rep_writer->data());

  if (num_clusters > 0) {
    try {
      detail::launch_compact_and_gather(
          static_cast<const int64_t*>(best_rows_ptr),
          static_cast<const int64_t*>(idx_in.data()), n_slots, idx_cols,
          rep_ptr, stream);
      check_cuda("compact_and_gather", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  }

  auto rep_committed = std::move(*rep_writer).commit();
  if (!rep_committed) {
    return make_unexpected(std::move(rep_committed).error());
  }
  auto rep_emitted = rep_idx.emit(
      std::move(*rep_loan),
      EmitOptions{.capture_time = context.activation_time()});
  if (!rep_emitted) {
    return make_unexpected(std::move(rep_emitted).error());
  }

  retained_labels_ = std::move(labels_in);
  retained_idx_ = std::move(idx_in);
  if (maybe_points.has_value()) {
    retained_points_ = std::move(*maybe_points);
  }
  retained_weights_ = std::move(weights_dev);
  return {};
}

}  // namespace holoscan::radar
