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
constexpr std::int64_t kMaxRepIdxCols = 10;

struct DetectionReductionStageConfig {
  RepresentativePolicy representative;
  ScoreMethod score_method;
  std::vector<double> feature_weights;
};

template <typename T>
struct DetectionReductionStagePlan {
  DetectionReductionStageConfig stage;
  std::int64_t num_points;
  std::int64_t idx_cols;
  std::int64_t num_features;
  const T* points;
  bool needs_weights;
};

inline expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

inline expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

inline expected<void*, Error> writable_ptr(holoscan::Tensor& tensor, const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              std::string(kOpName) + ": " + what + " is not writable: " + mut.error().message});
  }
  return *mut;
}

inline expected<DetectionReductionStageConfig, Error> resolve_params(
    Input<RepresentativePolicy>& representative, Input<ScoreMethod>& score_method,
    Input<Tensor>& feature_weights, const DetectionReductionParams& pinned,
    std::int64_t max_features) {
  DetectionReductionStageConfig stage;
  HOLOSCAN_TRY(stage.representative, resolve_live_override(representative, pinned.representative));
  HOLOSCAN_TRY(stage.score_method, resolve_live_override(score_method, pinned.score_method));
  HOLOSCAN_TRY(
      stage.feature_weights,
      resolve_live_override_via(
          feature_weights, pinned.feature_weights,
          [max_features](const Tensor& tensor) -> expected<std::vector<double>, Error> {
            if (max_features <= 0) {
              return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                           std::string(kOpName) +
                                               ": MaxTensorExtents.max_features must be positive "
                                               "to accept a feature_weights override"});
            }
            return read_host_config_vector<double>(tensor, kFloat64Dtype, max_features,
                                                   "feature_weights");
          }));
  return stage;
}

template <typename T, typename ScoreFn>
expected<DetectionReductionStagePlan<T>, Error> build_reduction_stage_plan(
    const Tensor& labels, const Tensor& detection_idx, const Tensor& points,
    DetectionReductionStageConfig stage) {
  try {
    require_cuda_accessible(labels, "labels", kOpName);
    require_contiguous(labels, "labels", kOpName);
    require_dtype(labels, kInt64Dtype, "labels", kOpName);
    require_cuda_accessible(detection_idx, "detection_idx", kOpName);
    require_contiguous(detection_idx, "detection_idx", kOpName);
    require_dtype(detection_idx, kInt64Dtype, "detection_idx", kOpName);
    require_cuda_accessible(points, "points", kOpName);
    require_contiguous(points, "points", kOpName);
    require_dtype(points, real_dtype<T>(), "points", kOpName);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }

  if (labels.shape().size() != 1) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": labels must have shape (num_points,)"});
  }
  const auto& idx_shape = detection_idx.shape();
  if (idx_shape.size() != 2 || idx_shape[1] < 2) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, std::string(kOpName) +
                                                                  ": detection_idx must have shape "
                                                                  "(num_points, 2 [+ n_batch])"});
  }
  const std::int64_t num_points = labels.shape()[0];
  if (idx_shape[0] != num_points) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": labels and detection_idx row counts must match"});
  }
  if (idx_shape[1] > kMaxRepIdxCols) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": detection_idx has too many columns for the rep_idx pool"});
  }

  const bool uses_custom_score = stage.score_method == ScoreMethod::kCustom &&
                                 (stage.representative == RepresentativePolicy::kPeak ||
                                  (stage.representative == RepresentativePolicy::kWeightedMean &&
                                   !stage.feature_weights.empty()));
  if (uses_custom_score && std::is_same_v<ScoreFn, NoCustomScore>) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) +
                                     ": score_method=custom requires a compile-time ScoreFn "
                                     "template argument (DetectionReductionOp<T, ScoreFn>)"});
  }

  const bool needs_weights =
      stage.representative == RepresentativePolicy::kMedoid ||
      (stage.representative == RepresentativePolicy::kPeak &&
       stage.score_method != ScoreMethod::kCustom) ||
      (stage.representative == RepresentativePolicy::kWeightedMean &&
       stage.score_method != ScoreMethod::kCustom && !stage.feature_weights.empty());
  if (points.shape().size() != 2 || points.shape()[0] != num_points) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, std::string(kOpName) +
                                                                  ": points must have shape "
                                                                  "(num_points, num_features)"});
  }

  const std::int64_t num_features = points.shape()[1];
  if (num_features <= 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": points must contain at least one feature column"});
  }
  const T* points_ptr = static_cast<const T*>(points.data());
  const bool validates_weight_shape =
      needs_weights || (stage.representative == RepresentativePolicy::kWeightedMean &&
                        !stage.feature_weights.empty());
  if (validates_weight_shape &&
      static_cast<std::int64_t>(stage.feature_weights.size()) != num_features) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": feature_weights length must equal num_features"});
  }

  return DetectionReductionStagePlan<T>{
      .stage = std::move(stage),
      .num_points = num_points,
      .idx_cols = idx_shape[1],
      .num_features = num_features,
      .points = points_ptr,
      .needs_weights = needs_weights,
  };
}

inline expected<void, Error> run_first_selection(const Tensor& labels, std::int64_t num_points,
                                                 void* best_rows, cudaStream_t stream) {
  try {
    detail::launch_select_first(static_cast<const std::int64_t*>(labels.data()), num_points,
                                static_cast<std::int64_t*>(best_rows), stream);
    check_cuda("select_first", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }
  return {};
}

template <typename T, typename ScoreFn>
expected<void, Error> run_peak_selection(const Tensor& labels, const T* points, const T* weights,
                                         std::int64_t num_points, std::int64_t num_features,
                                         std::int64_t n_slots, ScoreMethod score_method,
                                         void* best_rows, cudaStream_t stream) {
  auto best_score = make_device_tensor({n_slots}, DLDataType{kDLUInt, 64, 1});
  HOLOSCAN_TRY(auto best_score_ptr, writable_ptr(*best_score, "best_score"));
  try {
    detail::launch_init_u64(static_cast<detail::cuda_u64*>(best_score_ptr), n_slots, 0ull, stream);
    if (score_method == ScoreMethod::kCustom) {
      if constexpr (!std::is_same_v<ScoreFn, NoCustomScore>) {
        detail::launch_peak_select_custom<T, ScoreFn>(
            static_cast<const std::int64_t*>(labels.data()), points, num_points, num_features,
            ScoreFn{}, static_cast<detail::cuda_u64*>(best_score_ptr),
            static_cast<std::int64_t*>(best_rows), stream);
      }
    } else {
      detail::launch_peak_select<T>(static_cast<const std::int64_t*>(labels.data()), points,
                                    weights, num_points, num_features, score_method,
                                    static_cast<detail::cuda_u64*>(best_score_ptr),
                                    static_cast<std::int64_t*>(best_rows), stream);
    }
    check_cuda("select_peak", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }
  return {};
}

template <typename T>
expected<void, Error> run_medoid_selection(const Tensor& labels, const T* points, const T* weights,
                                           std::int64_t num_points, std::int64_t num_features,
                                           std::int64_t n_slots, void* best_rows,
                                           cudaStream_t stream) {
  auto counts = make_device_tensor({n_slots}, kInt64Dtype);
  auto sum_feat = make_device_tensor({n_slots * num_features}, real_dtype<T>());
  auto best_score = make_device_tensor({n_slots}, DLDataType{kDLUInt, 64, 1});
  HOLOSCAN_TRY(auto counts_ptr, writable_ptr(*counts, "counts"));
  HOLOSCAN_TRY(auto sum_feat_ptr, writable_ptr(*sum_feat, "sum_feat"));
  HOLOSCAN_TRY(auto best_score_ptr, writable_ptr(*best_score, "best_score"));

  if (cudaMemsetAsync(counts_ptr, 0, sizeof(std::int64_t) * n_slots, stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: medoid counts clear failed");
  }
  try {
    detail::launch_init_t_zero<T>(static_cast<T*>(sum_feat_ptr), n_slots * num_features, stream);
    detail::launch_init_u64(static_cast<detail::cuda_u64*>(best_score_ptr), n_slots, 0ull, stream);
    detail::launch_medoid_select<T>(static_cast<const std::int64_t*>(labels.data()), points,
                                    weights, static_cast<T*>(sum_feat_ptr),
                                    static_cast<std::int64_t*>(counts_ptr), num_points, n_slots,
                                    num_features, static_cast<detail::cuda_u64*>(best_score_ptr),
                                    static_cast<std::int64_t*>(best_rows), stream);
    check_cuda("select_medoid", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }
  return {};
}

template <typename T>
struct WeightedMeanWorkspace {
  std::shared_ptr<Tensor> point_means;
  std::shared_ptr<Tensor> index_means;
  std::shared_ptr<Tensor> invalid_weight;
};

template <typename T, typename ScoreFn>
expected<WeightedMeanWorkspace<T>, Error> run_weighted_mean_selection(
    const Tensor& labels, const Tensor& detection_idx, const T* points, const T* weights,
    bool uniform_weights, std::int64_t num_points, std::int64_t num_features, std::int64_t idx_cols,
    std::int64_t n_slots, ScoreMethod score_method, void* best_rows, cudaStream_t stream) {
  auto weight_sums = make_device_tensor({n_slots}, real_dtype<T>());
  auto point_means = make_device_tensor({n_slots * num_features}, real_dtype<T>());
  auto index_means = make_device_tensor({n_slots * idx_cols}, real_dtype<T>());
  auto counts = make_device_tensor({n_slots}, kInt64Dtype);
  auto invalid_weight = make_device_tensor({1}, DLDataType{kDLInt, 32, 1});
  auto best_score = make_device_tensor({n_slots}, DLDataType{kDLUInt, 64, 1});

  HOLOSCAN_TRY(auto weight_sums_ptr, writable_ptr(*weight_sums, "weight_sums"));
  HOLOSCAN_TRY(auto point_means_ptr, writable_ptr(*point_means, "point_means"));
  HOLOSCAN_TRY(auto index_means_ptr, writable_ptr(*index_means, "index_means"));
  HOLOSCAN_TRY(auto counts_ptr, writable_ptr(*counts, "counts"));
  HOLOSCAN_TRY(auto invalid_weight_ptr, writable_ptr(*invalid_weight, "invalid_weight"));
  HOLOSCAN_TRY(auto best_score_ptr, writable_ptr(*best_score, "best_score"));

  if (cudaMemsetAsync(invalid_weight_ptr, 0, sizeof(int), stream) != cudaSuccess) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "DetectionReductionOp: weighted-mean validation clear failed"});
  }
  try {
    detail::launch_init_t_zero<T>(static_cast<T*>(weight_sums_ptr), n_slots, stream);
    detail::launch_init_t_zero<T>(static_cast<T*>(point_means_ptr), n_slots * num_features, stream);
    detail::launch_init_t_zero<T>(static_cast<T*>(index_means_ptr), n_slots * idx_cols, stream);
    detail::launch_init_i64(static_cast<int64_t*>(counts_ptr), n_slots, 0, stream);
    detail::launch_init_u64(static_cast<detail::cuda_u64*>(best_score_ptr), n_slots, 0ull, stream);
    if (uniform_weights) {
      detail::launch_weighted_mean_select<T, detail::BuiltInRowScore<T>>(
          static_cast<const int64_t*>(labels.data()),
          static_cast<const int64_t*>(detection_idx.data()), points, num_points, n_slots, idx_cols,
          num_features,
          detail::BuiltInRowScore<T>{
              .weights = nullptr,
              .method = score_method,
              .uniform = true,
          },
          static_cast<T*>(weight_sums_ptr), static_cast<T*>(point_means_ptr),
          static_cast<T*>(index_means_ptr), static_cast<int64_t*>(counts_ptr),
          static_cast<int*>(invalid_weight_ptr), static_cast<detail::cuda_u64*>(best_score_ptr),
          static_cast<int64_t*>(best_rows), stream);
    } else if (score_method == ScoreMethod::kCustom) {
      if constexpr (!std::is_same_v<ScoreFn, NoCustomScore>) {
        detail::launch_weighted_mean_select<T, ScoreFn>(
            static_cast<const int64_t*>(labels.data()),
            static_cast<const int64_t*>(detection_idx.data()), points, num_points, n_slots,
            idx_cols, num_features, ScoreFn{}, static_cast<T*>(weight_sums_ptr),
            static_cast<T*>(point_means_ptr), static_cast<T*>(index_means_ptr),
            static_cast<int64_t*>(counts_ptr), static_cast<int*>(invalid_weight_ptr),
            static_cast<detail::cuda_u64*>(best_score_ptr), static_cast<int64_t*>(best_rows),
            stream);
      }
    } else {
      detail::launch_weighted_mean_select<T, detail::BuiltInRowScore<T>>(
          static_cast<const int64_t*>(labels.data()),
          static_cast<const int64_t*>(detection_idx.data()), points, num_points, n_slots, idx_cols,
          num_features,
          detail::BuiltInRowScore<T>{
              .weights = weights,
              .method = score_method,
              .uniform = uniform_weights,
          },
          static_cast<T*>(weight_sums_ptr), static_cast<T*>(point_means_ptr),
          static_cast<T*>(index_means_ptr), static_cast<int64_t*>(counts_ptr),
          static_cast<int*>(invalid_weight_ptr), static_cast<detail::cuda_u64*>(best_score_ptr),
          static_cast<int64_t*>(best_rows), stream);
    }
    check_cuda("select_weighted_mean", kOpName);
  } catch (const std::runtime_error& e) {
    return make_unexpected(Error{ErrorCode::kFailure, std::string(e.what())});
  }
  return WeightedMeanWorkspace<T>{
      .point_means = std::move(point_means),
      .index_means = std::move(index_means),
      .invalid_weight = std::move(invalid_weight),
  };
}

}  // namespace detection_reduction_op_detail

using detection_reduction_op_detail::build_reduction_stage_plan;
using detection_reduction_op_detail::fail_invalid;
using detection_reduction_op_detail::fail_runtime;
using detection_reduction_op_detail::kFloat64Dtype;
using detection_reduction_op_detail::kInt64Dtype;
using detection_reduction_op_detail::kMaxRepIdxCols;
using detection_reduction_op_detail::kOpName;
using detection_reduction_op_detail::resolve_params;
using detection_reduction_op_detail::run_first_selection;
using detection_reduction_op_detail::run_medoid_selection;
using detection_reduction_op_detail::run_peak_selection;
using detection_reduction_op_detail::run_weighted_mean_selection;
using detection_reduction_op_detail::WeightedMeanWorkspace;
using detection_reduction_op_detail::writable_ptr;

template <typename T, typename ScoreFn>
DetectionReductionOp<T, ScoreFn>::DetectionReductionOp(const MaxTensorExtents& extents) {
  const char* who = "DetectionReductionOp";
  const std::int64_t max_clusters = require_extent(extents.max_clusters, "max_clusters", who);
  max_features_ = require_extent(extents.max_features, "max_features", who);
  max_rep_idx_byte_span_ = byte_span_elems(product_extents({max_clusters, kMaxRepIdxCols}, who),
                                           sizeof(std::int64_t), who);
  max_reduced_points_byte_span_ =
      byte_span_elems(product_extents({max_clusters, max_features_}, who), sizeof(T), who);
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
  spec.input(points).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(rep_idx).produces_tensor(pooled_tensor_output(kDevice, 2U, max_rep_idx_byte_span_));
  spec.output(reduced_points)
      .produces_tensor(pooled_tensor_output(kDevice, 2U, max_reduced_points_byte_span_));
  // [live] Exact overrides, Params order (schema and POD alike).
  spec.input(representative).optional().queue_depth(2U);
  spec.input(score_method).optional().queue_depth(2U);
  spec.input(feature_weights).optional().queue_depth(2U).expects_tensor(tensor_input(kHostVector));
}

template <typename T, typename ScoreFn>
Contract DetectionReductionOp<T, ScoreFn>::contract() const {
  Contract result;
  result.trigger(OnAligned{.ports = ports(labels, detection_idx, points),
                           .window = std::chrono::nanoseconds{0}});
  result.input(labels, Exact{}).input(detection_idx, Exact{}).input(points, Exact{});
  add_live_override_to_contract(result, representative);
  add_live_override_to_contract(result, score_method);
  add_live_override_to_contract(result, feature_weights);
  return result;
}

template <typename T, typename ScoreFn>
expected<void, Error> DetectionReductionOp<T, ScoreFn>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  // Receive the aligned data plane.
  HOLOSCAN_TRY(auto labels_in, receive_tensor(labels, kOpName));
  HOLOSCAN_TRY(auto idx_in, receive_tensor(detection_idx, kOpName));
  HOLOSCAN_TRY(auto points_in, receive_tensor(points, kOpName));

  // Resolve this tick's Exact overrides onto pinned Params.
  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const DetectionReductionParams& pinned = context.params<DetectionReductionParams>();
  HOLOSCAN_TRY(auto stage, resolve_params(representative, score_method, feature_weights, pinned,
                                          max_features_));

  // Validate named shapes/dtypes and derive the stage geometry.
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(auto plan, (build_reduction_stage_plan<T, ScoreFn>(labels_in, idx_in, points_in,
                                                                  std::move(stage))));
  const auto effective_representative = plan.stage.representative;
  const auto effective_score_method = plan.stage.score_method;
  const auto& effective_feature_weights = plan.stage.feature_weights;
  const int64_t num_points = plan.num_points;
  const int64_t idx_cols = plan.idx_cols;
  const int64_t num_features = plan.num_features;
  const T* points_ptr = plan.points;
  const bool needs_weights = plan.needs_weights;
  if (num_features > max_features_) {
    return fail_invalid(
        "DetectionReductionOp: points feature count exceeds "
        "MaxTensorExtents.max_features");
  }

  // Obtain the runtime compute stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "DetectionReductionOp: no CUDA compute "
                                 "stream"});
  }

  // Discover max label and derive the reduction slot count.
  auto max_label_t = make_device_tensor({1}, kInt64Dtype);
  auto any_cluster_t = make_device_tensor({1}, DLDataType{kDLInt, 32, 1});
  HOLOSCAN_TRY(auto max_label_ptr, writable_ptr(*max_label_t, "max_label"));
  HOLOSCAN_TRY(auto any_cluster_ptr, writable_ptr(*any_cluster_t, "any_cluster"));
  {
    const int64_t zero = 0;
    const int zero_i = 0;
    if (cudaMemcpyAsync(max_label_ptr, &zero, sizeof(int64_t), cudaMemcpyHostToDevice, stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(any_cluster_ptr, &zero_i, sizeof(int), cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
      return fail_runtime("DetectionReductionOp: max-label clear failed");
    }
  }
  try {
    // Execute the device stage on the input runtime stream.
    detail::launch_find_max_label(static_cast<const int64_t*>(labels_in.data()), num_points,
                                  static_cast<int64_t*>(max_label_ptr),
                                  static_cast<int*>(any_cluster_ptr), stream);
    check_cuda("find_max_label", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  int64_t max_label = 0;
  int any_cluster = 0;
  if (cudaMemcpyAsync(&max_label, max_label_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess ||
      cudaMemcpyAsync(&any_cluster, any_cluster_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess) {
    return fail_runtime("DetectionReductionOp: max-label D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: max-label sync failed");
  }
  // Allocate exact-sized outputs before launching device work.
  if (any_cluster == 0) {
    std::optional<TensorOutputLoan> reduced_loan;
    if (reduced_points_output_.active()) {
      HOLOSCAN_TRY(
          auto prepared,
          prepare_device_tensor(reduced_points, context, std::vector<int64_t>{0, num_features},
                                real_dtype<T>(), [](void*, std::size_t) -> expected<void, Error> {
                                  return {};
                                }));
      reduced_loan.emplace(std::move(prepared));
    }
    HOLOSCAN_TRY(auto rep_loan, prepare_device_tensor(
                                    rep_idx, context, std::vector<int64_t>{0, idx_cols},
                                    kInt64Dtype, [](void*, std::size_t) -> expected<void, Error> {
                                      return {};
                                    }));
    if (reduced_loan.has_value()) {
      // Publish coherent outputs in dependency order and retain borrowed inputs.
      reduced_points_output_.emit(reduced_points, context, std::move(*reduced_loan),
                                  "DetectionReductionOp.reduced_points");
    }
    return emit_prepared_tensor(rep_idx, context, std::move(rep_loan));
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
    detail::launch_init_i64(static_cast<int64_t*>(best_rows_ptr), n_slots, detail::kEmptyRow,
                            stream);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  // Materialize the effective feature weights for the selected policy.
  std::shared_ptr<holoscan::Tensor> weights_dev;
  if (needs_weights) {
    std::vector<T> host_w(static_cast<size_t>(num_features));
    for (int64_t f = 0; f < num_features; ++f) {
      host_w[static_cast<size_t>(f)] =
          static_cast<T>(effective_feature_weights[static_cast<size_t>(f)]);
    }
    weights_dev = make_device_tensor({num_features}, real_dtype<T>());
    HOLOSCAN_TRY(auto weights_ptr, writable_ptr(*weights_dev, "weights"));
    if (cudaMemcpyAsync(weights_ptr, host_w.data(), sizeof(T) * host_w.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return fail_runtime("DetectionReductionOp: feature_weights H2D failed");
    }
  }

  std::optional<WeightedMeanWorkspace<T>> weighted_mean_workspace;

  // Select one representative row in each occupied label slot.
  switch (effective_representative) {
    case RepresentativePolicy::kFirst: {
      HOLOSCAN_TRY(run_first_selection(labels_in, num_points, best_rows_ptr, stream));
      break;
    }
    case RepresentativePolicy::kPeak: {
      HOLOSCAN_TRY((run_peak_selection<T, ScoreFn>(
          labels_in, points_ptr,
          needs_weights ? static_cast<const T*>(weights_dev->data()) : nullptr, num_points,
          num_features, n_slots, effective_score_method, best_rows_ptr, stream)));
      break;
    }
    case RepresentativePolicy::kMedoid: {
      HOLOSCAN_TRY((run_medoid_selection<T>(labels_in, points_ptr,
                                            static_cast<const T*>(weights_dev->data()), num_points,
                                            num_features, n_slots, best_rows_ptr, stream)));
      break;
    }
    case RepresentativePolicy::kWeightedMean: {
      const bool uniform_weights = effective_feature_weights.empty();
      HOLOSCAN_TRY(auto workspace,
                   (run_weighted_mean_selection<T, ScoreFn>(
                       labels_in, idx_in, points_ptr,
                       needs_weights ? static_cast<const T*>(weights_dev->data()) : nullptr,
                       uniform_weights, num_points, num_features, idx_cols, n_slots,
                       effective_score_method, best_rows_ptr, stream)));
      weighted_mean_workspace.emplace(std::move(workspace));
      break;
    }
  }

  // Count occupied slots so the required output can be exact-sized.
  auto cluster_count_t = make_device_tensor({1}, kInt64Dtype);
  HOLOSCAN_TRY(auto cluster_count_ptr, writable_ptr(*cluster_count_t, "cluster_count"));
  {
    const int64_t zero = 0;
    if (cudaMemcpyAsync(cluster_count_ptr, &zero, sizeof(int64_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return fail_runtime("DetectionReductionOp: cluster-count clear failed");
    }
  }
  try {
    detail::launch_count_occupied(static_cast<const int64_t*>(best_rows_ptr), n_slots,
                                  static_cast<int64_t*>(cluster_count_ptr), stream);
    check_cuda("count_occupied", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }
  int64_t num_clusters = 0;
  int invalid_weight = 0;
  if (cudaMemcpyAsync(&num_clusters, cluster_count_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: cluster-count D2H failed");
  }
  if (weighted_mean_workspace.has_value() &&
      cudaMemcpyAsync(&invalid_weight, weighted_mean_workspace->invalid_weight->data(), sizeof(int),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: weighted-mean validation D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return fail_runtime("DetectionReductionOp: cluster-count sync failed");
  }
  if (invalid_weight != 0) {
    return fail_invalid(
        "DetectionReductionOp: weighted_mean requires finite non-negative "
        "row weights and a positive finite weight sum per cluster");
  }

  // Prepare the optional feature reduction before the required trigger output.
  std::optional<TensorOutputLoan> reduced_loan;
  if (reduced_points_output_.active()) {
    const bool use_slot_points = weighted_mean_workspace.has_value();
    const T* slot_points = use_slot_points
                               ? static_cast<const T*>(weighted_mean_workspace->point_means->data())
                               : nullptr;
    HOLOSCAN_TRY(auto prepared,
                 prepare_device_tensor(
                     reduced_points, context, std::vector<int64_t>{num_clusters, num_features},
                     real_dtype<T>(),
                     [best_rows_ptr, points_ptr, slot_points, n_slots, num_features, num_clusters,
                      use_slot_points, stream](void* output, std::size_t) -> expected<void, Error> {
                       if (num_clusters == 0) {
                         return {};
                       }
                       try {
                         detail::launch_compact_points<T>(
                             static_cast<const int64_t*>(best_rows_ptr), points_ptr, slot_points,
                             n_slots, num_features, use_slot_points, static_cast<T*>(output),
                             stream);
                         check_cuda("compact_points", kOpName);
                       } catch (const std::runtime_error& e) {
                         return fail_runtime(e.what());
                       }
                       return {};
                     }));
    reduced_loan.emplace(std::move(prepared));
  }

  // Fill and publish the required real-cell output.
  const std::vector<int64_t> rep_shape = {num_clusters, idx_cols};
  HOLOSCAN_TRY(auto rep_loan,
               prepare_device_tensor(rep_idx, context, rep_shape, kInt64Dtype,
                                     [best_rows_ptr, &idx_in, n_slots, idx_cols, num_clusters,
                                      stream](void* output, std::size_t) -> expected<void, Error> {
                                       if (num_clusters == 0) {
                                         return {};
                                       }
                                       try {
                                         detail::launch_compact_and_gather(
                                             static_cast<const int64_t*>(best_rows_ptr),
                                             static_cast<const int64_t*>(idx_in.data()), n_slots,
                                             idx_cols, static_cast<int64_t*>(output), stream);
                                         check_cuda("compact_and_gather", kOpName);
                                       } catch (const std::runtime_error& e) {
                                         return fail_runtime(e.what());
                                       }
                                       return {};
                                     }));

  if (reduced_loan.has_value()) {
    reduced_points_output_.emit(reduced_points, context, std::move(*reduced_loan),
                                "DetectionReductionOp.reduced_points");
  }

  // Retain inputs and borrowed workspace across the async boundary.
  retained_labels_ = std::move(labels_in);
  retained_idx_ = std::move(idx_in);
  retained_points_ = std::move(points_in);
  retained_weights_ = std::move(weights_dev);
  return emit_prepared_tensor(rep_idx, context, std::move(rep_loan));
}

}  // namespace holoscan::radar
