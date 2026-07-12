// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CUDA-TU implementation of DetectionReductionOp<T, ScoreFn>.
// Include from a .cu that explicitly instantiates the needed specializations.

#pragma once

#include "detection/detection_reduction_op/detection_reduction_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "detection/detection_reduction_op/detection_reduction_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace reduction_op_detail {

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

inline std::shared_ptr<holoscan::Tensor> receive_single(
    holoscan::InputContext& op_input, const char* port) {
  auto maybe = op_input.receive<holoscan::TensorMap>(port);
  if (!maybe) {
    throw std::runtime_error(
        std::string("DetectionReductionOp: no message on '") + port + "'");
  }
  auto& map = maybe.value();
  if (map.size() != 1) {
    throw std::invalid_argument(
        std::string("DetectionReductionOp: port '") + port +
        "' expects a single-entry TensorMap");
  }
  return map.begin()->second;
}

inline std::shared_ptr<holoscan::Tensor> receive_optional(
    holoscan::InputContext& op_input, const char* port) {
  auto maybe = op_input.receive<holoscan::TensorMap>(port);
  if (!maybe) {
    return nullptr;
  }
  auto& map = maybe.value();
  if (map.empty()) {
    return nullptr;
  }
  if (map.size() != 1) {
    throw std::invalid_argument(
        std::string("DetectionReductionOp: port '") + port +
        "' expects a single-entry TensorMap");
  }
  return map.begin()->second;
}

inline void emit_single(holoscan::OutputContext& op_output, const char* port,
                 const std::shared_ptr<holoscan::Tensor>& tensor) {
  holoscan::TensorMap map;
  map.insert({port, tensor});
  op_output.emit(map, port);
}

inline void require_cuda(const std::shared_ptr<holoscan::Tensor>& tensor,
                  const char* port) {
  if (!tensor || !is_cuda_accessible(*tensor)) {
    throw std::invalid_argument(
        std::string("DetectionReductionOp: '") + port +
        "' must be a CUDA-accessible tensor");
  }
}

inline void require_contiguous(const std::shared_ptr<holoscan::Tensor>& tensor,
                        const char* port) {
  if (!tensor->is_contiguous()) {
    throw std::invalid_argument(
        std::string("DetectionReductionOp: '") + port +
        "' must be contiguous (compact row-major)");
  }
}

inline void require_dtype(const std::shared_ptr<holoscan::Tensor>& tensor,
                   DLDataType expected, const char* port) {
  const auto dt = tensor->dtype();
  if (dt.code != expected.code || dt.bits != expected.bits ||
      dt.lanes != expected.lanes) {
    throw std::invalid_argument(
        std::string("DetectionReductionOp: '") + port +
        "' has unexpected dtype");
  }
}

inline void check_cuda(const char* where) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("DetectionReductionOp: ") + where + ": " +
        cudaGetErrorString(err));
  }
}

}  // namespace reduction_op_detail

using reduction_op_detail::receive_single;
using reduction_op_detail::receive_optional;
using reduction_op_detail::emit_single;
using reduction_op_detail::require_cuda;
using reduction_op_detail::require_contiguous;
using reduction_op_detail::require_dtype;
using reduction_op_detail::check_cuda;
using reduction_op_detail::kInt64Dtype;

template <typename T, typename ScoreFn>
void DetectionReductionOp<T, ScoreFn>::initialize() {
  ensure_radar_arg_setters();
  register_any_argument_setter<RepresentativePolicy>(
      "holoscan::radar::RepresentativePolicy");
  register_any_argument_setter<ScoreMethod>(
      "holoscan::radar::ScoreMethod");
  register_any_argument_setter<std::vector<double>>("std::vector<double>");
  Operator::initialize();
}

template <typename T, typename ScoreFn>
void DetectionReductionOp<T, ScoreFn>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("labels");
  spec.input<holoscan::TensorMap>("detection_idx");
  spec.input<holoscan::TensorMap>("points")
      .condition(holoscan::ConditionType::kNone);

  spec.output<holoscan::TensorMap>("rep_idx");

  spec.param(representative_, "representative", "Representative policy",
             "peak | medoid | first", RepresentativePolicy::kPeak);
  spec.param(score_method_, "score_method", "Peak score method",
             "weighted_sum | weighted_l2 | weighted_max | custom",
             ScoreMethod::kWeightedSum);
  spec.param(feature_weights_, "feature_weights", "Feature weights",
             "Per-feature weights for peak/medoid (required for those "
             "policies unless peak+custom)",
             {});

  add_override_port<RepresentativePolicy>(spec, "representative");
  add_override_port<ScoreMethod>(spec, "score_method");
  add_override_port<std::vector<double>>(spec, "feature_weights");
}

template <typename T, typename ScoreFn>
void DetectionReductionOp<T, ScoreFn>::compute(InputContext& op_input,
                                               OutputContext& op_output,
                                               ExecutionContext&) {
  if (!seeded_) {
    effective_representative_ = representative_.get();
    effective_score_method_ = score_method_.get();
    effective_feature_weights_ = feature_weights_.get();
    seeded_ = true;
  }
  effective_representative_ = resolve_override_param(
      op_input, "representative", effective_representative_, nullptr);
  effective_score_method_ = resolve_override_param(
      op_input, "score_method", effective_score_method_, nullptr);
  effective_feature_weights_ = resolve_override_param(
      op_input, "feature_weights", effective_feature_weights_, nullptr);

  if (effective_representative_ == RepresentativePolicy::kPeak &&
      effective_score_method_ == ScoreMethod::kCustom &&
      std::is_same_v<ScoreFn, NoCustomScore>) {
    throw std::invalid_argument(
        "DetectionReductionOp: score_method=custom requires a compile-time "
        "ScoreFn template argument (DetectionReductionOp<T, ScoreFn>)");
  }

  auto labels = receive_single(op_input, "labels");
  auto detection_idx = receive_single(op_input, "detection_idx");
  auto points = receive_optional(op_input, "points");

  require_cuda(labels, "labels");
  require_contiguous(labels, "labels");
  require_dtype(labels, kInt64Dtype, "labels");
  require_cuda(detection_idx, "detection_idx");
  require_contiguous(detection_idx, "detection_idx");
  require_dtype(detection_idx, kInt64Dtype, "detection_idx");

  if (labels->shape().size() != 1) {
    throw std::invalid_argument(
        "DetectionReductionOp: labels must have shape (num_points,)");
  }
  const auto& idx_shape = detection_idx->shape();
  if (idx_shape.size() != 2 || idx_shape[1] < 2) {
    throw std::invalid_argument(
        "DetectionReductionOp: detection_idx must have shape "
        "(num_points, 2 [+ n_batch])");
  }
  const int64_t num_points = labels->shape()[0];
  if (idx_shape[0] != num_points) {
    throw std::invalid_argument(
        "DetectionReductionOp: labels and detection_idx row counts must "
        "match");
  }
  const int64_t idx_cols = idx_shape[1];

  const bool needs_points =
      effective_representative_ != RepresentativePolicy::kFirst;
  const bool needs_weights =
      effective_representative_ == RepresentativePolicy::kMedoid ||
      (effective_representative_ == RepresentativePolicy::kPeak &&
       effective_score_method_ != ScoreMethod::kCustom);

  if (needs_points && !points) {
    throw std::invalid_argument(
        std::string("DetectionReductionOp: points required for "
                    "representative='") +
        std::string(to_string(effective_representative_)) + "'");
  }

  int64_t num_features = 0;
  const T* points_ptr = nullptr;
  if (points) {
    require_cuda(points, "points");
    require_contiguous(points, "points");
    require_dtype(points, real_dtype<T>(), "points");
    if (points->shape().size() != 2 || points->shape()[0] != num_points) {
      throw std::invalid_argument(
          "DetectionReductionOp: points must have shape "
          "(num_points, num_features)");
    }
    num_features = points->shape()[1];
    points_ptr = static_cast<const T*>(points->data());
  }

  if (needs_weights) {
    if (static_cast<int64_t>(effective_feature_weights_.size()) !=
        num_features) {
      throw std::invalid_argument(
          "DetectionReductionOp: feature_weights length must equal "
          "num_features");
    }
  }

  auto stream = op_input.receive_cuda_stream("labels");

  if (num_points == 0) {
    auto rep = make_device_tensor({0, idx_cols}, kInt64Dtype);
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "rep_idx");
    }
    emit_single(op_output, "rep_idx", rep);
    retained_labels_ = std::move(labels);
    retained_idx_ = std::move(detection_idx);
    retained_points_ = std::move(points);
    retained_weights_.reset();
    return;
  }

  // Discover max label → slot count.
  auto max_label_t = make_device_tensor({1}, kInt64Dtype);
  auto any_cluster_t =
      make_device_tensor({1}, DLDataType{kDLInt, 32, 1});
  {
    const int64_t zero = 0;
    const int zero_i = 0;
    cudaMemcpyAsync(max_label_t->data(), &zero, sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(any_cluster_t->data(), &zero_i, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
  }
  detail::launch_find_max_label(
      static_cast<const int64_t*>(labels->data()), num_points,
      static_cast<int64_t*>(max_label_t->data()),
      static_cast<int*>(any_cluster_t->data()), stream);
  check_cuda("find_max_label");

  int64_t max_label = 0;
  int any_cluster = 0;
  cudaMemcpyAsync(&max_label, max_label_t->data(), sizeof(int64_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&any_cluster, any_cluster_t->data(), sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  if (any_cluster == 0) {
    // All noise.
    auto rep = make_device_tensor({0, idx_cols}, kInt64Dtype);
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "rep_idx");
    }
    emit_single(op_output, "rep_idx", rep);
    retained_labels_ = std::move(labels);
    retained_idx_ = std::move(detection_idx);
    retained_points_ = std::move(points);
    retained_weights_.reset();
    return;
  }
  const int64_t n_slots = max_label + 1;
  constexpr int64_t kMaxSlots = 1 << 20;
  if (n_slots > kMaxSlots) {
    throw std::invalid_argument(
        "DetectionReductionOp: max cluster label exceeds supported range "
        "(expected DBSCAN-style dense labels)");
  }

  auto best_rows = make_device_tensor({n_slots}, kInt64Dtype);
  detail::launch_init_i64(static_cast<int64_t*>(best_rows->data()), n_slots,
                          detail::kEmptyRow, stream);

  std::shared_ptr<holoscan::Tensor> weights_dev;
  if (needs_weights) {
    std::vector<T> host_w(static_cast<size_t>(num_features));
    for (int64_t f = 0; f < num_features; ++f) {
      host_w[static_cast<size_t>(f)] =
          static_cast<T>(effective_feature_weights_[static_cast<size_t>(f)]);
    }
    weights_dev = make_device_tensor({num_features}, real_dtype<T>());
    cudaMemcpyAsync(weights_dev->data(), host_w.data(),
                    sizeof(T) * host_w.size(), cudaMemcpyHostToDevice, stream);
  }

  if (effective_representative_ == RepresentativePolicy::kFirst) {
    detail::launch_select_first(static_cast<const int64_t*>(labels->data()),
                                num_points,
                                static_cast<int64_t*>(best_rows->data()),
                                stream);
  } else if (effective_representative_ == RepresentativePolicy::kPeak) {
    auto best_score =
        make_device_tensor({n_slots}, DLDataType{kDLUInt, 64, 1});
    detail::launch_init_u64(static_cast<detail::cuda_u64*>(best_score->data()),
                            n_slots, 0ull, stream);
    if (effective_score_method_ == ScoreMethod::kCustom) {
      if constexpr (!std::is_same_v<ScoreFn, NoCustomScore>) {
        detail::launch_peak_select_custom<T, ScoreFn>(
            static_cast<const int64_t*>(labels->data()), points_ptr,
            num_points, num_features, ScoreFn{},
            static_cast<detail::cuda_u64*>(best_score->data()),
            static_cast<int64_t*>(best_rows->data()), stream);
      }
    } else {
      detail::launch_peak_select<T>(
          static_cast<const int64_t*>(labels->data()), points_ptr,
          static_cast<const T*>(weights_dev->data()), num_points, num_features,
          effective_score_method_,
          static_cast<detail::cuda_u64*>(best_score->data()),
          static_cast<int64_t*>(best_rows->data()), stream);
    }
  } else {
    // medoid
    auto counts = make_device_tensor({n_slots}, kInt64Dtype);
    auto sum_feat =
        make_device_tensor({n_slots * num_features}, real_dtype<T>());
    cudaMemsetAsync(counts->data(), 0, sizeof(int64_t) * n_slots, stream);
    detail::launch_init_t_zero<T>(static_cast<T*>(sum_feat->data()),
                                  n_slots * num_features, stream);
    auto best_score =
        make_device_tensor({n_slots}, DLDataType{kDLUInt, 64, 1});
    detail::launch_init_u64(static_cast<detail::cuda_u64*>(best_score->data()),
                            n_slots, 0ull, stream);
    detail::launch_medoid_select<T>(
        static_cast<const int64_t*>(labels->data()), points_ptr,
        static_cast<const T*>(weights_dev->data()),
        static_cast<T*>(sum_feat->data()),
        static_cast<int64_t*>(counts->data()), num_points, n_slots,
        num_features, static_cast<detail::cuda_u64*>(best_score->data()),
        static_cast<int64_t*>(best_rows->data()), stream);
  }
  check_cuda("select_representative");

  auto cluster_count_t = make_device_tensor({1}, kInt64Dtype);
  {
    const int64_t zero = 0;
    cudaMemcpyAsync(cluster_count_t->data(), &zero, sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
  }
  detail::launch_count_occupied(static_cast<const int64_t*>(best_rows->data()),
                                n_slots,
                                static_cast<int64_t*>(cluster_count_t->data()),
                                stream);
  check_cuda("count_occupied");
  int64_t num_clusters = 0;
  cudaMemcpyAsync(&num_clusters, cluster_count_t->data(), sizeof(int64_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  auto rep_idx =
      make_device_tensor({num_clusters, idx_cols}, kInt64Dtype);
  if (num_clusters > 0) {
    detail::launch_compact_and_gather(
        static_cast<const int64_t*>(best_rows->data()),
        static_cast<const int64_t*>(detection_idx->data()), n_slots, idx_cols,
        static_cast<int64_t*>(rep_idx->data()), stream);
    check_cuda("compact_and_gather");
  }

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "rep_idx");
  }
  emit_single(op_output, "rep_idx", rep_idx);

  retained_labels_ = std::move(labels);
  retained_idx_ = std::move(detection_idx);
  retained_points_ = std::move(points);
  retained_weights_ = std::move(weights_dev);
}

}  // namespace holoscan::radar
