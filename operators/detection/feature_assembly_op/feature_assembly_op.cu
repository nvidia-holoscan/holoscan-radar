// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/feature_assembly_op/feature_assembly_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "detection/feature_assembly_op/feature_assembly_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};

std::shared_ptr<holoscan::Tensor> receive_single(
    holoscan::InputContext& op_input, const char* port) {
  auto maybe = op_input.receive<holoscan::TensorMap>(port);
  if (!maybe) {
    throw std::runtime_error(
        std::string("FeatureAssemblyOp: no message on '") + port + "'");
  }
  auto& map = maybe.value();
  if (map.size() != 1) {
    throw std::invalid_argument(
        std::string("FeatureAssemblyOp: port '") + port +
        "' expects a single-entry TensorMap");
  }
  return map.begin()->second;
}

std::shared_ptr<holoscan::Tensor> receive_optional(
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
        std::string("FeatureAssemblyOp: port '") + port +
        "' expects a single-entry TensorMap");
  }
  return map.begin()->second;
}

void emit_single(holoscan::OutputContext& op_output, const char* port,
                 const std::shared_ptr<holoscan::Tensor>& tensor) {
  holoscan::TensorMap map;
  map.insert({port, tensor});
  op_output.emit(map, port);
}

void require_cuda(const std::shared_ptr<holoscan::Tensor>& tensor,
                  const char* port) {
  if (!tensor || !is_cuda_accessible(*tensor)) {
    throw std::invalid_argument(
        std::string("FeatureAssemblyOp: '") + port +
        "' must be a CUDA-accessible tensor");
  }
}

void require_contiguous(const std::shared_ptr<holoscan::Tensor>& tensor,
                        const char* port) {
  if (!tensor->is_contiguous()) {
    throw std::invalid_argument(
        std::string("FeatureAssemblyOp: '") + port +
        "' must be contiguous (compact row-major)");
  }
}

void require_dtype(const std::shared_ptr<holoscan::Tensor>& tensor,
                   DLDataType expected, const char* port) {
  const auto dt = tensor->dtype();
  if (dt.code != expected.code || dt.bits != expected.bits ||
      dt.lanes != expected.lanes) {
    throw std::invalid_argument(
        std::string("FeatureAssemblyOp: '") + port +
        "' has unexpected dtype");
  }
}

int feature_to_id(AssemblyFeature feature) {
  switch (feature) {
    case AssemblyFeature::kRange: return detail::kFeatRange;
    case AssemblyFeature::kVelocity: return detail::kFeatVelocity;
    case AssemblyFeature::kScore: return detail::kFeatScore;
    case AssemblyFeature::kAz: return detail::kFeatAz;
    case AssemblyFeature::kEl: return detail::kFeatEl;
    case AssemblyFeature::kDopplerBin: return detail::kFeatDopplerBin;
    case AssemblyFeature::kRangeBin: return detail::kFeatRangeBin;
  }
  throw std::invalid_argument(
      std::string("FeatureAssemblyOp: unknown AssemblyFeature '") +
      std::string(to_string(feature)) + "'");
}

bool needs_score(const std::vector<AssemblyFeature>& features) {
  for (auto f : features) {
    if (f == AssemblyFeature::kScore) {
      return true;
    }
  }
  return false;
}

bool needs_aoa(const std::vector<AssemblyFeature>& features) {
  for (auto f : features) {
    if (f == AssemblyFeature::kAz || f == AssemblyFeature::kEl) {
      return true;
    }
  }
  return false;
}

const char* kernel_error_message(int code) {
  switch (code) {
    case detail::kErrBatchCoord:
      return "FeatureAssemblyOp: detection_idx preserved-batch coordinate "
             "out of range for detection_score leading dims";
    case detail::kErrRangeBin:
      return "FeatureAssemblyOp: range_bin out of range for range_bins";
    case detail::kErrDopplerBin:
      return "FeatureAssemblyOp: doppler_bin out of range for velocity_bins";
    case detail::kErrScoreCoord:
      return "FeatureAssemblyOp: (doppler_bin, range_bin) out of range for "
             "detection_score";
    default:
      return "FeatureAssemblyOp: assemble kernel reported an error";
  }
}

}  // namespace

template <typename T>
void FeatureAssemblyOp<T>::initialize() {
  ensure_radar_arg_setters();
  register_any_argument_setter<std::vector<AssemblyFeature>>(
      "std::vector<AssemblyFeature>");
  Operator::initialize();
}

template <typename T>
void FeatureAssemblyOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("detection_idx");
  spec.input<holoscan::TensorMap>("range_bins")
      .condition(holoscan::ConditionType::kNone);
  spec.input<holoscan::TensorMap>("velocity_bins")
      .condition(holoscan::ConditionType::kNone);
  spec.input<holoscan::TensorMap>("detection_score")
      .condition(holoscan::ConditionType::kNone);
  spec.input<holoscan::TensorMap>("aoa_estimates")
      .condition(holoscan::ConditionType::kNone);

  spec.output<holoscan::TensorMap>("points");

  // No default — required at compose time (contract: Default: none).
  spec.param(features_, "features", "Feature columns",
             "Ordered AssemblyFeature tokens selecting points columns "
             "(static; defines output layout)");
}

template <typename T>
void FeatureAssemblyOp<T>::compute(InputContext& op_input,
                                   OutputContext& op_output,
                                   ExecutionContext&) {
  const auto& features = features_.get();
  if (features.empty()) {
    throw std::invalid_argument(
        "FeatureAssemblyOp: features must be a non-empty list");
  }
  const int64_t num_features = static_cast<int64_t>(features.size());

  auto detection_idx = receive_single(op_input, "detection_idx");
  require_cuda(detection_idx, "detection_idx");
  require_contiguous(detection_idx, "detection_idx");
  require_dtype(detection_idx, kInt64Dtype, "detection_idx");

  const auto& idx_shape = detection_idx->shape();
  if (idx_shape.size() != 2 || idx_shape[1] < 2) {
    throw std::invalid_argument(
        "FeatureAssemblyOp: detection_idx must have shape "
        "(num_points, 2 [+ n_batch])");
  }
  const int64_t num_points = idx_shape[0];
  const int64_t idx_cols = idx_shape[1];
  const int64_t n_preserved = idx_cols - 2;

  auto range_bins = receive_optional(op_input, "range_bins");
  auto velocity_bins = receive_optional(op_input, "velocity_bins");
  auto detection_score = receive_optional(op_input, "detection_score");
  auto aoa_estimates = receive_optional(op_input, "aoa_estimates");

  if (needs_score(features) && !detection_score) {
    throw std::invalid_argument(
        std::string("FeatureAssemblyOp: features includes '") +
        std::string(to_string(AssemblyFeature::kScore)) +
        "' but detection_score input is absent");
  }
  if (needs_aoa(features) && !aoa_estimates) {
    throw std::invalid_argument(
        std::string("FeatureAssemblyOp: features includes '") +
        std::string(to_string(AssemblyFeature::kAz)) + "'/'" +
        std::string(to_string(AssemblyFeature::kEl)) +
        "' but aoa_estimates input is absent");
  }

  int64_t num_range = 0;
  const T* range_ptr = nullptr;
  if (range_bins) {
    require_cuda(range_bins, "range_bins");
    require_contiguous(range_bins, "range_bins");
    require_dtype(range_bins, real_dtype<T>(), "range_bins");
    if (range_bins->shape().size() != 1) {
      throw std::invalid_argument(
          "FeatureAssemblyOp: range_bins must be rank-1 (num_range,)");
    }
    num_range = range_bins->shape()[0];
    range_ptr = static_cast<const T*>(range_bins->data());
  }

  int64_t num_doppler = 0;
  const T* velocity_ptr = nullptr;
  if (velocity_bins) {
    require_cuda(velocity_bins, "velocity_bins");
    require_contiguous(velocity_bins, "velocity_bins");
    require_dtype(velocity_bins, real_dtype<T>(), "velocity_bins");
    if (velocity_bins->shape().size() != 1) {
      throw std::invalid_argument(
          "FeatureAssemblyOp: velocity_bins must be rank-1 (num_doppler,)");
    }
    num_doppler = velocity_bins->shape()[0];
    velocity_ptr = static_cast<const T*>(velocity_bins->data());
  }

  const T* score_ptr = nullptr;
  int64_t num_score_doppler = 0;
  int64_t num_score_range = 0;
  std::vector<int64_t> score_leading;
  int64_t plane_elems = 0;
  if (detection_score) {
    require_cuda(detection_score, "detection_score");
    require_contiguous(detection_score, "detection_score");
    require_dtype(detection_score, real_dtype<T>(), "detection_score");
    const auto& sshape = detection_score->shape();
    if (sshape.size() < 2) {
      throw std::invalid_argument(
          "FeatureAssemblyOp: detection_score must end with "
          "(num_doppler, num_range)");
    }
    num_score_doppler = sshape[sshape.size() - 2];
    num_score_range = sshape[sshape.size() - 1];
    plane_elems = num_score_doppler * num_score_range;
    score_leading.assign(sshape.begin(), sshape.end() - 2);
    score_ptr = static_cast<const T*>(detection_score->data());
    if (num_doppler == 0) {
      num_doppler = num_score_doppler;
    }
    if (num_range == 0) {
      num_range = num_score_range;
    }
  }

  const T* aoa_ptr = nullptr;
  if (aoa_estimates) {
    require_cuda(aoa_estimates, "aoa_estimates");
    require_contiguous(aoa_estimates, "aoa_estimates");
    require_dtype(aoa_estimates, real_dtype<T>(), "aoa_estimates");
    const auto& ashape = aoa_estimates->shape();
    if (ashape.size() != 2 || ashape[1] != 2) {
      throw std::invalid_argument(
          "FeatureAssemblyOp: aoa_estimates must have shape (num_points, 2)");
    }
    if (ashape[0] != num_points) {
      throw std::invalid_argument(
          "FeatureAssemblyOp: aoa_estimates row count must match "
          "detection_idx");
    }
    aoa_ptr = static_cast<const T*>(aoa_estimates->data());
  }

  auto stream = op_input.receive_cuda_stream("detection_idx");

  auto points = make_device_tensor({num_points, num_features}, real_dtype<T>());
  if (num_points == 0) {
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "points");
    }
    emit_single(op_output, "points", points);
    retained_idx_ = std::move(detection_idx);
    retained_range_bins_ = std::move(range_bins);
    retained_velocity_bins_ = std::move(velocity_bins);
    retained_score_ = std::move(detection_score);
    retained_aoa_ = std::move(aoa_estimates);
    retained_feature_ids_.reset();
    retained_score_offsets_.reset();
    return;
  }

  std::vector<int> feature_ids(static_cast<size_t>(num_features));
  for (int64_t i = 0; i < num_features; ++i) {
    feature_ids[static_cast<size_t>(i)] =
        feature_to_id(features[static_cast<size_t>(i)]);
  }
  if (!feature_ids_ready_ || !retained_feature_ids_ ||
      retained_feature_ids_->shape()[0] != num_features) {
    retained_feature_ids_ = make_device_tensor({num_features}, kInt32Dtype);
    if (cudaMemcpyAsync(retained_feature_ids_->data(), feature_ids.data(),
                        feature_ids.size() * sizeof(int),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error(
          "FeatureAssemblyOp: feature id H2D copy failed");
    }
    feature_ids_ready_ = true;
  }

  if (!retained_error_flag_) {
    retained_error_flag_ = make_device_tensor({1}, kInt32Dtype);
  }
  if (cudaMemsetAsync(retained_error_flag_->data(), 0, sizeof(int), stream) !=
      cudaSuccess) {
    throw std::runtime_error("FeatureAssemblyOp: error-flag clear failed");
  }
  int* error_flag = static_cast<int*>(retained_error_flag_->data());

  std::shared_ptr<holoscan::Tensor> score_offsets_dev;
  const int64_t* score_offsets_ptr = nullptr;
  const bool gather_score =
      detection_score != nullptr && needs_score(features);
  if (gather_score) {
    if (static_cast<int64_t>(score_leading.size()) != n_preserved) {
      throw std::invalid_argument(
          "FeatureAssemblyOp: detection_score leading rank must match "
          "detection_idx preserved batch columns");
    }
    if (n_preserved > detail::kMaxPreservedBatchDims) {
      throw std::invalid_argument(
          "FeatureAssemblyOp: too many preserved batch dims in "
          "detection_idx");
    }
  }
  if (gather_score && n_preserved > 0) {
    const bool leading_changed = cached_score_leading_ != score_leading;
    if (leading_changed || !retained_score_leading_ ||
        retained_score_leading_->shape()[0] != n_preserved) {
      retained_score_leading_ =
          make_device_tensor({n_preserved}, kInt64Dtype);
      if (cudaMemcpyAsync(retained_score_leading_->data(), score_leading.data(),
                          score_leading.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error(
            "FeatureAssemblyOp: score leading-dims H2D copy failed");
      }
      cached_score_leading_ = score_leading;
    }
    score_offsets_dev = make_device_tensor({num_points}, kInt64Dtype);
    detail::launch_score_plane_offsets(
        static_cast<const int64_t*>(detection_idx->data()), num_points,
        idx_cols, n_preserved,
        static_cast<const int64_t*>(retained_score_leading_->data()),
        plane_elems, static_cast<int64_t*>(score_offsets_dev->data()),
        error_flag, stream);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error(
          "FeatureAssemblyOp: score plane offset kernel failed");
    }
    score_offsets_ptr =
        static_cast<const int64_t*>(score_offsets_dev->data());
  }

  detail::launch_assemble_features<T>(
      static_cast<const int64_t*>(detection_idx->data()), num_points, idx_cols,
      n_preserved, range_ptr, num_range, range_bins != nullptr, velocity_ptr,
      num_doppler, velocity_bins != nullptr, score_ptr, num_score_doppler,
      num_score_range, score_offsets_ptr, gather_score, aoa_ptr,
      aoa_estimates != nullptr,
      static_cast<const int*>(retained_feature_ids_->data()), num_features,
      static_cast<T*>(points->data()), error_flag, stream);

  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("FeatureAssemblyOp: assemble kernel failed");
  }

  int host_error = 0;
  if (cudaMemcpyAsync(&host_error, retained_error_flag_->data(), sizeof(int),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error("FeatureAssemblyOp: error-flag D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("FeatureAssemblyOp: stream sync failed");
  }
  if (host_error != 0) {
    throw std::invalid_argument(kernel_error_message(host_error));
  }

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "points");
  }
  emit_single(op_output, "points", points);

  retained_idx_ = std::move(detection_idx);
  retained_range_bins_ = std::move(range_bins);
  retained_velocity_bins_ = std::move(velocity_bins);
  retained_score_ = std::move(detection_score);
  retained_aoa_ = std::move(aoa_estimates);
  retained_score_offsets_ = std::move(score_offsets_dev);
}

template class FeatureAssemblyOp<float>;
template class FeatureAssemblyOp<double>;

}  // namespace holoscan::radar
