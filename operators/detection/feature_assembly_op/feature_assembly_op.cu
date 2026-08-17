// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/feature_assembly_op/feature_assembly_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "detection/feature_assembly_op/feature_assembly_stage.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "FeatureAssemblyOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};

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

int feature_to_id(AssemblyFeature feature) {
  switch (feature) {
    case AssemblyFeature::kRange:
      return detail::kFeatRange;
    case AssemblyFeature::kVelocity:
      return detail::kFeatVelocity;
    case AssemblyFeature::kScore:
      return detail::kFeatScore;
    case AssemblyFeature::kAz:
      return detail::kFeatAz;
    case AssemblyFeature::kEl:
      return detail::kFeatEl;
    case AssemblyFeature::kAzBin:
      return detail::kFeatAzBin;
    case AssemblyFeature::kElBin:
      return detail::kFeatElBin;
    case AssemblyFeature::kDopplerBin:
      return detail::kFeatDopplerBin;
    case AssemblyFeature::kRangeBin:
      return detail::kFeatRangeBin;
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

bool needs_az_bin(const std::vector<AssemblyFeature>& features) {
  for (auto f : features) {
    if (f == AssemblyFeature::kAzBin) {
      return true;
    }
  }
  return false;
}

bool needs_el_bin(const std::vector<AssemblyFeature>& features) {
  for (auto f : features) {
    if (f == AssemblyFeature::kElBin) {
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
    case detail::kErrAzBin:
      return "FeatureAssemblyOp: az_bin out of range for az_angles";
    case detail::kErrElBin:
      return "FeatureAssemblyOp: el_bin out of range for el_angles";
    default:
      return "FeatureAssemblyOp: assemble kernel reported an error";
  }
}

}  // namespace

template <typename T>
FeatureAssemblyOp<T>::FeatureAssemblyOp(
    const MaxTensorExtents& extents) {
  const char* who = "FeatureAssemblyOp";
  const std::int64_t max_points =
      require_extent(extents.max_points, "max_points", who);
  const std::int64_t max_features =
      require_extent(extents.max_features, "max_features", who);
  extents_ = extents;
  max_byte_span_ = byte_span_real_cube<T>(
      {max_points, max_features}, who);
}

template <typename T>
void FeatureAssemblyOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorAllocationBounds bounds{
      .max_rank = 2U,
      .max_byte_span = max_byte_span_};
  spec.input(detection_idx).queue_depth(2U).expects_tensor(kDevice);
  // Optional gather sources: registered so they may be wired; omitted from
  // contract() so unconnected apps compile (HSDK5_PORT / Beamscan pattern).
  spec.input(range_bins).queue_depth(2U).expects_tensor(kDevice);
  spec.input(velocity_bins).queue_depth(2U).expects_tensor(kDevice);
  spec.input(detection_score).queue_depth(2U).expects_tensor(kDevice);
  spec.input(aoa_estimates).queue_depth(2U).expects_tensor(kDevice);
  spec.input(az_angles).queue_depth(2U).expects_tensor(kDevice);
  spec.input(el_angles).queue_depth(2U).expects_tensor(kDevice);
  spec.output(points).produces_tensor(kDevice).tensor_allocation(bounds);
}

template <typename T>
Contract FeatureAssemblyOp<T>::contract() const {
  // Trigger on the required detection_idx only. Optional gather ports are
  // intentionally omitted so unconnected graphs do not yield
  // TEMPORAL_INPUT_UNCONNECTED (SafeDefault alone is not enough on EA —
  // same pattern as BeamscanEstimatorOp.is_valid). [EA HSDK5 ONLY] see
  // docs/planning/HSDK5_PORT.md and OPEN_ITEMS.md.
  Contract result;
  result.trigger(OnEach{detection_idx});
  return result;
}

template <typename T>
expected<void, Error> FeatureAssemblyOp<T>::compute(
    ExecutionContext& context) {
  const FeatureAssemblyParams& pinned =
      context.params<FeatureAssemblyParams>();
  const std::vector<AssemblyFeature>& features = pinned.features;
  if (features.empty()) {
    return fail_invalid(
        "FeatureAssemblyOp: features must be a non-empty list");
  }
  const int64_t num_features = static_cast<int64_t>(features.size());

  HOLOSCAN_TRY(auto detection_idx_in, receive_tensor(detection_idx, kOpName));
  try {
    require_cuda_accessible(detection_idx_in, "detection_idx", kOpName);
    require_contiguous(detection_idx_in, "detection_idx", kOpName);
    require_dtype(detection_idx_in, kInt64Dtype, "detection_idx", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const auto& idx_shape = detection_idx_in.shape();
  if (idx_shape.size() != 2 || idx_shape[1] < 2) {
    return fail_invalid(
        "FeatureAssemblyOp: detection_idx must have shape "
        "(num_points, 2 [+ n_batch])");
  }
  const int64_t num_points = idx_shape[0];
  if (extents_.max_points.has_value() &&
      num_points > *extents_.max_points) {
    return fail_invalid(
        "FeatureAssemblyOp: detection_idx rows (" +
        std::to_string(num_points) +
        ") exceed MaxTensorExtents max_points (" +
        std::to_string(*extents_.max_points) + ")");
  }
  const int64_t idx_cols = idx_shape[1];
  const int64_t n_preserved = idx_cols - 2;

  HOLOSCAN_TRY(auto maybe_range, receive_optional_value(range_bins));
  HOLOSCAN_TRY(auto maybe_velocity, receive_optional_value(velocity_bins));
  HOLOSCAN_TRY(auto maybe_score, receive_optional_value(detection_score));
  HOLOSCAN_TRY(auto maybe_aoa, receive_optional_value(aoa_estimates));
  HOLOSCAN_TRY(auto maybe_az, receive_optional_value(az_angles));
  HOLOSCAN_TRY(auto maybe_el, receive_optional_value(el_angles));

  const bool have_range = maybe_range.has_value();
  const bool have_velocity = maybe_velocity.has_value();
  const bool have_score = maybe_score.has_value();
  const bool have_aoa = maybe_aoa.has_value();
  const bool have_az = maybe_az.has_value();
  const bool have_el = maybe_el.has_value();

  if (needs_score(features) && !have_score) {
    return fail_invalid(
        std::string("FeatureAssemblyOp: features includes '") +
        std::string(to_string(AssemblyFeature::kScore)) +
        "' but detection_score input is absent");
  }
  if (needs_aoa(features) && !have_aoa) {
    return fail_invalid(
        std::string("FeatureAssemblyOp: features includes '") +
        std::string(to_string(AssemblyFeature::kAz)) + "'/'" +
        std::string(to_string(AssemblyFeature::kEl)) +
        "' but aoa_estimates input is absent");
  }
  if (needs_az_bin(features) && !have_az) {
    return fail_invalid(
        "FeatureAssemblyOp: features includes 'az_bin' but az_angles input is "
        "absent");
  }
  if (needs_el_bin(features) && !have_el) {
    return fail_invalid(
        "FeatureAssemblyOp: features includes 'el_bin' but el_angles input is "
        "absent");
  }
  // Dense az/el bins live in the two preserved-batch columns immediately
  // before (doppler_bin, range_bin), so at least two preserved columns.
  if ((needs_az_bin(features) || needs_el_bin(features)) && n_preserved < 2) {
    return fail_invalid(
        "FeatureAssemblyOp: 'az_bin'/'el_bin' require detection_idx with at "
        "least two preserved-batch columns (…, az_bin, el_bin, doppler_bin, "
        "range_bin)");
  }

  int64_t num_range = 0;
  const T* range_ptr = nullptr;
  if (have_range) {
    try {
      require_cuda_accessible(*maybe_range, "range_bins", kOpName);
      require_contiguous(*maybe_range, "range_bins", kOpName);
      require_dtype(*maybe_range, real_dtype<T>(), "range_bins", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    if (maybe_range->shape().size() != 1) {
      return fail_invalid(
          "FeatureAssemblyOp: range_bins must be rank-1 (num_range,)");
    }
    num_range = maybe_range->shape()[0];
    range_ptr = static_cast<const T*>(maybe_range->data());
  }

  int64_t num_doppler = 0;
  const T* velocity_ptr = nullptr;
  if (have_velocity) {
    try {
      require_cuda_accessible(*maybe_velocity, "velocity_bins", kOpName);
      require_contiguous(*maybe_velocity, "velocity_bins", kOpName);
      require_dtype(*maybe_velocity, real_dtype<T>(), "velocity_bins",
                    kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    if (maybe_velocity->shape().size() != 1) {
      return fail_invalid(
          "FeatureAssemblyOp: velocity_bins must be rank-1 (num_doppler,)");
    }
    num_doppler = maybe_velocity->shape()[0];
    velocity_ptr = static_cast<const T*>(maybe_velocity->data());
  }

  const T* score_ptr = nullptr;
  int64_t num_score_doppler = 0;
  int64_t num_score_range = 0;
  std::vector<int64_t> score_leading;
  int64_t plane_elems = 0;
  if (have_score) {
    try {
      require_cuda_accessible(*maybe_score, "detection_score", kOpName);
      require_contiguous(*maybe_score, "detection_score", kOpName);
      require_dtype(*maybe_score, real_dtype<T>(), "detection_score", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    const auto& sshape = maybe_score->shape();
    if (sshape.size() < 2) {
      return fail_invalid(
          "FeatureAssemblyOp: detection_score must end with "
          "(num_doppler, num_range)");
    }
    num_score_doppler = sshape[sshape.size() - 2];
    num_score_range = sshape[sshape.size() - 1];
    plane_elems = num_score_doppler * num_score_range;
    score_leading.assign(sshape.begin(), sshape.end() - 2);
    score_ptr = static_cast<const T*>(maybe_score->data());
    if (num_doppler == 0) {
      num_doppler = num_score_doppler;
    }
    if (num_range == 0) {
      num_range = num_score_range;
    }
  }

  const T* aoa_ptr = nullptr;
  if (have_aoa) {
    try {
      require_cuda_accessible(*maybe_aoa, "aoa_estimates", kOpName);
      require_contiguous(*maybe_aoa, "aoa_estimates", kOpName);
      require_dtype(*maybe_aoa, real_dtype<T>(), "aoa_estimates", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    const auto& ashape = maybe_aoa->shape();
    if (ashape.size() != 2 || ashape[1] != 2) {
      return fail_invalid(
          "FeatureAssemblyOp: aoa_estimates must have shape (num_points, 2)");
    }
    if (ashape[0] != num_points) {
      return fail_invalid(
          "FeatureAssemblyOp: aoa_estimates row count must match "
          "detection_idx");
    }
    aoa_ptr = static_cast<const T*>(maybe_aoa->data());
  }

  int64_t num_az = 0;
  const T* az_ptr = nullptr;
  if (have_az) {
    try {
      require_cuda_accessible(*maybe_az, "az_angles", kOpName);
      require_contiguous(*maybe_az, "az_angles", kOpName);
      require_dtype(*maybe_az, real_dtype<T>(), "az_angles", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    if (maybe_az->shape().size() != 1) {
      return fail_invalid(
          "FeatureAssemblyOp: az_angles must be rank-1 (num_az,)");
    }
    num_az = maybe_az->shape()[0];
    az_ptr = static_cast<const T*>(maybe_az->data());
  }

  int64_t num_el = 0;
  const T* el_ptr = nullptr;
  if (have_el) {
    try {
      require_cuda_accessible(*maybe_el, "el_angles", kOpName);
      require_contiguous(*maybe_el, "el_angles", kOpName);
      require_dtype(*maybe_el, real_dtype<T>(), "el_angles", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    if (maybe_el->shape().size() != 1) {
      return fail_invalid(
          "FeatureAssemblyOp: el_angles must be rank-1 (num_el,)");
    }
    num_el = maybe_el->shape()[0];
    el_ptr = static_cast<const T*>(maybe_el->data());
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "FeatureAssemblyOp: no CUDA compute stream"});
  }

  const DLDataType out_dtype = real_dtype<T>();
  const std::vector<int64_t> points_shape = {num_points, num_features};

  if (num_points == 0) {
    HOLOSCAN_TRY(emit_device_tensor(
        points, context, points_shape, out_dtype,
        [](void*, std::size_t) -> expected<void, Error> { return {}; }));
    retained_idx_ = std::move(detection_idx_in);
    if (have_range) {
      retained_range_bins_ = std::move(*maybe_range);
    }
    if (have_velocity) {
      retained_velocity_bins_ = std::move(*maybe_velocity);
    }
    if (have_score) {
      retained_score_ = std::move(*maybe_score);
    }
    if (have_aoa) {
      retained_aoa_ = std::move(*maybe_aoa);
    }
    if (have_az) {
      retained_az_angles_ = std::move(*maybe_az);
    }
    if (have_el) {
      retained_el_angles_ = std::move(*maybe_el);
    }
    retained_feature_ids_.reset();
    retained_score_offsets_.reset();
    // Keep seeded_feature_ids_ — empty ticks must not unlock a mid-session
    // features permute.
    return {};
  }

  std::vector<int> feature_ids(static_cast<size_t>(num_features));
  try {
    for (int64_t i = 0; i < num_features; ++i) {
      feature_ids[static_cast<size_t>(i)] =
          feature_to_id(features[static_cast<size_t>(i)]);
    }
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  if (seeded_feature_ids_.has_value() &&
      *seeded_feature_ids_ != feature_ids) {
    return fail_invalid(
        "FeatureAssemblyOp: features is static and cannot change after the "
        "first successful activation; instantiate a separate operator per "
        "column layout");
  }
  // Upload device ids before the assemble kernel, but lock
  // seeded_feature_ids_ only after a successful points emit so a failed
  // first tick does not freeze the static features layout.
  if (!seeded_feature_ids_.has_value() || !retained_feature_ids_ ||
      retained_feature_ids_->shape()[0] != num_features) {
    retained_feature_ids_ = make_device_tensor({num_features}, kInt32Dtype);
    HOLOSCAN_TRY(auto feat_mut,
                 writable_ptr(*retained_feature_ids_, "feature_ids"));
    if (cudaMemcpyAsync(feat_mut, feature_ids.data(),
                        feature_ids.size() * sizeof(int),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return fail_runtime("FeatureAssemblyOp: feature id H2D copy failed");
    }
  }

  if (!retained_error_flag_) {
    retained_error_flag_ = make_device_tensor({1}, kInt32Dtype);
  }
  HOLOSCAN_TRY(auto error_flag_ptr,
               writable_ptr(*retained_error_flag_, "error_flag"));
  if (cudaMemsetAsync(error_flag_ptr, 0, sizeof(int), stream) != cudaSuccess) {
    return fail_runtime("FeatureAssemblyOp: error-flag clear failed");
  }
  int* error_flag = static_cast<int*>(error_flag_ptr);

  std::shared_ptr<holoscan::Tensor> score_offsets_dev;
  const int64_t* score_offsets_ptr = nullptr;
  const bool gather_score = have_score && needs_score(features);
  if (gather_score) {
    if (static_cast<int64_t>(score_leading.size()) != n_preserved) {
      return fail_invalid(
          "FeatureAssemblyOp: detection_score leading rank must match "
          "detection_idx preserved batch columns");
    }
    if (n_preserved > detail::kMaxPreservedBatchDims) {
      return fail_invalid(
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
      HOLOSCAN_TRY(auto lead_mut,
                   writable_ptr(*retained_score_leading_, "score_leading"));
      if (cudaMemcpyAsync(lead_mut, score_leading.data(),
                          score_leading.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return fail_runtime(
            "FeatureAssemblyOp: score leading-dims H2D copy failed");
      }
      cached_score_leading_ = score_leading;
    }
    score_offsets_dev = make_device_tensor({num_points}, kInt64Dtype);
    HOLOSCAN_TRY(auto off_mut,
                 writable_ptr(*score_offsets_dev, "score_offsets"));
    try {
      detail::launch_score_plane_offsets(
          static_cast<const int64_t*>(detection_idx_in.data()), num_points,
          idx_cols, n_preserved,
          static_cast<const int64_t*>(retained_score_leading_->data()),
          plane_elems, static_cast<int64_t*>(off_mut), error_flag, stream);
      check_cuda("score plane offset kernel", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
    score_offsets_ptr = static_cast<const int64_t*>(score_offsets_dev->data());
  }

  auto points_loan = points.allocate_tensor(
      TensorLoanRequest{.shape = points_shape, .dtype = out_dtype});
  if (!points_loan) {
    return make_unexpected(std::move(points_loan).error());
  }
  auto points_writer = points_loan->write(context.cuda_stream());
  if (!points_writer) {
    return make_unexpected(std::move(points_writer).error());
  }
  T* points_ptr = static_cast<T*>(points_writer->data());

  try {
    detail::launch_assemble_features<T>(
        static_cast<const int64_t*>(detection_idx_in.data()), num_points,
        idx_cols, n_preserved, range_ptr, num_range, have_range, velocity_ptr,
        num_doppler, have_velocity, score_ptr, num_score_doppler,
        num_score_range, score_offsets_ptr, gather_score, aoa_ptr, have_aoa,
        az_ptr, num_az, el_ptr, num_el,
        static_cast<const int*>(retained_feature_ids_->data()), num_features,
        points_ptr, error_flag, stream);
    check_cuda("assemble kernel", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  int host_error = 0;
  if (cudaMemcpyAsync(&host_error, error_flag, sizeof(int),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return fail_runtime("FeatureAssemblyOp: error-flag D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return fail_runtime("FeatureAssemblyOp: stream sync failed");
  }
  if (host_error != 0) {
    return fail_invalid(kernel_error_message(host_error));
  }

  auto points_committed = std::move(*points_writer).commit();
  if (!points_committed) {
    return make_unexpected(std::move(points_committed).error());
  }
  auto points_emitted = points.emit(
      std::move(*points_loan),
      EmitOptions{.capture_time = context.activation_time()});
  if (!points_emitted) {
    return make_unexpected(std::move(points_emitted).error());
  }
  if (!seeded_feature_ids_.has_value()) {
    seeded_feature_ids_ = feature_ids;
  }

  retained_idx_ = std::move(detection_idx_in);
  if (have_range) {
    retained_range_bins_ = std::move(*maybe_range);
  }
  if (have_velocity) {
    retained_velocity_bins_ = std::move(*maybe_velocity);
  }
  if (have_score) {
    retained_score_ = std::move(*maybe_score);
  }
  if (have_aoa) {
    retained_aoa_ = std::move(*maybe_aoa);
  }
  if (have_az) {
    retained_az_angles_ = std::move(*maybe_az);
  }
  if (have_el) {
    retained_el_angles_ = std::move(*maybe_el);
  }
  retained_score_offsets_ = std::move(score_offsets_dev);
  return {};
}

template class FeatureAssemblyOp<float>;
template class FeatureAssemblyOp<double>;

}  // namespace holoscan::radar
