// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/feature_assembly_op/feature_assembly_op.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
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

expected<void*, Error> writable_ptr(holoscan::Tensor& tensor, const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              std::string(kOpName) + ": " + what + " is not writable: " + mut.error().message});
  }
  return *mut;
}

expected<void, Error> require_device_dtype(const Tensor& tensor, const char* port,
                                           DLDataType dtype) {
  try {
    require_cuda_accessible(tensor, port, kOpName);
    require_contiguous(tensor, port, kOpName);
    require_dtype(tensor, dtype, port, kOpName);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }
  return {};
}

template <typename U>
expected<void, Error> copy_table(const std::shared_ptr<Tensor>& dst, std::size_t capacity,
                                 const std::vector<U>& host, const char* what,
                                 cudaStream_t stream) {
  if (host.size() > capacity) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": " + what + " length exceeds ctor-sized table"});
  }
  if (host.empty()) {
    return {};
  }
  auto mut = dst->mutable_data();
  if (!mut) {
    return make_unexpected(
        Error{ErrorCode::kFailure, std::string(kOpName) + ": " + what + " not writable"});
  }
  if (cudaMemcpyAsync(*mut, host.data(), host.size() * sizeof(U), cudaMemcpyHostToDevice, stream) !=
      cudaSuccess) {
    return make_unexpected(
        Error{ErrorCode::kFailure, std::string(kOpName) + ": " + what + " H2D failed"});
  }
  return {};
}

expected<int, Error> feature_to_id(AssemblyFeature feature) {
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
    case AssemblyFeature::kAzEstimate:
      return detail::kFeatAzEstimate;
    case AssemblyFeature::kElEstimate:
      return detail::kFeatElEstimate;
    case AssemblyFeature::kAzBin:
      return detail::kFeatAzBin;
    case AssemblyFeature::kElBin:
      return detail::kFeatElBin;
    case AssemblyFeature::kDopplerBin:
      return detail::kFeatDopplerBin;
    case AssemblyFeature::kRangeBin:
      return detail::kFeatRangeBin;
  }
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument, std::string(kOpName) + ": unknown AssemblyFeature '" +
                                             std::string(to_string(feature)) + "'"});
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
      return "FeatureAssemblyOp: az_bin out of range for az_grid";
    case detail::kErrElBin:
      return "FeatureAssemblyOp: el_bin out of range for el_grid";
    case detail::kErrRangeSource:
      return "FeatureAssemblyOp: range feature requires range_bins";
    case detail::kErrVelocitySource:
      return "FeatureAssemblyOp: velocity feature requires velocity_bins";
    default:
      return "FeatureAssemblyOp: assemble kernel reported an error";
  }
}

// Optional data-plane gathers for this tick. Not [live] overrides: presence
// follows from the static features list (and from what the app wired).
struct OptionalGathers {
  const Tensor* range_bins = nullptr;
  const Tensor* velocity_bins = nullptr;
  const Tensor* detection_score = nullptr;
  const Tensor* aoa_estimates = nullptr;
  const Tensor* az_grid = nullptr;
  const Tensor* el_grid = nullptr;
};

struct RequiredSources {
  bool range{false};
  bool velocity{false};
  bool score{false};
  bool aoa{false};
  bool az_grid{false};
  bool el_grid{false};
};

RequiredSources required_sources(const std::vector<AssemblyFeature>& features) {
  RequiredSources result;
  for (const AssemblyFeature feature : features) {
    result.range = result.range || feature == AssemblyFeature::kRange;
    result.velocity = result.velocity || feature == AssemblyFeature::kVelocity;
    result.score = result.score || feature == AssemblyFeature::kScore;
    result.aoa = result.aoa || feature == AssemblyFeature::kAzEstimate ||
                 feature == AssemblyFeature::kElEstimate;
    result.az_grid = result.az_grid || feature == AssemblyFeature::kAz;
    result.el_grid = result.el_grid || feature == AssemblyFeature::kEl;
  }
  return result;
}

// Stage knobs (the static features list) plus this tick's parsed layout
// and emit geometry. There is no resolve_params: no [live] ports, and the
// empty-features sentinel is rejected here, not mapped to an optional.
struct AssemblyStagePlan {
  std::vector<int> feature_ids;
  detail::DetectionIdxLayout layout;
  int64_t num_features{0};
  bool have_range{false};
  bool have_velocity{false};
  bool have_score{false};
  bool have_aoa{false};
  bool have_az{false};
  bool have_el{false};
  bool gather_score{false};
  int64_t num_range{0};
  int64_t num_doppler{0};
  int64_t num_az{0};
  int64_t num_el{0};
  int64_t num_score_doppler{0};
  int64_t num_score_range{0};
  int64_t plane_elems{0};
  std::vector<int64_t> score_leading;
  std::vector<int64_t> out_shape;
};

template <typename T>
expected<int64_t, Error> parse_rank1_real(const Tensor& tensor, const char* port,
                                          const char* rank_msg) {
  HOLOSCAN_TRY(require_device_dtype(tensor, port, real_dtype<T>()));
  if (tensor.shape().size() != 1) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(kOpName) + ": " + rank_msg});
  }
  return tensor.shape()[0];
}

template <typename T>
expected<AssemblyStagePlan, Error> build_assembly_stage_plan(
    const std::vector<AssemblyFeature>& features, const Tensor& detection_idx,
    const OptionalGathers& gathers, const MaxTensorExtents& extents) {
  AssemblyStagePlan plan;
  if (features.empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": features must be a non-empty list"});
  }
  plan.num_features = static_cast<int64_t>(features.size());
  plan.feature_ids.resize(static_cast<size_t>(plan.num_features));
  const RequiredSources required = required_sources(features);
  bool needs_dense_coordinates = false;
  for (int64_t i = 0; i < plan.num_features; ++i) {
    const AssemblyFeature token = features[static_cast<size_t>(i)];
    HOLOSCAN_TRY(plan.feature_ids[static_cast<size_t>(i)], feature_to_id(token));
    needs_dense_coordinates = needs_dense_coordinates || token == AssemblyFeature::kAz ||
                              token == AssemblyFeature::kEl || token == AssemblyFeature::kAzBin ||
                              token == AssemblyFeature::kElBin;
  }

  try {
    plan.layout = detail::parse_detection_idx_layout(detection_idx, "detection_idx");
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }

  if (plan.layout.num_points > *extents.max_points) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": detection_idx rows (" + std::to_string(plan.layout.num_points) +
            ") exceed MaxTensorExtents max_points (" + std::to_string(*extents.max_points) + ")"});
  }
  if (plan.num_features > *extents.max_features) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(kOpName) + ": features size (" +
                                               std::to_string(plan.num_features) +
                                               ") exceeds MaxTensorExtents max_features (" +
                                               std::to_string(*extents.max_features) + ")"});
  }

  plan.have_range = gathers.range_bins != nullptr;
  plan.have_velocity = gathers.velocity_bins != nullptr;
  plan.have_score = gathers.detection_score != nullptr;
  plan.have_aoa = gathers.aoa_estimates != nullptr;
  plan.have_az = gathers.az_grid != nullptr;
  plan.have_el = gathers.el_grid != nullptr;
  plan.gather_score = plan.have_score && required.score;

  if (required.range && !plan.have_range) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": features includes 'range' but range_bins input is absent"});
  }
  if (required.velocity && !plan.have_velocity) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) +
                                     ": features includes 'velocity' but velocity_bins input is "
                                     "absent"});
  }
  if (required.score && !plan.have_score) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(kOpName) + ": features includes '" +
                                               std::string(to_string(AssemblyFeature::kScore)) +
                                               "' but detection_score input is absent"});
  }
  if (required.aoa && !plan.have_aoa) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": features includes '" +
                                     std::string(to_string(AssemblyFeature::kAzEstimate)) + "'/'" +
                                     std::string(to_string(AssemblyFeature::kElEstimate)) +
                                     "' but aoa_estimates input is absent"});
  }
  if (required.az_grid && !plan.have_az) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": features includes 'az' but az_grid input is absent"});
  }
  if (required.el_grid && !plan.have_el) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": features includes 'el' but el_grid input is absent"});
  }
  if (needs_dense_coordinates && plan.layout.n_preserved < 2) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": dense angle features require detection_idx with at least two "
                               "preserved-batch columns (…, az_bin, el_bin, doppler_bin, "
                               "range_bin)"});
  }

  if (plan.have_range) {
    HOLOSCAN_TRY(plan.num_range, parse_rank1_real<T>(*gathers.range_bins, "range_bins",
                                                     "range_bins must be rank-1 "
                                                     "(num_range,)"));
  }
  if (plan.have_velocity) {
    HOLOSCAN_TRY(plan.num_doppler, parse_rank1_real<T>(*gathers.velocity_bins, "velocity_bins",
                                                       "velocity_bins must be rank-1 "
                                                       "(num_doppler,)"));
  }
  if (plan.have_score) {
    HOLOSCAN_TRY(
        require_device_dtype(*gathers.detection_score, "detection_score", real_dtype<T>()));
    const auto& sshape = gathers.detection_score->shape();
    if (sshape.size() < 2) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                std::string(kOpName) + ": detection_score must end with (num_doppler, num_range)"});
    }
    plan.num_score_doppler = sshape[sshape.size() - 2];
    plan.num_score_range = sshape[sshape.size() - 1];
    plan.plane_elems = plan.num_score_doppler * plan.num_score_range;
    plan.score_leading.assign(sshape.begin(), sshape.end() - 2);
    if (plan.num_doppler == 0) {
      plan.num_doppler = plan.num_score_doppler;
    }
    if (plan.num_range == 0) {
      plan.num_range = plan.num_score_range;
    }
  }
  if (plan.gather_score) {
    if (static_cast<int64_t>(plan.score_leading.size()) != plan.layout.n_preserved) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, std::string(kOpName) +
                                                 ": detection_score leading rank must match "
                                                 "detection_idx preserved batch columns"});
    }
    if (plan.layout.n_preserved > detail::kMaxPreservedBatchDims) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                std::string(kOpName) + ": too many preserved batch dims in detection_idx"});
    }
  }
  if (plan.have_aoa) {
    HOLOSCAN_TRY(require_device_dtype(*gathers.aoa_estimates, "aoa_estimates", real_dtype<T>()));
    const auto& ashape = gathers.aoa_estimates->shape();
    if (ashape.size() != 2 || ashape[1] != 2) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                std::string(kOpName) + ": aoa_estimates must have shape (num_points, 2)"});
    }
    if (ashape[0] != plan.layout.num_points) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                std::string(kOpName) + ": aoa_estimates row count must match detection_idx"});
    }
  }
  if (plan.have_az) {
    HOLOSCAN_TRY(plan.num_az, parse_rank1_real<T>(*gathers.az_grid, "az_grid",
                                                  "az_grid must be rank-1 (num_az,)"));
  }
  if (plan.have_el) {
    HOLOSCAN_TRY(plan.num_el, parse_rank1_real<T>(*gathers.el_grid, "el_grid",
                                                  "el_grid must be rank-1 (num_el,)"));
  }

  plan.out_shape = {plan.layout.num_points, plan.num_features};
  return plan;
}

}  // namespace

template <typename T>
FeatureAssemblyOp<T>::FeatureAssemblyOp(const MaxTensorExtents& extents,
                                        std::vector<AssemblyFeature> features)
: features_(std::move(features)) {
  const char* who = kOpName;
  const std::int64_t max_points = require_extent(extents.max_points, "max_points", who);
  const std::int64_t max_features = require_extent(extents.max_features, "max_features", who);
  extents_ = extents;
  max_byte_span_ = byte_span_real_cube<T>({max_points, max_features}, who);

  feature_ids_capacity_ = static_cast<std::size_t>(max_features);
  score_offsets_capacity_ = static_cast<std::size_t>(max_points);
  score_leading_capacity_ = static_cast<std::size_t>(detail::kMaxPreservedBatchDims);
  feature_ids_ = make_device_tensor({max_features}, kInt32Dtype);
  score_offsets_ = make_device_tensor({max_points}, kInt64Dtype);
  score_leading_ = make_device_tensor({detail::kMaxPreservedBatchDims}, kInt64Dtype);
  error_flag_ = make_device_tensor({1}, kInt32Dtype);
}

template <typename T>
void FeatureAssemblyOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorBounds bounds{.max_rank = 2U, .max_byte_span = max_byte_span_};
  const RequiredSources required = required_sources(features_);

  // Data plane: trigger tensor, optional gathers, required output.
  spec.input(detection_idx).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  auto& range_bins_spec = spec.input(range_bins).queue_depth(2U);
  if (!required.range) {
    range_bins_spec.optional();
  }
  range_bins_spec.expects_tensor(tensor_input(kDevice));
  auto& velocity_bins_spec = spec.input(velocity_bins).queue_depth(2U);
  if (!required.velocity) {
    velocity_bins_spec.optional();
  }
  velocity_bins_spec.expects_tensor(tensor_input(kDevice));
  auto& detection_score_spec = spec.input(detection_score).queue_depth(2U);
  if (!required.score) {
    detection_score_spec.optional();
  }
  detection_score_spec.expects_tensor(tensor_input(kDevice));
  auto& aoa_estimates_spec = spec.input(aoa_estimates).queue_depth(2U);
  if (!required.aoa) {
    aoa_estimates_spec.optional();
  }
  aoa_estimates_spec.expects_tensor(tensor_input(kDevice));
  auto& az_grid_spec = spec.input(az_grid).queue_depth(2U);
  if (!required.az_grid) {
    az_grid_spec.optional();
  }
  az_grid_spec.expects_tensor(tensor_input(kDevice));
  auto& el_grid_spec = spec.input(el_grid).queue_depth(2U);
  if (!required.el_grid) {
    el_grid_spec.optional();
  }
  el_grid_spec.expects_tensor(tensor_input(kDevice));
  spec.output(points).produces_tensor(pooled_tensor_output(kDevice, bounds));
  // No [live] Exact overrides.
}

template <typename T>
Contract FeatureAssemblyOp<T>::contract() const {
  // Static features defines both the output columns and the compiled join.
  // setup() remains shape-stable: all ports are always registered.
  const RequiredSources required = required_sources(features_);
  std::vector<PortRef> aligned{detection_idx};
  if (required.range) {
    aligned.emplace_back(range_bins);
  }
  if (required.velocity) {
    aligned.emplace_back(velocity_bins);
  }
  if (required.score) {
    aligned.emplace_back(detection_score);
  }
  if (required.aoa) {
    aligned.emplace_back(aoa_estimates);
  }
  if (required.az_grid) {
    aligned.emplace_back(az_grid);
  }
  if (required.el_grid) {
    aligned.emplace_back(el_grid);
  }

  Contract result;
  if (aligned.size() == 1U) {
    result.trigger(OnEach{detection_idx});
  } else {
    result.trigger(
        OnAligned{.ports = PortList{std::move(aligned)}, .window = std::chrono::nanoseconds{0}});
    result.input(detection_idx, Exact{});
  }

  const auto add_source = [&result](const Input<Tensor>& port, bool is_required) {
    result.input(port, Exact{});
    if (!is_required) {
      result.on_missing(port, SafeDefault{});
    }
  };
  add_source(range_bins, required.range);
  add_source(velocity_bins, required.velocity);
  add_source(detection_score, required.score);
  add_source(aoa_estimates, required.aoa);
  add_source(az_grid, required.az_grid);
  add_source(el_grid, required.el_grid);
  return result;
}

template <typename T>
expected<void, Error> FeatureAssemblyOp<T>::ensure_device_feature_ids(
    const std::vector<int>& feature_ids, cudaStream_t stream) {
  if (seeded_feature_ids_.has_value()) {
    if (*seeded_feature_ids_ != feature_ids) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   std::string(kOpName) +
                                       ": features is static and cannot change after the "
                                       "first successful activation; instantiate a separate "
                                       "operator per column layout"});
    }
    return {};
  }
  return copy_table(feature_ids_, feature_ids_capacity_, feature_ids, "feature_ids", stream);
}

template <typename T>
expected<void, Error> FeatureAssemblyOp<T>::ensure_score_leading(
    const std::vector<int64_t>& score_leading, cudaStream_t stream) {
  if (cached_score_leading_ == score_leading) {
    return {};
  }
  HOLOSCAN_TRY(
      copy_table(score_leading_, score_leading_capacity_, score_leading, "score_leading", stream));
  cached_score_leading_ = score_leading;
  return {};
}

template <typename T>
expected<void, Error> FeatureAssemblyOp<T>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  // Receive detection_idx; require a device-contiguous int64 table.
  HOLOSCAN_TRY(auto detection_idx_in, receive_tensor(detection_idx, kOpName));
  try {
    require_cuda_accessible(detection_idx_in, "detection_idx", kOpName);
    require_contiguous(detection_idx_in, "detection_idx", kOpName);
    require_dtype(detection_idx_in, kInt64Dtype, "detection_idx", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Optional data-plane gathers. Not a [live] merge.
  HOLOSCAN_TRY(auto maybe_range, receive_optional_value(range_bins));
  HOLOSCAN_TRY(auto maybe_velocity, receive_optional_value(velocity_bins));
  HOLOSCAN_TRY(auto maybe_score, receive_optional_value(detection_score));
  HOLOSCAN_TRY(auto maybe_aoa, receive_optional_value(aoa_estimates));
  HOLOSCAN_TRY(auto maybe_az, receive_optional_value(az_grid));
  HOLOSCAN_TRY(auto maybe_el, receive_optional_value(el_grid));
  const OptionalGathers gathers{.range_bins = maybe_range ? &*maybe_range : nullptr,
                                .velocity_bins = maybe_velocity ? &*maybe_velocity : nullptr,
                                .detection_score = maybe_score ? &*maybe_score : nullptr,
                                .aoa_estimates = maybe_aoa ? &*maybe_aoa : nullptr,
                                .az_grid = maybe_az ? &*maybe_az : nullptr,
                                .el_grid = maybe_el ? &*maybe_el : nullptr};

  // Trailing rank is (num_points, 2 [+ n_preserved]), not a
  // (chirps, samples) ADC cube — layout parse lives in the plan builder.
  // No [live] merge.
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(auto plan,
               build_assembly_stage_plan<T>(features_, detection_idx_in, gathers, extents_));

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, std::string(kOpName) + ": no CUDA compute stream"});
  }

  const bool nonempty = plan.layout.num_points > 0;
  if (nonempty) {
    HOLOSCAN_TRY(ensure_device_feature_ids(plan.feature_ids, stream));
    if (plan.gather_score && plan.layout.n_preserved > 0) {
      HOLOSCAN_TRY(ensure_score_leading(plan.score_leading, stream));
    }
  }

  const T* range_ptr = plan.have_range ? static_cast<const T*>(maybe_range->data()) : nullptr;
  const T* velocity_ptr =
      plan.have_velocity ? static_cast<const T*>(maybe_velocity->data()) : nullptr;
  const T* score_ptr = plan.have_score ? static_cast<const T*>(maybe_score->data()) : nullptr;
  const T* aoa_ptr = plan.have_aoa ? static_cast<const T*>(maybe_aoa->data()) : nullptr;
  const T* az_ptr = plan.have_az ? static_cast<const T*>(maybe_az->data()) : nullptr;
  const T* el_ptr = plan.have_el ? static_cast<const T*>(maybe_el->data()) : nullptr;

  // Required points: assemble, then publish. prepare + emit_prepared_tensor
  // (not emit_device_tensor) so the error-flag D2H can reject the tick
  // before the loan is published. fill stays stream-ordered.
  // Allocate exact-sized outputs before launching device work.
  HOLOSCAN_TRY(auto points_loan,
               prepare_device_tensor(
                   points, context, plan.out_shape, real_dtype<T>(),
                   [&](void* out_ptr, std::size_t) -> expected<void, Error> {
                     if (!nonempty) {
                       return {};
                     }
                     HOLOSCAN_TRY(auto error_flag_ptr, writable_ptr(*error_flag_, "error_flag"));
                     if (cudaMemsetAsync(error_flag_ptr, 0, sizeof(int), stream) != cudaSuccess) {
                       return fail_runtime(std::string(kOpName) + ": error-flag clear failed");
                     }
                     int* error_flag = static_cast<int*>(error_flag_ptr);

                     const int64_t* score_offsets_ptr = nullptr;
                     if (plan.gather_score && plan.layout.n_preserved > 0) {
                       HOLOSCAN_TRY(auto off_mut, writable_ptr(*score_offsets_, "score_offsets"));
                       try {
                         // Execute the device stage on the input runtime stream.
                         detail::launch_score_plane_offsets(
                             static_cast<const int64_t*>(detection_idx_in.data()),
                             plan.layout.num_points, plan.layout.idx_cols, plan.layout.n_preserved,
                             static_cast<const int64_t*>(score_leading_->data()), plan.plane_elems,
                             static_cast<int64_t*>(off_mut), error_flag, stream);
                         check_cuda("score plane offset kernel", kOpName);
                       } catch (const std::runtime_error& e) {
                         return fail_runtime(e.what());
                       }
                       score_offsets_ptr = static_cast<const int64_t*>(score_offsets_->data());
                     }

                     try {
                       detail::launch_assemble_features<T>(
                           static_cast<const int64_t*>(detection_idx_in.data()),
                           plan.layout.num_points, plan.layout.idx_cols, plan.layout.n_preserved,
                           range_ptr, plan.num_range, plan.have_range, velocity_ptr,
                           plan.num_doppler, plan.have_velocity, score_ptr, plan.num_score_doppler,
                           plan.num_score_range, score_offsets_ptr, plan.gather_score, aoa_ptr,
                           plan.have_aoa, az_ptr, plan.num_az, el_ptr, plan.num_el,
                           static_cast<const int*>(feature_ids_->data()), plan.num_features,
                           static_cast<T*>(out_ptr), error_flag, stream);
                       check_cuda("assemble kernel", kOpName);
                     } catch (const std::runtime_error& e) {
                       return fail_runtime(e.what());
                     }
                     return {};
                   }));

  if (nonempty) {
    int host_error = 0;
    if (cudaMemcpyAsync(&host_error, error_flag_->data(), sizeof(int), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
      return fail_runtime(std::string(kOpName) + ": error-flag D2H failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return fail_runtime(std::string(kOpName) + ": stream sync failed");
    }
    if (host_error != 0) {
      return fail_invalid(kernel_error_message(host_error));
    }
  }

  // Publish coherent outputs in dependency order and retain borrowed inputs.
  HOLOSCAN_TRY(emit_prepared_tensor(points, context, std::move(points_loan)));
  if (nonempty && !seeded_feature_ids_.has_value()) {
    seeded_feature_ids_ = plan.feature_ids;
  }

  // Keep received tensors alive until stream-ordered reads complete.
  retained_idx_ = std::move(detection_idx_in);
  if (plan.have_range) {
    retained_range_bins_ = std::move(*maybe_range);
  }
  if (plan.have_velocity) {
    retained_velocity_bins_ = std::move(*maybe_velocity);
  }
  if (plan.have_score) {
    retained_score_ = std::move(*maybe_score);
  }
  if (plan.have_aoa) {
    retained_aoa_ = std::move(*maybe_aoa);
  }
  if (plan.have_az) {
    retained_az_grid_ = std::move(*maybe_az);
  }
  if (plan.have_el) {
    retained_el_grid_ = std::move(*maybe_el);
  }
  return {};
}

template class FeatureAssemblyOp<float>;
template class FeatureAssemblyOp<double>;

}  // namespace holoscan::radar
