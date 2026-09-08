// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/logger/logger.hpp>

#include "angle_estimation/esprit_estimator_op/esprit_stage.hpp"
#include "radar_common/dsp/planar_aperture.hpp"
#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

template <typename T>
struct ESPRITRuntimeState {
  bool warned_rank_deficient{false};
  detail::DeviceAngleConvention<T> device_angle_convention{};
  std::optional<AngleConvention> seeded_angle_convention;
  SpatialSmoothingPlan smoothing_plan;
  std::string smoothing_key;
  std::string subarray_key;
  ShiftInvariantSubarrayPlan subarray_plan;
  std::unordered_set<std::string> seen_geometry_keys;
  std::shared_ptr<holoscan::Tensor> d_subarray_index;
  std::shared_ptr<holoscan::Tensor> d_fb_flip;
  std::shared_ptr<holoscan::Tensor> d_reference_indices;
  std::shared_ptr<holoscan::Tensor> d_shifted_indices;
  std::string ura_key;
  int64_t ura_y_pairs{0};
  int64_t ura_z_pairs{0};
  std::shared_ptr<holoscan::Tensor> d_y_reference;
  std::shared_ptr<holoscan::Tensor> d_y_shifted;
  std::shared_ptr<holoscan::Tensor> d_z_reference;
  std::shared_ptr<holoscan::Tensor> d_z_shifted;
  std::shared_ptr<holoscan::Tensor> covariance;
  std::shared_ptr<holoscan::Tensor> eigenvectors;
  std::shared_ptr<holoscan::Tensor> valid_counts;
  std::shared_ptr<holoscan::Tensor> model_order;
  std::shared_ptr<holoscan::Tensor> smoothed_snapshot;
  std::shared_ptr<holoscan::Tensor> smoothed_mask;
  std::size_t smoothed_snapshot_capacity{0U};
  std::size_t smoothed_mask_capacity{0U};
  std::array<std::shared_ptr<holoscan::Tensor>, 6> matrix_workspace;
};

namespace {

constexpr const char* kOpName = "ESPRITEstimatorOp";
constexpr int64_t kMaxSourcesCap = 64;

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

struct ESPRITStageConfig {
  ArrayGeometry geometry;
  AngleConvention convention;
  ShiftInvariantSubarrayPair subarray_spec;
  std::vector<int64_t> spatial_smoothing;
  int64_t num_sources{-1};
  double source_count_pfa{1e-2};
  int64_t max_sources{1};
  bool forward_backward{true};
};

struct ESPRITStagePlan {
  ESPRITStageConfig stage;
  int64_t num_items{0};
  int64_t num_channels{0};
  int64_t num_snapshots{0};
  std::vector<int64_t> aoa_shape;
  std::vector<int64_t> doa_shape;
  std::vector<int64_t> count_shape;
};

template <typename T>
struct ESPRITEstimationInput {
  const ArrayGeometry* geometry{nullptr};
  int64_t num_channels{0};
  int64_t num_snapshots{0};
  const cuda::std::complex<T>* data{nullptr};
  const unsigned char* mask{nullptr};
  std::shared_ptr<holoscan::Tensor> snapshot_keepalive;
  std::shared_ptr<holoscan::Tensor> mask_keepalive;
  bool smoothing_changed{false};
};

expected<ESPRITStageConfig, Error> resolve_params(
    Input<WireArrayGeometry>& geometry_port,
    Input<SpatialSmoothingOverride>& smoothing_port,
    Input<int64_t>& num_sources_port, Input<double>& pfa_port,
    Input<int64_t>& max_sources_port, const ESPRITParams& pinned,
    const MaxTensorExtents& extents) {
  HOLOSCAN_TRY(
      auto geometry,
      resolve_live_override_via(
          geometry_port, pinned.array_geometry,
          [&extents](const WireArrayGeometry& wire)
              -> expected<ArrayGeometry, Error> {
            auto converted = from_wire(wire);
            if (!converted) {
              return make_unexpected(std::move(converted).error());
            }
            if (extents.max_channels.has_value() &&
                converted->num_elements() > *extents.max_channels) {
              return make_unexpected(
                  Error{ErrorCode::kInvalidArgument,
                        "ESPRITEstimatorOp: "
                        "array_geometry.num_elements exceeds "
                        "MaxTensorExtents.max_channels"});
            }
            return std::move(*converted);
          }));
  HOLOSCAN_TRY(
      auto smoothing,
      resolve_live_override_via(
          smoothing_port, pinned.spatial_smoothing,
          [](const SpatialSmoothingOverride& wire) {
            return from_wire(wire);
          }));
  HOLOSCAN_TRY(
      auto resolved_num_sources,
      resolve_live_override(num_sources_port, pinned.num_sources));
  HOLOSCAN_TRY(
      auto pfa,
      resolve_live_override(pfa_port, pinned.source_count_pfa));
  HOLOSCAN_TRY(
      auto resolved_max_sources,
      resolve_live_override(max_sources_port, pinned.max_sources));
  return ESPRITStageConfig{
      .geometry = std::move(geometry),
      .convention = pinned.angle_convention,
      .subarray_spec = pinned.subarray_spec,
      .spatial_smoothing = std::move(smoothing),
      .num_sources = resolved_num_sources,
      .source_count_pfa = pfa,
      .max_sources = resolved_max_sources,
      .forward_backward = pinned.forward_backward,
  };
}

expected<ESPRITStagePlan, Error> build_esprit_stage_plan(
    const Tensor& snapshot, ESPRITStageConfig stage,
    const MaxTensorExtents& extents) {
  try {
    stage.geometry.validate();
    stage.subarray_spec.validate();
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  }
  if (extents.max_channels.has_value() &&
      stage.geometry.num_elements() > *extents.max_channels) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: array_geometry.num_elements exceeds "
              "MaxTensorExtents.max_channels"});
  }
  if (!(stage.source_count_pfa > 0.0) ||
      !(stage.source_count_pfa < 1.0)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: source_count_pfa must be in (0, 1)"});
  }
  if (stage.max_sources < 1 || stage.max_sources > kMaxSourcesCap) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: max_sources must be in [1, 64]"});
  }
  if (extents.max_sources.has_value() &&
      stage.max_sources > *extents.max_sources) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: max_sources exceeds "
              "MaxTensorExtents.max_sources"});
  }
  if (!stage.subarray_spec.is_auto() &&
      !stage.spatial_smoothing.empty()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: explicit subarray_spec cannot be combined "
              "with spatial_smoothing; use subarray_spec=auto on the reduced "
              "aperture"});
  }

  const auto& shape = snapshot.shape();
  if (shape.size() != 3U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: snapshot must have shape "
              "(num_items, num_channels, num_snapshots)"});
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape[2];
  if (extents.max_items.has_value() && num_items > *extents.max_items) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: num_items exceeds "
              "MaxTensorExtents.max_items"});
  }
  if (num_channels != stage.geometry.num_elements()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: snapshot num_channels must match "
              "array_geometry.num_elements"});
  }
  if (num_channels > std::numeric_limits<int32_t>::max()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: num_channels exceeds index capacity"});
  }

  const int64_t max_sources = stage.max_sources;
  return ESPRITStagePlan{
      .stage = std::move(stage),
      .num_items = num_items,
      .num_channels = num_channels,
      .num_snapshots = num_snapshots,
      .aoa_shape = {num_items, 2},
      .doa_shape = {num_items, max_sources, 2},
      .count_shape = {num_items},
  };
}

std::string geometry_key(const ArrayGeometry& geometry) {
  std::ostringstream key;
  key << std::setprecision(std::numeric_limits<double>::max_digits10);
  key << geometry.wavelength << '|'
      << (geometry.geometry_type.has_value()
              ? static_cast<int>(*geometry.geometry_type)
              : -1)
      << '|';
  for (double value : geometry.element_positions) {
    key << value << ',';
  }
  return key.str();
}

std::string pair_key(const ArrayGeometry& geometry,
                     const ShiftInvariantSubarrayPair& pair) {
  std::ostringstream key;
  key << geometry_key(geometry) << '|';
  for (int64_t index : pair.reference_indices) {
    key << index << ',';
  }
  key << '|';
  for (int64_t index : pair.shifted_indices) {
    key << index << ',';
  }
  return key.str();
}

template <typename T>
expected<ESPRITEstimationInput<T>, Error> prepare_smoothed_input(
    ESPRITRuntimeState<T>& state, const Tensor& snapshot,
    const unsigned char* mask, const ESPRITStagePlan& plan,
    cudaStream_t stream) {
  ESPRITEstimationInput<T> result{
      .geometry = &plan.stage.geometry,
      .num_channels = plan.num_channels,
      .num_snapshots = plan.num_snapshots,
      .data =
          static_cast<const cuda::std::complex<T>*>(snapshot.data()),
      .mask = mask,
  };
  if (plan.stage.spatial_smoothing.empty()) {
    return result;
  }

  std::ostringstream key;
  key << geometry_key(plan.stage.geometry) << '|';
  for (int64_t count : plan.stage.spatial_smoothing) {
    key << count << ',';
  }
  key << '|' << (plan.stage.forward_backward ? 1 : 0);
  const std::string candidate_key = key.str();
  if (candidate_key != state.smoothing_key) {
    result.smoothing_changed = !state.smoothing_key.empty();
    try {
      state.smoothing_plan = build_spatial_smoothing_plan(
          plan.stage.geometry, plan.stage.spatial_smoothing,
          plan.stage.forward_backward);
    } catch (const std::invalid_argument& error) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, error.what()});
    } catch (const std::runtime_error& error) {
      return make_unexpected(
          Error{ErrorCode::kFailure, error.what()});
    }
    state.smoothing_key = candidate_key;
    if (state.smoothing_plan.enabled) {
      const int64_t subarrays = state.smoothing_plan.num_subarrays;
      const int64_t subarray_size = state.smoothing_plan.subarray_size;
      state.d_subarray_index =
          make_device_tensor({subarrays * subarray_size},
                             DLDataType{kDLInt, 32, 1});
      HOLOSCAN_TRY(
          auto index_ptr,
          writable_ptr(*state.d_subarray_index, "subarray_index"));
      if (cudaMemcpyAsync(
              index_ptr, state.smoothing_plan.subarray_index.data(),
              state.smoothing_plan.subarray_index.size() * sizeof(int32_t),
              cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return make_unexpected(
            Error{ErrorCode::kFailure,
                  "ESPRITEstimatorOp: smoothing index upload failed"});
      }
      if (state.smoothing_plan.forward_backward) {
        state.d_fb_flip = make_device_tensor(
            {subarray_size}, DLDataType{kDLInt, 32, 1});
        HOLOSCAN_TRY(
            auto flip_ptr,
            writable_ptr(*state.d_fb_flip, "fb_flip"));
        if (cudaMemcpyAsync(
                flip_ptr, state.smoothing_plan.fb_flip.data(),
                state.smoothing_plan.fb_flip.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          return make_unexpected(
              Error{ErrorCode::kFailure,
                    "ESPRITEstimatorOp: smoothing flip upload failed"});
        }
      } else {
        state.d_fb_flip.reset();
      }
    }
  }

  if (!state.smoothing_plan.enabled) {
    return result;
  }
  const int64_t subarray_size = state.smoothing_plan.subarray_size;
  const int64_t looks =
      state.smoothing_plan.looks_per_snapshot() * plan.num_snapshots;
  const std::size_t snapshot_elements = product_extents(
      {plan.num_items, subarray_size, looks}, kOpName);
  if (!state.smoothed_snapshot ||
      snapshot_elements > state.smoothed_snapshot_capacity) {
    state.smoothed_snapshot = make_device_tensor(
        {plan.num_items, subarray_size, looks}, complex_dtype<T>());
    state.smoothed_snapshot_capacity = snapshot_elements;
  }
  result.snapshot_keepalive = state.smoothed_snapshot;
  HOLOSCAN_TRY(
      auto smoothed_ptr,
      writable_ptr(*result.snapshot_keepalive, "ss_snapshot"));
  unsigned char* smoothed_mask_ptr = nullptr;
  if (mask != nullptr) {
    const std::size_t mask_elements =
        product_extents({plan.num_items, looks}, kOpName);
    if (!state.smoothed_mask ||
        mask_elements > state.smoothed_mask_capacity) {
      state.smoothed_mask = make_device_tensor(
          {plan.num_items, looks}, DLDataType{kDLUInt, 8, 1});
      state.smoothed_mask_capacity = mask_elements;
    }
    result.mask_keepalive = state.smoothed_mask;
    HOLOSCAN_TRY(
        auto mask_out,
        writable_ptr(*result.mask_keepalive, "ss_mask"));
    smoothed_mask_ptr = static_cast<unsigned char*>(mask_out);
  }
  try {
    run_spatial_smoothing_gather<T>(
        static_cast<const cuda::std::complex<T>*>(snapshot.data()), mask,
        plan.num_items, plan.num_channels, plan.num_snapshots,
        static_cast<const int32_t*>(state.d_subarray_index->data()),
        state.d_fb_flip
            ? static_cast<const int32_t*>(state.d_fb_flip->data())
            : nullptr,
        state.smoothing_plan.num_subarrays, subarray_size,
        state.smoothing_plan.forward_backward,
        static_cast<cuda::std::complex<T>*>(smoothed_ptr),
        smoothed_mask_ptr, stream);
    check_cuda("spatial smoothing gather", kOpName);
  } catch (const std::runtime_error& error) {
    return make_unexpected(
        Error{ErrorCode::kFailure, error.what()});
  }
  result.geometry = &state.smoothing_plan.reduced_geometry;
  result.num_channels = subarray_size;
  result.num_snapshots = looks;
  result.data =
      static_cast<const cuda::std::complex<T>*>(smoothed_ptr);
  result.mask = smoothed_mask_ptr;
  return result;
}

}  // namespace

template <typename T>
ESPRITEstimatorOp<T>::ESPRITEstimatorOp(
    const MaxTensorExtents& extents) {
  state_ = std::make_shared<ESPRITRuntimeState<T>>();
  extents_ = extents;
  if (!extents.max_items.has_value() &&
      !extents.max_sources.has_value()) {
    return;
  }
  const char* who = "ESPRITEstimatorOp";
  const auto max_items =
      require_extent(extents.max_items, "max_items", who);
  const auto max_sources =
      require_extent(extents.max_sources, "max_sources", who);
  const auto max_channels =
      require_extent(extents.max_channels, "max_channels", who);
  aoa_max_byte_span_ =
      byte_span_real_cube<T>({max_items, 2}, who);
  doa_max_byte_span_ =
      byte_span_real_cube<T>({max_items, max_sources, 2}, who);
  count_max_byte_span_ = byte_span_elems(
      static_cast<std::size_t>(max_items), 4U, who);
  eig_max_byte_span_ =
      byte_span_real_cube<T>({max_items, max_channels}, who);
  state_->covariance = make_device_tensor(
      {max_items, max_channels, max_channels}, complex_dtype<T>());
  state_->eigenvectors = make_device_tensor(
      {max_items, max_channels, max_channels}, complex_dtype<T>());
  state_->valid_counts =
      make_device_tensor({max_items}, DLDataType{kDLInt, 64, 1});
  state_->model_order =
      make_device_tensor({max_items}, DLDataType{kDLInt, 32, 1});
  for (auto& workspace : state_->matrix_workspace) {
    workspace = make_device_tensor(
        {max_items, max_sources, max_sources}, complex_dtype<T>());
  }
}

template <typename T>
void ESPRITEstimatorOp<T>::stop() {
  detail::retire_music_matx_caches();
}

template <typename T>
void ESPRITEstimatorOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr std::uint32_t kMaxRank = 3U;
  spec.input(snapshot).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(is_valid).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(aoa_estimates)
      .produces_tensor(pooled_tensor_output(
          kDevice, kMaxRank, aoa_max_byte_span_));
  // Optional in OPERATOR_API.md — always emitted so unwired consumers do not
  // need Connectivity gating. No angle_spectrum (gridless).
  spec.output(doa_multi)
      .produces_tensor(pooled_tensor_output(
          kDevice, kMaxRank, doa_max_byte_span_));
  spec.output(source_count)
      .produces_tensor(pooled_tensor_output(
          kDevice, kMaxRank, count_max_byte_span_));
  spec.output(eigenvalues)
      .produces_tensor(pooled_tensor_output(
          kDevice, kMaxRank, eig_max_byte_span_));
  // [live] Exact overrides, Params order.
  spec.input(array_geometry).queue_depth(2U);
  spec.input(spatial_smoothing).queue_depth(2U);
  spec.input(num_sources).queue_depth(2U);
  spec.input(source_count_pfa).queue_depth(2U);
  spec.input(max_sources).queue_depth(2U);
}

template <typename T>
Contract ESPRITEstimatorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{snapshot})
      .input(is_valid, Exact{})
      .on_missing(is_valid, SafeDefault{});
  add_live_override_to_contract(result, array_geometry);
  add_live_override_to_contract(result, spatial_smoothing);
  add_live_override_to_contract(result, num_sources);
  add_live_override_to_contract(result, source_count_pfa);
  add_live_override_to_contract(result, max_sources);
  return result;
}

template <typename T>
expected<void, Error> ESPRITEstimatorOp<T>::compute(
    ExecutionContext& context) {
  // Receive and validate the required trigger tensor.
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));
  try {
    require_cuda_accessible(snap, "snapshot", kOpName);
    require_contiguous(snap, "snapshot", kOpName);
    require_dtype(snap, complex_dtype<T>(), "snapshot", kOpName);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  // Resolve this tick's Exact overrides, then validate named axes and derive
  // output shapes in one immutable plan.
  const ESPRITParams& pinned = context.params<ESPRITParams>();
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(array_geometry, spatial_smoothing, num_sources,
                     source_count_pfa, max_sources, pinned, extents_));
  HOLOSCAN_TRY(
      auto plan,
      build_esprit_stage_plan(snap, std::move(stage), extents_));
  auto& geometry = plan.stage.geometry;
  const int64_t effective_num_sources = plan.stage.num_sources;
  const double effective_pfa = plan.stage.source_count_pfa;
  const int64_t effective_max_sources = plan.stage.max_sources;

  if (state_->seeded_angle_convention.has_value() &&
      *state_->seeded_angle_convention != plan.stage.convention) {
    return fail_invalid(
        "ESPRITEstimatorOp: angle_convention is static and cannot change "
        "after the first successful activation; instantiate a separate "
        "operator per convention");
  }
  const ShiftInvariantSubarrayPair& requested_pair =
      plan.stage.subarray_spec;
  const int64_t num_items = plan.num_items;
  const int64_t num_channels = plan.num_channels;
  const int64_t num_snapshots = plan.num_snapshots;

  // Validate the optional same-tick mask against the planned snapshot axes.
  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  const unsigned char* mask_ptr = nullptr;
  if (maybe_valid.has_value()) {
    const DLDataType dt = maybe_valid->dtype();
    const bool ok_dtype = (dt.code == kDLBool || dt.code == kDLUInt) &&
                          dt.bits == 8 && dt.lanes == 1;
    if (!ok_dtype) {
      return fail_invalid(
          "ESPRITEstimatorOp: is_valid must have dtype bool or uint8");
    }
    const auto& vshape = maybe_valid->shape();
    if (vshape.size() != 2 || vshape[0] != num_items ||
        vshape[1] != num_snapshots) {
      return fail_invalid(
          "ESPRITEstimatorOp: is_valid must have shape "
          "(num_items, num_snapshots) matching snapshot");
    }
    try {
      require_cuda_accessible(*maybe_valid, "is_valid", kOpName);
      require_contiguous(*maybe_valid, "is_valid", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    mask_ptr = static_cast<const unsigned char*>(maybe_valid->data());
  }

  // Resolve the runtime stream before touching cached device artifacts.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "ESPRITEstimatorOp: no CUDA compute stream"});
  }

  // Tick-local convention until the first successful emit locks the static
  // seed — geometry / subarray validation below can still fail this tick.
  detail::DeviceAngleConvention<T> tick_convention{};
  const detail::DeviceAngleConvention<T>* convention =
      &state_->device_angle_convention;
  if (!state_->seeded_angle_convention.has_value()) {
    tick_convention =
        detail::make_device_angle_convention<T>(plan.stage.convention);
    convention = &tick_convention;
  }
  auto seed_angle_convention_after_emit = [&]() {
    if (!state_->seeded_angle_convention.has_value()) {
      state_->device_angle_convention = *convention;
      state_->seeded_angle_convention = plan.stage.convention;
    }
  };

  const DLDataType out_dtype = real_dtype<T>();
  const DLDataType cplx_dtype = complex_dtype<T>();
  const DLDataType i32_dtype{kDLInt, 32, 1};
  const DLDataType i64_dtype{kDLInt, 64, 1};
  const auto& aoa_shape = plan.aoa_shape;
  const auto& doa_shape = plan.doa_shape;
  const auto& count_shape = plan.count_shape;

  auto emit_empty = [&]() -> expected<void, Error> {
    // Estimation channel count equals full aperture when smoothing is off
    // (empty items never exercise the smoothing front-end).
    // Prepare every loan before any publish so a late allocate/commit
    // failure cannot leave a sibling port already downstream.
    const std::vector<int64_t> eig_shape = {num_items, num_channels};
    auto noop = [](void*, std::size_t) -> expected<void, Error> {
      return {};
    };
    HOLOSCAN_TRY(auto aoa_loan,
                 prepare_device_tensor(aoa_estimates, context, aoa_shape,
                                      out_dtype, noop));
    HOLOSCAN_TRY(auto doa_loan,
                 prepare_device_tensor(doa_multi, context, doa_shape,
                                      out_dtype, noop));
    HOLOSCAN_TRY(auto cnt_loan,
                 prepare_device_tensor(source_count, context, count_shape,
                                      i32_dtype, noop));
    HOLOSCAN_TRY(auto eig_loan,
                 prepare_device_tensor(eigenvalues, context, eig_shape,
                                      out_dtype, noop));
    HOLOSCAN_TRY(
        emit_prepared_tensor(aoa_estimates, context, std::move(aoa_loan)));
    // doa_multi, source_count and eigenvalues are optional/always
    // (OPERATOR_API.md): all three come from work the primary outputs need
    // anyway, so an unwired sink saves nothing and is not an error.
    (void)emit_prepared_tensor(doa_multi, context, std::move(doa_loan));
    (void)emit_prepared_tensor(source_count, context, std::move(cnt_loan));
    (void)emit_prepared_tensor(eigenvalues, context, std::move(eig_loan));
    return {};
  };

  if (num_items == 0) {
    HOLOSCAN_TRY(emit_empty());
    seed_angle_convention_after_emit();
    retained_snapshot_ = std::move(snap);
    if (maybe_valid.has_value()) {
      retained_is_valid_ = std::move(*maybe_valid);
    }
    return {};
  }

  // Warn once per previously unseen geometry key.
  const std::string geom_key = geometry_key(geometry);
  const bool geometry_is_new =
      state_->seen_geometry_keys.insert(geom_key).second;
  const bool geometry_changed =
      geometry_is_new && state_->seen_geometry_keys.size() > 1;

  // Prepare the effective covariance input: either the original snapshot or
  // gathered overlapping looks on the reduced smoothing aperture.
  HOLOSCAN_TRY(
      auto estimation,
      prepare_smoothed_input(*state_, snap, mask_ptr, plan, stream));
  const ArrayGeometry* estimation_geometry = estimation.geometry;
  const int64_t estimation_channels = estimation.num_channels;
  const int64_t estimation_snapshots = estimation.num_snapshots;
  const cuda::std::complex<T>* estimation_data = estimation.data;
  const unsigned char* estimation_mask = estimation.mask;
  const bool smoothing_changed = estimation.smoothing_changed;

  const std::vector<int64_t> eig_shape = {num_items, estimation_channels};

  // A filled URA with `subarray_spec: auto` takes the native 2-D
  // dual-invariance path; an explicit subarray pair always stays on the 1-D
  // path (and must never trigger planar-aperture analysis, which is defined
  // only for auto derivation).
  PlanarAperturePlan planar{};
  if (requested_pair.is_auto()) {
    try {
      planar = analyze_planar_aperture(*estimation_geometry);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  }
  if (planar.is_ura && requested_pair.is_auto()) {
    const int64_t y_pairs = planar.nz * (planar.ny - 1);
    const int64_t z_pairs = (planar.nz - 1) * planar.ny;
    if (effective_num_sources >= 0) {
      if (effective_num_sources > std::min(y_pairs, z_pairs)) {
        return fail_invalid(
            "ESPRITEstimatorOp: num_sources must not exceed the URA "
            "invariance subarray size");
      }
      if (effective_num_sources >= estimation_channels) {
        return fail_invalid(
            "ESPRITEstimatorOp: num_sources must leave at least one "
            "noise-subspace dimension");
      }
    }

    std::ostringstream ura_key;
    ura_key << geometry_key(*estimation_geometry) << '|' << planar.ny << 'x'
            << planar.nz;
    if (ura_key.str() != state_->ura_key) {
      UraInvariancePairs pairs;
      try {
        pairs = build_ura_dual_invariance(planar.ny, planar.nz);
      } catch (const std::invalid_argument& e) {
        return fail_invalid(e.what());
      } catch (const std::runtime_error& e) {
        return fail_runtime(e.what());
      }
      state_->ura_y_pairs =
          static_cast<int64_t>(pairs.y_reference.size());
      state_->ura_z_pairs =
          static_cast<int64_t>(pairs.z_reference.size());
      const auto upload =
          [&](const std::vector<int64_t>& src,
              std::shared_ptr<holoscan::Tensor>& dst,
              const char* what) -> expected<void, Error> {
        dst = make_device_tensor({static_cast<int64_t>(src.size())},
                                 DLDataType{kDLInt, 32, 1});
        HOLOSCAN_TRY(auto dst_ptr, writable_ptr(*dst, what));
        std::vector<int32_t> host(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
          host[i] = static_cast<int32_t>(src[i]);
        }
        if (cudaMemcpyAsync(dst_ptr, host.data(),
                            host.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          return fail_runtime(
              "ESPRITEstimatorOp: URA invariance index upload failed");
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
          return fail_runtime(
              "ESPRITEstimatorOp: URA invariance index sync failed");
        }
        return {};
      };
      HOLOSCAN_TRY(
          upload(pairs.y_reference, state_->d_y_reference, "y_reference"));
      HOLOSCAN_TRY(
          upload(pairs.y_shifted, state_->d_y_shifted, "y_shifted"));
      HOLOSCAN_TRY(
          upload(pairs.z_reference, state_->d_z_reference, "z_reference"));
      HOLOSCAN_TRY(
          upload(pairs.z_shifted, state_->d_z_shifted, "z_shifted"));
      state_->ura_key = ura_key.str();
    }

    if (geometry_changed) {
      HOLOSCAN_LOG_WARN(
          "ESPRITEstimatorOp: unseen array_geometry override; revalidating "
          "the URA dual-invariance pairs");
    }
    if (smoothing_changed) {
      HOLOSCAN_LOG_WARN(
          "ESPRITEstimatorOp: spatial smoothing override changes the ESPRIT "
          "aperture and derived subarray pairs");
    }

    const int64_t model_order_cap = std::min(
        {effective_max_sources, state_->ura_y_pairs, state_->ura_z_pairs,
         estimation_channels - 1});
    const int64_t workspace_dim = std::max<int64_t>(model_order_cap, 1);
    auto covariance =
        state_->covariance
            ? state_->covariance
            : make_device_tensor(
                  {num_items, estimation_channels, estimation_channels},
                  cplx_dtype);
    auto eigenvectors =
        state_->eigenvectors
            ? state_->eigenvectors
            : make_device_tensor(
                  {num_items, estimation_channels, estimation_channels},
                  cplx_dtype);
    auto valid_counts = state_->valid_counts
                            ? state_->valid_counts
                            : make_device_tensor({num_items}, i64_dtype);
    auto model_order = state_->model_order
                           ? state_->model_order
                           : make_device_tensor({num_items}, i32_dtype);
    std::array<std::shared_ptr<holoscan::Tensor>, 6> local_workspace;
    if (!state_->matrix_workspace[0]) {
      for (auto& workspace : local_workspace) {
        workspace = make_device_tensor(
            {num_items, workspace_dim, workspace_dim}, cplx_dtype);
      }
    }
    const auto& workspace =
        state_->matrix_workspace[0] ? state_->matrix_workspace
                                    : local_workspace;
    const auto& psi_y = workspace[0];
    const auto& psi_z = workspace[1];
    const auto& z_ws = workspace[2];
    const auto& q_ws = workspace[3];
    const auto& r_ws = workspace[4];
    const auto& tmp_ws = workspace[5];

    HOLOSCAN_TRY(auto cov_ptr, writable_ptr(*covariance, "covariance"));
    HOLOSCAN_TRY(auto evec_ptr, writable_ptr(*eigenvectors, "eigenvectors"));
    HOLOSCAN_TRY(auto vc_ptr, writable_ptr(*valid_counts, "valid_counts"));
    HOLOSCAN_TRY(auto mo_ptr, writable_ptr(*model_order, "model_order"));
    HOLOSCAN_TRY(auto psi_y_ptr, writable_ptr(*psi_y, "psi_y"));
    HOLOSCAN_TRY(auto psi_z_ptr, writable_ptr(*psi_z, "psi_z"));
    HOLOSCAN_TRY(auto z_ptr, writable_ptr(*z_ws, "z_ws"));
    HOLOSCAN_TRY(auto q_ptr, writable_ptr(*q_ws, "q_ws"));
    HOLOSCAN_TRY(auto r_ptr, writable_ptr(*r_ws, "r_ws"));
    HOLOSCAN_TRY(auto tmp_ptr, writable_ptr(*tmp_ws, "tmp_ws"));

    auto aoa_loan = aoa_estimates.allocate_tensor(
        TensorLoanRequest{.shape = aoa_shape, .dtype = out_dtype});
    if (!aoa_loan) {
      return make_unexpected(std::move(aoa_loan).error());
    }
    auto doa_loan = doa_multi.allocate_tensor(
        TensorLoanRequest{.shape = doa_shape, .dtype = out_dtype});
    if (!doa_loan) {
      return make_unexpected(std::move(doa_loan).error());
    }
    auto cnt_loan = source_count.allocate_tensor(
        TensorLoanRequest{.shape = count_shape, .dtype = i32_dtype});
    if (!cnt_loan) {
      return make_unexpected(std::move(cnt_loan).error());
    }
    auto eig_loan = eigenvalues.allocate_tensor(
        TensorLoanRequest{.shape = eig_shape, .dtype = out_dtype});
    if (!eig_loan) {
      return make_unexpected(std::move(eig_loan).error());
    }

    auto aoa_writer = aoa_loan->write(context.cuda_stream());
    if (!aoa_writer) {
      return make_unexpected(std::move(aoa_writer).error());
    }
    auto doa_writer = doa_loan->write(context.cuda_stream());
    if (!doa_writer) {
      return make_unexpected(std::move(doa_writer).error());
    }
    auto cnt_writer = cnt_loan->write(context.cuda_stream());
    if (!cnt_writer) {
      return make_unexpected(std::move(cnt_writer).error());
    }
    auto eig_writer = eig_loan->write(context.cuda_stream());
    if (!eig_writer) {
      return make_unexpected(std::move(eig_writer).error());
    }

    const double k = kTwoPi / estimation_geometry->wavelength;
    try {
      detail::run_esprit_2d_stage<T>(
          estimation_data, estimation_mask, num_items, estimation_channels,
          estimation_snapshots, effective_num_sources, effective_pfa,
          model_order_cap,
          static_cast<const int32_t*>(state_->d_y_reference->data()),
          static_cast<const int32_t*>(state_->d_y_shifted->data()),
          state_->ura_y_pairs,
          static_cast<const int32_t*>(state_->d_z_reference->data()),
          static_cast<const int32_t*>(state_->d_z_shifted->data()),
          state_->ura_z_pairs,
          workspace_dim, static_cast<T>(k * planar.dy),
          static_cast<T>(k * planar.dz), static_cast<T>(planar.y_axis[0]),
          static_cast<T>(planar.y_axis[1]), static_cast<T>(planar.y_axis[2]),
          static_cast<T>(planar.z_axis[0]), static_cast<T>(planar.z_axis[1]),
          static_cast<T>(planar.z_axis[2]), static_cast<T>(planar.normal[0]),
          static_cast<T>(planar.normal[1]), static_cast<T>(planar.normal[2]),
          *convention,
          static_cast<cuda::std::complex<T>*>(cov_ptr),
          static_cast<cuda::std::complex<T>*>(evec_ptr),
          static_cast<int64_t*>(vc_ptr),
          static_cast<int32_t*>(mo_ptr),
          static_cast<T*>(eig_writer->data()),
          static_cast<cuda::std::complex<T>*>(psi_y_ptr),
          static_cast<cuda::std::complex<T>*>(psi_z_ptr),
          static_cast<cuda::std::complex<T>*>(z_ptr),
          static_cast<cuda::std::complex<T>*>(q_ptr),
          static_cast<cuda::std::complex<T>*>(r_ptr),
          static_cast<cuda::std::complex<T>*>(tmp_ptr),
          static_cast<T*>(aoa_writer->data()),
          static_cast<T*>(doa_writer->data()), effective_max_sources,
          static_cast<int32_t*>(cnt_writer->data()), stream);
      check_cuda("ESPRIT 2-D stage", kOpName);
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }

    if (effective_num_sources < 0 && mask_ptr == nullptr &&
        estimation_snapshots < estimation_channels &&
        !state_->warned_rank_deficient) {
      HOLOSCAN_LOG_WARN(
          "ESPRITEstimatorOp: num_sources=auto with fewer snapshots than "
          "channels; falling back to one source");
      state_->warned_rank_deficient = true;
    }

    auto aoa_committed = std::move(*aoa_writer).commit();
    if (!aoa_committed) {
      return make_unexpected(std::move(aoa_committed).error());
    }
    auto doa_committed = std::move(*doa_writer).commit();
    if (!doa_committed) {
      return make_unexpected(std::move(doa_committed).error());
    }
    auto cnt_committed = std::move(*cnt_writer).commit();
    if (!cnt_committed) {
      return make_unexpected(std::move(cnt_committed).error());
    }
    auto eig_committed = std::move(*eig_writer).commit();
    if (!eig_committed) {
      return make_unexpected(std::move(eig_committed).error());
    }

    const EmitOptions emit_opts{.capture_time = context.activation_time()};
    auto aoa_emitted = aoa_estimates.emit(std::move(*aoa_loan), emit_opts);
    if (!aoa_emitted) {
      return make_unexpected(std::move(aoa_emitted).error());
    }
    // doa_multi, source_count and eigenvalues are optional/always
    // (OPERATOR_API.md): all three come from work the primary outputs need
    // anyway, so an unwired sink saves nothing and is not an error.
    (void)doa_multi.emit(std::move(*doa_loan), emit_opts);
    (void)source_count.emit(std::move(*cnt_loan), emit_opts);
    (void)eigenvalues.emit(std::move(*eig_loan), emit_opts);

    seed_angle_convention_after_emit();
    retained_snapshot_ = std::move(snap);
    if (maybe_valid.has_value()) {
      retained_is_valid_ = std::move(*maybe_valid);
    }
    return {};
  }

  const std::string current_pair_key =
      pair_key(*estimation_geometry, requested_pair);
  if (current_pair_key != state_->subarray_key) {
    try {
      state_->subarray_plan = build_shift_invariant_subarray_plan(
          *estimation_geometry, requested_pair);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
    state_->subarray_key = current_pair_key;
    const int64_t pair_size = static_cast<int64_t>(
        state_->subarray_plan.pair.reference_indices.size());
    state_->d_reference_indices =
        make_device_tensor({pair_size}, DLDataType{kDLInt, 32, 1});
    state_->d_shifted_indices =
        make_device_tensor({pair_size}, DLDataType{kDLInt, 32, 1});
    std::vector<int32_t> reference(static_cast<size_t>(pair_size));
    std::vector<int32_t> shifted(static_cast<size_t>(pair_size));
    for (int64_t index = 0; index < pair_size; ++index) {
      reference[static_cast<size_t>(index)] = static_cast<int32_t>(
          state_->subarray_plan.pair.reference_indices[
              static_cast<size_t>(index)]);
      shifted[static_cast<size_t>(index)] = static_cast<int32_t>(
          state_->subarray_plan.pair.shifted_indices[
              static_cast<size_t>(index)]);
    }
    HOLOSCAN_TRY(auto ref_ptr,
                 writable_ptr(*state_->d_reference_indices,
                              "reference_indices"));
    HOLOSCAN_TRY(auto sh_ptr,
                 writable_ptr(*state_->d_shifted_indices,
                              "shifted_indices"));
    if (cudaMemcpyAsync(ref_ptr, reference.data(),
                        reference.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(sh_ptr, shifted.data(),
                        shifted.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return fail_runtime(
          "ESPRITEstimatorOp: subarray index upload failed");
    }
  }

  const int64_t pair_size = static_cast<int64_t>(
      state_->subarray_plan.pair.reference_indices.size());
  if (effective_num_sources >= 0 && effective_num_sources > pair_size) {
    return fail_invalid(
        "ESPRITEstimatorOp: num_sources must not exceed the number of "
        "corresponding subarray elements");
  }
  if (effective_num_sources >= estimation_channels) {
    return fail_invalid(
        "ESPRITEstimatorOp: num_sources must leave at least one "
        "noise-subspace dimension");
  }
  if (geometry_changed) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: unseen array_geometry override; "
        "revalidating the shift-invariant subarray pair");
  }
  if (smoothing_changed) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: spatial smoothing override changes the ESPRIT "
        "aperture and derived subarray pair");
  }

  const int64_t model_order_cap =
      std::min(effective_max_sources, pair_size);
  const int64_t workspace_dim = std::max<int64_t>(model_order_cap, 1);
  auto covariance =
      state_->covariance
          ? state_->covariance
          : make_device_tensor(
                {num_items, estimation_channels, estimation_channels},
                cplx_dtype);
  auto eigenvectors =
      state_->eigenvectors
          ? state_->eigenvectors
          : make_device_tensor(
                {num_items, estimation_channels, estimation_channels},
                cplx_dtype);
  auto valid_counts = state_->valid_counts
                          ? state_->valid_counts
                          : make_device_tensor({num_items}, i64_dtype);
  auto model_order = state_->model_order
                         ? state_->model_order
                         : make_device_tensor({num_items}, i32_dtype);
  auto matrix_workspace =
      state_->matrix_workspace[0]
          ? state_->matrix_workspace[0]
          : make_device_tensor(
                {num_items, workspace_dim, workspace_dim}, cplx_dtype);
  auto q_workspace =
      state_->matrix_workspace[1]
          ? state_->matrix_workspace[1]
          : make_device_tensor(
                {num_items, workspace_dim, workspace_dim}, cplx_dtype);
  auto r_workspace =
      state_->matrix_workspace[2]
          ? state_->matrix_workspace[2]
          : make_device_tensor(
                {num_items, workspace_dim, workspace_dim}, cplx_dtype);

  HOLOSCAN_TRY(auto cov_ptr, writable_ptr(*covariance, "covariance"));
  HOLOSCAN_TRY(auto evec_ptr, writable_ptr(*eigenvectors, "eigenvectors"));
  HOLOSCAN_TRY(auto vc_ptr, writable_ptr(*valid_counts, "valid_counts"));
  HOLOSCAN_TRY(auto mo_ptr, writable_ptr(*model_order, "model_order"));
  HOLOSCAN_TRY(auto mat_ptr, writable_ptr(*matrix_workspace, "matrix_ws"));
  HOLOSCAN_TRY(auto q_ptr, writable_ptr(*q_workspace, "q_ws"));
  HOLOSCAN_TRY(auto r_ptr, writable_ptr(*r_workspace, "r_ws"));

  auto aoa_loan = aoa_estimates.allocate_tensor(
      TensorLoanRequest{.shape = aoa_shape, .dtype = out_dtype});
  if (!aoa_loan) {
    return make_unexpected(std::move(aoa_loan).error());
  }
  auto doa_loan = doa_multi.allocate_tensor(
      TensorLoanRequest{.shape = doa_shape, .dtype = out_dtype});
  if (!doa_loan) {
    return make_unexpected(std::move(doa_loan).error());
  }
  auto cnt_loan = source_count.allocate_tensor(
      TensorLoanRequest{.shape = count_shape, .dtype = i32_dtype});
  if (!cnt_loan) {
    return make_unexpected(std::move(cnt_loan).error());
  }
  auto eig_loan = eigenvalues.allocate_tensor(
      TensorLoanRequest{.shape = eig_shape, .dtype = out_dtype});
  if (!eig_loan) {
    return make_unexpected(std::move(eig_loan).error());
  }

  auto aoa_writer = aoa_loan->write(context.cuda_stream());
  if (!aoa_writer) {
    return make_unexpected(std::move(aoa_writer).error());
  }
  auto doa_writer = doa_loan->write(context.cuda_stream());
  if (!doa_writer) {
    return make_unexpected(std::move(doa_writer).error());
  }
  auto cnt_writer = cnt_loan->write(context.cuda_stream());
  if (!cnt_writer) {
    return make_unexpected(std::move(cnt_writer).error());
  }
  auto eig_writer = eig_loan->write(context.cuda_stream());
  if (!eig_writer) {
    return make_unexpected(std::move(eig_writer).error());
  }

  try {
    detail::run_esprit_stage<T>(
        estimation_data, estimation_mask, num_items, estimation_channels,
        estimation_snapshots, effective_num_sources, effective_pfa,
        model_order_cap,
        static_cast<const int32_t*>(state_->d_reference_indices->data()),
        static_cast<const int32_t*>(state_->d_shifted_indices->data()),
        pair_size,
        workspace_dim,
        static_cast<T>(kTwoPi * state_->subarray_plan.spacing /
                       estimation_geometry->wavelength),
        static_cast<T>(state_->subarray_plan.axis[0]),
        static_cast<T>(state_->subarray_plan.axis[1]),
        static_cast<T>(state_->subarray_plan.axis[2]),
        static_cast<T>(state_->subarray_plan.broadside[0]),
        static_cast<T>(state_->subarray_plan.broadside[1]),
        static_cast<T>(state_->subarray_plan.broadside[2]),
        *convention,
        static_cast<cuda::std::complex<T>*>(cov_ptr),
        static_cast<cuda::std::complex<T>*>(evec_ptr),
        static_cast<int64_t*>(vc_ptr),
        static_cast<int32_t*>(mo_ptr),
        static_cast<T*>(eig_writer->data()),
        static_cast<cuda::std::complex<T>*>(mat_ptr),
        static_cast<cuda::std::complex<T>*>(q_ptr),
        static_cast<cuda::std::complex<T>*>(r_ptr),
        static_cast<T*>(aoa_writer->data()),
        static_cast<T*>(doa_writer->data()), effective_max_sources,
        static_cast<int32_t*>(cnt_writer->data()), stream);
    check_cuda("ESPRIT stage", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  if (effective_num_sources < 0 && mask_ptr == nullptr &&
      estimation_snapshots < estimation_channels &&
      !state_->warned_rank_deficient) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: num_sources=auto with fewer snapshots than "
        "channels; falling back to one source");
    state_->warned_rank_deficient = true;
  }

  auto aoa_committed = std::move(*aoa_writer).commit();
  if (!aoa_committed) {
    return make_unexpected(std::move(aoa_committed).error());
  }
  auto doa_committed = std::move(*doa_writer).commit();
  if (!doa_committed) {
    return make_unexpected(std::move(doa_committed).error());
  }
  auto cnt_committed = std::move(*cnt_writer).commit();
  if (!cnt_committed) {
    return make_unexpected(std::move(cnt_committed).error());
  }
  auto eig_committed = std::move(*eig_writer).commit();
  if (!eig_committed) {
    return make_unexpected(std::move(eig_committed).error());
  }

  const EmitOptions emit_opts{.capture_time = context.activation_time()};
  auto aoa_emitted = aoa_estimates.emit(std::move(*aoa_loan), emit_opts);
  if (!aoa_emitted) {
    return make_unexpected(std::move(aoa_emitted).error());
  }
  // doa_multi, source_count and eigenvalues are optional/always
  // (OPERATOR_API.md): all three come from work the primary outputs need
  // anyway, so an unwired sink saves nothing and is not an error.
  (void)doa_multi.emit(std::move(*doa_loan), emit_opts);
  (void)source_count.emit(std::move(*cnt_loan), emit_opts);
  (void)eigenvalues.emit(std::move(*eig_loan), emit_opts);

  seed_angle_convention_after_emit();
  retained_snapshot_ = std::move(snap);
  if (maybe_valid.has_value()) {
    retained_is_valid_ = std::move(*maybe_valid);
  }
  return {};
}

template class ESPRITEstimatorOp<float>;
template class ESPRITEstimatorOp<double>;

}  // namespace holoscan::radar
