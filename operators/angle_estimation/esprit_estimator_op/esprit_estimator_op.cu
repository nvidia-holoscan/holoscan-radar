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
#include <type_traits>
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

using ESPRITTensorLoanWriter = std::remove_cvref_t<decltype(
    *std::declval<TensorOutputLoan&>().write(
        std::declval<ExecutionContext&>().cuda_stream()))>;

struct ESPRITPreparedOutputs {
  TensorOutputLoan aoa_loan;
  TensorOutputLoan doa_loan;
  TensorOutputLoan count_loan;
  TensorOutputLoan eigenvalue_loan;
  ESPRITTensorLoanWriter aoa_writer;
  ESPRITTensorLoanWriter doa_writer;
  ESPRITTensorLoanWriter count_writer;
  ESPRITTensorLoanWriter eigenvalue_writer;
  std::vector<std::shared_ptr<holoscan::Tensor>> workspace_keepalive;
};

template <typename T>
struct ESPRITPreparedCommonWorkspace {
  std::shared_ptr<holoscan::Tensor> covariance;
  std::shared_ptr<holoscan::Tensor> eigenvectors;
  std::shared_ptr<holoscan::Tensor> valid_counts;
  std::shared_ptr<holoscan::Tensor> model_order;
  cuda::std::complex<T>* covariance_ptr{nullptr};
  cuda::std::complex<T>* eigenvectors_ptr{nullptr};
  int64_t* valid_counts_ptr{nullptr};
  int32_t* model_order_ptr{nullptr};
};

expected<void, Error> emit_empty_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& doa_port,
    Output<Tensor>& count_port, Output<Tensor>& eigenvalue_port,
    ExecutionContext& context, const ESPRITStagePlan& plan,
    DLDataType output_dtype) {
  const std::vector<int64_t> eigenvalue_shape = {
      plan.num_items, plan.num_channels};
  const DLDataType i32_dtype{kDLInt, 32, 1};
  auto noop = [](void*, std::size_t) -> expected<void, Error> {
    return {};
  };
  HOLOSCAN_TRY(
      auto aoa_loan,
      prepare_device_tensor(aoa_port, context, plan.aoa_shape,
                            output_dtype, noop));
  HOLOSCAN_TRY(
      auto doa_loan,
      prepare_device_tensor(doa_port, context, plan.doa_shape,
                            output_dtype, noop));
  HOLOSCAN_TRY(
      auto count_loan,
      prepare_device_tensor(count_port, context, plan.count_shape,
                            i32_dtype, noop));
  HOLOSCAN_TRY(
      auto eigenvalue_loan,
      prepare_device_tensor(eigenvalue_port, context, eigenvalue_shape,
                            output_dtype, noop));
  HOLOSCAN_TRY(
      emit_prepared_tensor(aoa_port, context, std::move(aoa_loan)));
  (void)emit_prepared_tensor(doa_port, context, std::move(doa_loan));
  (void)emit_prepared_tensor(count_port, context, std::move(count_loan));
  (void)emit_prepared_tensor(
      eigenvalue_port, context, std::move(eigenvalue_loan));
  return {};
}

template <typename T>
expected<ESPRITPreparedCommonWorkspace<T>, Error>
prepare_esprit_common_workspace(ESPRITRuntimeState<T>& state,
                                int64_t num_items,
                                int64_t estimation_channels) {
  const DLDataType cplx_dtype = complex_dtype<T>();
  const DLDataType i32_dtype{kDLInt, 32, 1};
  const DLDataType i64_dtype{kDLInt, 64, 1};
  ESPRITPreparedCommonWorkspace<T> result;
  result.covariance =
      state.covariance
          ? state.covariance
          : make_device_tensor(
                {num_items, estimation_channels, estimation_channels},
                cplx_dtype);
  result.eigenvectors =
      state.eigenvectors
          ? state.eigenvectors
          : make_device_tensor(
                {num_items, estimation_channels, estimation_channels},
                cplx_dtype);
  result.valid_counts =
      state.valid_counts
          ? state.valid_counts
          : make_device_tensor({num_items}, i64_dtype);
  result.model_order =
      state.model_order
          ? state.model_order
          : make_device_tensor({num_items}, i32_dtype);
  HOLOSCAN_TRY(
      auto covariance,
      writable_ptr(*result.covariance, "covariance"));
  HOLOSCAN_TRY(
      auto eigenvectors,
      writable_ptr(*result.eigenvectors, "eigenvectors"));
  HOLOSCAN_TRY(
      auto valid_counts,
      writable_ptr(*result.valid_counts, "valid_counts"));
  HOLOSCAN_TRY(
      auto model_order,
      writable_ptr(*result.model_order, "model_order"));
  result.covariance_ptr =
      static_cast<cuda::std::complex<T>*>(covariance);
  result.eigenvectors_ptr =
      static_cast<cuda::std::complex<T>*>(eigenvectors);
  result.valid_counts_ptr = static_cast<int64_t*>(valid_counts);
  result.model_order_ptr = static_cast<int32_t*>(model_order);
  return result;
}

expected<ESPRITPreparedOutputs, Error> prepare_esprit_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& doa_port,
    Output<Tensor>& count_port, Output<Tensor>& eigenvalue_port,
    ExecutionContext& context, const ESPRITStagePlan& plan,
    int64_t estimation_channels, DLDataType output_dtype) {
  const std::vector<int64_t> eigenvalue_shape = {
      plan.num_items, estimation_channels};
  const DLDataType i32_dtype{kDLInt, 32, 1};
  HOLOSCAN_TRY(
      auto aoa_loan,
      aoa_port.allocate_tensor(
          TensorLoanRequest{.shape = plan.aoa_shape,
                            .dtype = output_dtype}));
  HOLOSCAN_TRY(
      auto doa_loan,
      doa_port.allocate_tensor(
          TensorLoanRequest{.shape = plan.doa_shape,
                            .dtype = output_dtype}));
  HOLOSCAN_TRY(
      auto count_loan,
      count_port.allocate_tensor(
          TensorLoanRequest{.shape = plan.count_shape,
                            .dtype = i32_dtype}));
  HOLOSCAN_TRY(
      auto eigenvalue_loan,
      eigenvalue_port.allocate_tensor(
          TensorLoanRequest{.shape = eigenvalue_shape,
                            .dtype = output_dtype}));
  HOLOSCAN_TRY(auto aoa_writer,
               aoa_loan.write(context.cuda_stream()));
  HOLOSCAN_TRY(auto doa_writer,
               doa_loan.write(context.cuda_stream()));
  HOLOSCAN_TRY(auto count_writer,
               count_loan.write(context.cuda_stream()));
  HOLOSCAN_TRY(auto eigenvalue_writer,
               eigenvalue_loan.write(context.cuda_stream()));
  return ESPRITPreparedOutputs{
      .aoa_loan = std::move(aoa_loan),
      .doa_loan = std::move(doa_loan),
      .count_loan = std::move(count_loan),
      .eigenvalue_loan = std::move(eigenvalue_loan),
      .aoa_writer = std::move(aoa_writer),
      .doa_writer = std::move(doa_writer),
      .count_writer = std::move(count_writer),
      .eigenvalue_writer = std::move(eigenvalue_writer),
  };
}

expected<void, Error> commit_and_publish_esprit_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& doa_port,
    Output<Tensor>& count_port, Output<Tensor>& eigenvalue_port,
    ExecutionContext& context, ESPRITPreparedOutputs outputs) {
  HOLOSCAN_TRY(std::move(outputs.aoa_writer).commit());
  HOLOSCAN_TRY(std::move(outputs.doa_writer).commit());
  HOLOSCAN_TRY(std::move(outputs.count_writer).commit());
  HOLOSCAN_TRY(std::move(outputs.eigenvalue_writer).commit());

  const EmitOptions emit_options{
      .capture_time = context.activation_time()};
  HOLOSCAN_TRY(
      aoa_port.emit(std::move(outputs.aoa_loan), emit_options));
  (void)doa_port.emit(std::move(outputs.doa_loan), emit_options);
  (void)count_port.emit(std::move(outputs.count_loan), emit_options);
  (void)eigenvalue_port.emit(
      std::move(outputs.eigenvalue_loan), emit_options);
  return {};
}

template <typename T>
expected<ESPRITPreparedOutputs, Error> execute_esprit_auto_ura(
    ESPRITRuntimeState<T>& state, const ESPRITStagePlan& plan,
    const ESPRITEstimationInput<T>& estimation,
    const PlanarAperturePlan& planar,
    const detail::DeviceAngleConvention<T>& convention,
    bool geometry_changed, Output<Tensor>& aoa_port,
    Output<Tensor>& doa_port, Output<Tensor>& count_port,
    Output<Tensor>& eigenvalue_port, ExecutionContext& context,
    cudaStream_t stream) {
  const int64_t y_pairs = planar.nz * (planar.ny - 1);
  const int64_t z_pairs = (planar.nz - 1) * planar.ny;
  if (plan.stage.num_sources >= 0) {
    if (plan.stage.num_sources > std::min(y_pairs, z_pairs)) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "ESPRITEstimatorOp: num_sources must not exceed the URA "
          "invariance subarray size"});
    }
    if (plan.stage.num_sources >= estimation.num_channels) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "ESPRITEstimatorOp: num_sources must leave at least one "
          "noise-subspace dimension"});
    }
  }

  std::ostringstream ura_key;
  ura_key << geometry_key(*estimation.geometry) << '|' << planar.ny
          << 'x' << planar.nz;
  if (ura_key.str() != state.ura_key) {
    UraInvariancePairs pairs;
    try {
      pairs = build_ura_dual_invariance(planar.ny, planar.nz);
    } catch (const std::invalid_argument& error) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, error.what()});
    } catch (const std::runtime_error& error) {
      return make_unexpected(
          Error{ErrorCode::kFailure, error.what()});
    }
    state.ura_y_pairs =
        static_cast<int64_t>(pairs.y_reference.size());
    state.ura_z_pairs =
        static_cast<int64_t>(pairs.z_reference.size());
    const auto upload =
        [&](const std::vector<int64_t>& source,
            std::shared_ptr<holoscan::Tensor>& destination,
            const char* what) -> expected<void, Error> {
      destination = make_device_tensor(
          {static_cast<int64_t>(source.size())},
          DLDataType{kDLInt, 32, 1});
      HOLOSCAN_TRY(
          auto destination_ptr,
          writable_ptr(*destination, what));
      std::vector<int32_t> host(source.size());
      for (size_t index = 0; index < source.size(); ++index) {
        host[index] = static_cast<int32_t>(source[index]);
      }
      if (cudaMemcpyAsync(
              destination_ptr, host.data(),
              host.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
              stream) != cudaSuccess) {
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
        upload(pairs.y_reference, state.d_y_reference, "y_reference"));
    HOLOSCAN_TRY(
        upload(pairs.y_shifted, state.d_y_shifted, "y_shifted"));
    HOLOSCAN_TRY(
        upload(pairs.z_reference, state.d_z_reference, "z_reference"));
    HOLOSCAN_TRY(
        upload(pairs.z_shifted, state.d_z_shifted, "z_shifted"));
    state.ura_key = ura_key.str();
  }

  if (geometry_changed) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: unseen array_geometry override; revalidating "
        "the URA dual-invariance pairs");
  }
  if (estimation.smoothing_changed) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: spatial smoothing override changes the ESPRIT "
        "aperture and derived subarray pairs");
  }

  const int64_t model_order_cap = std::min(
      {plan.stage.max_sources, state.ura_y_pairs, state.ura_z_pairs,
       estimation.num_channels - 1});
  const int64_t workspace_dim =
      std::max<int64_t>(model_order_cap, 1);
  HOLOSCAN_TRY(
      auto common,
      prepare_esprit_common_workspace<T>(
          state, plan.num_items, estimation.num_channels));

  const DLDataType cplx_dtype = complex_dtype<T>();
  std::array<std::shared_ptr<holoscan::Tensor>, 6> local_workspace;
  if (!state.matrix_workspace[0]) {
    for (auto& item : local_workspace) {
      item = make_device_tensor(
          {plan.num_items, workspace_dim, workspace_dim}, cplx_dtype);
    }
  }
  const auto& matrix =
      state.matrix_workspace[0] ? state.matrix_workspace : local_workspace;
  HOLOSCAN_TRY(auto psi_y, writable_ptr(*matrix[0], "psi_y"));
  HOLOSCAN_TRY(auto psi_z, writable_ptr(*matrix[1], "psi_z"));
  HOLOSCAN_TRY(auto z_workspace, writable_ptr(*matrix[2], "z_ws"));
  HOLOSCAN_TRY(auto q_workspace, writable_ptr(*matrix[3], "q_ws"));
  HOLOSCAN_TRY(auto r_workspace, writable_ptr(*matrix[4], "r_ws"));
  HOLOSCAN_TRY(auto temporary, writable_ptr(*matrix[5], "tmp_ws"));

  HOLOSCAN_TRY(
      auto outputs,
      prepare_esprit_outputs(
          aoa_port, doa_port, count_port, eigenvalue_port, context,
          plan, estimation.num_channels, real_dtype<T>()));

  const double k = kTwoPi / estimation.geometry->wavelength;
  try {
    detail::run_esprit_2d_stage<T>(
        estimation.data, estimation.mask, plan.num_items,
        estimation.num_channels, estimation.num_snapshots,
        plan.stage.num_sources, plan.stage.source_count_pfa,
        model_order_cap,
        static_cast<const int32_t*>(state.d_y_reference->data()),
        static_cast<const int32_t*>(state.d_y_shifted->data()),
        state.ura_y_pairs,
        static_cast<const int32_t*>(state.d_z_reference->data()),
        static_cast<const int32_t*>(state.d_z_shifted->data()),
        state.ura_z_pairs, workspace_dim,
        static_cast<T>(k * planar.dy), static_cast<T>(k * planar.dz),
        static_cast<T>(planar.y_axis[0]),
        static_cast<T>(planar.y_axis[1]),
        static_cast<T>(planar.y_axis[2]),
        static_cast<T>(planar.z_axis[0]),
        static_cast<T>(planar.z_axis[1]),
        static_cast<T>(planar.z_axis[2]),
        static_cast<T>(planar.normal[0]),
        static_cast<T>(planar.normal[1]),
        static_cast<T>(planar.normal[2]), convention,
        common.covariance_ptr, common.eigenvectors_ptr,
        common.valid_counts_ptr, common.model_order_ptr,
        static_cast<T*>(outputs.eigenvalue_writer.data()),
        static_cast<cuda::std::complex<T>*>(psi_y),
        static_cast<cuda::std::complex<T>*>(psi_z),
        static_cast<cuda::std::complex<T>*>(z_workspace),
        static_cast<cuda::std::complex<T>*>(q_workspace),
        static_cast<cuda::std::complex<T>*>(r_workspace),
        static_cast<cuda::std::complex<T>*>(temporary),
        static_cast<T*>(outputs.aoa_writer.data()),
        static_cast<T*>(outputs.doa_writer.data()),
        plan.stage.max_sources,
        static_cast<int32_t*>(outputs.count_writer.data()), stream);
    check_cuda("ESPRIT 2-D stage", kOpName);
  } catch (const std::runtime_error& error) {
    return make_unexpected(
        Error{ErrorCode::kFailure, error.what()});
  }

  if (plan.stage.num_sources < 0 && estimation.mask == nullptr &&
      estimation.num_snapshots < estimation.num_channels &&
      !state.warned_rank_deficient) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: num_sources=auto with fewer snapshots than "
        "channels; falling back to one source");
    state.warned_rank_deficient = true;
  }
  outputs.workspace_keepalive = {
      std::move(common.covariance), std::move(common.eigenvectors),
      std::move(common.valid_counts), std::move(common.model_order)};
  outputs.workspace_keepalive.insert(
      outputs.workspace_keepalive.end(), matrix.begin(), matrix.end());
  return outputs;
}

template <typename T>
expected<ESPRITPreparedOutputs, Error>
execute_esprit_shift_invariant_1d(
    ESPRITRuntimeState<T>& state, const ESPRITStagePlan& plan,
    const ESPRITEstimationInput<T>& estimation,
    const detail::DeviceAngleConvention<T>& convention,
    bool geometry_changed, Output<Tensor>& aoa_port,
    Output<Tensor>& doa_port, Output<Tensor>& count_port,
    Output<Tensor>& eigenvalue_port, ExecutionContext& context,
    cudaStream_t stream) {
  const std::string current_pair_key =
      pair_key(*estimation.geometry, plan.stage.subarray_spec);
  if (current_pair_key != state.subarray_key) {
    try {
      state.subarray_plan = build_shift_invariant_subarray_plan(
          *estimation.geometry, plan.stage.subarray_spec);
    } catch (const std::invalid_argument& error) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, error.what()});
    } catch (const std::runtime_error& error) {
      return make_unexpected(
          Error{ErrorCode::kFailure, error.what()});
    }
    state.subarray_key = current_pair_key;
    const int64_t pair_size = static_cast<int64_t>(
        state.subarray_plan.pair.reference_indices.size());
    state.d_reference_indices =
        make_device_tensor({pair_size}, DLDataType{kDLInt, 32, 1});
    state.d_shifted_indices =
        make_device_tensor({pair_size}, DLDataType{kDLInt, 32, 1});
    std::vector<int32_t> reference(static_cast<size_t>(pair_size));
    std::vector<int32_t> shifted(static_cast<size_t>(pair_size));
    for (int64_t index = 0; index < pair_size; ++index) {
      reference[static_cast<size_t>(index)] = static_cast<int32_t>(
          state.subarray_plan.pair.reference_indices[
              static_cast<size_t>(index)]);
      shifted[static_cast<size_t>(index)] = static_cast<int32_t>(
          state.subarray_plan.pair.shifted_indices[
              static_cast<size_t>(index)]);
    }
    HOLOSCAN_TRY(
        auto reference_ptr,
        writable_ptr(*state.d_reference_indices, "reference_indices"));
    HOLOSCAN_TRY(
        auto shifted_ptr,
        writable_ptr(*state.d_shifted_indices, "shifted_indices"));
    if (cudaMemcpyAsync(
            reference_ptr, reference.data(),
            reference.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
            stream) != cudaSuccess ||
        cudaMemcpyAsync(
            shifted_ptr, shifted.data(),
            shifted.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
            stream) != cudaSuccess) {
      return make_unexpected(
          Error{ErrorCode::kFailure,
                "ESPRITEstimatorOp: subarray index upload failed"});
    }
  }

  const int64_t pair_size = static_cast<int64_t>(
      state.subarray_plan.pair.reference_indices.size());
  if (plan.stage.num_sources >= 0 &&
      plan.stage.num_sources > pair_size) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: num_sources must not exceed the number of "
              "corresponding subarray elements"});
  }
  if (plan.stage.num_sources >= estimation.num_channels) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "ESPRITEstimatorOp: num_sources must leave at least one "
              "noise-subspace dimension"});
  }
  if (geometry_changed) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: unseen array_geometry override; "
        "revalidating the shift-invariant subarray pair");
  }
  if (estimation.smoothing_changed) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: spatial smoothing override changes the ESPRIT "
        "aperture and derived subarray pair");
  }

  const int64_t model_order_cap =
      std::min(plan.stage.max_sources, pair_size);
  const int64_t workspace_dim =
      std::max<int64_t>(model_order_cap, 1);
  HOLOSCAN_TRY(
      auto common,
      prepare_esprit_common_workspace<T>(
          state, plan.num_items, estimation.num_channels));
  const DLDataType cplx_dtype = complex_dtype<T>();
  auto matrix_workspace =
      state.matrix_workspace[0]
          ? state.matrix_workspace[0]
          : make_device_tensor(
                {plan.num_items, workspace_dim, workspace_dim},
                cplx_dtype);
  auto q_workspace =
      state.matrix_workspace[1]
          ? state.matrix_workspace[1]
          : make_device_tensor(
                {plan.num_items, workspace_dim, workspace_dim},
                cplx_dtype);
  auto r_workspace =
      state.matrix_workspace[2]
          ? state.matrix_workspace[2]
          : make_device_tensor(
                {plan.num_items, workspace_dim, workspace_dim},
                cplx_dtype);
  HOLOSCAN_TRY(
      auto matrix_ptr,
      writable_ptr(*matrix_workspace, "matrix_ws"));
  HOLOSCAN_TRY(auto q_ptr, writable_ptr(*q_workspace, "q_ws"));
  HOLOSCAN_TRY(auto r_ptr, writable_ptr(*r_workspace, "r_ws"));

  HOLOSCAN_TRY(
      auto outputs,
      prepare_esprit_outputs(
          aoa_port, doa_port, count_port, eigenvalue_port, context,
          plan, estimation.num_channels, real_dtype<T>()));
  try {
    detail::run_esprit_stage<T>(
        estimation.data, estimation.mask, plan.num_items,
        estimation.num_channels, estimation.num_snapshots,
        plan.stage.num_sources, plan.stage.source_count_pfa,
        model_order_cap,
        static_cast<const int32_t*>(state.d_reference_indices->data()),
        static_cast<const int32_t*>(state.d_shifted_indices->data()),
        pair_size, workspace_dim,
        static_cast<T>(kTwoPi * state.subarray_plan.spacing /
                       estimation.geometry->wavelength),
        static_cast<T>(state.subarray_plan.axis[0]),
        static_cast<T>(state.subarray_plan.axis[1]),
        static_cast<T>(state.subarray_plan.axis[2]),
        static_cast<T>(state.subarray_plan.broadside[0]),
        static_cast<T>(state.subarray_plan.broadside[1]),
        static_cast<T>(state.subarray_plan.broadside[2]), convention,
        common.covariance_ptr, common.eigenvectors_ptr,
        common.valid_counts_ptr, common.model_order_ptr,
        static_cast<T*>(outputs.eigenvalue_writer.data()),
        static_cast<cuda::std::complex<T>*>(matrix_ptr),
        static_cast<cuda::std::complex<T>*>(q_ptr),
        static_cast<cuda::std::complex<T>*>(r_ptr),
        static_cast<T*>(outputs.aoa_writer.data()),
        static_cast<T*>(outputs.doa_writer.data()),
        plan.stage.max_sources,
        static_cast<int32_t*>(outputs.count_writer.data()), stream);
    check_cuda("ESPRIT stage", kOpName);
  } catch (const std::runtime_error& error) {
    return make_unexpected(
        Error{ErrorCode::kFailure, error.what()});
  }

  if (plan.stage.num_sources < 0 && estimation.mask == nullptr &&
      estimation.num_snapshots < estimation.num_channels &&
      !state.warned_rank_deficient) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: num_sources=auto with fewer snapshots than "
        "channels; falling back to one source");
    state.warned_rank_deficient = true;
  }
  outputs.workspace_keepalive = {
      std::move(common.covariance), std::move(common.eigenvectors),
      std::move(common.valid_counts), std::move(common.model_order),
      std::move(matrix_workspace), std::move(q_workspace),
      std::move(r_workspace)};
  return outputs;
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
  // 1. Receive the snapshot cube and resolve this tick's live configuration
  // into one validated plan with all output shapes.
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));
  try {
    require_cuda_accessible(snap, "snapshot", kOpName);
    require_contiguous(snap, "snapshot", kOpName);
    require_dtype(snap, complex_dtype<T>(), "snapshot", kOpName);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  const ESPRITParams& pinned = context.params<ESPRITParams>();
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(array_geometry, spatial_smoothing, num_sources,
                     source_count_pfa, max_sources, pinned, extents_));
  HOLOSCAN_TRY(
      auto plan,
      build_esprit_stage_plan(snap, std::move(stage), extents_));

  if (state_->seeded_angle_convention.has_value() &&
      *state_->seeded_angle_convention != plan.stage.convention) {
    return fail_invalid(
        "ESPRITEstimatorOp: angle_convention is static and cannot change "
        "after the first successful activation; instantiate a separate "
        "operator per convention");
  }

  // 2. Validate the optional mask and runtime stream, then prepare the
  // convention representation that becomes permanent only after publication.
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
    if (vshape.size() != 2 || vshape[0] != plan.num_items ||
        vshape[1] != plan.num_snapshots) {
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

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "ESPRITEstimatorOp: no CUDA compute stream"});
  }

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

  // 3. Preserve the output contract for an empty item axis without entering
  // covariance, eigendecomposition, or rotational-invariance work.
  if (plan.num_items == 0) {
    HOLOSCAN_TRY(
        emit_empty_outputs(
            aoa_estimates, doa_multi, source_count, eigenvalues,
            context, plan, real_dtype<T>()));
    seed_angle_convention_after_emit();
    retained_snapshot_ = std::move(snap);
    if (maybe_valid.has_value()) {
      retained_is_valid_ = std::move(*maybe_valid);
    }
    return {};
  }

  // 4. Track runtime geometry, form the effective covariance input, and
  // analyze an automatic subarray request for the native filled-URA path.
  const std::string geom_key = geometry_key(plan.stage.geometry);
  const bool geometry_is_new =
      state_->seen_geometry_keys.insert(geom_key).second;
  const bool geometry_changed =
      geometry_is_new && state_->seen_geometry_keys.size() > 1;

  HOLOSCAN_TRY(
      auto estimation,
      prepare_smoothed_input(*state_, snap, mask_ptr, plan, stream));

  PlanarAperturePlan planar{};
  if (plan.stage.subarray_spec.is_auto()) {
    try {
      planar = analyze_planar_aperture(*estimation.geometry);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  }

  // 5. Dispatch the geometry-specific algorithm. Auto URA uses two invariance
  // axes and pairs them; all explicit or linear cases use one subarray pair.
  if (planar.is_ura && plan.stage.subarray_spec.is_auto()) {
    HOLOSCAN_TRY(
        auto outputs,
        execute_esprit_auto_ura(
            *state_, plan, estimation, planar, *convention,
            geometry_changed, aoa_estimates, doa_multi, source_count,
            eigenvalues, context, stream));
    HOLOSCAN_TRY(
        commit_and_publish_esprit_outputs(
            aoa_estimates, doa_multi, source_count, eigenvalues,
            context, std::move(outputs)));
  } else {
    HOLOSCAN_TRY(
        auto outputs,
        execute_esprit_shift_invariant_1d(
            *state_, plan, estimation, *convention, geometry_changed,
            aoa_estimates, doa_multi, source_count, eigenvalues, context,
            stream));
    HOLOSCAN_TRY(
        commit_and_publish_esprit_outputs(
            aoa_estimates, doa_multi, source_count, eigenvalues,
            context, std::move(outputs)));
  }

  // 6. Publication succeeded: lock the static convention and retain input
  // storage until asynchronous work from this tick has completed.
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
