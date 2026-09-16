// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/logger/logger.hpp>

#include "angle_estimation/mvdr_estimator_op/mvdr_stage.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"
#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/config_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

void retire_matx_solver_caches() {
  detail::retire_matx_solver_caches();
}

struct MVDRSteeringArtifact {
  std::shared_ptr<holoscan::Tensor> manifold;
  std::shared_ptr<holoscan::Tensor> az_grid;
  std::shared_ptr<holoscan::Tensor> el_grid;
};

template <typename T>
struct MVDRRuntimeState {
  ConfigCache<std::shared_ptr<MVDRSteeringArtifact>> steering_cache;
  SpatialSmoothingPlan smoothing_plan;
  std::string smoothing_key;
  std::shared_ptr<holoscan::Tensor> d_subarray_index;
  std::shared_ptr<holoscan::Tensor> d_fb_flip;
  std::shared_ptr<holoscan::Tensor> covariance;
  std::shared_ptr<holoscan::Tensor> inverse;
  std::shared_ptr<holoscan::Tensor> valid_counts;
  std::shared_ptr<holoscan::Tensor> smoothed_snapshot;
  std::shared_ptr<holoscan::Tensor> smoothed_mask;
  std::size_t smoothed_snapshot_capacity{0U};
  std::size_t smoothed_mask_capacity{0U};
};

namespace {

constexpr const char* kOpName = "MVDREstimatorOp";

// Upper bound on max_sources: bounds the per-block shared-memory selected-beam
// list in peak_extract_kernel. Generous for "number of Capon peaks".
constexpr int64_t kMaxSourcesCap = 64;

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

std::string spatial_smoothing_key(
    const ArrayGeometry& geometry, const std::vector<int64_t>& smoothing,
    bool forward_backward) {
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10);
  stream << geometry.wavelength << '|';
  for (double value : geometry.element_positions) {
    stream << value << ',';
  }
  stream << '|';
  for (int64_t value : smoothing) {
    stream << value << ',';
  }
  stream << '|' << (forward_backward ? 1 : 0);
  return stream.str();
}

template <typename T>
std::vector<T> cast_vector(const std::vector<double>& in) {
  std::vector<T> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = static_cast<T>(in[i]);
  }
  return out;
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

struct MVDRStageConfig {
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_angles;
  std::vector<double> el_angles;
  std::vector<int64_t> spatial_smoothing;
  double diagonal_loading{-1.0};
  int64_t max_sources{1};
  bool forward_backward{true};
};

struct MVDRStagePlan {
  MVDRStageConfig stage;
  int64_t num_items{0};
  int64_t num_channels{0};
  int64_t num_snapshots{0};
  int64_t num_az{0};
  int64_t num_el{0};
  std::vector<int64_t> aoa_shape;
  std::vector<int64_t> spectrum_shape;
  std::vector<int64_t> doa_shape;
  std::vector<int64_t> count_shape;
};

template <typename T>
struct MVDREstimationInput {
  const ArrayGeometry* geometry{nullptr};
  int64_t num_channels{0};
  int64_t num_snapshots{0};
  const cuda::std::complex<T>* data{nullptr};
  const unsigned char* mask{nullptr};
  std::shared_ptr<holoscan::Tensor> snapshot_keepalive;
  std::shared_ptr<holoscan::Tensor> mask_keepalive;
};

expected<MVDRStageConfig, Error> resolve_params(
    Input<WireArrayGeometry>& geometry_port, Input<Tensor>& az_port,
    Input<Tensor>& el_port,
    Input<SpatialSmoothingOverride>& smoothing_port,
    Input<double>& loading_port, Input<int64_t>& max_sources_port,
    const MVDRParams& pinned, const MaxTensorExtents& extents) {
  HOLOSCAN_TRY(
      auto geometry,
      resolve_live_override_via(
          geometry_port, pinned.array_geometry,
          [](const WireArrayGeometry& value) { return from_wire(value); }));
  HOLOSCAN_TRY(
      auto az,
      resolve_live_override_via(
          az_port, pinned.az_angles,
          [&extents](const Tensor& value) {
            return read_host_config_vector<double>(
                value, DLDataType{kDLFloat, 64U, 1U},
                require_extent(extents.max_az, "max_az", kOpName),
                "az_angles");
          }));
  HOLOSCAN_TRY(
      auto el,
      resolve_live_override_via(
          el_port, pinned.el_angles,
          [&extents](const Tensor& value) {
            return read_host_config_vector<double>(
                value, DLDataType{kDLFloat, 64U, 1U},
                require_extent(extents.max_el, "max_el", kOpName),
                "el_angles");
          }));
  HOLOSCAN_TRY(
      auto smoothing,
      resolve_live_override_via(
          smoothing_port, pinned.spatial_smoothing,
          [](const SpatialSmoothingOverride& value) {
            return from_wire(value);
          }));
  HOLOSCAN_TRY(
      auto loading,
      resolve_live_override(loading_port, pinned.diagonal_loading));
  HOLOSCAN_TRY(
      auto resolved_max_sources,
      resolve_live_override(max_sources_port, pinned.max_sources));
  return MVDRStageConfig{
      .geometry = std::move(geometry),
      .convention = pinned.angle_convention,
      .az_angles = std::move(az),
      .el_angles = std::move(el),
      .spatial_smoothing = std::move(smoothing),
      .diagonal_loading = loading,
      .max_sources = resolved_max_sources,
      .forward_backward = pinned.forward_backward,
  };
}

expected<MVDRStagePlan, Error> build_mvdr_stage_plan(
    const Tensor& snapshot, MVDRStageConfig stage,
    const MaxTensorExtents& extents) {
  try {
    stage.geometry.validate();
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  }
  if (stage.az_angles.empty() || stage.el_angles.empty()) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "MVDREstimatorOp: az_angles and el_angles must be non-empty"});
  }
  try {
    validate_scan_grid_axes(stage.convention, stage.az_angles, stage.el_angles,
                            "MVDREstimatorOp");
    (void)boundary_to_canonical(
        stage.convention, stage.az_angles.front(), stage.el_angles.front());
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  }
  const int64_t max_az =
      require_extent(extents.max_az, "max_az", kOpName);
  const int64_t max_el =
      require_extent(extents.max_el, "max_el", kOpName);
  const int64_t max_channels =
      require_extent(extents.max_channels, "max_channels", kOpName);
  if (static_cast<int64_t>(stage.az_angles.size()) > max_az ||
      static_cast<int64_t>(stage.el_angles.size()) > max_el ||
      stage.geometry.num_elements() > max_channels) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "MVDREstimatorOp: live configuration exceeds MaxTensorExtents"});
  }
  if (stage.max_sources < 1 || stage.max_sources > kMaxSourcesCap) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "MVDREstimatorOp: max_sources must be in [1, 64]"});
  }
  if (extents.max_sources.has_value() &&
      stage.max_sources > *extents.max_sources) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "MVDREstimatorOp: max_sources exceeds "
              "MaxTensorExtents.max_sources"});
  }

  const auto& shape = snapshot.shape();
  if (shape.size() != 3U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "MVDREstimatorOp: snapshot must have shape "
              "(num_items, num_channels, num_snapshots)"});
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape[2];
  if (extents.max_items.has_value() && num_items > *extents.max_items) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "MVDREstimatorOp: num_items exceeds "
              "MaxTensorExtents.max_items"});
  }
  if (num_channels != stage.geometry.num_elements()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "MVDREstimatorOp: snapshot num_channels must match "
              "array_geometry.num_elements"});
  }

  const int64_t num_az = static_cast<int64_t>(stage.az_angles.size());
  const int64_t num_el = static_cast<int64_t>(stage.el_angles.size());
  const int64_t max_sources = stage.max_sources;
  return MVDRStagePlan{
      .stage = std::move(stage),
      .num_items = num_items,
      .num_channels = num_channels,
      .num_snapshots = num_snapshots,
      .num_az = num_az,
      .num_el = num_el,
      .aoa_shape = {num_items, 2},
      .spectrum_shape = {num_items, num_az, num_el},
      .doa_shape = {num_items, max_sources, 2},
      .count_shape = {num_items},
  };
}

template <typename T>
expected<MVDREstimationInput<T>, Error> prepare_smoothed_input(
    MVDRRuntimeState<T>& state, const Tensor& snapshot,
    const unsigned char* mask, const MVDRStagePlan& plan,
    std::string candidate_key,
    std::optional<SpatialSmoothingPlan> candidate_plan,
    cudaStream_t stream) {
  MVDREstimationInput<T> result{
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

  if (candidate_key != state.smoothing_key) {
    if (!candidate_plan.has_value()) {
      return make_unexpected(
          Error{ErrorCode::kFailure,
                "MVDREstimatorOp: missing spatial-smoothing plan"});
    }
    state.smoothing_plan = std::move(*candidate_plan);
    state.smoothing_key = std::move(candidate_key);
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
                  "MVDREstimatorOp: spatial-smoothing index H2D failed"});
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
                    "MVDREstimatorOp: spatial-smoothing fb-flip H2D failed"});
        }
      } else {
        state.d_fb_flip.reset();
      }
      HOLOSCAN_LOG_WARN(
          "MVDREstimatorOp: spatial smoothing on — aperture {}->{} "
          "channels, {} sub-arrays{}",
          plan.num_channels, subarray_size, subarrays,
          state.smoothing_plan.forward_backward
              ? " (forward-backward)"
              : "");
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

expected<const unsigned char*, Error> prepare_mvdr_mask(
    std::optional<Tensor>& maybe_valid, int64_t num_items,
    int64_t num_snapshots) {
  if (!maybe_valid.has_value()) {
    return nullptr;
  }
  const DLDataType dtype = maybe_valid->dtype();
  const bool valid_dtype =
      (dtype.code == kDLBool || dtype.code == kDLUInt) &&
      dtype.bits == 8 && dtype.lanes == 1;
  if (!valid_dtype) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "MVDREstimatorOp: is_valid must have dtype bool or uint8"});
  }
  const auto& shape = maybe_valid->shape();
  if (shape.size() != 2 || shape[0] != num_items ||
      shape[1] != num_snapshots) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "MVDREstimatorOp: is_valid must have shape "
              "(num_items, num_snapshots) matching snapshot"});
  }
  try {
    require_cuda_accessible(*maybe_valid, "is_valid", kOpName);
    require_contiguous(*maybe_valid, "is_valid", kOpName);
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  }
  return static_cast<const unsigned char*>(maybe_valid->data());
}

template <typename T>
expected<void, Error> emit_empty_mvdr_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& spectrum_port,
    Output<Tensor>& doa_port, Output<Tensor>& count_port,
    Output<Tensor>& eigenvalues_port, LatchedOutput& doa_latch,
    LatchedOutput& count_latch, LatchedOutput& eigenvalues_latch,
    ExecutionContext& context, const MVDRStagePlan& plan) {
  const DLDataType output_dtype = real_dtype<T>();
  const DLDataType count_dtype{kDLInt, 32, 1};
  const std::vector<int64_t> eigenvalues_shape{
      plan.num_items, plan.num_channels};
  auto noop = [](void*, std::size_t) -> expected<void, Error> {
    return {};
  };

  HOLOSCAN_TRY(auto aoa_loan,
               prepare_device_tensor(aoa_port, context, plan.aoa_shape,
                                     output_dtype, noop));
  HOLOSCAN_TRY(auto spectrum_loan,
               prepare_device_tensor(spectrum_port, context,
                                     plan.spectrum_shape, output_dtype, noop));
  std::optional<TensorOutputLoan> doa_loan;
  std::optional<TensorOutputLoan> count_loan;
  std::optional<TensorOutputLoan> eigenvalues_loan;
  if (doa_latch.active()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_device_tensor(doa_port, context, plan.doa_shape,
                                       output_dtype, noop));
    doa_loan.emplace(std::move(loan));
  }
  if (count_latch.active()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_device_tensor(count_port, context, plan.count_shape,
                                       count_dtype, noop));
    count_loan.emplace(std::move(loan));
  }
  if (eigenvalues_latch.active()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_device_tensor(
                     eigenvalues_port, context, eigenvalues_shape,
                     output_dtype, noop));
    eigenvalues_loan.emplace(std::move(loan));
  }

  HOLOSCAN_TRY(
      emit_prepared_tensor(aoa_port, context, std::move(aoa_loan)));
  (void)emit_prepared_tensor(
      spectrum_port, context, std::move(spectrum_loan));
  if (doa_loan.has_value()) {
    doa_latch.emit(doa_port, context, std::move(*doa_loan),
                   "MVDREstimatorOp.doa_multi");
  }
  if (count_loan.has_value()) {
    count_latch.emit(count_port, context, std::move(*count_loan),
                     "MVDREstimatorOp.source_count");
  }
  if (eigenvalues_loan.has_value()) {
    eigenvalues_latch.emit(
        eigenvalues_port, context, std::move(*eigenvalues_loan),
        "MVDREstimatorOp.eigenvalues");
  }
  return {};
}

template <typename T>
expected<std::shared_ptr<MVDRSteeringArtifact>, Error>
prepare_mvdr_steering(
    MVDRRuntimeState<T>& state, const ArrayGeometry& geometry,
    const AngleConvention& convention, const std::vector<double>& azimuth,
    const std::vector<double>& elevation, int64_t num_az, int64_t num_el,
    cudaStream_t stream) {
  const DLDataType output_dtype = real_dtype<T>();
  const std::string cache_key = steering_cache_key(
      geometry, convention, azimuth, elevation, WindowSpec{});
  const std::shared_ptr<MVDRSteeringArtifact>& steering =
      state.steering_cache.get_or_build(
          cache_key, "steering manifold and scan grids", [&] {
            auto artifact = std::make_shared<MVDRSteeringArtifact>();
            artifact->manifold = build_steering_weights_device<T>(
                geometry, convention, azimuth, elevation, WindowSpec{},
                BeamOrder::kElMajor, SteeringLayout::kBeamMajor,
                /*conjugate=*/false, stream);
            const auto azimuth_host = cast_vector<T>(azimuth);
            const auto elevation_host = cast_vector<T>(elevation);
            artifact->az_grid =
                make_device_tensor({num_az}, output_dtype);
            artifact->el_grid =
                make_device_tensor({num_el}, output_dtype);
            auto azimuth_mut = artifact->az_grid->mutable_data();
            auto elevation_mut = artifact->el_grid->mutable_data();
            if (!azimuth_mut || !elevation_mut ||
                cudaMemcpyAsync(
                    *azimuth_mut, azimuth_host.data(),
                    azimuth_host.size() * sizeof(T), cudaMemcpyHostToDevice,
                    stream) != cudaSuccess ||
                cudaMemcpyAsync(
                    *elevation_mut, elevation_host.data(),
                    elevation_host.size() * sizeof(T), cudaMemcpyHostToDevice,
                    stream) != cudaSuccess) {
              return std::shared_ptr<MVDRSteeringArtifact>{};
            }
            return artifact;
          });
  if (!steering || !steering->manifold || !steering->az_grid ||
      !steering->el_grid) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              "MVDREstimatorOp: steering artifact build failed"});
  }
  return steering;
}

template <typename T>
struct MVDRPreparedWorkspace {
  std::shared_ptr<holoscan::Tensor> covariance;
  std::shared_ptr<holoscan::Tensor> inverse;
  std::shared_ptr<holoscan::Tensor> valid_counts;
  cuda::std::complex<T>* covariance_ptr{nullptr};
  cuda::std::complex<T>* inverse_ptr{nullptr};
  int64_t* valid_counts_ptr{nullptr};
};

template <typename T>
expected<MVDRPreparedWorkspace<T>, Error> prepare_mvdr_workspace(
    MVDRRuntimeState<T>& state, int64_t num_items, int64_t num_channels) {
  const DLDataType complex_type = complex_dtype<T>();
  const DLDataType int64_type{kDLInt, 64, 1};
  const int64_t inverse_items = num_items == 1 ? 2 : num_items;
  MVDRPreparedWorkspace<T> workspace;
  workspace.covariance =
      state.covariance
          ? state.covariance
          : make_device_tensor(
                {inverse_items, num_channels, num_channels}, complex_type);
  workspace.inverse =
      state.inverse
          ? state.inverse
          : make_device_tensor(
                {inverse_items, num_channels, num_channels}, complex_type);
  workspace.valid_counts =
      state.valid_counts
          ? state.valid_counts
          : make_device_tensor({num_items}, int64_type);
  HOLOSCAN_TRY(auto covariance_ptr,
               writable_ptr(*workspace.covariance, "cov"));
  HOLOSCAN_TRY(auto inverse_ptr,
               writable_ptr(*workspace.inverse, "rinv"));
  HOLOSCAN_TRY(auto valid_counts_ptr,
               writable_ptr(*workspace.valid_counts, "k_counts"));
  workspace.covariance_ptr =
      static_cast<cuda::std::complex<T>*>(covariance_ptr);
  workspace.inverse_ptr =
      static_cast<cuda::std::complex<T>*>(inverse_ptr);
  workspace.valid_counts_ptr = static_cast<int64_t*>(valid_counts_ptr);
  return workspace;
}

struct MVDRPreparedOutputs {
  std::optional<TensorOutputLoan> aoa_loan;
  std::optional<TensorOutputLoan> spectrum_loan;
  std::optional<TensorOutputLoan> doa_loan;
  std::optional<TensorOutputLoan> count_loan;
  std::optional<TensorOutputLoan> eigenvalues_loan;
  std::optional<TensorOutputWriteGuard> aoa_writer;
  std::optional<TensorOutputWriteGuard> spectrum_writer;
  std::optional<TensorOutputWriteGuard> doa_writer;
  std::optional<TensorOutputWriteGuard> count_writer;
  std::optional<TensorOutputWriteGuard> eigenvalues_writer;
};

template <typename T>
expected<MVDRPreparedOutputs, Error> prepare_mvdr_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& spectrum_port,
    Output<Tensor>& doa_port, Output<Tensor>& count_port,
    Output<Tensor>& eigenvalues_port, ExecutionContext& context,
    const MVDRStagePlan& plan, int64_t estimation_channels,
    bool want_peaks, bool want_eigenvalues) {
  const DLDataType output_dtype = real_dtype<T>();
  const DLDataType count_dtype{kDLInt, 32, 1};
  const std::vector<int64_t> eigenvalues_shape{
      plan.num_items, estimation_channels};
  MVDRPreparedOutputs outputs;

  HOLOSCAN_TRY(auto aoa,
               aoa_port.allocate_tensor(TensorLoanRequest{
                   .shape = plan.aoa_shape, .dtype = output_dtype}));
  outputs.aoa_loan.emplace(std::move(aoa));
  HOLOSCAN_TRY(auto spectrum,
               spectrum_port.allocate_tensor(TensorLoanRequest{
                   .shape = plan.spectrum_shape, .dtype = output_dtype}));
  outputs.spectrum_loan.emplace(std::move(spectrum));
  if (want_peaks) {
    HOLOSCAN_TRY(auto doa,
                 doa_port.allocate_tensor(TensorLoanRequest{
                     .shape = plan.doa_shape, .dtype = output_dtype}));
    outputs.doa_loan.emplace(std::move(doa));
    HOLOSCAN_TRY(auto count,
                 count_port.allocate_tensor(TensorLoanRequest{
                     .shape = plan.count_shape, .dtype = count_dtype}));
    outputs.count_loan.emplace(std::move(count));
  }
  if (want_eigenvalues) {
    HOLOSCAN_TRY(auto eigenvalues,
                 eigenvalues_port.allocate_tensor(TensorLoanRequest{
                     .shape = eigenvalues_shape, .dtype = output_dtype}));
    outputs.eigenvalues_loan.emplace(std::move(eigenvalues));
  }

  HOLOSCAN_TRY(auto aoa_writer,
               outputs.aoa_loan->write(context.cuda_stream()));
  outputs.aoa_writer.emplace(std::move(aoa_writer));
  HOLOSCAN_TRY(auto spectrum_writer,
               outputs.spectrum_loan->write(context.cuda_stream()));
  outputs.spectrum_writer.emplace(std::move(spectrum_writer));
  if (want_peaks) {
    HOLOSCAN_TRY(auto doa_writer,
                 outputs.doa_loan->write(context.cuda_stream()));
    outputs.doa_writer.emplace(std::move(doa_writer));
    HOLOSCAN_TRY(auto count_writer,
                 outputs.count_loan->write(context.cuda_stream()));
    outputs.count_writer.emplace(std::move(count_writer));
  }
  if (want_eigenvalues) {
    HOLOSCAN_TRY(auto eigenvalues_writer,
                 outputs.eigenvalues_loan->write(context.cuda_stream()));
    outputs.eigenvalues_writer.emplace(std::move(eigenvalues_writer));
  }
  return outputs;
}

template <typename T>
expected<void, Error> execute_mvdr(
    const MVDREstimationInput<T>& estimation, const MVDRStagePlan& plan,
    const MVDRSteeringArtifact& steering,
    const MVDRPreparedWorkspace<T>& workspace,
    MVDRPreparedOutputs& outputs, cudaStream_t stream) {
  try {
    detail::run_mvdr_stage<T>(
        estimation.data, estimation.mask, plan.num_items,
        estimation.num_channels, estimation.num_snapshots,
        plan.stage.diagonal_loading,
        static_cast<const cuda::std::complex<T>*>(
            steering.manifold->data()),
        plan.num_az, plan.num_el,
        static_cast<const T*>(steering.az_grid->data()),
        static_cast<const T*>(steering.el_grid->data()),
        plan.stage.max_sources, workspace.covariance_ptr,
        workspace.inverse_ptr, workspace.valid_counts_ptr,
        static_cast<T*>(outputs.spectrum_writer->data()),
        static_cast<T*>(outputs.aoa_writer->data()),
        outputs.eigenvalues_writer
            ? static_cast<T*>(outputs.eigenvalues_writer->data())
            : nullptr,
        outputs.doa_writer
            ? static_cast<T*>(outputs.doa_writer->data())
            : nullptr,
        outputs.count_writer
            ? static_cast<int32_t*>(outputs.count_writer->data())
            : nullptr,
        stream);
    check_cuda("mvdr stage", kOpName);
  } catch (const std::runtime_error& error) {
    return fail_runtime(error.what());
  }
  return {};
}

expected<void, Error> publish_mvdr_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& spectrum_port,
    Output<Tensor>& doa_port, Output<Tensor>& count_port,
    Output<Tensor>& eigenvalues_port, LatchedOutput& doa_latch,
    LatchedOutput& count_latch, LatchedOutput& eigenvalues_latch,
    ExecutionContext& context, MVDRPreparedOutputs outputs) {
  HOLOSCAN_TRY(std::move(*outputs.aoa_writer).commit());
  HOLOSCAN_TRY(std::move(*outputs.spectrum_writer).commit());
  if (outputs.doa_writer.has_value()) {
    HOLOSCAN_TRY(std::move(*outputs.doa_writer).commit());
  }
  if (outputs.count_writer.has_value()) {
    HOLOSCAN_TRY(std::move(*outputs.count_writer).commit());
  }
  if (outputs.eigenvalues_writer.has_value()) {
    HOLOSCAN_TRY(std::move(*outputs.eigenvalues_writer).commit());
  }

  const EmitOptions emit_options{
      .capture_time = context.activation_time()};
  HOLOSCAN_TRY(
      aoa_port.emit(std::move(*outputs.aoa_loan), emit_options));
  (void)spectrum_port.emit(
      std::move(*outputs.spectrum_loan), emit_options);
  if (outputs.doa_loan.has_value()) {
    doa_latch.emit(doa_port, context, std::move(*outputs.doa_loan),
                   "MVDREstimatorOp.doa_multi");
  }
  if (outputs.count_loan.has_value()) {
    count_latch.emit(count_port, context, std::move(*outputs.count_loan),
                     "MVDREstimatorOp.source_count");
  }
  if (outputs.eigenvalues_loan.has_value()) {
    eigenvalues_latch.emit(
        eigenvalues_port, context, std::move(*outputs.eigenvalues_loan),
        "MVDREstimatorOp.eigenvalues");
  }
  return {};
}

}  // namespace

template <typename T>
MVDREstimatorOp<T>::MVDREstimatorOp(
    const MaxTensorExtents& extents) {
  state_ = std::make_shared<MVDRRuntimeState<T>>();
  extents_ = extents;
  if (!extents.max_items.has_value() &&
      !extents.max_az.has_value()) {
    return;
  }
  const char* who = kOpName;
  const auto max_items =
      require_extent(extents.max_items, "max_items", who);
  const auto max_az =
      require_extent(extents.max_az, "max_az", who);
  const auto max_el =
      require_extent(extents.max_el, "max_el", who);
  const auto max_sources =
      require_extent(extents.max_sources, "max_sources", who);
  const auto max_channels =
      require_extent(extents.max_channels, "max_channels", who);
  spectrum_max_byte_span_ =
      byte_span_real_cube<T>({max_items, max_az, max_el}, who);
  aoa_max_byte_span_ =
      byte_span_real_cube<T>({max_items, 2}, who);
  doa_max_byte_span_ =
      byte_span_real_cube<T>({max_items, max_sources, 2}, who);
  count_max_byte_span_ = byte_span_elems(
      product_extents({max_items}, who), 4U, who);
  eig_max_byte_span_ =
      byte_span_real_cube<T>({max_items, max_channels}, who);
  const int64_t inverse_items = max_items < 2 ? 2 : max_items;
  state_->covariance = make_device_tensor(
      {inverse_items, max_channels, max_channels}, complex_dtype<T>());
  state_->inverse = make_device_tensor(
      {inverse_items, max_channels, max_channels}, complex_dtype<T>());
  state_->valid_counts =
      make_device_tensor({max_items}, DLDataType{kDLInt, 64, 1});
}

template <typename T>
void MVDREstimatorOp<T>::stop() {
  detail::retire_matx_solver_caches();
}

template <typename T>
void MVDREstimatorOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kHostF64Vector{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 64U, 1U},
      .rank = 1U,
  };
  spec.input(snapshot).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(is_valid)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kDevice));
  spec.output(aoa_estimates)
      .produces_tensor(pooled_tensor_output(kDevice, 2U, aoa_max_byte_span_));
  // Optional in OPERATOR_API.md — always emitted so unwired consumers do
  // not need Connectivity gating; tests bind/wire all five (OPEN_ITEMS).
  spec.output(angle_spectrum)
      .produces_tensor(pooled_tensor_output(
          kDevice, 3U, spectrum_max_byte_span_));
  spec.output(doa_multi)
      .produces_tensor(pooled_tensor_output(kDevice, 3U, doa_max_byte_span_));
  spec.output(source_count)
      .produces_tensor(pooled_tensor_output(kDevice, 1U, count_max_byte_span_));
  spec.output(eigenvalues)
      .produces_tensor(pooled_tensor_output(kDevice, 2U, eig_max_byte_span_));
  // [live] Exact overrides, Params order.
  spec.input(array_geometry).optional().queue_depth(2U);
  spec.input(az_angles)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostF64Vector));
  spec.input(el_angles)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostF64Vector));
  spec.input(spatial_smoothing).optional().queue_depth(2U);
  spec.input(diagonal_loading).optional().queue_depth(2U);
  spec.input(max_sources).optional().queue_depth(2U);
}

template <typename T>
Contract MVDREstimatorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{snapshot})
      .input(is_valid, Exact{})
      .on_missing(is_valid, SafeDefault{});
  add_live_override_to_contract(result, array_geometry);
  add_live_override_to_contract(result, az_angles);
  add_live_override_to_contract(result, el_angles);
  add_live_override_to_contract(result, spatial_smoothing);
  add_live_override_to_contract(result, diagonal_loading);
  add_live_override_to_contract(result, max_sources);
  return result;
}

template <typename T>
expected<void, Error> MVDREstimatorOp<T>::compute(ExecutionContext& context) {
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

  const MVDRParams& pinned = context.params<MVDRParams>();
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(array_geometry, az_angles, el_angles, spatial_smoothing,
                     diagonal_loading, max_sources, pinned, extents_));
  HOLOSCAN_TRY(
      auto plan,
      build_mvdr_stage_plan(snap, std::move(stage), extents_));

  // 2. Validate tick sidebands and prepare any host-side smoothing-plan
  // replacement before mutating cached device state.
  const std::string candidate_smoothing_key =
      plan.stage.spatial_smoothing.empty()
          ? std::string{}
          : spatial_smoothing_key(
                plan.stage.geometry, plan.stage.spatial_smoothing,
                plan.stage.forward_backward);
  std::optional<SpatialSmoothingPlan> candidate_smoothing_plan;
  if (!plan.stage.spatial_smoothing.empty() &&
      candidate_smoothing_key != state_->smoothing_key) {
    try {
      candidate_smoothing_plan = build_spatial_smoothing_plan(
          plan.stage.geometry, plan.stage.spatial_smoothing,
          plan.stage.forward_backward);
    } catch (const std::invalid_argument& error) {
      return fail_invalid(error.what());
    } catch (const std::runtime_error& error) {
      return fail_runtime(error.what());
    }
  }

  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  HOLOSCAN_TRY(auto mask_ptr,
               prepare_mvdr_mask(
                   maybe_valid, plan.num_items, plan.num_snapshots));

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "MVDREstimatorOp: no CUDA compute stream"});
  }

  // 3. Preserve the output contract for an empty item axis without entering
  // covariance inversion or scan work.
  if (plan.num_items == 0) {
    HOLOSCAN_TRY(emit_empty_mvdr_outputs<T>(
        aoa_estimates, angle_spectrum, doa_multi, source_count, eigenvalues,
        doa_multi_out_, source_count_out_, eigenvalues_out_, context, plan));
    retained_snapshot_ = std::move(snap);
    if (maybe_valid.has_value()) {
      retained_is_valid_ = std::move(*maybe_valid);
    }
    return {};
  }

  // 4. Form the effective covariance input. Spatial smoothing may replace the
  // original aperture and expand the number of looks.
  HOLOSCAN_TRY(
      auto estimation,
      prepare_smoothed_input(
          *state_, snap, mask_ptr, plan, candidate_smoothing_key,
          std::move(candidate_smoothing_plan), stream));

  // 5. Prepare immutable steering data, reusable covariance workspace, and
  // output buffers. Optional diagnostics gate eig and peak-only work.
  HOLOSCAN_TRY(
      auto steering,
      prepare_mvdr_steering<T>(
          *state_, *estimation.geometry, plan.stage.convention,
          plan.stage.az_angles, plan.stage.el_angles, plan.num_az,
          plan.num_el, stream));
  HOLOSCAN_TRY(
      auto workspace,
      prepare_mvdr_workspace<T>(
          *state_, plan.num_items, estimation.num_channels));

  const bool want_eigenvalues = eigenvalues_out_.active();
  const bool want_peaks =
      doa_multi_out_.active() || source_count_out_.active();
  HOLOSCAN_TRY(
      auto outputs,
      prepare_mvdr_outputs<T>(
          aoa_estimates, angle_spectrum, doa_multi, source_count,
          eigenvalues, context, plan, estimation.num_channels, want_peaks,
          want_eigenvalues));

  // 6. Run covariance → diagonal loading / inversion → Capon scan and
  // optional diagnostics, then publish outputs in contract order.
  HOLOSCAN_TRY(
      execute_mvdr(estimation, plan, *steering, workspace, outputs, stream));
  HOLOSCAN_TRY(publish_mvdr_outputs(
      aoa_estimates, angle_spectrum, doa_multi, source_count, eigenvalues,
      doa_multi_out_, source_count_out_, eigenvalues_out_, context,
      std::move(outputs)));

  // 7. Keep input storage alive until all asynchronous work from this tick has
  // completed on the runtime stream.
  retained_snapshot_ = std::move(snap);
  if (maybe_valid.has_value()) {
    retained_is_valid_ = std::move(*maybe_valid);
  }
  return {};
}

template class MVDREstimatorOp<float>;
template class MVDREstimatorOp<double>;

}  // namespace holoscan::radar
