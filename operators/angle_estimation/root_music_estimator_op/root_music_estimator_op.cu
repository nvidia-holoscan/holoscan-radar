// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
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

#include "angle_estimation/root_music_estimator_op/root_music_stage.hpp"
#include "radar_common/dsp/planar_aperture.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"
#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

struct RootMUSICDeviceBuffer {
  std::shared_ptr<holoscan::Tensor> tensor;
  std::size_t capacity{0U};
};

template <typename T>
struct RootMUSICAxisRuntimeBuffers {
  RootMUSICDeviceBuffer eigenvectors;
  RootMUSICDeviceBuffer eigenvalues;
  RootMUSICDeviceBuffer projector;
  RootMUSICDeviceBuffer model_order;
  RootMUSICDeviceBuffer coefficients;
  RootMUSICDeviceBuffer companion;
  RootMUSICDeviceBuffer roots;
};

template <typename T>
struct RootMUSICURARuntimeBuffers {
  RootMUSICDeviceBuffer cov_y;
  RootMUSICDeviceBuffer cov_z;
  RootMUSICAxisRuntimeBuffers<T> axis_y;
  RootMUSICAxisRuntimeBuffers<T> axis_z;
  RootMUSICDeviceBuffer u_candidates;
  RootMUSICDeviceBuffer v_candidates;
  RootMUSICDeviceBuffer u_counts;
  RootMUSICDeviceBuffer v_counts;
};

template <typename T>
struct RootMUSICRuntimeState {
  bool warned_rank_deficient{false};
  detail::DeviceAngleConvention<T> device_angle_convention{};
  std::optional<AngleConvention> seeded_angle_convention;
  SpatialSmoothingPlan smoothing_plan;
  std::string smoothing_key;
  std::unordered_set<std::string> seen_geometry_keys;
  std::shared_ptr<holoscan::Tensor> d_subarray_index;
  std::shared_ptr<holoscan::Tensor> d_fb_flip;
  RootMUSICDeviceBuffer smoothed_snapshot;
  RootMUSICDeviceBuffer smoothed_mask;
  std::shared_ptr<holoscan::Tensor> covariance;
  std::shared_ptr<holoscan::Tensor> eigenvectors;
  std::shared_ptr<holoscan::Tensor> projector;
  std::shared_ptr<holoscan::Tensor> valid_counts;
  std::shared_ptr<holoscan::Tensor> model_order;
  std::shared_ptr<holoscan::Tensor> coefficients;
  std::shared_ptr<holoscan::Tensor> companion;
  std::shared_ptr<holoscan::Tensor> roots;
  RootMUSICURARuntimeBuffers<T> ura;
  std::shared_ptr<detail::RootMUSICSolverState> solver;
  std::shared_ptr<detail::RootMUSICSolverState> solver_z;
};

namespace {

constexpr const char* kOpName = "RootMUSICEstimatorOp";
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

expected<void*, Error> prepare_runtime_buffer(
    RootMUSICDeviceBuffer& buffer, std::initializer_list<int64_t> shape,
    DLDataType dtype, const char* what) {
  const std::size_t elements = product_extents(shape, kOpName);
  if (!buffer.tensor || elements > buffer.capacity) {
    buffer.tensor = make_device_tensor(shape, dtype);
    buffer.capacity = elements;
  }
  return writable_ptr(*buffer.tensor, what);
}

struct RootMUSICStageConfig {
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<int64_t> spatial_smoothing;
  int64_t num_sources{-1};
  double source_count_pfa{1e-2};
  int64_t max_sources{1};
  bool forward_backward{true};
};

struct RootMUSICStagePlan {
  RootMUSICStageConfig stage;
  int64_t num_items{0};
  int64_t num_channels{0};
  int64_t num_snapshots{0};
  std::vector<int64_t> aoa_shape;
  std::vector<int64_t> doa_shape;
  std::vector<int64_t> count_shape;
};

expected<RootMUSICStageConfig, Error> resolve_params(
    Input<WireArrayGeometry>& geometry_port,
    Input<SpatialSmoothingOverride>& smoothing_port,
    Input<int64_t>& num_sources_port, Input<double>& pfa_port,
    Input<int64_t>& max_sources_port, const RootMUSICParams& pinned,
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
                        "RootMUSICEstimatorOp: "
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
  return RootMUSICStageConfig{
      .geometry = std::move(geometry),
      .convention = pinned.angle_convention,
      .spatial_smoothing = std::move(smoothing),
      .num_sources = resolved_num_sources,
      .source_count_pfa = pfa,
      .max_sources = resolved_max_sources,
      .forward_backward = pinned.forward_backward,
  };
}

expected<RootMUSICStagePlan, Error> build_root_music_stage_plan(
    const Tensor& snapshot, RootMUSICStageConfig stage,
    const MaxTensorExtents& extents) {
  try {
    stage.geometry.validate();
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  }
  if (extents.max_channels.has_value() &&
      stage.geometry.num_elements() > *extents.max_channels) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "RootMUSICEstimatorOp: array_geometry.num_elements exceeds "
              "MaxTensorExtents.max_channels"});
  }
  if (!(stage.source_count_pfa > 0.0) ||
      !(stage.source_count_pfa < 1.0)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "RootMUSICEstimatorOp: source_count_pfa must be in (0, 1)"});
  }
  if (stage.max_sources < 1 || stage.max_sources > kMaxSourcesCap) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "RootMUSICEstimatorOp: max_sources must be in [1, 64]"});
  }
  if (extents.max_sources.has_value() &&
      stage.max_sources > *extents.max_sources) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "RootMUSICEstimatorOp: max_sources exceeds "
              "MaxTensorExtents.max_sources"});
  }

  const auto& shape = snapshot.shape();
  if (shape.size() != 3U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "RootMUSICEstimatorOp: snapshot must have shape "
              "(num_items, num_channels, num_snapshots)"});
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape[2];
  if (extents.max_items.has_value() && num_items > *extents.max_items) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "RootMUSICEstimatorOp: num_items exceeds "
              "MaxTensorExtents.max_items"});
  }
  if (num_channels != stage.geometry.num_elements()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "RootMUSICEstimatorOp: snapshot num_channels must match "
              "array_geometry.num_elements"});
  }

  const int64_t max_sources = stage.max_sources;
  return RootMUSICStagePlan{
      .stage = std::move(stage),
      .num_items = num_items,
      .num_channels = num_channels,
      .num_snapshots = num_snapshots,
      .aoa_shape = {num_items, 2},
      .doa_shape = {num_items, max_sources, 2},
      .count_shape = {num_items},
  };
}

struct ULAParameters {
  double spacing{0.0};
  std::array<double, 3> axis{};
  std::array<double, 3> broadside{};
};

template <typename T>
struct RootMUSICEstimationInput {
  const ArrayGeometry* geometry{nullptr};
  int64_t num_channels{0};
  int64_t num_snapshots{0};
  const cuda::std::complex<T>* data{nullptr};
  const unsigned char* mask{nullptr};
  std::shared_ptr<holoscan::Tensor> snapshot_keepalive;
  std::shared_ptr<holoscan::Tensor> mask_keepalive;
};

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

template <typename T>
expected<RootMUSICEstimationInput<T>, Error> prepare_smoothed_input(
    RootMUSICRuntimeState<T>& state, const Tensor& snapshot,
    const unsigned char* mask, const RootMUSICStagePlan& plan,
    cudaStream_t stream) {
  RootMUSICEstimationInput<T> result{
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
    if (!state.smoothing_key.empty()) {
      HOLOSCAN_LOG_WARN(
          "RootMUSICEstimatorOp: spatial smoothing override changes the "
          "Root-MUSIC aperture and polynomial degree");
    }
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
                  "RootMUSICEstimatorOp: smoothing index upload failed"});
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
                    "RootMUSICEstimatorOp: smoothing flip upload failed"});
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
  HOLOSCAN_TRY(
      auto smoothed_ptr,
      prepare_runtime_buffer(
          state.smoothed_snapshot,
          {plan.num_items, subarray_size, looks}, complex_dtype<T>(),
          "smoothed_snapshot"));
  result.snapshot_keepalive = state.smoothed_snapshot.tensor;
  unsigned char* smoothed_mask_ptr = nullptr;
  if (mask != nullptr) {
    HOLOSCAN_TRY(
        auto mask_out,
        prepare_runtime_buffer(
            state.smoothed_mask, {plan.num_items, looks},
            DLDataType{kDLUInt, 8, 1}, "smoothed_mask"));
    result.mask_keepalive = state.smoothed_mask.tensor;
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

// Root-MUSIC's Vandermonde ordering requires channels to follow consecutive,
// equally spaced positions on one line. The direction associated with the
// root's spatial frequency is the forward-hemisphere point nearest canonical
// boresight (+x), which uniquely embeds the ULA's one resolved coordinate in
// the operator's two-coordinate boundary output.
ULAParameters analyze_ula(const ArrayGeometry& geometry) {
  geometry.validate();
  const int64_t channels = geometry.num_elements();
  if (channels < 2) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: array_geometry must contain at least two "
        "elements");
  }
  if (geometry.geometry_type &&
      *geometry.geometry_type != ArrayGeometryType::kUla) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: array_geometry must be a ULA");
  }

  const auto& p = geometry.element_positions;
  std::array<double, 3> delta{
      p[3] - p[0],
      p[4] - p[1],
      p[5] - p[2],
  };
  const double spacing =
      std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                delta[2] * delta[2]);
  if (!(spacing > 0.0) || !std::isfinite(spacing)) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: ULA spacing must be positive and finite");
  }
  const double scale =
      std::max(geometry.wavelength, spacing * static_cast<double>(channels));
  const double tolerance = 1e-6 * scale;
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      const double expected =
          p[static_cast<size_t>(coordinate)] +
          static_cast<double>(channel) * delta[static_cast<size_t>(coordinate)];
      const double actual =
          p[static_cast<size_t>(channel * 3 + coordinate)];
      if (std::abs(actual - expected) > tolerance) {
        throw std::invalid_argument(
            "RootMUSICEstimatorOp: array_geometry must be a uniformly spaced "
            "linear array in snapshot channel order; use "
            "SubarraySelectionOp upstream for non-uniform geometry");
      }
    }
  }

  ULAParameters result;
  result.spacing = spacing;
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    result.axis[static_cast<size_t>(coordinate)] =
        delta[static_cast<size_t>(coordinate)] / spacing;
  }

  // Normalize the component of canonical boresight orthogonal to the ULA.
  // This chooses the forward member of the cone consistent with the measured
  // one-dimensional direction cosine.
  const double projection = result.axis[0];
  result.broadside = {
      1.0 - projection * result.axis[0],
      -projection * result.axis[1],
      -projection * result.axis[2],
  };
  const double broadside_norm =
      std::sqrt(result.broadside[0] * result.broadside[0] +
                result.broadside[1] * result.broadside[1] +
                result.broadside[2] * result.broadside[2]);
  if (!(broadside_norm > 1e-8)) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: a ULA parallel to canonical boresight has no "
        "unique forward-hemisphere 1-D DOA embedding");
  }
  for (double& value : result.broadside) {
    value /= broadside_norm;
  }
  return result;
}

using TensorLoanWriter = std::remove_cvref_t<decltype(
    *std::declval<TensorOutputLoan&>().write(
        std::declval<ExecutionContext&>().cuda_stream()))>;

struct RootMUSICPreparedOutputs {
  TensorOutputLoan aoa_loan;
  TensorOutputLoan doa_loan;
  TensorOutputLoan count_loan;
  TensorOutputLoan eigenvalue_loan;
  TensorLoanWriter aoa_writer;
  TensorLoanWriter doa_writer;
  TensorLoanWriter count_writer;
  TensorLoanWriter eigenvalue_writer;
  std::vector<std::shared_ptr<holoscan::Tensor>> workspace_keepalive;
};

template <typename T>
struct RootMUSICPreparedWorkspace {
  std::shared_ptr<holoscan::Tensor> covariance;
  std::shared_ptr<holoscan::Tensor> eigenvectors;
  std::shared_ptr<holoscan::Tensor> projector;
  std::shared_ptr<holoscan::Tensor> valid_counts;
  std::shared_ptr<holoscan::Tensor> model_order;
  cuda::std::complex<T>* covariance_ptr{nullptr};
  cuda::std::complex<T>* eigenvectors_ptr{nullptr};
  cuda::std::complex<T>* projector_ptr{nullptr};
  int64_t* valid_counts_ptr{nullptr};
  int32_t* model_order_ptr{nullptr};
};

expected<void, Error> emit_empty_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& doa_port,
    Output<Tensor>& count_port, Output<Tensor>& eigenvalue_port,
    ExecutionContext& context, const RootMUSICStagePlan& plan,
    DLDataType output_dtype) {
  const std::vector<int64_t> eig_shape = {
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
      prepare_device_tensor(eigenvalue_port, context, eig_shape,
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
expected<RootMUSICPreparedWorkspace<T>, Error>
prepare_root_music_workspace(RootMUSICRuntimeState<T>& state,
                             int64_t num_items,
                             int64_t estimation_channels) {
  const DLDataType cplx_dtype = complex_dtype<T>();
  const DLDataType i32_dtype{kDLInt, 32, 1};
  const DLDataType i64_dtype{kDLInt, 64, 1};
  RootMUSICPreparedWorkspace<T> result;
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
  result.projector =
      state.projector
          ? state.projector
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
      auto projector,
      writable_ptr(*result.projector, "projector"));
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
  result.projector_ptr =
      static_cast<cuda::std::complex<T>*>(projector);
  result.valid_counts_ptr = static_cast<int64_t*>(valid_counts);
  result.model_order_ptr = static_cast<int32_t*>(model_order);
  return result;
}

template <typename T>
expected<detail::RootMUSICAxisWorkspace<T>, Error>
prepare_root_music_axis_workspace(
    RootMUSICAxisRuntimeBuffers<T>& buffers, int64_t num_items,
    int64_t axis_channels) {
  const int64_t degree = 2 * (axis_channels - 1);
  HOLOSCAN_TRY(
      auto eigenvectors,
      prepare_runtime_buffer(
          buffers.eigenvectors,
          {num_items, axis_channels, axis_channels}, complex_dtype<T>(),
          "axis eigenvectors"));
  HOLOSCAN_TRY(
      auto eigenvalues,
      prepare_runtime_buffer(
          buffers.eigenvalues, {num_items, axis_channels}, real_dtype<T>(),
          "axis eigenvalues"));
  HOLOSCAN_TRY(
      auto projector,
      prepare_runtime_buffer(
          buffers.projector,
          {num_items, axis_channels, axis_channels}, complex_dtype<T>(),
          "axis projector"));
  HOLOSCAN_TRY(
      auto model_order,
      prepare_runtime_buffer(
          buffers.model_order, {num_items}, DLDataType{kDLInt, 32, 1},
          "axis model order"));
  HOLOSCAN_TRY(
      auto coefficients,
      prepare_runtime_buffer(
          buffers.coefficients, {num_items, degree + 1}, complex_dtype<T>(),
          "axis coefficients"));
  HOLOSCAN_TRY(
      auto companion,
      prepare_runtime_buffer(
          buffers.companion, {num_items, degree, degree}, complex_dtype<T>(),
          "axis companion"));
  HOLOSCAN_TRY(
      auto roots,
      prepare_runtime_buffer(
          buffers.roots, {num_items, degree}, complex_dtype<T>(),
          "axis roots"));
  return detail::RootMUSICAxisWorkspace<T>{
      .eigenvectors =
          static_cast<cuda::std::complex<T>*>(eigenvectors),
      .eigenvalues = static_cast<T*>(eigenvalues),
      .projector = static_cast<cuda::std::complex<T>*>(projector),
      .model_order = static_cast<int32_t*>(model_order),
      .coefficients =
          static_cast<cuda::std::complex<T>*>(coefficients),
      .companion = static_cast<cuda::std::complex<T>*>(companion),
      .roots = static_cast<cuda::std::complex<T>*>(roots),
  };
}

template <typename T>
expected<detail::RootMUSICURAScratch<T>, Error>
prepare_root_music_ura_scratch(RootMUSICRuntimeState<T>& state,
                               int64_t num_items, int64_t ny, int64_t nz) {
  HOLOSCAN_TRY(
      auto cov_y,
      prepare_runtime_buffer(
          state.ura.cov_y, {num_items, ny, ny}, complex_dtype<T>(),
          "ura cov_y"));
  HOLOSCAN_TRY(
      auto cov_z,
      prepare_runtime_buffer(
          state.ura.cov_z, {num_items, nz, nz}, complex_dtype<T>(),
          "ura cov_z"));
  HOLOSCAN_TRY(
      auto axis_y,
      prepare_root_music_axis_workspace<T>(
          state.ura.axis_y, num_items, ny));
  HOLOSCAN_TRY(
      auto axis_z,
      prepare_root_music_axis_workspace<T>(
          state.ura.axis_z, num_items, nz));

  const int64_t u_cap =
      std::min<int64_t>(ny - 1, detail::kRootMusicPairCandCap);
  const int64_t v_cap =
      std::min<int64_t>(nz - 1, detail::kRootMusicPairCandCap);
  HOLOSCAN_TRY(
      auto u_candidates,
      prepare_runtime_buffer(
          state.ura.u_candidates, {num_items, u_cap}, real_dtype<T>(),
          "ura u candidates"));
  HOLOSCAN_TRY(
      auto v_candidates,
      prepare_runtime_buffer(
          state.ura.v_candidates, {num_items, v_cap}, real_dtype<T>(),
          "ura v candidates"));
  HOLOSCAN_TRY(
      auto u_counts,
      prepare_runtime_buffer(
          state.ura.u_counts, {num_items}, DLDataType{kDLInt, 32, 1},
          "ura u counts"));
  HOLOSCAN_TRY(
      auto v_counts,
      prepare_runtime_buffer(
          state.ura.v_counts, {num_items}, DLDataType{kDLInt, 32, 1},
          "ura v counts"));
  return detail::RootMUSICURAScratch<T>{
      .cov_y = static_cast<cuda::std::complex<T>*>(cov_y),
      .cov_z = static_cast<cuda::std::complex<T>*>(cov_z),
      .axis_y = axis_y,
      .axis_z = axis_z,
      .u_candidates = static_cast<T*>(u_candidates),
      .v_candidates = static_cast<T*>(v_candidates),
      .u_counts = static_cast<int32_t*>(u_counts),
      .v_counts = static_cast<int32_t*>(v_counts),
  };
}

expected<RootMUSICPreparedOutputs, Error> prepare_root_music_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& doa_port,
    Output<Tensor>& count_port, Output<Tensor>& eigenvalue_port,
    ExecutionContext& context, const RootMUSICStagePlan& plan,
    const std::vector<int64_t>& eigenvalue_shape,
    DLDataType output_dtype) {
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
  return RootMUSICPreparedOutputs{
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

template <typename T>
expected<void, Error> execute_root_music_ura(
    RootMUSICRuntimeState<T>& state,
    const RootMUSICStagePlan& plan,
    const RootMUSICEstimationInput<T>& estimation,
    const PlanarAperturePlan& planar,
    const detail::DeviceAngleConvention<T>& convention,
    RootMUSICPreparedWorkspace<T>& workspace,
    RootMUSICPreparedOutputs& outputs, bool warn_new_geometry,
    cudaStream_t stream) {
  if (estimation.num_channels > detail::kRootMusicPairChannelCap) {
    return fail_invalid(
        "RootMUSICEstimatorOp: 2-D URA pairing supports at most 64 channels "
        "(Ny*Nz); reduce the aperture upstream");
  }
  const int64_t axis_min = std::min(planar.ny, planar.nz);
  if (plan.stage.num_sources >= 0 &&
      plan.stage.num_sources > axis_min - 1) {
    return fail_invalid(
        "RootMUSICEstimatorOp: num_sources must leave at least one "
        "noise-subspace dimension on each URA aperture axis");
  }
  if (warn_new_geometry) {
    HOLOSCAN_LOG_WARN(
        "RootMUSICEstimatorOp: unseen array_geometry override; "
        "revalidating URA aperture and solver dimensions");
  }
  HOLOSCAN_TRY(
      auto scratch,
      prepare_root_music_ura_scratch<T>(
          state, plan.num_items, planar.ny, planar.nz));
  const double k = kTwoPi / estimation.geometry->wavelength;
  try {
    detail::run_root_music_2d_stage<T>(
        estimation.data, estimation.mask, plan.num_items, planar.ny,
        planar.nz, estimation.num_snapshots, plan.stage.num_sources,
        plan.stage.source_count_pfa, plan.stage.max_sources,
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
        workspace.covariance_ptr, workspace.eigenvectors_ptr,
        workspace.projector_ptr, workspace.valid_counts_ptr,
        workspace.model_order_ptr,
        static_cast<T*>(outputs.eigenvalue_writer.data()),
        static_cast<T*>(outputs.aoa_writer.data()),
        static_cast<T*>(outputs.doa_writer.data()),
        static_cast<int32_t*>(outputs.count_writer.data()),
        scratch, state.solver.get(), state.solver_z.get(), stream);
    check_cuda("root music stage", kOpName);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  } catch (const std::runtime_error& error) {
    return fail_runtime(error.what());
  }
  return {};
}

template <typename T>
expected<void, Error> execute_root_music_ula(
    RootMUSICRuntimeState<T>& state,
    const RootMUSICStagePlan& plan,
    const RootMUSICEstimationInput<T>& estimation,
    const detail::DeviceAngleConvention<T>& convention,
    RootMUSICPreparedWorkspace<T>& workspace,
    RootMUSICPreparedOutputs& outputs, bool warn_new_geometry,
    cudaStream_t stream) {
  ULAParameters ula;
  try {
    ula = analyze_ula(*estimation.geometry);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }
  if (plan.stage.num_sources >= 0 &&
      plan.stage.num_sources > estimation.num_channels - 1) {
    return fail_invalid(
        "RootMUSICEstimatorOp: num_sources must leave at least one "
        "noise-subspace dimension");
  }
  if (warn_new_geometry) {
    HOLOSCAN_LOG_WARN(
        "RootMUSICEstimatorOp: unseen array_geometry override; "
        "revalidating ULA and solver dimensions");
  }

  const DLDataType cplx_dtype = complex_dtype<T>();
  const int64_t degree = 2 * (estimation.num_channels - 1);
  auto coefficients =
      state.coefficients
          ? state.coefficients
          : make_device_tensor(
                {plan.num_items, degree + 1}, cplx_dtype);
  auto companion =
      state.companion
          ? state.companion
          : make_device_tensor(
                {plan.num_items, degree, degree}, cplx_dtype);
  auto roots =
      state.roots
          ? state.roots
          : make_device_tensor({plan.num_items, degree}, cplx_dtype);
  HOLOSCAN_TRY(
      auto coefficients_ptr,
      writable_ptr(*coefficients, "coefficients"));
  HOLOSCAN_TRY(
      auto companion_ptr,
      writable_ptr(*companion, "companion"));
  HOLOSCAN_TRY(auto roots_ptr, writable_ptr(*roots, "roots"));

  try {
    detail::run_root_music_stage<T>(
        estimation.data, estimation.mask, plan.num_items,
        estimation.num_channels, estimation.num_snapshots,
        plan.stage.num_sources, plan.stage.source_count_pfa,
        plan.stage.max_sources,
        static_cast<T>(kTwoPi * ula.spacing /
                       estimation.geometry->wavelength),
        static_cast<T>(ula.axis[0]), static_cast<T>(ula.axis[1]),
        static_cast<T>(ula.axis[2]),
        static_cast<T>(ula.broadside[0]),
        static_cast<T>(ula.broadside[1]),
        static_cast<T>(ula.broadside[2]), convention,
        workspace.covariance_ptr, workspace.eigenvectors_ptr,
        workspace.projector_ptr, workspace.valid_counts_ptr,
        workspace.model_order_ptr,
        static_cast<T*>(outputs.eigenvalue_writer.data()),
        static_cast<cuda::std::complex<T>*>(coefficients_ptr),
        static_cast<cuda::std::complex<T>*>(companion_ptr),
        static_cast<cuda::std::complex<T>*>(roots_ptr),
        static_cast<T*>(outputs.aoa_writer.data()),
        static_cast<T*>(outputs.doa_writer.data()),
        static_cast<int32_t*>(outputs.count_writer.data()),
        state.solver.get(), stream);
    check_cuda("root music stage", kOpName);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  } catch (const std::runtime_error& error) {
    return fail_runtime(error.what());
  }
  outputs.workspace_keepalive = {
      std::move(coefficients), std::move(companion), std::move(roots)};
  return {};
}

expected<void, Error> commit_and_publish_root_music_outputs(
    Output<Tensor>& aoa_port, Output<Tensor>& doa_port,
    Output<Tensor>& count_port, Output<Tensor>& eigenvalue_port,
    ExecutionContext& context, RootMUSICPreparedOutputs outputs) {
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

}  // namespace

template <typename T>
RootMUSICEstimatorOp<T>::RootMUSICEstimatorOp(
    const MaxTensorExtents& extents) {
  state_ = std::make_shared<RootMUSICRuntimeState<T>>();
  state_->solver = std::make_shared<detail::RootMUSICSolverState>();
  state_->solver_z = std::make_shared<detail::RootMUSICSolverState>();
  extents_ = extents;
  if (!extents.max_items.has_value() &&
      !extents.max_sources.has_value()) {
    return;
  }
  const char* who = "RootMUSICEstimatorOp";
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
  state_->projector = make_device_tensor(
      {max_items, max_channels, max_channels}, complex_dtype<T>());
  state_->valid_counts =
      make_device_tensor({max_items}, DLDataType{kDLInt, 64, 1});
  state_->model_order =
      make_device_tensor({max_items}, DLDataType{kDLInt, 32, 1});
  const int64_t degree = 2 * (max_channels - 1);
  state_->coefficients =
      make_device_tensor({max_items, degree + 1}, complex_dtype<T>());
  state_->companion =
      make_device_tensor({max_items, degree, degree}, complex_dtype<T>());
  state_->roots =
      make_device_tensor({max_items, degree}, complex_dtype<T>());
}

template <typename T>
void RootMUSICEstimatorOp<T>::stop() {
  detail::retire_music_matx_caches();
}

template <typename T>
void RootMUSICEstimatorOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr std::uint32_t kMaxRank = 3U;
  spec.input(snapshot).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(is_valid)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kDevice));
  spec.output(aoa_estimates)
      .produces_tensor(pooled_tensor_output(
          kDevice, kMaxRank, aoa_max_byte_span_));
  // Optional in OPERATOR_API.md — always emitted so unwired consumers do not
  // need Connectivity gating; no angle_spectrum (gridless).
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
  spec.input(array_geometry).optional().queue_depth(2U);
  spec.input(spatial_smoothing).optional().queue_depth(2U);
  spec.input(num_sources).optional().queue_depth(2U);
  spec.input(source_count_pfa).optional().queue_depth(2U);
  spec.input(max_sources).optional().queue_depth(2U);
}

template <typename T>
Contract RootMUSICEstimatorOp<T>::contract() const {
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
expected<void, Error> RootMUSICEstimatorOp<T>::compute(
    ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
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

  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const RootMUSICParams& pinned = context.params<RootMUSICParams>();
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(array_geometry, spatial_smoothing, num_sources,
                     source_count_pfa, max_sources, pinned, extents_));
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(
      auto plan,
      build_root_music_stage_plan(snap, std::move(stage), extents_));

  if (state_->seeded_angle_convention.has_value() &&
      *state_->seeded_angle_convention != plan.stage.convention) {
    return fail_invalid(
        "RootMUSICEstimatorOp: angle_convention is static and cannot change "
        "after the first successful activation; instantiate a separate "
        "operator per convention");
  }

  // 2. Validate the optional mask and runtime stream, then prepare the
  // convention representation that becomes permanent only after publication.
  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  const unsigned char* mask = nullptr;
  if (maybe_valid.has_value()) {
    const DLDataType dtype = maybe_valid->dtype();
    if (!((dtype.code == kDLBool || dtype.code == kDLUInt) &&
          dtype.bits == 8 && dtype.lanes == 1)) {
      return fail_invalid(
          "RootMUSICEstimatorOp: is_valid must have dtype bool or uint8");
    }
    if (maybe_valid->shape().size() != 2 ||
        maybe_valid->shape()[0] != plan.num_items ||
        maybe_valid->shape()[1] != plan.num_snapshots) {
      return fail_invalid(
          "RootMUSICEstimatorOp: is_valid must have shape "
          "(num_items, num_snapshots) matching snapshot");
    }
    try {
      require_cuda_accessible(*maybe_valid, "is_valid", kOpName);
      require_contiguous(*maybe_valid, "is_valid", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    mask = static_cast<const unsigned char*>(maybe_valid->data());
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "RootMUSICEstimatorOp: no CUDA compute stream"});
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
  // covariance, eigendecomposition, or polynomial-root work.
  // Publish coherent outputs in dependency order and retain borrowed inputs.
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

  // 4. Form the effective covariance input and classify its reduced or
  // original aperture as ULA or filled URA.
  // Allocate exact-sized outputs before launching device work.
  HOLOSCAN_TRY(
      auto estimation,
      prepare_smoothed_input(*state_, snap, mask, plan, stream));

  PlanarAperturePlan planar;
  try {
    planar = analyze_planar_aperture(*estimation.geometry);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const std::string gkey = geometry_key(plan.stage.geometry);
  const bool geometry_is_new =
      state_->seen_geometry_keys.insert(gkey).second;
  const bool warn_new_geometry =
      geometry_is_new && state_->seen_geometry_keys.size() > 1;

  // 5. Prepare covariance/eigenspace workspace and the complete output set
  // shared by both Root-MUSIC variants.
  HOLOSCAN_TRY(
      auto workspace,
      prepare_root_music_workspace<T>(
          *state_, plan.num_items, estimation.num_channels));
  const std::vector<int64_t> eigenvalue_shape = {
      plan.num_items, estimation.num_channels};
  HOLOSCAN_TRY(
      auto outputs,
      prepare_root_music_outputs(
          aoa_estimates, doa_multi, source_count, eigenvalues,
          context, plan, eigenvalue_shape, real_dtype<T>()));

  // 6. Dispatch the geometry-specific algorithm: one companion polynomial for
  // a ULA, or reduced-dimension per-axis rooting plus 2-D pairing for a URA.
  // Execute the device stage on the input runtime stream.
  if (planar.is_ura) {
    HOLOSCAN_TRY(
        execute_root_music_ura(
            *state_, plan, estimation, planar, *convention, workspace,
            outputs, warn_new_geometry, stream));
  } else {
    HOLOSCAN_TRY(
        execute_root_music_ula(
            *state_, plan, estimation, *convention, workspace, outputs,
            warn_new_geometry, stream));
  }

  if (plan.stage.num_sources < 0 && mask == nullptr &&
      estimation.num_snapshots < estimation.num_channels &&
      !state_->warned_rank_deficient) {
    HOLOSCAN_LOG_WARN(
        "RootMUSICEstimatorOp: num_sources=auto with fewer snapshots than "
        "channels; falling back to one source");
    state_->warned_rank_deficient = true;
  }

  // 7. Commit every sibling before required-first publication, lock the
  // convention after success, and retain asynchronous inputs.
  HOLOSCAN_TRY(
      commit_and_publish_root_music_outputs(
          aoa_estimates, doa_multi, source_count, eigenvalues,
          context, std::move(outputs)));
  seed_angle_convention_after_emit();
  retained_snapshot_ = std::move(snap);
  if (maybe_valid.has_value()) {
    retained_is_valid_ = std::move(*maybe_valid);
  }
  return {};
}

template class RootMUSICEstimatorOp<float>;
template class RootMUSICEstimatorOp<double>;

}  // namespace holoscan::radar
