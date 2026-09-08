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
  std::shared_ptr<holoscan::Tensor> covariance;
  std::shared_ptr<holoscan::Tensor> eigenvectors;
  std::shared_ptr<holoscan::Tensor> projector;
  std::shared_ptr<holoscan::Tensor> valid_counts;
  std::shared_ptr<holoscan::Tensor> model_order;
  std::shared_ptr<holoscan::Tensor> coefficients;
  std::shared_ptr<holoscan::Tensor> companion;
  std::shared_ptr<holoscan::Tensor> roots;
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
  result.snapshot_keepalive = make_device_tensor(
      {plan.num_items, subarray_size, looks}, complex_dtype<T>());
  HOLOSCAN_TRY(
      auto smoothed_ptr,
      writable_ptr(*result.snapshot_keepalive, "smoothed_snapshot"));
  unsigned char* smoothed_mask_ptr = nullptr;
  if (mask != nullptr) {
    result.mask_keepalive = make_device_tensor(
        {plan.num_items, looks}, DLDataType{kDLUInt, 8, 1});
    HOLOSCAN_TRY(
        auto mask_out,
        writable_ptr(*result.mask_keepalive, "smoothed_mask"));
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
  spec.input(is_valid).queue_depth(2U).expects_tensor(tensor_input(kDevice));
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
  spec.input(array_geometry).queue_depth(2U);
  spec.input(spatial_smoothing).queue_depth(2U);
  spec.input(num_sources).queue_depth(2U);
  spec.input(source_count_pfa).queue_depth(2U);
  spec.input(max_sources).queue_depth(2U);
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
  const RootMUSICParams& pinned = context.params<RootMUSICParams>();
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(array_geometry, spatial_smoothing, num_sources,
                     source_count_pfa, max_sources, pinned, extents_));
  HOLOSCAN_TRY(
      auto plan,
      build_root_music_stage_plan(snap, std::move(stage), extents_));
  auto& geometry = plan.stage.geometry;
  const int64_t effective_num_sources = plan.stage.num_sources;
  const double effective_pfa = plan.stage.source_count_pfa;
  const int64_t effective_max_sources = plan.stage.max_sources;

  if (state_->seeded_angle_convention.has_value() &&
      *state_->seeded_angle_convention != plan.stage.convention) {
    return fail_invalid(
        "RootMUSICEstimatorOp: angle_convention is static and cannot change "
        "after the first successful activation; instantiate a separate "
        "operator per convention");
  }

  const int64_t num_items = plan.num_items;
  const int64_t num_channels = plan.num_channels;
  const int64_t num_snapshots = plan.num_snapshots;

  // Validate the optional same-tick mask against the planned snapshot axes.
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
        maybe_valid->shape()[0] != num_items ||
        maybe_valid->shape()[1] != num_snapshots) {
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

  // Resolve the runtime stream before touching cached device artifacts.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "RootMUSICEstimatorOp: no CUDA compute stream"});
  }

  // Tick-local convention until the first successful emit locks the static
  // seed — geometry / subspace validation below can still fail this tick.
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

  // Prepare the effective covariance input: either the original snapshot or
  // gathered overlapping looks on the reduced smoothing aperture.
  HOLOSCAN_TRY(
      auto estimation,
      prepare_smoothed_input(*state_, snap, mask, plan, stream));
  const ArrayGeometry* estimation_geometry = estimation.geometry;
  const int64_t estimation_channels = estimation.num_channels;
  const int64_t estimation_snapshots = estimation.num_snapshots;
  const cuda::std::complex<T>* estimation_data = estimation.data;
  const unsigned char* estimation_mask = estimation.mask;

  PlanarAperturePlan planar;
  try {
    planar = analyze_planar_aperture(*estimation_geometry);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const std::string gkey = geometry_key(geometry);
  const bool geometry_is_new =
      state_->seen_geometry_keys.insert(gkey).second;
  const bool warn_new_geometry =
      geometry_is_new && state_->seen_geometry_keys.size() > 1;

  const std::vector<int64_t> eig_shape = {num_items, estimation_channels};
  auto covariance = state_->covariance
                        ? state_->covariance
                        : make_device_tensor(
                              {num_items, estimation_channels,
                               estimation_channels},
                              cplx_dtype);
  auto eigenvectors = state_->eigenvectors
                          ? state_->eigenvectors
                          : make_device_tensor(
                                {num_items, estimation_channels,
                                 estimation_channels},
                                cplx_dtype);
  auto projector = state_->projector
                       ? state_->projector
                       : make_device_tensor(
                             {num_items, estimation_channels,
                              estimation_channels},
                             cplx_dtype);
  auto valid_counts = state_->valid_counts
                          ? state_->valid_counts
                          : make_device_tensor({num_items}, i64_dtype);
  auto model_order = state_->model_order
                         ? state_->model_order
                         : make_device_tensor({num_items}, i32_dtype);
  HOLOSCAN_TRY(auto cov_ptr, writable_ptr(*covariance, "covariance"));
  HOLOSCAN_TRY(auto evec_ptr, writable_ptr(*eigenvectors, "eigenvectors"));
  HOLOSCAN_TRY(auto proj_ptr, writable_ptr(*projector, "projector"));
  HOLOSCAN_TRY(auto valid_ptr, writable_ptr(*valid_counts, "valid_counts"));
  HOLOSCAN_TRY(auto model_ptr, writable_ptr(*model_order, "model_order"));

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

  if (planar.is_ura) {
    if (estimation_channels > detail::kRootMusicPairChannelCap) {
      return fail_invalid(
          "RootMUSICEstimatorOp: 2-D URA pairing supports at most 64 channels "
          "(Ny*Nz); reduce the aperture upstream");
    }
    const int64_t axis_min = std::min(planar.ny, planar.nz);
    if (effective_num_sources >= 0 &&
        effective_num_sources > axis_min - 1) {
      return fail_invalid(
          "RootMUSICEstimatorOp: num_sources must leave at least one "
          "noise-subspace dimension on each URA aperture axis");
    }
    if (warn_new_geometry) {
      HOLOSCAN_LOG_WARN(
          "RootMUSICEstimatorOp: unseen array_geometry override; "
          "revalidating URA aperture and solver dimensions");
    }
    const double k = kTwoPi / estimation_geometry->wavelength;
    try {
      detail::run_root_music_2d_stage<T>(
          estimation_data, estimation_mask, num_items, planar.ny, planar.nz,
          estimation_snapshots, effective_num_sources, effective_pfa,
          effective_max_sources, static_cast<T>(k * planar.dy),
          static_cast<T>(k * planar.dz), static_cast<T>(planar.y_axis[0]),
          static_cast<T>(planar.y_axis[1]), static_cast<T>(planar.y_axis[2]),
          static_cast<T>(planar.z_axis[0]), static_cast<T>(planar.z_axis[1]),
          static_cast<T>(planar.z_axis[2]), static_cast<T>(planar.normal[0]),
          static_cast<T>(planar.normal[1]), static_cast<T>(planar.normal[2]),
          *convention,
          static_cast<cuda::std::complex<T>*>(cov_ptr),
          static_cast<cuda::std::complex<T>*>(evec_ptr),
          static_cast<cuda::std::complex<T>*>(proj_ptr),
          static_cast<int64_t*>(valid_ptr),
          static_cast<int32_t*>(model_ptr),
          static_cast<T*>(eig_writer->data()),
          static_cast<T*>(aoa_writer->data()),
          static_cast<T*>(doa_writer->data()),
          static_cast<int32_t*>(cnt_writer->data()), state_->solver.get(),
          state_->solver_z.get(), stream);
      check_cuda("root music stage", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  } else {
    ULAParameters ula;
    try {
      ula = analyze_ula(*estimation_geometry);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    if (effective_num_sources >= 0 &&
        effective_num_sources > estimation_channels - 1) {
      return fail_invalid(
          "RootMUSICEstimatorOp: num_sources must leave at least one "
          "noise-subspace dimension");
    }
    if (warn_new_geometry) {
      HOLOSCAN_LOG_WARN(
          "RootMUSICEstimatorOp: unseen array_geometry override; "
          "revalidating ULA and solver dimensions");
    }
    const int64_t degree = 2 * (estimation_channels - 1);
    auto coefficients =
        state_->coefficients
            ? state_->coefficients
            : make_device_tensor({num_items, degree + 1}, cplx_dtype);
    auto companion =
        state_->companion
            ? state_->companion
            : make_device_tensor({num_items, degree, degree}, cplx_dtype);
    auto roots = state_->roots
                     ? state_->roots
                     : make_device_tensor({num_items, degree}, cplx_dtype);
    HOLOSCAN_TRY(auto coef_ptr, writable_ptr(*coefficients, "coefficients"));
    HOLOSCAN_TRY(auto companion_ptr, writable_ptr(*companion, "companion"));
    HOLOSCAN_TRY(auto roots_ptr, writable_ptr(*roots, "roots"));
    try {
      detail::run_root_music_stage<T>(
          estimation_data, estimation_mask, num_items, estimation_channels,
          estimation_snapshots, effective_num_sources, effective_pfa,
          effective_max_sources,
          static_cast<T>(kTwoPi * ula.spacing /
                         estimation_geometry->wavelength),
          static_cast<T>(ula.axis[0]), static_cast<T>(ula.axis[1]),
          static_cast<T>(ula.axis[2]), static_cast<T>(ula.broadside[0]),
          static_cast<T>(ula.broadside[1]), static_cast<T>(ula.broadside[2]),
          *convention,
          static_cast<cuda::std::complex<T>*>(cov_ptr),
          static_cast<cuda::std::complex<T>*>(evec_ptr),
          static_cast<cuda::std::complex<T>*>(proj_ptr),
          static_cast<int64_t*>(valid_ptr),
          static_cast<int32_t*>(model_ptr),
          static_cast<T*>(eig_writer->data()),
          static_cast<cuda::std::complex<T>*>(coef_ptr),
          static_cast<cuda::std::complex<T>*>(companion_ptr),
          static_cast<cuda::std::complex<T>*>(roots_ptr),
          static_cast<T*>(aoa_writer->data()),
          static_cast<T*>(doa_writer->data()),
          static_cast<int32_t*>(cnt_writer->data()), state_->solver.get(),
          stream);
      check_cuda("root music stage", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  }

  if (effective_num_sources < 0 && mask == nullptr &&
      estimation_snapshots < estimation_channels &&
      !state_->warned_rank_deficient) {
    HOLOSCAN_LOG_WARN(
        "RootMUSICEstimatorOp: num_sources=auto with fewer snapshots than "
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

template class RootMUSICEstimatorOp<float>;
template class RootMUSICEstimatorOp<double>;

}  // namespace holoscan::radar
