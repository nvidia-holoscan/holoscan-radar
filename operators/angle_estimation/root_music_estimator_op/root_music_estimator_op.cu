// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
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

struct ULAParameters {
  double spacing{0.0};
  std::array<double, 3> axis{};
  std::array<double, 3> broadside{};
};

std::string geometry_key(const ArrayGeometry& geometry) {
  std::ostringstream key;
  key << geometry.wavelength << '|';
  for (double value : geometry.element_positions) {
    key << value << ',';
  }
  return key.str();
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
    const MaxTensorExtents& extents)
    : extents_(extents) {
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
}

template <typename T>
void RootMUSICEstimatorOp<T>::stop() {
  detail::retire_music_matx_caches();
}

template <typename T>
void RootMUSICEstimatorOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr std::uint32_t kMaxRank = 3U;
  spec.input(snapshot).queue_depth(2U).expects_tensor(kDevice);
  // Optional is_valid: omitted from contract so unconnected graphs compile.
  spec.input(is_valid).queue_depth(2U).expects_tensor(kDevice);
  spec.output(aoa_estimates)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = kMaxRank,
          .max_byte_span = aoa_max_byte_span_});
  // Optional in OPERATOR_API.md — always emitted so unwired consumers do not
  // need Connectivity gating; no angle_spectrum (gridless).
  spec.output(doa_multi)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = kMaxRank,
          .max_byte_span = doa_max_byte_span_});
  spec.output(source_count)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = kMaxRank,
          .max_byte_span = count_max_byte_span_});
  spec.output(eigenvalues)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = kMaxRank,
          .max_byte_span = eig_max_byte_span_});
  // [live] POD sticky overrides. Leaving unconnected is legal — compute()
  // seeds from Params. Descriptor / vector live params are Params-only (§9).
  spec.input(num_sources).queue_depth(2U);
  spec.input(source_count_pfa).queue_depth(2U);
  spec.input(max_sources).queue_depth(2U);
}

template <typename T>
Contract RootMUSICEstimatorOp<T>::contract() const {
  // Trigger on the data tensor only. Optional is_valid and [live] POD
  // overrides are omitted from the temporal contract so unconnected ports do
  // not yield TEMPORAL_INPUT_UNCONNECTED (same pattern as MUSIC / MVDR).
  Contract result;
  result.trigger(OnEach{snapshot});
  return result;
}

template <typename T>
expected<void, Error> RootMUSICEstimatorOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));

  const RootMUSICParams& pinned = context.params<RootMUSICParams>();
  HOLOSCAN_TRY(auto effective_num_sources,
               resolve_sticky_override(num_sources, pinned.num_sources,
                                       sticky_num_sources_));
  HOLOSCAN_TRY(auto effective_pfa,
               resolve_sticky_override(source_count_pfa,
                                       pinned.source_count_pfa,
                                       sticky_source_count_pfa_));
  HOLOSCAN_TRY(auto effective_max_sources,
               resolve_sticky_override(max_sources, pinned.max_sources,
                                       sticky_max_sources_));

  // Descriptor / vector live params are Params-only on EA (HSDK5_PORT §9).
  ArrayGeometry geometry = pinned.array_geometry;
  try {
    geometry.validate();
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const std::vector<int64_t>& smoothing = pinned.spatial_smoothing;
  const bool forward_backward = pinned.forward_backward;

  if (seeded_angle_convention_.has_value() &&
      *seeded_angle_convention_ != pinned.angle_convention) {
    return fail_invalid(
        "RootMUSICEstimatorOp: angle_convention is static and cannot change "
        "after the first successful activation; instantiate a separate "
        "operator per convention");
  }

  if (!(effective_pfa > 0.0) || !(effective_pfa < 1.0)) {
    return fail_invalid(
        "RootMUSICEstimatorOp: source_count_pfa must be in (0, 1)");
  }
  if (effective_max_sources < 1 || effective_max_sources > kMaxSourcesCap) {
    return fail_invalid(
        "RootMUSICEstimatorOp: max_sources must be in [1, 64]");
  }

  try {
    require_cuda_accessible(snap, "snapshot", kOpName);
    require_contiguous(snap, "snapshot", kOpName);
    require_dtype(snap, complex_dtype<T>(), "snapshot", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const auto& shape = snap.shape();
  if (shape.size() != 3) {
    return fail_invalid(
        "RootMUSICEstimatorOp: snapshot must have shape "
        "(num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape[2];
  if (num_channels != geometry.num_elements()) {
    return fail_invalid(
        "RootMUSICEstimatorOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

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
      &device_angle_convention_;
  if (!seeded_angle_convention_.has_value()) {
    tick_convention =
        detail::make_device_angle_convention<T>(pinned.angle_convention);
    convention = &tick_convention;
  }
  auto seed_angle_convention_after_emit = [&]() {
    if (!seeded_angle_convention_.has_value()) {
      device_angle_convention_ = *convention;
      seeded_angle_convention_ = pinned.angle_convention;
    }
  };

  const DLDataType out_dtype = real_dtype<T>();
  const DLDataType cplx_dtype = complex_dtype<T>();
  const DLDataType i32_dtype{kDLInt, 32, 1};
  const DLDataType i64_dtype{kDLInt, 64, 1};
  const std::vector<int64_t> aoa_shape = {num_items, 2};
  const std::vector<int64_t> doa_shape = {num_items, effective_max_sources, 2};
  const std::vector<int64_t> count_shape = {num_items};

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
    HOLOSCAN_TRY(
        emit_prepared_tensor(doa_multi, context, std::move(doa_loan)));
    HOLOSCAN_TRY(
        emit_prepared_tensor(source_count, context, std::move(cnt_loan)));
    HOLOSCAN_TRY(
        emit_prepared_tensor(eigenvalues, context, std::move(eig_loan)));
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

  const ArrayGeometry* estimation_geometry = &geometry;
  int64_t estimation_channels = num_channels;
  int64_t estimation_snapshots = num_snapshots;
  const auto* estimation_data =
      static_cast<const cuda::std::complex<T>*>(snap.data());
  const unsigned char* estimation_mask = mask;
  std::shared_ptr<holoscan::Tensor> smoothed_snapshot;
  std::shared_ptr<holoscan::Tensor> smoothed_mask;

  if (!smoothing.empty()) {
    std::ostringstream key;
    key << geometry_key(geometry) << '|';
    for (int64_t count : smoothing) {
      key << count << ',';
    }
    key << '|' << (forward_backward ? 1 : 0);
    if (key.str() != smoothing_key_) {
      if (!smoothing_key_.empty()) {
        HOLOSCAN_LOG_WARN(
            "RootMUSICEstimatorOp: spatial smoothing override changes the "
            "Root-MUSIC aperture and polynomial degree");
      }
      try {
        smoothing_plan_ = build_spatial_smoothing_plan(
            geometry, smoothing, forward_backward);
      } catch (const std::invalid_argument& e) {
        return fail_invalid(e.what());
      } catch (const std::runtime_error& e) {
        return fail_runtime(e.what());
      }
      smoothing_key_ = key.str();
      if (smoothing_plan_.enabled) {
        const int64_t subarrays = smoothing_plan_.num_subarrays;
        const int64_t subarray_size = smoothing_plan_.subarray_size;
        d_subarray_index_ =
            make_device_tensor({subarrays * subarray_size},
                               DLDataType{kDLInt, 32, 1});
        HOLOSCAN_TRY(auto idx_ptr,
                     writable_ptr(*d_subarray_index_, "subarray_index"));
        if (cudaMemcpyAsync(
                idx_ptr, smoothing_plan_.subarray_index.data(),
                smoothing_plan_.subarray_index.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          return fail_runtime(
              "RootMUSICEstimatorOp: smoothing index upload failed");
        }
        if (smoothing_plan_.forward_backward) {
          d_fb_flip_ = make_device_tensor({subarray_size},
                                          DLDataType{kDLInt, 32, 1});
          HOLOSCAN_TRY(auto flip_ptr, writable_ptr(*d_fb_flip_, "fb_flip"));
          if (cudaMemcpyAsync(
                  flip_ptr, smoothing_plan_.fb_flip.data(),
                  smoothing_plan_.fb_flip.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return fail_runtime(
                "RootMUSICEstimatorOp: smoothing flip upload failed");
          }
        } else {
          d_fb_flip_.reset();
        }
      }
    }
    if (smoothing_plan_.enabled) {
      const int64_t subarray_size = smoothing_plan_.subarray_size;
      const int64_t looks =
          smoothing_plan_.looks_per_snapshot() * num_snapshots;
      smoothed_snapshot = make_device_tensor(
          {num_items, subarray_size, looks}, cplx_dtype);
      HOLOSCAN_TRY(auto ss_ptr,
                   writable_ptr(*smoothed_snapshot, "smoothed_snapshot"));
      unsigned char* ss_mask_ptr = nullptr;
      if (mask != nullptr) {
        smoothed_mask = make_device_tensor(
            {num_items, looks}, DLDataType{kDLUInt, 8, 1});
        HOLOSCAN_TRY(auto mptr, writable_ptr(*smoothed_mask, "smoothed_mask"));
        ss_mask_ptr = static_cast<unsigned char*>(mptr);
      }
      try {
        run_spatial_smoothing_gather<T>(
            static_cast<const cuda::std::complex<T>*>(snap.data()), mask,
            num_items, num_channels, num_snapshots,
            static_cast<const int32_t*>(d_subarray_index_->data()),
            d_fb_flip_ ? static_cast<const int32_t*>(d_fb_flip_->data())
                       : nullptr,
            smoothing_plan_.num_subarrays, subarray_size,
            smoothing_plan_.forward_backward,
            static_cast<cuda::std::complex<T>*>(ss_ptr), ss_mask_ptr, stream);
        check_cuda("spatial smoothing gather", kOpName);
      } catch (const std::runtime_error& e) {
        return fail_runtime(e.what());
      }
      estimation_geometry = &smoothing_plan_.reduced_geometry;
      estimation_channels = subarray_size;
      estimation_snapshots = looks;
      estimation_data = static_cast<const cuda::std::complex<T>*>(ss_ptr);
      estimation_mask = ss_mask_ptr;
    }
  }

  PlanarAperturePlan planar;
  try {
    planar = analyze_planar_aperture(*estimation_geometry);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const std::string gkey = geometry_key(geometry);
  const bool geometry_is_new = seen_geometry_keys_.insert(gkey).second;
  const bool warn_new_geometry =
      geometry_is_new && seen_geometry_keys_.size() > 1;

  const std::vector<int64_t> eig_shape = {num_items, estimation_channels};
  auto covariance =
      make_device_tensor({num_items, estimation_channels, estimation_channels},
                         cplx_dtype);
  auto eigenvectors =
      make_device_tensor({num_items, estimation_channels, estimation_channels},
                         cplx_dtype);
  auto projector =
      make_device_tensor({num_items, estimation_channels, estimation_channels},
                         cplx_dtype);
  auto valid_counts = make_device_tensor({num_items}, i64_dtype);
  auto model_order = make_device_tensor({num_items}, i32_dtype);
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
    if (!solver_state_) {
      solver_state_ = std::make_shared<detail::RootMUSICSolverState>();
    }
    if (!solver_state_z_) {
      solver_state_z_ = std::make_shared<detail::RootMUSICSolverState>();
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
          static_cast<int32_t*>(cnt_writer->data()), solver_state_.get(),
          solver_state_z_.get(), stream);
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
        make_device_tensor({num_items, degree + 1}, cplx_dtype);
    auto companion =
        make_device_tensor({num_items, degree, degree}, cplx_dtype);
    auto roots = make_device_tensor({num_items, degree}, cplx_dtype);
    HOLOSCAN_TRY(auto coef_ptr, writable_ptr(*coefficients, "coefficients"));
    HOLOSCAN_TRY(auto companion_ptr, writable_ptr(*companion, "companion"));
    HOLOSCAN_TRY(auto roots_ptr, writable_ptr(*roots, "roots"));
    if (!solver_state_) {
      solver_state_ = std::make_shared<detail::RootMUSICSolverState>();
    }
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
          static_cast<int32_t*>(cnt_writer->data()), solver_state_.get(),
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
      !warned_rank_deficient_) {
    HOLOSCAN_LOG_WARN(
        "RootMUSICEstimatorOp: num_sources=auto with fewer snapshots than "
        "channels; falling back to one source");
    warned_rank_deficient_ = true;
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
  auto doa_emitted = doa_multi.emit(std::move(*doa_loan), emit_opts);
  if (!doa_emitted) {
    return make_unexpected(std::move(doa_emitted).error());
  }
  auto cnt_emitted = source_count.emit(std::move(*cnt_loan), emit_opts);
  if (!cnt_emitted) {
    return make_unexpected(std::move(cnt_emitted).error());
  }
  auto eig_emitted = eigenvalues.emit(std::move(*eig_loan), emit_opts);
  if (!eig_emitted) {
    return make_unexpected(std::move(eig_emitted).error());
  }

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
