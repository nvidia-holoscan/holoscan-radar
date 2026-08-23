// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
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

std::string geometry_key(const ArrayGeometry& geometry) {
  std::ostringstream key;
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

}  // namespace

template <typename T>
ESPRITEstimatorOp<T>::ESPRITEstimatorOp(
    const MaxTensorExtents& extents) {
  extents_ = extents;
  max_channels_ = extents.max_channels;
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
}

template <typename T>
void ESPRITEstimatorOp<T>::stop() {
  detail::retire_music_matx_caches();
}

template <typename T>
void ESPRITEstimatorOp<T>::setup(OperatorSpec& spec) {
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
  // need Connectivity gating. No angle_spectrum (gridless).
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
  // [live] Exact sticky overrides. Leaving unconnected is legal — compute()
  // seeds from Params.
  spec.input(array_geometry).queue_depth(2U);
  spec.input(spatial_smoothing).queue_depth(2U);
  spec.input(num_sources).queue_depth(2U);
  spec.input(source_count_pfa).queue_depth(2U);
  spec.input(max_sources).queue_depth(2U);
}

template <typename T>
Contract ESPRITEstimatorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{snapshot});
  maybe_add_live_override_to_contract(
      result, array_geometry, params().expect_array_geometry);
  maybe_add_live_override_to_contract(
      result, spatial_smoothing, params().expect_spatial_smoothing);
  maybe_add_live_override_to_contract(
      result, num_sources, params().expect_num_sources);
  maybe_add_live_override_to_contract(
      result, source_count_pfa, params().expect_source_count_pfa);
  maybe_add_live_override_to_contract(
      result, max_sources, params().expect_max_sources);
  return result;
}

template <typename T>
expected<void, Error> ESPRITEstimatorOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));

  const ESPRITParams& pinned = context.params<ESPRITParams>();
  HOLOSCAN_TRY(
      auto geometry,
      resolve_expected_override_via(
          pinned.expect_array_geometry, array_geometry, pinned.array_geometry,
          sticky_array_geometry_, pinned.sticky_array_geometry,
          [this](const WireArrayGeometry& wire)
              -> expected<ArrayGeometry, Error> {
            auto converted = from_wire(wire);
            if (!converted) {
              return make_unexpected(std::move(converted).error());
            }
            if (max_channels_.has_value() &&
                converted->num_elements() > *max_channels_) {
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
      resolve_expected_override_via(
          pinned.expect_spatial_smoothing, spatial_smoothing,
          pinned.spatial_smoothing, sticky_spatial_smoothing_,
          pinned.sticky_spatial_smoothing,
          [](const SpatialSmoothingOverride& wire) {
            return from_wire(wire);
          }));
  HOLOSCAN_TRY(auto effective_num_sources,
               resolve_expected_override(
                   pinned.expect_num_sources, num_sources, pinned.num_sources,
                   sticky_num_sources_, pinned.sticky_num_sources));
  HOLOSCAN_TRY(auto effective_pfa,
               resolve_expected_override(pinned.expect_source_count_pfa,
                                         source_count_pfa,
                                         pinned.source_count_pfa,
                                         sticky_source_count_pfa_,
                                         pinned.sticky_source_count_pfa));
  HOLOSCAN_TRY(auto effective_max_sources,
               resolve_expected_override(
                   pinned.expect_max_sources, max_sources, pinned.max_sources,
                   sticky_max_sources_, pinned.sticky_max_sources));

  try {
    geometry.validate();
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  if (seeded_angle_convention_.has_value() &&
      *seeded_angle_convention_ != pinned.angle_convention) {
    return fail_invalid(
        "ESPRITEstimatorOp: angle_convention is static and cannot change "
        "after the first successful activation; instantiate a separate "
        "operator per convention");
  }
  const bool forward_backward = pinned.forward_backward;
  const ShiftInvariantSubarrayPair& requested_pair = pinned.subarray_spec;
  if (max_channels_.has_value() &&
      geometry.num_elements() > *max_channels_) {
    return fail_invalid(
        "ESPRITEstimatorOp: array_geometry.num_elements exceeds "
        "MaxTensorExtents.max_channels");
  }

  if (!(effective_pfa > 0.0) || !(effective_pfa < 1.0)) {
    return fail_invalid(
        "ESPRITEstimatorOp: source_count_pfa must be in (0, 1)");
  }
  if (effective_max_sources < 1 || effective_max_sources > kMaxSourcesCap) {
    return fail_invalid(
        "ESPRITEstimatorOp: max_sources must be in [1, 64]");
  }
  try {
    requested_pair.validate();
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  if (!requested_pair.is_auto() && !smoothing.empty()) {
    return fail_invalid(
        "ESPRITEstimatorOp: explicit subarray_spec cannot be combined with "
        "spatial_smoothing; use subarray_spec=auto on the reduced aperture");
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
        "ESPRITEstimatorOp: snapshot must have shape "
        "(num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape[2];
  if (num_channels != geometry.num_elements()) {
    return fail_invalid(
        "ESPRITEstimatorOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }
  if (num_channels > std::numeric_limits<int32_t>::max()) {
    return fail_invalid(
        "ESPRITEstimatorOp: num_channels exceeds index capacity");
  }

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

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "ESPRITEstimatorOp: no CUDA compute stream"});
  }

  // Tick-local convention until the first successful emit locks the static
  // seed — geometry / subarray validation below can still fail this tick.
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

  // Warn once per previously unseen geometry key.
  const std::string geom_key = geometry_key(geometry);
  const bool geometry_is_new = seen_geometry_keys_.insert(geom_key).second;
  const bool geometry_changed =
      geometry_is_new && seen_geometry_keys_.size() > 1;

  // Spatial-smoothing front-end. When enabled, gather the snapshot into
  // overlapping sub-array looks and estimate on the reduced sub-array
  // geometry; the ESPRIT stage below is unchanged.
  const ArrayGeometry* estimation_geometry = &geometry;
  int64_t estimation_channels = num_channels;
  int64_t estimation_snapshots = num_snapshots;
  const cuda::std::complex<T>* estimation_data =
      static_cast<const cuda::std::complex<T>*>(snap.data());
  const unsigned char* estimation_mask = mask_ptr;
  std::shared_ptr<holoscan::Tensor> smoothed_snapshot;
  std::shared_ptr<holoscan::Tensor> smoothed_mask;
  bool smoothing_changed = false;

  if (!smoothing.empty()) {
    std::ostringstream key;
    key << geometry_key(geometry) << '|';
    for (int64_t count : smoothing) {
      key << count << ',';
    }
    key << '|' << (forward_backward ? 1 : 0);
    if (key.str() != smoothing_key_) {
      smoothing_changed = !smoothing_key_.empty();
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
              "ESPRITEstimatorOp: smoothing index upload failed");
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
                "ESPRITEstimatorOp: smoothing flip upload failed");
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
                   writable_ptr(*smoothed_snapshot, "ss_snapshot"));
      unsigned char* ss_mask_ptr = nullptr;
      if (mask_ptr != nullptr) {
        smoothed_mask = make_device_tensor(
            {num_items, looks}, DLDataType{kDLUInt, 8, 1});
        HOLOSCAN_TRY(auto mptr, writable_ptr(*smoothed_mask, "ss_mask"));
        ss_mask_ptr = static_cast<unsigned char*>(mptr);
      }
      try {
        run_spatial_smoothing_gather<T>(
            static_cast<const cuda::std::complex<T>*>(snap.data()), mask_ptr,
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
    if (ura_key.str() != ura_key_) {
      UraInvariancePairs pairs;
      try {
        pairs = build_ura_dual_invariance(planar.ny, planar.nz);
      } catch (const std::invalid_argument& e) {
        return fail_invalid(e.what());
      } catch (const std::runtime_error& e) {
        return fail_runtime(e.what());
      }
      ura_ny_ = planar.ny;
      ura_nz_ = planar.nz;
      ura_y_pairs_ = static_cast<int64_t>(pairs.y_reference.size());
      ura_z_pairs_ = static_cast<int64_t>(pairs.z_reference.size());
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
      HOLOSCAN_TRY(upload(pairs.y_reference, d_y_reference_, "y_reference"));
      HOLOSCAN_TRY(upload(pairs.y_shifted, d_y_shifted_, "y_shifted"));
      HOLOSCAN_TRY(upload(pairs.z_reference, d_z_reference_, "z_reference"));
      HOLOSCAN_TRY(upload(pairs.z_shifted, d_z_shifted_, "z_shifted"));
      ura_key_ = ura_key.str();
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
        {effective_max_sources, ura_y_pairs_, ura_z_pairs_,
         estimation_channels - 1});
    const int64_t workspace_dim = std::max<int64_t>(model_order_cap, 1);
    auto covariance = make_device_tensor(
        {num_items, estimation_channels, estimation_channels}, cplx_dtype);
    auto eigenvectors = make_device_tensor(
        {num_items, estimation_channels, estimation_channels}, cplx_dtype);
    auto valid_counts = make_device_tensor({num_items}, i64_dtype);
    auto model_order = make_device_tensor({num_items}, i32_dtype);
    const auto workspace = [&]() {
      return make_device_tensor({num_items, workspace_dim, workspace_dim},
                                cplx_dtype);
    };
    auto psi_y = workspace();
    auto psi_z = workspace();
    auto z_ws = workspace();
    auto q_ws = workspace();
    auto r_ws = workspace();
    auto tmp_ws = workspace();

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
          static_cast<const int32_t*>(d_y_reference_->data()),
          static_cast<const int32_t*>(d_y_shifted_->data()), ura_y_pairs_,
          static_cast<const int32_t*>(d_z_reference_->data()),
          static_cast<const int32_t*>(d_z_shifted_->data()), ura_z_pairs_,
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
        !warned_rank_deficient_) {
      HOLOSCAN_LOG_WARN(
          "ESPRITEstimatorOp: num_sources=auto with fewer snapshots than "
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

  const std::string current_pair_key =
      pair_key(*estimation_geometry, requested_pair);
  if (current_pair_key != subarray_key_) {
    try {
      subarray_plan_ = build_shift_invariant_subarray_plan(
          *estimation_geometry, requested_pair);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
    subarray_key_ = current_pair_key;
    const int64_t pair_size = static_cast<int64_t>(
        subarray_plan_.pair.reference_indices.size());
    d_reference_indices_ =
        make_device_tensor({pair_size}, DLDataType{kDLInt, 32, 1});
    d_shifted_indices_ =
        make_device_tensor({pair_size}, DLDataType{kDLInt, 32, 1});
    std::vector<int32_t> reference(static_cast<size_t>(pair_size));
    std::vector<int32_t> shifted(static_cast<size_t>(pair_size));
    for (int64_t index = 0; index < pair_size; ++index) {
      reference[static_cast<size_t>(index)] = static_cast<int32_t>(
          subarray_plan_.pair.reference_indices[static_cast<size_t>(index)]);
      shifted[static_cast<size_t>(index)] = static_cast<int32_t>(
          subarray_plan_.pair.shifted_indices[static_cast<size_t>(index)]);
    }
    HOLOSCAN_TRY(auto ref_ptr,
                 writable_ptr(*d_reference_indices_, "reference_indices"));
    HOLOSCAN_TRY(auto sh_ptr,
                 writable_ptr(*d_shifted_indices_, "shifted_indices"));
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
      subarray_plan_.pair.reference_indices.size());
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
  auto covariance = make_device_tensor(
      {num_items, estimation_channels, estimation_channels}, cplx_dtype);
  auto eigenvectors = make_device_tensor(
      {num_items, estimation_channels, estimation_channels}, cplx_dtype);
  auto valid_counts = make_device_tensor({num_items}, i64_dtype);
  auto model_order = make_device_tensor({num_items}, i32_dtype);
  auto matrix_workspace = make_device_tensor(
      {num_items, workspace_dim, workspace_dim}, cplx_dtype);
  auto q_workspace = make_device_tensor(
      {num_items, workspace_dim, workspace_dim}, cplx_dtype);
  auto r_workspace = make_device_tensor(
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
        static_cast<const int32_t*>(d_reference_indices_->data()),
        static_cast<const int32_t*>(d_shifted_indices_->data()), pair_size,
        workspace_dim,
        static_cast<T>(kTwoPi * subarray_plan_.spacing /
                       estimation_geometry->wavelength),
        static_cast<T>(subarray_plan_.axis[0]),
        static_cast<T>(subarray_plan_.axis[1]),
        static_cast<T>(subarray_plan_.axis[2]),
        static_cast<T>(subarray_plan_.broadside[0]),
        static_cast<T>(subarray_plan_.broadside[1]),
        static_cast<T>(subarray_plan_.broadside[2]),
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
      !warned_rank_deficient_) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: num_sources=auto with fewer snapshots than "
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

template class ESPRITEstimatorOp<float>;
template class ESPRITEstimatorOp<double>;

}  // namespace holoscan::radar
