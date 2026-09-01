// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
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

}  // namespace

template <typename T>
MVDREstimatorOp<T>::MVDREstimatorOp(
    const MaxTensorExtents& extents) {
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
  // Optional is_valid: omitted from contract so unconnected graphs compile.
  spec.input(is_valid).queue_depth(2U).expects_tensor(tensor_input(kDevice));
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
  // [live] overrides. Leaving one unconnected is legal; a wired port joins
  // snapshot through Exact + SafeDefault.
  spec.input(array_geometry).queue_depth(2U);
  spec.input(az_angles)
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostF64Vector));
  spec.input(el_angles)
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostF64Vector));
  spec.input(spatial_smoothing).queue_depth(2U);
  spec.input(diagonal_loading).queue_depth(2U);
  spec.input(max_sources).queue_depth(2U);
}

template <typename T>
Contract MVDREstimatorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{snapshot});
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
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));

  const MVDRParams& pinned = context.params<MVDRParams>();
  HOLOSCAN_TRY(
      auto geometry,
      resolve_live_override_via(
          array_geometry, pinned.array_geometry,
          [](const WireArrayGeometry& value) { return from_wire(value); }));
  HOLOSCAN_TRY(
      auto az,
      resolve_live_override_via(
          az_angles, pinned.az_angles,
          [this](const Tensor& value) {
            return read_host_config_vector<double>(
                value, DLDataType{kDLFloat, 64U, 1U},
                require_extent(extents_.max_az, "max_az", kOpName),
                "az_angles");
          }));
  HOLOSCAN_TRY(
      auto el,
      resolve_live_override_via(
          el_angles, pinned.el_angles,
          [this](const Tensor& value) {
            return read_host_config_vector<double>(
                value, DLDataType{kDLFloat, 64U, 1U},
                require_extent(extents_.max_el, "max_el", kOpName),
                "el_angles");
          }));
  HOLOSCAN_TRY(
      auto smoothing,
      resolve_live_override_via(
          spatial_smoothing, pinned.spatial_smoothing,
          [](const SpatialSmoothingOverride& value) {
            return from_wire(value);
          }));
  HOLOSCAN_TRY(auto effective_loading,
               resolve_live_override(
                   diagonal_loading, pinned.diagonal_loading));
  HOLOSCAN_TRY(auto effective_max_sources,
               resolve_live_override(
                   max_sources, pinned.max_sources));

  try {
    geometry.validate();
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const AngleConvention& convention = pinned.angle_convention;
  const bool forward_backward = pinned.forward_backward;

  if (az.empty() || el.empty()) {
    return fail_invalid(
        "MVDREstimatorOp: az_angles and el_angles must be non-empty");
  }
  const int64_t max_az =
      require_extent(extents_.max_az, "max_az", kOpName);
  const int64_t max_el =
      require_extent(extents_.max_el, "max_el", kOpName);
  const int64_t max_channels =
      require_extent(extents_.max_channels, "max_channels", kOpName);
  if (static_cast<int64_t>(az.size()) > max_az ||
      static_cast<int64_t>(el.size()) > max_el ||
      geometry.num_elements() > max_channels) {
    return fail_invalid(
        "MVDREstimatorOp: live configuration exceeds MaxTensorExtents");
  }
  if (effective_max_sources < 1 || effective_max_sources > kMaxSourcesCap) {
    return fail_invalid(
        "MVDREstimatorOp: max_sources must be in [1, 64]");
  }

  const std::string candidate_smoothing_key =
      smoothing.empty()
          ? std::string{}
          : spatial_smoothing_key(geometry, smoothing, forward_backward);
  std::optional<SpatialSmoothingPlan> candidate_smoothing_plan;
  if (!smoothing.empty() && candidate_smoothing_key != smoothing_key_) {
    try {
      candidate_smoothing_plan = build_spatial_smoothing_plan(
          geometry, smoothing, forward_backward);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  }

  try {
    require_cuda_accessible(snap, "snapshot", kOpName);
    require_contiguous(snap, "snapshot", kOpName);
    require_dtype(snap, complex_dtype<T>(), "snapshot", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const auto& shape = snap.shape();
  if (shape.size() != 2 && shape.size() != 3) {
    return fail_invalid(
        "MVDREstimatorOp: snapshot must have shape "
        "(num_items, num_channels) or "
        "(num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape.size() == 3 ? shape[2] : 1;
  if (num_channels != geometry.num_elements()) {
    return fail_invalid(
        "MVDREstimatorOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  const unsigned char* mask_ptr = nullptr;
  if (maybe_valid.has_value()) {
    const DLDataType dt = maybe_valid->dtype();
    const bool ok_dtype = (dt.code == kDLBool || dt.code == kDLUInt) &&
                          dt.bits == 8 && dt.lanes == 1;
    if (!ok_dtype) {
      return fail_invalid(
          "MVDREstimatorOp: is_valid must have dtype bool or uint8");
    }
    const auto& vshape = maybe_valid->shape();
    if (vshape.size() != 2 || vshape[0] != num_items ||
        vshape[1] != num_snapshots) {
      return fail_invalid(
          "MVDREstimatorOp: is_valid must have shape "
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
                                 "MVDREstimatorOp: no CUDA compute stream"});
  }

  const int64_t num_az = static_cast<int64_t>(az.size());
  const int64_t num_el = static_cast<int64_t>(el.size());
  const DLDataType out_dtype = real_dtype<T>();
  const DLDataType cplx_dtype = complex_dtype<T>();
  const DLDataType i32_dtype{kDLInt, 32, 1};
  const DLDataType i64_dtype{kDLInt, 64, 1};
  const std::vector<int64_t> aoa_shape = {num_items, 2};
  const std::vector<int64_t> spec_shape = {num_items, num_az, num_el};
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
    HOLOSCAN_TRY(auto spec_loan,
                 prepare_device_tensor(angle_spectrum, context, spec_shape,
                                      out_dtype, noop));
    std::optional<TensorOutputLoan> doa_loan;
    std::optional<TensorOutputLoan> cnt_loan;
    std::optional<TensorOutputLoan> eig_loan;
    if (doa_multi_out_.active()) {
      HOLOSCAN_TRY(auto loan,
                   prepare_device_tensor(doa_multi, context, doa_shape,
                                        out_dtype, noop));
      doa_loan.emplace(std::move(loan));
    }
    if (source_count_out_.active()) {
      HOLOSCAN_TRY(auto loan,
                   prepare_device_tensor(source_count, context, count_shape,
                                        i32_dtype, noop));
      cnt_loan.emplace(std::move(loan));
    }
    if (eigenvalues_out_.active()) {
      HOLOSCAN_TRY(auto loan,
                   prepare_device_tensor(eigenvalues, context, eig_shape,
                                        out_dtype, noop));
      eig_loan.emplace(std::move(loan));
    }
    HOLOSCAN_TRY(
        emit_prepared_tensor(aoa_estimates, context, std::move(aoa_loan)));
    // angle_spectrum is optional/always (OPERATOR_API.md).
    (void)emit_prepared_tensor(angle_spectrum, context, std::move(spec_loan));
    if (doa_loan.has_value()) {
      doa_multi_out_.emit(doa_multi, context, std::move(*doa_loan),
                          "MVDREstimatorOp.doa_multi");
    }
    if (cnt_loan.has_value()) {
      source_count_out_.emit(source_count, context, std::move(*cnt_loan),
                             "MVDREstimatorOp.source_count");
    }
    if (eig_loan.has_value()) {
      eigenvalues_out_.emit(eigenvalues, context, std::move(*eig_loan),
                            "MVDREstimatorOp.eigenvalues");
    }
    return {};
  };

  if (num_items == 0) {
    HOLOSCAN_TRY(emit_empty());
    retained_snapshot_ = std::move(snap);
    if (maybe_valid.has_value()) {
      retained_is_valid_ = std::move(*maybe_valid);
    }
    return {};
  }

  // Spatial-smoothing front-end. When enabled, gather the snapshot into
  // overlapping sub-array looks and estimate on the reduced sub-array
  // geometry; the covariance/eig/inverse/scan path below is unchanged.
  const ArrayGeometry* manifold_geom = &geometry;
  int64_t est_channels = num_channels;
  int64_t est_snapshots = num_snapshots;
  const cuda::std::complex<T>* est_snapshot =
      static_cast<const cuda::std::complex<T>*>(snap.data());
  const unsigned char* est_mask = mask_ptr;
  std::shared_ptr<holoscan::Tensor> ss_snapshot;  // keep-alive
  std::shared_ptr<holoscan::Tensor> ss_mask_t;    // keep-alive

  if (!smoothing.empty()) {
    if (candidate_smoothing_key != smoothing_key_) {
      smoothing_plan_ = std::move(*candidate_smoothing_plan);
      smoothing_key_ = candidate_smoothing_key;
      if (smoothing_plan_.enabled) {
        const int64_t p = smoothing_plan_.num_subarrays;
        const int64_t l = smoothing_plan_.subarray_size;
        d_subarray_index_ =
            make_device_tensor({p * l}, DLDataType{kDLInt, 32, 1});
        HOLOSCAN_TRY(auto idx_ptr,
                     writable_ptr(*d_subarray_index_, "subarray_index"));
        if (cudaMemcpyAsync(
                idx_ptr, smoothing_plan_.subarray_index.data(),
                smoothing_plan_.subarray_index.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          return fail_runtime(
              "MVDREstimatorOp: spatial-smoothing index H2D failed");
        }
        if (smoothing_plan_.forward_backward) {
          d_fb_flip_ = make_device_tensor({l}, DLDataType{kDLInt, 32, 1});
          HOLOSCAN_TRY(auto flip_ptr,
                       writable_ptr(*d_fb_flip_, "fb_flip"));
          if (cudaMemcpyAsync(
                  flip_ptr, smoothing_plan_.fb_flip.data(),
                  smoothing_plan_.fb_flip.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return fail_runtime(
                "MVDREstimatorOp: spatial-smoothing fb-flip H2D failed");
          }
        } else {
          d_fb_flip_.reset();
        }
        HOLOSCAN_LOG_WARN(
            "MVDREstimatorOp: spatial smoothing on — aperture {}->{} "
            "channels, {} sub-arrays{}",
            num_channels, l, p,
            smoothing_plan_.forward_backward ? " (forward-backward)" : "");
      }
    }

    if (smoothing_plan_.enabled) {
      const int64_t l = smoothing_plan_.subarray_size;
      const int64_t m = smoothing_plan_.looks_per_snapshot() * num_snapshots;
      ss_snapshot = make_device_tensor({num_items, l, m}, cplx_dtype);
      HOLOSCAN_TRY(auto ss_ptr, writable_ptr(*ss_snapshot, "ss_snapshot"));
      unsigned char* ss_mask_ptr = nullptr;
      if (mask_ptr != nullptr) {
        ss_mask_t = make_device_tensor({num_items, m}, DLDataType{kDLUInt, 8, 1});
        HOLOSCAN_TRY(auto mptr, writable_ptr(*ss_mask_t, "ss_mask"));
        ss_mask_ptr = static_cast<unsigned char*>(mptr);
      }
      try {
        run_spatial_smoothing_gather<T>(
            static_cast<const cuda::std::complex<T>*>(snap.data()), mask_ptr,
            num_items, num_channels, num_snapshots,
            static_cast<const int32_t*>(d_subarray_index_->data()),
            d_fb_flip_ ? static_cast<const int32_t*>(d_fb_flip_->data())
                       : nullptr,
            smoothing_plan_.num_subarrays, l, smoothing_plan_.forward_backward,
            static_cast<cuda::std::complex<T>*>(ss_ptr), ss_mask_ptr, stream);
        check_cuda("spatial smoothing gather", kOpName);
      } catch (const std::runtime_error& e) {
        return fail_runtime(e.what());
      }

      manifold_geom = &smoothing_plan_.reduced_geometry;
      est_channels = l;
      est_snapshots = m;
      est_snapshot = static_cast<const cuda::std::complex<T>*>(ss_ptr);
      est_mask = ss_mask_ptr;
    }
  }

  const std::string cache_key =
      steering_cache_key(*manifold_geom, convention, az, el, WindowSpec{});
  const std::shared_ptr<holoscan::Tensor>& manifold =
      steering_cache_.get_or_build(
          cache_key, "steering manifold", [&] {
            return build_steering_weights_device<T>(
                *manifold_geom, convention, az, el, WindowSpec{},
                BeamOrder::kElMajor, SteeringLayout::kBeamMajor,
                /*conjugate=*/false, stream);
          });
  if (!manifold) {
    return fail_runtime(
        "MVDREstimatorOp: internal error: steering manifold missing");
  }

  const auto az_host = cast_vector<T>(az);
  const auto el_host = cast_vector<T>(el);
  auto az_grid = make_device_tensor({num_az}, out_dtype);
  auto el_grid = make_device_tensor({num_el}, out_dtype);
  HOLOSCAN_TRY(auto az_ptr, writable_ptr(*az_grid, "az_grid"));
  HOLOSCAN_TRY(auto el_ptr, writable_ptr(*el_grid, "el_grid"));
  if (cudaMemcpyAsync(az_ptr, az_host.data(), az_host.size() * sizeof(T),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(el_ptr, el_host.data(), el_host.size() * sizeof(T),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return fail_runtime("MVDREstimatorOp: scan-grid H2D failed");
  }

  const std::vector<int64_t> eig_shape = {num_items, est_channels};
  auto cov = make_device_tensor({num_items, est_channels, est_channels},
                                cplx_dtype);
  auto rinv = make_device_tensor({num_items, est_channels, est_channels},
                                 cplx_dtype);
  auto k_counts = make_device_tensor({num_items}, i64_dtype);
  HOLOSCAN_TRY(auto cov_ptr, writable_ptr(*cov, "cov"));
  HOLOSCAN_TRY(auto rinv_ptr, writable_ptr(*rinv, "rinv"));
  HOLOSCAN_TRY(auto k_ptr, writable_ptr(*k_counts, "k_counts"));

  // Allocate the output loans, fill on the runtime stream, then commit/emit.
  // The latched optional ports (OPERATOR_API.md) drop out of all of it once
  // known unconsumed: eigenvalues then skips a batched Hermitian eig, and
  // doa_multi / source_count skip the peak-extraction pass. That one pass
  // writes both, so the pair drops out only once both have latched.
  const bool want_eig = eigenvalues_out_.active();
  const bool want_peaks =
      doa_multi_out_.active() || source_count_out_.active();

  auto aoa_loan = aoa_estimates.allocate_tensor(
      TensorLoanRequest{.shape = aoa_shape, .dtype = out_dtype});
  if (!aoa_loan) {
    return make_unexpected(std::move(aoa_loan).error());
  }
  auto spec_loan = angle_spectrum.allocate_tensor(
      TensorLoanRequest{.shape = spec_shape, .dtype = out_dtype});
  if (!spec_loan) {
    return make_unexpected(std::move(spec_loan).error());
  }
  std::optional<TensorOutputLoan> doa_loan;
  std::optional<TensorOutputLoan> cnt_loan;
  if (want_peaks) {
    auto doa = doa_multi.allocate_tensor(
        TensorLoanRequest{.shape = doa_shape, .dtype = out_dtype});
    if (!doa) {
      return make_unexpected(std::move(doa).error());
    }
    doa_loan.emplace(std::move(*doa));
    auto cnt = source_count.allocate_tensor(
        TensorLoanRequest{.shape = count_shape, .dtype = i32_dtype});
    if (!cnt) {
      return make_unexpected(std::move(cnt).error());
    }
    cnt_loan.emplace(std::move(*cnt));
  }
  std::optional<TensorOutputLoan> eig_loan;
  if (want_eig) {
    auto eig = eigenvalues.allocate_tensor(
        TensorLoanRequest{.shape = eig_shape, .dtype = out_dtype});
    if (!eig) {
      return make_unexpected(std::move(eig).error());
    }
    eig_loan.emplace(std::move(*eig));
  }

  auto aoa_writer = aoa_loan->write(context.cuda_stream());
  if (!aoa_writer) {
    return make_unexpected(std::move(aoa_writer).error());
  }
  auto spec_writer = spec_loan->write(context.cuda_stream());
  if (!spec_writer) {
    return make_unexpected(std::move(spec_writer).error());
  }
  std::optional<TensorOutputWriteGuard> doa_writer;
  std::optional<TensorOutputWriteGuard> cnt_writer;
  if (want_peaks) {
    auto doa = doa_loan->write(context.cuda_stream());
    if (!doa) {
      return make_unexpected(std::move(doa).error());
    }
    doa_writer.emplace(std::move(*doa));
    auto cnt = cnt_loan->write(context.cuda_stream());
    if (!cnt) {
      return make_unexpected(std::move(cnt).error());
    }
    cnt_writer.emplace(std::move(*cnt));
  }
  std::optional<TensorOutputWriteGuard> eig_writer;
  if (want_eig) {
    auto eig = eig_loan->write(context.cuda_stream());
    if (!eig) {
      return make_unexpected(std::move(eig).error());
    }
    eig_writer.emplace(std::move(*eig));
  }

  try {
    detail::run_mvdr_stage<T>(
        est_snapshot, est_mask, num_items, est_channels, est_snapshots,
        effective_loading,
        static_cast<const cuda::std::complex<T>*>(manifold->data()), num_az,
        num_el, static_cast<const T*>(az_grid->data()),
        static_cast<const T*>(el_grid->data()), effective_max_sources,
        static_cast<cuda::std::complex<T>*>(cov_ptr),
        static_cast<cuda::std::complex<T>*>(rinv_ptr),
        static_cast<int64_t*>(k_ptr), static_cast<T*>(spec_writer->data()),
        static_cast<T*>(aoa_writer->data()),
        eig_writer ? static_cast<T*>(eig_writer->data()) : nullptr,
        doa_writer ? static_cast<T*>(doa_writer->data()) : nullptr,
        cnt_writer ? static_cast<int32_t*>(cnt_writer->data()) : nullptr,
        stream);
    check_cuda("mvdr stage", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  auto aoa_committed = std::move(*aoa_writer).commit();
  if (!aoa_committed) {
    return make_unexpected(std::move(aoa_committed).error());
  }
  auto spec_committed = std::move(*spec_writer).commit();
  if (!spec_committed) {
    return make_unexpected(std::move(spec_committed).error());
  }
  if (doa_writer.has_value()) {
    auto doa_committed = std::move(*doa_writer).commit();
    if (!doa_committed) {
      return make_unexpected(std::move(doa_committed).error());
    }
  }
  if (cnt_writer.has_value()) {
    auto cnt_committed = std::move(*cnt_writer).commit();
    if (!cnt_committed) {
      return make_unexpected(std::move(cnt_committed).error());
    }
  }
  if (eig_writer.has_value()) {
    auto eig_committed = std::move(*eig_writer).commit();
    if (!eig_committed) {
      return make_unexpected(std::move(eig_committed).error());
    }
  }

  const EmitOptions emit_opts{.capture_time = context.activation_time()};
  auto aoa_emitted = aoa_estimates.emit(std::move(*aoa_loan), emit_opts);
  if (!aoa_emitted) {
    return make_unexpected(std::move(aoa_emitted).error());
  }
  // angle_spectrum is optional/always (OPERATOR_API.md): the Capon grid is the
  // same pass that yields aoa_estimates, so there is nothing to skip.
  (void)angle_spectrum.emit(std::move(*spec_loan), emit_opts);
  if (doa_loan.has_value()) {
    doa_multi_out_.emit(doa_multi, context, std::move(*doa_loan),
                        "MVDREstimatorOp.doa_multi");
  }
  if (cnt_loan.has_value()) {
    source_count_out_.emit(source_count, context, std::move(*cnt_loan),
                           "MVDREstimatorOp.source_count");
  }
  if (eig_loan.has_value()) {
    eigenvalues_out_.emit(eigenvalues, context, std::move(*eig_loan),
                          "MVDREstimatorOp.eigenvalues");
  }

  retained_snapshot_ = std::move(snap);
  if (maybe_valid.has_value()) {
    retained_is_valid_ = std::move(*maybe_valid);
  }
  return {};
}

template class MVDREstimatorOp<float>;
template class MVDREstimatorOp<double>;

}  // namespace holoscan::radar
