// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
#include <vector>

#include "angle_estimation/esprit_estimator_op/esprit_stage.hpp"
#include "radar_common/dsp/planar_aperture.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {
namespace {

constexpr const char* kOpName = "ESPRITEstimatorOp";
constexpr int64_t kMaxSourcesCap = 64;

std::string geometry_key(const ArrayGeometry& geometry) {
  std::ostringstream key;
  key << geometry.wavelength << '|';
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
void ESPRITEstimatorOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void ESPRITEstimatorOp<T>::stop() {
  detail::retire_music_matx_caches();
}

template <typename T>
void ESPRITEstimatorOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("snapshot");
  spec.input<holoscan::TensorMap>("is_valid")
      .condition(holoscan::ConditionType::kNone);
  spec.output<holoscan::TensorMap>("aoa_estimates");
  spec.output<holoscan::TensorMap>("doa_multi")
      .condition(holoscan::ConditionType::kNone);
  spec.output<holoscan::TensorMap>("source_count")
      .condition(holoscan::ConditionType::kNone);
  spec.output<holoscan::TensorMap>("eigenvalues")
      .condition(holoscan::ConditionType::kNone);

  spec.param(array_geometry_, "array_geometry", "Array geometry",
             "Shift-invariant array positions and wavelength",
             ArrayGeometry{});
  add_override_port<ArrayGeometry>(spec, "array_geometry");
  spec.param(angle_convention_, "angle_convention", "Angle convention",
             "Boundary labeling for the gridless DOA output",
             AngleConvention{});
  spec.param(num_sources_, "num_sources", "Number of sources",
             "Signal-subspace dimension; negative selects per-item EFT auto",
             int64_t{-1});
  add_override_port<int64_t>(spec, "num_sources");
  spec.param(source_count_pfa_, "source_count_pfa", "Source-count Pfa",
             "EFT over-estimation probability for num_sources: auto", 1e-2);
  add_override_port<double>(spec, "source_count_pfa");
  spec.param(max_sources_, "max_sources", "Max sources",
             "doa_multi capacity and auto source-count cap", int64_t{4});
  add_override_port<int64_t>(spec, "max_sources");
  spec.param(spatial_smoothing_, "spatial_smoothing", "Spatial smoothing",
             "ULA sub-array count ([P], empty = off)",
             std::vector<int64_t>{});
  add_override_port<std::vector<int64_t>>(spec, "spatial_smoothing");
  spec.param(forward_backward_, "forward_backward",
             "Forward-backward smoothing",
             "Include conjugate-reversed looks when smoothing", true);
  spec.param(subarray_spec_, "subarray_spec", "Shift-invariant subarrays",
             "Corresponding subarray indices; empty derives a ULA pair",
             ShiftInvariantSubarrayPair{});
}

template <typename T>
void ESPRITEstimatorOp<T>::compute(InputContext& op_input,
                                   OutputContext& op_output,
                                   ExecutionContext&) {
  if (!seeded_) {
    effective_geometry_ = array_geometry_.get();
    effective_geometry_.validate();
    effective_num_sources_ = num_sources_.get();
    effective_pfa_ = source_count_pfa_.get();
    effective_max_sources_ = max_sources_.get();
    effective_smoothing_ = spatial_smoothing_.get();
    effective_forward_backward_ = forward_backward_.get();
    device_angle_convention_ =
        detail::make_device_angle_convention<T>(angle_convention_.get());
    seen_geometry_keys_.insert(geometry_key(effective_geometry_));
    seeded_ = true;
  }

  bool geometry_changed = false;
  bool smoothing_changed = false;
  bool scalar_changed = false;
  effective_geometry_ = resolve_override_param(
      op_input, "array_geometry", effective_geometry_, &geometry_changed);
  effective_geometry_.validate();
  effective_num_sources_ = resolve_override_param(
      op_input, "num_sources", effective_num_sources_, &scalar_changed);
  effective_pfa_ = resolve_override_param(op_input, "source_count_pfa",
                                          effective_pfa_, &scalar_changed);
  effective_max_sources_ = resolve_override_param(
      op_input, "max_sources", effective_max_sources_, &scalar_changed);
  effective_smoothing_ = resolve_override_param(
      op_input, "spatial_smoothing", effective_smoothing_,
      &smoothing_changed);
  (void)scalar_changed;

  if (!(effective_pfa_ > 0.0) || !(effective_pfa_ < 1.0)) {
    throw std::invalid_argument(
        "ESPRITEstimatorOp: source_count_pfa must be in (0, 1)");
  }
  if (effective_max_sources_ < 1 ||
      effective_max_sources_ > kMaxSourcesCap) {
    throw std::invalid_argument(
        "ESPRITEstimatorOp: max_sources must be in [1, 64]");
  }
  const auto requested_pair = subarray_spec_.get();
  requested_pair.validate();
  if (!requested_pair.is_auto() && !effective_smoothing_.empty()) {
    throw std::invalid_argument(
        "ESPRITEstimatorOp: explicit subarray_spec cannot be combined with "
        "spatial_smoothing; use subarray_spec=auto on the reduced aperture");
  }

  auto snapshot = receive_single(op_input, "snapshot", kOpName);
  require_cuda_accessible(*snapshot, "snapshot", kOpName);
  require_contiguous(*snapshot, "snapshot", kOpName);
  require_dtype(*snapshot, complex_dtype<T>(), "snapshot", kOpName);
  const auto& shape = snapshot->shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "ESPRITEstimatorOp: snapshot must have shape "
        "(num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape[2];
  if (num_channels != effective_geometry_.num_elements()) {
    throw std::invalid_argument(
        "ESPRITEstimatorOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }
  if (num_channels > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument(
        "ESPRITEstimatorOp: num_channels exceeds index capacity");
  }

  auto is_valid = receive_optional(op_input, "is_valid", kOpName);
  const unsigned char* mask = nullptr;
  if (is_valid) {
    const DLDataType dtype = is_valid->dtype();
    if (!((dtype.code == kDLBool || dtype.code == kDLUInt) &&
          dtype.bits == 8 && dtype.lanes == 1)) {
      throw std::invalid_argument(
          "ESPRITEstimatorOp: is_valid must have dtype bool or uint8");
    }
    if (is_valid->shape().size() != 2 || is_valid->shape()[0] != num_items ||
        is_valid->shape()[1] != num_snapshots) {
      throw std::invalid_argument(
          "ESPRITEstimatorOp: is_valid must have shape "
          "(num_items, num_snapshots) matching snapshot");
    }
    require_cuda_accessible(*is_valid, "is_valid", kOpName);
    require_contiguous(*is_valid, "is_valid", kOpName);
    mask = static_cast<const unsigned char*>(is_valid->data());
  }

  auto stream = op_input.receive_cuda_stream("snapshot");
  const ArrayGeometry* estimation_geometry = &effective_geometry_;
  int64_t estimation_channels = num_channels;
  int64_t estimation_snapshots = num_snapshots;
  const auto* estimation_data =
      static_cast<const cuda::std::complex<T>*>(snapshot->data());
  const unsigned char* estimation_mask = mask;
  std::shared_ptr<holoscan::Tensor> smoothed_snapshot;
  std::shared_ptr<holoscan::Tensor> smoothed_mask;

  if (!effective_smoothing_.empty()) {
    std::ostringstream key;
    key << geometry_key(effective_geometry_) << '|';
    for (int64_t count : effective_smoothing_) {
      key << count << ',';
    }
    key << '|' << (effective_forward_backward_ ? 1 : 0);
    if (key.str() != smoothing_key_) {
      smoothing_plan_ = build_spatial_smoothing_plan(
          effective_geometry_, effective_smoothing_,
          effective_forward_backward_);
      smoothing_key_ = key.str();
      if (smoothing_plan_.enabled) {
        const int64_t subarrays = smoothing_plan_.num_subarrays;
        const int64_t subarray_size = smoothing_plan_.subarray_size;
        d_subarray_index_ =
            make_device_tensor({subarrays * subarray_size},
                               DLDataType{kDLInt, 32, 1});
        if (cudaMemcpyAsync(
                d_subarray_index_->data(),
                smoothing_plan_.subarray_index.data(),
                smoothing_plan_.subarray_index.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              "ESPRITEstimatorOp: smoothing index upload failed");
        }
        if (smoothing_plan_.forward_backward) {
          d_fb_flip_ = make_device_tensor({subarray_size},
                                          DLDataType{kDLInt, 32, 1});
          if (cudaMemcpyAsync(
                  d_fb_flip_->data(), smoothing_plan_.fb_flip.data(),
                  smoothing_plan_.fb_flip.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::runtime_error(
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
          {num_items, subarray_size, looks}, complex_dtype<T>());
      if (mask != nullptr) {
        smoothed_mask = make_device_tensor(
            {num_items, looks}, DLDataType{kDLUInt, 8, 1});
      }
      run_spatial_smoothing_gather<T>(
          static_cast<const cuda::std::complex<T>*>(snapshot->data()), mask,
          num_items, num_channels, num_snapshots,
          static_cast<const int32_t*>(d_subarray_index_->data()),
          d_fb_flip_ ? static_cast<const int32_t*>(d_fb_flip_->data())
                     : nullptr,
          smoothing_plan_.num_subarrays, subarray_size,
          smoothing_plan_.forward_backward,
          static_cast<cuda::std::complex<T>*>(smoothed_snapshot->data()),
          smoothed_mask
              ? static_cast<unsigned char*>(smoothed_mask->data())
              : nullptr,
          stream);
      check_cuda("spatial smoothing gather", kOpName);
      estimation_geometry = &smoothing_plan_.reduced_geometry;
      estimation_channels = subarray_size;
      estimation_snapshots = looks;
      estimation_data = static_cast<const cuda::std::complex<T>*>(
          smoothed_snapshot->data());
      estimation_mask =
          smoothed_mask
              ? static_cast<const unsigned char*>(smoothed_mask->data())
              : nullptr;
    }
  }

  // A filled URA with `subarray_spec: auto` takes the native 2-D
  // dual-invariance path; an explicit subarray pair always stays on the 1-D
  // path (and must never trigger planar-aperture analysis, which is defined
  // only for auto derivation).
  const PlanarAperturePlan planar =
      requested_pair.is_auto() ? analyze_planar_aperture(*estimation_geometry)
                               : PlanarAperturePlan{};
  if (planar.is_ura && requested_pair.is_auto()) {
    const int64_t y_pairs = planar.nz * (planar.ny - 1);
    const int64_t z_pairs = (planar.nz - 1) * planar.ny;
    if (effective_num_sources_ >= 0) {
      if (effective_num_sources_ > std::min(y_pairs, z_pairs)) {
        throw std::invalid_argument(
            "ESPRITEstimatorOp: num_sources must not exceed the URA "
            "invariance subarray size");
      }
      if (effective_num_sources_ >= estimation_channels) {
        throw std::invalid_argument(
            "ESPRITEstimatorOp: num_sources must leave at least one "
            "noise-subspace dimension");
      }
    }

    std::ostringstream ura_key;
    ura_key << geometry_key(*estimation_geometry) << '|' << planar.ny << 'x'
            << planar.nz;
    if (ura_key.str() != ura_key_) {
      const UraInvariancePairs pairs =
          build_ura_dual_invariance(planar.ny, planar.nz);
      ura_ny_ = planar.ny;
      ura_nz_ = planar.nz;
      ura_y_pairs_ = static_cast<int64_t>(pairs.y_reference.size());
      ura_z_pairs_ = static_cast<int64_t>(pairs.z_reference.size());
      const auto upload = [&](const std::vector<int64_t>& src,
                              std::shared_ptr<holoscan::Tensor>& dst) {
        dst = make_device_tensor({static_cast<int64_t>(src.size())},
                                 DLDataType{kDLInt, 32, 1});
        std::vector<int32_t> host(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
          host[i] = static_cast<int32_t>(src[i]);
        }
        if (cudaMemcpyAsync(dst->data(), host.data(),
                            host.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              "ESPRITEstimatorOp: URA invariance index upload failed");
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
          throw std::runtime_error(
              "ESPRITEstimatorOp: URA invariance index sync failed");
        }
      };
      upload(pairs.y_reference, d_y_reference_);
      upload(pairs.y_shifted, d_y_shifted_);
      upload(pairs.z_reference, d_z_reference_);
      upload(pairs.z_shifted, d_z_shifted_);
      ura_key_ = ura_key.str();
    }

    if (geometry_changed) {
      const std::string key = geometry_key(effective_geometry_);
      if (seen_geometry_keys_.insert(key).second) {
        HOLOSCAN_LOG_WARN(
            "ESPRITEstimatorOp: unseen array_geometry override; revalidating "
            "the URA dual-invariance pairs");
      }
    }
    if (smoothing_changed) {
      HOLOSCAN_LOG_WARN(
          "ESPRITEstimatorOp: spatial smoothing override changes the ESPRIT "
          "aperture and derived subarray pairs");
    }

    const int64_t model_order_cap = std::min(
        {effective_max_sources_, ura_y_pairs_, ura_z_pairs_,
         estimation_channels - 1});
    const int64_t workspace_dim = std::max<int64_t>(model_order_cap, 1);
    auto covariance = make_device_tensor(
        {num_items, estimation_channels, estimation_channels},
        complex_dtype<T>());
    auto eigenvectors = make_device_tensor(
        {num_items, estimation_channels, estimation_channels},
        complex_dtype<T>());
    auto eigenvalues =
        make_device_tensor({num_items, estimation_channels}, real_dtype<T>());
    auto valid_counts =
        make_device_tensor({num_items}, DLDataType{kDLInt, 64, 1});
    auto model_order =
        make_device_tensor({num_items}, DLDataType{kDLInt, 32, 1});
    const auto workspace = [&]() {
      return make_device_tensor({num_items, workspace_dim, workspace_dim},
                                complex_dtype<T>());
    };
    auto psi_y = workspace();
    auto psi_z = workspace();
    auto z_ws = workspace();
    auto q_ws = workspace();
    auto r_ws = workspace();
    auto tmp_ws = workspace();
    auto aoa = make_device_tensor({num_items, 2}, real_dtype<T>());
    auto doa_multi = make_device_tensor(
        {num_items, effective_max_sources_, 2}, real_dtype<T>());
    auto source_count =
        make_device_tensor({num_items}, DLDataType{kDLInt, 32, 1});

    const double k = kTwoPi / estimation_geometry->wavelength;
    detail::run_esprit_2d_stage<T>(
        estimation_data, estimation_mask, num_items, estimation_channels,
        estimation_snapshots, effective_num_sources_, effective_pfa_,
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
        device_angle_convention_,
        static_cast<cuda::std::complex<T>*>(covariance->data()),
        static_cast<cuda::std::complex<T>*>(eigenvectors->data()),
        static_cast<int64_t*>(valid_counts->data()),
        static_cast<int32_t*>(model_order->data()),
        static_cast<T*>(eigenvalues->data()),
        static_cast<cuda::std::complex<T>*>(psi_y->data()),
        static_cast<cuda::std::complex<T>*>(psi_z->data()),
        static_cast<cuda::std::complex<T>*>(z_ws->data()),
        static_cast<cuda::std::complex<T>*>(q_ws->data()),
        static_cast<cuda::std::complex<T>*>(r_ws->data()),
        static_cast<cuda::std::complex<T>*>(tmp_ws->data()),
        static_cast<T*>(aoa->data()), static_cast<T*>(doa_multi->data()),
        effective_max_sources_, static_cast<int32_t*>(source_count->data()),
        stream);
    check_cuda("ESPRIT 2-D stage", kOpName);

    if (effective_num_sources_ < 0 && mask == nullptr &&
        estimation_snapshots < estimation_channels &&
        !warned_rank_deficient_) {
      HOLOSCAN_LOG_WARN(
          "ESPRITEstimatorOp: num_sources=auto with fewer snapshots than "
          "channels; falling back to one source");
      warned_rank_deficient_ = true;
    }

    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "aoa_estimates");
      op_output.set_cuda_stream(stream, "doa_multi");
      op_output.set_cuda_stream(stream, "source_count");
      op_output.set_cuda_stream(stream, "eigenvalues");
    }
    emit_single(op_output, "aoa_estimates", aoa);
    emit_single(op_output, "doa_multi", doa_multi);
    emit_single(op_output, "source_count", source_count);
    emit_single(op_output, "eigenvalues", eigenvalues);
    return;
  }

  const std::string current_pair_key =
      pair_key(*estimation_geometry, requested_pair);
  if (current_pair_key != subarray_key_) {
    subarray_plan_ =
        build_shift_invariant_subarray_plan(*estimation_geometry,
                                            requested_pair);
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
    if (cudaMemcpyAsync(d_reference_indices_->data(), reference.data(),
                        reference.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_shifted_indices_->data(), shifted.data(),
                        shifted.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error(
          "ESPRITEstimatorOp: subarray index upload failed");
    }
  }

  const int64_t pair_size = static_cast<int64_t>(
      subarray_plan_.pair.reference_indices.size());
  if (effective_num_sources_ >= 0 &&
      effective_num_sources_ > pair_size) {
    throw std::invalid_argument(
        "ESPRITEstimatorOp: num_sources must not exceed the number of "
        "corresponding subarray elements");
  }
  if (effective_num_sources_ >= estimation_channels) {
    throw std::invalid_argument(
        "ESPRITEstimatorOp: num_sources must leave at least one "
        "noise-subspace dimension");
  }
  if (geometry_changed) {
    const std::string key = geometry_key(effective_geometry_);
    if (seen_geometry_keys_.insert(key).second) {
      HOLOSCAN_LOG_WARN(
          "ESPRITEstimatorOp: unseen array_geometry override; "
          "revalidating the shift-invariant subarray pair");
    }
  }
  if (smoothing_changed) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: spatial smoothing override changes the ESPRIT "
        "aperture and derived subarray pair");
  }

  const int64_t model_order_cap =
      std::min(effective_max_sources_, pair_size);
  const int64_t workspace_dim = std::max<int64_t>(model_order_cap, 1);
  auto covariance =
      make_device_tensor({num_items, estimation_channels, estimation_channels},
                         complex_dtype<T>());
  auto eigenvectors =
      make_device_tensor({num_items, estimation_channels, estimation_channels},
                         complex_dtype<T>());
  auto eigenvalues =
      make_device_tensor({num_items, estimation_channels}, real_dtype<T>());
  auto valid_counts =
      make_device_tensor({num_items}, DLDataType{kDLInt, 64, 1});
  auto model_order =
      make_device_tensor({num_items}, DLDataType{kDLInt, 32, 1});
  auto matrix_workspace =
      make_device_tensor({num_items, workspace_dim, workspace_dim},
                         complex_dtype<T>());
  auto q_workspace =
      make_device_tensor({num_items, workspace_dim, workspace_dim},
                         complex_dtype<T>());
  auto r_workspace =
      make_device_tensor({num_items, workspace_dim, workspace_dim},
                         complex_dtype<T>());
  auto aoa = make_device_tensor({num_items, 2}, real_dtype<T>());
  auto doa_multi = make_device_tensor(
      {num_items, effective_max_sources_, 2}, real_dtype<T>());
  auto source_count =
      make_device_tensor({num_items}, DLDataType{kDLInt, 32, 1});

  detail::run_esprit_stage<T>(
      estimation_data, estimation_mask, num_items, estimation_channels,
      estimation_snapshots, effective_num_sources_, effective_pfa_,
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
      device_angle_convention_,
      static_cast<cuda::std::complex<T>*>(covariance->data()),
      static_cast<cuda::std::complex<T>*>(eigenvectors->data()),
      static_cast<int64_t*>(valid_counts->data()),
      static_cast<int32_t*>(model_order->data()),
      static_cast<T*>(eigenvalues->data()),
      static_cast<cuda::std::complex<T>*>(matrix_workspace->data()),
      static_cast<cuda::std::complex<T>*>(q_workspace->data()),
      static_cast<cuda::std::complex<T>*>(r_workspace->data()),
      static_cast<T*>(aoa->data()), static_cast<T*>(doa_multi->data()),
      effective_max_sources_, static_cast<int32_t*>(source_count->data()),
      stream);
  check_cuda("ESPRIT stage", kOpName);

  if (effective_num_sources_ < 0 && mask == nullptr &&
      estimation_snapshots < estimation_channels &&
      !warned_rank_deficient_) {
    HOLOSCAN_LOG_WARN(
        "ESPRITEstimatorOp: num_sources=auto with fewer snapshots than "
        "channels; falling back to one source");
    warned_rank_deficient_ = true;
  }

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "aoa_estimates");
    op_output.set_cuda_stream(stream, "doa_multi");
    op_output.set_cuda_stream(stream, "source_count");
    op_output.set_cuda_stream(stream, "eigenvalues");
  }
  emit_single(op_output, "aoa_estimates", aoa);
  emit_single(op_output, "doa_multi", doa_multi);
  emit_single(op_output, "source_count", source_count);
  emit_single(op_output, "eigenvalues", eigenvalues);
}

template class ESPRITEstimatorOp<float>;
template class ESPRITEstimatorOp<double>;

}  // namespace holoscan::radar
