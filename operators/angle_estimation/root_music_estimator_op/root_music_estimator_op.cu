// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
#include <vector>

#include "angle_estimation/root_music_estimator_op/root_music_stage.hpp"
#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "RootMUSICEstimatorOp";
constexpr int64_t kMaxSourcesCap = 64;
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

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
void RootMUSICEstimatorOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void RootMUSICEstimatorOp<T>::setup(OperatorSpec& spec) {
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
             "Uniform linear array positions and wavelength", ArrayGeometry{});
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
}

template <typename T>
void RootMUSICEstimatorOp<T>::compute(InputContext& op_input,
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

  const auto device_angle_convention =
      detail::make_device_angle_convention<T>(angle_convention_.get());
  if (!(effective_pfa_ > 0.0) || !(effective_pfa_ < 1.0)) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: source_count_pfa must be in (0, 1)");
  }
  if (effective_max_sources_ < 1 ||
      effective_max_sources_ > kMaxSourcesCap) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: max_sources must be in [1, 64]");
  }

  auto snapshot = receive_single(op_input, "snapshot", kOpName);
  require_cuda_accessible(*snapshot, "snapshot", kOpName);
  require_contiguous(*snapshot, "snapshot", kOpName);
  require_dtype(*snapshot, complex_dtype<T>(), "snapshot", kOpName);
  const auto& shape = snapshot->shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: snapshot must have shape "
        "(num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape[2];
  if (num_channels != effective_geometry_.num_elements()) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

  auto is_valid = receive_optional(op_input, "is_valid", kOpName);
  const unsigned char* mask = nullptr;
  if (is_valid) {
    const DLDataType dtype = is_valid->dtype();
    if (!((dtype.code == kDLBool || dtype.code == kDLUInt) &&
          dtype.bits == 8 && dtype.lanes == 1)) {
      throw std::invalid_argument(
          "RootMUSICEstimatorOp: is_valid must have dtype bool or uint8");
    }
    if (is_valid->shape().size() != 2 || is_valid->shape()[0] != num_items ||
        is_valid->shape()[1] != num_snapshots) {
      throw std::invalid_argument(
          "RootMUSICEstimatorOp: is_valid must have shape "
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
              "RootMUSICEstimatorOp: smoothing index upload failed");
        }
        if (smoothing_plan_.forward_backward) {
          d_fb_flip_ = make_device_tensor({subarray_size},
                                          DLDataType{kDLInt, 32, 1});
          if (cudaMemcpyAsync(
                  d_fb_flip_->data(), smoothing_plan_.fb_flip.data(),
                  smoothing_plan_.fb_flip.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::runtime_error(
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

  const ULAParameters ula = analyze_ula(*estimation_geometry);
  if (effective_num_sources_ >= 0 &&
      effective_num_sources_ > estimation_channels - 1) {
    throw std::invalid_argument(
        "RootMUSICEstimatorOp: num_sources must leave at least one "
        "noise-subspace dimension");
  }

  if (geometry_changed) {
    const std::string key = geometry_key(effective_geometry_);
    if (seen_geometry_keys_.insert(key).second) {
      HOLOSCAN_LOG_WARN(
          "RootMUSICEstimatorOp: unseen array_geometry override; "
          "revalidating ULA and solver dimensions");
    }
  }
  if (smoothing_changed) {
    HOLOSCAN_LOG_WARN(
        "RootMUSICEstimatorOp: spatial smoothing override changes the "
        "Root-MUSIC aperture and polynomial degree");
  }

  const int64_t degree = 2 * (estimation_channels - 1);
  auto covariance =
      make_device_tensor({num_items, estimation_channels, estimation_channels},
                         complex_dtype<T>());
  auto eigenvectors =
      make_device_tensor({num_items, estimation_channels, estimation_channels},
                         complex_dtype<T>());
  auto projector =
      make_device_tensor({num_items, estimation_channels, estimation_channels},
                         complex_dtype<T>());
  auto eigenvalues =
      make_device_tensor({num_items, estimation_channels}, real_dtype<T>());
  auto valid_counts =
      make_device_tensor({num_items}, DLDataType{kDLInt, 64, 1});
  auto model_order =
      make_device_tensor({num_items}, DLDataType{kDLInt, 32, 1});
  auto coefficients =
      make_device_tensor({num_items, degree + 1}, complex_dtype<T>());
  auto companion =
      make_device_tensor({num_items, degree, degree}, complex_dtype<T>());
  auto roots = make_device_tensor({num_items, degree}, complex_dtype<T>());
  auto aoa = make_device_tensor({num_items, 2}, real_dtype<T>());
  auto doa_multi = make_device_tensor(
      {num_items, effective_max_sources_, 2}, real_dtype<T>());
  auto source_count =
      make_device_tensor({num_items}, DLDataType{kDLInt, 32, 1});
  if (!solver_state_) {
    solver_state_ = std::make_shared<detail::RootMUSICSolverState>();
  }

  detail::run_root_music_stage<T>(
      estimation_data, estimation_mask, num_items, estimation_channels,
      estimation_snapshots, effective_num_sources_, effective_pfa_,
      effective_max_sources_,
      static_cast<T>(kTwoPi * ula.spacing / estimation_geometry->wavelength),
      static_cast<T>(ula.axis[0]), static_cast<T>(ula.axis[1]),
      static_cast<T>(ula.axis[2]), static_cast<T>(ula.broadside[0]),
      static_cast<T>(ula.broadside[1]), static_cast<T>(ula.broadside[2]),
      device_angle_convention,
      static_cast<cuda::std::complex<T>*>(covariance->data()),
      static_cast<cuda::std::complex<T>*>(eigenvectors->data()),
      static_cast<cuda::std::complex<T>*>(projector->data()),
      static_cast<int64_t*>(valid_counts->data()),
      static_cast<int32_t*>(model_order->data()),
      static_cast<T*>(eigenvalues->data()),
      static_cast<cuda::std::complex<T>*>(coefficients->data()),
      static_cast<cuda::std::complex<T>*>(companion->data()),
      static_cast<cuda::std::complex<T>*>(roots->data()),
      static_cast<T*>(aoa->data()), static_cast<T*>(doa_multi->data()),
      static_cast<int32_t*>(source_count->data()), solver_state_.get(), stream);
  check_cuda("root music stage", kOpName);

  if (effective_num_sources_ < 0 && mask == nullptr &&
      estimation_snapshots < estimation_channels &&
      !warned_rank_deficient_) {
    HOLOSCAN_LOG_WARN(
        "RootMUSICEstimatorOp: num_sources=auto with fewer snapshots than "
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

template class RootMUSICEstimatorOp<float>;
template class RootMUSICEstimatorOp<double>;

}  // namespace holoscan::radar
