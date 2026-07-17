// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "angle_estimation/beamscan_estimator_op/beamscan_stage.hpp"
#include "angle_estimation/beamscan_estimator_op/is_valid_mask.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "BeamscanEstimatorOp";

template <typename T>
std::vector<T> cast_vector(const std::vector<double>& in) {
  std::vector<T> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = static_cast<T>(in[i]);
  }
  return out;
}

}  // namespace

template <typename T>
void BeamscanEstimatorOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void BeamscanEstimatorOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("snapshot");
  spec.input<holoscan::TensorMap>("is_valid")
      .condition(holoscan::ConditionType::kNone);

  spec.output<holoscan::TensorMap>("aoa_estimates");
  spec.output<holoscan::TensorMap>("angle_spectrum");

  spec.param(array_geometry_, "array_geometry", "Array geometry",
             "Element positions and wavelength", ArrayGeometry{});
  add_override_port<ArrayGeometry>(spec, "array_geometry");

  spec.param(angle_convention_, "angle_convention", "Angle convention",
             "Boundary AER labeling for scan grid and DOA output",
             AngleConvention{});
  spec.param(az_angles_, "az_angles", "Azimuth scan grid",
             "Azimuth grid (or u grid under direction_cosine)",
             std::vector<double>{});
  add_override_port<std::vector<double>>(spec, "az_angles");
  spec.param(el_angles_, "el_angles", "Elevation scan grid",
             "Elevation grid (or v grid under direction_cosine)",
             std::vector<double>{});
  add_override_port<std::vector<double>>(spec, "el_angles");
  spec.param(array_taper_, "array_taper", "Array taper",
             "Amplitude taper across elements", WindowSpec{});
  add_override_port<WindowSpec>(spec, "array_taper");
  spec.param(peak_interpolation_, "peak_interpolation", "Peak interpolation",
             "Sub-grid peak refinement", PeakInterpolation::kNone);
  add_override_port<PeakInterpolation>(spec, "peak_interpolation");
}

template <typename T>
void BeamscanEstimatorOp<T>::compute(InputContext& op_input,
                                   OutputContext& op_output,
                                   ExecutionContext&) {
  if (!seeded_) {
    effective_geometry_ = array_geometry_.get();
    effective_geometry_.validate();
    effective_az_ = az_angles_.get();
    effective_el_ = el_angles_.get();
    effective_taper_ = array_taper_.get();
    effective_peak_interp_ = peak_interpolation_.get();
    seeded_ = true;
  }

  bool geom_changed = false;
  bool grid_changed = false;
  bool taper_changed = false;
  bool peak_changed = false;
  effective_geometry_ = resolve_override_param(
      op_input, "array_geometry", effective_geometry_, &geom_changed);
  effective_geometry_.validate();
  effective_az_ = resolve_override_param(op_input, "az_angles", effective_az_,
                                         &grid_changed);
  effective_el_ = resolve_override_param(op_input, "el_angles", effective_el_,
                                         &grid_changed);
  effective_taper_ = resolve_override_param(op_input, "array_taper",
                                            effective_taper_, &taper_changed);
  effective_peak_interp_ = resolve_override_param(
      op_input, "peak_interpolation", effective_peak_interp_, &peak_changed);
  (void)peak_changed;

  if (effective_az_.empty() || effective_el_.empty()) {
    throw std::invalid_argument(
        "BeamscanEstimatorOp: az_angles and el_angles must be non-empty");
  }

  auto snapshot = receive_single(op_input, "snapshot", kOpName);
  require_cuda_accessible(*snapshot, "snapshot", kOpName);
  require_contiguous(*snapshot, "snapshot", kOpName);
  require_dtype(*snapshot, complex_dtype<T>(), "snapshot", kOpName);

  const auto& shape = snapshot->shape();
  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument(
        "BeamscanEstimatorOp: snapshot must have shape "
        "(num_items, num_channels) or "
        "(num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  int64_t num_snapshots = 1;
  if (shape.size() == 3) {
    num_snapshots = shape[2];
    if (num_snapshots != 1) {
      throw std::invalid_argument(
          "BeamscanEstimatorOp: snapshot num_snapshots must be 1 for "
          "single-snapshot Beamscan");
    }
  }
  if (num_channels != effective_geometry_.num_elements()) {
    throw std::invalid_argument(
        "BeamscanEstimatorOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

  auto is_valid = receive_optional(op_input, "is_valid", kOpName);
  if (is_valid && shape.size() == 3) {
    detail::check_is_valid_mask(*is_valid, num_items, num_snapshots, kOpName);
  }

  auto stream = op_input.receive_cuda_stream("snapshot");

  const bool warn_on_build = geom_changed || grid_changed || taper_changed;
  const std::string cache_key =
      steering_cache_key(effective_geometry_, angle_convention_.get(),
                         effective_az_, effective_el_, effective_taper_);
  // Conjugate-transposed weights Wᶜ, built on device, laid out
  // (num_channels, num_beams) with beams el-major to match the reduce kernel's
  // beam decode — drops straight into the X · Wᶜ GEMM.
  const std::shared_ptr<holoscan::Tensor>& weights =
      steering_cache_.get_or_build(
          cache_key, warn_on_build, "steering weights", [&] {
            return build_steering_weights_device<T>(
                effective_geometry_, angle_convention_.get(), effective_az_,
                effective_el_, effective_taper_, BeamOrder::kElMajor,
                SteeringLayout::kChannelMajor, /*conjugate=*/true, stream);
          });

  const int64_t num_az = static_cast<int64_t>(effective_az_.size());
  const int64_t num_el = static_cast<int64_t>(effective_el_.size());
  const int64_t num_beams = num_az * num_el;

  const auto az_host = cast_vector<T>(effective_az_);
  const auto el_host = cast_vector<T>(effective_el_);
  auto az_grid = make_device_tensor({num_az}, real_dtype<T>());
  auto el_grid = make_device_tensor({num_el}, real_dtype<T>());
  if (cudaMemcpyAsync(az_grid->data(), az_host.data(),
                      az_host.size() * sizeof(T), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(el_grid->data(), el_host.data(),
                      el_host.size() * sizeof(T), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error("BeamscanEstimatorOp: scan-grid H2D failed");
  }

  auto spectrum =
      make_device_tensor({num_items, num_az, num_el}, real_dtype<T>());
  auto aoa = make_device_tensor({num_items, 2}, real_dtype<T>());
  // GEMM scratch for the beamformer output X · Wᶜ (num_items, num_beams).
  auto beamformed =
      make_device_tensor({num_items, num_beams}, complex_dtype<T>());

  if (!weights) {
    throw std::runtime_error(
        "BeamscanEstimatorOp: internal error: steering weights missing");
  }

  detail::run_beamscan_stage<T>(
      static_cast<const cuda::std::complex<T>*>(snapshot->data()), num_items,
      num_channels,
      static_cast<const cuda::std::complex<T>*>(weights->data()),
      static_cast<cuda::std::complex<T>*>(beamformed->data()), num_az, num_el,
      static_cast<const T*>(az_grid->data()),
      static_cast<const T*>(el_grid->data()), effective_peak_interp_,
      static_cast<T*>(spectrum->data()), static_cast<T*>(aoa->data()), stream);
  check_cuda("beamscan stage", kOpName);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "aoa_estimates");
    op_output.set_cuda_stream(stream, "angle_spectrum");
  }
  emit_single(op_output, "aoa_estimates", aoa);
  emit_single(op_output, "angle_spectrum", spectrum);
}

template class BeamscanEstimatorOp<float>;
template class BeamscanEstimatorOp<double>;

}  // namespace holoscan::radar
