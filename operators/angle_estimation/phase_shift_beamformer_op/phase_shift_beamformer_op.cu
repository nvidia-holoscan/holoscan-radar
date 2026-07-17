// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_stage.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "PhaseShiftBeamformerOp";

}  // namespace

template <typename T>
void PhaseShiftBeamformerOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void PhaseShiftBeamformerOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("rd_maps");
  spec.output<holoscan::TensorMap>("rda_cube");

  spec.param(array_geometry_, "array_geometry", "Array geometry",
             "Element positions and wavelength", ArrayGeometry{});
  add_override_port<ArrayGeometry>(spec, "array_geometry");

  spec.param(angle_convention_, "angle_convention", "Angle convention",
             "Boundary AER labeling for scan grid and beam axes",
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
}

template <typename T>
void PhaseShiftBeamformerOp<T>::compute(InputContext& op_input,
                                        OutputContext& op_output,
                                        ExecutionContext&) {
  if (!seeded_) {
    effective_geometry_ = array_geometry_.get();
    effective_geometry_.validate();
    effective_az_ = az_angles_.get();
    effective_el_ = el_angles_.get();
    effective_taper_ = array_taper_.get();
    seeded_ = true;
  }

  bool geom_changed = false;
  bool grid_changed = false;
  bool taper_changed = false;
  effective_geometry_ = resolve_override_param(
      op_input, "array_geometry", effective_geometry_, &geom_changed);
  effective_geometry_.validate();
  effective_az_ = resolve_override_param(op_input, "az_angles", effective_az_,
                                         &grid_changed);
  effective_el_ = resolve_override_param(op_input, "el_angles", effective_el_,
                                         &grid_changed);
  effective_taper_ = resolve_override_param(op_input, "array_taper",
                                            effective_taper_, &taper_changed);

  if (effective_az_.empty() || effective_el_.empty()) {
    throw std::invalid_argument(
        "PhaseShiftBeamformerOp: az_angles and el_angles must be non-empty");
  }

  auto rd_maps = receive_single(op_input, "rd_maps", kOpName);
  require_cuda_accessible(*rd_maps, "rd_maps", kOpName);
  require_contiguous(*rd_maps, "rd_maps", kOpName);
  require_dtype(*rd_maps, complex_dtype<T>(), "rd_maps", kOpName);

  const auto& shape = rd_maps->shape();
  const int64_t rank = static_cast<int64_t>(shape.size());
  if (rank < 3) {
    throw std::invalid_argument(
        "PhaseShiftBeamformerOp: rd_maps must have shape "
        "(..., num_channels, num_doppler, num_range_bins) with rank >= 3");
  }
  const int64_t num_channels = shape[static_cast<size_t>(rank - 3)];
  const int64_t num_doppler = shape[static_cast<size_t>(rank - 2)];
  const int64_t num_range = shape[static_cast<size_t>(rank - 1)];
  int64_t batch = 1;
  for (int64_t i = 0; i < rank - 3; ++i) {
    batch *= shape[static_cast<size_t>(i)];
  }
  const int64_t num_cells = num_doppler * num_range;

  if (num_channels != effective_geometry_.num_elements()) {
    throw std::invalid_argument(
        "PhaseShiftBeamformerOp: rd_maps num_channels must match "
        "array_geometry.num_elements");
  }

  auto stream = op_input.receive_cuda_stream("rd_maps");

  const bool warn_on_build = geom_changed || grid_changed || taper_changed;
  const std::string cache_key =
      steering_cache_key(effective_geometry_, angle_convention_.get(),
                         effective_az_, effective_el_, effective_taper_);
  // Conjugated weights, built on device, az-major beams in a
  // (num_beams, num_channels) layout — drops straight into the GEMM.
  const std::shared_ptr<holoscan::Tensor>& weights =
      steering_cache_.get_or_build(
          cache_key, warn_on_build, "steering weights", [&] {
            return build_steering_weights_device<T>(
                effective_geometry_, angle_convention_.get(), effective_az_,
                effective_el_, effective_taper_, BeamOrder::kAzMajor,
                SteeringLayout::kBeamMajor, /*conjugate=*/true, stream);
          });

  const int64_t num_az = static_cast<int64_t>(effective_az_.size());
  const int64_t num_el = static_cast<int64_t>(effective_el_.size());
  const int64_t num_beams = num_az * num_el;

  // Output cube: original leading (batch) dims, then the beam grid, then the
  // range-Doppler plane. The contiguous buffer aliases [batch, num_beams,
  // num_cells] for the GEMM.
  std::vector<int64_t> out_shape;
  out_shape.reserve(static_cast<size_t>(rank + 1));
  for (int64_t i = 0; i < rank - 3; ++i) {
    out_shape.push_back(shape[static_cast<size_t>(i)]);
  }
  out_shape.push_back(num_az);
  out_shape.push_back(num_el);
  out_shape.push_back(num_doppler);
  out_shape.push_back(num_range);
  auto rda_cube = make_device_tensor(out_shape, complex_dtype<T>());

  if (!weights) {
    throw std::runtime_error(
        "PhaseShiftBeamformerOp: internal error: steering weights missing");
  }

  detail::run_phase_shift_beamform_stage<T>(
      static_cast<const cuda::std::complex<T>*>(rd_maps->data()),
      static_cast<cuda::std::complex<T>*>(rda_cube->data()), batch,
      num_channels, num_cells,
      static_cast<const cuda::std::complex<T>*>(weights->data()), num_beams,
      stream);
  check_cuda("phase-shift beamform stage", kOpName);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "rda_cube");
  }
  emit_single(op_output, "rda_cube", rda_cube);
}

template class PhaseShiftBeamformerOp<float>;
template class PhaseShiftBeamformerOp<double>;

}  // namespace holoscan::radar
