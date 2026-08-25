// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/backprojection_op/backprojection_op.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/port_connected.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/backprojection_op/backprojection_stage.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

template <typename T>
const std::string& op_name() {
  static const std::string name =
      std::string("BackprojectionOp<") + precision_name<T>() + ">";
  return name;
}

bool is_resetting(FormationControl control) {
  return control == FormationControl::kEmitAndReset ||
         control == FormationControl::kResetWithoutEmit ||
         control == FormationControl::kResetThenEmit;
}

bool is_pre_reset(FormationControl control) {
  return control == FormationControl::kResetWithoutEmit ||
         control == FormationControl::kResetThenEmit;
}

bool is_emit(FormationControl control) {
  return control != FormationControl::kResetWithoutEmit;
}

void require_linear(RangeInterpolation interpolation,
                    const std::string& who) {
  if (interpolation != RangeInterpolation::kLinear) {
    throw std::invalid_argument(
        who + ": interpolation '" +
        std::string(to_string(interpolation)) +
        "' is not implemented (only linear is)");
  }
}

void require_backend(BackprojectionBackend backend, const std::string& who) {
  if (backend == BackprojectionBackend::kMatxTaylorFast) {
    throw std::invalid_argument(
        who + ": backend 'matx_taylor_fast' is not in this MatX drop");
  }
  if (!detail::backend_is_implemented(backend)) {
    throw std::invalid_argument(
        who + ": backend '" + std::string(to_string(backend)) +
        "' is not implemented");
  }
}

std::optional<FormationControl> read_control(
    const holoscan::Tensor& tensor, const std::string& who) {
  require_dtype(tensor, DLDataType{kDLUInt, 8, 1}, "control", who);
  if (tensor.nbytes() < 1) {
    throw std::invalid_argument(who + ": control must have at least one "
                                      "element");
  }
  uint8_t value = 0;
  if (cudaMemcpy(&value, tensor.data(), 1, cudaMemcpyDefault) !=
      cudaSuccess) {
    throw std::runtime_error(who + ": failed to read control");
  }
  if (value > 3) {
    throw std::invalid_argument(
        who + ": control value " + std::to_string(value) +
        " is outside the FormationControl table");
  }
  return static_cast<FormationControl>(value);
}

void maybe_warn_locality(BackprojectionBackend backend, double D,
                         double r0_min, double frequency, bool* warned,
                         const std::string& who) {
  if (*warned || !is_taylor_backend(backend) || r0_min <= 0.0 ||
      !(D > 0.0)) {
    return;
  }
  const double bound = (D * D * D) / (2.0 * r0_min * r0_min);
  const double lambda = kSpeedOfLight / frequency;
  const double threshold = lambda / 16.0;
  if (bound > threshold) {
    HOLOSCAN_LOG_WARN(
        "{}: Taylor locality bound D^3/(2 R0^2) = {} m exceeds "
        "lambda/16 = {} m (D = {} m, R0 = {} m)",
        who, bound, threshold, D, r0_min);
  }
  *warned = true;
}

template <typename T>
void copy_sample_points_lattice(
    detail::BackprojectionState<T>& state, const holoscan::Tensor& points,
    int64_t num_v, int64_t num_u, cudaStream_t stream,
    const std::string& who) {
  const int64_t n = num_v * num_u * 3;
  state.lattice_xyz.ensure(static_cast<size_t>(n) * sizeof(T), stream);
  if (cudaMemcpyAsync(state.lattice_xyz.template as<T>(), points.data(),
                      sizeof(T) * static_cast<size_t>(n),
                      cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": sample_points copy failed");
  }
  state.lattice_num_v = num_v;
  state.lattice_num_u = num_u;
  state.lattice_valid = true;
  state.planes_valid = false;
}

}  // namespace

template <typename T>
BackprojectionOp<T>::~BackprojectionOp() = default;

template <typename T>
void BackprojectionOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
  metadata_policy(MetadataPolicy::kUpdate);

  const std::string& who = op_name<T>();
  control_connected_ = input_port_connected(this, "control");
  if (!control_connected_) {
    throw std::invalid_argument(
        who + ": input port 'control' must be connected (an unconnected "
              "control can never emit an image)");
  }
  sample_points_connected_ = input_port_connected(this, "sample_points");
  range_bins_connected_ = input_port_connected(this, "range_bins");

  effective_frame_ = coordinate_frame_.get();
  if (!effective_frame_.is_valid()) {
    throw std::invalid_argument(
        who + ": coordinate_frame is required");
  }
  effective_geometry_ = sample_geometry_.get();
  if (sample_points_connected_ && effective_geometry_.is_valid()) {
    throw std::invalid_argument(
        who + ": connecting sample_points and supplying sample_geometry "
              "selects both geometry modes; connect only one");
  }
  if (!sample_points_connected_ && !effective_geometry_.is_valid()) {
    throw std::invalid_argument(
        who + ": sample_geometry is required when sample_points is "
              "unconnected");
  }
  effective_spacing_ = range_bin_spacing_.try_get();
  effective_origin_ = range_bin_origin_.try_get();
  if (!range_bins_connected_ && !effective_spacing_.has_value()) {
    throw std::invalid_argument(
        who + ": range_bin_spacing is required when range_bins is "
              "unconnected");
  }
  effective_frequency_ = reference_frequency_.get();
  if (!(effective_frequency_ > 0.0) ||
      !std::isfinite(effective_frequency_)) {
    throw std::invalid_argument(
        who + ": reference_frequency must be finite and > 0");
  }
  effective_interpolation_ = interpolation_.get();
  require_linear(effective_interpolation_, who);
  backend_value_ = backend_.get();
  require_backend(backend_value_, who);
  geometry_change_value_ = geometry_change_.get();
  if (geometry_change_value_ == GeometryChange::kScroll) {
    throw std::invalid_argument(
        who + ": geometry_change 'scroll' is not implemented this slice");
  }
  state_ = std::make_shared<detail::BackprojectionState<T>>();
}

template <typename T>
void BackprojectionOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("range_profiles");
  spec.input<holoscan::TensorMap>("positions");
  spec.input<holoscan::TensorMap>("reference_position");
  spec.input<holoscan::TensorMap>("control")
      .condition(ConditionType::kNone);
  spec.input<holoscan::TensorMap>("range_bins")
      .condition(ConditionType::kNone);
  spec.input<holoscan::TensorMap>("sample_points")
      .condition(ConditionType::kNone);
  spec.input<holoscan::TensorMap>("weight")
      .condition(ConditionType::kNone);
  spec.input<holoscan::TensorMap>("initial_image")
      .condition(ConditionType::kNone);

  spec.output<holoscan::TensorMap>("image")
      .condition(ConditionType::kNone);
  spec.output<SampleGeometry>("sample_geometry")
      .condition(ConditionType::kNone);

  spec.param(coordinate_frame_, "coordinate_frame", "Coordinate frame",
             "Frame of every position in the graph. Static. Required.");
  spec.param(sample_geometry_, "sample_geometry", "Sample geometry",
             "Uniform image lattice. Ignored when sample_points is "
             "connected. Required otherwise.",
             SampleGeometry{});
  add_override_port<SampleGeometry>(spec, "sample_geometry");

  spec.param(range_bin_spacing_, "range_bin_spacing", "Range bin spacing",
             "Metres per range bin. Ignored when range_bins is connected.",
             ParameterFlag::kOptional);
  add_override_port<double>(spec, "range_bin_spacing");

  spec.param(range_bin_origin_, "range_bin_origin", "Range bin origin",
             "Range (m) of bin zero relative to the derived reference "
             "range. Default: centred, -floor(N/2)*spacing.",
             ParameterFlag::kOptional);
  add_override_port<double>(spec, "range_bin_origin");

  spec.param(reference_frequency_, "reference_frequency",
             "Reference frequency",
             "Transmit frequency (Hz) of the first fast-time sample.");
  add_override_port<double>(spec, "reference_frequency");

  spec.param(interpolation_, "interpolation", "Range interpolation",
             "Only linear is implemented this wave.",
             RangeInterpolation::kLinear);
  add_override_port<RangeInterpolation>(spec, "interpolation");

  spec.param(backend_, "backend", "Formation backend",
             "Internal arithmetic strategy. Static.",
             kDefaultBackprojectionBackend);

  spec.param(geometry_change_, "geometry_change", "Geometry change",
             "reset or scroll. scroll is rejected this slice.",
             GeometryChange::kReset);
}

template <typename T>
void BackprojectionOp<T>::compute(InputContext& op_input,
                                  OutputContext& op_output,
                                  [[maybe_unused]] ExecutionContext&
                                      context) {
  const std::string& who = op_name<T>();

  auto profiles = receive_single(op_input, "range_profiles", who);
  auto positions = receive_single(op_input, "positions", who);
  auto reference = receive_single(op_input, "reference_position", who);
  auto control_tensor = receive_optional(op_input, "control", who);
  auto range_bins = receive_optional(op_input, "range_bins", who);
  auto sample_points = receive_optional(op_input, "sample_points", who);
  auto weight = receive_optional(op_input, "weight", who);
  auto initial = receive_optional(op_input, "initial_image", who);

  if (initial) {
    throw std::invalid_argument(
        who + ": initial_image is not implemented this slice");
  }

  bool geometry_override = false;
  SampleGeometry pending_geometry;
  bool changed = false;
  SampleGeometry new_geometry = resolve_override_param(
      op_input, "sample_geometry", effective_geometry_, &changed);
  if (changed) {
    geometry_override = true;
    pending_geometry = std::move(new_geometry);
  }
  auto spacing_override = receive_override<double>(op_input,
                                                   "range_bin_spacing");
  if (spacing_override.has_value()) {
    effective_spacing_ = *spacing_override;
  }
  auto origin_override = receive_override<double>(op_input,
                                                 "range_bin_origin");
  if (origin_override.has_value()) {
    effective_origin_ = *origin_override;
  }
  effective_frequency_ = resolve_override_param(
      op_input, "reference_frequency", effective_frequency_, nullptr);
  effective_interpolation_ = resolve_override_param(
      op_input, "interpolation", effective_interpolation_, nullptr);
  require_linear(effective_interpolation_, who);

  std::optional<FormationControl> control;
  if (control_tensor) {
    control = read_control(*control_tensor, who);
  }

  classify_input_dtype<T>(*profiles, "range_profiles",
                          /*allow_real=*/false);
  require_dtype(*positions, real_dtype<T>(), "positions", who);
  require_dtype(*reference, real_dtype<T>(), "reference_position", who);
  require_contiguous(*profiles, "range_profiles", who);
  require_contiguous(*positions, "positions", who);
  require_contiguous(*reference, "reference_position", who);
  require_cuda_accessible(*profiles, "range_profiles", who);
  require_cuda_accessible(*positions, "positions", who);
  require_cuda_accessible(*reference, "reference_position", who);

  const BatchedShape cube = collapse_batch(*profiles, "range_profiles");
  const BatchedShape pos = collapse_batch(*positions, "positions");
  const BatchedShape ref = collapse_batch(*reference, "reference_position");
  if (pos.cols != 3 || ref.cols != 3) {
    throw std::invalid_argument(
        who + ": positions and reference_position last axis must be 3");
  }
  if (pos.rows != cube.rows || ref.rows != cube.rows ||
      pos.batch != cube.batch || ref.batch != cube.batch) {
    throw std::invalid_argument(
        who + ": positions / reference_position pulse and batch axes "
              "must match range_profiles");
  }

  cudaStream_t stream = op_input.receive_cuda_stream("range_profiles");

  bool points_changed = false;
  bool install_first_points = false;
  int64_t incoming_v = 0;
  int64_t incoming_u = 0;
  if (sample_points_connected_ && sample_points) {
    require_dtype(*sample_points, real_dtype<T>(), "sample_points", who);
    require_contiguous(*sample_points, "sample_points", who);
    require_cuda_accessible(*sample_points, "sample_points", who);
    if (sample_points->ndim() != 3 ||
        sample_points->shape()[2] != 3) {
      throw std::invalid_argument(
          who + ": sample_points must have shape (num_v, num_u, 3)");
    }
    incoming_v = sample_points->shape()[0];
    incoming_u = sample_points->shape()[1];
    if (!state_->lattice_valid) {
      // First lattice is not a change; install it this tick.
      install_first_points = true;
    } else if (state_->lattice_num_v == incoming_v &&
               state_->lattice_num_u == incoming_u) {
      const int64_t n = incoming_v * incoming_u * 3;
      if (detail::device_arrays_differ<T>(
              state_->lattice_xyz.template as<T>(),
              static_cast<const T*>(sample_points->data()), n,
              state_->cmp_flag, stream)) {
        points_changed = true;
      }
    } else {
      points_changed = true;
    }
  }

  const bool geometry_change = geometry_override || points_changed;
  if (geometry_change) {
    const bool resetting =
        control.has_value() && is_resetting(*control);
    if (!resetting) {
      throw std::invalid_argument(
          who + ": a geometry change must arrive with a resetting "
                "control message");
    }
  }
  // emit-and-reset applies the new lattice *after* this tick's emit
  // (the closing aperture was formed on the old geometry). Pre-reset
  // controls install before this tick's accumulate.
  const bool apply_before =
      geometry_change && control.has_value() && is_pre_reset(*control);
  const bool apply_after =
      geometry_change && control.has_value() &&
      *control == FormationControl::kEmitAndReset;

  auto install_pending_geometry = [&]() {
    if (geometry_override) {
      effective_geometry_ = pending_geometry;
      if (!sample_points_connected_) {
        state_->lattice_valid = false;
        state_->planes_valid = false;
      }
      locality_warned_ = false;
    }
    if (points_changed) {
      copy_sample_points_lattice(*state_, *sample_points, incoming_v,
                                 incoming_u, stream, who);
      locality_warned_ = false;
    }
  };

  if (apply_before) {
    install_pending_geometry();
  }
  if (install_first_points) {
    copy_sample_points_lattice(*state_, *sample_points, incoming_v,
                               incoming_u, stream, who);
  }

  if (control.has_value() && is_pre_reset(*control)) {
    if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
      detail::reset_matx_accum(*state_, stream);
#endif
    } else {
      detail::reset_accum(*state_, stream);
    }
    unemitted_pulses_ = 0;
  }

  const int64_t num_bins = cube.cols;
  const int64_t num_pulses = cube.rows;
  double spacing = 0.0;
  double origin = 0.0;
  if (range_bins) {
    require_dtype(*range_bins, real_dtype<T>(), "range_bins", who);
    require_contiguous(*range_bins, "range_bins", who);
    if (range_bins->ndim() != 1 ||
        range_bins->shape()[0] != num_bins) {
      throw std::invalid_argument(
          who + ": range_bins length must equal num_range_bins");
    }
    std::vector<T> bins(static_cast<size_t>(num_bins));
    if (cudaMemcpy(bins.data(), range_bins->data(),
                   sizeof(T) * static_cast<size_t>(num_bins),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(who + ": failed to read range_bins");
    }
    origin = static_cast<double>(bins[0]);
    if (num_bins < 2) {
      throw std::invalid_argument(
          who + ": range_bins must have at least two samples");
    }
    // Endpoint spacing. Uniformity is the spread of consecutive
    // steps, which stays ~0.5 ULP of the farthest bin for a
    // double-computed axis stored in T (measured 512..16384). A
    // reconstructed-line check against those endpoints accumulates
    // the endpoint rounding with N and failed at 2048 after passing
    // at 512. That is how SARRangeCompressionOp emits the axis this
    // graph wires (D21).
    const double last = static_cast<double>(
        bins[static_cast<size_t>(num_bins - 1)]);
    spacing = (last - origin) / static_cast<double>(num_bins - 1);
    const double ulp =
        static_cast<double>(std::numeric_limits<T>::epsilon());
    const double max_abs = std::max(
        {std::abs(origin), std::abs(last), 1.0});
    double min_step = spacing;
    double max_step = spacing;
    for (int64_t i = 1; i < num_bins; ++i) {
      const double step = static_cast<double>(
                              bins[static_cast<size_t>(i)]) -
                          static_cast<double>(
                              bins[static_cast<size_t>(i - 1)]);
      min_step = std::min(min_step, step);
      max_step = std::max(max_step, step);
    }
    if (max_step - min_step > 4.0 * ulp * max_abs) {
      throw std::invalid_argument(
          who + ": range_bins is not a uniform axis");
    }
    effective_spacing_ = spacing;
    effective_origin_ = origin;
  } else {
    if (!effective_spacing_.has_value()) {
      throw std::invalid_argument(
          who + ": range_bin_spacing is required when range_bins is "
                "unconnected");
    }
    spacing = *effective_spacing_;
    origin = effective_origin_.value_or(
        -std::floor(static_cast<double>(num_bins) / 2.0) * spacing);
  }
  if (!(spacing > 0.0) || !std::isfinite(spacing)) {
    throw std::invalid_argument(
        who + ": range bin spacing must be finite and > 0");
  }

  int64_t num_v = 0;
  int64_t num_u = 0;
  const T* lattice_ptr = nullptr;
  if (sample_points_connected_) {
    if (!state_->lattice_valid) {
      throw std::invalid_argument(
          who + ": sample_points is connected but no lattice has arrived");
    }
    num_v = state_->lattice_num_v;
    num_u = state_->lattice_num_u;
    lattice_ptr = state_->lattice_xyz.template as<T>();
  } else {
    if (!effective_geometry_.is_valid()) {
      throw std::invalid_argument(who + ": sample_geometry is required");
    }
    num_v = effective_geometry_.num_v();
    num_u = effective_geometry_.num_u();
    if (!state_->lattice_valid || state_->lattice_num_v != num_v ||
        state_->lattice_num_u != num_u) {
      // Uniform lattice: expand on device. NVRadar writes x/y/z planes
      // directly (no 1.61 GiB packed H2D). MatX still needs packed xyz.
      const bool packed = detail::backend_is_matx(backend_value_);
      detail::expand_geometry_on_device(*state_, effective_geometry_,
                                        packed, stream);
      state_->lattice_valid = true;
      state_->planes_valid = !packed;
      locality_warned_ = false;
    }
    if (detail::backend_is_matx(backend_value_)) {
      lattice_ptr = state_->lattice_xyz.template as<T>();
    }
  }

  if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
    detail::ensure_matx_accum(*state_, cube.batch, num_v, num_u, stream);
#endif
  } else {
    detail::ensure_accum(*state_, cube.batch, num_v, num_u, stream);
    if (!state_->planes_valid) {
      if (lattice_ptr == nullptr) {
        throw std::runtime_error(who + ": lattice planes were not built");
      }
      detail::gather_lattice(*state_, lattice_ptr, num_v * num_u,
                             stream);
      state_->planes_valid = true;
    }
  }

  const T* profile_ptr = static_cast<const T*>(profiles->data());
  if (weight) {
    require_dtype(*weight, real_dtype<T>(), "weight", who);
    require_contiguous(*weight, "weight", who);
    require_cuda_accessible(*weight, "weight", who);
    const auto wshape = weight->shape();
    if (wshape.empty() || wshape.back() != num_pulses) {
      throw std::invalid_argument(
          who + ": weight last axis must equal num_pulses");
    }
    int64_t wbatch = 1;
    for (size_t d = 0; d + 1 < wshape.size(); ++d) {
      wbatch *= wshape[d];
    }
    if (wbatch != cube.batch) {
      throw std::invalid_argument(
          who + ": weight leading axes must product to the cube batch");
    }
    const int64_t nweight = wbatch * num_pulses;
    std::vector<T> host_w(static_cast<size_t>(nweight));
    if (cudaMemcpyAsync(host_w.data(), weight->data(),
                        sizeof(T) * static_cast<size_t>(nweight),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      throw std::runtime_error(who + ": failed to read weight");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error(who + ": weight sync failed");
    }
    bool all_one = true;
    for (T w : host_w) {
      if (w != static_cast<T>(1)) {
        all_one = false;
        break;
      }
    }
    if (!all_one) {
      const int64_t n_batch = num_pulses * num_bins;
      const int64_t n = cube.batch * n_batch;
      state_->weight_scratch.ensure(
          static_cast<size_t>(n) * 2 * sizeof(T), stream);
      const int threads = 256;
      const int blocks =
          static_cast<int>((n_batch + threads - 1) / threads);
      for (int64_t b = 0; b < cube.batch; ++b) {
        const T* src = static_cast<const T*>(profiles->data()) +
                       b * n_batch * 2;
        const T* wptr = static_cast<const T*>(weight->data()) +
                        b * num_pulses;
        T* dst = state_->weight_scratch.template as<T>() +
                 b * n_batch * 2;
        detail::scale_interleaved_kernel<T>
            <<<blocks, threads, 0, stream>>>(
                src, wptr, dst, static_cast<int>(num_pulses),
                static_cast<int>(num_bins));
      }
      profile_ptr = state_->weight_scratch.template as<T>();
    }
  }

  const double pc_partial =
      4.0 * kPi * effective_frequency_ / kSpeedOfLight;
  const int64_t pulse_stride = num_pulses * 3;
  const int64_t profile_stride = num_pulses * num_bins * 2;
  for (int64_t b = 0; b < cube.batch; ++b) {
    const T* pos_b = static_cast<const T*>(positions->data()) +
                     b * pulse_stride;
    const T* ref_b = static_cast<const T*>(reference->data()) +
                     b * pulse_stride;
    const T* prof_b = profile_ptr + b * profile_stride;
    detail::prepare_pulses(*state_, pos_b, ref_b,
                           static_cast<int>(num_pulses), stream);
    if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
      detail::launch_matx<T>(
          backend_value_, *state_, prof_b, pos_b, lattice_ptr,
          static_cast<int>(num_v), static_cast<int>(num_u),
          static_cast<int>(num_pulses), static_cast<int>(num_bins),
          static_cast<int>(b), effective_frequency_, spacing, stream);
#else
      throw std::invalid_argument(
          who + ": backend '" + std::string(to_string(backend_value_)) +
          "' needs matx/kernels/sar_bp.cuh");
#endif
    } else {
      detail::launch_nvradar<T>(
          backend_value_, *state_, prof_b, static_cast<int>(num_v),
          static_cast<int>(num_u), static_cast<int>(num_pulses),
          static_cast<int>(num_bins), static_cast<int>(b), origin, spacing,
          pc_partial, stream);
    }
  }

  if (!locality_warned_ && is_taylor_backend(backend_value_)) {
    std::vector<double> r0_host(static_cast<size_t>(num_pulses));
    if (cudaMemcpyAsync(r0_host.data(), state_->r0.template as<double>(),
                        sizeof(double) * static_cast<size_t>(num_pulses),
                        cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess) {
      double r0_min = r0_host[0];
      for (double r : r0_host) {
        r0_min = std::min(r0_min, r);
      }
      double D = 0.0;
      if (sample_points_connected_) {
        if (!state_->planes_valid && lattice_ptr != nullptr) {
          detail::gather_lattice(*state_, lattice_ptr, num_v * num_u,
                                 stream);
          state_->planes_valid = true;
        }
        if (state_->planes_valid) {
          D = detail::max_tile_diameter(*state_, num_v, num_u, stream);
        }
      } else if (effective_geometry_.is_valid()) {
        const double sv = effective_geometry_.spacing()[0];
        const double su = effective_geometry_.spacing()[1];
        D = 16.0 * std::hypot(sv, su);
      }
      maybe_warn_locality(backend_value_, D, r0_min, effective_frequency_,
                          &locality_warned_, who);
    }
  }

  unemitted_pulses_ += num_pulses * cube.batch;

  if (control.has_value() && is_emit(*control)) {
    auto image = make_device_tensor(
        full_rank_shape(cube, num_v, num_u), complex_dtype<T>());
    if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
      detail::emit_matx_accum(*state_, static_cast<T*>(image->data()),
                              cube.batch, num_v, num_u, stream);
#endif
    } else {
      detail::emit_accum(*state_, static_cast<T*>(image->data()),
                         cube.batch, num_v, num_u, stream);
    }
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "image");
    }
    emit_single(op_output, "image", image);
    if (!sample_points_connected_ && effective_geometry_.is_valid()) {
      op_output.emit(effective_geometry_, "sample_geometry");
    }
    unemitted_pulses_ = 0;
  }

  if (control.has_value() &&
      *control == FormationControl::kEmitAndReset) {
    if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
      detail::reset_matx_accum(*state_, stream);
#endif
    } else {
      detail::reset_accum(*state_, stream);
    }
  }

  if (apply_after) {
    install_pending_geometry();
  }

  retained_profiles_ = std::move(profiles);
  retained_positions_ = std::move(positions);
  retained_reference_ = std::move(reference);
  if (sample_points) {
    retained_sample_points_ = std::move(sample_points);
  }
  if (weight) {
    retained_weight_ = std::move(weight);
  }
}

template <typename T>
void BackprojectionOp<T>::stop() {
  if (unemitted_pulses_ > 0) {
    HOLOSCAN_LOG_WARN(
        "{}: stopping with {} unemitted pulse(s) still in the "
        "accumulator",
        op_name<T>(), unemitted_pulses_);
  }
}

template class BackprojectionOp<float>;
template class BackprojectionOp<double>;

}  // namespace holoscan::radar
