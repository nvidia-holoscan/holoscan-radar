// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/backprojection_op/backprojection_op.hpp"

#include <algorithm>
#include <chrono>
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
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/logger/logger.hpp>
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/backprojection_op/backprojection_stage.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

template <typename T, typename TGeom>
const std::string& op_name() {
  // Name the geometry precision only when it differs, so the overwhelmingly
  // common matched instantiations keep the name their messages already use.
  static const std::string name = std::is_same_v<T, TGeom>
                                      ? std::string("BackprojectionOp<") + precision_name<T>() + ">"
                                      : std::string("BackprojectionOp<") + precision_name<T>() +
                                            ", " + precision_name<TGeom>() + ">";
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

void require_linear(RangeInterpolation interpolation, const std::string& who) {
  if (interpolation != RangeInterpolation::kLinear) {
    throw std::invalid_argument(who + ": interpolation '" + std::string(to_string(interpolation)) +
                                "' is not implemented (only linear is)");
  }
}

void require_backend(BackprojectionBackend backend, const std::string& who) {
  if (backend == BackprojectionBackend::kMatxTaylorFast) {
    throw std::invalid_argument(who + ": backend 'matx_taylor_fast' is not in this MatX drop");
  }
  if (!detail::backend_is_implemented(backend)) {
    throw std::invalid_argument(who + ": backend '" + std::string(to_string(backend)) +
                                "' is not implemented");
  }
}

int64_t scroll_translation_rows(const SampleGeometry& from, const SampleGeometry& to,
                                const std::string& who) {
  if (from.axis_u() != to.axis_u() || from.axis_v() != to.axis_v() ||
      from.spacing() != to.spacing() || from.counts() != to.counts()) {
    throw std::invalid_argument(who +
                                ": geometry_change 'scroll' requires a pure translation "
                                "along v (axes, spacing, and counts unchanged)");
  }
  const auto& av = from.axis_v();
  const auto& au = from.axis_u();
  const double sv = from.spacing()[0];
  double dv = 0.0;
  double du = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double d = to.origin()[i] - from.origin()[i];
    dv += d * av[i];
    du += d * au[i];
  }
  double dn2 = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double d = to.origin()[i] - from.origin()[i];
    const double recon = dv * av[i] + du * au[i];
    const double r = d - recon;
    dn2 += r * r;
  }
  const double tol = 1.0e-3 * std::max(sv, 1.0e-6);
  if (std::abs(du) > tol || std::sqrt(dn2) > tol) {
    throw std::invalid_argument(who +
                                ": geometry_change 'scroll' translation must be along v "
                                "with no u component");
  }
  const double k_real = dv / sv;
  const int64_t k = std::llround(k_real);
  if (std::abs(k_real - static_cast<double>(k)) * sv > tol) {
    throw std::invalid_argument(who +
                                ": geometry_change 'scroll' translation must be a whole "
                                "number of pixels along v");
  }
  if (k < 0) {
    throw std::invalid_argument(who + ": geometry_change 'scroll' cannot move backward along v");
  }
  if (k >= from.num_v()) {
    throw std::invalid_argument(who +
                                ": a scroll of k >= num_v retains nothing; send a "
                                "resetting control message");
  }
  return k;
}

std::optional<FormationControl> read_control(const holoscan::Tensor& tensor,
                                             const std::string& who) {
  require_dtype(tensor, DLDataType{kDLUInt, 8, 1}, "control", who);
  if (tensor.nbytes() < 1) {
    throw std::invalid_argument(who +
                                ": control must have at least one "
                                "element");
  }
  uint8_t value = 0;
  if (cudaMemcpy(&value, tensor.data(), 1, cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error(who + ": failed to read control");
  }
  if (value > 3) {
    throw std::invalid_argument(who + ": control value " + std::to_string(value) +
                                " is outside the FormationControl table");
  }
  return static_cast<FormationControl>(value);
}

void maybe_warn_locality(BackprojectionBackend backend, double D, double r0_min, double frequency,
                         bool* warned, const std::string& who) {
  if (*warned || !is_taylor_backend(backend) || r0_min <= 0.0 || !(D > 0.0)) {
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

template <typename T, typename TGeom>
void copy_sample_points_lattice(detail::BackprojectionState<T, TGeom>& state,
                                const holoscan::Tensor& points, int64_t num_v, int64_t num_u,
                                cudaStream_t stream, const std::string& who) {
  const int64_t n = num_v * num_u * 3;
  state.lattice_xyz.ensure(static_cast<size_t>(n) * sizeof(TGeom), stream);
  if (cudaMemcpyAsync(state.lattice_xyz.template as<TGeom>(), points.data(),
                      sizeof(TGeom) * static_cast<size_t>(n), cudaMemcpyDeviceToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error(who + ": sample_points copy failed");
  }
  state.lattice_num_v = num_v;
  state.lattice_num_u = num_u;
  state.lattice_valid = true;
  state.planes_valid = false;
}

}  // namespace

template <typename T, typename TGeom>
BackprojectionOp<T, TGeom>::BackprojectionOp(const MaxTensorExtents& extents,
                                             std::shared_ptr<FormationHud> hud)
: extents_(extents), hud_(std::move(hud)) {
  require_extent(extents.max_pulses, "max_pulses", "BackprojectionOp");
  require_extent(extents.max_range_bins, "max_range_bins", "BackprojectionOp");
  require_extent(extents.max_image_v, "max_image_v", "BackprojectionOp");
  require_extent(extents.max_image_u, "max_image_u", "BackprojectionOp");
}
template <typename T, typename TGeom>
BackprojectionOp<T, TGeom>::~BackprojectionOp() = default;

template <typename T, typename TGeom>
void BackprojectionOp<T, TGeom>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  constexpr TensorRepresentation host{
      .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLFloat, 64U, 1U}, .rank = 1U};
  // Setup default 16: continuous strip fans 4 BP ticks per reader cube. Per-edge
  // ConnectionOptions still set destination depth (other apps keep 4).
  spec.input(range_profiles).queue_depth(16U).expects_tensor(tensor_input(device));
  spec.input(positions).queue_depth(16U).expects_tensor(tensor_input(device));
  spec.input(reference_position).queue_depth(16U).expects_tensor(tensor_input(device));
  spec.input(control).optional().queue_depth(16U);
  spec.input(range_bins).optional().queue_depth(4U);
  spec.input(sample_points).optional().queue_depth(4U);
  spec.input(weight).optional().queue_depth(4U);
  spec.input(initial_image).optional().queue_depth(4U);
  spec.input(sample_geometry).optional().queue_depth(16U);
  spec.input(range_bin_spacing).optional().queue_depth(4U);
  spec.input(range_bin_origin).optional().queue_depth(4U);
  spec.input(reference_frequency).optional().queue_depth(4U);
  spec.input(interpolation).optional().queue_depth(4U);
  spec.output(image).produces_tensor(pooled_tensor_output(
      device, 2U,
      tensor_byte_span(std::vector<int64_t>{*extents_.max_image_v, *extents_.max_image_u},
                       complex_dtype<T>())));
  spec.output(sample_geometry_out)
      .produces_tensor(pooled_tensor_output(host, 1U, 11U * sizeof(double)));
}

template <typename T, typename TGeom>
Contract BackprojectionOp<T, TGeom>::contract() const {
  Contract result;
  result
      .trigger(OnAligned{.ports = ports(range_profiles, positions, reference_position),
                         .window = std::chrono::nanoseconds{0}})
      .input(range_profiles, Exact{})
      .input(positions, Exact{})
      .input(reference_position, Exact{});
  add_live_override_to_contract(result, control);
  add_live_override_to_contract(result, range_bins);
  add_live_override_to_contract(result, sample_points);
  add_live_override_to_contract(result, weight);
  add_live_override_to_contract(result, initial_image);
  add_live_override_to_contract(result, sample_geometry);
  add_live_override_to_contract(result, range_bin_spacing);
  add_live_override_to_contract(result, range_bin_origin);
  add_live_override_to_contract(result, reference_frequency);
  add_live_override_to_contract(result, interpolation);
  return result;
}

template <typename T, typename TGeom>
void BackprojectionOp<T, TGeom>::start() {
  const auto& p = params();
  const std::string& who = op_name<T, TGeom>();
  effective_frame_ = p.coordinate_frame;
  if (!effective_frame_.is_valid())
    throw std::invalid_argument(who + ": coordinate_frame required");
  effective_geometry_ = p.sample_geometry;
  effective_spacing_ = p.range_bin_spacing;
  effective_origin_ = p.range_bin_origin;
  effective_frequency_ = p.reference_frequency;
  if (!(effective_frequency_ > 0.0) || !std::isfinite(effective_frequency_))
    throw std::invalid_argument(who + ": reference_frequency must be positive");
  effective_interpolation_ = p.interpolation;
  require_linear(effective_interpolation_, who);
  backend_value_ = p.backend;
  require_backend(backend_value_, who);
  geometry_change_value_ = p.geometry_change;
  state_ = std::make_shared<detail::BackprojectionState<T, TGeom>>();
}

template <typename T, typename TGeom>
expected<void, Error> BackprojectionOp<T, TGeom>::compute(ExecutionContext& context) {
  try {
    const std::string& who = op_name<T, TGeom>();

    // Join this pulse block with same-tick controls and geometry overrides.
    HOLOSCAN_TRY(auto profiles_value, receive_tensor(range_profiles, who));
    HOLOSCAN_TRY(auto positions_value, receive_tensor(positions, who));
    HOLOSCAN_TRY(auto reference_value, receive_tensor(reference_position, who));
    auto profiles = std::make_shared<Tensor>(std::move(profiles_value));
    auto positions = std::make_shared<Tensor>(std::move(positions_value));
    auto reference = std::make_shared<Tensor>(std::move(reference_value));
    auto optional_tensor = [](Input<Tensor>& port) -> expected<std::shared_ptr<Tensor>, Error> {
      HOLOSCAN_TRY(auto value, receive_optional_value(port));
      if (!value)
        return std::shared_ptr<Tensor>{};
      return std::make_shared<Tensor>(std::move(*value));
    };
    HOLOSCAN_TRY(auto control_tensor, optional_tensor(control));
    HOLOSCAN_TRY(auto range_bins_value, optional_tensor(range_bins));
    HOLOSCAN_TRY(auto sample_points_value, optional_tensor(sample_points));
    HOLOSCAN_TRY(auto weight_value, optional_tensor(weight));
    HOLOSCAN_TRY(auto initial, optional_tensor(initial_image));
    auto range_bins = std::move(range_bins_value);
    auto sample_points = std::move(sample_points_value);
    auto weight = std::move(weight_value);
    if (initial)
      throw std::invalid_argument(who + ": initial_image is not implemented");
    if (sample_points)
      sample_points_connected_ = true;
    if (weight)
      weight_connected_ = true;
    if (weight_connected_ && geometry_change_value_ == GeometryChange::kScroll)
      throw std::invalid_argument(who + ": scroll does not support weight");
    std::optional<FormationControl> control_value;
    if (control_tensor)
      control_value = read_control(*control_tensor, who);
    // Latch the reader's last packet now: this tick's pulses are in hand, and
    // the live mailbox moves into the next collect while this emit is on GPU.
    std::optional<std::chrono::steady_clock::time_point> tick_last_packet;
    PlaybackStatus tick_playback{};
    if (hud_) {
      const auto snap = hud_->snapshot();
      tick_last_packet = snap.last_packet_at;
      tick_playback = snap.last_batch_playback;
    }
    const bool stage_after = control_value && *control_value == FormationControl::kEmitAndReset;
    // Whether this emission completes an aperture, which is not the same as
    // staging a reset after it. A reform replay resets first and then forms
    // the whole retained aperture in one pass (kResetThenEmit), so its image
    // is terminal even though nothing is staged behind it. Reported to the
    // HUD so the end-of-stream drain can recognise that last image.
    const bool completes_aperture =
        stage_after || (control_value && *control_value == FormationControl::kResetThenEmit);
    HOLOSCAN_TRY(auto geometry_override_wire, receive_optional_value(sample_geometry));
    SampleGeometry new_geometry = params().sample_geometry;
    if (geometry_override_wire) {
      HOLOSCAN_TRY(auto decoded, from_wire(*geometry_override_wire));
      new_geometry = std::move(decoded);
    }
    bool geometry_override = new_geometry.is_valid() && new_geometry != effective_geometry_;
    SampleGeometry pending_geometry = new_geometry;
    HOLOSCAN_TRY(auto spacing_override, receive_optional_value(range_bin_spacing));
    HOLOSCAN_TRY(auto origin_override, receive_optional_value(range_bin_origin));
    HOLOSCAN_TRY(auto frequency_override, receive_optional_value(reference_frequency));
    HOLOSCAN_TRY(auto interpolation_override, receive_optional_value(interpolation));
    std::optional<double> pending_spacing;
    std::optional<double> pending_origin;
    std::optional<double> pending_frequency;
    if (spacing_override) {
      if (stage_after)
        pending_spacing = *spacing_override;
      else
        effective_spacing_ = *spacing_override;
    } else {
      effective_spacing_ = params().range_bin_spacing;
    }
    if (origin_override) {
      if (stage_after)
        pending_origin = *origin_override;
      else
        effective_origin_ = *origin_override;
    } else {
      effective_origin_ = params().range_bin_origin;
    }
    if (frequency_override) {
      if (stage_after)
        pending_frequency = *frequency_override;
      else
        effective_frequency_ = *frequency_override;
    } else {
      effective_frequency_ = params().reference_frequency;
    }
    effective_interpolation_ = interpolation_override.value_or(params().interpolation);
    require_linear(effective_interpolation_, who);

    // Validate the named pulse and image axes before touching the accumulator.
    classify_input_dtype<T>(*profiles, "range_profiles",
                            /*allow_real=*/false);
    require_dtype(*positions, real_dtype<TGeom>(), "positions", who);
    require_dtype(*reference, real_dtype<TGeom>(), "reference_position", who);
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
      throw std::invalid_argument(who + ": positions and reference_position last axis must be 3");
    }
    if (pos.rows != cube.rows || ref.rows != cube.rows || pos.batch != cube.batch ||
        ref.batch != cube.batch) {
      throw std::invalid_argument(who +
                                  ": positions / reference_position pulse and batch axes "
                                  "must match range_profiles");
    }

    cudaStream_t stream = context.cuda_stream().get();

    bool points_changed = false;
    bool install_first_points = false;
    bool points_scroll = false;
    bool grid_scroll = false;
    int64_t scroll_k = 0;
    int64_t incoming_v = 0;
    int64_t incoming_u = 0;
    if (sample_points) {
      require_dtype(*sample_points, real_dtype<TGeom>(), "sample_points", who);
      require_contiguous(*sample_points, "sample_points", who);
      require_cuda_accessible(*sample_points, "sample_points", who);
      if (sample_points->ndim() != 3 || sample_points->shape()[2] != 3) {
        throw std::invalid_argument(who + ": sample_points must have shape (num_v, num_u, 3)");
      }
      incoming_v = sample_points->shape()[0];
      incoming_u = sample_points->shape()[1];
      if (!state_->lattice_valid) {
        // First lattice is not a change; install it this tick.
        install_first_points = true;
      } else if (geometry_change_value_ == GeometryChange::kScroll &&
                 incoming_u == state_->lattice_num_u && incoming_v > 0 &&
                 incoming_v < state_->lattice_num_v &&
                 !(control_value.has_value() && is_resetting(*control_value))) {
        points_scroll = true;
        scroll_k = incoming_v;
      } else if (geometry_change_value_ == GeometryChange::kScroll &&
                 incoming_u != state_->lattice_num_u &&
                 !(control_value.has_value() && is_resetting(*control_value))) {
        throw std::invalid_argument(who +
                                    ": sample_points num_u must match the retained lattice "
                                    "under geometry_change 'scroll'");
      } else if (geometry_change_value_ == GeometryChange::kScroll &&
                 incoming_v >= state_->lattice_num_v &&
                 !(control_value.has_value() && is_resetting(*control_value))) {
        throw std::invalid_argument(who +
                                    ": a scroll of k >= num_v retains nothing; send a "
                                    "resetting control message");
      } else if (state_->lattice_num_v == incoming_v && state_->lattice_num_u == incoming_u) {
        const int64_t n = incoming_v * incoming_u * 3;
        if (detail::device_arrays_differ<TGeom>(state_->lattice_xyz.template as<TGeom>(),
                                                static_cast<const TGeom*>(sample_points->data()), n,
                                                state_->cmp_flag, stream)) {
          points_changed = true;
        }
      } else {
        points_changed = true;
      }
    }

    const bool resetting = control_value.has_value() && is_resetting(*control_value);
    if (geometry_override && geometry_change_value_ == GeometryChange::kScroll && !resetting) {
      scroll_k = scroll_translation_rows(effective_geometry_, pending_geometry, who);
      grid_scroll = scroll_k > 0;
      geometry_override = false;
    }

    const bool geometry_change = geometry_override || points_changed;
    if (geometry_change) {
      if (!resetting) {
        throw std::invalid_argument(who +
                                    ": a geometry change must arrive with a resetting "
                                    "control message");
      }
    }
    // emit-and-reset applies the new lattice *after* this tick's emit
    // (the closing aperture was formed on the old geometry). Pre-reset
    // controls install before this tick's accumulate.
    const bool apply_before =
        geometry_change && control_value.has_value() && is_pre_reset(*control_value);
    const bool apply_after = geometry_change && control_value.has_value() &&
                             *control_value == FormationControl::kEmitAndReset;

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
        copy_sample_points_lattice(*state_, *sample_points, incoming_v, incoming_u, stream, who);
        locality_warned_ = false;
      }
    };

    if (apply_before) {
      install_pending_geometry();
    }
    if (install_first_points) {
      copy_sample_points_lattice(*state_, *sample_points, incoming_v, incoming_u, stream, who);
    }

    if (control_value.has_value() && is_pre_reset(*control_value)) {
      if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
        detail::reset_matx_accum(*state_, stream);
#endif
      } else {
        detail::reset_accum(*state_, stream);
      }
      unemitted_pulses_ = 0;
      aperture_compute_s_ = 0.0;
      state_->discard_tick_gpu_timing();
    }

    const int64_t num_bins = cube.cols;
    const int64_t num_pulses = cube.rows;
    double spacing = 0.0;
    double origin = 0.0;
    if (range_bins) {
      require_dtype(*range_bins, real_dtype<T>(), "range_bins", who);
      require_contiguous(*range_bins, "range_bins", who);
      if (range_bins->ndim() != 1 || range_bins->shape()[0] != num_bins) {
        throw std::invalid_argument(who + ": range_bins length must equal num_range_bins");
      }
      std::vector<T> bins(static_cast<size_t>(num_bins));
      if (cudaMemcpy(bins.data(), range_bins->data(), sizeof(T) * static_cast<size_t>(num_bins),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error(who + ": failed to read range_bins");
      }
      origin = static_cast<double>(bins[0]);
      if (num_bins < 2) {
        throw std::invalid_argument(who + ": range_bins must have at least two samples");
      }
      // Endpoint spacing. Uniformity is the spread of consecutive
      // steps, which stays ~0.5 ULP of the farthest bin for a
      // double-computed axis stored in T (measured 512..16384). A
      // reconstructed-line check against those endpoints accumulates
      // the endpoint rounding with N and failed at 2048 after passing
      // at 512. That is how SARRangeCompressionOp emits the axis this
      // graph wires (D21).
      const double last = static_cast<double>(bins[static_cast<size_t>(num_bins - 1)]);
      spacing = (last - origin) / static_cast<double>(num_bins - 1);
      const double ulp = static_cast<double>(std::numeric_limits<T>::epsilon());
      const double max_abs = std::max({std::abs(origin), std::abs(last), 1.0});
      double min_step = spacing;
      double max_step = spacing;
      for (int64_t i = 1; i < num_bins; ++i) {
        const double step = static_cast<double>(bins[static_cast<size_t>(i)]) -
                            static_cast<double>(bins[static_cast<size_t>(i - 1)]);
        min_step = std::min(min_step, step);
        max_step = std::max(max_step, step);
      }
      if (max_step - min_step > 4.0 * ulp * max_abs) {
        throw std::invalid_argument(who + ": range_bins is not a uniform axis");
      }
      effective_spacing_ = spacing;
      effective_origin_ = origin;
    } else {
      if (!effective_spacing_.has_value()) {
        throw std::invalid_argument(who +
                                    ": range_bin_spacing is required when range_bins is "
                                    "unconnected");
      }
      spacing = *effective_spacing_;
      origin =
          effective_origin_.value_or(-std::floor(static_cast<double>(num_bins) / 2.0) * spacing);
    }
    if (!(spacing > 0.0) || !std::isfinite(spacing)) {
      throw std::invalid_argument(who + ": range bin spacing must be finite and > 0");
    }

    int64_t num_v = 0;
    int64_t num_u = 0;
    const TGeom* lattice_ptr = nullptr;
    if (sample_points_connected_) {
      if (!state_->lattice_valid) {
        throw std::invalid_argument(who +
                                    ": sample_points is connected but no lattice has arrived");
      }
      num_v = state_->lattice_num_v;
      num_u = state_->lattice_num_u;
      lattice_ptr = state_->lattice_xyz.template as<TGeom>();
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
        detail::expand_geometry_on_device(*state_, effective_geometry_, packed, stream);
        state_->lattice_valid = true;
        state_->planes_valid = !packed;
        locality_warned_ = false;
      }
      if (detail::backend_is_matx(backend_value_)) {
        lattice_ptr = state_->lattice_xyz.template as<TGeom>();
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
        detail::gather_lattice(*state_, lattice_ptr, num_v * num_u, stream);
        state_->planes_valid = true;
      }
    }

    if (grid_scroll || points_scroll) {
      const int64_t k = scroll_k;
      const int64_t phys0 = state_->ring_offset;
      const bool packed = detail::backend_is_matx(backend_value_);
      if (packed) {
#if __has_include("matx/kernels/sar_bp.cuh")
        detail::zero_matx_rows(*state_, phys0, k, stream);
#endif
      } else {
        detail::zero_nvradar_rows(*state_, phys0, k, stream);
      }
      if (grid_scroll) {
        effective_geometry_ = pending_geometry;
        detail::expand_geometry_rows(*state_, effective_geometry_, packed, num_v - k, k, phys0,
                                     stream);
        if (!packed) {
          state_->planes_valid = true;
        }
      } else {
        detail::copy_entering_xyz(*state_, static_cast<const TGeom*>(sample_points->data()), k,
                                  num_u, phys0, stream);
        lattice_ptr = state_->lattice_xyz.template as<TGeom>();
        if (!packed) {
          detail::gather_lattice(*state_, lattice_ptr, num_v * num_u, stream);
          state_->planes_valid = true;
        }
      }
      state_->ring_offset = (phys0 + k) % num_v;
      // Deliberately *not* re-arming `locality_warned_` here. A scroll
      // advances the lattice by whole rows and does not change the locality
      // property the warning reports, so clearing it every tick turned a
      // warn-once into a warn-forever -- and, worse, re-ran the check itself,
      // which costs two blocking device-to-host synchronizations per tick on
      // the default Taylor backend. `continuous_strip` scrolls every tick.
    }

    const T* profile_ptr = static_cast<const T*>(profiles->data());
    if (weight) {
      require_dtype(*weight, real_dtype<T>(), "weight", who);
      require_contiguous(*weight, "weight", who);
      require_cuda_accessible(*weight, "weight", who);
      const auto wshape = weight->shape();
      if (wshape.empty() || wshape.back() != num_pulses) {
        throw std::invalid_argument(who + ": weight last axis must equal num_pulses");
      }
      int64_t wbatch = 1;
      for (size_t d = 0; d + 1 < wshape.size(); ++d) {
        wbatch *= wshape[d];
      }
      if (wbatch != cube.batch) {
        throw std::invalid_argument(who + ": weight leading axes must product to the cube batch");
      }
      const int64_t nweight = wbatch * num_pulses;
      std::vector<T> host_w(static_cast<size_t>(nweight));
      if (cudaMemcpyAsync(host_w.data(), weight->data(), sizeof(T) * static_cast<size_t>(nweight),
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
        state_->weight_scratch.ensure(static_cast<size_t>(n) * 2 * sizeof(T), stream);
        const int threads = 256;
        const int blocks = static_cast<int>((n_batch + threads - 1) / threads);
        for (int64_t b = 0; b < cube.batch; ++b) {
          const T* src = static_cast<const T*>(profiles->data()) + b * n_batch * 2;
          const T* wptr = static_cast<const T*>(weight->data()) + b * num_pulses;
          T* dst = state_->weight_scratch.template as<T>() + b * n_batch * 2;
          detail::scale_interleaved_kernel<T><<<blocks, threads, 0, stream>>>(
              src, wptr, dst, static_cast<int>(num_pulses), static_cast<int>(num_bins));
        }
        profile_ptr = state_->weight_scratch.template as<T>();
      }
    }

    state_->record_tick_gpu_start(stream);
    const double pc_partial = 4.0 * kPi * effective_frequency_ / kSpeedOfLight;
    const int64_t pulse_stride = num_pulses * 3;
    const int64_t profile_stride = num_pulses * num_bins * 2;
    for (int64_t b = 0; b < cube.batch; ++b) {
      const TGeom* pos_b = static_cast<const TGeom*>(positions->data()) + b * pulse_stride;
      const TGeom* ref_b = static_cast<const TGeom*>(reference->data()) + b * pulse_stride;
      const T* prof_b = profile_ptr + b * profile_stride;
      detail::prepare_pulses(*state_, pos_b, ref_b, static_cast<int>(num_pulses), stream);
      if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
        detail::launch_matx_window<T, TGeom>(
            backend_value_, *state_, prof_b, pos_b, lattice_ptr, static_cast<int>(num_v),
            static_cast<int>(num_u), static_cast<int>(num_pulses), static_cast<int>(num_bins),
            static_cast<int>(b), effective_frequency_, spacing, stream);
#else
        throw std::invalid_argument(who + ": backend '" + std::string(to_string(backend_value_)) +
                                    "' needs matx/kernels/sar_bp.cuh");
#endif
      } else {
        detail::launch_nvradar_window<T>(backend_value_, *state_, prof_b, static_cast<int>(num_v),
                                         static_cast<int>(num_u), static_cast<int>(num_pulses),
                                         static_cast<int>(num_bins), static_cast<int>(b), origin,
                                         spacing, pc_partial, stream);
      }
    }

    if (!locality_warned_ && is_taylor_backend(backend_value_)) {
      std::vector<double> r0_host(static_cast<size_t>(num_pulses));
      if (cudaMemcpyAsync(r0_host.data(), state_->r0.template as<double>(),
                          sizeof(double) * static_cast<size_t>(num_pulses), cudaMemcpyDeviceToHost,
                          stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess) {
        double r0_min = r0_host[0];
        for (double r : r0_host) {
          r0_min = std::min(r0_min, r);
        }
        double D = 0.0;
        if (sample_points_connected_) {
          if (!state_->planes_valid && lattice_ptr != nullptr) {
            detail::gather_lattice(*state_, lattice_ptr, num_v * num_u, stream);
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
        maybe_warn_locality(backend_value_, D, r0_min, effective_frequency_, &locality_warned_,
                            who);
      }
    }

    unemitted_pulses_ += num_pulses * cube.batch;

    bool emitted = false;
    int64_t emitted_pulses = 0;
    int64_t emitted_pixels = 0;
    std::optional<SampleGeometry> emitted_geometry;
    if (control_value.has_value() && is_emit(*control_value)) {
      // Keep the lattice of this image before EmitAndReset installs the next one.
      emitted_geometry.emplace(effective_geometry_);
      emitted = true;
      emitted_pulses = unemitted_pulses_;
      emitted_pixels = num_v * num_u;
      unemitted_pulses_ = 0;
    }

    if (apply_after) {
      install_pending_geometry();
    }
    if (stage_after) {
      if (pending_spacing.has_value()) {
        effective_spacing_ = *pending_spacing;
      }
      if (pending_origin.has_value()) {
        effective_origin_ = *pending_origin;
      }
      if (pending_frequency.has_value()) {
        effective_frequency_ = *pending_frequency;
      }
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

    state_->record_tick_gpu_stop(stream);

    // Synchronize only at an image boundary, when downstream reads the result.
    if (emitted) {
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error(who + ": emit stream sync failed");
      }
      aperture_compute_s_ = state_->take_tick_gpu_s();
    }
    const auto now = std::chrono::steady_clock::now();
    if (emitted) {
      // Timing is published before the image port so EventBased
      // downstream host code (the overlay) reads this emit, not the
      // previous collection's.
      last_image_compute_s_.store(aperture_compute_s_, std::memory_order_relaxed);
      last_image_pulses_.store(emitted_pulses, std::memory_order_relaxed);
      last_image_pixels_.store(emitted_pixels, std::memory_order_relaxed);
      last_image_num_v_.store(num_v, std::memory_order_relaxed);
      last_image_num_u_.store(num_u, std::memory_order_relaxed);
      const auto ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
      last_image_emit_ns_.store(ns, std::memory_order_relaxed);
      last_image_emit_valid_.store(true, std::memory_order_release);
      if (hud_) {
        hud_->publish_bp(FormationHudBp{
            .capture_time_ns = context.activation_time().timestamp_ns,
            .closes_aperture = completes_aperture,
            .compute_s = aperture_compute_s_,
            .pulses = emitted_pulses,
            .pixels = emitted_pixels,
            .num_v = num_v,
            .num_u = num_u,
            .emit_at = now,
            .last_packet_at = tick_last_packet,
            .playback = tick_playback,
        });
      }
      // Copy after GPU work is synced, and reset only after a successful
      // image emit: a failed Tensor emit consumes the loan.
      HOLOSCAN_RETURN_IF_ERROR(
          emit_required_tensor(image, context, "BackprojectionOp.image", false, [&] {
            return prepare_device_tensor(
                image, context, std::vector<int64_t>{num_v, num_u}, complex_dtype<T>(),
                [&](void* dst, std::size_t bytes) -> expected<void, Error> {
                  if (bytes != static_cast<std::size_t>(num_v) * static_cast<std::size_t>(num_u) *
                                   2U * sizeof(T)) {
                    return make_unexpected(
                        Error{ErrorCode::kFailure, who + ": image size mismatch"});
                  }
                  if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
                    detail::emit_matx_accum(*state_, static_cast<T*>(dst), 1, num_v, num_u, stream);
#endif
                  } else {
                    detail::emit_accum(*state_, static_cast<T*>(dst), 1, num_v, num_u, stream);
                  }
                  if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    return make_unexpected(
                        Error{ErrorCode::kFailure, who + ": image copy sync failed"});
                  }
                  return {};
                });
          }));
      if (control_value.has_value() && *control_value == FormationControl::kEmitAndReset) {
        if (detail::backend_is_matx(backend_value_)) {
#if __has_include("matx/kernels/sar_bp.cuh")
          detail::reset_matx_accum(*state_, stream);
#endif
        } else {
          detail::reset_accum(*state_, stream);
        }
      }
      if (!sample_points_connected_ && emitted_geometry->is_valid()) {
        if (geometry_output_.active()) {
          const auto packed = pack_sample_geometry(*emitted_geometry);
          auto loan = sample_geometry_out.allocate_tensor(TensorLoanRequest{
              .shape = std::vector<int64_t>{11}, .dtype = DLDataType{kDLFloat, 64U, 1U}});
          if (!loan)
            return make_unexpected(std::move(loan).error());
          auto written = loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
            std::copy(packed.begin(), packed.end(), writer.data_as<double>());
          });
          if (!written)
            return make_unexpected(std::move(written).error());
          geometry_output_.emit_schema_tensor(sample_geometry_out, context, std::move(*loan),
                                              to_wire(*emitted_geometry),
                                              "BackprojectionOp.sample_geometry_out");
        }
      }
      const double rate = aperture_compute_s_ > 0.0
                              ? static_cast<double>(emitted_pixels) *
                                    static_cast<double>(emitted_pulses) / aperture_compute_s_
                              : 0.0;
      HOLOSCAN_LOG_DEBUG("{}: {}x{} in {:.3f} s ({:.3e} px*pls/s, {} pulses)", who, num_v, num_u,
                         aperture_compute_s_, rate, emitted_pulses);
      aperture_compute_s_ = 0.0;
    }
    return {};
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  } catch (const std::exception& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }
}

template <typename T, typename TGeom>
void BackprojectionOp<T, TGeom>::stop() {
  if (unemitted_pulses_ > 0) {
    HOLOSCAN_LOG_WARN(
        "{}: stopping with {} unemitted pulse(s) still in the "
        "accumulator",
        op_name<T, TGeom>(), unemitted_pulses_);
  }
}

template class BackprojectionOp<float, float>;
template class BackprojectionOp<double, double>;
template class BackprojectionOp<float, double>;

}  // namespace holoscan::radar
