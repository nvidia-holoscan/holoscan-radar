// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_stage.hpp"
#include "radar_common/dsp/angle_convention.hpp"
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

namespace {

constexpr const char* kOpName = "PhaseShiftBeamformerOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

bool same_double_bytes(double a, double b) {
  return std::bit_cast<uint64_t>(a) == std::bit_cast<uint64_t>(b);
}

bool same_double_vector_bytes(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

bool same_geometry_bytes(const ArrayGeometry& a, const ArrayGeometry& b) {
  return a.geometry_type == b.geometry_type && same_double_bytes(a.wavelength, b.wavelength) &&
         same_double_vector_bytes(a.element_positions, b.element_positions);
}

bool same_window_bytes(const WindowSpec& a, const WindowSpec& b) {
  return a.type == b.type &&
         same_double_bytes(a.sidelobe_attenuation_db, b.sidelobe_attenuation_db) &&
         a.kaiser_beta.has_value() == b.kaiser_beta.has_value() &&
         (!a.kaiser_beta.has_value() || same_double_bytes(*a.kaiser_beta, *b.kaiser_beta)) &&
         a.taylor_nbar == b.taylor_nbar && same_double_bytes(a.tukey_alpha, b.tukey_alpha) &&
         same_double_vector_bytes(a.coefficients, b.coefficients) &&
         a.coherent_gain_normalization == b.coherent_gain_normalization;
}

struct PhaseShiftBeamformerStageConfig {
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_coordinates;
  std::vector<double> el_coordinates;
  WindowSpec taper;
};

struct PhaseShiftBeamformerStagePlan {
  PhaseShiftBeamformerStageConfig stage;
  int64_t num_channels{0};
  int64_t num_cells{0};
  int64_t num_beams{0};
  std::vector<int64_t> output_shape;
};

expected<PhaseShiftBeamformerStageConfig, Error> resolve_params(
    Input<WireArrayGeometry>& geometry_port, Input<Tensor>& az_port, Input<Tensor>& el_port,
    Input<WireWindowSpec>& taper_port, const PhaseShiftBeamformerParams& pinned, int64_t max_az,
    int64_t max_el) {
  HOLOSCAN_TRY(auto geometry, resolve_live_override_via(geometry_port, pinned.array_geometry,
                                                        [](const WireArrayGeometry& value) {
                                                          return from_wire(value);
                                                        }));
  HOLOSCAN_TRY(auto az, resolve_live_override_via(
                            az_port, pinned.az_coordinates, [max_az](const Tensor& value) {
                              return read_host_config_vector<double>(
                                  value, DLDataType{kDLFloat, 64U, 1U}, max_az, "az_coordinates");
                            }));
  HOLOSCAN_TRY(auto el, resolve_live_override_via(
                            el_port, pinned.el_coordinates, [max_el](const Tensor& value) {
                              return read_host_config_vector<double>(
                                  value, DLDataType{kDLFloat, 64U, 1U}, max_el, "el_coordinates");
                            }));
  HOLOSCAN_TRY(auto taper, resolve_live_override_via(taper_port, pinned.array_taper,
                                                     [](const WireWindowSpec& value) {
                                                       return from_wire(value);
                                                     }));
  return PhaseShiftBeamformerStageConfig{
      .geometry = std::move(geometry),
      .convention = pinned.angle_convention,
      .az_coordinates = std::move(az),
      .el_coordinates = std::move(el),
      .taper = std::move(taper),
  };
}

expected<PhaseShiftBeamformerStagePlan, Error> build_phase_shift_beamformer_stage_plan(
    const Tensor& rd, PhaseShiftBeamformerStageConfig stage, int64_t max_az, int64_t max_el,
    int64_t max_doppler, int64_t max_range) {
  const auto& shape = rd.shape();
  if (shape.size() != 3U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(kOpName) +
                                               ": rd_maps must have shape "
                                               "(num_channels, num_doppler, num_range_bins)"});
  }
  const int64_t num_channels = shape[0];
  const int64_t num_doppler = shape[1];
  const int64_t num_range = shape[2];
  if (num_channels <= 0 || num_doppler <= 0 || num_range <= 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": rd_maps axes must be positive"});
  }

  try {
    stage.geometry.validate();
    if (stage.az_coordinates.empty() || stage.el_coordinates.empty()) {
      throw std::invalid_argument(
          "PhaseShiftBeamformerOp: az_coordinates and el_coordinates must be non-empty");
    }
    validate_scan_grid_axes(stage.convention, stage.az_coordinates, stage.el_coordinates,
                            "PhaseShiftBeamformerOp");
    if (stage.az_coordinates.size() > static_cast<std::size_t>(max_az) ||
        stage.el_coordinates.size() > static_cast<std::size_t>(max_el) ||
        num_doppler > max_doppler || num_range > max_range) {
      throw std::invalid_argument(
          "PhaseShiftBeamformerOp: input or scan grid exceeds "
          "MaxTensorExtents");
    }
    generate_steering_taper(stage.geometry, stage.taper);
    // Also validates the signed-axis frame for a custom convention.
    (void)boundary_to_canonical(stage.convention, stage.az_coordinates.front(),
                                stage.el_coordinates.front());
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }
  if (num_channels != stage.geometry.num_elements()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": rd_maps num_channels must match "
                                                        "array_geometry.num_elements"});
  }

  const int64_t num_az = static_cast<int64_t>(stage.az_coordinates.size());
  const int64_t num_el = static_cast<int64_t>(stage.el_coordinates.size());
  std::size_t num_cells_size = 0U;
  std::size_t num_beams_size = 0U;
  try {
    num_cells_size = product_extents({num_doppler, num_range}, kOpName);
    num_beams_size = product_extents({num_az, num_el}, kOpName);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }
  constexpr std::size_t kInt64Max = static_cast<std::size_t>(std::numeric_limits<int64_t>::max());
  if (num_cells_size > kInt64Max || num_beams_size > kInt64Max) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": stage dimensions overflow int64"});
  }

  return PhaseShiftBeamformerStagePlan{
      .stage = std::move(stage),
      .num_channels = num_channels,
      .num_cells = static_cast<int64_t>(num_cells_size),
      .num_beams = static_cast<int64_t>(num_beams_size),
      .output_shape = {num_az, num_el, num_doppler, num_range},
  };
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_grid(const std::vector<double>& values,
                                                   cudaStream_t stream) {
  auto result = make_device_tensor({static_cast<int64_t>(values.size())}, real_dtype<T>());
  std::vector<T> cast(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    cast[i] = static_cast<T>(values[i]);
  }
  auto writable = result->mutable_data();
  if (!writable) {
    throw std::runtime_error("PhaseShiftBeamformerOp: scan-coordinate grid is not writable");
  }
  if (!cast.empty() && cudaMemcpyAsync(*writable, cast.data(), cast.size() * sizeof(T),
                                       cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("PhaseShiftBeamformerOp: angle-grid H2D copy failed");
  }
  return result;
}

template <typename T>
expected<const detail::PhaseShiftBeamformerArtifacts*, Error> prepare_steering_artifacts(
    ConfigCache<detail::PhaseShiftBeamformerArtifacts>& cache,
    const PhaseShiftBeamformerStageConfig& stage, cudaStream_t stream) {
  try {
    const std::string cache_key = steering_cache_key(
        stage.geometry, stage.convention, stage.az_coordinates, stage.el_coordinates, stage.taper);
    const detail::PhaseShiftBeamformerArtifacts& artifacts =
        cache.get_or_build(cache_key, "steering artifacts", [&] {
          return detail::PhaseShiftBeamformerArtifacts{
              .weights = build_steering_weights_device<T>(
                  stage.geometry, stage.convention, stage.az_coordinates, stage.el_coordinates,
                  stage.taper, BeamOrder::kAzMajor, SteeringLayout::kBeamMajor, /*conjugate=*/true,
                  stream),
              .az_grid = make_device_grid<T>(stage.az_coordinates, stream),
              .el_grid = make_device_grid<T>(stage.el_coordinates, stream)};
        });
    if (!artifacts.weights || !artifacts.az_grid || !artifacts.el_grid) {
      return make_unexpected(
          Error{ErrorCode::kFailure,
                std::string(kOpName) + ": internal error: steering artifacts missing"});
    }
    return &artifacts;
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  } catch (const std::runtime_error& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }
}

}  // namespace

template <typename T>
PhaseShiftBeamformerOp<T>::PhaseShiftBeamformerOp(const MaxTensorExtents& extents) {
  const std::int64_t max_az = require_extent(extents.max_az, "max_az", kOpName);
  const std::int64_t max_el = require_extent(extents.max_el, "max_el", kOpName);
  const std::int64_t max_doppler =
      require_extent(extents.max_doppler_bins, "max_doppler_bins", kOpName);
  const std::int64_t max_range = require_extent(extents.max_range_bins, "max_range_bins", kOpName);
  max_az_ = max_az;
  max_el_ = max_el;
  max_doppler_ = max_doppler;
  max_range_ = max_range;
  // rda_cube is (num_az, num_el, D, R) — beamforming collapses channels.
  max_byte_span_ = byte_span_complex_cube<T>({max_az, max_el, max_doppler, max_range}, kOpName);
  max_az_grid_byte_span_ = byte_span_real_cube<T>({max_az}, kOpName);
  max_el_grid_byte_span_ = byte_span_real_cube<T>({max_el}, kOpName);
}

template <typename T>
void PhaseShiftBeamformerOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kOutputDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = 4U,
  };
  constexpr TensorRepresentation kGridDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = real_dtype<T>(),
      .rank = 1U,
  };
  constexpr TensorRepresentation kHostF64{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 64U, 1U},
      .rank = 1U,
  };
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(rda_cube).produces_tensor(pooled_tensor_output(kOutputDevice, 4U, max_byte_span_));
  spec.output(az_grid).produces_tensor(
      pooled_tensor_output(kGridDevice, 2U, max_az_grid_byte_span_));
  spec.output(el_grid).produces_tensor(
      pooled_tensor_output(kGridDevice, 2U, max_el_grid_byte_span_));
  // [live] Exact overrides, Params order (schema and host-Tensor alike).
  spec.input(array_geometry).optional().queue_depth(2U);
  spec.input(az_coordinates).optional().queue_depth(2U).expects_tensor(tensor_input(kHostF64));
  spec.input(el_coordinates).optional().queue_depth(2U).expects_tensor(tensor_input(kHostF64));
  spec.input(array_taper).optional().queue_depth(2U);
}

template <typename T>
Contract PhaseShiftBeamformerOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  add_live_override_to_contract(result, array_geometry);
  add_live_override_to_contract(result, az_coordinates);
  add_live_override_to_contract(result, el_coordinates);
  add_live_override_to_contract(result, array_taper);
  return result;
}

template <typename T>
expected<void, Error> PhaseShiftBeamformerOp<T>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  // Receive and validate the trigger tensor.
  HOLOSCAN_TRY(auto rd, receive_tensor(rd_maps, kOpName));
  try {
    require_cuda_accessible(rd, "rd_maps", kOpName);
    require_contiguous(rd, "rd_maps", kOpName);
    require_complex<T>(rd, "rd_maps");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Resolve this tick's Exact overrides, then validate named axes and derive
  // output geometry.
  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const PhaseShiftBeamformerParams& pinned = context.params<PhaseShiftBeamformerParams>();
  HOLOSCAN_TRY(auto stage, resolve_params(array_geometry, az_coordinates, el_coordinates,
                                          array_taper, pinned, max_az_, max_el_));
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(auto plan, build_phase_shift_beamformer_stage_plan(
                              rd, std::move(stage), max_az_, max_el_, max_doppler_, max_range_));

  // Resolve stream and configuration-derived steering weights.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "PhaseShiftBeamformerOp: no CUDA compute stream"});
  }
  // Prepare or reuse cached artifacts without changing tick coherence.
  const bool last_steering_hit =
      last_steering_artifact_ != nullptr &&
      same_geometry_bytes(last_steering_geometry_, plan.stage.geometry) &&
      last_steering_convention_ == plan.stage.convention &&
      same_double_vector_bytes(last_steering_az_, plan.stage.az_coordinates) &&
      same_double_vector_bytes(last_steering_el_, plan.stage.el_coordinates) &&
      same_window_bytes(last_steering_taper_, plan.stage.taper);
  const detail::PhaseShiftBeamformerArtifacts* artifacts = last_steering_artifact_;
  if (!last_steering_hit) {
    HOLOSCAN_TRY(auto rebuilt, prepare_steering_artifacts<T>(artifact_cache_, plan.stage, stream));
    artifacts = rebuilt;
    last_steering_geometry_ = plan.stage.geometry;
    last_steering_convention_ = plan.stage.convention;
    last_steering_az_ = plan.stage.az_coordinates;
    last_steering_el_ = plan.stage.el_coordinates;
    last_steering_taper_ = plan.stage.taper;
    last_steering_artifact_ = artifacts;
  }

  // Prepare the required dense cube before publishing optional sidebands.
  const void* input_ptr = rd.data();
  const void* weights_ptr = artifacts->weights->data();
  // Allocate exact-sized outputs before launching device work.
  HOLOSCAN_TRY(auto cube_loan,
               prepare_device_tensor(rda_cube, context, plan.output_shape, complex_dtype<T>(),
                                     [&](void* out_ptr, std::size_t) -> expected<void, Error> {
                                       try {
                                         // Execute the device stage on the input runtime stream.
                                         detail::run_phase_shift_beamform_stage<T>(
                                             static_cast<const cuda::std::complex<T>*>(input_ptr),
                                             static_cast<cuda::std::complex<T>*>(out_ptr),
                                             /*batch=*/1, plan.num_channels, plan.num_cells,
                                             static_cast<const cuda::std::complex<T>*>(weights_ptr),
                                             plan.num_beams, stream);
                                         check_cuda("phase-shift beamform stage", kOpName);
                                       } catch (const std::invalid_argument& e) {
                                         return fail_invalid(e.what());
                                       } catch (const std::runtime_error& e) {
                                         return fail_runtime(e.what());
                                       }
                                       return {};
                                     }));

  // Publish effective grid sidebands before the cube that triggers downstream
  // work. The cached sources remain alive for all stream-ordered D2D copies.
  const auto emit_grid = [&](Output<Tensor>& output, LatchedOutput& latched,
                             const std::shared_ptr<Tensor>& source,
                             const char* port_name) -> expected<void, Error> {
    if (!latched.active()) {
      return {};
    }
    HOLOSCAN_TRY(auto loan, prepare_device_tensor(
                                output, context, source->shape(), real_dtype<T>(),
                                [&](void* dst, std::size_t bytes) -> expected<void, Error> {
                                  if (bytes != 0U && cudaMemcpyAsync(dst, source->data(), bytes,
                                                                     cudaMemcpyDeviceToDevice,
                                                                     stream) != cudaSuccess) {
                                    return fail_runtime(std::string(kOpName) + "." + port_name +
                                                        ": device copy failed");
                                  }
                                  return {};
                                }));
    // Publish coherent outputs in dependency order and retain borrowed inputs.
    latched.emit(output, context, std::move(loan), std::string(kOpName) + "." + port_name);
    return {};
  };
  HOLOSCAN_RETURN_IF_ERROR(emit_grid(az_grid, az_grid_out_, artifacts->az_grid, "az_grid"));
  HOLOSCAN_RETURN_IF_ERROR(emit_grid(el_grid, el_grid_out_, artifacts->el_grid, "el_grid"));
  HOLOSCAN_TRY(emit_prepared_tensor(rda_cube, context, std::move(cube_loan)));

  // Retain the borrowed input until the next activation.
  retained_input_ = std::move(rd);
  return {};
}

template class PhaseShiftBeamformerOp<float>;
template class PhaseShiftBeamformerOp<double>;

}  // namespace holoscan::radar
