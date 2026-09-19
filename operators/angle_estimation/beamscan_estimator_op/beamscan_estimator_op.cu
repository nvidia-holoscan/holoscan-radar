// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_stage.hpp"
#include "angle_estimation/is_valid_mask.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/sdk/config_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

struct BeamscanSteeringArtifact {
  std::shared_ptr<Tensor> weights;
  std::shared_ptr<Tensor> az_grid;
  std::shared_ptr<Tensor> el_grid;
};

struct BeamscanRuntimeState {
  ConfigCache<std::shared_ptr<BeamscanSteeringArtifact>> steering_cache;
  std::shared_ptr<Tensor> beamformed;
};

namespace {

constexpr const char* kOpName = "BeamscanEstimatorOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
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

struct BeamscanStageConfig {
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_coordinates;
  std::vector<double> el_coordinates;
  WindowSpec taper;
  PeakInterpolation peak_interpolation{PeakInterpolation::kNone};
};

struct BeamscanStagePlan {
  BeamscanStageConfig stage;
  int64_t num_items{0};
  int64_t num_channels{0};
  int64_t num_snapshots{1};
  int64_t num_beams{0};
  std::vector<int64_t> aoa_shape;
  std::vector<int64_t> spectrum_shape;
};

expected<BeamscanStageConfig, Error> resolve_params(
    Input<WireArrayGeometry>& geometry_port, Input<Tensor>& az_port,
    Input<Tensor>& el_port, Input<WireWindowSpec>& taper_port,
    Input<PeakInterpolation>& interpolation_port,
    const BeamscanParams& pinned, int64_t max_az, int64_t max_el) {
  HOLOSCAN_TRY(
      auto geometry,
      resolve_live_override_via(
          geometry_port, pinned.array_geometry,
          [](const WireArrayGeometry& value) { return from_wire(value); }));
  HOLOSCAN_TRY(
      auto az,
      resolve_live_override_via(
          az_port, pinned.az_coordinates,
          [max_az](const Tensor& value) {
            return read_host_config_vector<double>(
                value, DLDataType{kDLFloat, 64U, 1U}, max_az, "az_coordinates");
          }));
  HOLOSCAN_TRY(
      auto el,
      resolve_live_override_via(
          el_port, pinned.el_coordinates,
          [max_el](const Tensor& value) {
            return read_host_config_vector<double>(
                value, DLDataType{kDLFloat, 64U, 1U}, max_el, "el_coordinates");
          }));
  HOLOSCAN_TRY(
      auto taper,
      resolve_live_override_via(
          taper_port, pinned.array_taper,
          [](const WireWindowSpec& value) { return from_wire(value); }));
  HOLOSCAN_TRY(
      auto interpolation,
      resolve_live_override(interpolation_port, pinned.peak_interpolation));
  return BeamscanStageConfig{
      .geometry = std::move(geometry),
      .convention = pinned.angle_convention,
      .az_coordinates = std::move(az),
      .el_coordinates = std::move(el),
      .taper = std::move(taper),
      .peak_interpolation = interpolation,
  };
}

expected<BeamscanStagePlan, Error> build_beamscan_stage_plan(
    const Tensor& snapshot, BeamscanStageConfig stage, int64_t max_items,
    int64_t max_az, int64_t max_el) {
  const auto& shape = snapshot.shape();
  if (shape.size() != 2U && shape.size() != 3U) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": snapshot must have shape (num_items, num_channels) or "
            "(num_items, num_channels, num_snapshots)"});
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape.size() == 3U ? shape[2] : 1;
  if (num_items < 0 || num_channels <= 0) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": snapshot num_items must be non-negative and num_channels "
            "must be positive"});
  }
  if (num_snapshots <= 0) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": snapshot num_snapshots must be positive"});
  }
  try {
    stage.geometry.validate();
    if (stage.az_coordinates.empty() || stage.el_coordinates.empty()) {
      throw std::invalid_argument(
          "BeamscanEstimatorOp: az_coordinates and el_coordinates must be non-empty");
    }
    validate_scan_grid_axes(stage.convention, stage.az_coordinates, stage.el_coordinates,
                            "BeamscanEstimatorOp");
    if (num_items > max_items ||
        stage.az_coordinates.size() > static_cast<std::size_t>(max_az) ||
        stage.el_coordinates.size() > static_cast<std::size_t>(max_el)) {
      throw std::invalid_argument(
          "BeamscanEstimatorOp: input or scan grid exceeds MaxTensorExtents");
    }
    generate_steering_taper(stage.geometry, stage.taper);
    (void)boundary_to_canonical(
        stage.convention, stage.az_coordinates.front(), stage.el_coordinates.front());
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  }
  if (num_channels != stage.geometry.num_elements()) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": snapshot num_channels must match "
            "array_geometry.num_elements"});
  }

  std::size_t num_beams = 0U;
  try {
    num_beams = product_extents(
        {static_cast<int64_t>(stage.az_coordinates.size()),
         static_cast<int64_t>(stage.el_coordinates.size())},
        kOpName);
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  }
  if (num_beams >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": scan-grid size overflows int64"});
  }

  const int64_t num_az = static_cast<int64_t>(stage.az_coordinates.size());
  const int64_t num_el = static_cast<int64_t>(stage.el_coordinates.size());
  return BeamscanStagePlan{
      .stage = std::move(stage),
      .num_items = num_items,
      .num_channels = num_channels,
      .num_snapshots = num_snapshots,
      .num_beams = static_cast<int64_t>(num_beams),
      .aoa_shape = {num_items, 2},
      .spectrum_shape = {num_items, num_az, num_el},
  };
}

template <typename T>
expected<std::shared_ptr<BeamscanSteeringArtifact>, Error>
prepare_steering_artifact(BeamscanRuntimeState& state,
                          const BeamscanStageConfig& stage,
                          cudaStream_t stream) {
  try {
    const std::string key =
        steering_cache_key(stage.geometry, stage.convention, stage.az_coordinates,
                           stage.el_coordinates, stage.taper);
    return state.steering_cache.get_or_build(
        key, "steering weights and scan grids", [&] {
          auto artifact = std::make_shared<BeamscanSteeringArtifact>();
          artifact->weights = build_steering_weights_device<T>(
              stage.geometry, stage.convention, stage.az_coordinates,
              stage.el_coordinates, stage.taper, BeamOrder::kElMajor,
              SteeringLayout::kChannelMajor, /*conjugate=*/true, stream);
          const auto az_host = cast_vector<T>(stage.az_coordinates);
          const auto el_host = cast_vector<T>(stage.el_coordinates);
          artifact->az_grid =
              make_device_tensor({static_cast<int64_t>(az_host.size())},
                                 real_dtype<T>());
          artifact->el_grid =
              make_device_tensor({static_cast<int64_t>(el_host.size())},
                                 real_dtype<T>());
          auto az_ptr = writable_ptr(*artifact->az_grid, "az_grid");
          auto el_ptr = writable_ptr(*artifact->el_grid, "el_grid");
          if (!az_ptr || !el_ptr) {
            throw std::runtime_error(
                "BeamscanEstimatorOp: scan-grid tensor is not writable");
          }
          if (cudaMemcpyAsync(*az_ptr, az_host.data(),
                              az_host.size() * sizeof(T),
                              cudaMemcpyHostToDevice, stream) != cudaSuccess ||
              cudaMemcpyAsync(*el_ptr, el_host.data(),
                              el_host.size() * sizeof(T),
                              cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::runtime_error(
                "BeamscanEstimatorOp: scan-grid H2D failed");
          }
          return artifact;
        });
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  } catch (const std::runtime_error& error) {
    return make_unexpected(Error{ErrorCode::kFailure, error.what()});
  }
}

}  // namespace

template <typename T>
BeamscanEstimatorOp<T>::BeamscanEstimatorOp(
    const MaxTensorExtents& extents) {
  const std::int64_t items =
      require_extent(extents.max_items, "max_items", kOpName);
  const std::int64_t az =
      require_extent(extents.max_az, "max_az", kOpName);
  const std::int64_t el =
      require_extent(extents.max_el, "max_el", kOpName);
  max_items_ = items;
  max_az_ = az;
  max_el_ = el;
  spectrum_max_byte_span_ =
      byte_span_real_cube<T>({items, az, el}, kOpName);
  aoa_max_byte_span_ =
      byte_span_real_cube<T>({items, 2}, kOpName);
  state_ = std::make_shared<BeamscanRuntimeState>();
  state_->beamformed =
      make_device_tensor({items, az, el}, complex_dtype<T>());
}

template <typename T>
void BeamscanEstimatorOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kHostF64{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 64U, 1U},
      .rank = 1U,
  };
  spec.input(snapshot).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(is_valid)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kDevice));
  spec.output(aoa_estimates)
      .produces_tensor(pooled_tensor_output(kDevice, 2U, aoa_max_byte_span_));
  spec.output(angle_spectrum)
      .produces_tensor(pooled_tensor_output(
          kDevice, 3U, spectrum_max_byte_span_));
  // [live] Exact overrides, Params order (schema and host-Tensor alike).
  spec.input(array_geometry).optional().queue_depth(2U);
  spec.input(az_coordinates)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostF64));
  spec.input(el_coordinates)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostF64));
  spec.input(array_taper).optional().queue_depth(2U);
  spec.input(peak_interpolation).optional().queue_depth(2U);
}

template <typename T>
Contract BeamscanEstimatorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{snapshot})
      .input(is_valid, Exact{})
      .on_missing(is_valid, SafeDefault{});
  add_live_override_to_contract(result, array_geometry);
  add_live_override_to_contract(result, az_coordinates);
  add_live_override_to_contract(result, el_coordinates);
  add_live_override_to_contract(result, array_taper);
  add_live_override_to_contract(result, peak_interpolation);
  return result;
}

template <typename T>
expected<void, Error> BeamscanEstimatorOp<T>::compute(
    ExecutionContext& context) {
  // Receive and validate the trigger tensor.
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));
  try {
    require_cuda_accessible(snap, "snapshot", kOpName);
    require_contiguous(snap, "snapshot", kOpName);
    require_complex<T>(snap, "snapshot");
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  // Resolve this tick's Exact overrides, then validate named axes and derive
  // output geometry.
  const BeamscanParams& pinned = context.params<BeamscanParams>();
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(array_geometry, az_coordinates, el_coordinates, array_taper,
                     peak_interpolation, pinned, max_az_, max_el_));
  HOLOSCAN_TRY(
      auto plan,
      build_beamscan_stage_plan(
          snap, std::move(stage), max_items_, max_az_, max_el_));

  // Validate the optional same-tick mask against the resolved snapshot axes.
  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  const unsigned char* mask_ptr = nullptr;
  if (maybe_valid.has_value()) {
    try {
      detail::check_is_valid_mask(*maybe_valid, plan.num_items,
                                  plan.num_snapshots, kOpName);
    } catch (const std::invalid_argument& error) {
      return fail_invalid(error.what());
    }
    mask_ptr = static_cast<const unsigned char*>(maybe_valid->data());
  }

  // Resolve the runtime stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "BeamscanEstimatorOp: no CUDA compute stream"});
  }

  const DLDataType out_dtype = real_dtype<T>();
  auto retain_inputs = [&] {
    retained_snapshot_ = std::move(snap);
    if (maybe_valid.has_value()) {
      retained_is_valid_ = std::move(*maybe_valid);
    }
  };

  // Empty items still publish both contract shapes without building steering
  // artifacts or launching the stage.
  if (plan.num_items == 0) {
    auto noop = [](void*, std::size_t) -> expected<void, Error> { return {}; };
    HOLOSCAN_TRY(
        auto aoa_loan,
        prepare_device_tensor(
            aoa_estimates, context, plan.aoa_shape, out_dtype, noop));
    HOLOSCAN_TRY(
        auto spectrum_loan,
        prepare_device_tensor(
            angle_spectrum, context, plan.spectrum_shape, out_dtype, noop));
    HOLOSCAN_TRY(
        emit_prepared_tensor(aoa_estimates, context, std::move(aoa_loan)));
    // Optional/always: the spectrum is the required AOA pass's intermediate.
    (void)emit_prepared_tensor(
        angle_spectrum, context, std::move(spectrum_loan));
    retain_inputs();
    return {};
  }

  // Resolve cached steering weights and scan grids. An unseen effective
  // configuration builds all three artifacts once.
  HOLOSCAN_TRY(
      auto steering,
      prepare_steering_artifact<T>(*state_, plan.stage, stream));
  if (!steering || !steering->weights || !steering->az_grid ||
      !steering->el_grid) {
    return fail_runtime(
        "BeamscanEstimatorOp: internal steering artifact is incomplete");
  }

  // Prepare both loans before the shared stage writes either output.
  auto aoa_loan = aoa_estimates.allocate_tensor(
      TensorLoanRequest{.shape = plan.aoa_shape, .dtype = out_dtype});
  if (!aoa_loan) {
    return make_unexpected(std::move(aoa_loan).error());
  }
  auto spectrum_loan = angle_spectrum.allocate_tensor(
      TensorLoanRequest{.shape = plan.spectrum_shape, .dtype = out_dtype});
  if (!spectrum_loan) {
    return make_unexpected(std::move(spectrum_loan).error());
  }
  auto aoa_writer = aoa_loan->write(context.cuda_stream());
  if (!aoa_writer) {
    return make_unexpected(std::move(aoa_writer).error());
  }
  auto spectrum_writer = spectrum_loan->write(context.cuda_stream());
  if (!spectrum_writer) {
    return make_unexpected(std::move(spectrum_writer).error());
  }

  HOLOSCAN_TRY(
      auto beamformed_ptr,
      writable_ptr(*state_->beamformed, "beamformed workspace"));

  try {
    if (plan.num_snapshots == 1) {
      detail::run_beamscan_stage<T>(
          static_cast<const cuda::std::complex<T>*>(snap.data()),
          plan.num_items, plan.num_channels,
          static_cast<const cuda::std::complex<T>*>(steering->weights->data()),
          static_cast<cuda::std::complex<T>*>(beamformed_ptr),
          plan.spectrum_shape[1], plan.spectrum_shape[2],
          static_cast<const T*>(steering->az_grid->data()),
          static_cast<const T*>(steering->el_grid->data()),
          plan.stage.peak_interpolation,
          static_cast<T*>(spectrum_writer->data()),
          static_cast<T*>(aoa_writer->data()), stream);
      detail::blank_empty_beamscan_items(
          mask_ptr, plan.num_items, plan.num_snapshots, plan.num_beams,
          static_cast<T*>(spectrum_writer->data()),
          static_cast<T*>(aoa_writer->data()), stream);
    } else {
      detail::run_beamscan_multilook_stage<T>(
          static_cast<const cuda::std::complex<T>*>(snap.data()), mask_ptr,
          plan.num_items, plan.num_channels, plan.num_snapshots,
          static_cast<const cuda::std::complex<T>*>(steering->weights->data()),
          plan.spectrum_shape[1], plan.spectrum_shape[2],
          static_cast<const T*>(steering->az_grid->data()),
          static_cast<const T*>(steering->el_grid->data()),
          plan.stage.peak_interpolation,
          static_cast<T*>(spectrum_writer->data()),
          static_cast<T*>(aoa_writer->data()), stream);
    }
    check_cuda("beamscan stage", kOpName);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  } catch (const std::runtime_error& error) {
    return fail_runtime(error.what());
  }

  auto aoa_committed = std::move(*aoa_writer).commit();
  if (!aoa_committed) {
    return make_unexpected(std::move(aoa_committed).error());
  }
  auto spectrum_committed = std::move(*spectrum_writer).commit();
  if (!spectrum_committed) {
    return make_unexpected(std::move(spectrum_committed).error());
  }

  // Publish the required output first; declining the optional spectrum cannot
  // suppress the AOA result.
  HOLOSCAN_TRY(
      emit_prepared_tensor(aoa_estimates, context, std::move(*aoa_loan)));
  (void)emit_prepared_tensor(
      angle_spectrum, context, std::move(*spectrum_loan));
  retain_inputs();
  return {};
}

template class BeamscanEstimatorOp<float>;
template class BeamscanEstimatorOp<double>;

}  // namespace holoscan::radar
