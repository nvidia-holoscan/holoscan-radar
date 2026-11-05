// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/aperture_manager_op/aperture_manager_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_common/dsp/window.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {
namespace {
constexpr const char* kWho = "ApertureManagerOp";
constexpr TensorRepresentation kDevice{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = std::nullopt,
    .rank = std::nullopt};  // NOLINT(whitespace/indent_namespace)
constexpr TensorRepresentation kHostGeometry{.memory_kind = MemoryKind::kHost,
                                             .dtype = DLDataType{kDLFloat, 64U, 1U},
                                             .rank = 1U};  // NOLINT(whitespace/indent_namespace)

bool resets(FormationControl c) {
  return c == FormationControl::kEmitAndReset || c == FormationControl::kResetWithoutEmit ||
         c == FormationControl::kResetThenEmit;
}
bool pre_resets(FormationControl c) {
  return c == FormationControl::kResetWithoutEmit || c == FormationControl::kResetThenEmit;
}

template <typename Value>
expected<Value, Error> read_scalar(const Tensor& tensor, const char* name, cudaStream_t stream) {
  if (tensor.shape().size() != 1U || tensor.shape()[0] != 1 || tensor.nbytes() != sizeof(Value)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kWho) + ": " + name + " must have shape (1)"});
  }
  Value result{};
  if (cudaMemcpyAsync(&result, tensor.data(), sizeof(Value), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return make_unexpected(
        Error{ErrorCode::kFailure, std::string(kWho) + ": failed to read " + name});
  }
  return result;
}

expected<TensorOutputLoan, Error> prepare_copy(Output<Tensor>& output, ExecutionContext& context,
                                               const Tensor& source) {
  return prepare_device_tensor(
      output, context, source.shape(), source.dtype(),
      [&](void* dst, std::size_t bytes) -> expected<void, Error> {
        if (bytes != source.nbytes() ||
            cudaMemcpyAsync(dst, source.data(), bytes, cudaMemcpyDeviceToDevice,
                            context.cuda_stream().get()) != cudaSuccess) {
          return make_unexpected(
              Error{ErrorCode::kFailure, "ApertureManagerOp: device copy failed"});
        }
        return {};
      });
}

// Retained pulses must own their storage. Keeping received Tensor handles
// would hold reader and range-compression output-pool slots for the whole
// sliding aperture, eventually exhausting those finite pools.
PulseGroup copy_for_retention(const PulseGroup& source, cudaStream_t stream) {
  PulseGroup copy;
  copy.profiles = make_device_tensor(source.profiles->shape(), source.profiles->dtype());
  copy.positions = make_device_tensor(source.positions->shape(), source.positions->dtype());
  copy.reference = make_device_tensor(source.reference->shape(), source.reference->dtype());
  copy.count = source.count;
  copy_pulse_rows(*source.profiles, source.first, *copy.profiles, 0, source.count, stream, kWho);
  copy_pulse_rows(*source.positions, source.first, *copy.positions, 0, source.count, stream, kWho);
  copy_pulse_rows(*source.reference, source.first, *copy.reference, 0, source.count, stream, kWho);
  return copy;
}

template <typename T>
expected<TensorOutputLoan, Error> prepare_values(Output<Tensor>& output, ExecutionContext& context,
                                                 const std::vector<T>& values, DLDataType dtype) {
  return prepare_device_tensor(
      output, context, std::vector<int64_t>{static_cast<int64_t>(values.size())}, dtype,
      [&](void* dst, std::size_t bytes) -> expected<void, Error> {
        if (bytes != values.size() * sizeof(T) ||
            cudaMemcpyAsync(dst, values.data(), bytes, cudaMemcpyHostToDevice,
                            context.cuda_stream().get()) != cudaSuccess) {
          return make_unexpected(
              Error{ErrorCode::kFailure, "ApertureManagerOp: weight/control H2D failed"});
        }
        return {};
      });
}

expected<TensorOutputLoan, Error> prepare_geometry(Output<WireSampleGeometry>& output,
                                                   const SampleGeometry& geometry) {
  auto loan = output.allocate_tensor(
      TensorLoanRequest{.shape = std::vector<int64_t>{11}, .dtype = DLDataType{kDLFloat, 64U, 1U}});
  if (!loan)
    return make_unexpected(std::move(loan).error());
  const auto packed = pack_sample_geometry(geometry);
  auto written = loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
    std::copy(packed.begin(), packed.end(), writer.data_as<double>());
  });
  if (!written)
    return make_unexpected(std::move(written).error());
  return std::move(*loan);
}
}  // namespace

template <typename T, typename TGeom>
ApertureManagerOp<T, TGeom>::ApertureManagerOp(const MaxTensorExtents& extents)
: extents_(extents) {
  require_extent(extents.max_pulses, "max_pulses", kWho);
  require_extent(extents.max_range_bins, "max_range_bins", kWho);
}

template <typename T, typename TGeom>
void ApertureManagerOp<T, TGeom>::setup(OperatorSpec& spec) {
  const auto pulses = *extents_.max_pulses;
  const auto bins = *extents_.max_range_bins;
  // Setup default 16: continuous strip fans 4 slices per reader cube. Per-edge
  // ConnectionOptions still set destination depth (other apps keep 4).
  spec.input(range_profiles).queue_depth(16U).expects_tensor(tensor_input(kDevice));
  spec.input(positions).queue_depth(16U).expects_tensor(tensor_input(kDevice));
  spec.input(reference_position).queue_depth(16U).expects_tensor(tensor_input(kDevice));
  spec.input(stream_boundary).queue_depth(16U);
  spec.input(sample_geometry).optional().queue_depth(4U);
  spec.input(sample_points).optional().queue_depth(4U);
  spec.input(control).optional().queue_depth(4U);
  spec.input(evict).optional().queue_depth(4U);
  spec.output(range_profiles_out)
      .produces_tensor(pooled_tensor_output(
          kDevice, 2U, tensor_byte_span(std::vector<int64_t>{pulses, bins}, complex_dtype<T>())));
  for (Output<Tensor>* port : {&positions_out, &reference_position_out}) {
    spec.output(*port).produces_tensor(pooled_tensor_output(
        kDevice, 2U, tensor_byte_span(std::vector<int64_t>{pulses, 3}, real_dtype<TGeom>())));
  }
  spec.output(weight).produces_tensor(pooled_tensor_output(
      kDevice, 1U, tensor_byte_span(std::vector<int64_t>{pulses}, real_dtype<T>())));
  spec.output(sample_geometry_out)
      .produces_tensor(pooled_tensor_output(kHostGeometry, 1U, 11U * sizeof(double)));
  spec.output(sample_points_out)
      .produces_tensor(pooled_tensor_output(
          kDevice, 3U,
          tensor_byte_span(std::vector<int64_t>{*extents_.max_image_v, *extents_.max_image_u, 3},
                           real_dtype<TGeom>())));
  spec.output(control_out).produces_tensor(pooled_tensor_output(kDevice, 1U, sizeof(uint8_t)));
}

template <typename T, typename TGeom>
Contract ApertureManagerOp<T, TGeom>::contract() const {
  Contract result;
  result
      .trigger(
          OnAligned{.ports = ports(range_profiles, positions, reference_position, stream_boundary),
                    .window = std::chrono::nanoseconds{0}})
      .input(range_profiles, Exact{})
      .input(positions, Exact{})
      .input(reference_position, Exact{})
      .input(stream_boundary, Exact{});
  add_live_override_to_contract(result, sample_geometry);
  add_live_override_to_contract(result, sample_points);
  add_live_override_to_contract(result, control);
  add_live_override_to_contract(result, evict);
  return result;
}

template <typename T, typename TGeom>
void ApertureManagerOp<T, TGeom>::start() {
  const auto& p = params();
  if (p.max_aperture_pulses < 0 || p.emit_interval_pulses < 0 ||
      (p.emit_mode == ApertureEmitMode::kReform && p.max_aperture_pulses == 0)) {
    throw std::invalid_argument("ApertureManagerOp: invalid aperture capacity or cadence");
  }
  if (p.max_aperture_pulses > *extents_.max_pulses) {
    throw std::invalid_argument("ApertureManagerOp: max_aperture_pulses exceeds Tensor pool");
  }
  buffer_ = ApertureBuffer(p.max_aperture_pulses);
  stamp_window_.clear();
  const bool stamped = p.max_aperture_pulses == 0 && p.emit_mode == ApertureEmitMode::kReset &&
                       p.emit_interval_pulses > 0;
  if (p.aperture_window.type != WindowType::kNone) {
    // A taper needs final membership: known at admission (stamped) or at the
    // boundary (reform). Anywhere else it would weight against the wrong slots.
    if (!stamped && p.emit_mode != ApertureEmitMode::kReform) {
      throw std::invalid_argument(
          "ApertureManagerOp: aperture_window requires emit_mode reform, or "
          "max_aperture_pulses 0 with emit_mode reset and emit_interval_pulses > 0");
    }
    if (stamped)
      stamp_window_ = generate_window(p.aperture_window, p.emit_interval_pulses);
  }
}

template <typename T, typename TGeom>
expected<void, Error> ApertureManagerOp<T, TGeom>::compute(ExecutionContext& context) {
  keep_.clear();
  HOLOSCAN_TRY(auto raw_profiles, receive_tensor(range_profiles, kWho));
  HOLOSCAN_TRY(auto raw_positions, receive_tensor(positions, kWho));
  HOLOSCAN_TRY(auto raw_reference, receive_tensor(reference_position, kWho));
  HOLOSCAN_TRY(auto boundary, stream_boundary.receive_data());
  HOLOSCAN_TRY(auto request_tensor, receive_optional_value(control));
  HOLOSCAN_TRY(auto evict_tensor, receive_optional_value(evict));
  HOLOSCAN_TRY(auto geometry_wire, receive_optional_value(sample_geometry));
  HOLOSCAN_TRY(auto points_tensor, receive_optional_value(sample_points));
  try {
    require_complex<T>(raw_profiles, "range_profiles");
    require_dtype(raw_positions, real_dtype<TGeom>(), "positions", kWho);
    require_dtype(raw_reference, real_dtype<TGeom>(), "reference_position", kWho);
    require_contiguous(raw_profiles, "range_profiles", kWho);
    require_contiguous(raw_positions, "positions", kWho);
    require_contiguous(raw_reference, "reference_position", kWho);
    require_cuda_accessible(raw_profiles, "range_profiles", kWho);
    require_cuda_accessible(raw_positions, "positions", kWho);
    require_cuda_accessible(raw_reference, "reference_position", kWho);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }
  const auto shape = raw_profiles.shape();
  if (shape.size() != 2U || raw_positions.shape() != std::vector<int64_t>{shape[0], 3} ||
      raw_reference.shape() != std::vector<int64_t>{shape[0], 3} || shape[0] <= 0 ||
      shape[1] <= 0 || shape[0] > *extents_.max_pulses || shape[1] > *extents_.max_range_bins ||
      boundary > 2U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 "ApertureManagerOp: input shape or boundary is invalid"});
  }
  const auto& p = params();
  const int64_t np = shape[0];
  if (p.max_aperture_pulses > 0 && np > p.max_aperture_pulses) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 "ApertureManagerOp: input batch exceeds retention capacity"});
  }
  cudaStream_t stream = context.cuda_stream().get();
  std::optional<FormationControl> request;
  if (request_tensor) {
    HOLOSCAN_TRY(auto raw, read_scalar<uint8_t>(*request_tensor, "control", stream));
    if (raw > 3U) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "ApertureManagerOp: unknown FormationControl"});
    }
    request = static_cast<FormationControl>(raw);
  }
  int64_t evict_count = 0;
  if (evict_tensor) {
    HOLOSCAN_TRY(auto raw, read_scalar<uint32_t>(*evict_tensor, "evict", stream));
    evict_count = raw;
  }
  if (geometry_wire) {
    HOLOSCAN_TRY(auto geometry, from_wire(*geometry_wire));
    pending_geometry_ = std::move(geometry);
  }
  if (points_tensor)
    pending_points_ = std::make_shared<Tensor>(std::move(*points_tensor));

  // Update aperture membership before deciding whether this tick closes it.
  const bool reform = p.emit_mode == ApertureEmitMode::kReform;
  const bool pre_reset = request && pre_resets(*request) && !reform;
  if (pre_reset) {
    dropped_ += buffer_.occupancy();
    buffer_.clear();
    admitted_since_emit_ = slot_ = 0;
  }
  std::vector<PulseGroup> departing;
  if (!reform && evict_count > 0) {
    if (evict_count > buffer_.occupancy()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "ApertureManagerOp: evict exceeds occupancy"});
    }
    departing = buffer_.evict(evict_count);
    evicted_ += pulse_group_total(departing);
  }
  if (!reform && !evict_tensor && p.max_aperture_pulses > 0) {
    auto overflow = buffer_.evict_for_admission(np);
    evicted_ += pulse_group_total(overflow);
    departing.insert(departing.end(), overflow.begin(), overflow.end());
  }
  if (p.max_aperture_pulses > 0 && buffer_.occupancy() + np > p.max_aperture_pulses) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 "ApertureManagerOp: admission exceeds retention capacity"});
  }
  PulseGroup incoming{std::make_shared<Tensor>(std::move(raw_profiles)),
                      std::make_shared<Tensor>(std::move(raw_positions)),
                      std::make_shared<Tensor>(std::move(raw_reference)), 0, np};
  if (p.max_aperture_pulses > 0) {
    auto retained = copy_for_retention(incoming, stream);
    // Release the received pool-backed tensors only after their asynchronous
    // copies have completed.
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return make_unexpected(
          Error{ErrorCode::kFailure, "ApertureManagerOp: retention sync failed"});
    buffer_.admit(std::move(retained));
  }
  const int64_t slot0 = slot_;
  admitted_ += np;
  admitted_since_emit_ += np;
  slot_ += np;
  const bool cadence = p.emit_interval_pulses > 0 && admitted_since_emit_ >= p.emit_interval_pulses;
  const bool end_aperture = boundary != 0U || (request && resets(*request)) ||
                            (cadence && p.emit_mode != ApertureEmitMode::kContinue);
  const bool emit_now = cadence || request.has_value() || boundary != 0U;
  FormationControl output_control = FormationControl::kEmitAndContinue;
  if (reform && emit_now)
    output_control = FormationControl::kResetThenEmit;
  else if (boundary != 0U || (cadence && p.emit_mode == ApertureEmitMode::kReset))
    output_control = FormationControl::kEmitAndReset;
  else if (request)
    output_control = *request;

  // Build additions followed by signed departures, or replay the retained aperture.
  keep_.insert(keep_.end(), {incoming.profiles, incoming.positions, incoming.reference});
  std::vector<PulseGroup> output_groups;
  if (reform) {
    if (emit_now)
      output_groups = buffer_.retained();
  } else {
    output_groups.push_back(incoming);
    output_groups.insert(output_groups.end(), departing.begin(), departing.end());
  }
  std::optional<PulseGroup> output_batch;
  std::vector<T> weights;
  if (!output_groups.empty()) {
    const int64_t n_out = pulse_group_total(output_groups);
    if (n_out > *extents_.max_pulses) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "ApertureManagerOp: output exceeds max_pulses"});
    }
    PulseGroup batch = concat_pulse_groups(output_groups, stream, kWho);
    keep_.insert(keep_.end(), {batch.profiles, batch.positions, batch.reference});
    weights.assign(static_cast<size_t>(n_out), T{1});
    if (reform) {
      // A reform resets and replays the whole retained aperture, so every
      // pulse is an admission: +1, or the taper where one is configured
      // (D15 -- reform is the sanctioned tapered path, and unweighted is an
      // honest ~-13 dB rather than a placeholder). The additions/departures
      // layout below must not be reached: it would mark everything from `np`
      // on as a departure and tell the backprojector to subtract most of the
      // replay, and its index split does not even line up here, since
      // `retained()` is ordered oldest-first while `np` counts this tick's
      // arrivals. Keyed on `reform` rather than on the window type, which is
      // how an unwindowed reform used to fall through -- WindowSpec defaults
      // to kNone, so that was the default reform configuration.
      if (p.aperture_window.type != WindowType::kNone) {
        const auto coeff = generate_window(p.aperture_window, n_out);
        for (int64_t i = 0; i < n_out; ++i)
          weights[i] = static_cast<T>(coeff[i]);
      }
    } else {
      for (int64_t i = 0; i < np; ++i) {
        if (!stamp_window_.empty() && slot0 + i < static_cast<int64_t>(stamp_window_.size()))
          weights[i] = static_cast<T>(stamp_window_[slot0 + i]);
      }
      for (int64_t i = np; i < n_out; ++i)
        weights[i] = T{-1};
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return make_unexpected(Error{ErrorCode::kFailure, "ApertureManagerOp: stream sync failed"});
    output_batch.emplace(std::move(batch));
  }
  if (pending_geometry_ && geometry_output_.active()) {
    HOLOSCAN_TRY(auto loan, prepare_geometry(sample_geometry_out, *pending_geometry_));
    geometry_output_.emit_schema_tensor(sample_geometry_out, context, std::move(loan),
                                        to_wire(*pending_geometry_),
                                        "ApertureManagerOp.sample_geometry_out");
  }
  if (pending_points_ && points_output_.active()) {
    HOLOSCAN_TRY(auto loan, prepare_copy(sample_points_out, context, *pending_points_));
    points_output_.emit(sample_points_out, context, std::move(loan),
                        "ApertureManagerOp.sample_points_out");
  }
  if (emit_now && control_output_.active()) {
    const std::vector<uint8_t> value{static_cast<uint8_t>(output_control)};
    HOLOSCAN_TRY(auto loan,
                 prepare_values(control_out, context, value, DLDataType{kDLUInt, 8U, 1U}));
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return make_unexpected(Error{ErrorCode::kFailure, "ApertureManagerOp: control sync failed"});
    control_output_.emit(control_out, context, std::move(loan), "ApertureManagerOp.control_out");
    admitted_since_emit_ = 0;
  }
  if (output_batch) {
    auto emit_copy = [&](Output<Tensor>& port, const Tensor& source,
                         std::string_view what) -> expected<void, Error> {
      return emit_required_tensor(
          port, context, what, false, [&]() -> expected<TensorOutputLoan, Error> {
            HOLOSCAN_TRY(auto loan, prepare_copy(port, context, source));
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
              return make_unexpected(
                  Error{ErrorCode::kFailure, "ApertureManagerOp: device copy failed"});
            }
            return loan;
          });
    };
    if (weight_output_.active()) {
      HOLOSCAN_TRY(auto weight_loan, prepare_values(weight, context, weights, real_dtype<T>()));
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "ApertureManagerOp: weight/control H2D failed"});
      }
      weight_output_.emit(weight, context, std::move(weight_loan), "ApertureManagerOp.weight");
    }
    HOLOSCAN_TRY(
        emit_copy(positions_out, *output_batch->positions, "ApertureManagerOp.positions_out"));
    HOLOSCAN_TRY(emit_copy(reference_position_out, *output_batch->reference,
                           "ApertureManagerOp.reference_position_out"));
    HOLOSCAN_TRY(emit_copy(range_profiles_out, *output_batch->profiles,
                           "ApertureManagerOp.range_profiles_out"));
  }
  // A collection seam closes the aperture after its final output is published.
  if (end_aperture) {
    const bool emits_image = emit_now && output_control != FormationControl::kResetWithoutEmit;
    if (emits_image && !stamp_window_.empty() &&
        slot_ < static_cast<int64_t>(stamp_window_.size())) {
      HOLOSCAN_LOG_WARN(
          "{}: aperture closed after {} of {} pulses; its aperture_window was truncated "
          "(use emit_mode reform for an exact taper on a short aperture)",
          kWho, slot_, stamp_window_.size());
    }
    dropped_ += buffer_.occupancy();
    buffer_.clear();
    slot_ = admitted_since_emit_ = 0;
    pending_geometry_.reset();
    pending_points_.reset();
  }
  return {};
}

template class ApertureManagerOp<float, float>;
template class ApertureManagerOp<double, double>;
template class ApertureManagerOp<float, double>;
}  // namespace holoscan::radar
