// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/cphd_reader_op/cphd_reader_op.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

// The metadata key the boundary announcement rides on (D22). The reader is
// the only stage that knows where a collection ends, so it stamps the key on
// the messages of the batch that *is* the last one -- announcing on the
// following batch would be a boundary reported after the closing aperture
// had already been contaminated.
constexpr const char* kStreamBoundaryKey = "holoscan.radar.stream_boundary";
constexpr const char* kEndOfStream = "end_of_stream";

// Copy a host buffer to a freshly allocated device tensor of `shape`.
template <typename Element>
std::shared_ptr<holoscan::Tensor> upload(const std::vector<Element>& host,
                                         const std::vector<int64_t>& shape,
                                         DLDataType dtype, const char* who) {
  auto tensor = make_device_tensor(shape, dtype);
  if (cudaMemcpy(tensor->data(), host.data(), sizeof(Element) * host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error(std::string(who) + ": H2D upload failed");
  }
  return tensor;
}

// Narrow the file's doubles to the instantiated precision. PVP records are
// always float64 on disk; single-precision instances pay the cast here so
// the rest of the chain sees one dtype.
template <typename T>
std::vector<T> narrow(const std::vector<double>& values) {
  std::vector<T> narrowed(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    narrowed[i] = static_cast<T>(values[i]);
  }
  return narrowed;
}

}  // namespace

template <typename T>
CPHDReaderOp<T>::~CPHDReaderOp() = default;

template <typename T>
void CPHDReaderOp<T>::initialize() {
  ensure_radar_arg_setters();

  // The scheduling term this operator disables once the selected range is
  // exhausted. Created here rather than by the application so terminating is
  // the reader's own business, matching the SDK's source operators.
  stop_condition_ =
      fragment()->make_condition<holoscan::BooleanCondition>("stop_condition");
  add_arg(stop_condition_);

  Operator::initialize();
}

template <typename T>
void CPHDReaderOp<T>::setup(OperatorSpec& spec) {
  spec.output<holoscan::TensorMap>("phase_history");
  spec.output<holoscan::TensorMap>("positions");
  spec.output<holoscan::TensorMap>("reference_position");
  // Optional outputs: kNone so an unconnected port never gates scheduling.
  spec.output<holoscan::TensorMap>("sample_axis")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("pulse_times")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("velocities")
      .condition(ConditionType::kNone);
  spec.output<CoordinateFrame>("coordinate_frame")
      .condition(ConditionType::kNone);

  spec.param(path_, "path", "CPHD file path",
             "Path to a single CPHD file. The directory and ordered-list "
             "forms land with multi-file support (mode 3).");

  spec.param(coordinate_frame_, "coordinate_frame", "Coordinate frame",
             "Frame the emitted positions are expressed in. A frame "
             "disagreeing with the one the file declares is an error, not a "
             "silent reinterpretation.");

  spec.param(pulses_per_tick_, "pulses_per_tick", "Pulses per tick",
             "Batch size along the pulse axis. The final batch is short when "
             "the selected range is not a multiple of it.");

  spec.param(channel_, "channel", "CPHD channel index",
             "Which CPHD channel to read. One channel per instance is "
             "deliberate; see OPERATOR_API.md.",
             static_cast<int64_t>(0));

  spec.param(pulse_range_, "pulse_range", "Pulse range",
             "Half-open [first, last) pulse range to read. Default: the "
             "whole collection.",
             ParameterFlag::kOptional);
}

template <typename T>
void CPHDReaderOp<T>::start() {
  const std::string who =
      std::string("CPHDReaderOp<") + precision_name<T>() + ">";

  file_ = std::make_unique<cphd::CphdFile>(path_.get());
  const cphd::ChannelInfo& info = file_->channel(channel_.get());

  // CPHD carries positions in the earth-centred fixed frame, so anything
  // else the caller asks for would be a reinterpretation of the file's
  // numbers rather than a conversion of them.
  const CoordinateFrame& frame = coordinate_frame_.get();
  if (!frame.is_valid()) {
    throw std::runtime_error(who + ": coordinate_frame is not valid");
  }
  if (frame.frame() != CoordinateFrameType::kEcef) {
    throw std::runtime_error(
        who + ": file '" + path_.get() +
        "' declares its positions in ECF, but coordinate_frame is " +
        std::string(to_string(frame.frame())) +
        ". CPHD positions are ECEF; conversion is a downstream concern.");
  }

  if (file_->sample_domain() != cphd::SampleDomain::kFX) {
    throw std::runtime_error(
        who + ": file '" + path_.get() + "' is a " +
        to_string(file_->sample_domain()) +
        "-domain collection; only FX is implemented (TOA lands with the "
        "collections that force it)");
  }
  if (file_->has_pvp_field("AmpSF")) {
    throw std::runtime_error(
        who + ": file '" + path_.get() +
        "' carries per-vector amplitude scale factors (AmpSF), which are "
        "not applied yet (mode 3); reading it would understate amplitudes");
  }

  if (pulses_per_tick_.get() <= 0) {
    throw std::runtime_error(who + ": pulses_per_tick must be positive, got " +
                             std::to_string(pulses_per_tick_.get()));
  }

  first_pulse_ = 0;
  last_pulse_ = info.num_vectors;
  const auto range = pulse_range_.try_get();
  if (range.has_value()) {
    if (range->size() != 2) {
      throw std::runtime_error(who + ": pulse_range needs 2 entries, got " +
                               std::to_string(range->size()));
    }
    first_pulse_ = (*range)[0];
    last_pulse_ = (*range)[1];
    if (first_pulse_ < 0 || last_pulse_ > info.num_vectors ||
        first_pulse_ >= last_pulse_) {
      throw std::runtime_error(
          who + ": pulse_range [" + std::to_string(first_pulse_) + ", " +
          std::to_string(last_pulse_) + ") is out of range; the channel has " +
          std::to_string(info.num_vectors) + " pulses");
    }
  }
  next_pulse_ = first_pulse_;
  frame_emitted_ = false;

  HOLOSCAN_LOG_INFO(
      "{}: '{}' channel {} ({}): {} pulses x {} samples, reading [{}, {}) in "
      "batches of {}",
      who, path_.get(), channel_.get(), info.identifier, info.num_vectors,
      info.num_samples, first_pulse_, last_pulse_, pulses_per_tick_.get());
}

template <typename T>
void CPHDReaderOp<T>::compute([[maybe_unused]] InputContext& op_input,
                              OutputContext& op_output,
                              [[maybe_unused]] ExecutionContext& context) {
  const std::string who =
      std::string("CPHDReaderOp<") + precision_name<T>() + ">";
  const cphd::ChannelInfo& info = file_->channel(channel_.get());
  const int64_t num_samples = info.num_samples;

  // The final batch is short whenever the range is not a multiple of the
  // batch size; downstream infers the count from the tensor, never from
  // pulses_per_tick.
  const int64_t count =
      std::min(pulses_per_tick_.get(), last_pulse_ - next_pulse_);
  const int64_t first = next_pulse_;
  const bool is_final = (first + count >= last_pulse_);

  // 1. Phase history, read straight into the staging buffer at this
  // instance's precision. A single-precision instance makes no copy beyond
  // this one: CF8 is big-endian complex<float>, so the bytes land here and
  // are swapped in place.
  std::vector<std::complex<T>> history(
      static_cast<size_t>(count * num_samples));
  file_->read_signal_into(channel_.get(), first, count, history.data());

  // 2. Per-pulse geometry. Every field this tick needs comes out of one read
  // of the parameter block: the records are interleaved, so reading eight
  // fields separately would be eight passes over the same bytes.
  const cphd::PvpBlock pvp =
      file_->read_pvp_block(channel_.get(), first, count);

  // The aperture phase centre is the midpoint of the transmit and receive
  // phase centres, which reduces to either one for the monostatic
  // collections this wave covers. Which of the three readings the contract
  // means is a convention still to be pinned against the mode 3 collections
  // -- see this operator's OPEN_ITEMS.md.
  const std::vector<double> tx_pos = pvp.field("TxPos", 3);
  const std::vector<double> rcv_pos = pvp.field("RcvPos", 3);
  std::vector<double> apc(tx_pos.size());
  for (size_t i = 0; i < apc.size(); ++i) {
    apc[i] = 0.5 * (tx_pos[i] + rcv_pos[i]);
  }
  const std::vector<double> srp = pvp.field("SRPPos", 3);

  // Public collections do carry non-finite ephemeris on a minority of
  // vectors -- one surveyed Umbra file has NaN Tx fields on 32 of 4844, with
  // a fully populated signal array. Passing them through keeps a collection
  // that is otherwise good, but silently: the NaN reaches the image as blank
  // pixels with nothing to attribute them to. So say it once, with the count,
  // and leave the policy decision (drop? refuse?) to whoever hits a file bad
  // enough to need one.
  if (!nonfinite_warned_) {
    int64_t bad = 0;
    for (int64_t pulse = 0; pulse < count; ++pulse) {
      const size_t base = static_cast<size_t>(pulse) * 3;
      if (!std::isfinite(apc[base]) || !std::isfinite(apc[base + 1]) ||
          !std::isfinite(apc[base + 2])) {
        ++bad;
      }
    }
    if (bad > 0) {
      nonfinite_warned_ = true;
      HOLOSCAN_LOG_WARN(
          "{}: {} of {} pulses in this batch have non-finite phase-centre "
          "positions (NaN TxPos/RcvPos in the file); they are passed through "
          "and will appear as blank image regions. Warned once per instance.",
          who, bad, count);
    }
  }

  // 3. The fast-time sample axis. SC0 and SCSS are per-vector fields, but
  // the axis is one shared row, so their constancy over the batch is a
  // precondition rather than an assumption -- a collection that steps its
  // start frequency would otherwise be read at the wrong frequencies and
  // produce a plausible, wrong image.
  const std::vector<double> sc0 = pvp.field("SC0", 1);
  const std::vector<double> scss = pvp.field("SCSS", 1);
  for (int64_t pulse = 1; pulse < count; ++pulse) {
    if (sc0[pulse] != sc0[0] || scss[pulse] != scss[0]) {
      throw std::runtime_error(
          who + ": SC0/SCSS vary across pulses (pulse " +
          std::to_string(first + pulse) +
          " differs from the batch's first); a per-pulse frequency axis is "
          "not supported");
    }
  }
  std::vector<T> axis(static_cast<size_t>(num_samples));
  for (int64_t sample = 0; sample < num_samples; ++sample) {
    axis[static_cast<size_t>(sample)] =
        static_cast<T>(sc0[0] + static_cast<double>(sample) * scss[0]);
  }

  // 4. Emit. The boundary key is stamped before the emits so it rides the
  // messages of this batch rather than the next one.
  if (is_final && is_metadata_enabled()) {
    metadata()->set(kStreamBoundaryKey, std::string(kEndOfStream));
  }

  const std::vector<int64_t> pulse_shape{count, 3};
  emit_single(op_output, "phase_history",
              upload(history, {count, num_samples}, complex_dtype<T>(),
                     who.c_str()));
  emit_single(op_output, "positions",
              upload(narrow<T>(apc), pulse_shape, real_dtype<T>(),
                     who.c_str()));
  emit_single(op_output, "reference_position",
              upload(narrow<T>(srp), pulse_shape, real_dtype<T>(),
                     who.c_str()));
  emit_single(op_output, "sample_axis",
              upload(axis, {num_samples}, real_dtype<T>(), who.c_str()));

  // Pulse times stay float64 in both instantiations: a collection-relative
  // timestamp does not survive single precision to useful accuracy.
  const std::vector<double> times = pvp.field("TxTime", 1);
  emit_single(op_output, "pulse_times",
              upload(times, {count}, real_dtype<double>(), who.c_str()));

  if (pvp.has("TxVel") && pvp.has("RcvVel")) {
    const std::vector<double> tx_vel = pvp.field("TxVel", 3);
    const std::vector<double> rcv_vel = pvp.field("RcvVel", 3);
    std::vector<double> velocity(tx_vel.size());
    for (size_t i = 0; i < velocity.size(); ++i) {
      velocity[i] = 0.5 * (tx_vel[i] + rcv_vel[i]);
    }
    emit_single(op_output, "velocities",
                upload(narrow<T>(velocity), pulse_shape, real_dtype<T>(),
                       who.c_str()));
  }

  // The descriptor edge is emitted once, at the first tick: a consumer
  // wiring it into a [live] override needs the value once, overrides being
  // sticky.
  if (!frame_emitted_) {
    op_output.emit(coordinate_frame_.get(), "coordinate_frame");
    frame_emitted_ = true;
  }

  next_pulse_ = first + count;
  if (is_final) {
    HOLOSCAN_LOG_INFO("{}: end of stream after {} pulses", who,
                      last_pulse_ - first_pulse_);
    stop_condition_->disable_tick();
  }
}

template class CPHDReaderOp<float>;
template class CPHDReaderOp<double>;

}  // namespace holoscan::radar
