// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/cphd_reader_op/cphd_reader_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {
namespace {
constexpr const char* kWho = "CPHDReaderOp";
constexpr TensorRepresentation kDevice{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = std::nullopt,
    .rank = std::nullopt};  // NOLINT(whitespace/indent_namespace)
constexpr TensorRepresentation kFrameHost{.memory_kind = MemoryKind::kHost,
                                          .dtype = DLDataType{kDLFloat, 64U, 1U},
                                          .rank = 1U};  // NOLINT(whitespace/indent_namespace)

std::vector<std::string> resolve_paths(const CollectionPaths& source) {
  std::vector<std::string> result;
  for (const auto& entry : source.entries) {
    if (!std::filesystem::exists(entry)) {
      throw std::invalid_argument(std::string(kWho) + ": path does not exist: " + entry);
    }
    if (!std::filesystem::is_directory(entry)) {
      result.push_back(entry);
      continue;
    }
    std::vector<std::string> found;
    for (const auto& item : std::filesystem::directory_iterator(entry)) {
      if (item.is_regular_file() && item.path().extension() == ".cphd") {
        found.push_back(item.path().string());
      }
    }
    std::sort(found.begin(), found.end());
    result.insert(result.end(), found.begin(), found.end());
  }
  if (result.empty())
    throw std::invalid_argument(std::string(kWho) + ": path selects no CPHD files");
  return result;
}

template <typename Element>
expected<TensorOutputLoan, Error> upload(Output<Tensor>& output, ExecutionContext& context,
                                         const std::vector<Element>& values,
                                         const std::vector<int64_t>& shape, DLDataType dtype) {
  return prepare_device_tensor(
      output, context, shape, dtype, [&](void* dst, std::size_t bytes) -> expected<void, Error> {
        if (bytes != values.size() * sizeof(Element)) {
          return make_unexpected(
              Error{ErrorCode::kInvalidArgument, "CPHDReaderOp: upload shape and data differ"});
        }
        if (cudaMemcpyAsync(dst, values.data(), bytes, cudaMemcpyHostToDevice,
                            context.cuda_stream().get()) != cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure, "CPHDReaderOp: H2D upload failed"});
        }
        return {};
      });
}

expected<TensorOutputLoan, Error> prepare_frame(Output<WireCoordinateFrame>& output,
                                                const CoordinateFrame& frame) {
  auto loan = output.allocate_tensor(
      TensorLoanRequest{.shape = std::vector<int64_t>{3}, .dtype = DLDataType{kDLFloat, 64U, 1U}});
  if (!loan)
    return make_unexpected(std::move(loan).error());
  const std::array<double, 3> origin = frame.origin_llh().value_or(std::array<double, 3>{});
  auto committed = loan->write_host_and_commit([&origin](const TensorOutputWriteGuard& writer) {
    std::copy(origin.begin(), origin.end(), writer.data_as<double>());
  });
  if (!committed)
    return make_unexpected(std::move(committed).error());
  return std::move(*loan);
}

void check_sample_axis(const std::vector<double>& values, const char* name) {
  for (size_t i = 1; i < values.size(); ++i) {
    const double scale = std::max(std::abs(values[0]), std::abs(values[i]));
    if (!std::isfinite(values[i]) || std::abs(values[i] - values[0]) > 1e-10 * scale) {
      throw std::runtime_error(std::string(kWho) + ": " + name + " varies across a pulse batch");
    }
  }
}
}  // namespace

template <typename T>
CPHDReaderOp<T>::CPHDReaderOp(const MaxTensorExtents& extents) : extents_(extents) {
  require_extent(extents.max_pulses, "max_pulses", kWho);
  require_extent(extents.max_samples_per_pulse, "max_samples_per_pulse", kWho);
}
template <typename T>
CPHDReaderOp<T>::~CPHDReaderOp() = default;

template <typename T>
void CPHDReaderOp<T>::setup(OperatorSpec& spec) {
  const auto pulses = *extents_.max_pulses;
  const auto samples = *extents_.max_samples_per_pulse;
  spec.output(phase_history)
      .produces_tensor(pooled_tensor_output(
          kDevice, 2U,
          tensor_byte_span(std::vector<int64_t>{pulses, samples}, complex_dtype<T>())));
  spec.output(positions).produces_tensor(pooled_tensor_output(
      kDevice, 2U, tensor_byte_span(std::vector<int64_t>{pulses, 3}, real_dtype<T>())));
  spec.output(reference_position)
      .produces_tensor(pooled_tensor_output(
          kDevice, 2U, tensor_byte_span(std::vector<int64_t>{pulses, 3}, real_dtype<T>())));
  spec.output(sample_axis)
      .produces_tensor(pooled_tensor_output(
          kDevice, 1U, tensor_byte_span(std::vector<int64_t>{samples}, real_dtype<T>())));
  spec.output(pulse_times)
      .produces_tensor(pooled_tensor_output(
          kDevice, 1U, tensor_byte_span(std::vector<int64_t>{pulses}, real_dtype<double>())));
  spec.output(velocities)
      .produces_tensor(pooled_tensor_output(
          kDevice, 2U, tensor_byte_span(std::vector<int64_t>{pulses, 3}, real_dtype<T>())));
  spec.output(coordinate_frame)
      .produces_tensor(pooled_tensor_output(kFrameHost, 1U, 3U * sizeof(double)));
  spec.output(stream_boundary);
}

template <typename T>
Contract CPHDReaderOp<T>::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::microseconds(1)});
  return result;
}

template <typename T>
void CPHDReaderOp<T>::open_file() {
  const auto& p = params();
  file_ = std::make_unique<cphd::CphdFile>(files_[file_index_]);
  const auto& info = file_->channel(p.channel);
  if (p.coordinate_frame.frame() != CoordinateFrameType::kEcef) {
    throw std::invalid_argument("CPHDReaderOp: CPHD positions require an ECEF coordinate_frame");
  }
  for (const char* field : {"TxTime", "TxPos", "RcvPos", "SRPPos", "SC0", "SCSS"}) {
    if (!file_->has_pvp_field(field)) {
      throw std::invalid_argument(std::string("CPHDReaderOp: missing PVP field ") + field);
    }
  }
  first_pulse_ = p.pulse_range.empty() ? 0 : p.pulse_range[0];
  last_pulse_ = p.pulse_range.empty() ? info.num_vectors : p.pulse_range[1];
  if (first_pulse_ < 0 || first_pulse_ >= last_pulse_ || last_pulse_ > info.num_vectors ||
      info.num_samples > *extents_.max_samples_per_pulse) {
    throw std::invalid_argument("CPHDReaderOp: pulse range or sample count exceeds bounds");
  }
  next_pulse_ = first_pulse_;
  nonfinite_warned_ = false;
}

template <typename T>
void CPHDReaderOp<T>::start() {
  const auto& p = params();
  if (!p.coordinate_frame.is_valid() || p.pulses_per_tick <= 0 ||
      p.pulses_per_tick > *extents_.max_pulses ||
      (!p.pulse_range.empty() && p.pulse_range.size() != 2U) || !std::isfinite(p.playback_rate) ||
      p.playback_rate < 0.0 || !std::isfinite(p.duty_cycle) || p.duty_cycle <= 0.0 ||
      p.duty_cycle > 1.0) {
    throw std::invalid_argument("CPHDReaderOp: invalid reader parameters or tensor bounds");
  }
  files_ = resolve_paths(p.path);
  // Parse every header before producing data, including later collections.
  for (const auto& path : files_) {
    cphd::CphdFile check(path);
    check.channel(p.channel);
  }
  file_index_ = 0;
  open_file();
}

template <typename T>
expected<void, Error> CPHDReaderOp<T>::compute(ExecutionContext& context) {
  if (stopped_) {
    context.request_stop();
    return {};
  }
  try {
    if (pending_cooldown_s_ > 0.0) {
      spans_.back().quiet_start = std::chrono::steady_clock::now();
      std::this_thread::sleep_for(std::chrono::duration<double>(pending_cooldown_s_));
      spans_.back().quiet_end = std::chrono::steady_clock::now();
      pending_cooldown_s_ = 0.0;
    }
    const auto& p = params();
    const auto tick_start = std::chrono::steady_clock::now();
    const auto& info = file_->channel(p.channel);
    const int64_t count = std::min(p.pulses_per_tick, last_pulse_ - next_pulse_);
    const int64_t first = next_pulse_;
    const bool final = first + count == last_pulse_;
    const bool follows = p.loop || file_index_ + 1U < files_.size();
    if (first == first_pulse_) {
      collection_wall_start_ = tick_start;
      spans_.push_back(
          CollectionSpan{.file_index = static_cast<int>(file_index_), .dwell_start = tick_start});
      if (spans_.size() > 32U)
        spans_.erase(spans_.begin());
    }
    // Decode one bounded CPHD batch and derive the pulse geometry from its PVPs.
    auto pvp = file_->read_pvp_block(p.channel, first, count);
    std::vector<std::complex<T>> history(static_cast<size_t>(count * info.num_samples));
    file_->read_signal_into<T>(p.channel, first, count, history.data(), &pvp);
    const auto tx = pvp.field("TxPos", 3);
    const auto rx = pvp.field("RcvPos", 3);
    const auto srp = pvp.field("SRPPos", 3);
    std::vector<T> apc(tx.size()), reference(srp.size());
    std::array<double, 3> mean{};
    int64_t finite_count = 0;
    for (int64_t pulse = 0; pulse < count; ++pulse) {
      bool finite = true;
      for (int axis = 0; axis < 3; ++axis) {
        const size_t i = static_cast<size_t>(3 * pulse + axis);
        const double value = 0.5 * (tx[i] + rx[i]);
        apc[i] = static_cast<T>(value);
        reference[i] = static_cast<T>(srp[i]);
        finite = finite && std::isfinite(value);
      }
      if (finite) {
        for (int axis = 0; axis < 3; ++axis)
          mean[axis] += apc[3 * pulse + axis];
        ++finite_count;
      } else if (!nonfinite_warned_) {
        HOLOSCAN_LOG_WARN("{}: non-finite phase-centre positions passed through", kWho);
        nonfinite_warned_ = true;
      }
    }
    if (finite_count > 0) {
      for (double& value : mean)
        value /= static_cast<double>(finite_count);
      last_apc_ = {.valid = true, .file_index = static_cast<int64_t>(file_index_), .ecef = mean};
    }
    const auto sc0 = pvp.field("SC0", 1);
    const auto scss = pvp.field("SCSS", 1);
    check_sample_axis(sc0, "SC0");
    check_sample_axis(scss, "SCSS");
    std::vector<T> axis(static_cast<size_t>(info.num_samples));
    for (int64_t i = 0; i < info.num_samples; ++i) {
      axis[static_cast<size_t>(i)] = static_cast<T>(sc0[0] + i * scss[0]);
    }
    const auto times = pvp.field("TxTime", 1);
    if (first == first_pulse_)
      collection_first_time_ = times.front();
    collection_last_time_ = times.back();
    std::vector<T> vel;
    if (pvp.has("TxVel") && pvp.has("RcvVel")) {
      const auto tx_vel = pvp.field("TxVel", 3);
      const auto rx_vel = pvp.field("RcvVel", 3);
      vel.resize(tx_vel.size());
      for (size_t i = 0; i < vel.size(); ++i) {
        vel[i] = static_cast<T>(0.5 * (tx_vel[i] + rx_vel[i]));
      }
    }

    // Upload the required tensors before publishing any batch output.
    HOLOSCAN_TRY(auto history_loan, upload(phase_history, context, history,
                                           {count, info.num_samples}, complex_dtype<T>()));
    HOLOSCAN_TRY(auto positions_loan, upload(positions, context, apc, {count, 3}, real_dtype<T>()));
    HOLOSCAN_TRY(auto reference_loan,
                 upload(reference_position, context, reference, {count, 3}, real_dtype<T>()));
    if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "CPHDReaderOp: H2D sync failed"});
    }
    if (axis_out_.active()) {
      HOLOSCAN_TRY(auto loan,
                   upload(sample_axis, context, axis, {info.num_samples}, real_dtype<T>()));
      axis_out_.emit(sample_axis, context, std::move(loan), "CPHDReaderOp.sample_axis");
    }
    if (times_out_.active()) {
      HOLOSCAN_TRY(auto loan, upload(pulse_times, context, times, {count}, real_dtype<double>()));
      times_out_.emit(pulse_times, context, std::move(loan), "CPHDReaderOp.pulse_times");
    }
    if (velocities_out_.active() && !vel.empty()) {
      HOLOSCAN_TRY(auto loan, upload(velocities, context, vel, {count, 3}, real_dtype<T>()));
      velocities_out_.emit(velocities, context, std::move(loan), "CPHDReaderOp.velocities");
    }
    if (frame_out_.active()) {
      HOLOSCAN_TRY(auto loan, prepare_frame(coordinate_frame, p.coordinate_frame));
      frame_out_.emit_schema_tensor(coordinate_frame, context, std::move(loan),
                                    to_wire(p.coordinate_frame), "CPHDReaderOp.coordinate_frame");
    }
    HOLOSCAN_RETURN_IF_ERROR(emit_prepared_tensor(phase_history, context, std::move(history_loan)));
    HOLOSCAN_RETURN_IF_ERROR(emit_prepared_tensor(positions, context, std::move(positions_loan)));
    HOLOSCAN_RETURN_IF_ERROR(
        emit_prepared_tensor(reference_position, context, std::move(reference_loan)));
    // Stamp the closing boundary on this batch, including the last partial batch.
    const auto boundary =
        final ? (follows ? SarStreamBoundary::kCollectionBoundary : SarStreamBoundary::kEndOfStream)
              : SarStreamBoundary::kNone;
    HOLOSCAN_RETURN_IF_ERROR(stream_boundary.emit(
        static_cast<uint8_t>(boundary), EmitOptions{.capture_time = context.activation_time()}));
    last_packet_at_ = std::chrono::steady_clock::now();
    const double collect_s = collection_last_time_ - collection_first_time_;
    double wall_s =
        std::chrono::duration<double>(*last_packet_at_ - collection_wall_start_).count();
    double slack_s = p.playback_rate > 0.0 ? collect_s / p.playback_rate - wall_s : 0.0;
    playback_ = {.valid = true,
                 .paced = p.playback_rate > 0.0,
                 .collect_s = collect_s,
                 .wall_s = wall_s,
                 .behind_s = -slack_s,
                 .last_slack_s = slack_s};
    if (slack_s > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(slack_s));
      wall_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_wall_start_)
              .count();
      playback_.wall_s = wall_s;
      playback_.behind_s = collect_s / p.playback_rate - wall_s;
    }
    next_pulse_ += count;
    if (final) {
      spans_.back().dwell_end = std::chrono::steady_clock::now();
      if (follows) {
        const double duration = collection_last_time_ - collection_first_time_;
        pending_cooldown_s_ = duration * (1.0 / p.duty_cycle - 1.0);
        file_index_ = file_index_ + 1U == files_.size() ? 0U : file_index_ + 1U;
        open_file();
      } else {
        stopped_ = true;
      }
    }
    return {};
  } catch (const std::exception& error) {
    return make_unexpected(Error{ErrorCode::kFailure, error.what()});
  }
}

template class CPHDReaderOp<float>;
template class CPHDReaderOp<double>;
}  // namespace holoscan::radar
