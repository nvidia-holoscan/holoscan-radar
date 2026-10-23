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
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/logger/logger.hpp>

#include "io/cphd_reader_op/cphd_cf8_decode.hpp"
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
// Match the Gantt lag-band entry/exit so the console and HUD agree.
constexpr double kBehindWarnEnterS = 0.2;
constexpr double kBehindWarnExitS = 0.05;
constexpr auto kBehindWarnPeriod = std::chrono::seconds{1};

struct CallbackTimer {
  std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};
  ~CallbackTimer() {
    HOLOSCAN_LOG_DEBUG(
        "CPHDReaderOp: callback={:.6f}s",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
  }
};

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
                                         const std::vector<int64_t>& shape, DLDataType dtype,
                                         cudaStream_t stream) {
  return prepare_device_tensor(
      output, context, shape, dtype, [&](void* dst, std::size_t bytes) -> expected<void, Error> {
        if (bytes != values.size() * sizeof(Element)) {
          return make_unexpected(
              Error{ErrorCode::kInvalidArgument, "CPHDReaderOp: upload shape and data differ"});
        }
        if (cudaMemcpyAsync(dst, values.data(), bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
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

expected<void, Error> cuda_ok(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure, std::string("CPHDReaderOp: ") + what});
  }
  return {};
}

template <typename T>
expected<void, Error> fill_history(void* dst, std::size_t bytes, std::size_t nsamp,
                                   int64_t num_samples, bool cf8, const char* pinned,
                                   void* device_cf8, double* device_amp,
                                   const std::vector<double>& amp, cudaStream_t stream) {
  const std::size_t out_bytes = nsamp * sizeof(std::complex<T>);
  if (bytes != out_bytes) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CPHDReaderOp: phase_history shape and data differ"});
  }
  const double* d_amp = nullptr;
  if (!amp.empty()) {
    HOLOSCAN_TRY(cuda_ok(cudaMemcpyAsync(device_amp, amp.data(), amp.size() * sizeof(double),
                                         cudaMemcpyHostToDevice, stream),
                         "AmpSF H2D failed"));
    d_amp = device_amp;
  }
  if (!cf8) {
    return cuda_ok(cudaMemcpyAsync(dst, pinned, out_bytes, cudaMemcpyHostToDevice, stream),
                   "H2D upload failed");
  }
  const std::size_t stored = nsamp * 8U;
  if constexpr (std::is_same_v<T, float>) {
    (void)device_cf8;
    HOLOSCAN_TRY(cuda_ok(cudaMemcpyAsync(dst, pinned, stored, cudaMemcpyHostToDevice, stream),
                         "CF8 H2D failed"));
    return cuda_ok(cphd_decode_cf8_f32(dst, dst, nsamp, num_samples, d_amp, stream),
                   "CF8 decode failed");
  } else {
    if (device_cf8 == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "CPHDReaderOp: device CF8 scratch is not armed"});
    }
    HOLOSCAN_TRY(
        cuda_ok(cudaMemcpyAsync(device_cf8, pinned, stored, cudaMemcpyHostToDevice, stream),
                "CF8 H2D failed"));
    return cuda_ok(cphd_decode_cf8_f64(device_cf8, dst, nsamp, num_samples, d_amp, stream),
                   "CF8 decode failed");
  }
}

bool is_transient_pool_error(ErrorCode code) {
  return code == ErrorCode::kResourceExhausted || code == ErrorCode::kBackpressured;
}

expected<void, Error> sync_ingest(cudaStream_t h2d, cudaEvent_t h2d_done, cudaStream_t compute) {
  if (cudaEventRecord(h2d_done, h2d) != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure, "CPHDReaderOp: H2D event record failed"});
  }
  if (compute != nullptr && cudaStreamWaitEvent(compute, h2d_done, 0) != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure, "CPHDReaderOp: H2D stream wait failed"});
  }
  if (cudaStreamSynchronize(h2d) != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure, "CPHDReaderOp: H2D sync failed"});
  }
  return {};
}

void check_sample_axis(const std::vector<double>& values, const char* name) {
  // Umbra SNAPSHOT PVPs have a small recorded-grid ramp within a batch. A
  // relative 1e-5 bound admits that residual while rejecting a frequency step.
  for (size_t i = 0; i < values.size(); ++i) {
    const double scale = std::max(std::abs(values[0]), std::abs(values[i]));
    if (!std::isfinite(values[i]) || std::abs(values[i] - values[0]) > 1e-5 * scale) {
      throw std::runtime_error(std::string(kWho) + ": " + name + " varies across a pulse batch (" +
                               std::to_string(values[0]) + " vs " + std::to_string(values[i]) +
                               ")");
    }
  }
}
}  // namespace

template <typename T>
struct CPHDReaderOp<T>::Batch {
  int64_t first{}, count{}, samples{};
  bool final{}, follows{}, cf8{};
  double collect_span{};
  std::vector<T> apc, reference, axis, vel;
  std::vector<double> times, amp;
  TimePoint capture_time{};
  // Optional outputs are attempted once; required outputs retire independently.
  unsigned published{};
  bool retry_warned{};
};

template <typename T>
CPHDReaderOp<T>::CPHDReaderOp(const MaxTensorExtents& extents, std::shared_ptr<FormationHud> hud)
: extents_(extents), hud_(std::move(hud)) {
  require_extent(extents.max_pulses, "max_pulses", kWho);
  require_extent(extents.max_samples_per_pulse, "max_samples_per_pulse", kWho);
}

template <typename T>
CPHDReaderOp<T>::~CPHDReaderOp() {
  stop();
  release_resources();
}

template <typename T>
LifecycleStatus CPHDReaderOp<T>::on_arm(LifecycleContext& context) noexcept {
  auto sender = context.notification_sender(wake_);
  if (!sender)
    return LifecycleStatus::kFatalFailure;
  scheduler_.bind(std::move(*sender));
  return LifecycleStatus::kOk;
}

template <typename T>
void CPHDReaderOp<T>::stop() {
  scheduler_.stop();
}

template <typename T>
LifecycleStatus CPHDReaderOp<T>::on_release(LifecycleContext&) noexcept {
  scheduler_.release();
  batch_.reset();
  file_.reset();
  release_resources();
  return LifecycleStatus::kOk;
}

template <typename T>
void CPHDReaderOp<T>::release_resources() noexcept {
  if (h2d_done_ != 0) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(h2d_done_));
    h2d_done_ = 0;
  }
  if (h2d_stream_ != 0) {
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(h2d_stream_));
    h2d_stream_ = 0;
  }
  if (pinned_signal_ != 0) {
    cudaFreeHost(reinterpret_cast<void*>(pinned_signal_));
    pinned_signal_ = 0;
  }
  if (device_cf8_ != 0) {
    cudaFree(reinterpret_cast<void*>(device_cf8_));
    device_cf8_ = 0;
  }
  if (device_amp_ != 0) {
    cudaFree(reinterpret_cast<void*>(device_amp_));
    device_amp_ = 0;
  }
}

template <typename T>
void CPHDReaderOp<T>::setup(OperatorSpec& spec) {
  spec.notification_source(wake_, "playback").capacity(2U).sender_reference_capacity(1U);
  spec.lifecycle()
      .stage(LifecycleStage::kArm, &CPHDReaderOp::on_arm)
      .stage(LifecycleStage::kRelease, &CPHDReaderOp::on_release);
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
  result.trigger(OnNotified{.event = wake_});
  return result;
}

template <typename T>
void CPHDReaderOp<T>::publish_hud() {
  if (hud_)
    hud_->publish_reader(spans_, playback_, last_apc_, last_packet_at_, stopped_, capture_time_ns_);
}

template <typename T>
void CPHDReaderOp<T>::maybe_warn_behind() {
  if (!playback_.paced || !playback_.valid || !std::isfinite(playback_.behind_s))
    return;
  if (playback_.behind_s >= kBehindWarnEnterS) {
    behind_episode_ = true;
  } else if (playback_.behind_s <= kBehindWarnExitS) {
    behind_episode_ = false;
    last_behind_warn_at_.reset();
    return;
  }
  if (!behind_episode_)
    return;
  const auto now = std::chrono::steady_clock::now();
  if (last_behind_warn_at_.has_value() && now - *last_behind_warn_at_ < kBehindWarnPeriod)
    return;
  last_behind_warn_at_ = now;
  HOLOSCAN_LOG_WARN(
      "CPHDReaderOp: playback +{:.1f}s behind (target={:.3f}s wall={:.3f}s rate={:.3g})",
      playback_.behind_s, playback_.target_s(), playback_.wall_s, playback_.rate);
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
  behind_episode_ = false;
  last_behind_warn_at_.reset();
  collection_t_min_ = 0.0;
  collection_t_max_ = 0.0;
  collection_time_valid_ = false;
}

template <typename T>
void CPHDReaderOp<T>::start() {
  stopped_ = false;
  capture_time_ns_.reset();
  stage_ = Stage::kRead;
  batch_.reset();
  spans_.clear();
  pending_cooldown_s_ = 0.0;
  axis_out_ = {};
  times_out_ = {};
  velocities_out_ = {};
  frame_out_ = {};
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
  cudaStream_t stream{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    throw std::runtime_error("CPHDReaderOp: failed to create H2D stream");
  }
  h2d_stream_ = reinterpret_cast<std::uintptr_t>(stream);
  cudaEvent_t done{};
  if (cudaEventCreateWithFlags(&done, cudaEventDisableTiming) != cudaSuccess) {
    throw std::runtime_error("CPHDReaderOp: failed to create H2D event");
  }
  h2d_done_ = reinterpret_cast<std::uintptr_t>(done);
  // Size to one tick, not max_pulses: the shared SAR extents also cover the
  // retained aperture, which must not inflate pinned ingest to gigabytes.
  const auto pulses = static_cast<std::size_t>(p.pulses_per_tick);
  const auto samples = static_cast<std::size_t>(*extents_.max_samples_per_pulse);
  pinned_signal_bytes_ = pulses * samples * sizeof(std::complex<T>);
  void* pinned = nullptr;
  if (cudaMallocHost(&pinned, pinned_signal_bytes_) != cudaSuccess) {
    throw std::runtime_error("CPHDReaderOp: failed to allocate pinned ingest buffer");
  }
  pinned_signal_ = reinterpret_cast<std::uintptr_t>(pinned);
  if constexpr (std::is_same_v<T, double>) {
    void* scratch = nullptr;
    if (cudaMalloc(&scratch, pulses * samples * 8U) != cudaSuccess) {
      throw std::runtime_error("CPHDReaderOp: failed to allocate device CF8 scratch");
    }
    device_cf8_ = reinterpret_cast<std::uintptr_t>(scratch);
  }
  void* amp = nullptr;
  if (cudaMalloc(&amp, pulses * sizeof(double)) != cudaSuccess) {
    throw std::runtime_error("CPHDReaderOp: failed to allocate device AmpSF buffer");
  }
  device_amp_ = reinterpret_cast<std::uintptr_t>(amp);
  scheduler_.start();
  scheduler_.schedule(std::chrono::steady_clock::now());
}

template <typename T>
expected<void, Error> CPHDReaderOp<T>::compute(ExecutionContext& context) {
  const CallbackTimer timing;
  if (context.cancellation().stop_requested()) {
    context.request_stop();
    return {};
  }
  if (stopped_)
    return {};
  try {
    const auto& p = params();
    const bool wait = cphd_reader_waits_on_full(p);
    if (stage_ == Stage::kPacing) {
      playback_.wall_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_wall_start_)
              .count();
      playback_.behind_s = playback_.wall_s - playback_.target_s();
      playback_.last_slack_s = p.playback_rate > 0.0 ? -playback_.behind_s : 0.0;
      next_pulse_ += batch_->count;
      if (batch_->final) {
        spans_.back().dwell_end = std::chrono::steady_clock::now();
        if (!batch_->follows) {
          stopped_ = true;
          stage_ = Stage::kStopped;
          batch_.reset();
          publish_hud();
          return {};
        }
        pending_cooldown_s_ =
            cphd::quiet_seconds(batch_->collect_span, p.duty_cycle, p.playback_rate);
        file_index_ = (file_index_ + 1U) % files_.size();
        open_file();
      }
      batch_.reset();
      stage_ = Stage::kRead;
      if (pending_cooldown_s_ > 0.0) {
        spans_.back().quiet_start = std::chrono::steady_clock::now();
        stage_ = Stage::kQuiet;
        scheduler_.schedule(cphd::after_seconds(*spans_.back().quiet_start, pending_cooldown_s_));
        publish_hud();
        return {};
      }
    }
    if (stage_ == Stage::kQuiet) {
      spans_.back().quiet_end = std::chrono::steady_clock::now();
      pending_cooldown_s_ = 0.0;
      stage_ = Stage::kRead;
      publish_hud();
    }
    if (stage_ == Stage::kRead) {
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
      const auto times = pvp.field("TxTime", 1);
      // Umbra files can put NaN on a handful of TxTime/TxPos words (one surveyed
      // file had 32 of 4844). Front/back of the batch is not a span: a NaN on
      // vector 0 poisons collect time for the rest of the file, skips pace
      // sleep, and shrinks the Gantt dwell. Skip non-finite samples, matching
      // the HSDK 4 reader.
      for (double t : times) {
        if (!std::isfinite(t))
          continue;
        if (!collection_time_valid_) {
          collection_t_min_ = t;
          collection_t_max_ = t;
          collection_time_valid_ = true;
        } else {
          collection_t_min_ = std::min(collection_t_min_, t);
          collection_t_max_ = std::max(collection_t_max_, t);
        }
      }
      const double collect_s = collection_time_valid_ ? collection_t_max_ - collection_t_min_ : 0.0;
      const double collect_span = (collect_s > 0.0 && std::isfinite(collect_s)) ? collect_s : 0.0;
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
      std::vector<T> vel;
      if (pvp.has("TxVel") && pvp.has("RcvVel")) {
        const auto tx_vel = pvp.field("TxVel", 3);
        const auto rx_vel = pvp.field("RcvVel", 3);
        vel.resize(tx_vel.size());
        for (size_t i = 0; i < vel.size(); ++i) {
          vel[i] = static_cast<T>(0.5 * (tx_vel[i] + rx_vel[i]));
        }
      }

      char* const pinned = reinterpret_cast<char*>(pinned_signal_);
      if (pinned == nullptr) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "CPHDReaderOp: pinned ingest buffer is not armed"});
      }
      const bool cf8 = file_->sample_format() == cphd::SampleFormat::kCF8;
      std::vector<double> amp;
      if (cf8 && file_->has_amp_sf())
        amp = file_->amplitude_scale(&pvp, count);
      if (cf8) {
        file_->read_signal_bytes(p.channel, first, count, pinned, pinned_signal_bytes_);
      } else {
        file_->read_signal_into<T>(p.channel, first, count,
                                   reinterpret_cast<std::complex<T>*>(pinned), &pvp);
      }
      batch_ = std::make_unique<Batch>(Batch{.first = first,
                                             .count = count,
                                             .samples = info.num_samples,
                                             .final = final,
                                             .follows = follows,
                                             .cf8 = cf8,
                                             .collect_span = collect_span,
                                             .apc = std::move(apc),
                                             .reference = std::move(reference),
                                             .axis = std::move(axis),
                                             .vel = std::move(vel),
                                             .times = times,
                                             .amp = std::move(amp),
                                             .capture_time = context.activation_time()});
      capture_time_ns_ = batch_->capture_time.timestamp_ns;
      stage_ = Stage::kPublish;
    }
    auto& batch = *batch_;
    const auto count = batch.count;
    const auto samples = batch.samples;
    const bool final = batch.final;
    const double collect_span = batch.collect_span;
    // Publish collect as soon as TxTime is known so the HUD is live during
    // ingest rather than still showing the previous tick's collect.
    auto publish_playback = [&](bool live) {
      const auto now = std::chrono::steady_clock::now();
      const double wall_s = std::chrono::duration<double>(now - collection_wall_start_).count();
      const double slack_s = p.playback_rate > 0.0 && collect_span > 0.0
                                 ? collect_span / p.playback_rate - wall_s
                                 : 0.0;
      playback_ = {.valid = true,
                   .paced = p.playback_rate > 0.0,
                   .live = live,
                   .collect_s = collect_span,
                   .wall_s = wall_s,
                   .behind_s = p.playback_rate > 0.0 ? -slack_s : wall_s - collect_span,
                   .last_slack_s = slack_s,
                   .rate = p.playback_rate,
                   .wall_origin = collection_wall_start_};
      publish_hud();
      maybe_warn_behind();
    };
    publish_playback(!final);
    const EmitOptions options{.capture_time = batch.capture_time};
    const auto* pinned = reinterpret_cast<const char*>(pinned_signal_);
    const auto nsamp = static_cast<std::size_t>(count * samples);
    const auto h2d = reinterpret_cast<cudaStream_t>(h2d_stream_);
    const auto h2d_done = reinterpret_cast<cudaEvent_t>(h2d_done_);
    if (h2d == nullptr || h2d_done == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "CPHDReaderOp: H2D stream is not armed"});
    }

    const cudaStream_t compute = context.cuda_stream().get();
    // Each successful publication retires exactly once. A declined loan is
    // consumed by the SDK; retry uploads from retained host storage.
    const char* publishing_port = "sample_axis";
    auto publish = [&]() -> expected<void, Error> {
      auto required = [&](Output<Tensor>& port, unsigned bit,
                          auto&& prepare) -> expected<void, Error> {
        if (batch.published & bit)
          return {};
        publishing_port = port.name().data();
        HOLOSCAN_TRY(auto loan, prepare());
        HOLOSCAN_RETURN_IF_ERROR(sync_ingest(h2d, h2d_done, compute));
        last_packet_at_ = std::chrono::steady_clock::now();
        publish_hud();
        HOLOSCAN_RETURN_IF_ERROR(port.emit(std::move(loan), options));
        batch.published |= bit;
        return {};
      };
      // Optional outputs preserve their existing one-attempt, latched policy.
      if (!(batch.published & 1U)) {
        if (axis_out_.active()) {
          HOLOSCAN_TRY(auto loan,
                       upload(sample_axis, context, batch.axis, {samples}, real_dtype<T>(), h2d));
          HOLOSCAN_RETURN_IF_ERROR(sync_ingest(h2d, h2d_done, compute));
          axis_out_.emit(sample_axis, context, std::move(loan), "CPHDReaderOp.sample_axis",
                         options);
        }
        batch.published |= 1U;
      }
      publishing_port = "pulse_times";
      if (!(batch.published & 2U)) {
        if (times_out_.active()) {
          HOLOSCAN_TRY(auto loan, upload(pulse_times, context, batch.times, {count},
                                         real_dtype<double>(), h2d));
          HOLOSCAN_RETURN_IF_ERROR(sync_ingest(h2d, h2d_done, compute));
          times_out_.emit(pulse_times, context, std::move(loan), "CPHDReaderOp.pulse_times",
                          options);
        }
        batch.published |= 2U;
      }
      publishing_port = "velocities";
      if (!(batch.published & 4U)) {
        if (velocities_out_.active() && !batch.vel.empty()) {
          HOLOSCAN_TRY(auto loan,
                       upload(velocities, context, batch.vel, {count, 3}, real_dtype<T>(), h2d));
          HOLOSCAN_RETURN_IF_ERROR(sync_ingest(h2d, h2d_done, compute));
          velocities_out_.emit(velocities, context, std::move(loan), "CPHDReaderOp.velocities",
                               options);
        }
        batch.published |= 4U;
      }
      publishing_port = "coordinate_frame";
      if (!(batch.published & 8U)) {
        if (frame_out_.active()) {
          HOLOSCAN_TRY(auto loan, prepare_frame(coordinate_frame, p.coordinate_frame));
          frame_out_.emit_schema_tensor(coordinate_frame, context, std::move(loan),
                                        to_wire(p.coordinate_frame),
                                        "CPHDReaderOp.coordinate_frame", options);
        }
        batch.published |= 8U;
      }
      HOLOSCAN_RETURN_IF_ERROR(required(phase_history, 16U, [&] {
        return prepare_device_tensor(phase_history, context, std::vector<int64_t>{count, samples},
                                     complex_dtype<T>(), [&](void* dst, std::size_t bytes) {
                                       return fill_history<T>(
                                           dst, bytes, nsamp, samples, batch.cf8, pinned,
                                           reinterpret_cast<void*>(device_cf8_),
                                           reinterpret_cast<double*>(device_amp_), batch.amp, h2d);
                                     });
      }));
      HOLOSCAN_RETURN_IF_ERROR(required(positions, 32U, [&] {
        return upload(positions, context, batch.apc, {count, 3}, real_dtype<T>(), h2d);
      }));
      HOLOSCAN_RETURN_IF_ERROR(required(reference_position, 64U, [&] {
        return upload(reference_position, context, batch.reference, {count, 3}, real_dtype<T>(),
                      h2d);
      }));
      const auto boundary = final ? (batch.follows ? SarStreamBoundary::kCollectionBoundary
                                                   : SarStreamBoundary::kEndOfStream)
                                  : SarStreamBoundary::kNone;
      publishing_port = "stream_boundary";
      HOLOSCAN_RETURN_IF_ERROR(stream_boundary.emit(static_cast<uint8_t>(boundary), options));
      return {};
    };
    auto result = publish();
    if (!result) {
      if (wait && is_transient_pool_error(result.error().code)) {
        if (!batch.retry_warned) {
          HOLOSCAN_LOG_WARN("CPHDReaderOp.{}: waiting for tensor pool/queue ({})", publishing_port,
                            holoscan::error_code_name(result.error().code));
          batch.retry_warned = true;
        }
        scheduler_.schedule(std::chrono::steady_clock::now() + std::chrono::milliseconds{10});
        return {};
      }
      return emit_required_result(context, std::string("CPHDReaderOp.") + publishing_port, false,
                                  [&] {
                                    return result;
                                  });
    }
    last_packet_at_ = std::chrono::steady_clock::now();
    publish_playback(!final);
    HOLOSCAN_LOG_DEBUG("CPHDReaderOp: pulses {}-{} collect={:.3f}s wall={:.3f}s slack={:.3f}s",
                       batch.first, batch.first + count - 1, collect_span, playback_.wall_s,
                       playback_.last_slack_s);
    stage_ = Stage::kPacing;
    // The batch emits ASAP. Its accumulated radar span gates the NEXT read.
    scheduler_.schedule(p.playback_rate > 0.0 ? cphd::after_seconds(collection_wall_start_,
                                                                    collect_span / p.playback_rate)
                                              : std::chrono::steady_clock::now());
    publish_hud();
    return {};
  } catch (const std::exception& error) {
    return make_unexpected(Error{ErrorCode::kFailure, error.what()});
  }
}

template class CPHDReaderOp<float>;
template class CPHDReaderOp<double>;
}  // namespace holoscan::radar
