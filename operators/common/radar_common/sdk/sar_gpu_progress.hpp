// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "radar_common/sar/formation_progress.hpp"

namespace holoscan::radar::sar_progress {

// A fixed, best-effort event ring. The operator records events on its stream;
// the timeline queries completion at its display cadence. No kernel changes, host tensor
// copies, synchronization, callbacks on the CUDA stream, or waiting for slots.
class GpuProgress {
 public:
  static constexpr std::size_t kSlots = 16;
  explicit GpuProgress(std::function<void(const Work&, Clock::time_point)> completed = {})
  : completed_(std::move(completed)) {
    if (cudaGetDevice(&device_) != cudaSuccess)
      throw std::runtime_error("SAR progress: cannot identify compute device");
    for (auto& slot : slots_) {
      if (cudaEventCreate(&slot.start) != cudaSuccess ||
          cudaEventCreate(&slot.stop) != cudaSuccess) {
        release();
        throw std::runtime_error("SAR progress: cannot allocate bounded event ring");
      }
    }
  }
  ~GpuProgress() { release(); }
  GpuProgress(const GpuProgress&) = delete;
  GpuProgress& operator=(const GpuProgress&) = delete;

  std::optional<std::size_t> begin(cudaStream_t stream) {
    const std::lock_guard lock(mu_);
    poll();
    if (!valid_ || count_ == kSlots)
      return {};
    const auto index = (head_ + count_) % kSlots;
    if (cudaEventRecord(slots_[index].start, stream) != cudaSuccess) {
      valid_ = false;
      return {};
    }
    return index;
  }
  void end(std::size_t index, cudaStream_t stream, Work work) {
    const std::lock_guard lock(mu_);
    if (cudaEventRecord(slots_[index].stop, stream) != cudaSuccess) {
      valid_ = false;
      return;
    }
    slots_[index].work = work;
    ++count_;
  }
  // The image boundary already synchronizes its stream for the product. Use
  // that existing proof of completion even if tracking slots were exhausted.
  void completed_emit(const Work& work, Clock::time_point at = Clock::now()) {
    const std::lock_guard lock(mu_);
    poll();
    state_.completed = work;
    state_.completed_at = at;
    if (completed_)
      completed_(work, at);
  }
  GpuState snapshot() {
    const std::lock_guard lock(mu_);
    int prior = device_;
    if (cudaGetDevice(&prior) != cudaSuccess || cudaSetDevice(device_) != cudaSuccess)
      return {};
    poll();
    if (prior != device_)
      (void)cudaSetDevice(prior);
    auto result = state_;
    result.pending = count_ ? std::optional<Work>{slots_[head_].work} : std::nullopt;
    if (!valid_)
      result.pixel_pulses_per_s = 0;
    return result;
  }

 private:
  struct Slot {
    cudaEvent_t start{nullptr}, stop{nullptr};
    Work work;
  };
  void poll() {
    while (valid_ && count_ > 0) {
      auto& slot = slots_[head_];
      const auto status = cudaEventQuery(slot.stop);
      if (status == cudaErrorNotReady)
        break;
      if (status != cudaSuccess) {
        valid_ = false;
        break;
      }
      float ms = 0;
      if (cudaEventElapsedTime(&ms, slot.start, slot.stop) != cudaSuccess) {
        valid_ = false;
        break;
      }
      const auto& work = slot.work;
      // A reform is one full replay: admissions on its closing input are not
      // the replay's pulse count. Measure effective unique-pulse throughput.
      const int64_t unique = work.replay ? work.end - work.begin : work.admissions;
      if (ms > 0 && unique > 0 && work.pixels > 0) {
        const double seconds =
            work.replay
                ? std::max(ms * 1e-3,
                           std::chrono::duration<double>(Clock::now() - work.admitted_at).count())
                : ms * 1e-3;
        const double rate = static_cast<double>(unique) * work.pixels / seconds;
        if (state_.measured_pixels != work.pixels || !state_.completed ||
            state_.completed->collection.generation != work.collection.generation) {
          state_.pixel_pulses_per_s = rate;
        } else {
          state_.pixel_pulses_per_s = 0.7 * state_.pixel_pulses_per_s + 0.3 * rate;
        }
        state_.measured_pixels = work.pixels;
      }
      state_.completed = work;
      state_.completed_at = Clock::now();
      if (completed_)
        completed_(work, *state_.completed_at);
      head_ = (head_ + 1) % kSlots;
      --count_;
    }
  }
  void release() {
    int prior = device_;
    (void)cudaGetDevice(&prior);
    (void)cudaSetDevice(device_);
    for (auto& slot : slots_) {
      if (slot.start)
        (void)cudaEventDestroy(slot.start);
      if (slot.stop)
        (void)cudaEventDestroy(slot.stop);
    }
    if (prior != device_)
      (void)cudaSetDevice(prior);
  }
  std::mutex mu_;
  std::array<Slot, kSlots> slots_{};
  std::size_t head_{0}, count_{0};
  int device_{0};
  bool valid_{true};
  GpuState state_;
  std::function<void(const Work&, Clock::time_point)> completed_;
};

}  // namespace holoscan::radar::sar_progress
