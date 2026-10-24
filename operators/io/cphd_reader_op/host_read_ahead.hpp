// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include "io/cphd_reader_op/playback_scheduler.hpp"

namespace holoscan::radar::cphd {

// Exactly one read in progress or ready, in addition to the caller's current
// batch. The worker owns all I/O; the execution lane never waits for it.
// Errors occupy the next slot and become visible only when it is taken.
template <typename Batch>
class HostReadAhead {
 public:
  struct Result {
    std::unique_ptr<Batch> batch;
    std::exception_ptr error;
  };

  template <typename Read>
  void request(PlaybackScheduler& worker, Read read) {
    {
      const std::lock_guard lock(mutex_);
      if (outstanding_)
        throw std::logic_error("CPHDReaderOp: more than one read ahead requested");
      outstanding_ = true;
    }
    worker.submit([this, &worker, read = std::move(read)] {
      Result result;
      try {
        result.batch = read();
      } catch (...) {
        result.error = std::current_exception();
      }
      bool wake = false;
      {
        const std::lock_guard lock(mutex_);
        ready_.emplace(std::move(result));
        wake = waiting_;
        waiting_ = false;
      }
      if (wake)
        worker.schedule(PlaybackClock::now());
    });
  }

  std::optional<Result> take() {
    const std::lock_guard lock(mutex_);
    if (!ready_) {
      waiting_ = true;
      return std::nullopt;
    }
    auto result = std::move(ready_);
    ready_.reset();
    outstanding_ = false;
    return result;
  }

  // Call only after the worker has joined.
  void clear() {
    const std::lock_guard lock(mutex_);
    ready_.reset();
    outstanding_ = waiting_ = false;
  }

 private:
  std::mutex mutex_;
  std::optional<Result> ready_;
  bool outstanding_{false};
  bool waiting_{false};
};
}  // namespace holoscan::radar::cphd
