// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan::radar::cphd {

using PlaybackClock = std::chrono::steady_clock;

inline PlaybackClock::time_point after_seconds(PlaybackClock::time_point origin, double seconds) {
  return origin + std::chrono::duration_cast<PlaybackClock::duration>(
                      std::chrono::duration<double>{seconds});
}

inline double quiet_seconds(double span, double duty, double rate) {
  const double quiet = span * (1.0 / duty - 1.0);
  return rate > 0.0 ? quiet / rate : quiet;
}

// One logical wake at a time. The SDK's two credits allow a successor wake
// while the current callback retires; the worker never retains callback data.
class PlaybackScheduler {
 public:
  ~PlaybackScheduler() { stop(); }

  void bind(NotificationSender sender) { sender_ = std::move(sender); }

  void start() {
    worker_ = std::jthread([this](std::stop_token token) {
      std::unique_lock lock(mutex_);
      while (!token.stop_requested()) {
        changed_.wait(lock, token, [this] {
          return deadline_.has_value();
        });
        if (token.stop_requested())
          break;
        const auto deadline = *deadline_;
        changed_.wait_until(lock, token, deadline, [this, deadline] {
          return !deadline_ || *deadline_ != deadline;
        });
        if (token.stop_requested())
          break;
        if (!deadline_ || *deadline_ != deadline)
          continue;
        deadline_.reset();
        lock.unlock();
        while (!token.stop_requested()) {
          const auto posted = sender_.post();
          if (posted)
            break;
          const auto code = posted.error().code;
          if (code == ErrorCode::kClosed || code == ErrorCode::kStaleEndpoint)
            return;
          if (code != ErrorCode::kNotReady && code != ErrorCode::kBackpressured) {
            HOLOSCAN_LOG_ERROR("CPHDReaderOp: notification post failed: {}",
                               holoscan::runtime_error_message(posted.error()));
            return;
          }
          lock.lock();
          changed_.wait_for(lock, token, std::chrono::milliseconds{1}, [] {
            return false;
          });
          lock.unlock();
        }
        lock.lock();
      }
    });
  }

  void schedule(PlaybackClock::time_point deadline) {
    const std::lock_guard lock(mutex_);
    deadline_ = deadline;
    changed_.notify_all();
  }

  void stop() {
    if (worker_.joinable()) {
      worker_.request_stop();
      changed_.notify_all();
      worker_.join();
    }
    const std::lock_guard lock(mutex_);
    deadline_.reset();
  }

  void release() { sender_ = {}; }

 private:
  NotificationSender sender_;
  std::mutex mutex_;
  std::condition_variable_any changed_;
  std::optional<PlaybackClock::time_point> deadline_;
  std::jthread worker_;
};
}  // namespace holoscan::radar::cphd
