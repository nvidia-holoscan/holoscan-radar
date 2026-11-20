// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <signal.h>

#include <atomic>
#include <stdexcept>

namespace holoscan::radar::app {

// Application-owned process signals: safe to install after profiler/framework
// threads exist. No signal masks are changed. The handler only stores a lock-free
// flag; the normal control thread requests cooperative graph shutdown.
class ShutdownSignals {
 public:
  ShutdownSignals() {
    if (owned_.test_and_set(std::memory_order_acquire))
      throw std::runtime_error("Radar: shutdown signal handlers already owned");
    pending_.store(0, std::memory_order_relaxed);
    struct sigaction action {};
    action.sa_handler = &handle;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    if (sigaction(SIGINT, &action, &prior_int_) != 0) {
      owned_.clear(std::memory_order_release);
      throw std::runtime_error("Radar: cannot install SIGINT handler");
    }
    if (sigaction(SIGTERM, &action, &prior_term_) != 0) {
      (void)sigaction(SIGINT, &prior_int_, nullptr);
      owned_.clear(std::memory_order_release);
      throw std::runtime_error("Radar: cannot install SIGTERM handler");
    }
  }

  ~ShutdownSignals() {
    restore(SIGTERM, prior_term_);
    restore(SIGINT, prior_int_);
    owned_.clear(std::memory_order_release);
  }
  ShutdownSignals(const ShutdownSignals&) = delete;
  ShutdownSignals& operator=(const ShutdownSignals&) = delete;

  [[nodiscard]] bool requested() const { return pending_.load(std::memory_order_relaxed) != 0; }

 private:
  static_assert(std::atomic<int>::is_always_lock_free);
  static void handle(int value) noexcept { pending_.store(value, std::memory_order_relaxed); }
  static void restore(int value, const struct sigaction& prior) noexcept {
    struct sigaction current {};
    // Do not overwrite a handler deliberately replaced by another owner.
    if (sigaction(value, nullptr, &current) == 0 && !(current.sa_flags & SA_SIGINFO) &&
        current.sa_handler == &handle)
      (void)sigaction(value, &prior, nullptr);
  }
  inline static std::atomic<int> pending_{0};
  inline static std::atomic_flag owned_ = ATOMIC_FLAG_INIT;
  struct sigaction prior_int_ {};
  struct sigaction prior_term_ {};
};

}  // namespace holoscan::radar::app
