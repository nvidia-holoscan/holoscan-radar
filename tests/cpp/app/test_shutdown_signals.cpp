// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <signal.h>

#include <atomic>
#include <stdexcept>
#include <thread>

#include "radar_app/app/shutdown_signals.hpp"

namespace {
using holoscan::radar::app::ShutdownSignals;

TEST(ShutdownSignals, AcceptsSignalsOnThreadsThatPredateInstallation) {
  for (int value : {SIGINT, SIGTERM}) {
    std::atomic<bool> deliver{false};
    std::thread participant([&] {
      while (!deliver.load())
        std::this_thread::yield();
      raise(value);
    });
    {
      ShutdownSignals signals;
      EXPECT_FALSE(signals.requested());
      deliver.store(true);
      participant.join();
      EXPECT_TRUE(signals.requested());
    }
  }
}

TEST(ShutdownSignals, RestoresPriorDispositionAndLeavesMaskUnchanged) {
  struct sigaction before_int {
  }, before_term{}, after_int{}, after_term{};
  sigset_t before_mask, after_mask;
  ASSERT_EQ(sigaction(SIGINT, nullptr, &before_int), 0);
  ASSERT_EQ(sigaction(SIGTERM, nullptr, &before_term), 0);
  ASSERT_EQ(sigprocmask(SIG_SETMASK, nullptr, &before_mask), 0);
  {
    ShutdownSignals signals;
    EXPECT_THROW(ShutdownSignals duplicate, std::runtime_error);
  }
  ASSERT_EQ(sigaction(SIGINT, nullptr, &after_int), 0);
  ASSERT_EQ(sigaction(SIGTERM, nullptr, &after_term), 0);
  EXPECT_EQ(before_int.sa_handler, after_int.sa_handler);
  EXPECT_EQ(before_term.sa_handler, after_term.sa_handler);
  ASSERT_EQ(sigprocmask(SIG_SETMASK, nullptr, &after_mask), 0);
  for (int value : {SIGINT, SIGTERM})
    EXPECT_EQ(sigismember(&before_mask, value), sigismember(&after_mask, value));
  ShutdownSignals fresh;
  EXPECT_FALSE(fresh.requested());
}
}  // namespace
