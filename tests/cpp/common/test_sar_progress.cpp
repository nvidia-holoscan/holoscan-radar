// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "radar_common/sar/formation_progress.hpp"
#include "radar_common/sdk/sar_gpu_progress.hpp"

namespace {
namespace progress = holoscan::radar::sar_progress;

TEST(SarPulseProgress, ReceiptDoesNotClaimGpuCompletionOrInventAnEstimate) {
  progress::Monitor monitor(0, 20);
  monitor.begin_packet(1, 0, 0, 20, 100, 10, 0.1, true);
  monitor.received(1);
  monitor.admit(1, 20, false, true);
  auto view = monitor.snapshot();
  EXPECT_TRUE(view.valid);
  EXPECT_EQ(view.received, 20);
  EXPECT_EQ(view.formed, 0);
  EXPECT_EQ(view.total, 100);
  EXPECT_FALSE(view.eta_s);
  const auto work = monitor.take_work(1);
  ASSERT_TRUE(work);
  EXPECT_EQ(work->admissions, 20);
}

TEST(SarPulseProgress, EtaOverlapsPacingAndFormationAndHonorsFirstBatchRelease) {
  progress::Monitor monitor(0, 20);
  monitor.begin_packet(1, 0, 0, 20, 100, 99, 0.1, true);
  monitor.received(1);
  monitor.admit(1, 20, false, true);
  auto work = monitor.take_work(1);
  ASSERT_TRUE(work);
  monitor.submitted();
  work->pixels = 10;
  progress::GpuState gpu{.completed = work, .pixel_pulses_per_s = 100, .measured_pixels = 10};
  monitor.set_probe([&gpu] {
    return gpu;
  });
  const auto view = monitor.snapshot(work->collection.started);
  ASSERT_TRUE(view.eta_s);
  // Pulse 79 gates the final batch (not pulse 99). Acquisition takes 790s;
  // remaining GPU work is 8s, which overlaps rather than adding another 8s.
  EXPECT_DOUBLE_EQ(*view.eta_s, 790);
  EXPECT_EQ(view.formed, 20);

  gpu.pending = *work;
  gpu.pending->pixels = 40;
  EXPECT_FALSE(monitor.snapshot().eta_s) << "A changed grid needs a matching measurement";
}

TEST(SarPulseProgress, UnpacedEtaIncludesObservedIngestRate) {
  progress::Monitor monitor(0, 20);
  monitor.begin_packet(1, 0, 0, 20, 100, 99, 0, true);
  monitor.received(1);
  monitor.admit(1, 20, false, true);
  auto work = monitor.take_work(1);
  ASSERT_TRUE(work);
  monitor.submitted();
  work->pixels = 10;
  monitor.set_probe([work] {
    return progress::GpuState{.completed = work, .pixel_pulses_per_s = 100, .measured_pixels = 10};
  });
  const auto view = monitor.snapshot(work->collection.started + std::chrono::seconds{2});
  ASSERT_TRUE(view.eta_s);
  EXPECT_DOUBLE_EQ(*view.eta_s, 8);
}

TEST(SarPulseProgress, SlicesWithOneStampAdvanceImagesWithoutDoubleCounting) {
  progress::Monitor monitor(2, 8);
  monitor.begin_packet(10, 0, 0, 8, 8, 7, 0, true);
  monitor.received(10);
  for (int slice = 0; slice < 4; ++slice)
    monitor.admit(10, 2, true, true);
  progress::GpuState gpu;
  monitor.set_probe([&gpu] {
    return gpu;
  });
  for (int slice = 0; slice < 4; ++slice) {
    const auto work = monitor.take_work(10);
    ASSERT_TRUE(work);
    EXPECT_EQ(work->begin, slice * 2);
    EXPECT_EQ(work->end, (slice + 1) * 2);
    gpu.pending = work;
    monitor.submitted();
    const auto view = monitor.snapshot();
    EXPECT_EQ(view.total, 2);
    EXPECT_EQ(view.received, 2);
    EXPECT_EQ(view.formed, 0);
    gpu.completed = work;
  }
  gpu.pending.reset();
  EXPECT_TRUE(monitor.snapshot().image_complete);
  EXPECT_FALSE(monitor.take_work(10));
}

TEST(SarPulseProgress, ReplayAdmissionsDoNotClaimFormationUntilTheReplayCompletes) {
  progress::Monitor monitor(0, 4);
  monitor.begin_packet(1, 0, 0, 4, 8, 7, 0, true);
  monitor.received(1);
  monitor.admit(1, 4, false, false, true);
  EXPECT_EQ(monitor.snapshot().formed, 0);
  EXPECT_FALSE(monitor.take_work(1));
  monitor.begin_packet(2, 0, 4, 4, 8, 7, 0, false);
  monitor.received(2);
  monitor.admit(2, 4, true, true, true);
  const auto work = monitor.take_work(2);
  ASSERT_TRUE(work);
  EXPECT_EQ(work->end, 8);
  EXPECT_EQ(work->admissions, 4);
  EXPECT_TRUE(work->replay);
  EXPECT_EQ(monitor.snapshot().formed, 0);
  monitor.set_probe([work] {
    return progress::GpuState{.completed = work};
  });
  monitor.submitted();
  EXPECT_EQ(monitor.snapshot().formed, 8);
}

TEST(SarPulseProgress, RepeatedFileNamesDoNotMergeLoopGenerationsOrRelabelQueuedWork) {
  progress::Monitor monitor(0, 8);
  monitor.begin_packet(1, 0, 0, 8, 8, 7, 1, true);
  monitor.received(1);
  monitor.admit(1, 8, true, true);
  const auto old = monitor.take_work(1);
  ASSERT_TRUE(old);
  monitor.submitted();
  progress::GpuState gpu{.pending = old};
  monitor.set_probe([&gpu] {
    return gpu;
  });
  monitor.begin_packet(2, 0, 0, 4, 20, 19, 1, true);
  monitor.received(2);
  EXPECT_EQ(monitor.snapshot().total, 8);
  EXPECT_EQ(monitor.snapshot().received, 8);
  gpu.completed = old;
  gpu.pending.reset();
  const auto next = monitor.snapshot();
  EXPECT_EQ(next.total, 20);
  EXPECT_EQ(next.received, 4);
  EXPECT_EQ(next.formed, 0);
  EXPECT_FALSE(next.image_complete);
}

TEST(SarPulseProgress, PartialFinalCadenceAndExpiredHistoryRemainBounded) {
  progress::Monitor monitor(6, 4);
  monitor.begin_packet(1, 0, 0, 4, 10, 9, 1, true);
  monitor.admit(1, 4, false, true);
  EXPECT_EQ(monitor.snapshot().total, 8);
  monitor.begin_packet(2, 0, 4, 4, 10, 9, 1, false);
  monitor.admit(2, 4, true, true);
  monitor.begin_packet(3, 0, 8, 2, 10, 9, 1, false);
  monitor.admit(3, 2, true, true);
  const auto first = monitor.take_work(1);
  ASSERT_TRUE(first);
  monitor.take_work(2);
  const auto final = monitor.take_work(3);
  ASSERT_TRUE(final);
  EXPECT_EQ(final->begin, 8);
  EXPECT_EQ(final->target, 10);
  for (std::size_t i = 4; i < progress::Monitor::kHistory + 20; ++i)
    monitor.begin_packet(static_cast<int64_t>(i), 0, 0, 1, 1, 0, 0, true);
  monitor.set_probe([first] {
    return progress::GpuState{.pending = first};
  });
  EXPECT_FALSE(monitor.snapshot().valid);
  monitor.admit(1, 1, true, true);
  EXPECT_FALSE(monitor.take_work(1));
}

TEST(SarGpuProgress, QueriesAreNonblockingAndFullRingNeverWaitsOrClaimsQueuedWork) {
  struct Gate {
    std::mutex mu;
    std::condition_variable changed;
    bool open{false};
  } gate;
  cudaStream_t stream{};
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  progress::GpuProgress tracker;
  ASSERT_EQ(cudaLaunchHostFunc(
                stream,
                [](void* raw) {
                  auto& state = *static_cast<Gate*>(raw);
                  std::unique_lock lock(state.mu);
                  state.changed.wait(lock, [&state] {
                    return state.open;
                  });
                },
                &gate),
            cudaSuccess);
  bool allocated = true;
  for (std::size_t i = 0; i < progress::GpuProgress::kSlots; ++i) {
    const auto slot = tracker.begin(stream);
    allocated = allocated && slot.has_value();
    if (slot)
      tracker.end(*slot, stream,
                  {.collection = {.generation = 1, .total = 16},
                   .end = static_cast<int64_t>(i + 1),
                   .target = 16,
                   .admissions = 1,
                   .gpu_pulses = 1,
                   .pixels = 64});
  }
  const auto pending = tracker.snapshot();
  const bool full = !tracker.begin(stream);
  // Release before assertions so a failure cannot strand the test's CUDA stream.
  {
    const std::lock_guard lock(gate.mu);
    gate.open = true;
  }
  gate.changed.notify_one();
  const auto synced = cudaStreamSynchronize(stream);
  const auto done = tracker.snapshot();
  auto terminal = done.completed.value_or(progress::Work{});
  terminal.end = terminal.target = 17;
  terminal.emits_image = true;
  tracker.completed_emit(terminal);
  const auto terminal_state = tracker.snapshot();
  const bool reusable = tracker.begin(stream).has_value();
  const auto destroyed = cudaStreamDestroy(stream);
  EXPECT_TRUE(allocated);
  EXPECT_TRUE(full);
  EXPECT_FALSE(pending.completed);
  EXPECT_TRUE(pending.pending);
  EXPECT_EQ(synced, cudaSuccess);
  ASSERT_TRUE(done.completed);
  EXPECT_EQ(done.completed->end, 16);
  EXPECT_FALSE(done.pending);
  ASSERT_TRUE(terminal_state.completed);
  EXPECT_EQ(terminal_state.completed->end, 17);
  EXPECT_TRUE(reusable);
  EXPECT_EQ(destroyed, cudaSuccess);
}

TEST(SarPulseProgress, OverflowDoesNotMatchLateSlicesToEarlierGpuInvocations) {
  progress::Monitor monitor(1, 130);
  monitor.begin_packet(1, 0, 0, 130, 131, 130, 0, true);
  for (int i = 0; i < 130; ++i)
    monitor.admit(1, 1, true, true);
  EXPECT_FALSE(monitor.take_work(1));
  monitor.begin_packet(2, 0, 130, 1, 131, 130, 0, false);
  monitor.admit(2, 1, true, true);
  const auto recovered = monitor.take_work(2);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered->begin, 130);
  EXPECT_EQ(recovered->end, 131);
}
}  // namespace
