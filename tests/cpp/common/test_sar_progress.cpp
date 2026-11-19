// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#include "radar_common/sar/formation_progress.hpp"
#include "radar_common/sdk/sar_gpu_progress.hpp"

namespace {
namespace progress = holoscan::radar::sar_progress;

TEST(SarBatchHistory, PartialSlicesReceiptRetriesAndImageProofDoNotDuplicateTicks) {
  progress::Monitor monitor(0, 20);
  monitor.begin_packet(1, 0, 0, 20, 100, 10, 1, true);
  monitor.received(1);
  monitor.received(1);
  ASSERT_EQ(monitor.batch_history().size(), 1U);
  monitor.admit(1, 10, false, true);
  auto work = monitor.take_work(1);
  ASSERT_TRUE(work);
  monitor.processed(*work, progress::Clock::now());
  EXPECT_EQ(monitor.batch_history().size(), 1U);
  monitor.admit(1, 10, false, true);
  work = monitor.take_work(1);
  ASSERT_TRUE(work);
  monitor.processed(*work, progress::Clock::now());
  monitor.processed(*work, progress::Clock::now());
  const auto history = monitor.batch_history();
  ASSERT_EQ(history.size(), 2U);
  EXPECT_FALSE(history[0].processed);
  EXPECT_TRUE(history[1].processed);
  EXPECT_LE(history[0].at, history[1].at);
}

TEST(SarBatchHistory, ReplayCompletesRetainedBatchesAndLoopGenerationStaysSeparate) {
  progress::Monitor monitor(0, 10);
  monitor.begin_packet(1, 0, 0, 10, 20, 10, 1, true);
  monitor.received(1);
  monitor.admit(1, 10, false, false, true);
  monitor.begin_packet(2, 0, 10, 10, 20, 10, 1, false);
  monitor.received(2);
  monitor.admit(2, 10, true, true, true);
  auto work = monitor.take_work(2);
  ASSERT_TRUE(work);
  work->gpu_pulses = 20;
  monitor.begin_packet(3, 0, 0, 10, 20, 10, 1, true);
  monitor.received(3);
  monitor.admit(3, 10, false, false, true);
  monitor.processed(*work, progress::Clock::now());
  const auto history = monitor.batch_history();
  uint32_t received = 0, processed = 0;
  for (const auto& event : history)
    (event.processed ? processed : received) += event.count;
  EXPECT_EQ(received, 3U);
  EXPECT_EQ(processed, 2U);
}

TEST(SarBatchHistory, ReplayDoesNotClaimEvictedUnprocessedPulsesWereFormed) {
  progress::Monitor monitor(0, 10);
  for (int64_t i = 0; i < 3; ++i) {
    monitor.begin_packet(i, 0, 10 * i, 10, 30, 10, 1, i == 0);
    monitor.received(i);
    monitor.admit(i, 10, i == 2, i == 2, true);
  }
  auto work = monitor.take_work(2);
  ASSERT_TRUE(work);
  work->gpu_pulses = 20;  // The first ten pulses were evicted before any replay.
  monitor.processed(*work, progress::Clock::now());
  const auto events = monitor.batch_history();
  uint32_t received = 0, processed = 0;
  for (const auto& event : events)
    (event.processed ? processed : received) += event.count;
  EXPECT_EQ(received, 3U);
  EXPECT_EQ(processed, 2U);
}

TEST(SarBatchHistory, CompletionProbeCanRaceFinalReceiptPublicationWithoutLosingEvents) {
  progress::Monitor monitor(0, 20);
  monitor.begin_packet(1, 0, 0, 20, 20, 10, 1, true);
  monitor.admit(1, 20, true, true);
  const auto work = monitor.take_work(1);
  ASSERT_TRUE(work);
  monitor.set_probe([&monitor, work] {
    monitor.processed(*work, progress::Clock::now());
    return progress::GpuState{.completed = work};
  });
  EXPECT_TRUE(monitor.batch_history().empty());
  monitor.received(1);
  const auto events = monitor.batch_history();
  ASSERT_EQ(events.size(), 2U);
  EXPECT_FALSE(events[0].processed);
  EXPECT_TRUE(events[1].processed);
}

TEST(SarBatchHistory, ExpiredPacketMetadataSuppressesUnprovableCompletionTicks) {
  progress::Monitor monitor(0, 1);
  for (std::size_t i = 0; i <= progress::Monitor::kHistory; ++i) {
    monitor.begin_packet(static_cast<int64_t>(i), 0, static_cast<int64_t>(i), 1, 10000, 10, 1,
                         i == 0);
    monitor.received(static_cast<int64_t>(i));
    monitor.admit(static_cast<int64_t>(i), 1, false, i == progress::Monitor::kHistory, true);
  }
  auto work = monitor.take_work(progress::Monitor::kHistory);
  ASSERT_TRUE(work);
  work->gpu_pulses = progress::Monitor::kHistory + 1;
  monitor.processed(*work, progress::Clock::now());
  uint32_t processed = 0;
  for (const auto& event : monitor.batch_history())
    if (event.processed)
      processed += event.count;
  EXPECT_EQ(processed, progress::Monitor::kHistory);
}

TEST(SarBatchHistory, EventHistoryRemainsBoundedAcrossLongRuns) {
  progress::Monitor monitor(0, 1);
  const auto origin = progress::Clock::now();
  for (int64_t i = 0; i < 2 * progress::Monitor::kBatchEvents; ++i) {
    monitor.begin_packet(i, 0, i, 1, 10000, 10, 1, i == 0);
    monitor.received(i);
    monitor.admit(i, 1, false, true);
    const auto work = monitor.take_work(i);
    ASSERT_TRUE(work);
    // Advance the completion observer through distinct logical time bins.
    monitor.processed(*work,
                      origin + std::chrono::milliseconds(i * progress::Monitor::kBatchBinMs));
  }
  EXPECT_EQ(monitor.batch_history().size(), progress::Monitor::kBatchEvents);
}

TEST(SarBatchHistory, DenseEventsPreserveOlderActivityAndBatchCounts) {
  progress::Monitor monitor(0, 1);
  const auto earlier = progress::Clock::now() - std::chrono::seconds(10);
  for (int64_t i = 0; i < 2000; ++i) {
    monitor.begin_packet(i, 0, i, 1, 10000, 10, 1, i == 0);
    monitor.received(i);
    monitor.admit(i, 1, false, true);
    const auto work = monitor.take_work(i);
    ASSERT_TRUE(work);
    monitor.processed(*work, i == 0 ? earlier : progress::Clock::now());
  }
  uint32_t received = 0, processed = 0;
  bool preserved = false;
  for (const auto& event : monitor.batch_history()) {
    (event.processed ? processed : received) += event.count;
    preserved |= event.processed && event.at == earlier;
  }
  EXPECT_TRUE(preserved);
  EXPECT_EQ(received, 2000U);
  EXPECT_EQ(processed, 2000U);
}

TEST(SarBatchSpans, PendingSpanClosesOnlyAfterTheEntireSourceBatchCompletes) {
  progress::Monitor monitor(0, 20);
  const auto receipt = progress::Clock::now() - std::chrono::seconds(8);
  monitor.begin_packet(1, 0, 0, 20, 20, 10, 1, true);
  monitor.received(1, receipt);
  monitor.received(1);  // Publication retry must not move the receipt marker.
  auto spans = monitor.batch_spans();
  ASSERT_EQ(spans.size(), 1U);
  EXPECT_EQ(spans[0].received, receipt);
  EXPECT_FALSE(spans[0].processed);
  monitor.admit(1, 10, false, true);
  auto work = monitor.take_work(1);
  ASSERT_TRUE(work);
  monitor.processed(*work, receipt + std::chrono::seconds(2));
  ASSERT_EQ(monitor.batch_spans().size(), 1U);
  EXPECT_FALSE(monitor.batch_spans()[0].processed);
  monitor.admit(1, 10, true, true);
  work = monitor.take_work(1);
  ASSERT_TRUE(work);
  const auto completed = receipt + std::chrono::seconds(4);
  monitor.processed(*work, completed);
  spans = monitor.batch_spans();
  ASSERT_EQ(spans.size(), 1U);
  ASSERT_TRUE(spans[0].processed);
  EXPECT_EQ(spans[0].received, receipt);
  EXPECT_EQ(*spans[0].processed, completed);
}

TEST(SarBatchSpans, MatchedIntervalsPreserveIdleGapsAndDoNotJoinDifferentBatches) {
  progress::Monitor monitor(0, 10);
  const auto now = progress::Clock::now();
  for (int64_t i = 0; i < 2; ++i) {
    monitor.begin_packet(i, i, 0, 10, 10, 10, 1, true);
    monitor.received(i, now - std::chrono::seconds(12 - i * 6));
    monitor.admit(i, 10, true, true);
    const auto work = monitor.take_work(i);
    ASSERT_TRUE(work);
    monitor.processed(*work, now - std::chrono::seconds(10 - i * 6));
  }
  const auto spans = monitor.batch_spans();
  ASSERT_EQ(spans.size(), 2U);
  ASSERT_TRUE(spans[0].processed);
  ASSERT_TRUE(spans[1].processed);
  EXPECT_LT(*spans[0].processed, spans[1].received);
  EXPECT_GT(spans[1].received - *spans[0].processed, std::chrono::seconds(3));
}

TEST(SarBatchSpans, OverlappingPairsRemainSeparateAndNeverUseAnotherBatchEndpoint) {
  progress::Monitor monitor(0, 10);
  const auto now = progress::Clock::now();
  for (int64_t i = 0; i < 2; ++i) {
    monitor.begin_packet(i, i, 0, 10, 10, 10, 1, true);
    monitor.received(i, now - std::chrono::seconds(8 - i * 2));
    monitor.admit(i, 10, true, true);
    const auto work = monitor.take_work(i);
    ASSERT_TRUE(work);
    monitor.processed(*work, now - std::chrono::seconds(4 - i * 2));
  }
  const auto pairs = monitor.batch_spans();
  ASSERT_EQ(pairs.size(), 2U);
  for (int64_t i = 0; i < 2; ++i) {
    EXPECT_EQ(pairs[i].received, now - std::chrono::seconds(8 - i * 2));
    EXPECT_EQ(pairs[i].processed, now - std::chrono::seconds(4 - i * 2));
  }
}

TEST(SarBatchSpans, PassedUnprovenPacketsDoNotLookPendingForever) {
  progress::Monitor monitor(0, 10);
  for (int64_t i = 0; i < 3; ++i) {
    monitor.begin_packet(i, 0, 10 * i, 10, 30, 10, 1, i == 0);
    monitor.received(i);
    monitor.admit(i, 10, i == 2, i == 2, true);
  }
  auto work = monitor.take_work(2);
  ASSERT_TRUE(work);
  work->gpu_pulses = 20;  // Initial packet was evicted without being processed.
  monitor.processed(*work, progress::Clock::now());
  for (const auto& span : monitor.batch_spans())
    EXPECT_TRUE(span.processed) << "A discarded prefix is not still in flight";
  monitor.begin_packet(3, 0, 0, 10, 30, 10, 1, true);
  monitor.received(3);
  int pending = 0;
  for (const auto& span : monitor.batch_spans())
    pending += !span.processed;
  EXPECT_EQ(pending, 1) << "A new loop must not inherit the prior loop's completion";
}

TEST(SarBatchSpans, DenseLongIntervalsStayBoundedAndReceiptCompletionRaceIsNonnegative) {
  progress::Monitor monitor(0, 1);
  const auto origin = progress::Clock::now();
  for (int64_t i = 0; i < 2000; ++i) {
    monitor.begin_packet(i, 0, i, 1, 10000, 10, 1, i == 0);
    monitor.admit(i, 1, false, true);
    const auto work = monitor.take_work(i);
    ASSERT_TRUE(work);
    // Completion can race final source-publication bookkeeping.
    monitor.processed(*work, origin + std::chrono::milliseconds(i));
    monitor.received(i, origin + std::chrono::milliseconds(i + 1));
  }
  EXPECT_LE(monitor.batch_spans().size(),
            progress::Monitor::kBatchPairs + progress::Monitor::kHistory);
  for (const auto& span : monitor.batch_spans()) {
    ASSERT_TRUE(span.processed);
    EXPECT_GE(*span.processed, span.received);
  }
  monitor.begin_packet(2001, 1, 0, 1, 1, 10, 1, true);
  monitor.received(2001, origin);
  monitor.admit(2001, 1, true, true);
  const auto work = monitor.take_work(2001);
  ASSERT_TRUE(work);
  monitor.processed(*work, origin + std::chrono::hours(1));
  const auto spans = monitor.batch_spans();
  EXPECT_LE(spans.size(), progress::Monitor::kBatchPairs);
  const auto long_pair = std::find_if(spans.begin(), spans.end(), [&](const auto& span) {
    return span.received == origin && span.processed == origin + std::chrono::hours(1);
  });
  EXPECT_NE(long_pair, spans.end()) << "Clipping is a drawing concern; preserve exact endpoints";
}

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
  EXPECT_FALSE(view.giga_pixel_pulses);
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
  ASSERT_TRUE(view.giga_pixel_pulses);
  EXPECT_DOUBLE_EQ(*view.giga_pixel_pulses, 1000.0 / 1e9);

  gpu.pending = *work;
  gpu.pending->pixels = 40;
  EXPECT_FALSE(monitor.snapshot().eta_s) << "A changed grid needs a matching measurement";
  EXPECT_DOUBLE_EQ(*monitor.snapshot().giga_pixel_pulses, 4000.0 / 1e9);
  gpu.pending->replay = true;
  gpu.pending->gpu_pulses = 60;
  EXPECT_DOUBLE_EQ(*monitor.snapshot().giga_pixel_pulses, 2400.0 / 1e9)
      << "A replay processes the retained aperture, not just newly admitted pulses";
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

TEST(SarPulseProgress, CollectionElapsedAdvancesThenFreezesAtImageCompletion) {
  progress::Monitor monitor(0, 20);
  monitor.begin_packet(1, 0, 0, 20, 20, 19, 0.02, true);
  monitor.received(1);
  monitor.admit(1, 20, true, true);
  const auto work = monitor.take_work(1);
  ASSERT_TRUE(work);
  const auto start = work->collection.started;
  monitor.submitted();
  progress::GpuState gpu{.pending = work};
  monitor.set_probe([&gpu] {
    return gpu;
  });
  const auto live = monitor.snapshot(start + std::chrono::seconds{65});
  ASSERT_TRUE(live.elapsed_s);
  EXPECT_DOUBLE_EQ(*live.elapsed_s, 65);
  EXPECT_FALSE(live.image_complete);
  EXPECT_EQ(live.formed, 0);
  gpu.pending.reset();
  gpu.completed = work;
  gpu.completed_at = start + std::chrono::milliseconds{94700};
  const auto held = monitor.snapshot(start + std::chrono::seconds{150});
  ASSERT_TRUE(held.elapsed_s);
  EXPECT_TRUE(held.image_complete);
  EXPECT_DOUBLE_EQ(*held.elapsed_s, 94.7);
  // Repeated display polls never add hold time to the completed image.
  EXPECT_EQ(monitor.snapshot(start + std::chrono::hours{1}).elapsed_s, held.elapsed_s);
  gpu.completed_at.reset();
  EXPECT_FALSE(monitor.snapshot().elapsed_s) << "Missing completion timing must stay unknown";

  monitor.begin_packet(2, 0, 0, 20, 20, 19, 0.02, true);
  monitor.admit(2, 20, false, true);
  const auto next = monitor.take_work(2);
  ASSERT_TRUE(next);
  const auto loop = monitor.snapshot(next->collection.started + std::chrono::seconds{2});
  ASSERT_TRUE(loop.elapsed_s);
  EXPECT_DOUBLE_EQ(*loop.elapsed_s, 2);
  EXPECT_FALSE(loop.image_complete);
}

TEST(SarPulseProgress, PeriodicImagesKeepTheirCollectionStartRatherThanThePreviousEmit) {
  progress::Monitor monitor(20, 20);
  monitor.begin_packet(1, 0, 0, 20, 40, 39, 1, true);
  monitor.admit(1, 20, true, true);
  const auto first = monitor.take_work(1);
  ASSERT_TRUE(first);
  monitor.submitted();
  monitor.begin_packet(2, 0, 20, 20, 40, 39, 1, false);
  monitor.admit(2, 20, true, true);
  const auto next = monitor.take_work(2);
  ASSERT_TRUE(next);
  monitor.submitted();
  monitor.set_probe([next] {
    return progress::GpuState{.completed = next,
                              .completed_at = next->collection.started + std::chrono::seconds{30}};
  });
  const auto view = monitor.snapshot(first->collection.started + std::chrono::seconds{60});
  EXPECT_TRUE(view.image_complete);
  EXPECT_EQ(view.total, 20);
  ASSERT_TRUE(view.elapsed_s);
  EXPECT_DOUBLE_EQ(*view.elapsed_s, 30);
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

TEST(SarPulseProgress, ImageBudgetUsesExactTxTimeSeamsRegardlessOfPlaybackRate) {
  const std::array times{100., 101., 103., 106., 110., 115., 121., 128.};
  const std::array budgets{1., 5., 9., 13.};
  for (const double rate : {0., 0.2, 1., 2.}) {
    SCOPED_TRACE(rate);
    progress::Monitor monitor(2, 8);
    monitor.begin_packet(10, 0, 0, 8, 8, 28, rate, true, times);
    for (const double budget : budgets) {
      // Sliding/replayed work still advances only two new source pulses.
      monitor.admit(10, 2, true, true, true);
      const auto work = monitor.take_work(10);
      ASSERT_TRUE(work);
      ASSERT_TRUE(work->budget_s);
      EXPECT_DOUBLE_EQ(*work->budget_s, budget);
      EXPECT_EQ(work->admissions, 2);
      monitor.submitted();
    }
  }
}

TEST(SarPulseProgress, ImageBudgetSpansBatchesButNotPreviousImagesOrTheFullAperture) {
  progress::Monitor monitor(8, 4);
  const std::array first{10., 11., 12., 14.};
  const std::array second{16., 19., 23., 28.};
  const std::array final{34., 41.};
  monitor.begin_packet(1, 0, 0, 4, 10, 31, 0.2, true, first);
  monitor.admit(1, 4, false, true);
  ASSERT_TRUE(monitor.take_work(1));
  monitor.begin_packet(2, 0, 4, 4, 10, 31, 0.2, false, second);
  monitor.admit(2, 4, true, true);
  const auto image = monitor.take_work(2);
  ASSERT_TRUE(image);
  ASSERT_TRUE(image->budget_s);
  EXPECT_DOUBLE_EQ(*image->budget_s, 18);
  monitor.begin_packet(3, 0, 8, 2, 10, 31, 0.2, false, final);
  monitor.admit(3, 2, true, true, true);
  const auto partial = monitor.take_work(3);
  ASSERT_TRUE(partial);
  ASSERT_TRUE(partial->budget_s);
  EXPECT_DOUBLE_EQ(*partial->budget_s, 13);
}

TEST(SarPulseProgress, BudgetResetsAtLoopBoundariesAndStaysWithQueuedImages) {
  progress::Monitor monitor(2, 4);
  const std::array first{1000., 1001., 1003., 1006.};
  const std::array loop{100., 101., 103., 106.};
  monitor.begin_packet(1, 0, 0, 4, 4, 6, 1, true, first);
  monitor.admit(1, 2, true, true);
  monitor.admit(1, 2, true, true);
  monitor.begin_packet(2, 0, 0, 4, 4, 6, 0.2, true, loop);
  monitor.admit(2, 2, true, true);
  const auto old_first = monitor.take_work(1);
  const auto old_last = monitor.take_work(1);
  const auto new_first = monitor.take_work(2);
  ASSERT_TRUE(old_first && old_last && new_first);
  EXPECT_EQ(old_first->budget_s, std::optional{1.});
  EXPECT_EQ(old_last->budget_s, std::optional{5.});
  EXPECT_EQ(new_first->budget_s, std::optional{1.});
  EXPECT_NE(new_first->collection.generation, old_last->collection.generation);
}

TEST(SarPulseProgress, WholeCollectionBudgetUsesKnownDurationAndMissingTimesStayUnknown) {
  for (const double duration : {7., 0., -1., std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(duration);
    progress::Monitor monitor(0, 8);
    monitor.begin_packet(1, 0, 0, 8, 8, duration, 0.2, true);
    monitor.admit(1, 8, true, true);
    const auto work = monitor.take_work(1);
    ASSERT_TRUE(work);
    if (duration == 7.) {
      EXPECT_EQ(work->budget_s, std::optional{7.});
    } else {
      EXPECT_FALSE(work->budget_s);
    }
  }
  progress::Monitor monitor(2, 8);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::array times{4., 4., nan, nan, 6., 5., 9., 10.};
  monitor.begin_packet(1, 0, 0, 8, 8, 6, 1, true, times);
  for (int i = 0; i < 4; ++i) {
    monitor.admit(1, 2, true, true);
    const auto work = monitor.take_work(1);
    ASSERT_TRUE(work);
    if (i == 3) {
      EXPECT_EQ(work->budget_s, std::optional{5.});
    } else {
      EXPECT_FALSE(work->budget_s);
    }
  }
}

TEST(SarPulseProgress, OversizedTimingPacketsKeepEndpointsAndRecoverAfterUnknownSeams) {
  const auto count = static_cast<int64_t>(progress::Monitor::kPacketTimingLimit + 2);
  std::vector<double> times(count);
  for (int64_t i = 0; i < count; ++i)
    times[i] = static_cast<double>(i);
  progress::Monitor monitor(2, count);
  monitor.begin_packet(1, 0, 0, count, count + 2, count + 1, 1, true, times);
  monitor.admit(1, 2, true, true);
  const auto internal = monitor.take_work(1);
  ASSERT_TRUE(internal);
  EXPECT_FALSE(internal->budget_s);
  monitor.admit(1, count - 2, true, true);
  const auto endpoint = monitor.take_work(1);
  ASSERT_TRUE(endpoint);
  EXPECT_FALSE(endpoint->budget_s) << "Do not interpolate a missing previous seam";
  const std::array next{static_cast<double>(count), static_cast<double>(count + 1)};
  monitor.begin_packet(2, 0, count, 2, count + 2, count + 1, 1, false, next);
  monitor.admit(2, 2, true, true);
  const auto recovered = monitor.take_work(2);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered->budget_s, std::optional{2.});

  progress::Monitor whole(0, count);
  whole.begin_packet(1, 0, 0, count, count + 2, count + 1, 1, true, times);
  whole.admit(1, count, true, true);
  const auto known_endpoints = whole.take_work(1);
  ASSERT_TRUE(known_endpoints);
  EXPECT_EQ(known_endpoints->budget_s, std::optional{static_cast<double>(count - 1)});
}

TEST(SarPulseProgress, LostPacketHistoryDoesNotCombineBudgetsAcrossUnobservedImages) {
  progress::Monitor monitor(2, 2);
  const auto total = static_cast<int64_t>(progress::Monitor::kHistory + 2) * 2;
  for (int64_t i = 0; i < total / 2; ++i) {
    const std::array times{static_cast<double>(i * 2), static_cast<double>(i * 2 + 1)};
    monitor.begin_packet(i + 1, 0, i * 2, 2, total, total - 1, 1, i == 0, times);
  }
  monitor.admit(1, 2, true, true);  // Expired: its image seam cannot be recovered.
  EXPECT_FALSE(monitor.take_work(1));
  monitor.admit(3, 2, true, true);
  const auto unknown = monitor.take_work(3);
  ASSERT_TRUE(unknown);
  EXPECT_FALSE(unknown->budget_s);
  monitor.admit(4, 2, true, true);
  const auto recovered = monitor.take_work(4);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered->budget_s, std::optional{2.});
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
  std::vector<int64_t> observed;
  progress::GpuProgress tracker(
      [&observed](const progress::Work& work, progress::Clock::time_point) {
        observed.push_back(work.end);
      });
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
  const auto before_release = observed.size();
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
  const auto terminal_at = progress::Clock::now();
  tracker.completed_emit(terminal, terminal_at);
  const auto terminal_state = tracker.snapshot();
  const bool reusable = tracker.begin(stream).has_value();
  const auto destroyed = cudaStreamDestroy(stream);
  EXPECT_EQ(before_release, 0U);
  ASSERT_EQ(observed.size(), progress::GpuProgress::kSlots + 1);
  for (std::size_t i = 0; i < observed.size(); ++i)
    EXPECT_EQ(observed[i], static_cast<int64_t>(i + 1));
  EXPECT_TRUE(allocated);
  EXPECT_TRUE(full);
  EXPECT_FALSE(pending.completed);
  EXPECT_FALSE(pending.completed_at);
  EXPECT_TRUE(pending.pending);
  EXPECT_EQ(synced, cudaSuccess);
  ASSERT_TRUE(done.completed);
  EXPECT_TRUE(done.completed_at);
  EXPECT_EQ(done.completed->end, 16);
  EXPECT_FALSE(done.pending);
  ASSERT_TRUE(terminal_state.completed);
  EXPECT_EQ(terminal_state.completed_at, terminal_at);
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

TEST(SarBatchHistory, DistinctScrollSliceStampsRetainExactSourcePacketAndBudget) {
  progress::Monitor monitor(2, 8);
  const std::array times{0., 1., 2., 3., 4., 5., 6., 7.};
  monitor.begin_packet(100, 0, 0, 8, 8, 7, 0.2, true, times);
  monitor.received(100);
  monitor.split_packet(100, 4);
  for (int i = 0; i < 4; ++i) {
    monitor.admit(100 + i, 2, true, true);
    auto work = monitor.take_work(100 + i);
    ASSERT_TRUE(work);
    EXPECT_EQ(work->capture_ns, 100 + i);
    EXPECT_EQ(work->end, (i + 1) * 2);
    ASSERT_TRUE(work->budget_s);
    EXPECT_DOUBLE_EQ(*work->budget_s, i == 0 ? 1. : 2.);
    monitor.processed(*work, progress::Clock::now());
    if (i < 3)
      EXPECT_EQ(monitor.batch_history().size(), 1U);
  }
  const auto history = monitor.batch_history();
  ASSERT_EQ(history.size(), 2U);
  EXPECT_TRUE(history.back().processed);
  EXPECT_TRUE(monitor.batch_spans().front().processed);
  // Registration is explicit: the next stamp must not silently inherit this packet.
  monitor.admit(104, 2, true, true);
  EXPECT_FALSE(monitor.take_work(104));
}
