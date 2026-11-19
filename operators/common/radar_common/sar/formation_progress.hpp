// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace holoscan::radar::sar_progress {

using Clock = std::chrono::steady_clock;

// Application diagnostics only: never tensor payloads or config descriptors.
struct Collection {
  uint64_t generation{0};
  int64_t file_index{0}, total{0}, received{0};
  double duration_s{0}, playback_rate{0};
  Clock::time_point started{};
  std::optional<double> first_tx_s;
};

struct Work {
  Collection collection;
  int64_t capture_ns{0}, begin{0}, end{0}, target{0};
  int64_t admissions{0}, gpu_pulses{0}, pixels{0};
  bool emits_image{false}, replay{false};
  Clock::time_point admitted_at{};
  std::optional<double> budget_s;
};

struct GpuState {
  std::optional<Work> completed, pending;
  double pixel_pulses_per_s{0};
  int64_t measured_pixels{0};
  std::optional<Clock::time_point> completed_at;
};

struct BatchEvent {
  Clock::time_point at;
  bool processed;
  uint32_t count{1};
};

// A completed span retains one exact, proven receipt/completion pair.
// An absent end is a retained batch still awaiting observed GPU completion.
struct BatchSpan {
  Clock::time_point received;
  std::optional<Clock::time_point> processed;
};

struct View {
  bool valid{false}, image_complete{false};
  int64_t file_index{0}, received{0}, formed{0}, total{0};
  std::optional<double> eta_s;
  std::optional<double> elapsed_s;
  // Expected BP work for this image only; absent until its processing grid is known.
  std::optional<double> giga_pixel_pulses;
};

// This mailbox never holds a tensor, image, or blocking GPU operation. Lost
// diagnostic history suppresses an estimate rather than affecting the graph.
class Monitor {
 public:
  static constexpr std::size_t kHistory = 128;
  // Two lanes at 40 ms resolution retain at least the full 20 s plot even
  // when thousands of batches arrive each second. Counts are diagnostic only.
  static constexpr std::size_t kBatchEvents = 1024;
  static constexpr int64_t kBatchBinMs = 40;
  static constexpr std::size_t kBatchPairs = 512;
  // At most 4 MiB of pulse-time samples across the packet history. Oversized
  // packets keep their endpoints; unrecorded internal seams stay unknown.
  static constexpr std::size_t kPacketTimingLimit = 4096;
  explicit Monitor(int64_t cadence, int64_t reader_batch)
  : cadence_(cadence), reader_batch_(reader_batch) {}

  void begin_packet(int64_t stamp, int64_t file, int64_t offset, int64_t count, int64_t total,
                    double duration, double rate, bool new_collection,
                    std::span<const double> pulse_times = {}) {
    const std::lock_guard lock(mu_);
    if (new_collection || collections_.empty()) {
      if (collections_.size() == kHistory)
        collections_.pop_front();
      collections_.push_back(
          {.generation = ++generation_,
           .file_index = file,
           .total = total,
           .duration_s = duration,
           .playback_rate = rate,
           .started = Clock::now(),
           .first_tx_s =
               offset == 0 && count > 0 && pulse_times.size() == static_cast<std::size_t>(count)
                   ? finite_time(pulse_times.front())
                   : std::nullopt});
    }
    if (packets_.size() == kHistory)
      packets_.pop_front();
    Packet packet{stamp, generation_, offset, count, 0};
    packet.sliced_through = stamp;
    if (count > 0 && pulse_times.size() == static_cast<std::size_t>(count)) {
      packet.first_tx_s = finite_time(pulse_times.front());
      packet.last_tx_s = finite_time(pulse_times.back());
      if (pulse_times.size() <= kPacketTimingLimit)
        packet.times.assign(pulse_times.begin(), pulse_times.end());
    }
    packets_.push_back(std::move(packet));
  }

  // A scroll stager explicitly stamps its slices base, base+1, ... . Register
  // that exact interval before emitting any slice; never guess nearby stamps.
  // One endpoint per retained source packet keeps diagnostic storage bounded.
  void split_packet(int64_t stamp, int64_t slices) {
    const std::lock_guard lock(mu_);
    auto* packet = find_packet(stamp);
    if (packet && slices > 0 && stamp <= std::numeric_limits<int64_t>::max() - (slices - 1))
      packet->sliced_through = stamp + slices - 1;
  }

  void received(int64_t stamp, Clock::time_point at = Clock::now()) {
    const std::lock_guard lock(mu_);
    auto* packet = find_packet(stamp);
    if (packet) {
      if (!packet->received_at) {
        packet->received_at = at;
        record_batch(*packet->received_at, false);
        if (packet->processed_at) {
          record_batch(*packet->processed_at, true);
          record_span(*packet->received_at, *packet->processed_at);
        }
      }
      if (auto* collection = find_collection(packet->generation))
        collection->received = std::max(collection->received, packet->offset + packet->count);
    }
  }

  // Completion is reported only after an event query or the existing image
  // boundary synchronization proves GPU completion. A reader packet can be
  // split across several manager slices; it gets one processed tick only when
  // proven completed ranges cover all of that source packet.
  void processed(const Work& work, Clock::time_point at) {
    const std::lock_guard lock(mu_);
    const int64_t count = work.replay ? work.gpu_pulses : work.admissions;
    if (count <= 0)
      return;
    const int64_t begin = work.end - count;
    if (work.collection.generation > observed_generation_) {
      observed_generation_ = work.collection.generation;
      observed_end_ = work.end;
    } else if (work.collection.generation == observed_generation_) {
      observed_end_ = std::max(observed_end_, work.end);
    }
    for (auto& packet : packets_) {
      if (packet.generation != work.collection.generation || packet.processed_at)
        continue;
      const int64_t frontier = packet.offset + packet.formed;
      // A replay may have evicted an unprocessed prefix. Only contiguous
      // proven coverage advances this packet; a later frontier alone cannot
      // establish that discarded pulses ever went through backprojection.
      if (begin > frontier || work.end <= frontier)
        continue;
      packet.formed = std::min(packet.count, work.end - packet.offset);
      if (packet.formed == packet.count && packet.admitted == packet.count) {
        packet.processed_at = at;
        if (packet.received_at) {
          record_batch(at, true);
          record_span(*packet.received_at, at);
        }
      }
    }
  }

  // Poll at timeline cadence, independently of the slower ETA text. The probe
  // and its completion callback never execute while this mailbox is locked.
  std::vector<BatchEvent> batch_history() {
    std::function<GpuState()> probe;
    {
      const std::lock_guard lock(mu_);
      probe = probe_;
    }
    if (probe)
      (void)probe();
    const std::lock_guard lock(mu_);
    return {batch_events_.begin(), batch_events_.end()};
  }

  // batch_history() polls GPU proof first; this snapshot never queries CUDA.
  // Retain exact representative pairs, never match independently binned ticks.
  std::vector<BatchSpan> batch_spans() {
    const std::lock_guard lock(mu_);
    std::vector<BatchSpan> result;
    for (const auto& pair : completed_pairs_)
      if (pair)
        result.push_back(*pair);
    std::sort(result.begin(), result.end(), [](const BatchSpan& a, const BatchSpan& b) {
      return a.processed < b.processed;
    });
    for (const auto& packet : packets_) {
      if (!packet.received_at || packet.processed_at)
        continue;
      // Once observed ordered work has passed an unproven packet, its old
      // prefix may have been evicted or diagnostic proof may have expired.
      // Do not leave that unknown work looking pending forever.
      if (packet.generation < observed_generation_ ||
          (packet.generation == observed_generation_ &&
           packet.offset + packet.count <= observed_end_))
        continue;
      result.push_back({*packet.received_at, {}});
    }
    return result;
  }

  // Called once per manager input, before publishing its corresponding output.
  // Only admissions advance the frontier: signed evictions and replayed pulses
  // are GPU work, not additional acquired pulses. Slices may share one stamp.
  void admit(int64_t stamp, int64_t count, bool emits_image, bool has_output, bool replay = false) {
    const std::lock_guard lock(mu_);
    auto* packet = find_packet(stamp);
    if (!packet || count > packet->count - packet->admitted)
      return;
    const auto* collection = find_collection(packet->generation);
    if (!collection)
      return;
    if (manager_generation_ != packet->generation) {
      manager_generation_ = packet->generation;
      begin_ = 0;
      previous_emit_tx_s_ = collection->first_tx_s;
      budget_end_ = 0;
      budget_contiguous_ = true;
    }
    if (packet->offset + packet->admitted != budget_end_) {
      // Expired source history may have hidden an image boundary. Never give
      // the next image the combined budget of those unobserved updates.
      previous_emit_tx_s_.reset();
      budget_contiguous_ = false;
    }
    packet->admitted += count;
    const int64_t end = packet->offset + packet->admitted;
    budget_end_ = end;
    Work work{.collection = *collection,
              .capture_ns = stamp,
              .begin = begin_,
              .end = end,
              .target = next_target(begin_, count, collection->total),
              .admissions = count,
              .emits_image = emits_image,
              .replay = replay,
              .admitted_at = Clock::now()};
    if (emits_image) {
      work.target = end;  // Includes an early seam or a partial final cadence.
      const auto last_tx = packet_time(*packet, packet->admitted - 1);
      if (budget_contiguous_ && begin_ == 0 && end == collection->total &&
          std::isfinite(collection->duration_s) && collection->duration_s > 0) {
        work.budget_s = collection->duration_s;
      } else if (last_tx && previous_emit_tx_s_ && *last_tx > *previous_emit_tx_s_) {
        const double interval = *last_tx - *previous_emit_tx_s_;
        if (std::isfinite(interval))
          work.budget_s = interval;
      }
      previous_emit_tx_s_ = last_tx;
      budget_contiguous_ = true;
    }
    manager_work_ = work;
    if (has_output) {
      if (work_.size() == kHistory) {
        // Multiple slices can share a stamp. Never match a surviving late
        // slice to an earlier BP invocation after diagnostic overflow.
        discarded_through_ = stamp;
        work_.clear();
      }
      if (!discarded_through_ || stamp > *discarded_through_)
        work_.push_back(work);
    }
    if (emits_image)
      begin_ = end;
  }

  std::optional<Work> take_work(int64_t stamp) {
    const std::lock_guard lock(mu_);
    while (!work_.empty() && work_.front().capture_ns < stamp)
      work_.pop_front();
    if (work_.empty() || work_.front().capture_ns != stamp)
      return {};
    const auto result = work_.front();
    work_.pop_front();
    in_flight_ = result;
    return result;
  }

  void submitted() {
    const std::lock_guard lock(mu_);
    in_flight_.reset();
  }

  void set_probe(std::function<GpuState()> probe) {
    const std::lock_guard lock(mu_);
    probe_ = std::move(probe);
  }

  View snapshot(Clock::time_point now = Clock::now()) {
    std::function<GpuState()> probe;
    {
      const std::lock_guard lock(mu_);
      probe = probe_;
    }
    // Query outside the mailbox lock: the GPU tracker has its own short lock.
    const auto gpu = probe ? probe() : GpuState{};
    const std::lock_guard lock(mu_);
    std::optional<Work> active = gpu.pending;
    if (!active)
      active = in_flight_;
    if (!active && !work_.empty())
      active = work_.front();
    if (!active)
      active = manager_work_;
    if (!active && !collections_.empty())
      active = initial_work(collections_.back());
    if (!active)
      return {};
    // A completed emit advances to the next cadence, or to a newer collection.
    if (gpu.completed && same_image(*active, *gpu.completed) && gpu.completed->emits_image &&
        gpu.completed->end >= active->target) {
      if (!collections_.empty() && collections_.back().generation > active->collection.generation) {
        active = initial_work(collections_.back());
      } else if (active->target < active->collection.total) {
        active->begin = active->target;
        active->end = active->begin;
        active->target = next_target(active->begin, reader_batch_, active->collection.total);
        active->emits_image = false;
      }
    }
    const auto* collection = find_collection(active->collection.generation);
    if (!collection)
      return {};  // Never relabel an old image with the live reader's collection.
    View result{.valid = true,
                .file_index = collection->file_index,
                .total = std::max(int64_t{0}, active->target - active->begin)};
    result.received = std::clamp(collection->received - active->begin, int64_t{0}, result.total);
    if (gpu.completed && same_image(*active, *gpu.completed)) {
      result.formed = std::clamp(gpu.completed->end - active->begin, int64_t{0}, result.total);
      result.image_complete = gpu.completed->emits_image && result.formed == result.total;
    }
    // A downstream admission proves receipt even during the tiny publication
    // interval before the reader records its successful final port emission.
    result.received = std::max(result.received, result.formed);
    const Work* image_work = gpu.pending && same_image(*active, *gpu.pending) ? &*gpu.pending
                             : gpu.completed && same_image(*active, *gpu.completed)
                                 ? &*gpu.completed
                                 : nullptr;
    if (image_work && image_work->pixels > 0) {
      const auto pulses = image_work->replay ? image_work->gpu_pulses : result.total;
      if (pulses > 0)
        result.giga_pixel_pulses =
            static_cast<double>(image_work->pixels) * static_cast<double>(pulses) / 1e9;
    }
    // Collection elapsed includes acquisition/pacing and processing. At an
    // image boundary freeze at completion, never at a later display poll.
    const auto end = result.image_complete ? gpu.completed_at : std::optional{now};
    if (end)
      result.elapsed_s =
          std::max(0.0, std::chrono::duration<double>(*end - collection->started).count());
    if (result.image_complete || result.total <= 0)
      return result;
    const auto pixels =
        gpu.pending ? gpu.pending->pixels
        : gpu.completed && gpu.completed->collection.generation == active->collection.generation
            ? gpu.completed->pixels
            : 0;
    if (gpu.pixel_pulses_per_s <= 0 || pixels <= 0 || gpu.measured_pixels != pixels)
      return result;
    // Acquisition and computation overlap. Do not add both durations. Include
    // queued formation even if the reader has already reached the next collect.
    double ingest_s = 0;
    if (result.received < result.total) {
      if (collection->playback_rate > 0 && collection->duration_s > 0) {
        // The previous batch's final TxTime gates this batch; the first
        // batch is released ASAP. A cadence may split one reader batch.
        const auto batch = std::max(int64_t{1}, reader_batch_);
        const auto gate = std::max(int64_t{0}, ((active->target - 1) / batch) * batch - 1);
        const double target_s = collection->duration_s * static_cast<double>(gate) /
                                std::max(int64_t{1}, collection->total - 1) /
                                collection->playback_rate;
        ingest_s = std::max(
            0.0, target_s - std::chrono::duration<double>(now - collection->started).count());
      } else {
        // Observed wall rate covers unpaced I/O, scheduling, and downstream
        // stalls. It is deliberately unknown until some data has arrived.
        const double wall = std::chrono::duration<double>(now - collection->started).count();
        if (collection->received <= 0 || wall <= 0)
          return result;
        ingest_s = static_cast<double>(active->target - collection->received) * wall /
                   collection->received;
      }
    }
    const double formation_s =
        static_cast<double>(result.total - result.formed) * pixels / gpu.pixel_pulses_per_s;
    if (std::isfinite(formation_s) && std::isfinite(ingest_s))
      result.eta_s = std::max(1.0, std::max(ingest_s, formation_s));
    return result;
  }

 private:
  struct Packet {
    int64_t stamp;
    uint64_t generation;
    int64_t offset, count, admitted;
    std::optional<double> first_tx_s, last_tx_s;
    std::vector<double> times;
    std::optional<Clock::time_point> received_at, processed_at;
    int64_t formed{0};
    int64_t sliced_through{0};
  };
  static int64_t time_bin(Clock::time_point at) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(at.time_since_epoch()).count() /
           kBatchBinMs;
  }
  void record_span(Clock::time_point received, Clock::time_point processed) {
    // Completion can race source bookkeeping. Keep a nonnegative exact pair.
    const auto end = std::max(received, processed);
    const auto bin = time_bin(end);
    auto& pair = completed_pairs_[static_cast<uint64_t>(bin) % kBatchPairs];
    // Keep one actual representative per completion-time bin. Never merge
    // its endpoints with another batch or let old proof replace newer history.
    if (!pair || *pair->processed <= end)
      pair = BatchSpan{received, end};
  }
  void record_batch(Clock::time_point at, bool processed) {
    const auto bucket = time_bin(at);
    for (auto it = batch_events_.rbegin(); it != batch_events_.rend(); ++it) {
      if (it->processed == processed && time_bin(it->at) == bucket) {
        it->at = std::max(it->at, at);
        if (it->count != std::numeric_limits<uint32_t>::max())
          ++it->count;
        return;
      }
    }
    if (batch_events_.size() == kBatchEvents)
      batch_events_.pop_front();
    batch_events_.push_back({at, processed});
  }
  static std::optional<double> finite_time(double value) {
    return std::isfinite(value) ? std::optional{value} : std::nullopt;
  }
  static std::optional<double> packet_time(const Packet& packet, int64_t index) {
    if (index < 0 || index >= packet.count)
      return {};
    if (index == 0)
      return packet.first_tx_s;
    if (index == packet.count - 1)
      return packet.last_tx_s;
    if (static_cast<std::size_t>(index) < packet.times.size())
      return finite_time(packet.times[index]);
    return {};
  }
  static bool same_image(const Work& a, const Work& b) {
    return a.collection.generation == b.collection.generation && a.begin == b.begin;
  }
  int64_t next_target(int64_t begin, int64_t batch, int64_t total) const {
    if (cadence_ <= 0)
      return total;
    const auto step = std::max(int64_t{1}, batch);
    return begin + std::min(total - begin, ((cadence_ - 1) / step + 1) * step);
  }
  Work initial_work(const Collection& collection) const {
    return {.collection = collection, .target = next_target(0, reader_batch_, collection.total)};
  }
  Packet* find_packet(int64_t stamp) {
    const auto it = std::find_if(packets_.rbegin(), packets_.rend(), [stamp](const Packet& packet) {
      return packet.stamp <= stamp && stamp <= packet.sliced_through;
    });
    return it == packets_.rend() ? nullptr : &*it;
  }
  Collection* find_collection(uint64_t generation) {
    const auto it = std::find_if(collections_.begin(), collections_.end(),
                                 [generation](const Collection& collection) {
                                   return collection.generation == generation;
                                 });
    return it == collections_.end() ? nullptr : &*it;
  }
  std::mutex mu_;
  const int64_t cadence_, reader_batch_;
  uint64_t generation_{0}, manager_generation_{0};
  int64_t begin_{0};
  int64_t budget_end_{0};
  bool budget_contiguous_{true};
  std::optional<double> previous_emit_tx_s_;
  std::deque<Collection> collections_;
  std::deque<Packet> packets_;
  std::deque<BatchEvent> batch_events_;
  std::array<std::optional<BatchSpan>, kBatchPairs> completed_pairs_{};
  uint64_t observed_generation_{0};
  int64_t observed_end_{0};
  std::deque<Work> work_;
  std::optional<Work> manager_work_;
  std::optional<Work> in_flight_;
  std::optional<int64_t> discarded_through_;
  std::function<GpuState()> probe_;
};

}  // namespace holoscan::radar::sar_progress
