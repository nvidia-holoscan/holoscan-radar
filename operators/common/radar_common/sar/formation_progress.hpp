// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>

namespace holoscan::radar::sar_progress {

using Clock = std::chrono::steady_clock;

// Application diagnostics only: never tensor payloads or config descriptors.
struct Collection {
  uint64_t generation{0};
  int64_t file_index{0}, total{0}, received{0};
  double duration_s{0}, playback_rate{0};
  Clock::time_point started{};
};

struct Work {
  Collection collection;
  int64_t capture_ns{0}, begin{0}, end{0}, target{0};
  int64_t admissions{0}, gpu_pulses{0}, pixels{0};
  bool emits_image{false}, replay{false};
  Clock::time_point admitted_at{};
};

struct GpuState {
  std::optional<Work> completed, pending;
  double pixel_pulses_per_s{0};
  int64_t measured_pixels{0};
};

struct View {
  bool valid{false}, image_complete{false};
  int64_t file_index{0}, received{0}, formed{0}, total{0};
  std::optional<double> eta_s;
};

// This mailbox never holds a tensor, image, or blocking GPU operation. Lost
// diagnostic history suppresses an estimate rather than affecting the graph.
class Monitor {
 public:
  static constexpr std::size_t kHistory = 128;
  explicit Monitor(int64_t cadence, int64_t reader_batch)
  : cadence_(cadence), reader_batch_(reader_batch) {}

  void begin_packet(int64_t stamp, int64_t file, int64_t offset, int64_t count, int64_t total,
                    double duration, double rate, bool new_collection) {
    const std::lock_guard lock(mu_);
    if (new_collection || collections_.empty()) {
      if (collections_.size() == kHistory)
        collections_.pop_front();
      collections_.push_back({.generation = ++generation_,
                              .file_index = file,
                              .total = total,
                              .duration_s = duration,
                              .playback_rate = rate,
                              .started = Clock::now()});
    }
    if (packets_.size() == kHistory)
      packets_.pop_front();
    packets_.push_back({stamp, generation_, offset, count, 0});
  }

  void received(int64_t stamp) {
    const std::lock_guard lock(mu_);
    auto* packet = find_packet(stamp);
    if (packet) {
      if (auto* collection = find_collection(packet->generation))
        collection->received = std::max(collection->received, packet->offset + packet->count);
    }
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
    }
    packet->admitted += count;
    const int64_t end = packet->offset + packet->admitted;
    Work work{.collection = *collection,
              .capture_ns = stamp,
              .begin = begin_,
              .end = end,
              .target = next_target(begin_, count, collection->total),
              .admissions = count,
              .emits_image = emits_image,
              .replay = replay,
              .admitted_at = Clock::now()};
    if (emits_image)
      work.target = end;  // Includes an early seam or a partial final cadence.
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
  };
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
      return packet.stamp == stamp;
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
  std::deque<Collection> collections_;
  std::deque<Packet> packets_;
  std::deque<Work> work_;
  std::optional<Work> manager_work_;
  std::optional<Work> in_flight_;
  std::optional<int64_t> discarded_through_;
  std::function<GpuState()> probe_;
};

}  // namespace holoscan::radar::sar_progress
