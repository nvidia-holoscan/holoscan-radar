// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace holoscan::radar {

// Live formation diagnostics for SAR HUD / Gantt. Graph.op() records the
// authoring prototype, not the running instance, so overlay code cannot poll
// operator members. Apps pass one shared mailbox into the reader and
// backprojector ctors; both the prototype and the live clone share it.
struct CollectionSpan {
  int file_index{0};
  std::chrono::steady_clock::time_point dwell_start{};
  std::optional<std::chrono::steady_clock::time_point> dwell_end;
  std::optional<std::chrono::steady_clock::time_point> quiet_start;
  std::optional<std::chrono::steady_clock::time_point> quiet_end;
};

// Collect time vs wall. `behind_s` > 0 means wall is ahead of radar time
// (falling behind). While `live`, snapshot() recomputes wall from now() so a
// 100 ms timeline poll during pace-sleep sees current slack, matching the
// HSDK 4 `playback_status()` poll of the live reader.
struct PlaybackStatus {
  bool valid{false};
  bool paced{false};
  bool live{false};
  double collect_s{0.0};
  double wall_s{0.0};
  double behind_s{0.0};
  double last_slack_s{0.0};
  double rate{0.0};
  std::chrono::steady_clock::time_point wall_origin{};
};

struct LastMeanApc {
  bool valid{false};
  int64_t file_index{0};
  std::array<double, 3> ecef{};
};

struct FormationHudBp {
  double compute_s{0.0};
  int64_t pulses{0};
  int64_t pixels{0};
  int64_t num_v{0};
  int64_t num_u{0};
  std::optional<std::chrono::steady_clock::time_point> emit_at;
  // Last-pulse packet of the formed aperture, latched at the start of the
  // emit tick. The live reader last_packet_at advances into the next collect
  // while this image is still on the GPU.
  std::optional<std::chrono::steady_clock::time_point> last_packet_at;
  // Collect/wall of that same last packet. Frozen (`live` false) so a report
  // at image receive does not pick up the next collection's playback.
  PlaybackStatus playback{};
};

struct FormationHudSnapshot {
  std::vector<CollectionSpan> spans;
  PlaybackStatus playback{};
  PlaybackStatus last_batch_playback{};
  LastMeanApc apc{};
  std::optional<std::chrono::steady_clock::time_point> last_packet_at;
  FormationHudBp bp{};
};

class FormationHud {
 public:
  void publish_reader(std::vector<CollectionSpan> spans, PlaybackStatus playback, LastMeanApc apc,
                      std::optional<std::chrono::steady_clock::time_point> last_packet) {
    const std::lock_guard lock(mu_);
    spans_ = std::move(spans);
    playback_ = playback;
    apc_ = apc;
    if (last_packet.has_value() && last_packet != last_packet_at_) {
      last_batch_playback_ = playback;
      last_batch_playback_.live = false;
    }
    last_packet_at_ = last_packet;
  }

  void publish_bp(FormationHudBp bp) {
    const std::lock_guard lock(mu_);
    bp_ = bp;
  }

  [[nodiscard]] FormationHudSnapshot snapshot() const {
    const std::lock_guard lock(mu_);
    FormationHudSnapshot snap{.spans = spans_,
                              .playback = playback_,
                              .last_batch_playback = last_batch_playback_,
                              .apc = apc_,
                              .last_packet_at = last_packet_at_,
                              .bp = bp_};
    if (snap.playback.valid && snap.playback.live) {
      const auto now = std::chrono::steady_clock::now();
      snap.playback.wall_s =
          std::max(0.0, std::chrono::duration<double>(now - snap.playback.wall_origin).count());
      if (snap.playback.paced && snap.playback.rate > 0.0) {
        snap.playback.behind_s =
            snap.playback.wall_s - snap.playback.collect_s / snap.playback.rate;
      } else {
        snap.playback.behind_s = snap.playback.wall_s - snap.playback.collect_s;
      }
    }
    return snap;
  }

 private:
  mutable std::mutex mu_;
  std::vector<CollectionSpan> spans_;
  PlaybackStatus playback_{};
  PlaybackStatus last_batch_playback_{};
  LastMeanApc apc_{};
  std::optional<std::chrono::steady_clock::time_point> last_packet_at_;
  FormationHudBp bp_{};
};

}  // namespace holoscan::radar
