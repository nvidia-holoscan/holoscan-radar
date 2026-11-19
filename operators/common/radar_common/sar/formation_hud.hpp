// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/sar/formation_progress.hpp"

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

// Collect time vs wall. `collect_s` is the unscaled TxTime span (rate 1).
// `target_s()` is that span at `playback_rate`; `behind_s` is wall minus
// target (falling behind when > 0). While `live`, snapshot() recomputes
// wall from now() so a 100 ms timeline poll during pace-sleep sees current
// slack, matching the HSDK 4 `playback_status()` poll of the live reader.
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

  [[nodiscard]] double target_s() const {
    if (paced && rate > 0.0)
      return collect_s / rate;
    return collect_s;
  }
};

struct LastMeanApc {
  bool valid{false};
  int64_t file_index{0};
  std::array<double, 3> ecef{};
};

struct FormationHudBp {
  int64_t capture_time_ns{0};
  bool closes_aperture{false};
  double compute_s{0.0};
  // Wall time from this collection's first batch to this image's GPU completion.
  // Frozen with the formed image; absent if diagnostic history was unavailable.
  std::optional<double> collection_elapsed_s;
  // Original (1x) TxTime interval advanced by this image, not paced wall time.
  std::optional<double> budget_s;
  // Frozen with the same image; zero means unpaced, absent means unknown.
  std::optional<double> budget_playback_rate;
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

enum class AutofocusHudOutcome {
  kDisabled,
  kPending,
  kAccepted,
  kDeclined,
  kUnchanged,
  kOracle,
  kUnavailable
};

struct AutofocusHudStatus {
  AutofocusHudOutcome outcome{AutofocusHudOutcome::kDisabled};
  int32_t iterations{0};
};

struct FormationHudSnapshot {
  std::vector<CollectionSpan> spans;
  PlaybackStatus playback{};
  PlaybackStatus last_batch_playback{};
  LastMeanApc apc{};
  std::optional<std::chrono::steady_clock::time_point> last_packet_at;
  std::optional<int64_t> reader_capture_time_ns;
  // True after CPHDReaderOp becomes idle on kEndOfStream. Collection-boundary
  // dwell_end is not EOS: independent_images still has files to read.
  bool reader_stopped{false};
  FormationHudBp bp{};
  AutofocusHudStatus autofocus{};
  // Last INFO heartbeat rate line from SarImageReportOp. The HUD draws this
  // instead of live-polling `bp` after the LUT: visualization is drop-oldest
  // on another partition, so a live poll can read a later BP emit than the
  // console just logged.
  std::string rate_line;
};

// The last reader batch may fan out into multiple formation ticks. Wall time
// alone cannot identify its terminal image. Wait for a closing BP tick at or
// after that batch's capture stamp, and for reporting to acknowledge that tick.
inline bool sar_eof_drained(const FormationHudSnapshot& snapshot, int64_t reported_capture_ns) {
  return snapshot.reader_stopped && snapshot.reader_capture_time_ns.has_value() &&
         snapshot.bp.closes_aperture &&
         snapshot.bp.capture_time_ns >= *snapshot.reader_capture_time_ns &&
         reported_capture_ns >= snapshot.bp.capture_time_ns;
}

class FormationHud {
 public:
  void enable_autofocus(bool oracle = false) {
    const std::lock_guard lock(mu_);
    autofocus_.outcome = oracle ? AutofocusHudOutcome::kOracle : AutofocusHudOutcome::kPending;
  }
  void publish_autofocus(std::optional<int32_t> termination, std::optional<int32_t> iterations,
                         std::optional<int64_t> capture_ns = {}) {
    const std::lock_guard lock(mu_);
    if (capture_ns && autofocus_collection_stamp_ && *capture_ns < *autofocus_collection_stamp_)
      return;
    autofocus_ = {AutofocusHudOutcome::kUnavailable, iterations.value_or(0)};
    if (!termination || !iterations || *iterations < 0)
      return;
    switch (static_cast<AutofocusTermination>(*termination)) {
      case AutofocusTermination::kRefusedSharpness:
      case AutofocusTermination::kRefusedScatterers:
      case AutofocusTermination::kRefusedMaxCorrection:
        autofocus_.outcome = AutofocusHudOutcome::kDeclined;
        break;
      case AutofocusTermination::kConverged:
      case AutofocusTermination::kBudgetExhausted:
      case AutofocusTermination::kStoppedScatterers:
        autofocus_.outcome =
            *iterations > 0 ? AutofocusHudOutcome::kAccepted : AutofocusHudOutcome::kUnchanged;
        break;
    }
  }

  void enable_progress(int64_t cadence, int64_t reader_batch) {
    const std::lock_guard lock(mu_);
    progress_ = std::make_shared<sar_progress::Monitor>(cadence, reader_batch);
  }
  [[nodiscard]] std::shared_ptr<sar_progress::Monitor> progress() const {
    const std::lock_guard lock(mu_);
    return progress_;
  }
  void publish_reader(std::vector<CollectionSpan> spans, PlaybackStatus playback, LastMeanApc apc,
                      std::optional<std::chrono::steady_clock::time_point> last_packet,
                      bool reader_stopped = false, std::optional<int64_t> capture_time_ns = {}) {
    const std::lock_guard lock(mu_);
    if (autofocus_.outcome != AutofocusHudOutcome::kDisabled && !spans.empty() &&
        (spans_.empty() || spans.back().dwell_start != spans_.back().dwell_start)) {
      autofocus_collection_stamp_ = capture_time_ns;
      if (autofocus_.outcome != AutofocusHudOutcome::kOracle)
        autofocus_ = {AutofocusHudOutcome::kPending, 0};
    }
    spans_ = std::move(spans);
    playback_ = playback;
    apc_ = apc;
    if (last_packet.has_value() && last_packet != last_packet_at_) {
      last_batch_playback_ = playback;
      last_batch_playback_.live = false;
    }
    last_packet_at_ = last_packet;
    reader_stopped_ = reader_stopped;
    reader_capture_time_ns_ = capture_time_ns;
  }

  void publish_bp(FormationHudBp bp) {
    const std::lock_guard lock(mu_);
    bp_ = bp;
  }

  void publish_rate_line(std::string line) {
    const std::lock_guard lock(mu_);
    rate_line_ = std::move(line);
  }

  [[nodiscard]] FormationHudSnapshot snapshot() const {
    const std::lock_guard lock(mu_);
    FormationHudSnapshot snap{.spans = spans_,
                              .playback = playback_,
                              .last_batch_playback = last_batch_playback_,
                              .apc = apc_,
                              .last_packet_at = last_packet_at_,
                              .reader_capture_time_ns = reader_capture_time_ns_,
                              .reader_stopped = reader_stopped_,
                              .bp = bp_,
                              .autofocus = autofocus_,
                              .rate_line = rate_line_};
    if (snap.playback.valid && snap.playback.live) {
      const auto now = std::chrono::steady_clock::now();
      snap.playback.wall_s =
          std::max(0.0, std::chrono::duration<double>(now - snap.playback.wall_origin).count());
      snap.playback.behind_s = snap.playback.wall_s - snap.playback.target_s();
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
  std::optional<int64_t> reader_capture_time_ns_;
  bool reader_stopped_{false};
  FormationHudBp bp_{};
  AutofocusHudStatus autofocus_{};
  std::optional<int64_t> autofocus_collection_stamp_;
  std::string rate_line_;
  std::shared_ptr<sar_progress::Monitor> progress_;
};

}  // namespace holoscan::radar
