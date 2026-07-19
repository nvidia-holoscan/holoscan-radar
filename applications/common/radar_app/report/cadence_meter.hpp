// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>

#include <holoscan/logger/logger.hpp>

// Shared wall-clock cadence probe for the FMCW demo apps. The PeriodicCondition
// source is supposed to emit once per `frame_period`; when the pipeline (or the
// synthetic source itself) cannot keep up, Holoscan delays the next source
// tick. This meter answers two questions:
//
//   1. Source cadence — is the inter-emit gap exceeding period (+ slack)?
//   2. End-to-end — does DetectionReportOp finish each frame before the next
//      period elapses (FIFO-matched to the corresponding source emit)?
//
// Wire the same shared_ptr into the ADC source (on_source_emit after emit) and
// DetectionReportOp (on_sink_done at end of compute), then call log_summary()
// after Application::run() returns.
//
// Slack: leave `slack_s < 0` (default) for auto slack =
// max(10 ms, 10% of frame_period). A few ms of greedy-scheduler / logging
// jitter past the period is normal and is not a keep-up failure; the previous
// 2 ms floor false-alarmed on healthy demos.
namespace holoscan::radar::app {

class CadenceMeter {
 public:
  double frame_period_s{0.1};
  // < 0 → auto: max(10 ms, 10% of frame_period_s). Set >= 0 to override.
  double slack_s{-1.0};
  // 0 → warn on overdue/late only; N → also INFO every N completed sink frames.
  int64_t log_every{0};

  double effective_slack_s() const {
    if (slack_s >= 0.0) {
      return slack_s;
    }
    return std::max(0.010, 0.10 * frame_period_s);
  }

  double budget_s() const { return frame_period_s + effective_slack_s(); }

  void on_source_emit() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    std::lock_guard<std::mutex> lock(mu_);
    ++source_ticks_;
    if (have_prev_emit_) {
      const double inter =
          std::chrono::duration<double>(now - prev_emit_).count();
      max_inter_emit_s_ = std::max(max_inter_emit_s_, inter);
      const double budget = budget_s();
      if (inter > budget) {
        ++overdue_emits_;
        HOLOSCAN_LOG_WARN(
            "cadence: source emit overdue: inter={:.1f} ms > "
            "period+slack={:.1f} ms (emit {})",
            inter * 1e3, budget * 1e3, source_ticks_);
      }
    }
    prev_emit_ = now;
    have_prev_emit_ = true;
    emit_times_.push_back(now);
  }

  void on_sink_done() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    std::lock_guard<std::mutex> lock(mu_);
    ++sink_ticks_;
    if (emit_times_.empty()) {
      HOLOSCAN_LOG_WARN(
          "cadence: sink tick {} with no matching source emit", sink_ticks_);
      return;
    }
    const auto t0 = emit_times_.front();
    emit_times_.pop_front();
    const double e2e = std::chrono::duration<double>(now - t0).count();
    sum_e2e_s_ += e2e;
    max_e2e_s_ = std::max(max_e2e_s_, e2e);
    const double budget = budget_s();
    if (e2e > budget) {
      ++late_frames_;
      HOLOSCAN_LOG_WARN(
          "cadence: frame late: e2e={:.1f} ms > period+slack={:.1f} ms "
          "(sink {})",
          e2e * 1e3, budget * 1e3, sink_ticks_);
    }
    if (log_every > 0 && (sink_ticks_ % log_every) == 0) {
      HOLOSCAN_LOG_INFO(
          "cadence: frame {} e2e={:.1f} ms (period={:.1f} ms)", sink_ticks_,
          e2e * 1e3, frame_period_s * 1e3);
    }
  }

  void log_summary() const {
    std::lock_guard<std::mutex> lock(mu_);
    const double mean_e2e =
        sink_ticks_ > 0 ? (sum_e2e_s_ / static_cast<double>(sink_ticks_))
                        : 0.0;
    HOLOSCAN_LOG_INFO(
        "cadence summary: source_emits={} sink_frames={} overdue_emits={} "
        "late_e2e={} max_inter={:.1f} ms max_e2e={:.1f} ms mean_e2e={:.1f} ms "
        "period={:.1f} ms slack={:.1f} ms",
        source_ticks_, sink_ticks_, overdue_emits_, late_frames_,
        max_inter_emit_s_ * 1e3, max_e2e_s_ * 1e3, mean_e2e * 1e3,
        frame_period_s * 1e3, effective_slack_s() * 1e3);
  }

  int64_t source_ticks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return source_ticks_;
  }
  int64_t sink_ticks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return sink_ticks_;
  }
  int64_t overdue_emits() const {
    std::lock_guard<std::mutex> lock(mu_);
    return overdue_emits_;
  }
  int64_t late_frames() const {
    std::lock_guard<std::mutex> lock(mu_);
    return late_frames_;
  }

 private:
  mutable std::mutex mu_;
  bool have_prev_emit_{false};
  std::chrono::steady_clock::time_point prev_emit_{};
  std::deque<std::chrono::steady_clock::time_point> emit_times_;

  int64_t source_ticks_{0};
  int64_t sink_ticks_{0};
  int64_t overdue_emits_{0};
  int64_t late_frames_{0};
  double max_inter_emit_s_{0};
  double max_e2e_s_{0};
  double sum_e2e_s_{0};
};

}  // namespace holoscan::radar::app
