// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared per-scenario run skeleton. Every perf_<operator> consumer runs a
// scenario the same way: decide a frame plan (with or without a mid-session
// live-override window), run `repeats` graph sessions collecting one
// TimingCollector each, bucket the per-frame latencies into cold / steady /
// override-spike samples, then emit one "setup" row and one "compute" row.
//
// Only two pieces are operator-specific and stay in the consumer:
//   1. building + running one graph session (the RunRepFn passed to
//      collect_scenario_samples), and
//   2. finishing the compute row (throughput model + notes) after
//      fill_compute_row() has filled the timing fields.
//
// Keeping the bucketing here means the subtle override-window logic (shared
// verbatim by CFAR and DetectionReduction) is single-sourced.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "utils/perf_app.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_stats.hpp"

namespace holoscan::radar::perf {

// Frame plan for one scenario: how many frames to run and, when the operator
// has a [live] override, which frames form the post-override spike window.
struct RunPlan {
  int total_frames{0};
  int64_t override_at{-1};       // -1 => no live-override window
  int override_spike_frames{0};  // frames measured as the override spike
  bool has_override() const { return override_at >= 0; }
};

// Build the plan. With an override the spike lands mid-window (after warmup +
// half the measured frames) and total is stretched to fit the spike; without
// one it is simply warmup + measured frames.
inline RunPlan make_run_plan(const RunSettings& rs, bool has_override, int spike_frames = 3) {
  RunPlan p;
  if (has_override) {
    p.override_at = rs.warmup_frames + std::max(2, rs.measured_frames / 2);
    p.override_spike_frames = spike_frames;
    p.total_frames = std::max(rs.warmup_frames + rs.measured_frames,
                              static_cast<int>(p.override_at) + spike_frames);
  } else {
    p.override_at = -1;
    p.override_spike_frames = 0;
    p.total_frames = rs.warmup_frames + rs.measured_frames;
  }
  return p;
}

// Per-frame samples pooled across all repeats of one scenario.
struct ScenarioSamples {
  std::vector<double> setup;   // ready-to-tick time per repeat
  std::vector<double> cold;    // first frame latency per repeat
  std::vector<double> steady;  // post-warmup, pre-override frames
  double override_ms_best{std::numeric_limits<double>::quiet_NaN()};
};

// Run `rs.repeats` graph sessions and bucket their frames. `run_rep(rep, plan)`
// must build and run one graph for `plan.total_frames` frames and return the
// populated TimingCollector. Frame 0 is the cold sample; post-warmup frames
// before override_at are steady; frames in [override_at, override_at + spike)
// contribute the peak override latency; later frames are ignored (the plan
// sizes total so there are few or none).
template <typename RunRepFn>
ScenarioSamples collect_scenario_samples(const RunSettings& rs, const RunPlan& plan,
                                         RunRepFn run_rep) {
  ScenarioSamples s;
  for (int rep = 0; rep < rs.repeats; ++rep) {
    TimingCollector collector = run_rep(rep, plan);
    s.setup.push_back(collector.setup_ms());
    const auto lat = collector.frame_latencies_ms();
    if (!lat.empty()) {
      s.cold.push_back(lat.front());
    }
    for (size_t i = static_cast<size_t>(rs.warmup_frames); i < lat.size(); ++i) {
      const int64_t frame = static_cast<int64_t>(i);
      if (!plan.has_override() || frame < plan.override_at) {
        s.steady.push_back(lat[i]);
      } else if (frame < plan.override_at + plan.override_spike_frames) {
        const double v = lat[i];
        if (std::isnan(s.override_ms_best) || v > s.override_ms_best) {
          s.override_ms_best = v;
        }
      }
    }
  }
  return s;
}

// Append the standard "setup" row (median ready-to-tick time across repeats).
// The caller supplies a base row (scenario metadata already filled).
inline void append_setup_row(ResultRow row, const ScenarioSamples& s, PerfReport* report) {
  row.bucket = "setup";
  row.setup_ms = summarize(s.setup).median_ms;
  report->results.push_back(std::move(row));
}

// Fill the timing fields of a "compute" row (cold, steady latency stats, and
// the live-override peak/delta when present). The caller finishes the row with
// the operator-specific throughput model and notes, then appends it.
inline void fill_compute_row(ResultRow* row, const ScenarioSamples& s) {
  row->bucket = "compute";
  row->cold_ms = summarize(s.cold).median_ms;
  row->latency = summarize(s.steady);
  row->override_ms = s.override_ms_best;
  if (!std::isnan(s.override_ms_best) && !std::isnan(row->latency.median_ms)) {
    row->override_delta_ms = s.override_ms_best - row->latency.median_ms;
  }
}

}  // namespace holoscan::radar::perf
