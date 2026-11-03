// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared per-scenario run skeleton. Every perf_<operator> consumer runs a
// scenario the same way: decide a frame plan (with or without a mid-session
// live-override window), run `repeats` graph sessions collecting one
// TimingCollector each, bucket the per-frame latencies into cold / steady /
// changed-frame samples, then emit one "setup" row and one "compute" row.
//
// The typed recipe runner handles each Graph session. An adapter supplies
// valid inputs, operator construction and port wiring, then optionally adds
// throughput units and notes after fill_compute_row() fills timing fields.
//
// Keeping the bucketing here makes steady and changed-frame accounting consistent.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils/perf_app.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_stats.hpp"

namespace holoscan::radar::perf {

// Frame plan for one scenario: how many frames to run and, when the operator
// has a [live] override, which one frame receives the changed value.
struct RunPlan {
  int total_frames{0};
  int64_t override_at{-1};  // -1 => no live override
  bool has_override() const { return override_at >= 0; }
};

// Build the plan. Measure the requested number of steady frames after warmup.
// An override run adds one changed frame and one Params-reversion frame.
inline RunPlan make_run_plan(const RunSettings& rs, bool has_override) {
  RunPlan p;
  if (has_override) {
    p.override_at = rs.warmup_frames + rs.measured_frames;
    p.total_frames = static_cast<int>(p.override_at) + 2;
  } else {
    p.override_at = -1;
    p.total_frames = rs.warmup_frames + rs.measured_frames;
  }
  return p;
}

// Per-frame samples pooled across independent runs of one scenario.
struct ScenarioSamples {
  std::vector<int64_t> raw_steady_run_offsets;
  std::vector<double> setup;   // ready-to-tick time per repeat
  std::vector<double> cold;    // first frame latency per repeat
  std::vector<double> steady;  // post-warmup, pre-override frames
  std::vector<double> override;
  std::vector<double> override_delta;
  std::vector<double> achieved_fps;
  std::vector<double> source_fps;
};

// Run `rs.repeats` graph sessions and bucket their frames. `run_rep(rep, plan)`
// must build and run one graph for `plan.total_frames` frames and return the
// populated TimingCollector. Frame 0 is the cold sample; post-warmup frames
// before override_at are steady; the changed frame is an override sample.
// The following frame exercises Params reversion and is excluded from stats.
template <typename RunRepFn>
ScenarioSamples collect_scenario_samples(const RunSettings& rs, const RunPlan& plan,
                                         RunRepFn run_rep, bool expect_prime = false) {
  ScenarioSamples s;
  for (int rep = 0; rep < rs.repeats; ++rep) {
    if (rs.raw_frame_times)
      s.raw_steady_run_offsets.push_back(static_cast<int64_t>(s.steady.size()));
    TimingCollector collector = run_rep(rep, plan);
    // Cached-plan scenarios prime the changed value on frame zero, then
    // emit it once more on the measured changed frame.
    const std::vector<int64_t> expected = expect_prime ? std::vector<int64_t>{0, plan.override_at}
                                                       : std::vector<int64_t>{plan.override_at};
    if (plan.has_override() && collector.live_emissions != expected)
      throw std::runtime_error("performance live emission count or frame mismatch");
    if (!plan.has_override() && !collector.live_emissions.empty())
      throw std::runtime_error("performance baseline emitted a live override");
    s.setup.push_back(collector.setup_ms());
    const auto lat = collector.frame_latencies_ms();
    if (lat.size() != static_cast<size_t>(plan.total_frames) ||
        collector.starts.size() != lat.size() || collector.ends.size() != lat.size()) {
      throw std::runtime_error("performance frame timing count mismatch");
    }
    s.cold.push_back(lat.front());
    const size_t first = static_cast<size_t>(rs.warmup_frames);
    const size_t last =
        plan.has_override() ? static_cast<size_t>(plan.override_at - 1) : lat.size() - 1;
    if (last > first) {
      const double elapsed_s = to_ms(collector.ends[last] - collector.ends[first]) / 1000.0;
      const double source_s = to_ms(collector.starts[last] - collector.starts[first]) / 1000.0;
      if (elapsed_s > 0.0)
        s.achieved_fps.push_back(static_cast<double>(last - first) / elapsed_s);
      if (source_s > 0.0)
        s.source_fps.push_back(static_cast<double>(last - first) / source_s);
    }
    std::vector<double> run_steady;
    for (size_t i = static_cast<size_t>(rs.warmup_frames); i < lat.size(); ++i) {
      const int64_t frame = static_cast<int64_t>(i);
      if (!plan.has_override() || frame < plan.override_at) {
        s.steady.push_back(lat[i]);
        run_steady.push_back(lat[i]);
      } else if (frame == plan.override_at) {
        const double v = lat[i];
        s.override.push_back(v);
        s.override_delta.push_back(v - summarize(run_steady).median_ms);
      }
    }
  }
  if (rs.raw_frame_times)
    s.raw_steady_run_offsets.push_back(static_cast<int64_t>(s.steady.size()));
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
// the changed-frame latency/delta when present). The caller finishes the row with
// the operator-specific throughput model and notes, then appends it.
inline void fill_compute_row(ResultRow* row, const ScenarioSamples& s) {
  row->bucket = "compute";
  row->cold_ms = summarize(s.cold).median_ms;
  row->latency = summarize(s.steady);
  if (!s.raw_steady_run_offsets.empty()) {
    row->raw_steady_ms = s.steady;
    row->raw_steady_run_offsets = s.raw_steady_run_offsets;
  }
  row->override_ms = summarize(s.override).median_ms;
  row->achieved_frames_per_second = summarize(s.achieved_fps).median_ms;
  row->source_frames_per_second = summarize(s.source_fps).median_ms;
  row->override_delta_ms = summarize(s.override_delta).median_ms;
}

// Emit the standard setup and compute buckets. Adapters only annotate the
// compute row with throughput units and operator-specific notes.
template <typename Decorate>
void append_scenario_rows(ResultRow row, const ScenarioSamples& samples, PerfReport* report,
                          Decorate&& decorate) {
  append_setup_row(row, samples, report);
  fill_compute_row(&row, samples);
  std::forward<Decorate>(decorate)(row);
  report->results.push_back(std::move(row));
}

// One adapter invocation: build the frame plan, run independent Graph sessions,
// and write both report buckets. Only the Graph callback and metadata vary.
template <typename RunRep, typename Decorate>
void measure_scenario(const RunSettings& settings, bool has_override, RunRep&& run_rep,
                      ResultRow row, PerfReport* report, Decorate&& decorate,
                      bool expect_prime = false) {
  const RunPlan plan = make_run_plan(settings, has_override);
  const auto samples =
      collect_scenario_samples(settings, plan, std::forward<RunRep>(run_rep), expect_prime);
  append_scenario_rows(std::move(row), samples, report, std::forward<Decorate>(decorate));
}

}  // namespace holoscan::radar::perf
