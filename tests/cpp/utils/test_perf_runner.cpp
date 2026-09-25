// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <string>

#include "utils/perf_runner.hpp"

namespace {

namespace perf = holoscan::radar::perf;

TEST(PerfRunner, BucketsFullSteadyWindowAndOneChangedFramePerRun) {
  perf::RunSettings settings;
  settings.warmup_frames = 5;
  settings.measured_frames = 50;
  settings.repeats = 3;
  const auto plan = perf::make_run_plan(settings, true);
  EXPECT_EQ(plan.override_at, 55);
  EXPECT_EQ(plan.total_frames, 57);

  const auto samples =
      perf::collect_scenario_samples(settings, plan, [](int, const perf::RunPlan& current) {
        perf::TimingCollector collector;
        collector.t_construct = perf::Clock::time_point{};
        collector.t_first_tick = collector.t_construct + std::chrono::milliseconds{1};
        for (int frame = 0; frame < current.total_frames; ++frame) {
          const auto start = collector.t_construct + std::chrono::milliseconds{frame + 1};
          const auto latency = frame == current.override_at ? std::chrono::microseconds{200}
                                                            : std::chrono::microseconds{100};
          collector.starts.push_back(start);
          collector.ends.push_back(start + latency);
        }
        return collector;
      });
  EXPECT_EQ(samples.setup.size(), 3U);
  EXPECT_EQ(samples.cold.size(), 3U);
  EXPECT_EQ(samples.steady.size(), 150U);
  EXPECT_EQ(samples.override.size(), 3U);
  EXPECT_EQ(samples.achieved_fps.size(), 3U);
  EXPECT_EQ(samples.source_fps.size(), 3U);

  perf::ResultRow row;
  perf::fill_compute_row(&row, samples);
  EXPECT_NEAR(row.latency.median_ms, 0.1, 1e-9);
  EXPECT_NEAR(row.override_ms, 0.2, 1e-9);
  EXPECT_NEAR(row.override_delta_ms, 0.1, 1e-9);
  EXPECT_NEAR(row.achieved_frames_per_second, 1000.0, 1e-9);
  EXPECT_NEAR(row.source_frames_per_second, 1000.0, 1e-9);
  EXPECT_EQ(row.latency.samples, 150);
}

TEST(PerfRunner, PairsChangedFrameWithItsOwnRunBaseline) {
  perf::RunSettings settings;
  settings.warmup_frames = 0;
  settings.measured_frames = 3;
  settings.repeats = 3;
  const auto plan = perf::make_run_plan(settings, true);
  const auto samples =
      perf::collect_scenario_samples(settings, plan, [](int rep, const perf::RunPlan& current) {
        perf::TimingCollector collector;
        collector.t_construct = perf::Clock::time_point{};
        collector.t_first_tick = collector.t_construct;
        const int steady_us[] = {1000, 10000, 100000};
        const int changed_us[] = {2000, 110000, 101000};
        for (int frame = 0; frame < current.total_frames; ++frame) {
          const auto start = collector.t_construct + std::chrono::seconds{frame + 1};
          const auto duration = std::chrono::microseconds{
              frame == current.override_at ? changed_us[rep] : steady_us[rep]};
          collector.starts.push_back(start);
          collector.ends.push_back(start + duration);
        }
        return collector;
      });
  perf::ResultRow row;
  perf::fill_compute_row(&row, samples);
  EXPECT_NEAR(row.override_delta_ms, 1.0, 1e-9);
  EXPECT_NEAR(row.override_ms, 101.0, 1e-9);
  EXPECT_NEAR(row.latency.median_ms, 10.0, 1e-9);
}

TEST(PerfRunner, SerializesScopeAndMeasuredRate) {
  perf::PerfReport report;
  report.run_settings.scope = "overrides";
  perf::ResultRow row;
  row.scenario_id = "one-live-frame";
  row.bucket = "compute";
  row.live_port = "method";
  row.achieved_frames_per_second = 1000.0;
  row.source_frames_per_second = 1100.0;
  report.results.push_back(row);
  std::ostringstream stream;
  perf::write_json(stream, report);
  const std::string json = stream.str();
  EXPECT_NE(json.find(R"("scope": "overrides")"), std::string::npos);
  EXPECT_NE(json.find(R"("live_port": "method")"), std::string::npos);
  EXPECT_NE(json.find(R"("achieved_frames_per_second": 1000)"), std::string::npos);
}

}  // namespace
