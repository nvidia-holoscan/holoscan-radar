// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HSDK 5 Graph entry helper for FMCW demo apps (replaces Application-based
// run_fmcw_app while Holoviz / Python remain deferred).

#pragma once

#include <chrono>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "radar_app/app/config_path.hpp"
#include "radar_app/app/gpu_placement.hpp"
#include "radar_app/app/logging.hpp"
#include "radar_app/report/cadence_meter.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/viz/display_probe.hpp"

namespace holoscan::radar::app {

inline constexpr auto kAppStall = std::chrono::seconds{120};
inline constexpr auto kAppPoll = std::chrono::milliseconds{50};

// Wait until AppFrameBank captures `total_frames`, a diagnostic appears, or
// progress stalls for kAppStall (resets on each completed sink frame).
inline void wait_for_app_frames(holoscan::RunSession& session, int total_frames) {
  auto last_progress = std::chrono::steady_clock::now();
  int seen = 0;
  std::unique_lock lock(AppFrameBank::mutex);
  while (AppFrameBank::captured < total_frames && session.diagnostics().empty() &&
         session.termination() == holoscan::RunTermination::kRunning) {
    AppFrameBank::ready.wait_for(lock, kAppPoll);
    const auto now = std::chrono::steady_clock::now();
    if (AppFrameBank::captured != seen) {
      seen = AppFrameBank::captured;
      last_progress = now;
    } else if (now - last_progress >= kAppStall) {
      return;
    }
  }
}

// `build` owns Graph composition (ops, flows, bind_tensor_out, set_params).
// `cfg` must expose .num_frames / .visualize / .headless / .skip_display_check.
template <class Config, class BuildFn>
int run_fmcw_graph_app(int argc, char** argv, const char* log_prefix, const char* default_yaml,
                       std::function<Config(const std::filesystem::path&)> load_config,
                       BuildFn&& build) {
  configure_fmcw_logging();
  try {
    const auto launch = parse_fmcw_launch_options(argc, argv, default_yaml);
    Config cfg = load_config(launch.config_path);
    if (cfg.visualize && !cfg.headless && !cfg.skip_display_check && !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "{}: visualize is enabled but no usable display was detected; "
          "running without the visualization window",
          log_prefix);
      cfg.visualize = false;
    }

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
    cfg.visualization.x11_presentation =
        resolve_fmcw_x11_presentation(cfg.visualize, cfg.visualization.x11_presentation,
                                      launch.choose_presenter, launch.config_path.string());
    select_fmcw_application_device(cfg.visualize, cfg.visualize,
                                   cfg.visualization.x11_presentation.value());
#else
    select_application_device(false, false);
#endif
    AppFrameBank::reset(static_cast<int>(cfg.num_frames));

    holoscan::Graph graph{log_prefix};
    holoscan::CompileOptions options;
    std::shared_ptr<CadenceMeter> cadence;
    build(graph, options, cfg, &cadence);

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
      throw std::runtime_error(std::string(log_prefix) +
                               " compile failed: " + std::string(plan.json()));
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    wait_for_app_frames(session, static_cast<int>(cfg.num_frames));
    const bool already_stopped = session.termination() == holoscan::RunTermination::kStopped;
    session.request_stop();
    session.wait();
    if (!session.diagnostics().empty()) {
      std::string msg = std::string(log_prefix) + " run diagnostics:";
      for (const auto& d : session.diagnostics()) {
        msg += " [";
        msg += holoscan::format_diagnostic(d, plan.graph());
        msg += "]";
      }
      throw std::runtime_error(msg);
    }
    if (AppFrameBank::captured < static_cast<int>(cfg.num_frames) && !already_stopped) {
      throw std::runtime_error(std::string(log_prefix) + " stalled after " +
                               std::to_string(AppFrameBank::captured) + " / " +
                               std::to_string(cfg.num_frames) + " frames");
    }
    if (cadence) {
      cadence->log_summary();
    }
    return 0;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("{}: {}", log_prefix, e.what());
    return 1;
  }
}

}  // namespace holoscan::radar::app
