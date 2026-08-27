// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared application entry point for the FMCW example apps. Folds the identical
// main() scaffolding — config-path resolution, the app config load, the
// display-probe downgrade for visualize on GUI-less hosts, feeding the same
// file to Holoscan's native config() path, and run() inside one try/catch — so
// each app's main() collapses to a single call.
//
// Header-only: inline free function + template.

#pragma once

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/holoscan.hpp>

#include "radar_app/app/config_path.hpp"
#include "radar_app/viz/display_probe.hpp"

namespace holoscan::radar::app {

namespace detail {

// SFINAE: log CadenceMeter summary when App exposes a `cadence` member.
template <class App>
void log_cadence_if_present(App*, ...) {}

template <class App>
auto log_cadence_if_present(App* app, int) -> decltype(app->cadence, void()) {
  if (app->cadence) {
    app->cadence->log_summary();
  }
}

}  // namespace detail

// Load config, downgrade visualize on a GUI-less host, then config() + run()
// under one try/catch. `load_config` is the app's own loader
// ((const std::filesystem::path&) -> decltype(App::cfg)); `App::cfg` must
// expose .visualize / .headless / .skip_display_check (satisfied by deriving
// the app's AppConfig from CommonRunConfig).
//
// [EA HSDK5 ONLY] Application API is unavailable without Holoviz demos —
// prefer run_fmcw_graph_app (graph_driver.hpp) on this branch.
template <class App, class LoadFn>
int run_fmcw_app(int argc, char** argv, const char* log_prefix,
                 const char* default_yaml, LoadFn&& load_config) {
  auto app = holoscan::make_application<App>();
  try {
    const std::filesystem::path config_path =
        resolve_config_path(argc, argv, default_yaml);
    app->cfg = std::forward<LoadFn>(load_config)(config_path);
    // Headless rendering needs no display server, so the probe is moot there.
    if (app->cfg.visualize && !app->cfg.headless &&
        !app->cfg.skip_display_check && !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "{}: visualize is enabled but no usable display was detected "
          "(no usable display was detected); running without the "
          "visualization window",
          log_prefix);
      app->cfg.visualize = false;
    }
    // Same file feeds the native config path: compose() pulls the per-operator
    // blocks via from_config().
    app->config(config_path.string());
    // run() is inside the try: a graph failure (e.g. an operator rejecting an
    // invalid parameter mid-tick) is rethrown out of run(); catching it here
    // exits cleanly with the message instead of terminating on an uncaught
    // exception, which the runtime reports as a process abort.
    app->run();
    detail::log_cadence_if_present(app.get(), 0);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("{}: {}", log_prefix, e.what());
    return 1;
  }
  return 0;
}

}  // namespace holoscan::radar::app
