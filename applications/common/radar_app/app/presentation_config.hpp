// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include <holoscan/logger/logger.hpp>

#include "radar_app/app/gpu_placement.hpp"
#include "radar_app/app/gui_lifecycle.hpp"
#include "radar_app/viz/display_probe.hpp"

namespace holoscan::radar::app {

inline void validate_presenter(const std::string& mode) {
  if (!mode.empty() && mode != "x11" && mode != "native" && mode != "headless" && mode != "off")
    throw std::invalid_argument("presenter must be x11, native, headless, or off");
}

// Resolve exactly once before renderer discovery or graph construction. This
// function is also used by Python, so preferences and failure policy cannot drift.
template <typename Config>
void prepare_presentation(Config& cfg, const std::string& path, const std::string& mode = {},
                          bool choose_again = false) {
  validate_presenter(mode);
  authoring_frame_progress() = std::make_shared<FrameProgress>();
  auto& viz = cfg.visualization;
  if (!mode.empty() && choose_again) {
    HOLOSCAN_LOG_INFO("--presenter overrides --choose-presenter for this run");
    choose_again = false;
  }
  if (!mode.empty()) {
    viz.enabled = mode != "off";
    viz.headless = mode == "headless";
    if (mode == "x11" || mode == "native")
      viz.x11_presentation = mode == "x11";
  }
  cfg.visualize = viz.enabled;
  cfg.headless = viz.headless;
  cfg.skip_display_check = viz.skip_display_check;
  const bool window = cfg.visualize && !cfg.headless;
  if (window && !cfg.skip_display_check && !gui_available()) {
    if (viz.x11_presentation.has_value() || !mode.empty())
      throw std::runtime_error(
          "Radar presentation: cannot open display; check DISPLAY and X11 connectivity");
    HOLOSCAN_LOG_WARN("Radar: no usable display; disabling visualization");
    cfg.visualize = false;
    viz.enabled = false;
  }
  if (cfg.visualize && !cfg.headless) {
    viz.x11_presentation =
        resolve_fmcw_x11_presentation(true, viz.x11_presentation, choose_again, path);
    if (viz.x11_presentation.value() && !fmcw_uses_x11_presentation(true))
      throw std::runtime_error("Radar X11 presentation requires X11/XRender support and DISPLAY");
  }
  if (!cfg.visualize)
    HOLOSCAN_LOG_INFO("Radar presentation: off");
  else if (cfg.headless)
    HOLOSCAN_LOG_INFO("Radar presentation: headless GPU rendering");
  try {
    select_fmcw_application_device(cfg.visualize && !cfg.headless, cfg.visualize,
                                   viz.x11_presentation.value_or(true));
  } catch (const std::exception& error) {
    if (cfg.visualize && !cfg.headless && !viz.x11_presentation.value_or(true))
      throw std::runtime_error(
          std::string(error.what()) +
          "; native presentation requires a compatible Vulkan display. "
          "Use --presenter x11 for SSH/X11 forwarding, or --presenter headless without windows.");
    throw;
  }
}

}  // namespace holoscan::radar::app
