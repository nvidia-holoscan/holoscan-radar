// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>

#include <holoscan/holoscan.hpp>
#include <holoviz/holoviz.hpp>

namespace holoscan::radar::app {

// Holoviz needs a reachable display server and a Vulkan surface; checking
// DISPLAY/WAYLAND_DISPLAY is not enough (the variable may be set with no
// server behind it). Probe by briefly initializing the Holoviz module — the
// same GLFW + Vulkan stack HolovizOp uses — on the main thread, where the
// failure is a catchable exception. Inside the running graph the same
// failure is thrown from a GXF worker thread and terminates the process,
// which is exactly what this probe exists to avoid.
inline bool gui_available() {
  namespace viz = holoscan::viz;
  try {
    viz::Init(64, 64, "radar_app display probe");
    viz::Shutdown();
    return true;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_DEBUG("radar_app: display probe failed: {}", e.what());
    return false;
  }
}

}  // namespace holoscan::radar::app
