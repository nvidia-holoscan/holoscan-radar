// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <X11/Xlib.h>

#include <holoscan/holoscan.hpp>

namespace holoscan::radar::app {

// True when an X11 display server is reachable (DISPLAY / XOpenDisplay).
// Used to downgrade visualize=true on headless hosts without Holoviz.
inline bool x11_display_available() {
  Display* display = XOpenDisplay(nullptr);
  if (display == nullptr) {
    HOLOSCAN_LOG_DEBUG("radar_app: X11 display probe failed (XOpenDisplay)");
    return false;
  }
  XCloseDisplay(display);
  return true;
}

inline bool gui_available() { return x11_display_available(); }

}  // namespace holoscan::radar::app
