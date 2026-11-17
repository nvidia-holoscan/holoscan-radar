// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>
#if HOLOSCAN_RADAR_HAS_X11
#include <X11/Xlib.h>
#endif

namespace holoscan::radar::app::viz {
// A native SDK window has no public extent accessor. Isolate the existing X11
// title-based integration here; snapshot under the server lock to avoid races
// with temporary discovery windows and window-manager reparenting.
class WindowExtent {
 public:
  explicit WindowExtent(std::string title) : title_(std::move(title)) {}
  ~WindowExtent() { close(); }
  void close() {
#if HOLOSCAN_RADAR_HAS_X11
    if (display_)
      XCloseDisplay(display_);
    display_ = nullptr;
#endif
  }
  std::pair<int, int> query(int width, int height) {
#if HOLOSCAN_RADAR_HAS_X11
    if (!display_)
      display_ = XOpenDisplay(nullptr);
    if (display_) {
      XGrabServer(display_);
      unsigned int budget = 4096;
      find(DefaultRootWindow(display_), 0, budget, width, height);
      XUngrabServer(display_);
      XFlush(display_);
    }
#endif
    return {width, height};
  }

 private:
#if HOLOSCAN_RADAR_HAS_X11
  bool find(Window window, unsigned int depth, unsigned int& budget, int& width, int& height) {
    if (depth > 4 || budget == 0)
      return false;
    --budget;
    char* name = nullptr;
    XFetchName(display_, window, &name);
    const bool match = name && title_ == name;
    if (name)
      XFree(name);
    if (match) {
      Window root;
      int x, y;
      unsigned int w, h, border, bits;
      if (XGetGeometry(display_, window, &root, &x, &y, &w, &h, &border, &bits) && w && h) {
        width = static_cast<int>(w);
        height = static_cast<int>(h);
        return true;
      }
    }
    Window root, parent, *children = nullptr;
    unsigned int count = 0;
    if (!XQueryTree(display_, window, &root, &parent, &children, &count))
      return false;
    bool found = false;
    for (unsigned int i = 0; i < count && !found && budget; ++i)
      found = find(children[i], depth + 1, budget, width, height);
    if (children)
      XFree(children);
    return found;
  }
  Display* display_{nullptr};
#endif
  std::string title_;
};
}  // namespace holoscan::radar::app::viz
