// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/logger/logger.hpp>

#include "radar_app/viz/frame_viewport.hpp"

namespace holoscan::radar::app::viz {

// One managed window, two independently presented child surfaces. The parent
// never uploads or composites pixels; a timeline tick touches only its child.
class X11DockedWindow {
 public:
  X11DockedWindow(int width, int image_height, int footer_height, std::string title,
                  int status_width = 0, std::shared_ptr<FrameViewport> viewport = {})
  : width_(width),
    height_(image_height + footer_height),
    footer_height_(footer_height),
    status_width_(status_width),
    title_(std::move(title)),
    viewport_(std::move(viewport)) {}
  ~X11DockedWindow() {
    if (display_)
      XCloseDisplay(display_);
  }

  Window parent() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!display_) {
      display_ = XOpenDisplay(nullptr);
      if (!display_)
        throw std::runtime_error("Cannot open docked X11 display");
      window_ = XCreateSimpleWindow(display_, DefaultRootWindow(display_), 0, 0, width_, height_, 0,
                                    0, 0);
      XStoreName(display_, window_, title_.c_str());
      close_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
      XSetWMProtocols(display_, window_, &close_, 1);
      XSelectInput(display_, window_, StructureNotifyMask | KeyPressMask);
      XMapWindow(display_, window_);
      // Children are created on separate client connections. Publish the parent
      // on the server before handing its XID to either worker.
      XSync(display_, False);
    }
    return window_;
  }

  void attach(bool footer, Window child) {
    std::lock_guard<std::mutex> lock(mutex_);
    children_[footer ? 1 : 0] = child;
    layout();
  }

  bool poll() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!display_)
      return false;
    while (XPending(display_)) {
      XEvent event;
      XNextEvent(display_, &event);
      if ((event.type == ClientMessage && static_cast<Atom>(event.xclient.data.l[0]) == close_) ||
          (event.type == KeyPress && XLookupKeysym(&event.xkey, 0) == XK_Escape))
        return true;
      if (event.type == ConfigureNotify &&
          (width_ != event.xconfigure.width || height_ != event.xconfigure.height)) {
        width_ = event.xconfigure.width;
        height_ = event.xconfigure.height;
        layout();
      }
    }
    return false;
  }

  void uploaded(bool footer, std::uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++uploads_[footer ? 1 : 0];
    bytes_[footer ? 1 : 0] += bytes;
  }

  void release(bool footer, Display* child_display, Window child) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Serialize destruction against parent layout from the other worker.
    XDestroyWindow(child_display, child);
    XSync(child_display, False);
    children_[footer ? 1 : 0] = 0;
    if (!children_[0] && !children_[1] && display_) {
      HOLOSCAN_LOG_INFO(
          "Radar dock transport: image_uploads={} image_bytes={} "
          "timeline_uploads={} timeline_bytes={}",
          uploads_[0], bytes_[0], uploads_[1], bytes_[1]);
      XDestroyWindow(display_, window_);
      XCloseDisplay(display_);
      display_ = nullptr;
    }
  }

 private:
  void layout() {
    const int footer = std::min(footer_height_, std::max(1, height_ / 2));
    const int image = std::max(1, height_ - footer);
    const int status = std::min(status_width_, std::max(1, width_ / 2));
    if (viewport_)
      viewport_->set(width_ - status, footer);
    if (children_[0])
      XMoveResizeWindow(display_, children_[0], 0, 0, width_, status_width_ > 0 ? height_ : image);
    if (children_[1]) {
      XMoveResizeWindow(display_, children_[1], status, image, std::max(1, width_ - status),
                        footer);
      // The cached image/status surface spans the parent; timeline is above it.
      XRaiseWindow(display_, children_[1]);
    }
    XFlush(display_);
  }
  std::mutex mutex_;
  Display* display_{nullptr};
  Window window_{};
  Atom close_{};
  std::array<Window, 2> children_{};
  std::array<std::uint64_t, 2> uploads_{}, bytes_{};
  int width_, height_, footer_height_, status_width_;
  std::string title_;
  std::shared_ptr<FrameViewport> viewport_;
};
}  // namespace holoscan::radar::app::viz
