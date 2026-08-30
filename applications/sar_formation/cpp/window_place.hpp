// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Move a mapped top-level window to a work-area corner, or query
// the work-area size for GLFW client sizing. HolovizOp has no
// window-position parameter; GLFW is inside Holoviz, so this
// finds the window by title after it appears. X11 is loaded at
// runtime (libX11.so.6) so the app does not need libx11-dev.
// Include from one TU. Warns once if DISPLAY / libX11 / the title
// is missing (native Wayland without XWayland is a no-op).

#pragma once

#include <dlfcn.h>

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>

#include <holoscan/logger/logger.hpp>

namespace x11_place_detail {

// Xlib ABI. Keep in lockstep with X.h / Xlib.h / Xatom.h.
using X11Long = long;            // NOLINT(runtime/int)
using X11ULong = unsigned long;  // NOLINT(runtime/int)
using XDisplay = struct _XDisplay;
using XWindow = X11ULong;
using XAtom = X11ULong;

struct XTextProperty {
  unsigned char* value;
  XAtom encoding;
  int format;
  X11ULong nitems;
};

constexpr int kXSuccess = 0;  // X.h Success — not POSIX
constexpr XAtom kXaCardinal = 6;
constexpr XAtom kXaWindow = 33;

struct X11Api {
  void* lib = nullptr;
  XDisplay* (*XOpenDisplay)(const char*) = nullptr;
  int (*XCloseDisplay)(XDisplay*) = nullptr;
  int (*XDefaultScreen)(XDisplay*) = nullptr;
  XWindow (*XDefaultRootWindow)(XDisplay*) = nullptr;
  int (*XDisplayWidth)(XDisplay*, int) = nullptr;
  int (*XDisplayHeight)(XDisplay*, int) = nullptr;
  XAtom (*XInternAtom)(XDisplay*, const char*, int) = nullptr;
  int (*XGetWindowProperty)(XDisplay*, XWindow, XAtom, X11Long, X11Long,
                            int, XAtom, XAtom*, int*, X11ULong*, X11ULong*,
                            unsigned char**) = nullptr;
  int (*XFree)(void*) = nullptr;
  int (*XGetWMName)(XDisplay*, XWindow, XTextProperty*) = nullptr;
  int (*XGetGeometry)(XDisplay*, XWindow, XWindow*, int*, int*,
                      unsigned int*, unsigned int*, unsigned int*,
                      unsigned int*) = nullptr;
  int (*XMoveWindow)(XDisplay*, XWindow, int, int) = nullptr;
  int (*XFlush)(XDisplay*) = nullptr;
};

inline bool load_sym(void* lib, const char* name, void** out) {
  dlerror();
  *out = dlsym(lib, name);
  return *out != nullptr;
}

inline const X11Api* x11() {
  static const X11Api api = [] {
    X11Api loaded;
    loaded.lib = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
    if (loaded.lib == nullptr) {
      return loaded;
    }
    const auto sym = [&](const char* name, auto* fn) {
      void* p = nullptr;
      if (!load_sym(loaded.lib, name, &p)) {
        return false;
      }
      *fn = reinterpret_cast<std::remove_reference_t<decltype(*fn)>>(p);
      return true;
    };
    const bool ok =
        sym("XOpenDisplay", &loaded.XOpenDisplay) &&
        sym("XCloseDisplay", &loaded.XCloseDisplay) &&
        sym("XDefaultScreen", &loaded.XDefaultScreen) &&
        sym("XDefaultRootWindow", &loaded.XDefaultRootWindow) &&
        sym("XDisplayWidth", &loaded.XDisplayWidth) &&
        sym("XDisplayHeight", &loaded.XDisplayHeight) &&
        sym("XInternAtom", &loaded.XInternAtom) &&
        sym("XGetWindowProperty", &loaded.XGetWindowProperty) &&
        sym("XFree", &loaded.XFree) &&
        sym("XGetWMName", &loaded.XGetWMName) &&
        sym("XGetGeometry", &loaded.XGetGeometry) &&
        sym("XMoveWindow", &loaded.XMoveWindow) &&
        sym("XFlush", &loaded.XFlush);
    if (!ok) {
      dlclose(loaded.lib);
      loaded.lib = nullptr;
    }
    return loaded;
  }();
  return api.lib != nullptr ? &api : nullptr;
}

inline std::string x11_window_title(const X11Api& x, XDisplay* dpy,
                                    XWindow win) {
  const XAtom net_wm_name = x.XInternAtom(dpy, "_NET_WM_NAME", 0);
  const XAtom utf8 = x.XInternAtom(dpy, "UTF8_STRING", 0);
  XAtom actual = 0;
  int fmt = 0;
  X11ULong nitems = 0;
  X11ULong after = 0;
  unsigned char* data = nullptr;
  if (x.XGetWindowProperty(dpy, win, net_wm_name, 0, 256, 0, utf8,
                           &actual, &fmt, &nitems, &after, &data) ==
          kXSuccess &&
      data != nullptr && nitems > 0) {
    std::string title(reinterpret_cast<char*>(data), nitems);
    x.XFree(data);
    return title;
  }
  if (data != nullptr) {
    x.XFree(data);
  }
  XTextProperty prop{};
  if (x.XGetWMName(dpy, win, &prop) != 0 && prop.value != nullptr) {
    std::string title(reinterpret_cast<char*>(prop.value));
    x.XFree(prop.value);
    return title;
  }
  return {};
}

inline bool x11_work_area(const X11Api& x, XDisplay* dpy, XWindow root,
                          int* ox, int* oy, int* ow, int* oh) {
  const XAtom work = x.XInternAtom(dpy, "_NET_WORKAREA", 1);
  if (work == 0) {
    return false;
  }
  XAtom actual = 0;
  int fmt = 0;
  X11ULong nitems = 0;
  X11ULong after = 0;
  unsigned char* data = nullptr;
  if (x.XGetWindowProperty(dpy, root, work, 0, 4, 0, kXaCardinal,
                           &actual, &fmt, &nitems, &after, &data) !=
          kXSuccess ||
      data == nullptr || nitems < 4) {
    if (data != nullptr) {
      x.XFree(data);
    }
    return false;
  }
  const auto* vals = reinterpret_cast<const X11Long*>(data);
  *ox = static_cast<int>(vals[0]);
  *oy = static_cast<int>(vals[1]);
  *ow = static_cast<int>(vals[2]);
  *oh = static_cast<int>(vals[3]);
  x.XFree(data);
  return *ow > 0 && *oh > 0;
}

inline void x11_frame_top_bottom(const X11Api& x, XDisplay* dpy,
                                 XWindow win, int* top, int* bottom) {
  *top = 0;
  *bottom = 0;
  const XAtom ext = x.XInternAtom(dpy, "_NET_FRAME_EXTENTS", 1);
  if (ext == 0) {
    return;
  }
  XAtom actual = 0;
  int fmt = 0;
  X11ULong nitems = 0;
  X11ULong after = 0;
  unsigned char* data = nullptr;
  if (x.XGetWindowProperty(dpy, win, ext, 0, 4, 0, kXaCardinal, &actual,
                           &fmt, &nitems, &after, &data) != kXSuccess ||
      data == nullptr || nitems < 4) {
    if (data != nullptr) {
      x.XFree(data);
    }
    return;
  }
  const auto* vals = reinterpret_cast<const X11Long*>(data);
  *top = static_cast<int>(vals[2]);
  *bottom = static_cast<int>(vals[3]);
  x.XFree(data);
}

enum class WindowCorner { kBottomLeft, kTopLeft };

inline bool try_place_x11(const std::string& title,
                          WindowCorner corner) {
  const X11Api* x = x11();
  if (x == nullptr) {
    return false;
  }
  XDisplay* dpy = x->XOpenDisplay(nullptr);
  if (dpy == nullptr) {
    return false;
  }
  const XWindow root = x->XDefaultRootWindow(dpy);
  XWindow found = 0;
  const XAtom list = x->XInternAtom(dpy, "_NET_CLIENT_LIST", 1);
  XAtom actual = 0;
  int fmt = 0;
  X11ULong nitems = 0;
  X11ULong after = 0;
  unsigned char* data = nullptr;
  if (list != 0 &&
      x->XGetWindowProperty(dpy, root, list, 0, 1024, 0, kXaWindow,
                            &actual, &fmt, &nitems, &after, &data) ==
          kXSuccess &&
      data != nullptr) {
    const auto* wins = reinterpret_cast<const XWindow*>(data);
    for (X11ULong i = 0; i < nitems; ++i) {
      if (x11_window_title(*x, dpy, wins[i]) == title) {
        found = wins[i];
        break;
      }
    }
    x->XFree(data);
  }
  if (found == 0) {
    x->XCloseDisplay(dpy);
    return false;
  }

  const int screen = x->XDefaultScreen(dpy);
  int wx = 0;
  int wy = 0;
  int ww = x->XDisplayWidth(dpy, screen);
  int wh = x->XDisplayHeight(dpy, screen);
  x11_work_area(*x, dpy, root, &wx, &wy, &ww, &wh);

  int ft = 0;
  int fb = 0;
  x11_frame_top_bottom(*x, dpy, found, &ft, &fb);
  XWindow geom_root = 0;
  int gx = 0;
  int gy = 0;
  unsigned int gw = 0;
  unsigned int gh = 0;
  unsigned int border = 0;
  unsigned int depth = 0;
  if (x->XGetGeometry(dpy, found, &geom_root, &gx, &gy, &gw, &gh,
                      &border, &depth) == 0) {
    x->XCloseDisplay(dpy);
    return false;
  }
  const int total_h = static_cast<int>(gh) + ft + fb;
  int xpos = wx;
  int ypos = wy;
  if (corner == WindowCorner::kBottomLeft) {
    ypos = wy + wh - total_h;
    if (ypos < wy) {
      ypos = wy;
    }
  }
  x->XMoveWindow(dpy, found, xpos, ypos);
  x->XFlush(dpy);
  x->XCloseDisplay(dpy);
  return true;
}

// Work-area size for sizing a GLFW client. Falls back to the
// display size when _NET_WORKAREA is missing. False if X11 /
// DISPLAY is unavailable (caller keeps visualize_size).
inline bool desktop_work_area(int* width, int* height) {
  if (width == nullptr || height == nullptr) {
    return false;
  }
  const X11Api* x = x11();
  if (x == nullptr || std::getenv("DISPLAY") == nullptr) {
    return false;
  }
  XDisplay* dpy = x->XOpenDisplay(nullptr);
  if (dpy == nullptr) {
    return false;
  }
  const XWindow root = x->XDefaultRootWindow(dpy);
  const int screen = x->XDefaultScreen(dpy);
  int ox = 0;
  int oy = 0;
  int ww = x->XDisplayWidth(dpy, screen);
  int wh = x->XDisplayHeight(dpy, screen);
  x11_work_area(*x, dpy, root, &ox, &oy, &ww, &wh);
  x->XCloseDisplay(dpy);
  if (ww <= 0 || wh <= 0) {
    return false;
  }
  *width = ww;
  *height = wh;
  return true;
}

}  // namespace x11_place_detail

inline bool desktop_work_area(int* width, int* height) {
  return x11_place_detail::desktop_work_area(width, height);
}

inline void place_window(const std::string& title,
                         x11_place_detail::WindowCorner corner) {
  if (x11_place_detail::x11() == nullptr) {
    HOLOSCAN_LOG_WARN(
        "window_place: cannot move '{}': libX11.so.6 not loaded "
        "(native Wayland without XWayland is a no-op)",
        title);
    return;
  }
  if (std::getenv("DISPLAY") == nullptr) {
    HOLOSCAN_LOG_WARN(
        "window_place: cannot move '{}': DISPLAY is unset", title);
    return;
  }
  for (int i = 0; i < 50; ++i) {
    if (x11_place_detail::try_place_x11(title, corner)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  HOLOSCAN_LOG_WARN(
      "window_place: no window titled '{}' after 1 s; left unmoved "
      "(Holoviz may suffix the title, or the window is not X11)",
      title);
}

// Find `title` and move it to the bottom-left of the work area.
// Retries briefly: Holoviz maps the GLFW window a tick after
// start().
inline void place_window_bottom_left(const std::string& title) {
  place_window(title, x11_place_detail::WindowCorner::kBottomLeft);
}

inline void place_window_top_left(const std::string& title) {
  place_window(title, x11_place_detail::WindowCorner::kTopLeft);
}

// Non-blocking: Holoviz's size callback runs on the GLFW thread.
inline void spawn_place_window_bottom_left(const std::string& title) {
  static std::mutex mu;
  static bool started = false;
  std::lock_guard<std::mutex> lock(mu);
  if (started) {
    return;
  }
  started = true;
  std::thread([title] { place_window_bottom_left(title); }).detach();
}

inline void spawn_place_window_top_left(const std::string& title) {
  static std::mutex mu;
  static bool started = false;
  std::lock_guard<std::mutex> lock(mu);
  if (started) {
    return;
  }
  started = true;
  std::thread([title] { place_window_top_left(title); }).detach();
}
