// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_app/app/gpu_placement.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <string>
#include <optional>
#include <stdexcept>

#include <holoscan/logger/logger.hpp>

#include "radar_app/app/presentation_choice.hpp"

#if HOLOSCAN_RADAR_HAS_X11
#include <X11/Xlib.h>
#endif

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
#include <holoviz/adapter.hpp>
#include <holoviz/window.hpp>
#endif

namespace holoscan::radar::app {
namespace {

thread_local std::optional<int> display_device_ordinal;
thread_local std::optional<int> capture_device_ordinal;

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
// HSDK 5.0's public HolovizOp has no adapter query or selection parameter.
// Keep this dependency on its internal renderer isolated here and match the
// installed operator's first-adapter / first-presentable-adapter policy.
void discover_renderer_devices(bool window) {
  const auto adapters = holoviz::enumerate_adapters();
  if (!adapters)
    throw std::runtime_error(adapters.error().message);
  int capture_device = -1;
  for (const auto& adapter : *adapters) {
    if (adapter.cuda_device_index >= 0) {
      capture_device = adapter.cuda_device_index;
      break;
    }
  }
  capture_device_ordinal = capture_device;
  if (capture_device < 0)
    throw std::runtime_error("HoloViz: no CUDA-backed Vulkan adapter is available");
  if (!window)
    return;

  // Query the same window-system surface as WindowSinkDesc before compiling
  // CUDA pools. No logical device or swapchain is allocated by this probe.
  auto probe = holoviz::create_window({.title = "Radar GPU selection", .size = {1, 1}});
  if (!probe)
    throw std::runtime_error(probe.error().message);
  auto surface = probe->create_surface();
  if (!surface)
    throw std::runtime_error(surface.error().message);
  for (const auto& adapter : *adapters) {
    auto supported = adapter.supports_surface(*surface);
    if (!supported)
      throw std::runtime_error(supported.error().message);
    if (!*supported)
      continue;
    const int display_device = adapter.cuda_device_index;
    if (display_device < 0)
      throw std::runtime_error("HoloViz: window adapter has no CUDA device");
    display_device_ordinal = display_device;
    return;
  }
  throw std::runtime_error("HoloViz: no CUDA-backed Vulkan adapter can present to this display");
}
#endif

}  // namespace

int select_application_device(bool window, bool cuda_sink) {
  display_device_ordinal.reset();
  capture_device_ordinal.reset();
  const int selected = application_device().value;
#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
  if (window || cuda_sink)
    discover_renderer_devices(window);
#else
  if (window || cuda_sink)
    throw std::runtime_error("HoloViz is unavailable in this build");
#endif
  HOLOSCAN_LOG_INFO("Radar computation: CUDA device {}", selected);
  if (capture_device_ordinal && cuda_sink)
    HOLOSCAN_LOG_INFO("Radar composition: CUDA device {}", *capture_device_ordinal);
  if (display_device_ordinal)
    HOLOSCAN_LOG_INFO("Radar presentation: CUDA device {}", *display_device_ordinal);
  return selected;
}

int application_composer_device() {
  return capture_device_ordinal.value_or(application_device().value);
}

int application_display_device() {
  return display_device_ordinal.value_or(application_device().value);
}

bool fmcw_uses_x11_presentation(bool enabled) {
#if HOLOSCAN_RADAR_HAS_X11
  const char* display = std::getenv("DISPLAY");
  return enabled && display && display[0] != '\0';
#else
  static_cast<void>(enabled);
  return false;
#endif
}

bool resolve_fmcw_x11_presentation(bool window, std::optional<bool> configured, bool choose_again,
                                   const std::string& config_path) {
  // Cache per display and session type, rather than treating DISPLAY as proof
  // of SSH forwarding. Avoid SSH connection addresses so reconnects reuse it.
  const auto env = [](const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string{};
  };
  std::filesystem::path root = env("XDG_CACHE_HOME");
  if (root.empty() && !env("HOME").empty())
    root = std::filesystem::path(env("HOME")) / ".cache";
  const std::string setup =
      env("DISPLAY") + "\n" + env("WAYLAND_DISPLAY") + "\n" +
      (env("SSH_CONNECTION").empty() && env("SSH_TTY").empty() ? "local" : "ssh");
  // Stable FNV-1a keeps display names out of filenames across C++/Python runs.
  std::uint64_t key = 14695981039346656037ULL;
  for (const unsigned char byte : setup) {
    key ^= byte;
    key *= 1099511628211ULL;
  }
  const auto cache =
      root.empty() ? std::filesystem::path{}
                   : root / "holoscan-radar" / "presentation-v2" / (std::to_string(key) + ".json");
  return detail::choose_x11_presentation(
      detail::PresentationChoice{.window = window,
                                 .x11_available = fmcw_uses_x11_presentation(true),
                                 .interactive = ::isatty(STDIN_FILENO) && ::isatty(STDERR_FILENO),
                                 .configured = configured,
                                 .cache_file = cache,
                                 .setup_key = std::to_string(key),
                                 .choose_again = choose_again,
                                 .config_path = config_path},
      std::cin, std::cerr);
}

int select_fmcw_application_device(bool window, bool cuda_sink, bool x11_presentation) {
  const bool x11 = window && fmcw_uses_x11_presentation(x11_presentation);
#if HOLOSCAN_RADAR_HAS_X11
  // Initialize before either the main-thread probe or the SDK's GLFW opens X11.
  if (window && XInitThreads() == 0)
    throw std::runtime_error("X11 thread initialization failed");
#endif
  const int selected = select_application_device(window && !x11, cuda_sink);
  if (x11)
    HOLOSCAN_LOG_INFO("Radar presentation: X11 finished-image transport");
  return selected;
}

}  // namespace holoscan::radar::app
