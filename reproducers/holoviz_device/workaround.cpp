// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unsupported renderer APIs discover the adapter before configuring the same graph.
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <holoviz/adapter.hpp>
#include <holoviz/window.hpp>

#include "image_graph.hpp"  // NOLINT(build/include_subdir): standalone reproducer

namespace {
// Unsupported workaround: duplicate the GA operator's adapter-selection policy before compile().
// A supported resolve_render_device(sink) would replace this entire function and renderer linkage.
int discover_device(bool window) {
  const auto adapters = holoviz::enumerate_adapters();
  if (!adapters)
    throw std::runtime_error(adapters.error().message);
  if (!window) {
    for (const auto& adapter : *adapters)
      if (adapter.cuda_device_index >= 0)
        return adapter.cuda_device_index;
  } else {
    auto probe =
        holoviz::create_window({.title = "Device probe", .size = {repro::kSize, repro::kSize}});
    if (!probe)
      throw std::runtime_error(probe.error().message);
    auto surface = probe->create_surface();
    if (!surface)
      throw std::runtime_error(surface.error().message);
    for (const auto& adapter : *adapters) {
      auto supported = adapter.supports_surface(*surface);
      if (!supported)
        throw std::runtime_error(supported.error().message);
      if (*supported) {
        if (adapter.cuda_device_index < 0)
          throw std::runtime_error("First presentable adapter has no CUDA device");
        return adapter.cuda_device_index;
      }
    }
  }
  throw std::runtime_error("No suitable CUDA-backed Vulkan adapter");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: holoviz_workaround cuda|window");
    const std::string sink_name = argv[1];
    if (sink_name != "cuda" && sink_name != "window")
      throw std::invalid_argument("sink must be cuda or window");
    const bool window = sink_name == "window";
    const int device = discover_device(window);
    std::cout << "Internal discovery: CUDA device " << device << '\n';
    return repro::run(window, device);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
