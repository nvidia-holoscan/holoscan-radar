// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Public API only: tensor placement does not select HoloViz's rendering adapter.
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "image_graph.hpp"  // NOLINT(build/include_subdir): standalone reproducer

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("usage: holoviz_broken cuda|window DEVICE");
    const std::string sink_name = argv[1];
    const std::string device_name = argv[2];
    if (sink_name != "cuda" && sink_name != "window")
      throw std::invalid_argument("sink must be cuda or window");
    std::size_t end = 0;
    const int device = std::stoi(device_name, &end);
    if (end != device_name.size())
      throw std::invalid_argument("DEVICE must be a CUDA ordinal");
    return repro::run(sink_name == "window", device);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
