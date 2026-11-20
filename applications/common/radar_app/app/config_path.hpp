// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared config-path resolution for FMCW demo apps (Application and Graph).

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17

#include "radar_app/app/presentation_config.hpp"

namespace holoscan::radar::app {

struct FmcwLaunchOptions {
  std::filesystem::path config_path;
  bool choose_presenter{false};
  std::string presenter;
};

// Accept the presenter flag before or after the optional YAML path.
inline FmcwLaunchOptions parse_fmcw_launch_options(int argc, char** argv,
                                                   const char* default_yaml) {
  const auto bin_dir = std::filesystem::path(argv[0]).parent_path();
  FmcwLaunchOptions options{.config_path = bin_dir / default_yaml};
  bool has_config = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--choose-presenter") {
      options.choose_presenter = true;
      continue;
    }
    if (arg == "--presenter") {
      if (++i >= argc)
        throw std::invalid_argument("--presenter requires x11, native, headless, or off");
      options.presenter = argv[i];
      if (options.presenter.empty())
        throw std::invalid_argument("--presenter requires a mode");
      validate_presenter(options.presenter);
      continue;
    }
    if (arg.starts_with("--presenter=")) {
      options.presenter = std::string(arg.substr(12));
      if (options.presenter.empty())
        throw std::invalid_argument("--presenter requires a mode");
      validate_presenter(options.presenter);
      continue;
    }
    if (arg.starts_with("-") || has_config)
      throw std::invalid_argument("Usage: <application> [config.yaml] [--choose-presenter]");
    has_config = true;
    const std::filesystem::path given(arg);
    const auto beside_bin = bin_dir / given.filename();
    options.config_path = std::filesystem::is_regular_file(given)        ? given
                          : std::filesystem::is_regular_file(beside_bin) ? beside_bin
                                                                         : given;
  }
  return options;
}

}  // namespace holoscan::radar::app
