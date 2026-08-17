// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared config-path resolution for FMCW demo apps (Application and Graph).

#pragma once

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17

namespace holoscan::radar::app {

// Resolve the config path: the first CLI argument if given, otherwise
// `default_yaml` staged next to the executable. Bare names like
// `fmcw_mimo_ddm.yaml` also resolve beside the binary (CMake-staged YAMLs).
inline std::filesystem::path resolve_config_path(int argc, char** argv,
                                                  const char* default_yaml) {
  const auto bin_dir = std::filesystem::path(argv[0]).parent_path();
  if (argc > 1) {
    std::filesystem::path given(argv[1]);
    if (std::filesystem::is_regular_file(given)) {
      return given;
    }
    const auto beside_bin = bin_dir / given.filename();
    if (std::filesystem::is_regular_file(beside_bin)) {
      return beside_bin;
    }
    return given;  // load_config will report the original path
  }
  return bin_dir / default_yaml;
}

}  // namespace holoscan::radar::app
