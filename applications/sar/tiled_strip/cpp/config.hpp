// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>

#include "radar_app/sar/config.hpp"

inline AppConfig load_tiled_strip_config(const std::filesystem::path& path,
                                         const std::filesystem::path& cphd_override) {
  AppConfig cfg = load_sar_config(path, cphd_override);
  if (!(cfg.cphd_paths.size() == 1 && cfg.tile_geometries.size() > 1 && !cfg.scroll &&
        cfg.parent_geometry.is_valid()))
    throw std::invalid_argument("tiled_strip: configuration selects another SAR topology");
  return cfg;
}
