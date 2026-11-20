// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>

#include "radar_app/sar/config.hpp"

inline AppConfig load_spotlight_config(const std::filesystem::path& path,
                                       const std::filesystem::path& cphd_override) {
  AppConfig cfg = load_sar_config(path, cphd_override);
  if (!(cfg.cphd_paths.size() == 1 && cfg.tile_geometries.empty() && !cfg.scroll &&
        cfg.parent_geometry.is_valid() == false && !cfg.wire_weight &&
        cfg.emit_interval_pulses == 0))
    throw std::invalid_argument("spotlight: configuration selects another SAR topology");
  return cfg;
}
