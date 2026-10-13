// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17

#include "radar_app/fmcw/mimo_config.hpp"

namespace holoscan::radar::app::mimo_tdm {

using Config = mimo::TdmConfig;

inline Config load_config(const std::filesystem::path& path) {
  return mimo::load_tdm_config(path);
}

}  // namespace holoscan::radar::app::mimo_tdm
