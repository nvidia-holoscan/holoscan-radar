// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17

#include "radar_app/simple/mimo/config.hpp"

namespace holoscan::radar::app::mimo_ddm {

using Config = mimo::DdmConfig;

inline Config load_config(const std::filesystem::path& path) {
  return mimo::load_ddm_config(path);
}

}  // namespace holoscan::radar::app::mimo_ddm
