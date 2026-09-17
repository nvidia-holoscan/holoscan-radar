// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17

#include "radar_app/simple/simo/config.hpp"

namespace holoscan::radar::app::simo_beamform_then_detect {

using Config = simo::BeamformThenDetectConfig;

inline Config load_config(const std::filesystem::path& path) {
  return simo::load_beamform_then_detect_config(path);
}

}  // namespace holoscan::radar::app::simo_beamform_then_detect
