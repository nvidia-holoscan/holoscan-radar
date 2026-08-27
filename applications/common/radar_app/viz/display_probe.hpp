// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Unified display probe for run_fmcw_app visualize downgrade.
// Prefer Holoviz when present; otherwise X11; otherwise always false.

#if defined(HOLOSCAN_RADAR_HAS_HOLOVIZ)
#include "radar_app/viz/display.hpp"
#elif defined(HOLOSCAN_RADAR_HAS_X11)
#include "radar_app/viz/x11_display.hpp"
#else
namespace holoscan::radar::app {
inline bool gui_available() { return false; }
}  // namespace holoscan::radar::app
#endif
