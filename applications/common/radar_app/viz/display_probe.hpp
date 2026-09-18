// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Display probe for the HSDK 5 dashboard.

#if defined(HOLOSCAN_RADAR_HAS_X11)
#include "radar_app/viz/x11_display.hpp"
#else
namespace holoscan::radar::app {
inline bool gui_available() { return false; }
}  // namespace holoscan::radar::app
#endif
