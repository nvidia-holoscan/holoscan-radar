// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/logger/logger.hpp>

namespace holoscan::radar::app {

// Keep C++ and Python FMCW app logs comparable across hosts and SSH sessions.
// The SDK preserves an explicit HOLOSCAN_LOG_FORMAT environment override.
inline void configure_fmcw_logging() {
  holoscan::set_log_pattern("[%Y-%m-%d %H:%M:%S.%e %z] [%^%l%$] [%s:%#] %v");
}

}  // namespace holoscan::radar::app
