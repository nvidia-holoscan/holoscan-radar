# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is entered via add_holohub_package(holoscan-radar ...) in
# pkg/CMakeLists.txt, which only fires when PKG_holoscan_radar is ON.
include(holohub_configure_deb)

holohub_configure_deb(
    NAME        "holoscan-radar"
    DESCRIPTION "${PROJECT_DESCRIPTION}"
    VERSION     "${PROJECT_VERSION}"
    VENDOR      "NVIDIA"
    CONTACT     "jjomier@nvidia.com"
    DEPENDS     "holoscan (>= 5.0.0)"
)
