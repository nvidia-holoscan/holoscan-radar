# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CPack pre-build script: add the Debian copyright file to the DEB staging tree only.
if(CPACK_GENERATOR STREQUAL "DEB")
    set(_doc "${CPACK_TEMPORARY_INSTALL_DIRECTORY}/usr/share/doc/holoscan-radar")
    file(MAKE_DIRECTORY "${_doc}")
    file(COPY_FILE "${CPACK_HOLOSCAN_RADAR_COPYRIGHT_FILE}" "${_doc}/copyright")
    file(CHMOD "${_doc}/copyright" PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
endif()
