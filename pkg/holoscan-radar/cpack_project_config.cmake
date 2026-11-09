# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CPack loads this file once per generator. Archives are raw install trees of the
# packaged files: no top-level directory, and the same prefix as the DEB.
if(CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|TZST|ZIP)$")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    set(CPACK_ARCHIVE_FILE_NAME "${CPACK_HOLOSCAN_RADAR_ARCHIVE_FILE_NAME}")
endif()
