# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Default the CUDA architectures before project() enables the CUDA language.
# Development builds target the GPU present at build time (`native`), as the
# SDK's build does. Package builds (PKG_holoscan_radar for DEB/TGZ, SKBUILD for
# wheels) are redistributed and must configure without a GPU, so they default to
# `all-major`: SASS for every major architecture the CUDA toolkit supports, plus
# PTX for the newest. An explicit CMAKE_CUDA_ARCHITECTURES or the CUDAARCHS
# environment variable overrides both defaults. Numeric values below 70 are
# floored (MatX requires compute capability >= 7.0).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    if(PKG_holoscan_radar OR SKBUILD)
        set(CMAKE_CUDA_ARCHITECTURES "all-major")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$"
       AND CMAKE_CUDA_ARCHITECTURES LESS 70)
    message(STATUS "MatX requires CUDA compute capability >= 7.0; using 70")
    set(CMAKE_CUDA_ARCHITECTURES "70")
endif()
