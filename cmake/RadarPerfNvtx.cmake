# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# radar_enable_perf_nvtx(<target>)
#
# Opt an operator target into NVTX compute()-phase instrumentation. When the
# HOLOSCAN_RADAR_PERF_NVTX option is ON, this adds the RADAR_NVTX compile
# definition so radar_common/sdk/nvtx.hpp emits its scoped ranges, and links the
# header-only CUDA::nvtx3 target when present. Under CUDA 13 NVTX3 is
# header-only (the old libnvToolsExt.so is gone), so the link only supplies the
# include dir/defines; never link the deprecated CUDA::nvToolsExt. When the
# option is OFF this is a no-op and the operator's hot path is untouched.
#
# One line per operator, so templatizing the instrumentation across the operator
# set stays trivial.
function(radar_enable_perf_nvtx target)
    if(HOLOSCAN_RADAR_PERF_NVTX)
        target_compile_definitions(${target} PRIVATE RADAR_NVTX)
        if(TARGET CUDA::nvtx3)
            target_link_libraries(${target} PRIVATE CUDA::nvtx3)
        endif()
    endif()
endfunction()
