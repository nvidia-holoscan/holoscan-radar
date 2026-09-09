// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/dsp/virtual_array.hpp"

namespace holoscan::radar::detail {

constexpr int kSparseTdmErrCellCoord = 1;

template <typename T>
void run_tdm_sparse_compensation_stage(
    const void* rd_maps, const int64_t* detection_idx,
    int64_t num_detections, const TdmRdMapsLayout& layout,
    int64_t num_snapshots, int64_t doppler_half, int64_t range_half,
    SnapshotEdgeMode doppler_edge, SnapshotEdgeMode range_edge,
    const VirtualAssemblyPlan& virtual_plan, const void* correction,
    const int64_t* src_channel, const int64_t* member_offsets,
    const int64_t* member_indices, void* snapshot, uint8_t* is_valid,
    int* error_flag, cudaStream_t stream);

}  // namespace holoscan::radar::detail
