// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/dsp/virtual_array.hpp"

namespace holoscan::radar::detail {

constexpr int kSparseDdmErrCellCoord = 1;
constexpr int kSparseDdmErrFold = 2;

template <typename T>
void run_ddm_sparse_demultiplex_stage(
    const void* rd_maps, const int64_t* detection_idx, const int64_t* fold_idx,
    int64_t num_detections, const DdmRdMapsLayout& layout, int64_t width, int64_t num_tx,
    int64_t num_snapshots, int64_t doppler_half, int64_t range_half, SnapshotEdgeMode doppler_edge,
    SnapshotEdgeMode range_edge, const VirtualAssemblyPlan& virtual_plan,
    const int64_t* demux_table, const int64_t* tx_by_band, const void* initial_phase,
    const int64_t* src_channel, const int64_t* member_offsets, const int64_t* member_indices,
    void* snapshot, uint8_t* is_valid, int* error_flag, cudaStream_t stream);

}  // namespace holoscan::radar::detail
