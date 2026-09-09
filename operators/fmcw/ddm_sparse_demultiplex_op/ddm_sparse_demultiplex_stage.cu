// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_stage.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_stage.hpp"
#include "radar_common/tensor/cuda_check.hpp"

namespace holoscan::radar::detail {
namespace {

template <typename T>
using Complex = cuda::std::complex<T>;

template <typename T>
__device__ Complex<T> demultiplexed_channel(
    const Complex<T>* rd_maps, int64_t physical_channel, int64_t num_rx,
    int64_t num_doppler, int64_t num_range, int64_t width,
    int64_t folded_doppler, int64_t range, int64_t fold, int64_t num_tx,
    const int64_t* demux_table, const int64_t* tx_by_band,
    const Complex<T>* initial_phase) {
  const int64_t physical_tx = physical_channel / num_rx;
  const int64_t rx = physical_channel % num_rx;
  const int64_t source_band =
      (tx_by_band[physical_tx] + fold) % num_tx;
  const int64_t source_tx = tx_by_band[num_tx + source_band];
  const int64_t source_d =
      demux_table[source_tx * width + folded_doppler];
  const int64_t source =
      (rx * num_doppler + source_d) * num_range + range;
  return rd_maps[source] * initial_phase[physical_tx];
}

template <typename T>
__global__ void sparse_ddm_kernel(
    const Complex<T>* rd_maps, const int64_t* detection_idx,
    const int64_t* fold_idx, int64_t num_detections, int64_t num_rx,
    int64_t num_doppler, int64_t num_range, int64_t width, int64_t num_tx,
    int64_t num_virtual, int64_t num_snapshots, int64_t doppler_half,
    int64_t range_half, int doppler_edge, int range_edge, int overlap_policy,
    const int64_t* demux_table, const int64_t* tx_by_band,
    const Complex<T>* initial_phase, const int64_t* src_channel,
    const int64_t* member_offsets, const int64_t* member_indices,
    Complex<T>* snapshot, uint8_t* is_valid, int* error_flag) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  const int64_t virt = static_cast<int64_t>(blockIdx.y);
  if (item >= num_detections || virt >= num_virtual) {
    return;
  }
  const int64_t doppler_center = detection_idx[item * 2];
  const int64_t range_center = detection_idx[item * 2 + 1];
  const int64_t fold = fold_idx == nullptr ? 0 : fold_idx[item];
  if (doppler_center < 0 || doppler_center >= width ||
      range_center < 0 || range_center >= num_range) {
    if (virt == 0 && threadIdx.x == 0) {
      atomicMax(error_flag, kSparseDdmErrCellCoord);
    }
    return;
  }
  if (fold < 0 || fold >= num_tx) {
    if (virt == 0 && threadIdx.x == 0) {
      atomicMax(error_flag, kSparseDdmErrFold);
    }
    return;
  }

  for (int64_t slot = static_cast<int64_t>(threadIdx.x);
       slot < num_snapshots; slot += blockDim.x) {
    const int64_t range_len = 2 * range_half + 1;
    const int64_t od = slot / range_len - doppler_half;
    const int64_t orng = slot % range_len - range_half;
    bool valid_d = true;
    bool valid_r = true;
    const int64_t d = resolve_axis_index(
        doppler_center, od, doppler_half, width, doppler_edge, &valid_d);
    const int64_t r = resolve_axis_index(
        range_center, orng, range_half, num_range, range_edge, &valid_r);
    const bool valid = valid_d && valid_r;
    if (virt == 0) {
      is_valid[item * num_snapshots + slot] = static_cast<uint8_t>(valid);
    }
    Complex<T> value{};
    if (valid) {
      if (overlap_policy == static_cast<int>(OverlapPolicy::kAverage)) {
        const int64_t begin = member_offsets[virt];
        const int64_t end = member_offsets[virt + 1];
        for (int64_t member = begin; member < end; ++member) {
          value += demultiplexed_channel(
              rd_maps, member_indices[member], num_rx, num_doppler, num_range,
              width, d, r, fold, num_tx, demux_table, tx_by_band,
              initial_phase);
        }
        value /= static_cast<T>(end - begin);
      } else {
        value = demultiplexed_channel(
            rd_maps, src_channel[virt], num_rx, num_doppler, num_range, width,
            d, r, fold, num_tx, demux_table, tx_by_band, initial_phase);
      }
    }
    snapshot[(item * num_virtual + virt) * num_snapshots + slot] = value;
  }
}

}  // namespace

template <typename T>
void run_ddm_sparse_demultiplex_stage(
    const void* rd_maps, const int64_t* detection_idx,
    const int64_t* fold_idx, int64_t num_detections,
    const DdmRdMapsLayout& layout, int64_t width, int64_t num_tx,
    int64_t num_snapshots, int64_t doppler_half, int64_t range_half,
    SnapshotEdgeMode doppler_edge, SnapshotEdgeMode range_edge,
    const VirtualAssemblyPlan& virtual_plan, const int64_t* demux_table,
    const int64_t* tx_by_band, const void* initial_phase,
    const int64_t* src_channel, const int64_t* member_offsets,
    const int64_t* member_indices, void* snapshot, uint8_t* is_valid,
    int* error_flag, cudaStream_t stream) {
  if (num_detections == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned>(num_detections),
                  static_cast<unsigned>(virtual_plan.num_virtual));
  constexpr int kBlock = 128;
  sparse_ddm_kernel<T><<<grid, kBlock, 0, stream>>>(
      static_cast<const Complex<T>*>(rd_maps), detection_idx, fold_idx,
      num_detections, layout.num_rx, layout.num_doppler, layout.num_range,
      width, num_tx, virtual_plan.num_virtual, num_snapshots, doppler_half,
      range_half, edge_mode_id(doppler_edge), edge_mode_id(range_edge),
      static_cast<int>(virtual_plan.policy), demux_table, tx_by_band,
      static_cast<const Complex<T>*>(initial_phase), src_channel,
      member_offsets, member_indices, static_cast<Complex<T>*>(snapshot),
      is_valid, error_flag);
  check_cuda("sparse DDM demultiplex kernel", "DDMSparseDemultiplexOp");
}

template void run_ddm_sparse_demultiplex_stage<float>(
    const void*, const int64_t*, const int64_t*, int64_t,
    const DdmRdMapsLayout&, int64_t, int64_t, int64_t, int64_t, int64_t,
    SnapshotEdgeMode, SnapshotEdgeMode, const VirtualAssemblyPlan&,
    const int64_t*, const int64_t*, const void*, const int64_t*,
    const int64_t*, const int64_t*, void*, uint8_t*, int*, cudaStream_t);
template void run_ddm_sparse_demultiplex_stage<double>(
    const void*, const int64_t*, const int64_t*, int64_t,
    const DdmRdMapsLayout&, int64_t, int64_t, int64_t, int64_t, int64_t,
    SnapshotEdgeMode, SnapshotEdgeMode, const VirtualAssemblyPlan&,
    const int64_t*, const int64_t*, const void*, const int64_t*,
    const int64_t*, const int64_t*, void*, uint8_t*, int*, cudaStream_t);

}  // namespace holoscan::radar::detail
