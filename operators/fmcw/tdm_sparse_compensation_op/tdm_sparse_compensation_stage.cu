// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_stage.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_stage.hpp"
#include "radar_common/tensor/cuda_check.hpp"

namespace holoscan::radar::detail {

namespace {

template <typename T>
using Complex = cuda::std::complex<T>;

template <typename T>
__device__ Complex<T> corrected_channel(const Complex<T>* rd_maps, const Complex<T>* correction,
                                        const Complex<T>* fold_correction, int64_t fold,
                                        int64_t num_tx, int64_t channel, int64_t num_rx,
                                        int64_t num_doppler, int64_t num_range, int64_t doppler,
                                        int64_t range) {
  const int64_t tx = channel / num_rx;
  const int64_t src = ((channel * num_doppler + doppler) * num_range) + range;
  const Complex<T> corrected = rd_maps[src] * correction[tx * num_doppler + doppler];
  return fold == 0 ? corrected : corrected * fold_correction[fold * num_tx + tx];
}

__device__ int64_t positive_mod(int64_t value, int64_t modulus) {
  const int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

__device__ int64_t floor_div(int64_t value, int64_t divisor) {
  const int64_t quotient = value / divisor;
  const int64_t remainder = value % divisor;
  return remainder < 0 ? quotient - 1 : quotient;
}

template <typename T>
__device__ __forceinline__ void process_sparse_compensation_slot(
    const Complex<T>* rd_maps, const int64_t* detection_idx, const int64_t* fold_idx, int64_t item,
    int64_t virt, int64_t slot, int64_t num_tx, int64_t num_rx, int64_t num_doppler,
    int64_t num_range, int64_t num_virtual, int64_t num_snapshots, int64_t doppler_half,
    int64_t range_half, int doppler_edge, int range_edge, int overlap_policy,
    const Complex<T>* correction, const Complex<T>* fold_correction, const int64_t* src_channel,
    const int64_t* member_offsets, const int64_t* member_indices, Complex<T>* snapshot,
    uint8_t* is_valid, int* error_flag) {
  const int64_t doppler_center = detection_idx[item * 2];
  const int64_t range_center = detection_idx[item * 2 + 1];
  const int64_t center_fold = fold_idx == nullptr ? 0 : fold_idx[item];
  if (doppler_center < 0 || doppler_center >= num_doppler || range_center < 0 ||
      range_center >= num_range) {
    if (virt == 0 && slot == 0) {
      atomicMax(error_flag, kSparseTdmErrCellCoord);
    }
    return;
  }
  if (center_fold < 0 || center_fold >= num_tx) {
    if (virt == 0 && slot == 0) {
      atomicMax(error_flag, kSparseTdmErrFold);
    }
    return;
  }

  const int64_t range_len = 2 * range_half + 1;
  const int64_t od = slot / range_len - doppler_half;
  const int64_t orng = slot % range_len - range_half;
  bool valid_d = true;
  bool valid_r = true;
  const int64_t doppler =
      resolve_axis_index(doppler_center, od, doppler_half, num_doppler, doppler_edge, &valid_d);
  const int64_t range =
      resolve_axis_index(range_center, orng, range_half, num_range, range_edge, &valid_r);
  const bool valid = valid_d && valid_r;
  const int64_t unfolded_doppler = doppler_center + od;
  const int64_t wrap_count = doppler_edge == edge_mode_id(SnapshotEdgeMode::kWrap)
                                 ? floor_div(unfolded_doppler, num_doppler)
                                 : 0;
  const int64_t fold = positive_mod(center_fold + wrap_count, num_tx);
  if (virt == 0) {
    is_valid[item * num_snapshots + slot] = static_cast<uint8_t>(valid);
  }

  Complex<T> value{};
  if (valid) {
    if (overlap_policy == static_cast<int>(OverlapPolicy::kAverage)) {
      const int64_t begin = member_offsets[virt];
      const int64_t finish = member_offsets[virt + 1];
      for (int64_t member = begin; member < finish; ++member) {
        value += corrected_channel(rd_maps, correction, fold_correction, fold, num_tx,
                                   member_indices[member], num_rx, num_doppler, num_range, doppler,
                                   range);
      }
      value /= static_cast<T>(finish - begin);
    } else {
      value = corrected_channel(rd_maps, correction, fold_correction, fold, num_tx,
                                src_channel[virt], num_rx, num_doppler, num_range, doppler, range);
    }
  }
  snapshot[(item * num_virtual + virt) * num_snapshots + slot] = value;
}

template <typename T>
__global__ void sparse_compensation_kernel(
    const Complex<T>* rd_maps, const int64_t* detection_idx, const int64_t* fold_idx,
    int64_t num_detections, int64_t num_tx, int64_t num_rx, int64_t num_doppler, int64_t num_range,
    int64_t num_virtual, int64_t num_snapshots, int64_t doppler_half, int64_t range_half,
    int doppler_edge, int range_edge, int overlap_policy, const Complex<T>* correction,
    const Complex<T>* fold_correction, const int64_t* src_channel, const int64_t* member_offsets,
    const int64_t* member_indices, Complex<T>* snapshot, uint8_t* is_valid, int* error_flag) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  const int64_t virt = static_cast<int64_t>(blockIdx.y);
  if (item >= num_detections || virt >= num_virtual) {
    return;
  }
  for (int64_t slot = static_cast<int64_t>(threadIdx.x); slot < num_snapshots; slot += blockDim.x) {
    process_sparse_compensation_slot(
        rd_maps, detection_idx, fold_idx, item, virt, slot, num_tx, num_rx, num_doppler, num_range,
        num_virtual, num_snapshots, doppler_half, range_half, doppler_edge, range_edge,
        overlap_policy, correction, fold_correction, src_channel, member_offsets, member_indices,
        snapshot, is_valid, error_flag);
  }
}

template <typename T>
__global__ void sparse_compensation_kernel_point(
    const Complex<T>* rd_maps, const int64_t* detection_idx, const int64_t* fold_idx,
    int64_t num_detections, int64_t num_tx, int64_t num_rx, int64_t num_doppler, int64_t num_range,
    int64_t num_virtual, int64_t num_snapshots, int64_t doppler_half, int64_t range_half,
    int doppler_edge, int range_edge, int overlap_policy, const Complex<T>* correction,
    const Complex<T>* fold_correction, const int64_t* src_channel, const int64_t* member_offsets,
    const int64_t* member_indices, Complex<T>* snapshot, uint8_t* is_valid, int* error_flag) {
  const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t item = flat / num_virtual;
  const int64_t virt = flat % num_virtual;
  if (item >= num_detections || virt >= num_virtual) {
    return;
  }
  const int64_t slot = 0;
  process_sparse_compensation_slot(rd_maps, detection_idx, fold_idx, item, virt, slot, num_tx,
                                   num_rx, num_doppler, num_range, num_virtual, num_snapshots,
                                   doppler_half, range_half, doppler_edge, range_edge,
                                   overlap_policy, correction, fold_correction, src_channel,
                                   member_offsets, member_indices, snapshot, is_valid, error_flag);
}

}  // namespace

template <typename T>
void run_tdm_sparse_compensation_stage(const void* rd_maps, const int64_t* detection_idx,
                                       const int64_t* fold_idx, int64_t num_detections,
                                       const TdmRdMapsLayout& layout, int64_t num_snapshots,
                                       int64_t doppler_half, int64_t range_half,
                                       SnapshotEdgeMode doppler_edge, SnapshotEdgeMode range_edge,
                                       const VirtualAssemblyPlan& virtual_plan,
                                       const void* correction, const void* fold_correction,
                                       const int64_t* src_channel, const int64_t* member_offsets,
                                       const int64_t* member_indices, void* snapshot,
                                       uint8_t* is_valid, int* error_flag, cudaStream_t stream) {
  if (num_detections == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(num_detections),
                  static_cast<unsigned int>(virtual_plan.num_virtual));
  constexpr int kBlock = 128;
  if (num_snapshots == 1 && virtual_plan.num_virtual > 0 &&
      num_detections <= (2147483647LL * kBlock) / virtual_plan.num_virtual) {
    const int64_t total = num_detections * virtual_plan.num_virtual;
    const int point_grid = static_cast<int>((total + kBlock - 1) / kBlock);
    sparse_compensation_kernel_point<T><<<point_grid, kBlock, 0, stream>>>(
        static_cast<const Complex<T>*>(rd_maps), detection_idx, fold_idx, num_detections,
        layout.num_tx, layout.num_rx, layout.num_doppler, layout.num_range,
        virtual_plan.num_virtual, num_snapshots, doppler_half, range_half,
        edge_mode_id(doppler_edge), edge_mode_id(range_edge), static_cast<int>(virtual_plan.policy),
        static_cast<const Complex<T>*>(correction), static_cast<const Complex<T>*>(fold_correction),
        src_channel, member_offsets, member_indices, static_cast<Complex<T>*>(snapshot), is_valid,
        error_flag);
  } else {
    sparse_compensation_kernel<T><<<grid, kBlock, 0, stream>>>(
        static_cast<const Complex<T>*>(rd_maps), detection_idx, fold_idx, num_detections,
        layout.num_tx, layout.num_rx, layout.num_doppler, layout.num_range,
        virtual_plan.num_virtual, num_snapshots, doppler_half, range_half,
        edge_mode_id(doppler_edge), edge_mode_id(range_edge), static_cast<int>(virtual_plan.policy),
        static_cast<const Complex<T>*>(correction), static_cast<const Complex<T>*>(fold_correction),
        src_channel, member_offsets, member_indices, static_cast<Complex<T>*>(snapshot), is_valid,
        error_flag);
  }
  check_cuda("sparse TDM compensation kernel", "TDMSparseCompensationOp");
}

template void run_tdm_sparse_compensation_stage<float>(const void*, const int64_t*, const int64_t*,
                                                       int64_t, const TdmRdMapsLayout&, int64_t,
                                                       int64_t, int64_t, SnapshotEdgeMode,
                                                       SnapshotEdgeMode, const VirtualAssemblyPlan&,
                                                       const void*, const void*, const int64_t*,
                                                       const int64_t*, const int64_t*, void*,
                                                       uint8_t*, int*, cudaStream_t);
template void run_tdm_sparse_compensation_stage<double>(
    const void*, const int64_t*, const int64_t*, int64_t, const TdmRdMapsLayout&, int64_t, int64_t,
    int64_t, SnapshotEdgeMode, SnapshotEdgeMode, const VirtualAssemblyPlan&, const void*,
    const void*, const int64_t*, const int64_t*, const int64_t*, void*, uint8_t*, int*,
    cudaStream_t);

}  // namespace holoscan::radar::detail
