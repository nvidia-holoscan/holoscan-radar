// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/spatial_smoothing_device.hpp"

#include <cuda/std/complex>

#include <cstdint>
#include <stdexcept>

namespace holoscan::radar {

namespace {

// One thread per output element (item, l, col) of ss_snapshot (items, L, M).
// Forward columns [0, PK) copy snapshot[item, subarray_index[p,l], k]; backward
// columns [PK, 2PK) copy conj(snapshot[item, subarray_index[p, fb_flip[l]],
// k]). Column decodes as p = (col mod PK)/K, k = col mod K.
template <typename T>
__global__ void gather_kernel(
    const cuda::std::complex<T>* __restrict__ snapshot,
    const int32_t* __restrict__ subarray_index,
    const int32_t* __restrict__ fb_flip, int64_t num_items,
    int64_t num_channels, int64_t num_snapshots, int64_t num_subarrays,
    int64_t subarray_size, int64_t looks_cols,
    cuda::std::complex<T>* __restrict__ ss_snapshot) {
  const int64_t m_cols = looks_cols * num_snapshots;  // M
  const int64_t total = num_items * subarray_size * m_cols;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t item = idx / (subarray_size * m_cols);
    const int64_t rem = idx % (subarray_size * m_cols);
    const int64_t l = rem / m_cols;
    const int64_t col = rem % m_cols;

    const int64_t fwd_cols = num_subarrays * num_snapshots;  // P·K
    const bool backward = col >= fwd_cols;
    const int64_t bcol = backward ? col - fwd_cols : col;
    const int64_t p = bcol / num_snapshots;
    const int64_t k = bcol % num_snapshots;
    const int64_t local = backward ? fb_flip[l] : l;
    const int64_t ch = subarray_index[p * subarray_size + local];

    cuda::std::complex<T> v =
        snapshot[item * num_channels * num_snapshots + ch * num_snapshots + k];
    if (backward) {
      v = cuda::std::complex<T>(v.real(), -v.imag());
    }
    ss_snapshot[item * subarray_size * m_cols + l * m_cols + col] = v;
  }
}

// One thread per output element (item, col) of ss_mask (num_items, M): repeats
// the per-snapshot mask[item, k] across every sub-array / backward copy (k =
// col mod K, valid because P·K is a multiple of K).
__global__ void expand_mask_kernel(const unsigned char* __restrict__ mask,
                                   int64_t num_items, int64_t num_snapshots,
                                   int64_t m_cols,
                                   unsigned char* __restrict__ ss_mask) {
  const int64_t total = num_items * m_cols;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t item = idx / m_cols;
    const int64_t col = idx % m_cols;
    const int64_t k = col % num_snapshots;
    ss_mask[idx] = mask[item * num_snapshots + k];
  }
}

}  // namespace

template <typename T>
void run_spatial_smoothing_gather(
    const cuda::std::complex<T>* snapshot, const unsigned char* mask,
    int64_t num_items, int64_t num_channels, int64_t num_snapshots,
    const int32_t* subarray_index, const int32_t* fb_flip,
    int64_t num_subarrays, int64_t subarray_size, bool forward_backward,
    cuda::std::complex<T>* ss_snapshot, unsigned char* ss_mask,
    cudaStream_t stream) {
  if (num_items <= 0 || subarray_size <= 0 || num_subarrays <= 0 ||
      num_snapshots <= 0) {
    return;
  }
  const int64_t looks_cols =
      forward_backward ? 2 * num_subarrays : num_subarrays;
  const int64_t m_cols = looks_cols * num_snapshots;

  constexpr int block = 256;
  const int64_t total = num_items * subarray_size * m_cols;
  const int grid =
      static_cast<int>((total + block - 1) / block > 0
                           ? (total + block - 1) / block
                           : 1);
  gather_kernel<T><<<grid, block, 0, stream>>>(
      snapshot, subarray_index, fb_flip, num_items, num_channels, num_snapshots,
      num_subarrays, subarray_size, looks_cols, ss_snapshot);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error(
        "run_spatial_smoothing_gather: gather kernel launch failed");
  }

  if (mask != nullptr && ss_mask != nullptr) {
    const int64_t mtotal = num_items * m_cols;
    const int mgrid = static_cast<int>((mtotal + block - 1) / block > 0
                                           ? (mtotal + block - 1) / block
                                           : 1);
    expand_mask_kernel<<<mgrid, block, 0, stream>>>(
        mask, num_items, num_snapshots, m_cols, ss_mask);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error(
          "run_spatial_smoothing_gather: mask kernel launch failed");
    }
  }
}

template void run_spatial_smoothing_gather<float>(
    const cuda::std::complex<float>*, const unsigned char*, int64_t, int64_t,
    int64_t, const int32_t*, const int32_t*, int64_t, int64_t, bool,
    cuda::std::complex<float>*, unsigned char*, cudaStream_t);
template void run_spatial_smoothing_gather<double>(
    const cuda::std::complex<double>*, const unsigned char*, int64_t, int64_t,
    int64_t, const int32_t*, const int32_t*, int64_t, int64_t, bool,
    cuda::std::complex<double>*, unsigned char*, cudaStream_t);

}  // namespace holoscan::radar
