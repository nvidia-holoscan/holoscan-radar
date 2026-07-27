// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_app/viz/viz_kernels.hpp"

#include <cub/device/device_radix_sort.cuh>

#include <stdexcept>
#include <string>

namespace holoscan::radar::app {

namespace {

constexpr int kBlock = 256;

// |z| for complex input; pass-through for a real (power) surface. Kept as one
// kernel so the rest of the chain is dtype-agnostic.
__global__ void magnitude_kernel(const void* in, bool input_is_complex,
                                 int64_t count, float* out) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  if (input_is_complex) {
    const float2 z = static_cast<const float2*>(in)[i];
    out[i] = sqrtf(z.x * z.x + z.y * z.y);
  } else {
    out[i] = static_cast<const float*>(in)[i];
  }
}

// Median of each range column (one block per column). Selection by rank rather
// than a sort: for element i the rank is #(v_j < v_i) + #(v_j == v_i && j < i),
// so ranks are a permutation of [0, num_doppler) even with duplicates and
// exactly one thread holds rank num_doppler/2. That index is the same element
// std::nth_element(begin, begin + n/2, end) selected, so the result matches the
// former host code bit for bit.
__global__ void col_median_kernel(const float* mag, int64_t num_doppler,
                                  int64_t num_range, float* col_median) {
  extern __shared__ float column[];
  const int64_t r = blockIdx.x;
  if (r >= num_range) {
    return;
  }
  for (int64_t d = threadIdx.x; d < num_doppler; d += blockDim.x) {
    column[d] = mag[d * num_range + r];
  }
  __syncthreads();

  const int64_t target = num_doppler / 2;
  for (int64_t i = threadIdx.x; i < num_doppler; i += blockDim.x) {
    const float v = column[i];
    int64_t rank = 0;
    for (int64_t j = 0; j < num_doppler; ++j) {
      const float u = column[j];
      rank += (u < v) || (u == v && j < i) ? 1 : 0;
    }
    if (rank == target) {
      col_median[r] = v;
    }
  }
}

// In-place magnitude -> dB against the column median. The +1e-20 sits on the
// ratio (not inside the median clamp), matching the host original.
__global__ void to_db_kernel(float* image, const float* col_median,
                             int64_t num_range, int64_t count,
                             float db_factor) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float med = fmaxf(col_median[i % num_range], 1e-20F);
  image[i] = db_factor * log10f(image[i] / med + 1e-20F);
}

// Renormalize between the image's own median (floor) and max (peak), read
// straight off the sorted copy — sorted[n/2] is the median std::nth_element
// picked, sorted[n-1] the max the host loop tracked.
__global__ void normalize_kernel(float* image, const float* sorted,
                                 int64_t count, float lut_max) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float floor_db = sorted[count / 2];
  const float span_db = fmaxf(sorted[count - 1] - floor_db, 1.0F);
  image[i] =
      fminf(fmaxf((image[i] - floor_db) / span_db, 0.0F), 1.0F) * lut_max;
}

int64_t grid_for(int64_t count) {
  return (count + kBlock - 1) / kBlock;
}

}  // namespace

size_t rd_lut_image_sort_scratch_bytes(int64_t count) {
  size_t bytes = 0;
  // Sizing query: CUB writes the requirement and enqueues nothing when the
  // scratch pointer is null.
  cub::DeviceRadixSort::SortKeys(nullptr, bytes, static_cast<float*>(nullptr),
                                 static_cast<float*>(nullptr),
                                 static_cast<int>(count));
  return bytes;
}

void rd_map_to_lut_image(const void* rd_map, bool input_is_complex,
                         int64_t num_doppler, int64_t num_range,
                         float db_factor, float lut_max, float* image,
                         float* col_median, float* sorted, void* sort_scratch,
                         size_t sort_scratch_bytes, cudaStream_t stream) {
  if (num_doppler > kRdLutMaxDoppler) {
    throw std::runtime_error(
        "rd_map_to_lut_image: num_doppler " + std::to_string(num_doppler) +
        " exceeds the shared-memory column limit " +
        std::to_string(kRdLutMaxDoppler));
  }
  const int64_t count = num_doppler * num_range;

  magnitude_kernel<<<grid_for(count), kBlock, 0, stream>>>(
      rd_map, input_is_complex, count, image);

  const size_t shared_bytes =
      static_cast<size_t>(num_doppler) * sizeof(float);
  col_median_kernel<<<static_cast<unsigned>(num_range), kBlock, shared_bytes,
                      stream>>>(image, num_doppler, num_range, col_median);

  to_db_kernel<<<grid_for(count), kBlock, 0, stream>>>(image, col_median,
                                                       num_range, count,
                                                       db_factor);

  size_t bytes = sort_scratch_bytes;
  cub::DeviceRadixSort::SortKeys(sort_scratch, bytes, image, sorted,
                                 static_cast<int>(count), 0,
                                 static_cast<int>(sizeof(float) * 8), stream);

  normalize_kernel<<<grid_for(count), kBlock, 0, stream>>>(image, sorted, count,
                                                           lut_max);

  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("rd_map_to_lut_image: kernel launch failed");
  }
}

}  // namespace holoscan::radar::app
