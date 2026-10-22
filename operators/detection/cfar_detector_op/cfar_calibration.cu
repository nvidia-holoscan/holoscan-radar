// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_calibration.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>

namespace holoscan::radar::detail {
namespace {
constexpr int kBatch = 65536;
constexpr int kMaxTraining = 512;

void checked(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("CFAR calibration CUDA: ") + cudaGetErrorString(error));
}
struct Allocation {
  void* data{};
  explicit Allocation(std::size_t bytes) { checked(cudaMalloc(&data, bytes)); }
  ~Allocation() {
    if (data)
      cudaFree(data);
  }
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
};
struct Stream {
  cudaStream_t value{};
  Stream() { checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() {
    if (value) {
      cudaStreamSynchronize(value);
      cudaStreamDestroy(value);
    }
  }
};

__device__ double gamma_draw(std::int64_t k, curandStatePhilox4_32_10_t* state) {
  double value = 0;
  for (std::int64_t i = 0; i < k; ++i) {
    // A 64-bit uniform preserves rare upper tails. Near one, compute its
    // integer distance to the endpoint before converting to double so the
    // small exponential tail is not rounded to zero. Use fast log elsewhere.
    const auto high = static_cast<std::uint64_t>(curand(state));
    const auto bits = (high << 32) | static_cast<std::uint64_t>(curand(state));
    if (bits > (~std::uint64_t{0} - (std::uint64_t{1} << 40))) {
      const double complement = (static_cast<double>(~bits) + 0.5) * 0x1p-64;
      value -= log1p(-complement);
    } else {
      const double u = (static_cast<double>(bits) + 0.5) * 0x1p-64;
      value -= log(u);
    }
  }
  return value;
}
__device__ double combine(double neg, int nn, double pos, int np, bool go) {
  if (!nn)
    return np ? pos / np : 0;
  if (!np)
    return neg / nn;
  return go ? fmax(neg / nn, pos / np) : fmin(neg / nn, pos / np);
}
__device__ double select_rank(double* values, int n, int rank) {
  int left = 0, right = n - 1;
  while (left < right) {
    const double pivot = values[(left + right) / 2];
    int i = left, j = right;
    while (i <= j) {
      while (values[i] < pivot)
        ++i;
      while (values[j] > pivot)
        --j;
      if (i <= j) {
        double t = values[i];
        values[i++] = values[j];
        values[j--] = t;
      }
    }
    if (rank <= j)
      right = j;
    else if (rank >= i)
      left = i;
    else
      break;
  }
  return values[rank];
}
struct DrawConfig {
  int variant, nd, nr, gd, gr, td, tr, d, r;
  std::int64_t looks, n, os_rank;
  std::uint64_t seed;
};
__global__ void ratios(DrawConfig c, std::uint64_t begin, int count, double* output, int* invalid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  curandStatePhilox4_32_10_t rng;
  curand_init(c.seed, begin + static_cast<std::uint64_t>(i), 0, &rng);
  double noise = 0;
  if (c.variant == static_cast<int>(CfarVariant::kOS)) {
    double values[kMaxTraining];
    for (int j = 0; j < c.n; ++j)
      values[j] = gamma_draw(c.looks, &rng);
    noise = select_rank(values, static_cast<int>(c.n), static_cast<int>(min(c.os_rank, c.n - 1)));
  } else {
    double rn = 0, rp = 0, dn = 0, dp = 0;
    int rnc = 0, rpc = 0, dnc = 0, dpc = 0;
    if (c.d < 0) {
      for (int j = 0; j < c.n; ++j) {
        rn += gamma_draw(c.looks, &rng);
        rp += gamma_draw(c.looks, &rng);
        dn += gamma_draw(c.looks, &rng);
        dp += gamma_draw(c.looks, &rng);
      }
      rnc = rpc = dnc = dpc = static_cast<int>(c.n);
    } else {
      for (int dd = -c.gd - c.td; dd <= c.gd + c.td; ++dd) {
        if (static_cast<std::int64_t>(c.d) + dd < 0 || static_cast<std::int64_t>(c.d) + dd >= c.nd)
          continue;
        for (int rr = -c.gr - c.tr; rr <= c.gr + c.tr; ++rr) {
          if (static_cast<std::int64_t>(c.r) + rr < 0 ||
              static_cast<std::int64_t>(c.r) + rr >= c.nr || (abs(dd) <= c.gd && abs(rr) <= c.gr))
            continue;
          const double cell = gamma_draw(c.looks, &rng);
          if (rr < 0) {
            rn += cell;
            ++rnc;
          } else if (rr > 0) {
            rp += cell;
            ++rpc;
          }
          if (dd < 0) {
            dn += cell;
            ++dnc;
          } else if (dd > 0) {
            dp += cell;
            ++dpc;
          }
        }
      }
    }
    const bool go = c.variant == static_cast<int>(CfarVariant::kGO);
    const double rs = combine(rn, rnc, rp, rpc, go), ds = combine(dn, dnc, dp, dpc, go);
    noise = !(rnc + rpc) ? ds : !(dnc + dpc) ? rs : go ? fmax(rs, ds) : fmin(rs, ds);
  }
  const double ratio = gamma_draw(c.looks, &rng) / noise;
  if (!isfinite(ratio) || ratio <= 0)
    atomicExch(invalid, 1);
  output[i] = ratio;
}
}  // namespace

double calibrate_cfar_geometry_gpu(const CfarCalibrationPlan& plan,
                                   const CfarCalibrationGeometry& g,
                                   const std::function<void(std::uint64_t)>& done,
                                   std::stop_token stop, bool benchmark, std::uint32_t batch_size) {
  if (stop.stop_requested())
    throw CfarCalibrationCancelled{};
  if (!batch_size || batch_size > kBatch)
    throw std::invalid_argument("CFAR calibration invalid batch size");
  const auto trials =
      benchmark ? std::min<std::uint64_t>(kBatch, plan.budget.trials) : plan.budget.trials;
  const auto rank = std::min(plan.budget.tail_rank, trials);
  if (!trials || !rank ||
      rank > static_cast<std::uint64_t>(std::numeric_limits<int>::max() - kBatch))
    throw std::invalid_argument("CFAR calibration tail capacity exceeds supported GPU sort extent");
  const auto capacity = rank + kBatch;
  std::size_t free_bytes = 0, total_bytes = 0;
  checked(cudaMemGetInfo(&free_bytes, &total_bytes));
  // Reserve room for both key buffers and CUB scratch before starting a long build.
  if (capacity > free_bytes / (sizeof(double) * 6))
    throw std::invalid_argument("CFAR calibration tail workspace exceeds available GPU memory");
  Stream stream;
  Allocation in(capacity * sizeof(double)), out(capacity * sizeof(double)), invalid(sizeof(int));
  auto* input = static_cast<double*>(in.data);
  auto* output = static_cast<double*>(out.data);
  std::size_t temp_size = 0;
  checked(cub::DeviceRadixSort::SortKeysDescending(
      nullptr, temp_size, input, output, static_cast<int>(capacity), 0, 64, stream.value));
  Allocation temp(temp_size);
  const auto& p = *plan.params;
  auto geometry_seed = plan.seed;
  auto mix_seed = [&](std::uint64_t value) {
    geometry_seed += value + 0x9E3779B97F4A7C15ULL;
    geometry_seed = (geometry_seed ^ (geometry_seed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    geometry_seed = (geometry_seed ^ (geometry_seed >> 27)) * 0x94D049BB133111EBULL;
    geometry_seed ^= geometry_seed >> 31;
  };
  mix_seed(static_cast<std::uint32_t>(g.doppler + 1));
  mix_seed(static_cast<std::uint32_t>(g.range + 1));
  mix_seed(static_cast<std::uint64_t>(g.lut_index));
  DrawConfig cfg{static_cast<int>(p.cfar_variant),
                 static_cast<int>(plan.input_shape[plan.input_shape.size() - 2]),
                 static_cast<int>(plan.input_shape.back()),
                 static_cast<int>(p.guard_cells[0]),
                 static_cast<int>(p.guard_cells[1]),
                 static_cast<int>(p.training_cells[0]),
                 static_cast<int>(p.training_cells[1]),
                 g.doppler,
                 g.range,
                 plan.effective_looks,
                 g.lut_index,
                 p.os_rank,
                 geometry_seed};
  std::uint64_t retained = 0;
  for (std::uint64_t begin = 0; begin < trials;) {
    if (stop.stop_requested())
      throw CfarCalibrationCancelled{};
    const int batch = static_cast<int>(std::min<std::uint64_t>(batch_size, trials - begin));
    checked(cudaMemsetAsync(invalid.data, 0, sizeof(int), stream.value));
    ratios<<<(batch + 127) / 128, 128, 0, stream.value>>>(cfg, begin, batch, input + retained,
                                                          static_cast<int*>(invalid.data));
    checked(cudaGetLastError());
    checked(cub::DeviceRadixSort::SortKeysDescending(temp.data, temp_size, input, output,
                                                     static_cast<int>(retained) + batch, 0, 64,
                                                     stream.value));
    retained = std::min(rank, retained + batch);
    checked(cudaMemcpyAsync(input, output, retained * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream.value));
    int bad = 0;
    checked(cudaMemcpyAsync(&bad, invalid.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
    checked(cudaStreamSynchronize(stream.value));
    if (bad)
      throw std::runtime_error(
          "CFAR calibration encountered a nonfinite/zero ratio; cannot certify alpha");
    begin += batch;
    if (done)
      done(batch);
  }
  double alpha = 0;
  checked(cudaMemcpyAsync(&alpha, input + rank - 1, sizeof(double), cudaMemcpyDeviceToHost,
                          stream.value));
  checked(cudaStreamSynchronize(stream.value));
  return alpha;
}
}  // namespace holoscan::radar::detail
