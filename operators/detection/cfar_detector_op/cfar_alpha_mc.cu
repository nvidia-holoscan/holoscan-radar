// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_alpha_mc.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <mutex>

namespace holoscan::radar::detail {

namespace {

std::mutex g_gpu_mc_mutex;

__device__ uint64_t rng_step(uint64_t state) {
  return state * 6364136223846793005ULL + 1ULL;
}

__device__ double sample_exp(uint64_t* state) {
  *state = rng_step(*state);
  const double u = (static_cast<double>(*state >> 11) + 1.0) *
                   (1.0 / 9007199254740992.0);
  return -log(u);
}

__device__ double sample_gamma_k(int64_t k, uint64_t* state) {
  double sum = 0.0;
  for (int64_t i = 0; i < k; ++i) {
    sum += sample_exp(state);
  }
  return sum;
}

template <bool kGreatestOf>
__device__ double goso_side_dev(double neg_sum, int neg_cnt, double pos_sum,
                                int pos_cnt) {
  if (neg_cnt > 0 && pos_cnt > 0) {
    const double neg_mean = neg_sum / static_cast<double>(neg_cnt);
    const double pos_mean = pos_sum / static_cast<double>(pos_cnt);
    if (kGreatestOf) {
      return neg_mean > pos_mean ? neg_mean : pos_mean;
    }
    return neg_mean < pos_mean ? neg_mean : pos_mean;
  }
  if (neg_cnt > 0) {
    return neg_sum / static_cast<double>(neg_cnt);
  }
  if (pos_cnt > 0) {
    return pos_sum / static_cast<double>(pos_cnt);
  }
  return 0.0;
}

template <bool kGreatestOf>
__device__ double simulate_goso_noise_nsides_dev(int64_t n_side, int64_t k,
                                                 uint64_t* state) {
  auto side_mean = [&]() {
    double sum = 0.0;
    for (int64_t i = 0; i < n_side; ++i) {
      sum += sample_gamma_k(k, state);
    }
    return sum / static_cast<double>(n_side);
  };
  const double range_neg = side_mean();
  const double range_pos = side_mean();
  const double doppler_neg = side_mean();
  const double doppler_pos = side_mean();
  const double range_stat =
      goso_side_dev<kGreatestOf>(range_neg, 1, range_pos, 1);
  const double doppler_stat =
      goso_side_dev<kGreatestOf>(doppler_neg, 1, doppler_pos, 1);
  if (kGreatestOf) {
    return range_stat > doppler_stat ? range_stat : doppler_stat;
  }
  return range_stat < doppler_stat ? range_stat : doppler_stat;
}

template <bool kGreatestOf>
__device__ double simulate_goso_noise_at_cell_dev(int d, int r, int num_doppler,
                                                  int num_range, int guard_d,
                                                  int guard_r, int train_d,
                                                  int train_r, int64_t k,
                                                  uint64_t* state) {
  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  double range_neg_sum = 0.0;
  double range_pos_sum = 0.0;
  double doppler_neg_sum = 0.0;
  double doppler_pos_sum = 0.0;
  int range_neg_cnt = 0;
  int range_pos_cnt = 0;
  int doppler_neg_cnt = 0;
  int doppler_pos_cnt = 0;
  for (int dd = -half_d; dd <= half_d; ++dd) {
    const int nd = d + dd;
    if (nd < 0 || nd >= num_doppler) {
      continue;
    }
    const bool guard_row = (dd >= -guard_d && dd <= guard_d);
    for (int rr = -half_r; rr <= half_r; ++rr) {
      const int nr = r + rr;
      if (nr < 0 || nr >= num_range) {
        continue;
      }
      if (guard_row && rr >= -guard_r && rr <= guard_r) {
        continue;
      }
      const double cell = sample_gamma_k(k, state);
      if (rr < 0) {
        range_neg_sum += cell;
        ++range_neg_cnt;
      } else if (rr > 0) {
        range_pos_sum += cell;
        ++range_pos_cnt;
      }
      if (dd < 0) {
        doppler_neg_sum += cell;
        ++doppler_neg_cnt;
      } else if (dd > 0) {
        doppler_pos_sum += cell;
        ++doppler_pos_cnt;
      }
    }
  }
  const double range_stat = goso_side_dev<kGreatestOf>(
      range_neg_sum, range_neg_cnt, range_pos_sum, range_pos_cnt);
  const double doppler_stat = goso_side_dev<kGreatestOf>(
      doppler_neg_sum, doppler_neg_cnt, doppler_pos_sum, doppler_pos_cnt);
  if (kGreatestOf) {
    return range_stat > doppler_stat ? range_stat : doppler_stat;
  }
  return range_stat < doppler_stat ? range_stat : doppler_stat;
}

__device__ double os_select_rank_dev(double* samples, int count,
                                     int64_t os_rank) {
  if (count <= 0) {
    return 0.0;
  }
  const int rank =
      os_rank < static_cast<int64_t>(count) ? static_cast<int>(os_rank)
                                           : count - 1;
  for (int i = 1; i < count; ++i) {
    const double key = samples[i];
    int j = i - 1;
    while (j >= 0 && samples[j] > key) {
      samples[j + 1] = samples[j];
      --j;
    }
    samples[j + 1] = key;
  }
  return samples[rank];
}

__device__ double simulate_os_noise_n_dev(int64_t n, int64_t os_rank,
                                          int64_t k, uint64_t* state) {
  double samples[kMaxOsTrainingCells];
  const int count =
      n > static_cast<int64_t>(kMaxOsTrainingCells)
          ? kMaxOsTrainingCells
          : static_cast<int>(n);
  for (int i = 0; i < count; ++i) {
    samples[i] = sample_gamma_k(k, state);
  }
  return os_select_rank_dev(samples, count, os_rank);
}

__global__ void goso_mc_nsides_kernel(double alpha, int64_t n_side, int64_t k,
                                      int mc_trials, bool greatest_of,
                                      uint64_t seed_base, int* out_fa) {
  const int trial = blockIdx.x * blockDim.x + threadIdx.x;
  if (trial >= mc_trials) {
    return;
  }
  uint64_t state =
      seed_base ^ (static_cast<uint64_t>(trial) * 0xD6E8FEB86659FD93ULL);
  const double noise =
      greatest_of ? simulate_goso_noise_nsides_dev<true>(n_side, k, &state)
                  : simulate_goso_noise_nsides_dev<false>(n_side, k, &state);
  const double cut = sample_gamma_k(k, &state);
  if (cut > alpha * noise) {
    atomicAdd(out_fa, 1);
  }
}

__global__ void goso_mc_at_cell_kernel(double alpha, int64_t k, int d, int r,
                                       int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d,
                                       int train_r, int mc_trials,
                                       bool greatest_of, uint64_t seed_base,
                                       int* out_fa) {
  const int trial = blockIdx.x * blockDim.x + threadIdx.x;
  if (trial >= mc_trials) {
    return;
  }
  uint64_t state =
      seed_base ^ (static_cast<uint64_t>(trial) * 0xD6E8FEB86659FD93ULL);
  const double noise =
      greatest_of
          ? simulate_goso_noise_at_cell_dev<true>(
                d, r, num_doppler, num_range, guard_d, guard_r, train_d,
                train_r, k, &state)
          : simulate_goso_noise_at_cell_dev<false>(
                d, r, num_doppler, num_range, guard_d, guard_r, train_d,
                train_r, k, &state);
  const double cut = sample_gamma_k(k, &state);
  if (cut > alpha * noise) {
    atomicAdd(out_fa, 1);
  }
}

__global__ void os_mc_pfa_kernel(double alpha, int64_t n, int64_t os_rank,
                                 int64_t k, int mc_trials, uint64_t seed_base,
                                 int* out_fa) {
  const int trial = blockIdx.x * blockDim.x + threadIdx.x;
  if (trial >= mc_trials) {
    return;
  }
  uint64_t state =
      seed_base ^ (static_cast<uint64_t>(trial) * 0xA24BAED4963EE407ULL);
  const double noise = simulate_os_noise_n_dev(n, os_rank, k, &state);
  const double cut = sample_gamma_k(k, &state);
  if (cut > alpha * noise) {
    atomicAdd(out_fa, 1);
  }
}

int* device_false_alarm_counter() {
  static int* counter = nullptr;
  if (counter == nullptr) {
    if (cudaMalloc(&counter, sizeof(int)) != cudaSuccess) {
      return nullptr;
    }
  }
  return counter;
}

}  // namespace

bool gpu_mc_calibration_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

double goso_mc_pfa_from_alpha_gpu(double alpha, int64_t n_side, int64_t k,
                                  int mc_trials, int /*guard_d*/,
                                  int /*guard_r*/, int /*train_d*/,
                                  int /*train_r*/, bool greatest_of) {
  if (n_side <= 0 || k <= 0 || alpha <= 0.0 || mc_trials <= 0) {
    return alpha <= 0.0 ? 1.0 : 0.0;
  }
  std::lock_guard<std::mutex> lock(g_gpu_mc_mutex);
  int* counter = device_false_alarm_counter();
  if (counter == nullptr) {
    return -1.0;
  }
  if (cudaMemset(counter, 0, sizeof(int)) != cudaSuccess) {
    return -1.0;
  }
  const int block = 256;
  const int grid_x = (mc_trials + block - 1) / block;
  const uint64_t seed =
      0x474F43464152ULL ^ static_cast<uint64_t>(n_side) ^
      (static_cast<uint64_t>(k) << 32) ^
      (greatest_of ? 0ULL : 0x534F43464152ULL);
  goso_mc_nsides_kernel<<<dim3(grid_x), dim3(block)>>>(
      alpha, n_side, k, mc_trials, greatest_of, seed, counter);
  if (cudaGetLastError() != cudaSuccess) {
    return -1.0;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return -1.0;
  }
  int host_fa = 0;
  if (cudaMemcpy(&host_fa, counter, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return -1.0;
  }
  return static_cast<double>(host_fa) / static_cast<double>(mc_trials);
}

double goso_mc_pfa_at_cell_gpu(double alpha, int64_t k, int mc_trials, int d,
                               int r, int num_doppler, int num_range,
                               int guard_d, int guard_r, int train_d,
                               int train_r, bool greatest_of) {
  if (k <= 0 || alpha <= 0.0 || mc_trials <= 0) {
    return alpha <= 0.0 ? 1.0 : 0.0;
  }
  std::lock_guard<std::mutex> lock(g_gpu_mc_mutex);
  int* counter = device_false_alarm_counter();
  if (counter == nullptr) {
    return -1.0;
  }
  if (cudaMemset(counter, 0, sizeof(int)) != cudaSuccess) {
    return -1.0;
  }
  const int block = 256;
  const int grid_x = (mc_trials + block - 1) / block;
  const uint64_t seed =
      0x474F43464152ULL ^ (static_cast<uint64_t>(d) << 32) ^
      static_cast<uint64_t>(r) ^ (static_cast<uint64_t>(k) << 16) ^
      (greatest_of ? 0ULL : 0x534F43464152ULL);
  goso_mc_at_cell_kernel<<<dim3(grid_x), dim3(block)>>>(
      alpha, k, d, r, num_doppler, num_range, guard_d, guard_r, train_d,
      train_r, mc_trials, greatest_of, seed, counter);
  if (cudaGetLastError() != cudaSuccess) {
    return -1.0;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return -1.0;
  }
  int host_fa = 0;
  if (cudaMemcpy(&host_fa, counter, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return -1.0;
  }
  return static_cast<double>(host_fa) / static_cast<double>(mc_trials);
}

double os_mc_pfa_from_alpha_gpu(double alpha, int64_t n, int64_t os_rank,
                                int64_t k, int mc_trials, int /*guard_d*/,
                                int /*guard_r*/, int /*train_d*/,
                                int /*train_r*/) {
  if (n <= 0 || k <= 0 || alpha <= 0.0 || mc_trials <= 0) {
    return alpha <= 0.0 ? 1.0 : 0.0;
  }
  if (n > static_cast<int64_t>(kMaxOsTrainingCells)) {
    return -1.0;
  }
  const int64_t effective_rank = os_rank < n ? os_rank : n - 1;
  std::lock_guard<std::mutex> lock(g_gpu_mc_mutex);
  int* counter = device_false_alarm_counter();
  if (counter == nullptr) {
    return -1.0;
  }
  if (cudaMemset(counter, 0, sizeof(int)) != cudaSuccess) {
    return -1.0;
  }
  const int block = 256;
  const int grid_x = (mc_trials + block - 1) / block;
  const uint64_t seed =
      0x4F5343464152ULL ^ static_cast<uint64_t>(n) ^
      (static_cast<uint64_t>(effective_rank) << 16) ^
      (static_cast<uint64_t>(k) << 32);
  os_mc_pfa_kernel<<<dim3(grid_x), dim3(block)>>>(
      alpha, n, effective_rank, k, mc_trials, seed, counter);
  if (cudaGetLastError() != cudaSuccess) {
    return -1.0;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return -1.0;
  }
  int host_fa = 0;
  if (cudaMemcpy(&host_fa, counter, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return -1.0;
  }
  return static_cast<double>(host_fa) / static_cast<double>(mc_trials);
}

}  // namespace holoscan::radar::detail
