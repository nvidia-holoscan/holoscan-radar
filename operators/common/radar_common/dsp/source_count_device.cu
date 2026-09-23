// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/source_count_device.hpp"

#include <cmath>

namespace holoscan::radar {

namespace detail {

// Device mirrors of the host reference in source_count.cpp - kept
// byte-for-byte equivalent in formula (radar_common stays CUDA-free, so this
// cannot literally share the host translation unit); source_count.cpp is the
// spec these are validated against (see test_source_count.cpp).

__device__ inline double eft_profile_a_device(int64_t m_prime, int64_t n) {
  const double m = static_cast<double>(m_prime);
  const double nn = static_cast<double>(n);
  const double m2 = m * m;
  const double term1 = 15.0 / (m2 + 2.0);
  const double inner = term1 * term1 - (180.0 * m / nn) * ((m2 - 1.0) / (m2 + 2.0));
  const double safe_inner = inner > 0.0 ? inner : 0.0;
  return 0.5 * (term1 - sqrt(safe_inner));
}

__device__ inline double eft_profile_j_device(int64_t m_prime, double r) {
  if (m_prime <= 0) {
    return 1.0;
  }
  const double r_pow = pow(r, static_cast<double>(m_prime));
  const double denom = 1.0 - r_pow;
  if (fabs(denom) < 1e-12) {
    return 1.0 / static_cast<double>(m_prime);
  }
  return (1.0 - r) / denom;
}

// Acklam's rational approximation of the standard normal quantile - same
// coefficients as the host reference's detail::norm_inv.
__device__ inline double norm_inv_device(double p) {
  const double a0 = -3.969683028665376e+01, a1 = 2.209460984245205e+02;
  const double a2 = -2.759285104469687e+02, a3 = 1.383577518672690e+02;
  const double a4 = -3.066479806614716e+01, a5 = 2.506628277459239e+00;
  const double b0 = -5.447609879822406e+01, b1 = 1.615858368580409e+02;
  const double b2 = -1.556989798598866e+02, b3 = 6.680131188771972e+01;
  const double b4 = -1.328068155288572e+01;
  const double c0 = -7.784894002430293e-03, c1 = -3.223964580411365e-01;
  const double c2 = -2.400758277161838e+00, c3 = -2.549732539343734e+00;
  const double c4 = 4.374664141464968e+00, c5 = 2.938163982698783e+00;
  const double d0 = 7.784695709041462e-03, d1 = 3.224671290700398e-01;
  const double d2 = 2.445134137142996e+00, d3 = 3.754408661907416e+00;
  constexpr double kPLow = 0.02425;

  if (p < kPLow) {
    const double q = sqrt(-2.0 * log(p));
    return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) /
           ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
  }
  if (p > 1.0 - kPLow) {
    const double q = sqrt(-2.0 * log(1.0 - p));
    return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) /
           ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
  }
  const double q = p - 0.5;
  const double rr = q * q;
  return (((((a0 * rr + a1) * rr + a2) * rr + a3) * rr + a4) * rr + a5) * q /
         (((((b0 * rr + b1) * rr + b2) * rr + b3) * rr + b4) * rr + 1.0);
}

// One thread per item: sequential EFT recursion over its own (small,
// resident-in-registers-via-global-reads) eigenvalue array. num_channels is
// small (array element counts), so the O(num_channels) inner sums per step
// are cheap relative to the eig / covariance kernels upstream.
template <typename T>
__global__ void source_count_kernel(const T* __restrict__ eigenvalues,
                                    const int64_t* __restrict__ k_counts, int64_t num_items,
                                    int64_t num_channels, double pfa, int64_t max_sources,
                                    int32_t* __restrict__ source_count) {
  const int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= num_items) {
    return;
  }
  const T* eig = eigenvalues + item * num_channels;
  const int64_t m = num_channels;
  const int64_t num_snapshots = k_counts[item];

  // Rank-deficient covariance (too few valid snapshots to trust the small
  // eigenvalues at all) - contract fallback, no EFT recursion attempted.
  if (num_snapshots < num_channels) {
    source_count[item] = 1;
    return;
  }

  if (m <= 2) {
    source_count[item] = 1;
    return;
  }

  const double pfa_step = pfa / static_cast<double>(m - 1);
  const double z = detail::norm_inv_device(1.0 - pfa_step);

  int64_t noise_dim = 1;
  for (int64_t q = 1; q <= m - 1; ++q) {
    const int64_t candidate_size = q + 1;
    double sigma2 = 0.0;
    for (int64_t i = 0; i <= q; ++i) {
      sigma2 += static_cast<double>(eig[i]);
    }
    sigma2 /= static_cast<double>(candidate_size);

    const double a = detail::eft_profile_a_device(candidate_size, num_snapshots);
    const double r = exp(-2.0 * a);
    const double j = detail::eft_profile_j_device(candidate_size, r);
    const double predicted = static_cast<double>(candidate_size) * j * sigma2;
    const double observed = static_cast<double>(eig[q]);

    if (!(predicted > 0.0)) {
      break;
    }
    const double relative_excess = (observed - predicted) / predicted;
    const double relative_std =
        sqrt(1.0 / (static_cast<double>(num_snapshots) * static_cast<double>(candidate_size)));
    const double eta = z * relative_std;

    if (relative_excess > eta) {
      break;
    }
    noise_dim = candidate_size;
  }

  int64_t sources = m - noise_dim;
  sources = sources < 1 ? 1 : sources;
  sources = sources > max_sources ? max_sources : sources;
  source_count[item] = static_cast<int32_t>(sources);
}

}  // namespace detail

template <typename T>
void run_source_count_device(const T* eigenvalues, const int64_t* k_counts, int64_t num_items,
                             int64_t num_channels, double pfa, int64_t max_sources,
                             int32_t* source_count, cudaStream_t stream) {
  if (num_items <= 0 || num_channels <= 0) {
    return;
  }
  constexpr int kBlock = 128;
  const int grid = static_cast<int>((num_items + kBlock - 1) / kBlock);
  detail::source_count_kernel<T><<<grid, kBlock, 0, stream>>>(
      eigenvalues, k_counts, num_items, num_channels, pfa, max_sources, source_count);
}

template void run_source_count_device<float>(const float*, const int64_t*, int64_t, int64_t, double,
                                             int64_t, int32_t*, cudaStream_t);
template void run_source_count_device<double>(const double*, const int64_t*, int64_t, int64_t,
                                              double, int64_t, int32_t*, cudaStream_t);

}  // namespace holoscan::radar
