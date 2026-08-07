// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/source_count.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace holoscan::radar {

namespace detail {

// eq. (31): the exponential-profile decay-rate parameter `a` for a candidate
// noise subspace of size `m_prime`, given `n` snapshots. r = exp(-2a) is the
// profile's decay ratio (eq. 25); `a` -> 0 (r -> 1, flat profile) as N grows,
// matching "the exponential profile tends to a horizontal profile as the
// noise eigenvalues become equal" (finite-N correction vanishes).
inline double eft_profile_a(int64_t m_prime, int64_t n) {
  const double m = static_cast<double>(m_prime);
  const double nn = static_cast<double>(n);
  const double m2 = m * m;
  const double term1 = 15.0 / (m2 + 2.0);
  const double inner =
      term1 * term1 - (180.0 * m / nn) * ((m2 - 1.0) / (m2 + 2.0));
  // The order-4 expansion (eq. 30) that eq. (31) inverts can go slightly
  // negative at small N / small m_prime (outside its validity region);
  // clamp to the r -> 1 (flat-profile) limit rather than propagate a NaN.
  const double safe_inner = inner > 0.0 ? inner : 0.0;
  return 0.5 * (term1 - std::sqrt(safe_inner));
}

// eq. (27)/(32): J_{m_prime} = (1 - r) / (1 - r^m_prime), the normalization
// that makes the predicted profile's implied noise-floor average equal to the
// observed one (eq. 26). Well-defined in the r -> 1 limit (J -> 1/m_prime).
inline double eft_profile_j(int64_t m_prime, double r) {
  if (m_prime <= 0) {
    return 1.0;
  }
  const double r_pow = std::pow(r, static_cast<double>(m_prime));
  const double denom = 1.0 - r_pow;
  if (std::abs(denom) < 1e-12) {
    return 1.0 / static_cast<double>(m_prime);
  }
  return (1.0 - r) / denom;
}

// Standard normal quantile (inverse CDF), Acklam's rational approximation
// (relative error < 1.15e-9 over the full (0,1) domain). Self-contained: no
// <random> / statistics dependency for a single closed-form lookup.
inline double norm_inv(double p) {
  static constexpr double kA[6] = {
      -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
  static constexpr double kB[5] = {
      -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01};
  static constexpr double kC[6] = {
      -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
  static constexpr double kD[4] = {
      7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00};
  constexpr double kPLow = 0.02425;
  if (p <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (p >= 1.0) {
    return std::numeric_limits<double>::infinity();
  }
  if (p < kPLow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    const double num =
        (((((kC[0] * q + kC[1]) * q + kC[2]) * q + kC[3]) * q + kC[4]) * q +
        kC[5]);
    const double den =
        ((((kD[0] * q + kD[1]) * q + kD[2]) * q + kD[3]) * q + 1.0);
    return num / den;
  }
  if (p > 1.0 - kPLow) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    const double num =
        (((((kC[0] * q + kC[1]) * q + kC[2]) * q + kC[3]) * q + kC[4]) * q +
        kC[5]);
    const double den =
        ((((kD[0] * q + kD[1]) * q + kD[2]) * q + kD[3]) * q + 1.0);
    return -num / den;
  }
  const double q = p - 0.5;
  const double r = q * q;
  const double num =
      (((((kA[0] * r + kA[1]) * r + kA[2]) * r + kA[3]) * r + kA[4]) * r +
      kA[5]);
  const double den =
      (((((kB[0] * r + kB[1]) * r + kB[2]) * r + kB[3]) * r + kB[4]) * r +
      1.0);
  return num * q / den;
}

}  // namespace detail

int64_t eft_source_count(const std::vector<double>& eigenvalues_ascending,
                         int64_t num_snapshots, double pfa,
                         int64_t max_sources) {
  const int64_t m = static_cast<int64_t>(eigenvalues_ascending.size());
  if (m < 1) {
    throw std::invalid_argument("eft_source_count: need >= 1 eigenvalue");
  }
  if (num_snapshots < 1) {
    throw std::invalid_argument("eft_source_count: num_snapshots must be >= 1");
  }
  if (!(pfa > 0.0) || !(pfa < 1.0)) {
    throw std::invalid_argument("eft_source_count: pfa must be in (0, 1)");
  }
  if (max_sources < 1) {
    throw std::invalid_argument("eft_source_count: max_sources must be >= 1");
  }

  // Trivial arrays: at most one candidate step (M<=2) or none (M==1) - the
  // contract clamp still guarantees >= 1.
  if (m <= 2) {
    return std::min<int64_t>(1, max_sources);
  }

  // Pfa budget split uniformly over the up to (M-1) recursive steps
  // (mirrors eq. 37's per-step decomposition of the overall Pfa).
  const double pfa_step = pfa / static_cast<double>(m - 1);
  const double z = detail::norm_inv(1.0 - pfa_step);

  int64_t noise_dim = 1;  // the smallest eigenvalue alone, by assumption
  for (int64_t q = 1; q <= m - 1; ++q) {
    const int64_t candidate_size = q + 1;  // P+1 in the paper's notation
    double sigma2 = 0.0;
    for (int64_t i = 0; i <= q; ++i) {
      sigma2 += eigenvalues_ascending[static_cast<size_t>(i)];
    }
    sigma2 /= static_cast<double>(candidate_size);

    const double a = detail::eft_profile_a(candidate_size, num_snapshots);
    const double r = std::exp(-2.0 * a);
    const double j = detail::eft_profile_j(candidate_size, r);
    const double predicted = static_cast<double>(candidate_size) * j * sigma2;
    const double observed = eigenvalues_ascending[static_cast<size_t>(q)];

    if (!(predicted > 0.0)) {
      // Degenerate (all-zero) covariance: nothing left to discriminate.
      break;
    }
    const double relative_excess = (observed - predicted) / predicted;

    // First-order Gaussian approximation of the profile's relative
    // fluctuation (eq. 22-24: Var ~ sigma^4/N per eigenvalue; sigma^2 is
    // itself an average of `candidate_size` such terms) in place of the
    // paper's Monte-Carlo-calibrated eta_P - see the header's fidelity note.
    const double relative_std =
        std::sqrt(1.0 / (static_cast<double>(num_snapshots) *
                         static_cast<double>(candidate_size)));
    const double eta = z * relative_std;

    if (relative_excess > eta) {
      // Observed eigenvalue breaks from the noise profile: signal detected.
      break;
    }
    noise_dim = candidate_size;
  }

  const int64_t sources = m - noise_dim;
  return std::clamp<int64_t>(sources, 1, max_sources);
}

}  // namespace holoscan::radar
