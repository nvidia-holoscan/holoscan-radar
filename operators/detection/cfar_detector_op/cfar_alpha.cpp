// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_alpha.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace holoscan::radar::detail {

namespace {

constexpr double kEps = 1e-12;
constexpr int kMaxBetaIter = 200;
constexpr double kFpmin = 1e-30;

double log_gamma(double z) { return std::lgamma(z); }

// Continued-fraction representation of the incomplete beta function
// (Numerical Recipes betacf), for x in (0, 1).
double incomplete_beta_cf(double a, double b, double x) {
  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kFpmin) {
    d = kFpmin;
  }
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= kMaxBetaIter; ++m) {
    const double m2 = 2.0 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpmin) {
      d = kFpmin;
    }
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpmin) {
      c = kFpmin;
    }
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpmin) {
      d = kFpmin;
    }
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpmin) {
      c = kFpmin;
    }
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) {
      break;
    }
  }
  return h;
}

// Regularized incomplete beta I_x(a, b).
double regularized_incomplete_beta(double a, double b, double x) {
  if (x <= 0.0) {
    return 0.0;
  }
  if (x >= 1.0) {
    return 1.0;
  }
  const double bt =
      std::exp(log_gamma(a + b) - log_gamma(a) - log_gamma(b) +
               a * std::log(x) + b * std::log(1.0 - x));
  const bool symm = x < (a + 1.0) / (a + b + 2.0);
  if (symm) {
    return bt * incomplete_beta_cf(a, b, x) / a;
  }
  return 1.0 - bt * incomplete_beta_cf(b, a, 1.0 - x) / b;
}

}  // namespace

int64_t training_cell_count(int d, int r, int num_doppler, int num_range,
                            int guard_d, int guard_r, int train_d,
                            int train_r) {
  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  int64_t train_cnt = 0;
  for (int dd = -half_d; dd <= half_d; ++dd) {
    const int64_t nd = static_cast<int64_t>(d) + dd;
    if (nd < 0 || nd >= num_doppler) {
      continue;
    }
    const bool guard_row = (dd >= -guard_d && dd <= guard_d);
    for (int rr = -half_r; rr <= half_r; ++rr) {
      const int64_t nr = static_cast<int64_t>(r) + rr;
      if (nr < 0 || nr >= num_range) {
        continue;
      }
      if (guard_row && rr >= -guard_r && rr <= guard_r) {
        continue;
      }
      ++train_cnt;
    }
  }
  return train_cnt;
}

int64_t max_training_cell_count(int num_doppler, int num_range, int guard_d,
                                int guard_r, int train_d, int train_r) {
  int64_t max_cnt = 0;
  for (int64_t d = 0; d < num_doppler; ++d) {
    for (int64_t r = 0; r < num_range; ++r) {
      max_cnt = std::max(
          max_cnt, training_cell_count(static_cast<int>(d), static_cast<int>(r),
                                     num_doppler, num_range, guard_d, guard_r,
                                     train_d, train_r));
    }
  }
  return max_cnt;
}

double ca_cfar_pfa_from_alpha(double alpha, int64_t n, int64_t k) {
  if (n <= 0 || k <= 0) {
    return 0.0;
  }
  if (alpha <= 0.0) {
    return 1.0;
  }
  const double c = alpha / static_cast<double>(n);
  const double x = 1.0 / (1.0 + c);
  return regularized_incomplete_beta(static_cast<double>(n * k),
                                     static_cast<double>(k), x);
}

double ca_cfar_alpha(double pfa, int64_t n, int64_t k) {
  if (n <= 0 || k <= 0) {
    throw std::invalid_argument("ca_cfar_alpha: n and k must be positive");
  }
  if (pfa <= 0.0 || pfa >= 1.0) {
    throw std::invalid_argument(
        "ca_cfar_alpha: pfa must be in (0, 1) exclusive");
  }

  double lo = 0.0;
  double hi = 1.0;
  while (ca_cfar_pfa_from_alpha(hi, n, k) > pfa) {
    hi *= 2.0;
    if (hi > 1e12) {
      throw std::runtime_error("ca_cfar_alpha: failed to bracket alpha");
    }
  }

  for (int iter = 0; iter < 100; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (ca_cfar_pfa_from_alpha(mid, n, k) > pfa) {
      lo = mid;
    } else {
      hi = mid;
    }
    if (hi - lo <= kEps * std::max(1.0, hi)) {
      break;
    }
  }
  return 0.5 * (lo + hi);
}

std::vector<double> build_ca_alpha_lut(double pfa, int64_t fold,
                                       int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d,
                                       int train_r) {
  const int64_t max_n = max_training_cell_count(
      num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  std::vector<double> lut(static_cast<size_t>(max_n + 1), 0.0);
  for (int64_t n = 1; n <= max_n; ++n) {
    lut[static_cast<size_t>(n)] = ca_cfar_alpha(pfa, n, fold);
  }
  return lut;
}

}  // namespace holoscan::radar::detail
