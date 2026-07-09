// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_alpha.hpp"

#include <algorithm>
#include <cmath>
#include <random>
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

void go_training_side_counts(int d, int r, int num_doppler, int num_range,
                             int guard_d, int guard_r, int train_d,
                             int train_r, int64_t* range_neg,
                             int64_t* range_pos, int64_t* doppler_neg,
                             int64_t* doppler_pos) {
  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  int64_t rn = 0;
  int64_t rp = 0;
  int64_t dn = 0;
  int64_t dp = 0;
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
      if (rr < 0) {
        ++rn;
      } else if (rr > 0) {
        ++rp;
      }
      if (dd < 0) {
        ++dn;
      } else if (dd > 0) {
        ++dp;
      }
    }
  }
  *range_neg = rn;
  *range_pos = rp;
  *doppler_neg = dn;
  *doppler_pos = dp;
}

int64_t go_lut_side_index(int64_t neg_count, int64_t pos_count) {
  if (neg_count > 0 && pos_count > 0) {
    return std::min(neg_count, pos_count);
  }
  return std::max(neg_count, pos_count);
}

int64_t go_lut_index(int64_t range_neg, int64_t range_pos, int64_t doppler_neg,
                     int64_t doppler_pos) {
  const int64_t nr = go_lut_side_index(range_neg, range_pos);
  const int64_t nd = go_lut_side_index(doppler_neg, doppler_pos);
  if (nr > 0 && nd > 0) {
    return std::min(nr, nd);
  }
  return std::max(nr, nd);
}

int64_t max_go_lut_index(int num_doppler, int num_range, int guard_d,
                         int guard_r, int train_d, int train_r) {
  int64_t max_idx = 0;
  for (int64_t d = 0; d < num_doppler; ++d) {
    for (int64_t r = 0; r < num_range; ++r) {
      int64_t rn = 0;
      int64_t rp = 0;
      int64_t dn = 0;
      int64_t dp = 0;
      go_training_side_counts(static_cast<int>(d), static_cast<int>(r),
                            num_doppler, num_range, guard_d, guard_r, train_d,
                            train_r, &rn, &rp, &dn, &dp);
      max_idx = std::max(max_idx, go_lut_index(rn, rp, dn, dp));
    }
  }
  return max_idx;
}

namespace {

constexpr int kGoMcCalTrials = 4096;
constexpr int kGoMcVerifyTrials = 8192;

double sample_gamma_k(int64_t k, std::mt19937_64& rng) {
  std::exponential_distribution<double> dist(1.0);
  double sum = 0.0;
  for (int64_t i = 0; i < k; ++i) {
    sum += dist(rng);
  }
  return sum;
}

double go_side_statistic_host(double neg_sum, int64_t neg_cnt, double pos_sum,
                              int64_t pos_cnt) {
  if (neg_cnt > 0 && pos_cnt > 0) {
    const double neg_mean = neg_sum / static_cast<double>(neg_cnt);
    const double pos_mean = pos_sum / static_cast<double>(pos_cnt);
    return std::max(neg_mean, pos_mean);
  }
  if (neg_cnt > 0) {
    return neg_sum / static_cast<double>(neg_cnt);
  }
  if (pos_cnt > 0) {
    return pos_sum / static_cast<double>(pos_cnt);
  }
  return 0.0;
}

double simulate_gogo_noise_interior(int guard_d, int guard_r, int train_d,
                                    int train_r, int64_t k,
                                    std::mt19937_64& rng) {
  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  double range_neg_sum = 0.0;
  double range_pos_sum = 0.0;
  double doppler_neg_sum = 0.0;
  double doppler_pos_sum = 0.0;
  int64_t range_neg_cnt = 0;
  int64_t range_pos_cnt = 0;
  int64_t doppler_neg_cnt = 0;
  int64_t doppler_pos_cnt = 0;
  for (int dd = -half_d; dd <= half_d; ++dd) {
    const bool guard_row = (dd >= -guard_d && dd <= guard_d);
    for (int rr = -half_r; rr <= half_r; ++rr) {
      if (guard_row && rr >= -guard_r && rr <= guard_r) {
        continue;
      }
      const double cell = sample_gamma_k(k, rng);
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
  const double range_go = go_side_statistic_host(
      range_neg_sum, range_neg_cnt, range_pos_sum, range_pos_cnt);
  const double doppler_go = go_side_statistic_host(
      doppler_neg_sum, doppler_neg_cnt, doppler_pos_sum, doppler_pos_cnt);
  return std::max(range_go, doppler_go);
}

double go_cfar_pfa_from_alpha_impl(double alpha, int64_t n_side, int64_t k,
                                   int mc_trials, int guard_d, int guard_r,
                                   int train_d, int train_r) {
  if (n_side <= 0 || k <= 0) {
    return 0.0;
  }
  if (alpha <= 0.0) {
    return 1.0;
  }
  std::mt19937_64 rng(0x474F43464152ULL ^ static_cast<uint64_t>(n_side) ^
                      (static_cast<uint64_t>(k) << 32));
  int64_t false_alarms = 0;
  for (int trial = 0; trial < mc_trials; ++trial) {
    const double noise = simulate_gogo_noise_interior(
        guard_d, guard_r, train_d, train_r, k, rng);
    const double cut = sample_gamma_k(k, rng);
    if (cut > alpha * noise) {
      ++false_alarms;
    }
  }
  return static_cast<double>(false_alarms) / static_cast<double>(mc_trials);
}

}  // namespace

double go_cfar_pfa_from_alpha(double alpha, int64_t n_side, int64_t k) {
  return go_cfar_pfa_from_alpha_impl(alpha, n_side, k, kGoMcVerifyTrials, 1, 1,
                                     2, 2);
}

double go_cfar_alpha(double pfa, int64_t n_side, int64_t k, int guard_d,
                     int guard_r, int train_d, int train_r) {
  if (n_side <= 0 || k <= 0) {
    throw std::invalid_argument("go_cfar_alpha: n_side and k must be positive");
  }
  if (pfa <= 0.0 || pfa >= 1.0) {
    throw std::invalid_argument(
        "go_cfar_alpha: pfa must be in (0, 1) exclusive");
  }

  // GO-CFAR is more conservative than CA on the same side count; seed the
  // bracket from the CA closed-form inversion so Monte Carlo noise at tiny Pfa
  // does not collapse alpha toward zero.
  double lo = ca_cfar_alpha(pfa, n_side, k);
  double hi = lo * 8.0;
  while (go_cfar_pfa_from_alpha_impl(hi, n_side, k, kGoMcCalTrials, guard_d,
                                     guard_r, train_d, train_r) > pfa) {
    hi *= 2.0;
    if (hi > 1e12) {
      throw std::runtime_error("go_cfar_alpha: failed to bracket alpha");
    }
  }
  while (lo > 0.0 &&
         go_cfar_pfa_from_alpha_impl(lo, n_side, k, kGoMcCalTrials, guard_d,
                                     guard_r, train_d, train_r) <= pfa) {
    lo *= 0.5;
  }

  for (int iter = 0; iter < 100; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (go_cfar_pfa_from_alpha_impl(mid, n_side, k, kGoMcCalTrials, guard_d,
                                     guard_r, train_d, train_r) > pfa) {
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

std::vector<double> build_go_alpha_lut(double pfa, int64_t fold,
                                       int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d,
                                       int train_r) {
  const int64_t max_n = max_go_lut_index(num_doppler, num_range, guard_d,
                                         guard_r, train_d, train_r);
  std::vector<double> lut(static_cast<size_t>(max_n + 1), 0.0);
  for (int64_t n = 1; n <= max_n; ++n) {
    lut[static_cast<size_t>(n)] =
        go_cfar_alpha(pfa, n, fold, guard_d, guard_r, train_d, train_r);
  }
  return lut;
}

}  // namespace holoscan::radar::detail
