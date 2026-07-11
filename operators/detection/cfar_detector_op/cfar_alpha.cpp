// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_alpha_mc.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace holoscan::radar::detail {

constexpr int kMcBisectionMaxIter = 100;

namespace {

constexpr double kEps = 1e-12;
constexpr int kMaxBetaIter = 200;
constexpr double kFpmin = 1e-30;

// Fill lut[n] for n in [1, max_n] by calling `calibrate(n)` on a pool of
// worker threads. Each calibration is independent and uses its own
// deterministically-seeded RNG, so results are identical regardless of how
// the indices are distributed across threads.
template <typename Calibrate>
void parallel_fill_lut(std::vector<double>& lut, int64_t max_n,
                       Calibrate&& calibrate) {
  if (max_n <= 0) {
    return;
  }
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    hw = 1;
  }
  const int64_t workers = std::min<int64_t>(static_cast<int64_t>(hw), max_n);
  if (workers <= 1) {
    for (int64_t n = 1; n <= max_n; ++n) {
      lut[static_cast<size_t>(n)] = calibrate(n);
    }
    return;
  }
  std::atomic<int64_t> next{1};
  const auto run = [&]() {
    for (;;) {
      const int64_t n = next.fetch_add(1);
      if (n > max_n) {
        break;
      }
      lut[static_cast<size_t>(n)] = calibrate(n);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers - 1));
  for (int64_t w = 0; w < workers - 1; ++w) {
    pool.emplace_back(run);
  }
  run();
  for (std::thread& t : pool) {
    t.join();
  }
}

// Fill lut[n] only for n in indices (sorted unique, all in [1, lut.size())).
template <typename Calibrate>
void parallel_fill_lut_indices(std::vector<double>& lut,
                               const std::vector<int64_t>& indices,
                               Calibrate&& calibrate) {
  if (indices.empty()) {
    return;
  }
  const size_t count = indices.size();
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    hw = 1;
  }
  const size_t workers =
      std::min<size_t>(static_cast<size_t>(hw), count);
  if (workers <= 1) {
    for (int64_t n : indices) {
      lut[static_cast<size_t>(n)] = calibrate(n);
    }
    return;
  }
  std::atomic<size_t> next{0};
  const auto run = [&]() {
    for (;;) {
      const size_t i = next.fetch_add(1);
      if (i >= count) {
        break;
      }
      const int64_t n = indices[i];
      lut[static_cast<size_t>(n)] = calibrate(n);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (size_t w = 0; w < workers - 1; ++w) {
    pool.emplace_back(run);
  }
  run();
  for (std::thread& t : pool) {
    t.join();
  }
}

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
  // The training count for a cell is (in-bounds window cells) - (in-bounds
  // guard+CUT box cells). Both are axis-separable rectangles, so clipping at a
  // border only ever reduces the count relative to a fully in-bounds cell.
  // The maximum therefore occurs at any interior cell whose whole window fits,
  // which exists exactly when the map is at least as large as the window in
  // each axis. In that (overwhelmingly common) case max_N is closed-form and
  // independent of map size; only sub-window maps need the exhaustive scan
  // (and there D*R is tiny, so the scan is cheap).
  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  const int64_t win_d = static_cast<int64_t>(2 * half_d + 1);
  const int64_t win_r = static_cast<int64_t>(2 * half_r + 1);
  if (num_doppler >= win_d && num_range >= win_r) {
    const int64_t window = win_d * win_r;
    const int64_t guard_box =
        static_cast<int64_t>(2 * guard_d + 1) * (2 * guard_r + 1);
    return window - guard_box;
  }
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

std::vector<int64_t> reachable_training_cell_counts(int num_doppler,
                                                    int num_range, int guard_d,
                                                    int guard_r, int train_d,
                                                    int train_r) {
  std::unordered_set<int64_t> seen;
  seen.reserve(64);
  for (int64_t d = 0; d < num_doppler; ++d) {
    for (int64_t r = 0; r < num_range; ++r) {
      const int64_t n = training_cell_count(
          static_cast<int>(d), static_cast<int>(r), num_doppler, num_range,
          guard_d, guard_r, train_d, train_r);
      if (n > 0) {
        seen.insert(n);
      }
    }
  }
  std::vector<int64_t> indices(seen.begin(), seen.end());
  std::sort(indices.begin(), indices.end());
  return indices;
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

  for (int iter = 0; iter < kMcBisectionMaxIter; ++iter) {
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

namespace {

struct GoLutReachability {
  int64_t max_index{0};
  std::vector<int64_t> reachable_indices;
  // First cell on the map that produces each LUT index (for cell-faithful MC).
  std::unordered_map<int64_t, std::pair<int, int>> representative_cell;
};

GoLutReachability go_lut_reachability(int num_doppler, int num_range,
                                      int guard_d, int guard_r, int train_d,
                                      int train_r) {
  std::unordered_set<int64_t> seen;
  seen.reserve(64);
  GoLutReachability out;
  for (int64_t d = 0; d < num_doppler; ++d) {
    for (int64_t r = 0; r < num_range; ++r) {
      int64_t rn = 0;
      int64_t rp = 0;
      int64_t dn = 0;
      int64_t dp = 0;
      go_training_side_counts(static_cast<int>(d), static_cast<int>(r),
                              num_doppler, num_range, guard_d, guard_r, train_d,
                              train_r, &rn, &rp, &dn, &dp);
      const int64_t idx = go_lut_index(rn, rp, dn, dp);
      out.max_index = std::max(out.max_index, idx);
      if (idx > 0 && seen.insert(idx).second) {
        out.representative_cell.emplace(
            idx, std::make_pair(static_cast<int>(d), static_cast<int>(r)));
      }
    }
  }
  out.reachable_indices.assign(seen.begin(), seen.end());
  std::sort(out.reachable_indices.begin(), out.reachable_indices.end());
  return out;
}

}  // namespace

int64_t max_go_lut_index(int num_doppler, int num_range, int guard_d,
                         int guard_r, int train_d, int train_r) {
  // Same interior-vs-border argument as max_training_cell_count: clipping at a
  // border never increases a GO/SO LUT index relative to a fully in-bounds
  // cell, so the maximum is the interior index whenever the map fits the
  // window. Sub-window maps fall back to a scan (D*R is tiny there).
  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  const int64_t win_d = static_cast<int64_t>(2 * half_d + 1);
  const int64_t win_r = static_cast<int64_t>(2 * half_r + 1);
  if (num_doppler >= win_d && num_range >= win_r) {
    // Interior side counts (axis-separable; matches go_training_side_counts
    // for a fully in-bounds CUT):
    //   range_neg = train_r * (2*half_d+1) + 2*guard_r*train_d
    //   doppler_neg = train_d * (2*half_r+1) + 2*guard_d*train_r
    // and range_pos == range_neg, doppler_pos == doppler_neg by symmetry.
    const int64_t range_side =
        static_cast<int64_t>(train_r) * win_d +
        2 * static_cast<int64_t>(guard_r) * train_d;
    const int64_t doppler_side =
        static_cast<int64_t>(train_d) * win_r +
        2 * static_cast<int64_t>(guard_d) * train_r;
    return go_lut_index(range_side, range_side, doppler_side, doppler_side);
  }
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

std::vector<int64_t> reachable_go_lut_indices(int num_doppler, int num_range,
                                              int guard_d, int guard_r,
                                              int train_d, int train_r) {
  return go_lut_reachability(num_doppler, num_range, guard_d, guard_r, train_d,
                             train_r)
      .reachable_indices;
}

namespace {

constexpr int kMcCalTrials = 4096;
constexpr int kMcVerifyTrials = 8192;

int mc_trials_for_calibration(int64_t k) {
  return k > 1 ? kMcVerifyTrials : kMcCalTrials;
}

int64_t effective_os_rank(int64_t n, int64_t os_rank) {
  return os_rank < n ? os_rank : n - 1;
}

// OS-CFAR K=1 closed form: pooled training cells, ascending 0-based rank.
//   Pfa = prod_{i=0}^{rank} (N - i) / (N - i + alpha)
// Matches cfar_os_kernel when each cell is Exp(1) under homogeneous noise.
double os_cfar_pfa_closed_k1(double alpha, int64_t n, int64_t os_rank) {
  if (n <= 0) {
    return 0.0;
  }
  if (alpha <= 0.0) {
    return 1.0;
  }
  const int64_t rank = effective_os_rank(n, os_rank);
  double pfa = 1.0;
  for (int64_t i = 0; i <= rank; ++i) {
    pfa *= static_cast<double>(n - i) /
           (static_cast<double>(n - i) + alpha);
  }
  return pfa;
}

double os_cfar_alpha_closed_k1(double pfa, int64_t n, int64_t os_rank) {
  double lo = ca_cfar_alpha(pfa, n, 1);
  double hi = lo * 8.0;
  while (os_cfar_pfa_closed_k1(hi, n, os_rank) > pfa) {
    hi *= 2.0;
    if (hi > 1e12) {
      throw std::runtime_error("os_cfar_alpha_closed_k1: failed to bracket");
    }
  }
  while (lo > 0.0 && os_cfar_pfa_closed_k1(lo, n, os_rank) <= pfa) {
    lo *= 0.5;
  }
  for (int iter = 0; iter < kMcBisectionMaxIter; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (os_cfar_pfa_closed_k1(mid, n, os_rank) > pfa) {
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

double sample_gamma_k(int64_t k, std::mt19937_64& rng) {
  std::exponential_distribution<double> dist(1.0);
  double sum = 0.0;
  for (int64_t i = 0; i < k; ++i) {
    sum += dist(rng);
  }
  return sum;
}

template <bool kGreatestOf>
double goso_side_statistic_host(double neg_sum, int64_t neg_cnt, double pos_sum,
                                int64_t pos_cnt) {
  if (neg_cnt > 0 && pos_cnt > 0) {
    const double neg_mean = neg_sum / static_cast<double>(neg_cnt);
    const double pos_mean = pos_sum / static_cast<double>(pos_cnt);
    if (kGreatestOf) {
      return std::max(neg_mean, pos_mean);
    }
    return std::min(neg_mean, pos_mean);
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
double simulate_goso_noise_nsides(int64_t n_side, int64_t k,
                                  std::mt19937_64& rng) {
  auto side_mean = [&]() {
    double sum = 0.0;
    for (int64_t i = 0; i < n_side; ++i) {
      sum += sample_gamma_k(k, rng);
    }
    return sum / static_cast<double>(n_side);
  };
  const double range_stat = goso_side_statistic_host<kGreatestOf>(
      side_mean(), 1, side_mean(), 1);
  const double doppler_stat = goso_side_statistic_host<kGreatestOf>(
      side_mean(), 1, side_mean(), 1);
  if (kGreatestOf) {
    return std::max(range_stat, doppler_stat);
  }
  return std::min(range_stat, doppler_stat);
}

template <bool kGreatestOf>
double simulate_goso_noise_at_cell(int d, int r, int num_doppler, int num_range,
                                   int guard_d, int guard_r, int train_d,
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
  const double range_stat = goso_side_statistic_host<kGreatestOf>(
      range_neg_sum, range_neg_cnt, range_pos_sum, range_pos_cnt);
  const double doppler_stat = goso_side_statistic_host<kGreatestOf>(
      doppler_neg_sum, doppler_neg_cnt, doppler_pos_sum, doppler_pos_cnt);
  if (kGreatestOf) {
    return std::max(range_stat, doppler_stat);
  }
  return std::min(range_stat, doppler_stat);
}

double goso_cfar_pfa_from_alpha_impl(double alpha, int64_t n_side, int64_t k,
                                     int mc_trials, bool greatest_of) {
  if (n_side <= 0 || k <= 0) {
    return 0.0;
  }
  if (alpha <= 0.0) {
    return 1.0;
  }
  // Prefer GPU MC when CUDA is available. The GPU path uses an LCG; the host
  // fallback uses mt19937_64 — LUT entries can differ slightly between paths
  // (and between GPU vs CPU-only machines) but both stay within MC tolerance.
  if (gpu_mc_calibration_available()) {
    const double gpu_pfa = goso_mc_pfa_from_alpha_gpu(
        alpha, n_side, k, mc_trials, /*guard_d=*/0, /*guard_r=*/0,
        /*train_d=*/0, /*train_r=*/0, greatest_of);
    if (gpu_pfa >= 0.0) {
      return gpu_pfa;
    }
  }
  std::mt19937_64 rng(0x474F43464152ULL ^ static_cast<uint64_t>(n_side) ^
                      (static_cast<uint64_t>(k) << 32) ^
                      (greatest_of ? 0ULL : 0x534F43464152ULL));
  int64_t false_alarms = 0;
  for (int trial = 0; trial < mc_trials; ++trial) {
    const double noise =
        greatest_of ? simulate_goso_noise_nsides<true>(n_side, k, rng)
                    : simulate_goso_noise_nsides<false>(n_side, k, rng);
    const double cut = sample_gamma_k(k, rng);
    if (cut > alpha * noise) {
      ++false_alarms;
    }
  }
  return static_cast<double>(false_alarms) / static_cast<double>(mc_trials);
}

double goso_cfar_pfa_at_cell_impl(double alpha, int64_t k, int mc_trials, int d,
                                  int r, int num_doppler, int num_range,
                                  int guard_d, int guard_r, int train_d,
                                  int train_r, bool greatest_of) {
  if (k <= 0) {
    return 0.0;
  }
  if (alpha <= 0.0) {
    return 1.0;
  }
  if (gpu_mc_calibration_available()) {
    const double gpu_pfa = goso_mc_pfa_at_cell_gpu(
        alpha, k, mc_trials, d, r, num_doppler, num_range, guard_d, guard_r,
        train_d, train_r, greatest_of);
    if (gpu_pfa >= 0.0) {
      return gpu_pfa;
    }
  }
  std::mt19937_64 rng(0x474F43464152ULL ^ (static_cast<uint64_t>(d) << 32) ^
                      static_cast<uint64_t>(r) ^
                      (static_cast<uint64_t>(k) << 16) ^
                      (greatest_of ? 0ULL : 0x534F43464152ULL));
  int64_t false_alarms = 0;
  for (int trial = 0; trial < mc_trials; ++trial) {
    const double noise =
        greatest_of
            ? simulate_goso_noise_at_cell<true>(d, r, num_doppler, num_range,
                                                guard_d, guard_r, train_d,
                                                train_r, k, rng)
            : simulate_goso_noise_at_cell<false>(d, r, num_doppler, num_range,
                                                 guard_d, guard_r, train_d,
                                                 train_r, k, rng);
    const double cut = sample_gamma_k(k, rng);
    if (cut > alpha * noise) {
      ++false_alarms;
    }
  }
  return static_cast<double>(false_alarms) / static_cast<double>(mc_trials);
}

double go_cfar_pfa_from_alpha_impl(double alpha, int64_t n_side, int64_t k,
                                   int mc_trials, int /*guard_d*/,
                                   int /*guard_r*/, int /*train_d*/,
                                   int /*train_r*/) {
  return goso_cfar_pfa_from_alpha_impl(alpha, n_side, k, mc_trials, true);
}

double so_cfar_pfa_from_alpha_impl(double alpha, int64_t n_side, int64_t k,
                                   int mc_trials, int /*guard_d*/,
                                   int /*guard_r*/, int /*train_d*/,
                                   int /*train_r*/) {
  return goso_cfar_pfa_from_alpha_impl(alpha, n_side, k, mc_trials, false);
}

double os_select_rank_host(std::vector<double>& samples, int64_t os_rank) {
  if (samples.empty()) {
    return 0.0;
  }
  const int64_t rank =
      os_rank < static_cast<int64_t>(samples.size())
          ? os_rank
          : static_cast<int64_t>(samples.size()) - 1;
  std::nth_element(samples.begin(), samples.begin() + rank, samples.end());
  return samples[static_cast<size_t>(rank)];
}

double simulate_os_noise_interior(int guard_d, int guard_r, int train_d,
                                  int train_r, int64_t os_rank, int64_t k,
                                  std::mt19937_64& rng) {
  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>((2 * half_d + 1) * (2 * half_r + 1)));
  for (int dd = -half_d; dd <= half_d; ++dd) {
    const bool guard_row = (dd >= -guard_d && dd <= guard_d);
    for (int rr = -half_r; rr <= half_r; ++rr) {
      if (guard_row && rr >= -guard_r && rr <= guard_r) {
        continue;
      }
      samples.push_back(sample_gamma_k(k, rng));
    }
  }
  return os_select_rank_host(samples, os_rank);
}

double os_cfar_pfa_from_alpha_impl(double alpha, int64_t n, int64_t os_rank,
                                   int64_t k, int mc_trials, int guard_d,
                                   int guard_r, int train_d, int train_r) {
  if (n <= 0 || k <= 0) {
    return 0.0;
  }
  if (alpha <= 0.0) {
    return 1.0;
  }
  const int64_t effective_rank = effective_os_rank(n, os_rank);
  if (k == 1) {
    return os_cfar_pfa_closed_k1(alpha, n, effective_rank);
  }
  // Same GPU-vs-host RNG note as goso_cfar_pfa_from_alpha_impl below.
  if (gpu_mc_calibration_available()) {
    const double gpu_pfa = os_mc_pfa_from_alpha_gpu(
        alpha, n, effective_rank, k, mc_trials, guard_d, guard_r, train_d,
        train_r);
    if (gpu_pfa >= 0.0) {
      return gpu_pfa;
    }
  }
  std::mt19937_64 rng(0x4F5343464152ULL ^ static_cast<uint64_t>(n) ^
                      (static_cast<uint64_t>(effective_rank) << 16) ^
                      (static_cast<uint64_t>(k) << 32));
  int64_t false_alarms = 0;
  for (int trial = 0; trial < mc_trials; ++trial) {
    const double noise = simulate_os_noise_interior(
        guard_d, guard_r, train_d, train_r, effective_rank, k, rng);
    const double cut = sample_gamma_k(k, rng);
    if (cut > alpha * noise) {
      ++false_alarms;
    }
  }
  return static_cast<double>(false_alarms) / static_cast<double>(mc_trials);
}

}  // namespace

double go_cfar_pfa_from_alpha(double alpha, int64_t n_side, int64_t k) {
  return go_cfar_pfa_from_alpha_impl(alpha, n_side, k, kMcVerifyTrials, 1, 1,
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
  const int mc_trials = mc_trials_for_calibration(k);
  while (go_cfar_pfa_from_alpha_impl(hi, n_side, k, mc_trials, guard_d,
                                     guard_r, train_d, train_r) > pfa) {
    hi *= 2.0;
    if (hi > 1e12) {
      throw std::runtime_error("go_cfar_alpha: failed to bracket alpha");
    }
  }
  while (lo > 0.0 &&
         go_cfar_pfa_from_alpha_impl(lo, n_side, k, mc_trials, guard_d,
                                     guard_r, train_d, train_r) <= pfa) {
    lo *= 0.5;
  }

  for (int iter = 0; iter < kMcBisectionMaxIter; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (go_cfar_pfa_from_alpha_impl(mid, n_side, k, mc_trials, guard_d,
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

double go_cfar_alpha_at_cell(double pfa, int64_t k, int d, int r,
                             int num_doppler, int num_range, int guard_d,
                             int guard_r, int train_d, int train_r) {
  if (k <= 0) {
    throw std::invalid_argument("go_cfar_alpha_at_cell: k must be positive");
  }
  if (pfa <= 0.0 || pfa >= 1.0) {
    throw std::invalid_argument(
        "go_cfar_alpha_at_cell: pfa must be in (0, 1) exclusive");
  }
  int64_t rn = 0;
  int64_t rp = 0;
  int64_t dn = 0;
  int64_t dp = 0;
  go_training_side_counts(d, r, num_doppler, num_range, guard_d, guard_r,
                          train_d, train_r, &rn, &rp, &dn, &dp);
  const int64_t n_side = go_lut_index(rn, rp, dn, dp);
  if (n_side <= 0) {
    throw std::invalid_argument(
        "go_cfar_alpha_at_cell: cell has no training sides");
  }
  double lo = ca_cfar_alpha(pfa, n_side, k);
  double hi = lo * 8.0;
  const int mc_trials = mc_trials_for_calibration(k);
  auto pfa_at = [&](double alpha) {
    return goso_cfar_pfa_at_cell_impl(alpha, k, mc_trials, d, r, num_doppler,
                                      num_range, guard_d, guard_r, train_d,
                                      train_r, true);
  };
  while (pfa_at(hi) > pfa) {
    hi *= 2.0;
    if (hi > 1e12) {
      throw std::runtime_error("go_cfar_alpha_at_cell: failed to bracket alpha");
    }
  }
  while (lo > 0.0 && pfa_at(lo) <= pfa) {
    lo *= 0.5;
  }
  for (int iter = 0; iter < kMcBisectionMaxIter; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (pfa_at(mid) > pfa) {
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
  const GoLutReachability reach = go_lut_reachability(
      num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  std::vector<double> lut(static_cast<size_t>(reach.max_index + 1), 0.0);
  parallel_fill_lut_indices(lut, reach.reachable_indices, [&](int64_t n) {
    const auto it = reach.representative_cell.find(n);
    if (it == reach.representative_cell.end()) {
      return go_cfar_alpha(pfa, n, fold, guard_d, guard_r, train_d, train_r);
    }
    return go_cfar_alpha_at_cell(pfa, fold, it->second.first, it->second.second,
                                 num_doppler, num_range, guard_d, guard_r,
                                 train_d, train_r);
  });
  return lut;
}

double so_cfar_pfa_from_alpha(double alpha, int64_t n_side, int64_t k) {
  return so_cfar_pfa_from_alpha_impl(alpha, n_side, k, kMcVerifyTrials, 1, 1,
                                     2, 2);
}

double so_cfar_alpha(double pfa, int64_t n_side, int64_t k, int guard_d,
                     int guard_r, int train_d, int train_r) {
  if (n_side <= 0 || k <= 0) {
    throw std::invalid_argument("so_cfar_alpha: n_side and k must be positive");
  }
  if (pfa <= 0.0 || pfa >= 1.0) {
    throw std::invalid_argument(
        "so_cfar_alpha: pfa must be in (0, 1) exclusive");
  }

  // SO-CFAR is less conservative than CA; seed the bracket from the CA
  // closed-form inversion (same strategy as GO and OS). SO alpha > CA alpha,
  // so ca_cfar_alpha is a valid lower bracket; avoids nesting a full GO MC
  // inversion per LUT index.
  double lo = ca_cfar_alpha(pfa, n_side, k);
  double hi = lo * 8.0;
  const int mc_trials = mc_trials_for_calibration(k);
  while (so_cfar_pfa_from_alpha_impl(hi, n_side, k, mc_trials, guard_d,
                                     guard_r, train_d, train_r) > pfa) {
    hi *= 2.0;
    if (hi > 1e12) {
      throw std::runtime_error("so_cfar_alpha: failed to bracket alpha");
    }
  }
  while (lo > 0.0 &&
         so_cfar_pfa_from_alpha_impl(lo, n_side, k, mc_trials, guard_d,
                                     guard_r, train_d, train_r) <= pfa) {
    lo *= 0.5;
  }

  for (int iter = 0; iter < kMcBisectionMaxIter; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (so_cfar_pfa_from_alpha_impl(mid, n_side, k, mc_trials, guard_d,
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

double so_cfar_alpha_at_cell(double pfa, int64_t k, int d, int r,
                             int num_doppler, int num_range, int guard_d,
                             int guard_r, int train_d, int train_r) {
  if (k <= 0) {
    throw std::invalid_argument("so_cfar_alpha_at_cell: k must be positive");
  }
  if (pfa <= 0.0 || pfa >= 1.0) {
    throw std::invalid_argument(
        "so_cfar_alpha_at_cell: pfa must be in (0, 1) exclusive");
  }
  int64_t rn = 0;
  int64_t rp = 0;
  int64_t dn = 0;
  int64_t dp = 0;
  go_training_side_counts(d, r, num_doppler, num_range, guard_d, guard_r,
                          train_d, train_r, &rn, &rp, &dn, &dp);
  const int64_t n_side = go_lut_index(rn, rp, dn, dp);
  if (n_side <= 0) {
    throw std::invalid_argument(
        "so_cfar_alpha_at_cell: cell has no training sides");
  }
  double lo = ca_cfar_alpha(pfa, n_side, k);
  double hi = lo * 8.0;
  const int mc_trials = mc_trials_for_calibration(k);
  auto pfa_at = [&](double alpha) {
    return goso_cfar_pfa_at_cell_impl(alpha, k, mc_trials, d, r, num_doppler,
                                      num_range, guard_d, guard_r, train_d,
                                      train_r, false);
  };
  while (pfa_at(hi) > pfa) {
    hi *= 2.0;
    if (hi > 1e12) {
      throw std::runtime_error("so_cfar_alpha_at_cell: failed to bracket alpha");
    }
  }
  while (lo > 0.0 && pfa_at(lo) <= pfa) {
    lo *= 0.5;
  }
  for (int iter = 0; iter < kMcBisectionMaxIter; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (pfa_at(mid) > pfa) {
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

std::vector<double> build_so_alpha_lut(double pfa, int64_t fold,
                                       int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d,
                                       int train_r) {
  const GoLutReachability reach = go_lut_reachability(
      num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  std::vector<double> lut(static_cast<size_t>(reach.max_index + 1), 0.0);
  parallel_fill_lut_indices(lut, reach.reachable_indices, [&](int64_t n) {
    const auto it = reach.representative_cell.find(n);
    if (it == reach.representative_cell.end()) {
      return so_cfar_alpha(pfa, n, fold, guard_d, guard_r, train_d, train_r);
    }
    return so_cfar_alpha_at_cell(pfa, fold, it->second.first, it->second.second,
                                 num_doppler, num_range, guard_d, guard_r,
                                 train_d, train_r);
  });
  return lut;
}

double os_cfar_pfa_from_alpha(double alpha, int64_t n, int64_t os_rank,
                              int64_t k) {
  return os_cfar_pfa_from_alpha_impl(alpha, n, os_rank, k, kMcVerifyTrials, 1,
                                     1, 2, 2);
}

double os_cfar_alpha(double pfa, int64_t n, int64_t os_rank, int64_t k,
                     int guard_d, int guard_r, int train_d, int train_r) {
  if (n <= 0 || k <= 0) {
    throw std::invalid_argument("os_cfar_alpha: n and k must be positive");
  }
  if (os_rank < 0) {
    throw std::invalid_argument("os_cfar_alpha: os_rank must be non-negative");
  }
  if (pfa <= 0.0 || pfa >= 1.0) {
    throw std::invalid_argument(
        "os_cfar_alpha: pfa must be in (0, 1) exclusive");
  }

  if (k == 1) {
    return os_cfar_alpha_closed_k1(pfa, n, os_rank);
  }

  double lo = ca_cfar_alpha(pfa, n, k);
  double hi = lo * 8.0;
  const int mc_trials = mc_trials_for_calibration(k);
  while (os_cfar_pfa_from_alpha_impl(hi, n, os_rank, k, mc_trials, guard_d,
                                     guard_r, train_d, train_r) > pfa) {
    hi *= 2.0;
    if (hi > 1e12) {
      throw std::runtime_error("os_cfar_alpha: failed to bracket alpha");
    }
  }
  while (lo > 0.0 &&
         os_cfar_pfa_from_alpha_impl(lo, n, os_rank, k, mc_trials, guard_d,
                                     guard_r, train_d, train_r) <= pfa) {
    lo *= 0.5;
  }

  for (int iter = 0; iter < kMcBisectionMaxIter; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (os_cfar_pfa_from_alpha_impl(mid, n, os_rank, k, mc_trials, guard_d,
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

std::vector<double> build_os_alpha_lut(double pfa, int64_t fold,
                                       int64_t os_rank, int num_doppler,
                                       int num_range, int guard_d, int guard_r,
                                       int train_d, int train_r) {
  const int64_t max_n = max_training_cell_count(
      num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  std::vector<double> lut(static_cast<size_t>(max_n + 1), 0.0);
  if (fold == 1) {
    // Closed-form inversion per N; no map scan — calibrate the full 1..max_n
    // range cheaply (reachable-only would add a costly D×R scan for no gain).
    for (int64_t n = 1; n <= max_n; ++n) {
      const int64_t effective_rank = effective_os_rank(n, os_rank);
      lut[static_cast<size_t>(n)] =
          os_cfar_alpha_closed_k1(pfa, n, effective_rank);
    }
    return lut;
  }
  const std::vector<int64_t> indices = reachable_training_cell_counts(
      num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  parallel_fill_lut_indices(lut, indices, [&](int64_t n) {
    const int64_t effective_rank = effective_os_rank(n, os_rank);
    return os_cfar_alpha(pfa, n, effective_rank, fold, guard_d, guard_r,
                         train_d, train_r);
  });
  return lut;
}

struct CfarAlphaLutKeyHash {
  size_t operator()(const CfarAlphaLutKey& key) const {
    auto h = std::hash<int>{}(static_cast<int>(key.variant));
    auto mix = [](size_t seed, size_t value) {
      return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    };
    uint64_t pfa_bits = 0;
    static_assert(sizeof(pfa_bits) == sizeof(key.pfa));
    std::memcpy(&pfa_bits, &key.pfa, sizeof(key.pfa));
    h = mix(h, pfa_bits);
    h = mix(h, static_cast<size_t>(key.fold));
    h = mix(h, static_cast<size_t>(key.os_rank));
    h = mix(h, static_cast<size_t>(key.guard_d));
    h = mix(h, static_cast<size_t>(key.guard_r));
    h = mix(h, static_cast<size_t>(key.train_d));
    h = mix(h, static_cast<size_t>(key.train_r));
    h = mix(h, static_cast<size_t>(key.num_doppler));
    h = mix(h, static_cast<size_t>(key.num_range));
    return h;
  }
};

using AlphaLut = std::vector<double>;
using AlphaLutFuture = std::shared_future<AlphaLut>;

// Bound on the process-wide LUT cache. Each entry is tiny (a few hundred
// doubles) and distinct keys are rare in practice (one per (variant, pfa,
// footprint, os_rank, map size) combination), so a generous FIFO cap keeps
// a long-running pipeline that sweeps many configurations from growing the
// cache without bound.
constexpr size_t kMaxCachedAlphaLuts = 64;

std::mutex g_alpha_lut_cache_mutex;
std::unordered_map<CfarAlphaLutKey, AlphaLut, CfarAlphaLutKeyHash>
    g_alpha_lut_cache;  // NOLINT(whitespace/indent_namespace)
// Insertion order for FIFO eviction once the cache exceeds its cap.
std::deque<CfarAlphaLutKey>
    g_alpha_lut_order;  // NOLINT(whitespace/indent_namespace)
// Keys whose LUT is currently being built. Concurrent callers for the same
// key wait on the shared future instead of duplicating the (expensive) MC
// calibration.
std::unordered_map<CfarAlphaLutKey, AlphaLutFuture, CfarAlphaLutKeyHash>
    g_alpha_lut_in_flight;  // NOLINT(whitespace/indent_namespace)

std::vector<double> build_alpha_lut(const CfarAlphaLutKey& key) {
  switch (key.variant) {
    case CfarVariant::kCA:
      return build_ca_alpha_lut(key.pfa, key.fold, key.num_doppler,
                                key.num_range, key.guard_d, key.guard_r,
                                key.train_d, key.train_r);
    case CfarVariant::kGO:
      return build_go_alpha_lut(key.pfa, key.fold, key.num_doppler,
                                key.num_range, key.guard_d, key.guard_r,
                                key.train_d, key.train_r);
    case CfarVariant::kSO:
      return build_so_alpha_lut(key.pfa, key.fold, key.num_doppler,
                                key.num_range, key.guard_d, key.guard_r,
                                key.train_d, key.train_r);
    case CfarVariant::kOS:
      return build_os_alpha_lut(key.pfa, key.fold, key.os_rank,
                                key.num_doppler, key.num_range, key.guard_d,
                                key.guard_r, key.train_d, key.train_r);
    default:
      throw std::invalid_argument(
          std::string("get_cached_alpha_lut: variant '") +
          std::string(to_string(key.variant)) + "' is not implemented");
  }
}

void commit_cache_entry(const CfarAlphaLutKey& key, AlphaLut built) {
  std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
  if (g_alpha_lut_cache.emplace(key, built).second) {
    g_alpha_lut_order.push_back(key);
  }
  g_alpha_lut_in_flight.erase(key);
  while (g_alpha_lut_cache.size() > kMaxCachedAlphaLuts &&
         !g_alpha_lut_order.empty()) {
    const CfarAlphaLutKey oldest = g_alpha_lut_order.front();
    if (oldest == key) {
      break;
    }
    g_alpha_lut_cache.erase(oldest);
    g_alpha_lut_order.pop_front();
  }
}

void run_lut_build(const CfarAlphaLutKey& key,
                   const std::shared_ptr<std::promise<AlphaLut>>& promise) {
  try {
    AlphaLut built = build_alpha_lut(key);
    commit_cache_entry(key, built);
    promise->set_value(std::move(built));
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
      g_alpha_lut_in_flight.erase(key);
    }
    promise->set_exception(std::current_exception());
  }
}

// Look up `key` under the cache lock. If a build for this key just finished
// between the cache miss and the in-flight miss, a second cache check closes
// that race without waiting on unrelated keys.
bool try_get_cached_lut_locked(const CfarAlphaLutKey& key, AlphaLut* out) {
  const auto it = g_alpha_lut_cache.find(key);
  if (it == g_alpha_lut_cache.end()) {
    return false;
  }
  *out = it->second;
  return true;
}

AlphaLutSnapshot poll_cached_alpha_lut(const CfarAlphaLutKey& key) {
  AlphaLut cached;
  AlphaLutFuture future;
  bool have_future = false;
  {
    std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
    if (try_get_cached_lut_locked(key, &cached)) {
      return {true, std::move(cached)};
    }
    const auto in_flight = g_alpha_lut_in_flight.find(key);
    if (in_flight != g_alpha_lut_in_flight.end()) {
      future = in_flight->second;
      have_future = true;
    } else if (try_get_cached_lut_locked(key, &cached)) {
      // Builder committed and erased in-flight between the two lookups.
      return {true, std::move(cached)};
    }
  }
  if (!have_future) {
    return {};
  }
  // Non-blocking: only this key's future; never wait on other builds.
  if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    return {};
  }
  return {true, future.get()};
}

void ensure_alpha_lut_build(const CfarAlphaLutKey& key) {
  std::shared_ptr<std::promise<AlphaLut>> promise;
  {
    std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
    if (g_alpha_lut_cache.find(key) != g_alpha_lut_cache.end()) {
      return;
    }
    if (g_alpha_lut_in_flight.find(key) != g_alpha_lut_in_flight.end()) {
      return;
    }
    promise = std::make_shared<std::promise<AlphaLut>>();
    g_alpha_lut_in_flight.emplace(key, promise->get_future().share());
  }
  std::thread([key, promise]() { run_lut_build(key, promise); }).detach();
}

std::vector<double> get_cached_alpha_lut(const CfarAlphaLutKey& key) {
  std::shared_ptr<std::promise<AlphaLut>> promise;
  AlphaLutFuture future;
  AlphaLut cached;
  {
    std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
    if (try_get_cached_lut_locked(key, &cached)) {
      return cached;
    }
    const auto in_flight = g_alpha_lut_in_flight.find(key);
    if (in_flight != g_alpha_lut_in_flight.end()) {
      future = in_flight->second;
    } else if (try_get_cached_lut_locked(key, &cached)) {
      return cached;
    } else {
      promise = std::make_shared<std::promise<AlphaLut>>();
      future = promise->get_future().share();
      g_alpha_lut_in_flight.emplace(key, future);
    }
  }

  if (!promise) {
    // Wait only on this key's in-flight build.
    return future.get();
  }

  run_lut_build(key, promise);
  return future.get();
}

}  // namespace holoscan::radar::detail
