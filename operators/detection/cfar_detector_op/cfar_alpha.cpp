// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_calibration.hpp"
#include "detection/cfar_detector_op/cfar_alpha_mc.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <set>
#include <stop_token>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
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
void parallel_fill_lut(std::vector<double>& lut, int64_t max_n, Calibrate&& calibrate) {
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
void parallel_fill_lut_indices(std::vector<double>& lut, const std::vector<int64_t>& indices,
                               Calibrate&& calibrate) {
  if (indices.empty()) {
    return;
  }
  const size_t count = indices.size();
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    hw = 1;
  }
  const size_t workers = std::min<size_t>(static_cast<size_t>(hw), count);
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

double log_gamma(double z) {
  return std::lgamma(z);
}

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
  const double bt = std::exp(log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * std::log(x) +
                             b * std::log(1.0 - x));
  const bool symm = x < (a + 1.0) / (a + b + 2.0);
  if (symm) {
    return bt * incomplete_beta_cf(a, b, x) / a;
  }
  return 1.0 - bt * incomplete_beta_cf(b, a, 1.0 - x) / b;
}

}  // namespace

int64_t training_cell_count(int d, int r, int num_doppler, int num_range, int guard_d, int guard_r,
                            int train_d, int train_r) {
  // The in-bounds training footprint is the clipped window rectangle minus
  // the clipped guard+CUT rectangle. Count each axis without visiting offsets.
  const auto extent = [](int position, int size, int64_t radius) -> int64_t {
    return 1 + std::min<int64_t>(position, radius) +
           std::min<int64_t>(static_cast<int64_t>(size) - 1 - position, radius);
  };
  return extent(d, num_doppler, static_cast<int64_t>(guard_d) + train_d) *
             extent(r, num_range, static_cast<int64_t>(guard_r) + train_r) -
         extent(d, num_doppler, guard_d) * extent(r, num_range, guard_r);
}

int64_t max_training_cell_count(int num_doppler, int num_range, int guard_d, int guard_r,
                                int train_d, int train_r) {
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
    const int64_t guard_box = static_cast<int64_t>(2 * guard_d + 1) * (2 * guard_r + 1);
    return window - guard_box;
  }
  int64_t max_cnt = 0;
  for (int64_t d = 0; d < num_doppler; ++d) {
    for (int64_t r = 0; r < num_range; ++r) {
      max_cnt = std::max(max_cnt,
                         training_cell_count(static_cast<int>(d), static_cast<int>(r), num_doppler,
                                             num_range, guard_d, guard_r, train_d, train_r));
    }
  }
  return max_cnt;
}

std::vector<int64_t> reachable_training_cell_counts(int num_doppler, int num_range, int guard_d,
                                                    int guard_r, int train_d, int train_r) {
  std::unordered_set<int64_t> seen;
  seen.reserve(64);
  for (int64_t d = 0; d < num_doppler; ++d) {
    for (int64_t r = 0; r < num_range; ++r) {
      const int64_t n = training_cell_count(static_cast<int>(d), static_cast<int>(r), num_doppler,
                                            num_range, guard_d, guard_r, train_d, train_r);
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
  return regularized_incomplete_beta(static_cast<double>(n * k), static_cast<double>(k), x);
}

double ca_cfar_alpha(double pfa, int64_t n, int64_t k) {
  if (n <= 0 || k <= 0) {
    throw std::invalid_argument("ca_cfar_alpha: n and k must be positive");
  }
  if (pfa <= 0.0 || pfa >= 1.0) {
    throw std::invalid_argument("ca_cfar_alpha: pfa must be in (0, 1) exclusive");
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

std::vector<double> build_ca_alpha_lut(double pfa, int64_t fold, int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d, int train_r) {
  const int64_t max_n =
      max_training_cell_count(num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  std::vector<double> lut(static_cast<size_t>(max_n + 1), 0.0);
  for (int64_t n = 1; n <= max_n; ++n) {
    lut[static_cast<size_t>(n)] = ca_cfar_alpha(pfa, n, fold);
  }
  return lut;
}

void go_training_side_counts(int d, int r, int num_doppler, int num_range, int guard_d, int guard_r,
                             int train_d, int train_r, int64_t* range_neg, int64_t* range_pos,
                             int64_t* doppler_neg, int64_t* doppler_pos) {
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

struct GoSideCounts {
  int64_t range_neg{0};
  int64_t range_pos{0};
  int64_t doppler_neg{0};
  int64_t doppler_pos{0};

  bool operator==(const GoSideCounts& other) const {
    return range_neg == other.range_neg && range_pos == other.range_pos &&
           doppler_neg == other.doppler_neg && doppler_pos == other.doppler_pos;
  }
};

struct GoSideCountsHash {
  size_t operator()(const GoSideCounts& s) const {
    auto mix = [](size_t seed, size_t value) {
      return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    };
    size_t h = mix(0, static_cast<size_t>(s.range_neg));
    h = mix(h, static_cast<size_t>(s.range_pos));
    h = mix(h, static_cast<size_t>(s.doppler_neg));
    h = mix(h, static_cast<size_t>(s.doppler_pos));
    return h;
  }
};

struct GoLutReachability {
  int64_t max_index{0};
  std::vector<int64_t> reachable_indices;
  // One representative cell per distinct side-count geometry that maps to
  // each LUT index. Compact go_lut_index aliases many geometries onto one
  // slot; calibrating only the first cell leaves the others mis-scaled.
  std::unordered_map<int64_t, std::vector<std::pair<int, int>>> representative_cells;
};

GoLutReachability go_lut_reachability(int num_doppler, int num_range, int guard_d, int guard_r,
                                      int train_d, int train_r) {
  std::unordered_set<int64_t> seen;
  seen.reserve(64);
  std::unordered_map<int64_t, std::unordered_set<GoSideCounts, GoSideCountsHash>> seen_geometries;
  GoLutReachability out;
  for (int64_t d = 0; d < num_doppler; ++d) {
    for (int64_t r = 0; r < num_range; ++r) {
      int64_t rn = 0;
      int64_t rp = 0;
      int64_t dn = 0;
      int64_t dp = 0;
      go_training_side_counts(static_cast<int>(d), static_cast<int>(r), num_doppler, num_range,
                              guard_d, guard_r, train_d, train_r, &rn, &rp, &dn, &dp);
      const int64_t idx = go_lut_index(rn, rp, dn, dp);
      out.max_index = std::max(out.max_index, idx);
      if (idx <= 0) {
        continue;
      }
      seen.insert(idx);
      const GoSideCounts sides{rn, rp, dn, dp};
      if (seen_geometries[idx].insert(sides).second) {
        out.representative_cells[idx].emplace_back(static_cast<int>(d), static_cast<int>(r));
      }
    }
  }
  out.reachable_indices.assign(seen.begin(), seen.end());
  std::sort(out.reachable_indices.begin(), out.reachable_indices.end());
  return out;
}

}  // namespace

int64_t max_go_lut_index(int num_doppler, int num_range, int guard_d, int guard_r, int train_d,
                         int train_r) {
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
        static_cast<int64_t>(train_r) * win_d + 2 * static_cast<int64_t>(guard_r) * train_d;
    const int64_t doppler_side =
        static_cast<int64_t>(train_d) * win_r + 2 * static_cast<int64_t>(guard_d) * train_r;
    return go_lut_index(range_side, range_side, doppler_side, doppler_side);
  }
  int64_t max_idx = 0;
  for (int64_t d = 0; d < num_doppler; ++d) {
    for (int64_t r = 0; r < num_range; ++r) {
      int64_t rn = 0;
      int64_t rp = 0;
      int64_t dn = 0;
      int64_t dp = 0;
      go_training_side_counts(static_cast<int>(d), static_cast<int>(r), num_doppler, num_range,
                              guard_d, guard_r, train_d, train_r, &rn, &rp, &dn, &dp);
      max_idx = std::max(max_idx, go_lut_index(rn, rp, dn, dp));
    }
  }
  return max_idx;
}

std::vector<int64_t> reachable_go_lut_indices(int num_doppler, int num_range, int guard_d,
                                              int guard_r, int train_d, int train_r) {
  return go_lut_reachability(num_doppler, num_range, guard_d, guard_r, train_d, train_r)
      .reachable_indices;
}

namespace {

constexpr int kMcVerifyTrials = 8192;

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
    pfa *= static_cast<double>(n - i) / (static_cast<double>(n - i) + alpha);
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
double goso_side_statistic_host(double neg_sum, int64_t neg_cnt, double pos_sum, int64_t pos_cnt) {
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
double simulate_goso_noise_nsides(int64_t n_side, int64_t k, std::mt19937_64& rng) {
  auto side_mean = [&]() {
    double sum = 0.0;
    for (int64_t i = 0; i < n_side; ++i) {
      sum += sample_gamma_k(k, rng);
    }
    return sum / static_cast<double>(n_side);
  };
  const double range_stat = goso_side_statistic_host<kGreatestOf>(side_mean(), 1, side_mean(), 1);
  const double doppler_stat = goso_side_statistic_host<kGreatestOf>(side_mean(), 1, side_mean(), 1);
  if (kGreatestOf) {
    return std::max(range_stat, doppler_stat);
  }
  return std::min(range_stat, doppler_stat);
}

template <bool kGreatestOf>
double simulate_goso_noise_at_cell(int d, int r, int num_doppler, int num_range, int guard_d,
                                   int guard_r, int train_d, int train_r, int64_t k,
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
  const double range_stat = goso_side_statistic_host<kGreatestOf>(range_neg_sum, range_neg_cnt,
                                                                  range_pos_sum, range_pos_cnt);
  const double doppler_stat = goso_side_statistic_host<kGreatestOf>(
      doppler_neg_sum, doppler_neg_cnt, doppler_pos_sum, doppler_pos_cnt);
  if (kGreatestOf) {
    return std::max(range_stat, doppler_stat);
  }
  return std::min(range_stat, doppler_stat);
}

double goso_cfar_pfa_from_alpha_impl(double alpha, int64_t n_side, int64_t k, int mc_trials,
                                     bool greatest_of) {
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
    const double gpu_pfa =
        goso_mc_pfa_from_alpha_gpu(alpha, n_side, k, mc_trials, /*guard_d=*/0, /*guard_r=*/0,
                                   /*train_d=*/0, /*train_r=*/0, greatest_of);
    if (gpu_pfa >= 0.0) {
      return gpu_pfa;
    }
  }
  std::mt19937_64 rng(0x474F43464152ULL ^ static_cast<uint64_t>(n_side) ^
                      (static_cast<uint64_t>(k) << 32) ^ (greatest_of ? 0ULL : 0x534F43464152ULL));
  int64_t false_alarms = 0;
  for (int trial = 0; trial < mc_trials; ++trial) {
    const double noise = greatest_of ? simulate_goso_noise_nsides<true>(n_side, k, rng)
                                     : simulate_goso_noise_nsides<false>(n_side, k, rng);
    const double cut = sample_gamma_k(k, rng);
    if (cut > alpha * noise) {
      ++false_alarms;
    }
  }
  return static_cast<double>(false_alarms) / static_cast<double>(mc_trials);
}

double goso_cfar_pfa_at_cell_impl(double alpha, int64_t k, int mc_trials, int d, int r,
                                  int num_doppler, int num_range, int guard_d, int guard_r,
                                  int train_d, int train_r, bool greatest_of) {
  if (k <= 0) {
    return 0.0;
  }
  if (alpha <= 0.0) {
    return 1.0;
  }
  if (gpu_mc_calibration_available()) {
    const double gpu_pfa =
        goso_mc_pfa_at_cell_gpu(alpha, k, mc_trials, d, r, num_doppler, num_range, guard_d, guard_r,
                                train_d, train_r, greatest_of);
    if (gpu_pfa >= 0.0) {
      return gpu_pfa;
    }
  }
  std::mt19937_64 rng(0x474F43464152ULL ^ (static_cast<uint64_t>(d) << 32) ^
                      static_cast<uint64_t>(r) ^ (static_cast<uint64_t>(k) << 16) ^
                      (greatest_of ? 0ULL : 0x534F43464152ULL));
  int64_t false_alarms = 0;
  for (int trial = 0; trial < mc_trials; ++trial) {
    const double noise =
        greatest_of ? simulate_goso_noise_at_cell<true>(d, r, num_doppler, num_range, guard_d,
                                                        guard_r, train_d, train_r, k, rng)
                    : simulate_goso_noise_at_cell<false>(d, r, num_doppler, num_range, guard_d,
                                                         guard_r, train_d, train_r, k, rng);
    const double cut = sample_gamma_k(k, rng);
    if (cut > alpha * noise) {
      ++false_alarms;
    }
  }
  return static_cast<double>(false_alarms) / static_cast<double>(mc_trials);
}

double go_cfar_pfa_from_alpha_impl(double alpha, int64_t n_side, int64_t k, int mc_trials,
                                   int /*guard_d*/, int /*guard_r*/, int /*train_d*/,
                                   int /*train_r*/) {
  return goso_cfar_pfa_from_alpha_impl(alpha, n_side, k, mc_trials, true);
}

double so_cfar_pfa_from_alpha_impl(double alpha, int64_t n_side, int64_t k, int mc_trials,
                                   int /*guard_d*/, int /*guard_r*/, int /*train_d*/,
                                   int /*train_r*/) {
  return goso_cfar_pfa_from_alpha_impl(alpha, n_side, k, mc_trials, false);
}

double os_select_rank_host(std::vector<double>& samples, int64_t os_rank) {
  if (samples.empty()) {
    return 0.0;
  }
  const int64_t rank = os_rank < static_cast<int64_t>(samples.size())
                           ? os_rank
                           : static_cast<int64_t>(samples.size()) - 1;
  std::nth_element(samples.begin(), samples.begin() + rank, samples.end());
  return samples[static_cast<size_t>(rank)];
}

double simulate_os_noise_n(int64_t n, int64_t os_rank, int64_t k, std::mt19937_64& rng) {
  std::vector<double> samples(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    samples[static_cast<size_t>(i)] = sample_gamma_k(k, rng);
  }
  return os_select_rank_host(samples, os_rank);
}

double os_cfar_pfa_from_alpha_impl(double alpha, int64_t n, int64_t os_rank, int64_t k,
                                   int mc_trials, int /*guard_d*/, int /*guard_r*/, int /*train_d*/,
                                   int /*train_r*/) {
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
  // Draw exactly `n` i.i.d. training cells (matches the clipped OS window
  // count); guard/train only defined the physical stencil that produced n.
  if (gpu_mc_calibration_available()) {
    const double gpu_pfa = os_mc_pfa_from_alpha_gpu(alpha, n, effective_rank, k, mc_trials,
                                                    /*guard_d=*/0, /*guard_r=*/0,
                                                    /*train_d=*/0, /*train_r=*/0);
    if (gpu_pfa >= 0.0) {
      return gpu_pfa;
    }
  }
  std::mt19937_64 rng(0x4F5343464152ULL ^ static_cast<uint64_t>(n) ^
                      (static_cast<uint64_t>(effective_rank) << 16) ^
                      (static_cast<uint64_t>(k) << 32));
  int64_t false_alarms = 0;
  for (int trial = 0; trial < mc_trials; ++trial) {
    const double noise = simulate_os_noise_n(n, effective_rank, k, rng);
    const double cut = sample_gamma_k(k, rng);
    if (cut > alpha * noise) {
      ++false_alarms;
    }
  }
  return static_cast<double>(false_alarms) / static_cast<double>(mc_trials);
}

}  // namespace

namespace {
double single_mc_alpha(CfarVariant variant, double pfa, std::int64_t k, std::int64_t index, int gd,
                       int gr, int td, int tr, int d = -1, int r = -1, int nd = 1, int nr = 1,
                       std::int64_t os_rank = 0) {
  if (k <= 0 || index <= 0 || (variant == CfarVariant::kOS && index > detail::kMaxOsTrainingCells))
    throw std::invalid_argument("CFAR calibration invalid looks or training count");
  CfarCalibrationPlan plan;
  auto params = std::make_shared<CFARParams>();
  plan.params = params;
  params->cfar_variant = variant;
  params->pfa = pfa;
  params->os_rank = os_rank;
  params->guard_cells = {gd, gr};
  params->training_cells = {td, tr};
  plan.input_shape = {nd, nr};
  plan.effective_looks = k;
  plan.budget = cfar_calibration_budget(pfa);
  return calibrate_cfar_geometry_gpu(plan, {index, d, r}, {}, {});
}
}  // namespace

double go_cfar_pfa_from_alpha(double alpha, int64_t n_side, int64_t k) {
  return go_cfar_pfa_from_alpha_impl(alpha, n_side, k, kMcVerifyTrials, 1, 1, 2, 2);
}

double go_cfar_alpha(double pfa, int64_t n_side, int64_t k, int guard_d, int guard_r, int train_d,
                     int train_r) {
  return single_mc_alpha(CfarVariant::kGO, pfa, k, n_side, guard_d, guard_r, train_d, train_r);
}

double go_cfar_alpha_at_cell(double pfa, int64_t k, int d, int r, int num_doppler, int num_range,
                             int guard_d, int guard_r, int train_d, int train_r) {
  std::int64_t rn, rp, dn, dp;
  go_training_side_counts(d, r, num_doppler, num_range, guard_d, guard_r, train_d, train_r, &rn,
                          &rp, &dn, &dp);
  return single_mc_alpha(CfarVariant::kGO, pfa, k, go_lut_index(rn, rp, dn, dp), guard_d, guard_r,
                         train_d, train_r, d, r, num_doppler, num_range);
}

std::vector<double> build_go_alpha_lut(double pfa, int64_t fold, int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d, int train_r) {
  CFARParams p;
  p.cfar_variant = CfarVariant::kGO;
  p.pfa = pfa;
  p.preintegrated_looks = fold;
  p.guard_cells = {guard_d, guard_r};
  p.training_cells = {train_d, train_r};
  return calibrate_cfar(p, {num_doppler, num_range}).alpha_lut;
}

double so_cfar_pfa_from_alpha(double alpha, int64_t n_side, int64_t k) {
  return so_cfar_pfa_from_alpha_impl(alpha, n_side, k, kMcVerifyTrials, 1, 1, 2, 2);
}

double so_cfar_alpha(double pfa, int64_t n_side, int64_t k, int guard_d, int guard_r, int train_d,
                     int train_r) {
  return single_mc_alpha(CfarVariant::kSO, pfa, k, n_side, guard_d, guard_r, train_d, train_r);
}

double so_cfar_alpha_at_cell(double pfa, int64_t k, int d, int r, int num_doppler, int num_range,
                             int guard_d, int guard_r, int train_d, int train_r) {
  std::int64_t rn, rp, dn, dp;
  go_training_side_counts(d, r, num_doppler, num_range, guard_d, guard_r, train_d, train_r, &rn,
                          &rp, &dn, &dp);
  return single_mc_alpha(CfarVariant::kSO, pfa, k, go_lut_index(rn, rp, dn, dp), guard_d, guard_r,
                         train_d, train_r, d, r, num_doppler, num_range);
}

std::vector<double> build_so_alpha_lut(double pfa, int64_t fold, int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d, int train_r) {
  CFARParams p;
  p.cfar_variant = CfarVariant::kSO;
  p.pfa = pfa;
  p.preintegrated_looks = fold;
  p.guard_cells = {guard_d, guard_r};
  p.training_cells = {train_d, train_r};
  return calibrate_cfar(p, {num_doppler, num_range}).alpha_lut;
}

double os_cfar_pfa_from_alpha(double alpha, int64_t n, int64_t os_rank, int64_t k) {
  return os_cfar_pfa_from_alpha_impl(alpha, n, os_rank, k, kMcVerifyTrials, 1, 1, 2, 2);
}

double os_cfar_alpha(double pfa, int64_t n, int64_t os_rank, int64_t k, int guard_d, int guard_r,
                     int train_d, int train_r) {
  if (k == 1)
    return os_cfar_alpha_closed_k1(pfa, n, os_rank);
  return single_mc_alpha(CfarVariant::kOS, pfa, k, n, guard_d, guard_r, train_d, train_r, -1, -1, 1,
                         1, os_rank);
}

std::vector<double> build_os_alpha_lut(double pfa, int64_t fold, int64_t os_rank, int num_doppler,
                                       int num_range, int guard_d, int guard_r, int train_d,
                                       int train_r) {
  if (fold > 1) {
    CFARParams p;
    p.cfar_variant = CfarVariant::kOS;
    p.pfa = pfa;
    p.preintegrated_looks = fold;
    p.os_rank = os_rank;
    p.guard_cells = {guard_d, guard_r};
    p.training_cells = {train_d, train_r};
    return calibrate_cfar(p, {num_doppler, num_range}).alpha_lut;
  }

  const int64_t max_n =
      max_training_cell_count(num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  std::vector<double> lut(static_cast<size_t>(max_n + 1), 0.0);
  if (fold == 1) {
    // Closed-form inversion per N; no map scan — calibrate the full 1..max_n
    // range cheaply (reachable-only would add a costly D×R scan for no gain).
    for (int64_t n = 1; n <= max_n; ++n) {
      const int64_t effective_rank = effective_os_rank(n, os_rank);
      lut[static_cast<size_t>(n)] = os_cfar_alpha_closed_k1(pfa, n, effective_rank);
    }
    return lut;
  }
  const std::vector<int64_t> indices =
      reachable_training_cell_counts(num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  parallel_fill_lut_indices(lut, indices, [&](int64_t n) {
    const int64_t effective_rank = effective_os_rank(n, os_rank);
    return os_cfar_alpha(pfa, n, effective_rank, fold, guard_d, guard_r, train_d, train_r);
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
    std::uint64_t error_bits = 0, confidence_bits = 0;
    std::memcpy(&error_bits, &key.relative_error, sizeof(error_bits));
    std::memcpy(&confidence_bits, &key.confidence, sizeof(confidence_bits));
    h = mix(h, error_bits);
    h = mix(h, confidence_bits);
    h = mix(h, key.algorithm_version);
    h = mix(h, key.calibration_trials);
    return h;
  }
};

using AlphaLut = std::vector<double>;

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
std::deque<CfarAlphaLutKey> g_alpha_lut_order;  // NOLINT(whitespace/indent_namespace)
// Keys whose LUT is currently being built. Concurrent callers for the same
// key wait on the shared future instead of duplicating the (expensive) MC
// calibration.
struct AlphaLutJob {
  std::promise<AlphaLut> promise;
  AlphaLutFuture future{promise.get_future().share()};
  std::stop_source stop;
  std::set<const void*> owners;
};
std::unordered_map<CfarAlphaLutKey, std::shared_ptr<AlphaLutJob>, CfarAlphaLutKeyHash> g_jobs;

std::vector<double> build_alpha_lut(const CfarAlphaLutKey& key, std::stop_token stop) {
  CFARParams p;
  p.cfar_variant = key.variant;
  p.pfa = key.pfa;
  p.preintegrated_looks = key.fold;
  p.os_rank = key.os_rank;
  p.guard_cells = {key.guard_d, key.guard_r};
  p.training_cells = {key.train_d, key.train_r};
  p.calibration_relative_error = key.relative_error;
  p.calibration_confidence = key.confidence;
  p.calibration_trials = key.calibration_trials;
  return calibrate_cfar(p, {key.num_doppler, key.num_range}, {}, stop).alpha_lut;
}

void run_lut_build(CfarAlphaLutKey key, std::shared_ptr<AlphaLutJob> job, int device) {
  try {
    if (cudaSetDevice(device) != cudaSuccess)
      throw std::runtime_error("CFAR calibration cannot select CUDA device");
    auto lut = build_alpha_lut(key, job->stop.get_token());
    {
      std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
      if (job->stop.stop_requested())
        throw CfarCalibrationCancelled{};
      if (g_alpha_lut_cache.emplace(key, lut).second)
        g_alpha_lut_order.push_back(key);
      while (g_alpha_lut_order.size() > kMaxCachedAlphaLuts) {
        g_alpha_lut_cache.erase(g_alpha_lut_order.front());
        g_alpha_lut_order.pop_front();
      }
      job->promise.set_value(std::move(lut));
      if (auto it = g_jobs.find(key); it != g_jobs.end() && it->second == job)
        g_jobs.erase(it);
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
    job->promise.set_exception(std::current_exception());
    // Existing waiters retain this job and its exception through shared_ptr.
    // Fresh callers must be free to retry without waiting for those owners to stop.
    if (auto it = g_jobs.find(key); it != g_jobs.end() && it->second == job)
      g_jobs.erase(it);
  }
}

AlphaLutSnapshot poll_alpha_lut_build(const AlphaLutFuture& build) {
  if (build.wait_for(std::chrono::seconds{0}) == std::future_status::ready)
    return {true, build.get()};
  return {false, {}, true};
}

AlphaLutSnapshot poll_cached_alpha_lut(const CfarAlphaLutKey& key) {
  std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
  if (auto it = g_alpha_lut_cache.find(key); it != g_alpha_lut_cache.end())
    return {true, it->second};
  if (auto it = g_jobs.find(key); it != g_jobs.end())
    return poll_alpha_lut_build(it->second->future);
  return {};
}

struct AlphaLutAcquisition {
  std::shared_ptr<AlphaLutJob> job;
  AlphaLutSnapshot cached;
};

AlphaLutAcquisition acquire_job(const CfarAlphaLutKey& key, const void* owner) {
  std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
  // Copy under the same lock as the lookup: FIFO eviction may remove the entry
  // as soon as this operation returns, so a second poll cannot recover it safely.
  if (auto it = g_alpha_lut_cache.find(key); it != g_alpha_lut_cache.end())
    return {{}, {true, it->second}};
  if (auto it = g_jobs.find(key); it != g_jobs.end() && !it->second->stop.stop_requested()) {
    it->second->owners.insert(owner);
    return {it->second, {}};
  }
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    throw std::runtime_error("CFAR calibration requires CUDA");
  auto job = std::make_shared<AlphaLutJob>();
  job->owners.insert(owner);
  g_jobs[key] = job;
  try {
    std::thread(run_lut_build, key, job, device).detach();
  } catch (...) {
    g_jobs.erase(key);
    throw;
  }
  return {job, {}};
}

AlphaLutFuture ensure_alpha_lut_build(const CfarAlphaLutKey& key, const void* owner) {
  auto acquired = acquire_job(key, owner);
  if (acquired.job)
    return acquired.job->future;
  std::promise<AlphaLut> ready;
  ready.set_value(std::move(acquired.cached.lut));
  return ready.get_future().share();
}

void release_alpha_lut_builds(const void* owner) {
  std::vector<std::shared_ptr<AlphaLutJob>> retired;
  {
    std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
    for (const auto& [key, job] : g_jobs) {
      if (job->owners.erase(owner) && job->owners.empty()) {
        job->stop.request_stop();
        retired.push_back(job);
      }
    }
  }
  for (const auto& job : retired) {
    job->future.wait();
    std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
    for (auto it = g_jobs.begin(); it != g_jobs.end();) {
      if (it->second == job && job->owners.empty())
        it = g_jobs.erase(it);
      else
        ++it;
    }
  }
}

std::vector<double> get_cached_alpha_lut(const CfarAlphaLutKey& key,
                                         std::function<bool()> cancelled, const void* owner,
                                         AlphaLutFuture build) {
  int local_owner = 0;
  const void* subscriber = owner ? owner : &local_owner;
  if (!build.valid()) {
    auto acquired = acquire_job(key, subscriber);
    if (acquired.cached.ready)
      return std::move(acquired.cached.lut);
    build = acquired.job->future;
  }
  try {
    while (build.wait_for(std::chrono::milliseconds{10}) != std::future_status::ready) {
      if (cancelled && cancelled())
        throw CfarCalibrationCancelled{};
    }
    auto lut = build.get();
    if (!owner)
      release_alpha_lut_builds(subscriber);
    return lut;
  } catch (...) {
    release_alpha_lut_builds(subscriber);
    throw;
  }
}

void join_inflight_alpha_lut_builds() {
  std::vector<AlphaLutFuture> futures;
  {
    std::lock_guard<std::mutex> lock(g_alpha_lut_cache_mutex);
    for (const auto& [key, job] : g_jobs)
      futures.push_back(job->future);
  }
  for (auto& f : futures)
    f.wait();
}

}  // namespace holoscan::radar::detail
