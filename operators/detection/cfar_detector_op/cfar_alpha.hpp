// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <future>
#include <vector>

namespace holoscan::radar {

struct CfarAlphaLutKey;

namespace detail {

// Maximum in-bounds training cells for OS-CFAR on a map. Shared by the fused
// OS stencil (cfar_stage.hpp), compute()-time validation, and the GPU Monte
// Carlo calibration kernel (cfar_alpha_mc.cu).
constexpr int kMaxOsTrainingCells = 512;

// Non-blocking LUT lookup for async live-override paths.
struct AlphaLutSnapshot {
  bool ready{false};
  std::vector<double> lut{};
  bool building{false};  // An unfinished job exists; distinguishes cancellation from pending work.
};

// Return a cached LUT or the current background-job state without queuing work.
AlphaLutSnapshot poll_cached_alpha_lut(const CfarAlphaLutKey& key);

// Retain the outcome for a subscriber even after a failed job leaves the registry.
using AlphaLutFuture = std::shared_future<std::vector<double>>;
AlphaLutSnapshot poll_alpha_lut_build(const AlphaLutFuture& build);

// Idempotently queue a build and retain its result, including any failure.
AlphaLutFuture ensure_alpha_lut_build(const CfarAlphaLutKey& key, const void* owner = nullptr);

// Drop only this detector's subscriptions; cancel and drain jobs with no remaining users.
void release_alpha_lut_builds(const void* owner);

// Block until every in-flight background alpha-LUT build has finished. Called
// from CFARDetectorOp::stop() so a detached GPU Monte-Carlo build cannot launch
// kernels after the operator / CUDA context is torn down (fixes the L15 race).
void join_inflight_alpha_lut_builds();

// Blocking LUT lookup (first tick or when async rebuild is disabled).
// A supplied build preserves that subscriber's result even after registry replacement.
std::vector<double> get_cached_alpha_lut(const CfarAlphaLutKey& key,
                                         std::function<bool()> cancelled = {},
                                         const void* owner = nullptr, AlphaLutFuture build = {});

// Training-cell count for one (doppler, range) cell under the CA window
// footprint (matches cfar_ca_kernel).
int64_t training_cell_count(int d, int r, int num_doppler, int num_range, int guard_d, int guard_r,
                            int train_d, int train_r);

// Maximum training-cell count over all cells on a map with the given
// footprint (interior and border-inclusive).
int64_t max_training_cell_count(int num_doppler, int num_range, int guard_d, int guard_r,
                                int train_d, int train_r);

// Distinct CA-window training-cell counts that appear for at least one cell on
// the map (exhaustive scan). Used to validate sparse external alpha_lut tables
// for OS K>1, whose LUTs fill only these reachable counts.
std::vector<int64_t> reachable_training_cell_counts(int num_doppler, int num_range, int guard_d,
                                                    int guard_r, int train_d, int train_r);

// False-alarm probability for CA-CFAR on Gamma(K) integrated clutter:
//   Pfa = I_{1/(1 + alpha/N)}(N*K, K)
double ca_cfar_pfa_from_alpha(double alpha, int64_t n, int64_t k);

// Invert ca_cfar_pfa_from_alpha for alpha (host-only, monotone bisection).
double ca_cfar_alpha(double pfa, int64_t n, int64_t k);

// Build alpha_lut[n] for n in [1, max_n] on the map. Index 0 is unused.
std::vector<double> build_ca_alpha_lut(double pfa, int64_t fold, int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d, int train_r);

// GO-CFAR (2-D GOGO): per-side reference counts for one (doppler, range)
// cell (matches cfar_go_kernel).
void go_training_side_counts(int d, int r, int num_doppler, int num_range, int guard_d, int guard_r,
                             int train_d, int train_r, int64_t* range_neg, int64_t* range_pos,
                             int64_t* doppler_neg, int64_t* doppler_pos);

// LUT index from side counts (symmetric interior → n; border-asymmetric →
// min of the two active sides per axis, then min across axes when both
// active).
int64_t go_lut_side_index(int64_t neg_count, int64_t pos_count);

int64_t go_lut_index(int64_t range_neg, int64_t range_pos, int64_t doppler_neg,
                     int64_t doppler_pos);

// Maximum GO/SO LUT index on a map. Closed-form for maps that fit the window
// (interior side-count index); exhaustive only for sub-window maps.
int64_t max_go_lut_index(int num_doppler, int num_range, int guard_d, int guard_r, int train_d,
                         int train_r);

// Distinct GO/SO LUT indices that appear for at least one cell on the map
// (exhaustive scan; used to validate sparse external alpha_lut tables).
std::vector<int64_t> reachable_go_lut_indices(int num_doppler, int num_range, int guard_d,
                                              int guard_r, int train_d, int train_r);

// False-alarm probability for symmetric GOGO-CFAR with `n_side` i.i.d.
// reference cells on each of the four half-sides (range±, Doppler±) and
// Gamma(`k`) integrated clutter.
double go_cfar_pfa_from_alpha(double alpha, int64_t n_side, int64_t k);

// Invert go_cfar_pfa_from_alpha (Monte Carlo + bisection). The `guard_*` /
// `train_*` arguments are retained for call-site compatibility; the noise
// model is parameterized by `n_side` alone (four half-sides of that length).
double go_cfar_alpha(double pfa, int64_t n_side, int64_t k, int guard_d, int guard_r, int train_d,
                     int train_r);

// Invert Pfa at a specific map cell by Monte Carlo matching cfar_go_kernel's
// clipped window at (d, r). Used when building per-map alpha LUTs.
double go_cfar_alpha_at_cell(double pfa, int64_t k, int d, int r, int num_doppler, int num_range,
                             int guard_d, int guard_r, int train_d, int train_r);

std::vector<double> build_go_alpha_lut(double pfa, int64_t fold, int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d, int train_r);

// SO-CFAR shares GOGO window geometry and LUT indexing; only the side and
// axis combine uses min instead of max.
double so_cfar_pfa_from_alpha(double alpha, int64_t n_side, int64_t k);

double so_cfar_alpha(double pfa, int64_t n_side, int64_t k, int guard_d, int guard_r, int train_d,
                     int train_r);

double so_cfar_alpha_at_cell(double pfa, int64_t k, int d, int r, int num_doppler, int num_range,
                             int guard_d, int guard_r, int train_d, int train_r);

std::vector<double> build_so_alpha_lut(double pfa, int64_t fold, int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d, int train_r);

// OS-CFAR: os_rank is a 0-based index into ascending-sorted training powers.
// For K>1, Monte Carlo draws exactly `n` i.i.d. Gamma(K) training cells
// (matching the clipped window count); guard/train are retained on
// os_cfar_alpha for call-site compatibility only.
double os_cfar_pfa_from_alpha(double alpha, int64_t n, int64_t os_rank, int64_t k);

double os_cfar_alpha(double pfa, int64_t n, int64_t os_rank, int64_t k, int guard_d, int guard_r,
                     int train_d, int train_r);

std::vector<double> build_os_alpha_lut(double pfa, int64_t fold, int64_t os_rank, int num_doppler,
                                       int num_range, int guard_d, int guard_r, int train_d,
                                       int train_r);

}  // namespace detail
}  // namespace holoscan::radar
