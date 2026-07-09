// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace holoscan::radar::detail {

// Training-cell count for one (doppler, range) cell under the CA window
// footprint (matches cfar_ca_kernel).
int64_t training_cell_count(int d, int r, int num_doppler, int num_range,
                            int guard_d, int guard_r, int train_d, int train_r);

// Maximum training-cell count over all cells on a map with the given
// footprint (interior and border-inclusive).
int64_t max_training_cell_count(int num_doppler, int num_range, int guard_d,
                                int guard_r, int train_d, int train_r);

// False-alarm probability for CA-CFAR on Gamma(K) integrated clutter:
//   Pfa = I_{1/(1 + alpha/N)}(N*K, K)
double ca_cfar_pfa_from_alpha(double alpha, int64_t n, int64_t k);

// Invert ca_cfar_pfa_from_alpha for alpha (host-only, monotone bisection).
double ca_cfar_alpha(double pfa, int64_t n, int64_t k);

// Build alpha_lut[n] for n in [1, max_n] on the map. Index 0 is unused.
std::vector<double> build_ca_alpha_lut(double pfa, int64_t fold,
                                       int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d,
                                       int train_r);

// GO-CFAR (2-D GOGO): per-side reference counts for one (doppler, range)
// cell (matches cfar_go_kernel).
void go_training_side_counts(int d, int r, int num_doppler, int num_range,
                             int guard_d, int guard_r, int train_d,
                             int train_r, int64_t* range_neg,
                             int64_t* range_pos, int64_t* doppler_neg,
                             int64_t* doppler_pos);

// LUT index from side counts (symmetric interior → n; border-asymmetric →
// min of the two active sides per axis, then min across axes when both
// active).
int64_t go_lut_side_index(int64_t neg_count, int64_t pos_count);

int64_t go_lut_index(int64_t range_neg, int64_t range_pos, int64_t doppler_neg,
                     int64_t doppler_pos);

int64_t max_go_lut_index(int num_doppler, int num_range, int guard_d,
                         int guard_r, int train_d, int train_r);

// False-alarm probability for symmetric GOGO-CFAR with `n_side` reference
// cells on each range/doppler side and Gamma(`k`) integrated clutter.
double go_cfar_pfa_from_alpha(double alpha, int64_t n_side, int64_t k);

// Invert go_cfar_pfa_from_alpha (host Monte Carlo + bisection). Uses the
// interior 2-D window geometry so the simulator matches cfar_go_kernel.
double go_cfar_alpha(double pfa, int64_t n_side, int64_t k, int guard_d,
                     int guard_r, int train_d, int train_r);

std::vector<double> build_go_alpha_lut(double pfa, int64_t fold,
                                       int num_doppler, int num_range,
                                       int guard_d, int guard_r, int train_d,
                                       int train_r);

}  // namespace holoscan::radar::detail
