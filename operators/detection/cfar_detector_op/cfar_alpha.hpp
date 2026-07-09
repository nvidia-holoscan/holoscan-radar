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

}  // namespace holoscan::radar::detail
