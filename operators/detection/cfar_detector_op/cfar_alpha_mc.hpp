// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoscan::radar::detail {

// True when at least one CUDA device is available for MC calibration.
bool gpu_mc_calibration_available();

// Monte Carlo false-alarm rate for symmetric GOGO/SOSO with `n_side` i.i.d.
// cells on each of the four half-sides. Returns false_alarms / mc_trials, or
// <0 when CUDA is unavailable.
double goso_mc_pfa_from_alpha_gpu(double alpha, int64_t n_side, int64_t k, int mc_trials,
                                  int guard_d, int guard_r, int train_d, int train_r,
                                  bool greatest_of);

// Monte Carlo false-alarm rate matching cfar_goso_kernel at cell (d, r).
double goso_mc_pfa_at_cell_gpu(double alpha, int64_t k, int mc_trials, int d, int r,
                               int num_doppler, int num_range, int guard_d, int guard_r,
                               int train_d, int train_r, bool greatest_of);

// Monte Carlo false-alarm rate for OS-CFAR with exactly `n` i.i.d. training
// cells (clipped-window count). guard/train args are unused (API compat).
double os_mc_pfa_from_alpha_gpu(double alpha, int64_t n, int64_t os_rank, int64_t k, int mc_trials,
                                int guard_d, int guard_r, int train_d, int train_r);

}  // namespace holoscan::radar::detail
