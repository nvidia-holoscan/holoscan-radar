// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoscan::radar::detail {

// True when at least one CUDA device is available for MC calibration.
bool gpu_mc_calibration_available();

// Monte Carlo false-alarm rate for interior GOGO/SOSO geometry on the GPU.
// Returns false_alarms / mc_trials. Falls back to the host path when CUDA is
// unavailable (callers should check gpu_mc_calibration_available() first).
double goso_mc_pfa_from_alpha_gpu(double alpha, int64_t n_side, int64_t k,
                                  int mc_trials, int guard_d, int guard_r,
                                  int train_d, int train_r, bool greatest_of);

// Monte Carlo false-alarm rate for interior OS geometry on the GPU.
double os_mc_pfa_from_alpha_gpu(double alpha, int64_t n, int64_t os_rank,
                                int64_t k, int mc_trials, int guard_d,
                                int guard_r, int train_d, int train_r);

}  // namespace holoscan::radar::detail
