// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Process-level warmup helper for performance consumers.
//
// Scenario #1 in a fresh process otherwise absorbs CUDA primary-context init
// and Holoscan's first Application tick, inflating cold_ms relative to later
// scenarios. Call run_process_warmup() once before the measured matrix with a
// discarded operator graph; do not record its rows in the report.

#pragma once

#include <cuda_runtime_api.h>

#include <functional>
#include <iostream>
#include <stdexcept>

namespace holoscan::radar::perf {

// `prime` should build and run a minimal graph for the operator under test
// (a few frames is enough). Results must not be appended to the published
// PerfReport.
inline void run_process_warmup(const std::function<void()>& prime) {
  // Force primary CUDA context creation before Holoscan graph setup.
  if (cudaFree(0) != cudaSuccess) {
    throw std::runtime_error(
        "perf process warmup: cudaFree(0) failed (no usable CUDA device?)");
  }
  std::cerr << "process warmup (discarded)...\n";
  prime();
  if (cudaDeviceSynchronize() != cudaSuccess) {
    throw std::runtime_error(
        "perf process warmup: cudaDeviceSynchronize failed");
  }
}

}  // namespace holoscan::radar::perf
