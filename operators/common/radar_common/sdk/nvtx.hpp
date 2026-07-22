// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Optional NVTX instrumentation for attributing time inside an operator's
// compute() to named phases (resolve-overrides / classify-validate /
// stage-kernel / emit) under Nsight Systems. Everything here compiles to a
// no-op unless the build defines RADAR_NVTX, so normal and CI builds keep a
// zero-overhead, untouched hot path — the macros drop their arguments
// entirely, so not even the phase-name string literals reach the binary. The
// performance/ harness turns it on with the CMake option
// HOLOSCAN_RADAR_PERF_NVTX (see cmake/RadarPerfNvtx.cmake, performance/agent/,
// and the perf-analyze skill).
//
// Two forms:
//
//   RADAR_NVTX_RANGE(name)      — a scoped RAII range for a whole function
//                                 body or a { } block (e.g. setup()).
//
//   RADAR_NVTX_PHASE_SCOPE();   — declare a re-armable phase marker in the
//   RADAR_NVTX_PHASE("phase-a"); current scope, then advance it through a
//   ... work ...                sequence of non-overlapping phases that share
//   RADAR_NVTX_PHASE("phase-b"); local variables (so they cannot each take
//   ... work ...                their own { } block). Each RADAR_NVTX_PHASE
//                               closes the previous phase and opens the next;
//                               the marker's destructor closes the last, even
//                               on an early throw mid-compute().
//
// Coarse per-operator attribution (the whole compute() tick) already comes from
// Holoscan's GXF wrapper NVTX range; these ranges add the intra-compute split.
// Note: a range around a kernel enqueue bounds the launch, not the async GPU
// execution — join NVTX phases with the CUDA GPU-kernel timeline for true
// kernel duration (nsys nvtx_gpu_proj_sum / cuda_gpu_kern_sum), never treat
// NVTX-phase wall-time as GPU time.

#ifdef RADAR_NVTX

#include <optional>

#include <nvtx3/nvtx3.hpp>

// Two-level concat so __LINE__ expands before pasting (a unique variable name
// per RADAR_NVTX_RANGE use site).
#define RADAR_NVTX_DETAIL_CONCAT_INNER(a, b) a##b
#define RADAR_NVTX_DETAIL_CONCAT(a, b) RADAR_NVTX_DETAIL_CONCAT_INNER(a, b)

#define RADAR_NVTX_RANGE(name)                                       \
  ::nvtx3::scoped_range RADAR_NVTX_DETAIL_CONCAT(radar_nvtx_scope_,  \
                                                 __LINE__) {         \
    name                                                             \
  }

namespace holoscan::radar::nvtx {

// A re-armable NVTX phase marker. Holds the current phase's scoped_range in an
// optional; begin() emplaces a new one, which destroys the previous (ending
// that phase) before constructing the next (starting it) — so the phases are
// strictly sequential and non-overlapping. The destructor ends the final
// phase, including when the enclosing compute() throws mid-tick. Drive it
// through the RADAR_NVTX_PHASE_SCOPE / RADAR_NVTX_PHASE macros, never directly,
// so non-instrumented builds elide it (and its phase-name literals) entirely.
class Phase {
 public:
  Phase() = default;
  void begin(const char* name) { range_.emplace(name); }

 private:
  std::optional<::nvtx3::scoped_range> range_;
};

}  // namespace holoscan::radar::nvtx

#define RADAR_NVTX_PHASE_SCOPE() ::holoscan::radar::nvtx::Phase radar_nvtx_phase
#define RADAR_NVTX_PHASE(name) radar_nvtx_phase.begin(name)

#else  // RADAR_NVTX

// No-ops that consume their arguments without emitting them, so an off build
// carries neither NVTX calls nor the phase-name string literals.
#define RADAR_NVTX_RANGE(name) \
  do {                         \
  } while (0)
#define RADAR_NVTX_PHASE_SCOPE() \
  do {                           \
  } while (0)
#define RADAR_NVTX_PHASE(name) \
  do {                         \
  } while (0)

#endif  // RADAR_NVTX
