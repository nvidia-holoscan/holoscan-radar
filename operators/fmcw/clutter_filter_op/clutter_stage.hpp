// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Clutter-suppression stage: the pure kernel behind ClutterFilterOp.
//
// The fused FMCWRangeDopplerProcessorOp calls this same code path, which is
// what guarantees the split-vs-fused parity property by
// construction. The signatures below are therefore FROZEN for the parallel
// implementation phase: implement the bodies, do not change the interfaces
// (escalate to the orchestrator if a change seems necessary).
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cstdint>
#include <stdexcept>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// Output chirp count for a method, per the locked contract. kNone passes
// through unchanged (fused operator's inline stage only).
inline int64_t clutter_num_chirps_out(ClutterFilterMethod method,
                                      int64_t num_chirps) {
  switch (method) {
    case ClutterFilterMethod::kNone:
    case ClutterFilterMethod::kMeanSubtraction:
      return num_chirps;
    case ClutterFilterMethod::kSingleDelay:
      return num_chirps - 1;
    case ClutterFilterMethod::kDoubleDelay:
      return num_chirps - 2;
  }
  return num_chirps;
}

// Run slow-time clutter suppression over a collapsed batch view.
//
//   in  — device pointer, (batch, num_chirps, num_range_bins),
//         cuda::std::complex<T>.
//   out — device pointer, (batch, num_chirps_out, num_range_bins),
//         cuda::std::complex<T>, preallocated by the caller;
//         num_chirps_out must equal clutter_num_chirps_out(method, ...).
//
// Methods (per the locked contract):
//   kMeanSubtraction — subtract the per-(batch, range-bin) slow-time mean.
//   kSingleDelay     — first difference  x[k+1] - x[k]        (1st-order MTI)
//   kDoubleDelay     — x[k] - 2 x[k+1] + x[k+2]               (2nd-order MTI)
//   kNone            — copy through (fused inline stage only).
//
// Stateless (no setup cost). All work is enqueued on `stream`; no
// synchronization. Throws std::invalid_argument when num_chirps is too
// small for the selected method (single_delay needs >= 2, double_delay
// needs >= 3).
template <typename T>
void run_clutter_stage(ClutterFilterMethod method, const void* in,
                       int64_t batch, int64_t num_chirps,
                       int64_t num_range_bins, void* out,
                       cudaStream_t stream);

}  // namespace holoscan::radar::detail
