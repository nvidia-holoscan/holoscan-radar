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
#include <string>

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
                       cudaStream_t stream) {
  if (method == ClutterFilterMethod::kSingleDelay && num_chirps < 2) {
    throw std::invalid_argument(
        "clutter stage: method 'single_delay' needs num_chirps >= 2, got " +
        std::to_string(num_chirps));
  }
  if (method == ClutterFilterMethod::kDoubleDelay && num_chirps < 3) {
    throw std::invalid_argument(
        "clutter stage: method 'double_delay' needs num_chirps >= 3, got " +
        std::to_string(num_chirps));
  }

  auto in_view =
      make_const_complex_view3<T>(in, batch, num_chirps, num_range_bins);
  const int64_t num_chirps_out = clutter_num_chirps_out(method, num_chirps);
  auto out_view =
      make_complex_view3<T>(out, batch, num_chirps_out, num_range_bins);
  matx::cudaExecutor exec{stream};

  switch (method) {
    case ClutterFilterMethod::kNone:
      // Fused operator's inline passthrough only.
      (out_view = in_view).run(exec);
      break;
    case ClutterFilterMethod::kMeanSubtraction: {
      // Subtract the per-(batch, range-bin) slow-time mean, broadcast back
      // over the chirp axis. Shape preserved.
      auto slow_time_mean = matx::mean(in_view, {1});  // (batch, bins)
      auto mean_bcast = matx::clone<3>(
          slow_time_mean,
          {matx::matxKeepDim, num_chirps, matx::matxKeepDim});
      (out_view = in_view - mean_bcast).run(exec);
      break;
    }
    case ClutterFilterMethod::kSingleDelay: {
      // out[k] = in[k+1] - in[k], k in [0, num_chirps - 1).
      auto head = matx::slice(
          in_view, {0, 0, 0},
          {matx::matxEnd, num_chirps - 1, matx::matxEnd});
      auto tail = matx::slice(in_view, {0, 1, 0},
                              {matx::matxEnd, num_chirps, matx::matxEnd});
      (out_view = tail - head).run(exec);
      break;
    }
    case ClutterFilterMethod::kDoubleDelay: {
      // out[k] = in[k] - 2 in[k+1] + in[k+2], k in [0, num_chirps - 2).
      auto x0 = matx::slice(
          in_view, {0, 0, 0},
          {matx::matxEnd, num_chirps - 2, matx::matxEnd});
      auto x1 = matx::slice(
          in_view, {0, 1, 0},
          {matx::matxEnd, num_chirps - 1, matx::matxEnd});
      auto x2 = matx::slice(in_view, {0, 2, 0},
                            {matx::matxEnd, num_chirps, matx::matxEnd});
      const cuda::std::complex<T> two{static_cast<T>(2), T{0}};
      (out_view = x0 - two * x1 + x2).run(exec);
      break;
    }
  }
}

}  // namespace holoscan::radar::detail
