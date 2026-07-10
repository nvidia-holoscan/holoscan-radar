// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Clutter-suppression stage: the pure kernel behind ClutterFilterOp.
//
// with_clutter_expr() is the single source of the clutter math, consumed by
// both the split ClutterFilterOp (materialized via run_clutter_stage) and
// the fused FMCWRangeDopplerProcessorOp (composed into the Doppler pre-FFT
// expression) — which is what guarantees the split-vs-fused parity property
// by construction, at expression granularity. The signatures below are the
// stable stage API: extend via additive overloads; byte-identical
// fused-vs-split behavior is enforced by the BitExact parity tests.
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

// Build the clutter-suppression expression for `method` over `in` — a
// rank-3 MatX operator, (batch, num_chirps, num_range_bins),
// cuda::std::complex — and invoke fn(expr). The expression's chirp axis has
// clutter_num_chirps_out(method, num_chirps) elements. kNone invokes fn(in)
// directly (zero added kernels). Single source of the clutter math for both
// the split ClutterFilterOp (materialized below) and the fused operator's
// Doppler pre-FFT expression.
//
// Methods (per the locked contract):
//   kMeanSubtraction — subtract the per-(batch, range-bin) slow-time mean.
//   kSingleDelay     — first difference  x[k+1] - x[k]        (1st-order MTI)
//   kDoubleDelay     — x[k] - 2 x[k+1] + x[k+2]               (2nd-order MTI)
//   kNone            — pass through (fused inline stage only).
//
// Throws std::invalid_argument when num_chirps is too small for the
// selected method (single_delay needs >= 2, double_delay needs >= 3).
template <typename InOp, typename Fn>
void with_clutter_expr(ClutterFilterMethod method, const InOp& in,
                       int64_t num_chirps, Fn&& fn) {
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

  switch (method) {
    case ClutterFilterMethod::kNone:
      fn(in);
      break;
    case ClutterFilterMethod::kMeanSubtraction: {
      // Subtract the per-(batch, range-bin) slow-time mean, broadcast back
      // over the chirp axis. Shape preserved. matx::mean is a transform op:
      // the reduction materializes into a small stream-ordered temp during
      // the consuming expression's PreRun.
      auto slow_time_mean = matx::mean(in, {1});  // (batch, bins)
      auto mean_bcast = matx::clone<3>(
          slow_time_mean,
          {matx::matxKeepDim, num_chirps, matx::matxKeepDim});
      fn(in - mean_bcast);
      break;
    }
    case ClutterFilterMethod::kSingleDelay: {
      // out[k] = in[k+1] - in[k], k in [0, num_chirps - 1).
      auto head = matx::slice(
          in, {0, 0, 0},
          {matx::matxEnd, num_chirps - 1, matx::matxEnd});
      auto tail = matx::slice(in, {0, 1, 0},
                              {matx::matxEnd, num_chirps, matx::matxEnd});
      fn(tail - head);
      break;
    }
    case ClutterFilterMethod::kDoubleDelay: {
      // out[k] = in[k] - 2 in[k+1] + in[k+2], k in [0, num_chirps - 2).
      auto x0 = matx::slice(
          in, {0, 0, 0},
          {matx::matxEnd, num_chirps - 2, matx::matxEnd});
      auto x1 = matx::slice(
          in, {0, 1, 0},
          {matx::matxEnd, num_chirps - 1, matx::matxEnd});
      auto x2 = matx::slice(in, {0, 2, 0},
                            {matx::matxEnd, num_chirps, matx::matxEnd});
      using value_type = typename InOp::value_type;
      const value_type two{static_cast<typename value_type::value_type>(2),
                           {}};
      fn(x0 - two * x1 + x2);
      break;
    }
  }
}

// Run slow-time clutter suppression over a collapsed batch view.
//
//   in  — device pointer, (batch, num_chirps, num_range_bins),
//         cuda::std::complex<T>.
//   out — device pointer, (batch, num_chirps_out, num_range_bins),
//         cuda::std::complex<T>, preallocated by the caller;
//         num_chirps_out must equal clutter_num_chirps_out(method, ...).
//
// Stateless (no setup cost). All work is enqueued on `stream`; no
// synchronization. Validation and method semantics: see with_clutter_expr.
template <typename T>
void run_clutter_stage(ClutterFilterMethod method, const void* in,
                       int64_t batch, int64_t num_chirps,
                       int64_t num_range_bins, void* out,
                       cudaStream_t stream) {
  auto in_view =
      make_const_complex_view3<T>(in, batch, num_chirps, num_range_bins);
  const int64_t num_chirps_out = clutter_num_chirps_out(method, num_chirps);
  auto out_view =
      make_complex_view3<T>(out, batch, num_chirps_out, num_range_bins);
  matx::cudaExecutor exec{stream};
  with_clutter_expr(method, in_view, num_chirps,
                    [&](const auto& expr) { (out_view = expr).run(exec); });
}

}  // namespace holoscan::radar::detail
