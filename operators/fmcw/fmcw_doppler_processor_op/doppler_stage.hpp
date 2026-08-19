// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Doppler-compression stage: the pure kernel behind FMCWDopplerProcessorOp.
//
// The fused FMCWRangeDopplerProcessorOp consumes the same code path — the
// expression entry point below, fed the single-source clutter expression
// (clutter_stage.hpp with_clutter_expr) — which is what guarantees the
// split-vs-fused parity property by construction, at expression
// granularity, and it is enforced by the UlpParity tests. The
// signatures below are the stable stage API: extend via additive overloads.
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/tensor/matx_utils.hpp"
#include "radar_common/dsp/window.hpp"
#include "radar_common/descriptors/window_spec.hpp"

namespace holoscan::radar::detail {

// Effective (override-resolved) Doppler-stage configuration for one tick.
struct DopplerStageConfig {
  WindowSpec window{};
  bool center_zero_doppler{true};
  std::optional<FMCWWaveformDescriptor> waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  std::optional<int64_t> nfft{};  // >= num_chirps when set
};

// Kernel-side caches, owned by the operator across ticks (see
// RangeStageState for the caching/warn-once contract).
template <typename T>
struct DopplerStageState {
  ConfigCache<matx::tensor_t<T, 1>> windows;
  ConfigCache<bool> seen_fft_keys;
};

// Number of Doppler bins produced: nfft when set, else num_chirps.
inline int64_t doppler_num_bins(int64_t num_chirps,
                                const std::optional<int64_t>& nfft) {
  return nfft.has_value() ? *nfft : num_chirps;
}

// Run Doppler compression over a rank-3 MatX operator expression.
//
//   in_op       — rank-3 MatX operator, (batch, num_chirps,
//                 num_range_bins), value_type cuda::std::complex<T>.
//                 num_chirps is the POST-clutter chirp count when the caller
//                 composed a clutter expression (with_clutter_expr).
//   out         — device pointer, (batch, num_doppler, num_range_bins),
//                 cuda::std::complex<T>, preallocated by the caller;
//                 num_doppler must equal doppler_num_bins(...).
//
// Cached artifacts (window taps, FFT keys) are rebuilt only for unseen
// configurations; ConfigCache warns once when it sees a new one after the
// first.
//
// Steps (per the locked contract): window multiply along slow-time, FFT
// along the slow-time (second-to-last) axis with nfft zero-padding and
// cfg.fft_norm normalization, then fftshift along the Doppler axis when
// cfg.center_zero_doppler (matching numpy.fft.fftshift semantics, including
// odd lengths); else DC stays at bin 0. All work is enqueued on `stream`;
// no synchronization.
//
// Throws std::invalid_argument for contract violations
// (nfft < num_chirps, ...).
template <typename T, typename InOp>
void run_doppler_stage(const DopplerStageConfig& cfg,
                       DopplerStageState<T>& state, const InOp& in_op,
                       void* out, int64_t num_doppler, cudaStream_t stream) {
  static_assert(matx::is_matx_op<InOp>(),
                "doppler stage: in_op must be a MatX operator or tensor");
  static_assert(InOp::Rank() == 3,
                "doppler stage: in_op must be rank-3 (batch, chirps, bins)");
  static_assert(
      std::is_same_v<typename InOp::value_type, cuda::std::complex<T>>,
      "doppler stage: in_op value_type must be cuda::std::complex<T>");
  const int64_t batch = in_op.Size(0);
  const int64_t num_chirps = in_op.Size(1);
  const int64_t num_range_bins = in_op.Size(2);
  if (batch < 1 || num_chirps < 1 || num_range_bins < 1) {
    throw std::invalid_argument(
        "doppler stage: input view (batch, num_chirps, num_range_bins) "
        "must be non-empty, got (" + std::to_string(batch) + ", " +
        std::to_string(num_chirps) + ", " + std::to_string(num_range_bins) +
        ")");
  }
  if (cfg.nfft.has_value() && *cfg.nfft < num_chirps) {
    throw std::invalid_argument(
        "doppler stage: nfft (" + std::to_string(*cfg.nfft) +
        ") must be >= num_chirps (" + std::to_string(num_chirps) + ")");
  }
  if (num_doppler != doppler_num_bins(num_chirps, cfg.nfft)) {
    throw std::invalid_argument(
        "doppler stage: output num_doppler (" + std::to_string(num_doppler) +
        ") does not match doppler_num_bins(num_chirps, nfft) (" +
        std::to_string(doppler_num_bins(num_chirps, cfg.nfft)) + ")");
  }

  auto out_view =
      make_complex_view3<T>(out, batch, num_doppler, num_range_bins);
  matx::cudaExecutor exec{stream};

  // FFT plan reuse comes from MatX's internal plan cache; seen_fft_keys
  // exists purely to decide when a reconfigured length warns.
  const std::string fft_key =
      "nfft=" + std::to_string(num_doppler) +
      ",len=" + std::to_string(num_chirps) +
      ",batch=" + std::to_string(batch * num_range_bins) +
      ",dtype=" + std::string(sizeof(T) == 4 ? "complex64" : "complex128");
  state.seen_fft_keys.get_or_build(fft_key, "Doppler FFT plan",
                                   []() { return true; });

  // The slow-time axis is permuted innermost so the window multiply, FFT,
  // and fftshift all act along the last axis. The pre-FFT expression feeds
  // matx::fft directly: MatX materializes operator inputs into a
  // stream-ordered temp in a single fused pass (and zero-pads it up to
  // num_doppler when nfft > num_chirps). The FFT output is permuted back
  // into the caller's layout through a nested expression — the FFT runs
  // into an internal stream-ordered temp during PreRun, and the
  // permute(+shift) is one elementwise kernel into out_view.
  const auto fft_into_out = [&](const auto& pre) {
    auto spectrum = matx::fft(pre, static_cast<uint64_t>(num_doppler),
                              to_matx_norm(cfg.fft_norm));
    if (cfg.center_zero_doppler) {
      // matx::fftshift1D maps output index i to input (i + (n + 1) / 2) % n
      // along the last axis, which is exactly numpy.fft.fftshift's
      // out[i] = in[(i + ceil(n / 2)) % n] for both even and odd n.
      (out_view = matx::permute(matx::fftshift1D(spectrum), {0, 2, 1}))
          .run(exec);
    } else {
      (out_view = matx::permute(spectrum, {0, 2, 1})).run(exec);
    }
  };

  if (cfg.window.type != WindowType::kNone) {
    const std::string key = window_cache_key(cfg.window, num_chirps);
    const auto& window = state.windows.get_or_build(
        key, "Doppler window", [&]() {
          return upload_coefficients<T>(generate_window(cfg.window,
                                                        num_chirps),
                                        stream);
        });
    auto window_bcast = matx::clone<3>(
        window, {batch, num_range_bins, matx::matxKeepDim});
    fft_into_out(matx::permute(in_op, {0, 2, 1}) * window_bcast);
  } else {
    fft_into_out(matx::permute(in_op, {0, 2, 1}));
  }
}

// Run Doppler compression over a collapsed batch view.
//
//   in — device pointer, (batch, num_chirps, num_range_bins),
//        cuda::std::complex<T>.
//
// Thin wrapper over the expression entry point above, which holds all
// validation and the math — the two entry points cannot drift.
template <typename T>
void run_doppler_stage(const DopplerStageConfig& cfg,
                       DopplerStageState<T>& state, const void* in,
                       int64_t batch, int64_t num_chirps,
                       int64_t num_range_bins, void* out, int64_t num_doppler,
                       cudaStream_t stream) {
  run_doppler_stage<T>(
      cfg, state,
      make_const_complex_view3<T>(in, batch, num_chirps, num_range_bins),
      out, num_doppler, stream);
}

// Compute the velocity_bins axis (m/s) for the current configuration into a
// device buffer of length num_doppler: v = -f_d * lambda / 2 with
// f_d the per-bin Doppler frequency (PRF = 1 / chirp_repetition_interval),
// ordered to match the rd_maps Doppler axis (i.e. accounting for
// cfg.center_zero_doppler). Requires cfg.waveform. Enqueued on `stream`.
template <typename T>
void fill_velocity_bins(const DopplerStageConfig& cfg, int64_t num_doppler,
                        T* out, cudaStream_t stream) {
  if (!cfg.waveform.has_value()) {
    throw std::invalid_argument(
        "doppler stage: velocity_bins requires a waveform descriptor");
  }
  if (num_doppler < 1) {
    throw std::invalid_argument(
        "doppler stage: velocity_bins length must be >= 1, got " +
        std::to_string(num_doppler));
  }
  if (!cfg.waveform->is_valid()) {
    throw std::invalid_argument(
        "doppler stage: velocity_bins requires a valid (non-empty) waveform "
        "descriptor");
  }
  const double prf = 1.0 / cfg.waveform->chirp_repetition_interval();
  const double wavelength = cfg.waveform->wavelength();

  // Per-bin Doppler frequency on the (possibly centered) axis, then
  // v = -f_d * lambda / 2. This must agree exactly with the independent
  // reference in tests/cpp/utils/bin_model.hpp (expected_velocity_axis).
  std::vector<T> axis(static_cast<size_t>(num_doppler));
  for (int64_t i = 0; i < num_doppler; ++i) {
    int64_t idx = i;
    if (cfg.center_zero_doppler) {
      // Position i on the shifted axis corresponds to unshifted bin
      // (i - n/2) mod n (numpy.fft.fftshift semantics).
      idx = (i - num_doppler / 2 + num_doppler) % num_doppler;
    }
    double f = static_cast<double>(idx) / static_cast<double>(num_doppler) *
               prf;
    if (f >= prf / 2.0) {
      f -= prf;
    }
    axis[static_cast<size_t>(i)] = static_cast<T>(-f * wavelength / 2.0);
  }

  if (cudaMemcpyAsync(out, axis.data(), sizeof(T) * axis.size(),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error(
        "doppler stage: velocity_bins host-to-device upload failed");
  }
  // The host staging vector is freed on return: block until the copy lands
  // (mirrors upload_coefficients; callers enqueue this before the Doppler
  // FFT so the wait does not cover the hot path).
  cudaStreamSynchronize(stream);
}

}  // namespace holoscan::radar::detail
