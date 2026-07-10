// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Range-compression stage: the pure kernel behind FMCWRangeProcessorOp.
//
// The fused FMCWRangeDopplerProcessorOp calls this same code path, which is
// what guarantees the split-vs-fused parity property by construction. The
// signatures below are the stable stage API: extend via additive overloads.
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/tensor/matx_utils.hpp"
#include "radar_common/dsp/window.hpp"
#include "radar_common/descriptors/window_spec.hpp"

namespace holoscan::radar::detail {

// Effective (override-resolved) range-stage configuration for one tick.
struct RangeStageConfig {
  WindowSpec window{};
  bool apply_dechirp{false};
  std::optional<FMCWWaveformDescriptor> waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  double propagation_speed{kDefaultPropagationSpeed};
  std::optional<int64_t> nfft{};  // >= samples_per_chirp when set
};

// Kernel-side caches, owned by the operator across ticks; rebuilt only for
// unseen configurations (ConfigCache implements the warn-once rule). The
// complex-input path folds the window taps and the conjugated dechirp
// reference into one cached per-sample multiplier (`premultipliers`); the
// real-input path cannot dechirp and caches plain window taps. FFT plan
// reuse itself comes from MatX's internal plan cache, so `seen_fft_keys`
// exists only to decide when to warn.
template <typename T>
struct RangeStageState {
  ConfigCache<matx::tensor_t<cuda::std::complex<T>, 1>> premultipliers;
  ConfigCache<matx::tensor_t<T, 1>> windows;
  ConfigCache<bool> seen_fft_keys;
};

// Number of range bins produced for a given input, per the locked contract:
// complex IF → nfft (default samples_per_chirp); real IF → nfft/2 + 1
// (default samples_per_chirp/2 + 1).
inline int64_t range_num_bins(bool complex_input, int64_t samples_per_chirp,
                              const std::optional<int64_t>& nfft) {
  const int64_t n = nfft.has_value() ? *nfft : samples_per_chirp;
  return complex_input ? n : n / 2 + 1;
}

// Run range compression over a collapsed batch view.
//
//   in       — device pointer, (batch, num_chirps, samples_per_chirp),
//              cuda::std::complex<T> when complex_input else T.
//   out      — device pointer, (batch, num_chirps, num_bins),
//              cuda::std::complex<T>, preallocated by the caller.
//   num_bins — must equal range_num_bins(complex_input, samples, cfg.nfft).
//   warn_on_rebuild — true when this tick's config came from an
//              override that differs from the previous effective config.
//
// Steps (per the locked contract): optional dechirp (complex input only;
// conjugate-multiply against the cached reference chirp), window multiply
// along fast-time, FFT along the last axis with nfft zero-padding and
// cfg.fft_norm normalization. No fftshift (bin 0 = zero range). All work is
// enqueued on `stream`; no synchronization.
//
// Dechirp reference convention (must match the point-target simulator,
// radar_common/sim/point_target_sim.hpp): when cfg.waveform has
// reference_samples they are used directly; otherwise the reference is
// synthesized as r[s] = exp(+j*pi*slope*t_s^2) with t_s = s / sample_rate,
// and the dechirped signal is in[s] * conj(r[s]).
//
// Throws std::invalid_argument for contract violations (dechirp without a
// waveform or with real input, nfft < samples_per_chirp, ...).
template <typename T>
void run_range_stage(const RangeStageConfig& cfg, RangeStageState<T>& state,
                     const void* in, bool complex_input, int64_t batch,
                     int64_t num_chirps, int64_t samples_per_chirp, void* out,
                     int64_t num_bins, bool warn_on_rebuild,
                     cudaStream_t stream);

// Compute the range_bins axis (meters, relative to zero beat) for the
// current configuration into a device buffer of length num_bins:
//   range_bins[i] = propagation_speed * (i * fs / nfft) / (2 * slope).
// Requires cfg.waveform. Enqueued on `stream`.
template <typename T>
void fill_range_bins(const RangeStageConfig& cfg, int64_t samples_per_chirp,
                     int64_t num_bins, T* out, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline constexpr double kRangeStagePi = 3.14159265358979323846;

// Exact-round-trip textual form of a double, for canonical cache keys.
inline std::string range_stage_double_key(double value) {
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return std::string(buffer);
}

// FNV-1a over raw bytes; used to key measured dechirp references without
// serializing the full sample vector.
inline uint64_t range_stage_fnv1a(const void* data, size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= static_cast<uint64_t>(p[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

// Canonical ConfigCache key for the dechirp reference: only the descriptor
// fields the reference actually consumes (measured samples, or slope +
// sample rate for the synthesized form), plus the chirp length.
inline std::string dechirp_cache_key(const FMCWWaveformDescriptor& waveform,
                                     int64_t samples_per_chirp) {
  std::string key = "n=" + std::to_string(samples_per_chirp);
  if (!waveform.reference_samples().empty()) {
    key += ";ref_len=" + std::to_string(waveform.reference_samples().size());
    key += ";ref_hash=" +
           std::to_string(range_stage_fnv1a(
               waveform.reference_samples().data(),
               waveform.reference_samples().size() *
                   sizeof(std::complex<double>)));
  } else {
    key += ";slope=" + range_stage_double_key(waveform.slope());
    key += ";fs=" + range_stage_double_key(waveform.sample_rate());
  }
  return key;
}

// Host-side dechirp reference for one chirp (double precision): measured
// samples when present (validated against samples_per_chirp), otherwise
// synthesized as r[s] = exp(+j*pi*slope*t_s^2), t_s = s / sample_rate (the
// simulator's locked sign convention; dechirp is in * conj(r)).
inline std::vector<std::complex<double>> host_dechirp_reference(
    const FMCWWaveformDescriptor& waveform, int64_t samples_per_chirp) {
  if (!waveform.reference_samples().empty()) {
    if (static_cast<int64_t>(waveform.reference_samples().size()) !=
        samples_per_chirp) {
      throw std::invalid_argument(
          "range stage: waveform.reference_samples length (" +
          std::to_string(waveform.reference_samples().size()) +
          ") does not match samples_per_chirp (" +
          std::to_string(samples_per_chirp) + ")");
    }
    return waveform.reference_samples();
  }
  if (!waveform.is_valid()) {
    throw std::invalid_argument(
        "range stage: an empty (default-constructed) waveform descriptor "
        "cannot synthesize the dechirp reference");
  }
  const double slope = waveform.slope();
  std::vector<std::complex<double>> host(
      static_cast<size_t>(samples_per_chirp));
  for (int64_t s = 0; s < samples_per_chirp; ++s) {
    const double t = static_cast<double>(s) / waveform.sample_rate();
    const double phase = kRangeStagePi * slope * t * t;
    host[static_cast<size_t>(s)] =
        std::complex<double>(std::cos(phase), std::sin(phase));
  }
  return host;
}

// Cache key for the combined pre-FFT multiplier: exactly the window and
// dechirp inputs build_premultiplier consumes.
inline std::string premultiplier_cache_key(const RangeStageConfig& cfg,
                                           int64_t samples_per_chirp) {
  std::string key = "win=";
  key += cfg.window.type != WindowType::kNone
             ? window_cache_key(cfg.window, samples_per_chirp)
             : "none";
  key += ";dechirp=";
  key += cfg.apply_dechirp
             ? dechirp_cache_key(*cfg.waveform, samples_per_chirp)
             : "none";
  return key;
}

// Combined per-sample pre-FFT multiplier for the complex-input path:
// conj(reference) (dechirp), the window taps, or their product — combined
// in double precision once per configuration, then cast and uploaded.
// Requires cfg.apply_dechirp or an active window.
template <typename T>
matx::tensor_t<cuda::std::complex<T>, 1> build_premultiplier(
    const RangeStageConfig& cfg, int64_t samples_per_chirp,
    cudaStream_t stream) {
  std::vector<std::complex<double>> mult(
      static_cast<size_t>(samples_per_chirp), {1.0, 0.0});
  if (cfg.apply_dechirp) {
    const auto reference =
        host_dechirp_reference(*cfg.waveform, samples_per_chirp);
    for (size_t s = 0; s < mult.size(); ++s) {
      mult[s] = std::conj(reference[s]);
    }
  }
  if (cfg.window.type != WindowType::kNone) {
    const auto taps = generate_window(cfg.window, samples_per_chirp);
    for (size_t s = 0; s < mult.size(); ++s) {
      mult[s] *= taps[s];
    }
  }
  return upload_complex_coefficients<T>(mult, stream);
}

template <typename T>
void run_range_stage(const RangeStageConfig& cfg, RangeStageState<T>& state,
                     const void* in, bool complex_input, int64_t batch,
                     int64_t num_chirps, int64_t samples_per_chirp, void* out,
                     int64_t num_bins, bool warn_on_rebuild,
                     cudaStream_t stream) {
  const int64_t nfft = cfg.nfft.value_or(samples_per_chirp);
  if (nfft < samples_per_chirp) {
    throw std::invalid_argument(
        "range stage: nfft (" + std::to_string(nfft) +
        ") must be >= samples_per_chirp (" +
        std::to_string(samples_per_chirp) + ")");
  }
  if (num_bins != range_num_bins(complex_input, samples_per_chirp,
                                 cfg.nfft)) {
    throw std::invalid_argument(
        "range stage: num_bins (" + std::to_string(num_bins) +
        ") does not match the configured FFT length (expected " +
        std::to_string(range_num_bins(complex_input, samples_per_chirp,
                                      cfg.nfft)) +
        ")");
  }
  if (cfg.apply_dechirp) {
    if (!cfg.waveform.has_value()) {
      throw std::invalid_argument(
          "range stage: apply_dechirp = true requires a waveform "
          "descriptor to derive the dechirp reference");
    }
    if (!complex_input) {
      throw std::invalid_argument(
          "range stage: apply_dechirp = true requires complex-IF input "
          "(software dechirp of real-IF samples is not supported)");
    }
  }

  matx::cudaExecutor exec{stream};
  const matx::FFTNorm norm = to_matx_norm(cfg.fft_norm);
  auto out_view = make_complex_view3<T>(out, batch, num_chirps, num_bins);

  // Warn-once bookkeeping for override-selected FFT lengths; the plan
  // itself is cached inside MatX.
  {
    const std::string fft_key =
        std::string(complex_input ? "c2c" : "r2c") +
        ":nfft=" + std::to_string(nfft) +
        ":len=" + std::to_string(samples_per_chirp) +
        ":batch=" + std::to_string(batch * num_chirps) +
        ":dtype=" + (sizeof(T) == 4 ? "f32" : "f64");
    state.seen_fft_keys.get_or_build(fft_key, warn_on_rebuild, "FFT length",
                                     [] { return true; });
  }

  const bool apply_window = cfg.window.type != WindowType::kNone;
  const uint64_t fft_size = static_cast<uint64_t>(nfft);
  const auto bcast = [&](const auto& coeffs) {
    return matx::clone<3>(coeffs, {batch, num_chirps, matx::matxKeepDim});
  };
  // matx::fft accepts operator expressions: non-tensor inputs are
  // materialized into a stream-ordered internal temp (and zero-padding
  // allocates its own temp either way), so no manual staging tensor is
  // needed. Plain tensor views feed cuFFT with no extra copy.
  const auto fft_into_out = [&](const auto& pre) {
    (out_view = matx::fft(pre, fft_size, norm)).run(exec);
  };

  if (complex_input) {
    auto in_view = make_const_complex_view3<T>(in, batch, num_chirps,
                                               samples_per_chirp);
    if (cfg.apply_dechirp || apply_window) {
      const auto& mult = state.premultipliers.get_or_build(
          premultiplier_cache_key(cfg, samples_per_chirp), warn_on_rebuild,
          "pre-FFT multiplier", [&] {
            return build_premultiplier<T>(cfg, samples_per_chirp, stream);
          });
      fft_into_out(in_view * bcast(mult));
    } else {
      fft_into_out(in_view);
    }
  } else {
    auto in_view = make_real_view3<T>(in, batch, num_chirps,
                                      samples_per_chirp);
    if (apply_window) {
      const auto& taps = state.windows.get_or_build(
          window_cache_key(cfg.window, samples_per_chirp), warn_on_rebuild,
          "window coefficient", [&] {
            return upload_coefficients<T>(
                generate_window(cfg.window, samples_per_chirp), stream);
          });
      fft_into_out(in_view * bcast(taps));
    } else {
      fft_into_out(in_view);
    }
  }
}

template <typename T>
void fill_range_bins(const RangeStageConfig& cfg, int64_t samples_per_chirp,
                     int64_t num_bins, T* out, cudaStream_t stream) {
  if (!cfg.waveform.has_value()) {
    throw std::invalid_argument(
        "range stage: range_bins requires a waveform descriptor");
  }
  if (!cfg.waveform->is_valid()) {
    throw std::invalid_argument(
        "range stage: range_bins requires a valid (non-empty) waveform "
        "descriptor");
  }
  const double fs = cfg.waveform->sample_rate();
  const double slope = cfg.waveform->slope();
  const int64_t nfft = cfg.nfft.value_or(samples_per_chirp);

  std::vector<double> host(static_cast<size_t>(num_bins));
  for (int64_t i = 0; i < num_bins; ++i) {
    const double beat = static_cast<double>(i) * fs /
                        static_cast<double>(nfft);
    host[static_cast<size_t>(i)] =
        cfg.propagation_speed * beat / (2.0 * slope);
  }
  // Cast to the instantiation precision and copy straight into the output
  // buffer, stream-ordered; sync before the host staging vector goes out of
  // scope (mirrors fill_velocity_bins in doppler_stage.hpp).
  std::vector<T> cast(host.begin(), host.end());
  if (cudaMemcpyAsync(out, cast.data(),
                      sizeof(T) * static_cast<size_t>(num_bins),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("range stage: range_bins H2D copy failed");
  }
  cudaStreamSynchronize(stream);
}

}  // namespace holoscan::radar::detail
