// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Range-compression stage: the pure kernel behind FMCWRangeProcessorOp.
//
// The fused FMCWRangeDopplerProcessorOp calls this same code path, which is
// what guarantees the split-vs-fused parity property by
// construction. The signatures below are therefore FROZEN for the parallel
// implementation phase: implement the bodies, do not change the interfaces
// (escalate to the orchestrator if a change seems necessary).
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

// Kernel-side caches, owned by the operator across ticks. Window
// coefficients and dechirp references are rebuilt only for unseen
// configurations (ConfigCache implements the warn-once rule); FFT plan
// reuse itself comes from MatX's internal plan cache, so `seen_fft_keys`
// exists only to decide when to warn.
template <typename T>
struct RangeStageState {
  ConfigCache<matx::tensor_t<T, 1>> windows;
  ConfigCache<matx::tensor_t<cuda::std::complex<T>, 1>> dechirp_references;
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

// Build the device dechirp reference for one chirp: measured samples when
// present (validated against samples_per_chirp), otherwise synthesized as
// r[s] = exp(+j*pi*slope*t_s^2), t_s = s / sample_rate (the simulator's
// locked sign convention; dechirp is in * conj(r)).
template <typename T>
matx::tensor_t<cuda::std::complex<T>, 1> build_dechirp_reference(
    const FMCWWaveformDescriptor& waveform, int64_t samples_per_chirp,
    cudaStream_t stream) {
  std::vector<std::complex<double>> host;
  if (!waveform.reference_samples().empty()) {
    if (static_cast<int64_t>(waveform.reference_samples().size()) !=
        samples_per_chirp) {
      throw std::invalid_argument(
          "range stage: waveform.reference_samples length (" +
          std::to_string(waveform.reference_samples().size()) +
          ") does not match samples_per_chirp (" +
          std::to_string(samples_per_chirp) + ")");
    }
    host = waveform.reference_samples();
  } else {
    if (!waveform.is_valid()) {
      throw std::invalid_argument(
          "range stage: an empty (default-constructed) waveform descriptor "
          "cannot synthesize the dechirp reference");
    }
    const double slope = waveform.slope();
    host.resize(static_cast<size_t>(samples_per_chirp));
    for (int64_t s = 0; s < samples_per_chirp; ++s) {
      const double t = static_cast<double>(s) / waveform.sample_rate();
      const double phase = kRangeStagePi * slope * t * t;
      host[static_cast<size_t>(s)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }
  return upload_complex_coefficients<T>(host, stream);
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
  const matx::tensor_t<T, 1>* window = nullptr;
  if (apply_window) {
    window = &state.windows.get_or_build(
        window_cache_key(cfg.window, samples_per_chirp), warn_on_rebuild,
        "window coefficient", [&] {
          return upload_coefficients<T>(
              generate_window(cfg.window, samples_per_chirp), stream);
        });
  }

  const uint64_t fft_size = static_cast<uint64_t>(nfft);
  if (complex_input) {
    auto in_view = make_const_complex_view3<T>(in, batch, num_chirps,
                                               samples_per_chirp);
    if (cfg.apply_dechirp || apply_window) {
      // Materialize the pre-FFT product into a stream-ordered temporary so
      // the FFT (and its zero-padding path) consumes a plain tensor.
      auto pre = matx::make_tensor<cuda::std::complex<T>>(
          {batch, num_chirps, samples_per_chirp},
          matx::MATX_ASYNC_DEVICE_MEMORY, stream);
      if (cfg.apply_dechirp) {
        const auto& reference = state.dechirp_references.get_or_build(
            dechirp_cache_key(*cfg.waveform, samples_per_chirp),
            warn_on_rebuild, "dechirp reference", [&] {
              return build_dechirp_reference<T>(*cfg.waveform,
                                                samples_per_chirp, stream);
            });
        auto conj_ref = matx::clone<3>(
            matx::conj(reference), {batch, num_chirps, matx::matxKeepDim});
        if (apply_window) {
          auto taps = matx::clone<3>(*window,
                                     {batch, num_chirps, matx::matxKeepDim});
          (pre = in_view * conj_ref * taps).run(exec);
        } else {
          (pre = in_view * conj_ref).run(exec);
        }
      } else {
        auto taps = matx::clone<3>(*window,
                                   {batch, num_chirps, matx::matxKeepDim});
        (pre = in_view * taps).run(exec);
      }
      (out_view = matx::fft(pre, fft_size, norm)).run(exec);
    } else {
      (out_view = matx::fft(in_view, fft_size, norm)).run(exec);
    }
  } else {
    auto in_view = make_real_view3<T>(in, batch, num_chirps,
                                      samples_per_chirp);
    if (apply_window) {
      auto pre = matx::make_tensor<T>({batch, num_chirps, samples_per_chirp},
                                      matx::MATX_ASYNC_DEVICE_MEMORY, stream);
      auto taps = matx::clone<3>(*window,
                                 {batch, num_chirps, matx::matxKeepDim});
      (pre = in_view * taps).run(exec);
      (out_view = matx::fft(pre, fft_size, norm)).run(exec);
    } else {
      (out_view = matx::fft(in_view, fft_size, norm)).run(exec);
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
  auto device = upload_coefficients<T>(host, stream);
  cudaMemcpyAsync(out, device.Data(),
                  sizeof(T) * static_cast<size_t>(num_bins),
                  cudaMemcpyDeviceToDevice, stream);
}

}  // namespace holoscan::radar::detail
