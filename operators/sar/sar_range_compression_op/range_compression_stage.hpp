// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SAR range-compression stage: the pure kernel behind SARRangeCompressionOp.
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/window.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// Effective (override-resolved) configuration for one tick.
struct RangeCompressionConfig {
  WindowSpec window{};
  std::optional<LFMWaveformDescriptor> waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  std::optional<int64_t> nfft{};  // >= num_samples when set
  double propagation_speed{kDefaultPropagationSpeed};
};

// Caches owned by the operator across ticks; rebuilt only for unseen
// configurations (ConfigCache implements the warn-once rule). Transform-plan
// reuse itself comes from MatX's internal plan cache, so `seen_fft_keys`
// exists only to decide when to warn.
//
// `scratch` holds the un-shifted transform output. The centring shift is a
// gather, and reading and writing one buffer through a permuted index is not
// safe in place, so the transform lands here and the shift copies out of it.
// Cached by shape rather than allocated per tick: a per-tick cudaMalloc costs
// ~0.09 ms flat, which the CPHDReaderOp harness measured and which would
// dominate this stage at small apertures.
template <typename T>
struct RangeCompressionState {
  ConfigCache<matx::tensor_t<T, 1>> windows;
  ConfigCache<bool> seen_fft_keys;
  ConfigCache<matx::tensor_t<cuda::std::complex<T>, 3>> scratch;
};

// Output bin count: the transform length, defaulting to the input's fast-time
// extent. Unlike the FMCW stage there is no real-input half-spectrum case —
// phase history is complex by contract.
inline int64_t compression_num_bins(int64_t num_samples,
                                    const std::optional<int64_t>& nfft) {
  return nfft.has_value() ? *nfft : num_samples;
}

// Fast-time frequency sample spacing, in Hz.
//
// **`bandwidth / (num_samples - 1)`, not `bandwidth / num_samples`**: the
// bandwidth is the span from the first sample's frequency to the last one's,
// so N samples divide it into N-1 steps. That is what CPHD's `FxBand` against
// per-vector `SC0` / `SCSS` describes, and what the in-repo simulator
// (`tools/sar_sim/sar_sim/signal.py` → `frequency_axis`) generates. The two
// spellings differ by one part in N, which is a range-scale error that a
// single-target test at the reference range cannot see -- so the axis test
// places a target away from centre.
inline double frequency_spacing(const LFMWaveformDescriptor& waveform,
                                int64_t num_samples) {
  if (num_samples < 2) {
    throw std::invalid_argument(
        "range compression stage: the range axis needs at least 2 fast-time "
        "samples to have a frequency spacing");
  }
  return waveform.bandwidth() / static_cast<double>(num_samples - 1);
}

// Range bin spacing in metres: a delay bin of 1 / (nfft * df) is a two-way
// delay, so the one-way range step is c/2 times it. Zero-padding refines this
// while leaving the total span (nfft * spacing) unchanged, which is what makes
// nfft an interpolation control rather than a resolution one.
inline double range_bin_spacing(const RangeCompressionConfig& cfg,
                                int64_t num_samples, int64_t num_bins) {
  const double df = frequency_spacing(*cfg.waveform, num_samples);
  return cfg.propagation_speed /
         (2.0 * static_cast<double>(num_bins) * df);
}

// Run range compression over a collapsed batch view.
//
//   in       — device pointer, (batch, num_pulses, num_samples),
//              cuda::std::complex<T>.
//   out      — device pointer, (batch, num_pulses, num_bins),
//              cuda::std::complex<T>, preallocated by the caller.
//   num_bins — must equal compression_num_bins(num_samples, cfg.nfft).
//   warn_on_rebuild — true when this tick's config came from an override that
//              differs from the previous effective config.
//
// Steps, per the locked contract: window along fast-time, zero-pad to nfft,
// inverse-transform under cfg.fft_norm, and fftshift the range axis so zero
// differential range lands at bin num_bins/2. All work is enqueued on
// `stream`; no synchronization.
template <typename T>
void run_range_compression(const RangeCompressionConfig& cfg,
                           RangeCompressionState<T>& state, const void* in,
                           int64_t batch, int64_t num_pulses,
                           int64_t num_samples, void* out, int64_t num_bins,
                           bool warn_on_rebuild, cudaStream_t stream);

// Fill the centred range axis (metres, relative to the derived reference
// range) into a device buffer of length num_bins:
//   range_bins[i] = (i - num_bins/2) * range_bin_spacing.
// Requires cfg.waveform. Enqueued on `stream`.
template <typename T>
void fill_compressed_range_bins(const RangeCompressionConfig& cfg,
                                int64_t num_samples, int64_t num_bins, T* out,
                                cudaStream_t stream);

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

template <typename T>
void run_range_compression(const RangeCompressionConfig& cfg,
                           RangeCompressionState<T>& state, const void* in,
                           int64_t batch, int64_t num_pulses,
                           int64_t num_samples, void* out, int64_t num_bins,
                           bool warn_on_rebuild, cudaStream_t stream) {
  const int64_t nfft = cfg.nfft.value_or(num_samples);
  if (nfft < num_samples) {
    throw std::invalid_argument(
        "range compression stage: nfft (" + std::to_string(nfft) +
        ") must be >= num_samples (" + std::to_string(num_samples) + ")");
  }
  if (num_bins != compression_num_bins(num_samples, cfg.nfft)) {
    throw std::invalid_argument(
        "range compression stage: num_bins (" + std::to_string(num_bins) +
        ") does not match the configured transform length (expected " +
        std::to_string(compression_num_bins(num_samples, cfg.nfft)) + ")");
  }

  matx::cudaExecutor exec{stream};
  const matx::FFTNorm norm = to_matx_norm(cfg.fft_norm);
  auto in_view =
      make_const_complex_view3<T>(in, batch, num_pulses, num_samples);
  auto out_view = make_complex_view3<T>(out, batch, num_pulses, num_bins);

  // Warn-once bookkeeping for override-selected transform lengths; the plan
  // itself is cached inside MatX.
  {
    const std::string fft_key =
        "c2c:nfft=" + std::to_string(nfft) +
        ":len=" + std::to_string(num_samples) +
        ":batch=" + std::to_string(batch * num_pulses) +
        ":dtype=" + (sizeof(T) == 4 ? "f32" : "f64");
    state.seen_fft_keys.get_or_build(fft_key, warn_on_rebuild,
                                     "transform length", [] { return true; });
  }

  const std::string scratch_key = "b=" + std::to_string(batch) + ";p=" +
                                  std::to_string(num_pulses) + ";n=" +
                                  std::to_string(num_bins);
  // Not warn-worthy: this is our own staging buffer, not a configuration the
  // caller chose, so it is cached silently.
  // Plain device memory, not stream-ordered: the cache outlives every tick
  // and is freed when the operator is destroyed, by which time the Holoscan
  // stream this tick ran on is gone. A stream-ordered allocation would free
  // itself on that dead stream.
  auto& scratch = const_cast<matx::tensor_t<cuda::std::complex<T>, 3>&>(
      state.scratch.get_or_build(scratch_key, false, "range scratch", [&] {
        return matx::make_tensor<cuda::std::complex<T>>(
            {batch, num_pulses, num_bins}, matx::MATX_DEVICE_MEMORY);
      }));

  const bool apply_window = cfg.window.type != WindowType::kNone;
  const uint64_t fft_size = static_cast<uint64_t>(nfft);
  if (apply_window) {
    const auto& taps = state.windows.get_or_build(
        window_cache_key(cfg.window, num_samples), warn_on_rebuild,
        "window coefficient", [&] {
          return upload_coefficients<T>(
              generate_window(cfg.window, num_samples), stream);
        });
    const auto bcast =
        matx::clone<3>(taps, {batch, num_pulses, matx::matxKeepDim});
    (scratch = matx::ifft(in_view * bcast, fft_size, norm)).run(exec);
  } else {
    (scratch = matx::ifft(in_view, fft_size, norm)).run(exec);
  }

  // The centring shift, applied as a separate gather out of the scratch
  // buffer. MatX's fftshift1D rotates the last axis by (N + 1) / 2, which is
  // numpy's fftshift and puts the transform's bin 0 -- zero differential
  // range -- at output bin N/2 for even N. `N/2`, not `(N - 1)/2`: the two
  // differ by half a bin, which is a phase error a magnitude test cannot see.
  (out_view = matx::fftshift1D(scratch)).run(exec);
}

template <typename T>
void fill_compressed_range_bins(const RangeCompressionConfig& cfg,
                                int64_t num_samples, int64_t num_bins, T* out,
                                cudaStream_t stream) {
  if (!cfg.waveform.has_value() || !cfg.waveform->is_valid()) {
    throw std::invalid_argument(
        "range compression stage: range_bins requires a valid (non-empty) "
        "waveform descriptor");
  }
  const double spacing = range_bin_spacing(cfg, num_samples, num_bins);
  const int64_t centre = num_bins / 2;

  std::vector<T> host(static_cast<size_t>(num_bins));
  for (int64_t i = 0; i < num_bins; ++i) {
    host[static_cast<size_t>(i)] =
        static_cast<T>(static_cast<double>(i - centre) * spacing);
  }
  // Stream-ordered copy, synchronized before the host staging vector goes out
  // of scope (mirrors fill_range_bins in the FMCW range stage).
  if (cudaMemcpyAsync(out, host.data(),
                      sizeof(T) * static_cast<size_t>(num_bins),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error(
        "range compression stage: range_bins H2D copy failed");
  }
  cudaStreamSynchronize(stream);
}

}  // namespace holoscan::radar::detail
