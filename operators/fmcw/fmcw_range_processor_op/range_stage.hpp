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

#include <cstdint>
#include <optional>
#include <string>
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
// Dechirp reference convention (must match the test simulator,
// tests/cpp/utils/point_target_sim.hpp): when cfg.waveform has
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

}  // namespace holoscan::radar::detail
