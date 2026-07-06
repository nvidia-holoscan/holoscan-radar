// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Doppler-compression stage: the pure kernel behind FMCWDopplerProcessorOp.
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

// Run Doppler compression over a collapsed batch view.
//
//   in          — device pointer, (batch, num_chirps, num_range_bins),
//                 cuda::std::complex<T>.
//   out         — device pointer, (batch, num_doppler, num_range_bins),
//                 cuda::std::complex<T>, preallocated by the caller;
//                 num_doppler must equal doppler_num_bins(...).
//   warn_on_rebuild — true when this tick's config came from an
//                 override that differs from the previous effective config.
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
template <typename T>
void run_doppler_stage(const DopplerStageConfig& cfg,
                       DopplerStageState<T>& state, const void* in,
                       int64_t batch, int64_t num_chirps,
                       int64_t num_range_bins, void* out, int64_t num_doppler,
                       bool warn_on_rebuild, cudaStream_t stream);

// Compute the velocity_bins axis (m/s) for the current configuration into a
// device buffer of length num_doppler: v = -f_d * lambda / 2 with
// f_d the per-bin Doppler frequency (PRF = 1 / chirp_repetition_interval),
// ordered to match the rd_maps Doppler axis (i.e. accounting for
// cfg.center_zero_doppler). Requires cfg.waveform. Enqueued on `stream`.
template <typename T>
void fill_velocity_bins(const DopplerStageConfig& cfg, int64_t num_doppler,
                        T* out, cudaStream_t stream);

}  // namespace holoscan::radar::detail
