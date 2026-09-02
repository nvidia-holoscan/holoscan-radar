// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"

namespace holoscan::radar::detail {

// Kernel knobs for one tick. multiplexing_scheme and ddm_code are
// Params-only (static). waveform is the live-merged descriptor: empty
// Params / override is rejected in resolve_params, so this is always
// valid after that call. TDM uses PRI/carrier; DDM validates and ignores.
struct CompensationStageConfig {
  MultiplexingScheme multiplexing_scheme{MultiplexingScheme::kUnspecified};
  FMCWWaveformDescriptor waveform{};
  DDMCodeDescriptor ddm_code{};
};

struct TdmRdMapsLayout {
  std::vector<int64_t> leading;
  int64_t batch{1};
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t num_doppler{0};
  int64_t num_range{0};
};

TdmRdMapsLayout parse_tdm_rd_maps_layout(const holoscan::Tensor& tensor,
                                       const char* port_name);

std::vector<int64_t> tdm_output_rd_shape(const TdmRdMapsLayout& in);

// Per-(tx, doppler) complex correction factors for uniform TDM inter-TX
// alignment. Assumes the Doppler axis matches FMCWDopplerProcessorOp with
// center_zero_doppler=true (the operator default).
std::vector<std::complex<double>> build_tdm_correction_table(
    const FMCWWaveformDescriptor& waveform, int64_t num_tx,
    int64_t num_doppler);

std::string tdm_correction_cache_key(const FMCWWaveformDescriptor& waveform,
                                     int64_t num_tx, int64_t num_doppler);

template <typename T>
void run_tdm_compensation_stage(
    const void* input, void* output, const TdmRdMapsLayout& layout,
    const void* d_correction, cudaStream_t stream);

// DDM (Doppler-division) input layout: the M TX sit in distinct Doppler
// sub-bands, so the input has no explicit M_tx axis
// (..., N_rx, num_doppler, num_range).
struct DdmRdMapsLayout {
  std::vector<int64_t> leading;
  int64_t batch{1};
  int64_t num_rx{0};
  int64_t num_doppler{0};
  int64_t num_range{0};
};

DdmRdMapsLayout parse_ddm_rd_maps_layout(const holoscan::Tensor& tensor,
                                         const char* port_name);

// Classification of DDM code type: linear-phase codes reduce to Doppler
// offsets (gather path); non-linear codes require conjugate correlation.
enum class DdmCodeMode { kLinear, kNonLinear };

// Detect whether phase_codes are linear-phase (constant per-chirp increment
// within 1e-3 rad) or non-linear. Returns kLinear for doppler_offsets form.
DdmCodeMode ddm_classify_codes(const DDMCodeDescriptor& code);

// Per-TX fractional Doppler offsets (cycles/chirp, i.e. fraction of PRF in
// [0, 1)) resolved from the descriptor: the `doppler_offsets` compact form is
// used directly; a linear-phase `phase_codes` form is reduced to offsets.
// Throws if codes are non-linear (caller must check ddm_classify_codes first).
// Requires a validated descriptor.
std::vector<double> ddm_resolve_offsets(const DDMCodeDescriptor& code);

// num_doppler_per_tx = num_doppler / num_tx - guard_bands. Throws if
// num_doppler is not divisible by num_tx (equal-width bands) or the
// resulting band width is < 1.
int64_t ddm_num_doppler_per_tx(int64_t num_doppler, int64_t num_tx,
                               int64_t guard_bands);

std::vector<int64_t> ddm_output_rd_shape(const DdmRdMapsLayout& in,
                                         int64_t num_tx,
                                         int64_t num_doppler_per_tx);

// Gather table sized num_tx * num_doppler_per_tx: for each (tx, out_bin) the
// source Doppler index into the input. Input and output Doppler axes are
// center-zero (matching FMCWDopplerProcessorOp center_zero_doppler=true); the
// table encapsulates the offset shift + fftshift bookkeeping + guard discard so
// the kernel is a pure gather and every lane is aligned to the common reduced
// Doppler axis.
std::vector<int64_t> build_ddm_demux_table(const std::vector<double>& offsets,
                                           int64_t num_tx, int64_t num_doppler,
                                           int64_t num_doppler_per_tx);

std::string ddm_demux_cache_key(const DDMCodeDescriptor& code,
                                int64_t num_doppler);

template <typename T>
void run_ddm_demux_stage(const void* input, void* output,
                         const DdmRdMapsLayout& layout, int64_t num_tx,
                         int64_t num_doppler_per_tx, const int64_t* d_table,
                         cudaStream_t stream);

// DDM non-linear correlate path output shape: full Doppler axis preserved.
std::vector<int64_t> ddm_correlate_output_shape(const DdmRdMapsLayout& in,
                                                int64_t num_tx);

// FFT-correlation weights for the non-linear demux. With num_chirps == D,
//   out = ifft(fft(Y) * W_m)   (MatX/NumPy BACKWARD norms)
// equals (1/D) * circular conjugate-correlation of Y with C_m = DFT(c_m), and
// W_m[p] = conj(c_m[(-p) mod D]). Returns num_tx * num_doppler values,
// row-major (m, p).
std::vector<std::complex<double>> build_ddm_corr_weights(
    const DDMCodeDescriptor& code, int64_t num_doppler);

// Cache key for the non-linear correlate path (includes full code content).
std::string ddm_correlate_cache_key(const DDMCodeDescriptor& code,
                                    int64_t num_doppler);

// Release per-operator cuFFT plan / tile workspace owned by `owner`
// (pass the MIMOCoherentCompensationOp `this` pointer). Safe no-op if absent.
void release_ddm_correlate_scratch(void* owner);

// Conjugate-correlation DDM demux via FFT:
//   out[..., m, rx, :, r] = ifft(fft(Y[..., rx, :, r]) * W_m)
// with W from build_ddm_corr_weights. Equivalent to
//   out[..., m, rx, d, r] = (1/D) * sum_k Y[..., rx, k, r] *
//                            conj(C_m[(k-d) mod D]),  C_m = DFT(c_m).
// `owner` keys per-instance cuFFT/workspace scratch (not shared across ops).
template <typename T>
void run_ddm_correlate_stage(const void* input, void* output,
                             const DdmRdMapsLayout& layout, int64_t num_tx,
                             const void* d_corr_weights, void* owner,
                             cudaStream_t stream);

}  // namespace holoscan::radar::detail
