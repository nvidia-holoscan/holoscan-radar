// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"

namespace holoscan::radar::detail {

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

// Per-TX fractional Doppler offsets (cycles/chirp, i.e. fraction of PRF in
// [0, 1)) resolved from the descriptor: the `doppler_offsets` compact form is
// used directly; a linear-phase `phase_codes` form is reduced to offsets;
// arbitrary / non-linear `phase_codes` are rejected (see OPEN_ITEMS). Requires
// a validated descriptor.
std::vector<double> ddm_resolve_offsets(const DDMCodeDescriptor& code);

// num_doppler_per_tx = num_doppler / num_tx - guard_bands. Throws if
// num_doppler is not divisible by num_tx (imposed layout requirement) or the
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

}  // namespace holoscan::radar::detail
