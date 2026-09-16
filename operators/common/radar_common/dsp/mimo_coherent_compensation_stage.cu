// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "radar_common/math/constants.hpp"

namespace holoscan::radar::detail {

namespace {

constexpr int64_t kBlock = 256;

template <typename T>
using Complex = cuda::std::complex<T>;

double doppler_frequency_hz(int64_t shifted_bin, int64_t num_doppler,
                              double prf) {
  int64_t idx = (shifted_bin - num_doppler / 2 + num_doppler) % num_doppler;
  double f = static_cast<double>(idx) / static_cast<double>(num_doppler) * prf;
  if (f >= prf / 2.0) {
    f -= prf;
  }
  return f;
}

template <typename T>
__global__ void ddm_demux_gather_kernel(const Complex<T>* input,
                                        Complex<T>* output, int64_t num_rows,
                                        int64_t num_rx, int64_t num_doppler,
                                        int64_t num_range, int64_t num_tx,
                                        int64_t num_doppler_per_tx,
                                        const int64_t* table,
                                        const Complex<T>* phase_weights) {
  // Each block handles one output row (batch, tx, rx, out_doppler) of length
  // num_range. The source Doppler bin is constant along the row (one table
  // lookup per row); the inner loop is a fully coalesced gather along range.
  for (int64_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    int64_t rem = row;
    const int64_t out_d = rem % num_doppler_per_tx;
    rem /= num_doppler_per_tx;
    const int64_t rx = rem % num_rx;
    rem /= num_rx;
    const int64_t tx = rem % num_tx;
    rem /= num_tx;
    const int64_t b = rem;
    const int64_t src_d = table[tx * num_doppler_per_tx + out_d];
    const int64_t in_base =
        ((b * num_rx + rx) * num_doppler + src_d) * num_range;
    const int64_t out_base = row * num_range;
    for (int64_t r = threadIdx.x; r < num_range;
         r += static_cast<int64_t>(blockDim.x)) {
      output[out_base + r] = input[in_base + r] * phase_weights[tx];
    }
  }
}

template <typename T>
__global__ void apply_tdm_correction_kernel(
    const Complex<T>* input, Complex<T>* output, int64_t num_rows,
    int64_t num_tx, int64_t num_rx, int64_t num_doppler, int64_t num_range,
    const Complex<T>* correction) {
  // Each block handles one or more contiguous (batch, tx, rx, doppler) rows of
  // length num_range. The correction factor is constant along a row, so the
  // (tx, doppler) decode and the table lookup happen once per row (not per
  // element); the inner loop is a fully coalesced load / multiply / store.
  for (int64_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    int64_t rem = row;
    const int64_t d = rem % num_doppler;
    rem /= num_doppler;
    rem /= num_rx;
    const int64_t tx = rem % num_tx;
    const Complex<T> factor = correction[tx * num_doppler + d];
    const int64_t base = row * num_range;
    for (int64_t r = threadIdx.x; r < num_range;
         r += static_cast<int64_t>(blockDim.x)) {
      output[base + r] = input[base + r] * factor;
    }
  }
}

}  // namespace

TdmRdMapsLayout parse_tdm_rd_maps_layout(const holoscan::Tensor& tensor,
                                       const char* port_name) {
  const auto& shape = tensor.shape();
  if (shape.size() < 4) {
    throw std::invalid_argument(std::string(port_name) +
                                ": expected rank >= 4 "
                                "(..., M_tx, N_rx, num_doppler, num_range)");
  }
  TdmRdMapsLayout layout;
  layout.num_range = shape.back();
  layout.num_doppler = shape[shape.size() - 2];
  layout.num_rx = shape[shape.size() - 3];
  layout.num_tx = shape[shape.size() - 4];
  if (layout.num_tx <= 0 || layout.num_rx <= 0 || layout.num_doppler <= 0 ||
      layout.num_range <= 0) {
    throw std::invalid_argument(std::string(port_name) +
                                ": trailing axes must be positive");
  }
  layout.leading.assign(shape.begin(), shape.end() - 4);
  layout.batch = 1;
  for (int64_t dim : layout.leading) {
    if (dim <= 0) {
      throw std::invalid_argument(std::string(port_name) +
                                  ": leading batch axes must be positive");
    }
    layout.batch *= dim;
  }
  return layout;
}

std::vector<int64_t> tdm_output_rd_shape(const TdmRdMapsLayout& in) {
  std::vector<int64_t> shape = in.leading;
  shape.push_back(in.num_tx);
  shape.push_back(in.num_rx);
  shape.push_back(in.num_doppler);
  shape.push_back(in.num_range);
  return shape;
}

std::vector<int64_t> tdm_firing_slots(const TXSchedule& schedule,
                                      int64_t num_tx) {
  if (num_tx <= 0) {
    throw std::invalid_argument(
        "dense MIMO stage: num_tx must be positive");
  }
  if (schedule.num_tx == 0 && schedule.tx_order.empty() &&
      schedule.chirp_tx_index.empty()) {
    std::vector<int64_t> slots(static_cast<size_t>(num_tx));
    for (int64_t m = 0; m < num_tx; ++m) {
      slots[static_cast<size_t>(m)] = m;
    }
    return slots;
  }
  if (schedule.num_tx != num_tx) {
    throw std::invalid_argument(
        "dense MIMO stage: tx_schedule.num_tx (" +
        std::to_string(schedule.num_tx) + ") must match input M_tx (" +
        std::to_string(num_tx) + ")");
  }
  return schedule.firing_slot_by_tx();
}

std::vector<std::complex<double>> build_tdm_correction_table(
    const FMCWWaveformDescriptor& waveform, int64_t num_tx,
    int64_t num_doppler, const std::vector<int64_t>& slot_by_tx) {
  if (!waveform.is_valid()) {
    throw std::invalid_argument(
        "dense MIMO stage: waveform must be valid");
  }
  if (num_tx <= 0 || num_doppler <= 0) {
    throw std::invalid_argument(
        "dense MIMO stage: num_tx and num_doppler must be positive");
  }
  std::vector<int64_t> slots = slot_by_tx;
  if (slots.empty()) {
    slots.resize(static_cast<size_t>(num_tx));
    for (int64_t m = 0; m < num_tx; ++m) {
      slots[static_cast<size_t>(m)] = m;
    }
  } else if (static_cast<int64_t>(slots.size()) != num_tx) {
    throw std::invalid_argument(
        "dense MIMO stage: slot_by_tx length must equal num_tx");
  }
  const double pri = waveform.chirp_repetition_interval();
  const double prf = 1.0 / pri;
  const double t_inter_chirp = pri / static_cast<double>(num_tx);

  std::vector<std::complex<double>> table(
      static_cast<size_t>(num_tx * num_doppler));
  for (int64_t d = 0; d < num_doppler; ++d) {
    const double f_d = doppler_frequency_hz(d, num_doppler, prf);
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      const int64_t slot = slots[static_cast<size_t>(tx)];
      if (slot < 0 || slot >= num_tx) {
        throw std::invalid_argument(
            "dense MIMO stage: firing slot out of range [0, num_tx)");
      }
      const double phase =
          -2.0 * kPi * f_d * static_cast<double>(slot) * t_inter_chirp;
      table[static_cast<size_t>(tx * num_doppler + d)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }
  return table;
}

std::string tdm_correction_cache_key(const FMCWWaveformDescriptor& waveform,
                                     int64_t num_tx, int64_t num_doppler,
                                     const std::vector<int64_t>& slot_by_tx) {
  std::ostringstream os;
  os << num_tx << "|" << num_doppler << "|" << waveform.carrier_frequency()
     << "|" << waveform.chirp_repetition_interval() << "|slots";
  for (int64_t slot : slot_by_tx) {
    os << ":" << slot;
  }
  return os.str();
}

template <typename T>
void run_tdm_compensation_stage(const void* input, void* output,
                                const TdmRdMapsLayout& layout,
                                const void* d_correction, cudaStream_t stream) {
  const int64_t elems = layout.batch * layout.num_tx * layout.num_rx *
                        layout.num_doppler * layout.num_range;
  if (elems == 0) {
    return;
  }
  // One block per contiguous num_range row, grid-strided so the grid stays
  // bounded for large tensors while every block does coalesced work.
  const int64_t num_rows = layout.batch * layout.num_tx * layout.num_rx *
                           layout.num_doppler;
  constexpr int64_t kMaxBlocks = 65535;
  const int grid =
      static_cast<int>(num_rows < kMaxBlocks ? num_rows : kMaxBlocks);
  apply_tdm_correction_kernel<T><<<grid, kBlock, 0, stream>>>(
      static_cast<const Complex<T>*>(input),
      static_cast<Complex<T>*>(output), num_rows, layout.num_tx,
      layout.num_rx, layout.num_doppler, layout.num_range,
      static_cast<const Complex<T>*>(d_correction));
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("tdm compensation stage: kernel launch failed");
  }
}

template void run_tdm_compensation_stage<float>(
    const void* input, void* output, const TdmRdMapsLayout& layout,
    const void* d_correction, cudaStream_t stream);
template void run_tdm_compensation_stage<double>(
    const void* input, void* output, const TdmRdMapsLayout& layout,
    const void* d_correction, cudaStream_t stream);

DdmRdMapsLayout parse_ddm_rd_maps_layout(const holoscan::Tensor& tensor,
                                         const char* port_name) {
  const auto& shape = tensor.shape();
  if (shape.size() < 3) {
    throw std::invalid_argument(std::string(port_name) +
                                ": expected rank >= 3 "
                                "(..., N_rx, num_doppler, num_range) for DDM");
  }
  DdmRdMapsLayout layout;
  layout.num_range = shape.back();
  layout.num_doppler = shape[shape.size() - 2];
  layout.num_rx = shape[shape.size() - 3];
  if (layout.num_rx <= 0 || layout.num_doppler <= 0 || layout.num_range <= 0) {
    throw std::invalid_argument(std::string(port_name) +
                                ": trailing axes must be positive");
  }
  layout.leading.assign(shape.begin(), shape.end() - 3);
  layout.batch = 1;
  for (int64_t dim : layout.leading) {
    if (dim <= 0) {
      throw std::invalid_argument(std::string(port_name) +
                                  ": leading batch axes must be positive");
    }
    layout.batch *= dim;
  }
  return layout;
}

DdmCodeMode ddm_classify_codes(const DDMCodeDescriptor& code) {
  code.validate();
  if (!code.doppler_offsets.empty()) {
    return DdmCodeMode::kLinear;
  }
  if (code.phase_codes.empty()) {
    return DdmCodeMode::kLinear;  // will throw in resolve_offsets anyway
  }
  const int64_t num_tx = code.num_tx;
  const int64_t num_chirps =
      static_cast<int64_t>(code.phase_codes.size()) / num_tx;
  if (num_chirps < 2) {
    return DdmCodeMode::kNonLinear;
  }
  for (int64_t m = 0; m < num_tx; ++m) {
    const auto* code_m =
        &code.phase_codes[static_cast<size_t>(m * num_chirps)];
    const std::complex<double> ref_step = code_m[1] * std::conj(code_m[0]);
    const double delta = std::arg(ref_step);
    for (int64_t k = 0; k + 1 < num_chirps; ++k) {
      const std::complex<double> step = code_m[k + 1] * std::conj(code_m[k]);
      double d = std::arg(step) - delta;
      d -= kTwoPi * std::round(d / kTwoPi);
      if (std::abs(d) > 1e-3) {
        return DdmCodeMode::kNonLinear;
      }
    }
  }
  return DdmCodeMode::kLinear;
}

std::vector<double> ddm_resolve_offsets(const DDMCodeDescriptor& code) {
  code.validate();
  const int64_t num_tx = code.num_tx;
  std::vector<double> offsets(static_cast<size_t>(num_tx), 0.0);

  if (!code.doppler_offsets.empty()) {
    for (int64_t m = 0; m < num_tx; ++m) {
      double o = code.doppler_offsets[static_cast<size_t>(m)];
      o -= std::floor(o);
      offsets[static_cast<size_t>(m)] = o;
    }
    return offsets;
  }

  if (!code.phase_codes.empty()) {
    const int64_t num_chirps =
        static_cast<int64_t>(code.phase_codes.size()) / num_tx;
    if (num_chirps < 2) {
      throw std::invalid_argument(
          "DDMCodeDescriptor: phase_codes need >= 2 chirps per TX to infer a "
          "linear-phase Doppler offset");
    }
    for (int64_t m = 0; m < num_tx; ++m) {
      const auto* code_m =
          &code.phase_codes[static_cast<size_t>(m * num_chirps)];
      const std::complex<double> ref_step =
          code_m[1] * std::conj(code_m[0]);
      const double delta = std::arg(ref_step);
      for (int64_t k = 0; k + 1 < num_chirps; ++k) {
        const std::complex<double> step = code_m[k + 1] * std::conj(code_m[k]);
        double d = std::arg(step) - delta;
        d -= kTwoPi * std::round(d / kTwoPi);
        if (std::abs(d) > 1e-3) {
          throw std::invalid_argument(
              "dense MIMO stage: DDM phase_codes are not linear-"
              "phase; use the correlate path for non-linear codes");
        }
      }
      double o = delta / kTwoPi;
      o -= std::floor(o);
      offsets[static_cast<size_t>(m)] = o;
    }
    return offsets;
  }

  throw std::invalid_argument(
      "dense MIMO stage: DDM requires ddm_code with doppler_offsets "
      "or phase_codes");
}

std::vector<int64_t> ddm_exact_band_indices(const DDMCodeDescriptor& code,
                                            int64_t num_doppler) {
  if (ddm_classify_codes(code) != DdmCodeMode::kLinear) {
    throw std::invalid_argument(
        "DDM linear M-band layout does not accept non-linear phase_codes");
  }
  const int64_t num_tx = code.num_tx;
  (void)ddm_num_doppler_per_tx(
      num_doppler, num_tx, code.guard_bands);
  const int64_t band_width = num_doppler / num_tx;
  const auto offsets = ddm_resolve_offsets(code);
  std::vector<int64_t> band_by_tx(static_cast<size_t>(num_tx), -1);
  std::vector<bool> occupied(static_cast<size_t>(num_tx), false);
  constexpr double kBinTolerance = 1e-8;
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    const double bins =
        offsets[static_cast<size_t>(tx)] * static_cast<double>(num_doppler);
    const int64_t integer_bin = static_cast<int64_t>(std::llround(bins));
    if (std::abs(bins - static_cast<double>(integer_bin)) > kBinTolerance) {
      throw std::invalid_argument(
          "DDM linear offset for TX " + std::to_string(tx) +
          " is fractional for num_doppler=" + std::to_string(num_doppler));
    }
    const int64_t wrapped_bin =
        ((integer_bin % num_doppler) + num_doppler) % num_doppler;
    if (wrapped_bin % band_width != 0) {
      throw std::invalid_argument(
          "DDM linear offset for TX " + std::to_string(tx) +
          " is not aligned to an M-band origin");
    }
    const int64_t band = wrapped_bin / band_width;
    if (occupied[static_cast<size_t>(band)]) {
      throw std::invalid_argument(
          "DDM linear offsets must occupy every M-band exactly once");
    }
    occupied[static_cast<size_t>(band)] = true;
    band_by_tx[static_cast<size_t>(tx)] = band;
  }
  return band_by_tx;
}

std::vector<std::complex<double>> ddm_initial_phase_weights(
    const DDMCodeDescriptor& code) {
  if (ddm_classify_codes(code) != DdmCodeMode::kLinear) {
    throw std::invalid_argument(
        "DDM initial phase weights require linear phase codes");
  }
  std::vector<std::complex<double>> weights(
      static_cast<size_t>(code.num_tx), {1.0, 0.0});
  if (code.phase_codes.empty()) {
    return weights;
  }
  const int64_t num_chirps =
      static_cast<int64_t>(code.phase_codes.size()) / code.num_tx;
  for (int64_t tx = 0; tx < code.num_tx; ++tx) {
    weights[static_cast<size_t>(tx)] =
        std::conj(code.phase_codes[static_cast<size_t>(tx * num_chirps)]);
  }
  return weights;
}

int64_t ddm_num_doppler_per_tx(int64_t num_doppler, int64_t num_tx,
                               int64_t guard_bands) {
  if (num_tx <= 0) {
    throw std::invalid_argument(
        "dense MIMO stage: DDM num_tx must be positive");
  }
  if (num_doppler % num_tx != 0) {
    throw std::invalid_argument(
        "dense MIMO stage: DDM requires num_doppler (" +
        std::to_string(num_doppler) + ") divisible by num_tx (" +
        std::to_string(num_tx) + ")");
  }
  const int64_t band = num_doppler / num_tx;
  const int64_t width = band - guard_bands;
  if (width < 1) {
    throw std::invalid_argument(
        "dense MIMO stage: DDM guard_bands (" +
        std::to_string(guard_bands) + ") leaves no Doppler bins in a band of "
        "width " + std::to_string(band));
  }
  return width;
}

std::vector<int64_t> ddm_output_rd_shape(const DdmRdMapsLayout& in,
                                         int64_t num_tx,
                                         int64_t num_doppler_per_tx) {
  std::vector<int64_t> shape = in.leading;
  shape.push_back(num_tx);
  shape.push_back(in.num_rx);
  shape.push_back(num_doppler_per_tx);
  shape.push_back(in.num_range);
  return shape;
}

std::vector<int64_t> build_ddm_demux_table(const std::vector<double>& offsets,
                                           int64_t num_tx, int64_t num_doppler,
                                           int64_t num_doppler_per_tx) {
  const int64_t D = num_doppler;
  const int64_t W = num_doppler_per_tx;
  // fftshift centers: fftshift1D maps center-zero index i to fft-order bin
  // (i + c) % n; the inverse (fft-order j to center-zero index) is (j - c) % n.
  const int64_t c = (D + 1) / 2;
  const int64_t cW = (W + 1) / 2;
  std::vector<int64_t> table(static_cast<size_t>(num_tx * W));
  for (int64_t m = 0; m < num_tx; ++m) {
    const int64_t start =
        ((static_cast<int64_t>(std::llround(
              offsets[static_cast<size_t>(m)] * static_cast<double>(D))) %
          D) +
         D) %
        D;
    for (int64_t o = 0; o < W; ++o) {
      // output center-zero bin -> reduced fft-order bin
      const int64_t p = (o + cW) % W;
      // reduced fft-order bin -> absolute input fft-order bin (offset shift)
      const int64_t j = (start + p) % D;
      // input fft-order bin -> input center-zero index
      const int64_t in_cz = ((j - c) % D + D) % D;
      table[static_cast<size_t>(m * W + o)] = in_cz;
    }
  }
  return table;
}

std::string ddm_demux_cache_key(const DDMCodeDescriptor& code,
                                int64_t num_doppler) {
  std::ostringstream os;
  os << "ddm|" << code.num_tx << "|D=" << num_doppler << "|g="
     << code.guard_bands << "|off=";
  if (!code.doppler_offsets.empty()) {
    for (double o : code.doppler_offsets) {
      os << o << ",";
    }
  } else {
    os << "codes:" << code.phase_codes.size();
    for (const auto& z : code.phase_codes) {
      os << z.real() << ":" << z.imag() << ",";
    }
  }
  return os.str();
}

template <typename T>
void run_ddm_demux_stage(const void* input, void* output,
                         const DdmRdMapsLayout& layout, int64_t num_tx,
                         int64_t num_doppler_per_tx, const int64_t* d_table,
                         const void* d_initial_phase_weights,
                         cudaStream_t stream) {
  const int64_t num_rows =
      layout.batch * num_tx * layout.num_rx * num_doppler_per_tx;
  if (num_rows == 0 || layout.num_range == 0) {
    return;
  }
  constexpr int64_t kMaxBlocks = 65535;
  const int grid =
      static_cast<int>(num_rows < kMaxBlocks ? num_rows : kMaxBlocks);
  ddm_demux_gather_kernel<T><<<grid, kBlock, 0, stream>>>(
      static_cast<const Complex<T>*>(input),
      static_cast<Complex<T>*>(output), num_rows, layout.num_rx,
      layout.num_doppler, layout.num_range, num_tx, num_doppler_per_tx,
      d_table, static_cast<const Complex<T>*>(d_initial_phase_weights));
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("ddm demux stage: kernel launch failed");
  }
}

template void run_ddm_demux_stage<float>(const void* input, void* output,
                                         const DdmRdMapsLayout& layout,
                                         int64_t num_tx,
                                         int64_t num_doppler_per_tx,
                                         const int64_t* d_table,
                                         const void* d_initial_phase_weights,
                                         cudaStream_t stream);
template void run_ddm_demux_stage<double>(const void* input, void* output,
                                          const DdmRdMapsLayout& layout,
                                          int64_t num_tx,
                                          int64_t num_doppler_per_tx,
                                          const int64_t* d_table,
                                          const void* d_initial_phase_weights,
                                          cudaStream_t stream);

template <typename T>
void run_ddm_velocity_axis_stage(const void* input, void* output,
                                 int64_t num_doppler,
                                 int64_t num_doppler_out,
                                 cudaStream_t stream) {
  if (input == nullptr || output == nullptr || num_doppler <= 0 ||
      num_doppler_out <= 0 || num_doppler_out > num_doppler) {
    throw std::invalid_argument(
        "DDM velocity axis: invalid input/output extent");
  }
  const int64_t input_zero = num_doppler / 2;
  const int64_t output_zero = num_doppler_out / 2;
  const int64_t start = input_zero - output_zero;
  const auto* source = static_cast<const T*>(input) + start;
  const size_t bytes =
      static_cast<size_t>(num_doppler_out) * sizeof(T);
  if (cudaMemcpyAsync(
          output, source, bytes, cudaMemcpyDeviceToDevice, stream) !=
      cudaSuccess) {
    throw std::runtime_error("DDM velocity axis: D2D copy failed");
  }
}

template void run_ddm_velocity_axis_stage<float>(
    const void* input, void* output, int64_t num_doppler,
    int64_t num_doppler_out, cudaStream_t stream);
template void run_ddm_velocity_axis_stage<double>(
    const void* input, void* output, int64_t num_doppler,
    int64_t num_doppler_out, cudaStream_t stream);

// ---- Non-linear DDM correlate path ----------------------------------------

std::vector<int64_t> ddm_correlate_output_shape(const DdmRdMapsLayout& in,
                                                int64_t num_tx) {
  std::vector<int64_t> shape = in.leading;
  shape.push_back(num_tx);
  shape.push_back(in.num_rx);
  shape.push_back(in.num_doppler);
  shape.push_back(in.num_range);
  return shape;
}

std::vector<std::complex<double>> build_ddm_corr_weights(
    const DDMCodeDescriptor& code, int64_t num_doppler) {
  code.validate();
  const int64_t num_tx = code.num_tx;
  const int64_t num_chirps =
      static_cast<int64_t>(code.phase_codes.size()) / num_tx;
  const int64_t D = num_doppler;
  if (num_chirps != D) {
    throw std::invalid_argument(
        "build_ddm_corr_weights: num_chirps must equal num_doppler");
  }
  // Unnormalized DFT identity: DFT(DFT(c))[p] = D * c[(-p) mod D], so the
  // BACKWARD-norm weights conj(DFT(C))/D collapse to conj(c[(-p) mod D]).
  std::vector<std::complex<double>> weights(static_cast<size_t>(num_tx * D));
  for (int64_t m = 0; m < num_tx; ++m) {
    const auto* c_m = &code.phase_codes[static_cast<size_t>(m * D)];
    for (int64_t p = 0; p < D; ++p) {
      const int64_t idx = ((-p) % D + D) % D;
      weights[static_cast<size_t>(m * D + p)] = std::conj(c_m[idx]);
    }
  }
  return weights;
}

std::string ddm_correlate_cache_key(const DDMCodeDescriptor& code,
                                    int64_t num_doppler) {
  std::ostringstream os;
  os << "ddm_corr|" << code.num_tx << "|D=" << num_doppler << "|codes:";
  for (const auto& z : code.phase_codes) {
    os << z.real() << ":" << z.imag() << ",";
  }
  return os.str();
}

namespace {

// Pack center-zero RD tile (batch_rx, D, R) → contiguous (batch_rx, t, D)
// in fft-order for cuFFT correlation. Block (b, d_fft); threads walk range.
// MatX/numpy ifftshift with c = (D + 1) / 2:
//   out[..., d_fft] = in[..., (d_fft - c) mod D].
template <typename T>
__global__ void pack_rd_tile_kernel(const Complex<T>* in, Complex<T>* out,
                                    int64_t batch_rx, int64_t D, int64_t R,
                                    int64_t r0, int64_t t) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);
  const int64_t d_fft = static_cast<int64_t>(blockIdx.y);
  if (b >= batch_rx || d_fft >= D) {
    return;
  }
  const int64_t c = (D + 1) / 2;
  const int64_t d_cz = (d_fft - c + D) % D;
  const Complex<T>* in_row = in + (b * D + d_cz) * R + r0;
  for (int64_t ri = static_cast<int64_t>(threadIdx.x); ri < t;
       ri += static_cast<int64_t>(blockDim.x)) {
    out[(b * t + ri) * D + d_fft] = in_row[ri];
  }
}

// work[i, k] = y_hat[i, k] * W[k] * inv_D  (folds NumPy/MatX BACKWARD 1/D).
template <typename T>
__global__ void mul_corr_weights_kernel(const Complex<T>* y_hat,
                                        const Complex<T>* W, Complex<T>* work,
                                        int64_t nbatch, int64_t D, T inv_D) {
  const int64_t n = nbatch * D;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     static_cast<int64_t>(threadIdx.x);
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t k = idx % D;
    work[idx] = y_hat[idx] * W[k] * inv_D;
  }
}

// Scatter fft-order (Nrx, t, D) → center-zero (Nrx, D, R). Block (rx, d_fft).
// MatX fftshift1D maps fft-order j → center-zero (j - c) mod D with
// c = (D + 1) / 2 (same map as pack; for even D this equals (j + D/2) mod D).
template <typename T>
__global__ void scatter_corr_tile_kernel(const Complex<T>* corr,
                                         Complex<T>* out_m, int64_t Nrx,
                                         int64_t D, int64_t R, int64_t r0,
                                         int64_t t) {
  const int64_t rx = static_cast<int64_t>(blockIdx.x);
  const int64_t d_fft = static_cast<int64_t>(blockIdx.y);
  if (rx >= Nrx || d_fft >= D) {
    return;
  }
  const int64_t c = (D + 1) / 2;
  const int64_t d_cz = (d_fft - c + D) % D;
  Complex<T>* out_row = out_m + (rx * D + d_cz) * R + r0;
  const Complex<T>* c_ptr = corr + rx * t * D + d_fft;
  for (int64_t ri = static_cast<int64_t>(threadIdx.x); ri < t;
       ri += static_cast<int64_t>(blockDim.x)) {
    out_row[ri] = c_ptr[ri * D];
  }
}

struct CufftPlanCache {
  cufftHandle plan{0};
  int nbatch{0};
  int nfft{0};
  int is_float{-1};

  ~CufftPlanCache() {
    if (plan != 0) {
      cufftDestroy(plan);
      plan = 0;
    }
  }

  void ensure(int batch, int fft_len, bool use_float, cudaStream_t stream) {
    const int want = use_float ? 1 : 0;
    if (plan != 0 && nbatch == batch && nfft == fft_len && is_float == want) {
      if (cufftSetStream(plan, stream) != CUFFT_SUCCESS) {
        throw std::runtime_error("ddm correlate: cufftSetStream failed");
      }
      return;
    }
    if (plan != 0) {
      cufftDestroy(plan);
      plan = 0;
    }
    int n = fft_len;
    const cufftType typ = use_float ? CUFFT_C2C : CUFFT_Z2Z;
    if (cufftPlanMany(&plan, 1, &n, /*inembed=*/&n, /*istride=*/1,
                      /*idist=*/fft_len, /*onembed=*/&n, /*ostride=*/1,
                      /*odist=*/fft_len, typ, batch) != CUFFT_SUCCESS) {
      throw std::runtime_error("ddm correlate: cufftPlanMany failed");
    }
    if (cufftSetStream(plan, stream) != CUFFT_SUCCESS) {
      throw std::runtime_error("ddm correlate: cufftSetStream failed");
    }
    nbatch = batch;
    nfft = fft_len;
    is_float = want;
  }
};

template <typename T>
struct TileWorkspace {
  Complex<T>* yhat{nullptr};
  Complex<T>* work{nullptr};
  size_t bytes{0};

  ~TileWorkspace() {
    if (yhat != nullptr) {
      cudaFree(yhat);
      yhat = nullptr;
    }
    if (work != nullptr) {
      cudaFree(work);
      work = nullptr;
    }
    bytes = 0;
  }

  void ensure(size_t need, cudaStream_t stream) {
    if (bytes >= need) {
      return;
    }
    if (yhat != nullptr) {
      cudaFreeAsync(yhat, stream);
      yhat = nullptr;
    }
    if (work != nullptr) {
      cudaFreeAsync(work, stream);
      work = nullptr;
    }
    if (cudaMallocAsync(reinterpret_cast<void**>(&yhat), need, stream) !=
            cudaSuccess ||
        cudaMallocAsync(reinterpret_cast<void**>(&work), need, stream) !=
            cudaSuccess) {
      if (yhat != nullptr) {
        cudaFreeAsync(yhat, stream);
        yhat = nullptr;
      }
      bytes = 0;
      throw std::runtime_error("ddm correlate: tile workspace alloc failed");
    }
    bytes = need;
  }
};

// Per-operator-instance scratch (keyed by DDMDenseDemultiplexOp*).
// Not shared across instances so interleaved compute() cannot destroy another
// op's in-flight cuFFT plan.
struct CorrelateScratch {
  CufftPlanCache plan;
  TileWorkspace<float> ws_f;
  TileWorkspace<double> ws_d;
};

std::unordered_map<void*, CorrelateScratch>& correlate_scratch_map() {
  static thread_local std::unordered_map<void*, CorrelateScratch> map;
  return map;
}

CorrelateScratch& scratch_for(void* owner) {
  if (owner == nullptr) {
    throw std::invalid_argument("ddm correlate: owner must be non-null");
  }
  return correlate_scratch_map()[owner];
}

template <typename T>
void cufft_c2c(cufftHandle plan, Complex<T>* data, int direction) {
  if constexpr (sizeof(T) == 4) {
    if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(data),
                     reinterpret_cast<cufftComplex*>(data),
                     direction) != CUFFT_SUCCESS) {
      throw std::runtime_error("ddm correlate: cufftExecC2C failed");
    }
  } else {
    if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(data),
                     reinterpret_cast<cufftDoubleComplex*>(data),
                     direction) != CUFFT_SUCCESS) {
      throw std::runtime_error("ddm correlate: cufftExecZ2Z failed");
    }
  }
}

}  // namespace

void release_ddm_correlate_scratch(void* owner) {
  if (owner == nullptr) {
    return;
  }
  correlate_scratch_map().erase(owner);
}

template <typename T>
void run_ddm_correlate_stage(const void* input, void* output,
                             const DdmRdMapsLayout& layout, int64_t num_tx,
                             const void* d_corr_weights, void* owner,
                             cudaStream_t stream) {
  const int64_t B = layout.batch;
  const int64_t Nrx = layout.num_rx;
  const int64_t D = layout.num_doppler;
  const int64_t R = layout.num_range;
  if (B == 0 || Nrx == 0 || D == 0 || R == 0 || num_tx == 0) {
    return;
  }

  // FFT circular correlation with range tiling + cuFFT. Input/output Doppler
  // axes are center-zero (FMCWDopplerProcessorOp default); pack/scatter apply
  // ifftshift/fftshift so the cuFFT correlate runs in fft-order — matching
  // the linear DDM gather path's center-zero contract.
  //   out = ifft(fft(Y) * W) / D   (cuFFT unnormalized both ways; /D matches
  //   MatX/NumPy BACKWARD). Pack/scatter avoid MatX permute temps; persistent
  //   ~32 MiB tiles keep Ŷ L2-hot across the per-TX IFFTs.
  using ComplexT = Complex<T>;
  const int64_t batch_rx = B * Nrx;
  const T inv_D = static_cast<T>(1.0) / static_cast<T>(D);

  constexpr int64_t kTargetTileBytes = 32LL * 1024 * 1024;
  const int64_t bytes_per_range =
      batch_rx * D * static_cast<int64_t>(sizeof(ComplexT));
  int64_t tile_r =
      bytes_per_range > 0 ? (kTargetTileBytes / bytes_per_range) : R;
  tile_r = std::max<int64_t>(1, std::min(R, tile_r));

  const auto* in = static_cast<const ComplexT*>(input);
  const auto* weights = static_cast<const ComplexT*>(d_corr_weights);
  auto* out_base = static_cast<ComplexT*>(output);

  const size_t tile_bytes =
      static_cast<size_t>(batch_rx * tile_r * D) * sizeof(ComplexT);
  CorrelateScratch& scratch = scratch_for(owner);
  ComplexT* d_yhat = nullptr;
  ComplexT* d_work = nullptr;
  if constexpr (sizeof(T) == 4) {
    scratch.ws_f.ensure(tile_bytes, stream);
    d_yhat = scratch.ws_f.yhat;
    d_work = scratch.ws_f.work;
  } else {
    scratch.ws_d.ensure(tile_bytes, stream);
    d_yhat = scratch.ws_d.yhat;
    d_work = scratch.ws_d.work;
  }

  constexpr int kBlock = 256;
  for (int64_t r0 = 0; r0 < R; r0 += tile_r) {
    const int64_t t = std::min(tile_r, R - r0);
    const int nbatch = static_cast<int>(batch_rx * t);
    scratch.plan.ensure(nbatch, static_cast<int>(D), sizeof(T) == 4, stream);

    {
      dim3 grid(static_cast<unsigned>(batch_rx), static_cast<unsigned>(D));
      const int threads = static_cast<int>(std::min<int64_t>(kBlock, t));
      pack_rd_tile_kernel<T><<<grid, threads, 0, stream>>>(
          in, d_yhat, batch_rx, D, R, r0, t);
    }
    cufft_c2c<T>(scratch.plan.plan, d_yhat, CUFFT_FORWARD);

    for (int64_t m = 0; m < num_tx; ++m) {
      const int mul_blocks = static_cast<int>(
          std::min<int64_t>(65535, (static_cast<int64_t>(nbatch) * D +
                                    kBlock - 1) /
                                       kBlock));
      mul_corr_weights_kernel<T><<<mul_blocks, kBlock, 0, stream>>>(
          d_yhat, weights + m * D, d_work, nbatch, D, inv_D);
      cufft_c2c<T>(scratch.plan.plan, d_work, CUFFT_INVERSE);

      for (int64_t b = 0; b < B; ++b) {
        ComplexT* out_bm = out_base + ((b * num_tx + m) * Nrx * D * R);
        const ComplexT* corr_b = d_work + (b * Nrx * t * D);
        dim3 grid(static_cast<unsigned>(Nrx), static_cast<unsigned>(D));
        const int threads = static_cast<int>(std::min<int64_t>(kBlock, t));
        scatter_corr_tile_kernel<T><<<grid, threads, 0, stream>>>(
            corr_b, out_bm, Nrx, D, R, r0, t);
      }
    }
  }
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("ddm correlate: CUDA kernel failed");
  }
}

template void run_ddm_correlate_stage<float>(const void* input, void* output,
                                             const DdmRdMapsLayout& layout,
                                             int64_t num_tx,
                                             const void* d_corr_weights,
                                             void* owner, cudaStream_t stream);
template void run_ddm_correlate_stage<double>(const void* input, void* output,
                                              const DdmRdMapsLayout& layout,
                                              int64_t num_tx,
                                              const void* d_corr_weights,
                                              void* owner,
                                              cudaStream_t stream);

}  // namespace holoscan::radar::detail
