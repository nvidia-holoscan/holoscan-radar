// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_stage.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::radar::detail {

namespace {

constexpr int64_t kBlock = 256;
constexpr double kPi = 3.141592653589793;

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
                                        const int64_t* table) {
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
      output[out_base + r] = input[in_base + r];
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

std::vector<std::complex<double>> build_tdm_correction_table(
    const FMCWWaveformDescriptor& waveform, int64_t num_tx,
    int64_t num_doppler) {
  if (!waveform.is_valid()) {
    throw std::invalid_argument(
        "MIMOCoherentCompensationOp: waveform must be valid");
  }
  if (num_tx <= 0 || num_doppler <= 0) {
    throw std::invalid_argument(
        "MIMOCoherentCompensationOp: num_tx and num_doppler must be positive");
  }
  const double pri = waveform.chirp_repetition_interval();
  const double prf = 1.0 / pri;
  const double t_inter_chirp = pri / static_cast<double>(num_tx);

  std::vector<std::complex<double>> table(
      static_cast<size_t>(num_tx * num_doppler));
  for (int64_t d = 0; d < num_doppler; ++d) {
    const double f_d = doppler_frequency_hz(d, num_doppler, prf);
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      const double phase =
          -2.0 * kPi * f_d * static_cast<double>(tx) * t_inter_chirp;
      table[static_cast<size_t>(tx * num_doppler + d)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }
  return table;
}

std::string tdm_correction_cache_key(const FMCWWaveformDescriptor& waveform,
                                     int64_t num_tx, int64_t num_doppler) {
  std::ostringstream os;
  os << num_tx << "|" << num_doppler << "|" << waveform.carrier_frequency()
     << "|" << waveform.chirp_repetition_interval();
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

std::vector<double> ddm_resolve_offsets(const DDMCodeDescriptor& code) {
  code.validate();
  const int64_t num_tx = code.num_tx;
  std::vector<double> offsets(static_cast<size_t>(num_tx), 0.0);

  if (!code.doppler_offsets.empty()) {
    // Compact form: offsets are already fractions of PRF (cycles/chirp).
    for (int64_t m = 0; m < num_tx; ++m) {
      double o = code.doppler_offsets[static_cast<size_t>(m)];
      o -= std::floor(o);  // fold into [0, 1)
      offsets[static_cast<size_t>(m)] = o;
    }
    return offsets;
  }

  if (!code.phase_codes.empty()) {
    // Linear-phase form: recover the constant per-chirp phase increment for
    // each TX and express it as a fractional Doppler offset. Reject arbitrary /
    // non-linear codes (a genuinely different, deferred demux; see OPEN_ITEMS).
    const int64_t num_chirps =
        static_cast<int64_t>(code.phase_codes.size()) / num_tx;
    if (num_chirps < 2) {
      throw std::invalid_argument(
          "DDMCodeDescriptor: phase_codes need >= 2 chirps per TX to infer a "
          "linear-phase Doppler offset");
    }
    constexpr double kTwoPi = 2.0 * kPi;
    for (int64_t m = 0; m < num_tx; ++m) {
      const auto* code_m =
          &code.phase_codes[static_cast<size_t>(m * num_chirps)];
      // Reference increment from the first chirp pair.
      const std::complex<double> ref_step =
          code_m[1] * std::conj(code_m[0]);
      const double delta = std::arg(ref_step);
      for (int64_t k = 0; k + 1 < num_chirps; ++k) {
        const std::complex<double> step = code_m[k + 1] * std::conj(code_m[k]);
        double d = std::arg(step) - delta;
        d -= kTwoPi * std::round(d / kTwoPi);  // wrap to (-pi, pi]
        if (std::abs(d) > 1e-3) {
          throw std::invalid_argument(
              "MIMOCoherentCompensationOp: DDM phase_codes are not linear-"
              "phase; arbitrary / non-linear code demux is not implemented "
              "(supply doppler_offsets or a linear-phase code). See "
              "OPEN_ITEMS.");
        }
      }
      double o = delta / kTwoPi;
      o -= std::floor(o);  // fold into [0, 1)
      offsets[static_cast<size_t>(m)] = o;
    }
    return offsets;
  }

  throw std::invalid_argument(
      "MIMOCoherentCompensationOp: DDM requires ddm_code with doppler_offsets "
      "or phase_codes");
}

int64_t ddm_num_doppler_per_tx(int64_t num_doppler, int64_t num_tx,
                               int64_t guard_bands) {
  if (num_tx <= 0) {
    throw std::invalid_argument(
        "MIMOCoherentCompensationOp: DDM num_tx must be positive");
  }
  if (num_doppler % num_tx != 0) {
    throw std::invalid_argument(
        "MIMOCoherentCompensationOp: DDM requires num_doppler (" +
        std::to_string(num_doppler) + ") divisible by num_tx (" +
        std::to_string(num_tx) + "); see OPEN_ITEMS imposed layout");
  }
  const int64_t band = num_doppler / num_tx;
  const int64_t width = band - guard_bands;
  if (width < 1) {
    throw std::invalid_argument(
        "MIMOCoherentCompensationOp: DDM guard_bands (" +
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
      d_table);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("ddm demux stage: kernel launch failed");
  }
}

template void run_ddm_demux_stage<float>(const void* input, void* output,
                                         const DdmRdMapsLayout& layout,
                                         int64_t num_tx,
                                         int64_t num_doppler_per_tx,
                                         const int64_t* d_table,
                                         cudaStream_t stream);
template void run_ddm_demux_stage<double>(const void* input, void* output,
                                          const DdmRdMapsLayout& layout,
                                          int64_t num_tx,
                                          int64_t num_doppler_per_tx,
                                          const int64_t* d_table,
                                          cudaStream_t stream);

}  // namespace holoscan::radar::detail
