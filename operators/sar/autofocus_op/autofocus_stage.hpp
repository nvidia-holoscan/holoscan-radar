// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Phase-gradient autofocus stage behind AutofocusOp. CUDA-TU only
// (includes <matx.h> via matx_utils.hpp).
//
// Algorithm (OPERATOR_API.md → AutofocusOp), on a (batch, n_range,
// n_az) working cube whose last axis is the declared azimuth:
//   1. Brightest scatterer per range bin; keep the brightest
//      min_scatterers whose peak power is 10× the bin mean. Extra
//      rows that pass the gate are clutter on a long azimuth axis
//      and would average the LUE to zero.
//   2. Circularly shift each selected scatterer to azimuth index 0
//      (so the LUE sees a small phase step, not a π checkerboard).
//   3. Rectangular window of the current width about index 0.
//   4. FFT the windowed n_az line -- zeros outside the window
//      included -- and LUE + integrate on all n_az bins, with the
//      occupied-energy skip. The window narrows the *image* support;
//      it must not narrow the transform. Packing the window to
//      `width` and FFTing that instead decimates the phase history
//      by n_az/width, so each LUE difference spans that many bins
//      and the estimator's small-angle form saturates -- which is
//      the narrow-window edge loss that used to be fault 1.
//   5. Integrate, drop linear + constant (unobservable), apply the
//      conjugate to the full (unwindowed) image. Keep the candidate
//      only when image sharpness does not fall — later geometric
//      window shrinks otherwise overfit and defocus. First width is
//      window_fraction × n_az; subsequent widths halve.
//   6. Accumulate accepted per-iteration corrections. The in-image
//      apply uses that accumulator in the azimuth-FFT domain. The
//      emitted phase_correction is its negation: a backprojected
//      lattice (WORKFLOWS Figure 2) has an azimuth FFT conjugated
//      versus pulse history, and the port is pulse-indexed so
//      PhaseCorrectionOp sense=subtract undoes the residual.
//
// A failed estimate (too few scatterers, or peak-to-peak above
// max_correction) copies the input image through, writes a zero
// phase, and fills sharpness from the uncorrected image.

#pragma once

#include <cuda/std/complex>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// Peak-to-mean power ratio that counts as "above local background".
inline constexpr double kPgaMinPeakToMean = 10.0;
// Fraction of azimuth-FFT energy that defines the occupied
// low-frequency support. If that support is already most of n_az,
// use every bin so a 1-sample-per-pulse helper is unchanged.
inline constexpr double kPgaOccupiedEnergy = 0.99;
inline constexpr double kPgaAllBinsFrac = 0.85;

struct AutofocusConfig {
  AutofocusAzimuthAxis azimuth_axis{AutofocusAzimuthAxis::kUnspecified};
  int64_t num_iterations{8};
  double window_fraction{0.5};
  int64_t min_scatterers{16};
  double max_correction{std::numeric_limits<double>::infinity()};
};

// What the run did, for the `termination` and `iterations` ports. Host-side
// because every branch that decides it is already a host-side decision taken
// after a device-to-host sync, so there is nothing to bring back.
//
// `iterations_accepted` is the number whose phase is in `phase_correction`,
// which is not the number attempted: the loop applies an iteration, scores it,
// and discards it if sharpness fell, so the last attempt is routinely not in
// the estimate.
struct AutofocusOutcome {
  AutofocusTermination termination{AutofocusTermination::kConverged};
  int64_t iterations_accepted{0};
};

template <typename T>
struct AutofocusState {
  std::string key;
  cuda::std::complex<T>* work{nullptr};
  cuda::std::complex<T>* spec{nullptr};
  cuda::std::complex<T>* scratch{nullptr};
  int32_t* peaks{nullptr};
  uint8_t* strong{nullptr};
  T* peak_pwr{nullptr};
  T* grad{nullptr};
  T* bin_pwr{nullptr};
  T* phase_iter{nullptr};
  T* phase_acc{nullptr};
  int32_t* counts{nullptr};
  T* p2p{nullptr};
  T* sharp_tmp{nullptr};

  AutofocusState() = default;
  AutofocusState(const AutofocusState&) = delete;
  AutofocusState& operator=(const AutofocusState&) = delete;
  ~AutofocusState() { clear(); }

  void clear() {
    cudaFree(work);
    cudaFree(spec);
    cudaFree(scratch);
    cudaFree(peaks);
    cudaFree(strong);
    cudaFree(peak_pwr);
    cudaFree(grad);
    cudaFree(bin_pwr);
    cudaFree(phase_iter);
    cudaFree(phase_acc);
    cudaFree(counts);
    cudaFree(p2p);
    cudaFree(sharp_tmp);
    work = nullptr;
    spec = nullptr;
    scratch = nullptr;
    peaks = nullptr;
    strong = nullptr;
    peak_pwr = nullptr;
    grad = nullptr;
    bin_pwr = nullptr;
    phase_iter = nullptr;
    phase_acc = nullptr;
    counts = nullptr;
    p2p = nullptr;
    sharp_tmp = nullptr;
    key.clear();
  }
};

template <typename U>
U* autofocus_malloc(size_t n, const char* what) {
  U* ptr = nullptr;
  if (n == 0) {
    return nullptr;
  }
  if (cudaMalloc(&ptr, n * sizeof(U)) != cudaSuccess) {
    throw std::runtime_error(std::string("AutofocusOp workspace: ") + what + " cudaMalloc failed");
  }
  return ptr;
}

inline int64_t pga_window_width(int64_t n_az, double fraction, int iter) {
  double width = fraction * static_cast<double>(n_az);
  for (int i = 0; i < iter; ++i) {
    width *= 0.5;
  }
  int64_t w = static_cast<int64_t>(std::llround(width));
  if (w < 2) {
    w = 2;
  }
  if (w > n_az) {
    w = n_az;
  }
  return w;
}

template <typename T>
void ensure_autofocus_workspace(AutofocusState<T>& state, int64_t batch, int64_t n_range,
                                int64_t n_az) {
  const std::string key =
      "b=" + std::to_string(batch) + ";r=" + std::to_string(n_range) + ";a=" + std::to_string(n_az);
  if (state.key == key && state.work != nullptr) {
    return;
  }
  state.clear();
  const int64_t n_bins = batch * n_range;
  const int64_t n_pix = n_bins * n_az;
  state.work = autofocus_malloc<cuda::std::complex<T>>(static_cast<size_t>(n_pix), "work");
  state.spec = autofocus_malloc<cuda::std::complex<T>>(static_cast<size_t>(n_pix), "spec");
  state.scratch = autofocus_malloc<cuda::std::complex<T>>(static_cast<size_t>(n_pix), "scratch");
  state.peaks = autofocus_malloc<int32_t>(static_cast<size_t>(n_bins), "peaks");
  state.strong = autofocus_malloc<uint8_t>(static_cast<size_t>(n_bins), "strong");
  state.peak_pwr = autofocus_malloc<T>(static_cast<size_t>(n_bins), "peak_pwr");
  state.grad = autofocus_malloc<T>(static_cast<size_t>(batch * n_az), "grad");
  state.bin_pwr = autofocus_malloc<T>(static_cast<size_t>(batch * n_az), "bin_pwr");
  state.phase_iter = autofocus_malloc<T>(static_cast<size_t>(batch * n_az), "phase_iter");
  state.phase_acc = autofocus_malloc<T>(static_cast<size_t>(batch * n_az), "phase_acc");
  state.counts = autofocus_malloc<int32_t>(static_cast<size_t>(batch), "counts");
  state.p2p = autofocus_malloc<T>(static_cast<size_t>(batch), "p2p");
  state.sharp_tmp = autofocus_malloc<T>(static_cast<size_t>(batch), "sharp_tmp");
  state.key = key;
}

template <typename T>
__device__ void sincos_t(T x, T* s, T* c) {
  if constexpr (std::is_same_v<T, float>) {
    sincosf(x, s, c);
  } else {
    sincos(x, s, c);
  }
}

template <typename T>
__global__ void peak_find_kernel(const cuda::std::complex<T>* in, int32_t* peaks, uint8_t* strong,
                                 T* peak_pwr, int32_t* counts, int64_t n_bins, int64_t n_range,
                                 int64_t n_az, T min_ratio) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_bins) {
    return;
  }
  const int64_t b = idx / n_range;
  const int64_t base = idx * n_az;
  T best = static_cast<T>(-1);
  T sum = static_cast<T>(0);
  int32_t arg = 0;
  for (int64_t a = 0; a < n_az; ++a) {
    const auto z = in[base + a];
    const T p = z.real() * z.real() + z.imag() * z.imag();
    sum += p;
    if (p > best) {
      best = p;
      arg = static_cast<int32_t>(a);
    }
  }
  const T mean = sum / static_cast<T>(n_az);
  peaks[idx] = arg;
  peak_pwr[idx] = best;
  const uint8_t ok = (best > min_ratio * mean) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
  strong[idx] = ok;
  if (ok != 0) {
    atomicAdd(counts + b, 1);
  }
}

// Keep the brightest min_scatterers range bins. A 10x peak-to-mean
// gate on a long azimuth axis marks almost every row (Atlanta aux:
// 512/512), and averaging those in the LUE cancels the estimate.
template <typename T>
void cap_strong_scatterers(uint8_t* strong, const T* peak_pwr, int32_t* host_counts, int64_t batch,
                           int64_t n_range, int64_t keep, cudaStream_t stream,
                           const std::string& who, int iter) {
  if (keep < 1) {
    return;
  }
  bool any = false;
  for (int64_t b = 0; b < batch; ++b) {
    if (host_counts[static_cast<size_t>(b)] > keep) {
      any = true;
    }
  }
  if (!any) {
    return;
  }
  const int64_t n_bins = batch * n_range;
  std::vector<T> pwr(static_cast<size_t>(n_bins));
  std::vector<uint8_t> mask(static_cast<size_t>(n_bins));
  if (cudaMemcpyAsync(pwr.data(), peak_pwr, static_cast<size_t>(n_bins) * sizeof(T),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": peak_pwr D2H failed");
  }
  if (cudaMemcpyAsync(mask.data(), strong, static_cast<size_t>(n_bins) * sizeof(uint8_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": strong D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error(who + ": strong sync failed");
  }
  int32_t saw = host_counts[0];
  for (int64_t b = 0; b < batch; ++b) {
    const int32_t n = host_counts[static_cast<size_t>(b)];
    saw = std::max(saw, n);
    if (static_cast<int64_t>(n) <= keep) {
      continue;
    }
    std::vector<int64_t> idx;
    idx.reserve(static_cast<size_t>(n));
    const int64_t base = b * n_range;
    for (int64_t r = 0; r < n_range; ++r) {
      if (mask[static_cast<size_t>(base + r)] != 0) {
        idx.push_back(r);
      }
    }
    std::nth_element(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(keep), idx.end(),
                     [&](int64_t i, int64_t j) {
                       return pwr[static_cast<size_t>(base + i)] >
                              pwr[static_cast<size_t>(base + j)];
                     });
    for (size_t k = static_cast<size_t>(keep); k < idx.size(); ++k) {
      mask[static_cast<size_t>(base + idx[k])] = 0;
    }
    host_counts[static_cast<size_t>(b)] = static_cast<int32_t>(keep);
  }
  if (cudaMemcpyAsync(strong, mask.data(), static_cast<size_t>(n_bins) * sizeof(uint8_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": strong H2D failed");
  }
  if (iter == 0) {
    HOLOSCAN_LOG_INFO("{}: using {} of {} strong range bins", who, keep, saw);
  }
}

template <typename T>
__global__ void circshift_window_kernel(const cuda::std::complex<T>* in, cuda::std::complex<T>* out,
                                        const int32_t* peaks, const uint8_t* strong, int64_t n_bins,
                                        int64_t n_az, int64_t width) {
  const int64_t n = n_bins * n_az;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t bin = i / n_az;
  const int64_t a = i - bin * n_az;
  if (strong[bin] == 0) {
    out[i] = cuda::std::complex<T>(static_cast<T>(0), static_cast<T>(0));
    return;
  }
  // Peak → index 0; keep a rectangular window about 0 (wraps).
  const int64_t half = width / 2;
  const int64_t hi = n_az - (width - half);
  if (a >= half && a < hi) {
    out[i] = cuda::std::complex<T>(static_cast<T>(0), static_cast<T>(0));
    return;
  }
  int64_t src = a + static_cast<int64_t>(peaks[bin]);
  src %= n_az;
  if (src < 0) {
    src += n_az;
  }
  out[i] = in[bin * n_az + src];
}

// The pair at k = n_fft-1 wraps to bin 0. Those two bins are adjacent *pulses*
// -- the azimuth-FFT axis is the pulse axis rotated by half the aperture, so
// bin n-1 and bin 0 straddle the dwell centre -- and zeroing that gradient
// instead of estimating it left the integration below without the one link it
// genuinely needed. See integrate_detrend_kernel.
template <typename T>
__global__ void lue_kernel(const cuda::std::complex<T>* spec, T* grad, int64_t batch,
                           int64_t n_range, int64_t n_fft) {
  const int64_t n = batch * n_fft;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t b = i / n_fft;
  const int64_t k = i - b * n_fft;
  const int64_t k1 = (k + 1) % n_fft;
  T num = static_cast<T>(0);
  T den = static_cast<T>(0);
  for (int64_t r = 0; r < n_range; ++r) {
    const int64_t row = (b * n_range + r) * n_fft;
    const auto g0 = spec[row + k];
    const auto g1 = spec[row + k1];
    num += g1.imag() * g0.real() - g1.real() * g0.imag();
    den += g0.real() * g0.real() + g0.imag() * g0.imag();
  }
  const T eps = static_cast<T>(1e-20);
  grad[b * n_fft + k] = (den > eps) ? (num / den) : static_cast<T>(0);
}

template <typename T>
__global__ void bin_power_kernel(const cuda::std::complex<T>* spec, T* pwr, int64_t batch,
                                 int64_t n_range, int64_t n_az) {
  const int64_t n = batch * n_az;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t b = i / n_az;
  const int64_t k = i - b * n_az;
  T acc = static_cast<T>(0);
  for (int64_t r = 0; r < n_range; ++r) {
    const auto z = spec[(b * n_range + r) * n_az + k];
    acc += z.real() * z.real() + z.imag() * z.imag();
  }
  pwr[i] = acc;
}

template <typename T>
__device__ T lue_pair(const cuda::std::complex<T>* spec, int64_t b, int64_t n_range, int64_t n_az,
                      int64_t k0, int64_t k1) {
  T num = static_cast<T>(0);
  T den = static_cast<T>(0);
  for (int64_t r = 0; r < n_range; ++r) {
    const int64_t row = (b * n_range + r) * n_az;
    const auto g0 = spec[row + k0];
    const auto g1 = spec[row + k1];
    num += g1.imag() * g0.real() - g1.real() * g0.imag();
    den += g0.real() * g0.real() + g0.imag() * g0.imag();
  }
  const T eps = static_cast<T>(1e-20);
  return (den > eps) ? (num / den) : static_cast<T>(0);
}

template <typename T>
__device__ int64_t pga_occupied_nkeep(const T* pwr, int64_t n_az) {
  T tot = static_cast<T>(0);
  for (int64_t k = 0; k < n_az; ++k) {
    tot += pwr[k];
  }
  if (tot <= static_cast<T>(0)) {
    return n_az;
  }
  const T target = static_cast<T>(kPgaOccupiedEnergy) * tot;
  int64_t n_keep = 0;
  T acc = static_cast<T>(0);
  while (n_keep < n_az && acc < target) {
    int64_t k = 0;
    if (n_keep % 2 == 0) {
      k = n_keep / 2;
    } else {
      k = n_az - (n_keep + 1) / 2;
    }
    acc += pwr[k];
    ++n_keep;
  }
  if (n_keep < 2) {
    n_keep = 2;
  }
  if (n_keep < n_az && (n_keep % 2) == 1) {
    ++n_keep;
  }
  if (n_keep > n_az) {
    n_keep = n_az;
  }
  if (static_cast<double>(n_keep) >= kPgaAllBinsFrac * static_cast<double>(n_az)) {
    return n_az;
  }
  return n_keep;
}

__device__ inline int64_t occupied_chain_index(int64_t i, int64_t n_keep, int64_t n_az) {
  const int64_t half = n_keep / 2;
  if (i < half) {
    return i;
  }
  const int64_t tail = n_keep - half;
  return n_az - tail + (i - half);
}

template <typename T>
__global__ void integrate_detrend_kernel(const cuda::std::complex<T>* spec, const T* pwr,
                                         const T* grad, T* phase, int64_t n_range, int64_t n_az) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);
  T* phi = phase + b * n_az;
  const T* g = grad + b * n_az;
  const T* bp = pwr + b * n_az;
  if (threadIdx.x != 0) {
    return;
  }
  const int64_t keep = min(pga_occupied_nkeep(bp, n_az), n_az);
  for (int64_t k = 0; k < n_az; ++k) {
    phi[k] = static_cast<T>(0);
  }
  // Integrate outward from DC, never along the bin axis. The azimuth-FFT bin
  // axis is the pulse axis rotated by half the aperture: bins 0..n/2-1 run
  // from the dwell centre to its end, bins n/2..n-1 from its start back to
  // the centre. So bins n/2-1 and n/2 are *not* adjacent pulses and the
  // gradient between them carries no information, while bins n-1 and 0 are
  // adjacent and were being skipped entirely.
  //
  // Integrating straight up the bin axis therefore took the second half of the
  // aperture's constant of integration across that wrap, and emitted the two
  // halves offset from each other by a step at the dwell centre. The step is
  // small when the error is even about the centre -- a quadratic, a cosine --
  // and is the whole error when it is odd -- a cubic, a sine. Every sweep
  // before 2026-10-29 injected a quadratic, which is why this looked healthy:
  // the same sinusoid shifted a quarter cycle went from 0.160 rad residual to
  // 5.079 rad, with a 17.34 rad step at pulse n/2 against a 0.008 rad median
  // step elsewhere.
  const int64_t n_pos = (keep + 1) / 2;
  const int64_t n_neg = keep - n_pos;
  for (int64_t k = 0; k + 1 < n_pos; ++k) {
    phi[k + 1] = phi[k] + g[k];
  }
  for (int64_t j = 0; j < n_neg; ++j) {
    const int64_t k = n_az - 1 - j;
    phi[k] = phi[(k + 1) % n_az] - g[k];
  }
  // Detrend against signed frequency rather than position along the chain,
  // for the same reason: a line in bin index is not a line in pulse index, and
  // removing one would reintroduce a step at the centre. Signed frequency runs
  // -n_neg..n_pos-1 contiguously and is the pulse offset from the dwell
  // centre, so a line in it is the image shift PGA is blind to.
  T sum_p = static_cast<T>(0);
  T sum_x = static_cast<T>(0);
  T sum_xp = static_cast<T>(0);
  T sum_xx = static_cast<T>(0);
  for (int64_t i = 0; i < keep; ++i) {
    const int64_t k = (i < n_pos) ? i : (n_az - 1 - (i - n_pos));
    const T x = static_cast<T>((i < n_pos) ? i : (k - n_az));
    const T p = phi[k];
    sum_p += p;
    sum_x += x;
    sum_xp += x * p;
    sum_xx += x * x;
  }
  const T n = static_cast<T>(keep);
  const T denom = sum_xx - sum_x * sum_x / n;
  T slope = static_cast<T>(0);
  if (denom != static_cast<T>(0)) {
    slope = (sum_xp - sum_x * sum_p / n) / denom;
  }
  const T intercept = (sum_p - slope * sum_x) / n;
  for (int64_t i = 0; i < keep; ++i) {
    const int64_t k = (i < n_pos) ? i : (n_az - 1 - (i - n_pos));
    const T x = static_cast<T>((i < n_pos) ? i : (k - n_az));
    phi[k] -= intercept + slope * x;
  }
}

template <typename T>
__global__ void add_phase_kernel(T* acc, const T* iter, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    acc[i] += iter[i];
  }
}

// `phase_acc` is indexed by azimuth-FFT bin with fftfreq semantics -- bin 0 is
// DC, which is the centre of the dwell. The contract's `phase_correction` is
// indexed by pulse, which runs monotonically across the dwell, so the two
// differ by an fftshift *and by the handedness of the axis*: the image's
// azimuth axis is the Fourier conjugate of slow time, which runs the opposite
// way, so pulse p reads bin (n - 1 - p + n/2) mod n.
//
// The reversal was missing until 2026-10-30 and is invisible on an error that
// is even about the dwell centre, because reversing an even function is a
// no-op -- and reversing an odd one is indistinguishable from negating it, so
// `AutofocusScoreOp`'s sign search absorbed those too. Only an error that is
// neither exposed it: a sinusoid at 45 degrees scored 4.92 rad as emitted
// against a 4.75 rad baseline (no recovery at all, reported with a normal
// sign) and 0.178 rad reversed. Measured against injected truth on Kourou over
// a quadratic, a cubic, a sine and that 45-degree sinusoid, this mapping wins
// on every one; see autofocus_op/OPEN_ITEMS.md.
//
// Only the exported vector was affected. The in-image correction multiplies
// bin k by phase_acc[k] in the spectrum domain and never consults this
// mapping, so WORKFLOWS Figure 3 was always right while Figure 2 handed
// `PhaseCorrectionOp` a time-reversed correction.
template <typename T>
__global__ void fftshift_phase_kernel(const T* in, T* out, int64_t batch, int64_t n_az) {
  const int64_t n = batch * n_az;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t b = i / n_az;
  const int64_t p = i - b * n_az;
  out[i] = in[b * n_az + (n_az - 1 - p + (n_az + 1) / 2) % n_az];
}

template <typename T>
__global__ void mul_phasor_kernel(cuda::std::complex<T>* spec, const T* phase, int64_t batch,
                                  int64_t n_range, int64_t n_az, T sign) {
  const int64_t n = batch * n_range * n_az;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t a = i % n_az;
  const int64_t tmp = i / n_az;
  const int64_t b = tmp / n_range;
  T s = 0;
  T c = 0;
  sincos_t(sign * phase[b * n_az + a], &s, &c);
  const auto z = spec[i];
  spec[i] = cuda::std::complex<T>(z.real() * c - z.imag() * s, z.real() * s + z.imag() * c);
}

template <typename T>
__global__ void sharpness_kernel(const cuda::std::complex<T>* in, T* out, int64_t batch,
                                 int64_t pixels) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);
  const cuda::std::complex<T>* row = in + b * pixels;
  T acc = static_cast<T>(0);
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < pixels;
       i += static_cast<int64_t>(blockDim.x)) {
    const auto z = row[i];
    const T p = z.real() * z.real() + z.imag() * z.imag();
    acc += p * p;
  }
  __shared__ T smem[256];
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[b] = smem[0];
  }
}

template <typename T>
__global__ void p2p_kernel(const T* phase, T* out, int64_t n_az) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);
  const T* row = phase + b * n_az;
  T lo = row[0];
  T hi = row[0];
  for (int64_t k = static_cast<int64_t>(threadIdx.x); k < n_az;
       k += static_cast<int64_t>(blockDim.x)) {
    const T v = row[k];
    lo = (v < lo) ? v : lo;
    hi = (v > hi) ? v : hi;
  }
  __shared__ T slo[256];
  __shared__ T shi[256];
  slo[threadIdx.x] = lo;
  shi[threadIdx.x] = hi;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) {
      slo[threadIdx.x] = (slo[threadIdx.x + offset] < slo[threadIdx.x]) ? slo[threadIdx.x + offset]
                                                                        : slo[threadIdx.x];
      shi[threadIdx.x] = (shi[threadIdx.x + offset] > shi[threadIdx.x]) ? shi[threadIdx.x + offset]
                                                                        : shi[threadIdx.x];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[b] = shi[0] - slo[0];
  }
}

template <typename T>
__global__ void fill_sharpness_kernel(T* dest, const T* src, int64_t batch, int64_t n_slots,
                                      int64_t slot) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b < batch) {
    dest[b * n_slots + slot] = src[b];
  }
}

template <typename T>
__global__ void broadcast_sharpness_kernel(T* dest, const T* src, int64_t batch, int64_t n_slots) {
  const int64_t n = batch * n_slots;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dest[i] = src[i / n_slots];
  }
}

template <typename T>
__global__ void transpose_to_az_last_kernel(const cuda::std::complex<T>* in,
                                            cuda::std::complex<T>* out, int64_t batch, int64_t n_az,
                                            int64_t n_range) {
  // in (batch, n_az, n_range) → out (batch, n_range, n_az)
  const int64_t n = batch * n_range * n_az;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t a = i % n_az;
  const int64_t tmp = i / n_az;
  const int64_t r = tmp % n_range;
  const int64_t b = tmp / n_range;
  const int64_t src = (b * n_az + a) * n_range + r;
  out[i] = in[src];
}

template <typename T>
__global__ void transpose_from_az_last_kernel(const cuda::std::complex<T>* in,
                                              cuda::std::complex<T>* out, int64_t batch,
                                              int64_t n_az, int64_t n_range) {
  // in (batch, n_range, n_az) → out (batch, n_az, n_range)
  const int64_t n = batch * n_range * n_az;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t r = i % n_range;
  const int64_t tmp = i / n_range;
  const int64_t a = tmp % n_az;
  const int64_t b = tmp / n_az;
  const int64_t src = (b * n_range + r) * n_az + a;
  out[i] = in[src];
}

template <typename T>
__global__ void copy_n_kernel(const cuda::std::complex<T>* in, cuda::std::complex<T>* out,
                              int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i];
  }
}

// One element per thread: every kernel here indexes with a bare
// blockIdx.x * blockDim.x + threadIdx.x and no grid-stride loop, so the launch
// has to cover n exactly. Clamping the grid instead silently drops the tail --
// on a 2478 x 8000 auxiliary image that was the last 380 range rows, which came
// back out of image_out as zeros. 65535 is the limit for gridDim.y and .z; .x
// is capped at 2^31-1 on every GPU this module supports.
inline int grid_for(int64_t n, int threads) {
  if (n <= 0) {
    return 1;
  }
  const int64_t g = (n + threads - 1) / threads;
  if (g > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("AutofocusOp: image is too large to launch over");
  }
  return static_cast<int>(g);
}

template <typename T>
void copy_to_work(const void* in, AutofocusAzimuthAxis axis, int64_t batch, int64_t rows,
                  int64_t cols, cuda::std::complex<T>* work, int64_t n_range, int64_t n_az,
                  cudaStream_t stream, const std::string& who) {
  constexpr int kT = 256;
  const int64_t n = batch * n_range * n_az;
  if (axis == AutofocusAzimuthAxis::kU) {
    copy_n_kernel<T><<<grid_for(n, kT), kT, 0, stream>>>(
        static_cast<const cuda::std::complex<T>*>(in), work, n);
    check_cuda("copy to work", who);
    return;
  }
  (void)rows;
  (void)cols;
  transpose_to_az_last_kernel<T><<<grid_for(n, kT), kT, 0, stream>>>(
      static_cast<const cuda::std::complex<T>*>(in), work, batch, n_az, n_range);
  check_cuda("transpose to az-last", who);
}

template <typename T>
void copy_from_work(const cuda::std::complex<T>* work, void* out, AutofocusAzimuthAxis axis,
                    int64_t batch, int64_t n_range, int64_t n_az, cudaStream_t stream,
                    const std::string& who) {
  constexpr int kT = 256;
  const int64_t n = batch * n_range * n_az;
  if (axis == AutofocusAzimuthAxis::kU) {
    copy_n_kernel<T>
        <<<grid_for(n, kT), kT, 0, stream>>>(work, static_cast<cuda::std::complex<T>*>(out), n);
    check_cuda("copy from work", who);
    return;
  }
  transpose_from_az_last_kernel<T><<<grid_for(n, kT), kT, 0, stream>>>(
      work, static_cast<cuda::std::complex<T>*>(out), batch, n_az, n_range);
  check_cuda("transpose from az-last", who);
}

template <typename T>
void passthrough_outputs(const void* in, void* image_out, T* phase_out, T* sharpness_out,
                         int64_t batch, int64_t rows, int64_t cols, int64_t n_az, int64_t n_slots,
                         cudaStream_t stream, AutofocusState<T>& state, const std::string& who) {
  const int64_t n_img = batch * rows * cols;
  const size_t img_bytes = static_cast<size_t>(n_img) * sizeof(cuda::std::complex<T>);
  if (cudaMemcpyAsync(image_out, in, img_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": passthrough image copy failed");
  }
  const int64_t n_phase = batch * n_az;
  if (cudaMemsetAsync(phase_out, 0, static_cast<size_t>(n_phase) * sizeof(T), stream) !=
      cudaSuccess) {
    throw std::runtime_error(who + ": zero phase_correction failed");
  }
  constexpr int kT = 256;
  sharpness_kernel<T><<<static_cast<int>(batch), kT, 0, stream>>>(
      static_cast<const cuda::std::complex<T>*>(in), state.sharp_tmp, batch, rows * cols);
  check_cuda("passthrough sharpness", who);
  broadcast_sharpness_kernel<T><<<grid_for(batch * n_slots, kT), kT, 0, stream>>>(
      sharpness_out, state.sharp_tmp, batch, n_slots);
  check_cuda("broadcast sharpness", who);
}

// in / image_out: native (batch, rows, cols) = (..., v, u).
// Diagnostic escape hatch for the sharpness guard, off unless the environment
// sets it. When an estimate lowers sharpness the guard discards it and stops,
// which is correct but leaves the rejected estimate unobservable: phase_acc is
// never accumulated, so the emitted correction is zero and anything scoring it
// downstream sees the uncorrected error rather than the estimate that was
// refused. Setting SAR_AF_IGNORE_DROP keeps the estimate so it can be scored
// against a known truth. It is not a tuning knob and not a parameter -- the
// contract says the guard holds -- and it will leave the image worse.
inline bool autofocus_ignore_drop() {
  static const bool ignore = std::getenv("SAR_AF_IGNORE_DROP") != nullptr;
  return ignore;
}

// phase_out: (batch, n_az). sharpness_out: (batch, n_iter+1). outcome: what
// the run did, for the provenance ports; never null.
template <typename T>
void run_autofocus(const AutofocusConfig& cfg, AutofocusState<T>& state, const void* in,
                   int64_t batch, int64_t rows, int64_t cols, void* image_out, T* phase_out,
                   T* sharpness_out, AutofocusOutcome* outcome, cudaStream_t stream,
                   const std::string& who) {
  // Set on every exit below, including the three that return early.
  *outcome = AutofocusOutcome{};
  const bool az_u = cfg.azimuth_axis == AutofocusAzimuthAxis::kU;
  const int64_t n_range = az_u ? rows : cols;
  const int64_t n_az = az_u ? cols : rows;
  if (n_az < 2 || n_range < 1) {
    throw std::invalid_argument(who + ": image azimuth axis must have length >= 2");
  }
  if (cfg.num_iterations < 1) {
    throw std::invalid_argument(who + ": num_iterations must be >= 1");
  }
  ensure_autofocus_workspace(state, batch, n_range, n_az);

  cuda::std::complex<T>* work = state.work;
  cuda::std::complex<T>* spec = state.spec;
  cuda::std::complex<T>* scratch = state.scratch;
  const int64_t n_bins = batch * n_range;
  const int64_t n_pix = n_bins * n_az;
  const int64_t n_slots = cfg.num_iterations + 1;
  constexpr int kT = 256;

  copy_to_work<T>(in, cfg.azimuth_axis, batch, rows, cols, work, n_range, n_az, stream, who);

  if (cudaMemsetAsync(state.phase_acc, 0, static_cast<size_t>(batch * n_az) * sizeof(T), stream) !=
      cudaSuccess) {
    throw std::runtime_error(who + ": zero phase_acc failed");
  }

  sharpness_kernel<T>
      <<<static_cast<int>(batch), kT, 0, stream>>>(work, state.sharp_tmp, batch, n_range * n_az);
  check_cuda("sharpness 0", who);
  fill_sharpness_kernel<T>
      <<<grid_for(batch, kT), kT, 0, stream>>>(sharpness_out, state.sharp_tmp, batch, n_slots, 0);
  check_cuda("store sharpness 0", who);

  std::vector<T> host_sharp(static_cast<size_t>(batch));
  std::vector<T> host_sharp_new(static_cast<size_t>(batch));
  if (cudaMemcpyAsync(host_sharp.data(), state.sharp_tmp, static_cast<size_t>(batch) * sizeof(T),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": sharpness D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error(who + ": sharpness sync failed");
  }

  matx::cudaExecutor exec{stream};
  const uint64_t nfft = static_cast<uint64_t>(n_az);
  auto work_v = make_complex_view3<T>(work, batch, n_range, n_az);
  auto spec_v = make_complex_view3<T>(spec, batch, n_range, n_az);
  auto scratch_v = make_complex_view3<T>(scratch, batch, n_range, n_az);

  std::vector<int32_t> host_counts(static_cast<size_t>(batch));
  std::vector<T> host_p2p(static_cast<size_t>(batch));

  for (int iter = 0; iter < static_cast<int>(cfg.num_iterations); ++iter) {
    if (cudaMemsetAsync(state.counts, 0, static_cast<size_t>(batch) * sizeof(int32_t), stream) !=
        cudaSuccess) {
      throw std::runtime_error(who + ": zero counts failed");
    }
    peak_find_kernel<T><<<grid_for(n_bins, kT), kT, 0, stream>>>(
        work, state.peaks, state.strong, state.peak_pwr, state.counts, n_bins, n_range, n_az,
        static_cast<T>(kPgaMinPeakToMean));
    check_cuda("peak find", who);
    if (cudaMemcpyAsync(host_counts.data(), state.counts,
                        static_cast<size_t>(batch) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
      throw std::runtime_error(who + ": counts D2H failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error(who + ": counts sync failed");
    }
    cap_strong_scatterers<T>(state.strong, state.peak_pwr, host_counts.data(), batch, n_range,
                             cfg.min_scatterers, stream, who, iter);
    bool weak = false;
    int32_t seen = host_counts[0];
    for (int64_t b = 0; b < batch; ++b) {
      seen = std::min(seen, host_counts[static_cast<size_t>(b)]);
      if (host_counts[static_cast<size_t>(b)] < cfg.min_scatterers) {
        weak = true;
      }
    }
    if (weak) {
      if (iter == 0) {
        HOLOSCAN_LOG_WARN(
            "{}: fewer than {} strong scatterers (saw {}); passing "
            "the image through unchanged",
            who, cfg.min_scatterers, seen);
        outcome->termination = AutofocusTermination::kRefusedScatterers;
        passthrough_outputs<T>(in, image_out, phase_out, sharpness_out, batch, rows, cols, n_az,
                               n_slots, stream, state, who);
        return;
      }
      HOLOSCAN_LOG_WARN(
          "{}: fewer than {} strong scatterers (saw {}) on "
          "iteration {}; keeping the estimate from prior iterations",
          who, cfg.min_scatterers, seen, iter);
      outcome->termination = AutofocusTermination::kStoppedScatterers;
      for (int64_t s = iter + 1; s < n_slots; ++s) {
        fill_sharpness_kernel<T><<<grid_for(batch, kT), kT, 0, stream>>>(
            sharpness_out, state.sharp_tmp, batch, n_slots, s);
      }
      check_cuda("fill remaining sharpness", who);
      break;
    }

    const int64_t width = pga_window_width(n_az, cfg.window_fraction, iter);
    circshift_window_kernel<T><<<grid_for(n_pix, kT), kT, 0, stream>>>(
        work, scratch, state.peaks, state.strong, n_bins, n_az, width);
    check_cuda("circshift window", who);
    if (iter == 0) {
      HOLOSCAN_LOG_INFO("{}: LUE over {} azimuth bins, image windowed to {}", who, n_az, width);
    }

    // The window narrows the image support, not the transform: FFT the full
    // n_az line with its zeros so adjacent bins stay one phase-history sample
    // apart. See the header note on why packing to `width` loses the edges.
    (scratch_v = matx::fft(scratch_v, nfft, matx::FFTNorm::BACKWARD)).run(exec);

    lue_kernel<T>
        <<<grid_for(batch * n_az, kT), kT, 0, stream>>>(scratch, state.grad, batch, n_range, n_az);
    check_cuda("LUE gradient", who);
    bin_power_kernel<T><<<grid_for(batch * n_az, kT), kT, 0, stream>>>(scratch, state.bin_pwr,
                                                                       batch, n_range, n_az);
    check_cuda("azimuth-FFT bin power", who);

    integrate_detrend_kernel<T><<<static_cast<int>(batch), 1, 0, stream>>>(
        scratch, state.bin_pwr, state.grad, state.phase_iter, n_range, n_az);
    check_cuda("integrate detrend", who);

    // `spec` is free from here; use it as a backup of work so a sharpness
    // drop can restore the last accepted image.
    copy_n_kernel<T><<<grid_for(n_pix, kT), kT, 0, stream>>>(work, spec, n_pix);
    check_cuda("backup work", who);

    (scratch_v = matx::fft(work_v, nfft, matx::FFTNorm::BACKWARD)).run(exec);
    mul_phasor_kernel<T><<<grid_for(n_pix, kT), kT, 0, stream>>>(scratch, state.phase_iter, batch,
                                                                 n_range, n_az, static_cast<T>(-1));
    check_cuda("apply phase", who);
    (work_v = matx::ifft(scratch_v, nfft, matx::FFTNorm::BACKWARD)).run(exec);

    sharpness_kernel<T>
        <<<static_cast<int>(batch), kT, 0, stream>>>(work, state.sharp_tmp, batch, n_range * n_az);
    check_cuda("sharpness iter", who);
    if (cudaMemcpyAsync(host_sharp_new.data(), state.sharp_tmp,
                        static_cast<size_t>(batch) * sizeof(T), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
      throw std::runtime_error(who + ": sharpness-new D2H failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error(who + ": sharpness-new sync failed");
    }
    bool drop = false;
    for (int64_t b = 0; b < batch; ++b) {
      if (host_sharp_new[static_cast<size_t>(b)] < host_sharp[static_cast<size_t>(b)]) {
        drop = true;
      }
    }
    if (drop && autofocus_ignore_drop()) {
      HOLOSCAN_LOG_WARN(
          "{}: iteration {} sharpness dropped ({:g} -> {:g}); keeping it anyway "
          "(SAR_AF_IGNORE_DROP) -- diagnostic only, the image will be worse",
          who, iter, static_cast<double>(host_sharp[0]), static_cast<double>(host_sharp_new[0]));
      drop = false;
    }
    if (drop) {
      HOLOSCAN_LOG_WARN(
          "{}: iteration {} sharpness dropped ({:g} -> {:g}); "
          "keeping the prior estimate",
          who, iter, static_cast<double>(host_sharp[0]), static_cast<double>(host_sharp_new[0]));
      // Nothing accepted yet means nothing applied at all, which is a refusal;
      // a drop after at least one accepted iteration is ordinary convergence.
      outcome->termination =
          (iter == 0) ? AutofocusTermination::kRefusedSharpness : AutofocusTermination::kConverged;
      copy_n_kernel<T><<<grid_for(n_pix, kT), kT, 0, stream>>>(spec, work, n_pix);
      check_cuda("restore work", who);
      if (cudaMemcpyAsync(state.sharp_tmp, host_sharp.data(),
                          static_cast<size_t>(batch) * sizeof(T), cudaMemcpyHostToDevice,
                          stream) != cudaSuccess) {
        throw std::runtime_error(who + ": restore sharpness H2D failed");
      }
      for (int64_t s = iter + 1; s < n_slots; ++s) {
        fill_sharpness_kernel<T><<<grid_for(batch, kT), kT, 0, stream>>>(
            sharpness_out, state.sharp_tmp, batch, n_slots, s);
      }
      check_cuda("fill remaining sharpness", who);
      break;
    }
    host_sharp = host_sharp_new;
    add_phase_kernel<T><<<grid_for(batch * n_az, kT), kT, 0, stream>>>(
        state.phase_acc, state.phase_iter, batch * n_az);
    check_cuda("accumulate phase", who);
    // Counted here rather than from the loop index: this is the only point an
    // iteration's phase enters the estimate.
    ++outcome->iterations_accepted;
    if (outcome->iterations_accepted == cfg.num_iterations) {
      outcome->termination = AutofocusTermination::kBudgetExhausted;
    }
    fill_sharpness_kernel<T><<<grid_for(batch, kT), kT, 0, stream>>>(sharpness_out, state.sharp_tmp,
                                                                     batch, n_slots, iter + 1);
    check_cuda("store sharpness iter", who);
  }

  p2p_kernel<T><<<static_cast<int>(batch), kT, 0, stream>>>(state.phase_acc, state.p2p, n_az);
  check_cuda("phase peak-to-peak", who);
  if (cudaMemcpyAsync(host_p2p.data(), state.p2p, static_cast<size_t>(batch) * sizeof(T),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": p2p D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error(who + ": p2p sync failed");
  }
  if (std::isfinite(cfg.max_correction)) {
    bool over = false;
    T peak = static_cast<T>(0);
    for (int64_t b = 0; b < batch; ++b) {
      peak = std::max(peak, host_p2p[static_cast<size_t>(b)]);
      if (static_cast<double>(host_p2p[static_cast<size_t>(b)]) > cfg.max_correction) {
        over = true;
      }
    }
    if (over) {
      HOLOSCAN_LOG_WARN(
          "{}: estimated correction peak-to-peak {} rad exceeds "
          "max_correction {}; passing the image through unchanged",
          who, static_cast<double>(peak), cfg.max_correction);
      // The estimate is discarded along with the iterations that built it, so
      // the count goes back to zero to match the zero phase_correction emitted.
      outcome->termination = AutofocusTermination::kRefusedMaxCorrection;
      outcome->iterations_accepted = 0;
      passthrough_outputs<T>(in, image_out, phase_out, sharpness_out, batch, rows, cols, n_az,
                             n_slots, stream, state, who);
      return;
    }
  }

  copy_from_work<T>(work, image_out, cfg.azimuth_axis, batch, n_range, n_az, stream, who);
  // `image_out` was corrected in the spectrum domain, where `phase_acc`'s bin
  // order is already right. Only the emitted vector moves to pulse order, and
  // it is emitted with the sign of the error so `PhaseCorrectionOp` with
  // sense=subtract removes it. Both halves of that used to be wrong -- no
  // shift, and a negate -- and they partially cancel on a symmetric quadratic,
  // which is why the pair survived a 0.7 peak-recovery gate. On Kourou the
  // residual against a known injected error is 0.146 rad corrected, 1.246 rad
  // with the old pair, against a 2.400 rad uncorrected baseline.
  fftshift_phase_kernel<T>
      <<<grid_for(batch * n_az, kT), kT, 0, stream>>>(state.phase_acc, phase_out, batch, n_az);
  check_cuda("fftshift phase onto the pulse axis", who);
}

}  // namespace holoscan::radar::detail
