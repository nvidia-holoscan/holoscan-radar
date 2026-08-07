// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MUSIC (subspace) sparse-path DOA estimator stage.
//
// Per item i the pipeline is
//   1. R_i = (1/K) sum_{s valid} x_{i,s} x_{i,s}^H       (covariance_kernel)
//   2. (E_i, lambda_i) = eig(R_i)  (ascending, VECTOR)   (batched matx::eig)
//   3. d_i = model order (fixed, or per-item EFT on lambda_i, with the
//      rank-deficient K < channels fallback baked in)    (resolve_model_order)
//   4. P_n,i = sum_{j=0}^{C-d_i-1} e_j e_j^H              (projector_kernel)
//      (the num_channels - d_i smallest-eigenvalue eigenvectors)
//   5. P_i(theta) = 1 / (a(theta)^H P_n,i a(theta))      (music_reduce_kernel)
//      argmax over theta -> aoa_estimates
//   6. top-min(d_i, max_sources) peaks of P_i(theta)       (music_peak_extract)
// where a(theta) is the untapered steering manifold (num_beams, num_channels),
// beam el-major (beam = ie*num_az + ia).
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>

#include "radar_common/dsp/source_count_device.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// Floor on the MUSIC denominator a^H P_n a (a projector, so this is always
// in [0, num_channels] for the untapered unit-magnitude-entry steering
// manifold). MUSIC's sharp peaks are exactly the near-zero denominators, so
// (unlike Capon) a zero denominator must NOT be treated as "no power" - it
// is floored, not gated, to avoid inf/nan while preserving the peak.
template <typename T>
inline constexpr T kMusicDenomFloor = static_cast<T>(1e-6);

// See mvdr_stage.hpp's identical cache-retirement rationale (MatX inv/eig
// plans keyed on device pointers that do not outlive Holoscan's per-tick
// scratch). MUSIC only ever uses eig (VECTOR mode), no inv.
inline std::atomic<bool>& music_eig_plans_live_flag() {
  static std::atomic<bool> live{false};
  return live;
}

inline void mark_music_eig_cache_live() {
  music_eig_plans_live_flag().store(true, std::memory_order_relaxed);
}

inline void clear_music_eig_cache() {
  matx::detail::GetCache().Clear<matx::detail::eig_cuda_cache_t>(
      matx::detail::GetCacheIdFromType<matx::detail::eig_cuda_cache_t>());
}

// Per-tick: clear the eig plan cache (keys are freed-at-return scratch
// pointers) to bound growth — WITHOUT draining MatX's global MemTracker. The
// global drain frees every MatX-tracked buffer in the process, including
// sibling operators' cached cuFFT workspaces (FMCWRange/DopplerProcessorOp),
// which they reuse next tick → out-of-bounds reads and corrupted output.
// Leaves the live flag set so the teardown retire still drains. See
// mvdr_stage.hpp for the identical split.
inline void clear_music_matx_caches() {
  const bool live =
      music_eig_plans_live_flag().load(std::memory_order_relaxed);
  if (!live) {
    return;
  }
  cudaDeviceSynchronize();
  clear_music_eig_cache();
}

// Teardown: clear the eig plan cache and drain MatX's MemTracker while CUDA is
// still valid. Call once from the operator's stop() (and from Python atexit as
// backup) — NOT per tick (see clear_music_matx_caches above).
inline void retire_music_matx_caches() {
  const bool live =
      music_eig_plans_live_flag().exchange(false, std::memory_order_relaxed);
  if (!live) {
    return;
  }
  cudaDeviceSynchronize();
  clear_music_eig_cache();
  auto& tracker = matx::GetAllocMap();
  while (tracker.size() > 0) {
    tracker.deallocate(tracker.allocationMap.begin()->first);
  }
}

// One block per item. Forms the sample covariance R_i (num_channels x
// num_channels, row-major) over the valid snapshots. Identical to MVDR's
// covariance_kernel (mvdr_stage.hpp) - duplicated rather than shared because
// radar_common stays CUDA-free and per-operator kernels are not (yet) a
// shared target; see OPEN_ITEMS.md if a third consumer wants to factor this
// out. `mask` (num_items x num_snapshots, uint8/bool) selects valid slots;
// nullptr -> all valid. Empty items (K_i == 0) yield a zero matrix;
// `k_counts[item]` records K_i.
template <typename T>
__global__ void music_covariance_kernel(
    const cuda::std::complex<T>* __restrict__ x,
    const unsigned char* __restrict__ mask, int64_t num_items,
    int64_t num_channels, int64_t num_snapshots,
    cuda::std::complex<T>* __restrict__ cov, int64_t* __restrict__ k_counts) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }

  __shared__ int64_t k_valid;
  if (threadIdx.x == 0) {
    int64_t count = 0;
    for (int64_t s = 0; s < num_snapshots; ++s) {
      const bool valid =
          mask == nullptr || mask[item * num_snapshots + s] != 0;
      count += valid ? 1 : 0;
    }
    k_valid = count;
    k_counts[item] = count;
  }
  __syncthreads();

  const int64_t k_count = k_valid;
  const cuda::std::complex<T>* xi = x + item * num_channels * num_snapshots;
  const int64_t entries = num_channels * num_channels;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  for (int64_t e = tid; e < entries; e += nthreads) {
    const int64_t c1 = e / num_channels;
    const int64_t c2 = e % num_channels;
    cuda::std::complex<T> acc{T{0}, T{0}};
    for (int64_t s = 0; s < num_snapshots; ++s) {
      const bool valid =
          mask == nullptr || mask[item * num_snapshots + s] != 0;
      if (!valid) {
        continue;
      }
      const cuda::std::complex<T> a = xi[c1 * num_snapshots + s];
      const cuda::std::complex<T> b = xi[c2 * num_snapshots + s];
      acc += cuda::std::complex<T>(a.real() * b.real() + a.imag() * b.imag(),
                                   a.imag() * b.real() - a.real() * b.imag());
    }
    if (k_count > 0) {
      const T scale = T{1} / static_cast<T>(k_count);
      acc = cuda::std::complex<T>(acc.real() * scale, acc.imag() * scale);
    } else {
      acc = cuda::std::complex<T>(T{0}, T{0});
    }
    cov[item * entries + e] = acc;
  }
}

// Broadcasts a fixed model order into every item's slot (num_sources: fixed
// mode - no per-item EFT decision needed).
__global__ inline void fill_model_order_kernel(int32_t* __restrict__ out,
                                               int64_t num_items,
                                               int32_t value) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < num_items) {
    out[i] = value;
  }
}

// One block per item. Builds the noise-subspace projector
// P_n = sum_{j=0}^{noise_dim-1} e_j e_j^H from the ascending eigenvectors
// (columns of `evec`, row-major (num_channels, num_channels) per item) and
// the per-item `model_order` (noise_dim = num_channels - model_order,
// clamped to >= 0 defensively - callers validate model_order <=
// num_channels - 1 upstream).
template <typename T>
__global__ void projector_kernel(
    const cuda::std::complex<T>* __restrict__ evec,
    const int32_t* __restrict__ model_order, int64_t num_items,
    int64_t num_channels, cuda::std::complex<T>* __restrict__ proj) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }
  const int64_t entries = num_channels * num_channels;
  const cuda::std::complex<T>* e = evec + item * entries;
  cuda::std::complex<T>* p = proj + item * entries;

  int64_t noise_dim = num_channels - static_cast<int64_t>(model_order[item]);
  noise_dim = noise_dim < 0 ? 0 : noise_dim;
  noise_dim = noise_dim > num_channels ? num_channels : noise_dim;

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  for (int64_t idx = tid; idx < entries; idx += nthreads) {
    const int64_t r = idx / num_channels;
    const int64_t c = idx % num_channels;
    cuda::std::complex<T> acc{T{0}, T{0}};
    for (int64_t j = 0; j < noise_dim; ++j) {
      const cuda::std::complex<T> er = e[r * num_channels + j];
      const cuda::std::complex<T> ec = e[c * num_channels + j];
      // er * conj(ec)
      const T re = er.real() * ec.real() + er.imag() * ec.imag();
      const T im = er.imag() * ec.real() - er.real() * ec.imag();
      acc += cuda::std::complex<T>(re, im);
    }
    p[idx] = acc;
  }
}

// spectrum is stored (num_az, num_el) per item; beams are el-major
// (beam = ie*num_az + ia) matching the steering-manifold build.
template <typename T>
__device__ inline T& music_spectrum_at(T* spec, int64_t ia, int64_t ie,
                                       int64_t num_el) {
  return spec[ia * num_el + ie];
}

// One block per item. Threads sweep the beam grid computing the MUSIC power
// P_b = 1 / (a_b^H P_n a_b), writing the (num_az, num_el) spectrum and
// keeping a running argmax (lowest beam index wins ties). Unlike Capon, a
// near-zero denominator is a genuine (sharp) MUSIC peak, so it is floored
// (kMusicDenomFloor), never gated to zero power.
//
// `Staged`: when true, P_n (num_channels^2 complex) is cooperatively copied
// into shared memory once per block, then read from there in the beam loop.
// ncu on the xl case (RTX PRO 6000 Blackwell) showed this kernel is L1-
// bandwidth bound (72% memory / 7% compute; 81% of warp stalls are L1TEX
// scoreboard waits) because P_n is re-read from L1 for every one of the
// thousands of beams; staging it into shared moves that traffic off the L1TEX
// load path. (MVDR's capon_reduce_kernel carries an older "shared was slower,
// kernel is latency-bound" note from a different GPU - it does not hold here;
// see OPEN_ITEMS.md.) The launch falls back to Staged=false when P_n does not
// fit the shared-memory budget. The inner reduction over c2 is carried in
// four independent accumulators so the FMAs pipeline.
template <typename T, bool Staged>
__global__ void music_reduce_kernel(
    const cuda::std::complex<T>* __restrict__ proj,
    const cuda::std::complex<T>* __restrict__ manifold, int64_t num_items,
    int64_t num_channels, int64_t num_az, int64_t num_el,
    const T* __restrict__ az_grid, const T* __restrict__ el_grid,
    T* __restrict__ spectrum, T* __restrict__ aoa) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }
  const int64_t num_beams = num_az * num_el;
  const int64_t cc = num_channels * num_channels;
  const cuda::std::complex<T>* Pn_g = proj + item * cc;
  T* spec = spectrum + item * num_beams;

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  // Shared layout: [P_n (Staged only)] then [power T[blockDim]] then
  // [index int64_t[blockDim]]. P_n (complex, 16-byte aligned for double) leads
  // so the reduction buffers keep their natural alignment.
  extern __shared__ __align__(16) unsigned char smem[];
  const cuda::std::complex<T>* Pn;
  T* red_pow;
  int64_t* red_idx;
  if constexpr (Staged) {
    auto* Pn_s = reinterpret_cast<cuda::std::complex<T>*>(smem);
    for (int64_t idx = tid; idx < cc; idx += nthreads) {
      Pn_s[idx] = Pn_g[idx];
    }
    red_pow = reinterpret_cast<T*>(Pn_s + cc);
    red_idx = reinterpret_cast<int64_t*>(red_pow + nthreads);
    __syncthreads();
    Pn = Pn_s;
  } else {
    red_pow = reinterpret_cast<T*>(smem);
    red_idx = reinterpret_cast<int64_t*>(red_pow + nthreads);
    Pn = Pn_g;
  }

  T best_power = T{0};
  int64_t best_beam = -1;
  for (int64_t beam = tid; beam < num_beams; beam += nthreads) {
    const cuda::std::complex<T>* a = manifold + beam * num_channels;
    // Invisible beams (direction-cosine out-of-visible-region) carry an
    // all-zero manifold entry, so a^H a == 0 - guard this BEFORE the
    // denominator floor below, else the zero-vector "peak" (0/eps) would be
    // reported as a huge spurious power instead of the contract's zero.
    T a_norm = T{0};
    for (int64_t c = 0; c < num_channels; ++c) {
      a_norm += a[c].real() * a[c].real() + a[c].imag() * a[c].imag();
    }
    if (!(a_norm > kMusicDenomFloor<T>)) {
      music_spectrum_at(spec, beam % num_az, beam / num_az, num_el) = T{0};
      continue;
    }
    // aH_P_n_a = sum_c conj(a[c]) * (sum_c2 Pn[c,c2]*a[c2]). The inner
    // reduction over c2 is carried in four independent accumulators so the
    // FMAs pipeline instead of stalling on one serial chain.
    T denom = T{0};
    for (int64_t c = 0; c < num_channels; ++c) {
      const cuda::std::complex<T>* row = Pn + c * num_channels;
      T q0r = T{0}, q0i = T{0}, q1r = T{0}, q1i = T{0};
      T q2r = T{0}, q2i = T{0}, q3r = T{0}, q3i = T{0};
      int64_t c2 = 0;
      for (; c2 + 3 < num_channels; c2 += 4) {
        const T r0r = row[c2].real(), r0i = row[c2].imag();
        const T a0r = a[c2].real(), a0i = a[c2].imag();
        q0r += r0r * a0r - r0i * a0i;
        q0i += r0r * a0i + r0i * a0r;
        const T r1r = row[c2 + 1].real(), r1i = row[c2 + 1].imag();
        const T a1r = a[c2 + 1].real(), a1i = a[c2 + 1].imag();
        q1r += r1r * a1r - r1i * a1i;
        q1i += r1r * a1i + r1i * a1r;
        const T r2r = row[c2 + 2].real(), r2i = row[c2 + 2].imag();
        const T a2r = a[c2 + 2].real(), a2i = a[c2 + 2].imag();
        q2r += r2r * a2r - r2i * a2i;
        q2i += r2r * a2i + r2i * a2r;
        const T r3r = row[c2 + 3].real(), r3i = row[c2 + 3].imag();
        const T a3r = a[c2 + 3].real(), a3i = a[c2 + 3].imag();
        q3r += r3r * a3r - r3i * a3i;
        q3i += r3r * a3i + r3i * a3r;
      }
      T qr = q0r + q1r + q2r + q3r;
      T qi = q0i + q1i + q2i + q3i;
      for (; c2 < num_channels; ++c2) {
        const T rr = row[c2].real(), ri = row[c2].imag();
        const T ar = a[c2].real(), ai = a[c2].imag();
        qr += rr * ar - ri * ai;
        qi += rr * ai + ri * ar;
      }
      // Re(conj(a[c]) * q)
      denom += a[c].real() * qr + a[c].imag() * qi;
    }
    const T floor = kMusicDenomFloor<T>;
    const T power = T{1} / (denom > floor ? denom : floor);
    const int64_t ia = beam % num_az;
    const int64_t ie = beam / num_az;
    music_spectrum_at(spec, ia, ie, num_el) = power;
    if (power > best_power) {
      best_power = power;
      best_beam = beam;
    }
  }

  red_pow[tid] = best_power;
  red_idx[tid] = best_beam;
  __syncthreads();
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const T other_pow = red_pow[tid + stride];
      const int64_t other_idx = red_idx[tid + stride];
      const bool take = other_pow > red_pow[tid] ||
                        (other_pow == red_pow[tid] && other_idx >= 0 &&
                         (red_idx[tid] < 0 || other_idx < red_idx[tid]));
      if (take) {
        red_pow[tid] = other_pow;
        red_idx[tid] = other_idx;
      }
    }
    __syncthreads();
  }
  if (tid != 0) {
    return;
  }

  best_power = red_pow[0];
  best_beam = red_idx[0];
  if (best_beam < 0 || best_power <= T{0}) {
    const T nan = T{0} / T{0};
    aoa[item * 2] = nan;
    aoa[item * 2 + 1] = nan;
  } else {
    const int64_t ia = best_beam % num_az;
    const int64_t ie = best_beam / num_az;
    aoa[item * 2] = az_grid[ia];
    aoa[item * 2 + 1] = el_grid[ie];
  }
}

// Strict 2-D local maximum - identical rule to MVDR's mvdr_is_local_max.
template <typename T>
__device__ inline bool music_is_local_max(const T* __restrict__ spec,
                                          int64_t ia, int64_t ie,
                                          int64_t num_az, int64_t num_el,
                                          T p) {
  if (ia > 0 && spec[(ia - 1) * num_el + ie] >= p) {
    return false;
  }
  if (ia < num_az - 1 && spec[(ia + 1) * num_el + ie] >= p) {
    return false;
  }
  if (ie > 0 && spec[ia * num_el + (ie - 1)] >= p) {
    return false;
  }
  if (ie < num_el - 1 && spec[ia * num_el + (ie + 1)] >= p) {
    return false;
  }
  return true;
}

// One block per item. Extracts up to `target_count_i = min(model_order[i],
// max_sources)` MUSIC-spectrum peaks into doa_multi (num_items, max_sources,
// 2), with source_count (num_items,) the number actually found (<=
// target_count_i, itself <= max_sources); unused rows are NaN.
//
// Pass 0 is a *global* argmax (matches music_reduce_kernel's tie-break) so
// doa_multi[.,0,.] equals aoa_estimates even on flat plateaus. Passes k>=1
// require a strict 2-D local maximum so a single lobe is not double-counted.
// Differs from MVDR's peak_extract_kernel only in the per-item target count
// (MUSIC targets the resolved model order, not always max_sources).
template <typename T>
__global__ void music_peak_extract_kernel(
    const T* __restrict__ spectrum, const int32_t* __restrict__ model_order,
    int64_t num_items, int64_t num_az, int64_t num_el, int64_t max_sources,
    const T* __restrict__ az_grid, const T* __restrict__ el_grid,
    T* __restrict__ doa_multi, int32_t* __restrict__ source_count) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }
  const int64_t num_beams = num_az * num_el;
  const T* spec = spectrum + item * num_beams;
  int64_t target = static_cast<int64_t>(model_order[item]);
  target = target < 1 ? 1 : target;
  target = target > max_sources ? max_sources : target;

  extern __shared__ __align__(16) unsigned char smem_pk[];
  auto* red_pow = reinterpret_cast<T*>(smem_pk);
  auto* red_idx =
      reinterpret_cast<int64_t*>(red_pow + static_cast<int64_t>(blockDim.x));
  int64_t* sel = red_idx + static_cast<int64_t>(blockDim.x);

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  int64_t found = 0;
  for (int64_t k = 0; k < target; ++k) {
    T best_power = T{0};
    int64_t best_beam = -1;
    for (int64_t beam = tid; beam < num_beams; beam += nthreads) {
      const int64_t ia = beam % num_az;
      const int64_t ie = beam / num_az;
      const T p = spec[ia * num_el + ie];
      if (p <= T{0}) {
        continue;
      }
      if (k > 0 && !music_is_local_max(spec, ia, ie, num_az, num_el, p)) {
        continue;
      }
      bool taken = false;
      for (int64_t j = 0; j < k; ++j) {
        if (sel[j] == beam) {
          taken = true;
          break;
        }
      }
      if (taken) {
        continue;
      }
      if (p > best_power ||
          (p == best_power && (best_beam < 0 || beam < best_beam))) {
        best_power = p;
        best_beam = beam;
      }
    }

    red_pow[tid] = best_power;
    red_idx[tid] = best_beam;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const T other_pow = red_pow[tid + stride];
        const int64_t other_idx = red_idx[tid + stride];
        const bool take = other_pow > red_pow[tid] ||
                          (other_pow == red_pow[tid] && other_idx >= 0 &&
                           (red_idx[tid] < 0 || other_idx < red_idx[tid]));
        if (take) {
          red_pow[tid] = other_pow;
          red_idx[tid] = other_idx;
        }
      }
      __syncthreads();
    }

    const T chosen_pow = red_pow[0];
    const int64_t chosen = red_idx[0];
    if (tid == 0) {
      sel[k] = (chosen >= 0 && chosen_pow > T{0}) ? chosen : -1;
    }
    __syncthreads();
    if (chosen < 0 || chosen_pow <= T{0}) {
      break;
    }
    found = k + 1;
  }

  if (tid != 0) {
    return;
  }
  const T nan = T{0} / T{0};
  for (int64_t k = 0; k < max_sources; ++k) {
    const int64_t base = (item * max_sources + k) * 2;
    if (k < found && sel[k] >= 0) {
      const int64_t ia = sel[k] % num_az;
      const int64_t ie = sel[k] / num_az;
      doa_multi[base] = az_grid[ia];
      doa_multi[base + 1] = el_grid[ie];
    } else {
      doa_multi[base] = nan;
      doa_multi[base + 1] = nan;
    }
  }
  source_count[item] = static_cast<int32_t>(found);
}

// One block per item. When K_i == 0 the covariance was empty: zero the MUSIC
// spectrum and NaN-fill DOA outputs. Identical role to MVDR's
// blank_empty_items_kernel.
template <typename T>
__global__ void music_blank_empty_items_kernel(
    const int64_t* __restrict__ k_counts, int64_t num_items, int64_t num_az,
    int64_t num_el, int64_t max_sources, T* __restrict__ spectrum,
    T* __restrict__ aoa, T* __restrict__ doa_multi,
    int32_t* __restrict__ source_count) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items || k_counts[item] > 0) {
    return;
  }
  const int64_t num_beams = num_az * num_el;
  T* spec = spectrum + item * num_beams;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  for (int64_t b = tid; b < num_beams; b += nthreads) {
    spec[b] = T{0};
  }
  if (tid != 0) {
    return;
  }
  const T nan = T{0} / T{0};
  aoa[item * 2] = nan;
  aoa[item * 2 + 1] = nan;
  for (int64_t k = 0; k < max_sources; ++k) {
    const int64_t base = (item * max_sources + k) * 2;
    doa_multi[base] = nan;
    doa_multi[base + 1] = nan;
  }
  source_count[item] = 0;
}

// Run the full MUSIC estimate for one tick: covariance -> eig (VECTOR) ->
// model-order resolution -> noise-subspace projector -> MUSIC scan ->
// multi-peak extraction. All work is enqueued on `stream`; no
// synchronization. `cov`/`evec` are caller-owned scratch of shape
// (num_items, num_channels, num_channels); `proj` likewise. `k_counts` is
// caller-owned scratch of shape (num_items,) for per-item valid snapshot
// counts. `fixed_num_sources < 0` selects auto (per-item EFT via
// `pfa`/`max_sources`); >= 0 broadcasts that fixed model order to every item
// (validated by the caller to be in [1, num_channels - 1]).
//
//   snapshots — device ptr, [num_items, num_channels, num_snapshots]
//               cuda::std::complex<T>.
//   mask      — device ptr, [num_items, num_snapshots] uint8/bool, or nullptr.
//   manifold  — device ptr, [num_beams, num_channels] cuda::std::complex<T>
//               untapered steering vectors, beam el-major.
//   eigenvalues — device ptr, [num_items, num_channels] T. Ascending.
//   model_order — device ptr, [num_items] int32 scratch (resolved order).
//   doa_multi   — device ptr, [num_items, max_sources, 2] T.
//   source_count — device ptr, [num_items] int32. Peaks found.
template <typename T>
inline void run_music_stage(const cuda::std::complex<T>* snapshots,
                            const unsigned char* mask, int64_t num_items,
                            int64_t num_channels, int64_t num_snapshots,
                            int64_t fixed_num_sources, double pfa,
                            const cuda::std::complex<T>* manifold,
                            int64_t num_az, int64_t num_el,
                            const T* az_grid, const T* el_grid,
                            int64_t max_sources,
                            cuda::std::complex<T>* cov,
                            cuda::std::complex<T>* evec,
                            cuda::std::complex<T>* proj, int64_t* k_counts,
                            int32_t* model_order, T* spectrum, T* aoa,
                            T* eigenvalues, T* doa_multi,
                            int32_t* source_count, cudaStream_t stream) {
  if (num_items <= 0 || num_channels <= 0) {
    return;
  }

  constexpr int cov_block = 256;
  const int grid = static_cast<int>(num_items);
  music_covariance_kernel<T><<<grid, cov_block, 0, stream>>>(
      snapshots, mask, num_items, num_channels, num_snapshots, cov, k_counts);

  {
    auto cov_v = matx::make_tensor<cuda::std::complex<T>, 3>(
        cov, {num_items, num_channels, num_channels});
    auto evec_v = matx::make_tensor<cuda::std::complex<T>, 3>(
        evec, {num_items, num_channels, num_channels});
    auto eval_v =
        matx::make_tensor<T, 2>(eigenvalues, {num_items, num_channels});
    matx::cudaExecutor eig_exec{stream};
    (matx::mtie(evec_v, eval_v) = matx::eig(cov_v, matx::EigenMode::VECTOR))
        .run(eig_exec);
  }

  if (fixed_num_sources < 0) {
    run_source_count_device<T>(eigenvalues, k_counts, num_items, num_channels,
                               pfa, max_sources, model_order, stream);
  } else {
    constexpr int fill_block = 128;
    const int fill_grid =
        static_cast<int>((num_items + fill_block - 1) / fill_block);
    fill_model_order_kernel<<<fill_grid, fill_block, 0, stream>>>(
        model_order, num_items, static_cast<int32_t>(fixed_num_sources));
  }

  constexpr int proj_block = 256;
  projector_kernel<T><<<grid, proj_block, 0, stream>>>(
      evec, model_order, num_items, num_channels, proj);

  constexpr int reduce_block = 256;
  const size_t red_smem =
      static_cast<size_t>(reduce_block) * (sizeof(T) + sizeof(int64_t));
  // Staging P_n (num_channels^2 complex) in shared lifts the reduce kernel's
  // L1-bandwidth bottleneck (P_n is otherwise re-read from L1 for every beam;
  // ncu measured 72%->relieved memory throughput and -18.7% on xl). But the
  // cooperative copy + __syncthreads only pays off once the grid saturates the
  // GPU: below one full occupancy wave the device is occupancy/latency-bound
  // and the barrier is pure overhead (measured +6.6% on lg, which is ~half a
  // wave here). So gate on (a) P_n fitting a conservative shared budget (stays
  // within the 48 KB default cap without an opt-in attribute) and (b) the grid
  // spanning >= one full occupancy wave, both queried at runtime so the gate
  // adapts to the GPU rather than hard-coding a size threshold.
  const size_t pn_smem =
      static_cast<size_t>(num_channels) * static_cast<size_t>(num_channels) *
      sizeof(cuda::std::complex<T>);
  constexpr size_t kStagedSmemBudget = 40 * 1024;
  bool stage = pn_smem + red_smem <= kStagedSmemBudget;
  if (stage) {
    static const int sm_count = [] {
      int dev = 0;
      cudaGetDevice(&dev);
      int n = 0;
      cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
      return n > 0 ? n : 1;
    }();
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, music_reduce_kernel<T, true>, reduce_block,
        pn_smem + red_smem);
    const int64_t full_wave =
        static_cast<int64_t>(blocks_per_sm) * sm_count;
    stage = blocks_per_sm > 0 && num_items >= full_wave;
  }
  if (stage) {
    music_reduce_kernel<T, true>
        <<<grid, reduce_block, pn_smem + red_smem, stream>>>(
            proj, manifold, num_items, num_channels, num_az, num_el, az_grid,
            el_grid, spectrum, aoa);
  } else {
    music_reduce_kernel<T, false><<<grid, reduce_block, red_smem, stream>>>(
        proj, manifold, num_items, num_channels, num_az, num_el, az_grid,
        el_grid, spectrum, aoa);
  }

  constexpr int peak_block = 256;
  const size_t peak_smem =
      static_cast<size_t>(peak_block) * (sizeof(T) + sizeof(int64_t)) +
      static_cast<size_t>(max_sources) * sizeof(int64_t);
  music_peak_extract_kernel<T><<<grid, peak_block, peak_smem, stream>>>(
      spectrum, model_order, num_items, num_az, num_el, max_sources, az_grid,
      el_grid, doa_multi, source_count);

  constexpr int blank_block = 256;
  music_blank_empty_items_kernel<T><<<grid, blank_block, 0, stream>>>(
      k_counts, num_items, num_az, num_el, max_sources, spectrum, aoa,
      doa_multi, source_count);

  mark_music_eig_cache_live();
  clear_music_matx_caches();
}

}  // namespace holoscan::radar::detail
