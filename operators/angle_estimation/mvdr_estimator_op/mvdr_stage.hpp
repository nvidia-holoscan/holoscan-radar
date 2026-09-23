// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MVDR (Capon) adaptive-beamforming stage.
//
// Per item i the pipeline is
//   1. R̂_i = (1/K_i) Σ_{s valid} x_{i,s}·x_{i,s}ᴴ        (covariance_kernel)
//   2. λ_i = eig(R̂_i)  (raw covariance, ascending)       (batched matx::eig)
//   3. R̂_i ← R̂_i + δ_i·I                                (load_diagonal_kernel)
//   4. R̂⁻¹_i = inv(R̂_i)                                  (batched matx::inv)
//   5. P_i(θ) = 1 / (a(θ)ᴴ R̂⁻¹_i a(θ)), argmax over θ    (capon_reduce_kernel)
//   6. top-max_sources peaks of P_i(θ)                   (peak_extract_kernel)
// where a(θ) is the untapered steering manifold (num_beams, num_channels),
// beam el-major (beam = ie·num_az + ia). δ_i is either a fixed absolute value
// or, in auto mode, a small multiple of the per-item noise-floor proxy
// trace(R̂_i)/num_channels.
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>

#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// Auto diagonal-loading factor: δ = kAutoLoadingFactor · trace(R̂)/C. Modest
// regularization that guarantees invertibility (including the rank-deficient
// K < C / single-snapshot case) without dominating a well-conditioned R̂.
template <typename T>
inline constexpr T kAutoLoadingFactor = static_cast<T>(1e-2);

// MatX caches inv plans by (A, A_inv) device pointers (eig by shape/stream).
// MVDR's covariance and inverse buffers have stable, operator-owned addresses,
// so plans remain valid across ticks and are retired by the operator's stop().
// Clear() asserts if that cache type was never created; gate with the flags.
inline std::atomic<bool>& matx_eig_plans_live_flag() {
  static std::atomic<bool> live{false};
  return live;
}
inline std::atomic<bool>& matx_inv_plans_live_flag() {
  static std::atomic<bool> live{false};
  return live;
}

inline void mark_matx_solver_caches_live(bool eig_used) {
  if (eig_used) {
    matx_eig_plans_live_flag().store(true, std::memory_order_relaxed);
  }
  matx_inv_plans_live_flag().store(true, std::memory_order_relaxed);
}

inline void clear_matx_eig_cache() {
  matx::detail::GetCache().Clear<matx::detail::eig_cuda_cache_t>(
      matx::detail::GetCacheIdFromType<matx::detail::eig_cuda_cache_t>());
}

inline void clear_matx_inv_cache() {
  matx::detail::GetCache().Clear<matx::detail::inv_cache_t>(
      matx::detail::GetCacheIdFromType<matx::detail::inv_cache_t>());
}

// Teardown: destroy cached eig/inv plans and drain MatX's MemTracker while CUDA
// is still valid. Call once from the operator's stop() (and from Python atexit
// as backup) — never per tick. Safe no-op if solvers were never used.
inline void retire_matx_solver_caches() {
  const bool eig = matx_eig_plans_live_flag().exchange(false, std::memory_order_relaxed);
  const bool inv = matx_inv_plans_live_flag().exchange(false, std::memory_order_relaxed);
  if (!eig && !inv) {
    return;
  }
  cudaDeviceSynchronize();
  if (eig) {
    clear_matx_eig_cache();
  }
  if (inv) {
    clear_matx_inv_cache();
  }
  // Plan Clear can leave small MatX-tracked temps (seen as 64B). ~MemTracker
  // frees those at process exit via cudaFreeAsync — after CuPy has torn down
  // CUDA under pytest → SIGSEGV. Drain while the context is still valid.
  auto& tracker = matx::GetAllocMap();
  while (tracker.size() > 0) {
    tracker.deallocate(tracker.allocationMap.begin()->first);
  }
}

// One block per item. Forms the sample covariance R̂_i (num_channels ×
// num_channels, row-major) over the valid snapshots. `mask` (num_items ×
// num_snapshots, uint8/bool) selects valid slots; nullptr → all valid. Empty
// items (K_i == 0) yield a zero matrix; `k_counts[item]` records K_i so the
// stage can blank Capon / DOA outputs after a safe absolute load + inv.
template <typename T>
__global__ void covariance_kernel(const cuda::std::complex<T>* __restrict__ x,
                                  const unsigned char* __restrict__ mask, int64_t num_items,
                                  int64_t num_channels, int64_t num_snapshots,
                                  cuda::std::complex<T>* __restrict__ cov,
                                  int64_t* __restrict__ k_counts) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }

  __shared__ int64_t k_valid;
  if (threadIdx.x == 0) {
    int64_t count = 0;
    for (int64_t s = 0; s < num_snapshots; ++s) {
      const bool valid = mask == nullptr || mask[item * num_snapshots + s] != 0;
      count += valid ? 1 : 0;
    }
    k_valid = count;
    k_counts[item] = count;
  }
  __syncthreads();

  const int64_t k_count = k_valid;  // valid-snapshot count (covariance divisor)
  const cuda::std::complex<T>* xi = x + item * num_channels * num_snapshots;
  const int64_t entries = num_channels * num_channels;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  for (int64_t e = tid; e < entries; e += nthreads) {
    const int64_t c1 = e / num_channels;
    const int64_t c2 = e % num_channels;
    cuda::std::complex<T> acc{T{0}, T{0}};
    for (int64_t s = 0; s < num_snapshots; ++s) {
      const bool valid = mask == nullptr || mask[item * num_snapshots + s] != 0;
      if (!valid) {
        continue;
      }
      const cuda::std::complex<T> a = xi[c1 * num_snapshots + s];
      const cuda::std::complex<T> b = xi[c2 * num_snapshots + s];
      // a · conj(b)
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

// One block per item. Adds the diagonal load δ_i to R̂_i in place. In auto mode
// (loading < 0) δ_i = kAutoLoadingFactor · trace(R̂_i)/C; otherwise δ_i is the
// fixed absolute `loading`. Block reduces the real diagonal to get the trace.
template <typename T>
__global__ void load_diagonal_kernel(cuda::std::complex<T>* __restrict__ cov, int64_t num_items,
                                     int64_t num_channels, double loading) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }
  const int64_t stride = num_channels * num_channels;
  cuda::std::complex<T>* R = cov + item * stride;

  extern __shared__ __align__(16) unsigned char smem_dl[];
  auto* red = reinterpret_cast<T*>(smem_dl);
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  T partial = T{0};
  for (int64_t c = tid; c < num_channels; c += nthreads) {
    partial += R[c * num_channels + c].real();
  }
  red[tid] = partial;
  __syncthreads();
  for (int stride2 = nthreads / 2; stride2 > 0; stride2 >>= 1) {
    if (tid < stride2) {
      red[tid] += red[tid + stride2];
    }
    __syncthreads();
  }
  const T trace = red[0];

  T delta;
  if (loading < 0.0) {
    delta = kAutoLoadingFactor<T> * trace / static_cast<T>(num_channels);
    // Empty / all-zero covariances have trace == 0 → auto δ would be 0 and
    // leave a singular R̂. Use an absolute floor so the batched inv stays
    // defined; blank_empty_items_kernel then zeros Capon / DOA for K_i == 0.
    if (!(delta > T{0})) {
      delta = kAutoLoadingFactor<T>;
    }
  } else {
    delta = static_cast<T>(loading);
  }
  for (int64_t c = tid; c < num_channels; c += nthreads) {
    cuda::std::complex<T>& d = R[c * num_channels + c];
    d = cuda::std::complex<T>(d.real() + delta, d.imag());
  }
}

// spectrum is stored (num_az, num_el) per item; beams are el-major
// (beam = ie·num_az + ia) matching the steering-manifold build.
template <typename T>
__device__ inline T& mvdr_spectrum_at(T* spec, int64_t ia, int64_t ie, int64_t num_el) {
  return spec[ia * num_el + ie];
}

// One block per item. Threads sweep the beam grid computing the Capon power
// P_b = 1 / (a_bᴴ R̂⁻¹ a_b), writing the (num_az, num_el) spectrum and keeping a
// running argmax (lowest beam index wins ties). Invalid beams have a zero
// manifold → zero power. Dynamic shared: [power T[blockDim]] then
// [index int64_t[blockDim]].
//
// R̂⁻¹ is read from global; because every thread reads the same R̂⁻¹[c,c2] in
// lockstep it broadcasts from L1/L2 and is not the limiter (staging it in
// shared measured slower — occupancy loss outweighs the already-cached reads).
template <typename T>
__global__ void capon_reduce_kernel(const cuda::std::complex<T>* __restrict__ rinv,
                                    const cuda::std::complex<T>* __restrict__ manifold,
                                    int64_t num_items, int64_t num_channels, int64_t num_az,
                                    int64_t num_el, const T* __restrict__ az_grid,
                                    const T* __restrict__ el_grid, T* __restrict__ spectrum,
                                    T* __restrict__ aoa) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }
  const int64_t num_beams = num_az * num_el;
  const int64_t cc = num_channels * num_channels;
  const cuda::std::complex<T>* Ri = rinv + item * cc;
  T* spec = spectrum + item * num_beams;

  extern __shared__ __align__(16) unsigned char smem[];
  auto* red_pow = reinterpret_cast<T*>(smem);
  auto* red_idx = reinterpret_cast<int64_t*>(red_pow + static_cast<int64_t>(blockDim.x));

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  T best_power = T{0};
  int64_t best_beam = -1;
  for (int64_t beam = tid; beam < num_beams; beam += nthreads) {
    const cuda::std::complex<T>* a = manifold + beam * num_channels;
    // denom = aᴴ R̂⁻¹ a = Σ_c conj(a[c]) · (Σ_c2 R̂⁻¹[c,c2]·a[c2]). The inner
    // reduction over c2 is carried in four independent accumulators so the
    // FMAs pipeline instead of stalling on one serial chain (this kernel is
    // latency-bound: warps resident but not issuing; DRAM idle). Real scalar
    // accumulators keep the register allocation tight.
    T denom = T{0};
    for (int64_t c = 0; c < num_channels; ++c) {
      const cuda::std::complex<T>* row = Ri + c * num_channels;
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
      // Re(conj(a[c]) · q)
      denom += a[c].real() * qr + a[c].imag() * qi;
    }
    const T power = denom > T{0} ? T{1} / denom : T{0};
    const int64_t ia = beam % num_az;
    const int64_t ie = beam / num_az;
    mvdr_spectrum_at(spec, ia, ie, num_el) = power;
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
      const bool take =
          other_pow > red_pow[tid] || (other_pow == red_pow[tid] && other_idx >= 0 &&
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

// Strict 2-D local maximum of the Capon spectrum at (ia, ie): power `p` is
// greater than every in-grid 4-neighbor (az±1, el±1). Boundary cells compare
// only against the neighbors that exist, so an edge cell can be a peak; a flat
// plateau (neighbor == p) is not (avoids double-reporting the same lobe).
template <typename T>
__device__ inline bool mvdr_is_local_max(const T* __restrict__ spec, int64_t ia, int64_t ie,
                                         int64_t num_az, int64_t num_el, T p) {
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

// One block per item. Extracts the `max_sources` strongest Capon-spectrum
// peaks into doa_multi (num_items, max_sources, 2), with source_count
// (num_items,) the number actually found (≤ max_sources); unused rows are NaN.
//
// Pass 0 is a *global* argmax (same tie-break as capon_reduce_kernel: highest
// power, lowest beam index) so doa_multi[·,0,·] matches aoa_estimates even on
// flat plateaus. Passes k≥1 require a strict 2-D local maximum (neighbor ≥ p
// rejects plateaus) so a single lobe is not double-counted as multiple sources.
// Cheap relative to the Capon scan — re-reads the resident spectrum. Dynamic
// shared: [power T[blockDim]] [index int64_t[blockDim]]
// [sel int64_t[max_sources]].
template <typename T>
__global__ void peak_extract_kernel(const T* __restrict__ spectrum, int64_t num_items,
                                    int64_t num_az, int64_t num_el, int64_t max_sources,
                                    const T* __restrict__ az_grid, const T* __restrict__ el_grid,
                                    T* __restrict__ doa_multi, int32_t* __restrict__ source_count) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }
  const int64_t num_beams = num_az * num_el;
  const T* spec = spectrum + item * num_beams;

  extern __shared__ __align__(16) unsigned char smem_pk[];
  auto* red_pow = reinterpret_cast<T*>(smem_pk);
  auto* red_idx = reinterpret_cast<int64_t*>(red_pow + static_cast<int64_t>(blockDim.x));
  int64_t* sel = red_idx + static_cast<int64_t>(blockDim.x);

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  int64_t found = 0;
  for (int64_t k = 0; k < max_sources; ++k) {
    T best_power = T{0};
    int64_t best_beam = -1;
    for (int64_t beam = tid; beam < num_beams; beam += nthreads) {
      const int64_t ia = beam % num_az;
      const int64_t ie = beam / num_az;
      const T p = spec[ia * num_el + ie];
      if (p <= T{0}) {
        continue;
      }
      // k == 0: any beam (global argmax). k ≥ 1: strict local maxima only.
      if (k > 0 && !mvdr_is_local_max(spec, ia, ie, num_az, num_el, p)) {
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
      if (p > best_power || (p == best_power && (best_beam < 0 || beam < best_beam))) {
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
        const bool take =
            other_pow > red_pow[tid] || (other_pow == red_pow[tid] && other_idx >= 0 &&
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
      break;  // no more peaks — remaining rows stay unselected (NaN-filled)
    }
    found = k + 1;
  }

  if (tid != 0) {
    return;
  }
  const T nan = T{0} / T{0};
  if (doa_multi != nullptr) {
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
  }
  if (source_count != nullptr) {
    source_count[item] = static_cast<int32_t>(found);
  }
}

// One block per item. When K_i == 0 the covariance was empty: zero the Capon
// spectrum and NaN-fill DOA outputs so callers never see the loaded-identity
// Capon response that the absolute auto-load floor leaves behind for inv.
template <typename T>
__global__ void blank_empty_items_kernel(const int64_t* __restrict__ k_counts, int64_t num_items,
                                         int64_t num_az, int64_t num_el, int64_t max_sources,
                                         T* __restrict__ spectrum, T* __restrict__ aoa,
                                         T* __restrict__ doa_multi,
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
  if (doa_multi != nullptr) {
    for (int64_t k = 0; k < max_sources; ++k) {
      const int64_t base = (item * max_sources + k) * 2;
      doa_multi[base] = nan;
      doa_multi[base + 1] = nan;
    }
  }
  if (source_count != nullptr) {
    source_count[item] = 0;
  }
}

// Run the full MVDR estimate for one tick: eigenvalues (raw covariance) →
// diagonal load → batched inverse → Capon scan → multi-peak extraction. All
// work is enqueued on `stream`; no synchronization. `cov` and `rinv` are
// caller-owned scratch with room for
// (max(num_items, 2), num_channels, num_channels). The extra matrix keeps a
// single public item off MatX's non-repeatable cached single-batch inverse
// backend; eig and every downstream kernel still operate on `num_items`.
// `k_counts` is caller-owned scratch of shape (num_items,) for per-item valid
// snapshot counts (empty items are blanked after Capon).
//
//   snapshots — device ptr, [num_items, num_channels, num_snapshots]
//               cuda::std::complex<T>; a single look uses a trailing extent 1.
//   mask      — device ptr, [num_items, num_snapshots] uint8/bool, or nullptr.
//   manifold  — device ptr, [num_beams, num_channels] cuda::std::complex<T>
//               untapered steering vectors, beam el-major (invalid beams zero).
//   eigenvalues — device ptr, [num_items, num_channels] T. Ascending eigenvalue
//                 spectrum of the RAW (pre-loading) sample covariance.
//   doa_multi   — device ptr, [num_items, max_sources, 2] T. Top-`max_sources`
//                 Capon peaks (strongest first); unused rows NaN.
//   source_count — device ptr, [num_items] int32. Peaks found (≤ max_sources).
//
// Those last three back latched optional ports and may be null, which skips
// the work that produces them: a null `eigenvalues` skips the batched eig, and
// a null `doa_multi` *and* `source_count` skips peak extraction. `spectrum`
// and `aoa` are required.
template <typename T>
inline void run_mvdr_stage(const cuda::std::complex<T>* snapshots, const unsigned char* mask,
                           int64_t num_items, int64_t num_channels, int64_t num_snapshots,
                           double loading, const cuda::std::complex<T>* manifold, int64_t num_az,
                           int64_t num_el, const T* az_grid, const T* el_grid, int64_t max_sources,
                           cuda::std::complex<T>* cov, cuda::std::complex<T>* rinv,
                           int64_t* k_counts, T* spectrum, T* aoa, T* eigenvalues, T* doa_multi,
                           int32_t* source_count, cudaStream_t stream) {
  if (num_items <= 0 || num_channels <= 0) {
    return;
  }

  constexpr int cov_block = 256;
  const int grid = static_cast<int>(num_items);
  covariance_kernel<T><<<grid, cov_block, 0, stream>>>(snapshots, mask, num_items, num_channels,
                                                       num_snapshots, cov, k_counts);

  // The solver plans reference stable operator-owned covariance/inverse
  // buffers. Mark them before first use so stop() retires any plan that MatX
  // creates, including when a later solve step throws.
  mark_matx_solver_caches_live(eigenvalues != nullptr);

  // Eigenvalues of the RAW sample covariance (before diagonal loading) for
  // subspace / source-count diagnostics. matx::eig copies `cov` into the
  // eigenvector output (we hand it the `rinv` scratch, overwritten by the
  // inverse below) and solves in place, so `cov` survives for the load +
  // inverse. NO_VECTOR: eigenvalues only. Batched cuSOLVER heevd.
  //
  // Nothing downstream reads these, so a null `eigenvalues` skips the solve
  // outright — the single largest saving available to a latched port here.
  if (eigenvalues != nullptr) {
    auto cov_v =
        matx::make_tensor<cuda::std::complex<T>, 3>(cov, {num_items, num_channels, num_channels});
    auto evec_scratch =
        matx::make_tensor<cuda::std::complex<T>, 3>(rinv, {num_items, num_channels, num_channels});
    auto eval_v = matx::make_tensor<T, 2>(eigenvalues, {num_items, num_channels});
    matx::cudaExecutor eig_exec{stream};
    (matx::mtie(evec_scratch, eval_v) = matx::eig(cov_v, matx::EigenMode::NO_VECTOR)).run(eig_exec);
  }

  const int dl_block = 256;
  const size_t dl_smem = static_cast<size_t>(dl_block) * sizeof(T);
  load_diagonal_kernel<T>
      <<<grid, dl_block, dl_smem, stream>>>(cov, num_items, num_channels, loading);

  // MatX 1.0's cached batch_size==1 inverse selects a cuSolver getrf/getrs
  // path that is not repeatable across executions. Promote only that inverse
  // to a two-matrix batch by duplicating the fully loaded covariance. MatX
  // then selects its reusable cuBLAS batched path; downstream consumes only
  // the first inverse and retains the public one-item shape.
  const int64_t inverse_items = num_items == 1 ? 2 : num_items;
  if (num_items == 1) {
    const size_t matrix_bytes = static_cast<size_t>(num_channels) *
                                static_cast<size_t>(num_channels) * sizeof(cuda::std::complex<T>);
    (void)cudaMemcpyAsync(cov + num_channels * num_channels, cov, matrix_bytes,
                          cudaMemcpyDeviceToDevice, stream);
  }

  // Batched Hermitian inverse (cuBLAS matinv for C <= 32, else getrf/getri).
  auto cov_view =
      matx::make_tensor<cuda::std::complex<T>, 3>(cov, {inverse_items, num_channels, num_channels});
  auto rinv_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      rinv, {inverse_items, num_channels, num_channels});
  matx::cudaExecutor exec{stream};
  (rinv_view = matx::inv(cov_view)).run(exec);

  constexpr int capon_block = 256;
  const size_t capon_smem = static_cast<size_t>(capon_block) * (sizeof(T) + sizeof(int64_t));
  capon_reduce_kernel<T><<<grid, capon_block, capon_smem, stream>>>(
      rinv, manifold, num_items, num_channels, num_az, num_el, az_grid, el_grid, spectrum, aoa);

  // Multi-source: top-max_sources Capon peaks → doa_multi + source_count.
  // One pass writes both, so it runs unless both are null.
  if (doa_multi != nullptr || source_count != nullptr) {
    constexpr int peak_block = 256;
    const size_t peak_smem = static_cast<size_t>(peak_block) * (sizeof(T) + sizeof(int64_t)) +
                             static_cast<size_t>(max_sources) * sizeof(int64_t);
    peak_extract_kernel<T>
        <<<grid, peak_block, peak_smem, stream>>>(spectrum, num_items, num_az, num_el, max_sources,
                                                  az_grid, el_grid, doa_multi, source_count);
  }

  constexpr int blank_block = 256;
  blank_empty_items_kernel<T><<<grid, blank_block, 0, stream>>>(
      k_counts, num_items, num_az, num_el, max_sources, spectrum, aoa, doa_multi, source_count);
}

}  // namespace holoscan::radar::detail
