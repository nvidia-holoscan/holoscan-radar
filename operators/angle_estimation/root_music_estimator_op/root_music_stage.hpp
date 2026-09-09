// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Root-MUSIC GPU stage. Includes MatX/cuSolver and must only be included by a
// CUDA translation unit. Polynomial roots come from the companion matrix via
// cuSolverDnXgeev (one matrix per item on the operator stream). A fully
// device-resident custom root solver is tracked in OPEN_ITEMS.md.

#pragma once

#include <cuda/std/complex>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/math/constants.hpp"

#include "angle_estimation/angle_boundary_device.hpp"
#include "angle_estimation/music_estimator_op/music_stage.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::detail {

inline void check_cusolver(cusolverStatus_t status, const char* what) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("RootMUSICEstimatorOp: ") + what +
                             " failed (cuSolver status " +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}

struct RootMUSICSolverState {
  cusolverDnHandle_t handle{nullptr};
  cusolverDnParams_t params{nullptr};
  int64_t degree{-1};
  cudaDataType data_type{CUDA_C_32F};
  size_t device_workspace_bytes{0};
  size_t host_workspace_bytes{0};
  std::shared_ptr<holoscan::Tensor> device_workspace{};
  std::shared_ptr<holoscan::Tensor> info{};
  std::vector<std::byte> host_workspace{};

  RootMUSICSolverState() {
    check_cusolver(cusolverDnCreate(&handle), "cusolverDnCreate");
    check_cusolver(cusolverDnCreateParams(&params), "cusolverDnCreateParams");
  }

  ~RootMUSICSolverState() {
    if (params != nullptr) {
      cusolverDnDestroyParams(params);
    }
    if (handle != nullptr) {
      cusolverDnDestroy(handle);
    }
  }

  RootMUSICSolverState(const RootMUSICSolverState&) = delete;
  RootMUSICSolverState& operator=(const RootMUSICSolverState&) = delete;
};

template <typename T>
inline constexpr cudaDataType root_complex_cuda_type();

template <>
inline constexpr cudaDataType root_complex_cuda_type<float>() {
  return CUDA_C_32F;
}

template <>
inline constexpr cudaDataType root_complex_cuda_type<double>() {
  return CUDA_C_64F;
}

// HSDK 5: Tensor::data() is const void*; writers must use mutable_data().
// Prefer mutable_data() + cast over mutable_data_as<U>() — the latter rejects
// complex DLPack dtypes used by make_device_tensor(complex_dtype<T>()).
template <typename U>
inline U* require_mutable(holoscan::Tensor& tensor, const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    throw std::runtime_error(
        std::string("RootMUSICEstimatorOp: ") + what +
        " is not writable: " + mut.error().message);
  }
  return static_cast<U*>(*mut);
}

inline void* require_mutable_void(holoscan::Tensor& tensor, const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    throw std::runtime_error(
        std::string("RootMUSICEstimatorOp: ") + what +
        " is not writable: " + mut.error().message);
  }
  return *mut;
}

template <typename T>
__global__ void root_music_coefficients_kernel(
    const cuda::std::complex<T>* __restrict__ projector, int64_t num_items,
    int64_t channels, cuda::std::complex<T>* __restrict__ coefficients) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }
  const int64_t degree = 2 * (channels - 1);
  const int64_t coefficient_count = degree + 1;
  for (int64_t ci = threadIdx.x; ci < coefficient_count; ci += blockDim.x) {
    const int64_t lag = ci - (channels - 1);
    cuda::std::complex<T> sum{T{0}, T{0}};
    const int64_t first = lag < 0 ? -lag : 0;
    const int64_t last = lag > 0 ? channels - lag : channels;
    const auto* p = projector + item * channels * channels;
    for (int64_t row = first; row < last; ++row) {
      const int64_t col = row + lag;
      sum += p[row * channels + col];
    }
    coefficients[item * coefficient_count + ci] = sum;
  }
}

template <typename T>
__global__ void root_music_companion_kernel(
    const cuda::std::complex<T>* __restrict__ coefficients,
    const int64_t* __restrict__ valid_counts, int64_t num_items, int64_t degree,
    cuda::std::complex<T>* __restrict__ companion) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  const int64_t entries = degree * degree;
  auto* matrix = companion + item * entries;
  const auto* coeff = coefficients + item * (degree + 1);
  const cuda::std::complex<T> leading = coeff[degree];
  const T leading_norm =
      leading.real() * leading.real() + leading.imag() * leading.imag();
  const T floor = std::is_same_v<T, float> ? static_cast<T>(1e-20)
                                           : static_cast<T>(1e-40);

  for (int64_t index = threadIdx.x; index < entries; index += blockDim.x) {
    matrix[index] = cuda::std::complex<T>{T{0}, T{0}};
  }
  __syncthreads();

  // Keep the matrix finite so one empty item cannot make the per-item geev
  // call fail before the selection kernel blanks its outputs.
  if (valid_counts[item] <= 0) {
    return;
  }

  if (!(leading_norm > floor)) {
    if (threadIdx.x == 0) {
      const T nan = T{0} / T{0};
      for (int64_t index = 0; index < entries; ++index) {
        matrix[index] = cuda::std::complex<T>{nan, nan};
      }
    }
    return;
  }

  for (int64_t col = threadIdx.x; col < degree - 1; col += blockDim.x) {
    matrix[(col + 1) + col * degree] = cuda::std::complex<T>{T{1}, T{0}};
  }
  for (int64_t row = threadIdx.x; row < degree; row += blockDim.x) {
    matrix[row + (degree - 1) * degree] = -coeff[row] / leading;
  }
}

template <typename T>
__device__ inline T wrapped_phase_distance(T a, T b) {
  const T two_pi = static_cast<T>(kTwoPi);
  T d = fabs(a - b);
  return d > two_pi - d ? two_pi - d : d;
}

template <typename T>
__global__ void root_music_select_kernel(
    const cuda::std::complex<T>* __restrict__ roots,
    const int32_t* __restrict__ model_order,
    const int64_t* __restrict__ valid_counts, int64_t num_items,
    int64_t degree, int64_t max_sources, T wave_number_spacing,
    T axis_x, T axis_y, T axis_z, T broadside_x, T broadside_y,
    T broadside_z, DeviceAngleConvention<T> convention, T* __restrict__ aoa,
    T* __restrict__ doa_multi, int32_t* __restrict__ source_count) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || item >= num_items) {
    return;
  }

  const T nan = T{0} / T{0};
  aoa[item * 2] = nan;
  aoa[item * 2 + 1] = nan;
  for (int64_t k = 0; k < max_sources; ++k) {
    doa_multi[(item * max_sources + k) * 2] = nan;
    doa_multi[(item * max_sources + k) * 2 + 1] = nan;
  }
  source_count[item] = 0;
  if (valid_counts[item] <= 0) {
    return;
  }

  int64_t target = static_cast<int64_t>(model_order[item]);
  target = target < 1 ? 1 : target;
  target = target > max_sources ? max_sources : target;
  const auto* item_roots = roots + item * degree;

  int64_t found = 0;
  int64_t selected_indices[64];
  for (int64_t out = 0; out < target; ++out) {
    int64_t best = -1;
    T best_score = static_cast<T>(1e30);
    T best_phase = T{0};
    for (int64_t r = 0; r < degree; ++r) {
      const T re = item_roots[r].real();
      const T im = item_roots[r].imag();
      const T magnitude = hypot(re, im);
      if (!isfinite(magnitude) || magnitude > T{1}) {
        continue;
      }
      const T phase = atan2(im, re);
      bool duplicate = false;
      for (int64_t prior = 0; prior < found; ++prior) {
        const auto selected = item_roots[selected_indices[prior]];
        if (wrapped_phase_distance(phase,
                                   atan2(selected.imag(), selected.real())) <
            static_cast<T>(1e-4)) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        continue;
      }
      const T score = fabs(T{1} - magnitude);
      if (score < best_score) {
        best_score = score;
        best = r;
        best_phase = phase;
      }
    }
    if (best < 0) {
      break;
    }

    selected_indices[found] = best;
    T spatial_projection = best_phase / wave_number_spacing;
    spatial_projection =
        spatial_projection < T{-1} ? T{-1} : spatial_projection;
    spatial_projection =
        spatial_projection > T{1} ? T{1} : spatial_projection;
    const T transverse =
        sqrt(fmax(T{0}, T{1} - spatial_projection * spatial_projection));
    const T x = spatial_projection * axis_x + transverse * broadside_x;
    const T y = spatial_projection * axis_y + transverse * broadside_y;
    const T z = spatial_projection * axis_z + transverse * broadside_z;
    T out0 = nan;
    T out1 = nan;
    angle_canonical_to_boundary(convention, x, y, z, &out0, &out1);
    const int64_t base = (item * max_sources + found) * 2;
    doa_multi[base] = out0;
    doa_multi[base + 1] = out1;
    if (found == 0) {
      aoa[item * 2] = out0;
      aoa[item * 2 + 1] = out1;
    }
    ++found;
  }
  source_count[item] = static_cast<int32_t>(found);
}

template <typename T>
inline void configure_root_solver(
    RootMUSICSolverState* state, int64_t degree,
    cuda::std::complex<T>* companion, cuda::std::complex<T>* roots,
    int64_t num_items) {
  const cudaDataType type = root_complex_cuda_type<T>();
  if (state->degree == degree && state->data_type == type && state->info &&
      state->info->shape()[0] >= num_items) {
    return;
  }

  size_t device_bytes = 0;
  size_t host_bytes = 0;
  check_cusolver(
      cusolverDnXgeev_bufferSize(
          state->handle, state->params, CUSOLVER_EIG_MODE_NOVECTOR,
          CUSOLVER_EIG_MODE_NOVECTOR, degree, type, companion, degree, type,
          roots, type, nullptr, 1, type, nullptr, 1, type, &device_bytes,
          &host_bytes),
      "cusolverDnXgeev_bufferSize");
  state->device_workspace = make_device_tensor(
      {static_cast<int64_t>(std::max<size_t>(device_bytes, size_t{1}))},
      DLDataType{kDLUInt, 8, 1});
  state->info = make_device_tensor({std::max<int64_t>(num_items, 1)},
                                   DLDataType{kDLInt, 32, 1});
  state->host_workspace.resize(std::max<size_t>(host_bytes, size_t{1}));
  state->degree = degree;
  state->data_type = type;
  state->device_workspace_bytes = device_bytes;
  state->host_workspace_bytes = host_bytes;
}

// One companion-matrix eigen-solve per item on the operator stream (see the
// serial-hybrid note in OPEN_ITEMS.md). Configures/reuses the cuSolver
// workspace, runs cusolverDnXgeev per item, then downloads and checks the
// per-item info codes (synchronizes `stream`). A per-item non-convergence
// (info > 0, e.g. a degenerate NaN companion matrix) blanks that item's roots
// so the selection kernels skip it; only an illegal argument (info < 0) throws.
template <typename T>
inline void solve_companion_roots(RootMUSICSolverState* solver, int64_t degree,
                                  cuda::std::complex<T>* companion,
                                  cuda::std::complex<T>* roots,
                                  int64_t num_items, cudaStream_t stream) {
  if (num_items <= 0 || degree <= 0) {
    return;
  }
  configure_root_solver<T>(solver, degree, companion, roots, num_items);
  check_cusolver(cusolverDnSetStream(solver->handle, stream),
                 "cusolverDnSetStream");
  for (int64_t item = 0; item < num_items; ++item) {
    check_cusolver(
        cusolverDnXgeev(
            solver->handle, solver->params, CUSOLVER_EIG_MODE_NOVECTOR,
            CUSOLVER_EIG_MODE_NOVECTOR, degree, root_complex_cuda_type<T>(),
            companion + item * degree * degree, degree,
            root_complex_cuda_type<T>(), roots + item * degree,
            root_complex_cuda_type<T>(), nullptr, 1,
            root_complex_cuda_type<T>(), nullptr, 1,
            root_complex_cuda_type<T>(),
            require_mutable_void(*solver->device_workspace,
                                 "geev device workspace"),
            solver->device_workspace_bytes, solver->host_workspace.data(),
            solver->host_workspace_bytes,
            require_mutable<int>(*solver->info, "geev info") + item),
        "cusolverDnXgeev");
  }
  std::vector<int> info_host(
      static_cast<size_t>(std::max(num_items, int64_t{1})));
  if (cudaMemcpyAsync(info_host.data(), solver->info->data(),
                      static_cast<size_t>(num_items) * sizeof(int),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error(
        "RootMUSICEstimatorOp: cusolverDnXgeev info download failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error(
        "RootMUSICEstimatorOp: cusolverDnXgeev info sync failed");
  }
  for (int64_t item = 0; item < num_items; ++item) {
    const int info = info_host[static_cast<size_t>(item)];
    if (info < 0) {
      // info < 0: the |info|-th argument to geev was illegal -- a programming
      // error in how we drive the solver, not data-dependent. Fail loudly.
      throw std::runtime_error(
          "RootMUSICEstimatorOp: cusolverDnXgeev illegal argument for item " +
          std::to_string(item) + " (info " + std::to_string(info) + ")");
    }
    if (info > 0) {
      // info > 0: QR failed to converge for this item -- the expected outcome
      // for a numerically degenerate companion matrix (e.g. root_music_
      // companion_kernel deliberately writes a NaN matrix when the leading
      // coefficient underflows). Blank just this item's roots with an
      // all-ones bit pattern (a NaN for both float and double); the selection
      // kernels skip non-finite roots and emit zero sources for it, so one bad
      // item no longer aborts the whole tick.
      if (cudaMemsetAsync(roots + item * degree, 0xFF,
                          static_cast<size_t>(degree) *
                              sizeof(cuda::std::complex<T>),
                          stream) != cudaSuccess) {
        throw std::runtime_error(
            "RootMUSICEstimatorOp: failed to blank degenerate roots for item " +
            std::to_string(item));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2-D URA path: reduced-dimension per-axis rooting + cross-axis pairing.
//
// A filled URA (element order (iz outer, iy inner)) is separable: the array
// manifold is a Kronecker product of a y-axis (azimuth) and a z-axis
// (elevation) Vandermonde. Row/column subarray-averaging the full covariance
// yields an Ny x Ny azimuth covariance and an Nz x Nz elevation covariance;
// the existing 1-D rooting machinery on each yields candidate direction
// cosines, paired by the full 2-D MUSIC subspace metric. See OPERATOR_API.md
// and OPEN_ITEMS.md for the method rationale (no canonical 2-D Root-MUSIC).
// ---------------------------------------------------------------------------

// The in-kernel pairing stages the full URA steering vector and the candidate
// bookkeeping in per-thread local arrays, so the URA channel count and the
// per-axis candidate count are bounded (validated on the host).
inline constexpr int64_t kRootMusicPairChannelCap = 64;
inline constexpr int64_t kRootMusicPairCandCap = 8;

template <typename T>
struct RootMUSICAxisWorkspace {
  cuda::std::complex<T>* eigenvectors{nullptr};
  T* eigenvalues{nullptr};
  cuda::std::complex<T>* projector{nullptr};
  int32_t* model_order{nullptr};
  cuda::std::complex<T>* coefficients{nullptr};
  cuda::std::complex<T>* companion{nullptr};
  cuda::std::complex<T>* roots{nullptr};
};

template <typename T>
struct RootMUSICURAScratch {
  cuda::std::complex<T>* cov_y{nullptr};
  cuda::std::complex<T>* cov_z{nullptr};
  RootMUSICAxisWorkspace<T> axis_y;
  RootMUSICAxisWorkspace<T> axis_z;
  T* u_candidates{nullptr};
  T* v_candidates{nullptr};
  int32_t* u_counts{nullptr};
  int32_t* v_counts{nullptr};
};

// One block per item. Row/column subarray-averages the full covariance R
// (channels = ny*nz, row-major) into an ny x ny azimuth covariance and an
// nz x nz elevation covariance (forward spatial averaging over the orthogonal
// aperture axis).
template <typename T>
__global__ void ura_axis_covariance_kernel(
    const cuda::std::complex<T>* __restrict__ cov, int64_t num_items,
    int64_t ny, int64_t nz, cuda::std::complex<T>* __restrict__ cov_y,
    cuda::std::complex<T>* __restrict__ cov_z) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }
  const int64_t channels = ny * nz;
  const auto* R = cov + item * channels * channels;
  const T inv_nz = T{1} / static_cast<T>(nz);
  const T inv_ny = T{1} / static_cast<T>(ny);
  for (int64_t e = threadIdx.x; e < ny * ny; e += blockDim.x) {
    const int64_t a = e / ny;
    const int64_t b = e % ny;
    cuda::std::complex<T> acc{T{0}, T{0}};
    for (int64_t iz = 0; iz < nz; ++iz) {
      acc += R[(iz * ny + a) * channels + (iz * ny + b)];
    }
    cov_y[item * ny * ny + e] =
        cuda::std::complex<T>{acc.real() * inv_nz, acc.imag() * inv_nz};
  }
  for (int64_t e = threadIdx.x; e < nz * nz; e += blockDim.x) {
    const int64_t a = e / nz;
    const int64_t b = e % nz;
    cuda::std::complex<T> acc{T{0}, T{0}};
    for (int64_t iy = 0; iy < ny; ++iy) {
      acc += R[(a * ny + iy) * channels + (b * ny + iy)];
    }
    cov_z[item * nz * nz + e] =
        cuda::std::complex<T>{acc.real() * inv_ny, acc.imag() * inv_ny};
  }
}

// Clamp a per-item model order into [1, cap] (a URA source count cannot exceed
// a single aperture axis's noise-subspace budget when rooting that axis).
__global__ inline void clamp_model_order_kernel(const int32_t* __restrict__ in,
                                                int32_t* __restrict__ out,
                                                int64_t num_items,
                                                int32_t cap) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int32_t value = in[i];
  value = value < 1 ? 1 : value;
  value = value > cap ? cap : value;
  out[i] = value;
}

// One block per item (thread 0). Like root_music_select_kernel, but instead of
// mapping to a boundary angle it emits up to `cand_cap` candidate direction
// cosines (root phase / wave-number spacing) for one aperture axis, closest to
// the unit circle first, with the same phase de-duplication.
template <typename T>
__global__ void root_music_axis_candidates_kernel(
    const cuda::std::complex<T>* __restrict__ roots,
    const int32_t* __restrict__ axis_model_order,
    const int64_t* __restrict__ valid_counts, int64_t num_items,
    int64_t degree, int64_t cand_cap, T wave_number_spacing,
    T* __restrict__ cand, int32_t* __restrict__ cand_count) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || item >= num_items) {
    return;
  }
  for (int64_t j = 0; j < cand_cap; ++j) {
    cand[item * cand_cap + j] = T{0};
  }
  cand_count[item] = 0;
  if (valid_counts[item] <= 0) {
    return;
  }

  int64_t target = static_cast<int64_t>(axis_model_order[item]);
  target = target < 1 ? 1 : target;
  target = target > cand_cap ? cand_cap : target;
  const auto* item_roots = roots + item * degree;

  int64_t found = 0;
  int64_t selected_indices[64];
  for (int64_t out = 0; out < target; ++out) {
    int64_t best = -1;
    T best_score = static_cast<T>(1e30);
    T best_phase = T{0};
    for (int64_t r = 0; r < degree; ++r) {
      const T re = item_roots[r].real();
      const T im = item_roots[r].imag();
      const T magnitude = hypot(re, im);
      if (!isfinite(magnitude) || magnitude > T{1}) {
        continue;
      }
      const T phase = atan2(im, re);
      bool duplicate = false;
      for (int64_t prior = 0; prior < found; ++prior) {
        const auto selected = item_roots[selected_indices[prior]];
        if (wrapped_phase_distance(
                phase, atan2(selected.imag(), selected.real())) <
            static_cast<T>(1e-4)) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        continue;
      }
      const T score = fabs(T{1} - magnitude);
      if (score < best_score) {
        best_score = score;
        best = r;
        best_phase = phase;
      }
    }
    if (best < 0) {
      break;
    }
    selected_indices[found] = best;
    T projection = best_phase / wave_number_spacing;
    projection = projection < T{-1} ? T{-1} : projection;
    projection = projection > T{1} ? T{1} : projection;
    cand[item * cand_cap + found] = projection;
    ++found;
  }
  cand_count[item] = static_cast<int32_t>(found);
}

// One block per item (thread 0). Pairs the per-axis candidate cosines by
// evaluating the full 2-D MUSIC subspace metric 1/(a^H P_n a) at each
// (u_i, v_k) and greedily assigning the `target` best mutually-exclusive
// pairs; each selected (u, v) is mapped through the aperture-axis frame to a
// canonical direction and then to boundary angles.
template <typename T>
__global__ void root_music_pair_kernel(
    const T* __restrict__ u_cand, const int32_t* __restrict__ u_count,
    const T* __restrict__ v_cand, const int32_t* __restrict__ v_count,
    const cuda::std::complex<T>* __restrict__ projector,
    const int32_t* __restrict__ model_order,
    const int64_t* __restrict__ valid_counts, int64_t num_items, int64_t ny,
    int64_t nz, int64_t u_cap, int64_t v_cap, T ky_spacing, T kz_spacing,
    T yax_x, T yax_y, T yax_z, T zax_x, T zax_y, T zax_z, T nrm_x, T nrm_y,
    T nrm_z, DeviceAngleConvention<T> convention, int64_t max_sources,
    T* __restrict__ aoa, T* __restrict__ doa_multi,
    int32_t* __restrict__ source_count) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || item >= num_items) {
    return;
  }
  const int64_t channels = ny * nz;
  const T nan = T{0} / T{0};
  aoa[item * 2] = nan;
  aoa[item * 2 + 1] = nan;
  for (int64_t k = 0; k < max_sources; ++k) {
    doa_multi[(item * max_sources + k) * 2] = nan;
    doa_multi[(item * max_sources + k) * 2 + 1] = nan;
  }
  source_count[item] = 0;
  if (valid_counts[item] <= 0) {
    return;
  }

  int64_t nu = static_cast<int64_t>(u_count[item]);
  int64_t nv = static_cast<int64_t>(v_count[item]);
  nu = nu > u_cap ? u_cap : nu;
  nv = nv > v_cap ? v_cap : nv;
  int64_t target = static_cast<int64_t>(model_order[item]);
  target = target < 1 ? 1 : target;
  target = target > max_sources ? max_sources : target;
  target = target > nu ? nu : target;
  target = target > nv ? nv : target;
  if (target < 1) {
    return;
  }

  const auto* Pn = projector + item * channels * channels;
  T metric[kRootMusicPairCandCap * kRootMusicPairCandCap];
  cuda::std::complex<T> a[kRootMusicPairChannelCap];
  const T floor = kMusicDenomFloor<T>;
  for (int64_t i = 0; i < nu; ++i) {
    const T u = u_cand[item * u_cap + i];
    for (int64_t k = 0; k < nv; ++k) {
      const T v = v_cand[item * v_cap + k];
      for (int64_t c = 0; c < channels; ++c) {
        const int64_t iy = c % ny;
        const int64_t iz = c / ny;
        const T phase = static_cast<T>(iy) * ky_spacing * u +
                        static_cast<T>(iz) * kz_spacing * v;
        a[c] = cuda::std::complex<T>{cos(phase), sin(phase)};
      }
      T denom = T{0};
      for (int64_t c = 0; c < channels; ++c) {
        const auto* row = Pn + c * channels;
        cuda::std::complex<T> q{T{0}, T{0}};
        for (int64_t c2 = 0; c2 < channels; ++c2) {
          q += row[c2] * a[c2];
        }
        denom += a[c].real() * q.real() + a[c].imag() * q.imag();
      }
      metric[i * nv + k] = T{1} / (denom > floor ? denom : floor);
    }
  }

  bool u_used[kRootMusicPairCandCap];
  bool v_used[kRootMusicPairCandCap];
  for (int64_t i = 0; i < nu; ++i) {
    u_used[i] = false;
  }
  for (int64_t k = 0; k < nv; ++k) {
    v_used[k] = false;
  }

  int64_t found = 0;
  for (int64_t out = 0; out < target; ++out) {
    T best = static_cast<T>(-1);
    int64_t best_i = -1;
    int64_t best_k = -1;
    for (int64_t i = 0; i < nu; ++i) {
      if (u_used[i]) {
        continue;
      }
      for (int64_t k = 0; k < nv; ++k) {
        if (v_used[k]) {
          continue;
        }
        const T m = metric[i * nv + k];
        if (m > best) {
          best = m;
          best_i = i;
          best_k = k;
        }
      }
    }
    if (best_i < 0) {
      break;
    }
    u_used[best_i] = true;
    v_used[best_k] = true;
    const T u = u_cand[item * u_cap + best_i];
    const T v = v_cand[item * v_cap + best_k];
    const T w = sqrt(fmax(T{0}, T{1} - u * u - v * v));
    const T x = u * yax_x + v * zax_x + w * nrm_x;
    const T y = u * yax_y + v * zax_y + w * nrm_y;
    const T z = u * yax_z + v * zax_z + w * nrm_z;
    T out0 = nan;
    T out1 = nan;
    angle_canonical_to_boundary(convention, x, y, z, &out0, &out1);
    const int64_t base = (item * max_sources + found) * 2;
    doa_multi[base] = out0;
    doa_multi[base + 1] = out1;
    if (found == 0) {
      aoa[item * 2] = out0;
      aoa[item * 2 + 1] = out1;
    }
    ++found;
  }
  source_count[item] = static_cast<int32_t>(found);
}

// Root one aperture axis: eig its subarray-averaged covariance, build the
// noise-subspace projector at the per-axis-clamped model order, form the
// Root-MUSIC polynomial, solve for its roots, and emit candidate cosines.
template <typename T>
inline void root_music_axis_roots(cuda::std::complex<T>* cov_axis,
                                  int64_t num_items, int64_t axis_channels,
                                  const int64_t* valid_counts,
                                  const int32_t* model_order_full,
                                  int64_t cand_cap, T wave_number_spacing,
                                  const RootMUSICAxisWorkspace<T>& workspace,
                                  RootMUSICSolverState* solver, T* cand,
                                  int32_t* cand_count, cudaStream_t stream) {
  if (num_items <= 0 || axis_channels < 2) {
    return;
  }
  constexpr int block = 256;
  const int grid = static_cast<int>(num_items);
  const int64_t degree = 2 * (axis_channels - 1);

  auto cov_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      cov_axis, {num_items, axis_channels, axis_channels});
  auto evec_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      workspace.eigenvectors, {num_items, axis_channels, axis_channels});
  auto eval_view = matx::make_tensor<T, 2>(
      workspace.eigenvalues, {num_items, axis_channels});
  matx::cudaExecutor eig_exec{stream};
  (matx::mtie(evec_view, eval_view) =
       matx::eig(cov_view, matx::EigenMode::VECTOR))
      .run(eig_exec);

  constexpr int clamp_block = 128;
  const int clamp_grid =
      static_cast<int>((num_items + clamp_block - 1) / clamp_block);
  clamp_model_order_kernel<<<clamp_grid, clamp_block, 0, stream>>>(
      model_order_full, workspace.model_order, num_items,
      static_cast<int32_t>(axis_channels - 1));
  projector_kernel<T><<<grid, block, 0, stream>>>(
      workspace.eigenvectors, workspace.model_order, num_items,
      axis_channels, workspace.projector);
  root_music_coefficients_kernel<T><<<grid, block, 0, stream>>>(
      workspace.projector, num_items, axis_channels, workspace.coefficients);
  root_music_companion_kernel<T><<<grid, block, 0, stream>>>(
      workspace.coefficients, valid_counts, num_items, degree,
      workspace.companion);
  solve_companion_roots<T>(solver, degree, workspace.companion,
                           workspace.roots, num_items, stream);
  root_music_axis_candidates_kernel<T><<<grid, 1, 0, stream>>>(
      workspace.roots, workspace.model_order, valid_counts, num_items, degree,
      cand_cap, wave_number_spacing, cand, cand_count);
}

// Full 2-D URA Root-MUSIC for one tick. `covariance`, `eigenvectors`,
// `projector`, `eigenvalues`, `valid_counts`, `model_order`, `aoa`,
// `doa_multi`, and `source_count` are operator-owned scratch sized for the
// full ny*nz aperture (same shapes the 1-D stage uses). The remaining
// per-axis and pairing scratch is operator-owned and passed explicitly.
template <typename T>
inline void run_root_music_2d_stage(
    const cuda::std::complex<T>* snapshots, const unsigned char* mask,
    int64_t num_items, int64_t ny, int64_t nz, int64_t num_snapshots,
    int64_t fixed_num_sources, double pfa, int64_t max_sources, T ky_spacing,
    T kz_spacing, T yax_x, T yax_y, T yax_z, T zax_x, T zax_y, T zax_z,
    T nrm_x, T nrm_y, T nrm_z, DeviceAngleConvention<T> convention,
    cuda::std::complex<T>* covariance, cuda::std::complex<T>* eigenvectors,
    cuda::std::complex<T>* projector, int64_t* valid_counts,
    int32_t* model_order, T* eigenvalues, T* aoa, T* doa_multi,
    int32_t* source_count, const RootMUSICURAScratch<T>& scratch,
    RootMUSICSolverState* solver_y,
    RootMUSICSolverState* solver_z, cudaStream_t stream) {
  const int64_t channels = ny * nz;
  if (num_items <= 0 || channels < 4) {
    return;
  }
  constexpr int block = 256;
  const int grid = static_cast<int>(num_items);
  music_covariance_kernel<T><<<grid, block, 0, stream>>>(
      snapshots, mask, num_items, channels, num_snapshots, covariance,
      valid_counts);

  auto covariance_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      covariance, {num_items, channels, channels});
  auto eigenvector_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      eigenvectors, {num_items, channels, channels});
  auto eigenvalue_view =
      matx::make_tensor<T, 2>(eigenvalues, {num_items, channels});
  matx::cudaExecutor eig_exec{stream};
  (matx::mtie(eigenvector_view, eigenvalue_view) =
       matx::eig(covariance_view, matx::EigenMode::VECTOR))
      .run(eig_exec);

  if (fixed_num_sources < 0) {
    run_source_count_device<T>(eigenvalues, valid_counts, num_items, channels,
                               pfa, max_sources, model_order, stream);
  } else {
    constexpr int fill_block = 128;
    const int fill_grid =
        static_cast<int>((num_items + fill_block - 1) / fill_block);
    fill_model_order_kernel<<<fill_grid, fill_block, 0, stream>>>(
        model_order, num_items, static_cast<int32_t>(fixed_num_sources));
  }
  projector_kernel<T><<<grid, block, 0, stream>>>(
      eigenvectors, model_order, num_items, channels, projector);

  ura_axis_covariance_kernel<T><<<grid, block, 0, stream>>>(
      covariance, num_items, ny, nz, scratch.cov_y, scratch.cov_z);

  const int64_t u_cap = std::min<int64_t>(ny - 1, kRootMusicPairCandCap);
  const int64_t v_cap = std::min<int64_t>(nz - 1, kRootMusicPairCandCap);

  root_music_axis_roots<T>(
      scratch.cov_y, num_items, ny, valid_counts, model_order, u_cap,
      ky_spacing, scratch.axis_y, solver_y, scratch.u_candidates,
      scratch.u_counts, stream);
  root_music_axis_roots<T>(
      scratch.cov_z, num_items, nz, valid_counts, model_order, v_cap,
      kz_spacing, scratch.axis_z, solver_z, scratch.v_candidates,
      scratch.v_counts, stream);

  root_music_pair_kernel<T><<<grid, 1, 0, stream>>>(
      scratch.u_candidates, scratch.u_counts, scratch.v_candidates,
      scratch.v_counts, projector, model_order, valid_counts, num_items, ny,
      nz, u_cap, v_cap, ky_spacing, kz_spacing, yax_x, yax_y, yax_z, zax_x,
      zax_y, zax_z, nrm_x, nrm_y, nrm_z, convention, max_sources, aoa,
      doa_multi, source_count);

  mark_music_eig_cache_live();
  clear_music_matx_caches();
}

template <typename T>
inline void run_root_music_stage(
    const cuda::std::complex<T>* snapshots, const unsigned char* mask,
    int64_t num_items, int64_t num_channels, int64_t num_snapshots,
    int64_t fixed_num_sources, double pfa, int64_t max_sources,
    T wave_number_spacing, T axis_x, T axis_y, T axis_z, T broadside_x,
    T broadside_y, T broadside_z, DeviceAngleConvention<T> convention,
    cuda::std::complex<T>* covariance, cuda::std::complex<T>* eigenvectors,
    cuda::std::complex<T>* projector, int64_t* valid_counts,
    int32_t* model_order, T* eigenvalues,
    cuda::std::complex<T>* coefficients,
    cuda::std::complex<T>* companion, cuda::std::complex<T>* roots, T* aoa,
    T* doa_multi, int32_t* source_count, RootMUSICSolverState* solver,
    cudaStream_t stream) {
  if (num_items <= 0 || num_channels < 2) {
    return;
  }
  constexpr int block = 256;
  const int grid = static_cast<int>(num_items);
  music_covariance_kernel<T><<<grid, block, 0, stream>>>(
      snapshots, mask, num_items, num_channels, num_snapshots, covariance,
      valid_counts);

  auto covariance_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      covariance, {num_items, num_channels, num_channels});
  auto eigenvector_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      eigenvectors, {num_items, num_channels, num_channels});
  auto eigenvalue_view =
      matx::make_tensor<T, 2>(eigenvalues, {num_items, num_channels});
  matx::cudaExecutor eig_exec{stream};
  (matx::mtie(eigenvector_view, eigenvalue_view) =
       matx::eig(covariance_view, matx::EigenMode::VECTOR))
      .run(eig_exec);

  if (fixed_num_sources < 0) {
    run_source_count_device<T>(eigenvalues, valid_counts, num_items,
                               num_channels, pfa, max_sources, model_order,
                               stream);
  } else {
    constexpr int fill_block = 128;
    const int fill_grid =
        static_cast<int>((num_items + fill_block - 1) / fill_block);
    fill_model_order_kernel<<<fill_grid, fill_block, 0, stream>>>(
        model_order, num_items, static_cast<int32_t>(fixed_num_sources));
  }
  projector_kernel<T><<<grid, block, 0, stream>>>(
      eigenvectors, model_order, num_items, num_channels, projector);

  const int64_t degree = 2 * (num_channels - 1);
  root_music_coefficients_kernel<T><<<grid, block, 0, stream>>>(
      projector, num_items, num_channels, coefficients);
  root_music_companion_kernel<T><<<grid, block, 0, stream>>>(
      coefficients, valid_counts, num_items, degree, companion);

  solve_companion_roots<T>(solver, degree, companion, roots, num_items,
                           stream);

  root_music_select_kernel<T><<<grid, 1, 0, stream>>>(
      roots, model_order, valid_counts, num_items, degree, max_sources,
      wave_number_spacing, axis_x, axis_y, axis_z, broadside_x, broadside_y,
      broadside_z, convention, aoa, doa_multi, source_count);

  mark_music_eig_cache_live();
  clear_music_matx_caches();
}

}  // namespace holoscan::radar::detail
