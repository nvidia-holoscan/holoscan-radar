// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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

#include "angle_estimation/angle_boundary_device.hpp"
#include "angle_estimation/music_estimator_op/music_stage.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"

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
  constexpr T kTwoPi =
      static_cast<T>(6.28318530717958647692528676655900577);
  T d = fabs(a - b);
  return d > kTwoPi - d ? kTwoPi - d : d;
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
            root_complex_cuda_type<T>(), solver->device_workspace->data(),
            solver->device_workspace_bytes, solver->host_workspace.data(),
            solver->host_workspace_bytes,
            static_cast<int*>(solver->info->data()) + item),
        "cusolverDnXgeev");
  }

  root_music_select_kernel<T><<<grid, 1, 0, stream>>>(
      roots, model_order, valid_counts, num_items, degree, max_sources,
      wave_number_spacing, axis_x, axis_y, axis_z, broadside_x, broadside_y,
      broadside_z, convention, aoa, doa_multi, source_count);

  mark_music_eig_cache_live();
  retire_music_matx_caches();
}

}  // namespace holoscan::radar::detail
