// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ESPRIT GPU stage. The covariance/eigenspace/EFT front-end is shared with
// MUSIC. One block per item then forms the source-space least-squares rotation
// and solves its small general complex eigenproblem in-device.

#pragma once

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "angle_estimation/music_estimator_op/music_stage.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/source_count_device.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

template <typename T>
__device__ inline T esprit_norm_squared(cuda::std::complex<T> value) {
  return value.real() * value.real() + value.imag() * value.imag();
}

template <typename T>
__device__ inline cuda::std::complex<T> esprit_conj(
    cuda::std::complex<T> value) {
  return {value.real(), -value.imag()};
}

template <typename T>
__device__ inline void esprit_canonical_to_boundary(
    AngleConventionPreset preset, T x, T y, T z_coord, T* out0, T* out1) {
  if (preset == AngleConventionPreset::kDirectionCosine) {
    *out0 = y;
    *out1 = z_coord;
    return;
  }
  T az = atan2(y, x);
  T el = atan2(z_coord, hypot(x, y));
  if (preset == AngleConventionPreset::kAerospaceNed ||
      preset == AngleConventionPreset::kGeodeticNed) {
    az = -az;
    el = -el;
  } else if (preset == AngleConventionPreset::kGeodeticEnu) {
    az = atan2(x, -y);
  } else if (preset == AngleConventionPreset::kPhysicsSpherical) {
    constexpr T kHalfPi =
        static_cast<T>(1.57079632679489661923132169163975144);
    el = kHalfPi - el;
  }
  *out0 = az;
  *out1 = el;
}

template <typename T>
__device__ bool esprit_qr_eigenvalues(cuda::std::complex<T>* matrix,
                                      cuda::std::complex<T>* q,
                                      cuda::std::complex<T>* r, int64_t d) {
  if (d <= 1) {
    return true;
  }
  const T floor = std::is_same_v<T, float> ? static_cast<T>(1e-12)
                                           : static_cast<T>(1e-24);
  const T tolerance = std::is_same_v<T, float> ? static_cast<T>(2e-5)
                                               : static_cast<T>(1e-11);
  constexpr int kMaxIterations = 192;

  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    cuda::std::complex<T> shift = matrix[(d - 1) * d + (d - 1)];
    if ((iteration + 1) % 32 == 0) {
      const T perturb = tolerance * static_cast<T>(iteration + 1);
      shift += cuda::std::complex<T>{perturb, perturb};
    }
    for (int64_t index = 0; index < d * d; ++index) {
      q[index] = {T{0}, T{0}};
      r[index] = {T{0}, T{0}};
    }

    for (int64_t col = 0; col < d; ++col) {
      for (int64_t row = 0; row < d; ++row) {
        q[row * d + col] =
            matrix[row * d + col] -
            (row == col ? shift : cuda::std::complex<T>{T{0}, T{0}});
      }
      for (int64_t prior = 0; prior < col; ++prior) {
        cuda::std::complex<T> projection{T{0}, T{0}};
        for (int64_t row = 0; row < d; ++row) {
          projection += esprit_conj(q[row * d + prior]) *
                        q[row * d + col];
        }
        r[prior * d + col] = projection;
        for (int64_t row = 0; row < d; ++row) {
          q[row * d + col] -= q[row * d + prior] * projection;
        }
      }

      T column_norm_squared = T{0};
      for (int64_t row = 0; row < d; ++row) {
        column_norm_squared += esprit_norm_squared(q[row * d + col]);
      }
      if (!(column_norm_squared > floor)) {
        bool completed = false;
        for (int64_t candidate = 0; candidate < d && !completed; ++candidate) {
          for (int64_t row = 0; row < d; ++row) {
            q[row * d + col] =
                {row == candidate ? T{1} : T{0}, T{0}};
          }
          for (int64_t prior = 0; prior < col; ++prior) {
            cuda::std::complex<T> projection{T{0}, T{0}};
            for (int64_t row = 0; row < d; ++row) {
              projection += esprit_conj(q[row * d + prior]) *
                            q[row * d + col];
            }
            for (int64_t row = 0; row < d; ++row) {
              q[row * d + col] -= q[row * d + prior] * projection;
            }
          }
          column_norm_squared = T{0};
          for (int64_t row = 0; row < d; ++row) {
            column_norm_squared += esprit_norm_squared(q[row * d + col]);
          }
          completed = column_norm_squared > floor;
        }
        if (!completed) {
          return false;
        }
        r[col * d + col] = {T{0}, T{0}};
      } else {
        r[col * d + col] = {sqrt(column_norm_squared), T{0}};
      }
      const T inverse_norm = T{1} / sqrt(column_norm_squared);
      for (int64_t row = 0; row < d; ++row) {
        q[row * d + col] *= inverse_norm;
      }
    }

    T diagonal_scale = T{0};
    T subdiagonal = T{0};
    for (int64_t row = 0; row < d; ++row) {
      for (int64_t col = 0; col < d; ++col) {
        cuda::std::complex<T> value{T{0}, T{0}};
        for (int64_t inner = 0; inner < d; ++inner) {
          value += r[row * d + inner] * q[inner * d + col];
        }
        if (row == col) {
          value += shift;
          diagonal_scale += sqrt(esprit_norm_squared(value));
        } else if (row > col) {
          subdiagonal += sqrt(esprit_norm_squared(value));
        }
        matrix[row * d + col] = value;
      }
    }
    if (subdiagonal <= tolerance * (diagonal_scale + T{1})) {
      return true;
    }
  }
  return false;
}

template <typename T>
__global__ void esprit_solve_kernel(
    const cuda::std::complex<T>* __restrict__ eigenvectors,
    const int32_t* __restrict__ model_order,
    const int64_t* __restrict__ valid_counts,
    const int32_t* __restrict__ reference_indices,
    const int32_t* __restrict__ shifted_indices, int64_t num_items,
    int64_t channels, int64_t pair_size, int64_t workspace_dim,
    T wave_number_spacing, T axis_x, T axis_y, T axis_z, T broadside_x,
    T broadside_y, T broadside_z, AngleConventionPreset preset,
    cuda::std::complex<T>* __restrict__ matrix_workspace,
    cuda::std::complex<T>* __restrict__ q_workspace,
    cuda::std::complex<T>* __restrict__ r_workspace, T* __restrict__ aoa,
    T* __restrict__ doa_multi, int64_t max_sources,
    int32_t* __restrict__ source_count) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || item >= num_items) {
    return;
  }

  const T nan = T{0} / T{0};
  aoa[item * 2] = nan;
  aoa[item * 2 + 1] = nan;
  for (int64_t source = 0; source < max_sources; ++source) {
    doa_multi[(item * max_sources + source) * 2] = nan;
    doa_multi[(item * max_sources + source) * 2 + 1] = nan;
  }
  source_count[item] = 0;
  if (valid_counts[item] <= 0) {
    return;
  }

  int64_t d = static_cast<int64_t>(model_order[item]);
  d = d > pair_size ? pair_size : d;
  d = d > max_sources ? max_sources : d;
  if (d < 1) {
    return;
  }
  const int64_t stride = workspace_dim * workspace_dim;
  auto* matrix = matrix_workspace + item * stride;
  auto* q = q_workspace + item * stride;
  auto* r = r_workspace + item * stride;
  const auto* evec = eigenvectors + item * channels * channels;
  const int64_t signal_first = channels - d;

  for (int64_t row = 0; row < d; ++row) {
    for (int64_t col = 0; col < d; ++col) {
      cuda::std::complex<T> gram{T{0}, T{0}};
      cuda::std::complex<T> cross{T{0}, T{0}};
      for (int64_t pair_row = 0; pair_row < pair_size; ++pair_row) {
        const int64_t reference = reference_indices[pair_row];
        const int64_t shifted = shifted_indices[pair_row];
        const auto left =
            evec[reference * channels + signal_first + row];
        gram += esprit_conj(left) *
                evec[reference * channels + signal_first + col];
        cross += esprit_conj(left) *
                 evec[shifted * channels + signal_first + col];
      }
      q[row * d + col] = gram;
      matrix[row * d + col] = cross;
    }
  }

  const T pivot_floor = std::is_same_v<T, float> ? static_cast<T>(1e-10)
                                                 : static_cast<T>(1e-20);
  for (int64_t pivot = 0; pivot < d; ++pivot) {
    int64_t best = pivot;
    T best_norm = esprit_norm_squared(q[pivot * d + pivot]);
    for (int64_t row = pivot + 1; row < d; ++row) {
      const T candidate = esprit_norm_squared(q[row * d + pivot]);
      if (candidate > best_norm) {
        best = row;
        best_norm = candidate;
      }
    }
    if (!(best_norm > pivot_floor)) {
      return;
    }
    if (best != pivot) {
      for (int64_t col = 0; col < d; ++col) {
        const auto gram_tmp = q[pivot * d + col];
        q[pivot * d + col] = q[best * d + col];
        q[best * d + col] = gram_tmp;
        const auto rhs_tmp = matrix[pivot * d + col];
        matrix[pivot * d + col] = matrix[best * d + col];
        matrix[best * d + col] = rhs_tmp;
      }
    }
    const auto diagonal = q[pivot * d + pivot];
    for (int64_t col = 0; col < d; ++col) {
      q[pivot * d + col] /= diagonal;
      matrix[pivot * d + col] /= diagonal;
    }
    for (int64_t row = 0; row < d; ++row) {
      if (row == pivot) {
        continue;
      }
      const auto factor = q[row * d + pivot];
      for (int64_t col = 0; col < d; ++col) {
        q[row * d + col] -= factor * q[pivot * d + col];
        matrix[row * d + col] -= factor * matrix[pivot * d + col];
      }
    }
  }

  if (!esprit_qr_eigenvalues(matrix, q, r, d)) {
    return;
  }

  int64_t selected[64];
  for (int64_t output = 0; output < d; ++output) {
    int64_t best = -1;
    T best_score = static_cast<T>(1e30);
    T best_phase = T{0};
    for (int64_t candidate = 0; candidate < d; ++candidate) {
      bool used = false;
      for (int64_t prior = 0; prior < output; ++prior) {
        used = used || selected[prior] == candidate;
      }
      if (used) {
        continue;
      }
      const auto root = matrix[candidate * d + candidate];
      const T magnitude = hypot(root.real(), root.imag());
      if (!isfinite(magnitude)) {
        continue;
      }
      const T phase = atan2(root.imag(), root.real());
      const T score = fabs(T{1} - magnitude);
      if (score < best_score ||
          (score == best_score && phase < best_phase)) {
        best = candidate;
        best_score = score;
        best_phase = phase;
      }
    }
    if (best < 0) {
      break;
    }
    selected[output] = best;
    T projection = best_phase / wave_number_spacing;
    projection = projection < T{-1} ? T{-1} : projection;
    projection = projection > T{1} ? T{1} : projection;
    const T transverse = sqrt(fmax(T{0}, T{1} - projection * projection));
    const T x = projection * axis_x + transverse * broadside_x;
    const T y = projection * axis_y + transverse * broadside_y;
    const T z = projection * axis_z + transverse * broadside_z;
    T out0 = nan;
    T out1 = nan;
    esprit_canonical_to_boundary(preset, x, y, z, &out0, &out1);
    const int64_t base = (item * max_sources + output) * 2;
    doa_multi[base] = out0;
    doa_multi[base + 1] = out1;
    if (output == 0) {
      aoa[item * 2] = out0;
      aoa[item * 2 + 1] = out1;
    }
    source_count[item] = static_cast<int32_t>(output + 1);
  }
}

template <typename T>
inline void run_esprit_stage(
    const cuda::std::complex<T>* snapshots, const unsigned char* mask,
    int64_t num_items, int64_t channels, int64_t num_snapshots,
    int64_t fixed_num_sources, double pfa, int64_t model_order_cap,
    const int32_t* reference_indices, const int32_t* shifted_indices,
    int64_t pair_size, int64_t workspace_dim, T wave_number_spacing, T axis_x,
    T axis_y, T axis_z, T broadside_x, T broadside_y, T broadside_z,
    AngleConventionPreset preset, cuda::std::complex<T>* covariance,
    cuda::std::complex<T>* eigenvectors, int64_t* valid_counts,
    int32_t* model_order, T* eigenvalues,
    cuda::std::complex<T>* matrix_workspace,
    cuda::std::complex<T>* q_workspace,
    cuda::std::complex<T>* r_workspace, T* aoa, T* doa_multi,
    int64_t max_sources, int32_t* source_count, cudaStream_t stream) {
  if (num_items <= 0 || channels < 2) {
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
                               pfa, model_order_cap, model_order, stream);
  } else {
    constexpr int fill_block = 128;
    const int fill_grid =
        static_cast<int>((num_items + fill_block - 1) / fill_block);
    fill_model_order_kernel<<<fill_grid, fill_block, 0, stream>>>(
        model_order, num_items, static_cast<int32_t>(fixed_num_sources));
  }

  esprit_solve_kernel<T><<<grid, 1, 0, stream>>>(
      eigenvectors, model_order, valid_counts, reference_indices,
      shifted_indices, num_items, channels, pair_size, workspace_dim,
      wave_number_spacing, axis_x, axis_y, axis_z, broadside_x, broadside_y,
      broadside_z, preset, matrix_workspace, q_workspace, r_workspace, aoa,
      doa_multi, max_sources, source_count);

  mark_music_eig_cache_live();
  retire_music_matx_caches();
}

}  // namespace holoscan::radar::detail
