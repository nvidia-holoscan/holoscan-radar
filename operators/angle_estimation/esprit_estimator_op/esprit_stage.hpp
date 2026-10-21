// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "angle_estimation/angle_boundary_device.hpp"
#include "angle_estimation/music_estimator_op/music_stage.hpp"
#include "radar_common/dsp/source_count_device.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

template <typename T>
__device__ inline T esprit_norm_squared(cuda::std::complex<T> value) {
  return value.real() * value.real() + value.imag() * value.imag();
}

template <typename T>
__device__ inline cuda::std::complex<T> esprit_conj(cuda::std::complex<T> value) {
  return {value.real(), -value.imag()};
}

template <typename T>
__device__ bool esprit_qr_eigenvalues(cuda::std::complex<T>* matrix, cuda::std::complex<T>* q,
                                      cuda::std::complex<T>* r, int64_t d) {
  if (d <= 1) {
    return true;
  }
  const T floor = std::is_same_v<T, float> ? static_cast<T>(1e-12) : static_cast<T>(1e-24);
  const T tolerance = std::is_same_v<T, float> ? static_cast<T>(2e-5) : static_cast<T>(1e-11);
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
            matrix[row * d + col] - (row == col ? shift : cuda::std::complex<T>{T{0}, T{0}});
      }
      for (int64_t prior = 0; prior < col; ++prior) {
        cuda::std::complex<T> projection{T{0}, T{0}};
        for (int64_t row = 0; row < d; ++row) {
          projection += esprit_conj(q[row * d + prior]) * q[row * d + col];
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
            q[row * d + col] = {row == candidate ? T{1} : T{0}, T{0}};
          }
          for (int64_t prior = 0; prior < col; ++prior) {
            cuda::std::complex<T> projection{T{0}, T{0}};
            for (int64_t row = 0; row < d; ++row) {
              projection += esprit_conj(q[row * d + prior]) * q[row * d + col];
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

// ---------------------------------------------------------------------------
// 2-D URA path: dual-invariance 2-D ESPRIT with joint-diagonalization pairing.
//
// The signal subspace yields two rotation operators Psi_y (azimuth y-shift)
// and Psi_z (elevation z-shift). In the noiseless model they share
// eigenvectors (they commute), so the unitary Schur basis Z that triangulizes
// Psi_y (Z^H Psi_y Z upper-triangular) simultaneously triangulizes Psi_z; the
// paired diagonals diag(Z^H Psi_y Z) and diag(Z^H Psi_z Z) give the
// automatically-paired azimuth/elevation eigenvalues. See OPERATOR_API.md.
// ---------------------------------------------------------------------------

// Es_ref^H Es_ref -> gram (d x d) and Es_ref^H Es_shift -> cross (d x d), where
// the signal subspace is the last d eigenvectors (columns signal_first..).
template <typename T>
__device__ inline void esprit_build_gram_cross(const cuda::std::complex<T>* __restrict__ evec,
                                               const int32_t* __restrict__ reference_indices,
                                               const int32_t* __restrict__ shifted_indices,
                                               int64_t pair_size, int64_t d, int64_t signal_first,
                                               int64_t channels, cuda::std::complex<T>* gram,
                                               cuda::std::complex<T>* cross) {
  for (int64_t row = 0; row < d; ++row) {
    for (int64_t col = 0; col < d; ++col) {
      cuda::std::complex<T> g{T{0}, T{0}};
      cuda::std::complex<T> c{T{0}, T{0}};
      for (int64_t pr = 0; pr < pair_size; ++pr) {
        const int64_t reference = reference_indices[pr];
        const int64_t shifted = shifted_indices[pr];
        const auto left = evec[reference * channels + signal_first + row];
        g += esprit_conj(left) * evec[reference * channels + signal_first + col];
        c += esprit_conj(left) * evec[shifted * channels + signal_first + col];
      }
      gram[row * d + col] = g;
      cross[row * d + col] = c;
    }
  }
}

// Gaussian elimination with partial pivoting: solve a * x = b in place
// (b <- x). Returns false if a is numerically singular.
template <typename T>
__device__ inline bool esprit_lu_solve(cuda::std::complex<T>* a, cuda::std::complex<T>* b,
                                       int64_t d) {
  const T pivot_floor = std::is_same_v<T, float> ? static_cast<T>(1e-10) : static_cast<T>(1e-20);
  for (int64_t pivot = 0; pivot < d; ++pivot) {
    int64_t best = pivot;
    T best_norm = esprit_norm_squared(a[pivot * d + pivot]);
    for (int64_t row = pivot + 1; row < d; ++row) {
      const T candidate = esprit_norm_squared(a[row * d + pivot]);
      if (candidate > best_norm) {
        best = row;
        best_norm = candidate;
      }
    }
    if (!(best_norm > pivot_floor)) {
      return false;
    }
    if (best != pivot) {
      for (int64_t col = 0; col < d; ++col) {
        const auto a_tmp = a[pivot * d + col];
        a[pivot * d + col] = a[best * d + col];
        a[best * d + col] = a_tmp;
        const auto b_tmp = b[pivot * d + col];
        b[pivot * d + col] = b[best * d + col];
        b[best * d + col] = b_tmp;
      }
    }
    const auto diagonal = a[pivot * d + pivot];
    for (int64_t col = 0; col < d; ++col) {
      a[pivot * d + col] /= diagonal;
      b[pivot * d + col] /= diagonal;
    }
    for (int64_t row = 0; row < d; ++row) {
      if (row == pivot) {
        continue;
      }
      const auto factor = a[row * d + pivot];
      for (int64_t col = 0; col < d; ++col) {
        a[row * d + col] -= factor * a[pivot * d + col];
        b[row * d + col] -= factor * b[pivot * d + col];
      }
    }
  }
  return true;
}

// Shifted QR reduction of `a` (d x d) to upper-triangular Schur form in place,
// accumulating the unitary Schur vectors into `z` (Z^H a0 Z = a_final). `q`,
// `r`, and `tmp` are d x d scratch. Mirrors esprit_qr_eigenvalues but also
// forms z (init identity, z <- z * Q_k each iteration).
template <typename T>
__device__ bool esprit_schur_vectors(cuda::std::complex<T>* matrix, cuda::std::complex<T>* z,
                                     cuda::std::complex<T>* q, cuda::std::complex<T>* r,
                                     cuda::std::complex<T>* tmp, int64_t d) {
  for (int64_t index = 0; index < d * d; ++index) {
    z[index] = {T{0}, T{0}};
  }
  for (int64_t i = 0; i < d; ++i) {
    z[i * d + i] = {T{1}, T{0}};
  }
  if (d <= 1) {
    return true;
  }
  const T floor = std::is_same_v<T, float> ? static_cast<T>(1e-12) : static_cast<T>(1e-24);
  const T tolerance = std::is_same_v<T, float> ? static_cast<T>(2e-5) : static_cast<T>(1e-11);
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
            matrix[row * d + col] - (row == col ? shift : cuda::std::complex<T>{T{0}, T{0}});
      }
      for (int64_t prior = 0; prior < col; ++prior) {
        cuda::std::complex<T> projection{T{0}, T{0}};
        for (int64_t row = 0; row < d; ++row) {
          projection += esprit_conj(q[row * d + prior]) * q[row * d + col];
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
            q[row * d + col] = {row == candidate ? T{1} : T{0}, T{0}};
          }
          for (int64_t prior = 0; prior < col; ++prior) {
            cuda::std::complex<T> projection{T{0}, T{0}};
            for (int64_t row = 0; row < d; ++row) {
              projection += esprit_conj(q[row * d + prior]) * q[row * d + col];
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

    // z <- z * Q_k
    for (int64_t row = 0; row < d; ++row) {
      for (int64_t col = 0; col < d; ++col) {
        cuda::std::complex<T> value{T{0}, T{0}};
        for (int64_t inner = 0; inner < d; ++inner) {
          value += z[row * d + inner] * q[inner * d + col];
        }
        tmp[row * d + col] = value;
      }
    }
    for (int64_t index = 0; index < d * d; ++index) {
      z[index] = tmp[index];
    }

    if (subdiagonal <= tolerance * (diagonal_scale + T{1})) {
      return true;
    }
  }
  return false;
}

// One block per item (thread 0). Solves the two invariance rotations Psi_y,
// Psi_z from the signal subspace, joint-diagonalizes them (Schur of Psi_y with
// vectors, applied to Psi_z), and maps the paired eigenvalues to (u, v) and
// then to boundary angles through the aperture-axis frame.
template <typename T>
__global__ void esprit_solve_2d_kernel(
    const cuda::std::complex<T>* __restrict__ eigenvectors, const int32_t* __restrict__ model_order,
    const int64_t* __restrict__ valid_counts, const int32_t* __restrict__ y_reference,
    const int32_t* __restrict__ y_shifted, int64_t y_pairs, const int32_t* __restrict__ z_reference,
    const int32_t* __restrict__ z_shifted, int64_t z_pairs, int64_t num_items, int64_t channels,
    int64_t max_model, int64_t workspace_dim, T ky_spacing, T kz_spacing, T yax_x, T yax_y, T yax_z,
    T zax_x, T zax_y, T zax_z, T nrm_x, T nrm_y, T nrm_z, DeviceAngleConvention<T> convention,
    cuda::std::complex<T>* __restrict__ psi_y_ws, cuda::std::complex<T>* __restrict__ psi_z_ws,
    cuda::std::complex<T>* __restrict__ z_ws, cuda::std::complex<T>* __restrict__ q_ws,
    cuda::std::complex<T>* __restrict__ r_ws, cuda::std::complex<T>* __restrict__ tmp_ws,
    T* __restrict__ aoa, T* __restrict__ doa_multi, int64_t max_sources,
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
  d = d > max_model ? max_model : d;
  d = d > max_sources ? max_sources : d;
  d = d > y_pairs ? y_pairs : d;
  d = d > z_pairs ? z_pairs : d;
  if (d < 1) {
    return;
  }
  const int64_t stride = workspace_dim * workspace_dim;
  auto* psi_y = psi_y_ws + item * stride;
  auto* psi_z = psi_z_ws + item * stride;
  auto* z = z_ws + item * stride;
  auto* q = q_ws + item * stride;
  auto* r = r_ws + item * stride;
  auto* tmp = tmp_ws + item * stride;
  const int64_t signal_first = channels - d;

  // Psi_y = (Es_ref^H Es_ref)^-1 (Es_ref^H Es_shift): gram in q, cross in
  // psi_y.
  esprit_build_gram_cross<T>(eigenvectors + item * channels * channels, y_reference, y_shifted,
                             y_pairs, d, signal_first, channels, q, psi_y);
  if (!esprit_lu_solve<T>(q, psi_y, d)) {
    return;
  }
  esprit_build_gram_cross<T>(eigenvectors + item * channels * channels, z_reference, z_shifted,
                             z_pairs, d, signal_first, channels, q, psi_z);
  if (!esprit_lu_solve<T>(q, psi_z, d)) {
    return;
  }

  // Schur of Psi_y with vectors; diag(Psi_y) -> mu_y; diag(z^H Psi_z z) ->
  // mu_z.
  if (!esprit_schur_vectors<T>(psi_y, z, q, r, tmp, d)) {
    return;
  }

  int64_t found = 0;
  for (int64_t i = 0; i < d; ++i) {
    const cuda::std::complex<T> mu_y = psi_y[i * d + i];
    cuda::std::complex<T> mu_z{T{0}, T{0}};
    for (int64_t p = 0; p < d; ++p) {
      cuda::std::complex<T> psi_z_col{T{0}, T{0}};
      for (int64_t c = 0; c < d; ++c) {
        psi_z_col += psi_z[p * d + c] * z[c * d + i];
      }
      mu_z += esprit_conj(z[p * d + i]) * psi_z_col;
    }
    const T mag_y = hypot(mu_y.real(), mu_y.imag());
    const T mag_z = hypot(mu_z.real(), mu_z.imag());
    if (!isfinite(mag_y) || !isfinite(mag_z)) {
      continue;
    }
    T u = atan2(mu_y.imag(), mu_y.real()) / ky_spacing;
    T v = atan2(mu_z.imag(), mu_z.real()) / kz_spacing;
    u = u < T{-1} ? T{-1} : u;
    u = u > T{1} ? T{1} : u;
    v = v < T{-1} ? T{-1} : v;
    v = v > T{1} ? T{1} : v;
    const T w = sqrt(fmax(T{0}, T{1} - u * u - v * v));
    const T x = u * yax_x + v * zax_x + w * nrm_x;
    const T y = u * yax_y + v * zax_y + w * nrm_y;
    const T zc = u * yax_z + v * zax_z + w * nrm_z;
    T out0 = nan;
    T out1 = nan;
    angle_canonical_to_boundary(convention, x, y, zc, &out0, &out1);
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
__global__ void esprit_solve_kernel(
    const cuda::std::complex<T>* __restrict__ eigenvectors, const int32_t* __restrict__ model_order,
    const int64_t* __restrict__ valid_counts, const int32_t* __restrict__ reference_indices,
    const int32_t* __restrict__ shifted_indices, int64_t num_items, int64_t channels,
    int64_t pair_size, int64_t workspace_dim, T wave_number_spacing, T axis_x, T axis_y, T axis_z,
    T broadside_x, T broadside_y, T broadside_z, DeviceAngleConvention<T> convention,
    cuda::std::complex<T>* __restrict__ matrix_workspace,
    cuda::std::complex<T>* __restrict__ q_workspace,
    cuda::std::complex<T>* __restrict__ r_workspace, T* __restrict__ aoa, T* __restrict__ doa_multi,
    int64_t max_sources, int32_t* __restrict__ source_count) {
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
        const auto left = evec[reference * channels + signal_first + row];
        gram += esprit_conj(left) * evec[reference * channels + signal_first + col];
        cross += esprit_conj(left) * evec[shifted * channels + signal_first + col];
      }
      q[row * d + col] = gram;
      matrix[row * d + col] = cross;
    }
  }

  const T pivot_floor = std::is_same_v<T, float> ? static_cast<T>(1e-10) : static_cast<T>(1e-20);
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
      if (score < best_score || (score == best_score && phase < best_phase)) {
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
    angle_canonical_to_boundary(convention, x, y, z, &out0, &out1);
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
    const cuda::std::complex<T>* snapshots, const unsigned char* mask, int64_t num_items,
    int64_t channels, int64_t num_snapshots, int64_t fixed_num_sources, double pfa,
    int64_t model_order_cap, const int32_t* reference_indices, const int32_t* shifted_indices,
    int64_t pair_size, int64_t workspace_dim, T wave_number_spacing, T axis_x, T axis_y, T axis_z,
    T broadside_x, T broadside_y, T broadside_z, DeviceAngleConvention<T> convention,
    cuda::std::complex<T>* covariance, cuda::std::complex<T>* eigenvectors, int64_t* valid_counts,
    int32_t* model_order, T* eigenvalues, cuda::std::complex<T>* matrix_workspace,
    cuda::std::complex<T>* q_workspace, cuda::std::complex<T>* r_workspace, T* aoa, T* doa_multi,
    int64_t max_sources, int32_t* source_count, cudaStream_t stream) {
  if (num_items <= 0 || channels < 2) {
    return;
  }
  constexpr int block = 256;
  const int grid = static_cast<int>(num_items);
  music_covariance_kernel<T><<<grid, block, 0, stream>>>(snapshots, mask, num_items, channels,
                                                         num_snapshots, covariance, valid_counts);

  auto covariance_view =
      matx::make_tensor<cuda::std::complex<T>, 3>(covariance, {num_items, channels, channels});
  auto eigenvector_view =
      matx::make_tensor<cuda::std::complex<T>, 3>(eigenvectors, {num_items, channels, channels});
  auto eigenvalue_view = matx::make_tensor<T, 2>(eigenvalues, {num_items, channels});
  matx::cudaExecutor eig_exec{stream};
  (matx::mtie(eigenvector_view, eigenvalue_view) =
       matx::eig(covariance_view, matx::EigenMode::VECTOR))
      .run(eig_exec);

  if (fixed_num_sources < 0) {
    run_source_count_device<T>(eigenvalues, valid_counts, num_items, channels, pfa, model_order_cap,
                               model_order, stream);
  } else {
    constexpr int fill_block = 128;
    const int fill_grid = static_cast<int>((num_items + fill_block - 1) / fill_block);
    fill_model_order_kernel<<<fill_grid, fill_block, 0, stream>>>(
        model_order, num_items, static_cast<int32_t>(fixed_num_sources));
  }

  esprit_solve_kernel<T><<<grid, 1, 0, stream>>>(
      eigenvectors, model_order, valid_counts, reference_indices, shifted_indices, num_items,
      channels, pair_size, workspace_dim, wave_number_spacing, axis_x, axis_y, axis_z, broadside_x,
      broadside_y, broadside_z, convention, matrix_workspace, q_workspace, r_workspace, aoa,
      doa_multi, max_sources, source_count);

  // Reuse dimension/stream-keyed eig plans; stop() retires their owned workspaces.
  mark_music_eig_cache_live();
}

// Full 2-D URA ESPRIT for one tick: shared covariance / Hermitian eig / model
// order front-end (identical to run_esprit_stage), then the dual-invariance
// joint-diagonalization solve. `covariance`/`eigenvectors`/`eigenvalues`/
// `valid_counts`/`model_order`/`aoa`/`doa_multi`/`source_count` are
// operator-owned scratch; the six d x d workspaces hold Psi_y, Psi_z, the
// Schur vectors, and QR scratch.
template <typename T>
inline void run_esprit_2d_stage(
    const cuda::std::complex<T>* snapshots, const unsigned char* mask, int64_t num_items,
    int64_t channels, int64_t num_snapshots, int64_t fixed_num_sources, double pfa,
    int64_t model_order_cap, const int32_t* y_reference, const int32_t* y_shifted, int64_t y_pairs,
    const int32_t* z_reference, const int32_t* z_shifted, int64_t z_pairs, int64_t workspace_dim,
    T ky_spacing, T kz_spacing, T yax_x, T yax_y, T yax_z, T zax_x, T zax_y, T zax_z, T nrm_x,
    T nrm_y, T nrm_z, DeviceAngleConvention<T> convention, cuda::std::complex<T>* covariance,
    cuda::std::complex<T>* eigenvectors, int64_t* valid_counts, int32_t* model_order,
    T* eigenvalues, cuda::std::complex<T>* psi_y_ws, cuda::std::complex<T>* psi_z_ws,
    cuda::std::complex<T>* z_ws, cuda::std::complex<T>* q_ws, cuda::std::complex<T>* r_ws,
    cuda::std::complex<T>* tmp_ws, T* aoa, T* doa_multi, int64_t max_sources, int32_t* source_count,
    cudaStream_t stream) {
  if (num_items <= 0 || channels < 4) {
    return;
  }
  constexpr int block = 256;
  const int grid = static_cast<int>(num_items);
  music_covariance_kernel<T><<<grid, block, 0, stream>>>(snapshots, mask, num_items, channels,
                                                         num_snapshots, covariance, valid_counts);

  auto covariance_view =
      matx::make_tensor<cuda::std::complex<T>, 3>(covariance, {num_items, channels, channels});
  auto eigenvector_view =
      matx::make_tensor<cuda::std::complex<T>, 3>(eigenvectors, {num_items, channels, channels});
  auto eigenvalue_view = matx::make_tensor<T, 2>(eigenvalues, {num_items, channels});
  matx::cudaExecutor eig_exec{stream};
  (matx::mtie(eigenvector_view, eigenvalue_view) =
       matx::eig(covariance_view, matx::EigenMode::VECTOR))
      .run(eig_exec);

  if (fixed_num_sources < 0) {
    run_source_count_device<T>(eigenvalues, valid_counts, num_items, channels, pfa, model_order_cap,
                               model_order, stream);
  } else {
    constexpr int fill_block = 128;
    const int fill_grid = static_cast<int>((num_items + fill_block - 1) / fill_block);
    fill_model_order_kernel<<<fill_grid, fill_block, 0, stream>>>(
        model_order, num_items, static_cast<int32_t>(fixed_num_sources));
  }

  esprit_solve_2d_kernel<T><<<grid, 1, 0, stream>>>(
      eigenvectors, model_order, valid_counts, y_reference, y_shifted, y_pairs, z_reference,
      z_shifted, z_pairs, num_items, channels, model_order_cap, workspace_dim, ky_spacing,
      kz_spacing, yax_x, yax_y, yax_z, zax_x, zax_y, zax_z, nrm_x, nrm_y, nrm_z, convention,
      psi_y_ws, psi_z_ws, z_ws, q_ws, r_ws, tmp_ws, aoa, doa_multi, max_sources, source_count);

  // Reuse dimension/stream-keyed eig plans; stop() retires their owned workspaces.
  mark_music_eig_cache_live();
}

}  // namespace holoscan::radar::detail
