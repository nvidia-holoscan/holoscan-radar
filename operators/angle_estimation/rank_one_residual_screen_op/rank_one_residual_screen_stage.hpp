// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/math/constants.hpp"

namespace holoscan::radar::detail {

struct DeviceDirection {
  double x;
  double y;
  double z;
};

__device__ inline DeviceDirection screen_axis(SignedAxis axis) {
  switch (axis) {
    case SignedAxis::kPlusX:
      return {1.0, 0.0, 0.0};
    case SignedAxis::kMinusX:
      return {-1.0, 0.0, 0.0};
    case SignedAxis::kPlusY:
      return {0.0, 1.0, 0.0};
    case SignedAxis::kMinusY:
      return {0.0, -1.0, 0.0};
    case SignedAxis::kPlusZ:
      return {0.0, 0.0, 1.0};
    case SignedAxis::kMinusZ:
      return {0.0, 0.0, -1.0};
  }
  return {1.0, 0.0, 0.0};
}

__device__ inline DeviceDirection screen_direction(const AngleConvention& convention, double az,
                                                   double el) {
  if (convention.preset == AngleConventionPreset::kDirectionCosine) {
    const double r2 = az * az + el * el;
    return r2 <= 1.0 ? DeviceDirection{sqrt(fmax(0.0, 1.0 - r2)), az, el}
                     : DeviceDirection{0.0, 0.0, 0.0};
  }
  if (convention.preset == AngleConventionPreset::kAerospaceNed ||
      convention.preset == AngleConventionPreset::kGeodeticNed) {
    az = -az;
    el = -el;
  } else if (convention.preset == AngleConventionPreset::kGeodeticEnu) {
    az -= kHalfPi;
  } else if (convention.preset == AngleConventionPreset::kPhysicsSpherical) {
    el = kHalfPi - el;
  } else if (convention.preset == AngleConventionPreset::kCustom) {
    const auto b = screen_axis(convention.boresight_axis);
    const auto a = screen_axis(convention.azimuth_positive_axis);
    const auto e = screen_axis(convention.elevation_positive_axis);
    const double eh =
        convention.elevation_reference == ElevationReference::kZenith ? kHalfPi - el : el;
    const double ce = cos(eh);
    const double se = sin(eh);
    const double ca = cos(az);
    const double sa = sin(az);
    return {ce * ca * b.x + ce * sa * a.x + se * e.x, ce * ca * b.y + ce * sa * a.y + se * e.y,
            ce * ca * b.z + ce * sa * a.z + se * e.z};
  }
  const double ce = cos(el);
  return {ce * cos(az), ce * sin(az), sin(el)};
}

template <typename T>
__global__ void residual_screen_kernel(const cuda::std::complex<T>* snapshot, const T* aoa,
                                       const unsigned char* valid, const double* positions,
                                       int64_t num_items, int64_t channels, int64_t snapshots,
                                       double wavelength, AngleConvention convention, T threshold,
                                       T* scores, unsigned char* selected) {
  const int64_t item =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
  if (item >= num_items) {
    return;
  }
  const double az = static_cast<double>(aoa[item * 2]);
  const double el = static_cast<double>(aoa[item * 2 + 1]);
  if (!isfinite(az) || !isfinite(el)) {
    scores[item] = T{0} / T{0};
    selected[item] = 0;
    return;
  }
  const auto d = screen_direction(convention, az, el);
  const double dnorm = d.x * d.x + d.y * d.y + d.z * d.z;
  if (dnorm < 0.5) {
    scores[item] = T{0} / T{0};
    selected[item] = 0;
    return;
  }
  const double wave_number = kTwoPi / wavelength;
  double projected = 0.0;
  double total = 0.0;
  int64_t looks = 0;
  for (int64_t s = 0; s < snapshots; ++s) {
    if (valid != nullptr && valid[item * snapshots + s] == 0) {
      continue;
    }
    ++looks;
    double sum_re = 0.0;
    double sum_im = 0.0;
    for (int64_t c = 0; c < channels; ++c) {
      const auto x = snapshot[(item * channels + c) * snapshots + s];
      const double xr = static_cast<double>(x.real());
      const double xi = static_cast<double>(x.imag());
      total += xr * xr + xi * xi;
      const double phase = wave_number * (positions[c * 3] * d.x + positions[c * 3 + 1] * d.y +
                                          positions[c * 3 + 2] * d.z);
      const double cr = cos(phase);
      const double ci = -sin(phase);
      sum_re += cr * xr - ci * xi;
      sum_im += cr * xi + ci * xr;
    }
    projected += sum_re * sum_re + sum_im * sum_im;
  }
  if (looks == 0 || total <= 0.0) {
    scores[item] = T{0} / T{0};
    selected[item] = 0;
    return;
  }
  double score = 1.0 - projected / (static_cast<double>(channels) * total);
  score = fmin(1.0, fmax(0.0, score));
  scores[item] = static_cast<T>(score);
  selected[item] = score >= static_cast<double>(threshold) ? 1 : 0;
}

template <typename T>
__global__ void compact_screen_kernel(const cuda::std::complex<T>* snapshot,
                                      const unsigned char* valid, const unsigned char* selected,
                                      const uint32_t* offsets, int64_t num_items, int64_t channels,
                                      int64_t snapshots, cuda::std::complex<T>* snapshot_out,
                                      unsigned char* valid_out, int64_t* selected_idx) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items || selected[item] == 0) {
    return;
  }
  const int64_t dst_item = static_cast<int64_t>(offsets[item]);
  if (threadIdx.x == 0) {
    selected_idx[dst_item] = item;
  }
  for (int64_t i = threadIdx.x; i < channels * snapshots; i += blockDim.x) {
    snapshot_out[dst_item * channels * snapshots + i] = snapshot[item * channels * snapshots + i];
  }
  for (int64_t s = threadIdx.x; s < snapshots; s += blockDim.x) {
    valid_out[dst_item * snapshots + s] = valid == nullptr ? 1 : valid[item * snapshots + s];
  }
}

}  // namespace holoscan::radar::detail
