// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Deterministic device-tensor generation for the performance harness. Builds
// a real (already-formed power) map on the device so benchmarks exercise the
// real-input CFAR path with a fixed, reproducible payload. The map is
// preallocated once per scenario and reused across every frame.
//
// FMCW consumers additionally need a complex IQ cube (adc_cube / range_profiles
// payload); make_iq_cube_device<T>() builds one with a deterministic multi-tone
// pattern so the FFT stages exercise real spectral structure, not zeros.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/domain/tensor_map.hpp>

#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::perf {

// Single-entry TensorMap (the module's port payload convention).
inline holoscan::TensorMap single_map(std::shared_ptr<holoscan::Tensor> tensor,
                                      const std::string& key) {
  holoscan::TensorMap map;
  map.insert({key, std::move(tensor)});
  return map;
}

// Fill a host buffer with a deterministic noise floor plus a sparse, fixed
// grid of stronger returns so detection_idx compaction has non-trivial (but
// reproducible) work every frame. Values are patterned, not random, so the
// same shape yields the same payload across commits on the same box.
template <typename T>
std::vector<T> deterministic_power_host(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  std::vector<T> host(static_cast<size_t>(count));
  const T floor = static_cast<T>(1.0);
  // ~0.1% of cells carry a strong return; the stride is coprime-ish with
  // common row widths so peaks do not all land in one column.
  const int64_t stride = 997;
  for (int64_t i = 0; i < count; ++i) {
    host[static_cast<size_t>(i)] =
        (i % stride == 0) ? static_cast<T>(500.0) : floor;
  }
  return host;
}

// Allocate a device tensor of `shape` and upload the deterministic power map.
template <typename T>
std::shared_ptr<holoscan::Tensor> make_power_map_device(
    const std::vector<int64_t>& shape) {
  const auto host = deterministic_power_host<T>(shape);
  auto tensor = make_device_tensor(shape, real_dtype<T>());
  if (cudaMemcpy(tensor->data(), host.data(), sizeof(T) * host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("make_power_map_device: cudaMemcpy H2D failed");
  }
  return tensor;
}

// Deterministic complex IQ payload for the FMCW stages. The last axis is the
// transform axis (fast-time samples for adc_cube, slow-time chirps once
// transposed for Doppler); a few fixed tones along it give the FFT real
// spectral structure. Patterned (not random) so the same shape yields the same
// payload across commits on the same box.
template <typename T>
std::vector<std::complex<T>> deterministic_iq_host(
    const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  const int64_t n = shape.empty() ? 1 : shape.back();  // transform length
  std::vector<std::complex<T>> host(static_cast<size_t>(count));
  // Two coprime-ish tone indices plus a small DC term; the pattern repeats
  // every `n` elements so each row along the transform axis is identical.
  const double k1 = 5.0;
  const double k2 = 17.0;
  for (int64_t i = 0; i < count; ++i) {
    const int64_t s = (n > 0) ? (i % n) : 0;
    const double phase1 = kTwoPi * k1 * static_cast<double>(s) /
                          static_cast<double>(n > 0 ? n : 1);
    const double phase2 = kTwoPi * k2 * static_cast<double>(s) /
                          static_cast<double>(n > 0 ? n : 1);
    const double re = std::cos(phase1) + 0.5 * std::cos(phase2) + 0.1;
    const double im = std::sin(phase1) + 0.5 * std::sin(phase2);
    host[static_cast<size_t>(i)] =
        std::complex<T>(static_cast<T>(re), static_cast<T>(im));
  }
  return host;
}

// Allocate a complex64/complex128 device tensor of `shape` and upload the
// deterministic IQ payload. Used for adc_cube (range/fused) and range_profiles
// (Doppler/clutter) inputs.
template <typename T>
std::shared_ptr<holoscan::Tensor> make_iq_cube_device(
    const std::vector<int64_t>& shape) {
  const auto host = deterministic_iq_host<T>(shape);
  auto tensor = make_device_tensor(shape, complex_dtype<T>());
  if (cudaMemcpy(tensor->data(), host.data(),
                 sizeof(std::complex<T>) * host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("make_iq_cube_device: cudaMemcpy H2D failed");
  }
  return tensor;
}

}  // namespace holoscan::radar::perf
