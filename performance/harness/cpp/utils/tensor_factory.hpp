// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::perf {

//  TensorMap is gone; ports carry a plain Tensor. Keep a
// shared_ptr alias helper so unported consumers can migrate incrementally.
inline std::shared_ptr<holoscan::Tensor> as_tensor(std::shared_ptr<holoscan::Tensor> tensor) {
  return tensor;
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
    host[static_cast<size_t>(i)] = (i % stride == 0) ? static_cast<T>(500.0) : floor;
  }
  return host;
}

// Allocate a device tensor of `shape` and upload the deterministic power map.
template <typename T>
std::shared_ptr<holoscan::Tensor> make_power_map_device(const std::vector<int64_t>& shape) {
  const auto host = deterministic_power_host<T>(shape);
  auto tensor = make_device_tensor(shape, real_dtype<T>());
  auto mut = tensor->mutable_data();
  if (!mut) {
    throw std::runtime_error(std::string("make_power_map_device: tensor not writable: ") +
                             mut.error().message);
  }
  if (cudaMemcpy(mut.value(), host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
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
std::vector<std::complex<T>> deterministic_iq_host(const std::vector<int64_t>& shape) {
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
    const double phase1 = kTwoPi * k1 * static_cast<double>(s) / static_cast<double>(n > 0 ? n : 1);
    const double phase2 = kTwoPi * k2 * static_cast<double>(s) / static_cast<double>(n > 0 ? n : 1);
    const double re = std::cos(phase1) + 0.5 * std::cos(phase2) + 0.1;
    const double im = std::sin(phase1) + 0.5 * std::sin(phase2);
    host[static_cast<size_t>(i)] = std::complex<T>(static_cast<T>(re), static_cast<T>(im));
  }
  return host;
}

// Deterministic defocused SAR image: one dominant scatterer per range row,
// smeared in azimuth by a quadratic phase error on the phase history.
//
// Two properties the autofocus stage needs, neither of which `iq_cube`
// provides. First, PGA gates each range bin on peak power reaching 10x the row
// mean; `iq_cube`'s smooth multi-tone rows miss that by an order of magnitude,
// so the operator finds no strong scatterers and copies the image through,
// which measures a memcpy rather than the estimator. Second -- and easier to
// miss -- the stage breaks out of its iteration loop the moment an iteration
// fails to improve sharpness. An image built from clean deltas is already
// focused, so iteration 0 drops, the loop exits after one pass, and a sweep
// over `num_iterations` reads flat no matter what it is set to. The phase
// error here gives every iteration something to recover.
//
// Built the way the operator's own tests build theirs: take a delta, transform
// to the phase-history domain, apply exp(i*phi), and come back. Only one row
// is synthesized -- every range row is that row circularly shifted, so the
// scatterer walks along azimuth and the stage's own circular shift does real
// work. The transform is a direct O(n^2) sum against an exact twiddle table
// rather than a stepped recurrence; at these lengths a recurrence drifts
// enough to blur the peak the gate is looking for.
template <typename T>
std::vector<std::complex<T>> deterministic_scatterer_image_host(const std::vector<int64_t>& shape) {
  if (shape.size() != 2)
    throw std::invalid_argument("scatterer_image requires a rank-2 shape");
  const int64_t rows = shape[0];
  const int64_t cols = shape[1];
  if (rows < 1 || cols < 2)
    throw std::invalid_argument("scatterer_image needs at least two azimuth samples");

  std::vector<std::complex<double>> twiddle(static_cast<size_t>(cols));
  for (int64_t m = 0; m < cols; ++m) {
    const double arg = kTwoPi * static_cast<double>(m) / static_cast<double>(cols);
    twiddle[static_cast<size_t>(m)] = std::complex<double>(std::cos(arg), std::sin(arg));
  }

  // The quadratic scales with the aperture so the smear stays a fixed
  // *fraction* of the azimuth extent (about a sixteenth) rather than a fixed
  // number of samples. A fixed few radians would smear across ~3 samples at
  // every profile, and the stage's window -- half the extent by default --
  // would then be 4000 samples of clutter around a 3-sample response at the
  // large profile, which is far outside the regime PGA works in: the estimate
  // degrades, iteration 0 regresses sharpness, and the loop aborts after one
  // pass. Scaling keeps the window-to-smear ratio constant, so every profile
  // exercises the same iteration behaviour and the shape sweep compares like
  // with like. A quadratic of hundreds of radians is unremarkable here; the
  // gradient estimator works on the difference between adjacent bins, which
  // stays small however large the total.
  const double half = 0.5 * static_cast<double>(cols);
  const double amplitude = kPi * static_cast<double>(cols) / 32.0;
  std::vector<std::complex<double>> history(static_cast<size_t>(cols));
  for (int64_t k = 0; k < cols; ++k) {
    const double x = (static_cast<double>(k) - half) / half;
    const double phi = amplitude * x * x;
    history[static_cast<size_t>(k)] = std::complex<double>(std::cos(phi), std::sin(phi));
  }

  std::vector<std::complex<double>> base(static_cast<size_t>(cols));
  const double norm = 1.0 / static_cast<double>(cols);
  for (int64_t c = 0; c < cols; ++c) {
    std::complex<double> acc{0.0, 0.0};
    for (int64_t k = 0; k < cols; ++k) {
      acc += history[static_cast<size_t>(k)] * twiddle[static_cast<size_t>((k * c) % cols)];
    }
    base[static_cast<size_t>(c)] = acc * norm;
  }

  // A noise floor, without which every row carries the identical error and one
  // iteration recovers it exactly -- the stage then exits on convergence and
  // any sweep over `num_iterations` measures the same two passes. Scaled to
  // roughly a tenth of the peak, the level the operator's tests treat as
  // ordinary. Patterned, not random, so a shape maps to one payload forever.
  std::vector<std::complex<T>> host(static_cast<size_t>(rows * cols));
  double peak_mag = 0.0;
  for (const auto& z : base) {
    peak_mag = (std::abs(z) > peak_mag) ? std::abs(z) : peak_mag;
  }
  const double noise = 0.1 * peak_mag;
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t peak = (r * 7 + 3) % cols;
    for (int64_t c = 0; c < cols; ++c) {
      const auto z = base[static_cast<size_t>(((c - peak) % cols + cols) % cols)];
      // A scramble, not a stride: a linear step in `c` is a pure tone, i.e. a
      // coherent interferer the estimator will happily lock onto.
      uint64_t h = static_cast<uint64_t>(r * cols + c) + 0x9E3779B97F4A7C15ULL;
      h ^= h >> 30;
      h *= 0xBF58476D1CE4E5B9ULL;
      h ^= h >> 27;
      h *= 0x94D049BB133111EBULL;
      h ^= h >> 31;
      const double arg = kTwoPi * static_cast<double>(h >> 11) / 9007199254740992.0;
      host[static_cast<size_t>(r * cols + c)] =
          std::complex<T>(static_cast<T>(z.real() + noise * std::cos(arg)),
                          static_cast<T>(z.imag() + noise * std::sin(arg)));
    }
  }
  return host;
}

// Allocate a complex64/complex128 device tensor of `shape` and upload the
// deterministic scatterer image.
template <typename T>
std::shared_ptr<holoscan::Tensor> make_scatterer_image_device(const std::vector<int64_t>& shape) {
  const auto host = deterministic_scatterer_image_host<T>(shape);
  auto tensor = make_device_tensor(shape, complex_dtype<T>());
  auto mut = tensor->mutable_data();
  if (!mut) {
    throw std::runtime_error(std::string("make_scatterer_image_device: tensor not writable: ") +
                             mut.error().message);
  }
  if (cudaMemcpy(mut.value(), host.data(), sizeof(std::complex<T>) * host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("make_scatterer_image_device: cudaMemcpy H2D failed");
  }
  return tensor;
}

// Allocate a complex64/complex128 device tensor of `shape` and upload the
// deterministic IQ payload. Used for adc_cube (range/fused) and range_profiles
// (Doppler/clutter) inputs.
template <typename T>
std::shared_ptr<holoscan::Tensor> make_iq_cube_device(const std::vector<int64_t>& shape) {
  const auto host = deterministic_iq_host<T>(shape);
  auto tensor = make_device_tensor(shape, complex_dtype<T>());
  auto mut = tensor->mutable_data();
  if (!mut) {
    throw std::runtime_error(std::string("make_iq_cube_device: tensor not writable: ") +
                             mut.error().message);
  }
  if (cudaMemcpy(mut.value(), host.data(), sizeof(std::complex<T>) * host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("make_iq_cube_device: cudaMemcpy H2D failed");
  }
  return tensor;
}

}  // namespace holoscan::radar::perf
