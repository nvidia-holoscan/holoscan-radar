// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared scaffolding for the sparse DOA-estimator operator tests (MUSIC,
// Root-MUSIC, ESPRIT). These operators share an identical front-end contract
// (snapshot in, ArrayGeometry + AngleConvention parameters), so their tests
// build the same synthetic inputs: ULA/URA geometries, analytic steering
// vectors, one/two-source snapshot blocks, and CUDA-accessible tensor wrappers.
// Kept here (rather than copied per file) so a fix to the signal model or the
// DLPack lifetime handling lands once.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace holoscan::radar::testing {

// Re-exported so the estimator tests can keep saying `testing::kPi`; the single
// definition lives in radar_common/math/constants.hpp.
using holoscan::radar::kPi;

// Owns a CUDA-accessible (managed or pinned-host) buffer for the lifetime of
// the holoscan::Tensor that wraps it; freed by the DLPack deleter.
namespace cuda_accessible {

struct BufferHolder {
  void* data{nullptr};
  std::vector<int64_t> shape;
  DLDevice device{kDLCUDA, 0};
  DLManagedTensor dl{};

  ~BufferHolder() {
    if (device.device_type == kDLCUDAHost) {
      cudaFreeHost(data);
    } else {
      cudaFree(data);
    }
  }
};

template <typename ElementT>
std::shared_ptr<holoscan::Tensor> make_tensor(
    const std::vector<ElementT>& host, const std::vector<int64_t>& shape,
    DLDataType dtype, DLDeviceType device_type) {
  auto holder = std::make_unique<BufferHolder>();
  holder->shape = shape;
  const size_t nbytes = host.size() * sizeof(ElementT);
  int device_id = 0;
  cudaGetDevice(&device_id);
  holder->device = DLDevice{device_type, device_id};
  if (device_type == kDLCUDAManaged) {
    if (cudaMallocManaged(&holder->data, nbytes) != cudaSuccess) {
      throw std::runtime_error("cuda_accessible: cudaMallocManaged failed");
    }
  } else if (device_type == kDLCUDAHost) {
    if (cudaHostAlloc(&holder->data, nbytes, cudaHostAllocMapped) !=
        cudaSuccess) {
      throw std::runtime_error("cuda_accessible: cudaHostAlloc failed");
    }
  } else {
    throw std::invalid_argument("cuda_accessible: unsupported device type");
  }
  std::memcpy(holder->data, host.data(), nbytes);
  holder->dl.dl_tensor.data = holder->data;
  holder->dl.dl_tensor.device = holder->device;
  holder->dl.dl_tensor.ndim = static_cast<int32_t>(shape.size());
  holder->dl.dl_tensor.dtype = dtype;
  holder->dl.dl_tensor.shape = holder->shape.data();
  holder->dl.dl_tensor.strides = nullptr;
  holder->dl.dl_tensor.byte_offset = 0;
  holder->dl.manager_ctx = holder.get();
  holder->dl.deleter = [](DLManagedTensor* self) {
    delete static_cast<BufferHolder*>(self->manager_ctx);
  };
  auto tensor = std::make_shared<holoscan::Tensor>(&holder->dl);
  holder.release();
  return tensor;
}

}  // namespace cuda_accessible

// Half-wavelength-spaced ULA along +y (broadside +x), tagged kUla.
inline ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing,
                                       double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = ArrayGeometryType::kUla;
  geom.element_positions.reserve(static_cast<size_t>(num_elements * 3));
  for (int64_t m = 0; m < num_elements; ++m) {
    geom.element_positions.push_back(0.0);
    geom.element_positions.push_back(static_cast<double>(m) * spacing);
    geom.element_positions.push_back(0.0);
  }
  return geom;
}

// Filled URA on the y-z plane (element order iz outer, iy inner), tagged kUra.
inline ArrayGeometry make_planar_ura_geometry(int64_t ny, int64_t nz,
                                              double spacing,
                                              double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = ArrayGeometryType::kUra;
  geom.element_positions.reserve(static_cast<size_t>(ny * nz * 3));
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      geom.element_positions.push_back(0.0);
      geom.element_positions.push_back(static_cast<double>(iy) * spacing);
      geom.element_positions.push_back(static_cast<double>(iz) * spacing);
    }
  }
  return geom;
}

// Unit-amplitude array manifold vector a(az, el) for the given geometry and
// angle convention (boundary angles mapped to the canonical direction first).
inline std::vector<std::complex<double>> steering_vector(
    const ArrayGeometry& geom, const AngleConvention& conv, double az,
    double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(conv, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> a(
      static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase = k * (geom.element_positions[base] * dir.x +
                            geom.element_positions[base + 1] * dir.y +
                            geom.element_positions[base + 2] * dir.z);
    a[static_cast<size_t>(m)] = {std::cos(phase), std::sin(phase)};
  }
  return a;
}

// One-source snapshot block (channels x snapshots, row-major) with a
// deterministic per-snapshot source phase plus complex Gaussian noise.
inline std::vector<std::complex<double>> make_block(const ArrayGeometry& geom,
                                                    const AngleConvention& conv,
                                                    double az, double el,
                                                    int64_t S, double noise,
                                                    unsigned seed) {
  const auto a = steering_vector(geom, conv, az, el);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double gp =
        2.0 * kPi * static_cast<double>(s) / static_cast<double>(S);
    const std::complex<double> g{std::cos(gp), std::sin(gp)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v = a[static_cast<size_t>(c)] * g;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
}

// Two-source snapshot block with independent uniform-random per-snapshot
// source phases (decorrelated sources) plus complex Gaussian noise.
inline std::vector<std::complex<double>> make_block_two(
    const ArrayGeometry& geom, const AngleConvention& conv, double az0,
    double el0, double az1, double el1, int64_t S, double noise,
    unsigned seed) {
  const auto a0 = steering_vector(geom, conv, az0, el0);
  const auto a1 = steering_vector(geom, conv, az1, el1);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double p0 = phase(rng);
    const double p1 = phase(rng);
    const std::complex<double> g0{std::cos(p0), std::sin(p0)};
    const std::complex<double> g1{std::cos(p1), std::sin(p1)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v =
          a0[static_cast<size_t>(c)] * g0 + a1[static_cast<size_t>(c)] * g1;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
}

// Cast a complex<double> host block to the instantiation precision T and
// upload it as a complex device tensor with the given shape.
template <typename T>
std::shared_ptr<holoscan::Tensor> block_to_device(
    const std::vector<std::complex<double>>& host,
    const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    cast[i] = {static_cast<T>(host[i].real()), static_cast<T>(host[i].imag())};
  }
  return to_device_tensor(cast, shape, complex_dtype<T>());
}

}  // namespace holoscan::radar::testing
