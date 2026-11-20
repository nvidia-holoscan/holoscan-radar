// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/steering_weights_device.hpp"

#include <cuda/std/complex>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

// One thread per (beam, element). Computes taper[m]·exp(±j·k·(pₘ·dir_b)) and
// writes it into the requested layout; invalid beams write zero.
template <typename T>
__global__ void steering_weights_kernel(const double* __restrict__ elem_pos,
                                        const double* __restrict__ dirs,
                                        const unsigned char* __restrict__ beam_valid,
                                        const double* __restrict__ taper, int64_t num_beams,
                                        int64_t num_channels, double k, bool conjugate,
                                        bool channel_major,
                                        cuda::std::complex<T>* __restrict__ out) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_beams * num_channels;
  if (tid >= total) {
    return;
  }
  const int64_t beam = tid / num_channels;
  const int64_t m = tid % num_channels;

  cuda::std::complex<T> val{T{0}, T{0}};
  if (beam_valid[beam] != 0) {
    const double dx = dirs[beam * 3];
    const double dy = dirs[beam * 3 + 1];
    const double dz = dirs[beam * 3 + 2];
    const double px = elem_pos[m * 3];
    const double py = elem_pos[m * 3 + 1];
    const double pz = elem_pos[m * 3 + 2];
    const double phase = k * (px * dx + py * dy + pz * dz);
    const double scale = taper[m];
    double re = scale * cos(phase);
    double im = scale * sin(phase);
    if (conjugate) {
      im = -im;
    }
    val = cuda::std::complex<T>(static_cast<T>(re), static_cast<T>(im));
  }

  const int64_t out_idx = channel_major ? (m * num_beams + beam) : (beam * num_channels + m);
  out[out_idx] = val;
}

// Stream-ordered device scratch that frees itself on the same stream (so the
// free orders after the launch that reads it — no host sync in the builder).
struct DeviceScratch {
  void* ptr{nullptr};
  cudaStream_t stream{};
  DeviceScratch(size_t bytes, cudaStream_t s) : stream(s) {
    if (cudaMallocAsync(&ptr, bytes, stream) != cudaSuccess) {
      throw std::runtime_error("build_steering_weights_device: cudaMallocAsync failed");
    }
  }
  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;
  ~DeviceScratch() {
    if (ptr != nullptr) {
      cudaFreeAsync(ptr, stream);
    }
  }
};

void upload_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("build_steering_weights_device: scratch H2D failed");
  }
}

}  // namespace

template <typename T>
std::shared_ptr<holoscan::Tensor> build_steering_weights_device(
    const ArrayGeometry& geometry, const AngleConvention& convention,
    const std::vector<double>& az_coordinates, const std::vector<double>& el_coordinates,
    const WindowSpec& array_taper, BeamOrder beam_order, SteeringLayout layout, bool conjugate,
    cudaStream_t stream) {
  geometry.validate();
  if (az_coordinates.empty() || el_coordinates.empty()) {
    throw std::invalid_argument(
        "build_steering_weights_device: az_coordinates and el_coordinates must be "
        "non-empty");
  }
  validate_scan_grid_axes(convention, az_coordinates, el_coordinates,
                          "build_steering_weights_device");

  const int64_t num_channels = geometry.num_elements();
  const int64_t num_az = static_cast<int64_t>(az_coordinates.size());
  const int64_t num_el = static_cast<int64_t>(el_coordinates.size());
  const int64_t num_beams = num_az * num_el;

  // Host: geometry-aware taper (O(num_channels)) and per-beam directions +
  // validity (O(num_beams), the convention-dependent part) in the requested
  // beam order.
  const std::vector<double> taper = generate_steering_taper(geometry, array_taper);
  std::vector<double> dirs(static_cast<size_t>(num_beams * 3));
  std::vector<unsigned char> beam_valid(static_cast<size_t>(num_beams));
  const bool dc = convention.uses_direction_cosine();
  for (int64_t ia = 0; ia < num_az; ++ia) {
    for (int64_t ie = 0; ie < num_el; ++ie) {
      const int64_t beam =
          (beam_order == BeamOrder::kAzMajor) ? ia * num_el + ie : ie * num_az + ia;
      const double az_b = az_coordinates[static_cast<size_t>(ia)];
      const double el_b = el_coordinates[static_cast<size_t>(ie)];
      const CanonicalDirection d = boundary_to_canonical(convention, az_b, el_b);
      const bool valid =
          dc ? is_physical_direction_cosine(az_b, el_b) : (d.x * d.x + d.y * d.y + d.z * d.z) > 0.5;
      dirs[static_cast<size_t>(beam * 3)] = d.x;
      dirs[static_cast<size_t>(beam * 3 + 1)] = d.y;
      dirs[static_cast<size_t>(beam * 3 + 2)] = d.z;
      beam_valid[static_cast<size_t>(beam)] = valid ? 1 : 0;
    }
  }

  const std::vector<int64_t> shape = (layout == SteeringLayout::kChannelMajor)
                                         ? std::vector<int64_t>{num_channels, num_beams}
                                         : std::vector<int64_t>{num_beams, num_channels};
  auto out = make_device_tensor(shape, complex_dtype<T>());
  if (num_beams == 0 || num_channels == 0) {
    return out;
  }

  DeviceScratch d_pos(static_cast<size_t>(num_channels * 3) * sizeof(double), stream);
  DeviceScratch d_dirs(static_cast<size_t>(num_beams * 3) * sizeof(double), stream);
  DeviceScratch d_valid(static_cast<size_t>(num_beams) * sizeof(unsigned char), stream);
  DeviceScratch d_taper(static_cast<size_t>(num_channels) * sizeof(double), stream);
  upload_async(d_pos.ptr, geometry.element_positions.data(),
               static_cast<size_t>(num_channels * 3) * sizeof(double), stream);
  upload_async(d_dirs.ptr, dirs.data(), static_cast<size_t>(num_beams * 3) * sizeof(double),
               stream);
  upload_async(d_valid.ptr, beam_valid.data(),
               static_cast<size_t>(num_beams) * sizeof(unsigned char), stream);
  upload_async(d_taper.ptr, taper.data(), static_cast<size_t>(num_channels) * sizeof(double),
               stream);

  auto out_mut = out->mutable_data();
  if (!out_mut) {
    throw std::runtime_error("build_steering_weights_device: output tensor is not writable");
  }
  const double k = kTwoPi / geometry.wavelength;
  const int64_t total = num_beams * num_channels;
  constexpr int block = 256;
  const int grid = static_cast<int>((total + block - 1) / block);
  steering_weights_kernel<T><<<grid, block, 0, stream>>>(
      static_cast<const double*>(d_pos.ptr), static_cast<const double*>(d_dirs.ptr),
      static_cast<const unsigned char*>(d_valid.ptr), static_cast<const double*>(d_taper.ptr),
      num_beams, num_channels, k, conjugate, layout == SteeringLayout::kChannelMajor,
      static_cast<cuda::std::complex<T>*>(*out_mut));
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("build_steering_weights_device: kernel launch failed");
  }
  return out;
}

template std::shared_ptr<holoscan::Tensor> build_steering_weights_device<float>(
    const ArrayGeometry&, const AngleConvention&, const std::vector<double>&,
    const std::vector<double>&, const WindowSpec&, BeamOrder, SteeringLayout, bool, cudaStream_t);
template std::shared_ptr<holoscan::Tensor> build_steering_weights_device<double>(
    const ArrayGeometry&, const AngleConvention&, const std::vector<double>&,
    const std::vector<double>&, const WindowSpec&, BeamOrder, SteeringLayout, bool, cudaStream_t);

}  // namespace holoscan::radar
