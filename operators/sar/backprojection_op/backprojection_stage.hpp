// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/math/constants.hpp"
#include "sar/backprojection_op/nvradar_kernels.cuh"
#include "sar/backprojection_op/nvradar_kernels_fp32.cuh"
#include "sar/backprojection_op/nvradar_kernels_lut.cuh"
#include "sar/backprojection_op/nvradar_kernels_taylor.cuh"

namespace holoscan::radar::detail {

// Cached device allocation. Reuses capacity; never stream-ordered (the
// range-compression teardown crash). Per-tick cudaMalloc is the ~0.1 ms
// floor already measured on the reader.
struct DeviceBuffer {
  void* ptr{nullptr};
  size_t bytes{0};

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  // `stream` is the stream that last used this buffer. Grow waits on
  // it before cudaFree so a larger lattice cannot free in-flight work.
  void ensure(size_t need, cudaStream_t stream) {
    if (need <= bytes) {
      return;
    }
    if (ptr != nullptr) {
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("BackprojectionOp: stream sync before realloc failed");
      }
      cudaFree(ptr);
      ptr = nullptr;
      bytes = 0;
    }
    if (need == 0) {
      return;
    }
    if (cudaMalloc(&ptr, need) != cudaSuccess) {
      throw std::runtime_error("BackprojectionOp: device allocation failed");
    }
    bytes = need;
  }

  template <typename U>
  U* as() {
    return static_cast<U*>(ptr);
  }
};

inline bool backend_is_matx(BackprojectionBackend backend) {
  switch (backend) {
    case BackprojectionBackend::kMatxDouble:
    case BackprojectionBackend::kMatxDoublePhaseLut:
    case BackprojectionBackend::kMatxMixed:
    case BackprojectionBackend::kMatxMixedPhaseLut:
    case BackprojectionBackend::kMatxFloatFloat:
    case BackprojectionBackend::kMatxFloat:
    case BackprojectionBackend::kMatxFloatPhaseLut:
      return true;
    default:
      return false;
  }
}

inline bool backend_is_implemented(BackprojectionBackend backend) {
  if (backend == BackprojectionBackend::kMatxTaylorFast) {
    return false;
  }
  if (backend_is_matx(backend)) {
#if __has_include("matx/kernels/sar_bp.cuh")
    return true;
#else
    return false;
#endif
  }
  return true;
}

template <typename T, typename TGeom = T>
struct BackprojectionState {
  DeviceBuffer lattice_xyz;  // (num_v * num_u * 3) TGeom, cached
  DeviceBuffer x_plane;      // double, num_pixels
  DeviceBuffer y_plane;
  DeviceBuffer z_plane;
  DeviceBuffer apc;        // double, 3 * num_pulses
  DeviceBuffer mcp;        // double, 3 * num_pulses (reference)
  DeviceBuffer r0;         // double, num_pulses
  DeviceBuffer x_plane_f;  // float lattice / APC / R0 for fp32 family
  DeviceBuffer y_plane_f;
  DeviceBuffer z_plane_f;
  DeviceBuffer apc_f;
  DeviceBuffer mcp_f;
  DeviceBuffer r0_f;
  DeviceBuffer phase_lut;  // interleaved complex64, 2 * Nfft
  double lut_pc{0.0};
  double lut_rmin{0.0};
  double lut_dr{0.0};
  int lut_nfft{0};
  DeviceBuffer weight_scratch;  // complex<T>, profiles
  DeviceBuffer vph_cast;        // upcast/downcast profiles
  DeviceBuffer accum;           // float, 2 * batch * num_v * num_u (NVRadar)
  DeviceBuffer matx_accum;      // complex<T>, batch * num_v * num_u
  DeviceBuffer cmp_flag;        // int, sample_points equality
  DeviceBuffer tile_diam;       // double, I7 points-mode tile diameters
  int64_t lattice_num_v{0};     // xyz / planes, independent of accum
  int64_t lattice_num_u{0};
  int64_t cached_num_v{0};  // accumulator shape
  int64_t cached_num_u{0};
  int64_t cached_batch{0};
  // Physical row of logical v=0. Non-zero only under scroll.
  int64_t ring_offset{0};
  // Slab start for the current kernel launch (two launches wrap).
  int64_t launch_row0{0};
  bool lattice_valid{false};
  bool planes_valid{false};

  // Per-tick GPU spans for px*pls/s. Host wall around compute() only
  // sees launch overhead on non-emit ticks (the runtime finishes the
  // partition stream after return); video SAR emits every N ticks, so
  // that under-counts time by ~N and inflates the HUD rate.
  struct GpuTickEvents {
    cudaEvent_t start{nullptr};
    cudaEvent_t stop{nullptr};
  };
  // Caps CUDA event growth when control is unwired and no image is emitted.
  // Do not fold with take_tick_gpu_s() on the record path: that GPU-syncs.
  static constexpr size_t kMaxTickGpuEvents = 16384;
  std::vector<GpuTickEvents> tick_events;
  size_t tick_event_n{0};
  bool tick_gpu_armed_{false};

  ~BackprojectionState() {
    for (auto& ev : tick_events) {
      if (ev.start != nullptr)
        cudaEventDestroy(ev.start);
      if (ev.stop != nullptr)
        cudaEventDestroy(ev.stop);
    }
  }

  void discard_tick_gpu_timing() {
    tick_event_n = 0;
    tick_gpu_armed_ = false;
  }

  void record_tick_gpu_start(cudaStream_t stream) {
    if (tick_event_n >= kMaxTickGpuEvents) {
      tick_gpu_armed_ = false;
      return;
    }
    if (tick_event_n >= tick_events.size()) {
      GpuTickEvents ev;
      if (cudaEventCreate(&ev.start) != cudaSuccess || cudaEventCreate(&ev.stop) != cudaSuccess) {
        throw std::runtime_error("BackprojectionOp: cudaEventCreate failed");
      }
      tick_events.push_back(ev);
    }
    if (cudaEventRecord(tick_events[tick_event_n].start, stream) != cudaSuccess) {
      throw std::runtime_error("BackprojectionOp: cudaEventRecord start failed");
    }
    tick_gpu_armed_ = true;
  }

  void record_tick_gpu_stop(cudaStream_t stream) {
    if (!tick_gpu_armed_)
      return;
    if (cudaEventRecord(tick_events[tick_event_n].stop, stream) != cudaSuccess) {
      throw std::runtime_error("BackprojectionOp: cudaEventRecord stop failed");
    }
    ++tick_event_n;
    tick_gpu_armed_ = false;
  }

  double take_tick_gpu_s() {
    float total_ms = 0.0F;
    for (size_t i = 0; i < tick_event_n; ++i) {
      float ms = 0.0F;
      if (cudaEventElapsedTime(&ms, tick_events[i].start, tick_events[i].stop) != cudaSuccess) {
        throw std::runtime_error("BackprojectionOp: cudaEventElapsedTime failed");
      }
      total_ms += ms;
    }
    tick_event_n = 0;
    tick_gpu_armed_ = false;
    return static_cast<double>(total_ms) * 1.0e-3;
  }

  double* x_d() { return x_plane.as<double>() + static_cast<size_t>(launch_row0 * lattice_num_u); }
  double* y_d() { return y_plane.as<double>() + static_cast<size_t>(launch_row0 * lattice_num_u); }
  double* z_d() { return z_plane.as<double>() + static_cast<size_t>(launch_row0 * lattice_num_u); }
  float* x_f() { return x_plane_f.as<float>() + static_cast<size_t>(launch_row0 * lattice_num_u); }
  float* y_f() { return y_plane_f.as<float>() + static_cast<size_t>(launch_row0 * lattice_num_u); }
  float* z_f() { return z_plane_f.as<float>() + static_cast<size_t>(launch_row0 * lattice_num_u); }
};

__global__ void gather_xyz_kernel(const float* lattice, int64_t n, double* x, double* y,
                                  double* z) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  x[i] = static_cast<double>(lattice[i * 3 + 0]);
  y[i] = static_cast<double>(lattice[i * 3 + 1]);
  z[i] = static_cast<double>(lattice[i * 3 + 2]);
}

__global__ void gather_xyz_kernel(const double* lattice, int64_t n, double* x, double* y,
                                  double* z) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  x[i] = lattice[i * 3 + 0];
  y[i] = lattice[i * 3 + 1];
  z[i] = lattice[i * 3 + 2];
}

template <typename T>
__global__ void mark_if_differ_kernel(const T* a, const T* b, int64_t n, int* flag) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (a[i] != b[i]) {
    *flag = 1;
  }
}

// AABB diagonal of each 16x16 image tile — I7 points-mode block
// diameter. Conservative vs pairwise max, cheap, and matches the
// Taylor kernel's 16x16 launch.
__global__ void tile_aabb_diameter_kernel(const double* x, const double* y, const double* z,
                                          int num_v, int num_u, double* out) {
  const int tile_u = (num_u + 15) / 16;
  const int tx = static_cast<int>(blockIdx.x);
  const int ty = static_cast<int>(blockIdx.y);
  const int i0 = ty * 16;
  const int j0 = tx * 16;
  double minx = 0.0;
  double miny = 0.0;
  double minz = 0.0;
  double maxx = 0.0;
  double maxy = 0.0;
  double maxz = 0.0;
  bool any = false;
  for (int dy = 0; dy < 16; ++dy) {
    const int iv = i0 + dy;
    if (iv >= num_v) {
      break;
    }
    for (int dx = 0; dx < 16; ++dx) {
      const int iu = j0 + dx;
      if (iu >= num_u) {
        break;
      }
      const int idx = iv * num_u + iu;
      const double px = x[idx];
      const double py = y[idx];
      const double pz = z[idx];
      if (!any) {
        minx = maxx = px;
        miny = maxy = py;
        minz = maxz = pz;
        any = true;
      } else {
        minx = fmin(minx, px);
        miny = fmin(miny, py);
        minz = fmin(minz, pz);
        maxx = fmax(maxx, px);
        maxy = fmax(maxy, py);
        maxz = fmax(maxz, pz);
      }
    }
  }
  const int o = ty * tile_u + tx;
  if (!any) {
    out[o] = 0.0;
    return;
  }
  const double dx = maxx - minx;
  const double dy = maxy - miny;
  const double dz = maxz - minz;
  out[o] = sqrt(dx * dx + dy * dy + dz * dz);
}

template <typename T>
bool device_arrays_differ(const T* a, const T* b, int64_t n, DeviceBuffer& flag,
                          cudaStream_t stream) {
  flag.ensure(sizeof(int), stream);
  if (cudaMemsetAsync(flag.ptr, 0, sizeof(int), stream) != cudaSuccess) {
    throw std::runtime_error("BackprojectionOp: sample_points compare memset failed");
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + static_cast<int64_t>(threads) - 1) / threads);
  mark_if_differ_kernel<T><<<blocks, threads, 0, stream>>>(a, b, n, flag.template as<int>());
  int host = 0;
  if (cudaMemcpyAsync(&host, flag.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream) !=
      cudaSuccess) {
    throw std::runtime_error("BackprojectionOp: sample_points compare copy failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("BackprojectionOp: sample_points compare sync failed");
  }
  return host != 0;
}

// Worst-case 16x16-tile diameter over x/y/z planes.
template <typename T, typename TGeom>
double max_tile_diameter(BackprojectionState<T, TGeom>& state, int64_t num_v, int64_t num_u,
                         cudaStream_t stream) {
  const int tiles_v = static_cast<int>((num_v + 15) / 16);
  const int tiles_u = static_cast<int>((num_u + 15) / 16);
  const int ntiles = tiles_v * tiles_u;
  state.tile_diam.ensure(static_cast<size_t>(ntiles) * sizeof(double), stream);
  const dim3 grid(static_cast<unsigned>(tiles_u), static_cast<unsigned>(tiles_v));
  tile_aabb_diameter_kernel<<<grid, 1, 0, stream>>>(
      state.x_d(), state.y_d(), state.z_d(), static_cast<int>(num_v), static_cast<int>(num_u),
      state.tile_diam.template as<double>());
  std::vector<double> host(static_cast<size_t>(ntiles));
  if (cudaMemcpyAsync(host.data(), state.tile_diam.ptr, host.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error("BackprojectionOp: tile-diameter copy failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("BackprojectionOp: tile-diameter sync failed");
  }
  double dmax = 0.0;
  for (double d : host) {
    dmax = std::max(dmax, d);
  }
  return dmax;
}

// Device-side SampleGeometry::sample_position:
//   origin + iv * spacing[0] * axis_v + iu * spacing[1] * axis_u
// iv is the slow (row) index, iu the fast (column) index.
__global__ void expand_lattice_planes_kernel(double3 origin, double3 axis_v, double3 axis_u,
                                             double spacing_v, double spacing_u, int64_t num_v,
                                             int64_t num_u, double* x, double* y, double* z) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = num_v * num_u;
  if (i >= n) {
    return;
  }
  const int64_t iv = i / num_u;
  const int64_t iu = i - iv * num_u;
  const double dv = static_cast<double>(iv) * spacing_v;
  const double du = static_cast<double>(iu) * spacing_u;
  x[i] = origin.x + dv * axis_v.x + du * axis_u.x;
  y[i] = origin.y + dv * axis_v.y + du * axis_u.y;
  z[i] = origin.z + dv * axis_v.z + du * axis_u.z;
}

template <typename T>
__global__ void expand_lattice_packed_kernel(double3 origin, double3 axis_v, double3 axis_u,
                                             double spacing_v, double spacing_u, int64_t num_v,
                                             int64_t num_u, T* out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = num_v * num_u;
  if (i >= n) {
    return;
  }
  const int64_t iv = i / num_u;
  const int64_t iu = i - iv * num_u;
  const double dv = static_cast<double>(iv) * spacing_v;
  const double du = static_cast<double>(iu) * spacing_u;
  out[i * 3 + 0] = static_cast<T>(origin.x + dv * axis_v.x + du * axis_u.x);
  out[i * 3 + 1] = static_cast<T>(origin.y + dv * axis_v.y + du * axis_u.y);
  out[i * 3 + 2] = static_cast<T>(origin.z + dv * axis_v.z + du * axis_u.z);
}

// Write n_rows of a SampleGeometry into a ring-ordered plane buffer.
// logical_v0 is the first logical row (in the new geometry);
// phys_v0 is the physical slot those rows occupy (may wrap).
__global__ void expand_lattice_plane_rows_kernel(double3 origin, double3 axis_v, double3 axis_u,
                                                 double spacing_v, double spacing_u, int64_t num_v,
                                                 int64_t num_u, int64_t logical_v0, int64_t n_rows,
                                                 int64_t phys_v0, double* x, double* y, double* z) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = n_rows * num_u;
  if (i >= n) {
    return;
  }
  const int64_t r = i / num_u;
  const int64_t iu = i - r * num_u;
  const int64_t iv = logical_v0 + r;
  int64_t pv = phys_v0 + r;
  if (pv >= num_v) {
    pv -= num_v;
  }
  const int64_t o = pv * num_u + iu;
  const double dv = static_cast<double>(iv) * spacing_v;
  const double du = static_cast<double>(iu) * spacing_u;
  x[o] = origin.x + dv * axis_v.x + du * axis_u.x;
  y[o] = origin.y + dv * axis_v.y + du * axis_u.y;
  z[o] = origin.z + dv * axis_v.z + du * axis_u.z;
}

template <typename T>
__global__ void expand_lattice_packed_rows_kernel(double3 origin, double3 axis_v, double3 axis_u,
                                                  double spacing_v, double spacing_u, int64_t num_v,
                                                  int64_t num_u, int64_t logical_v0, int64_t n_rows,
                                                  int64_t phys_v0, T* out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = n_rows * num_u;
  if (i >= n) {
    return;
  }
  const int64_t r = i / num_u;
  const int64_t iu = i - r * num_u;
  const int64_t iv = logical_v0 + r;
  int64_t pv = phys_v0 + r;
  if (pv >= num_v) {
    pv -= num_v;
  }
  const int64_t o = pv * num_u + iu;
  const double dv = static_cast<double>(iv) * spacing_v;
  const double du = static_cast<double>(iu) * spacing_u;
  out[o * 3 + 0] = static_cast<T>(origin.x + dv * axis_v.x + du * axis_u.x);
  out[o * 3 + 1] = static_cast<T>(origin.y + dv * axis_v.y + du * axis_u.y);
  out[o * 3 + 2] = static_cast<T>(origin.z + dv * axis_v.z + du * axis_u.z);
}

template <typename T>
__global__ void compute_r0_apc_kernel(const T* pos, const T* ref, int Np, double* apc, double* mcp,
                                      double* r0) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= Np) {
    return;
  }
  const double px = static_cast<double>(pos[p * 3 + 0]);
  const double py = static_cast<double>(pos[p * 3 + 1]);
  const double pz = static_cast<double>(pos[p * 3 + 2]);
  const double rx = static_cast<double>(ref[p * 3 + 0]);
  const double ry = static_cast<double>(ref[p * 3 + 1]);
  const double rz = static_cast<double>(ref[p * 3 + 2]);
  apc[p] = px;
  apc[p + Np] = py;
  apc[p + 2 * Np] = pz;
  mcp[p] = rx;
  mcp[p + Np] = ry;
  mcp[p + 2 * Np] = rz;
  const double dx = px - rx;
  const double dy = py - ry;
  const double dz = pz - rz;
  r0[p] = sqrt(dx * dx + dy * dy + dz * dz);
}

// Interleaved complex scale in place on a scratch copy.
template <typename T>
__global__ void scale_interleaved_kernel(const T* src, const T* weight, T* dst, int Np, int N) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = Np * N;
  if (idx >= total) {
    return;
  }
  const int p = idx / N;
  const T w = weight[p];
  dst[idx * 2] = src[idx * 2] * w;
  dst[idx * 2 + 1] = src[idx * 2 + 1] * w;
}

__global__ void upcast_vph_kernel(const float* src, double* dst, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  dst[i] = static_cast<double>(src[i]);
}

__global__ void downcast_vph_kernel(const double* src, float* dst, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  dst[i] = static_cast<float>(src[i]);
}

__global__ void downcast_real_kernel(const double* src, float* dst, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  dst[i] = static_cast<float>(src[i]);
}

__global__ void zero_float_kernel(float* dst, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] = 0.0f;
}

__global__ void promote_image_kernel(const float* src, float* dst, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] = src[i];
}

__global__ void promote_image_kernel(const float* src, double* dst, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] = static_cast<double>(src[i]);
}

// Host-side pulse tile for kernel launches. Not a contract parameter:
// the operator still consumes the whole tick. 1024 keeps a mode-5
// pulses_per_tick tick as one launch (256 forced an extra simg write
// per inner smem chunk). Ticks with Np <= this stay one launch.
inline constexpr int kKernelPulseTile = 1024;

inline dim3 bp_grid(int img_width, int img_height) {
  return dim3((img_width + 15) / 16, (img_height + 15) / 16, 1);
}

inline dim3 bp_block() {
  return dim3(16, 16, 1);
}

template <typename T, typename TGeom>
void ensure_accum(BackprojectionState<T, TGeom>& state, int64_t batch, int64_t num_v, int64_t num_u,
                  cudaStream_t stream) {
  const int64_t n = batch * num_v * num_u * 2;
  const bool resize =
      state.cached_batch != batch || state.cached_num_v != num_v || state.cached_num_u != num_u;
  state.accum.ensure(static_cast<size_t>(n) * sizeof(float), stream);
  if (resize) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    zero_float_kernel<<<blocks, threads, 0, stream>>>(state.accum.template as<float>(), n);
    state.cached_batch = batch;
    state.cached_num_v = num_v;
    state.cached_num_u = num_u;
    state.ring_offset = 0;
    state.launch_row0 = 0;
  }
}

template <typename T, typename TGeom>
void reset_accum(BackprojectionState<T, TGeom>& state, cudaStream_t stream) {
  if (state.accum.ptr == nullptr || state.cached_batch == 0) {
    return;
  }
  const int64_t n = state.cached_batch * state.cached_num_v * state.cached_num_u * 2;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  zero_float_kernel<<<blocks, threads, 0, stream>>>(state.accum.template as<float>(), n);
  state.ring_offset = 0;
  state.launch_row0 = 0;
}

inline dim3 lattice_launch(int64_t num_pixels) {
  const int threads = 256;
  const int blocks = static_cast<int>((num_pixels + threads - 1) / threads);
  return dim3(static_cast<unsigned>(blocks), 1, 1);
}

inline void geometry_to_double3(const SampleGeometry& geometry, double3* origin, double3* axis_v,
                                double3* axis_u, double* spacing_v, double* spacing_u) {
  const auto& o = geometry.origin();
  const auto& av = geometry.axis_v();
  const auto& au = geometry.axis_u();
  const auto& sp = geometry.spacing();
  *origin = make_double3(o[0], o[1], o[2]);
  *axis_v = make_double3(av[0], av[1], av[2]);
  *axis_u = make_double3(au[0], au[1], au[2]);
  *spacing_v = sp[0];
  *spacing_u = sp[1];
}

// Expand a uniform lattice on device. NVRadar consumes x/y/z planes, so
// `packed == false` writes those directly and never materializes the
// (N, 3) buffer (1.61 GiB at Atlanta 8192x8192 double). MatX consumes packed
// xyz, so `packed == true` writes that instead.
template <typename T, typename TGeom>
void expand_geometry_on_device(BackprojectionState<T, TGeom>& state, const SampleGeometry& geometry,
                               bool packed, cudaStream_t stream) {
  const int64_t num_v = geometry.num_v();
  const int64_t num_u = geometry.num_u();
  const int64_t pixels = num_v * num_u;
  double3 origin{};
  double3 axis_v{};
  double3 axis_u{};
  double spacing_v = 0.0;
  double spacing_u = 0.0;
  geometry_to_double3(geometry, &origin, &axis_v, &axis_u, &spacing_v, &spacing_u);
  state.lattice_num_v = num_v;
  state.lattice_num_u = num_u;
  const dim3 grid = lattice_launch(pixels);
  constexpr int kThreads = 256;
  if (packed) {
    state.lattice_xyz.ensure(static_cast<size_t>(pixels) * 3 * sizeof(TGeom), stream);
    expand_lattice_packed_kernel<TGeom>
        <<<grid, kThreads, 0, stream>>>(origin, axis_v, axis_u, spacing_v, spacing_u, num_v, num_u,
                                        state.lattice_xyz.template as<TGeom>());
    return;
  }
  state.x_plane.ensure(static_cast<size_t>(pixels) * sizeof(double), stream);
  state.y_plane.ensure(static_cast<size_t>(pixels) * sizeof(double), stream);
  state.z_plane.ensure(static_cast<size_t>(pixels) * sizeof(double), stream);
  expand_lattice_planes_kernel<<<grid, kThreads, 0, stream>>>(origin, axis_v, axis_u, spacing_v,
                                                              spacing_u, num_v, num_u, state.x_d(),
                                                              state.y_d(), state.z_d());
}

template <typename T, typename TGeom>
void expand_geometry_rows(BackprojectionState<T, TGeom>& state, const SampleGeometry& geometry,
                          bool packed, int64_t logical_v0, int64_t n_rows, int64_t phys_v0,
                          cudaStream_t stream) {
  const int64_t num_v = geometry.num_v();
  const int64_t num_u = geometry.num_u();
  double3 origin{};
  double3 axis_v{};
  double3 axis_u{};
  double spacing_v = 0.0;
  double spacing_u = 0.0;
  geometry_to_double3(geometry, &origin, &axis_v, &axis_u, &spacing_v, &spacing_u);
  const int64_t pixels = n_rows * num_u;
  const dim3 grid = lattice_launch(pixels);
  constexpr int kThreads = 256;
  if (packed) {
    expand_lattice_packed_rows_kernel<TGeom><<<grid, kThreads, 0, stream>>>(
        origin, axis_v, axis_u, spacing_v, spacing_u, num_v, num_u, logical_v0, n_rows, phys_v0,
        state.lattice_xyz.template as<TGeom>());
    return;
  }
  expand_lattice_plane_rows_kernel<<<grid, kThreads, 0, stream>>>(
      origin, axis_v, axis_u, spacing_v, spacing_u, num_v, num_u, logical_v0, n_rows, phys_v0,
      state.x_plane.template as<double>(), state.y_plane.template as<double>(),
      state.z_plane.template as<double>());
}

inline void memset_row_span(void* base, int64_t phys0, int64_t n_rows, int64_t num_v,
                            int64_t row_bytes, cudaStream_t stream, const char* who) {
  if (n_rows <= 0 || row_bytes <= 0) {
    return;
  }
  auto one = [&](int64_t row, int64_t n) {
    unsigned char* p = static_cast<unsigned char*>(base) +
                       static_cast<size_t>(row) * static_cast<size_t>(row_bytes);
    if (cudaMemsetAsync(p, 0, static_cast<size_t>(n) * static_cast<size_t>(row_bytes), stream) !=
        cudaSuccess) {
      throw std::runtime_error(std::string(who) + ": row memset");
    }
  };
  if (phys0 + n_rows <= num_v) {
    one(phys0, n_rows);
    return;
  }
  const int64_t first = num_v - phys0;
  one(phys0, first);
  one(0, n_rows - first);
}

template <typename T, typename TGeom>
void zero_nvradar_rows(BackprojectionState<T, TGeom>& state, int64_t phys0, int64_t n_rows,
                       cudaStream_t stream) {
  const int64_t nv = state.cached_num_v;
  const int64_t nu = state.cached_num_u;
  const int64_t batch = state.cached_batch;
  const int64_t row_bytes = nu * 2 * static_cast<int64_t>(sizeof(float));
  for (int64_t b = 0; b < batch; ++b) {
    float* base = state.accum.template as<float>() + static_cast<size_t>(b) * nv * nu * 2;
    memset_row_span(base, phys0, n_rows, nv, row_bytes, stream, "BackprojectionOp");
  }
}

template <typename T, typename TGeom>
void copy_entering_xyz(BackprojectionState<T, TGeom>& state, const TGeom* src, int64_t k,
                       int64_t num_u, int64_t phys0, cudaStream_t stream) {
  const int64_t nv = state.lattice_num_v;
  const int64_t nu = num_u;
  TGeom* dst = state.lattice_xyz.template as<TGeom>();
  auto copy_rows = [&](int64_t drow, int64_t srow, int64_t n) {
    if (cudaMemcpyAsync(dst + (drow * nu * 3), src + (srow * nu * 3),
                        sizeof(TGeom) * static_cast<size_t>(n * nu * 3), cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
      throw std::runtime_error("BackprojectionOp: entering sample_points copy");
    }
  };
  if (phys0 + k <= nv) {
    copy_rows(phys0, 0, k);
    return;
  }
  const int64_t first = nv - phys0;
  copy_rows(phys0, 0, first);
  copy_rows(0, first, k - first);
}

template <typename T, typename TGeom>
void gather_lattice(BackprojectionState<T, TGeom>& state, const TGeom* lattice, int64_t num_pixels,
                    cudaStream_t stream) {
  state.x_plane.ensure(static_cast<size_t>(num_pixels) * sizeof(double), stream);
  state.y_plane.ensure(static_cast<size_t>(num_pixels) * sizeof(double), stream);
  state.z_plane.ensure(static_cast<size_t>(num_pixels) * sizeof(double), stream);
  const int threads = 256;
  const int blocks = static_cast<int>((num_pixels + threads - 1) / threads);
  gather_xyz_kernel<<<blocks, threads, 0, stream>>>(lattice, num_pixels, state.x_d(), state.y_d(),
                                                    state.z_d());
}

template <typename T, typename TGeom>
void prepare_pulses(BackprojectionState<T, TGeom>& state, const TGeom* positions,
                    const TGeom* reference, int Np, cudaStream_t stream) {
  state.apc.ensure(static_cast<size_t>(Np) * 3 * sizeof(double), stream);
  state.mcp.ensure(static_cast<size_t>(Np) * 3 * sizeof(double), stream);
  state.r0.ensure(static_cast<size_t>(Np) * sizeof(double), stream);
  const int threads = 256;
  const int blocks = (Np + threads - 1) / threads;
  compute_r0_apc_kernel<TGeom><<<blocks, threads, 0, stream>>>(
      positions, reference, Np, state.apc.template as<double>(), state.mcp.template as<double>(),
      state.r0.template as<double>());
}

template <typename T, typename TGeom>
void prepare_float_scene(BackprojectionState<T, TGeom>& state, int64_t num_pixels, int Np,
                         cudaStream_t stream) {
  const int64_t full = state.lattice_num_v * state.lattice_num_u;
  const int64_t off = state.launch_row0 * state.lattice_num_u;
  auto downcast = [&](DeviceBuffer& src, DeviceBuffer& dst, int n, size_t bytes, int64_t poff) {
    dst.ensure(bytes, stream);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    downcast_real_kernel<<<blocks, threads, 0, stream>>>(src.template as<double>() + poff,
                                                         dst.template as<float>() + poff, n);
  };
  const size_t plane_bytes = static_cast<size_t>(full) * sizeof(float);
  downcast(state.x_plane, state.x_plane_f, static_cast<int>(num_pixels), plane_bytes, off);
  downcast(state.y_plane, state.y_plane_f, static_cast<int>(num_pixels), plane_bytes, off);
  downcast(state.z_plane, state.z_plane_f, static_cast<int>(num_pixels), plane_bytes, off);
  downcast(state.apc, state.apc_f, Np * 3, static_cast<size_t>(Np * 3) * sizeof(float), 0);
  downcast(state.mcp, state.mcp_f, Np * 3, static_cast<size_t>(Np * 3) * sizeof(float), 0);
  downcast(state.r0, state.r0_f, Np, static_cast<size_t>(Np) * sizeof(float), 0);
}

template <typename T, typename TGeom>
void ensure_nvradar_lut(BackprojectionState<T, TGeom>& state, double pc_partial, double rmin,
                        double dr, int Nfft, cudaStream_t stream) {
  const bool same = state.lut_nfft == Nfft && state.lut_pc == pc_partial &&
                    state.lut_rmin == rmin && state.lut_dr == dr;
  state.phase_lut.ensure(static_cast<size_t>(Nfft) * 2 * sizeof(float), stream);
  if (same && state.phase_lut.ptr != nullptr) {
    return;
  }
  const int threads = 256;
  const int blocks = (Nfft + threads - 1) / threads;
  nvradar_fill_phase_lut<<<blocks, threads, 0, stream>>>(state.phase_lut.template as<float>(),
                                                         pc_partial, rmin, dr, Nfft);
  state.lut_pc = pc_partial;
  state.lut_rmin = rmin;
  state.lut_dr = dr;
  state.lut_nfft = Nfft;
}

template <typename T, typename TGeom>
void launch_nvradar(BackprojectionBackend backend, BackprojectionState<T, TGeom>& state,
                    const void* profiles, int num_v, int num_u, int Np, int Nfft, int batch_index,
                    double rmin, double dr, double pc_partial, cudaStream_t stream, int row0 = 0,
                    int full_v = -1) {
  if (full_v < 1) {
    full_v = num_v;
  }
  struct LaunchRowGuard {
    BackprojectionState<T, TGeom>& state;
    explicit LaunchRowGuard(BackprojectionState<T, TGeom>& s, int64_t row) : state(s) {
      state.launch_row0 = row;
    }
    ~LaunchRowGuard() { state.launch_row0 = 0; }
  };
  LaunchRowGuard guard(state, row0);
  const int img_width = num_u;
  const int img_height = num_v;
  const int pixels = num_v * num_u;
  float* simg =
      state.accum.template as<float>() +
      (static_cast<size_t>(batch_index) * static_cast<size_t>(full_v) + static_cast<size_t>(row0)) *
          static_cast<size_t>(num_u) * 2;
  const dim3 grid = bp_grid(img_width, img_height);
  const dim3 block = bp_block();
  const int stride = Np;
  const size_t vph_pulse = static_cast<size_t>(Nfft) * 2;
  // Kernels += into simg. Planar APC/R0 keep the tick stride; packed
  // vph is a pointer offset. Np <= kKernelPulseTile is one launch.
  auto tiles = [&](auto&& launch) {
    for (int p0 = 0; p0 < Np; p0 += kKernelPulseTile) {
      const int n = std::min(kKernelPulseTile, Np - p0);
      launch(p0, n);
    }
  };

  if (backend == BackprojectionBackend::kNvradarFp64Reference) {
    const double* vph = nullptr;
    if constexpr (std::is_same_v<T, double>) {
      vph = static_cast<const double*>(profiles);
    } else {
      const int n = Np * Nfft * 2;
      state.vph_cast.ensure(static_cast<size_t>(n) * sizeof(double), stream);
      const int threads = 256;
      const int blocks = (n + threads - 1) / threads;
      upcast_vph_kernel<<<blocks, threads, 0, stream>>>(static_cast<const float*>(profiles),
                                                        state.vph_cast.template as<double>(), n);
      vph = state.vph_cast.template as<double>();
    }
    tiles([&](int p0, int n) {
      backprojection_kernel_fp64_ref<<<grid, block, 0, stream>>>(
          0, img_width, img_height, n, stride, rmin, dr, pc_partial, simg, nullptr, state.x_d(),
          state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
          state.r0.template as<double>() + p0, vph + p0 * vph_pulse, Nfft);
    });
    return;
  }

  auto float_vph = [&]() -> const float* {
    if constexpr (std::is_same_v<T, float>) {
      return static_cast<const float*>(profiles);
    }
    const int n = Np * Nfft * 2;
    state.vph_cast.ensure(static_cast<size_t>(n) * sizeof(float), stream);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    downcast_vph_kernel<<<blocks, threads, 0, stream>>>(static_cast<const double*>(profiles),
                                                        state.vph_cast.template as<float>(), n);
    return state.vph_cast.template as<float>();
  };

  if (backend == BackprojectionBackend::kNvradarCudaFp32 ||
      backend == BackprojectionBackend::kNvradarCudaFp32Hq ||
      backend == BackprojectionBackend::kNvradarCudaFp32HqFast) {
    prepare_float_scene(state, pixels, Np, stream);
    const float* vph = float_vph();
    const float rmin_f = static_cast<float>(rmin);
    const float dr_inv_f = static_cast<float>(1.0 / dr);
    const float pc_f = static_cast<float>(pc_partial);
    if (backend == BackprojectionBackend::kNvradarCudaFp32) {
      tiles([&](int p0, int n) {
        backprojection_kernel_fp32<<<grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin_f, dr_inv_f, pc_f, simg, nullptr, state.x_f(),
            state.y_f(), state.z_f(), state.apc_f.template as<float>() + p0,
            state.r0_f.template as<float>() + p0, vph + p0 * vph_pulse, Nfft);
      });
      return;
    }
    if (backend == BackprojectionBackend::kNvradarCudaFp32Hq) {
      tiles([&](int p0, int n) {
        backprojection_kernel_fp32_hq<<<grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, stride, rmin_f, dr_inv_f, pc_f, simg, nullptr,
            state.x_f(), state.y_f(), state.z_f(), state.apc_f.template as<float>() + p0,
            state.mcp_f.template as<float>() + p0, state.r0_f.template as<float>() + p0,
            vph + p0 * vph_pulse, Nfft);
      });
      return;
    }
    tiles([&](int p0, int n) {
      backprojection_kernel_fp32_hq_fast<<<grid, block, 0, stream>>>(
          0, img_width, img_height, n, stride, stride, rmin_f, dr_inv_f, pc_f, simg, nullptr,
          state.x_f(), state.y_f(), state.z_f(), state.apc_f.template as<float>() + p0,
          state.mcp_f.template as<float>() + p0, state.r0_f.template as<float>() + p0,
          vph + p0 * vph_pulse, Nfft);
    });
    return;
  }

  if (backend == BackprojectionBackend::kNvradarBlockAnchored) {
    prepare_float_scene(state, pixels, Np, stream);
    const float* vph = float_vph();
    tiles([&](int p0, int n) {
      backprojection_kernel_fp32_hq_block<<<grid, block, 0, stream>>>(
          0, img_width, img_height, n, stride, stride, static_cast<float>(rmin),
          static_cast<float>(1.0 / dr), static_cast<float>(pc_partial), simg, nullptr, state.x_f(),
          state.y_f(), state.z_f(), state.apc_f.template as<float>() + p0,
          state.mcp_f.template as<float>() + p0, state.r0_f.template as<float>() + p0,
          vph + p0 * vph_pulse, Nfft);
    });
    return;
  }

  if (backend == BackprojectionBackend::kNvradarBlockAnchoredRefinedThroughput) {
    const float* vph = nullptr;
    if constexpr (std::is_same_v<T, float>) {
      vph = static_cast<const float*>(profiles);
    } else {
      const int n = Np * Nfft * 2;
      state.vph_cast.ensure(static_cast<size_t>(n) * sizeof(float), stream);
      const int threads = 256;
      const int blocks = (n + threads - 1) / threads;
      downcast_vph_kernel<<<blocks, threads, 0, stream>>>(static_cast<const double*>(profiles),
                                                          state.vph_cast.template as<float>(), n);
      vph = state.vph_cast.template as<float>();
    }
    const double dr_inv_d = 1.0 / dr;
    const float dr_inv = static_cast<float>(dr_inv_d);
    tiles([&](int p0, int n) {
      backprojection_kernel_fp64_phiwrap_f32mf_f32vph_anchor_taylor_phi32_vphf_drinvd<<<  // NOLINT
          grid, block, 0, stream>>>(
          0, img_width, img_height, n, stride, rmin, dr_inv_d, dr_inv, pc_partial, simg, nullptr,
          state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
          state.r0.template as<double>() + p0, vph + p0 * vph_pulse, Nfft);
    });
    return;
  }

  if (backend == BackprojectionBackend::kNvradarBlockAnchoredRefinedQuality) {
    const float* vph = float_vph();
    const float dr_inv = static_cast<float>(1.0 / dr);
    tiles([&](int p0, int n) {
      backprojection_kernel_fp64_phiwrap_f32mf_f32vph_anchor_taylor_phi32df_dr_cw1nw<<<  // NOLINT
          grid, block, 0, stream>>>(
          0, img_width, img_height, n, stride, rmin, dr, dr_inv, pc_partial, simg, nullptr,
          state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
          state.r0.template as<double>() + p0, vph + p0 * vph_pulse, Nfft);
    });
    return;
  }

  if (backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput ||
      backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedQuality) {
    const float* vph = float_vph();
    const float dr_inv = static_cast<float>(1.0 / dr);
    if (backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput) {
      tiles([&](int p0, int n) {
        backprojection_kernel_fp64_phiwrap_f32mf_f32vph_anchor_taylor_phi32df_dr_in_cw1nw_fma<<<  // NOLINT
            grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin, dr, dr_inv, pc_partial, simg, nullptr,
            state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
            state.r0.template as<double>() + p0, vph + p0 * vph_pulse, Nfft);
      });
    } else {
      tiles([&](int p0, int n) {
        backprojection_kernel_fp64_phiwrap_f32mf_f32vph_anchor_taylor_phi32df_dr_in_cw1nw<<<  // NOLINT
            grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin, dr, dr_inv, pc_partial, simg, nullptr,
            state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
            state.r0.template as<double>() + p0, vph + p0 * vph_pulse, Nfft);
      });
    }
    return;
  }

  if (backend == BackprojectionBackend::kNvradarBlockAnchoredFp64Trajectory) {
    prepare_float_scene(state, pixels, Np, stream);
    const float* vph = float_vph();
    tiles([&](int p0, int n) {
      backprojection_kernel_fp32_hq_block3_fp64in<<<grid, block, 0, stream>>>(
          0, img_width, img_height, n, stride, stride, static_cast<float>(rmin),
          static_cast<float>(1.0 / dr), static_cast<float>(pc_partial), simg, nullptr, state.x_f(),
          state.y_f(), state.z_f(), state.apc.template as<double>() + p0,
          state.mcp.template as<double>() + p0, state.r0.template as<double>() + p0,
          vph + p0 * vph_pulse, Nfft);
    });
    return;
  }

  const bool lut_backend =
      backend == BackprojectionBackend::kNvradarBlockAnchoredPhaseLut ||
      backend == BackprojectionBackend::kNvradarBlockAnchoredRefinedQualityPhaseLut ||
      backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLut ||
      backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVec ||
      backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0 ||
      backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedQualityPhaseLut ||
      backend == BackprojectionBackend::kNvradarBlockAnchoredPhaseLutLegacy ||
      backend == BackprojectionBackend::kNvradarBlockAnchoredFp64TrajectoryPhaseLut;
  if (lut_backend) {
    const float* vph = float_vph();
    const float dr_inv = static_cast<float>(1.0 / dr);
    const double dr_inv_d = 1.0 / dr;
    const float pc_dr = static_cast<float>(pc_partial * dr);
    ensure_nvradar_lut(state, pc_partial, rmin, dr, Nfft, stream);
    const float* lut = state.phase_lut.template as<float>();
    if (backend == BackprojectionBackend::kNvradarBlockAnchoredPhaseLut) {
      tiles([&](int p0, int n) {
        backprojection_kernel_anchor_taylor_phase_lut<<<grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin, dr_inv_d, dr_inv, pc_dr, simg, nullptr,
            state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
            state.r0.template as<double>() + p0, vph + p0 * vph_pulse, lut, Nfft);
      });
      return;
    }
    if (backend == BackprojectionBackend::kNvradarBlockAnchoredRefinedQualityPhaseLut) {
      tiles([&](int p0, int n) {
        backprojection_kernel_anchor_taylor_quality_phase_lut<<<grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin, dr, dr_inv, pc_dr, simg, nullptr,
            state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
            state.r0.template as<double>() + p0, vph + p0 * vph_pulse, lut, Nfft);
      });
      return;
    }
    if (backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLut) {
      tiles([&](int p0, int n) {
        backprojection_kernel_anchor_expanded_throughput_phase_lut<<<grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin, dr, dr_inv, pc_dr, simg, nullptr,
            state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
            state.r0.template as<double>() + p0, vph + p0 * vph_pulse, lut, Nfft);
      });
      return;
    }
    if (backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVec) {
      tiles([&](int p0, int n) {
        backprojection_kernel_anchor_expanded_throughput_phase_lut_vec<<<  // NOLINT
            grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin, dr, dr_inv, pc_dr, simg, nullptr,
            state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
            state.r0.template as<double>() + p0, vph + p0 * vph_pulse, lut, Nfft);
      });
      return;
    }
    if (backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0) {
      tiles([&](int p0, int n) {
        backprojection_kernel_anchor_expanded_throughput_phase_lut_vec_z0<<<  // NOLINT
            grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin, dr, dr_inv, pc_dr, simg, nullptr,
            state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
            state.r0.template as<double>() + p0, vph + p0 * vph_pulse, lut, Nfft);
      });
      return;
    }
    if (backend == BackprojectionBackend::kNvradarBlockAnchoredExpandedQualityPhaseLut) {
      tiles([&](int p0, int n) {
        backprojection_kernel_anchor_expanded_quality_phase_lut<<<grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, rmin, dr, dr_inv, pc_dr, simg, nullptr,
            state.x_d(), state.y_d(), state.z_d(), state.apc.template as<double>() + p0,
            state.r0.template as<double>() + p0, vph + p0 * vph_pulse, lut, Nfft);
      });
      return;
    }
    if (backend == BackprojectionBackend::kNvradarBlockAnchoredPhaseLutLegacy) {
      prepare_float_scene(state, pixels, Np, stream);
      tiles([&](int p0, int n) {
        backprojection_kernel_hq_block_phase_lut<<<grid, block, 0, stream>>>(
            0, img_width, img_height, n, stride, stride, static_cast<float>(rmin), dr_inv, pc_dr,
            simg, nullptr, state.x_f(), state.y_f(), state.z_f(),
            state.apc_f.template as<float>() + p0, state.mcp_f.template as<float>() + p0,
            state.r0_f.template as<float>() + p0, vph + p0 * vph_pulse, lut, Nfft);
      });
      return;
    }
    prepare_float_scene(state, pixels, Np, stream);
    tiles([&](int p0, int n) {
      backprojection_kernel_hq_block3_fp64in_phase_lut<<<grid, block, 0, stream>>>(
          0, img_width, img_height, n, stride, stride, static_cast<float>(rmin), dr_inv, pc_dr,
          simg, nullptr, state.x_f(), state.y_f(), state.z_f(),
          state.apc.template as<double>() + p0, state.mcp.template as<double>() + p0,
          state.r0.template as<double>() + p0, vph + p0 * vph_pulse, lut, Nfft);
    });
    return;
  }

  throw std::invalid_argument(std::string("BackprojectionOp: backend '") +
                              std::string(to_string(backend)) + "' is not implemented");
}

template <typename T, typename TGeom>
void launch_nvradar_window(BackprojectionBackend backend, BackprojectionState<T, TGeom>& state,
                           const void* profiles, int num_v, int num_u, int Np, int Nfft,
                           int batch_index, double rmin, double dr, double pc_partial,
                           cudaStream_t stream) {
  const int offset = static_cast<int>(state.ring_offset);
  if (offset <= 0 || offset >= num_v) {
    launch_nvradar(backend, state, profiles, num_v, num_u, Np, Nfft, batch_index, rmin, dr,
                   pc_partial, stream);
    return;
  }
  launch_nvradar(backend, state, profiles, num_v - offset, num_u, Np, Nfft, batch_index, rmin, dr,
                 pc_partial, stream, offset, num_v);
  launch_nvradar(backend, state, profiles, offset, num_u, Np, Nfft, batch_index, rmin, dr,
                 pc_partial, stream, 0, num_v);
}

template <typename T, typename TGeom>
void emit_accum(BackprojectionState<T, TGeom>& state, T* dst, int64_t batch, int64_t num_v,
                int64_t num_u, cudaStream_t stream) {
  const int threads = 256;
  auto promote = [&](const float* src, T* out, int64_t n) {
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    promote_image_kernel<<<blocks, threads, 0, stream>>>(src, out, n);
  };
  const int64_t off = state.ring_offset;
  if (off <= 0 || off >= num_v) {
    const int64_t n = batch * num_v * num_u * 2;
    promote(state.accum.template as<float>(), dst, n);
    return;
  }
  const int64_t row = num_u * 2;
  const int64_t h1 = num_v - off;
  for (int64_t b = 0; b < batch; ++b) {
    const float* src = state.accum.template as<float>() + static_cast<size_t>(b) * num_v * row;
    T* out = dst + static_cast<size_t>(b) * num_v * row;
    promote(src + off * row, out, h1 * row);
    promote(src, out + h1 * row, off * row);
  }
}

}  // namespace holoscan::radar::detail

#if __has_include("matx/kernels/sar_bp.cuh")
#include <vector_types.h>

#include "radar_common/tensor/matx_utils.hpp"
#include "matx/operators/sar_bp.h"
#endif

namespace holoscan::radar::detail {

#if __has_include("matx/kernels/sar_bp.cuh")
inline matx::SarBpParams matx_params(BackprojectionBackend backend, double frequency, double dr) {
  matx::SarBpParams params;
  params.center_frequency = frequency;
  params.del_r = dr;
  params.features = matx::SarBpFeature::None;
  switch (backend) {
    case BackprojectionBackend::kMatxDouble:
      params.compute_type = matx::SarBpComputeType::Double;
      break;
    case BackprojectionBackend::kMatxDoublePhaseLut:
      params.compute_type = matx::SarBpComputeType::Double;
      params.features = matx::SarBpFeature::PhaseLUTOptimization;
      break;
    case BackprojectionBackend::kMatxMixed:
      params.compute_type = matx::SarBpComputeType::Mixed;
      break;
    case BackprojectionBackend::kMatxMixedPhaseLut:
      params.compute_type = matx::SarBpComputeType::Mixed;
      params.features = matx::SarBpFeature::PhaseLUTOptimization;
      break;
    case BackprojectionBackend::kMatxFloatFloat:
      params.compute_type = matx::SarBpComputeType::FloatFloat;
      params.features = matx::SarBpFeature::PhaseLUTOptimization;
      break;
    case BackprojectionBackend::kMatxFloat:
      params.compute_type = matx::SarBpComputeType::Float;
      break;
    case BackprojectionBackend::kMatxFloatPhaseLut:
      params.compute_type = matx::SarBpComputeType::Float;
      params.features = matx::SarBpFeature::PhaseLUTOptimization;
      break;
    default:
      params.compute_type = matx::SarBpComputeType::Mixed;
      break;
  }
  return params;
}

template <typename T, typename TGeom>
void ensure_matx_accum(BackprojectionState<T, TGeom>& state, int64_t batch, int64_t num_v,
                       int64_t num_u, cudaStream_t stream) {
  const int64_t n = batch * num_v * num_u;
  const bool resize =
      state.cached_batch != batch || state.cached_num_v != num_v || state.cached_num_u != num_u;
  state.matx_accum.ensure(static_cast<size_t>(n) * sizeof(cuda::std::complex<T>), stream);
  if (resize) {
    if (cudaMemsetAsync(state.matx_accum.ptr, 0, state.matx_accum.bytes, stream) != cudaSuccess) {
      throw std::runtime_error("BackprojectionOp: matx accum memset");
    }
    state.cached_batch = batch;
    state.cached_num_v = num_v;
    state.cached_num_u = num_u;
    state.ring_offset = 0;
    state.launch_row0 = 0;
  }
}

template <typename T, typename TGeom>
void reset_matx_accum(BackprojectionState<T, TGeom>& state, cudaStream_t stream) {
  if (state.matx_accum.ptr == nullptr) {
    return;
  }
  if (cudaMemsetAsync(state.matx_accum.ptr, 0, state.matx_accum.bytes, stream) != cudaSuccess) {
    throw std::runtime_error("BackprojectionOp: matx accum reset");
  }
  state.ring_offset = 0;
  state.launch_row0 = 0;
}

template <typename T, typename TGeom>
void launch_matx(BackprojectionBackend backend, BackprojectionState<T, TGeom>& state,
                 const void* profiles, const TGeom* positions, const TGeom* lattice, int num_v,
                 int num_u, int Np, int Nfft, int batch_index, double frequency, double dr,
                 cudaStream_t stream, int row0 = 0, int full_v = -1) {
  using C = cuda::std::complex<T>;
  using Vec = std::conditional_t<std::is_same_v<TGeom, float>, float3, double3>;
  if (full_v < 1) {
    full_v = num_v;
  }
  const int pixels_full = full_v * num_u;
  C* img_ptr = state.matx_accum.template as<C>() + static_cast<size_t>(batch_index) * pixels_full +
               static_cast<size_t>(row0) * num_u;
  const TGeom* lat = lattice + static_cast<size_t>(row0) * num_u * 3;
  auto img = matx::make_tensor<C, 2>(img_ptr, {num_v, num_u});
  auto vox =
      matx::make_tensor<Vec, 2>(reinterpret_cast<Vec*>(const_cast<TGeom*>(lat)), {num_v, num_u});
  const auto params = matx_params(backend, frequency, dr);
  auto* prof_base = static_cast<C*>(const_cast<void*>(profiles));
  auto* pos_base = reinterpret_cast<Vec*>(const_cast<TGeom*>(positions));
  double* r0_base = state.r0.template as<double>();
  for (int p0 = 0; p0 < Np; p0 += kKernelPulseTile) {
    const int n = std::min(kKernelPulseTile, Np - p0);
    auto prof = matx::make_tensor<C, 2>(prof_base + p0 * Nfft, {n, Nfft});
    auto pos = matx::make_tensor<Vec, 1>(pos_base + p0, {n});
    auto r0 = matx::make_tensor<double, 1>(r0_base + p0, {n});
    (img = matx::experimental::sar_bp(img, prof, pos, vox, r0, params)).run(stream);
  }
}

template <typename T, typename TGeom>
void launch_matx_window(BackprojectionBackend backend, BackprojectionState<T, TGeom>& state,
                        const void* profiles, const TGeom* positions, const TGeom* lattice,
                        int num_v, int num_u, int Np, int Nfft, int batch_index, double frequency,
                        double dr, cudaStream_t stream) {
  const int offset = static_cast<int>(state.ring_offset);
  if (offset <= 0 || offset >= num_v) {
    launch_matx(backend, state, profiles, positions, lattice, num_v, num_u, Np, Nfft, batch_index,
                frequency, dr, stream);
    return;
  }
  launch_matx(backend, state, profiles, positions, lattice, num_v - offset, num_u, Np, Nfft,
              batch_index, frequency, dr, stream, offset, num_v);
  launch_matx(backend, state, profiles, positions, lattice, offset, num_u, Np, Nfft, batch_index,
              frequency, dr, stream, 0, num_v);
}

template <typename T, typename TGeom>
void zero_matx_rows(BackprojectionState<T, TGeom>& state, int64_t phys0, int64_t n_rows,
                    cudaStream_t stream) {
  using C = cuda::std::complex<T>;
  const int64_t nv = state.cached_num_v;
  const int64_t nu = state.cached_num_u;
  const int64_t batch = state.cached_batch;
  const int64_t row_bytes = nu * static_cast<int64_t>(sizeof(C));
  for (int64_t b = 0; b < batch; ++b) {
    C* base = state.matx_accum.template as<C>() + static_cast<size_t>(b) * nv * nu;
    memset_row_span(base, phys0, n_rows, nv, row_bytes, stream, "BackprojectionOp");
  }
}

template <typename T, typename TGeom>
void emit_matx_accum(BackprojectionState<T, TGeom>& state, T* dst, int64_t batch, int64_t num_v,
                     int64_t num_u, cudaStream_t stream) {
  using C = cuda::std::complex<T>;
  const int64_t off = state.ring_offset;
  auto copy = [&](const C* src, T* out, int64_t n_complex) {
    const size_t bytes = static_cast<size_t>(n_complex) * sizeof(C);
    if (cudaMemcpyAsync(out, src, bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      throw std::runtime_error("BackprojectionOp: matx emit copy");
    }
  };
  if (off <= 0 || off >= num_v) {
    copy(state.matx_accum.template as<C>(), dst, batch * num_v * num_u);
    return;
  }
  const int64_t h1 = num_v - off;
  for (int64_t b = 0; b < batch; ++b) {
    const C* src = state.matx_accum.template as<C>() + static_cast<size_t>(b) * num_v * num_u;
    T* out = dst + static_cast<size_t>(b) * num_v * num_u * 2;
    copy(src + off * num_u, out, h1 * num_u);
    copy(src, out + h1 * num_u * 2, off * num_u);
  }
}
#endif  // sar_bp.cuh

}  // namespace holoscan::radar::detail
