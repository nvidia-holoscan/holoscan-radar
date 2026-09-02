// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::radar::detail {

namespace {

constexpr int64_t kBlock = 256;
constexpr double kPositionTol = 1e-9;

bool positions_equal(const double* a, const double* b) {
  for (int i = 0; i < 3; ++i) {
    if (std::abs(a[i] - b[i]) > kPositionTol) {
      return false;
    }
  }
  return true;
}

template <typename T>
using Complex = cuda::std::complex<T>;

template <typename T>
__global__ void gather_single_source_kernel(
    const Complex<T>* input, Complex<T>* output, int64_t num_doppler,
    int64_t num_range, const int64_t* src_channel) {
  const int64_t plane = num_doppler * num_range;
  const int64_t v = static_cast<int64_t>(blockIdx.x);
  const int64_t src = src_channel[v];
  const int64_t in_base = src * plane;
  const int64_t out_base = v * plane;
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < plane;
       i += blockDim.x) {
    output[out_base + i] = input[in_base + i];
  }
}

template <typename T>
__global__ void gather_average_kernel(
    const Complex<T>* input, Complex<T>* output, int64_t num_doppler,
    int64_t num_range, const int64_t* member_offsets,
    const int64_t* member_indices) {
  const int64_t plane = num_doppler * num_range;
  const int64_t v = static_cast<int64_t>(blockIdx.x);
  const int64_t begin = member_offsets[v];
  const int64_t end = member_offsets[v + 1];
  const int64_t count = end - begin;
  if (count <= 0) {
    return;
  }

  const int64_t out_base = v * plane;
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < plane;
       i += blockDim.x) {
    Complex<T> sum{};
    for (int64_t m = begin; m < end; ++m) {
      const int64_t src = member_indices[m];
      sum += input[src * plane + i];
    }
    output[out_base + i] = sum / static_cast<T>(count);
  }
}

}  // namespace

RdMapsLayout parse_rd_maps_layout(const holoscan::Tensor& tensor,
                                  const char* port_name) {
  const auto& shape = tensor.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument(
        std::string(port_name) +
        ": expected rank 4 (M_tx, N_rx, num_doppler, num_range), got " +
        std::to_string(shape.size()));
  }
  RdMapsLayout layout;
  layout.num_tx = shape[0];
  layout.num_rx = shape[1];
  layout.num_doppler = shape[2];
  layout.num_range = shape[3];
  if (layout.num_tx <= 0 || layout.num_rx <= 0 || layout.num_doppler <= 0 ||
      layout.num_range <= 0) {
    throw std::invalid_argument(std::string(port_name) +
                                ": trailing axes must be positive");
  }
  return layout;
}

std::vector<int64_t> assembled_rd_shape(const RdMapsLayout& in,
                                        int64_t num_virtual) {
  return {num_virtual, in.num_doppler, in.num_range};
}

VirtualAssemblyPlan build_virtual_plan(
    const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, int64_t num_tx, int64_t num_rx,
    OverlapPolicy policy) {
  if (tx_positions.size() != static_cast<size_t>(num_tx * 3)) {
    throw std::invalid_argument(
        "VirtualArrayAssemblyOp: tx_positions length must be num_tx * 3");
  }
  if (rx_positions.size() != static_cast<size_t>(num_rx * 3)) {
    throw std::invalid_argument(
        "VirtualArrayAssemblyOp: rx_positions length must be num_rx * 3");
  }

  VirtualAssemblyPlan plan;
  plan.policy = policy;
  plan.num_tx = num_tx;
  plan.num_rx = num_rx;

  const int64_t num_channels = num_tx * num_rx;
  if (policy == OverlapPolicy::kKeepAll) {
    plan.num_virtual = num_channels;
    plan.virtual_positions.resize(static_cast<size_t>(num_channels * 3));
    plan.src_channel.resize(static_cast<size_t>(num_channels));
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        const int64_t ch = tx * num_rx + rx;
        const size_t out = static_cast<size_t>(ch * 3);
        for (int i = 0; i < 3; ++i) {
          plan.virtual_positions[out + static_cast<size_t>(i)] =
              tx_positions[static_cast<size_t>(tx * 3 + i)] +
              rx_positions[static_cast<size_t>(rx * 3 + i)];
        }
        plan.src_channel[static_cast<size_t>(ch)] = ch;
      }
    }
    return plan;
  }

  struct Group {
    double position[3];
    int64_t rep_channel{0};
    std::vector<int64_t> members;
  };
  std::vector<Group> groups;
  groups.reserve(static_cast<size_t>(num_channels));

  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      const int64_t ch = tx * num_rx + rx;
      double candidate[3];
      for (int i = 0; i < 3; ++i) {
        candidate[i] =
            tx_positions[static_cast<size_t>(tx * 3 + i)] +
            rx_positions[static_cast<size_t>(rx * 3 + i)];
      }
      int64_t group_idx = -1;
      for (size_t g = 0; g < groups.size(); ++g) {
        if (positions_equal(groups[g].position, candidate)) {
          group_idx = static_cast<int64_t>(g);
          break;
        }
      }
      if (group_idx < 0) {
        Group group;
        for (int i = 0; i < 3; ++i) {
          group.position[i] = candidate[i];
        }
        group.rep_channel = ch;
        group.members.push_back(ch);
        groups.push_back(std::move(group));
      } else {
        groups[static_cast<size_t>(group_idx)].members.push_back(ch);
      }
    }
  }

  plan.num_virtual = static_cast<int64_t>(groups.size());
  plan.virtual_positions.resize(static_cast<size_t>(plan.num_virtual * 3));
  plan.member_offsets.reserve(static_cast<size_t>(plan.num_virtual + 1));
  plan.member_offsets.push_back(0);
  for (int64_t g = 0; g < plan.num_virtual; ++g) {
    const Group& group = groups[static_cast<size_t>(g)];
    for (int i = 0; i < 3; ++i) {
      plan.virtual_positions[static_cast<size_t>(g * 3 + i)] =
          group.position[i];
    }
    if (policy == OverlapPolicy::kDedup) {
      plan.src_channel.push_back(group.rep_channel);
    } else {
      for (int64_t ch : group.members) {
        plan.member_indices.push_back(ch);
      }
      plan.member_offsets.push_back(
          static_cast<int64_t>(plan.member_indices.size()));
    }
  }
  return plan;
}

ArrayGeometry make_array_geometry(const VirtualAssemblyPlan& plan,
                                double wavelength) {
  ArrayGeometry geom;
  geom.element_positions = plan.virtual_positions;
  geom.wavelength = wavelength;
  geom.validate();
  return geom;
}

template <typename T>
void run_virtual_assembly_stage(
    const void* input, void* output, const RdMapsLayout& layout,
    const VirtualAssemblyPlan& plan, const int64_t* d_src_channel,
    const int64_t* d_member_offsets, const int64_t* d_member_indices,
    cudaStream_t stream) {
  const int grid = static_cast<int>(plan.num_virtual);
  const auto* in = static_cast<const Complex<T>*>(input);
  auto* out = static_cast<Complex<T>*>(output);

  if (plan.policy == OverlapPolicy::kAverage) {
    gather_average_kernel<T><<<grid, kBlock, 0, stream>>>(
        in, out, layout.num_doppler, layout.num_range, d_member_offsets,
        d_member_indices);
  } else {
    gather_single_source_kernel<T><<<grid, kBlock, 0, stream>>>(
        in, out, layout.num_doppler, layout.num_range, d_src_channel);
  }
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error(
        "virtual assembly stage: kernel launch failed");
  }
}

template void run_virtual_assembly_stage<float>(
    const void* input, void* output, const RdMapsLayout& layout,
    const VirtualAssemblyPlan& plan, const int64_t* d_src_channel,
    const int64_t* d_member_offsets, const int64_t* d_member_indices,
    cudaStream_t stream);
template void run_virtual_assembly_stage<double>(
    const void* input, void* output, const RdMapsLayout& layout,
    const VirtualAssemblyPlan& plan, const int64_t* d_src_channel,
    const int64_t* d_member_offsets, const int64_t* d_member_indices,
    cudaStream_t stream);

}  // namespace holoscan::radar::detail
