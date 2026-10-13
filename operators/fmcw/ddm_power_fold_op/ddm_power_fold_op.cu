// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/ddm_power_fold_op/ddm_power_fold_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {
namespace {

constexpr const char* kOpName = "DDMPowerFoldOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

template <typename T>
__global__ void power_fold_kernel(const cuda::std::complex<T>* input, T* output, int64_t num_rx,
                                  int64_t num_doppler, int64_t num_range, int64_t num_tx,
                                  int64_t width, const int64_t* table) {
  const int64_t total = width * num_range;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t d = index / num_range;
    const int64_t r = index % num_range;
    T sum = 0;
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      const int64_t source_d = table[tx * width + d];
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        const auto value = input[(rx * num_doppler + source_d) * num_range + r];
        sum += value.real() * value.real() + value.imag() * value.imag();
      }
    }
    output[index] = sum;
  }
}

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

}  // namespace

template <typename T>
DDMPowerFoldOp<T>::DDMPowerFoldOp(const MaxTensorExtents& extents) {
  (void)require_extent(extents.max_rx, "max_rx", kOpName);
  const auto max_doppler = require_extent(extents.max_doppler_bins, "max_doppler_bins", kOpName);
  const auto max_range = require_extent(extents.max_range_bins, "max_range_bins", kOpName);
  extents_ = extents;
  max_byte_span_ =
      byte_span_elems(product_extents({max_doppler, max_range}, kOpName), sizeof(T), kOpName);
}

template <typename T>
void DDMPowerFoldOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(power_map).produces_tensor(pooled_tensor_output(kDevice, 2U, max_byte_span_));
}

template <typename T>
Contract DDMPowerFoldOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  return result;
}

template <typename T>
expected<void, Error> DDMPowerFoldOp<T>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));
  try {
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
    require_complex<T>(input, "rd_maps");
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  detail::DdmRdMapsLayout layout;
  int64_t width = 0;
  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const auto& code = context.params<DDMPowerFoldParams>().ddm_code;
  try {
    layout = detail::parse_ddm_rd_maps_layout(input, "rd_maps");
    if (!layout.leading.empty()) {
      return fail_invalid("DDMPowerFoldOp: rd_maps must have rank 3 (N_rx, D, R)");
    }
    (void)detail::ddm_exact_band_indices(code, layout.num_doppler);
    width = detail::ddm_num_doppler_per_tx(layout.num_doppler, code.num_tx, code.guard_bands);
    if (layout.num_rx > *extents_.max_rx || layout.num_doppler > *extents_.max_doppler_bins ||
        layout.num_range > *extents_.max_range_bins) {
      return fail_invalid("DDMPowerFoldOp: input exceeds MaxTensorExtents");
    }
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "DDMPowerFoldOp: no CUDA compute stream"});
  }
  // Prepare or reuse cached artifacts without changing tick coherence.
  const auto key = detail::ddm_demux_cache_key(code, layout.num_doppler);
  const CachedTable& cached = table_cache_.get_or_build(key, "DDM power-fold table", [&] {
    CachedTable result;
    const auto host = detail::build_ddm_demux_table(detail::ddm_resolve_offsets(code), code.num_tx,
                                                    layout.num_doppler, width);
    result.device_table = make_device_tensor({code.num_tx, width}, kInt64Dtype);
    auto pointer = result.device_table->mutable_data();
    if (!pointer || cudaMemcpyAsync(*pointer, host.data(), host.size() * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error("DDMPowerFoldOp: table H2D failed");
    }
    return result;
  });

  const int64_t total = width * layout.num_range;
  const std::vector<int64_t> output_shape{width, layout.num_range};
  // Publish coherent outputs in dependency order and retain borrowed inputs.
  HOLOSCAN_TRY(emit_device_tensor(
      power_map, context, output_shape, real_dtype<T>(),
      [&](void* output, std::size_t) -> expected<void, Error> {
        constexpr int kBlock = 256;
        const int blocks =
            static_cast<int>(std::min<int64_t>(65535, (total + kBlock - 1) / kBlock));
        power_fold_kernel<T><<<blocks, kBlock, 0, stream>>>(
            static_cast<const cuda::std::complex<T>*>(input.data()), static_cast<T*>(output),
            layout.num_rx, layout.num_doppler, layout.num_range, code.num_tx, width,
            static_cast<const int64_t*>(cached.device_table->data()));
        if (cudaGetLastError() != cudaSuccess) {
          return make_unexpected(
              Error{ErrorCode::kFailure, "DDMPowerFoldOp: CUDA kernel launch failed"});
        }
        return {};
      }));
  retained_input_ = std::move(input);
  return {};
}

template class DDMPowerFoldOp<float>;
template class DDMPowerFoldOp<double>;

}  // namespace holoscan::radar
