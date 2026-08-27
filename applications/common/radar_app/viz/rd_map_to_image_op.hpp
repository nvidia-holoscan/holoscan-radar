// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_app/viz/colormap.hpp"
#include "radar_app/viz/viz_kernels.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app {

// HSDK 5 Graph: convert a rank-2 RD map into a float32 LUT-index image
// (num_doppler, num_range, 1) for Holoviz or the X11 stand-in viewer.
//
// Accepts complex64 (amplitude → 20*log10) or float32 power (10*log10).
// Fat output pool sizing requires MaxTensorExtents.max_doppler_bins and
// max_range_bins at graph.op().
class RdMapToImageOp : public holoscan::Operator<> {
 public:
  explicit RdMapToImageOp(const MaxTensorExtents& extents) {
    const int64_t max_d =
        require_extent(extents.max_doppler_bins, "max_doppler_bins",
                       "RdMapToImageOp");
    const int64_t max_r =
        require_extent(extents.max_range_bins, "max_range_bins",
                       "RdMapToImageOp");
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_d, max_r, 1}, real_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(rd_maps).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.output(rd_image)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 3U, max_byte_span_));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{rd_maps});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    HOLOSCAN_TRY(auto rd_map, receive_tensor(rd_maps, "RdMapToImageOp"));
    retained_input_ = rd_map;

    const auto& shape = rd_map.shape();
    if (shape.size() != 2) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "RdMapToImageOp: expected rank-2 rd_map (doppler, range), got "
          "rank " +
              std::to_string(shape.size())});
    }
    const int64_t num_doppler = shape[0];
    const int64_t num_range = shape[1];
    const int64_t count = num_doppler * num_range;

    const DLDataType dtype = rd_map.dtype();
    const bool is_complex =
        dldatatype_equals(dtype, complex_dtype<float>());
    if (!is_complex && !dldatatype_equals(dtype, real_dtype<float>())) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "RdMapToImageOp: expected complex64 or float32 rd_map, got " +
              dldatatype_name(dtype)});
    }

    try {
      ensure_scratch(count, num_range);
    } catch (const std::exception& e) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("RdMapToImageOp: scratch alloc failed: ") + e.what()});
    }

    const std::vector<int64_t> out_shape{num_doppler, num_range, 1};
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kNotSupported,
          "RdMapToImageOp: no CUDA compute stream"});
    }
    return emit_device_tensor(
        rd_image, context, out_shape, real_dtype<float>(),
        [&](void* dst, std::size_t) -> holoscan::expected<void, holoscan::Error> {
          auto col = col_median_->mutable_data();
          auto sorted = sorted_->mutable_data();
          if (!col || !sorted) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                "RdMapToImageOp: scratch tensors not writable"});
          }
          void* scratch = nullptr;
          if (sort_scratch_ != nullptr) {
            auto sc = sort_scratch_->mutable_data();
            if (!sc) {
              return holoscan::make_unexpected(holoscan::Error{
                  holoscan::ErrorCode::kFailure,
                  "RdMapToImageOp: sort scratch not writable"});
            }
            scratch = *sc;
          }
          rd_map_to_lut_image(
              rd_map.data(), is_complex, num_doppler, num_range,
              is_complex ? 20.0F : 10.0F,
              static_cast<float>(kLutSize - 1), static_cast<float*>(dst),
              static_cast<float*>(*col), static_cast<float*>(*sorted),
              scratch, sort_scratch_bytes_, stream);
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> rd_maps{"rd_maps"};
  holoscan::Output<holoscan::Tensor> rd_image{"rd_image"};

 private:
  void ensure_scratch(int64_t count, int64_t num_range) {
    if (scratch_count_ == count && scratch_range_ == num_range) {
      return;
    }
    col_median_ = make_device_tensor({num_range}, real_dtype<float>());
    sorted_ = make_device_tensor({count}, real_dtype<float>());
    sort_scratch_bytes_ = rd_lut_image_sort_scratch_bytes(count);
    sort_scratch_ =
        sort_scratch_bytes_ > 0
            ? make_device_tensor(
                  {static_cast<int64_t>(sort_scratch_bytes_)},
                  DLDataType{kDLUInt, 8, 1})
            : nullptr;
    scratch_count_ = count;
    scratch_range_ = num_range;
  }

  std::size_t max_byte_span_{0U};
  holoscan::Tensor retained_input_;
  std::shared_ptr<holoscan::Tensor> col_median_;
  std::shared_ptr<holoscan::Tensor> sorted_;
  std::shared_ptr<holoscan::Tensor> sort_scratch_;
  size_t sort_scratch_bytes_{0};
  int64_t scratch_count_{0};
  int64_t scratch_range_{0};
};

}  // namespace holoscan::radar::app
