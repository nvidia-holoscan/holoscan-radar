// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// HSDK 5 Graph: collapse a rank>=3 RD cube (..., D, R) to a rank-2 float32
// power map by max over leading planes. Complex input reduces on |z|^2;
// float32 power is reduced as-is. Matches Application RdAngleMaxOp in viz.hpp
// (rd_map_max_over_planes). Used by the dense SIMO demo to project CFAR's
// per-beam cfar_power cube onto the RD plane for the X11 viewer.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

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

#include "radar_app/viz/viz_kernels.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app {

class RdAngleMaxOp : public holoscan::Operator<> {
 public:
  explicit RdAngleMaxOp(const MaxTensorExtents& extents) {
    const int64_t max_d = require_extent(extents.max_doppler_bins,
                                         "max_doppler_bins", "RdAngleMaxOp");
    const int64_t max_r =
        require_extent(extents.max_range_bins, "max_range_bins",
                       "RdAngleMaxOp");
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_d, max_r}, real_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(rd_maps).queue_depth(2U).expects_tensor(kDevice);
    spec.output(rd_map)
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 2U, .max_byte_span = max_byte_span_});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{rd_maps});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "RdAngleMaxOp";
    HOLOSCAN_TRY(auto cube, receive_tensor(rd_maps, who));
    retained_input_ = cube;

    const auto& shape = cube.shape();
    if (shape.size() < 3) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) + ": expected rank >= 3 (..., D, R), got rank " +
              std::to_string(shape.size())});
    }
    const int64_t num_doppler = shape[shape.size() - 2];
    const int64_t num_range = shape[shape.size() - 1];
    int64_t num_planes = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
      num_planes *= shape[i];
    }

    const DLDataType dtype = cube.dtype();
    const bool is_complex =
        dldatatype_equals(dtype, complex_dtype<float>());
    if (!is_complex && !dldatatype_equals(dtype, real_dtype<float>())) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) +
              ": expected complex64 or float32 rd_maps, got " +
              dldatatype_name(dtype)});
    }

    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kNotSupported,
          std::string(who) + ": no CUDA compute stream"});
    }

    const std::vector<int64_t> out_shape{num_doppler, num_range};
    return emit_device_tensor(
        rd_map, context, out_shape, real_dtype<float>(),
        [&](void* dst, std::size_t) -> holoscan::expected<void, holoscan::Error> {
          try {
            rd_map_max_over_planes(cube.data(), is_complex, num_planes,
                                   num_doppler, num_range,
                                   static_cast<float*>(dst), stream);
          } catch (const std::exception& e) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": " + e.what()});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> rd_maps{"rd_maps"};
  holoscan::Output<holoscan::Tensor> rd_map{"rd_map"};

 private:
  std::size_t max_byte_span_{0U};
  holoscan::Tensor retained_input_;
};

}  // namespace holoscan::radar::app
