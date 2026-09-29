// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>

namespace holoscan::radar::app::viz {

// Holoviz consumes ImageT. Copy into a dedicated output pool at the display
// boundary so its CUDA stream can hold frames independently of the compositor.
class RgbaImageBridgeOp final : public Operator<> {
 public:
  RgbaImageBridgeOp(std::int32_t width, std::int32_t height) : width_(width), height_(height) {}

  void setup(OperatorSpec& spec) override {
    const TensorRepresentation format{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 3U,
        .strides = TensorStridePolicy::kContiguousOnly,
    };
    const TensorBounds bounds{
        .max_rank = 3U,
        .max_byte_span = static_cast<std::uint64_t>(width_) * height_ * 4U,
    };
    spec.input(rgba).queue_depth(2U).expects_tensor(TensorInputSpec{
        .representation = format,
        .bounds = bounds,
    });
    spec.output(image)
        .tensor_field(schema::ImageTensorFields::data)
        .produces_tensor(TensorOutputSpec{
            .representation = format,
            .bounds = bounds,
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{rgba});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = rgba.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    const auto shape = sample->data.shape();
    if (shape.size() != 3U || shape[0] != height_ || shape[1] != width_ || shape[2] != 4) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "dashboard RGBA shape mismatch"});
    }

    const std::array<std::int64_t, 3U> image_shape{height_, width_, 4};
    auto loan = image.allocate_tensor(TensorLoanRequest{
        .shape = image_shape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    const auto stream = context.cuda_stream();
    auto writer = loan->write(stream);
    if (!writer) {
      return make_unexpected(std::move(writer).error());
    }
    const std::size_t bytes = static_cast<std::size_t>(width_) * height_ * 4U;
    if (writer->byte_size() != bytes ||
        cudaMemcpyAsync(writer->data(), sample->data.data(), bytes, cudaMemcpyDeviceToDevice,
                        stream.get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "dashboard RGBA device copy failed"});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }

    schema::ImageT descriptor;
    descriptor.width = width_;
    descriptor.height = height_;
    descriptor.encoding = schema::ImageEncoding_RGBA8;
    descriptor.color_space = schema::ColorSpace_UNSPECIFIED;
    return image.emit_tensor(std::move(*loan), descriptor, sensor_io::derive_from(*sample));
  }

  Input<Tensor> rgba{"rgba"};
  Output<schema::ImageT> image{"image"};

 private:
  std::int32_t width_;
  std::int32_t height_;
};

}  // namespace holoscan::radar::app::viz
