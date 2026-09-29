// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <dlpack/dlpack.h>

#include <cstdint>
#include <memory>
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
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>

namespace holoscan::radar::app::viz {

// Holoviz's image layer consumes ImageT. Keep the radar compositor's output a
// plain Tensor and add the image descriptor only at the visualization boundary.
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
            .storage = TensorOutputStorage::kPassThrough,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{rgba});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = rgba.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    const auto shape = sample->data.shape();
    if (shape.size() != 3U || shape[0] != height_ || shape[1] != width_ || shape[2] != 4) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "dashboard RGBA shape mismatch"});
    }

    schema::ImageT descriptor;
    descriptor.width = width_;
    descriptor.height = height_;
    descriptor.encoding = schema::ImageEncoding_RGBA8;
    descriptor.color_space = schema::ColorSpace_UNSPECIFIED;
    descriptor.data = std::make_shared<Tensor>(std::move(sample->data));
    return image.emit(Sample<schema::ImageT>{std::move(descriptor), sample->metadata});
  }

  Input<Tensor> rgba{"rgba"};
  Output<schema::ImageT> image{"image"};

 private:
  std::int32_t width_;
  std::int32_t height_;
};

}  // namespace holoscan::radar::app::viz
