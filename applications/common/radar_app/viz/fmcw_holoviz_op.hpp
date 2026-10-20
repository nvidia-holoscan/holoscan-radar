// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "radar_app/viz/dashboard_text.hpp"

namespace holoscan::radar::app::viz {

inline visualization::LayerStack dashboard_image_layers(std::int32_t width, std::int32_t height) {
  visualization::LayerStack layers;
  layers.add_image_layer(
      "image", visualization::ImageLayerDesc{.format = visualization::ImageFormat::kR8G8B8A8Unorm,
                                             .max_width = static_cast<std::uint32_t>(width),
                                             .max_height = static_cast<std::uint32_t>(height)});
  return layers;
}

// Compose native SDK text and the radar canvas into one CUDA image. The display
// scales that complete frame, so resizing cannot separate labels from their axes.
class FmcwFrameComposerOp final : public holoscan::visualization::HolovizOp {
 public:
  FmcwFrameComposerOp(std::int32_t width, std::int32_t height)
  : HolovizOp(make_layers(width, height),
              visualization::CudaSinkDesc{.width = static_cast<std::uint32_t>(width),
                                          .height = static_cast<std::uint32_t>(height)}),
    width_(width),
    height_(height) {}

  void setup(OperatorSpec& spec) override {
    // Keep app-authored selectors and declarations in the same module: the
    // staging SDK has DSO-local type tokens for these typed image ports.
    const TensorRepresentation image_format{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 3U,
        .strides = TensorStridePolicy::kContiguousOnly,
    };
    spec.input(image_layer_input(0), "image")
        .queue_depth(2U)
        .tensor_field(schema::ImageTensorFields::data)
        .expects_tensor(TensorInputSpec{.representation = image_format});
    const TensorRepresentation text_format{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 2U,
        .strides = TensorStridePolicy::kContiguousOnly,
    };
    spec.input(tensor_layer_input(1), "text")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{.representation = text_format});
    spec.output(output(), "output")
        .tensor_field(schema::ImageTensorFields::data)
        .produces_tensor(TensorOutputSpec{
            .representation = image_format,
            .bounds =
                TensorBounds{.max_rank = 3U,
                             .max_byte_span = static_cast<std::uint64_t>(width_) * height_ * 4U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    // Join before activation: an OnEach image driver with an Exact text side
    // selector can strand a stale image after display pressure evicts its text.
    Contract result;
    result
        .trigger(OnAligned{.ports = ports(image_layer_input(0), tensor_layer_input(1)),
                           .window = std::chrono::nanoseconds{0}})
        .input(image_layer_input(0), Exact{})
        .input(tensor_layer_input(1), Exact{});
    return result;
  }

 private:
  static visualization::LayerStack make_layers(std::int32_t width, std::int32_t height) {
    auto layers = dashboard_image_layers(width, height);
    layers.add_text_layer("text",
                          visualization::TextLayerDesc{.max_glyphs = detail::kDashboardTextRows *
                                                                     detail::kDashboardTextBytes});
    return layers;
  }

  std::int32_t width_;
  std::int32_t height_;
};

// Present only complete frames. The SDK owns window events and swapchain resize.
class FmcwHolovizOp final : public holoscan::visualization::HolovizOp {
 public:
  FmcwHolovizOp(std::int32_t width, std::int32_t height, std::string title)
  : HolovizOp(dashboard_image_layers(width, height),
              visualization::WindowSinkDesc{.title = std::move(title),
                                            .width = static_cast<std::uint32_t>(width),
                                            .height = static_cast<std::uint32_t>(height)}) {}
};

}  // namespace holoscan::radar::app::viz
