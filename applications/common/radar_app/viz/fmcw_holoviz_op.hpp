// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "radar_app/viz/dashboard_text.hpp"

namespace holoscan::radar::app::viz {

// Require the image and text from one completed dashboard frame. The SDK's
// default side-layer selector is best-effort and may otherwise omit labels or
// select the next frame's counter when presentation falls behind.
class FmcwHolovizOp final : public holoscan::visualization::HolovizOp {
 public:
  FmcwHolovizOp(std::int32_t width, std::int32_t height, std::string title)
  : HolovizOp(make_layers(width, height),
              visualization::WindowSinkDesc{.title = std::move(title),
                                            .width = static_cast<std::uint32_t>(width),
                                            .height = static_cast<std::uint32_t>(height)}) {}

  void setup(OperatorSpec& spec) override {
    // Declare the payloads and selectors in the same module. Mixing the SDK's
    // setup() with app-authored selectors uses different DSO-local type tokens
    // in this staging image and fails temporal admission.
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
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{image_layer_input(0)})
        .input(tensor_layer_input(1), Exact{})
        .on_missing(tensor_layer_input(1), Skip{});
    return result;
  }

 private:
  static visualization::LayerStack make_layers(std::int32_t width, std::int32_t height) {
    visualization::LayerStack layers;
    layers.add_image_layer(
        "image", visualization::ImageLayerDesc{.format = visualization::ImageFormat::kR8G8B8A8Unorm,
                                               .max_width = static_cast<std::uint32_t>(width),
                                               .max_height = static_cast<std::uint32_t>(height)});
    // Pixel coordinates are the verified text mode in the staging SDK image.
    layers.add_text_layer("text",
                          visualization::TextLayerDesc{.max_glyphs = detail::kDashboardTextRows *
                                                                     detail::kDashboardTextBytes});
    return layers;
  }
};

}  // namespace holoscan::radar::app::viz
