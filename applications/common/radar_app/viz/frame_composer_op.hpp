// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "radar_app/viz/dashboard_text.hpp"

namespace holoscan::radar::app::viz {

inline visualization::LayerStack dashboard_image_layers(std::int32_t width, std::int32_t height,
                                                        float image_fraction = 1.0F) {
  visualization::LayerStack layers;
  layers.add_image_layer(
      "image",
      visualization::ImageLayerDesc{
          .format = visualization::ImageFormat::kR8G8B8A8Unorm,
          .max_width = static_cast<std::uint32_t>(width),
          .max_height = static_cast<std::uint32_t>(height),
          .view = image_fraction == 1.0F
                      ? std::nullopt
                      : std::optional{visualization::ViewTransform{.scale_y = image_fraction}}});
  return layers;
}

// Compose native SDK text and the radar canvas into one CUDA image. The display
// scales that complete frame, so resizing cannot separate labels from their axes.
class FrameComposerOp : public holoscan::visualization::HolovizOp {
 public:
  FrameComposerOp(std::int32_t width, std::int32_t height,
                  std::uint32_t max_glyphs = detail::kDashboardTextRows *
                                             detail::kDashboardTextBytes,
                  bool normalized_text = false, std::uint32_t max_lines = 0,
                  std::int32_t image_max_side = 0, std::int32_t image_view_height = 0)
  : HolovizOp(make_layers(width, height, max_glyphs, normalized_text, max_lines, image_max_side,
                          image_view_height),
              visualization::CudaSinkDesc{.width = static_cast<std::uint32_t>(width),
                                          .height = static_cast<std::uint32_t>(height)}),
    width_(width),
    height_(height),
    max_lines_(max_lines) {}

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
    if (max_lines_ != 0) {
      spec.input(tensor_layer_input(2), "geometry")
          .queue_depth(2U)
          .expects_tensor(TensorInputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                                             .dtype = DLDataType{kDLFloat, 32U, 1U},
                                                             .rank = 1U}});
    }
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
    if (max_lines_ != 0) {
      result
          .trigger(OnAligned{
              .ports = ports(image_layer_input(0), tensor_layer_input(1), tensor_layer_input(2)),
              .window = std::chrono::nanoseconds{0}})
          .input(image_layer_input(0), Exact{})
          .input(tensor_layer_input(1), Exact{})
          .input(tensor_layer_input(2), Exact{});
    } else {
      result
          .trigger(OnAligned{.ports = ports(image_layer_input(0), tensor_layer_input(1)),
                             .window = std::chrono::nanoseconds{0}})
          .input(image_layer_input(0), Exact{})
          .input(tensor_layer_input(1), Exact{});
    }
    return result;
  }

 private:
  static visualization::LayerStack make_layers(std::int32_t width, std::int32_t height,
                                               std::uint32_t max_glyphs, bool normalized_text,
                                               std::uint32_t max_lines, std::int32_t image_max_side,
                                               std::int32_t image_view_height) {
    const auto side = std::max({width, height, image_max_side});
    auto layers = dashboard_image_layers(
        side, side, image_view_height > 0 ? static_cast<float>(image_view_height) / height : 1.0F);
    layers.add_text_layer("text",
                          visualization::TextLayerDesc{.max_glyphs = max_glyphs,
                                                       .normalized_coordinates = normalized_text});
    if (max_lines != 0)
      layers.add_geometry_layer("geometry",
                                visualization::GeometryLayerDesc{
                                    .topologies = {visualization::GeometryTopology::kLineList},
                                    .max_primitives = max_lines,
                                    .default_line_width = 2.0F});
    return layers;
  }

  std::int32_t width_;
  std::int32_t height_;
  std::uint32_t max_lines_;
};

}  // namespace holoscan::radar::app::viz
