// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "radar_app/viz/frame_composer_op.hpp"

namespace holoscan::radar::app::viz {
class NativeImagePresenterOp final : public holoscan::visualization::HolovizOp {
 public:
  NativeImagePresenterOp(std::int32_t width, std::int32_t height, std::string title)
  : HolovizOp(dashboard_image_layers(width, height),
              visualization::WindowSinkDesc{.title = std::move(title),
                                            .width = static_cast<std::uint32_t>(width),
                                            .height = static_cast<std::uint32_t>(height),
                                            .close_on_escape = true}) {}
};
}  // namespace holoscan::radar::app::viz
