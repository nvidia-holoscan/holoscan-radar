// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <functional>
#include <chrono>
#include <cstdint>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "radar_app/sar/sar_overlay.hpp"

namespace holoscan::radar::app::sar {
// GPU LUT of a formed SAR image for Holoviz. `display_side` downsamples the
// display raster only. Do not sleep here: this operator shares a Graph
// partition with Holoviz, so a hold would block presentation of the frame it
// just emitted. Finite `visualize_hold_s` is the session wait in
// `run_sar_graph_app`. Optional `overlay` fills host Holoviz text (`labels`)
// and line-list (`overlay`) layers. The docked layout places text in a footer
// viewport while keeping the APC marker over the image. `TimelineRasterOp`
// independently renders the Gantt panel. Interactive runs enable a clocked,
// one-shot startup canvas; after handoff, only new images activate GPU rendering.
class SarImageVizOp : public Operator<> {
 public:
  // `product_extent` is the formed product's size when the tensor on `image`
  // is a decimated view of it, as a scroll mosaic publishes. The HUD banner
  // names the product; the view's own size is a transport detail. {0, 0}
  // reports the tensor.
  SarImageVizOp(int64_t max_v, int64_t max_u, int64_t display_side, bool transpose_axes,
                OverlayConfig overlay = {}, std::array<int64_t, 2> product_extent = {0, 0},
                double display_rate_hz = 60.0, std::function<bool()> terminal = {},
                bool startup_placeholder = false);
  ~SarImageVizOp() override;
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  Input<Tensor> image{"image"};
  Output<schema::ImageT> frame{"frame"};
  Output<Tensor> labels{"labels"};
  Output<Tensor> overlay{"overlay"};

 private:
  expected<void, Error> emit_startup(ExecutionContext& context);
  bool startup_placeholder_{false}, startup_emitted_{false};
  int64_t max_v_{0}, max_u_{0}, display_side_{0};
  bool transpose_axes_{false};
  OverlayConfig overlay_{};
  std::array<int64_t, 2> product_extent_{0, 0};
  int overlay_index_{0};
  double display_rate_hz_;
  std::function<bool()> terminal_;
  std::optional<std::chrono::steady_clock::time_point> last_display_at_;
  std::chrono::steady_clock::time_point overlay_rate_at_{};
  bool have_overlay_rate_{false};
  std::string held_rate_line_{};
  float* sampled_magnitude_{nullptr};
};
}  // namespace holoscan::radar::app::sar
