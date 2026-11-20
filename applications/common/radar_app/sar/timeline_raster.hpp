// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "radar_common/sar/formation_hud.hpp"
#include "radar_app/app/gui_lifecycle.hpp"
#include "radar_app/viz/frame_viewport.hpp"

namespace holoscan::radar::app::sar {

// 20 s history ending at the latest source-batch receipt or observed GPU
// completion, with checkpoint progress and a 1 Hz optional playback-lag HUD.
// Polls FormationHud (live mailbox), not Graph handles.
class TimelineRasterOp : public Operator<> {
 public:
  TimelineRasterOp(std::shared_ptr<radar::FormationHud> hud, std::vector<std::string> scene_names,
                   bool show_playback_lag, double rate = 10.0, int64_t width = kWidth,
                   int64_t height = kHeight, std::shared_ptr<FrameProgress> progress = {},
                   std::shared_ptr<viz::FrameViewport> viewport = {});
  ~TimelineRasterOp() override;
  void start() override;
  void stop() override;
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  Output<schema::ImageT> frame{"frame"};
  Output<Tensor> labels{"labels"};

  static constexpr int64_t kWidth = 1280;
  static constexpr int64_t kHeight = 112;
  static constexpr double kWindowS = 20.0;

 private:
  std::shared_ptr<radar::FormationHud> hud_;
  std::vector<std::string> scene_names_;
  bool show_playback_lag_{false};
  std::shared_ptr<FrameProgress> progress_;
  std::shared_ptr<radar::sar_progress::Monitor> pulse_progress_;
  radar::sar_progress::View pulse_view_;
  std::chrono::nanoseconds period_;
  int64_t width_, height_;
  std::shared_ptr<viz::FrameViewport> viewport_;
  void* commands_{nullptr};
  struct RenderCache;
  std::unique_ptr<RenderCache> rendered_;
  std::chrono::steady_clock::time_point hud_at_{};
  bool hud_held_{false};
  std::string lag_hud_;
  std::array<uint8_t, 3> lag_hud_fg_{{80, 200, 80}};
  int lag_band_{0};
};

}  // namespace holoscan::radar::app::sar
