// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <memory>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "radar_app/app/graph_driver.hpp"
#include "radar_app/app/visualization_config.hpp"

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
#error This regression must compile the driver without optional HoloViz support
#endif

namespace {
namespace app = holoscan::radar::app;
struct Config {
  int num_frames{1};
  bool visualize{true}, headless{false}, skip_display_check{false};
  app::VisualizationConfig visualization;
};
class CompleteFrameOp : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec&) override {}
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    app::AppFrameBank::last_capture_ns.store(1);
    app::AppFrameBank::mark_done();
    return {};
  }
};

TEST(FmcwWithoutHoloviz, SkipsDisplayDrainAndIndefiniteWindowHold) {
  for (const double hold : {0.0, -1.0}) {
    Config requested;
    requested.visualization.final_image_hold_s = hold;
    char name[] = "no-holoviz-regression";
    char* argv[] = {name};
    const auto loader = [requested](const std::filesystem::path&) {
      return requested;
    };
    const auto build = [](holoscan::Graph& graph, holoscan::CompileOptions&, const Config& cfg,
                          std::shared_ptr<app::CadenceMeter>*) {
      EXPECT_FALSE(cfg.visualize);
      EXPECT_FALSE(cfg.visualization.enabled);
      graph.op<CompleteFrameOp>("complete");
    };
    EXPECT_EQ(app::run_fmcw_graph_app<Config>(1, argv, "no-holoviz-regression", "unused.yaml",
                                              loader, build),
              0);
  }
}
}  // namespace
