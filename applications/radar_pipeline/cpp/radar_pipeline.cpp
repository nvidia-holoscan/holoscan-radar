// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// C++ pipeline example for radar.
// TODO: replace the stub with your actual pipeline topology.
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/radar_pipeline/radar_pipeline

#include <holoscan/holoscan.hpp>
#include <radar_op/radar_op.hpp>

namespace mm = holoscan::radar;

class RadarPipelineApp : public holoscan::Application {
 public:
  void compose() override {
    // TODO: instantiate and connect your operators, e.g.:
    auto op = make_operator<mm::RadarOp>("radar_op");
    // add_flow(source, op, ('out', 'in'));
    
    HOLOSCAN_LOG_INFO("TODO: RadarPipelineApp is not yet implemented");
  }
};

int main(int, char**) {
  auto app = holoscan::make_application<RadarPipelineApp>();
  app->run();
  return 0;
}
