// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TileMosaicOp: blit tiles into a parent, wrap-clear, and ping-pong
// output slots so a held previous emit is not overwritten on the
// next tick.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "custom_ops.hpp"  // NOLINT(build/include_subdir)
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::to_host;
using holoscan::radar::testing::wire_data_source;

SampleGeometry geom_at(double y0, int64_t nv, int64_t nu) {
  SampleGeometrySpec spec;
  spec.origin = {0.0, y0, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {1.0, 1.0};
  spec.counts = {nv, nu};
  return SampleGeometry(spec);
}

holoscan::TensorMap tile_filled(int64_t nv, int64_t nu,
                                std::complex<double> value) {
  const size_t n = static_cast<size_t>(nv * nu);
  std::vector<std::complex<double>> host(n, value);
  return single_map(
      holoscan::radar::testing::complex_cube_to_device<double>(
          host, {nv, nu}),
      "image");
}

std::complex<double> at(const HostTensor& t, int64_t v, int64_t u) {
  const int64_t nu = t.shape[1];
  return t.as<std::complex<double>>()[v * nu + u];
}

struct HeldEmits {
  std::vector<std::shared_ptr<holoscan::Tensor>> device;
  std::vector<HostTensor> at_recv;
  std::vector<HostTensor> prev_after_this;

  std::vector<std::function<void(const holoscan::TensorMap&)>>
  validators() {
    return {[this](const holoscan::TensorMap& map) {
      auto tensor = map.begin()->second;
      if (!device.empty()) {
        prev_after_this.push_back(to_host(*device.back()));
      }
      device.push_back(tensor);
      at_recv.push_back(to_host(*tensor));
    }};
  }
};

class MosaicApp : public holoscan::Application {
 public:
  SampleGeometry parent;
  std::vector<SampleGeometry> tiles;
  std::vector<holoscan::TensorMap> frames;
  bool loop{false};
  HeldEmits out;

  void compose() override {
    auto mosaic = make_operator<TileMosaicOp>("mosaic");
    mosaic->parent = parent;
    mosaic->tiles = tiles;
    mosaic->loop = loop;
    wire_data_source(this, mosaic, "image", frames);
    auto sink = make_operator<
        holoscan::test::TestHarnessSinkOp<holoscan::TensorMap>>(
        "sink");
    sink->set_validators(out.validators());
    add_flow(mosaic, sink, {{"image", "input"}});
  }
};

TEST(TileMosaic, AccumulatesAndHoldsPriorEmit) {
  auto app = holoscan::make_application<MosaicApp>();
  app->parent = geom_at(0.0, 8, 4);
  app->tiles = {geom_at(0.0, 4, 4), geom_at(4.0, 4, 4)};
  app->frames = {tile_filled(4, 4, {1.0, 0.0}),
                 tile_filled(4, 4, {2.0, 0.0})};
  app->run();

  ASSERT_EQ(app->out.at_recv.size(), 2u);
  const auto& f0 = app->out.at_recv[0];
  const auto& f1 = app->out.at_recv[1];
  ASSERT_EQ(f0.shape.size(), 2u);
  EXPECT_EQ(f0.shape[0], 8);
  EXPECT_EQ(f0.shape[1], 4);
  EXPECT_DOUBLE_EQ(at(f0, 0, 0).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(f0, 3, 3).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(f0, 4, 0).real(), 0.0);
  EXPECT_DOUBLE_EQ(at(f1, 0, 0).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(f1, 4, 0).real(), 2.0);
  EXPECT_DOUBLE_EQ(at(f1, 7, 3).real(), 2.0);

  ASSERT_EQ(app->out.prev_after_this.size(), 1u);
  const auto& held0 = app->out.prev_after_this[0];
  EXPECT_DOUBLE_EQ(at(held0, 0, 0).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(held0, 4, 0).real(), 0.0);
  EXPECT_NE(app->out.device[0]->data(), app->out.device[1]->data());
}

TEST(TileMosaic, WrapClearsParent) {
  auto app = holoscan::make_application<MosaicApp>();
  app->parent = geom_at(0.0, 8, 4);
  app->tiles = {geom_at(0.0, 4, 4), geom_at(4.0, 4, 4)};
  app->loop = true;
  app->frames = {tile_filled(4, 4, {1.0, 0.0}),
                 tile_filled(4, 4, {2.0, 0.0}),
                 tile_filled(4, 4, {3.0, 0.0})};
  app->run();

  ASSERT_EQ(app->out.at_recv.size(), 3u);
  const auto& wrap = app->out.at_recv[2];
  EXPECT_DOUBLE_EQ(at(wrap, 0, 0).real(), 3.0);
  EXPECT_DOUBLE_EQ(at(wrap, 3, 3).real(), 3.0);
  EXPECT_DOUBLE_EQ(at(wrap, 4, 0).real(), 0.0);
  EXPECT_EQ(app->out.device[0]->data(), app->out.device[2]->data());
}

}  // namespace
