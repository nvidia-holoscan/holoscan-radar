// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TileMosaicOp / ScrollMosaicOp: blit a chip into a parent, wrap-
// clear, and ping-pong output slots so a held previous emit is
// not overwritten on the next tick.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "custom_ops.hpp"  // NOLINT(build/include_subdir)
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sdk/stream_boundary.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::kStreamBoundaryCollection;
using holoscan::radar::kStreamBoundaryKey;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::to_host;
using holoscan::radar::testing::TensorMapCapture;
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

class ScrollMosaicApp : public holoscan::Application {
 public:
  SampleGeometry parent;
  std::vector<SampleGeometry> windows;
  std::vector<holoscan::TensorMap> frames;
  bool loop{false};
  HeldEmits out;

  void compose() override {
    auto mosaic = make_operator<ScrollMosaicOp>("scroll_mosaic");
    mosaic->parent = parent;
    mosaic->loop = loop;
    wire_data_source(this, mosaic, "image", frames);
    auto geom_src = make_operator<
        holoscan::test::TestHarnessSourceOp<SampleGeometry>>(
        "geom_source",
        make_condition<holoscan::CountCondition>(
            static_cast<int64_t>(windows.size())));
    geom_src->set_test_data(windows);
    add_flow(geom_src, mosaic, {{"output", "sample_geometry"}});
    auto sink = make_operator<
        holoscan::test::TestHarnessSinkOp<holoscan::TensorMap>>(
        "sink");
    sink->set_validators(out.validators());
    add_flow(mosaic, sink, {{"image", "input"}});
  }
};

TEST(ScrollMosaic, AccumulatesOverlapAndHoldsPriorEmit) {
  auto app = holoscan::make_application<ScrollMosaicApp>();
  app->parent = geom_at(0.0, 8, 4);
  app->windows = {geom_at(0.0, 4, 4), geom_at(2.0, 4, 4)};
  app->frames = {tile_filled(4, 4, {1.0, 0.0}),
                 tile_filled(4, 4, {2.0, 0.0})};
  app->run();

  ASSERT_EQ(app->out.at_recv.size(), 2u);
  const auto& f0 = app->out.at_recv[0];
  const auto& f1 = app->out.at_recv[1];
  EXPECT_EQ(f0.shape[0], 8);
  EXPECT_EQ(f0.shape[1], 4);
  EXPECT_DOUBLE_EQ(at(f0, 0, 0).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(f0, 3, 3).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(f0, 4, 0).real(), 0.0);
  EXPECT_DOUBLE_EQ(at(f1, 0, 0).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(f1, 1, 3).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(f1, 2, 0).real(), 2.0);
  EXPECT_DOUBLE_EQ(at(f1, 5, 3).real(), 2.0);
  EXPECT_DOUBLE_EQ(at(f1, 6, 0).real(), 0.0);

  ASSERT_EQ(app->out.prev_after_this.size(), 1u);
  const auto& held0 = app->out.prev_after_this[0];
  EXPECT_DOUBLE_EQ(at(held0, 0, 0).real(), 1.0);
  EXPECT_DOUBLE_EQ(at(held0, 4, 0).real(), 0.0);
  EXPECT_NE(app->out.device[0]->data(), app->out.device[1]->data());
}

TEST(ScrollMosaic, WrapClearsParent) {
  auto app = holoscan::make_application<ScrollMosaicApp>();
  app->parent = geom_at(0.0, 8, 4);
  app->windows = {geom_at(0.0, 4, 4), geom_at(4.0, 4, 4),
                  geom_at(0.0, 4, 4)};
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

holoscan::TensorMap xyz_y(const std::vector<double>& y) {
  const int64_t np = static_cast<int64_t>(y.size());
  std::vector<double> host(static_cast<size_t>(np * 3), 0.0);
  for (int64_t p = 0; p < np; ++p) {
    host[static_cast<size_t>(p * 3 + 1)] = y[static_cast<size_t>(p)];
  }
  return single_map(real_cube_to_device<double>(host, {np, 3}),
                    "reference_position");
}

holoscan::TensorMap profiles_n(int64_t np) {
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(np * 2), {1.0, 0.0});
  for (int64_t p = 0; p < np; ++p) {
    cube[static_cast<size_t>(p * 2)] = {
        static_cast<double>(p), 0.0};
  }
  return single_map(
      holoscan::radar::testing::complex_cube_to_device<double>(
          cube, {np, 2}),
      "range_profiles");
}

class StagerCaptureOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StagerCaptureOp)
  StagerCaptureOp() = default;
  TensorMapCapture* profiles{nullptr};
  std::vector<SampleGeometry>* geometries{nullptr};
  int64_t capacity{8};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>(
            "range_profiles", holoscan::IOSpec::IOSize{capacity})
        .condition(holoscan::ConditionType::kMessageAvailable,
                   holoscan::Arg("min_size") = uint64_t{1});
    spec.input<SampleGeometry>(
            "sample_geometry", holoscan::IOSpec::IOSize{capacity})
        .condition(holoscan::ConditionType::kNone);
  }

  void initialize() override {
    if (spec() != nullptr && capacity > 1) {
      auto raise = [&](const char* name) {
        auto it = spec()->inputs().find(name);
        if (it != spec()->inputs().end() && it->second) {
          it->second->queue_size(capacity);
        }
      };
      raise("range_profiles");
      raise("sample_geometry");
    }
    Operator::initialize();
  }

  void start() override {
    metadata_policy(holoscan::MetadataPolicy::kUpdate);
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto maybe = op_input.receive<holoscan::TensorMap>(
        "range_profiles");
    if (maybe && profiles != nullptr) {
      profiles->validators()[0](maybe.value());
    }
    auto geom = op_input.receive<SampleGeometry>("sample_geometry");
    if (geom && geometries != nullptr) {
      geometries->push_back(*geom);
    }
  }
};

class StampTickOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StampTickOp)
  StampTickOp() = default;
  std::vector<std::string> stamps;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in");
    spec.output<holoscan::TensorMap>("out");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    auto tensor = holoscan::radar::receive_single(
        op_input, "in", "StampTickOp");
    if (tick_ < stamps.size() && !stamps[tick_].empty()) {
      metadata()->set(kStreamBoundaryKey, stamps[tick_]);
    }
    ++tick_;
    holoscan::radar::emit_single(op_output, "out", tensor);
  }

 private:
  size_t tick_{0};
};

class ScrollStagerApp : public holoscan::Application {
 public:
  SampleGeometry parent;
  SampleGeometry window0;
  int64_t anchor{0};
  int64_t slice_pulses{0};
  int64_t dispatch_capacity{4};
  bool loop{false};
  std::vector<holoscan::TensorMap> profile_frames;
  std::vector<holoscan::TensorMap> position_frames;
  std::vector<holoscan::TensorMap> reference_frames;
  std::vector<std::string> stamps;
  TensorMapCapture profiles;
  std::vector<SampleGeometry> geometries;
  ScrollGeometryStagerOp* stager{nullptr};

  void compose() override {
    auto op = make_operator<ScrollGeometryStagerOp>("stager");
    op->parent = parent;
    op->window0 = window0;
    op->anchor = anchor;
    op->loop = loop;
    op->slice_pulses = slice_pulses;
    op->dispatch_capacity = dispatch_capacity;
    stager = op.get();

    if (!stamps.empty()) {
      auto stamp = make_operator<StampTickOp>("stamp");
      stamp->stamps = stamps;
      wire_data_source(this, stamp, "in", profile_frames);
      add_flow(stamp, op, {{"out", "range_profiles"}});
    } else {
      wire_data_source(this, op, "range_profiles", profile_frames);
    }
    wire_data_source(this, op, "positions", position_frames);
    wire_data_source(this, op, "reference_position",
                     reference_frames);

    auto sink = make_operator<StagerCaptureOp>("sink");
    sink->capacity = dispatch_capacity;
    sink->profiles = &profiles;
    sink->geometries = &geometries;
    add_flow(op, sink, {{"range_profiles", "range_profiles"}});
    add_flow(op, sink, {{"sample_geometry", "sample_geometry"}});
  }
};

TEST(ScrollGeometryStager, PassThroughOneWindow) {
  auto app = holoscan::make_application<ScrollStagerApp>();
  app->parent = geom_at(0.0, 16, 2);
  app->window0 = geom_at(0.0, 4, 2);
  app->profile_frames = {profiles_n(4)};
  app->position_frames = {xyz_y({0.0, 1.0, 2.0, 3.0})};
  app->reference_frames = {xyz_y({0.0, 1.0, 2.0, 3.0})};
  app->run();

  ASSERT_EQ(app->profiles.frames.size(), 1u);
  EXPECT_EQ(app->profiles.frames[0].shape[0], 4);
  ASSERT_EQ(app->geometries.size(), 1u);
  EXPECT_EQ(app->geometries[0], geom_at(3.0, 4, 2));
}

TEST(ScrollGeometryStager, SlicesAndAdvancesPerSlice) {
  auto app = holoscan::make_application<ScrollStagerApp>();
  app->parent = geom_at(0.0, 16, 2);
  app->window0 = geom_at(0.0, 4, 2);
  app->slice_pulses = 2;
  app->dispatch_capacity = 2;
  app->profile_frames = {profiles_n(4)};
  app->position_frames = {xyz_y({0.0, 1.0, 2.0, 3.0})};
  app->reference_frames = {xyz_y({0.0, 1.0, 2.0, 3.0})};
  app->run();

  ASSERT_EQ(app->profiles.frames.size(), 2u);
  EXPECT_EQ(app->profiles.frames[0].shape[0], 2);
  EXPECT_EQ(app->profiles.frames[1].shape[0], 2);
  ASSERT_EQ(app->geometries.size(), 2u);
  EXPECT_EQ(app->geometries[0], geom_at(1.0, 4, 2));
  EXPECT_EQ(app->geometries[1], geom_at(3.0, 4, 2));
}

TEST(ScrollGeometryStager, ShortFinalSlice) {
  auto app = holoscan::make_application<ScrollStagerApp>();
  app->parent = geom_at(0.0, 16, 2);
  app->window0 = geom_at(0.0, 4, 2);
  app->slice_pulses = 2;
  app->dispatch_capacity = 3;
  app->profile_frames = {profiles_n(5)};
  app->position_frames = {xyz_y({0.0, 1.0, 2.0, 3.0, 4.0})};
  app->reference_frames = {xyz_y({0.0, 1.0, 2.0, 3.0, 4.0})};
  app->run();

  ASSERT_EQ(app->profiles.frames.size(), 3u);
  EXPECT_EQ(app->profiles.frames[0].shape[0], 2);
  EXPECT_EQ(app->profiles.frames[1].shape[0], 2);
  EXPECT_EQ(app->profiles.frames[2].shape[0], 1);
  ASSERT_EQ(app->geometries.size(), 3u);
  EXPECT_EQ(app->geometries[2], geom_at(4.0, 4, 2));
}

TEST(ScrollGeometryStager, WrapRestagesOnLastSlice) {
  auto app = holoscan::make_application<ScrollStagerApp>();
  app->parent = geom_at(0.0, 16, 2);
  app->window0 = geom_at(0.0, 4, 2);
  app->slice_pulses = 2;
  app->dispatch_capacity = 2;
  app->loop = true;
  app->stamps = {"", std::string(kStreamBoundaryCollection)};
  app->profile_frames = {profiles_n(4), profiles_n(4)};
  app->position_frames = {xyz_y({0.0, 1.0, 2.0, 3.0}),
                          xyz_y({4.0, 5.0, 6.0, 7.0})};
  app->reference_frames = {xyz_y({0.0, 1.0, 2.0, 3.0}),
                           xyz_y({4.0, 5.0, 6.0, 7.0})};
  app->run();

  ASSERT_GE(app->geometries.size(), 3u);
  EXPECT_EQ(app->geometries.back(), geom_at(0.0, 4, 2));
  EXPECT_EQ(app->stager->current_window(), geom_at(0.0, 4, 2));
}

}  // namespace
