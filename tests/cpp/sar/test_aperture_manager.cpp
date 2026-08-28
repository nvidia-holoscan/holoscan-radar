// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ApertureManagerOp unit tests. Pulse identity is the first range bin
// (real part = pulse id) and positions[:, 0] (same id). Assertions are
// on emitted identities, weights, and control — no GPU image.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/window.hpp"
#include "radar_common/sdk/stream_boundary.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "sar/aperture_manager_op/aperture_manager_op.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ApertureEmitMode;
using holoscan::radar::ApertureManagerOp;
using holoscan::radar::FormationControl;
using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::kStreamBoundaryCollection;
using holoscan::radar::kStreamBoundaryEndOfStream;
using holoscan::radar::kStreamBoundaryKey;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_uint8_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::TestHarnessSinkOp;
using holoscan::test::TestHarnessSourceOp;

constexpr int64_t kBins = 4;

SampleGeometry make_geom(double origin_x) {
  SampleGeometrySpec spec;
  spec.origin = {origin_x, 0.0, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {1.0, 1.0};
  spec.counts = {2, 2};
  return SampleGeometry(spec);
}

template <typename T>
holoscan::TensorMap profiles_ids(const std::vector<int64_t>& ids) {
  const int64_t np = static_cast<int64_t>(ids.size());
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(np * kBins), {0.0, 0.0});
  for (int64_t p = 0; p < np; ++p) {
    cube[static_cast<size_t>(p * kBins)] = {
        static_cast<double>(ids[static_cast<size_t>(p)]), 0.0};
  }
  return single_map(
      holoscan::radar::testing::complex_cube_to_device<T>(
          cube, {np, kBins}),
      "range_profiles");
}

template <typename T>
holoscan::TensorMap xyz_ids(const std::vector<int64_t>& ids,
                            const char* key) {
  const int64_t np = static_cast<int64_t>(ids.size());
  std::vector<double> host(static_cast<size_t>(np * 3), 0.0);
  for (int64_t p = 0; p < np; ++p) {
    host[static_cast<size_t>(p * 3)] =
        static_cast<double>(ids[static_cast<size_t>(p)]);
  }
  return single_map(real_cube_to_device<T>(host, {np, 3}), key);
}

std::vector<int64_t> pulse_ids(const HostTensor& positions) {
  std::vector<int64_t> ids;
  const int64_t np = positions.shape.empty() ? 0 : positions.shape[0];
  const auto* data = positions.as<double>();
  // Tests run the double instantiation unless noted.
  if (positions.dtype.bits == 32) {
    const auto* f = positions.as<float>();
    for (int64_t p = 0; p < np; ++p) {
      ids.push_back(static_cast<int64_t>(f[p * 3] + 0.5f));
    }
    return ids;
  }
  for (int64_t p = 0; p < np; ++p) {
    ids.push_back(static_cast<int64_t>(data[p * 3] + 0.5));
  }
  return ids;
}

template <typename T>
std::vector<T> weights_of(const HostTensor& tensor) {
  const auto* data = tensor.as<T>();
  return std::vector<T>(data, data + tensor.size());
}

FormationControl control_of(const HostTensor& tensor) {
  return static_cast<FormationControl>(tensor.as<uint8_t>()[0]);
}

class StampBoundaryOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StampBoundaryOp)
  StampBoundaryOp() = default;
  std::vector<std::string> stamps;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in");
    spec.output<holoscan::TensorMap>("out");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    auto tensor = holoscan::radar::receive_single(
        op_input, "in", "StampBoundaryOp");
    if (tick_ < stamps.size() && !stamps[tick_].empty()) {
      metadata()->set(kStreamBoundaryKey, stamps[tick_]);
    }
    ++tick_;
    holoscan::radar::emit_single(op_output, "out", tensor);
  }

 private:
  size_t tick_{0};
};

template <typename T>
class ManagerApp : public holoscan::Application {
 public:
  std::vector<std::vector<int64_t>> batches;
  std::vector<std::string> stamps;
  std::vector<holoscan::Arg> args;
  std::vector<holoscan::TensorMap> evict_frames;
  std::vector<holoscan::TensorMap> control_frames;
  std::vector<SampleGeometry> geometry_overrides;
  TensorMapCapture profiles;
  TensorMapCapture positions;
  TensorMapCapture reference;
  TensorMapCapture weight;
  TensorMapCapture control;
  std::vector<SampleGeometry> forwarded;
  ApertureManagerOp<T>* op{nullptr};

  void compose() override {
    std::vector<holoscan::TensorMap> prof;
    std::vector<holoscan::TensorMap> pos;
    std::vector<holoscan::TensorMap> ref;
    for (const auto& ids : batches) {
      prof.push_back(profiles_ids<T>(ids));
      pos.push_back(xyz_ids<T>(ids, "positions"));
      ref.push_back(xyz_ids<T>(ids, "reference_position"));
    }
    holoscan::ArgList arg_list;
    for (const holoscan::Arg& arg : args) {
      arg_list.add(arg);
    }
    auto manager = make_operator<ApertureManagerOp<T>>(
        "manager", arg_list);
    op = manager.get();

    auto stamp = make_operator<StampBoundaryOp>("stamp");
    stamp->stamps = stamps;

    auto src_p = make_operator<
        TestHarnessSourceOp<holoscan::TensorMap>>(
        "prof_src",
        make_condition<holoscan::CountCondition>(
            static_cast<int64_t>(prof.size())));
    src_p->set_test_data(std::move(prof));
    add_flow(src_p, stamp, {{"output", "in"}});
    add_flow(stamp, manager, {{"out", "range_profiles"}});

    wire_data_source(this, manager, "positions", std::move(pos));
    wire_data_source(this, manager, "reference_position",
                     std::move(ref));
    if (!evict_frames.empty()) {
      wire_data_source(this, manager, "evict", evict_frames);
    }
    if (!control_frames.empty()) {
      wire_data_source(this, manager, "control", control_frames);
    }
    if (!geometry_overrides.empty()) {
      wire_leading_override(this, manager, "sample_geometry",
                            geometry_overrides);
    }
    wire_capture_sink(this, manager, "range_profiles", &profiles);
    wire_capture_sink(this, manager, "positions", &positions);
    wire_capture_sink(this, manager, "reference_position",
                      &reference);
    wire_capture_sink(this, manager, "weight", &weight);
    wire_capture_sink(this, manager, "control", &control);

    auto geom_sink = make_operator<TestHarnessSinkOp<SampleGeometry>>(
        "geom_sink");
    geom_sink->set_validators(
        {[this](const SampleGeometry& geom) {
          forwarded.push_back(geom);
        }});
    add_flow(manager, geom_sink, {{"sample_geometry", "input"}});
  }
};

template <typename T>
std::unique_ptr<ManagerApp<T>> run_app(ManagerApp<T>* raw) {
  auto app = std::unique_ptr<ManagerApp<T>>(raw);
  app->run();
  return app;
}

holoscan::TensorMap u32_map(uint32_t value) {
  auto tensor = holoscan::radar::make_device_tensor(
      {1}, DLDataType{kDLUInt, 32, 1});
  if (cudaMemcpy(tensor->data(), &value, sizeof(value),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("u32_map: H2D failed");
  }
  return single_map(tensor, "evict");
}

holoscan::TensorMap ctrl_map(FormationControl value) {
  return single_map(
      to_device_uint8_tensor({static_cast<uint8_t>(value)}, {1}),
      "control");
}

TEST(ApertureManager, PassThroughD22EmitAndReset) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}, {4}};
  raw->stamps = {"", "", std::string(kStreamBoundaryEndOfStream)};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
               holoscan::Arg("emit_interval_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kReset};
  auto app = run_app(raw);
  ASSERT_EQ(app->positions.frames.size(), 3u);
  EXPECT_EQ(pulse_ids(app->positions.frames[0]),
            (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(pulse_ids(app->positions.frames[1]),
            (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(pulse_ids(app->positions.frames[2]),
            (std::vector<int64_t>{4}));
  ASSERT_EQ(app->control.frames.size(), 1u);
  EXPECT_EQ(control_of(app->control.frames[0]),
            FormationControl::kEmitAndReset);
  EXPECT_EQ(app->op->retained_pulses(), 0);
  EXPECT_EQ(app->op->admitted_pulses(), 5);
}

TEST(ApertureManager, IntervalZeroEmitsOnNoCount) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
               holoscan::Arg("emit_interval_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kReset};
  auto app = run_app(raw);
  EXPECT_TRUE(app->control.frames.empty());
  EXPECT_EQ(app->positions.frames.size(), 2u);
}

TEST(ApertureManager, CadenceReset) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
               holoscan::Arg("emit_interval_pulses") = int64_t{2},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kReset};
  auto app = run_app(raw);
  ASSERT_EQ(app->control.frames.size(), 2u);
  EXPECT_EQ(control_of(app->control.frames[0]),
            FormationControl::kEmitAndReset);
  EXPECT_EQ(control_of(app->control.frames[1]),
            FormationControl::kEmitAndReset);
}

TEST(ApertureManager, GrowingApertureCadenceContinue) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}, {4}};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
               holoscan::Arg("emit_interval_pulses") = int64_t{2},
               holoscan::Arg("emit_mode") =
                   ApertureEmitMode::kContinue};
  auto app = run_app(raw);
  ASSERT_EQ(app->positions.frames.size(), 3u);
  EXPECT_EQ(pulse_ids(app->positions.frames[0]),
            (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(pulse_ids(app->positions.frames[1]),
            (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(pulse_ids(app->positions.frames[2]),
            (std::vector<int64_t>{4}));
  ASSERT_EQ(app->control.frames.size(), 2u);
  EXPECT_EQ(control_of(app->control.frames[0]),
            FormationControl::kEmitAndContinue);
  EXPECT_EQ(control_of(app->control.frames[1]),
            FormationControl::kEmitAndContinue);
  EXPECT_EQ(app->op->retained_pulses(), 0);
  EXPECT_EQ(app->op->evicted_pulses(), 0);
  for (const auto& frame : app->weight.frames) {
    for (double w : weights_of<double>(frame)) {
      EXPECT_DOUBLE_EQ(w, 1.0);
    }
  }
}

TEST(ApertureManager, ManagerOwnedSlidingEvict) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}, {4, 5}};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{4},
               holoscan::Arg("emit_interval_pulses") = int64_t{2},
               holoscan::Arg("emit_mode") =
                   ApertureEmitMode::kContinue};
  auto app = run_app(raw);
  ASSERT_EQ(app->positions.frames.size(), 3u);
  EXPECT_EQ(pulse_ids(app->positions.frames[0]),
            (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(pulse_ids(app->positions.frames[1]),
            (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(pulse_ids(app->positions.frames[2]),
            (std::vector<int64_t>{4, 5, 0, 1}));
  const auto w2 = weights_of<double>(app->weight.frames[2]);
  ASSERT_EQ(w2.size(), 4u);
  EXPECT_DOUBLE_EQ(w2[0], 1.0);
  EXPECT_DOUBLE_EQ(w2[1], 1.0);
  EXPECT_DOUBLE_EQ(w2[2], -1.0);
  EXPECT_DOUBLE_EQ(w2[3], -1.0);
  EXPECT_EQ(app->op->retained_pulses(), 4);
  EXPECT_EQ(app->op->evicted_pulses(), 2);
}

TEST(ApertureManager, AppOwnedEvictAndOverflow) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}};
  raw->evict_frames = {u32_map(0), u32_map(0)};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{2},
               holoscan::Arg("emit_interval_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") =
                   ApertureEmitMode::kContinue};
  EXPECT_THROW(run_app(raw), std::invalid_argument);
}

TEST(ApertureManager, AppOwnedEvictRetiresOldest) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}};
  raw->evict_frames = {u32_map(0), u32_map(2)};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{4},
               holoscan::Arg("emit_interval_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") =
                   ApertureEmitMode::kContinue};
  auto app = run_app(raw);
  ASSERT_EQ(app->positions.frames.size(), 2u);
  EXPECT_EQ(pulse_ids(app->positions.frames[1]),
            (std::vector<int64_t>{2, 3, 0, 1}));
  EXPECT_EQ(app->op->retained_pulses(), 2);
}

TEST(ApertureManager, ResetDropsWithoutEviction) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}};
  raw->control_frames = {
      ctrl_map(FormationControl::kEmitAndContinue),
      ctrl_map(FormationControl::kEmitAndReset)};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{8},
               holoscan::Arg("emit_interval_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") =
                   ApertureEmitMode::kContinue};
  auto app = run_app(raw);
  ASSERT_EQ(app->weight.frames.size(), 2u);
  for (double w : weights_of<double>(app->weight.frames[1])) {
    EXPECT_DOUBLE_EQ(w, 1.0);
  }
  EXPECT_EQ(app->op->evicted_pulses(), 0);
  EXPECT_EQ(app->op->retained_pulses(), 0);
  EXPECT_EQ(app->op->dropped_pulses(), 4);
}

TEST(ApertureManager, GeometryHeldUntilBoundary) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}};
  raw->stamps = {"", std::string(kStreamBoundaryCollection)};
  raw->geometry_overrides = {make_geom(10.0)};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
               holoscan::Arg("emit_interval_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kReset};
  auto app = run_app(raw);
  ASSERT_EQ(app->forwarded.size(), 1u);
  EXPECT_DOUBLE_EQ(app->forwarded[0].origin()[0], 10.0);
  ASSERT_EQ(app->control.frames.size(), 1u);
}

TEST(ApertureManager, ReformHoldsUntilBoundary) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}};
  raw->stamps = {"", std::string(kStreamBoundaryEndOfStream)};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{8},
               holoscan::Arg("emit_interval_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kReform};
  auto app = run_app(raw);
  ASSERT_EQ(app->positions.frames.size(), 1u);
  EXPECT_EQ(pulse_ids(app->positions.frames[0]),
            (std::vector<int64_t>{0, 1, 2, 3}));
  ASSERT_EQ(app->control.frames.size(), 1u);
  EXPECT_EQ(control_of(app->control.frames[0]),
            FormationControl::kResetThenEmit);
  EXPECT_EQ(app->op->retained_pulses(), 0);
}

TEST(ApertureManager, RequestEqualsD22) {
  auto* raw_d22 = new ManagerApp<double>();
  raw_d22->batches = {{0, 1}, {2, 3}};
  raw_d22->stamps = {"", std::string(kStreamBoundaryEndOfStream)};
  raw_d22->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
                   holoscan::Arg("emit_interval_pulses") = int64_t{0},
                   holoscan::Arg("emit_mode") =
                       ApertureEmitMode::kReset};
  auto d22 = run_app(raw_d22);

  auto* raw_req = new ManagerApp<double>();
  raw_req->batches = {{0, 1}, {2, 3}};
  raw_req->control_frames = {
      ctrl_map(FormationControl::kEmitAndContinue),
      ctrl_map(FormationControl::kEmitAndReset)};
  raw_req->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
                   holoscan::Arg("emit_interval_pulses") = int64_t{0},
                   holoscan::Arg("emit_mode") =
                       ApertureEmitMode::kReset};
  auto req = run_app(raw_req);
  ASSERT_EQ(d22->control.frames.size(), 1u);
  ASSERT_EQ(req->control.frames.size(), 2u);
  EXPECT_EQ(control_of(d22->control.frames[0]),
            control_of(req->control.frames[1]));
}

TEST(ApertureManager, Conservation) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}, {2, 3}, {4, 5}};
  raw->stamps = {"", "", std::string(kStreamBoundaryEndOfStream)};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{4},
               holoscan::Arg("emit_interval_pulses") = int64_t{2},
               holoscan::Arg("emit_mode") =
                   ApertureEmitMode::kContinue};
  auto app = run_app(raw);
  EXPECT_EQ(app->op->admitted_pulses(),
            app->op->evicted_pulses() + app->op->dropped_pulses() +
                app->op->retained_pulses());
}

TEST(ApertureManager, FloatDoublePassThrough) {
  auto* raw = new ManagerApp<float>();
  raw->batches = {{7, 8}};
  raw->stamps = {std::string(kStreamBoundaryEndOfStream)};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
               holoscan::Arg("emit_interval_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kReset};
  auto app = run_app(raw);
  EXPECT_EQ(pulse_ids(app->positions.frames[0]),
            (std::vector<int64_t>{7, 8}));
}

TEST(ApertureManager, ReformRejectedAtZeroCapacity) {
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kReform};
  EXPECT_THROW(run_app(raw), std::invalid_argument);
}

TEST(ApertureManager, WindowOnSlidingRejected) {
  WindowSpec window;
  window.type = WindowType::kHann;
  auto* raw = new ManagerApp<double>();
  raw->batches = {{0, 1}};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{4},
               holoscan::Arg("emit_interval_pulses") = int64_t{2},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kContinue,
               holoscan::Arg("aperture_window") = window};
  EXPECT_THROW(run_app(raw), std::invalid_argument);
}

TEST(ApertureManager, StampedTaperOnBlock) {
  WindowSpec window;
  window.type = WindowType::kHann;
  window.coherent_gain_normalization = false;
  auto* raw = new ManagerApp<double>();
  // Length-2 Hann is [0, 0] (symmetric endpoints). Four pulses
  // is the smallest block that is both non-rectangular and
  // non-constant, so the stamp is observable.
  raw->batches = {{0, 1, 2, 3}};
  raw->args = {holoscan::Arg("max_aperture_pulses") = int64_t{0},
               holoscan::Arg("emit_interval_pulses") = int64_t{4},
               holoscan::Arg("emit_mode") = ApertureEmitMode::kReset,
               holoscan::Arg("aperture_window") = window};
  auto app = run_app(raw);
  ASSERT_EQ(app->weight.frames.size(), 1u);
  const auto w = weights_of<double>(app->weight.frames[0]);
  const auto expected =
      holoscan::radar::generate_window(window, 4);
  ASSERT_EQ(w.size(), expected.size());
  for (size_t i = 0; i < w.size(); ++i) {
    EXPECT_DOUBLE_EQ(w[i], expected[i]);
  }
}

class EvictAtZeroApp : public holoscan::Application {
 public:
  void compose() override {
    auto manager = make_operator<ApertureManagerOp<double>>(
        "manager",
        holoscan::Arg("max_aperture_pulses") = int64_t{0});
    std::vector<int64_t> ids{0, 1};
    wire_data_source(this, manager, "range_profiles",
                     {profiles_ids<double>(ids)});
    wire_data_source(this, manager, "positions",
                     {xyz_ids<double>(ids, "positions")});
    wire_data_source(this, manager, "reference_position",
                     {xyz_ids<double>(ids, "reference_position")});
    wire_data_source(this, manager, "evict", {u32_map(0)});
    TensorMapCapture unused;
    wire_capture_sink(this, manager, "range_profiles", &unused);
    wire_capture_sink(this, manager, "positions", &unused);
    wire_capture_sink(this, manager, "reference_position", &unused);
    wire_capture_sink(this, manager, "weight", &unused);
    wire_capture_sink(this, manager, "control", &unused);
  }
};

TEST(ApertureManager, EvictWithZeroCapacityRejected) {
  auto app = holoscan::make_application<EvictAtZeroApp>();
  EXPECT_THROW(app->run(), std::invalid_argument);
}

}  // namespace
