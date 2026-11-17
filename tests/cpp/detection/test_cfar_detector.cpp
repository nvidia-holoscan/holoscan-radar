// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for CFARDetectorOp (CA variant and shared live behavior) on the
// HSDK 5 Graph harness.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/integration_graph_utils.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::CFARDetectorOp;
using holoscan::radar::CFARParams;
using holoscan::radar::CfarCellWindowOverride;
using holoscan::radar::CfarPeakMode;
using holoscan::radar::CfarVariant;
using holoscan::radar::AssemblyFeature;
using holoscan::radar::FeatureAssemblyOp;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kDoppler = 16;
constexpr int64_t kRange = 32;
constexpr DLDataType kInt64ConfigDtype{kDLInt, 64, 1};
constexpr DLDataType kFloat64ConfigDtype{kDLFloat, 64, 1};

struct CfarLiveTick {
  CfarLiveTick() = default;
  CfarLiveTick(std::shared_ptr<holoscan::Tensor> frame, std::optional<CfarCellWindowOverride> guard,
               std::optional<CfarCellWindowOverride> training,
               std::optional<std::vector<int64_t>> axes, std::optional<std::vector<double>> lut,
               std::vector<int64_t> axes_shape)
  : rd_maps(std::move(frame)),
    guard_cells(std::move(guard)),
    training_cells(std::move(training)),
    nci_axes(std::move(axes)),
    alpha_lut(std::move(lut)),
    nci_shape(std::move(axes_shape)) {}

  std::shared_ptr<holoscan::Tensor> rd_maps;
  std::optional<CfarCellWindowOverride> guard_cells;
  std::optional<CfarCellWindowOverride> training_cells;
  std::optional<std::vector<int64_t>> nci_axes;
  std::optional<std::vector<double>> alpha_lut;
  std::vector<int64_t> nci_shape;
  std::optional<double> pfa;
  std::optional<double> threshold_scale;
  std::optional<int64_t> os_rank;
  std::optional<int64_t> max_detections;
};

struct CfarOutputs {
  HostTensor mask;
  HostTensor score;
  HostTensor noise;
  HostTensor idx;
  HostTensor power;
};

std::vector<double> power_map(int64_t num_doppler, int64_t num_range, double floor,
                              const std::vector<std::pair<int64_t, int64_t>>& targets,
                              double peak) {
  std::vector<double> map(static_cast<size_t>(num_doppler * num_range), floor);
  for (const auto& [d, r] : targets) {
    map[static_cast<size_t>(d * num_range + r)] = peak;
  }
  return map;
}

std::vector<double> power_map(double floor, const std::vector<std::pair<int64_t, int64_t>>& targets,
                              double peak) {
  return power_map(kDoppler, kRange, floor, targets, peak);
}

CFARParams ca_params(double pfa) {
  CFARParams p{};
  p.cfar_variant = CfarVariant::kCA;
  p.guard_cells = {1, 1};
  p.training_cells = {2, 2};
  p.pfa = pfa;
  return p;
}

template <typename T>
holoscan::testing::TestResult run_cfar_result(
    std::shared_ptr<holoscan::Tensor> input, CFARParams params,
    const holoscan::radar::MaxTensorExtents& extents =
        holoscan::radar::testing::default_test_extents()) {
  using Op = CFARDetectorOp<T>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  fixture.push(&Op::rd_maps, *input);
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"detection_mask", "detection_score", "noise_estimate",
                                      "detection_idx", "cfar_power"};
  for (const char* output : kOutputs) {
    fixture.template capture_snapshot<holoscan::Tensor>(output, 1U, kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  for (const char* output : kOutputs) {
    (void)completion.wait_for_output(running, output, 1U);
  }
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

template <typename T>
CfarOutputs run_cfar(std::shared_ptr<holoscan::Tensor> input, CFARParams params,
                     const holoscan::radar::MaxTensorExtents& extents =
                         holoscan::radar::testing::default_test_extents()) {
  const auto result = run_cfar_result<T>(std::move(input), std::move(params), extents);
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return {
      holoscan::radar::testing::to_host(
          result.template snapshot<holoscan::Tensor>("detection_mask")),
      holoscan::radar::testing::to_host(
          result.template snapshot<holoscan::Tensor>("detection_score")),
      holoscan::radar::testing::to_host(
          result.template snapshot<holoscan::Tensor>("noise_estimate")),
      holoscan::radar::testing::to_host(
          result.template snapshot<holoscan::Tensor>("detection_idx")),
      holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("cfar_power"))};
}

// Which override ports the graph wires, and so which controls its sources
// emit. The operator declares every entry unconditionally.
struct CfarLiveWiring {
  bool pfa{false};
  bool threshold_scale{false};
  bool os_rank{false};
  bool max_detections{false};
  bool guard_cells{false};
  bool training_cells{false};
  bool nci_axes{false};
  bool alpha_lut{false};
};

holoscan::testing::TestResult run_cfar_live(std::vector<CfarLiveTick> ticks, CFARParams params,
                                            CfarLiveWiring wiring,
                                            std::function<void()> before_stop = {}) {
  using Op = CFARDetectorOp<double>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  std::vector<std::shared_ptr<holoscan::Tensor>> host_tensors;
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    holoscan::testing::TestFrame frame;
    const auto append = [&frame](holoscan::testing::TestFrame part) {
      for (auto& field : part.fields) {
        frame.fields.push_back(std::move(field));
      }
    };
    const auto& tick = ticks[index];
    if (wiring.pfa && tick.pfa.has_value()) {
      append(holoscan::testing::frame_at(capture_time, holoscan::testing::input("pfa", *tick.pfa)));
    }
    if (wiring.threshold_scale && tick.threshold_scale.has_value()) {
      append(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("threshold_scale", *tick.threshold_scale)));
    }
    if (wiring.os_rank && tick.os_rank.has_value()) {
      append(holoscan::testing::frame_at(capture_time,
                                         holoscan::testing::input("os_rank", *tick.os_rank)));
    }
    if (wiring.max_detections && tick.max_detections.has_value()) {
      append(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("max_detections", *tick.max_detections)));
    }
    if (wiring.guard_cells && tick.guard_cells.has_value()) {
      append(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("guard_cells", *tick.guard_cells)));
    }
    if (wiring.training_cells && tick.training_cells.has_value()) {
      append(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("training_cells", *tick.training_cells)));
    }
    if (wiring.nci_axes && tick.nci_axes.has_value()) {
      const auto shape = tick.nci_shape.empty()
                             ? std::vector<int64_t>{static_cast<int64_t>(tick.nci_axes->size())}
                             : tick.nci_shape;
      host_tensors.push_back(holoscan::radar::testing::to_pageable_host_tensor(
          *tick.nci_axes, shape, kInt64ConfigDtype));
      append(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("nci_axes", *host_tensors.back())));
    }
    if (wiring.alpha_lut && tick.alpha_lut.has_value()) {
      host_tensors.push_back(holoscan::radar::testing::to_pageable_host_tensor(
          *tick.alpha_lut, {static_cast<int64_t>(tick.alpha_lut->size())}, kFloat64ConfigDtype));
      append(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("alpha_lut", *host_tensors.back())));
    }
    append(holoscan::testing::frame_at(capture_time,
                                       holoscan::testing::input("rd_maps", *tick.rd_maps)));
    fixture.push_frame(frame);
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"detection_mask", "detection_score", "noise_estimate",
                                      "detection_idx", "cfar_power"};
  for (const char* output : kOutputs) {
    fixture.capture_snapshot<holoscan::Tensor>(output, ticks.size(), kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  for (const char* output : kOutputs) {
    (void)completion.wait_for_output(running, output, ticks.size());
  }
  if (before_stop)
    before_stop();
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

std::vector<HostTensor> cfar_frames(const holoscan::testing::TestResult& result, const char* port) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.snapshots<holoscan::Tensor>(port)) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

int64_t num_detections(const HostTensor& idx) {
  return idx.shape.empty() ? 0 : idx.shape[0];
}

bool idx_contains(const HostTensor& idx, int64_t doppler, int64_t range,
                  const std::vector<int64_t>& batch = {}) {
  if (idx.shape.size() < 2) {
    return false;
  }
  const int64_t n_cols = idx.shape[1];
  const int64_t n_preserved = n_cols - 2;
  if (static_cast<int64_t>(batch.size()) != n_preserved) {
    return false;
  }
  const auto* rows = idx.as<int64_t>();
  for (int64_t i = 0; i < num_detections(idx); ++i) {
    const int64_t* row = rows + i * n_cols;
    bool batch_match = true;
    for (int64_t b = 0; b < n_preserved; ++b) {
      if (row[b] != batch[static_cast<size_t>(b)]) {
        batch_match = false;
        break;
      }
    }
    if (!batch_match) {
      continue;
    }
    if (row[n_preserved] == doppler && row[n_preserved + 1] == range) {
      return true;
    }
  }
  return false;
}

int64_t linear_index(const std::vector<int64_t>& shape, const std::vector<int64_t>& coords) {
  int64_t idx = 0;
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    idx += coords[static_cast<size_t>(i)] * stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return idx;
}

std::vector<double> power_cube(const std::vector<int64_t>& shape, double floor,
                               const std::vector<std::pair<std::vector<int64_t>, double>>& cells) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  std::vector<double> data(static_cast<size_t>(count), floor);
  for (const auto& [coords, value] : cells) {
    data[static_cast<size_t>(linear_index(shape, coords))] = value;
  }
  return data;
}

template <typename T>
void expect_mask_matches_score_interior(const HostTensor& mask, const HostTensor& score,
                                        int64_t num_doppler, int64_t num_range) {
  ASSERT_EQ(mask.shape, score.shape);
  const int64_t margin = 3;
  const auto* mask_bytes = mask.as<uint8_t>();
  const auto* scores = score.as<T>();
  for (int64_t d = margin; d < num_doppler - margin; ++d) {
    for (int64_t r = margin; r < num_range - margin; ++r) {
      const int64_t i = d * num_range + r;
      const bool detected = scores[i] > T{1};
      EXPECT_EQ(mask_bytes[i] != 0, detected)
          << "cell (" << d << ", " << r << ") score=" << scores[i];
    }
  }
}

double score_at(const HostTensor& score, int64_t d, int64_t r) {
  return score.as<double>()[d * kRange + r];
}

}  // namespace

TEST(CFARDetectorOpTest, FeatureAssemblyWaitsForCfarScoreAndIndex) {
  namespace harness = holoscan::radar::testing::integration;
  harness::FrameBank::reset();
  harness::FrameBank::input_frames = {to_device_tensor<double>(
      power_map(1.0, {{8, 16}}, 100.0), {kDoppler, kRange}, real_dtype<double>())};

  holoscan::Graph graph{"cfar-feature-assembly"};
  const auto source = graph.op<harness::TensorSourceOp>("source");
  const auto extents = holoscan::radar::testing::default_test_extents();
  auto cfar = graph.op<CFARDetectorOp<double>>("cfar", extents);
  cfar.set_params(ca_params(1e-3));
  auto features = graph.op<FeatureAssemblyOp<double>>(
      "features", extents, std::vector<AssemblyFeature>{AssemblyFeature::kScore});
  const auto sink = graph.op<harness::TensorCaptureOp>("sink");
  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->out, cfar->rd_maps, queue);
  graph.add_flow(cfar->detection_idx, features->detection_idx, queue);
  graph.add_flow(cfar->detection_score, features->detection_score, queue);
  graph.add_flow(features->points, sink->in, queue);
  graph.partition("integration").add(source).add(cfar).add(features).add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  const auto bind = [&options](const char* op, const char* port) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  };
  bind("source", "out");
  bind("cfar", "detection_mask");
  bind("cfar", "detection_score");
  bind("cfar", "noise_estimate");
  bind("cfar", "detection_idx");
  bind("cfar", "cfar_power");
  bind("features", "points");

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + harness::kHarnessDeadline;
  {
    std::unique_lock lock(harness::FrameBank::mutex);
    while (harness::FrameBank::captured.empty() && session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      harness::FrameBank::ready.wait_for(lock, harness::kHarnessPoll);
    }
  }
  session.request_stop();
  session.wait();
  ASSERT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(harness::FrameBank::mutex);
  ASSERT_EQ(harness::FrameBank::captured.size(), 1U);
  const HostTensor& points = harness::FrameBank::captured.front();
  ASSERT_EQ(points.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_GT(points.as<double>()[0], 1.0);
}

TEST(CFARDetectorOpTest, DetectsSingleTargetRealPath) {
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);

  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[dt * kRange + rt], 1);
  const auto* score = out.score.as<double>();
  EXPECT_GT(score[dt * kRange + rt], 1.0);
}

TEST(CFARDetectorOpTest, NoiseEstimateTracksFloor) {
  const double floor = 3.5;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
}

TEST(CFARDetectorOpTest, NoFalseAlarmsOnFlatMap) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-6));

  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpTest, ComplexInputFormsPower) {
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(static_cast<size_t>(kDoppler * kRange),
                                         std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] = std::complex<double>(10.0, 0.0);
  auto input =
      to_device_tensor<std::complex<double>>(cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);
  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[0], 1.0, 1e-9);
}

TEST(CFARDetectorOpTest, CfarPowerIsSquareLawPowerWithoutNci) {
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(static_cast<size_t>(kDoppler * kRange),
                                         std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] = std::complex<double>(3.0, 4.0);
  auto input =
      to_device_tensor<std::complex<double>>(cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(out.power.shape, (std::vector<int64_t>{kDoppler, kRange}));
  ASSERT_EQ(out.power.shape, out.score.shape);
  const auto* p = out.power.as<double>();
  for (int64_t i = 0; i < kDoppler * kRange; ++i) {
    EXPECT_NEAR(p[i], std::norm(cube[static_cast<size_t>(i)]), 1e-9) << i;
  }
  EXPECT_NEAR(p[dt * kRange + rt], 25.0, 1e-9);
}

TEST(CFARDetectorOpTest, CfarPowerIsNciSumOverFold) {
  constexpr int64_t kChannels = 4;
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(static_cast<size_t>(kChannels * kDoppler * kRange));
  for (int64_t k = 0; k < kChannels; ++k) {
    for (int64_t i = 0; i < kDoppler * kRange; ++i) {
      cube[static_cast<size_t>(k * kDoppler * kRange + i)] =
          std::complex<double>(static_cast<double>(k + 1), 0.0);
    }
    cube[static_cast<size_t>(k * kDoppler * kRange + dt * kRange + rt)] =
        std::complex<double>(10.0, 0.0);
  }
  auto input = to_device_tensor<std::complex<double>>(cube, {kChannels, kDoppler, kRange},
                                                      complex_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  params.nci_fold = kChannels;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(out.power.shape, (std::vector<int64_t>{kDoppler, kRange}));
  ASSERT_EQ(out.power.shape, out.score.shape);
  const auto* p = out.power.as<double>();
  EXPECT_NEAR(p[0], 30.0, 1e-9);
  EXPECT_NEAR(p[dt * kRange + rt], 400.0, 1e-9);

  ASSERT_EQ(num_detections(out.idx), 1);
  const int64_t peak = static_cast<int64_t>(std::max_element(p, p + kDoppler * kRange) - p);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(peak, rows[0] * kRange + rows[1]);
}

TEST(CFARDetectorOpTest, MaxDetectionsCap) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.max_detections = 1;
  const auto out = run_cfar<double>(input, params);

  EXPECT_EQ(num_detections(out.idx), 1);
}

TEST(CFARDetectorOpTest, FloatDoubleAgree) {
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);

  std::vector<float> mapf(map.begin(), map.end());
  auto in_f = to_device_tensor<float>(mapf, {kDoppler, kRange}, real_dtype<float>());
  auto in_d = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out_f = run_cfar<float>(in_f, ca_params(1e-3));
  const auto out_d = run_cfar<double>(in_d, ca_params(1e-3));

  ASSERT_EQ(num_detections(out_f.idx), num_detections(out_d.idx));
  const auto* rf = out_f.idx.as<int64_t>();
  const auto* rd = out_d.idx.as<int64_t>();
  EXPECT_EQ(rf[0], rd[0]);
  EXPECT_EQ(rf[1], rd[1]);
}

TEST(CFARDetectorOpTest, DetectsOnLargeMap) {
  constexpr int64_t kLargeDoppler = 256;
  constexpr int64_t kLargeRange = 512;
  const int64_t dt = 128;
  const int64_t rt = 256;
  const auto map = power_map(kLargeDoppler, kLargeRange, 1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kLargeDoppler, kLargeRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
  expect_mask_matches_score_interior<double>(out.mask, out.score, kLargeDoppler, kLargeRange);
}

TEST(CFARDetectorOpTest, MaskMatchesScoreInterior) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler, kRange);
  const auto* scores = out.score.as<double>();
  EXPECT_GT(scores[8 * kRange + 16], 1.0);
  const int64_t far = 4 * kRange + 4;
  EXPECT_LT(scores[far], 1.0);
}

TEST(CFARDetectorOpTest, RealAndComplexPathsAgree) {
  const int64_t dt = 10;
  const int64_t rt = 22;
  const double floor = 2.0;
  const double peak = 80.0;

  const auto real_map = power_map(floor, {{dt, rt}}, peak);
  auto real_input = to_device_tensor<double>(real_map, {kDoppler, kRange}, real_dtype<double>());

  std::vector<std::complex<double>> complex_map(static_cast<size_t>(kDoppler * kRange),
                                                std::complex<double>(std::sqrt(floor), 0.0));
  complex_map[static_cast<size_t>(dt * kRange + rt)] = std::complex<double>(std::sqrt(peak), 0.0);
  auto complex_input = to_device_tensor<std::complex<double>>(complex_map, {kDoppler, kRange},
                                                              complex_dtype<double>());

  const auto out_real = run_cfar<double>(real_input, ca_params(1e-3));
  const auto out_complex = run_cfar<double>(complex_input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out_real.idx), num_detections(out_complex.idx));
  EXPECT_TRUE(idx_contains(out_real.idx, dt, rt));
  EXPECT_TRUE(idx_contains(out_complex.idx, dt, rt));
}

TEST(CFARDetectorOpTest, BorderCellsUseInBoundsTrainingOnly) {
  const double floor = 4.0;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-6));

  const auto* noise = out.noise.as<double>();
  const auto* scores = out.score.as<double>();
  EXPECT_NEAR(noise[0], floor, 1e-6);
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
  EXPECT_LE(scores[0], 1.0);
  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpTest, DetectsTwoTargets) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, 4, 8));
  EXPECT_TRUE(idx_contains(out.idx, 12, 24));
}

TEST(CFARDetectorOpTest, DetectionIdxRasterOrder) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  ASSERT_EQ(out.idx.shape.size(), 2u);
  ASSERT_EQ(out.idx.shape[1], 2);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], 4);
  EXPECT_EQ(rows[1], 8);
  EXPECT_EQ(rows[2], 12);
  EXPECT_EQ(rows[3], 24);
}

TEST(CFARDetectorOpTest, SingleLeadingAxisUsesMaxChannels) {
  constexpr int64_t kTx = 2;
  constexpr int64_t kRx = 4;
  constexpr int64_t kVirtual = kTx * kRx;
  const std::vector<int64_t> shape{kVirtual, kDoppler, kRange};
  const auto map = power_cube(shape, 1.0, {});
  auto input = to_device_tensor<double>(map, shape, real_dtype<double>());

  holoscan::radar::MaxTensorExtents extents;
  extents.max_doppler_bins = kDoppler;
  extents.max_range_bins = kRange;
  extents.max_tx = kTx;
  extents.max_rx = kRx;
  extents.max_channels = kVirtual;
  extents.max_detections = kVirtual * kDoppler * kRange;

  const auto result = run_cfar_result<double>(input, ca_params(1e-3), extents);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "detection_mask").size(), 1U);
  EXPECT_EQ(cfar_frames(result, "detection_mask").front().shape, shape);
}

TEST(CFARDetectorOpTest, RejectsOpaqueLeadingBatch) {
  constexpr int64_t kOpaque = 2;
  const std::vector<int64_t> shape{kOpaque, kOpaque, kOpaque, kDoppler, kRange};
  const auto map = power_cube(shape, 1.0, {});
  auto input = to_device_tensor<double>(map, shape, real_dtype<double>());
  holoscan::radar::MaxTensorExtents extents;
  extents.max_doppler_bins = kDoppler;
  extents.max_range_bins = kRange;
  extents.max_tx = kOpaque;
  extents.max_rx = kOpaque;
  extents.max_detections = kOpaque * kOpaque * kOpaque * kDoppler * kRange;

  const auto result = run_cfar_result<double>(input, ca_params(1e-3), extents);
  EXPECT_FALSE(result.ok());
}

TEST(CFARDetectorOpTest, OptionalOutputsMayBeLeftUnwired) {
  constexpr std::size_t kFrames = 8;
  const auto make_inputs = [] {
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs;
    inputs.reserve(kFrames);
    for (std::size_t i = 0; i < kFrames; ++i) {
      inputs.push_back(to_device_tensor<double>(power_map(1.0, {{8, 16}}, 100.0),
                                                {kDoppler, kRange}, real_dtype<double>()));
    }
    return inputs;
  };
  const auto run_selected = [&](const char* captured_port) {
    using Op = CFARDetectorOp<double>;
    auto fixture = holoscan::testing::create_operator_test<Op>(
        holoscan::radar::testing::default_test_extents());
    fixture.set_params(ca_params(1e-3));
    const auto inputs = make_inputs();
    for (const auto& input : inputs) {
      fixture.push(&Op::rd_maps, *input);
    }
    constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
    fixture.capture_snapshot<holoscan::Tensor>(captured_port, kFrames, kMaximumBytes);
    for (const char* output :
         {"detection_mask", "detection_score", "noise_estimate", "detection_idx", "cfar_power"}) {
      fixture.output_device(output, holoscan::DeviceId{0});
    }
    fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
    holoscan::radar::testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    (void)completion.wait_for_output(running, captured_port, kFrames);
    running.request_stop();
    auto fixture_result = running.finish();
    completion.check(fixture_result);
    return fixture_result;
  };

  const auto idx_only = run_selected("detection_idx");
  ASSERT_TRUE(idx_only.ok()) << idx_only.describe_failure();
  EXPECT_EQ(idx_only.snapshots<holoscan::Tensor>("detection_idx").size(), kFrames);

  const auto power_only = run_selected("cfar_power");
  ASSERT_TRUE(power_only.ok()) << power_only.describe_failure();
  EXPECT_EQ(power_only.snapshots<holoscan::Tensor>("cfar_power").size(), kFrames);
}

TEST(CFARDetectorOpTest, LiveThresholdScaleOverrideMidStream) {
  auto input = to_device_tensor<double>(power_map(1.0, {{8, 16}}, 100.0), {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  CfarLiveTick missing{};
  missing.rd_maps = input;
  CfarLiveTick override = missing;
  override.threshold_scale = 1e6;
  const auto result =
      run_cfar_live({missing, override, missing}, params, {.threshold_scale = true});

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "detection_idx").size(), 3U);
  EXPECT_GE(num_detections(cfar_frames(result, "detection_idx")[0]), 1);
  EXPECT_EQ(num_detections(cfar_frames(result, "detection_idx")[1]), 0);
  EXPECT_EQ(num_detections(cfar_frames(result, "detection_idx")[2]),
            num_detections(cfar_frames(result, "detection_idx")[0]));
}

TEST(CFARDetectorOpTest, LivePfaOverrideMidStream) {
  constexpr int64_t kPeak = 8 * kRange + 16;
  auto input = to_device_tensor<double>(power_map(1.0, {{8, 16}}, 26.0), {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  CfarLiveTick missing{};
  missing.rd_maps = input;
  CfarLiveTick override = missing;
  override.pfa = 1e-9;
  const auto result = run_cfar_live({missing, override, missing}, params, {.pfa = true});

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "detection_score").size(), 3U);
  const double params_score = cfar_frames(result, "detection_score")[0].as<double>()[kPeak];
  const double override_score = cfar_frames(result, "detection_score")[1].as<double>()[kPeak];
  EXPECT_LT(override_score, params_score);
  EXPECT_DOUBLE_EQ(cfar_frames(result, "detection_score")[2].as<double>()[kPeak], params_score);
}

TEST(CFARDetectorOpTest, NonBlockingPfaOverrideServesStaleLut) {
  constexpr int64_t kPeak = 8 * kRange + 16;
  auto input = to_device_tensor<double>(power_map(1.0, {{8, 16}}, 26.0), {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = false;
  CfarLiveTick missing{};
  missing.rd_maps = input;
  CfarLiveTick override = missing;
  override.pfa = 7.3e-8;
  const auto result = run_cfar_live({missing, override, missing}, params, {.pfa = true});

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "detection_score").size(), 3U);
  const double initial_score = cfar_frames(result, "detection_score")[0].as<double>()[kPeak];
  EXPECT_DOUBLE_EQ(cfar_frames(result, "detection_score")[1].as<double>()[kPeak], initial_score);
  EXPECT_DOUBLE_EQ(cfar_frames(result, "detection_score")[2].as<double>()[kPeak], initial_score);
}

TEST(CFARDetectorOpTest, SupersededCalibrationSubscriptionsAreReleasedBeforeStop) {
  namespace detail = holoscan::radar::detail;
  using holoscan::radar::CfarAlphaLutKey;
  auto input = to_device_tensor<double>(power_map(1.0, {{8, 16}}, 26.0), {kDoppler, kRange},
                                        real_dtype<double>());
  auto params = ca_params(0.1);
  params.cfar_variant = CfarVariant::kGO;
  CfarAlphaLutKey base{CfarVariant::kGO, params.pfa, 1, params.os_rank, 1, 1, 2, 2,
                       kDoppler,         kRange};
  const auto lut = detail::get_cached_alpha_lut(base);
  for (bool shared : {false, true}) {
    auto first_key = base;
    first_key.pfa = shared ? 1.23e-9 : 1.24e-9;
    auto second_key = base;
    second_key.pfa = shared ? 2.23e-9 : 2.24e-9;
    int other_detector = 0;
    if (shared)
      detail::ensure_alpha_lut_build(first_key, &other_detector);
    CfarLiveTick initial{};
    initial.rd_maps = input;
    auto first = initial;
    first.pfa = first_key.pfa;
    auto second = initial;
    second.pfa = second_key.pfa;
    // Check while the detector is still running: stop() would mask leaked subscriptions.
    const auto result =
        run_cfar_live({initial, first, second, initial}, params, {.pfa = true}, [&] {
          EXPECT_EQ(detail::poll_cached_alpha_lut(first_key).building, shared);
          EXPECT_FALSE(detail::poll_cached_alpha_lut(second_key).building);
          detail::release_alpha_lut_builds(&other_detector);
          EXPECT_FALSE(detail::poll_cached_alpha_lut(first_key).building);
        });
    ASSERT_TRUE(result.ok()) << result.describe_failure();
  }

  auto pending_key = base;
  pending_key.pfa = 3.24e-9;
  CfarLiveTick initial{};
  initial.rd_maps = input;
  auto pending = initial;
  pending.pfa = pending_key.pfa;
  auto external = initial;
  external.alpha_lut = lut;
  const auto result =
      run_cfar_live({initial, pending, external}, params, {.pfa = true, .alpha_lut = true}, [&] {
        EXPECT_FALSE(detail::poll_cached_alpha_lut(pending_key).building);
      });
  ASSERT_TRUE(result.ok()) << result.describe_failure();
}

TEST(CFARDetectorOpTest, ExternalLutReleasesSupersededStartupCalibration) {
  namespace detail = holoscan::radar::detail;
  auto input =
      to_device_tensor<double>(power_map(1.0, {}, 1.0), {kDoppler, kRange}, real_dtype<double>());
  auto params = ca_params(4.24e-9);
  params.cfar_variant = CfarVariant::kOS;
  params.os_rank = 2;
  params.preintegrated_looks = 2;
  holoscan::radar::CfarAlphaLutKey warm{CfarVariant::kOS, params.pfa, 2, 2, 1, 1, 2, 2, 7, 7};
  CfarLiveTick external{};
  external.rd_maps = input;
  external.alpha_lut = std::vector<double>(49, 1.0);
  const auto result = run_cfar_live({external}, params, {.alpha_lut = true}, [&] {
    EXPECT_FALSE(detail::poll_cached_alpha_lut(warm).building);
  });
  ASSERT_TRUE(result.ok()) << result.describe_failure();
}

TEST(CFARDetectorOpTest, LiveMaxDetectionsOverrideMidStream) {
  auto input = to_device_tensor<double>(power_map(1.0, {{4, 8}, {12, 24}}, 100.0),
                                        {kDoppler, kRange}, real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  CfarLiveTick missing{};
  missing.rd_maps = input;
  CfarLiveTick override = missing;
  override.max_detections = 1;
  const auto result = run_cfar_live({missing, override, missing}, params, {.max_detections = true});

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "detection_idx").size(), 3U);
  EXPECT_EQ(num_detections(cfar_frames(result, "detection_idx")[0]), 2);
  EXPECT_EQ(num_detections(cfar_frames(result, "detection_idx")[1]), 1);
  EXPECT_EQ(num_detections(cfar_frames(result, "detection_idx")[2]), 2);
}

TEST(CFARDetectorOpTest, LiveOsRankOverrideMidStream) {
  constexpr int64_t kPeak = 8 * kRange + 16;
  std::vector<double> map(static_cast<size_t>(kDoppler * kRange));
  for (size_t i = 0; i < map.size(); ++i) {
    map[i] = 1.0 + static_cast<double>(i % 17) * 0.03;
  }
  map[kPeak] = 50.0;
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.cfar_variant = CfarVariant::kOS;
  params.os_rank = 5;
  params.blocking_alpha_lut_rebuild = true;
  CfarLiveTick missing{};
  missing.rd_maps = input;
  CfarLiveTick override = missing;
  override.os_rank = 30;
  const auto result = run_cfar_live({missing, override, missing}, params, {.os_rank = true});

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "noise_estimate").size(), 3U);
  const double params_noise = cfar_frames(result, "noise_estimate")[0].as<double>()[kPeak];
  const double override_noise = cfar_frames(result, "noise_estimate")[1].as<double>()[kPeak];
  EXPECT_LT(params_noise, override_noise);
  EXPECT_DOUBLE_EQ(cfar_frames(result, "noise_estimate")[2].as<double>()[kPeak], params_noise);
}

TEST(CFARDetectorOpTest, GuardCellsConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  map[static_cast<size_t>(dt * kRange + rt + 2)] = 10000.0;
  auto input = to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams narrow = ca_params(1e-3);
  narrow.blocking_alpha_lut_rebuild = true;
  CFARParams wide = narrow;
  wide.guard_cells = {1, 2};
  const auto n = run_cfar<double>(input, narrow);
  const auto w = run_cfar<double>(input, wide);
  EXPECT_LT(score_at(n.score, dt, rt), score_at(w.score, dt, rt));
}

TEST(CFARDetectorOpTest, TrainingCellsConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(6 * kRange + rt)] = 10000.0;
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  auto input = to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams wide = ca_params(1e-3);
  wide.blocking_alpha_lut_rebuild = true;
  wide.training_cells = {3, 3};
  CFARParams narrow = ca_params(1e-3);
  narrow.blocking_alpha_lut_rebuild = true;
  narrow.training_cells = {1, 1};
  const auto w = run_cfar<double>(input, wide);
  const auto n = run_cfar<double>(input, narrow);
  EXPECT_GT(score_at(w.score, dt, rt), score_at(n.score, dt, rt));
}

TEST(CFARDetectorOpTest, LiveGuardCellsOverrideAppliesForOneTick) {
  constexpr int64_t kChannels = 2;
  const auto cube =
      power_cube({kChannels, kDoppler, kRange}, 1.0, {{{0, 8, 16}, 100.0}, {{1, 8, 16}, 100.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());
  const auto lut = holoscan::radar::detail::build_ca_alpha_lut(
      1e-3, kChannels, static_cast<int>(kDoppler), static_cast<int>(kRange), 1, 2, 2, 3);

  CFARParams params = ca_params(1e-3);
  params.training_cells = {2, 3};
  params.nci_axes = {0};
  params.nci_fold = kChannels;
  params.alpha_lut = lut;
  const CfarLiveTick first{
      input, CfarCellWindowOverride{1, 2}, std::nullopt, std::nullopt, std::nullopt, {}};
  const CfarLiveTick missing{input, std::nullopt, std::nullopt, std::nullopt, std::nullopt, {}};
  const auto result = run_cfar_live({missing, first, missing}, params, {.guard_cells = true});

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "detection_score").size(), 3U);
  EXPECT_EQ(cfar_frames(result, "detection_score")[0].shape,
            (std::vector<int64_t>{kDoppler, kRange}));
  EXPECT_EQ(cfar_frames(result, "detection_score")[1].shape,
            cfar_frames(result, "detection_score")[0].shape);
  EXPECT_EQ(cfar_frames(result, "detection_score")[2].shape,
            cfar_frames(result, "detection_score")[0].shape);
  constexpr int64_t kPeak = 8 * kRange + 16;
  const double params_score = cfar_frames(result, "detection_score")[0].as<double>()[kPeak];
  const double override_score = cfar_frames(result, "detection_score")[1].as<double>()[kPeak];
  const double reverted_score = cfar_frames(result, "detection_score")[2].as<double>()[kPeak];
  // The wider guard window changes the noise estimate, so the middle tick is
  // distinguishable; the silent tick after it is back on the Params window.
  EXPECT_NE(override_score, params_score);
  EXPECT_DOUBLE_EQ(reverted_score, params_score);
}

TEST(CFARDetectorOpTest, RejectsMalformedLiveCellWindow) {
  auto input = to_device_tensor<double>(power_map(1.0, {{8, 16}}, 100.0), {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  const CfarLiveTick malformed{
      input, CfarCellWindowOverride{-1, 1}, std::nullopt, std::nullopt, std::nullopt, {}};
  const auto result = run_cfar_live({malformed}, params, {.guard_cells = true});
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(cfar_frames(result, "detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

// A rank-2 nci_axes never reaches compute: the source's rank-1 pool refuses
// the loan, so the malformed shape surfaces as a run diagnostic rather than
// through the operator's own vector guard. Declaring the source rank-2 to
// smuggle it past is no longer possible either — HSDK 5 rejects a producer
// whose rank capacity exceeds the consumer's exact rank when the graph
// compiles (GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT).
TEST(CFARDetectorOpTest, RejectsMalformedLiveNciAxesTensor) {
  constexpr int64_t kChannels = 2;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0, {});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  const CfarLiveTick malformed{input,        std::nullopt, std::nullopt, std::vector<int64_t>{0},
                               std::nullopt, {1, 1}};
  EXPECT_THROW((void)run_cfar_live({malformed}, params, {.nci_axes = true}),
               holoscan::RuntimeError);
}

TEST(CFARDetectorOpTest, LiveNciAxesOverrideAppliesForOneTick) {
  constexpr int64_t kChannels = 2;
  const auto cube =
      power_cube({kChannels, kDoppler, kRange}, 1.0, {{{0, 8, 16}, 50.0}, {{1, 8, 16}, 50.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  CfarLiveTick missing{};
  missing.rd_maps = input;
  CfarLiveTick override = missing;
  override.nci_axes = std::vector<int64_t>{0};
  const auto result = run_cfar_live({missing, override, missing}, params, {.nci_axes = true});

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "detection_mask").size(), 3U);
  EXPECT_EQ(cfar_frames(result, "detection_mask")[0].shape,
            (std::vector<int64_t>{kChannels, kDoppler, kRange}));
  EXPECT_EQ(cfar_frames(result, "detection_mask")[1].shape,
            (std::vector<int64_t>{kDoppler, kRange}));
  EXPECT_EQ(cfar_frames(result, "detection_mask")[2].shape,
            cfar_frames(result, "detection_mask")[0].shape);
}

TEST(CFARDetectorOpTest, LiveAlphaLutOverrideAndClearing) {
  auto input = to_device_tensor<double>(power_map(1.0, {{8, 16}}, 100.0), {kDoppler, kRange},
                                        real_dtype<double>());
  auto huge_lut = holoscan::radar::detail::build_ca_alpha_lut(1e-3, 1, static_cast<int>(kDoppler),
                                                              static_cast<int>(kRange), 1, 1, 2, 2);
  for (size_t i = 1; i < huge_lut.size(); ++i) {
    huge_lut[i] *= 1e6;
  }
  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  const CfarLiveTick external{input, std::nullopt, std::nullopt, std::nullopt, huge_lut, {}};
  const CfarLiveTick clear{input,        std::nullopt,          std::nullopt,
                           std::nullopt, std::vector<double>{}, {}};
  const CfarLiveTick missing{input, std::nullopt, std::nullopt, std::nullopt, std::nullopt, {}};
  const auto result = run_cfar_live({external, clear, missing}, params, {.alpha_lut = true});

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cfar_frames(result, "detection_idx").size(), 3U);
  // The huge external LUT suppresses everything; the explicit empty vector
  // clears back to the internally built LUT; the silent third tick reverts to
  // the Params alpha_lut, which is also empty, so it agrees with tick 1.
  EXPECT_EQ(num_detections(cfar_frames(result, "detection_idx")[0]), 0);
  EXPECT_GE(num_detections(cfar_frames(result, "detection_idx")[1]), 1);
  EXPECT_EQ(num_detections(cfar_frames(result, "detection_idx")[2]),
            num_detections(cfar_frames(result, "detection_idx")[1]));
}

TEST(CFARDetectorOpTest, RejectsMalformedLiveAlphaLut) {
  auto input = to_device_tensor<double>(power_map(1.0, {{8, 16}}, 100.0), {kDoppler, kRange},
                                        real_dtype<double>());
  auto lut = holoscan::radar::detail::build_ca_alpha_lut(1e-3, 1, static_cast<int>(kDoppler),
                                                         static_cast<int>(kRange), 1, 1, 2, 2);
  lut[1] = std::numeric_limits<double>::quiet_NaN();
  CFARParams params = ca_params(1e-3);
  const CfarLiveTick malformed{input, std::nullopt, std::nullopt, std::nullopt, lut, {}};
  const auto result = run_cfar_live({malformed}, params, {.alpha_lut = true});
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(cfar_frames(result, "detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorOpTest, NciIntegratesChannelsForDetection) {
  constexpr int64_t kChannels = 4;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube =
      power_cube({kChannels, kDoppler, kRange}, 1.0,
                 {{{0, dt, rt}, 5.0}, {{1, dt, rt}, 5.0}, {{2, dt, rt}, 5.0}, {{3, dt, rt}, 5.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
}

TEST(CFARDetectorOpTest, NciSumsAllChannelPower) {
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0, {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(out.power.shape, (std::vector<int64_t>{kDoppler, kRange}));
  const auto* p = out.power.as<double>();
  EXPECT_NEAR(p[8 * kRange + 16], 43.0, 1e-9);
}

TEST(CFARDetectorOpTest, NciPreservesBatchAxis) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kChannels = 3;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube = power_cube({kBatch, kChannels, kDoppler, kRange}, 1.0,
                               {{{0, 0, dt, rt}, 20.0},
                                {{0, 1, dt, rt}, 20.0},
                                {{0, 2, dt, rt}, 20.0},
                                {{1, 0, dt, rt}, 20.0},
                                {{1, 1, dt, rt}, 20.0},
                                {{1, 2, dt, rt}, 20.0}});
  auto input =
      to_device_tensor<double>(cube, {kBatch, kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {1};
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpTest, BatchIdxIncludesPreservedAxes) {
  constexpr int64_t kBatch = 2;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube =
      power_cube({kBatch, kDoppler, kRange}, 1.0, {{{0, dt, rt}, 100.0}, {{1, dt, rt}, 100.0}});
  auto input = to_device_tensor<double>(cube, {kBatch, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpTest, NciMaskMatchesScoreOnIntegratedMap) {
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0, {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  const auto out = run_cfar<double>(input, params);

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler, kRange);
}

TEST(CFARDetectorOpTest, RejectsMissingGuardCells) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams params{};
  params.cfar_variant = CfarVariant::kCA;
  params.training_cells = {2, 2};
  params.pfa = 1e-3;
  const auto result = run_cfar_result<double>(input, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(cfar_frames(result, "detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorOpTest, RejectsInvalidNciAxis) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  const auto result = run_cfar_result<double>(input, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(cfar_frames(result, "detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorOpTest, CaAlphaK1MatchesContractClosedForm) {
  constexpr int64_t kN = 40;
  constexpr double kPfa = 1e-2;
  const double alpha = holoscan::radar::detail::ca_cfar_alpha(kPfa, kN, 1);
  const double closed = kN * (std::pow(kPfa, -1.0 / kN) - 1.0);
  EXPECT_NEAR(alpha, closed, 1e-6 * closed);
  EXPECT_NEAR(holoscan::radar::detail::ca_cfar_pfa_from_alpha(alpha, kN, 1), kPfa, 1e-9);
}

TEST(CFARDetectorOpTest, PeakModeNoneMatchesRawMask) {
  const auto map = power_map(1.0, {{8, 15}, {8, 16}, {8, 17}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto raw = run_cfar<double>(input, ca_params(1e-2));
  CFARParams off = ca_params(1e-2);
  off.peak_mode = CfarPeakMode::kNone;
  const auto out = run_cfar<double>(input, off);

  EXPECT_EQ(num_detections(raw.idx), 3);
  EXPECT_EQ(num_detections(out.idx), num_detections(raw.idx));
  for (const auto r : {15, 16, 17}) {
    EXPECT_TRUE(idx_contains(out.idx, 8, r));
  }
}

TEST(CFARDetectorOpTest, PeakModeRangeDopplerKeepsSingleMax) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(8 * kRange + 17)] = 50.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-2);
  params.peak_mode = CfarPeakMode::kRangeDoppler;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, 8, 16));
  EXPECT_FALSE(idx_contains(out.idx, 8, 17));
  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[8 * kRange + 16], 1);
  EXPECT_EQ(mask[8 * kRange + 17], 0);
}

TEST(CFARDetectorOpTest, PeakModeRangeDopplerUsesRasterTiebreak) {
  const auto map = power_map(1.0, {{8, 15}, {8, 16}, {8, 17}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-2);
  params.peak_mode = CfarPeakMode::kRangeDoppler;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, 8, 15));
}

TEST(CFARDetectorOpTest, PeakModeRangeDopplerUsesConfiguredRadius) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(9 * kRange + 17)] = 90.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange}, real_dtype<double>());

  CFARParams doppler_only = ca_params(1e-2);
  doppler_only.peak_mode = CfarPeakMode::kRangeDoppler;
  doppler_only.peak_radius = {1, 0};
  CFARParams square = doppler_only;
  square.peak_radius = {1, 1};
  CFARParams all = square;
  all.peak_mode = CfarPeakMode::kAll;

  const auto out_doppler_only = run_cfar<double>(input, doppler_only);
  const auto out_square = run_cfar<double>(input, square);
  const auto out_all = run_cfar<double>(input, all);

  EXPECT_EQ(num_detections(out_doppler_only.idx), 2);
  EXPECT_EQ(num_detections(out_square.idx), 1);
  EXPECT_EQ(num_detections(out_all.idx), 1);
  EXPECT_TRUE(idx_contains(out_square.idx, 8, 16));
  EXPECT_TRUE(idx_contains(out_all.idx, 8, 16));
}

TEST(CFARDetectorOpTest, PeakRadiusSuppressesWiderRangeNeighbors) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(8 * kRange + 18)] = 90.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange}, real_dtype<double>());
  CFARParams radius_one = ca_params(1e-2);
  radius_one.peak_mode = CfarPeakMode::kRangeDoppler;
  radius_one.peak_radius = {1};
  CFARParams radius_two = radius_one;
  radius_two.peak_radius = {1, 2};

  EXPECT_EQ(num_detections(run_cfar<double>(input, radius_one).idx), 2);
  const auto wider = run_cfar<double>(input, radius_two);
  ASSERT_EQ(num_detections(wider.idx), 1);
  EXPECT_TRUE(idx_contains(wider.idx, 8, 16));
}

TEST(CFARDetectorOpTest, PeakRadiusRejectsInvalidEncodings) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  for (const std::vector<int64_t>& radius :
       {std::vector<int64_t>{-1}, std::vector<int64_t>{0, 0}, std::vector<int64_t>{1, 1, 1}}) {
    CFARParams params = ca_params(1e-2);
    params.peak_mode = CfarPeakMode::kRangeDoppler;
    params.peak_radius = radius;
    const auto result = run_cfar_result<double>(input, params);
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(cfar_frames(result, "detection_idx").empty());
    EXPECT_FALSE(result.diagnostics().empty());
  }
}

TEST(CFARDetectorOpTest, PeakRadiusMustMatchPostNciOutputRank) {
  constexpr int64_t kChannels = 2;
  std::vector<double> data(static_cast<size_t>(kChannels * kDoppler * kRange), 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(kDoppler * kRange + 8 * kRange + 16)] = 100.0;
  auto input = to_device_tensor<double>(data, {kChannels, kDoppler, kRange}, real_dtype<double>());
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_rx = kChannels;
  CFARParams params = ca_params(1e-2);
  params.nci_axes = {0};
  params.peak_mode = CfarPeakMode::kAll;
  params.peak_radius = {1, 1, 1};

  const auto result = run_cfar_result<double>(input, params, extents);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(cfar_frames(result, "detection_idx").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorOpTest, PeakModeAllSuppressesAcrossEveryRank4Axis) {
  constexpr int64_t kAz = 2;
  constexpr int64_t kEl = 2;
  constexpr int64_t kD = 16;
  constexpr int64_t kR = 32;
  constexpr int64_t kPeakD = 8;
  constexpr int64_t kPeakR = 16;
  const std::vector<int64_t> shape{kAz, kEl, kD, kR};
  std::vector<double> data(static_cast<size_t>(kAz * kEl * kD * kR), 1.0);
  const auto offset = [](int64_t az, int64_t el) {
    return static_cast<size_t>(((az * kEl + el) * kD + kPeakD) * kR + kPeakR);
  };
  data[offset(0, 0)] = 100.0;
  data[offset(0, 1)] = 110.0;
  data[offset(1, 0)] = 120.0;
  data[offset(1, 1)] = 130.0;
  auto input = to_device_tensor<double>(data, shape, real_dtype<double>());
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_tx = kAz;
  extents.max_rx = kEl;
  extents.max_detections = kAz * kEl * kD * kR;

  CFARParams none = ca_params(1e-2);
  CFARParams rd = none;
  rd.peak_mode = CfarPeakMode::kRangeDoppler;
  CFARParams all = none;
  all.peak_mode = CfarPeakMode::kAll;
  const auto out_none = run_cfar<double>(input, none, extents);
  const auto out_rd = run_cfar<double>(input, rd, extents);
  const auto out_all = run_cfar<double>(input, all, extents);

  EXPECT_EQ(num_detections(out_none.idx), 4);
  EXPECT_EQ(num_detections(out_rd.idx), 4);
  ASSERT_EQ(num_detections(out_all.idx), 1);
  EXPECT_EQ(std::count(out_all.mask.as<uint8_t>(), out_all.mask.as<uint8_t>() + out_all.mask.size(),
                       uint8_t{1}),
            1);
  ASSERT_EQ(out_all.idx.shape, (std::vector<int64_t>{1, 4}));
  const int64_t* index = out_all.idx.as<int64_t>();
  EXPECT_EQ(std::vector<int64_t>(index, index + 4), (std::vector<int64_t>{1, 1, kPeakD, kPeakR}));

  EXPECT_EQ(out_none.score.data, out_all.score.data);
  EXPECT_EQ(out_none.noise.data, out_all.noise.data);
  EXPECT_EQ(out_none.power.data, out_all.power.data);
}

TEST(CFARDetectorOpTest, PeakModeAllRank3TiesAndDoesNotWrapAxes) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kD = 16;
  constexpr int64_t kR = 32;
  const std::vector<int64_t> shape{kBatch, kD, kR};
  std::vector<float> host(static_cast<size_t>(kBatch * kD * kR), 1.0F);
  const auto offset = [](int64_t batch, int64_t doppler, int64_t range) {
    return static_cast<size_t>((batch * kD + doppler) * kR + range);
  };
  host[offset(0, 8, 16)] = 100.0F;
  host[offset(1, 8, 16)] = 100.0F;
  host[offset(0, 4, kR - 1)] = 90.0F;
  host[offset(0, 5, 0)] = 80.0F;
  auto input = to_device_tensor<float>(host, shape, real_dtype<float>());
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_rx = kBatch;
  extents.max_detections = kBatch * kD * kR;
  CFARParams params = ca_params(1e-2);
  params.peak_mode = CfarPeakMode::kAll;
  const auto out = run_cfar<float>(input, params, extents);

  ASSERT_EQ(num_detections(out.idx), 3);
  EXPECT_EQ(
      std::count(out.mask.as<uint8_t>(), out.mask.as<uint8_t>() + out.mask.size(), uint8_t{1}), 3);
  const int64_t* indices = out.idx.as<int64_t>();
  EXPECT_EQ(std::vector<int64_t>(indices, indices + 3), (std::vector<int64_t>{0, 4, kR - 1}));
  EXPECT_EQ(std::vector<int64_t>(indices + 3, indices + 6), (std::vector<int64_t>{0, 5, 0}));
  EXPECT_EQ(std::vector<int64_t>(indices + 6, indices + 9), (std::vector<int64_t>{0, 8, 16}));
}

TEST(CFARDetectorOpTest, ThresholdScaleConfigurableFromParams) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams detect = ca_params(1e-3);
  detect.threshold_scale = 1.0;
  CFARParams suppress = ca_params(1e-3);
  suppress.threshold_scale = 1e4;
  EXPECT_EQ(num_detections(run_cfar<double>(input, detect).idx), 1);
  EXPECT_EQ(num_detections(run_cfar<double>(input, suppress).idx), 0);
}

TEST(CFARDetectorOpTest, MaxDetectionsConfigurableFromParams) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.max_detections = 1;
  EXPECT_EQ(num_detections(run_cfar<double>(input, params).idx), 1);
}

TEST(CFARDetectorOpTest, PfaConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 26.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams loose = ca_params(1e-2);
  loose.blocking_alpha_lut_rebuild = true;
  CFARParams tight = ca_params(1e-8);
  tight.blocking_alpha_lut_rebuild = true;
  EXPECT_GT(score_at(run_cfar<double>(input, loose).score, dt, rt),
            score_at(run_cfar<double>(input, tight).score, dt, rt));
}
