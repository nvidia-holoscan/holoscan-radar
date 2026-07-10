// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GO-CFAR contract tests for CFARDetectorOp (mirrors the CA matrix in
// test_cfar_detector.cpp with cfar_variant = GO).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::CFARDetectorOp;
using holoscan::radar::CfarVariant;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

constexpr int64_t kDoppler = 16;
constexpr int64_t kRange = 32;

struct CfarOutputs {
  HostTensor mask;
  HostTensor score;
  HostTensor noise;
  HostTensor idx;
};

std::vector<double> power_map(int64_t num_doppler, int64_t num_range,
                              double floor,
                              const std::vector<std::pair<int64_t, int64_t>>&
                                  targets,
                              double peak) {
  std::vector<double> map(static_cast<size_t>(num_doppler * num_range), floor);
  for (const auto& [d, r] : targets) {
    map[static_cast<size_t>(d * num_range + r)] = peak;
  }
  return map;
}

std::vector<double> power_map(
    double floor,
    const std::vector<std::pair<int64_t, int64_t>>& targets, double peak) {
  return power_map(kDoppler, kRange, floor, targets, peak);
}

template <typename T, typename... Args>
CfarOutputs run_cfar(std::shared_ptr<holoscan::Tensor> input, Args&&... args) {
  auto harness = create_operator_test<CFARDetectorOp<T>>(
      std::forward<Args>(args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps", {single_map(std::move(input), "rd_maps")});

  TensorMapCapture mask_cap;
  TensorMapCapture score_cap;
  TensorMapCapture noise_cap;
  TensorMapCapture idx_cap;
  harness->template add_output_port<holoscan::TensorMap>(
      "detection_mask", mask_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "detection_score", score_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "noise_estimate", noise_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "detection_idx", idx_cap.validators());

  harness->run_test();

  EXPECT_EQ(mask_cap.frames.size(), 1u);
  EXPECT_EQ(score_cap.frames.size(), 1u);
  EXPECT_EQ(noise_cap.frames.size(), 1u);
  EXPECT_EQ(idx_cap.frames.size(), 1u);
  return {std::move(mask_cap.frames.front()),
          std::move(score_cap.frames.front()),
          std::move(noise_cap.frames.front()),
          std::move(idx_cap.frames.front())};
}

auto go_args(double pfa) {
  return std::make_tuple(
      holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = pfa);
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

int64_t linear_index(const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& coords) {
  int64_t idx = 0;
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    idx += coords[static_cast<size_t>(i)] * stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return idx;
}

std::vector<double> power_cube(
    const std::vector<int64_t>& shape, double floor,
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
void expect_mask_matches_score_interior(const HostTensor& mask,
                                        const HostTensor& score,
                                        int64_t num_doppler,
                                        int64_t num_range) {
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

void wire_noop_cfar_output_sinks(holoscan::Fragment* app,
                                 const std::shared_ptr<holoscan::Operator>& op,
                                 bool capture_score) {
  const auto noop = [](const holoscan::TensorMap&) {};
  for (const char* port :
       {"detection_mask", "detection_score", "noise_estimate"}) {
    if (capture_score && std::string(port) == "detection_score") {
      continue;
    }
    auto sink = app->make_operator<
        holoscan::test::TestHarnessSinkOp<holoscan::TensorMap>>(
        std::string(port) + "_sink");
    sink->set_validators({noop});
    app->add_flow(op, sink, {{port, "input"}});
  }
}

class LiveGoCfarApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  TensorMapCapture* idx_capture = nullptr;
  TensorMapCapture* score_capture = nullptr;
  std::function<void(holoscan::Fragment*,
                     const std::shared_ptr<holoscan::Operator>&)>
      wire_overrides;

  void compose() override {
    auto op = make_operator<CFARDetectorOp<double>>(
        "op_under_test", holoscan::Arg("cfar_variant") = CfarVariant::kGO,
        holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
        holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
        holoscan::Arg("pfa") = 1e-3,
        holoscan::Arg("threshold_scale") = 1.0,
        holoscan::Arg("nci_axes") = std::vector<int64_t>{},
        holoscan::Arg("blocking_alpha_lut_rebuild") = true);
    wire_data_source(this, op, "rd_maps", frames);
    if (wire_overrides) {
      wire_overrides(this, op);
    }
    wire_capture_sink(this, op, "detection_idx", idx_capture);
    if (score_capture != nullptr) {
      wire_capture_sink(this, op, "detection_score", score_capture);
      wire_noop_cfar_output_sinks(this, op, /*capture_score=*/true);
    } else {
      wire_noop_cfar_output_sinks(this, op, /*capture_score=*/false);
    }
  }
};

void run_three_tick_live_test(
    std::vector<holoscan::TensorMap> frames,
    const std::function<void(holoscan::Fragment*,
                             const std::shared_ptr<holoscan::Operator>&)>&
        wire_overrides,
    const std::function<void(const std::vector<HostTensor>&)>& expect,
    TensorMapCapture* score_capture = nullptr) {
  TensorMapCapture capture;
  auto app = holoscan::make_application<LiveGoCfarApp>();
  app->frames = std::move(frames);
  app->wire_overrides = wire_overrides;
  app->idx_capture = &capture;
  app->score_capture = score_capture;
  app->run();
  ASSERT_EQ(capture.frames.size(), 3u);
  if (score_capture != nullptr) {
    ASSERT_EQ(score_capture->frames.size(), 3u);
  }
  expect(capture.frames);
}

double score_at(const HostTensor& score, int64_t doppler, int64_t range) {
  return score.as<double>()[doppler * kRange + range];
}

class LiveGoThresholdScaleApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  std::vector<double> threshold_scale_overrides;
  TensorMapCapture* idx_capture = nullptr;

  void compose() override {
    auto op = make_operator<CFARDetectorOp<double>>(
        "op_under_test", holoscan::Arg("cfar_variant") = CfarVariant::kGO,
        holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
        holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
        holoscan::Arg("pfa") = 1e-3,
        holoscan::Arg("threshold_scale") = 1.0);
    wire_data_source(this, op, "rd_maps", frames);
    wire_leading_override<double>(this, op, "threshold_scale",
                                  threshold_scale_overrides);
    wire_capture_sink(this, op, "detection_idx", idx_capture);
    wire_noop_cfar_output_sinks(this, op, /*capture_score=*/false);
  }
};

}  // namespace

TEST(CFARDetectorOpGoTest, DetectsSingleTargetRealPath) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);
  EXPECT_EQ(out.mask.as<uint8_t>()[dt * kRange + rt], 1);
  EXPECT_GT(out.score.as<double>()[dt * kRange + rt], 1.0);
}

TEST(CFARDetectorOpGoTest, NoiseEstimateTracksFloor) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const double floor = 3.5;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  EXPECT_NEAR(out.noise.as<double>()[8 * kRange + 16], floor, 1e-9);
}

TEST(CFARDetectorOpGoTest, NoFalseAlarmsOnFlatMap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-6));

  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpGoTest, ComplexInputFormsPower) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(kDoppler * kRange), std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] = std::complex<double>(10.0, 0.0);
  auto input = to_device_tensor<std::complex<double>>(
      cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_NEAR(out.noise.as<double>()[0], 1.0, 1e-9);
}

TEST(CFARDetectorOpGoTest, MaxDetectionsCap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("max_detections") = int64_t{1});

  EXPECT_EQ(num_detections(out.idx), 1);
}

TEST(CFARDetectorOpGoTest, FloatDoubleAgree) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);

  std::vector<float> mapf(map.begin(), map.end());
  auto in_f = to_device_tensor<float>(mapf, {kDoppler, kRange},
                                      real_dtype<float>());
  auto in_d = to_device_tensor<double>(map, {kDoppler, kRange},
                                       real_dtype<double>());

  const auto out_f = std::apply(
      [&](auto&&... a) {
        return run_cfar<float>(in_f, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));
  const auto out_d = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(in_d, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  ASSERT_EQ(num_detections(out_f.idx), num_detections(out_d.idx));
  EXPECT_EQ(out_f.idx.as<int64_t>()[0], out_d.idx.as<int64_t>()[0]);
  EXPECT_EQ(out_f.idx.as<int64_t>()[1], out_d.idx.as<int64_t>()[1]);
}

TEST(CFARDetectorOpGoTest, DetectsOnLargeMap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kLargeDoppler = 256;
  constexpr int64_t kLargeRange = 512;
  const int64_t dt = 128;
  const int64_t rt = 256;
  const auto map =
      power_map(kLargeDoppler, kLargeRange, 1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kLargeDoppler, kLargeRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
  expect_mask_matches_score_interior<double>(out.mask, out.score,
                                             kLargeDoppler, kLargeRange);
}

TEST(CFARDetectorOpGoTest, MaskMatchesScoreInterior) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler,
                                             kRange);
}

TEST(CFARDetectorOpGoTest, RealAndComplexPathsAgree) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t dt = 10;
  const int64_t rt = 22;
  const double floor = 2.0;
  const double peak = 80.0;

  const auto real_map = power_map(floor, {{dt, rt}}, peak);
  auto real_input = to_device_tensor<double>(real_map, {kDoppler, kRange},
                                             real_dtype<double>());

  std::vector<std::complex<double>> complex_map(
      static_cast<size_t>(kDoppler * kRange),
      std::complex<double>(std::sqrt(floor), 0.0));
  complex_map[static_cast<size_t>(dt * kRange + rt)] =
      std::complex<double>(std::sqrt(peak), 0.0);
  auto complex_input = to_device_tensor<std::complex<double>>(
      complex_map, {kDoppler, kRange}, complex_dtype<double>());

  const auto out_real = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(real_input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));
  const auto out_complex = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(complex_input,
                               std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  ASSERT_EQ(num_detections(out_real.idx), num_detections(out_complex.idx));
  EXPECT_TRUE(idx_contains(out_real.idx, dt, rt));
  EXPECT_TRUE(idx_contains(out_complex.idx, dt, rt));
}

TEST(CFARDetectorOpGoTest, BorderCellsUseInBoundsTrainingOnly) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const double floor = 4.0;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-6));

  const auto* noise = out.noise.as<double>();
  const auto* scores = out.score.as<double>();
  EXPECT_NEAR(noise[0], floor, 1e-6);
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
  EXPECT_LE(scores[0], 1.0);
  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpGoTest, DetectsTwoTargets) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, 4, 8));
  EXPECT_TRUE(idx_contains(out.idx, 12, 24));
}

TEST(CFARDetectorOpGoTest, MoreConservativeThanCaWithOneSidedInterferer) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 50.0;
  map[static_cast<size_t>(dt * kRange + rt + 3)] = 10000.0;
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out_ca = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);
  const auto out_go = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      go_args(1e-3));

  EXPECT_GT(score_at(out_ca.score, dt, rt), score_at(out_go.score, dt, rt));
  EXPECT_GT(out_go.noise.as<double>()[dt * kRange + rt],
            out_ca.noise.as<double>()[dt * kRange + rt]);
}

TEST(CFARDetectorOpGoTest, LiveThresholdScaleOverrideMidStreamAndSticky) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto frame = [&]() {
    return single_map(to_device_tensor(map, {kDoppler, kRange},
                                       real_dtype<double>()),
                      "rd_maps");
  };

  TensorMapCapture capture;
  auto app = holoscan::make_application<LiveGoThresholdScaleApp>();
  app->frames = {frame(), frame(), frame()};
  app->threshold_scale_overrides = {1.0, 1.0e4};
  app->idx_capture = &capture;
  app->run();

  ASSERT_EQ(capture.frames.size(), 3u);
  EXPECT_EQ(num_detections(capture.frames[0]), 1);
  EXPECT_EQ(num_detections(capture.frames[1]), 0);
  EXPECT_EQ(num_detections(capture.frames[2]), 0);
}

TEST(CFARDetectorOpGoTest, HarnessGuardCellsOverridePerTick) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  map[static_cast<size_t>(dt * kRange + rt + 2)] = 10000.0;
  auto input = to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  auto frame = [&]() { return single_map(input, "rd_maps"); };

  TensorMapCapture score_cap;
  auto harness = create_operator_test<CFARDetectorOp<double>>(
      holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);
  harness->add_input_port<holoscan::TensorMap>("rd_maps",
                                               {frame(), frame(), frame()});
  harness->add_input_port<std::vector<int64_t>>(
      "guard_cells",
      {std::vector<int64_t>{1, 1}, std::vector<int64_t>{1, 2},
       std::vector<int64_t>{1, 2}});
  harness->add_output_port<holoscan::TensorMap>("detection_score",
                                                score_cap.validators());
  TensorMapCapture mask_cap;
  TensorMapCapture noise_cap;
  TensorMapCapture idx_cap;
  harness->add_output_port<holoscan::TensorMap>("detection_mask",
                                                mask_cap.validators());
  harness->add_output_port<holoscan::TensorMap>("noise_estimate",
                                                noise_cap.validators());
  harness->add_output_port<holoscan::TensorMap>("detection_idx",
                                                idx_cap.validators());
  harness->run_test();
  ASSERT_EQ(score_cap.frames.size(), 3u);
  const double s0 = score_at(score_cap.frames[0], dt, rt);
  const double s1 = score_at(score_cap.frames[1], dt, rt);
  const double s2 = score_at(score_cap.frames[2], dt, rt);
  EXPECT_LT(s0, s1);
  EXPECT_DOUBLE_EQ(s1, s2);
}

TEST(CFARDetectorOpGoTest, LivePfaOverrideMidStreamAndSticky) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 26.0);
  auto frame = [&]() {
    return single_map(to_device_tensor(map, {kDoppler, kRange},
                                       real_dtype<double>()),
                      "rd_maps");
  };
  TensorMapCapture score_cap;
  run_three_tick_live_test(
      {frame(), frame(), frame()},
      [](holoscan::Fragment* app,
         const std::shared_ptr<holoscan::Operator>& op) {
        wire_leading_override<double>(app, op, "pfa", {1e-2, 1e-8});
      },
      [&](const std::vector<HostTensor>&) {
        const double s0 = score_at(score_cap.frames[0], dt, rt);
        const double s1 = score_at(score_cap.frames[1], dt, rt);
        const double s2 = score_at(score_cap.frames[2], dt, rt);
        EXPECT_GT(s0, s1);
        EXPECT_DOUBLE_EQ(s1, s2);
      },
      &score_cap);
}

TEST(CFARDetectorOpGoTest, LiveMaxDetectionsOverrideMidStreamAndSticky) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto frame = [&]() {
    return single_map(to_device_tensor(map, {kDoppler, kRange},
                                       real_dtype<double>()),
                      "rd_maps");
  };
  run_three_tick_live_test(
      {frame(), frame(), frame()},
      [](holoscan::Fragment* app,
         const std::shared_ptr<holoscan::Operator>& op) {
        wire_leading_override<int64_t>(app, op, "max_detections",
                                       {int64_t{0}, int64_t{1}});
      },
      [](const std::vector<HostTensor>& idx) {
        EXPECT_EQ(num_detections(idx[0]), 2);
        EXPECT_EQ(num_detections(idx[1]), 1);
        EXPECT_EQ(num_detections(idx[2]), 1);
      });
}

TEST(CFARDetectorOpGoTest, LiveGuardCellsOverrideMidStreamAndSticky) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  map[static_cast<size_t>(dt * kRange + rt + 2)] = 10000.0;
  auto frame = [&]() {
    return single_map(to_device_tensor(map, {kDoppler, kRange},
                                       real_dtype<double>()),
                      "rd_maps");
  };
  TensorMapCapture score_cap;
  struct LiveGuardApp : public LiveGoCfarApp {
    void compose() override {
      auto op = make_operator<CFARDetectorOp<double>>(
          "op_under_test", holoscan::Arg("cfar_variant") = CfarVariant::kGO,
          holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
          holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
          holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);
      wire_data_source(this, op, "rd_maps", frames);
      wire_leading_override<std::vector<int64_t>>(
          this, op, "guard_cells",
          {std::vector<int64_t>{1, 1}, std::vector<int64_t>{1, 2}});
      wire_capture_sink(this, op, "detection_idx", idx_capture);
      wire_capture_sink(this, op, "detection_score", score_capture);
      wire_noop_cfar_output_sinks(this, op, /*capture_score=*/true);
    }
  };
  TensorMapCapture idx_cap;
  auto app = holoscan::make_application<LiveGuardApp>();
  app->frames = {frame(), frame(), frame()};
  app->idx_capture = &idx_cap;
  app->score_capture = &score_cap;
  app->run();
  ASSERT_EQ(score_cap.frames.size(), 3u);
  EXPECT_LT(score_at(score_cap.frames[0], dt, rt),
            score_at(score_cap.frames[1], dt, rt));
  EXPECT_DOUBLE_EQ(score_at(score_cap.frames[1], dt, rt),
                   score_at(score_cap.frames[2], dt, rt));
}

TEST(CFARDetectorOpGoTest, LiveTrainingCellsOverrideMidStreamAndSticky) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(6 * kRange + rt)] = 10000.0;
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  auto frame = [&]() {
    return single_map(to_device_tensor(map, {kDoppler, kRange},
                                       real_dtype<double>()),
                      "rd_maps");
  };
  TensorMapCapture score_cap;
  struct LiveTrainingApp : public LiveGoCfarApp {
    void compose() override {
      auto op = make_operator<CFARDetectorOp<double>>(
          "op_under_test", holoscan::Arg("cfar_variant") = CfarVariant::kGO,
          holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
          holoscan::Arg("training_cells") = std::vector<int64_t>{1, 1},
          holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);
      wire_data_source(this, op, "rd_maps", frames);
      wire_leading_override<std::vector<int64_t>>(
          this, op, "training_cells",
          {std::vector<int64_t>{1, 1}, std::vector<int64_t>{3, 3}});
      wire_capture_sink(this, op, "detection_idx", idx_capture);
      wire_capture_sink(this, op, "detection_score", score_capture);
      wire_noop_cfar_output_sinks(this, op, /*capture_score=*/true);
    }
  };
  TensorMapCapture idx_cap;
  auto app = holoscan::make_application<LiveTrainingApp>();
  app->frames = {frame(), frame(), frame()};
  app->idx_capture = &idx_cap;
  app->score_capture = &score_cap;
  app->run();
  ASSERT_EQ(score_cap.frames.size(), 3u);
  EXPECT_LT(score_at(score_cap.frames[0], dt, rt),
            score_at(score_cap.frames[1], dt, rt));
  EXPECT_DOUBLE_EQ(score_at(score_cap.frames[1], dt, rt),
                   score_at(score_cap.frames[2], dt, rt));
}

TEST(CFARDetectorOpGoTest, LiveNciAxesOverrideMidStreamAndSticky) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kChannels = 8;
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const double floor = 1.0;
  // Flat cube: per-channel GO must not fire before nci_axes integrates.
  const auto cube = power_cube({kChannels, kDoppler, kRange}, floor, {});
  auto frame = [&]() {
    return single_map(
        to_device_tensor(cube, {kChannels, kDoppler, kRange},
                         real_dtype<double>()),
        "rd_maps");
  };
  TensorMapCapture score_cap;
  struct LiveNciApp : public LiveGoCfarApp {
    void compose() override {
      auto op = make_operator<CFARDetectorOp<double>>(
          "op_under_test", holoscan::Arg("cfar_variant") = CfarVariant::kGO,
          holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
          holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
          holoscan::Arg("pfa") = 1e-6,
          holoscan::Arg("nci_axes") = std::vector<int64_t>{},
          holoscan::Arg("blocking_alpha_lut_rebuild") = true);
      wire_data_source(this, op, "rd_maps", frames);
      wire_leading_override<std::vector<int64_t>>(
          this, op, "nci_axes",
          {std::vector<int64_t>{}, std::vector<int64_t>{0}});
      wire_capture_sink(this, op, "detection_idx", idx_capture);
      wire_capture_sink(this, op, "detection_score", score_capture);
      wire_noop_cfar_output_sinks(this, op, /*capture_score=*/true);
    }
  };
  TensorMapCapture idx_cap;
  auto app = holoscan::make_application<LiveNciApp>();
  app->frames = {frame(), frame(), frame()};
  app->idx_capture = &idx_cap;
  app->score_capture = &score_cap;
  app->run();
  ASSERT_EQ(idx_cap.frames.size(), 3u);
  EXPECT_EQ(num_detections(idx_cap.frames[0]), 0);
  EXPECT_EQ(score_cap.frames[0].shape,
            (std::vector<int64_t>{kChannels, kDoppler, kRange}));
  EXPECT_EQ(score_cap.frames[1].shape,
            (std::vector<int64_t>{kDoppler, kRange}));
  EXPECT_EQ(score_cap.frames[2].shape, score_cap.frames[1].shape);
  EXPECT_DOUBLE_EQ(score_at(score_cap.frames[1], dt, rt),
                   score_at(score_cap.frames[2], dt, rt));
}

TEST(CFARDetectorOpGoTest, NciIntegratesChannelsForDetection) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kChannels = 8;
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const double floor = 1.0;
  const double weak_peak = 14.0;

  const auto flat_cube =
      power_cube({kChannels, kDoppler, kRange}, floor, {});
  auto flat_input = to_device_tensor<double>(
      flat_cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  const auto per_channel = run_cfar<double>(
      flat_input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-6);
  EXPECT_EQ(num_detections(per_channel.idx), 0);

  std::vector<std::pair<std::vector<int64_t>, double>> cells;
  for (int64_t c = 0; c < kChannels; ++c) {
    cells.push_back({{c, dt, rt}, c == 0 ? weak_peak : floor});
  }
  const auto cube = power_cube({kChannels, kDoppler, kRange}, floor, cells);
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange},
                                        real_dtype<double>());

  const auto integrated = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{0});

  EXPECT_GT(integrated.score.as<double>()[dt * kRange + rt], 1.0);
  ASSERT_GE(num_detections(integrated.idx), 1);
  EXPECT_TRUE(idx_contains(integrated.idx, dt, rt));
}

TEST(CFARDetectorOpGoTest, NciSumsAllChannelPower) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kChannels = 4;
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  std::vector<std::pair<std::vector<int64_t>, double>> cells;
  for (int64_t c = 0; c < kChannels; ++c) {
    cells.push_back({{c, dt, rt}, 10.0});
  }
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0, cells);
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{0});

  EXPECT_GT(out.score.as<double>()[dt * kRange + rt], 1.0);
  ASSERT_EQ(num_detections(out.idx), 1);
}

TEST(CFARDetectorOpGoTest, NciPreservesBatchAxis) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kBatch = 2;
  constexpr int64_t kChannels = 3;
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const double floor = 1.0;
  const double peak = 12.0;

  std::vector<std::pair<std::vector<int64_t>, double>> cells;
  for (int64_t c = 0; c < kChannels; ++c) {
    cells.push_back({{1, c, dt, rt}, peak});
  }
  const auto cube =
      power_cube({kBatch, kChannels, kDoppler, kRange}, floor, cells);
  auto input = to_device_tensor<double>(
      cube, {kBatch, kChannels, kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{1});

  EXPECT_EQ(out.mask.shape, (std::vector<int64_t>{kBatch, kDoppler, kRange}));
  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpGoTest, BatchIdxIncludesPreservedAxes) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kBatch = 2;
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const auto cube = power_cube(
      {kBatch, kDoppler, kRange}, 1.0,
      {{{0, dt, rt}, 100.0}, {{1, dt, rt}, 100.0}});
  auto input = to_device_tensor<double>(cube, {kBatch, kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpGoTest, NciMaskMatchesScoreOnIntegratedMap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0,
                               {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{0});

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler,
                                             kRange);
}

TEST(CFARDetectorOpGoTest, RejectsMissingGuardCells) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  EXPECT_THROW(
      run_cfar<double>(input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
                       holoscan::Arg("training_cells") =
                           std::vector<int64_t>{2, 2},
                       holoscan::Arg("pfa") = 1e-3),
      std::invalid_argument);
}

TEST(CFARDetectorOpGoTest, RejectsInvalidNciAxis) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  EXPECT_THROW(
      run_cfar<double>(input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
                       holoscan::Arg("guard_cells") =
                           std::vector<int64_t>{1, 1},
                       holoscan::Arg("training_cells") =
                           std::vector<int64_t>{2, 2},
                       holoscan::Arg("pfa") = 1e-3,
                       holoscan::Arg("nci_axes") = std::vector<int64_t>{0}),
      std::invalid_argument);
}

TEST(CFARDetectorOpGoTest, GoAlphaInvertsMonteCarloPfa) {
  constexpr int64_t kN = 8;
  constexpr double kPfa = 1e-2;
  const double alpha =
      holoscan::radar::detail::go_cfar_alpha(kPfa, kN, 1, 1, 1, 2, 2);
  const double recovered =
      holoscan::radar::detail::go_cfar_pfa_from_alpha(alpha, kN, 1);
  EXPECT_NEAR(recovered, kPfa, 0.02);
}
