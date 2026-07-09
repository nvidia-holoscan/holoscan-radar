// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for CFARDetectorOp (v1: CA variant). Driven through the SDK-native
// holoscan::test::OperatorTestHarness with radar helpers (sdk_harness.hpp).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

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

// Captured CFAR outputs for one tick.
struct CfarOutputs {
  HostTensor mask;
  HostTensor score;
  HostTensor noise;
  HostTensor idx;
};

// A flat power map at `floor` with optional (doppler, range) peaks at `peak`.
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

// Run the detector once over a prepared frame and capture all four outputs.
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

// Common CA arguments: guard (1,1), training (2,2).
auto ca_args(double pfa) {
  return std::make_tuple(
      holoscan::Arg("cfar_variant") = CfarVariant::kCA,
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

// Build a real power cube with a constant floor and optional cell overrides.
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

// Interior cells are far enough from the map edge that the full training
// window is in-bounds (guard 1, training 2 → half extent 3).
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

// Data ticks outnumber the override values, so the trailing tick exercises
// sticky overrides (effective until the next override arrives).
class LiveThresholdScaleApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  std::vector<double> threshold_scale_overrides;
  TensorMapCapture* idx_capture = nullptr;

  void compose() override {
    auto op = make_operator<CFARDetectorOp<double>>(
        "op_under_test", holoscan::Arg("cfar_variant") = CfarVariant::kCA,
        holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
        holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
        holoscan::Arg("pfa") = 1e-3,
        holoscan::Arg("threshold_scale") = 1.0);
    wire_data_source(this, op, "rd_maps", frames);
    wire_leading_override<double>(this, op, "threshold_scale",
                                  threshold_scale_overrides);
    wire_capture_sink(this, op, "detection_idx", idx_capture);
    // All four outputs must be consumed or the operator never ticks.
    const auto noop = [](const holoscan::TensorMap&) {};
    for (const char* port :
         {"detection_mask", "detection_score", "noise_estimate"}) {
      auto sink = make_operator<
          holoscan::test::TestHarnessSinkOp<holoscan::TensorMap>>(
          std::string(port) + "_sink");
      sink->set_validators({noop});
      add_flow(op, sink, {{port, "input"}});
    }
  }
};

}  // namespace

// --- single-target detection (real power path) ---------------------------

TEST(CFARDetectorOpTest, DetectsSingleTargetRealPath) {
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
      ca_args(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);

  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[dt * kRange + rt], 1);
  const auto* score = out.score.as<double>();
  EXPECT_GT(score[dt * kRange + rt], 1.0);
}

// --- noise estimate tracks the local floor -------------------------------

TEST(CFARDetectorOpTest, NoiseEstimateTracksFloor) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const double floor = 3.5;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      ca_args(1e-3));

  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
}

// --- flat map → no false alarms ------------------------------------------

TEST(CFARDetectorOpTest, NoFalseAlarmsOnFlatMap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      ca_args(1e-6));

  EXPECT_EQ(num_detections(out.idx), 0);
}

// --- complex input: square-law power path --------------------------------

TEST(CFARDetectorOpTest, ComplexInputFormsPower) {
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
      ca_args(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);
  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[0], 1.0, 1e-9);
}

// --- max_detections cap ---------------------------------------------------

TEST(CFARDetectorOpTest, MaxDetectionsCap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("max_detections") = int64_t{1});

  EXPECT_EQ(num_detections(out.idx), 1);
}

// --- float/double agreement ----------------------------------------------

TEST(CFARDetectorOpTest, FloatDoubleAgree) {
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
      ca_args(1e-3));
  const auto out_d = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(in_d, std::forward<decltype(a)>(a)...);
      },
      ca_args(1e-3));

  ASSERT_EQ(num_detections(out_f.idx), num_detections(out_d.idx));
  const auto* rf = out_f.idx.as<int64_t>();
  const auto* rd = out_d.idx.as<int64_t>();
  EXPECT_EQ(rf[0], rd[0]);
  EXPECT_EQ(rf[1], rd[1]);
}

// --- larger RD map -------------------------------------------------------

TEST(CFARDetectorOpTest, DetectsOnLargeMap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kLargeDoppler = 256;
  constexpr int64_t kLargeRange = 512;
  const int64_t dt = 128;
  const int64_t rt = 256;
  const auto map = power_map(kLargeDoppler, kLargeRange, 1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kLargeDoppler, kLargeRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      ca_args(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
  expect_mask_matches_score_interior<double>(out.mask, out.score,
                                             kLargeDoppler, kLargeRange);
}

// --- score / mask consistency (TESTING.md) -------------------------------

TEST(CFARDetectorOpTest, MaskMatchesScoreInterior) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      ca_args(1e-3));

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler,
                                             kRange);
  const auto* scores = out.score.as<double>();
  EXPECT_GT(scores[8 * kRange + 16], 1.0);
  const int64_t far = 4 * kRange + 4;
  EXPECT_LT(scores[far], 1.0);
}

// --- real vs complex parity (TESTING.md) ---------------------------------

TEST(CFARDetectorOpTest, RealAndComplexPathsAgree) {
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
      ca_args(1e-3));
  const auto out_complex = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(complex_input,
                               std::forward<decltype(a)>(a)...);
      },
      ca_args(1e-3));

  ASSERT_EQ(num_detections(out_real.idx), num_detections(out_complex.idx));
  EXPECT_TRUE(idx_contains(out_real.idx, dt, rt));
  EXPECT_TRUE(idx_contains(out_complex.idx, dt, rt));
}

// --- border / corner cells -----------------------------------------------

TEST(CFARDetectorOpTest, BorderCellsUseInBoundsTrainingOnly) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const double floor = 4.0;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      ca_args(1e-6));

  const auto* noise = out.noise.as<double>();
  const auto* scores = out.score.as<double>();
  // Corner uses a partial but valid training window; interior uses the full
  // count. Both should track the flat floor and stay below threshold.
  EXPECT_NEAR(noise[0], floor, 1e-6);
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
  EXPECT_LE(scores[0], 1.0);
  EXPECT_EQ(num_detections(out.idx), 0);
}

// --- two targets in detection_idx ----------------------------------------

TEST(CFARDetectorOpTest, DetectsTwoTargets) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = std::apply(
      [&](auto&&... a) {
        return run_cfar<double>(input, std::forward<decltype(a)>(a)...);
      },
      ca_args(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, 4, 8));
  EXPECT_TRUE(idx_contains(out.idx, 12, 24));
}

// --- [live] sticky threshold_scale override ------------------------------

TEST(CFARDetectorOpTest, LiveThresholdScaleOverrideMidStreamAndSticky) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto frame = [&]() {
    return single_map(to_device_tensor(map, {kDoppler, kRange},
                                       real_dtype<double>()),
                      "rd_maps");
  };

  TensorMapCapture capture;
  auto app = holoscan::make_application<LiveThresholdScaleApp>();
  app->frames = {frame(), frame(), frame()};
  // Tick 0: scale 1.0 detects. Tick 1: override to 1e4 suppresses.
  // Tick 2: no override message — tick-1 value remains sticky.
  app->threshold_scale_overrides = {1.0, 1.0e4};
  app->idx_capture = &capture;
  app->run();

  ASSERT_EQ(capture.frames.size(), 3u);
  EXPECT_EQ(num_detections(capture.frames[0]), 1);
  EXPECT_EQ(num_detections(capture.frames[1]), 0);
  EXPECT_EQ(num_detections(capture.frames[2]), 0);
}

// --- non-coherent integration (nci_axes) --------------------------------

TEST(CFARDetectorOpTest, NciIntegratesChannelsForDetection) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kChannels = 8;
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const double floor = 1.0;
  const double weak_peak = 3.0;  // per-channel marginal; integrated CUT = 10.

  std::vector<std::pair<std::vector<int64_t>, double>> cells;
  for (int64_t c = 0; c < kChannels; ++c) {
    cells.push_back({{c, dt, rt}, c == 0 ? weak_peak : floor});
  }
  const auto cube = power_cube({kChannels, kDoppler, kRange}, floor, cells);
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange},
                                        real_dtype<double>());

  const auto per_channel = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-6);
  EXPECT_EQ(num_detections(per_channel.idx), 0);

  const auto integrated = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{0});

  EXPECT_EQ(integrated.mask.shape,
            (std::vector<int64_t>{kDoppler, kRange}));
  const auto* scores = integrated.score.as<double>();
  EXPECT_GT(scores[dt * kRange + rt], 1.0);
  ASSERT_GE(num_detections(integrated.idx), 1);
  EXPECT_TRUE(idx_contains(integrated.idx, dt, rt));
}

TEST(CFARDetectorOpTest, NciSumsAllChannelPower) {
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
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{0});

  const auto* scores = out.score.as<double>();
  EXPECT_GT(scores[dt * kRange + rt], 1.0);
  ASSERT_EQ(num_detections(out.idx), 1);
}

TEST(CFARDetectorOpTest, NciPreservesBatchAxis) {
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
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{1});

  EXPECT_EQ(out.mask.shape,
            (std::vector<int64_t>{kBatch, kDoppler, kRange}));
  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_EQ(out.idx.shape, (std::vector<int64_t>{1, 3}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpTest, BatchIdxIncludesPreservedAxes) {
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
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpTest, NciMaskMatchesScoreOnIntegratedMap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0,
                               {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{0});

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler,
                                             kRange);
}

// --- error paths ---------------------------------------------------------

TEST(CFARDetectorOpTest, RejectsMissingGuardCells) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  EXPECT_THROW(
      run_cfar<double>(input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
                       holoscan::Arg("training_cells") =
                           std::vector<int64_t>{2, 2},
                       holoscan::Arg("pfa") = 1e-3),
      std::invalid_argument);
}

TEST(CFARDetectorOpTest, RejectsInvalidNciAxis) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  EXPECT_THROW(
      run_cfar<double>(input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
                       holoscan::Arg("guard_cells") =
                           std::vector<int64_t>{1, 1},
                       holoscan::Arg("training_cells") =
                           std::vector<int64_t>{2, 2},
                       holoscan::Arg("pfa") = 1e-3,
                       holoscan::Arg("nci_axes") = std::vector<int64_t>{0}),
      std::invalid_argument);
}

TEST(CFARDetectorOpTest, RejectsUnimplementedGoVariant) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  EXPECT_THROW(
      run_cfar<double>(input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
                       holoscan::Arg("guard_cells") =
                           std::vector<int64_t>{1, 1},
                       holoscan::Arg("training_cells") =
                           std::vector<int64_t>{2, 2},
                       holoscan::Arg("pfa") = 1e-3),
      std::invalid_argument);
}

TEST(CFARDetectorOpTest, RejectsUnimplementedSoVariant) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  EXPECT_THROW(
      run_cfar<double>(input, holoscan::Arg("cfar_variant") = CfarVariant::kSO,
                       holoscan::Arg("guard_cells") =
                           std::vector<int64_t>{1, 1},
                       holoscan::Arg("training_cells") =
                           std::vector<int64_t>{2, 2},
                       holoscan::Arg("pfa") = 1e-3),
      std::invalid_argument);
}

TEST(CFARDetectorOpTest, RejectsUnimplementedOsVariant) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  EXPECT_THROW(
      run_cfar<double>(input, holoscan::Arg("cfar_variant") = CfarVariant::kOS,
                       holoscan::Arg("guard_cells") =
                           std::vector<int64_t>{1, 1},
                       holoscan::Arg("training_cells") =
                           std::vector<int64_t>{2, 2},
                       holoscan::Arg("pfa") = 1e-3,
                       holoscan::Arg("os_rank") = int64_t{5}),
      std::invalid_argument);
}
