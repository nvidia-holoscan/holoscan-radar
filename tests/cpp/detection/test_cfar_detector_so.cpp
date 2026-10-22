// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SO-CFAR contract tests for CFARDetectorOp on the HSDK 5 Graph harness.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::CFARDetectorOp;
using holoscan::radar::CFARParams;
using holoscan::radar::CfarPeakMode;
using holoscan::radar::CfarVariant;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kDoppler = 16;
constexpr int64_t kRange = 32;

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

// Numerical calibration cost scales with 1/Pfa. Shape/noise/score tests use
// affordable probabilities; rare-event budgeting and cancellation have dedicated tests.
CFARParams so_params(double pfa) {
  CFARParams p{};
  p.cfar_variant = CfarVariant::kSO;
  p.guard_cells = {1, 1};
  p.training_cells = {2, 2};
  p.pfa = pfa;
  return p;
}

template <typename T>
holoscan::testing::TestResult run_cfar_result(std::shared_ptr<holoscan::Tensor> input,
                                              CFARParams params) {
  using Op = CFARDetectorOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
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
  bool observed = true;
  for (const char* output : kOutputs) {
    observed = completion.wait_for_output(running, output, 1U) && observed;
  }
  (void)observed;
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

template <typename T>
CfarOutputs run_cfar(std::shared_ptr<holoscan::Tensor> input, CFARParams params) {
  const auto result = run_cfar_result<T>(std::move(input), std::move(params));
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

TEST(CFARDetectorOpSoTest, DetectsSingleTargetRealPath) {
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);

  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[dt * kRange + rt], 1);
  const auto* score = out.score.as<double>();
  EXPECT_GT(score[dt * kRange + rt], 1.0);
}

TEST(CFARDetectorOpSoTest, NoiseEstimateTracksFloor) {
  const double floor = 3.5;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
}

TEST(CFARDetectorOpSoTest, NoFalseAlarmsOnFlatMap) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpSoTest, ComplexInputFormsPower) {
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(static_cast<size_t>(kDoppler * kRange),
                                         std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] = std::complex<double>(10.0, 0.0);
  auto input =
      to_device_tensor<std::complex<double>>(cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);
  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[0], 1.0, 1e-9);
}

TEST(CFARDetectorOpSoTest, CfarPowerIsSquareLawPowerWithoutNci) {
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(static_cast<size_t>(kDoppler * kRange),
                                         std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] = std::complex<double>(3.0, 4.0);
  auto input =
      to_device_tensor<std::complex<double>>(cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  ASSERT_EQ(out.power.shape, (std::vector<int64_t>{kDoppler, kRange}));
  ASSERT_EQ(out.power.shape, out.score.shape);
  const auto* p = out.power.as<double>();
  for (int64_t i = 0; i < kDoppler * kRange; ++i) {
    EXPECT_NEAR(p[i], std::norm(cube[static_cast<size_t>(i)]), 1e-9) << i;
  }
  EXPECT_NEAR(p[dt * kRange + rt], 25.0, 1e-9);
}

TEST(CFARDetectorOpSoTest, CfarPowerIsNciSumOverFold) {
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

  CFARParams params = so_params(1e-3);
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

TEST(CFARDetectorOpSoTest, MaxDetectionsCap) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  CFARParams params = so_params(1e-3);
  params.max_detections = 1;
  const auto out = run_cfar<double>(input, params);

  EXPECT_EQ(num_detections(out.idx), 1);
}

TEST(CFARDetectorOpSoTest, FloatDoubleAgree) {
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);

  std::vector<float> mapf(map.begin(), map.end());
  auto in_f = to_device_tensor<float>(mapf, {kDoppler, kRange}, real_dtype<float>());
  auto in_d = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out_f = run_cfar<float>(in_f, so_params(1e-3));
  const auto out_d = run_cfar<double>(in_d, so_params(1e-3));

  ASSERT_EQ(num_detections(out_f.idx), num_detections(out_d.idx));
  const auto* rf = out_f.idx.as<int64_t>();
  const auto* rd = out_d.idx.as<int64_t>();
  EXPECT_EQ(rf[0], rd[0]);
  EXPECT_EQ(rf[1], rd[1]);
}

TEST(CFARDetectorOpSoTest, DetectsOnLargeMap) {
  constexpr int64_t kLargeDoppler = 256;
  constexpr int64_t kLargeRange = 512;
  const int64_t dt = 128;
  const int64_t rt = 256;
  const auto map = power_map(kLargeDoppler, kLargeRange, 1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kLargeDoppler, kLargeRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
  expect_mask_matches_score_interior<double>(out.mask, out.score, kLargeDoppler, kLargeRange);
}

TEST(CFARDetectorOpSoTest, MaskMatchesScoreInterior) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler, kRange);
}

TEST(CFARDetectorOpSoTest, RealAndComplexPathsAgree) {
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

  const auto out_real = run_cfar<double>(real_input, so_params(1e-3));
  const auto out_complex = run_cfar<double>(complex_input, so_params(1e-3));

  ASSERT_EQ(num_detections(out_real.idx), num_detections(out_complex.idx));
  EXPECT_TRUE(idx_contains(out_real.idx, dt, rt));
  EXPECT_TRUE(idx_contains(out_complex.idx, dt, rt));
}

TEST(CFARDetectorOpSoTest, BorderCellsUseInBoundsTrainingOnly) {
  const double floor = 4.0;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  const auto* noise = out.noise.as<double>();
  const auto* scores = out.score.as<double>();
  EXPECT_NEAR(noise[0], floor, 1e-6);
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
  EXPECT_LE(scores[0], 1.0);
  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpSoTest, DetectsTwoTargets) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, 4, 8));
  EXPECT_TRUE(idx_contains(out.idx, 12, 24));
}

TEST(CFARDetectorOpSoTest, DetectionIdxRasterOrder) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, so_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  ASSERT_EQ(out.idx.shape.size(), 2u);
  ASSERT_EQ(out.idx.shape[1], 2);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], 4);
  EXPECT_EQ(rows[1], 8);
  EXPECT_EQ(rows[2], 12);
  EXPECT_EQ(rows[3], 24);
}

TEST(CFARDetectorOpSoTest, LessConservativeThanGoWithOneSidedInterferer) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 50.0;
  map[static_cast<size_t>(dt * kRange + rt + 3)] = 10000.0;
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  CFARParams go_params{};
  go_params.cfar_variant = CfarVariant::kGO;
  go_params.guard_cells = {1, 1};
  go_params.training_cells = {2, 2};
  go_params.pfa = 1e-3;
  go_params.blocking_alpha_lut_rebuild = true;
  const auto out_go = run_cfar<double>(input, go_params);
  const auto out_so = run_cfar<double>(input, so_params(1e-3));

  EXPECT_GT(score_at(out_so.score, dt, rt), score_at(out_go.score, dt, rt));
  EXPECT_LT(out_so.noise.as<double>()[dt * kRange + rt],
            out_go.noise.as<double>()[dt * kRange + rt]);
}

TEST(CFARDetectorOpSoTest, GuardCellsConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  map[static_cast<size_t>(dt * kRange + rt + 2)] = 10000.0;
  auto input = to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams narrow = so_params(1e-3);
  narrow.blocking_alpha_lut_rebuild = true;
  CFARParams wide = narrow;
  wide.guard_cells = {1, 2};
  const auto n = run_cfar<double>(input, narrow);
  const auto w = run_cfar<double>(input, wide);
  // DIAG: CA direction
  EXPECT_LT(score_at(n.score, dt, rt), score_at(w.score, dt, rt))
      << "narrow=" << score_at(n.score, dt, rt) << " wide=" << score_at(w.score, dt, rt);
}

TEST(CFARDetectorOpSoTest, TrainingCellsConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(6 * kRange + rt)] = 10000.0;
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  auto input = to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams wide = so_params(1e-3);
  wide.blocking_alpha_lut_rebuild = true;
  wide.training_cells = {3, 3};
  CFARParams narrow = so_params(1e-3);
  narrow.blocking_alpha_lut_rebuild = true;
  narrow.training_cells = {1, 1};
  const auto w = run_cfar<double>(input, wide);
  const auto n = run_cfar<double>(input, narrow);
  EXPECT_GT(score_at(w.score, dt, rt), score_at(n.score, dt, rt))
      << "wide=" << score_at(w.score, dt, rt) << " narrow=" << score_at(n.score, dt, rt);
}

TEST(CFARDetectorOpSoTest, NciIntegratesChannelsForDetection) {
  constexpr int64_t kChannels = 8;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const double floor = 1.0;
  const double weak_peak = 20.0;

  const auto flat_cube = power_cube({kChannels, kDoppler, kRange}, floor, {});
  auto flat_input =
      to_device_tensor<double>(flat_cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams flat_params = so_params(1e-3);
  const auto per_channel = run_cfar<double>(flat_input, flat_params);
  EXPECT_EQ(num_detections(per_channel.idx), 0);

  std::vector<std::pair<std::vector<int64_t>, double>> cells;
  for (int64_t c = 0; c < kChannels; ++c) {
    cells.push_back({{c, dt, rt}, c == 0 ? weak_peak : floor});
  }
  const auto cube = power_cube({kChannels, kDoppler, kRange}, floor, cells);
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = so_params(1e-3);
  params.nci_axes = {0};
  const auto integrated = run_cfar<double>(input, params);

  EXPECT_GT(integrated.score.as<double>()[dt * kRange + rt], 1.0);
  ASSERT_GE(num_detections(integrated.idx), 1);
  EXPECT_TRUE(idx_contains(integrated.idx, dt, rt));
}

TEST(CFARDetectorOpSoTest, NciSumsAllChannelPower) {
  constexpr int64_t kChannels = 4;
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  std::vector<std::pair<std::vector<int64_t>, double>> cells;
  for (int64_t c = 0; c < kChannels; ++c) {
    cells.push_back({{c, dt, rt}, 10.0});
  }
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0, cells);
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = so_params(1e-3);
  params.nci_axes = {0};
  const auto out = run_cfar<double>(input, params);

  EXPECT_GT(out.score.as<double>()[dt * kRange + rt], 1.0);
  ASSERT_EQ(num_detections(out.idx), 1);
}

TEST(CFARDetectorOpSoTest, NciPreservesBatchAxis) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kChannels = 3;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const double floor = 1.0;
  const double peak = 12.0;

  std::vector<std::pair<std::vector<int64_t>, double>> cells;
  for (int64_t c = 0; c < kChannels; ++c) {
    cells.push_back({{1, c, dt, rt}, peak});
  }
  const auto cube = power_cube({kBatch, kChannels, kDoppler, kRange}, floor, cells);
  auto input =
      to_device_tensor<double>(cube, {kBatch, kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = so_params(1e-3);
  params.nci_axes = {1};
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  EXPECT_EQ(out.mask.shape, (std::vector<int64_t>{kBatch, kDoppler, kRange}));
  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpSoTest, BatchIdxIncludesPreservedAxes) {
  constexpr int64_t kBatch = 2;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube =
      power_cube({kBatch, kDoppler, kRange}, 1.0, {{{0, dt, rt}, 100.0}, {{1, dt, rt}, 100.0}});
  auto input = to_device_tensor<double>(cube, {kBatch, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = so_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpSoTest, NciMaskMatchesScoreOnIntegratedMap) {
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0, {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = so_params(1e-3);
  params.nci_axes = {0};
  const auto out = run_cfar<double>(input, params);

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler, kRange);
}

TEST(CFARDetectorOpSoTest, RejectsMissingGuardCells) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams params{};
  params.cfar_variant = CfarVariant::kSO;
  params.training_cells = {2, 2};
  params.pfa = 1e-3;
  const auto result = run_cfar_result<double>(input, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorOpSoTest, RejectsInvalidNciAxis) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams params = so_params(1e-3);
  params.nci_axes = {0};
  const auto result = run_cfar_result<double>(input, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorOpSoTest, SoAlphaInvertsMonteCarloPfa) {
  constexpr int64_t kN = 8;
  constexpr double kPfa = 1e-2;
  const double alpha = holoscan::radar::detail::so_cfar_alpha(kPfa, kN, 1, 1, 1, 2, 2);
  const double recovered = holoscan::radar::detail::so_cfar_pfa_from_alpha(alpha, kN, 1);
  EXPECT_NEAR(recovered, kPfa, 0.02);
}

TEST(CFARDetectorOpSoTest, SoAlphaDependsOnSideCount) {
  constexpr double kPfa = 1e-2;
  const double alpha_small = holoscan::radar::detail::so_cfar_alpha(kPfa, 8, 1, 1, 1, 2, 2);
  const double alpha_large = holoscan::radar::detail::so_cfar_alpha(kPfa, 18, 1, 1, 1, 2, 2);
  ASSERT_GT(alpha_large, 0.0);
  EXPECT_GT(std::abs(alpha_small - alpha_large) / alpha_large, 0.05)
      << "alpha_small=" << alpha_small << " alpha_large=" << alpha_large;
}

TEST(CFARDetectorOpSoTest, SoLutUsesCellFaithfulAlpha) {
  constexpr double kPfa = 1e-2;
  const auto lut = holoscan::radar::detail::build_so_alpha_lut(kPfa, 1, 32, 32, 1, 1, 2, 2);
  ASSERT_GT(lut.size(), 10u);
  ASSERT_GT(lut[10], 0.0);
  const double at_corner =
      holoscan::radar::detail::so_cfar_alpha_at_cell(kPfa, 1, 0, 0, 32, 32, 1, 1, 2, 2);
  const double at_edge =
      holoscan::radar::detail::so_cfar_alpha_at_cell(kPfa, 1, 0, 16, 32, 32, 1, 1, 2, 2);
  const double four_side = holoscan::radar::detail::so_cfar_alpha(kPfa, 10, 1, 1, 1, 2, 2);
  EXPECT_GE(lut[10], at_corner * 0.95) << "lut=" << lut[10] << " at_corner=" << at_corner;
  EXPECT_GE(lut[10], at_edge * 0.95) << "lut=" << lut[10] << " at_edge=" << at_edge;
  EXPECT_GT(std::abs(lut[10] - four_side) / lut[10], 0.05)
      << "lut=" << lut[10] << " four_side_n10=" << four_side;
}

TEST(CFARDetectorOpSoTest, PeakModeNoneMatchesRawMask) {
  const auto map = power_map(1.0, {{8, 15}, {8, 16}, {8, 17}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto raw = run_cfar<double>(input, so_params(1e-2));
  CFARParams off = so_params(1e-2);
  off.peak_mode = CfarPeakMode::kNone;
  const auto out = run_cfar<double>(input, off);

  EXPECT_EQ(num_detections(raw.idx), 3);
  EXPECT_EQ(num_detections(out.idx), num_detections(raw.idx));
  for (const auto r : {15, 16, 17}) {
    EXPECT_TRUE(idx_contains(out.idx, 8, r));
  }
}

TEST(CFARDetectorOpSoTest, PeakModeRangeDopplerKeepsSingleMax) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(8 * kRange + 17)] = 50.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange}, real_dtype<double>());

  CFARParams params = so_params(1e-2);
  params.peak_mode = CfarPeakMode::kRangeDoppler;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, 8, 16));
  EXPECT_FALSE(idx_contains(out.idx, 8, 17));
  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[8 * kRange + 16], 1);
  EXPECT_EQ(mask[8 * kRange + 17], 0);
}

TEST(CFARDetectorOpSoTest, PeakModeRangeDopplerUsesRasterTiebreak) {
  const auto map = power_map(1.0, {{8, 15}, {8, 16}, {8, 17}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  CFARParams params = so_params(1e-2);
  params.peak_mode = CfarPeakMode::kRangeDoppler;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, 8, 15));
}

TEST(CFARDetectorOpSoTest, PeakModeRangeDopplerUsesConfiguredRadius) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(9 * kRange + 17)] = 90.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange}, real_dtype<double>());

  CFARParams doppler_only = so_params(1e-2);
  doppler_only.peak_mode = CfarPeakMode::kRangeDoppler;
  doppler_only.peak_radius = {1, 0};
  CFARParams square = doppler_only;
  square.peak_radius = {1, 1};

  const auto out_doppler_only = run_cfar<double>(input, doppler_only);
  const auto out_square = run_cfar<double>(input, square);

  EXPECT_EQ(num_detections(out_doppler_only.idx), 2);
  EXPECT_EQ(num_detections(out_square.idx), 1);
  EXPECT_TRUE(idx_contains(out_square.idx, 8, 16));
}

TEST(CFARDetectorOpSoTest, ThresholdScaleConfigurableFromParams) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams detect = so_params(1e-3);
  detect.threshold_scale = 1.0;
  CFARParams suppress = so_params(1e-3);
  suppress.threshold_scale = 1e4;
  EXPECT_EQ(num_detections(run_cfar<double>(input, detect).idx), 1);
  EXPECT_EQ(num_detections(run_cfar<double>(input, suppress).idx), 0);
}

TEST(CFARDetectorOpSoTest, MaxDetectionsConfigurableFromParams) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams params = so_params(1e-3);
  params.max_detections = 1;
  EXPECT_EQ(num_detections(run_cfar<double>(input, params).idx), 1);
}

TEST(CFARDetectorOpSoTest, PfaConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 26.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams loose = so_params(1e-2);
  loose.blocking_alpha_lut_rebuild = true;
  CFARParams tight = so_params(1e-3);
  tight.blocking_alpha_lut_rebuild = true;
  EXPECT_GT(score_at(run_cfar<double>(input, loose).score, dt, rt),
            score_at(run_cfar<double>(input, tight).score, dt, rt));
}
