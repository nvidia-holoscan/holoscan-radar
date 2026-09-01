// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for CFARDetectorOp (OS variant) on the HSDK 5 Graph harness.
// Scalar live mid-stream Graph coverage remains deferred (HSDK5_PORT.md §1).

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

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::CFARDetectorOp;
using holoscan::radar::CFARParams;
using holoscan::radar::CfarPeakNeighborhood;
using holoscan::radar::CfarVariant;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::FiveOutGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_in_five_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kDoppler = 16;
constexpr int64_t kRange = 32;
constexpr int64_t kOsRank = 30;

struct CfarOutputs {
  HostTensor mask;
  HostTensor score;
  HostTensor noise;
  HostTensor idx;
  HostTensor power;
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

CFARParams os_params(double pfa) {
  CFARParams p{};
  p.cfar_variant = CfarVariant::kOS;
  p.os_rank = kOsRank;
  p.guard_cells = {1, 1};
  p.training_cells = {2, 2};
  p.pfa = pfa;
  return p;
}

// cpplint misreads the wrapped template header as an indented declaration
// inside the anonymous namespace; the parameter list cannot fit on one line.
template <typename Source, typename OpRef, typename S0, typename S1,
          typename S2, typename S3, typename S4>  // NOLINT(whitespace/indent_namespace)
void wire_cfar(holoscan::Graph& graph, const Source& source, OpRef& op,
               const S0& sink0, const S1& sink1, const S2& sink2,
               const S3& sink3, const S4& sink4) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->rd_maps, q);
  graph.add_flow(op->detection_mask, sink0->in, q);
  graph.add_flow(op->detection_score, sink1->in, q);
  graph.add_flow(op->noise_estimate, sink2->in, q);
  graph.add_flow(op->detection_idx, sink3->in, q);
  graph.add_flow(op->cfar_power, sink4->in, q);
}

template <typename T>
FiveOutGraphRunResult run_cfar_result(std::shared_ptr<holoscan::Tensor> input,
                                      CFARParams params) {
  return run_single_in_five_out_operator_graph_result<CFARDetectorOp<T>>(
      {std::move(input)}, std::move(params), "detection_mask",
      "detection_score", "noise_estimate", "detection_idx", "cfar_power",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1, const auto& sink2,
         const auto& sink3, const auto& sink4) {
        wire_cfar(graph, source, op, sink0, sink1, sink2, sink3, sink4);
      });
}

template <typename T>
CfarOutputs run_cfar(std::shared_ptr<holoscan::Tensor> input,
                     CFARParams params) {
  FiveOutGraphRunResult result =
      run_cfar_result<T>(std::move(input), std::move(params));
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames0.size() != 1U || result.frames1.size() != 1U ||
      result.frames2.size() != 1U || result.frames3.size() != 1U ||
      result.frames4.size() != 1U) {
    throw std::runtime_error(
        "CFARDetectorOp test: expected one output frame set");
  }
  return {std::move(result.frames0.front()), std::move(result.frames1.front()),
          std::move(result.frames2.front()), std::move(result.frames3.front()),
          std::move(result.frames4.front())};
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

double score_at(const HostTensor& score, int64_t d, int64_t r) {
  return score.as<double>()[d * kRange + r];
}

}  // namespace

TEST(CFARDetectorOpOsTest, DetectsSingleTargetRealPath) {
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);

  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[dt * kRange + rt], 1);
  const auto* score = out.score.as<double>();
  EXPECT_GT(score[dt * kRange + rt], 1.0);
}

TEST(CFARDetectorOpOsTest, NoiseEstimateTracksFloor) {
  const double floor = 3.5;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
}

TEST(CFARDetectorOpOsTest, NoFalseAlarmsOnFlatMap) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-6));

  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpOsTest, ComplexInputFormsPower) {
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(kDoppler * kRange),
      std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] =
      std::complex<double>(10.0, 0.0);
  auto input = to_device_tensor<std::complex<double>>(
      cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);
  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[0], 1.0, 1e-9);
}

TEST(CFARDetectorOpOsTest, CfarPowerIsSquareLawPowerWithoutNci) {
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(kDoppler * kRange),
      std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] =
      std::complex<double>(3.0, 4.0);
  auto input = to_device_tensor<std::complex<double>>(
      cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  ASSERT_EQ(out.power.shape,
            (std::vector<int64_t>{kDoppler, kRange}));
  ASSERT_EQ(out.power.shape, out.score.shape);
  const auto* p = out.power.as<double>();
  for (int64_t i = 0; i < kDoppler * kRange; ++i) {
    EXPECT_NEAR(p[i], std::norm(cube[static_cast<size_t>(i)]),
                1e-9)
        << i;
  }
  EXPECT_NEAR(p[dt * kRange + rt], 25.0, 1e-9);
}

TEST(CFARDetectorOpOsTest, CfarPowerIsNciSumOverFold) {
  constexpr int64_t kChannels = 4;
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(kChannels * kDoppler * kRange));
  for (int64_t k = 0; k < kChannels; ++k) {
    for (int64_t i = 0; i < kDoppler * kRange; ++i) {
      cube[static_cast<size_t>(k * kDoppler * kRange + i)] =
          std::complex<double>(static_cast<double>(k + 1), 0.0);
    }
    cube[static_cast<size_t>(k * kDoppler * kRange +
                             dt * kRange + rt)] =
        std::complex<double>(10.0, 0.0);
  }
  auto input = to_device_tensor<std::complex<double>>(
      cube, {kChannels, kDoppler, kRange}, complex_dtype<double>());

  CFARParams params = os_params(1e-3);
  params.nci_axes = {0};
  params.nci_fold = kChannels;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(out.power.shape,
            (std::vector<int64_t>{kDoppler, kRange}));
  ASSERT_EQ(out.power.shape, out.score.shape);
  const auto* p = out.power.as<double>();
  EXPECT_NEAR(p[0], 30.0, 1e-9);
  EXPECT_NEAR(p[dt * kRange + rt], 400.0, 1e-9);

  ASSERT_EQ(num_detections(out.idx), 1);
  const int64_t peak = static_cast<int64_t>(
      std::max_element(p, p + kDoppler * kRange) - p);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(peak, rows[0] * kRange + rows[1]);
}

TEST(CFARDetectorOpOsTest, MaxDetectionsCap) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = os_params(1e-3);
  params.max_detections = 1;
  const auto out = run_cfar<double>(input, params);

  EXPECT_EQ(num_detections(out.idx), 1);
}

TEST(CFARDetectorOpOsTest, FloatDoubleAgree) {
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);

  std::vector<float> mapf(map.begin(), map.end());
  auto in_f = to_device_tensor<float>(mapf, {kDoppler, kRange},
                                      real_dtype<float>());
  auto in_d = to_device_tensor<double>(map, {kDoppler, kRange},
                                       real_dtype<double>());

  const auto out_f = run_cfar<float>(in_f, os_params(1e-3));
  const auto out_d = run_cfar<double>(in_d, os_params(1e-3));

  ASSERT_EQ(num_detections(out_f.idx), num_detections(out_d.idx));
  const auto* rf = out_f.idx.as<int64_t>();
  const auto* rd = out_d.idx.as<int64_t>();
  EXPECT_EQ(rf[0], rd[0]);
  EXPECT_EQ(rf[1], rd[1]);
}

TEST(CFARDetectorOpOsTest, DetectsOnLargeMap) {
  constexpr int64_t kLargeDoppler = 256;
  constexpr int64_t kLargeRange = 512;
  const int64_t dt = 128;
  const int64_t rt = 256;
  const auto map =
      power_map(kLargeDoppler, kLargeRange, 1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(
      map, {kLargeDoppler, kLargeRange}, real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
  expect_mask_matches_score_interior<double>(
      out.mask, out.score, kLargeDoppler, kLargeRange);
}

TEST(CFARDetectorOpOsTest, MaskMatchesScoreInterior) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  expect_mask_matches_score_interior<double>(out.mask, out.score,
                                             kDoppler, kRange);
  const auto* scores = out.score.as<double>();
  EXPECT_GT(scores[8 * kRange + 16], 1.0);
  const int64_t far = 4 * kRange + 4;
  EXPECT_LT(scores[far], 1.0);
}

TEST(CFARDetectorOpOsTest, RealAndComplexPathsAgree) {
  const int64_t dt = 10;
  const int64_t rt = 22;
  const double floor = 2.0;
  const double peak = 80.0;

  const auto real_map = power_map(floor, {{dt, rt}}, peak);
  auto real_input = to_device_tensor<double>(
      real_map, {kDoppler, kRange}, real_dtype<double>());

  std::vector<std::complex<double>> complex_map(
      static_cast<size_t>(kDoppler * kRange),
      std::complex<double>(std::sqrt(floor), 0.0));
  complex_map[static_cast<size_t>(dt * kRange + rt)] =
      std::complex<double>(std::sqrt(peak), 0.0);
  auto complex_input = to_device_tensor<std::complex<double>>(
      complex_map, {kDoppler, kRange}, complex_dtype<double>());

  const auto out_real =
      run_cfar<double>(real_input, os_params(1e-3));
  const auto out_complex =
      run_cfar<double>(complex_input, os_params(1e-3));

  ASSERT_EQ(num_detections(out_real.idx),
            num_detections(out_complex.idx));
  EXPECT_TRUE(idx_contains(out_real.idx, dt, rt));
  EXPECT_TRUE(idx_contains(out_complex.idx, dt, rt));
}

TEST(CFARDetectorOpOsTest, BorderCellsUseInBoundsTrainingOnly) {
  const double floor = 4.0;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-6));

  const auto* noise = out.noise.as<double>();
  const auto* scores = out.score.as<double>();
  EXPECT_NEAR(noise[0], floor, 1e-6);
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
  EXPECT_LE(scores[0], 1.0);
  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpOsTest, DetectsTwoTargets) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, 4, 8));
  EXPECT_TRUE(idx_contains(out.idx, 12, 24));
}

TEST(CFARDetectorOpOsTest, DetectionIdxRasterOrder) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  ASSERT_EQ(out.idx.shape.size(), 2u);
  ASSERT_EQ(out.idx.shape[1], 2);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], 4);
  EXPECT_EQ(rows[1], 8);
  EXPECT_EQ(rows[2], 12);
  EXPECT_EQ(rows[3], 24);
}

TEST(CFARDetectorOpOsTest, NoiseEstimateMatchesOrderStatistic) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 50.0;
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, os_params(1e-3));

  const double noise =
      out.noise.as<double>()[dt * kRange + rt];
  EXPECT_GT(noise, 0.0);
  EXPECT_LT(noise, 50.0);
}

TEST(CFARDetectorOpOsTest,
     LiveThresholdScaleOverrideMidStream) {
  GTEST_SKIP() << "[EA HSDK5 ONLY] Graph mid-stream live "
                  "harness deferred (HSDK5_PORT.md §1)";
}

TEST(CFARDetectorOpOsTest, LivePfaOverrideMidStream) {
  GTEST_SKIP() << "[EA HSDK5 ONLY] Graph mid-stream live "
                  "harness deferred (HSDK5_PORT.md §1)";
}

TEST(CFARDetectorOpOsTest,
     LiveMaxDetectionsOverrideMidStream) {
  GTEST_SKIP() << "[EA HSDK5 ONLY] Graph mid-stream live "
                  "harness deferred (HSDK5_PORT.md §1)";
}

TEST(CFARDetectorOpOsTest,
     LiveOsRankOverrideMidStream) {
  GTEST_SKIP() << "[EA HSDK5 ONLY] Graph mid-stream live "
                  "harness deferred (HSDK5_PORT.md §1)";
}

TEST(CFARDetectorOpOsTest, OsRankConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  // Non-uniform background so order statistics differ by rank (flat
  // power makes every rank select the same noise; see LiveOsRank).
  std::vector<double> map(static_cast<size_t>(kDoppler * kRange));
  for (size_t i = 0; i < map.size(); ++i) {
    map[i] = 1.0 + static_cast<double>(i % 17) * 0.03;
  }
  map[static_cast<size_t>(dt * kRange + rt)] = 50.0;
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams low = os_params(1e-3);
  low.blocking_alpha_lut_rebuild = true;
  low.os_rank = 5;
  CFARParams high = os_params(1e-3);
  high.blocking_alpha_lut_rebuild = true;
  high.os_rank = kOsRank;  // 30
  const auto lo = run_cfar<double>(input, low);
  const auto hi = run_cfar<double>(input, high);
  // Ascending OS: higher rank → higher noise estimate.
  EXPECT_LT(lo.noise.as<double>()[dt * kRange + rt],
            hi.noise.as<double>()[dt * kRange + rt]);
  EXPECT_NE(score_at(lo.score, dt, rt), score_at(hi.score, dt, rt));
}

TEST(CFARDetectorOpOsTest, GuardCellsConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  map[static_cast<size_t>(dt * kRange + rt + 2)] = 10000.0;
  auto input = to_device_tensor(map, {kDoppler, kRange},
                                real_dtype<double>());
  CFARParams narrow = os_params(1e-3);
  narrow.blocking_alpha_lut_rebuild = true;
  CFARParams wide = narrow;
  wide.guard_cells = {1, 2};
  const auto n = run_cfar<double>(input, narrow);
  const auto w = run_cfar<double>(input, wide);
  // OS α grows with training count / shrinks when the interferer is
  // excluded by a wider guard — opposite score direction from CA/GO.
  EXPECT_GT(score_at(n.score, dt, rt),
            score_at(w.score, dt, rt));
}

TEST(CFARDetectorOpOsTest, TrainingCellsConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(6 * kRange + rt)] = 10000.0;
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  auto input = to_device_tensor(map, {kDoppler, kRange},
                                real_dtype<double>());
  CFARParams narrow = os_params(1e-3);
  narrow.blocking_alpha_lut_rebuild = true;
  CFARParams wide = os_params(1e-3);
  wide.blocking_alpha_lut_rebuild = true;
  wide.training_cells = {4, 4};
  const auto n = run_cfar<double>(input, narrow);
  const auto w = run_cfar<double>(input, wide);
  // OS: more training cells → higher α / noise → lower CUT score.
  EXPECT_LT(score_at(w.score, dt, rt),
            score_at(n.score, dt, rt));
}

TEST(CFARDetectorOpOsTest, NciIntegratesChannelsForDetection) {
  constexpr int64_t kChannels = 4;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube = power_cube(
      {kChannels, kDoppler, kRange}, 1.0,
      {{{0, dt, rt}, 10.0},
       {{1, dt, rt}, 10.0},
       {{2, dt, rt}, 10.0},
       {{3, dt, rt}, 10.0}});
  auto input = to_device_tensor<double>(
      cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = os_params(1e-3);
  params.nci_axes = {0};
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
}

TEST(CFARDetectorOpOsTest, NciSumsAllChannelPower) {
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0,
                               {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(
      cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = os_params(1e-3);
  params.nci_axes = {0};
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(out.power.shape,
            (std::vector<int64_t>{kDoppler, kRange}));
  const auto* p = out.power.as<double>();
  EXPECT_NEAR(p[8 * kRange + 16], 43.0, 1e-9);
}

TEST(CFARDetectorOpOsTest, NciPreservesBatchAxis) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kChannels = 3;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube = power_cube(
      {kBatch, kChannels, kDoppler, kRange}, 1.0,
      {{{0, 0, dt, rt}, 20.0},
       {{0, 1, dt, rt}, 20.0},
       {{0, 2, dt, rt}, 20.0},
       {{1, 0, dt, rt}, 20.0},
       {{1, 1, dt, rt}, 20.0},
       {{1, 2, dt, rt}, 20.0}});
  auto input = to_device_tensor<double>(
      cube, {kBatch, kChannels, kDoppler, kRange},
      real_dtype<double>());

  CFARParams params = os_params(1e-3);
  params.nci_axes = {1};
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpOsTest, BatchIdxIncludesPreservedAxes) {
  constexpr int64_t kBatch = 2;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube =
      power_cube({kBatch, kDoppler, kRange}, 1.0,
                 {{{0, dt, rt}, 100.0}, {{1, dt, rt}, 100.0}});
  auto input = to_device_tensor<double>(
      cube, {kBatch, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = os_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpOsTest, NciMaskMatchesScoreOnIntegratedMap) {
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0,
                               {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(
      cube, {kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = os_params(1e-3);
  params.nci_axes = {0};
  const auto out = run_cfar<double>(input, params);

  expect_mask_matches_score_interior<double>(out.mask, out.score,
                                             kDoppler, kRange);
}

TEST(CFARDetectorOpOsTest, RejectsMissingGuardCells) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params{};
  params.cfar_variant = CfarVariant::kOS;
  params.os_rank = kOsRank;
  params.training_cells = {2, 2};
  params.pfa = 1e-3;
  const FiveOutGraphRunResult result =
      run_cfar_result<double>(input, params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(CFARDetectorOpOsTest, RejectsInvalidNciAxis) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = os_params(1e-3);
  params.nci_axes = {0};
  const FiveOutGraphRunResult result =
      run_cfar_result<double>(input, params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(CFARDetectorOpOsTest, RejectsOsRankOutOfRange) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = os_params(1e-3);
  params.os_rank = 1000;
  const FiveOutGraphRunResult result =
      run_cfar_result<double>(input, params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(CFARDetectorOpOsTest, OsAlphaInvertsMonteCarloPfa) {
  constexpr int64_t kN = 40;
  constexpr int64_t kRank = 30;
  constexpr double kPfa = 1e-2;
  const double alpha = holoscan::radar::detail::os_cfar_alpha(
      kPfa, kN, kRank, 1, 1, 1, 2, 2);
  const double recovered =
      holoscan::radar::detail::os_cfar_pfa_from_alpha(
          alpha, kN, kRank, 1);
  EXPECT_NEAR(recovered, kPfa, 0.02);
}

TEST(CFARDetectorOpOsTest,
     OsAlphaDependsOnTrainingCountForKGreaterThan1) {
  constexpr double kPfa = 1e-2;
  constexpr int64_t kRank = 5;
  constexpr int64_t kFold = 2;
  const double alpha_small =
      holoscan::radar::detail::os_cfar_alpha(
          kPfa, 8, kRank, kFold, 1, 1, 2, 2);
  const double alpha_large =
      holoscan::radar::detail::os_cfar_alpha(
          kPfa, 40, kRank, kFold, 1, 1, 2, 2);
  ASSERT_GT(alpha_large, 0.0);
  EXPECT_GT(std::abs(alpha_small - alpha_large) / alpha_large,
            0.05)
      << "alpha_small=" << alpha_small
      << " alpha_large=" << alpha_large;
}

TEST(CFARDetectorOpOsTest, PeakOnlyOffMatchesRawMask) {
  const auto map =
      power_map(1.0, {{8, 15}, {8, 16}, {8, 17}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto raw = run_cfar<double>(input, os_params(1e-2));
  CFARParams off = os_params(1e-2);
  off.peak_only = false;
  const auto out = run_cfar<double>(input, off);

  EXPECT_EQ(num_detections(raw.idx), 3);
  EXPECT_EQ(num_detections(out.idx), num_detections(raw.idx));
  for (const auto r : {15, 16, 17}) {
    EXPECT_TRUE(idx_contains(out.idx, 8, r));
  }
}

TEST(CFARDetectorOpOsTest, PeakOnlyKeepsSingleMax) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(8 * kRange + 17)] = 50.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = os_params(1e-2);
  params.peak_only = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, 8, 16));
  EXPECT_FALSE(idx_contains(out.idx, 8, 17));
  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[8 * kRange + 16], 1);
  EXPECT_EQ(mask[8 * kRange + 17], 0);
}

TEST(CFARDetectorOpOsTest,
     PeakOnlyCollapsesStripRasterTiebreak) {
  const auto map =
      power_map(1.0, {{8, 15}, {8, 16}, {8, 17}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = os_params(1e-2);
  params.peak_only = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, 8, 15));
}

TEST(CFARDetectorOpOsTest,
     PeakOnlyNeighborhoodCross4VsSquare8) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(9 * kRange + 17)] = 90.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams cross = os_params(1e-2);
  cross.peak_only = true;
  cross.peak_neighborhood = CfarPeakNeighborhood::kCross4;
  CFARParams square = cross;
  square.peak_neighborhood = CfarPeakNeighborhood::kSquare8;

  const auto out_cross = run_cfar<double>(input, cross);
  const auto out_square = run_cfar<double>(input, square);

  EXPECT_EQ(num_detections(out_cross.idx), 2);
  EXPECT_EQ(num_detections(out_square.idx), 1);
  EXPECT_TRUE(idx_contains(out_square.idx, 8, 16));
}

TEST(CFARDetectorOpOsTest,
     ThresholdScaleConfigurableFromParams) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams detect = os_params(1e-3);
  detect.threshold_scale = 1.0;
  CFARParams suppress = os_params(1e-3);
  suppress.threshold_scale = 1e4;
  EXPECT_EQ(
      num_detections(run_cfar<double>(input, detect).idx), 1);
  EXPECT_EQ(
      num_detections(run_cfar<double>(input, suppress).idx), 0);
}

TEST(CFARDetectorOpOsTest,
     MaxDetectionsConfigurableFromParams) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = os_params(1e-3);
  params.max_detections = 1;
  EXPECT_EQ(
      num_detections(run_cfar<double>(input, params).idx), 1);
}

TEST(CFARDetectorOpOsTest, PfaConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 26.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams loose = os_params(1e-2);
  loose.blocking_alpha_lut_rebuild = true;
  CFARParams tight = os_params(1e-8);
  tight.blocking_alpha_lut_rebuild = true;
  EXPECT_GT(
      score_at(run_cfar<double>(input, loose).score, dt, rt),
      score_at(run_cfar<double>(input, tight).score, dt, rt));
}
