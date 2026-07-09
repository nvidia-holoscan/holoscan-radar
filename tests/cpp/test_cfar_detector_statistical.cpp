// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Monte-Carlo false-alarm calibration tests for CFARDetectorOp (TESTING.md
// statistical requirements). Labeled "statistical" — excluded from per-commit
// CI (`ctest -L unit`); run nightly via `ctest -L statistical`.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <tuple>
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
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::test::create_operator_test;

constexpr int64_t kGuard = 1;
constexpr int64_t kTrain = 2;
constexpr int64_t kMargin = kGuard + kTrain;

struct PfaTrialResult {
  int64_t interior_cells{0};
  int64_t false_alarms{0};
};

struct CfarOutputs {
  HostTensor score;
};

template <typename T, typename... Args>
CfarOutputs run_cfar_score(std::shared_ptr<holoscan::Tensor> input,
                           Args&&... args) {
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

  EXPECT_EQ(score_cap.frames.size(), 1u);
  return {std::move(score_cap.frames.front())};
}

template <typename T, typename... Args>
std::vector<HostTensor> run_cfar_scores(std::vector<holoscan::TensorMap> frames,
                                        Args&&... args) {
  auto harness = create_operator_test<CFARDetectorOp<T>>(
      std::forward<Args>(args)...);
  harness->template add_input_port<holoscan::TensorMap>("rd_maps",
                                                        std::move(frames));

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

  EXPECT_EQ(mask_cap.frames.size(), score_cap.frames.size());
  EXPECT_EQ(noise_cap.frames.size(), score_cap.frames.size());
  EXPECT_EQ(idx_cap.frames.size(), score_cap.frames.size());
  return std::move(score_cap.frames);
}

auto ca_args(double pfa) {
  return std::make_tuple(
      holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{kGuard, kGuard},
      holoscan::Arg("training_cells") = std::vector<int64_t>{kTrain, kTrain},
      holoscan::Arg("pfa") = pfa);
}

auto go_args(double pfa) {
  return std::make_tuple(
      holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{kGuard, kGuard},
      holoscan::Arg("training_cells") = std::vector<int64_t>{kTrain, kTrain},
      holoscan::Arg("pfa") = pfa);
}

std::vector<double> exponential_power_map(int64_t num_doppler,
                                          int64_t num_range, double mean,
                                          std::mt19937_64& rng) {
  std::exponential_distribution<double> dist(1.0 / mean);
  std::vector<double> map(static_cast<size_t>(num_doppler * num_range));
  for (double& cell : map) {
    cell = dist(rng);
  }
  return map;
}

std::vector<double> exponential_power_cube(
    const std::vector<int64_t>& shape, double mean, std::mt19937_64& rng) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  std::exponential_distribution<double> dist(1.0 / mean);
  std::vector<double> data(static_cast<size_t>(count));
  for (double& cell : data) {
    cell = dist(rng);
  }
  return data;
}

template <typename T>
PfaTrialResult count_interior_false_alarms(const HostTensor& score,
                                           int64_t num_doppler,
                                           int64_t num_range) {
  PfaTrialResult result;
  const auto* scores = score.as<T>();
  for (int64_t d = kMargin; d < num_doppler - kMargin; ++d) {
    for (int64_t r = kMargin; r < num_range - kMargin; ++r) {
      ++result.interior_cells;
      if (scores[d * num_range + r] > T{1}) {
        ++result.false_alarms;
      }
    }
  }
  return result;
}

void expect_realized_pfa_near_configured(double realized, double configured,
                                         int64_t total_cells) {
  ASSERT_GT(total_cells, 0);
  const double expected = configured * static_cast<double>(total_cells);
  ASSERT_GE(expected, 50.0)
      << "Monte Carlo design too weak: expected false alarms " << expected
      << " < 50";

  const double ratio = realized / configured;
  // ~4-sigma envelope on a binomial proportion (conservative for correlated
  // neighboring CUTs sharing training cells).
  const double sigma =
      std::sqrt(configured * (1.0 - configured) /
                static_cast<double>(total_cells));
  const double tol = std::max(4.0 * sigma / configured, 0.15);
  EXPECT_NEAR(ratio, 1.0, tol)
      << "realized_pfa=" << realized << " configured_pfa=" << configured
      << " total_interior_cells=" << total_cells;
}

PfaTrialResult run_monte_carlo_pfa_k1(int num_trials, int64_t num_doppler,
                                      int64_t num_range, double configured_pfa,
                                      uint64_t seed, CfarVariant variant) {
  PfaTrialResult aggregate;
  std::mt19937_64 rng(seed);
  const auto args =
      (variant == CfarVariant::kGO) ? go_args(configured_pfa)
                                    : ca_args(configured_pfa);
  std::vector<holoscan::TensorMap> frames;
  frames.reserve(static_cast<size_t>(num_trials));
  for (int trial = 0; trial < num_trials; ++trial) {
    const auto map =
        exponential_power_map(num_doppler, num_range, 1.0, rng);
    auto input = to_device_tensor<double>(map, {num_doppler, num_range},
                                          real_dtype<double>());
    frames.push_back(single_map(std::move(input), "rd_maps"));
  }
  const auto scores = std::apply(
      [&](auto&&... a) {
        return run_cfar_scores<double>(std::move(frames),
                                       std::forward<decltype(a)>(a)...);
      },
      args);
  for (const HostTensor& score : scores) {
    const PfaTrialResult trial_result =
        count_interior_false_alarms<double>(score, num_doppler, num_range);
    aggregate.interior_cells += trial_result.interior_cells;
    aggregate.false_alarms += trial_result.false_alarms;
  }
  return aggregate;
}

PfaTrialResult run_monte_carlo_pfa_nci(int num_trials, int64_t num_channels,
                                       int64_t num_doppler, int64_t num_range,
                                       double configured_pfa, uint64_t seed,
                                       CfarVariant variant) {
  PfaTrialResult aggregate;
  std::mt19937_64 rng(seed);
  std::vector<holoscan::TensorMap> frames;
  frames.reserve(static_cast<size_t>(num_trials));
  for (int trial = 0; trial < num_trials; ++trial) {
    const auto cube = exponential_power_cube(
        {num_channels, num_doppler, num_range}, 1.0, rng);
    auto input = to_device_tensor<double>(
        cube, {num_channels, num_doppler, num_range}, real_dtype<double>());
    frames.push_back(single_map(std::move(input), "rd_maps"));
  }
  const auto scores = run_cfar_scores<double>(
      std::move(frames), holoscan::Arg("cfar_variant") = variant,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{kGuard, kGuard},
      holoscan::Arg("training_cells") = std::vector<int64_t>{kTrain, kTrain},
      holoscan::Arg("pfa") = configured_pfa,
      holoscan::Arg("nci_axes") = std::vector<int64_t>{0});
  for (const HostTensor& score : scores) {
    const PfaTrialResult trial_result =
        count_interior_false_alarms<double>(score, num_doppler, num_range);
    aggregate.interior_cells += trial_result.interior_cells;
    aggregate.false_alarms += trial_result.false_alarms;
  }
  return aggregate;
}

}  // namespace

TEST(CFARDetectorOpStatisticalTest, CaMonteCarloPfaK1) {
  RADAR_TEST_SKIP_IF_NO_GPU();

  constexpr int kTrials = 30;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_k1(
      kTrials, kMapDoppler, kMapRange, kPfa, 0xCAFEBABE0001ULL,
      CfarVariant::kCA);
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}

TEST(CFARDetectorOpStatisticalTest, GoMonteCarloPfaK1) {
  RADAR_TEST_SKIP_IF_NO_GPU();

  constexpr int kTrials = 30;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_k1(
      kTrials, kMapDoppler, kMapRange, kPfa, 0x474F434641520001ULL,
      CfarVariant::kGO);
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}

TEST(CFARDetectorOpStatisticalTest, CaMonteCarloPfaNciK8) {
  RADAR_TEST_SKIP_IF_NO_GPU();

  constexpr int kTrials = 30;
  constexpr int64_t kChannels = 8;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_nci(
      kTrials, kChannels, kMapDoppler, kMapRange, kPfa, 0xCAFEBABE0002ULL,
      CfarVariant::kCA);
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}

TEST(CFARDetectorOpStatisticalTest, GoMonteCarloPfaNciK8) {
  RADAR_TEST_SKIP_IF_NO_GPU();

  constexpr int kTrials = 30;
  constexpr int64_t kChannels = 8;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_nci(
      kTrials, kChannels, kMapDoppler, kMapRange, kPfa, 0x474F434641520002ULL,
      CfarVariant::kGO);
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}
