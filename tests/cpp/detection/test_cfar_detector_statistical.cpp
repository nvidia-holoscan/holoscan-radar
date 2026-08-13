// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Monte-Carlo false-alarm calibration tests for CFARDetectorOp (TESTING.md
// statistical requirements). Labeled "statistical" — excluded from per-commit
// CI (`ctest -L unit`); run nightly via `ctest -L statistical`.
//
// Uses the HSDK 5 Graph harness: builds a vector of noise-map tensors, passes
// them to run_single_in_five_out_operator_graph_result (one graph run
// processes all frames), and collects detection_score tensors from frames1.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::CFARDetectorOp;
using holoscan::radar::CFARParams;
using holoscan::radar::CfarVariant;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::FiveOutGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_in_five_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kGuard = 1;
constexpr int64_t kTrain = 2;
constexpr int64_t kMargin = kGuard + kTrain;

struct PfaTrialResult {
  int64_t interior_cells{0};
  int64_t false_alarms{0};
};

CFARParams ca_params(double pfa) {
  CFARParams p{};
  p.cfar_variant = CfarVariant::kCA;
  p.guard_cells = {kGuard, kGuard};
  p.training_cells = {kTrain, kTrain};
  p.pfa = pfa;
  return p;
}

CFARParams go_params(double pfa) {
  CFARParams p{};
  p.cfar_variant = CfarVariant::kGO;
  p.guard_cells = {kGuard, kGuard};
  p.training_cells = {kTrain, kTrain};
  p.pfa = pfa;
  return p;
}

CFARParams so_params(double pfa) {
  CFARParams p{};
  p.cfar_variant = CfarVariant::kSO;
  p.guard_cells = {kGuard, kGuard};
  p.training_cells = {kTrain, kTrain};
  p.pfa = pfa;
  return p;
}

CFARParams os_params(double pfa) {
  CFARParams p{};
  p.cfar_variant = CfarVariant::kOS;
  p.os_rank = 30;
  p.guard_cells = {kGuard, kGuard};
  p.training_cells = {kTrain, kTrain};
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
                                         int64_t total_cells,
                                         double min_ratio_tol = 0.15) {
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
  const double tol = std::max(4.0 * sigma / configured, min_ratio_tol);
  EXPECT_NEAR(ratio, 1.0, tol)
      << "realized_pfa=" << realized << " configured_pfa=" << configured
      << " total_interior_cells=" << total_cells;
}

std::vector<HostTensor> run_cfar_scores(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs,
    CFARParams params) {
  FiveOutGraphRunResult result =
      run_single_in_five_out_operator_graph_result<CFARDetectorOp<double>>(
          std::move(inputs), std::move(params), "detection_mask",
          "detection_score", "noise_estimate", "detection_idx", "cfar_power",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1, const auto& sink2,
             const auto& sink3, const auto& sink4) {
            wire_cfar(graph, source, op, sink0, sink1, sink2, sink3, sink4);
          });
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
  return std::move(result.frames1);
}

PfaTrialResult run_monte_carlo_pfa_k1(int num_trials, int64_t num_doppler,
                                      int64_t num_range, double configured_pfa,
                                      uint64_t seed, CFARParams params) {
  PfaTrialResult aggregate;
  std::mt19937_64 rng(seed);
  std::vector<std::shared_ptr<holoscan::Tensor>> inputs;
  inputs.reserve(static_cast<size_t>(num_trials));
  for (int trial = 0; trial < num_trials; ++trial) {
    const auto map =
        exponential_power_map(num_doppler, num_range, 1.0, rng);
    inputs.push_back(to_device_tensor<double>(map, {num_doppler, num_range},
                                              real_dtype<double>()));
  }
  const std::vector<HostTensor> scores =
      run_cfar_scores(std::move(inputs), std::move(params));
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
                                       CFARParams params) {
  PfaTrialResult aggregate;
  std::mt19937_64 rng(seed);
  params.nci_axes = {0};
  std::vector<std::shared_ptr<holoscan::Tensor>> inputs;
  inputs.reserve(static_cast<size_t>(num_trials));
  for (int trial = 0; trial < num_trials; ++trial) {
    const auto cube = exponential_power_cube(
        {num_channels, num_doppler, num_range}, 1.0, rng);
    inputs.push_back(to_device_tensor<double>(
        cube, {num_channels, num_doppler, num_range}, real_dtype<double>()));
  }
  const std::vector<HostTensor> scores =
      run_cfar_scores(std::move(inputs), std::move(params));
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
  constexpr int kTrials = 30;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_k1(
      kTrials, kMapDoppler, kMapRange, kPfa, 0xCAFEBABE0001ULL,
      ca_params(kPfa));
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}

TEST(CFARDetectorOpStatisticalTest, GoMonteCarloPfaK1) {
  constexpr int kTrials = 30;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_k1(
      kTrials, kMapDoppler, kMapRange, kPfa, 0x474F434641520001ULL,
      go_params(kPfa));
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}

TEST(CFARDetectorOpStatisticalTest, CaMonteCarloPfaNciK8) {
  constexpr int kTrials = 30;
  constexpr int64_t kChannels = 8;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_nci(
      kTrials, kChannels, kMapDoppler, kMapRange, kPfa, 0xCAFEBABE0002ULL,
      ca_params(kPfa));
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}

TEST(CFARDetectorOpStatisticalTest, GoMonteCarloPfaNciK8) {
  constexpr int kTrials = 30;
  constexpr int64_t kChannels = 8;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_nci(
      kTrials, kChannels, kMapDoppler, kMapRange, kPfa, 0x474F434641520002ULL,
      go_params(kPfa));
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}

TEST(CFARDetectorOpStatisticalTest, SoMonteCarloPfaK1) {
  constexpr int kTrials = 30;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_k1(
      kTrials, kMapDoppler, kMapRange, kPfa, 0x534F434641520001ULL,
      so_params(kPfa));
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  // SO min-of-side noise is more sensitive to shared training-cell correlation
  // across neighboring CUTs than GO/CA; relax the ratio envelope accordingly.
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells,
                                      0.25);
}

TEST(CFARDetectorOpStatisticalTest, SoMonteCarloPfaNciK8) {
  constexpr int kTrials = 30;
  constexpr int64_t kChannels = 8;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_nci(
      kTrials, kChannels, kMapDoppler, kMapRange, kPfa, 0x534F434641520002ULL,
      so_params(kPfa));
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells,
                                      0.25);
}

TEST(CFARDetectorOpStatisticalTest, OsMonteCarloPfaK1) {
  constexpr int kTrials = 30;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_k1(
      kTrials, kMapDoppler, kMapRange, kPfa, 0x4F53434641520001ULL,
      os_params(kPfa));
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}

TEST(CFARDetectorOpStatisticalTest, OsMonteCarloPfaNciK8) {
  constexpr int kTrials = 30;
  constexpr int64_t kChannels = 8;
  constexpr int64_t kMapDoppler = 64;
  constexpr int64_t kMapRange = 64;
  constexpr double kPfa = 1e-2;

  const PfaTrialResult result = run_monte_carlo_pfa_nci(
      kTrials, kChannels, kMapDoppler, kMapRange, kPfa, 0x4F53434641520002ULL,
      os_params(kPfa));
  ASSERT_GT(result.interior_cells, 0);
  const double realized =
      static_cast<double>(result.false_alarms) /
      static_cast<double>(result.interior_cells);
  expect_realized_pfa_near_configured(realized, kPfa, result.interior_cells);
}
