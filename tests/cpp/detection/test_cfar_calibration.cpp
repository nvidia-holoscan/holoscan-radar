// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_calibration.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"

namespace {
namespace radar = holoscan::radar;
namespace detail = radar::detail;
radar::CFARParams params(radar::CfarVariant variant = radar::CfarVariant::kGO) {
  radar::CFARParams p;
  p.cfar_variant = variant;
  p.guard_cells = {1, 1};
  p.training_cells = {1, 1};
  p.pfa = 0.01;
  return p;
}

TEST(CfarCalibration, RarePfaBudgetsAreNotSilentlyTruncated) {
  auto b = radar::cfar_calibration_budget(1e-9);
  EXPECT_GT(b.trials, 380000000000ULL);
  EXPECT_LT(b.trials, 430000000000ULL);
  EXPECT_GE(b.coverage, 0.95);
  EXPECT_GE(b.tail_rank, 380U);
  EXPECT_LT(b.tail_rank, 430U);
  EXPECT_GT(radar::cfar_calibration_budget(1e-9, 0.05).trials, b.trials * 3);
  EXPECT_GT(radar::cfar_calibration_budget(1e-9, 0.1, 0.99).trials, b.trials);
  EXPECT_GT(radar::cfar_calibration_budget(1e-12).trials, b.trials * 900);
  EXPECT_THROW(radar::cfar_calibration_budget(1e-30), std::invalid_argument);
  for (double p : {0.0, 1.0, -1.0, std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(radar::cfar_calibration_budget(p), std::invalid_argument);
  EXPECT_THROW(radar::cfar_calibration_budget(0.01, 0.0), std::invalid_argument);
  EXPECT_THROW(radar::cfar_calibration_budget(0.01, 0.1, 1.0), std::invalid_argument);
}

TEST(CfarCalibration, ExplicitTrialsAreExactAndReportActualCoverage) {
  for (std::uint64_t trials : {1ULL, 17ULL, 4097ULL, 1000000ULL}) {
    const auto b = radar::cfar_calibration_budget(1e-9, 0.1, 0.95, trials);
    EXPECT_EQ(b.trials, trials);
    EXPECT_EQ(b.tail_rank, 1U);
    // Rank 1 selects the sample maximum; independently evaluate its coverage.
    const auto expected = std::pow(1 - 0.9e-9L, trials) - std::pow(1 - 1.1e-9L, trials);
    EXPECT_NEAR(b.coverage, expected, 1e-12);
    EXPECT_LT(b.coverage, 0.95);
  }
  const auto automatic = radar::cfar_calibration_budget(0.01);
  const auto fixed = radar::cfar_calibration_budget(0.01, 0.1, 0.95, automatic.trials);
  EXPECT_EQ(automatic.trials, fixed.trials);
  EXPECT_EQ(automatic.tail_rank, fixed.tail_rank);
  EXPECT_EQ(automatic.coverage, fixed.coverage);
  // An explicit budget also bypasses an otherwise unrepresentable automatic budget.
  EXPECT_EQ(radar::cfar_calibration_budget(1e-30, 0.1, 0.95, 17).trials, 17U);
}

TEST(CfarCalibration, TrialOverridePropagatesThroughRuntimeCache) {
  auto p = params();
  p.pfa = 1e-9;
  p.calibration_trials = 257;
  auto plan = detail::prepare_cfar_calibration(p, {8, 8});
  EXPECT_EQ(plan.budget.trials, 257U);
  EXPECT_EQ(plan.total_trials, 257U * plan.geometries.size());
  radar::CfarAlphaLutKey key{radar::CfarVariant::kGO, p.pfa, 1, -1, 1, 1, 1, 1, 8, 8};
  key.calibration_trials = p.calibration_trials;
  EXPECT_EQ(detail::get_cached_alpha_lut(key), radar::calibrate_cfar(p, {8, 8}).alpha_lut);
  auto changed = key;
  changed.calibration_trials *= 2;
  EXPECT_NE(changed, key);
  EXPECT_FALSE(detail::poll_cached_alpha_lut(changed).ready);
  p.calibration_trials = changed.calibration_trials;
  EXPECT_EQ(detail::get_cached_alpha_lut(changed), radar::calibrate_cfar(p, {8, 8}).alpha_lut);
  p.calibration_trials = -1;
  EXPECT_THROW(detail::prepare_cfar_calibration(p, {8, 8}), std::invalid_argument);
}

TEST(CfarCalibration, RareBudgetCoverageAgreesWithIndependentBinomialSum) {
  const auto b = radar::cfar_calibration_budget(1e-9);
  auto cdf = [&](long double q) {
    long double term = std::exp(static_cast<long double>(b.trials) * std::log1p(-q));
    long double sum = term;
    for (std::uint64_t j = 0; j + 1 < b.tail_rank; ++j) {
      term *= static_cast<long double>(b.trials - j) / (j + 1) * q / (1 - q);
      sum += term;
    }
    return sum;
  };
  const long double independent = cdf(0.9e-9L) - cdf(1.1e-9L);
  EXPECT_NEAR(b.coverage, static_cast<double>(independent), 1e-10);
  EXPECT_GE(independent, 0.95L);
}

TEST(CfarCalibration, PlanResolvesLooksAndEnumeratesEveryClippedSideGeometry) {
  auto p = params();
  p.nci_axes = {0};
  p.preintegrated_looks = 2;
  p.pfa = 1e-9;
  const auto plan = detail::prepare_cfar_calibration(p, {4, 16, 16});
  EXPECT_EQ(plan.effective_looks, 8);
  EXPECT_GT(plan.total_trials, plan.budget.trials);
  EXPECT_EQ(plan.total_trials, plan.budget.trials * plan.geometries.size());
  for (int d = 0; d < 16; ++d)
    for (int r = 0; r < 16; ++r) {
      std::int64_t rn, rp, dn, dp;
      detail::go_training_side_counts(d, r, 16, 16, 1, 1, 1, 1, &rn, &rp, &dn, &dp);
      bool found = false;
      for (auto g : plan.geometries) {
        std::int64_t a, b, c, e;
        detail::go_training_side_counts(g.doppler, g.range, 16, 16, 1, 1, 1, 1, &a, &b, &c, &e);
        found |= (rn == a && rp == b && dn == c && dp == e);
      }
      EXPECT_TRUE(found);
    }
  p.nci_axes = {0, 0};
  EXPECT_THROW(detail::prepare_cfar_calibration(p, {4, 16, 16}), std::invalid_argument);
}

TEST(CfarCalibration, AnalyticAndExternalTablesBypassMonteCarlo) {
  auto p = params(radar::CfarVariant::kCA);
  p.calibration_trials = 1;  // Ignored by analytic and explicit-table paths.
  auto result = radar::calibrate_cfar(p, {16, 16});
  EXPECT_EQ(result.plan.method, "analytic");
  EXPECT_EQ(result.plan.total_trials, 0U);
  EXPECT_EQ(result.alpha_lut, detail::build_ca_alpha_lut(p.pfa, 1, 16, 16, 1, 1, 1, 1));
  p.alpha_lut = result.alpha_lut;
  p.pfa = 1e-9;
  auto external = radar::calibrate_cfar(p, {16, 16});
  EXPECT_EQ(external.plan.method, "external");
  EXPECT_EQ(external.plan.total_trials, 0U);
  EXPECT_EQ(external.alpha_lut, result.alpha_lut);
  p.alpha_lut[1] = 0;
  EXPECT_THROW(radar::calibrate_cfar(p, {16, 16}), std::invalid_argument);
}

TEST(CfarCalibration, NoTrainingCellsNeedNoCalibration) {
  for (auto variant : {radar::CfarVariant::kCA, radar::CfarVariant::kGO, radar::CfarVariant::kSO}) {
    auto p = params(variant);
    p.training_cells = {0, 0};
    const auto result = radar::calibrate_cfar(p, {16, 16});
    EXPECT_EQ(result.plan.method, "empty");
    EXPECT_EQ(result.plan.total_trials, 0U);
    EXPECT_EQ(result.alpha_lut, (std::vector<double>{0.0}));
  }
}

TEST(CfarCalibration, StreamedTailIsIndependentOfBatchBoundaries) {
  auto p = params();
  p.pfa = 0.1;
  auto plan = detail::prepare_cfar_calibration(p, {16, 16});
  const auto g = plan.geometries.back();
  const auto one = detail::calibrate_cfar_geometry_gpu(plan, g, {}, {}, false, 65536);
  const auto many = detail::calibrate_cfar_geometry_gpu(plan, g, {}, {}, false, 257);
  EXPECT_EQ(one, many);
  EXPECT_TRUE(std::isfinite(one));
  EXPECT_GT(one, 0);
}

// Independent host noise draws validate the calibrated distribution rather
// than reusing the GPU's PRNG or its quantile-selection implementation.
double realized_pfa(radar::CfarVariant variant, int looks, double alpha) {
  std::mt19937_64 rng(123456);
  std::gamma_distribution<double> gamma(looks, 1.0);
  std::uint64_t alarms = 0;
  constexpr int trials = 70000;
  for (int i = 0; i < trials; ++i) {
    double rn = 0, rp = 0, dn = 0, dp = 0;
    int nrn = 0, nrp = 0, ndn = 0, ndp = 0;
    std::vector<double> cells;
    for (int dd = -2; dd <= 2; ++dd)
      for (int rr = -2; rr <= 2; ++rr) {
        if (std::abs(dd) <= 1 && std::abs(rr) <= 1)
          continue;
        const auto v = gamma(rng);
        cells.push_back(v);
        if (rr < 0) {
          rn += v;
          ++nrn;
        } else if (rr > 0) {
          rp += v;
          ++nrp;
        }
        if (dd < 0) {
          dn += v;
          ++ndn;
        } else if (dd > 0) {
          dp += v;
          ++ndp;
        }
      }
    double noise = 0;
    if (variant == radar::CfarVariant::kOS) {
      std::nth_element(cells.begin(), cells.begin() + 10, cells.end());
      noise = cells[10];
    } else if (variant == radar::CfarVariant::kGO) {
      noise = std::max({rn / nrn, rp / nrp, dn / ndn, dp / ndp});
    } else {
      noise = std::min({rn / nrn, rp / nrp, dn / ndn, dp / ndp});
    }
    alarms += gamma(rng) > alpha * noise;
  }
  return static_cast<double>(alarms) / trials;
}
TEST(CfarCalibration, IndependentNoiseVerifiesGoSoAndMultiLookOs) {
  for (auto variant : {radar::CfarVariant::kGO, radar::CfarVariant::kSO, radar::CfarVariant::kOS}) {
    auto p = params(variant);
    p.os_rank = 10;
    if (variant == radar::CfarVariant::kOS)
      p.preintegrated_looks = 4;
    const auto result = radar::calibrate_cfar(p, {16, 16});
    std::int64_t rn, rp, dn, dp;
    detail::go_training_side_counts(8, 8, 16, 16, 1, 1, 1, 1, &rn, &rp, &dn, &dp);
    const auto index =
        variant == radar::CfarVariant::kOS ? 16 : detail::go_lut_index(rn, rp, dn, dp);
    EXPECT_NEAR(realized_pfa(variant, p.preintegrated_looks, result.alpha_lut[index]), p.pfa,
                0.003);
    EXPECT_EQ(result.geometry_alphas.size(), result.plan.geometries.size());
    EXPECT_FALSE(result.started_at.empty());
    EXPECT_FALSE(result.finished_at.empty());
  }
}

TEST(CfarCalibration, CancellationInterruptsRarePfaAfterOneBatch) {
  auto p = params();
  p.pfa = 1e-9;
  std::uint64_t completed = 0;
  EXPECT_THROW(radar::calibrate_cfar(p, {16, 16},
                                     [&](const auto& event) {
                                       if (event.status == "running") {
                                         completed = event.completed_trials;
                                         throw radar::CfarCalibrationCancelled{};
                                       }
                                     }),
               radar::CfarCalibrationCancelled);
  EXPECT_EQ(completed, 65536U);
  std::stop_source source;
  source.request_stop();
  EXPECT_THROW(radar::calibrate_cfar(p, {16, 16}, {}, source.get_token()),
               radar::CfarCalibrationCancelled);
}

TEST(CfarCalibration, SharedBuildSurvivesOneConsumerAndCancelsAfterLast) {
  radar::CfarAlphaLutKey key{radar::CfarVariant::kGO, 1e-9, 1, 0, 1, 1, 1, 1, 16, 16};
  int a = 0, b = 0;
  detail::ensure_alpha_lut_build(key, &a);
  detail::ensure_alpha_lut_build(key, &b);
  detail::release_alpha_lut_builds(&a);
  EXPECT_FALSE(detail::poll_cached_alpha_lut(key).ready);
  const auto start = std::chrono::steady_clock::now();
  detail::release_alpha_lut_builds(&b);
  EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), 2.0);
  EXPECT_FALSE(detail::poll_cached_alpha_lut(key).ready);
  auto changed = key;
  changed.relative_error = 0.2;
  EXPECT_NE(changed, key);
  changed = key;
  changed.confidence = 0.99;
  EXPECT_NE(changed, key);
}
TEST(CfarCalibration, FailedCalibrationDoesNotPublishACachedTable) {
  radar::CfarAlphaLutKey key{radar::CfarVariant::kGO, 1e-40, 1, 0, 1, 1, 1, 1, 16, 16};
  EXPECT_THROW(detail::get_cached_alpha_lut(key), std::invalid_argument);
  EXPECT_FALSE(detail::poll_cached_alpha_lut(key).ready);
}

}  // namespace
