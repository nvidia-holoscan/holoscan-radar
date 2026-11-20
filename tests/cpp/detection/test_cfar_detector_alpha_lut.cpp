// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// alpha_lut fast-path contract tests for CFARDetectorOp on the HSDK 5
// Graph harness.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
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
using holoscan::radar::CfarVariant;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kDoppler = 16;
constexpr int64_t kRange = 32;
constexpr int kGuardD = 1;
constexpr int kGuardR = 1;
constexpr int kTrainD = 2;
constexpr int kTrainR = 2;

struct CfarOutputs {
  HostTensor mask;
  HostTensor score;
  HostTensor noise;
  HostTensor idx;
  HostTensor power;
};

std::vector<double> power_map(double floor, const std::vector<std::pair<int64_t, int64_t>>& targets,
                              double peak) {
  std::vector<double> map(static_cast<size_t>(kDoppler * kRange), floor);
  for (const auto& [d, r] : targets) {
    map[static_cast<size_t>(d * kRange + r)] = peak;
  }
  return map;
}

int64_t num_detections(const HostTensor& idx) {
  return idx.shape.empty() ? 0 : idx.shape[0];
}

CFARParams base_params(CfarVariant variant, double pfa) {
  CFARParams p{};
  p.cfar_variant = variant;
  p.guard_cells = {kGuardD, kGuardR};
  p.training_cells = {kTrainD, kTrainR};
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

std::vector<double> build_variant_lut(CfarVariant variant, double pfa, int64_t fold,
                                      int64_t os_rank) {
  switch (variant) {
    case CfarVariant::kCA:
      return holoscan::radar::detail::build_ca_alpha_lut(pfa, fold, static_cast<int>(kDoppler),
                                                         static_cast<int>(kRange), kGuardD, kGuardR,
                                                         kTrainD, kTrainR);
    case CfarVariant::kGO:
      return holoscan::radar::detail::build_go_alpha_lut(pfa, fold, static_cast<int>(kDoppler),
                                                         static_cast<int>(kRange), kGuardD, kGuardR,
                                                         kTrainD, kTrainR);
    case CfarVariant::kSO:
      return holoscan::radar::detail::build_so_alpha_lut(pfa, fold, static_cast<int>(kDoppler),
                                                         static_cast<int>(kRange), kGuardD, kGuardR,
                                                         kTrainD, kTrainR);
    case CfarVariant::kOS:
      return holoscan::radar::detail::build_os_alpha_lut(
          pfa, fold, os_rank, static_cast<int>(kDoppler), static_cast<int>(kRange), kGuardD,
          kGuardR, kTrainD, kTrainR);
    default:
      return {};
  }
}

std::vector<double> inflated_lut(const std::vector<double>& lut, double scale) {
  std::vector<double> out = lut;
  for (size_t i = 1; i < out.size(); ++i) {
    out[i] *= scale;
  }
  return out;
}

std::vector<int64_t> reachable_lut_indices(CfarVariant variant) {
  if (variant == CfarVariant::kGO || variant == CfarVariant::kSO) {
    return holoscan::radar::detail::reachable_go_lut_indices(
        static_cast<int>(kDoppler), static_cast<int>(kRange), kGuardD, kGuardR, kTrainD, kTrainR);
  }
  return holoscan::radar::detail::reachable_training_cell_counts(
      static_cast<int>(kDoppler), static_cast<int>(kRange), kGuardD, kGuardR, kTrainD, kTrainR);
}

std::vector<double> sparse_external_lut(CfarVariant variant) {
  const int64_t max_index = variant == CfarVariant::kOS
                                ? holoscan::radar::detail::max_training_cell_count(
                                      static_cast<int>(kDoppler), static_cast<int>(kRange), kGuardD,
                                      kGuardR, kTrainD, kTrainR)
                                : holoscan::radar::detail::max_go_lut_index(
                                      static_cast<int>(kDoppler), static_cast<int>(kRange), kGuardD,
                                      kGuardR, kTrainD, kTrainR);
  std::vector<double> lut(static_cast<size_t>(max_index + 1), 0.0);
  for (const int64_t idx : reachable_lut_indices(variant)) {
    lut[static_cast<size_t>(idx)] = 1.0;
  }
  return lut;
}

TEST(CFARDetectorAlphaLutTest, AlphaLutPrecedenceOverPfaCA) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto pfa_lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  CFARParams with_pfa = base_params(CfarVariant::kCA, 1e-3);
  with_pfa.blocking_alpha_lut_rebuild = true;
  EXPECT_GE(num_detections(run_cfar<double>(input, with_pfa).idx), 1);

  CFARParams with_lut = base_params(CfarVariant::kCA, 1e-3);
  with_lut.alpha_lut = huge_lut;
  EXPECT_EQ(num_detections(run_cfar<double>(input, with_lut).idx), 0);
}

TEST(CFARDetectorAlphaLutTest, AlphaLutPrecedenceOverPfaGO) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto pfa_lut = build_variant_lut(CfarVariant::kGO, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  CFARParams with_pfa = base_params(CfarVariant::kGO, 1e-3);
  with_pfa.blocking_alpha_lut_rebuild = true;
  EXPECT_GE(num_detections(run_cfar<double>(input, with_pfa).idx), 1);

  CFARParams with_lut = base_params(CfarVariant::kGO, 1e-3);
  with_lut.alpha_lut = huge_lut;
  EXPECT_EQ(num_detections(run_cfar<double>(input, with_lut).idx), 0);
}

TEST(CFARDetectorAlphaLutTest, AlphaLutPrecedenceOverPfaSO) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  const auto pfa_lut = build_variant_lut(CfarVariant::kSO, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  CFARParams with_pfa = base_params(CfarVariant::kSO, 1e-3);
  with_pfa.blocking_alpha_lut_rebuild = true;
  EXPECT_GE(num_detections(run_cfar<double>(input, with_pfa).idx), 1);

  CFARParams with_lut = base_params(CfarVariant::kSO, 1e-3);
  with_lut.alpha_lut = huge_lut;
  EXPECT_EQ(num_detections(run_cfar<double>(input, with_lut).idx), 0);
}

TEST(CFARDetectorAlphaLutTest, AlphaLutPrecedenceOverPfaOS) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());

  constexpr int64_t kOsRank = 18;
  const auto pfa_lut = build_variant_lut(CfarVariant::kOS, 1e-3, 1, kOsRank);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  CFARParams with_pfa = base_params(CfarVariant::kOS, 1e-3);
  with_pfa.os_rank = kOsRank;
  with_pfa.blocking_alpha_lut_rebuild = true;
  EXPECT_GE(num_detections(run_cfar<double>(input, with_pfa).idx), 1);

  CFARParams with_lut = base_params(CfarVariant::kOS, 1e-3);
  with_lut.os_rank = kOsRank;
  with_lut.alpha_lut = huge_lut;
  EXPECT_EQ(num_detections(run_cfar<double>(input, with_lut).idx), 0);
}

TEST(CFARDetectorAlphaLutTest, RejectsTooShortAlphaLut) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams params = base_params(CfarVariant::kCA, 1e-3);
  params.alpha_lut = {0.0, 1.0};
  const auto result = run_cfar_result<double>(input, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorAlphaLutTest, RejectsZeroAlphaLutEntry) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  auto lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  ASSERT_GT(lut.size(), 1u);
  lut[1] = 0.0;
  CFARParams params = base_params(CfarVariant::kCA, 1e-3);
  params.alpha_lut = lut;
  const auto result = run_cfar_result<double>(input, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorAlphaLutTest, SparseExternalLutAllowsOnlyUnreachableZeros) {
  constexpr int64_t kFold = 2;
  const std::vector<double> cube(static_cast<size_t>(kFold * kDoppler * kRange), 1.0);
  for (const CfarVariant variant : {CfarVariant::kGO, CfarVariant::kSO, CfarVariant::kOS}) {
    auto input = to_device_tensor<double>(cube, {kFold, kDoppler, kRange}, real_dtype<double>());
    auto lut = sparse_external_lut(variant);
    ASSERT_NE(std::find(lut.begin() + 1, lut.end(), 0.0), lut.end());
    CFARParams params = base_params(variant, 1e-3);
    params.nci_axes = {0};
    params.alpha_lut = std::move(lut);
    if (variant == CfarVariant::kOS) {
      params.os_rank = 18;
    }
    const auto result = run_cfar_result<double>(input, params);
    EXPECT_TRUE(result.ok()) << result.describe_failure();
    EXPECT_EQ(result.snapshots<holoscan::Tensor>("detection_mask").size(), 1U);
  }
}

TEST(CFARDetectorAlphaLutTest, SparseExternalLutRejectsReachableZero) {
  constexpr int64_t kFold = 2;
  const std::vector<double> cube(static_cast<size_t>(kFold * kDoppler * kRange), 1.0);
  for (const CfarVariant variant : {CfarVariant::kGO, CfarVariant::kSO, CfarVariant::kOS}) {
    auto lut = sparse_external_lut(variant);
    const std::vector<int64_t> reachable = reachable_lut_indices(variant);
    ASSERT_FALSE(reachable.empty());
    lut[static_cast<size_t>(reachable.front())] = 0.0;

    auto input = to_device_tensor<double>(cube, {kFold, kDoppler, kRange}, real_dtype<double>());
    CFARParams params = base_params(variant, 1e-3);
    params.nci_axes = {0};
    params.alpha_lut = std::move(lut);
    if (variant == CfarVariant::kOS) {
      params.os_rank = 18;
    }
    const auto result = run_cfar_result<double>(input, params);
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.snapshots<holoscan::Tensor>("detection_mask").empty());
    EXPECT_FALSE(result.diagnostics().empty());
  }
}

TEST(CFARDetectorAlphaLutTest, SparseExternalLutRejectsNegativeAtUnreachableIndex) {
  constexpr int64_t kFold = 2;
  const std::vector<double> cube(static_cast<size_t>(kFold * kDoppler * kRange), 1.0);
  for (const CfarVariant variant : {CfarVariant::kGO, CfarVariant::kSO, CfarVariant::kOS}) {
    auto lut = sparse_external_lut(variant);
    const auto unreachable = std::find(lut.begin() + 1, lut.end(), 0.0);
    ASSERT_NE(unreachable, lut.end());
    *unreachable = -1.0;

    auto input = to_device_tensor<double>(cube, {kFold, kDoppler, kRange}, real_dtype<double>());
    CFARParams params = base_params(variant, 1e-3);
    params.nci_axes = {0};
    params.alpha_lut = std::move(lut);
    if (variant == CfarVariant::kOS) {
      params.os_rank = 18;
    }
    const auto result = run_cfar_result<double>(input, params);
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.snapshots<holoscan::Tensor>("detection_mask").empty());
    EXPECT_FALSE(result.diagnostics().empty());
  }
}

TEST(CFARDetectorAlphaLutTest, RejectsNonFiniteAlphaLutEntry) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  auto lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  ASSERT_GT(lut.size(), 1u);
  lut[1] = std::numeric_limits<double>::quiet_NaN();
  CFARParams params = base_params(CfarVariant::kCA, 1e-3);
  params.alpha_lut = lut;
  const auto result = run_cfar_result<double>(input, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("detection_mask").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(CFARDetectorAlphaLutTest, FloatDoubleAgreeWithExternalAlphaLut) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  std::vector<float> mapf(map.begin(), map.end());

  CFARParams params = base_params(CfarVariant::kCA, 1e-3);
  params.alpha_lut = lut;
  const auto f_out = run_cfar<float>(
      to_device_tensor<float>(mapf, {kDoppler, kRange}, real_dtype<float>()), params);
  const auto d_out = run_cfar<double>(
      to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>()), params);

  EXPECT_EQ(num_detections(f_out.idx), num_detections(d_out.idx));
}

TEST(CFARDetectorAlphaLutTest, PfaIgnoredWhileExternalAlphaLutActive) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input = to_device_tensor<double>(map, {kDoppler, kRange}, real_dtype<double>());
  const auto pfa_lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  CFARParams params = base_params(CfarVariant::kCA, 0.5);
  params.alpha_lut = huge_lut;
  EXPECT_EQ(num_detections(run_cfar<double>(input, params).idx), 0);
}

int64_t brute_max_go_lut_index(int num_doppler, int num_range, int guard_d, int guard_r,
                               int train_d, int train_r) {
  int64_t max_idx = 0;
  for (int d = 0; d < num_doppler; ++d) {
    for (int r = 0; r < num_range; ++r) {
      int64_t rn = 0;
      int64_t rp = 0;
      int64_t dn = 0;
      int64_t dp = 0;
      holoscan::radar::detail::go_training_side_counts(
          d, r, num_doppler, num_range, guard_d, guard_r, train_d, train_r, &rn, &rp, &dn, &dp);
      max_idx = std::max(max_idx, holoscan::radar::detail::go_lut_index(rn, rp, dn, dp));
    }
  }
  return max_idx;
}

TEST(CFARDetectorAlphaLutTest, MaxGoLutIndexMatchesBruteForce) {
  const std::vector<std::array<int, 4>> footprints = {
      {1, 1, 2, 2}, {2, 2, 4, 4}, {1, 1, 1, 4}, {0, 0, 3, 3}, {1, 0, 2, 0}};
  for (const auto& fp : footprints) {
    const int gd = fp[0];
    const int gr = fp[1];
    const int td = fp[2];
    const int tr = fp[3];
    const int win_d = 2 * (gd + td) + 1;
    const int win_r = 2 * (gr + tr) + 1;
    for (int D : {win_d, win_d + 3, std::max(1, win_d - 1), 3}) {
      for (int R : {win_r, win_r + 3, std::max(1, win_r - 1), 3}) {
        const auto formula = holoscan::radar::detail::max_go_lut_index(D, R, gd, gr, td, tr);
        const auto brute = brute_max_go_lut_index(D, R, gd, gr, td, tr);
        EXPECT_EQ(formula, brute) << "fp=(" << gd << "," << gr << "," << td << "," << tr
                                  << ") map=" << D << "x" << R;
      }
    }
  }
}

TEST(CFARDetectorAlphaLutTest, CanonicalWindowLutEqualsLargerMapCaOs) {
  namespace detail = holoscan::radar::detail;
  const int win_d = 2 * (kGuardD + kTrainD) + 1;
  const int win_r = 2 * (kGuardR + kTrainR) + 1;
  const int big_d = static_cast<int>(kDoppler);
  const int big_r = static_cast<int>(kRange);
  const double pfa = 1e-3;

  const auto expect_eq_vec = [](const std::vector<double>& a, const std::vector<double>& b,
                                const char* what) {
    ASSERT_EQ(a.size(), b.size()) << what << " length";
    for (size_t i = 0; i < a.size(); ++i) {
      EXPECT_DOUBLE_EQ(a[i], b[i]) << what << " index " << i;
    }
  };

  EXPECT_EQ(detail::max_training_cell_count(win_d, win_r, kGuardD, kGuardR, kTrainD, kTrainR),
            detail::max_training_cell_count(big_d, big_r, kGuardD, kGuardR, kTrainD, kTrainR));
  EXPECT_EQ(
      detail::reachable_training_cell_counts(win_d, win_r, kGuardD, kGuardR, kTrainD, kTrainR),
      detail::reachable_training_cell_counts(big_d, big_r, kGuardD, kGuardR, kTrainD, kTrainR));

  expect_eq_vec(
      detail::build_ca_alpha_lut(pfa, 1, win_d, win_r, kGuardD, kGuardR, kTrainD, kTrainR),
      detail::build_ca_alpha_lut(pfa, 1, big_d, big_r, kGuardD, kGuardR, kTrainD, kTrainR), "CA");
  expect_eq_vec(
      detail::build_os_alpha_lut(pfa, 1, /*os_rank=*/20, win_d, win_r, kGuardD, kGuardR, kTrainD,
                                 kTrainR),
      detail::build_os_alpha_lut(pfa, 1, 20, big_d, big_r, kGuardD, kGuardR, kTrainD, kTrainR),
      "OS");
}

TEST(CFARDetectorAlphaLutTest, SubWindowMapStillDetects) {
  constexpr int64_t kSmall = 5;
  std::vector<double> map(static_cast<size_t>(kSmall * kSmall), 1.0);
  map[static_cast<size_t>(2 * kSmall + 2)] = 100.0;
  auto input = to_device_tensor<double>(map, {kSmall, kSmall}, real_dtype<double>());

  CFARParams params = base_params(CfarVariant::kCA, 1e-3);
  const auto out = run_cfar<double>(std::move(input), params);
  EXPECT_GE(num_detections(out.idx), 1);
}

}  // namespace
