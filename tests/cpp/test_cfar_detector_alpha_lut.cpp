// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// alpha_lut fast-path contract tests for CFARDetectorOp.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
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
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

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

constexpr int64_t kDoppler = 16;
constexpr int64_t kRange = 32;
constexpr int kGuardD = 1;
constexpr int kGuardR = 1;
constexpr int kTrainD = 2;
constexpr int kTrainR = 2;

struct CfarOutputs {
  HostTensor idx;
};

std::vector<double> power_map(double floor,
                              const std::vector<std::pair<int64_t, int64_t>>&
                                  targets,
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

template <typename T, typename... Args>
CfarOutputs run_cfar_idx(std::shared_ptr<holoscan::Tensor> input,
                         Args&&... args) {
  auto harness = create_operator_test<CFARDetectorOp<T>>(
      std::forward<Args>(args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps", {single_map(std::move(input), "rd_maps")});

  TensorMapCapture idx_cap;
  const auto noop = [](const holoscan::TensorMap&) {};
  harness->template add_output_port<holoscan::TensorMap>(
      "detection_idx", idx_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "detection_mask", {noop});
  harness->template add_output_port<holoscan::TensorMap>(
      "detection_score", {noop});
  harness->template add_output_port<holoscan::TensorMap>(
      "noise_estimate", {noop});

  harness->run_test();
  EXPECT_EQ(idx_cap.frames.size(), 1u);
  return {std::move(idx_cap.frames.front())};
}

std::vector<double> build_variant_lut(CfarVariant variant, double pfa,
                                      int64_t fold, int64_t os_rank) {
  switch (variant) {
    case CfarVariant::kCA:
      return holoscan::radar::detail::build_ca_alpha_lut(
          pfa, fold, static_cast<int>(kDoppler), static_cast<int>(kRange),
          kGuardD, kGuardR, kTrainD, kTrainR);
    case CfarVariant::kGO:
      return holoscan::radar::detail::build_go_alpha_lut(
          pfa, fold, static_cast<int>(kDoppler), static_cast<int>(kRange),
          kGuardD, kGuardR, kTrainD, kTrainR);
    case CfarVariant::kSO:
      return holoscan::radar::detail::build_so_alpha_lut(
          pfa, fold, static_cast<int>(kDoppler), static_cast<int>(kRange),
          kGuardD, kGuardR, kTrainD, kTrainR);
    case CfarVariant::kOS:
      return holoscan::radar::detail::build_os_alpha_lut(
          pfa, fold, os_rank, static_cast<int>(kDoppler),
          static_cast<int>(kRange), kGuardD, kGuardR, kTrainD, kTrainR);
  }
  return {};
}

std::vector<double> inflated_lut(const std::vector<double>& lut,
                                 double scale) {
  std::vector<double> out = lut;
  for (size_t i = 1; i < out.size(); ++i) {
    out[i] *= scale;
  }
  return out;
}

TEST(CFARDetectorAlphaLutTest, AlphaLutPrecedenceOverPfaCA) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input =
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());

  const auto pfa_lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  const auto with_pfa = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);
  EXPECT_GE(num_detections(with_pfa.idx), 1);

  const auto with_lut = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("alpha_lut") = huge_lut);
  EXPECT_EQ(num_detections(with_lut.idx), 0);
}

TEST(CFARDetectorAlphaLutTest, AlphaLutPrecedenceOverPfaGO) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input =
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());

  const auto pfa_lut = build_variant_lut(CfarVariant::kGO, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  const auto with_pfa = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);
  EXPECT_GE(num_detections(with_pfa.idx), 1);

  const auto with_lut = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kGO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("alpha_lut") = huge_lut);
  EXPECT_EQ(num_detections(with_lut.idx), 0);
}

TEST(CFARDetectorAlphaLutTest, AlphaLutPrecedenceOverPfaSO) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input =
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());

  const auto pfa_lut = build_variant_lut(CfarVariant::kSO, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  const auto with_pfa = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kSO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);
  EXPECT_GE(num_detections(with_pfa.idx), 1);

  const auto with_lut = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kSO,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("alpha_lut") = huge_lut);
  EXPECT_EQ(num_detections(with_lut.idx), 0);
}

TEST(CFARDetectorAlphaLutTest, AlphaLutPrecedenceOverPfaOS) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input =
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());

  constexpr int64_t kOsRank = 18;
  const auto pfa_lut = build_variant_lut(CfarVariant::kOS, 1e-3, 1, kOsRank);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  const auto with_pfa = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kOS,
      holoscan::Arg("os_rank") = kOsRank,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("blocking_alpha_lut_rebuild") = true);
  EXPECT_GE(num_detections(with_pfa.idx), 1);

  const auto with_lut = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kOS,
      holoscan::Arg("os_rank") = kOsRank,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 1e-3,
      holoscan::Arg("alpha_lut") = huge_lut);
  EXPECT_EQ(num_detections(with_lut.idx), 0);
}

TEST(CFARDetectorAlphaLutTest, RejectsTooShortAlphaLut) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  const auto input =
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());

  EXPECT_THROW(
      run_cfar_idx<double>(
          input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
          holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
          holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
          holoscan::Arg("pfa") = 1e-3,
          holoscan::Arg("alpha_lut") = std::vector<double>{0.0, 1.0}),
      std::invalid_argument);
}

TEST(CFARDetectorAlphaLutTest, RejectsZeroAlphaLutEntry) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  const auto input =
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  auto lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  ASSERT_GT(lut.size(), 1u);
  lut[1] = 0.0;

  EXPECT_THROW(
      run_cfar_idx<double>(
          input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
          holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
          holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
          holoscan::Arg("alpha_lut") = lut),
      std::invalid_argument);
}

TEST(CFARDetectorAlphaLutTest, RejectsNonFiniteAlphaLutEntry) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {}, 1.0);
  const auto input =
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  auto lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  ASSERT_GT(lut.size(), 1u);
  lut[1] = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(
      run_cfar_idx<double>(
          input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
          holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
          holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
          holoscan::Arg("alpha_lut") = lut),
      std::invalid_argument);
}

TEST(CFARDetectorAlphaLutTest, FloatDoubleAgreeWithExternalAlphaLut) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  std::vector<float> mapf(map.begin(), map.end());

  const auto f_out = run_cfar_idx<float>(
      to_device_tensor(mapf, {kDoppler, kRange}, real_dtype<float>()),
      holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("alpha_lut") = lut);
  const auto d_out = run_cfar_idx<double>(
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>()),
      holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("alpha_lut") = lut);

  EXPECT_EQ(num_detections(f_out.idx), num_detections(d_out.idx));
}

TEST(CFARDetectorAlphaLutTest, PfaIgnoredWhileExternalAlphaLutActive) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto input =
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  const auto pfa_lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  // Loose pfa would detect on this map, but the huge external LUT must win.
  const auto with_loose_pfa = run_cfar_idx<double>(
      input, holoscan::Arg("cfar_variant") = CfarVariant::kCA,
      holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
      holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
      holoscan::Arg("pfa") = 0.5,
      holoscan::Arg("alpha_lut") = huge_lut);
  EXPECT_EQ(num_detections(with_loose_pfa.idx), 0);
}

struct LiveAlphaLutApp : public holoscan::Application {
  std::vector<holoscan::TensorMap> frames;
  std::vector<std::vector<double>> alpha_lut_overrides;
  TensorMapCapture* idx_capture = nullptr;

  void compose() override {
    auto op = make_operator<CFARDetectorOp<double>>(
        "op_under_test", holoscan::Arg("cfar_variant") = CfarVariant::kCA,
        holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
        holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
        holoscan::Arg("pfa") = 1e-3);
    wire_data_source(this, op, "rd_maps", frames);
    wire_leading_override<std::vector<double>>(this, op, "alpha_lut",
                                               alpha_lut_overrides);
    wire_capture_sink(this, op, "detection_idx", idx_capture);
    wire_noop_cfar_output_sinks(this, op, /*capture_score=*/false);
  }
};

TEST(CFARDetectorAlphaLutTest, LiveAlphaLutOverrideStickyAndClearing) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  const auto pfa_lut = build_variant_lut(CfarVariant::kCA, 1e-3, 1, 0);
  const auto huge_lut = inflated_lut(pfa_lut, 1e6);

  holoscan::TensorMap frame = single_map(
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>()),
      "rd_maps");

  TensorMapCapture capture;
  auto app = holoscan::make_application<LiveAlphaLutApp>();
  app->frames = {frame, frame, frame};
  app->alpha_lut_overrides = {huge_lut, huge_lut, std::vector<double>()};
  app->idx_capture = &capture;
  app->run();

  ASSERT_EQ(capture.frames.size(), 3u);
  EXPECT_EQ(num_detections(capture.frames[0]), 0);
  EXPECT_EQ(num_detections(capture.frames[1]), 0);
  EXPECT_GE(num_detections(capture.frames[2]), 1);
}

struct LivePfaNonBlockingApp : public holoscan::Application {
  std::vector<holoscan::TensorMap> frames;
  std::vector<double> pfa_overrides;
  TensorMapCapture* idx_capture = nullptr;

  void compose() override {
    auto op = make_operator<CFARDetectorOp<double>>(
        "op_under_test", holoscan::Arg("cfar_variant") = CfarVariant::kGO,
        holoscan::Arg("guard_cells") = std::vector<int64_t>{1, 1},
        holoscan::Arg("training_cells") = std::vector<int64_t>{2, 2},
        holoscan::Arg("pfa") = 1e-3,
        holoscan::Arg("blocking_alpha_lut_rebuild") = false);
    wire_data_source(this, op, "rd_maps", frames);
    wire_leading_override<double>(this, op, "pfa", pfa_overrides);
    wire_capture_sink(this, op, "detection_idx", idx_capture);
    wire_noop_cfar_output_sinks(this, op, /*capture_score=*/false);
  }
};

// After a warm LUT exists, a non-blocking pfa change must serve the previous
// LUT immediately (GO MC rebuild is multi-second; a sub-second tick proves
// the stale path ran).
TEST(CFARDetectorAlphaLutTest, NonBlockingPfaOverrideServesStaleLut) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  holoscan::TensorMap frame = single_map(
      to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>()),
      "rd_maps");

  TensorMapCapture capture;
  auto app = holoscan::make_application<LivePfaNonBlockingApp>();
  app->frames = {frame, frame};
  // Frame 0: establish warm LUT at pfa=1e-3 (blocks once: no prior LUT).
  // Frame 1: override to a new pfa — must not wait for the GO MC rebuild.
  app->pfa_overrides = {1e-3, 1e-6};
  app->idx_capture = &capture;

  const auto t0 = std::chrono::steady_clock::now();
  app->run();
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0)
          .count();

  ASSERT_EQ(capture.frames.size(), 2u);
  EXPECT_GE(num_detections(capture.frames[0]), 1);
  EXPECT_GE(num_detections(capture.frames[1]), 1);
  // Warm GO LUT ~2–3 s on this footprint. A blocking second build would push
  // the total near 2× that; the stale path keeps total near one warm build.
  EXPECT_LT(elapsed_ms, 3500.0) << "elapsed_ms=" << elapsed_ms;
}

int64_t brute_max_go_lut_index(int num_doppler, int num_range, int guard_d,
                               int guard_r, int train_d, int train_r) {
  int64_t max_idx = 0;
  for (int d = 0; d < num_doppler; ++d) {
    for (int r = 0; r < num_range; ++r) {
      int64_t rn = 0;
      int64_t rp = 0;
      int64_t dn = 0;
      int64_t dp = 0;
      holoscan::radar::detail::go_training_side_counts(
          d, r, num_doppler, num_range, guard_d, guard_r, train_d, train_r, &rn,
          &rp, &dn, &dp);
      max_idx = std::max(
          max_idx, holoscan::radar::detail::go_lut_index(rn, rp, dn, dp));
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
        EXPECT_EQ(holoscan::radar::detail::max_go_lut_index(D, R, gd, gr, td,
                                                            tr),
                  brute_max_go_lut_index(D, R, gd, gr, td, tr))
            << "fp=(" << gd << "," << gr << "," << td << "," << tr
            << ") map=" << D << "x" << R;
      }
    }
  }
}

}  // namespace
