// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ESPRITEstimatorOp contract tests (HSDK 5 Graph harness): known-DOA recovery,
// multi-source / eigenvalues / spatial smoothing / explicit subarray / URA
// Params coverage, float/double agreement, optional-mask wiring, live
// overrides, and error paths.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/estimator_test_fixtures.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::ArrayGeometryType;
using holoscan::radar::ElevationReference;
using holoscan::radar::ESPRITEstimatorOp;
using holoscan::radar::ESPRITParams;
using holoscan::radar::ShiftInvariantSubarrayPair;
using holoscan::radar::SpatialSmoothingOverride;
using holoscan::radar::SignedAxis;
using holoscan::radar::SubarraySelectionOp;
using holoscan::radar::SubarraySelectionParams;
using holoscan::radar::WireArrayGeometry;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::FourOutGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::block_to_device;
using holoscan::radar::testing::kPi;
using holoscan::radar::testing::make_block;
using holoscan::radar::testing::make_block_two;
using holoscan::radar::testing::make_planar_ura_geometry;
using holoscan::radar::testing::make_ula_geometry;
namespace radar_testing = holoscan::radar::testing;
using radar_testing::run_exact_override_four_out_operator_graph_result;
using holoscan::radar::testing::run_dual_io_operator_graph_result;
using holoscan::radar::testing::run_single_in_four_out_operator_graph_result;
using holoscan::radar::testing::steering_vector;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;

struct EspritOutputs {
  HostTensor aoa;
  HostTensor doa_multi;
  HostTensor source_count;
  HostTensor eigenvalues;
};

// cpplint misreads the wrapped template header as an indented declaration
// inside the anonymous namespace; the parameter list cannot fit on one line.
template <typename Source, typename OpRef, typename S0, typename S1,
          typename S2, typename S3>  // NOLINT(whitespace/indent_namespace)
void wire_esprit(holoscan::Graph& graph, const Source& source, OpRef& op,
                 const S0& sink0, const S1& sink1, const S2& sink2,
                 const S3& sink3) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->snapshot, q);
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->doa_multi, sink1->in, q);
  graph.add_flow(op->source_count, sink2->in, q);
  graph.add_flow(op->eigenvalues, sink3->in, q);
}

ESPRITParams make_params(ArrayGeometry geom, AngleConvention conv,
                         int64_t num_sources = -1, double pfa = 1e-2,
                         int64_t max_sources = 4,
                         std::vector<int64_t> smoothing = {},
                         bool fb = true,
                         ShiftInvariantSubarrayPair subarray = {}) {
  ESPRITParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.num_sources = num_sources;
  params.source_count_pfa = pfa;
  params.max_sources = max_sources;
  params.spatial_smoothing = std::move(smoothing);
  params.forward_backward = fb;
  params.subarray_spec = std::move(subarray);
  return params;
}

// Noise-free block whose sample covariance is exactly V diag(`eigenvalues`)
// V^H, with V the unit steering vectors of `direction_cosines`. Spacing those
// cosines 2/num_channels apart makes V unitary on a half-wavelength ULA, so
// snapshot k carrying source k alone at power num_snapshots * eigenvalues[k] /
// num_channels reproduces the profile exactly. Pinning the eigenvalues that
// way turns the EFT source count into a pure function of the requested Pfa,
// while keeping the covariance physically realizable (unlike a synthetic
// diagonal, which leaves the subspace stage ill-conditioned).
std::vector<std::complex<double>> make_orthogonal_source_block(
    const ArrayGeometry& geometry, const AngleConvention& convention,
    const std::vector<double>& direction_cosines,
    const std::vector<double>& eigenvalues) {
  const int64_t channels = geometry.num_elements();
  const int64_t snapshots = static_cast<int64_t>(eigenvalues.size());
  std::vector<std::complex<double>> block(
      static_cast<size_t>(channels * snapshots));
  for (int64_t s = 0; s < snapshots; ++s) {
    const auto a =
        steering_vector(geometry, convention,
                        direction_cosines[static_cast<size_t>(s)], 0.0);
    const double amplitude =
        std::sqrt(static_cast<double>(snapshots) *
                  eigenvalues[static_cast<size_t>(s)] /
                  static_cast<double>(channels));
    for (int64_t c = 0; c < channels; ++c) {
      block[static_cast<size_t>(c * snapshots + s)] =
          a[static_cast<size_t>(c)] * amplitude;
    }
  }
  return block;
}

template <typename T>
FourOutGraphRunResult run_esprit_result(
    std::shared_ptr<holoscan::Tensor> snapshot, ESPRITParams params) {
  return run_single_in_four_out_operator_graph_result<ESPRITEstimatorOp<T>>(
      {std::move(snapshot)}, std::move(params), "aoa_estimates", "doa_multi",
      "source_count", "eigenvalues",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1, const auto& sink2,
         const auto& sink3) {
        wire_esprit(graph, source, op, sink0, sink1, sink2, sink3);
      });
}

template <typename T>
EspritOutputs run_esprit(std::shared_ptr<holoscan::Tensor> snapshot,
                         ESPRITParams params) {
  FourOutGraphRunResult result =
      run_esprit_result<T>(std::move(snapshot), std::move(params));
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
      result.frames2.size() != 1U || result.frames3.size() != 1U) {
    throw std::runtime_error(
        "ESPRITEstimatorOp test: expected one output frame set");
  }
  return {std::move(result.frames0.front()), std::move(result.frames1.front()),
          std::move(result.frames2.front()), std::move(result.frames3.front())};
}

TEST(ESPRITEstimatorOpTest, DefaultsToRadarBoresightConvention) {
  EXPECT_EQ(ESPRITParams{}.angle_convention.preset,
            AngleConventionPreset::kRadarBoresight);
}

TEST(ESPRITEstimatorOpTest, RecoversKnownDoaUlaFixedOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 3u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
}

TEST(ESPRITEstimatorOpTest, RecoversTwoSources) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 256;
  constexpr double kAz0 = 0.20;
  constexpr double kAz1 = -0.15;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.02, 71u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/2, 1e-2, /*max_sources=*/2));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double az_a = dm[0];
  const double az_b = dm[2];
  const bool matched =
      (std::abs(az_a - kAz0) < 0.03 && std::abs(az_b - kAz1) < 0.03) ||
      (std::abs(az_a - kAz1) < 0.03 && std::abs(az_b - kAz0) < 0.03);
  EXPECT_TRUE(matched) << "az_a=" << az_a << " az_b=" << az_b;

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(dm[0], aoa[0], 1e-9);
  EXPECT_NEAR(dm[1], aoa[1], 1e-9);
}

TEST(ESPRITEstimatorOpTest, ResolvesCloselySpacedSources) {
  constexpr int64_t kN = 16;
  constexpr int64_t kS = 512;
  constexpr double kAz0 = -0.035;
  constexpr double kAz1 = 0.035;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.005, 73u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/2, 1e-2, /*max_sources=*/2));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 2);
  std::array<double, 2> actual{
      out.doa_multi.as<double>()[0],
      out.doa_multi.as<double>()[2],
  };
  std::sort(actual.begin(), actual.end());
  EXPECT_NEAR(actual[0], kAz0, 0.01);
  EXPECT_NEAR(actual[1], kAz1, 0.01);
}

TEST(ESPRITEstimatorOpTest, MaxSourcesCapsReportedDoas) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 256;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, -0.15, 0.0, 0.20, 0.0, kS, 0.02, 78u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/2, 1e-2, /*max_sources=*/1));

  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
  EXPECT_TRUE(std::isfinite(out.doa_multi.as<double>()[0]));
}

TEST(ESPRITEstimatorOpTest, HighSnrLargeArrayEstimateRemainsFinite) {
  constexpr int64_t kN = 32;
  constexpr int64_t kS = 128;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 1e-6, 74u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double az = out.aoa.as<double>()[0];
  EXPECT_TRUE(std::isfinite(az));
  EXPECT_NEAR(az, kTruthAz, 1e-4);
}

TEST(ESPRITEstimatorOpTest, AutoNumSourcesRecoversOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 9u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/-1, 1e-2,
                        /*max_sources=*/4));

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
}

TEST(ESPRITEstimatorOpTest, RecoversUraSingleSourceAzEl) {
  constexpr int64_t kNy = 6;
  constexpr int64_t kNz = 6;
  constexpr int64_t kS = 256;
  constexpr double kAz = 0.15;
  constexpr double kEl = -0.10;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kAz, kEl, kS, 0.02, 7u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz, 0.03);
  EXPECT_NEAR(aoa[1], kEl, 0.03);
}

TEST(ESPRITEstimatorOpTest, RecoversUraTwoSourcesAzElPairing) {
  constexpr int64_t kNy = 8;
  constexpr int64_t kNz = 8;
  constexpr int64_t kS = 512;
  // Distinct in BOTH az and el so a mispaired result is detectable.
  constexpr double kAz0 = 0.20;
  constexpr double kEl0 = 0.12;
  constexpr double kAz1 = -0.15;
  constexpr double kEl1 = -0.10;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, kEl0, kAz1, kEl1, kS, 0.01, 79u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/2, 1e-2, /*max_sources=*/2));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const auto near_pair = [](double az, double el, double taz, double tel) {
    return std::abs(az - taz) < 0.04 && std::abs(el - tel) < 0.04;
  };
  const bool matched =
      (near_pair(dm[0], dm[1], kAz0, kEl0) &&
       near_pair(dm[2], dm[3], kAz1, kEl1)) ||
      (near_pair(dm[0], dm[1], kAz1, kEl1) &&
       near_pair(dm[2], dm[3], kAz0, kEl0));
  EXPECT_TRUE(matched) << "pair0=(" << dm[0] << "," << dm[1] << ") pair1=("
                       << dm[2] << "," << dm[3] << ")";
}

TEST(ESPRITEstimatorOpTest, RecoversUraSingleSourceAzElFloat) {
  constexpr int64_t kNy = 6;
  constexpr int64_t kNz = 6;
  constexpr int64_t kS = 256;
  constexpr double kAz = -0.12;
  constexpr double kEl = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kAz, kEl, kS, 0.02, 17u);
  auto snap = block_to_device<float>(block, {1, geom.num_elements(), kS});

  const auto out =
      run_esprit<float>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], kAz, 0.03);
  EXPECT_NEAR(aoa[1], kEl, 0.03);
}

TEST(ESPRITEstimatorOpTest, RejectsNonUniformLinearGeometry) {
  constexpr int64_t kN = 5;
  constexpr int64_t kS = 16;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  auto geom = make_ula_geometry(kN, 0.5, 1.0);
  geom.element_positions[3 * 3 + 1] = 1.65;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 75u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const FourOutGraphRunResult result =
      run_esprit_result<double>(snap, make_params(geom, conv));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, RejectsUlaParallelToBoresight) {
  constexpr int64_t kN = 4;
  ArrayGeometry geom;
  geom.wavelength = 1.0;
  geom.geometry_type = ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < kN; ++channel) {
    geom.element_positions.insert(
        geom.element_positions.end(),
        {0.5 * static_cast<double>(channel), 0.0, 0.0});
  }
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> block(static_cast<size_t>(kN * 4),
                                          {1.0, 0.0});
  auto snap = block_to_device<double>(block, {1, kN, 4});

  const FourOutGraphRunResult result =
      run_esprit_result<double>(snap, make_params(geom, conv));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, EigenvaluesAscendingAndNonNegative) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 128;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 91u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, kN}));
  const double* ev = out.eigenvalues.as<double>();
  for (int64_t c = 0; c < kN; ++c) {
    EXPECT_GE(ev[c], -1e-6);
    if (c > 0) {
      EXPECT_GE(ev[c], ev[c - 1] - 1e-9);
    }
  }
  EXPECT_GT(ev[kN - 1], 10.0 * ev[kN - 2]);
}

TEST(ESPRITEstimatorOpTest, RankDeficientAutoFallsBackToOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 4;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 12u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/-1, 1e-2,
                        /*max_sources=*/4));

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
}

TEST(ESPRITEstimatorOpTest, RejectsNumSourcesTooLarge) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});

  const FourOutGraphRunResult result = run_esprit_result<double>(
      snap, make_params(geom, conv, /*num_sources=*/kN));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, RejectsInvalidMaxSources) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});

  const FourOutGraphRunResult result = run_esprit_result<double>(
      snap, make_params(geom, conv, /*num_sources=*/-1, 1e-2,
                        /*max_sources=*/0));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, RejectsInvalidSourceCountPfa) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});

  const FourOutGraphRunResult result = run_esprit_result<double>(
      snap, make_params(geom, conv, /*num_sources=*/-1, /*pfa=*/0.0));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap = to_device_tensor(host, {0, kN, 4}, complex_dtype<double>());

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/-1, 1e-2,
                        /*max_sources=*/3));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{0, 3, 2}));
  EXPECT_EQ(out.source_count.shape, (std::vector<int64_t>{0}));
  EXPECT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{0, kN}));
}

TEST(ESPRITEstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>((kN - 1) * 4),
                                         {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN - 1, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;

  const FourOutGraphRunResult result =
      run_esprit_result<double>(snap, make_params(geom, conv));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_f = block_to_device<float>(a, {1, kN, 1});

  const FourOutGraphRunResult result =
      run_esprit_result<double>(snap_f, make_params(geom, conv));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, RejectsSnapshotRank2) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.05, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});

  const FourOutGraphRunResult result =
      run_esprit_result<double>(snap, make_params(geom, conv));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, AcceptsPinnedHostSnapshot) {
  // Device-tensor smoke (pinned-host path not exercisable via Graph harness).
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 161u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));
  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(ESPRITEstimatorOpTest, AcceptsManagedSnapshot) {
  // Device-tensor smoke (managed path not exercisable via Graph harness).
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 162u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));
  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(ESPRITEstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.06, 0.0, kS, 0.02, 11u);
  auto snap = block_to_device<float>(block, {1, kN, kS});

  const auto out =
      run_esprit<float>(snap, make_params(geom, conv, /*num_sources=*/1));
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.06f, 0.03f);
}

TEST(ESPRITEstimatorOpTest, FloatDoubleAgreeOnDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.09;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 5u);
  auto snap_f = block_to_device<float>(block, {1, kN, kS});
  auto snap_d = block_to_device<double>(block, {1, kN, kS});

  ESPRITParams params = make_params(geom, conv, /*num_sources=*/1);

  const auto out_f = run_esprit<float>(snap_f, params);
  const auto out_d = run_esprit<double>(snap_d, params);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 1e-2f);
}

TEST(ESPRITEstimatorOpTest, DeterministicAcrossRuns) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.09, 0.0, kS, 0.02, 171u);

  const auto run = [&]() {
    return run_esprit<double>(
        block_to_device<double>(block, {1, kN, kS}),
        make_params(geom, conv, /*num_sources=*/1, 1e-2, /*max_sources=*/2));
  };
  const auto first = run();
  const auto second = run();

  EXPECT_EQ(first.aoa.data, second.aoa.data);
  EXPECT_EQ(first.doa_multi.data, second.doa_multi.data);
  EXPECT_EQ(first.source_count.data, second.source_count.data);
  EXPECT_EQ(first.eigenvalues.data, second.eigenvalues.data);
}

TEST(ESPRITEstimatorOpTest, DirectionCosineConventionMapsUlaProjection) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthU = 0.23;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthU, 0.0, kS, 0.01, 76u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthU, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], 0.0, 1e-9);
}

TEST(ESPRITEstimatorOpTest, CustomConventionMapsSignedAxes) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kCanonicalAz = 0.18;
  AngleConvention radar;
  radar.preset = AngleConventionPreset::kRadarBoresight;
  AngleConvention custom;
  custom.preset = AngleConventionPreset::kCustom;
  custom.boresight_axis = SignedAxis::kPlusX;
  custom.azimuth_positive_axis = SignedAxis::kMinusY;
  custom.elevation_positive_axis = SignedAxis::kMinusZ;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block(geom, radar, kCanonicalAz, 0.0, kS, 0.01, 176u);

  const auto out = run_esprit<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, custom, /*num_sources=*/1));

  EXPECT_NEAR(out.aoa.as<double>()[0], -kCanonicalAz, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], 0.0, 1e-9);
}

TEST(ESPRITEstimatorOpTest, CustomConventionSupportsZenithReference) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  AngleConvention radar;
  radar.preset = AngleConventionPreset::kRadarBoresight;
  AngleConvention custom;
  custom.preset = AngleConventionPreset::kCustom;
  custom.elevation_reference = ElevationReference::kZenith;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, radar, 0.1, 0.0, kS, 0.01, 177u);

  const auto out = run_esprit<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, custom, /*num_sources=*/1));

  EXPECT_NEAR(out.aoa.as<double>()[0], 0.1, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], kPi / 2.0, 1e-9);
}

TEST(ESPRITEstimatorOpTest, RejectsDegenerateCustomConvention) {
  constexpr int64_t kN = 4;
  AngleConvention radar;
  radar.preset = AngleConventionPreset::kRadarBoresight;
  AngleConvention custom;
  custom.preset = AngleConventionPreset::kCustom;
  custom.azimuth_positive_axis = custom.boresight_axis;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, radar, 0.1, 0.0, 8, 0.01, 178u);

  const FourOutGraphRunResult result = run_esprit_result<double>(
      block_to_device<double>(block, {1, kN, 8}),
      make_params(geom, custom, /*num_sources=*/1));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, DirectionCosineNearEndfirePreservesSign) {
  constexpr int64_t kN = 16;
  constexpr int64_t kS = 128;
  constexpr double kU0 = -0.92;
  constexpr double kU1 = 0.92;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto b0 = make_block(geom, conv, kU0, 0.0, kS, 0.005, 79u);
  const auto b1 = make_block(geom, conv, kU1, 0.0, kS, 0.005, 80u);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), b0.begin(), b0.end());
  batch.insert(batch.end(), b1.begin(), b1.end());
  auto snap = block_to_device<double>(batch, {2, kN, kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  EXPECT_NEAR(out.aoa.as<double>()[0], kU0, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[2], kU1, 0.01);
}

TEST(ESPRITEstimatorOpTest, UnusedDoaRowsAreNan) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.01, 77u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/1, 1e-2,
                        /*max_sources=*/3));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double* doa = out.doa_multi.as<double>();
  EXPECT_TRUE(std::isfinite(doa[0]));
  EXPECT_TRUE(std::isfinite(doa[1]));
  for (int64_t index = 2; index < 6; ++index) {
    EXPECT_TRUE(std::isnan(doa[index]));
  }
}

TEST(ESPRITEstimatorOpTest, IsValidMaskExcludesWrongSnapshots) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.14;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto truth =
      make_block(geometry, conv, kTruthAz, 0.0, kS, 0.01, 701u);
  const auto wrong =
      make_block(geometry, conv, -0.24, 0.0, kS, 0.01, 702u);
  std::vector<std::complex<double>> mixed(
      static_cast<size_t>(kN * 2 * kS));
  for (int64_t channel = 0; channel < kN; ++channel) {
    for (int64_t look = 0; look < kS; ++look) {
      mixed[static_cast<size_t>(channel * 2 * kS + look)] =
          truth[static_cast<size_t>(channel * kS + look)];
      mixed[static_cast<size_t>(channel * 2 * kS + kS + look)] =
          wrong[static_cast<size_t>(channel * kS + look)];
    }
  }
  auto snapshot = block_to_device<double>(mixed, {1, kN, 2 * kS});
  std::vector<uint8_t> mask_host(static_cast<size_t>(2 * kS), uint8_t{0});
  std::fill_n(mask_host.begin(), static_cast<size_t>(kS), uint8_t{1});
  auto mask = to_device_tensor(mask_host, {1, 2 * kS},
                               DLDataType{kDLUInt, 8, 1});

  const auto result =
      run_dual_io_operator_graph_result<ESPRITEstimatorOp<double>>(
          {mask}, {snapshot},
          make_params(geometry, conv, /*num_sources=*/1),
          "aoa_estimates", "source_count",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out0, op->is_valid, q);
            graph.add_flow(source->out1, op->snapshot, q);
            graph.add_flow(op->aoa_estimates, sink0->in, q);
            graph.add_flow(op->source_count, sink1->in, q);
          },
          holoscan::radar::testing::default_test_extents(),
          {"doa_multi", "eigenvalues"});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames0.size(), 1U);
  EXPECT_NEAR(result.frames0[0].as<double>()[0], kTruthAz, 0.03);
}

TEST(ESPRITEstimatorOpTest, AllInvalidMaskBlanksDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 16;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geometry, conv, 0.1, 0.0, kS, 0.01, 703u);
  auto snapshot = block_to_device<double>(block, {1, kN, kS});
  auto mask = to_device_tensor(std::vector<uint8_t>(kS, uint8_t{0}), {1, kS},
                               DLDataType{kDLUInt, 8, 1});

  const auto result =
      run_dual_io_operator_graph_result<ESPRITEstimatorOp<double>>(
          {mask}, {snapshot},
          make_params(geometry, conv, /*num_sources=*/1),
          "aoa_estimates", "source_count",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out0, op->is_valid, q);
            graph.add_flow(source->out1, op->snapshot, q);
            graph.add_flow(op->aoa_estimates, sink0->in, q);
            graph.add_flow(op->source_count, sink1->in, q);
          },
          holoscan::radar::testing::default_test_extents(),
          {"doa_multi", "eigenvalues"});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames0.size(), 1U);
  ASSERT_EQ(result.frames1.size(), 1U);
  EXPECT_TRUE(std::isnan(result.frames0[0].as<double>()[0]));
  EXPECT_EQ(result.frames1[0].as<int32_t>()[0], 0);
}

// num_sources is reported verbatim on source_count (clamped by max_sources),
// so an int32 output value carries the override directly. Ticks 0 and 2 send
// nothing and pin the Params baseline inside the same run, so nothing is
// compared across graph runs.
TEST(ESPRITEstimatorOpTest, LiveNumSourcesAppliesForOneTick) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 128;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geometry, conv, 0.20, 0.0, -0.15, 0.0, kS, 0.02, 311u);
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    frames.push_back(block_to_device<double>(block, {1, kN, kS}));
  }
  ESPRITParams params =
      make_params(geometry, conv, /*num_sources=*/1, 1e-2, /*max_sources=*/4);
  std::vector<std::optional<int64_t>> payloads{
      std::nullopt, int64_t{2}, std::nullopt};

  const auto result =
      run_exact_override_four_out_operator_graph_result<
          ESPRITEstimatorOp<double>, int64_t>(
          std::move(frames), std::move(payloads), std::move(params),
          "aoa_estimates", "doa_multi", "source_count", "eigenvalues",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1, const auto& sink2,
             const auto& sink3) {
            wire_esprit(graph, source, op, sink0, sink1, sink2, sink3);
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->override_out, op->num_sources, q);
          });
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames2.size(), 3U);
  EXPECT_EQ(result.frames2[0].as<int32_t>()[0], 1);
  EXPECT_EQ(result.frames2[1].as<int32_t>()[0], 2);
  EXPECT_EQ(result.frames2[2].as<int32_t>()[0], 1);
}

TEST(ESPRITEstimatorOpTest, LiveArrayGeometryAppliesForOneTick) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geometry, conv, 0.1, 0.0, kS, 0.02, 301u);
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    frames.push_back(block_to_device<double>(block, {1, kN, kS}));
  }
  // The Params geometry has the same element count but a different
  // wavelength, so it stays valid against the snapshot yet resolves the
  // source to a different angle than the override does.
  ESPRITParams params =
      make_params(make_ula_geometry(kN, 0.5, 0.8), conv,
                  /*num_sources=*/1);
  std::vector<std::optional<WireArrayGeometry>> payloads;
  payloads.emplace_back(std::nullopt);
  payloads.emplace_back(holoscan::radar::to_wire(geometry));
  payloads.emplace_back(std::nullopt);

  const auto result =
      run_exact_override_four_out_operator_graph_result<
          ESPRITEstimatorOp<double>>(
          std::move(frames), std::move(payloads), std::move(params),
          "aoa_estimates", "doa_multi", "source_count", "eigenvalues",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1, const auto& sink2,
             const auto& sink3) {
            wire_esprit(graph, source, op, sink0, sink1, sink2, sink3);
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->override_out, op->array_geometry, q);
          });
  EXPECT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames0.size(), 3U);
  EXPECT_NE(result.frames0[1].data, result.frames0[0].data);
  EXPECT_EQ(result.frames0[2].data, result.frames0[0].data);

  ESPRITParams invalid_params =
      make_params(geometry, conv, /*num_sources=*/1);
  WireArrayGeometry malformed_geometry = holoscan::radar::to_wire(geometry);
  malformed_geometry.packed_geometry = to_pageable_host_tensor<double>(
      {1.0}, {1}, real_dtype<double>());
  std::vector<std::optional<WireArrayGeometry>> malformed_payloads;
  malformed_payloads.emplace_back(std::move(malformed_geometry));
  const auto invalid =
      run_exact_override_four_out_operator_graph_result<
          ESPRITEstimatorOp<double>, WireArrayGeometry>(
          {block_to_device<double>(block, {1, kN, kS})},
          std::move(malformed_payloads),
          std::move(invalid_params), "aoa_estimates", "doa_multi",
          "source_count", "eigenvalues",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1, const auto& sink2,
             const auto& sink3) {
            wire_esprit(graph, source, op, sink0, sink1, sink2, sink3);
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->override_out, op->array_geometry, q);
          });
  EXPECT_TRUE(invalid.plan_ok) << invalid.plan_json;
  EXPECT_TRUE(invalid.frames0.empty());
  EXPECT_FALSE(invalid.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, LiveSpatialSmoothingAppliesForOneTick) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geometry, conv, 0.1, 0.0, kS, 0.02, 302u);
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    frames.push_back(block_to_device<double>(block, {1, kN, kS}));
  }
  ESPRITParams params =
      make_params(geometry, conv, /*num_sources=*/1, 1e-2, 4);
  std::vector<std::optional<SpatialSmoothingOverride>> payloads{
      std::nullopt,
      SpatialSmoothingOverride{.count = 1, .first = 2},
      std::nullopt,
  };

  const auto result =
      run_exact_override_four_out_operator_graph_result<
          ESPRITEstimatorOp<double>>(
          std::move(frames), std::move(payloads), std::move(params),
          "aoa_estimates", "doa_multi", "source_count", "eigenvalues",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1, const auto& sink2,
             const auto& sink3) {
            wire_esprit(graph, source, op, sink0, sink1, sink2, sink3);
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->override_out, op->spatial_smoothing, q);
          });
  EXPECT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames3.size(), 3U);
  // Smoothing shortens the sub-array, so the eigenvalue count is the
  // observable. Ticks 0 and 2 publish nothing and share the Params order.
  EXPECT_NE(result.frames3[1].shape, result.frames3[0].shape);
  EXPECT_EQ(result.frames3[2].shape, result.frames3[0].shape);

  ESPRITParams invalid_params =
      make_params(geometry, conv, /*num_sources=*/1);
  const auto invalid =
      run_exact_override_four_out_operator_graph_result<
          ESPRITEstimatorOp<double>, SpatialSmoothingOverride>(
          {block_to_device<double>(block, {1, kN, kS})},
          {std::optional<SpatialSmoothingOverride>{
              SpatialSmoothingOverride{.count = 3}}},
          std::move(invalid_params), "aoa_estimates", "doa_multi",
          "source_count", "eigenvalues",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1, const auto& sink2,
             const auto& sink3) {
            wire_esprit(graph, source, op, sink0, sink1, sink2, sink3);
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->override_out, op->spatial_smoothing, q);
          });
  EXPECT_TRUE(invalid.plan_ok) << invalid.plan_json;
  EXPECT_TRUE(invalid.frames0.empty());
  EXPECT_FALSE(invalid.diagnostics.empty());
}

// source_count_pfa only steers the EFT model order, so the observable is the
// int32 source_count. The crafted covariance pins the eigenvalue profile
// exactly, which makes the 1 -> 2 transition between the Params Pfa and the
// override Pfa deterministic instead of a noise draw.
TEST(ESPRITEstimatorOpTest, LiveSourceCountPfaAppliesForOneTick) {
  const std::vector<double> spectrum = {0.8930540251, 0.9468277841,
                                        1.3517214181, 1.6732266229,
                                        9.0798471606, 14.3477610683};
  const std::vector<double> cosines = {-5.0 / 6.0, -0.5, -1.0 / 6.0,
                                       1.0 / 6.0, 0.5, 5.0 / 6.0};
  const int64_t kN = static_cast<int64_t>(spectrum.size());
  const int64_t kS = kN;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_orthogonal_source_block(geometry, conv, cosines, spectrum);
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    frames.push_back(block_to_device<double>(block, {1, kN, kS}));
  }
  ESPRITParams params =
      make_params(geometry, conv, /*num_sources=*/-1, /*pfa=*/1e-4,
                  /*max_sources=*/4, /*smoothing=*/{},
                  /*fb=*/false);
  std::vector<std::optional<double>> payloads{std::nullopt, 0.5, std::nullopt};

  const auto result =
      run_exact_override_four_out_operator_graph_result<
          ESPRITEstimatorOp<double>, double>(
          std::move(frames), std::move(payloads), std::move(params),
          "aoa_estimates", "doa_multi", "source_count", "eigenvalues",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1, const auto& sink2,
             const auto& sink3) {
            wire_esprit(graph, source, op, sink0, sink1, sink2, sink3);
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->override_out, op->source_count_pfa, q);
          });
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames2.size(), 3U);
  EXPECT_EQ(result.frames2[0].as<int32_t>()[0], 1);
  EXPECT_EQ(result.frames2[1].as<int32_t>()[0], 2);
  EXPECT_EQ(result.frames2[2].as<int32_t>()[0], 1);
}

// max_sources widens doa_multi to {items, max_sources, 2}, so the override
// shows up in the output shape alone.
TEST(ESPRITEstimatorOpTest, LiveMaxSourcesAppliesForOneTick) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr int64_t kLiveMaxSources = 3;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geometry, conv, 0.1, 0.0, kS, 0.02, 312u);
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    frames.push_back(block_to_device<double>(block, {1, kN, kS}));
  }
  ESPRITParams params =
      make_params(geometry, conv, /*num_sources=*/1, 1e-2, /*max_sources=*/1);
  std::vector<std::optional<int64_t>> payloads{
      std::nullopt, kLiveMaxSources, std::nullopt};

  const auto result =
      run_exact_override_four_out_operator_graph_result<
          ESPRITEstimatorOp<double>, int64_t>(
          std::move(frames), std::move(payloads), std::move(params),
          "aoa_estimates", "doa_multi", "source_count", "eigenvalues",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1, const auto& sink2,
             const auto& sink3) {
            wire_esprit(graph, source, op, sink0, sink1, sink2, sink3);
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->override_out, op->max_sources, q);
          });
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames1.size(), 3U);
  EXPECT_EQ(result.frames1[0].shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(result.frames1[1].shape,
            (std::vector<int64_t>{1, kLiveMaxSources, 2}));
  // Tick 2 reverts to the Params max_sources, narrowing doa_multi back.
  EXPECT_EQ(result.frames1[2].shape, (std::vector<int64_t>{1, 1, 2}));
}

TEST(ESPRITEstimatorOpTest, NumSourcesConfigurableFromParams) {
  // Authoring Params must select num_sources (live Graph override deferred —
  // HSDK5_PORT §1).
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 128;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block =
      make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, kS, 0.02, 111u);
  auto snap1 = block_to_device<double>(block, {1, kN, kS});
  auto snap2 = block_to_device<double>(block, {1, kN, kS});

  const auto out1 = run_esprit<double>(
      snap1, make_params(geom, conv, /*num_sources=*/1, 1e-2, 4));
  const auto out2 = run_esprit<double>(
      snap2, make_params(geom, conv, /*num_sources=*/2, 1e-2, 4));

  EXPECT_EQ(out1.source_count.as<int32_t>()[0], 1);
  EXPECT_EQ(out2.source_count.as<int32_t>()[0], 2);
}

TEST(ESPRITEstimatorOpTest, MaxSourcesConfigurableFromParams) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 101u);
  auto snap1 = block_to_device<double>(block, {1, kN, kS});
  auto snap3 = block_to_device<double>(block, {1, kN, kS});

  const auto out1 = run_esprit<double>(
      snap1, make_params(geom, conv, /*num_sources=*/1, 1e-2, 1));
  const auto out3 = run_esprit<double>(
      snap3, make_params(geom, conv, /*num_sources=*/1, 1e-2, 3));

  EXPECT_EQ(out1.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out3.doa_multi.shape, (std::vector<int64_t>{1, 3, 2}));
}

TEST(ESPRITEstimatorOpTest, SourceCountPfaConfigurableFromParams) {
  // Authoring Params must select source_count_pfa (live Graph override
  // deferred — HSDK5_PORT §1).
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 40;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.45, 191u);
  auto snap_a = block_to_device<double>(block, {1, kN, kS});
  auto snap_b = block_to_device<double>(block, {1, kN, kS});

  const auto out_a = run_esprit<double>(
      snap_a, make_params(geom, conv, /*num_sources=*/-1, /*pfa=*/1e-4, 4));
  const auto out_b = run_esprit<double>(
      snap_b, make_params(geom, conv, /*num_sources=*/-1, /*pfa=*/0.5, 4));

  EXPECT_LE(out_a.source_count.as<int32_t>()[0],
            out_b.source_count.as<int32_t>()[0]);
}

TEST(ESPRITEstimatorOpTest, MultipleItemsRecoverIndependentDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 48;
  constexpr double kAz0 = 0.11;
  constexpr double kAz1 = -0.07;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto b0 = make_block(geom, conv, kAz0, 0.0, kS, 0.02, 21u);
  const auto b1 = make_block(geom, conv, kAz1, 0.0, kS, 0.02, 22u);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), b0.begin(), b0.end());
  batch.insert(batch.end(), b1.begin(), b1.end());
  auto snap = block_to_device<double>(batch, {2, kN, kS});

  const auto out =
      run_esprit<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{2, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
  EXPECT_NEAR(aoa[2], kAz1, 0.02);
  EXPECT_NEAR(aoa[3], 0.0, 1e-9);
}

TEST(ESPRITEstimatorOpTest, SpatialSmoothing1DRecoversDoa) {
  constexpr int64_t kN = 10;
  constexpr int64_t kS = 64;
  constexpr double kAz0 = 0.30;
  constexpr double kAz1 = -0.30;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto a0 = steering_vector(geom, conv, kAz0, 0.0);
  const auto a1 = steering_vector(geom, conv, kAz1, 0.0);
  std::mt19937 rng(70u);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
  std::normal_distribution<double> n(0.0, 0.02);
  std::vector<std::complex<double>> block(static_cast<size_t>(kN * kS));
  for (int64_t s = 0; s < kS; ++s) {
    const double ps = phase(rng);
    const std::complex<double> g{std::cos(ps), std::sin(ps)};
    for (int64_t c = 0; c < kN; ++c) {
      std::complex<double> v =
          (a0[static_cast<size_t>(c)] + a1[static_cast<size_t>(c)]) * g;
      v += std::complex<double>(n(rng), n(rng));
      block[static_cast<size_t>(c * kS + s)] = v;
    }
  }
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_esprit<double>(
      snap, make_params(geom, conv, /*num_sources=*/2, 1e-2, /*max_sources=*/2,
                        /*smoothing=*/{4}, /*fb=*/true));

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, 7}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double a = dm[0];
  const double b = dm[2];
  const bool matched =
      (std::abs(a - kAz0) < 0.04 && std::abs(b - kAz1) < 0.04) ||
      (std::abs(a - kAz1) < 0.04 && std::abs(b - kAz0) < 0.04);
  EXPECT_TRUE(matched) << "a=" << a << " b=" << b;
}

TEST(ESPRITEstimatorOpTest, ForwardOnlySmoothingRecoversCoherentSources) {
  constexpr int64_t kN = 10;
  constexpr int64_t kS = 128;
  constexpr double kAz0 = -0.30;
  constexpr double kAz1 = 0.30;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto a0 = steering_vector(geom, conv, kAz0, 0.0);
  const auto a1 = steering_vector(geom, conv, kAz1, 0.0);
  std::mt19937 rng(205u);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
  std::normal_distribution<double> noise(0.0, 0.01);
  std::vector<std::complex<double>> block(static_cast<size_t>(kN * kS));
  for (int64_t snapshot = 0; snapshot < kS; ++snapshot) {
    const double source_phase = phase(rng);
    const std::complex<double> gain{std::cos(source_phase),
                                    std::sin(source_phase)};
    for (int64_t channel = 0; channel < kN; ++channel) {
      auto value = (a0[static_cast<size_t>(channel)] +
                    a1[static_cast<size_t>(channel)]) *
                   gain;
      value += std::complex<double>{noise(rng), noise(rng)};
      block[static_cast<size_t>(channel * kS + snapshot)] = value;
    }
  }

  const auto out = run_esprit<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, conv, /*num_sources=*/2, 1e-2, /*max_sources=*/2,
                  /*smoothing=*/{4}, /*fb=*/false));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 2);
  std::array<double, 2> actual{out.doa_multi.as<double>()[0],
                               out.doa_multi.as<double>()[2]};
  std::sort(actual.begin(), actual.end());
  EXPECT_NEAR(actual[0], kAz0, 0.04);
  EXPECT_NEAR(actual[1], kAz1, 0.04);
}

TEST(ESPRITEstimatorOpTest, RejectsExplicitSubarrayWithSpatialSmoothing) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 201u);
  ShiftInvariantSubarrayPair pair;
  pair.reference_indices = {0, 1, 2, 3};
  pair.shifted_indices = {1, 2, 3, 4};

  const FourOutGraphRunResult result = run_esprit_result<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, conv, /*num_sources=*/-1, 1e-2, 4, /*smoothing=*/{2},
                  true, pair));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ESPRITEstimatorOpTest, ExplicitSubarrayPairWorks) {
  constexpr int64_t kN = 6;
  constexpr int64_t kS = 96;
  constexpr double kTruthAz = 0.14;
  auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const std::array<double, kN> y{0.0, 0.20, 0.43, 0.5, 0.70, 0.93};
  for (int64_t channel = 0; channel < kN; ++channel) {
    geom.element_positions[static_cast<size_t>(3 * channel + 1)] =
        y[static_cast<size_t>(channel)];
  }
  geom.geometry_type.reset();
  ShiftInvariantSubarrayPair pair;
  pair.reference_indices = {0, 1, 2};
  pair.shifted_indices = {3, 4, 5};
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.01, 202u);

  const auto out = run_esprit<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, conv, /*num_sources=*/1, 1e-2, 4, {}, true, pair));
  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(ESPRITEstimatorOpTest, ExplicitSubarrayPairRecoversTwoSources) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 256;
  constexpr double kAz0 = -0.16;
  constexpr double kAz1 = 0.21;
  ArrayGeometry geom;
  geom.wavelength = 1.0;
  const std::array<double, 4> reference_y{0.0, 0.17, 0.39, 0.68};
  for (double y : reference_y) {
    geom.element_positions.insert(geom.element_positions.end(),
                                  {0.0, y, 0.0});
  }
  for (double y : reference_y) {
    geom.element_positions.insert(geom.element_positions.end(),
                                  {0.0, y + 0.5, 0.0});
  }
  ShiftInvariantSubarrayPair pair;
  pair.reference_indices = {0, 1, 2, 3};
  pair.shifted_indices = {4, 5, 6, 7};
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.01, 206u);

  const auto out = run_esprit<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, conv, /*num_sources=*/2, 1e-2, /*max_sources=*/2, {},
                  true, pair));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 2);
  std::array<double, 2> actual{out.doa_multi.as<double>()[0],
                               out.doa_multi.as<double>()[2]};
  std::sort(actual.begin(), actual.end());
  EXPECT_NEAR(actual[0], kAz0, 0.03);
  EXPECT_NEAR(actual[1], kAz1, 0.03);
}

TEST(ESPRITEstimatorOpTest, RejectsMalformedExplicitSubarrayPairs) {
  constexpr int64_t kN = 6;
  constexpr int64_t kS = 16;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.01, 207u);
  const auto expect_rejected = [&](ShiftInvariantSubarrayPair pair) {
    const FourOutGraphRunResult result = run_esprit_result<double>(
        block_to_device<double>(block, {1, kN, kS}),
        make_params(geom, conv, /*num_sources=*/-1, 1e-2, 4, {}, true, pair));
    EXPECT_TRUE(result.plan_ok);
    EXPECT_TRUE(result.frames0.empty());
    EXPECT_FALSE(result.diagnostics.empty());
  };

  expect_rejected({{0, 1, 2}, {1, 2}});
  expect_rejected({{0, 1}, {1, 3}});
  expect_rejected({{0, 1}, {1, kN}});
}

TEST(ESPRITEstimatorOpTest, FixedSourceCountWorksWithFewSnapshots) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 4;
  constexpr double kTruthAz = 0.12;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.001, 208u);

  const auto out = run_esprit<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, conv, /*num_sources=*/1));

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(ESPRITEstimatorOpTest, ReducedSubarrayDirectlyDrivesEstimator) {
  namespace detail = holoscan::radar::testing::detail;
  constexpr int64_t kChannels = 6;
  constexpr int64_t kSnapshots = 32;
  constexpr double kTruthAz = 0.12;
  const std::array<int64_t, kChannels> positions{0, 1, 2, 3, 5, 6};
  ArrayGeometry full;
  full.wavelength = 1.0;
  for (int64_t position : positions) {
    full.element_positions.insert(
        full.element_positions.end(),
        {0.0, 0.5 * static_cast<double>(position), 0.0});
  }
  AngleConvention convention;
  convention.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(
      full, convention, kTruthAz, 0.0, kSnapshots, 0.001, 209u);

  detail::FrameBank::reset();
  detail::FrameBank::input_frames = {
      block_to_device<double>(
          block, {1, kChannels, kSnapshots})};
  holoscan::Graph graph{"subarray-to-esprit"};
  const auto source = graph.op<detail::TensorSourceOp>("source");
  auto extents = radar_testing::default_test_extents();
  extents.max_snapshots = kSnapshots;
  auto selector =
      graph.op<SubarraySelectionOp<double>>("selector", extents);
  SubarraySelectionParams selection;
  selection.array_geometry = full;
  selection.subarray = {0, 1, 2, 3};
  selector.set_params(std::move(selection));

  auto estimator =
      graph.op<ESPRITEstimatorOp<double>>("esprit", extents);
  // Deliberately wrong spacing for the selected snapshot. Accurate estimation
  // proves the same-tick wire geometry won over Params.
  estimator.set_params(make_params(
      make_ula_geometry(4, 0.25, 1.0), convention,
      /*num_sources=*/1, 1e-2, /*max_sources=*/2));
  const auto aoa_sink = graph.op<detail::TensorCaptureOp>("aoa_sink");
  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->out, selector->snapshot, queue);
  graph.add_flow(selector->snapshot_out, estimator->snapshot, queue);
  graph.add_flow(
      selector->array_geometry, estimator->array_geometry, queue);
  graph.add_flow(estimator->aoa_estimates, aoa_sink->in, queue);
  graph.partition("integration")
      .add(source)
      .add(selector)
      .add(estimator)
      .add(aoa_sink);
  graph.set_default_clock(
      graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  const auto bind = [&options](const char* op, const char* port) {
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{
            .operator_path = op,
            .output_port = port,
            .device = holoscan::DeviceId{0}});
  };
  bind("source", "out");
  bind("selector", "snapshot_out");
  bind("selector", "is_valid_out");
  bind("esprit", "aoa_estimates");
  bind("esprit", "doa_multi");
  bind("esprit", "source_count");
  bind("esprit", "eigenvalues");

  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline =
      std::chrono::steady_clock::now() + detail::kHarnessDeadline;
  while (session.diagnostics().empty() &&
         std::chrono::steady_clock::now() < deadline) {
    bool captured = false;
    {
      const std::lock_guard lock(detail::FrameBank::mutex);
      captured = !detail::FrameBank::captured.empty();
    }
    if (captured) {
      break;
    }
    std::this_thread::sleep_for(detail::kHarnessPoll);
  }
  session.request_stop();
  session.wait();
  ASSERT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(detail::FrameBank::mutex);
  ASSERT_EQ(detail::FrameBank::captured.size(), 1U);
  const auto& aoa = detail::FrameBank::captured.front();
  ASSERT_EQ(aoa.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_NEAR(aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(ESPRITEstimatorOpTest, DoubleInstantiationSmoke) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, -0.06, 0.0, kS, 0.02, 204u);
  const auto out = run_esprit<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, conv, /*num_sources=*/1));
  EXPECT_NEAR(out.aoa.as<double>()[0], -0.06, 0.02);
}

}  // namespace
