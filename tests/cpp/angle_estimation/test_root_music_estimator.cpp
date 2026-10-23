// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RootMUSICEstimatorOp contract tests (HSDK 5 Graph harness): known-DOA
// recovery, multi-source / eigenvalues / spatial smoothing / EFT Params
// coverage, float/double agreement, optional-mask wiring, live overrides,
// and error paths.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/estimator_test_fixtures.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::ArrayGeometryType;
using holoscan::radar::ElevationReference;
using holoscan::radar::RootMUSICEstimatorOp;
using holoscan::radar::RootMUSICParams;
using holoscan::radar::SpatialSmoothingOverride;
using holoscan::radar::SignedAxis;
using holoscan::radar::WireArrayGeometry;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::block_to_device;
using holoscan::radar::testing::kPi;
using holoscan::radar::testing::make_block;
using holoscan::radar::testing::make_block_two;
using holoscan::radar::testing::make_planar_ura_geometry;
using holoscan::radar::testing::make_ula_geometry;
using holoscan::radar::testing::steering_vector;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;

struct RootMusicOutputs {
  HostTensor aoa;
  HostTensor doa_multi;
  HostTensor source_count;
  HostTensor eigenvalues;
};

RootMUSICParams make_params(ArrayGeometry geom, AngleConvention conv, int64_t num_sources = -1,
                            double source_count_pfa = 1e-2, int64_t max_sources = 4,
                            std::vector<int64_t> smoothing = {}, bool forward_backward = true) {
  RootMUSICParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.num_sources = num_sources;
  params.source_count_pfa = source_count_pfa;
  params.max_sources = max_sources;
  params.spatial_smoothing = std::move(smoothing);
  params.forward_backward = forward_backward;
  return params;
}

// Noise-free block whose sample covariance is exactly V diag(`eigenvalues`)
// V^H, with V the unit steering vectors of `direction_cosines`. Spacing those
// cosines 2/num_channels apart makes V unitary on a half-wavelength ULA, so
// snapshot k carrying source k alone at power num_snapshots * eigenvalues[k] /
// num_channels reproduces the profile exactly. Pinning the eigenvalues that
// way turns the EFT source count into a pure function of the requested Pfa,
// while keeping the covariance physically realizable (unlike a synthetic
// diagonal, which leaves the rooting stage ill-conditioned).
std::vector<std::complex<double>> make_orthogonal_source_block(
    const ArrayGeometry& geometry, const AngleConvention& convention,
    const std::vector<double>& direction_cosines, const std::vector<double>& eigenvalues) {
  const int64_t channels = geometry.num_elements();
  const int64_t snapshots = static_cast<int64_t>(eigenvalues.size());
  std::vector<std::complex<double>> block(static_cast<size_t>(channels * snapshots));
  for (int64_t s = 0; s < snapshots; ++s) {
    const auto a =
        steering_vector(geometry, convention, direction_cosines[static_cast<size_t>(s)], 0.0);
    const double amplitude =
        std::sqrt(static_cast<double>(snapshots) * eigenvalues[static_cast<size_t>(s)] /
                  static_cast<double>(channels));
    for (int64_t c = 0; c < channels; ++c) {
      block[static_cast<size_t>(c * snapshots + s)] = a[static_cast<size_t>(c)] * amplitude;
    }
  }
  return block;
}

template <typename T>
holoscan::testing::TestResult run_root_music_result(std::shared_ptr<holoscan::Tensor> snapshot,
                                                    RootMUSICParams params) {
  using Op = RootMUSICEstimatorOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  fixture.push(&Op::snapshot, *snapshot);
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"aoa_estimates", "doa_multi", "source_count", "eigenvalues"};
  for (const char* output : kOutputs) {
    fixture.template capture_snapshot<holoscan::Tensor>(output, 1U, kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  for (const char* output : kOutputs) {
    (void)running.wait_for_output(output, 1U, std::chrono::seconds{5});
  }
  running.request_stop();
  return running.finish();
}

template <typename T>
RootMusicOutputs run_root_music(std::shared_ptr<holoscan::Tensor> snapshot,
                                RootMUSICParams params) {
  const auto result = run_root_music_result<T>(std::move(snapshot), std::move(params));
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return {
      holoscan::radar::testing::to_host(
          result.template snapshot<holoscan::Tensor>("aoa_estimates")),
      holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("doa_multi")),
      holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("source_count")),
      holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("eigenvalues"))};
}

template <typename Override, typename T = double>
holoscan::testing::TestResult run_root_music_override(
    std::vector<std::shared_ptr<holoscan::Tensor>> frames,
    std::vector<std::optional<Override>> payloads, RootMUSICParams params,
    const char* override_port) {
  using Op = RootMUSICEstimatorOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    if (payloads[index].has_value()) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input(override_port, *payloads[index]),
          holoscan::testing::input("snapshot", *frames[index])));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("snapshot", *frames[index])));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"aoa_estimates", "doa_multi", "source_count", "eigenvalues"};
  for (const char* output : kOutputs) {
    fixture.template capture_snapshot<holoscan::Tensor>(output, frames.size(), kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  for (const char* output : kOutputs) {
    (void)running.wait_for_output(output, frames.size(), std::chrono::seconds{5});
  }
  running.request_stop();
  return running.finish();
}

holoscan::testing::TestResult run_root_music_with_mask(std::shared_ptr<holoscan::Tensor> mask,
                                                       std::shared_ptr<holoscan::Tensor> snapshot,
                                                       RootMUSICParams params) {
  using Op = RootMUSICEstimatorOp<double>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::milliseconds{100},
                                                 holoscan::testing::input("is_valid", *mask),
                                                 holoscan::testing::input("snapshot", *snapshot)));
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"aoa_estimates", "doa_multi", "source_count", "eigenvalues"};
  for (const char* output : kOutputs) {
    fixture.capture_snapshot<holoscan::Tensor>(output, 1U, kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  fixture.input_source_device("is_valid", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  for (const char* output : kOutputs) {
    (void)running.wait_for_output(output, 1U, std::chrono::seconds{5});
  }
  running.request_stop();
  return running.finish();
}

std::vector<HostTensor> root_music_frames(const holoscan::testing::TestResult& result,
                                          const char* port) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.snapshots<holoscan::Tensor>(port)) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

TEST(RootMUSICEstimatorOpTest, DefaultsToRadarBoresightConvention) {
  EXPECT_EQ(RootMUSICParams{}.angle_convention.preset, AngleConventionPreset::kRadarBoresight);
}

TEST(RootMUSICEstimatorOpTest, RecoversKnownDoaUlaFixedOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 3u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, RecoversTwoSources) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 256;
  constexpr double kAz0 = 0.20;
  constexpr double kAz1 = -0.15;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.02, 71u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/2, 1e-2,
                                                            /*max_sources=*/2));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double az_a = dm[0];
  const double az_b = dm[2];
  const bool matched = (std::abs(az_a - kAz0) < 0.03 && std::abs(az_b - kAz1) < 0.03) ||
                       (std::abs(az_a - kAz1) < 0.03 && std::abs(az_b - kAz0) < 0.03);
  EXPECT_TRUE(matched) << "az_a=" << az_a << " az_b=" << az_b;

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(dm[0], aoa[0], 1e-9);
  EXPECT_NEAR(dm[1], aoa[1], 1e-9);
}

TEST(RootMUSICEstimatorOpTest, ResolvesCloselySpacedSources) {
  constexpr int64_t kN = 16;
  constexpr int64_t kS = 512;
  constexpr double kAz0 = -0.035;
  constexpr double kAz1 = 0.035;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.005, 73u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/2, 1e-2,
                                                            /*max_sources=*/2));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 2);
  std::array<double, 2> actual{
      out.doa_multi.as<double>()[0],
      out.doa_multi.as<double>()[2],
  };
  std::sort(actual.begin(), actual.end());
  EXPECT_NEAR(actual[0], kAz0, 0.01);
  EXPECT_NEAR(actual[1], kAz1, 0.01);
}

TEST(RootMUSICEstimatorOpTest, MaxSourcesCapsReportedRoots) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 256;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block_two(geom, conv, -0.15, 0.0, 0.20, 0.0, kS, 0.02, 78u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/2, 1e-2,
                                                            /*max_sources=*/1));

  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
  EXPECT_TRUE(std::isfinite(out.doa_multi.as<double>()[0]));
}

TEST(RootMUSICEstimatorOpTest, HighSnrLargeArrayRootRemainsFinite) {
  constexpr int64_t kN = 32;
  constexpr int64_t kS = 128;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 1e-6, 74u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double az = out.aoa.as<double>()[0];
  EXPECT_TRUE(std::isfinite(az));
  EXPECT_NEAR(az, kTruthAz, 1e-4);
}

TEST(RootMUSICEstimatorOpTest, AutoNumSourcesRecoversOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 9u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/-1, 1e-2,
                                                            /*max_sources=*/4));

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
}

TEST(RootMUSICEstimatorOpTest, RecoversUraSingleSourceAzEl) {
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

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz, 0.03);
  EXPECT_NEAR(aoa[1], kEl, 0.03);
}

TEST(RootMUSICEstimatorOpTest, RecoversUraTwoSourcesAzElPairing) {
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
  const auto block = make_block_two(geom, conv, kAz0, kEl0, kAz1, kEl1, kS, 0.01, 79u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/2, 1e-2,
                                                            /*max_sources=*/2));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const auto near_pair = [](double az, double el, double taz, double tel) {
    return std::abs(az - taz) < 0.04 && std::abs(el - tel) < 0.04;
  };
  const bool matched =
      (near_pair(dm[0], dm[1], kAz0, kEl0) && near_pair(dm[2], dm[3], kAz1, kEl1)) ||
      (near_pair(dm[0], dm[1], kAz1, kEl1) && near_pair(dm[2], dm[3], kAz0, kEl0));
  EXPECT_TRUE(matched) << "pair0=(" << dm[0] << "," << dm[1] << ") pair1=(" << dm[2] << "," << dm[3]
                       << ")";
}

TEST(RootMUSICEstimatorOpTest, RecoversUraTwoSourcesWithCommonElevation) {
  constexpr int64_t kNy = 8;
  constexpr int64_t kNz = 8;
  constexpr int64_t kS = 512;
  constexpr double kAz0 = -0.15;
  constexpr double kAz1 = 0.20;
  constexpr double kEl = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block_two(geom, conv, kAz0, kEl, kAz1, kEl, kS, 0.01, 83u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});
  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/2, 1e-2,
                                                            /*max_sources=*/2));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 2);
  const double* dm = out.doa_multi.as<double>();
  std::array<double, 2> az{dm[0], dm[2]};
  std::sort(az.begin(), az.end());
  EXPECT_NEAR(az[0], kAz0, 0.04);
  EXPECT_NEAR(az[1], kAz1, 0.04);
  EXPECT_NEAR(dm[1], kEl, 0.04);
  EXPECT_NEAR(dm[3], kEl, 0.04);
}

TEST(RootMUSICEstimatorOpTest, RecoversUraSingleSourceAzElFloat) {
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

  const auto out = run_root_music<float>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], kAz, 0.03);
  EXPECT_NEAR(aoa[1], kEl, 0.03);
}

TEST(RootMUSICEstimatorOpTest, RejectsNonUniformLinearGeometry) {
  constexpr int64_t kN = 5;
  constexpr int64_t kS = 16;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  auto geom = make_ula_geometry(kN, 0.5, 1.0);
  geom.element_positions[3 * 3 + 1] = 1.65;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 75u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto result = run_root_music_result<double>(snap, make_params(geom, conv));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, RejectsUlaParallelToBoresight) {
  constexpr int64_t kN = 4;
  ArrayGeometry geom;
  geom.wavelength = 1.0;
  geom.geometry_type = ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < kN; ++channel) {
    geom.element_positions.insert(geom.element_positions.end(),
                                  {0.5 * static_cast<double>(channel), 0.0, 0.0});
  }
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> block(static_cast<size_t>(kN * 4), {1.0, 0.0});
  auto snap = block_to_device<double>(block, {1, kN, 4});

  const auto result = run_root_music_result<double>(snap, make_params(geom, conv));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, EigenvaluesAscendingAndNonNegative) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 128;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 91u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));

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

TEST(RootMUSICEstimatorOpTest, RankDeficientAutoFallsBackToOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 4;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 12u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/-1, 1e-2,
                                                            /*max_sources=*/4));

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
}

TEST(RootMUSICEstimatorOpTest, RejectsNumSourcesTooLarge) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});

  const auto result =
      run_root_music_result<double>(snap, make_params(geom, conv, /*num_sources=*/kN));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, RejectsInvalidMaxSources) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});
  const auto result =
      run_root_music_result<double>(snap, make_params(geom, conv, /*num_sources=*/-1, 1e-2,
                                                      /*max_sources=*/0));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, RejectsInvalidSourceCountPfa) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});
  const auto result =
      run_root_music_result<double>(snap, make_params(geom, conv, /*num_sources=*/-1, /*pfa=*/0.0));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap = to_device_tensor(host, {0, kN, 4}, complex_dtype<double>());

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/-1, 1e-2,
                                                            /*max_sources=*/3));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{0, 3, 2}));
  EXPECT_EQ(out.source_count.shape, (std::vector<int64_t>{0}));
  EXPECT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{0, kN}));
}

TEST(RootMUSICEstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>((kN - 1) * 4), {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN - 1, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto result = run_root_music_result<double>(snap, make_params(geom, conv));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_f = block_to_device<float>(a, {1, kN, 1});
  const auto result = run_root_music_result<double>(snap_f, make_params(geom, conv));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, RejectsSnapshotRank2) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.05, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});
  const auto result = run_root_music_result<double>(snap, make_params(geom, conv));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, AcceptsPinnedHostSnapshot) {
  // Graph harness always copies inputs to device; keep as device-path smoke
  // (pinned-host Tensor path not exercisable via FourOut harness).
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 161u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));
  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(RootMUSICEstimatorOpTest, AcceptsManagedSnapshot) {
  // Graph harness always copies inputs to device; keep as device-path smoke
  // (managed Tensor path not exercisable via FourOut harness).
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 162u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));
  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(RootMUSICEstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.06, 0.0, kS, 0.02, 11u);
  auto snap = block_to_device<float>(block, {1, kN, kS});

  const auto out = run_root_music<float>(snap, make_params(geom, conv, /*num_sources=*/1));
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.06f, 0.03f);
}

TEST(RootMUSICEstimatorOpTest, FloatDoubleAgreeOnDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.09;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 5u);
  auto snap_f = block_to_device<float>(block, {1, kN, kS});
  auto snap_d = block_to_device<double>(block, {1, kN, kS});

  RootMUSICParams params = make_params(geom, conv, /*num_sources=*/1);

  const auto out_f = run_root_music<float>(snap_f, params);
  const auto out_d = run_root_music<double>(snap_d, params);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 1e-2f);
}

TEST(RootMUSICEstimatorOpTest, DeterministicAcrossRuns) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.09, 0.0, kS, 0.02, 171u);
  RootMUSICParams params = make_params(geom, conv, /*num_sources=*/1, 1e-2, /*max_sources=*/2);

  const auto run = [&]() {
    return run_root_music<double>(block_to_device<double>(block, {1, kN, kS}), params);
  };
  const auto first = run();
  const auto second = run();

  EXPECT_EQ(first.aoa.data, second.aoa.data);
  EXPECT_EQ(first.doa_multi.data, second.doa_multi.data);
  EXPECT_EQ(first.source_count.data, second.source_count.data);
  EXPECT_EQ(first.eigenvalues.data, second.eigenvalues.data);
}

TEST(RootMUSICEstimatorOpTest, DirectionCosineConventionMapsUlaProjection) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthU = 0.23;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthU, 0.0, kS, 0.01, 76u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthU, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], 0.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, CustomConventionMapsSignedAxes) {
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
  const auto block = make_block(geom, radar, kCanonicalAz, 0.0, kS, 0.01, 176u);

  const auto out = run_root_music<double>(block_to_device<double>(block, {1, kN, kS}),
                                          make_params(geom, custom, /*num_sources=*/1));

  EXPECT_NEAR(out.aoa.as<double>()[0], -kCanonicalAz, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], 0.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, CustomConventionSupportsZenithReference) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  AngleConvention radar;
  radar.preset = AngleConventionPreset::kRadarBoresight;
  AngleConvention custom;
  custom.preset = AngleConventionPreset::kCustom;
  custom.elevation_reference = ElevationReference::kZenith;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, radar, 0.1, 0.0, kS, 0.01, 177u);

  const auto out = run_root_music<double>(block_to_device<double>(block, {1, kN, kS}),
                                          make_params(geom, custom, /*num_sources=*/1));

  EXPECT_NEAR(out.aoa.as<double>()[0], 0.1, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], kPi / 2.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, RejectsDegenerateCustomConvention) {
  constexpr int64_t kN = 4;
  AngleConvention radar;
  radar.preset = AngleConventionPreset::kRadarBoresight;
  AngleConvention custom;
  custom.preset = AngleConventionPreset::kCustom;
  custom.azimuth_positive_axis = custom.boresight_axis;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, radar, 0.1, 0.0, 8, 0.01, 178u);

  const auto result = run_root_music_result<double>(block_to_device<double>(block, {1, kN, 8}),
                                                    make_params(geom, custom, /*num_sources=*/1));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(root_music_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, DirectionCosineNearEndfirePreservesSign) {
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

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  EXPECT_NEAR(out.aoa.as<double>()[0], kU0, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[2], kU1, 0.01);
}

TEST(RootMUSICEstimatorOpTest, UnusedDoaRowsAreNan) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.01, 77u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1, 1e-2,
                                                            /*max_sources=*/3));

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double* doa = out.doa_multi.as<double>();
  EXPECT_TRUE(std::isfinite(doa[0]));
  EXPECT_TRUE(std::isfinite(doa[1]));
  for (int64_t index = 2; index < 6; ++index) {
    EXPECT_TRUE(std::isnan(doa[index]));
  }
}

TEST(RootMUSICEstimatorOpTest, IsValidMaskExcludesWrongSnapshots) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.14;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto truth = make_block(geometry, conv, kTruthAz, 0.0, kS, 0.01, 701u);
  const auto wrong = make_block(geometry, conv, -0.24, 0.0, kS, 0.01, 702u);
  std::vector<std::complex<double>> mixed(static_cast<size_t>(kN * 2 * kS));
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
  auto mask = to_device_tensor(mask_host, {1, 2 * kS}, DLDataType{kDLUInt, 8, 1});

  const auto result =
      run_root_music_with_mask(mask, snapshot, make_params(geometry, conv, /*num_sources=*/1));
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto aoa = root_music_frames(result, "aoa_estimates");
  ASSERT_EQ(aoa.size(), 1U);
  EXPECT_NEAR(aoa[0].as<double>()[0], kTruthAz, 0.03);
}

TEST(RootMUSICEstimatorOpTest, AllInvalidMaskBlanksDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 16;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geometry, conv, 0.1, 0.0, kS, 0.01, 703u);
  auto snapshot = block_to_device<double>(block, {1, kN, kS});
  auto mask =
      to_device_tensor(std::vector<uint8_t>(kS, uint8_t{0}), {1, kS}, DLDataType{kDLUInt, 8, 1});

  const auto result =
      run_root_music_with_mask(mask, snapshot, make_params(geometry, conv, /*num_sources=*/1));
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto aoa = root_music_frames(result, "aoa_estimates");
  const auto source_count = root_music_frames(result, "source_count");
  ASSERT_EQ(aoa.size(), 1U);
  ASSERT_EQ(source_count.size(), 1U);
  EXPECT_TRUE(std::isnan(aoa[0].as<double>()[0]));
  EXPECT_EQ(source_count[0].as<int32_t>()[0], 0);
}

// num_sources is reported verbatim on source_count (clamped by max_sources),
// so an int32 output value carries the override directly. Ticks 0 and 2 send
// nothing and pin the Params baseline inside the same run, so nothing is
// compared across graph runs.
TEST(RootMUSICEstimatorOpTest, LiveNumSourcesAppliesForOneTick) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 256;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block_two(geometry, conv, 0.20, 0.0, -0.15, 0.0, kS, 0.02, 311u);
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    frames.push_back(block_to_device<double>(block, {1, kN, kS}));
  }
  RootMUSICParams params = make_params(geometry, conv, /*num_sources=*/1, 1e-2,
                                       /*max_sources=*/4);
  std::vector<std::optional<int64_t>> payloads{std::nullopt, int64_t{2}, std::nullopt};

  const auto result = run_root_music_override(std::move(frames), std::move(payloads),
                                              std::move(params), "num_sources");
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(root_music_frames(result, "source_count").size(), 3U);
  EXPECT_EQ(root_music_frames(result, "source_count")[0].as<int32_t>()[0], 1);
  EXPECT_EQ(root_music_frames(result, "source_count")[1].as<int32_t>()[0], 2);
  EXPECT_EQ(root_music_frames(result, "source_count")[2].as<int32_t>()[0], 1);
}

TEST(RootMUSICEstimatorOpTest, LiveArrayGeometryAppliesForOneTick) {
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
  RootMUSICParams params = make_params(make_ula_geometry(kN, 0.5, 0.8), conv,
                                       /*num_sources=*/1);
  std::vector<std::optional<WireArrayGeometry>> payloads;
  payloads.emplace_back(std::nullopt);
  payloads.emplace_back(holoscan::radar::to_wire(geometry));
  payloads.emplace_back(std::nullopt);

  const auto result = run_root_music_override(std::move(frames), std::move(payloads),
                                              std::move(params), "array_geometry");
  EXPECT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(root_music_frames(result, "aoa_estimates").size(), 3U);
  EXPECT_NE(root_music_frames(result, "aoa_estimates")[1].data,
            root_music_frames(result, "aoa_estimates")[0].data);
  EXPECT_EQ(root_music_frames(result, "aoa_estimates")[2].data,
            root_music_frames(result, "aoa_estimates")[0].data);

  RootMUSICParams invalid_params = make_params(geometry, conv, /*num_sources=*/1);
  WireArrayGeometry malformed_geometry = holoscan::radar::to_wire(geometry);
  malformed_geometry.packed_geometry =
      to_pageable_host_tensor<double>({1.0}, {1}, real_dtype<double>());
  std::vector<std::optional<WireArrayGeometry>> malformed_payloads;
  malformed_payloads.emplace_back(std::move(malformed_geometry));
  const auto invalid = run_root_music_override(
      std::vector<std::shared_ptr<holoscan::Tensor>>{block_to_device<double>(block, {1, kN, kS})},
      std::move(malformed_payloads), std::move(invalid_params), "array_geometry");
  EXPECT_FALSE(invalid.ok());
  EXPECT_TRUE(root_music_frames(invalid, "aoa_estimates").empty());
  EXPECT_FALSE(invalid.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, LiveSpatialSmoothingAppliesForOneTick) {
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
  RootMUSICParams params = make_params(geometry, conv, /*num_sources=*/1, 1e-2, 4,
                                       /*smoothing=*/{2});
  std::vector<std::optional<SpatialSmoothingOverride>> payloads{
      std::nullopt,
      SpatialSmoothingOverride{.count = 1, .first = 4},
      std::nullopt,
  };

  const auto result = run_root_music_override(std::move(frames), std::move(payloads),
                                              std::move(params), "spatial_smoothing");
  EXPECT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(root_music_frames(result, "eigenvalues").size(), 3U);
  // Every tick is smoothed. The override grows the smoothing count and
  // shrinks the effective aperture; tick 2 reuses the Params-sized buffers.
  EXPECT_NE(root_music_frames(result, "eigenvalues")[1].shape,
            root_music_frames(result, "eigenvalues")[0].shape);
  EXPECT_EQ(root_music_frames(result, "eigenvalues")[2].shape,
            root_music_frames(result, "eigenvalues")[0].shape);

  RootMUSICParams invalid_params = make_params(geometry, conv, /*num_sources=*/1);
  const auto invalid = run_root_music_override(
      std::vector<std::shared_ptr<holoscan::Tensor>>{block_to_device<double>(block, {1, kN, kS})},
      std::vector<std::optional<SpatialSmoothingOverride>>{SpatialSmoothingOverride{.count = 3}},
      std::move(invalid_params), "spatial_smoothing");
  EXPECT_FALSE(invalid.ok());
  EXPECT_TRUE(root_music_frames(invalid, "aoa_estimates").empty());
  EXPECT_FALSE(invalid.diagnostics().empty());
}

TEST(RootMUSICEstimatorOpTest, RepeatedUraSmoothingSupportsChangingEffectiveAperture) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 64;
  constexpr double kAz = 0.12;
  constexpr double kEl = -0.08;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geometry = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geometry, conv, kAz, kEl, kS, 0.02, 303u);
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    frames.push_back(block_to_device<double>(block, {1, geometry.num_elements(), kS}));
  }
  RootMUSICParams params = make_params(geometry, conv, /*num_sources=*/1, 1e-2, 4,
                                       /*smoothing=*/{1, 1});
  std::vector<std::optional<SpatialSmoothingOverride>> payloads{
      std::nullopt,
      SpatialSmoothingOverride{.count = 2, .first = 2, .second = 2},
      std::nullopt,
  };

  const auto result = run_root_music_override(std::move(frames), std::move(payloads),
                                              std::move(params), "spatial_smoothing");
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(root_music_frames(result, "aoa_estimates").size(), 3U);
  ASSERT_EQ(root_music_frames(result, "eigenvalues").size(), 3U);
  EXPECT_EQ(root_music_frames(result, "eigenvalues")[0].shape, (std::vector<int64_t>{1, 16}));
  EXPECT_EQ(root_music_frames(result, "eigenvalues")[1].shape, (std::vector<int64_t>{1, 9}));
  EXPECT_EQ(root_music_frames(result, "eigenvalues")[2].shape,
            root_music_frames(result, "eigenvalues")[0].shape);
  for (const auto& frame : root_music_frames(result, "aoa_estimates")) {
    EXPECT_TRUE(std::isfinite(frame.as<double>()[0]));
    EXPECT_TRUE(std::isfinite(frame.as<double>()[1]));
  }
  EXPECT_EQ(root_music_frames(result, "source_count")[0].as<int32_t>()[0], 1);
  EXPECT_EQ(root_music_frames(result, "source_count")[1].as<int32_t>()[0], 1);
  EXPECT_EQ(root_music_frames(result, "source_count")[2].as<int32_t>()[0], 1);
  EXPECT_EQ(root_music_frames(result, "aoa_estimates")[2].data,
            root_music_frames(result, "aoa_estimates")[0].data);
}

// source_count_pfa only steers the EFT model order, so the observable is the
// int32 source_count. The crafted covariance pins the eigenvalue profile
// exactly, which makes the 1 -> 2 transition between the Params Pfa and the
// override Pfa deterministic instead of a noise draw.
TEST(RootMUSICEstimatorOpTest, LiveSourceCountPfaAppliesForOneTick) {
  const std::vector<double> spectrum = {0.8930540251, 0.9468277841, 1.3517214181,
                                        1.6732266229, 9.0798471606, 14.3477610683};
  const std::vector<double> cosines = {-5.0 / 6.0, -0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5, 5.0 / 6.0};
  const int64_t kN = static_cast<int64_t>(spectrum.size());
  const int64_t kS = kN;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geometry = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_orthogonal_source_block(geometry, conv, cosines, spectrum);
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    frames.push_back(block_to_device<double>(block, {1, kN, kS}));
  }
  RootMUSICParams params = make_params(geometry, conv, /*num_sources=*/-1, /*pfa=*/1e-4,
                                       /*max_sources=*/4, /*smoothing=*/{},
                                       /*forward_backward=*/false);
  std::vector<std::optional<double>> payloads{std::nullopt, 0.5, std::nullopt};

  const auto result = run_root_music_override(std::move(frames), std::move(payloads),
                                              std::move(params), "source_count_pfa");
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(root_music_frames(result, "source_count").size(), 3U);
  EXPECT_EQ(root_music_frames(result, "source_count")[0].as<int32_t>()[0], 1);
  EXPECT_EQ(root_music_frames(result, "source_count")[1].as<int32_t>()[0], 2);
  EXPECT_EQ(root_music_frames(result, "source_count")[2].as<int32_t>()[0], 1);
}

// max_sources widens doa_multi to {items, max_sources, 2}, so the override
// shows up in the output shape alone.
TEST(RootMUSICEstimatorOpTest, LiveMaxSourcesAppliesForOneTick) {
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
  RootMUSICParams params = make_params(geometry, conv, /*num_sources=*/1, 1e-2,
                                       /*max_sources=*/1);
  std::vector<std::optional<int64_t>> payloads{std::nullopt, kLiveMaxSources, std::nullopt};

  const auto result = run_root_music_override(std::move(frames), std::move(payloads),
                                              std::move(params), "max_sources");
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(root_music_frames(result, "doa_multi").size(), 3U);
  EXPECT_EQ(root_music_frames(result, "doa_multi")[0].shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(root_music_frames(result, "doa_multi")[1].shape,
            (std::vector<int64_t>{1, kLiveMaxSources, 2}));
  // Tick 2 reverts to the Params max_sources, narrowing doa_multi back.
  EXPECT_EQ(root_music_frames(result, "doa_multi")[2].shape, (std::vector<int64_t>{1, 1, 2}));
}

TEST(RootMUSICEstimatorOpTest, NumSourcesConfigurableFromParams) {
  // Exercise the authoring-time Params value for num_sources.
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 128;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, kS, 0.02, 111u);
  auto snap1 = block_to_device<double>(block, {1, kN, kS});
  auto snap2 = block_to_device<double>(block, {1, kN, kS});

  const auto out1 = run_root_music<double>(snap1, make_params(geom, conv, /*num_sources=*/1, 1e-2,
                                                              /*max_sources=*/4));
  const auto out2 = run_root_music<double>(snap2, make_params(geom, conv, /*num_sources=*/2, 1e-2,
                                                              /*max_sources=*/4));

  EXPECT_EQ(out1.source_count.as<int32_t>()[0], 1);
  EXPECT_EQ(out2.source_count.as<int32_t>()[0], 2);
}

TEST(RootMUSICEstimatorOpTest, MaxSourcesConfigurableFromParams) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 101u);
  auto snap1 = block_to_device<double>(block, {1, kN, kS});
  auto snap3 = block_to_device<double>(block, {1, kN, kS});

  const auto out1 = run_root_music<double>(snap1, make_params(geom, conv, /*num_sources=*/1, 1e-2,
                                                              /*max_sources=*/1));
  const auto out3 = run_root_music<double>(snap3, make_params(geom, conv, /*num_sources=*/1, 1e-2,
                                                              /*max_sources=*/3));

  EXPECT_EQ(out1.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out3.doa_multi.shape, (std::vector<int64_t>{1, 3, 2}));
}

TEST(RootMUSICEstimatorOpTest, SourceCountPfaConfigurableFromParams) {
  // Exercise the authoring-time Params value for source_count_pfa.
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 40;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.45, 191u);
  auto snap_a = block_to_device<double>(block, {1, kN, kS});
  auto snap_b = block_to_device<double>(block, {1, kN, kS});

  const auto out_a =
      run_root_music<double>(snap_a, make_params(geom, conv, /*num_sources=*/-1, /*pfa=*/1e-4,
                                                 /*max_sources=*/4));
  const auto out_b =
      run_root_music<double>(snap_b, make_params(geom, conv, /*num_sources=*/-1, /*pfa=*/0.5,
                                                 /*max_sources=*/4));

  const int32_t tight = out_a.source_count.as<int32_t>()[0];
  const int32_t loose = out_b.source_count.as<int32_t>()[0];
  EXPECT_LE(tight, loose);
}

TEST(RootMUSICEstimatorOpTest, MultipleItemsRecoverIndependentDoa) {
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

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{2, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
  EXPECT_NEAR(aoa[2], kAz1, 0.02);
  EXPECT_NEAR(aoa[3], 0.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, SpatialSmoothing1DRecoversDoa) {
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
      std::complex<double> v = (a0[static_cast<size_t>(c)] + a1[static_cast<size_t>(c)]) * g;
      v += std::complex<double>(n(rng), n(rng));
      block[static_cast<size_t>(c * kS + s)] = v;
    }
  }
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(snap, make_params(geom, conv, /*num_sources=*/2, 1e-2,
                                                            /*max_sources=*/2, /*smoothing=*/{4},
                                                            /*forward_backward=*/true));

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, 7}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double a = dm[0];
  const double b = dm[2];
  const bool matched = (std::abs(a - kAz0) < 0.04 && std::abs(b - kAz1) < 0.04) ||
                       (std::abs(a - kAz1) < 0.04 && std::abs(b - kAz0) < 0.04);
  EXPECT_TRUE(matched) << "a=" << a << " b=" << b;
}

}  // namespace

// Campaign reproducer: finite equal-channel IQ at three amplitudes. Float
// roundoff can change the EFT-selected null-space basis on the second frame,
// giving an exactly zero leading coefficient. Accept either a blanked item or
// finite AoA; later frames must still execute with the same operator and solver state.
TEST(RootMUSICEstimatorOpTest, FloatScaledRankOneFramesDoNotAbort) {
  constexpr int64_t kChannels = 4;
  constexpr int64_t kSnapshots = 16;
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (int tick = 0; tick < 3; ++tick) {
    std::vector<std::complex<float>> block(kChannels * kSnapshots);
    for (int64_t channel = 0; channel < kChannels; ++channel) {
      for (int64_t sample = 0; sample < kSnapshots; ++sample) {
        const double phase1 = 2.0 * kPi * 5.0 * static_cast<double>(sample) / kSnapshots;
        const double phase2 = 2.0 * kPi * 17.0 * static_cast<double>(sample) / kSnapshots;
        const float scale = 1.0F + static_cast<float>(tick) * 0.125F;
        block[channel * kSnapshots + sample] =
            std::complex<float>{static_cast<float>(std::cos(phase1) + 0.5 * std::cos(phase2) + 0.1),
                                static_cast<float>(std::sin(phase1) + 0.5 * std::sin(phase2))} *
            scale;
      }
    }
    frames.push_back(to_device_tensor(block, {1, kChannels, kSnapshots}, complex_dtype<float>()));
  }
  auto params = make_params(make_ula_geometry(kChannels, 0.5, 1.0), AngleConvention{});
  const auto result = run_root_music_override<int64_t, float>(
      std::move(frames), std::vector<std::optional<int64_t>>(3), std::move(params), "num_sources");
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto counts = root_music_frames(result, "source_count");
  const auto aoa = root_music_frames(result, "aoa_estimates");
  const auto eigenvalues = root_music_frames(result, "eigenvalues");
  ASSERT_EQ(counts.size(), 3U);
  ASSERT_EQ(aoa.size(), 3U);
  ASSERT_EQ(eigenvalues.size(), 3U);
  EXPECT_GT(counts[0].as<int32_t>()[0], 0);
  const int32_t second_count = counts[1].as<int32_t>()[0];
  EXPECT_GE(second_count, 0);
  if (second_count == 0) {
    EXPECT_TRUE(std::isnan(aoa[1].as<float>()[0]));
    EXPECT_TRUE(std::isnan(aoa[1].as<float>()[1]));
  } else {
    EXPECT_TRUE(std::isfinite(aoa[1].as<float>()[0]));
    EXPECT_TRUE(std::isfinite(aoa[1].as<float>()[1]));
  }
  EXPECT_GT(counts[2].as<int32_t>()[0], 0);
  for (const auto& values : eigenvalues) {
    for (int64_t channel = 0; channel < kChannels; ++channel) {
      EXPECT_TRUE(std::isfinite(values.as<float>()[channel]));
    }
  }
}
