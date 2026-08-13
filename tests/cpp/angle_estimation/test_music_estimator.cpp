// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MUSICEstimatorOp contract tests (HSDK 5 Graph harness): known-DOA recovery,
// multi-source / eigenvalues / spatial smoothing / EFT Params coverage,
// Beamscan interchange + super-resolution, float/double agreement, and error
// paths. Live / is_valid Graph wiring deferred.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/music_estimator_op/music_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::BeamscanEstimatorOp;
using holoscan::radar::BeamscanParams;
using holoscan::radar::MUSICEstimatorOp;
using holoscan::radar::MUSICParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::kPi;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::FiveOutGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;
using holoscan::radar::testing::run_single_in_five_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;

ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing,
                                double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = holoscan::radar::ArrayGeometryType::kUla;
  geom.element_positions.reserve(static_cast<size_t>(num_elements * 3));
  for (int64_t m = 0; m < num_elements; ++m) {
    geom.element_positions.push_back(0.0);
    geom.element_positions.push_back(static_cast<double>(m) * spacing);
    geom.element_positions.push_back(0.0);
  }
  return geom;
}

ArrayGeometry make_planar_ura_geometry(int64_t ny, int64_t nz, double spacing,
                                       double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = holoscan::radar::ArrayGeometryType::kUra;
  geom.element_positions.reserve(static_cast<size_t>(ny * nz * 3));
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      geom.element_positions.push_back(0.0);
      geom.element_positions.push_back(static_cast<double>(iy) * spacing);
      geom.element_positions.push_back(static_cast<double>(iz) * spacing);
    }
  }
  return geom;
}

std::vector<std::complex<double>> steering_vector(const ArrayGeometry& geom,
                                                  const AngleConvention& conv,
                                                  double az, double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(conv, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> a(
      static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase = k * (geom.element_positions[base] * dir.x +
                              geom.element_positions[base + 1] * dir.y +
                              geom.element_positions[base + 2] * dir.z);
    a[static_cast<size_t>(m)] = {std::cos(phase), std::sin(phase)};
  }
  return a;
}

std::vector<std::complex<double>> make_block(const ArrayGeometry& geom,
                                             const AngleConvention& conv,
                                             double az, double el, int64_t S,
                                             double noise, unsigned seed) {
  const auto a = steering_vector(geom, conv, az, el);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double gp =
        2.0 * kPi * static_cast<double>(s) / static_cast<double>(S);
    const std::complex<double> g{std::cos(gp), std::sin(gp)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v = a[static_cast<size_t>(c)] * g;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
}

std::vector<std::complex<double>> make_block_two(
    const ArrayGeometry& geom, const AngleConvention& conv, double az0,
    double el0, double az1, double el1, int64_t S, double noise,
    unsigned seed) {
  const auto a0 = steering_vector(geom, conv, az0, el0);
  const auto a1 = steering_vector(geom, conv, az1, el1);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double p0 = phase(rng);
    const double p1 = phase(rng);
    const std::complex<double> g0{std::cos(p0), std::sin(p0)};
    const std::complex<double> g1{std::cos(p1), std::sin(p1)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v =
          a0[static_cast<size_t>(c)] * g0 + a1[static_cast<size_t>(c)] * g1;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
}

std::vector<double> linspace(double start, double stop, int64_t n) {
  std::vector<double> out(static_cast<size_t>(n));
  if (n == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(n - 1);
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = start + step * static_cast<double>(i);
  }
  return out;
}

double spectrum_max(const HostTensor& spec) {
  const double* values = spec.as<double>();
  const int64_t n = spec.size();
  double best = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    best = std::max(best, values[static_cast<size_t>(i)]);
  }
  return best;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> block_to_device(
    const std::vector<std::complex<double>>& host,
    const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    cast[i] = {static_cast<T>(host[i].real()), static_cast<T>(host[i].imag())};
  }
  return to_device_tensor(cast, shape, complex_dtype<T>());
}

struct MusicOutputs {
  HostTensor aoa;
  HostTensor spectrum;
  HostTensor doa_multi;
  HostTensor source_count;
  HostTensor eigenvalues;
};

template <typename Source, typename OpRef, typename S0, typename S1,
          typename S2, typename S3, typename S4>
void wire_music(holoscan::Graph& graph, const Source& source, OpRef& op,
                const S0& sink0, const S1& sink1, const S2& sink2,
                const S3& sink3, const S4& sink4) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->snapshot, q);
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->angle_spectrum, sink1->in, q);
  graph.add_flow(op->doa_multi, sink2->in, q);
  graph.add_flow(op->source_count, sink3->in, q);
  graph.add_flow(op->eigenvalues, sink4->in, q);
}

MUSICParams make_params(ArrayGeometry geom, AngleConvention conv,
                        std::vector<double> az, std::vector<double> el,
                        int64_t num_sources = -1,
                        double source_count_pfa = 1e-2,
                        int64_t max_sources = 4,
                        std::vector<int64_t> smoothing = {},
                        bool forward_backward = true) {
  MUSICParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.az_angles = std::move(az);
  params.el_angles = std::move(el);
  params.num_sources = num_sources;
  params.source_count_pfa = source_count_pfa;
  params.max_sources = max_sources;
  params.spatial_smoothing = std::move(smoothing);
  params.forward_backward = forward_backward;
  return params;
}

template <typename T>
FiveOutGraphRunResult run_music_result(std::shared_ptr<holoscan::Tensor> snapshot,
                                       MUSICParams params) {
  return run_single_in_five_out_operator_graph_result<MUSICEstimatorOp<T>>(
      {std::move(snapshot)}, std::move(params), "aoa_estimates",
      "angle_spectrum", "doa_multi", "source_count", "eigenvalues",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1, const auto& sink2,
         const auto& sink3, const auto& sink4) {
        wire_music(graph, source, op, sink0, sink1, sink2, sink3, sink4);
      });
}

template <typename T>
MusicOutputs run_music(std::shared_ptr<holoscan::Tensor> snapshot,
                       MUSICParams params) {
  FiveOutGraphRunResult result =
      run_music_result<T>(std::move(snapshot), std::move(params));
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
        "MUSICEstimatorOp test: expected one output frame set");
  }
  return {std::move(result.frames0.front()), std::move(result.frames1.front()),
          std::move(result.frames2.front()), std::move(result.frames3.front()),
          std::move(result.frames4.front())};
}

template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_beamscan(holoscan::Graph& graph, const Source& source, OpRef& op,
                   const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->snapshot, q);
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->angle_spectrum, sink1->in, q);
}

BeamscanParams make_beamscan_params(ArrayGeometry geom, AngleConvention conv,
                                    std::vector<double> az,
                                    std::vector<double> el) {
  BeamscanParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.az_angles = std::move(az);
  params.el_angles = std::move(el);
  return params;
}

template <typename T>
DualIOGraphRunResult run_beamscan_result(
    std::shared_ptr<holoscan::Tensor> snapshot, BeamscanParams params) {
  return run_single_in_dual_out_operator_graph_result<BeamscanEstimatorOp<T>>(
      {std::move(snapshot)}, std::move(params), "aoa_estimates",
      "angle_spectrum",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_beamscan(graph, source, op, sink0, sink1);
      });
}

template <typename T>
std::pair<HostTensor, HostTensor> run_beamscan(
    std::shared_ptr<holoscan::Tensor> snapshot, BeamscanParams params) {
  DualIOGraphRunResult result =
      run_beamscan_result<T>(std::move(snapshot), std::move(params));
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
  if (result.frames0.size() != 1U || result.frames1.size() != 1U) {
    throw std::runtime_error(
        "BeamscanEstimatorOp test: expected one output frame pair");
  }
  return {std::move(result.frames0.front()),
          std::move(result.frames1.front())};
}

int64_t count_prominent_local_maxima_1d(const double* spec, int64_t num_az,
                                        double rel_threshold = 0.3) {
  double global_max = 0.0;
  for (int64_t i = 0; i < num_az; ++i) {
    global_max = std::max(global_max, spec[i]);
  }
  if (!(global_max > 0.0)) {
    return 0;
  }
  const double floor = rel_threshold * global_max;
  int64_t count = 0;
  for (int64_t i = 0; i < num_az; ++i) {
    const double p = spec[i];
    if (p < floor) {
      continue;
    }
    const bool left_ok = (i == 0) || spec[i - 1] < p;
    const bool right_ok = (i == num_az - 1) || spec[i + 1] < p;
    if (left_ok && right_ok) {
      ++count;
    }
  }
  return count;
}

TEST(MUSICEstimatorOpTest, RecoversKnownDoaUlaAzOnlyFixedOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 3u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1));

  ASSERT_EQ(out.spectrum.shape, (std::vector<int64_t>{1, 61, 1}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
}

TEST(MUSICEstimatorOpTest, SharedSparseSlotInterchangesWithBeamscan) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.08;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto look = steering_vector(geom, conv, kTruthAz, 0.0);
  const auto az_grid = linspace(-0.3, 0.3, 61);

  const auto beamscan = run_beamscan<double>(
      block_to_device<double>(look, {1, kN, 1}),
      make_beamscan_params(geom, conv, az_grid, {0.0}));
  const auto music = run_music<double>(
      block_to_device<double>(look, {1, kN, 1}),
      make_params(geom, conv, az_grid, {0.0}, 1));

  EXPECT_NEAR(beamscan.first.as<double>()[0], kTruthAz, 0.02);
  EXPECT_NEAR(music.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(MUSICEstimatorOpTest, RecoversKnownDoaUraFixedOneSource) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 48;
  constexpr double kTruthAz = 0.12;
  constexpr double kTruthEl = 0.05;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, kTruthEl, kS, 0.02, 7u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.5, 0.5, 101),
                        linspace(-0.3, 0.3, 61), 1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.02);
}

TEST(MUSICEstimatorOpTest, AutoNumSourcesRecoversOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 9u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1, 1e-2,
                        4));

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
}

TEST(MUSICEstimatorOpTest, MultiSourceRecoversTwoDoa) {
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

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.5, 0.5, 201), {0.0}, 2, 1e-2,
                        2));

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

TEST(MUSICEstimatorOpTest, MaxSourcesCapsBelowNumSourcesNanFillsRest) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 256;
  constexpr double kAz0 = 0.20;
  constexpr double kAz1 = -0.15;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.02, 72u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.5, 0.5, 201), {0.0}, 2, 1e-2,
                        1));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
}

TEST(MUSICEstimatorOpTest, SpatialSmoothingResolvesCoherentSourcesUla) {
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

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.6, 0.6, 241), {0.0}, 2, 1e-2, 2,
                        {4}, true));

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

TEST(MUSICEstimatorOpTest, SpatialSmoothingRejectsNonUniformGeometry) {
  ArrayGeometry geom;
  geom.wavelength = 1.0;
  const double ys[4] = {0.0, 0.5, 1.0, 2.0};
  for (double y : ys) {
    geom.element_positions.insert(geom.element_positions.end(), {0.0, y, 0.0});
  }
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, geom.num_elements()});

  const FiveOutGraphRunResult result = run_music_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}, 1, 1e-2, 4, {2}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MUSICEstimatorOpTest, EigenvaluesAscendingAndNonNegative) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 128;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 91u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, 1));

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

TEST(MUSICEstimatorOpTest, RankDeficientAutoFallsBackToOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 4;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 12u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, -1, 1e-2,
                        4));

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
}

TEST(MUSICEstimatorOpTest, RejectsNumSourcesTooLarge) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});
  const FiveOutGraphRunResult result = run_music_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}, kN));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MUSICEstimatorOpTest, RejectsInvalidMaxSources) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});
  const FiveOutGraphRunResult result = run_music_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}, 1, 1e-2, 0));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MUSICEstimatorOpTest, RejectsInvalidSourceCountPfa) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});
  const FiveOutGraphRunResult result = run_music_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}, -1, 0.0));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MUSICEstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap = to_device_tensor(host, {0, kN, 4}, complex_dtype<double>());

  const auto out = run_music<double>(
      snap, make_params(geom, conv, {-0.1, 0.0, 0.1}, {0.0}, 1));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{0, 3, 1}));
  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{0, 4, 2}));
  EXPECT_EQ(out.source_count.shape, (std::vector<int64_t>{0}));
  EXPECT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{0, kN}));
}

TEST(MUSICEstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>((kN - 1) * 4),
                                         {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN - 1, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result = run_music_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}, 1));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MUSICEstimatorOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN * 4),
                                         {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result = run_music_result<double>(
      snap, make_params(geom, conv, {}, {0.0}, 1));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MUSICEstimatorOpTest, RejectsInvalidSnapshotRank) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap = to_device_tensor(host, {kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result = run_music_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}, 1));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MUSICEstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_f = block_to_device<float>(a, {1, kN});
  const FiveOutGraphRunResult result = run_music_result<double>(
      snap_f, make_params(geom, conv, {0.0}, {0.0}, 1));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MUSICEstimatorOpTest, RejectsPageableHostSnapshot) {
  GTEST_SKIP() << "HSDK 5 EA: pageable-host rejection not exercisable via "
                  "device Graph harness (same as MVDREstimator)";
}

TEST(MUSICEstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.06, 0.0, kS, 0.02, 11u);
  auto snap = block_to_device<float>(block, {1, kN, kS});

  const auto out = run_music<float>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1));
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.06f, 0.03f);
}

TEST(MUSICEstimatorOpTest, FloatDoubleAgreeOnDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.09;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 5u);
  auto snap_f = block_to_device<float>(block, {1, kN, kS});
  auto snap_d = block_to_device<double>(block, {1, kN, kS});

  MUSICParams params =
      make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1);

  const auto out_f = run_music<float>(snap_f, params);
  const auto out_d = run_music<double>(snap_d, params);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 1e-2f);
}

TEST(MUSICEstimatorOpTest, DeterministicAcrossRuns) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.09, 0.0, kS, 0.02, 171u);
  MUSICParams params =
      make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1, 1e-2, 2);

  const auto first =
      run_music<double>(block_to_device<double>(block, {1, kN, kS}), params);
  const auto second =
      run_music<double>(block_to_device<double>(block, {1, kN, kS}), params);

  EXPECT_EQ(first.aoa.data, second.aoa.data);
  EXPECT_EQ(first.spectrum.data, second.spectrum.data);
  EXPECT_EQ(first.doa_multi.data, second.doa_multi.data);
  EXPECT_EQ(first.source_count.data, second.source_count.data);
  EXPECT_EQ(first.eigenvalues.data, second.eigenvalues.data);
}

TEST(MUSICEstimatorOpTest, DirectionCosineMasksInvisibleRegion) {
  constexpr int64_t kN = 4;
  constexpr int64_t kS = 16;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto block = make_block(geom, conv, 0.0, 0.0, kS, 0.02, 2u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-1.2, 1.2, 5),
                        linspace(-1.2, 1.2, 5), 1));

  const double* spec = out.spectrum.as<double>();
  EXPECT_NEAR(spec[0], 0.0, 1e-3);
}

TEST(MUSICEstimatorOpTest, NumSourcesConfigurableFromParams) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 128;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block =
      make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, kS, 0.02, 111u);
  auto snap1 = block_to_device<double>(block, {1, kN, kS});
  auto snap2 = block_to_device<double>(block, {1, kN, kS});

  const auto out1 = run_music<double>(
      snap1, make_params(geom, conv, linspace(-0.5, 0.5, 101), {0.0}, 1, 1e-2,
                         4));
  const auto out2 = run_music<double>(
      snap2, make_params(geom, conv, linspace(-0.5, 0.5, 101), {0.0}, 2, 1e-2,
                         4));

  EXPECT_EQ(out1.source_count.as<int32_t>()[0], 1);
  EXPECT_EQ(out2.source_count.as<int32_t>()[0], 2);
}

TEST(MUSICEstimatorOpTest, SourceCountPfaConfigurableFromParams) {
  constexpr int64_t kN = 6;
  constexpr int64_t kS = 6;
  const std::vector<double> eigenvalues = {
      0.8930540251, 0.9468277841, 1.3517214181,
      1.6732266229, 9.0798471606, 14.3477610683};
  std::vector<std::complex<double>> block(static_cast<size_t>(kN * kS));
  for (int64_t c = 0; c < kN; ++c) {
    block[static_cast<size_t>(c * kS + c)] = {
        std::sqrt(static_cast<double>(kS) *
                  eigenvalues[static_cast<size_t>(c)]),
        0.0};
  }
  auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  auto snap_a = block_to_device<double>(block, {1, kN, kS});
  auto snap_b = block_to_device<double>(block, {1, kN, kS});

  const auto out_a = run_music<double>(
      snap_a, make_params(geom, conv, linspace(-0.5, 0.5, 51), {0.0}, -1, 1e-4,
                          4));
  const auto out_b = run_music<double>(
      snap_b, make_params(geom, conv, linspace(-0.5, 0.5, 51), {0.0}, -1, 0.5,
                          4));

  EXPECT_NE(spectrum_max(out_a.spectrum), spectrum_max(out_b.spectrum));
}

TEST(MUSICEstimatorOpTest, MaxSourcesConfigurableFromParams) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 101u);
  auto snap1 = block_to_device<double>(block, {1, kN, kS});
  auto snap3 = block_to_device<double>(block, {1, kN, kS});

  const auto out1 = run_music<double>(
      snap1, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1, 1e-2,
                         1));
  const auto out3 = run_music<double>(
      snap3, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1, 1e-2,
                         3));

  EXPECT_EQ(out1.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out3.doa_multi.shape, (std::vector<int64_t>{1, 3, 2}));
}

TEST(MUSICEstimatorOpTest, ResolvesCloselySpacedSourcesBeamscanCannot) {
  constexpr int64_t kN = 16;
  constexpr int64_t kS = 256;
  constexpr double kAz0 = 0.025;
  constexpr double kAz1 = -0.025;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.01, 200u);
  const auto az_grid = linspace(-0.3, 0.3, 241);

  const auto a0 = steering_vector(geom, conv, kAz0, 0.0);
  const auto a1 = steering_vector(geom, conv, kAz1, 0.0);
  std::vector<std::complex<double>> single_look(static_cast<size_t>(kN));
  for (int64_t c = 0; c < kN; ++c) {
    single_look[static_cast<size_t>(c)] =
        a0[static_cast<size_t>(c)] + a1[static_cast<size_t>(c)];
  }
  const auto beamscan = run_beamscan<double>(
      block_to_device<double>(single_look, {1, kN}),
      make_beamscan_params(geom, conv, az_grid, {0.0}));
  const int64_t beamscan_peaks = count_prominent_local_maxima_1d(
      beamscan.second.as<double>(), static_cast<int64_t>(az_grid.size()));
  EXPECT_EQ(beamscan_peaks, 1)
      << "Beamscan should show a single blended lobe for sources this close";

  const auto out = run_music<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, conv, az_grid, {0.0}, 2, 1e-2, 2));
  const int64_t music_peaks = count_prominent_local_maxima_1d(
      out.spectrum.as<double>(), static_cast<int64_t>(az_grid.size()));
  EXPECT_GE(music_peaks, 2)
      << "MUSIC should resolve two distinct lobes where Beamscan cannot";
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);
  const double* dm = out.doa_multi.as<double>();
  const bool matched =
      (std::abs(dm[0] - kAz0) < 0.015 && std::abs(dm[2] - kAz1) < 0.015) ||
      (std::abs(dm[0] - kAz1) < 0.015 && std::abs(dm[2] - kAz0) < 0.015);
  EXPECT_TRUE(matched) << "dm[0]=" << dm[0] << " dm[2]=" << dm[2];
}

TEST(MUSICEstimatorOpTest, AutoNumSourcesResolvesTwoCloselySpacedSources) {
  constexpr int64_t kN = 16;
  constexpr int64_t kS = 256;
  constexpr double kAz0 = 0.025;
  constexpr double kAz1 = -0.025;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.01, 201u);
  const auto az_grid = linspace(-0.3, 0.3, 241);

  const auto out = run_music<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, conv, az_grid, {0.0}, -1, 1e-2, 2));

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);
  const int64_t music_peaks = count_prominent_local_maxima_1d(
      out.spectrum.as<double>(), static_cast<int64_t>(az_grid.size()));
  EXPECT_GE(music_peaks, 2);
  const double* dm = out.doa_multi.as<double>();
  const bool matched =
      (std::abs(dm[0] - kAz0) < 0.015 && std::abs(dm[2] - kAz1) < 0.015) ||
      (std::abs(dm[0] - kAz1) < 0.015 && std::abs(dm[2] - kAz0) < 0.015);
  EXPECT_TRUE(matched) << "dm[0]=" << dm[0] << " dm[2]=" << dm[2];
}

TEST(MUSICEstimatorOpTest, MultipleItemsRecoverIndependentDoa) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 48;
  constexpr double kAz0 = 0.11;
  constexpr double kEl0 = 0.04;
  constexpr double kAz1 = -0.07;
  constexpr double kEl1 = -0.02;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto b0 = make_block(geom, conv, kAz0, kEl0, kS, 0.02, 21u);
  const auto b1 = make_block(geom, conv, kAz1, kEl1, kS, 0.02, 22u);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), b0.begin(), b0.end());
  batch.insert(batch.end(), b1.begin(), b1.end());
  auto snap = block_to_device<double>(batch, {2, geom.num_elements(), kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.4, 0.4, 81),
                        linspace(-0.2, 0.2, 41), 1));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{2, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.02);
  EXPECT_NEAR(aoa[1], kEl0, 0.02);
  EXPECT_NEAR(aoa[2], kAz1, 0.02);
  EXPECT_NEAR(aoa[3], kEl1, 0.02);
}

TEST(MUSICEstimatorOpTest, Accepts2DSingleSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.05, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}, 1));
  EXPECT_EQ(out.spectrum.shape[0], 1);
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], 0.05, 0.03);
}

TEST(MUSICEstimatorOpTest, SpatialSmoothing2DUraRecoversDoa) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.10;
  constexpr double kTruthEl = 0.06;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, kTruthEl, kS, 0.05, 44u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.4, 0.4, 81),
                        linspace(-0.3, 0.3, 61), 1, 1e-2, 4, {2, 2}, true));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.03);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.03);
}

TEST(MUSICEstimatorOpTest,
     SpatialSmoothingForwardOnlyReducesEigenvalueChannels) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 45u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_music<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1, 1e-2, 4,
                        {3}, false));

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, 6}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], 0.1, 0.03);
}

TEST(MUSICEstimatorOpTest, IsValidMaskExcludesWrongDoaSnapshots) {
  GTEST_SKIP() << "HSDK 5 EA: optional is_valid Graph wiring deferred "
                  "(HSDK5_PORT.md §1 / OPEN_ITEMS.md)";
}

TEST(MUSICEstimatorOpTest, AllInvalidMaskBlanksDoaAndSpectrum) {
  GTEST_SKIP() << "HSDK 5 EA: optional is_valid Graph wiring deferred "
                  "(HSDK5_PORT.md §1 / OPEN_ITEMS.md)";
}

TEST(MUSICEstimatorOpTest, LiveAzGridStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live az_angles override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(MUSICEstimatorOpTest, LiveElGridStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live el_angles override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(MUSICEstimatorOpTest, LiveArrayGeometryStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live array_geometry override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(MUSICEstimatorOpTest, LiveSpatialSmoothingStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live spatial_smoothing override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(MUSICEstimatorOpTest, LiveNumSourcesStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live num_sources override harness deferred "
                  "(HSDK5_PORT.md §1)";
}

TEST(MUSICEstimatorOpTest, LiveSourceCountPfaStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live source_count_pfa override harness "
                  "deferred (HSDK5_PORT.md §1)";
}

TEST(MUSICEstimatorOpTest, LiveMaxSourcesStickyOverrideReallocates) {
  GTEST_SKIP() << "HSDK 5 EA: live max_sources override harness deferred "
                  "(HSDK5_PORT.md §1)";
}

}  // namespace
