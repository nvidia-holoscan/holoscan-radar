// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MVDREstimatorOp contract tests (HSDK 5 Graph harness): known-DOA recovery,
// multi-source / eigenvalues / spatial smoothing Params coverage, float/
// double agreement, and error paths. Live / is_valid Graph wiring deferred.

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

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
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
using holoscan::radar::MVDREstimatorOp;
using holoscan::radar::MVDRParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::kPi;
using holoscan::radar::testing::FiveOutGraphRunResult;
using holoscan::radar::testing::HostTensor;
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

std::vector<std::complex<double>> make_block_two_coherent(
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
    const double ps = phase(rng);
    const std::complex<double> g{std::cos(ps), std::sin(ps)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v =
          (a0[static_cast<size_t>(c)] + a1[static_cast<size_t>(c)]) * g;
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

struct MvdrOutputs {
  HostTensor aoa;
  HostTensor spectrum;
  HostTensor doa_multi;
  HostTensor source_count;
  HostTensor eigenvalues;
};

template <typename Source, typename OpRef, typename S0, typename S1,
          typename S2, typename S3, typename S4>
void wire_mvdr(holoscan::Graph& graph, const Source& source, OpRef& op,
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

MVDRParams make_params(ArrayGeometry geom, AngleConvention conv,
                       std::vector<double> az, std::vector<double> el,
                       double loading = -1.0, int64_t max_sources = 1,
                       std::vector<int64_t> smoothing = {},
                       bool forward_backward = true) {
  MVDRParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.az_angles = std::move(az);
  params.el_angles = std::move(el);
  params.diagonal_loading = loading;
  params.max_sources = max_sources;
  params.spatial_smoothing = std::move(smoothing);
  params.forward_backward = forward_backward;
  return params;
}

template <typename T>
FiveOutGraphRunResult run_mvdr_result(std::shared_ptr<holoscan::Tensor> snapshot,
                                      MVDRParams params) {
  return run_single_in_five_out_operator_graph_result<MVDREstimatorOp<T>>(
      {std::move(snapshot)}, std::move(params), "aoa_estimates",
      "angle_spectrum", "doa_multi", "source_count", "eigenvalues",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1, const auto& sink2,
         const auto& sink3, const auto& sink4) {
        wire_mvdr(graph, source, op, sink0, sink1, sink2, sink3, sink4);
      });
}

template <typename T>
MvdrOutputs run_mvdr(std::shared_ptr<holoscan::Tensor> snapshot,
                     MVDRParams params) {
  FiveOutGraphRunResult result =
      run_mvdr_result<T>(std::move(snapshot), std::move(params));
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
        "MVDREstimatorOp test: expected one output frame set");
  }
  return {std::move(result.frames0.front()), std::move(result.frames1.front()),
          std::move(result.frames2.front()), std::move(result.frames3.front()),
          std::move(result.frames4.front())};
}

TEST(MVDREstimatorOpTest, SpatialSmoothingResolvesCoherentSourcesUla) {
  constexpr int64_t kN = 10;
  constexpr int64_t kS = 64;
  constexpr double kAz0 = 0.30;
  constexpr double kAz1 = -0.30;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two_coherent(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.02, 71u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.6, 0.6, 241), {0.0}, -1.0, 2,
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

TEST(MVDREstimatorOpTest, SpatialSmoothing2DUraRecoversDoa) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.10;
  constexpr double kTruthEl = 0.06;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block =
      make_block(geom, conv, kTruthAz, kTruthEl, kS, 0.05, 44u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.4, 0.4, 81),
                        linspace(-0.3, 0.3, 61), -1.0, 1, {2, 2}, true));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.03);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.03);
}

TEST(MVDREstimatorOpTest, SpatialSmoothingReducesEigenvalueChannels) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 45u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1.0, 1,
                        {3}, false));

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, 6}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], 0.1, 0.03);
}

TEST(MVDREstimatorOpTest, SpatialSmoothingRejectsNonUniformGeometry) {
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

  const FiveOutGraphRunResult result = run_mvdr_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}, -1.0, 1, {2}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RecoversKnownDoaUra) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.12;
  constexpr double kTruthEl = 0.05;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, kTruthEl, kS, 0.05, 7u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.5, 0.5, 101),
                        linspace(-0.3, 0.3, 61)));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.02);
}

TEST(MVDREstimatorOpTest, RecoversKnownDoaUlaAzOnly) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 3u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}));

  ASSERT_EQ(out.spectrum.shape, (std::vector<int64_t>{1, 61, 1}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
}

TEST(MVDREstimatorOpTest, FixedDiagonalLoadingRecoversDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 9u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1e-2));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
}

TEST(MVDREstimatorOpTest, DiagonalLoadingConfigurableFromParams) {
  // Authoring Params must select diagonal_loading (live Graph override
  // deferred — HSDK5_PORT §1).
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 51u);
  auto snap_a = block_to_device<double>(block, {1, kN, kS});
  auto snap_b = block_to_device<double>(block, {1, kN, kS});

  const auto out_a = run_mvdr<double>(
      snap_a, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, 1e-3));
  const auto out_b = run_mvdr<double>(
      snap_b, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, 1.0));

  EXPECT_EQ(out_a.spectrum.shape, out_b.spectrum.shape);
  EXPECT_NE(out_a.spectrum.data, out_b.spectrum.data);
}

TEST(MVDREstimatorOpTest, MaxSourcesConfigurableFromParams) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 101u);
  auto snap1 = block_to_device<double>(block, {1, kN, kS});
  auto snap3 = block_to_device<double>(block, {1, kN, kS});

  const auto out1 = run_mvdr<double>(
      snap1, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1.0, 1));
  const auto out3 = run_mvdr<double>(
      snap3, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1.0, 3));

  EXPECT_EQ(out1.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out3.doa_multi.shape, (std::vector<int64_t>{1, 3, 2}));
}

TEST(MVDREstimatorOpTest, Accepts2DSingleSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.05, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}));
  EXPECT_EQ(out.spectrum.shape[0], 1);
}

TEST(MVDREstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.06, 0.0, kS, 0.05, 11u);
  auto snap = block_to_device<float>(block, {1, kN, kS});

  const auto out = run_mvdr<float>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}));
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.06f, 0.03f);
}

TEST(MVDREstimatorOpTest, FloatDoubleAgreeOnDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.09;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 5u);
  auto snap_f = block_to_device<float>(block, {1, kN, kS});
  auto snap_d = block_to_device<double>(block, {1, kN, kS});

  MVDRParams params =
      make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0});

  const auto out_f = run_mvdr<float>(snap_f, params);
  const auto out_d = run_mvdr<double>(snap_d, params);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 1e-2f);
}

TEST(MVDREstimatorOpTest, DirectionCosineMasksInvisibleRegion) {
  constexpr int64_t kN = 4;
  constexpr int64_t kS = 16;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto block = make_block(geom, conv, 0.0, 0.0, kS, 0.05, 2u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-1.2, 1.2, 5),
                        linspace(-1.2, 1.2, 5)));

  const double* spec = out.spectrum.as<double>();
  EXPECT_NEAR(spec[0], 0.0, 1e-9);
}

TEST(MVDREstimatorOpTest, MultipleItemsRecoverIndependentDoa) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 24;
  constexpr double kAz0 = 0.11;
  constexpr double kEl0 = 0.04;
  constexpr double kAz1 = -0.07;
  constexpr double kEl1 = -0.02;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto b0 = make_block(geom, conv, kAz0, kEl0, kS, 0.05, 21u);
  const auto b1 = make_block(geom, conv, kAz1, kEl1, kS, 0.05, 22u);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), b0.begin(), b0.end());
  batch.insert(batch.end(), b1.begin(), b1.end());
  auto snap = block_to_device<double>(batch, {2, geom.num_elements(), kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.4, 0.4, 81),
                        linspace(-0.2, 0.2, 41)));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{2, 2}));
  ASSERT_EQ(out.spectrum.shape, (std::vector<int64_t>{2, 81, 41}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.02);
  EXPECT_NEAR(aoa[1], kEl0, 0.02);
  EXPECT_NEAR(aoa[2], kAz1, 0.02);
  EXPECT_NEAR(aoa[3], kEl1, 0.02);
}

TEST(MVDREstimatorOpTest, MultiSourceRecoversTwoDoa) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 128;
  constexpr double kAz0 = 0.20;
  constexpr double kAz1 = -0.15;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.02, 71u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.5, 0.5, 201), {0.0}, -1.0, 2));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  ASSERT_EQ(out.source_count.shape, (std::vector<int64_t>{1}));
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

TEST(MVDREstimatorOpTest, MaxSourcesCapacityNanFillsUnusedRows) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 81u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1.0, 4));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 4, 2}));
  const int32_t count = out.source_count.as<int32_t>()[0];
  EXPECT_GE(count, 1);
  EXPECT_LE(count, 4);

  const double* dm = out.doa_multi.as<double>();
  EXPECT_NEAR(dm[0], kTruthAz, 0.02);
  for (int64_t k = 0; k < 4; ++k) {
    if (k < count) {
      EXPECT_FALSE(std::isnan(dm[k * 2]));
    } else {
      EXPECT_TRUE(std::isnan(dm[k * 2]));
      EXPECT_TRUE(std::isnan(dm[k * 2 + 1]));
    }
  }
}

TEST(MVDREstimatorOpTest, EigenvaluesRankOneSourceDominant) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 128;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 91u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}));

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

TEST(MVDREstimatorOpTest, RejectsInvalidMaxSources) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});
  const FiveOutGraphRunResult result = run_mvdr_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}, -1.0, 0));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap = to_device_tensor(host, {0, kN, 4}, complex_dtype<double>());

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, {-0.1, 0.0, 0.1}, {0.0}));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{0, 3, 1}));
  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(out.source_count.shape, (std::vector<int64_t>{0}));
  EXPECT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{0, kN}));
}

TEST(MVDREstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>((kN - 1) * 4),
                                         {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN - 1, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result = run_mvdr_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN * 4),
                                         {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result = run_mvdr_result<double>(
      snap, make_params(geom, conv, {}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RejectsInvalidSnapshotRank) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap = to_device_tensor(host, {kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result = run_mvdr_result<double>(
      snap, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_f = block_to_device<float>(a, {1, kN});
  const FiveOutGraphRunResult result = run_mvdr_result<double>(
      snap_f, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RejectsPageableHostSnapshot) {
  // Graph harness copies FrameBank tensors into a device Output loan before
  // the operator receives them, so MVDR never observes a pageable-host Tensor.
  GTEST_SKIP() << "HSDK 5 EA: pageable-host rejection not exercisable via "
                  "device Graph harness (same as BeamscanEstimator)";
}

TEST(MVDREstimatorOpTest, IsValidMaskExcludesWrongDoaSnapshots) {
  GTEST_SKIP() << "HSDK 5 EA: optional is_valid Graph wiring deferred "
                  "(HSDK5_PORT.md §1 / OPEN_ITEMS.md)";
}

TEST(MVDREstimatorOpTest, AllInvalidMaskBlanksDoaAndSpectrum) {
  GTEST_SKIP() << "HSDK 5 EA: optional is_valid Graph wiring deferred "
                  "(HSDK5_PORT.md §1 / OPEN_ITEMS.md)";
}

TEST(MVDREstimatorOpTest, LiveAzGridStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live az_angles override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(MVDREstimatorOpTest, LiveDiagonalLoadingStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live diagonal_loading override harness "
                  "deferred (HSDK5_PORT.md §1)";
}

TEST(MVDREstimatorOpTest, LiveArrayGeometryStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live array_geometry override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(MVDREstimatorOpTest, LiveMaxSourcesStickyOverrideReallocates) {
  GTEST_SKIP() << "HSDK 5 EA: live max_sources override harness deferred "
                  "(HSDK5_PORT.md §1)";
}

TEST(MVDREstimatorOpTest, LiveSpatialSmoothingStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live spatial_smoothing override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

}  // namespace
