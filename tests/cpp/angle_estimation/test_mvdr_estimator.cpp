// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MVDREstimatorOp contract tests (adaptive/Capon sparse-path DOA).

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

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_common/math/constants.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::complex_dtype;
using holoscan::radar::MVDREstimatorOp;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_bool_tensor;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;
using holoscan::test::create_operator_test;

using holoscan::radar::kPi;

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

// Multi-snapshot block for one item: x[c,s] = a(θ)·g_s + noise, laid out
// (num_channels, num_snapshots) row-major (c*S + s). g_s is a unit tone with a
// per-snapshot phase (so the sample covariance approaches a·aᴴ); a small
// deterministic complex noise makes the covariance full rank.
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

double spectrum_max(const HostTensor& spec) {
  const double* values = spec.as<double>();
  const int64_t n = spec.size();
  double best = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    best = std::max(best, values[static_cast<size_t>(i)]);
  }
  return best;
}

void wire_noop_output(holoscan::Fragment* app,
                      const std::shared_ptr<holoscan::Operator>& op,
                      const std::string& port) {
  const auto noop = [](const holoscan::TensorMap&) {};
  auto sink = app->make_operator<
      holoscan::test::TestHarnessSinkOp<holoscan::TensorMap>>(port + "_noop");
  sink->set_validators({noop});
  app->add_flow(op, sink, {{port, "input"}});
}

struct MvdrOutputs {
  HostTensor aoa;
  HostTensor spectrum;
};

template <typename T, typename... ExtraArgs>
MvdrOutputs run_mvdr(std::shared_ptr<holoscan::Tensor> snapshot,
                     ExtraArgs&&... extra_args) {
  auto harness = create_operator_test<MVDREstimatorOp<T>>(
      std::forward<ExtraArgs>(extra_args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "snapshot", {single_map(std::move(snapshot), "snapshot")});

  TensorMapCapture aoa_cap;
  TensorMapCapture spec_cap;
  harness->template add_output_port<holoscan::TensorMap>(
      "aoa_estimates", aoa_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "angle_spectrum", spec_cap.validators());
  harness->run_test();

  if (aoa_cap.frames.size() != 1u || spec_cap.frames.size() != 1u) {
    throw std::runtime_error("MVDREstimatorOp test: expected one frame");
  }
  return {aoa_cap.frames.front(), spec_cap.frames.front()};
}

struct MvdrFullOutputs {
  HostTensor aoa;
  HostTensor spectrum;
  HostTensor doa_multi;
  HostTensor source_count;
  HostTensor eigenvalues;
};

// As run_mvdr but also captures the optional multi-source / diagnostic ports.
template <typename T, typename... ExtraArgs>
MvdrFullOutputs run_mvdr_full(std::shared_ptr<holoscan::Tensor> snapshot,
                              ExtraArgs&&... extra_args) {
  auto harness = create_operator_test<MVDREstimatorOp<T>>(
      std::forward<ExtraArgs>(extra_args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "snapshot", {single_map(std::move(snapshot), "snapshot")});

  TensorMapCapture aoa_cap, spec_cap, doa_cap, cnt_cap, eig_cap;
  harness->template add_output_port<holoscan::TensorMap>(
      "aoa_estimates", aoa_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "angle_spectrum", spec_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "doa_multi", doa_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "source_count", cnt_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "eigenvalues", eig_cap.validators());
  harness->run_test();

  if (aoa_cap.frames.size() != 1u || doa_cap.frames.size() != 1u ||
      cnt_cap.frames.size() != 1u || eig_cap.frames.size() != 1u) {
    throw std::runtime_error("MVDREstimatorOp test: expected one frame");
  }
  return {aoa_cap.frames.front(), spec_cap.frames.front(),
          doa_cap.frames.front(), cnt_cap.frames.front(),
          eig_cap.frames.front()};
}

// One item with two mutually uncorrelated sources: x[c,s] = a0·g0_s + a1·g1_s
// + noise, with independent random per-snapshot phases so the sample covariance
// approaches a0·a0ᴴ + a1·a1ᴴ (rank 2 → two Capon peaks).
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

// One item with two fully-COHERENT sources: both share the same per-snapshot
// gain g_s (e.g. specular multipath), so x[c,s] = (a0 + a1)·g_s + noise and the
// sample covariance collapses toward rank 1 ((a0+a1)(a0+a1)ᴴ). Plain MVDR
// cannot separate them; spatial smoothing restores the rank.
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
    const std::complex<double> g{std::cos(ps), std::sin(ps)};  // shared gain
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v =
          (a0[static_cast<size_t>(c)] + a1[static_cast<size_t>(c)]) * g;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
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

  // Forward-backward smoothing with 4 sub-arrays (L = 7) decorrelates the pair.
  const auto out = run_mvdr_full<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.6, 0.6, 241),
      holoscan::Arg("el_angles") = std::vector<double>{0.0},
      holoscan::Arg("spatial_smoothing") = std::vector<int64_t>{4},
      holoscan::Arg("forward_backward") = true,
      holoscan::Arg("max_sources") = int64_t{2});

  // Smoothing reduces the estimation aperture to L = 7 channels.
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

  // 2-D smoothing: 2 sub-arrays per axis → 3x3 reduced sub-array (L = 9).
  const auto out = run_mvdr<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.4, 0.4, 81),
      holoscan::Arg("el_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("spatial_smoothing") = std::vector<int64_t>{2, 2},
      holoscan::Arg("forward_backward") = true);

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

  const auto out = run_mvdr_full<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("el_angles") = std::vector<double>{0.0},
      holoscan::Arg("spatial_smoothing") = std::vector<int64_t>{3},
      holoscan::Arg("forward_backward") = false);

  // 3 sub-arrays on 8 channels → L = 6 reduced channels.
  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, 6}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], 0.1, 0.03);
}

TEST(MVDREstimatorOpTest, SpatialSmoothingRejectsNonUniformGeometry) {
  // Holey ULA (gap between the 3rd and 4th element) — smoothing must reject it.
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

  EXPECT_THROW(
      run_mvdr<double>(
          snap, holoscan::Arg("array_geometry") = geom,
          holoscan::Arg("angle_convention") = conv,
          holoscan::Arg("az_angles") = std::vector<double>{0.0},
          holoscan::Arg("el_angles") = std::vector<double>{0.0},
          holoscan::Arg("spatial_smoothing") = std::vector<int64_t>{2}),
      std::invalid_argument);
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
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.5, 0.5, 101),
      holoscan::Arg("el_angles") = linspace(-0.3, 0.3, 61));

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
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

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
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("el_angles") = std::vector<double>{0.0},
      holoscan::Arg("diagonal_loading") = 1e-2);

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
}

TEST(MVDREstimatorOpTest, Accepts2DSingleSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.05, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});

  const auto out = run_mvdr<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.2, 0.2, 21),
      holoscan::Arg("el_angles") = std::vector<double>{0.0});
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
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("el_angles") = std::vector<double>{0.0});
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

  const auto args = std::make_tuple(
      holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

  const auto out_f = std::apply(
      [&](auto&&... a) { return run_mvdr<float>(snap_f, a...); }, args);
  const auto out_d = std::apply(
      [&](auto&&... a) { return run_mvdr<double>(snap_d, a...); }, args);

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
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-1.2, 1.2, 5),
      holoscan::Arg("el_angles") = linspace(-1.2, 1.2, 5));

  const double* spec = out.spectrum.as<double>();
  EXPECT_NEAR(spec[0], 0.0, 1e-9);  // corner (u,v)=(-1.2,-1.2): invisible
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
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.4, 0.4, 81),
      holoscan::Arg("el_angles") = linspace(-0.2, 0.2, 41));

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

  const auto out = run_mvdr_full<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.5, 0.5, 201),
      holoscan::Arg("el_angles") = std::vector<double>{0.0},
      holoscan::Arg("max_sources") = int64_t{2});

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  ASSERT_EQ(out.source_count.shape, (std::vector<int64_t>{1}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double az_a = dm[0];  // strongest
  const double az_b = dm[2];  // second
  // The two reported peaks match the two truths (either order).
  const bool matched =
      (std::abs(az_a - kAz0) < 0.03 && std::abs(az_b - kAz1) < 0.03) ||
      (std::abs(az_a - kAz1) < 0.03 && std::abs(az_b - kAz0) < 0.03);
  EXPECT_TRUE(matched) << "az_a=" << az_a << " az_b=" << az_b;

  // doa_multi[0] is the strongest — equals aoa_estimates.
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

  const auto out = run_mvdr_full<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("el_angles") = std::vector<double>{0.0},
      holoscan::Arg("max_sources") = int64_t{4});

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 4, 2}));
  const int32_t count = out.source_count.as<int32_t>()[0];
  EXPECT_GE(count, 1);
  EXPECT_LE(count, 4);

  const double* dm = out.doa_multi.as<double>();
  // Strongest peak is the true DOA.
  EXPECT_NEAR(dm[0], kTruthAz, 0.02);
  // Rows at/after source_count are NaN; rows before are finite.
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

  const auto out = run_mvdr_full<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 31),
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, kN}));
  const double* ev = out.eigenvalues.as<double>();
  // Ascending (cuSOLVER heevd convention) and non-negative (Hermitian PSD).
  for (int64_t c = 0; c < kN; ++c) {
    EXPECT_GE(ev[c], -1e-6);
    if (c > 0) {
      EXPECT_GE(ev[c], ev[c - 1] - 1e-9);
    }
  }
  // A single dominant source → the top eigenvalue towers over the noise floor.
  EXPECT_GT(ev[kN - 1], 10.0 * ev[kN - 2]);
}

TEST(MVDREstimatorOpTest, RejectsInvalidMaxSources) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});
  EXPECT_THROW(
      run_mvdr<double>(snap, holoscan::Arg("array_geometry") = geom,
                       holoscan::Arg("angle_convention") = conv,
                       holoscan::Arg("az_angles") = std::vector<double>{0.0},
                       holoscan::Arg("el_angles") = std::vector<double>{0.0},
                       holoscan::Arg("max_sources") = int64_t{0}),
      std::invalid_argument);
}

TEST(MVDREstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap = to_device_tensor(host, {0, kN, 4}, complex_dtype<double>());

  const auto out = run_mvdr<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = std::vector<double>{-0.1, 0.0, 0.1},
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{0, 3, 1}));
}

TEST(MVDREstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>((kN - 1) * 4),
                                         {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN - 1, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  EXPECT_THROW(
      run_mvdr<double>(snap, holoscan::Arg("array_geometry") = geom,
                       holoscan::Arg("angle_convention") = conv,
                       holoscan::Arg("az_angles") = std::vector<double>{0.0},
                       holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(MVDREstimatorOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN * 4),
                                         {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  EXPECT_THROW(
      run_mvdr<double>(snap, holoscan::Arg("array_geometry") = geom,
                       holoscan::Arg("angle_convention") = conv,
                       holoscan::Arg("az_angles") = std::vector<double>{},
                       holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(MVDREstimatorOpTest, RejectsInvalidSnapshotRank) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap = to_device_tensor(host, {kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  EXPECT_THROW(
      run_mvdr<double>(snap, holoscan::Arg("array_geometry") = geom,
                       holoscan::Arg("angle_convention") = conv,
                       holoscan::Arg("az_angles") = std::vector<double>{0.0},
                       holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(MVDREstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_f = block_to_device<float>(a, {1, kN});
  EXPECT_THROW(
      run_mvdr<double>(snap_f, holoscan::Arg("array_geometry") = geom,
                       holoscan::Arg("angle_convention") = conv,
                       holoscan::Arg("az_angles") = std::vector<double>{0.0},
                       holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(MVDREstimatorOpTest, RejectsPageableHostSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_cpu = to_pageable_host_tensor(a, {1, kN}, complex_dtype<double>());
  EXPECT_THROW(
      run_mvdr<double>(snap_cpu, holoscan::Arg("array_geometry") = geom,
                       holoscan::Arg("angle_convention") = conv,
                       holoscan::Arg("az_angles") = std::vector<double>{0.0},
                       holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

// Emits a fixed snapshot + is_valid pair for graph-testing the mask exclusion.
class SnapshotMaskRelayOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SnapshotMaskRelayOp)

  std::shared_ptr<holoscan::Tensor> snapshot;
  std::shared_ptr<holoscan::Tensor> is_valid;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("snapshot");
    spec.output<holoscan::TensorMap>("is_valid");
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    holoscan::radar::emit_single(op_output, "snapshot", snapshot);
    holoscan::radar::emit_single(op_output, "is_valid", is_valid);
  }
};

class IsValidMaskApp : public holoscan::Application {
 public:
  std::shared_ptr<holoscan::Tensor> snapshot;
  std::shared_ptr<holoscan::Tensor> is_valid;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;
  TensorMapCapture* aoa = nullptr;
  TensorMapCapture* spectrum = nullptr;
  TensorMapCapture* doa_multi = nullptr;
  TensorMapCapture* source_count = nullptr;

  void compose() override {
    using holoscan::Arg;
    auto relay = make_operator<SnapshotMaskRelayOp>(
        "relay", make_condition<holoscan::CountCondition>(1));
    relay->snapshot = snapshot;
    relay->is_valid = is_valid;

    auto mvdr = make_operator<MVDREstimatorOp<double>>(
        "mvdr", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid, Arg("max_sources") = int64_t{2});

    add_flow(relay, mvdr,
             {{"snapshot", "snapshot"}, {"is_valid", "is_valid"}});
    holoscan::radar::testing::wire_capture_sink(this, mvdr, "aoa_estimates",
                                                aoa);
    if (spectrum != nullptr) {
      holoscan::radar::testing::wire_capture_sink(this, mvdr, "angle_spectrum",
                                                  spectrum);
    } else {
      wire_noop_output(this, mvdr, "angle_spectrum");
    }
    if (doa_multi != nullptr) {
      holoscan::radar::testing::wire_capture_sink(this, mvdr, "doa_multi",
                                                  doa_multi);
    } else {
      wire_noop_output(this, mvdr, "doa_multi");
    }
    if (source_count != nullptr) {
      holoscan::radar::testing::wire_capture_sink(this, mvdr, "source_count",
                                                  source_count);
    } else {
      wire_noop_output(this, mvdr, "source_count");
    }
    wire_noop_output(this, mvdr, "eigenvalues");
  }
};

TEST(MVDREstimatorOpTest, IsValidMaskExcludesWrongDoaSnapshots) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.1;
  constexpr double kWrongAz = -0.2;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);

  // First half of the snapshots carry the true DOA, second half a decoy; the
  // mask flags the decoy slots invalid, so the covariance sees only the truth.
  const auto good = make_block(geom, conv, kTruthAz, 0.0, kS / 2, 0.02, 31u);
  const auto bad = make_block(geom, conv, kWrongAz, 0.0, kS / 2, 0.02, 32u);
  std::vector<std::complex<double>> block(static_cast<size_t>(kN * kS));
  std::vector<uint8_t> mask(static_cast<size_t>(kS));
  for (int64_t c = 0; c < kN; ++c) {
    for (int64_t s = 0; s < kS / 2; ++s) {
      block[static_cast<size_t>(c * kS + s)] =
          good[static_cast<size_t>(c * (kS / 2) + s)];
      block[static_cast<size_t>(c * kS + (kS / 2) + s)] =
          bad[static_cast<size_t>(c * (kS / 2) + s)];
    }
  }
  for (int64_t s = 0; s < kS; ++s) {
    mask[static_cast<size_t>(s)] = s < kS / 2 ? 1 : 0;
  }

  IsValidMaskApp app;
  app.snapshot = block_to_device<double>(block, {1, kN, kS});
  app.is_valid = to_device_bool_tensor(mask, {1, kS});
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.4, 0.4, 81);
  app.el_grid = std::vector<double>{0.0};

  TensorMapCapture aoa_cap;
  app.aoa = &aoa_cap;
  app.run();

  ASSERT_EQ(aoa_cap.frames.size(), 1u);
  const double* aoa = aoa_cap.frames[0].as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.03);
}

TEST(MVDREstimatorOpTest, AllInvalidMaskBlanksDoaAndSpectrum) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 16;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 33u);
  std::vector<uint8_t> mask(static_cast<size_t>(kS), 0);

  IsValidMaskApp app;
  app.snapshot = block_to_device<double>(block, {1, kN, kS});
  app.is_valid = to_device_bool_tensor(mask, {1, kS});
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.4, 0.4, 41);
  app.el_grid = std::vector<double>{0.0};

  TensorMapCapture aoa_cap;
  TensorMapCapture spec_cap;
  TensorMapCapture doa_cap;
  TensorMapCapture cnt_cap;
  app.aoa = &aoa_cap;
  app.spectrum = &spec_cap;
  app.doa_multi = &doa_cap;
  app.source_count = &cnt_cap;
  app.run();

  ASSERT_EQ(aoa_cap.frames.size(), 1u);
  ASSERT_EQ(spec_cap.frames.size(), 1u);
  ASSERT_EQ(doa_cap.frames.size(), 1u);
  ASSERT_EQ(cnt_cap.frames.size(), 1u);

  const double* aoa = aoa_cap.frames[0].as<double>();
  EXPECT_TRUE(std::isnan(aoa[0]));
  EXPECT_TRUE(std::isnan(aoa[1]));

  const double* spec = spec_cap.frames[0].as<double>();
  for (int64_t i = 0; i < spec_cap.frames[0].size(); ++i) {
    EXPECT_NEAR(spec[i], 0.0, 0.0);
  }

  EXPECT_EQ(cnt_cap.frames[0].as<int32_t>()[0], 0);
  const double* dm = doa_cap.frames[0].as<double>();
  for (int64_t k = 0; k < 2; ++k) {
    EXPECT_TRUE(std::isnan(dm[k * 2]));
    EXPECT_TRUE(std::isnan(dm[k * 2 + 1]));
  }
}

class AzGridOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<std::vector<double>> az_overrides;
  TensorMapCapture* spectrum = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<MVDREstimatorOp<double>>(
        "mvdr", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention,
        Arg("az_angles") = az_overrides.front(), Arg("el_angles") = el_grid);
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "az_angles",
                                                    az_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "angle_spectrum",
                                                spectrum);
    wire_noop_output(this, op, "aoa_estimates");
  }
};

TEST(MVDREstimatorOpTest, LiveAzGridStickyOverride) {
  constexpr int64_t kN = 4;
  constexpr int64_t kS = 16;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.05, 0.0, kS, 0.05, 41u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  AzGridOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.el_grid = std::vector<double>{0.0};
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};
  app.az_overrides = {std::vector<double>{-0.1, 0.0, 0.1},
                      linspace(-0.2, 0.2, 11)};

  TensorMapCapture spec_cap;
  app.spectrum = &spec_cap;
  app.run();

  ASSERT_EQ(spec_cap.frames.size(), 2u);
  EXPECT_EQ(spec_cap.frames[0].shape[1], 3);
  EXPECT_EQ(spec_cap.frames[1].shape[1], 11);
}

class LoadingOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<double> loading_overrides;
  TensorMapCapture* spectrum = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<MVDREstimatorOp<double>>(
        "mvdr", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid,
        Arg("diagonal_loading") = loading_overrides.front());
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(this, op,
                                                    "diagonal_loading",
                                                    loading_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "angle_spectrum",
                                                spectrum);
    wire_noop_output(this, op, "aoa_estimates");
  }
};

TEST(MVDREstimatorOpTest, LiveDiagonalLoadingStickyOverride) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 51u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  LoadingOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.3, 0.3, 31);
  app.el_grid = std::vector<double>{0.0};
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};
  // Tiny then large fixed loading — a large δ broadens the Capon peak, so the
  // spectrum shape (and peak power) changes between ticks.
  app.loading_overrides = {1e-3, 1.0};

  TensorMapCapture spec_cap;
  app.spectrum = &spec_cap;
  app.run();

  ASSERT_EQ(spec_cap.frames.size(), 2u);
  EXPECT_EQ(spec_cap.frames[0].shape, spec_cap.frames[1].shape);
  EXPECT_NE(spectrum_max(spec_cap.frames[0]),
            spectrum_max(spec_cap.frames[1]));
}

class ArrayGeometryOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<ArrayGeometry> geometry_overrides;
  TensorMapCapture* spectrum = nullptr;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<MVDREstimatorOp<double>>(
        "mvdr", Arg("array_geometry") = geometry_overrides.front(),
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid);
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(
        this, op, "array_geometry", geometry_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "angle_spectrum",
                                                spectrum);
    wire_noop_output(this, op, "aoa_estimates");
  }
};

TEST(MVDREstimatorOpTest, LiveArrayGeometryStickyOverride) {
  constexpr int64_t kN = 4;
  constexpr int64_t kS = 16;
  const auto geom_a = make_ula_geometry(kN, 0.5, 1.0);
  auto geom_b = make_ula_geometry(kN, 0.5, 1.0);
  geom_b.wavelength = 0.8;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom_a, conv, 0.05, 0.0, kS, 0.05, 61u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  ArrayGeometryOverrideApp app;
  app.convention = conv;
  app.az_grid = linspace(-0.2, 0.2, 11);
  app.el_grid = std::vector<double>{0.0};
  app.geometry_overrides = {geom_a, geom_b};
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};

  TensorMapCapture spec_cap;
  app.spectrum = &spec_cap;
  app.run();

  ASSERT_EQ(spec_cap.frames.size(), 2u);
  EXPECT_EQ(spec_cap.frames[0].shape, spec_cap.frames[1].shape);
  EXPECT_NE(spectrum_max(spec_cap.frames[0]),
            spectrum_max(spec_cap.frames[1]));
}

class MaxSourcesOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<int64_t> max_sources_overrides;
  TensorMapCapture* doa_multi = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<MVDREstimatorOp<double>>(
        "mvdr", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid,
        Arg("max_sources") = max_sources_overrides.front());
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "max_sources",
                                                    max_sources_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "doa_multi",
                                                doa_multi);
    wire_noop_output(this, op, "aoa_estimates");
  }
};

TEST(MVDREstimatorOpTest, LiveMaxSourcesStickyOverrideReallocates) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 101u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  MaxSourcesOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.3, 0.3, 61);
  app.el_grid = std::vector<double>{0.0};
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};
  app.max_sources_overrides = {int64_t{1}, int64_t{3}};

  TensorMapCapture doa_cap;
  app.doa_multi = &doa_cap;
  app.run();

  ASSERT_EQ(doa_cap.frames.size(), 2u);
  EXPECT_EQ(doa_cap.frames[0].shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(doa_cap.frames[1].shape, (std::vector<int64_t>{1, 3, 2}));
}

}  // namespace
