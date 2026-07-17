// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PhaseShiftBeamformerOp contract tests (conventional dense-path Bartlett
// beamforming: rd_maps -> rda_cube).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::PhaseShiftBeamformerOp;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;
using holoscan::test::create_operator_test;

constexpr double kPi = 3.141592653589793;

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

// The per-channel plane-wave vector for a source at (az, el) — one
// range-Doppler cell's channel snapshot when a single point target illuminates
// the array.
std::vector<std::complex<double>> make_steering_vector(
    const ArrayGeometry& geom, const AngleConvention& c, double az,
    double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(c, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> v(
      static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase = k * (geom.element_positions[base] * dir.x +
                              geom.element_positions[base + 1] * dir.y +
                              geom.element_positions[base + 2] * dir.z);
    v[static_cast<size_t>(m)] = {std::cos(phase), std::sin(phase)};
  }
  return v;
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
std::shared_ptr<holoscan::Tensor> vector_to_rd_maps(
    const std::vector<std::complex<double>>& v,
    const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    cast[i] = {static_cast<T>(v[i].real()), static_cast<T>(v[i].imag())};
  }
  return to_device_tensor(cast, shape, complex_dtype<T>());
}

template <typename T>
double cube_power(const HostTensor& cube, int64_t ia, int64_t ie, int64_t d,
                  int64_t r, int64_t num_el, int64_t num_doppler,
                  int64_t num_range) {
  const int64_t idx =
      (((ia * num_el + ie) * num_doppler + d) * num_range + r);
  const std::complex<T> v =
      cube.template as<std::complex<T>>()[static_cast<size_t>(idx)];
  return static_cast<double>(v.real()) * static_cast<double>(v.real()) +
         static_cast<double>(v.imag()) * static_cast<double>(v.imag());
}

// (ia, ie) of the peak-power beam at range-Doppler cell (d, r).
template <typename T>
std::pair<int64_t, int64_t> peak_beam_at(const HostTensor& cube, int64_t num_az,
                                         int64_t num_el, int64_t num_doppler,
                                         int64_t num_range, int64_t d,
                                         int64_t r) {
  double best = -1.0;
  std::pair<int64_t, int64_t> arg{-1, -1};
  for (int64_t ia = 0; ia < num_az; ++ia) {
    for (int64_t ie = 0; ie < num_el; ++ie) {
      const double p = cube_power<T>(cube, ia, ie, d, r, num_el, num_doppler,
                                     num_range);
      if (p > best) {
        best = p;
        arg = {ia, ie};
      }
    }
  }
  return arg;
}

template <typename T, typename... ExtraArgs>
HostTensor run_beamformer(std::shared_ptr<holoscan::Tensor> rd_maps,
                          ExtraArgs&&... extra_args) {
  auto harness = create_operator_test<PhaseShiftBeamformerOp<T>>(
      std::forward<ExtraArgs>(extra_args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps", {single_map(std::move(rd_maps), "rd_maps")});
  TensorMapCapture cube_cap;
  harness->template add_output_port<holoscan::TensorMap>("rda_cube",
                                                         cube_cap.validators());
  harness->run_test();
  if (cube_cap.frames.size() != 1u) {
    throw std::runtime_error("PhaseShiftBeamformerOp test: expected one frame");
  }
  return cube_cap.frames.front();
}

TEST(PhaseShiftBeamformerOpTest, RecoversKnownBeamBin2D) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = 0.12;
  constexpr double kTruthEl = 0.05;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto vec = make_steering_vector(geom, conv, kTruthAz, kTruthEl);
  auto rd = vector_to_rd_maps<double>(vec, {geom.num_elements(), 1, 1});

  const auto az = linspace(-0.5, 0.5, 101);
  const auto el = linspace(-0.3, 0.3, 61);
  const auto cube = run_beamformer<double>(
      rd, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = az, holoscan::Arg("el_angles") = el);

  ASSERT_EQ(cube.shape, (std::vector<int64_t>{101, 61, 1, 1}));
  const auto [ia, ie] = peak_beam_at<double>(cube, 101, 61, 1, 1, 0, 0);
  EXPECT_NEAR(az[static_cast<size_t>(ia)], kTruthAz, 0.02);
  EXPECT_NEAR(el[static_cast<size_t>(ie)], kTruthEl, 0.02);
}

TEST(PhaseShiftBeamformerOpTest, RecoversKnownBeamAzOnly1DScan) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto vec = make_steering_vector(geom, conv, kTruthAz, 0.0);
  auto rd = vector_to_rd_maps<double>(vec, {kN, 1, 1});

  const auto az = linspace(-0.3, 0.3, 61);
  const auto cube = run_beamformer<double>(
      rd, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = az,
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

  ASSERT_EQ(cube.shape, (std::vector<int64_t>{61, 1, 1, 1}));
  const auto [ia, ie] = peak_beam_at<double>(cube, 61, 1, 1, 1, 0, 0);
  EXPECT_EQ(ie, 0);
  EXPECT_NEAR(az[static_cast<size_t>(ia)], kTruthAz, 0.011);
}

TEST(PhaseShiftBeamformerOpTest, ConventionRelabelsSamePhysicalDirection) {
  // The same physical direction is reported at the expected (az, el) labels
  // under two different presets.
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = -0.1;
  constexpr double kTruthEl = -0.03;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);

  for (const auto preset : {AngleConventionPreset::kRadarBoresight,
                            AngleConventionPreset::kAerospaceNed}) {
    AngleConvention conv;
    conv.preset = preset;
    const auto vec = make_steering_vector(geom, conv, kTruthAz, kTruthEl);
    auto rd = vector_to_rd_maps<double>(vec, {geom.num_elements(), 1, 1});
    const auto az = linspace(-0.5, 0.5, 101);
    const auto el = linspace(-0.3, 0.3, 61);
    const auto cube = run_beamformer<double>(
        rd, holoscan::Arg("array_geometry") = geom,
        holoscan::Arg("angle_convention") = conv,
        holoscan::Arg("az_angles") = az, holoscan::Arg("el_angles") = el);
    const auto [ia, ie] = peak_beam_at<double>(cube, 101, 61, 1, 1, 0, 0);
    EXPECT_NEAR(az[static_cast<size_t>(ia)], kTruthAz, 0.02);
    EXPECT_NEAR(el[static_cast<size_t>(ie)], kTruthEl, 0.02);
  }
}

TEST(PhaseShiftBeamformerOpTest, DirectionCosineMasksInvisibleRegion) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);  // u=0,v=0
  auto rd = vector_to_rd_maps<double>(vec, {kN, 1, 1});

  const auto uu = linspace(-1.2, 1.2, 5);
  const auto vv = linspace(-1.2, 1.2, 5);
  const auto cube = run_beamformer<double>(
      rd, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = uu, holoscan::Arg("el_angles") = vv);

  // Corner (u,v)=(-1.2,-1.2): u^2+v^2 > 1 -> invisible -> zero weight -> zero.
  EXPECT_NEAR(cube_power<double>(cube, 0, 0, 0, 0, 5, 1, 1), 0.0, 1e-9);
  // Center (u,v)=(0,0) carries the target energy.
  EXPECT_GT(cube_power<double>(cube, 2, 2, 0, 0, 5, 1, 1), 0.0);
}

TEST(PhaseShiftBeamformerOpTest, ArrayTaperLowersSidelobes) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);  // boresight
  const auto az = linspace(-1.0, 1.0, 201);
  const std::vector<double> el{0.0};

  const auto peak_and_sidelobe = [&](const WindowSpec& taper) {
    auto rd = vector_to_rd_maps<double>(vec, {kN, 1, 1});
    const auto cube = run_beamformer<double>(
        rd, holoscan::Arg("array_geometry") = geom,
        holoscan::Arg("angle_convention") = conv,
        holoscan::Arg("az_angles") = az, holoscan::Arg("el_angles") = el,
        holoscan::Arg("array_taper") = taper);
    double peak = 0.0;
    double sidelobe = 0.0;
    for (int64_t ia = 0; ia < 201; ++ia) {
      const double p = cube_power<double>(cube, ia, 0, 0, 0, 1, 1, 1);
      peak = std::max(peak, p);
      // Beyond both windows' mainlobes (Hann roughly doubles the mainlobe
      // width, first null near az ~ 0.52 rad for this N=8 half-wave array).
      if (std::abs(az[static_cast<size_t>(ia)]) > 0.65) {
        sidelobe = std::max(sidelobe, p);
      }
    }
    return sidelobe / peak;  // normalized peak sidelobe ratio
  };

  WindowSpec hann;
  hann.type = WindowType::kHann;
  const double rect_ratio = peak_and_sidelobe(WindowSpec{});
  const double hann_ratio = peak_and_sidelobe(hann);
  EXPECT_GT(rect_ratio, 0.0);
  EXPECT_LT(hann_ratio, rect_ratio);
}

TEST(PhaseShiftBeamformerOpTest, PreservesRangeDopplerPlanePerCell) {
  // Two point targets at distinct (doppler, range) cells with distinct DOAs;
  // each cell's peak beam must recover its own DOA (validates the per-cell
  // beamforming and the (az, el, doppler, range) output layout).
  constexpr int64_t kN = 8;
  constexpr int64_t kD = 2;
  constexpr int64_t kR = 2;
  constexpr double kAz0 = 0.1;
  constexpr double kAz1 = -0.15;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto v0 = make_steering_vector(geom, conv, kAz0, 0.0);
  const auto v1 = make_steering_vector(geom, conv, kAz1, 0.0);

  // rd_maps layout (channel, doppler, range); target0 at (0,0), target1 (1,1).
  std::vector<std::complex<double>> host(
      static_cast<size_t>(kN * kD * kR), {0.0, 0.0});
  const auto set = [&](int64_t c, int64_t d, int64_t r,
                       std::complex<double> val) {
    host[static_cast<size_t>((c * kD + d) * kR + r)] = val;
  };
  for (int64_t c = 0; c < kN; ++c) {
    set(c, 0, 0, v0[static_cast<size_t>(c)]);
    set(c, 1, 1, v1[static_cast<size_t>(c)]);
  }
  auto rd = to_device_tensor(
      std::vector<std::complex<double>>(host), {kN, kD, kR},
      complex_dtype<double>());

  const auto az = linspace(-0.4, 0.4, 81);
  const auto cube = run_beamformer<double>(
      rd, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = az,
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

  ASSERT_EQ(cube.shape, (std::vector<int64_t>{81, 1, kD, kR}));
  const auto [ia0, ie0] = peak_beam_at<double>(cube, 81, 1, kD, kR, 0, 0);
  const auto [ia1, ie1] = peak_beam_at<double>(cube, 81, 1, kD, kR, 1, 1);
  EXPECT_NEAR(az[static_cast<size_t>(ia0)], kAz0, 0.02);
  EXPECT_NEAR(az[static_cast<size_t>(ia1)], kAz1, 0.02);
  // The empty (0,1) and (1,0) cells hold no energy.
  double empty = 0.0;
  for (int64_t ia = 0; ia < 81; ++ia) {
    empty = std::max(empty, cube_power<double>(cube, ia, 0, 0, 1, 1, kD, kR));
  }
  EXPECT_NEAR(empty, 0.0, 1e-9);
}

TEST(PhaseShiftBeamformerOpTest, BatchLeadingDimsRecoverIndependentDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kBatch = 2;
  constexpr double kAz0 = 0.12;
  constexpr double kAz1 = -0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto v0 = make_steering_vector(geom, conv, kAz0, 0.0);
  const auto v1 = make_steering_vector(geom, conv, kAz1, 0.0);

  // (batch, channel, doppler=1, range=1).
  std::vector<std::complex<double>> host;
  host.insert(host.end(), v0.begin(), v0.end());
  host.insert(host.end(), v1.begin(), v1.end());
  auto rd = to_device_tensor(host, {kBatch, kN, 1, 1}, complex_dtype<double>());

  const auto az = linspace(-0.4, 0.4, 81);
  const auto cube = run_beamformer<double>(
      rd, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = az,
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

  ASSERT_EQ(cube.shape, (std::vector<int64_t>{kBatch, 81, 1, 1, 1}));
  // Peak beam per batch slice (each slice is 81*1*1*1 elements).
  const auto peak_in_slice = [&](int64_t b) {
    double best = -1.0;
    int64_t arg = -1;
    const auto* data = cube.as<std::complex<double>>();
    for (int64_t ia = 0; ia < 81; ++ia) {
      const int64_t idx = b * 81 + ia;
      const auto v = data[static_cast<size_t>(idx)];
      const double p = v.real() * v.real() + v.imag() * v.imag();
      if (p > best) {
        best = p;
        arg = ia;
      }
    }
    return arg;
  };
  EXPECT_NEAR(az[static_cast<size_t>(peak_in_slice(0))], kAz0, 0.02);
  EXPECT_NEAR(az[static_cast<size_t>(peak_in_slice(1))], kAz1, 0.02);
}

TEST(PhaseShiftBeamformerOpTest, FloatDoubleAgreeOnPeakBeam) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.09;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto vec = make_steering_vector(geom, conv, kTruthAz, 0.0);
  const auto az = linspace(-0.3, 0.3, 61);

  auto rd_f = vector_to_rd_maps<float>(vec, {kN, 1, 1});
  auto rd_d = vector_to_rd_maps<double>(vec, {kN, 1, 1});
  const auto cube_f = run_beamformer<float>(
      rd_f, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = az,
      holoscan::Arg("el_angles") = std::vector<double>{0.0});
  const auto cube_d = run_beamformer<double>(
      rd_d, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = az,
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

  const auto [ia_f, ie_f] = peak_beam_at<float>(cube_f, 61, 1, 1, 1, 0, 0);
  const auto [ia_d, ie_d] = peak_beam_at<double>(cube_d, 61, 1, 1, 1, 0, 0);
  EXPECT_EQ(ia_f, ia_d);
  EXPECT_EQ(ie_f, ie_d);
}

TEST(PhaseShiftBeamformerOpTest, RejectsRankTooLow) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto rd = to_device_tensor(host, {kN, 1}, complex_dtype<double>());
  EXPECT_THROW(
      run_beamformer<double>(rd, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv,
                             holoscan::Arg("az_angles") =
                                 std::vector<double>{0.0},
                             holoscan::Arg("el_angles") =
                                 std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(PhaseShiftBeamformerOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN - 1),
                                         {1.0, 0.0});
  auto rd = to_device_tensor(host, {kN - 1, 1, 1}, complex_dtype<double>());
  EXPECT_THROW(
      run_beamformer<double>(rd, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv,
                             holoscan::Arg("az_angles") =
                                 std::vector<double>{0.0},
                             holoscan::Arg("el_angles") =
                                 std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(PhaseShiftBeamformerOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto rd = to_device_tensor(host, {kN, 1, 1}, complex_dtype<double>());
  EXPECT_THROW(
      run_beamformer<double>(rd, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv,
                             holoscan::Arg("az_angles") =
                                 std::vector<double>{},
                             holoscan::Arg("el_angles") =
                                 std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(PhaseShiftBeamformerOpTest, RejectsDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);
  auto rd_f = vector_to_rd_maps<float>(vec, {kN, 1, 1});
  EXPECT_THROW(
      run_beamformer<double>(rd_f, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv,
                             holoscan::Arg("az_angles") =
                                 std::vector<double>{0.0},
                             holoscan::Arg("el_angles") =
                                 std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(PhaseShiftBeamformerOpTest, RejectsPageableHostInput) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto rd_cpu =
      to_pageable_host_tensor(host, {kN, 1, 1}, complex_dtype<double>());
  EXPECT_THROW(
      run_beamformer<double>(rd_cpu, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv,
                             holoscan::Arg("az_angles") =
                                 std::vector<double>{0.0},
                             holoscan::Arg("el_angles") =
                                 std::vector<double>{0.0}),
      std::invalid_argument);
}

// ---- [live] sticky overrides -------------------------------------------

class AzGridOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> rd_frames;
  std::vector<std::vector<double>> az_overrides;
  TensorMapCapture* cube = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<PhaseShiftBeamformerOp<double>>(
        "beamformer", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention,
        Arg("az_angles") = az_overrides.front(), Arg("el_angles") = el_grid);
    holoscan::radar::testing::wire_data_source(this, op, "rd_maps", rd_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "az_angles",
                                                    az_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "rda_cube", cube);
  }
};

TEST(PhaseShiftBeamformerOpTest, LiveAzGridStickyOverride) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom, conv, 0.05, 0.0);
  auto rd = vector_to_rd_maps<double>(vec, {kN, 1, 1});

  AzGridOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.el_grid = {0.0};
  app.rd_frames = {single_map(rd, "rd_maps"), single_map(rd, "rd_maps")};
  app.az_overrides = {std::vector<double>{-0.1, 0.0, 0.1},
                      linspace(-0.2, 0.2, 11)};

  TensorMapCapture cube_cap;
  app.cube = &cube_cap;
  app.run();

  ASSERT_EQ(cube_cap.frames.size(), 2u);
  EXPECT_EQ(cube_cap.frames[0].shape, (std::vector<int64_t>{3, 1, 1, 1}));
  EXPECT_EQ(cube_cap.frames[1].shape, (std::vector<int64_t>{11, 1, 1, 1}));
}

class ElGridOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> rd_frames;
  std::vector<std::vector<double>> el_overrides;
  TensorMapCapture* cube = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<PhaseShiftBeamformerOp<double>>(
        "beamformer", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_overrides.front());
    holoscan::radar::testing::wire_data_source(this, op, "rd_maps", rd_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "el_angles",
                                                    el_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "rda_cube", cube);
  }
};

TEST(PhaseShiftBeamformerOpTest, LiveElGridStickyOverride) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom, conv, 0.1, 0.05);
  auto rd = vector_to_rd_maps<double>(vec, {geom.num_elements(), 1, 1});

  ElGridOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.3, 0.3, 31);
  app.el_overrides = {std::vector<double>{0.0}, linspace(-0.2, 0.2, 11)};
  app.rd_frames = {single_map(rd, "rd_maps"), single_map(rd, "rd_maps")};

  TensorMapCapture cube_cap;
  app.cube = &cube_cap;
  app.run();

  ASSERT_EQ(cube_cap.frames.size(), 2u);
  EXPECT_EQ(cube_cap.frames[0].shape, (std::vector<int64_t>{31, 1, 1, 1}));
  EXPECT_EQ(cube_cap.frames[1].shape, (std::vector<int64_t>{31, 11, 1, 1}));
}

class ArrayGeometryOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> rd_frames;
  std::vector<ArrayGeometry> geometry_overrides;
  TensorMapCapture* cube = nullptr;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<PhaseShiftBeamformerOp<double>>(
        "beamformer", Arg("array_geometry") = geometry_overrides.front(),
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid);
    holoscan::radar::testing::wire_data_source(this, op, "rd_maps", rd_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "array_geometry",
                                                    geometry_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "rda_cube", cube);
  }
};

TEST(PhaseShiftBeamformerOpTest, LiveArrayGeometryStickyOverride) {
  constexpr int64_t kN = 4;
  const auto geom_a = make_ula_geometry(kN, 0.5, 1.0);
  auto geom_b = make_ula_geometry(kN, 0.5, 1.0);
  geom_b.wavelength = 0.8;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom_a, conv, 0.05, 0.0);
  auto rd = vector_to_rd_maps<double>(vec, {kN, 1, 1});

  ArrayGeometryOverrideApp app;
  app.convention = conv;
  app.az_grid = linspace(-0.2, 0.2, 11);
  app.el_grid = {0.0};
  app.geometry_overrides = {geom_a, geom_b};
  app.rd_frames = {single_map(rd, "rd_maps"), single_map(rd, "rd_maps")};

  TensorMapCapture cube_cap;
  app.cube = &cube_cap;
  app.run();

  ASSERT_EQ(cube_cap.frames.size(), 2u);
  EXPECT_EQ(cube_cap.frames[0].shape, cube_cap.frames[1].shape);
  // A changed wavelength rebuilds the steering weights -> a different cube.
  const auto* a = cube_cap.frames[0].as<std::complex<double>>();
  const auto* b = cube_cap.frames[1].as<std::complex<double>>();
  double max_abs_diff = 0.0;
  for (int64_t i = 0; i < cube_cap.frames[0].size(); ++i) {
    max_abs_diff = std::max(max_abs_diff,
                            std::abs(a[static_cast<size_t>(i)] -
                                     b[static_cast<size_t>(i)]));
  }
  EXPECT_GT(max_abs_diff, 1e-6);
}

class ArrayTaperOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> rd_frames;
  std::vector<WindowSpec> taper_overrides;
  TensorMapCapture* cube = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<PhaseShiftBeamformerOp<double>>(
        "beamformer", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid,
        Arg("array_taper") = taper_overrides.front());
    holoscan::radar::testing::wire_data_source(this, op, "rd_maps", rd_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "array_taper",
                                                    taper_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "rda_cube", cube);
  }
};

TEST(PhaseShiftBeamformerOpTest, LiveArrayTaperStickyOverride) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);
  auto rd = vector_to_rd_maps<double>(vec, {kN, 1, 1});

  WindowSpec hann;
  hann.type = WindowType::kHann;

  ArrayTaperOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-1.0, 1.0, 201);
  app.el_grid = {0.0};
  app.taper_overrides = {WindowSpec{}, hann};
  app.rd_frames = {single_map(rd, "rd_maps"), single_map(rd, "rd_maps")};

  TensorMapCapture cube_cap;
  app.cube = &cube_cap;
  app.run();

  ASSERT_EQ(cube_cap.frames.size(), 2u);
  const auto sidelobe_ratio = [](const HostTensor& cube,
                                 const std::vector<double>& az) {
    double peak = 0.0;
    double sidelobe = 0.0;
    const auto* data = cube.as<std::complex<double>>();
    for (int64_t ia = 0; ia < 201; ++ia) {
      const auto v = data[static_cast<size_t>(ia)];
      const double p = v.real() * v.real() + v.imag() * v.imag();
      peak = std::max(peak, p);
      if (std::abs(az[static_cast<size_t>(ia)]) > 0.65) {
        sidelobe = std::max(sidelobe, p);
      }
    }
    return sidelobe / peak;
  };
  const double rect_ratio = sidelobe_ratio(cube_cap.frames[0], app.az_grid);
  const double hann_ratio = sidelobe_ratio(cube_cap.frames[1], app.az_grid);
  EXPECT_LT(hann_ratio, rect_ratio);
}

}  // namespace
