// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device steering-weight builder tests: the GPU-built table must match the
// host build_steering_table reference (conjugated, in the requested beam
// order / layout) within precision tolerance.

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/dsp/window.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::BeamOrder;
using holoscan::radar::generate_steering_taper;
using holoscan::radar::generate_window;
using holoscan::radar::build_steering_table;
using holoscan::radar::build_steering_weights_device;
using holoscan::radar::SteeringLayout;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_host;

ArrayGeometry make_ula(int64_t n, double spacing, double wavelength) {
  ArrayGeometry g;
  g.wavelength = wavelength;
  g.geometry_type = holoscan::radar::ArrayGeometryType::kUla;
  for (int64_t m = 0; m < n; ++m) {
    g.element_positions.push_back(0.0);
    g.element_positions.push_back(static_cast<double>(m) * spacing);
    g.element_positions.push_back(0.0);
  }
  return g;
}

ArrayGeometry make_ura(int64_t ny, int64_t nz, double spacing, double wl) {
  ArrayGeometry g;
  g.wavelength = wl;
  g.geometry_type = holoscan::radar::ArrayGeometryType::kUra;
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      g.element_positions.push_back(0.0);
      g.element_positions.push_back(static_cast<double>(iy) * spacing);
      g.element_positions.push_back(static_cast<double>(iz) * spacing);
    }
  }
  return g;
}

std::vector<double> linspace(double a, double b, int64_t n) {
  std::vector<double> out(static_cast<size_t>(n));
  if (n == 1) {
    out[0] = a;
    return out;
  }
  const double step = (b - a) / static_cast<double>(n - 1);
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = a + step * static_cast<double>(i);
  }
  return out;
}

TEST(SteeringTaperTest, UraUsesSeparableAxisProduct) {
  const auto geom = make_ura(4, 3, 0.5, 1.0);
  WindowSpec taper;
  taper.type = WindowType::kTaylor;
  taper.sidelobe_attenuation_db = 35.0;
  const auto actual = generate_steering_taper(geom, taper);
  const auto taper_y = generate_window(taper, 4);
  const auto taper_z = generate_window(taper, 3);
  const auto flat = generate_window(taper, 12);

  ASSERT_EQ(actual.size(), 12U);
  bool differs_from_flat = false;
  for (int64_t iz = 0; iz < 3; ++iz) {
    for (int64_t iy = 0; iy < 4; ++iy) {
      const size_t index = static_cast<size_t>(iz * 4 + iy);
      EXPECT_NEAR(actual[index],
                  taper_y[static_cast<size_t>(iy)] *
                      taper_z[static_cast<size_t>(iz)],
                  1e-12);
      differs_from_flat =
          differs_from_flat || std::abs(actual[index] - flat[index]) > 1e-6;
    }
  }
  EXPECT_TRUE(differs_from_flat);
}

TEST(SteeringTaperTest, UlaPreservesElementWiseWindow) {
  const auto geom = make_ula(8, 0.5, 1.0);
  WindowSpec taper;
  taper.type = WindowType::kHann;
  const auto actual = generate_steering_taper(geom, taper);
  const auto expected = generate_window(taper, 8);
  EXPECT_EQ(actual, expected);
}

TEST(SteeringTaperTest, UraCustomCoefficientsRemainPerElement) {
  const auto geom = make_ura(4, 3, 0.5, 1.0);
  WindowSpec taper;
  taper.type = WindowType::kCustom;
  taper.coherent_gain_normalization = false;
  taper.coefficients = {0.1, 0.2, 0.3, 0.4, 1.1, 1.2,
                        1.3, 1.4, 2.1, 2.2, 2.3, 2.4};
  EXPECT_EQ(generate_steering_taper(geom, taper), taper.coefficients);
}

TEST(SteeringTaperTest, UraCoherentGainNormalizationHasUnitMean) {
  const auto geom = make_ura(4, 3, 0.5, 1.0);
  WindowSpec taper;
  taper.type = WindowType::kHamming;
  taper.coherent_gain_normalization = true;
  const auto actual = generate_steering_taper(geom, taper);
  const double mean =
      std::accumulate(actual.begin(), actual.end(), 0.0) /
      static_cast<double>(actual.size());
  EXPECT_NEAR(mean, 1.0, 1e-12);
}

// Compare the device build against the host reference for a given order/layout.
template <typename T>
void expect_matches_host(const ArrayGeometry& geom, const AngleConvention& conv,
                         const std::vector<double>& az,
                         const std::vector<double>& el,
                         const WindowSpec& taper, BeamOrder order,
                         SteeringLayout layout, double tol) {
  const auto ref = build_steering_table(geom, conv, az, el, taper);
  auto dev = build_steering_weights_device<T>(geom, conv, az, el, taper, order,
                                              layout, /*conjugate=*/true,
                                              /*stream=*/0);
  ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
  const HostTensor host = to_host(*dev);
  const auto* d = host.as<std::complex<T>>();

  const int64_t num_az = static_cast<int64_t>(az.size());
  const int64_t num_el = static_cast<int64_t>(el.size());
  const int64_t num_beams = num_az * num_el;
  const int64_t nch = geom.num_elements();

  for (int64_t ia = 0; ia < num_az; ++ia) {
    for (int64_t ie = 0; ie < num_el; ++ie) {
      const int64_t beam_el = ie * num_az + ia;  // host reference order
      const int64_t beam_dev =
          (order == BeamOrder::kAzMajor) ? ia * num_el + ie : beam_el;
      for (int64_t c = 0; c < nch; ++c) {
        const std::complex<double> w =
            ref.weights[static_cast<size_t>(beam_el * nch + c)];
        const int64_t idx = (layout == SteeringLayout::kChannelMajor)
                                ? c * num_beams + beam_dev
                                : beam_dev * nch + c;
        const std::complex<T> got = d[static_cast<size_t>(idx)];
        EXPECT_NEAR(static_cast<double>(got.real()), w.real(), tol);
        EXPECT_NEAR(static_cast<double>(got.imag()), -w.imag(), tol);
      }
    }
  }
}

TEST(SteeringWeightsDeviceTest, BeamMajorAzMajorMatchesHostUra) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ura(4, 4, 0.5, 1.0);
  expect_matches_host<double>(geom, conv, linspace(-0.5, 0.5, 21),
                              linspace(-0.3, 0.3, 11), WindowSpec{},
                              BeamOrder::kAzMajor, SteeringLayout::kBeamMajor,
                              1e-9);
}

TEST(SteeringWeightsDeviceTest, ChannelMajorElMajorMatchesHostUla) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula(8, 0.5, 1.0);
  expect_matches_host<double>(geom, conv, linspace(-0.4, 0.4, 31),
                              std::vector<double>{0.0}, WindowSpec{},
                              BeamOrder::kElMajor,
                              SteeringLayout::kChannelMajor, 1e-9);
}

TEST(SteeringWeightsDeviceTest, WithHannTaperMatchesHost) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula(8, 0.5, 1.0);
  WindowSpec hann;
  hann.type = WindowType::kHann;
  expect_matches_host<double>(geom, conv, linspace(-0.4, 0.4, 21),
                              std::vector<double>{0.0}, hann,
                              BeamOrder::kAzMajor, SteeringLayout::kBeamMajor,
                              1e-9);
}

TEST(SteeringWeightsDeviceTest, WithTaylorTaperMatchesHostUra) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ura(4, 3, 0.5, 1.0);
  WindowSpec taylor;
  taylor.type = WindowType::kTaylor;
  taylor.sidelobe_attenuation_db = 35.0;
  expect_matches_host<double>(geom, conv, linspace(-0.4, 0.4, 17),
                              linspace(-0.3, 0.3, 13), taylor,
                              BeamOrder::kAzMajor, SteeringLayout::kBeamMajor,
                              1e-9);
}

TEST(SteeringWeightsDeviceTest, FloatPrecisionMatchesHost) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ura(4, 4, 0.5, 1.0);
  expect_matches_host<float>(geom, conv, linspace(-0.5, 0.5, 17),
                             linspace(-0.3, 0.3, 9), WindowSpec{},
                             BeamOrder::kAzMajor, SteeringLayout::kBeamMajor,
                             2e-5);
}

TEST(SteeringWeightsDeviceTest, DirectionCosineMasksInvisibleBeams) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geom = make_ula(4, 0.5, 1.0);
  const auto uu = linspace(-1.0, 1.0, 5);
  const auto vv = linspace(-1.0, 1.0, 5);
  auto dev = build_steering_weights_device<double>(
      geom, conv, uu, vv, WindowSpec{}, BeamOrder::kAzMajor,
      SteeringLayout::kBeamMajor, /*conjugate=*/true, /*stream=*/0);
  ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
  const HostTensor host = to_host(*dev);
  const auto* d = host.as<std::complex<double>>();
  const int64_t nch = geom.num_elements();
  const int64_t num_el = 5;
  // Corner (u,v)=(-1,-1): u^2+v^2 > 1 -> invisible -> zero weights.
  const int64_t beam_corner = 0 * num_el + 0;  // ia=0, ie=0 (az-major)
  for (int64_t c = 0; c < nch; ++c) {
    const std::complex<double> w =
        d[static_cast<size_t>(beam_corner * nch + c)];
    EXPECT_NEAR(w.real(), 0.0, 1e-12);
    EXPECT_NEAR(w.imag(), 0.0, 1e-12);
  }
  // Center (u,v)=(0,0): visible -> non-zero.
  const int64_t beam_center = 2 * num_el + 2;
  double mag = 0.0;
  for (int64_t c = 0; c < nch; ++c) {
    const std::complex<double> w =
        d[static_cast<size_t>(beam_center * nch + c)];
    mag += std::abs(w);
  }
  EXPECT_GT(mag, 0.0);
}

TEST(SteeringWeightsDeviceTest, HostAndDeviceRejectInvalidScanCoordinates) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geom = make_ula(4, 0.5, 1.0);
  EXPECT_THROW(build_steering_table(geom, conv, {1.01}, {0.0}, WindowSpec{}),
               std::invalid_argument);
  EXPECT_THROW(
      build_steering_weights_device<double>(
          geom, conv, {1.01}, {0.0}, WindowSpec{}, BeamOrder::kAzMajor,
          SteeringLayout::kBeamMajor, /*conjugate=*/true, /*stream=*/0),
      std::invalid_argument);
}

TEST(SteeringWeightsDeviceTest, NonConjugateFlipsImaginarySign) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula(6, 0.5, 1.0);
  const auto az = linspace(0.1, 0.3, 5);
  const std::vector<double> el{0.0};
  auto conj = build_steering_weights_device<double>(
      geom, conv, az, el, WindowSpec{}, BeamOrder::kAzMajor,
      SteeringLayout::kBeamMajor, /*conjugate=*/true, 0);
  auto plain = build_steering_weights_device<double>(
      geom, conv, az, el, WindowSpec{}, BeamOrder::kAzMajor,
      SteeringLayout::kBeamMajor, /*conjugate=*/false, 0);
  ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
  const HostTensor hc = to_host(*conj);
  const HostTensor hp = to_host(*plain);
  const auto* c = hc.as<std::complex<double>>();
  const auto* p = hp.as<std::complex<double>>();
  for (int64_t i = 0; i < hc.size(); ++i) {
    EXPECT_NEAR(c[static_cast<size_t>(i)].real(),
                p[static_cast<size_t>(i)].real(), 1e-12);
    EXPECT_NEAR(c[static_cast<size_t>(i)].imag(),
                -p[static_cast<size_t>(i)].imag(), 1e-12);
  }
}

TEST(SteeringWeightsDeviceTest, RejectsEmptyScanGrid) {
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula(4, 0.5, 1.0);
  EXPECT_THROW(
      build_steering_weights_device<double>(
          geom, conv, std::vector<double>{}, std::vector<double>{0.0},
          WindowSpec{}, BeamOrder::kAzMajor, SteeringLayout::kBeamMajor, true,
          0),
      std::invalid_argument);
}

}  // namespace
