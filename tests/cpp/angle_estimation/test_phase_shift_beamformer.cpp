// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PhaseShiftBeamformerOp contract tests (HSDK 5 Graph harness): conventional
// dense-path Bartlett beamforming (rd_maps → rda_cube), Params coverage, and
// error paths, including single-tick Exact live overrides.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::PhaseShiftBeamformerOp;
using holoscan::radar::PhaseShiftBeamformerParams;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

using holoscan::radar::kPi;

ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing, double wavelength) {
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

ArrayGeometry make_planar_ura_geometry(int64_t ny, int64_t nz, double spacing, double wavelength) {
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
std::vector<std::complex<double>> make_steering_vector(const ArrayGeometry& geom,
                                                       const AngleConvention& c, double az,
                                                       double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(c, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> v(static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase =
        k * (geom.element_positions[base] * dir.x + geom.element_positions[base + 1] * dir.y +
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
std::shared_ptr<holoscan::Tensor> vector_to_rd_maps(const std::vector<std::complex<double>>& v,
                                                    const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    cast[i] = {static_cast<T>(v[i].real()), static_cast<T>(v[i].imag())};
  }
  return to_device_tensor(cast, shape, complex_dtype<T>());
}

template <typename T>
double cube_power(const HostTensor& cube, int64_t ia, int64_t ie, int64_t d, int64_t r,
                  int64_t num_el, int64_t num_doppler, int64_t num_range) {
  const int64_t idx = (((ia * num_el + ie) * num_doppler + d) * num_range + r);
  const std::complex<T> v = cube.template as<std::complex<T>>()[static_cast<size_t>(idx)];
  return static_cast<double>(v.real()) * static_cast<double>(v.real()) +
         static_cast<double>(v.imag()) * static_cast<double>(v.imag());
}

// (ia, ie) of the peak-power beam at range-Doppler cell (d, r).
template <typename T>
std::pair<int64_t, int64_t> peak_beam_at(const HostTensor& cube, int64_t num_az, int64_t num_el,
                                         int64_t num_doppler, int64_t num_range, int64_t d,
                                         int64_t r) {
  double best = -1.0;
  std::pair<int64_t, int64_t> arg{-1, -1};
  for (int64_t ia = 0; ia < num_az; ++ia) {
    for (int64_t ie = 0; ie < num_el; ++ie) {
      const double p = cube_power<T>(cube, ia, ie, d, r, num_el, num_doppler, num_range);
      if (p > best) {
        best = p;
        arg = {ia, ie};
      }
    }
  }
  return arg;
}

PhaseShiftBeamformerParams make_params(ArrayGeometry geom, AngleConvention conv,
                                       std::vector<double> az, std::vector<double> el,
                                       WindowSpec taper = {}) {
  PhaseShiftBeamformerParams params;
  params.array_geometry = std::move(geom);
  params.angle_convention = conv;
  params.az_coordinates = std::move(az);
  params.el_coordinates = std::move(el);
  params.array_taper = std::move(taper);
  return params;
}

template <typename T>
holoscan::testing::TestResult run_beamformer_result(
    std::shared_ptr<holoscan::Tensor> rd_maps, PhaseShiftBeamformerParams params,
    std::optional<holoscan::radar::MaxTensorExtents> extents = std::nullopt) {
  using Op = PhaseShiftBeamformerOp<T>;
  const auto rd_shape = rd_maps->shape();
  const auto sized = extents.value_or(holoscan::radar::testing::extents_for_scan_grid(
      params.az_coordinates.size(), params.el_coordinates.size(),
      rd_shape.size() > 1U ? std::optional<std::size_t>(static_cast<std::size_t>(rd_shape[1]))
                           : std::nullopt,
      rd_shape.size() > 2U ? std::optional<std::size_t>(static_cast<std::size_t>(rd_shape[2]))
                           : std::nullopt));
  auto fixture = holoscan::testing::create_operator_test<Op>(sized);
  fixture.set_params(std::move(params));
  fixture.push(&Op::rd_maps, *rd_maps);
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"rda_cube", "az_grid", "el_grid"};
  for (const char* output : kOutputs) {
    fixture.template capture_snapshot<holoscan::Tensor>(output, 1U, kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  for (const char* output : kOutputs) {
    (void)running.wait_for_output(output, 1U, std::chrono::seconds{5});
  }
  running.request_stop();
  return running.finish();
}

template <typename T>
holoscan::testing::TestResult run_beamformer_with_grids(std::shared_ptr<holoscan::Tensor> rd_maps,
                                                        PhaseShiftBeamformerParams params) {
  return run_beamformer_result<T>(std::move(rd_maps), std::move(params));
}

std::vector<HostTensor> beamformer_frames(const holoscan::testing::TestResult& result,
                                          const char* port) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.template snapshots<holoscan::Tensor>(port)) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

template <typename T>
HostTensor run_beamformer(std::shared_ptr<holoscan::Tensor> rd_maps,
                          PhaseShiftBeamformerParams params) {
  const auto result = run_beamformer_result<T>(std::move(rd_maps), std::move(params));
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("rda_cube"));
}
struct LiveBeamformerTick {
  std::shared_ptr<holoscan::Tensor> rd_maps;
  std::optional<ArrayGeometry> geometry;
  std::optional<std::vector<double>> az;
  std::optional<std::vector<double>> el;
  std::optional<WindowSpec> taper;
  bool malformed_geometry{false};
};

holoscan::testing::TestResult run_live_beamformer_graph(
    std::vector<LiveBeamformerTick> ticks, PhaseShiftBeamformerParams params,
    std::optional<holoscan::radar::MaxTensorExtents> extents = std::nullopt) {
  using Op = PhaseShiftBeamformerOp<double>;
  const auto sized = [&] {
    if (extents)
      return *extents;
    std::size_t max_az = params.az_coordinates.size();
    std::size_t max_el = params.el_coordinates.size();
    std::size_t max_doppler = 1U;
    std::size_t max_range = 1U;
    for (const auto& tick : ticks) {
      if (tick.az)
        max_az = std::max(max_az, tick.az->size());
      if (tick.el)
        max_el = std::max(max_el, tick.el->size());
      const auto& shape = tick.rd_maps->shape();
      if (shape.size() > 1U) {
        max_doppler = std::max(max_doppler, static_cast<std::size_t>(shape[1]));
      }
      if (shape.size() > 2U) {
        max_range = std::max(max_range, static_cast<std::size_t>(shape[2]));
      }
    }
    return holoscan::radar::testing::extents_for_scan_grid(max_az, max_el, max_doppler, max_range);
  }();
  auto fixture = holoscan::testing::create_operator_test<Op>(sized);
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto& tick = ticks[index];
    const auto maps = holoscan::testing::input("rd_maps", *tick.rd_maps);
    if (tick.geometry.has_value()) {
      auto wire = holoscan::radar::to_wire(*tick.geometry);
      if (tick.malformed_geometry) {
        wire.packed_geometry = holoscan::radar::testing::to_pageable_host_tensor<double>(
            {1.0}, {1}, holoscan::radar::real_dtype<double>());
      }
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("array_geometry", std::move(wire)), maps));
    } else if (tick.az.has_value()) {
      auto coordinates = holoscan::radar::testing::to_pageable_host_tensor<double>(
          *tick.az, {static_cast<int64_t>(tick.az->size())}, holoscan::radar::real_dtype<double>());
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("az_coordinates", *coordinates), maps));
    } else if (tick.el.has_value()) {
      auto coordinates = holoscan::radar::testing::to_pageable_host_tensor<double>(
          *tick.el, {static_cast<int64_t>(tick.el->size())}, holoscan::radar::real_dtype<double>());
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("el_coordinates", *coordinates), maps));
    } else if (tick.taper.has_value()) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time,
          holoscan::testing::input("array_taper", holoscan::radar::to_wire(*tick.taper)), maps));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(capture_time, maps));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"rda_cube", "az_grid", "el_grid"};
  for (const char* output : kOutputs) {
    fixture.capture_snapshot<holoscan::Tensor>(output, ticks.size(), kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  for (const char* output : kOutputs) {
    (void)running.wait_for_output(output, ticks.size(), std::chrono::seconds{5});
  }
  running.request_stop();
  return running.finish();
}

TEST(PhaseShiftBeamformerOpTest, DefaultsToRadarBoresightConvention) {
  EXPECT_EQ(PhaseShiftBeamformerParams{}.angle_convention.preset,
            AngleConventionPreset::kRadarBoresight);
}

template <typename T>
void expect_effective_grids() {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const std::vector<double> az{-0.2, 0.0, 0.3};
  const std::vector<double> el{-0.1, 0.2};
  auto rd =
      vector_to_rd_maps<T>(make_steering_vector(geom, AngleConvention{}, 0.0, 0.0), {kN, 1, 1});
  const auto out =
      run_beamformer_with_grids<T>(std::move(rd), make_params(geom, AngleConvention{}, az, el));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamformer_frames(out, "rda_cube").size(), 1U);
  ASSERT_EQ(beamformer_frames(out, "az_grid").size(), 1U);
  ASSERT_EQ(beamformer_frames(out, "el_grid").size(), 1U);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[0].shape, (std::vector<int64_t>{3, 2, 1, 1}));
  EXPECT_EQ(beamformer_frames(out, "az_grid")[0].shape, (std::vector<int64_t>{3}));
  EXPECT_EQ(beamformer_frames(out, "el_grid")[0].shape, (std::vector<int64_t>{2}));
  for (std::size_t i = 0; i < az.size(); ++i) {
    EXPECT_NEAR(beamformer_frames(out, "az_grid")[0].template as<T>()[i], az[i], 1e-6);
  }
  for (std::size_t i = 0; i < el.size(); ++i) {
    EXPECT_NEAR(beamformer_frames(out, "el_grid")[0].template as<T>()[i], el[i], 1e-6);
  }
}

TEST(PhaseShiftBeamformerOpTest, PublishesEffectiveGridsFloat) {
  expect_effective_grids<float>();
}

TEST(PhaseShiftBeamformerOpTest, PublishesEffectiveGridsDouble) {
  expect_effective_grids<double>();
}

TEST(PhaseShiftBeamformerOpTest, OptionalGridsMayRemainUnwired) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto rd = vector_to_rd_maps<double>(make_steering_vector(geom, AngleConvention{}, 0.0, 0.0),
                                      {kN, 1, 1});
  const HostTensor cube = run_beamformer<double>(
      std::move(rd), make_params(geom, AngleConvention{}, {-0.1, 0.0, 0.1}, {0.0}));
  EXPECT_EQ(cube.shape, (std::vector<int64_t>{3, 1, 1, 1}));
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
  const auto cube = run_beamformer<double>(rd, make_params(geom, conv, az, el));

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
  const auto cube =
      run_beamformer<double>(rd, make_params(geom, conv, az, std::vector<double>{0.0}));

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

  for (const auto preset :
       {AngleConventionPreset::kRadarBoresight, AngleConventionPreset::kAerospaceNed}) {
    AngleConvention conv;
    conv.preset = preset;
    const auto vec = make_steering_vector(geom, conv, kTruthAz, kTruthEl);
    auto rd = vector_to_rd_maps<double>(vec, {geom.num_elements(), 1, 1});
    const auto az = linspace(-0.5, 0.5, 101);
    const auto el = linspace(-0.3, 0.3, 61);
    const auto cube = run_beamformer<double>(rd, make_params(geom, conv, az, el));
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

  const auto uu = linspace(-1.0, 1.0, 5);
  const auto vv = linspace(-1.0, 1.0, 5);
  const auto cube = run_beamformer<double>(rd, make_params(geom, conv, uu, vv));

  // Corner (u,v)=(-1,-1): u^2+v^2 > 1 -> invisible -> zero weight -> zero.
  EXPECT_NEAR(cube_power<double>(cube, 0, 0, 0, 0, 5, 1, 1), 0.0, 1e-9);
  // Center (u,v)=(0,0) carries the target energy.
  EXPECT_GT(cube_power<double>(cube, 2, 2, 0, 0, 5, 1, 1), 0.0);
}

TEST(PhaseShiftBeamformerOpTest, RejectsInvalidScanCoordinates) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  AngleConvention direction_cosine;
  direction_cosine.preset = AngleConventionPreset::kDirectionCosine;
  auto out_of_range =
      run_beamformer_result<double>(to_device_tensor(host, {kN, 1, 1}, complex_dtype<double>()),
                                    make_params(geom, direction_cosine, {1.01}, {0.0}));
  EXPECT_FALSE(out_of_range.diagnostics().empty());

  auto non_finite = run_beamformer_result<double>(
      to_device_tensor(host, {kN, 1, 1}, complex_dtype<double>()),
      make_params(geom, AngleConvention{}, {std::numeric_limits<double>::infinity()}, {0.0}));
  EXPECT_FALSE(non_finite.diagnostics().empty());
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
    const auto cube = run_beamformer<double>(rd, make_params(geom, conv, az, el, taper));
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

TEST(PhaseShiftBeamformerOpTest, UraArrayTaperLowersBothAxisSidelobes) {
  constexpr int64_t kNy = 8;
  constexpr int64_t kNz = 8;
  constexpr int64_t kGrid = 201;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);
  const auto angles = linspace(-1.0, 1.0, kGrid);

  const auto sidelobe_ratio = [&](const WindowSpec& taper, bool scan_azimuth) {
    auto rd = vector_to_rd_maps<double>(vec, {kNy * kNz, 1, 1});
    const std::vector<double> az = scan_azimuth ? angles : std::vector<double>{0.0};
    const std::vector<double> el = scan_azimuth ? std::vector<double>{0.0} : angles;
    const auto cube = run_beamformer<double>(rd, make_params(geom, conv, az, el, taper));
    double peak = 0.0;
    double sidelobe = 0.0;
    for (int64_t i = 0; i < kGrid; ++i) {
      const int64_t ia = scan_azimuth ? i : 0;
      const int64_t ie = scan_azimuth ? 0 : i;
      const double power =
          cube_power<double>(cube, ia, ie, 0, 0, static_cast<int64_t>(el.size()), 1, 1);
      peak = std::max(peak, power);
      if (std::abs(angles[static_cast<size_t>(i)]) > 0.65) {
        sidelobe = std::max(sidelobe, power);
      }
    }
    return sidelobe / peak;
  };

  WindowSpec hann;
  hann.type = WindowType::kHann;
  const double rect_az = sidelobe_ratio(WindowSpec{}, true);
  const double rect_el = sidelobe_ratio(WindowSpec{}, false);
  const double tapered_az = sidelobe_ratio(hann, true);
  const double tapered_el = sidelobe_ratio(hann, false);
  EXPECT_GT(rect_az, 0.0);
  EXPECT_GT(rect_el, 0.0);
  EXPECT_LT(tapered_az, rect_az);
  EXPECT_LT(tapered_el, rect_el);
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
  std::vector<std::complex<double>> host(static_cast<size_t>(kN * kD * kR), {0.0, 0.0});
  const auto set = [&](int64_t c, int64_t d, int64_t r, std::complex<double> val) {
    host[static_cast<size_t>((c * kD + d) * kR + r)] = val;
  };
  for (int64_t c = 0; c < kN; ++c) {
    set(c, 0, 0, v0[static_cast<size_t>(c)]);
    set(c, 1, 1, v1[static_cast<size_t>(c)]);
  }
  auto rd = to_device_tensor(std::vector<std::complex<double>>(host), {kN, kD, kR},
                             complex_dtype<double>());

  const auto az = linspace(-0.4, 0.4, 81);
  const auto cube =
      run_beamformer<double>(rd, make_params(geom, conv, az, std::vector<double>{0.0}));

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

TEST(PhaseShiftBeamformerOpTest, RejectsOpaqueLeadingBatch) {
  // Contract is (num_channels, D, R); an extra leading dim is opaque batch.
  constexpr int64_t kN = 8;
  constexpr int64_t kBatch = 2;
  constexpr double kAz0 = 0.12;
  constexpr double kAz1 = -0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto v0 = make_steering_vector(geom, conv, kAz0, 0.0);
  const auto v1 = make_steering_vector(geom, conv, kAz1, 0.0);

  std::vector<std::complex<double>> host;
  host.insert(host.end(), v0.begin(), v0.end());
  host.insert(host.end(), v1.begin(), v1.end());
  auto rd = to_device_tensor(host, {kBatch, kN, 1, 1}, complex_dtype<double>());

  const auto az = linspace(-0.4, 0.4, 81);
  const auto result =
      run_beamformer_result<double>(rd, make_params(geom, conv, az, std::vector<double>{0.0}));
  EXPECT_TRUE(beamformer_frames(result, "rda_cube").empty());
  EXPECT_TRUE(!result.ok() || !result.diagnostics().empty());
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
  const auto params = make_params(geom, conv, az, std::vector<double>{0.0});
  const auto cube_f = run_beamformer<float>(rd_f, params);
  const auto cube_d = run_beamformer<double>(rd_d, params);

  const auto [ia_f, ie_f] = peak_beam_at<float>(cube_f, 61, 1, 1, 1, 0, 0);
  const auto [ia_d, ie_d] = peak_beam_at<double>(cube_d, 61, 1, 1, 1, 0, 0);
  EXPECT_EQ(ia_f, ia_d);
  EXPECT_EQ(ie_f, ie_d);
}

TEST(PhaseShiftBeamformerOpTest, ScanGridConfigurableFromParams) {
  // Authoring Params (set_params) must configure az/el grid sizes.
  constexpr int64_t kN = 4;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);
  auto rd = vector_to_rd_maps<double>(vec, {kN, 1, 1});
  const auto cube = run_beamformer<double>(
      rd, make_params(geom, conv, linspace(-0.1, 0.1, 7), std::vector<double>{0.0, 0.05}));
  ASSERT_EQ(cube.shape, (std::vector<int64_t>{7, 2, 1, 1}));
}

TEST(PhaseShiftBeamformerOpTest, ArrayTaperConfigurableFromParams) {
  constexpr int64_t kN = 8;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);
  const auto az = linspace(-1.0, 1.0, 201);
  WindowSpec hann;
  hann.type = WindowType::kHann;
  auto rd_rect = vector_to_rd_maps<double>(vec, {kN, 1, 1});
  auto rd_hann = vector_to_rd_maps<double>(vec, {kN, 1, 1});
  const auto rect =
      run_beamformer<double>(rd_rect, make_params(geom, conv, az, std::vector<double>{0.0}));
  const auto tapered =
      run_beamformer<double>(rd_hann, make_params(geom, conv, az, std::vector<double>{0.0}, hann));
  const auto sidelobe_ratio = [](const HostTensor& cube, const std::vector<double>& grid) {
    double peak = 0.0;
    double sidelobe = 0.0;
    const auto* data = cube.as<std::complex<double>>();
    for (int64_t ia = 0; ia < 201; ++ia) {
      const auto v = data[static_cast<size_t>(ia)];
      const double p = v.real() * v.real() + v.imag() * v.imag();
      peak = std::max(peak, p);
      if (std::abs(grid[static_cast<size_t>(ia)]) > 0.65) {
        sidelobe = std::max(sidelobe, p);
      }
    }
    return sidelobe / peak;
  };
  EXPECT_LT(sidelobe_ratio(tapered, az), sidelobe_ratio(rect, az));
}

TEST(PhaseShiftBeamformerOpTest, RejectsRankTooLow) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto rd = to_device_tensor(host, {kN, 1}, complex_dtype<double>());
  const auto result = run_beamformer_result<double>(
      rd, make_params(geom, conv, std::vector<double>{0.0}, std::vector<double>{0.0}));
  EXPECT_TRUE(beamformer_frames(result, "rda_cube").empty());
  EXPECT_TRUE(!result.ok() || !result.diagnostics().empty());
}

TEST(PhaseShiftBeamformerOpTest, RejectsInputBeyondCapturedExtents) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto steering = make_steering_vector(geom, AngleConvention{}, 0.0, 0.0);
  std::vector<std::complex<double>> host;
  host.reserve(static_cast<std::size_t>(kN * 2 * 3));
  for (const auto value : steering) {
    host.insert(host.end(), 6U, value);
  }
  auto rd = to_device_tensor(host, {kN, 2, 3}, complex_dtype<double>());
  auto extents =
      holoscan::radar::testing::extents_for_scan_grid(1, 1, /*max_doppler=*/1, /*max_range=*/3);
  const auto result = run_beamformer_result<double>(
      rd, make_params(geom, AngleConvention{}, {0.0}, {0.0}), extents);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamformer_frames(result, "rda_cube").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(PhaseShiftBeamformerOpTest, CacheKeyIncludesCustomConventionAxes) {
  const auto geom = make_ula_geometry(4, 0.5, 1.0);
  AngleConvention first;
  first.preset = AngleConventionPreset::kCustom;
  AngleConvention second = first;
  second.boresight_axis = holoscan::radar::SignedAxis::kPlusY;
  second.azimuth_positive_axis = holoscan::radar::SignedAxis::kMinusX;

  const std::string first_key =
      holoscan::radar::steering_cache_key(geom, first, {0.0}, {0.0}, WindowSpec{});
  const std::string second_key =
      holoscan::radar::steering_cache_key(geom, second, {0.0}, {0.0}, WindowSpec{});
  EXPECT_NE(first_key, second_key);

  const std::string close_grid_key =
      holoscan::radar::steering_cache_key(geom, first, {0.000000000000001}, {0.0}, WindowSpec{});
  EXPECT_NE(first_key, close_grid_key);
}

TEST(PhaseShiftBeamformerOpTest, RejectsDegenerateCustomConvention) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto rd = vector_to_rd_maps<double>(make_steering_vector(geom, AngleConvention{}, 0.0, 0.0),
                                      {kN, 1, 1});
  AngleConvention invalid;
  invalid.preset = AngleConventionPreset::kCustom;
  invalid.azimuth_positive_axis = invalid.boresight_axis;
  const auto result = run_beamformer_result<double>(rd, make_params(geom, invalid, {0.0}, {0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamformer_frames(result, "rda_cube").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(PhaseShiftBeamformerOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN - 1), {1.0, 0.0});
  auto rd = to_device_tensor(host, {kN - 1, 1, 1}, complex_dtype<double>());
  const auto result = run_beamformer_result<double>(
      rd, make_params(geom, conv, std::vector<double>{0.0}, std::vector<double>{0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamformer_frames(result, "rda_cube").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(PhaseShiftBeamformerOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto rd = to_device_tensor(host, {kN, 1, 1}, complex_dtype<double>());
  const auto result = run_beamformer_result<double>(
      rd, make_params(geom, conv, std::vector<double>{}, std::vector<double>{0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamformer_frames(result, "rda_cube").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(PhaseShiftBeamformerOpTest, RejectsDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);
  auto rd_f = vector_to_rd_maps<float>(vec, {kN, 1, 1});
  const auto result = run_beamformer_result<double>(
      rd_f, make_params(geom, conv, std::vector<double>{0.0}, std::vector<double>{0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamformer_frames(result, "rda_cube").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(PhaseShiftBeamformerOpTest, LiveAzCoordinatesAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto vec = make_steering_vector(geom, AngleConvention{}, 0.0, 0.0);
  auto rd = vector_to_rd_maps<double>(vec, {kN, 1, 1});
  auto params = make_params(geom, AngleConvention{}, {-0.1, 0.0, 0.1}, {0.0});
  const auto out = run_live_beamformer_graph(
      {{rd}, {rd, std::nullopt, linspace(-0.2, 0.2, 5)}, {rd}}, std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamformer_frames(out, "rda_cube").size(), 3U);
  ASSERT_EQ(beamformer_frames(out, "az_grid").size(), 3U);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[0].shape[0], 3);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[1].shape[0], 5);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[2].shape[0], 3);
  EXPECT_EQ(beamformer_frames(out, "az_grid")[0].shape[0], 3);
  EXPECT_EQ(beamformer_frames(out, "az_grid")[1].shape[0], 5);
  EXPECT_EQ(beamformer_frames(out, "az_grid")[2].shape[0], 3);
  EXPECT_DOUBLE_EQ(beamformer_frames(out, "az_grid")[0].as<double>()[0], -0.1);
  EXPECT_DOUBLE_EQ(beamformer_frames(out, "az_grid")[1].as<double>()[0], -0.2);
  EXPECT_DOUBLE_EQ(beamformer_frames(out, "az_grid")[2].as<double>()[0], -0.1);
}

TEST(PhaseShiftBeamformerOpTest, LiveElCoordinatesAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto rd = vector_to_rd_maps<double>(make_steering_vector(geom, AngleConvention{}, 0.0, 0.0),
                                      {kN, 1, 1});
  auto params = make_params(geom, AngleConvention{}, {0.0}, {0.0});
  const auto out = run_live_beamformer_graph(
      {{rd}, {rd, std::nullopt, std::nullopt, std::vector<double>{-0.1, 0.0, 0.1}}, {rd}},
      std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamformer_frames(out, "rda_cube").size(), 3U);
  ASSERT_EQ(beamformer_frames(out, "el_grid").size(), 3U);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[0].shape[1], 1);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[1].shape[1], 3);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[2].shape[1], 1);
  EXPECT_EQ(beamformer_frames(out, "el_grid")[0].shape[0], 1);
  EXPECT_EQ(beamformer_frames(out, "el_grid")[1].shape[0], 3);
  EXPECT_EQ(beamformer_frames(out, "el_grid")[2].shape[0], 1);
  EXPECT_DOUBLE_EQ(beamformer_frames(out, "el_grid")[0].as<double>()[0], 0.0);
  EXPECT_DOUBLE_EQ(beamformer_frames(out, "el_grid")[1].as<double>()[0], -0.1);
  EXPECT_DOUBLE_EQ(beamformer_frames(out, "el_grid")[2].as<double>()[0], 0.0);
}

TEST(PhaseShiftBeamformerOpTest, LiveArrayGeometryAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto changed = make_ula_geometry(kN, 0.25, 1.0);
  auto rd = vector_to_rd_maps<double>(make_steering_vector(geom, AngleConvention{}, 0.2, 0.0),
                                      {kN, 1, 1});
  auto params = make_params(geom, AngleConvention{}, linspace(-0.4, 0.4, 9), {0.0});
  const auto out = run_live_beamformer_graph({{rd}, {rd, changed}, {rd}}, std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamformer_frames(out, "rda_cube").size(), 3U);
  EXPECT_NE(beamformer_frames(out, "rda_cube")[0].data, beamformer_frames(out, "rda_cube")[1].data);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[0].data, beamformer_frames(out, "rda_cube")[2].data);
}

TEST(PhaseShiftBeamformerOpTest, LiveArrayTaperAppliesForOneTick) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto rd = vector_to_rd_maps<double>(make_steering_vector(geom, AngleConvention{}, 0.15, 0.0),
                                      {kN, 1, 1});
  WindowSpec hann;
  hann.type = WindowType::kHann;
  auto params = make_params(geom, AngleConvention{}, linspace(-0.4, 0.4, 17), {0.0});
  const auto out = run_live_beamformer_graph(
      {{rd}, {rd, std::nullopt, std::nullopt, std::nullopt, hann}, {rd}}, std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamformer_frames(out, "rda_cube").size(), 3U);
  EXPECT_NE(beamformer_frames(out, "rda_cube")[0].data, beamformer_frames(out, "rda_cube")[1].data);
  EXPECT_EQ(beamformer_frames(out, "rda_cube")[0].data, beamformer_frames(out, "rda_cube")[2].data);
}

TEST(PhaseShiftBeamformerOpTest, LiveOverridesRejectMalformedAndBounds) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto rd = vector_to_rd_maps<double>(make_steering_vector(geom, AngleConvention{}, 0.0, 0.0),
                                      {kN, 1, 1});
  auto params = make_params(geom, AngleConvention{}, {0.0}, {0.0});
  const auto malformed = run_live_beamformer_graph(
      {{rd, geom, std::nullopt, std::nullopt, std::nullopt, true}}, params);
  ASSERT_FALSE(malformed.ok());
  EXPECT_TRUE(beamformer_frames(malformed, "rda_cube").empty());
  EXPECT_FALSE(malformed.diagnostics().empty());

  auto extents = holoscan::radar::testing::extents_for_scan_grid(2U, 1U, 1U, 1U);
  const auto bounded = run_live_beamformer_graph(
      {{rd, std::nullopt, std::vector<double>{-0.1, 0.0, 0.1}}}, std::move(params), extents);
  ASSERT_FALSE(bounded.ok());
  EXPECT_TRUE(beamformer_frames(bounded, "rda_cube").empty());
  EXPECT_FALSE(bounded.diagnostics().empty());
}

}  // namespace
