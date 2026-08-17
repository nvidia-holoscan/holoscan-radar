// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PhaseShiftBeamformerOp contract tests (HSDK 5 Graph harness): conventional
// dense-path Bartlett beamforming (rd_maps → rda_cube), Params coverage, and
// error paths. Live Graph sticky overrides deferred (HSDK5_PORT.md §1 / §9).

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

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
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
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

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

PhaseShiftBeamformerParams make_params(ArrayGeometry geom,
                                       AngleConvention conv,
                                       std::vector<double> az,
                                       std::vector<double> el,
                                       WindowSpec taper = {}) {
  PhaseShiftBeamformerParams params;
  params.array_geometry = std::move(geom);
  params.angle_convention = conv;
  params.az_angles = std::move(az);
  params.el_angles = std::move(el);
  params.array_taper = std::move(taper);
  return params;
}

template <typename Source, typename OpRef, typename Sink>
void wire_beamformer(holoscan::Graph& graph, const Source& source, OpRef& op,
                     const Sink& sink) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->rd_maps, q);
  graph.add_flow(op->rda_cube, sink->in, q);
}

template <typename T>
GraphRunResult run_beamformer_result(
    std::shared_ptr<holoscan::Tensor> rd_maps,
    PhaseShiftBeamformerParams params,
    const holoscan::radar::MaxTensorExtents& extents =
        holoscan::radar::testing::default_test_extents()) {
  return run_single_io_operator_graph_result<PhaseShiftBeamformerOp<T>>(
      {std::move(rd_maps)}, std::move(params), "rda_cube",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink) { wire_beamformer(graph, source, op, sink); },
      {},
      extents);
}

template <typename T>
HostTensor run_beamformer(std::shared_ptr<holoscan::Tensor> rd_maps,
                          PhaseShiftBeamformerParams params) {
  GraphRunResult result =
      run_beamformer_result<T>(std::move(rd_maps), std::move(params));
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
  if (result.frames.size() != 1U) {
    throw std::runtime_error(
        "PhaseShiftBeamformerOp test: expected one frame, got " +
        std::to_string(result.frames.size()));
  }
  return std::move(result.frames.front());
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
      rd, make_params(geom, conv, az, el));

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
      rd, make_params(geom, conv, az, std::vector<double>{0.0}));

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
        rd, make_params(geom, conv, az, el));
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
      rd, make_params(geom, conv, uu, vv));

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
        rd, make_params(geom, conv, az, el, taper));
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
      rd, make_params(geom, conv, az, std::vector<double>{0.0}));

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

TEST(PhaseShiftBeamformerOpTest,
     RejectsOpaqueLeadingBatchWhenExtentsOmitRxTx) {
  // Contract is (num_channels, D, R); an extra leading dim is opaque batch.
  // Omit max_rx / max_tx / max_channels so named leading rank is 0.
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
  auto rd = to_device_tensor(host, {kBatch, kN, 1, 1},
                             complex_dtype<double>());

  holoscan::radar::MaxTensorExtents extents =
      holoscan::radar::testing::default_test_extents();
  extents.max_rx.reset();
  extents.max_tx.reset();
  extents.max_channels.reset();

  const auto az = linspace(-0.4, 0.4, 81);
  const GraphRunResult result = run_beamformer_result<double>(
      rd, make_params(geom, conv, az, std::vector<double>{0.0}), extents);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
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
  const auto params =
      make_params(geom, conv, az, std::vector<double>{0.0});
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
      rd, make_params(geom, conv, linspace(-0.1, 0.1, 7),
                      std::vector<double>{0.0, 0.05}));
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
  const auto rect = run_beamformer<double>(
      rd_rect, make_params(geom, conv, az, std::vector<double>{0.0}));
  const auto tapered = run_beamformer<double>(
      rd_hann,
      make_params(geom, conv, az, std::vector<double>{0.0}, hann));
  const auto sidelobe_ratio = [](const HostTensor& cube,
                                 const std::vector<double>& grid) {
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
  const GraphRunResult result = run_beamformer_result<double>(
      rd, make_params(geom, conv, std::vector<double>{0.0},
                      std::vector<double>{0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(PhaseShiftBeamformerOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN - 1),
                                         {1.0, 0.0});
  auto rd = to_device_tensor(host, {kN - 1, 1, 1}, complex_dtype<double>());
  const GraphRunResult result = run_beamformer_result<double>(
      rd, make_params(geom, conv, std::vector<double>{0.0},
                      std::vector<double>{0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(PhaseShiftBeamformerOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto rd = to_device_tensor(host, {kN, 1, 1}, complex_dtype<double>());
  const GraphRunResult result = run_beamformer_result<double>(
      rd, make_params(geom, conv, std::vector<double>{},
                      std::vector<double>{0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(PhaseShiftBeamformerOpTest, RejectsDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto vec = make_steering_vector(geom, conv, 0.0, 0.0);
  auto rd_f = vector_to_rd_maps<float>(vec, {kN, 1, 1});
  const GraphRunResult result = run_beamformer_result<double>(
      rd_f, make_params(geom, conv, std::vector<double>{0.0},
                        std::vector<double>{0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(PhaseShiftBeamformerOpTest, RejectsPageableHostInput) {
  // [EA HSDK5 ONLY] Graph harness TensorSourceOp always allocates a device
  // loan and memcpy's into it (bind_tensor_output_device), so the operator
  // never observes a pageable-host Tensor. require_cuda_accessible remains
  // in compute(); restore a direct-injection check if the harness gains a
  // host-producer path.
  GTEST_SKIP() << "HSDK 5 EA: pageable-host rejection not exercisable via "
                  "device-bound Graph harness";
}

// ---- [live] sticky overrides (Graph harness deferred) --------------------

TEST(PhaseShiftBeamformerOpTest, LiveAzGridStickyOverride) {
  // [EA HSDK5 ONLY] See docs/planning/HSDK5_PORT.md §1 / §9 — live az_angles
  // is unbounded std::vector (not PublishedPayload); Params authoring covers
  // grid selection (ScanGridConfigurableFromParams).
  GTEST_SKIP() << "HSDK 5 EA: live az_angles override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(PhaseShiftBeamformerOpTest, LiveElGridStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live el_angles override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(PhaseShiftBeamformerOpTest, LiveArrayGeometryStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live array_geometry override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(PhaseShiftBeamformerOpTest, LiveArrayTaperStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live array_taper override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

}  // namespace
