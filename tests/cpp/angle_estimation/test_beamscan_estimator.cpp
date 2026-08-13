// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// BeamscanEstimatorOp contract tests (HSDK 5 Graph harness): known-DOA
// recovery, scan-grid / peak-interp Params coverage, optional is_valid,
// float/double agreement, and error paths.

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

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/beamscan_estimator_op/is_valid_mask.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
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
using holoscan::radar::PeakInterpolation;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::complex_dtype;
using holoscan::radar::detail::check_is_valid_mask;
using holoscan::radar::kPi;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;
using holoscan::radar::testing::to_device_bool_tensor;
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

std::vector<std::complex<double>> make_steering_snapshot(
    const ArrayGeometry& geom, const AngleConvention& conv, double az,
    double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(conv, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> snap(
      static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase = k * (geom.element_positions[base] * dir.x +
                              geom.element_positions[base + 1] * dir.y +
                              geom.element_positions[base + 2] * dir.z);
    snap[static_cast<size_t>(m)] = {std::cos(phase), std::sin(phase)};
  }
  return snap;
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
std::shared_ptr<holoscan::Tensor> snapshot_to_device(
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

struct BeamscanOutputs {
  HostTensor aoa;
  HostTensor spectrum;
};

template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_beamscan(holoscan::Graph& graph, const Source& source, OpRef& op,
                   const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->snapshot, q);
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->angle_spectrum, sink1->in, q);
}

BeamscanParams make_params(ArrayGeometry geom, AngleConvention conv,
                           std::vector<double> az, std::vector<double> el,
                           PeakInterpolation peak = PeakInterpolation::kNone,
                           WindowSpec taper = {}) {
  BeamscanParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.az_angles = std::move(az);
  params.el_angles = std::move(el);
  params.peak_interpolation = peak;
  params.array_taper = std::move(taper);
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
BeamscanOutputs run_beamscan(std::shared_ptr<holoscan::Tensor> snapshot,
                             BeamscanParams params) {
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
        "BeamscanEstimatorOp test: expected one output frame pair, got " +
        std::to_string(result.frames0.size()) + "/" +
        std::to_string(result.frames1.size()));
  }
  return {std::move(result.frames0.front()),
          std::move(result.frames1.front())};
}

template <typename T>
DualIOGraphRunResult run_beamscan_with_mask_result(
    std::shared_ptr<holoscan::Tensor> snapshot,
    std::shared_ptr<holoscan::Tensor> is_valid, BeamscanParams params) {
  return run_dual_io_operator_graph_result<BeamscanEstimatorOp<T>>(
      {std::move(snapshot)}, {std::move(is_valid)}, std::move(params),
      "aoa_estimates", "angle_spectrum",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_beamscan_with_is_valid(graph, source, op, sink0, sink1);
      });
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoa2D) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = 0.12;
  constexpr double kTruthEl = 0.05;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap_host =
      make_steering_snapshot(geom, conv, kTruthAz, kTruthEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(-0.3, 0.3, 61),
                  PeakInterpolation::kParabolic));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.01);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.01);
}

TEST(BeamscanEstimatorOpTest, Accepts3DSnapshotWithOneLook) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.0, 0.0);
  auto snap_t =
      to_device_tensor(snap_host, {1, kN, 1}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}));
  EXPECT_EQ(out.spectrum.shape[0], 1);
}

TEST(BeamscanEstimatorOpTest, RejectsMultiSnapshotInput) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN * 3),
                                         {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN, 3}, complex_dtype<double>());
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_t, make_params(geom, AngleConvention{}, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, DirectionCosineMasksInvisibleRegion) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.0, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-1.2, 1.2, 5),
                          linspace(-1.2, 1.2, 5)));

  const double* spec = out.spectrum.as<double>();
  EXPECT_NEAR(spec[0], 0.0, 1e-6);
}

TEST(BeamscanEstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0);
  auto snap_t = snapshot_to_device<float>(snap_host, {1, kN});

  const auto out = run_beamscan<float>(
      snap_t, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}));
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.05f, 0.02f);
}

TEST(BeamscanEstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN - 1),
                                         {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN - 1}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_t, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_t, make_params(geom, conv, {}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, PeakInterpolationConfigurableFromParams) {
  // Authoring Params must select peak_interpolation (live Graph override
  // deferred — HSDK5_PORT §1).
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_none =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_para =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out_none = run_beamscan<double>(
      snap_none,
      make_params(geom, conv, linspace(-0.2, 0.2, 11), linspace(-0.1, 0.1, 3),
                  PeakInterpolation::kNone));
  const auto out_para = run_beamscan<double>(
      snap_para,
      make_params(geom, conv, linspace(-0.2, 0.2, 11), linspace(-0.1, 0.1, 3),
                  PeakInterpolation::kParabolic));

  const double* none_az = out_none.aoa.as<double>();
  const double* para_az = out_para.aoa.as<double>();
  EXPECT_NEAR(none_az[0], 0.16, 1e-6);
  EXPECT_NEAR(para_az[0], kTruthAz, 0.02);
  EXPECT_GT(std::abs(para_az[0] - none_az[0]), 5e-3);
}

TEST(BeamscanEstimatorOpTest, LivePeakInterpolationStickyOverride) {
  // [EA HSDK5 ONLY] See docs/planning/HSDK5_PORT.md §1 — Graph mid-stream
  // sticky override harness fails on EA device/partition binding.
  // Params authoring covers peak_interpolation; restore once §1 is closed.
  GTEST_SKIP() << "HSDK 5 EA: live peak_interpolation override harness "
                  "deferred (HSDK5_PORT.md §1)";
}

TEST(BeamscanEstimatorOpTest, LiveAzGridStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live az_angles override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(BeamscanEstimatorOpTest, LiveElGridStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live el_angles override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(BeamscanEstimatorOpTest, LiveArrayGeometryStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live array_geometry override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(BeamscanEstimatorOpTest, LiveArrayTaperStickyOverride) {
  GTEST_SKIP() << "HSDK 5 EA: live array_taper override deferred "
                  "(HSDK5_PORT.md §1 / §9)";
}

TEST(BeamscanEstimatorOpTest, ArrayTaperConfigurableFromParams) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.12, 0.0);
  auto snap_rect =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_hann =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  WindowSpec hann;
  hann.type = WindowType::kHann;

  const auto out_rect = run_beamscan<double>(
      snap_rect,
      make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}));
  const auto out_hann = run_beamscan<double>(
      snap_hann,
      make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0},
                  PeakInterpolation::kNone, hann));

  EXPECT_GT(spectrum_max(out_rect.spectrum), 0.0);
  EXPECT_GT(spectrum_max(out_hann.spectrum), 0.0);
  EXPECT_NE(out_rect.spectrum.data, out_hann.spectrum.data);
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoaAzOnly1DScan) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0},
                  PeakInterpolation::kParabolic));

  ASSERT_EQ(out.spectrum.shape, (std::vector<int64_t>{1, 61, 1}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.01);
  EXPECT_NEAR(aoa[1], 0.0, 1e-6);
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoaPhysicsSpherical) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = 0.1;
  constexpr double kHorizonEl = 0.04;
  constexpr double kPolarEl = kPi / 2.0 - kHorizonEl;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kPhysicsSpherical;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap_host =
      make_steering_snapshot(geom, conv, kTruthAz, kPolarEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(1.2, 1.7, 61),
                  PeakInterpolation::kParabolic));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.01);
  EXPECT_NEAR(aoa[1], kPolarEl, 0.02);
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoaAerospaceNed) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = -0.1;
  constexpr double kTruthEl = -0.03;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kAerospaceNed;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap_host =
      make_steering_snapshot(geom, conv, kTruthAz, kTruthEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(-0.3, 0.3, 61),
                  PeakInterpolation::kParabolic));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.01);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.01);
}

TEST(BeamscanEstimatorOpTest, FloatDoubleAgreeOnDoa) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.09;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_f = snapshot_to_device<float>(snap_host, {1, kN});
  auto snap_d = snapshot_to_device<double>(snap_host, {1, kN});

  BeamscanParams params = make_params(
      geom, conv, linspace(-0.3, 0.3, 51), {0.0},
      PeakInterpolation::kParabolic);

  const auto out_f = run_beamscan<float>(snap_f, params);
  const auto out_d = run_beamscan<double>(snap_d, params);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 2e-3f);
  EXPECT_NEAR(aoa_f[1], static_cast<float>(aoa_d[1]), 2e-3f);
}

TEST(BeamscanEstimatorOpTest, PeakInterpolationNoneReturnsGridCell) {
  constexpr int64_t kN = 8;
  const double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.2, 0.2, 41), {0.0},
                  PeakInterpolation::kNone));

  const double* aoa = out.aoa.as<double>();
  EXPECT_DOUBLE_EQ(aoa[0], kTruthAz);
  EXPECT_DOUBLE_EQ(aoa[1], 0.0);
}

TEST(BeamscanEstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap_t = to_device_tensor(host, {0, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, {-0.1, 0.0, 0.1}, {0.0}));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{0, 3, 1}));
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidSnapshotRank) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_t, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(
      geom, AngleConvention{AngleConventionPreset::kRadarBoresight}, 0.0, 0.0);
  auto snap_f = snapshot_to_device<float>(snap_host, {1, kN});
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_f, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, RejectsPageableHostSnapshot) {
  // Graph harness copies FrameBank tensors into a device Output loan before
  // the operator receives them, so BeamscanEstimatorOp never observes a
  // pageable-host Tensor. require_cuda_accessible remains in compute().
  GTEST_SKIP() << "HSDK 5 EA: pageable-host rejection not exercisable via "
                  "device Graph harness (same as PhaseShiftBeamformer)";
}

TEST(BeamscanEstimatorOpTest, SpectrumShapeMatchesScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), snap_host.begin(), snap_host.end());
  batch.insert(batch.end(), snap_host.begin(), snap_host.end());
  auto snap_t = to_device_tensor(batch, {2, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.2, 0.2, 17), linspace(-0.1, 0.1, 9)));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{2, 17, 9}));
  EXPECT_GT(spectrum_max(out.spectrum), 0.0);
}

TEST(BeamscanEstimatorOpTest, MultipleItemsRecoverIndependentDoa) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kAz0 = 0.11;
  constexpr double kEl0 = 0.04;
  constexpr double kAz1 = -0.07;
  constexpr double kEl1 = -0.02;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap0 = make_steering_snapshot(geom, conv, kAz0, kEl0);
  const auto snap1 = make_steering_snapshot(geom, conv, kAz1, kEl1);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), snap0.begin(), snap0.end());
  batch.insert(batch.end(), snap1.begin(), snap1.end());
  auto snap_t = to_device_tensor(batch, {2, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.4, 0.4, 81), linspace(-0.2, 0.2, 41),
                  PeakInterpolation::kParabolic));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.01);
  EXPECT_NEAR(aoa[1], kEl0, 0.01);
  EXPECT_NEAR(aoa[2], kAz1, 0.01);
  EXPECT_NEAR(aoa[3], kEl1, 0.01);
}

TEST(BeamscanEstimatorOpTest, AcceptsValidIsValidMaskOn3DSnapshot) {
  // [EA HSDK5 ONLY] Wiring optional is_valid beside snapshot yields
  // RUNTIME_OPERATOR_REPLAY_MISMATCH when the port is omitted from contract
  // (required for unwired graphs — TEMPORAL_INPUT_UNCONNECTED otherwise).
  // Mask validation remains covered by CheckIsValidMask* unit tests.
  GTEST_SKIP() << "HSDK 5 EA: optional is_valid Graph wiring deferred "
                  "(HSDK5_PORT.md §1 / OPEN_ITEMS.md §3)";
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidIsValidMaskOn3DSnapshot) {
  GTEST_SKIP() << "HSDK 5 EA: optional is_valid Graph wiring deferred "
                  "(HSDK5_PORT.md §1 / OPEN_ITEMS.md §3)";
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskAcceptsTrueBoolHost) {
  auto mask =
      to_pageable_host_tensor<uint8_t>({1}, {1, 1}, DLDataType{kDLBool, 8, 1});
  EXPECT_NO_THROW(check_is_valid_mask(*mask, 1, 1, "test"));
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskRejectsFalseBoolHost) {
  auto mask =
      to_pageable_host_tensor<uint8_t>({0}, {1, 1}, DLDataType{kDLBool, 8, 1});
  EXPECT_THROW(check_is_valid_mask(*mask, 1, 1, "test"), std::invalid_argument);
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskRejectsFalseBoolDevice) {
  auto mask = to_device_bool_tensor({0}, {1, 1});
  EXPECT_THROW(check_is_valid_mask(*mask, 1, 1, "test"), std::invalid_argument);
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskRejectsShapeMismatch) {
  auto mask = to_pageable_host_tensor<uint8_t>(
      {1, 1}, {2, 1}, DLDataType{kDLBool, 8, 1});
  EXPECT_THROW(check_is_valid_mask(*mask, 1, 1, "test"), std::invalid_argument);
}

}  // namespace
