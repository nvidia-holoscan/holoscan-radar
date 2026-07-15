// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// BeamscanEstimatorOp contract tests (conventional sparse-path DOA).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/beamscan_estimator_op/is_valid_mask.hpp"
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
using holoscan::radar::BeamscanEstimatorOp;
using holoscan::radar::PeakInterpolation;
using holoscan::radar::WindowSpec;
using holoscan::radar::detail::check_is_valid_mask;
using holoscan::radar::WindowType;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_bool_tensor;
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

std::vector<std::complex<double>> make_steering_snapshot(
    const ArrayGeometry& geom, const AngleConvention& conv, double az,
    double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(conv, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> snap(static_cast<size_t>(geom.num_elements()));
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

void wire_noop_output(holoscan::Fragment* app,
                      const std::shared_ptr<holoscan::Operator>& op,
                      const std::string& port) {
  const auto noop = [](const holoscan::TensorMap&) {};
  auto sink = app->make_operator<
      holoscan::test::TestHarnessSinkOp<holoscan::TensorMap>>(port + "_noop");
  sink->set_validators({noop});
  app->add_flow(op, sink, {{port, "input"}});
}

// Emits a fixed snapshot + is_valid pair each tick for graph-testing Beamscan's
// is_valid exclusion on 3-D snapshots (device masks are D2H-copied in compute).
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

struct BeamscanOutputs {
  HostTensor aoa;
  HostTensor spectrum;
};

template <typename T, typename... ExtraArgs>
BeamscanOutputs run_beamscan(std::shared_ptr<holoscan::Tensor> snapshot,
                             ExtraArgs&&... extra_args) {
  auto harness = create_operator_test<BeamscanEstimatorOp<T>>(
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
    throw std::runtime_error("BeamscanEstimatorOp test: expected one frame");
  }
  return {aoa_cap.frames.front(), spec_cap.frames.front()};
}

class IsValidMaskGraphApp : public holoscan::Application {
 public:
  std::shared_ptr<holoscan::Tensor> snapshot;
  std::shared_ptr<holoscan::Tensor> is_valid;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;
  TensorMapCapture* aoa = nullptr;

  void compose() override {
    using holoscan::Arg;
    auto relay = make_operator<SnapshotMaskRelayOp>(
        "relay", make_condition<holoscan::CountCondition>(1));
    relay->snapshot = snapshot;
    relay->is_valid = is_valid;

    auto beamscan = make_operator<BeamscanEstimatorOp<double>>(
        "beamscan", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid);

    add_flow(relay, beamscan, {{"snapshot", "snapshot"},
                               {"is_valid", "is_valid"}});
    holoscan::radar::testing::wire_capture_sink(this, beamscan, "aoa_estimates",
                                                aoa);
    wire_noop_output(this, beamscan, "angle_spectrum");
  }
};

TEST(BeamscanEstimatorOpTest, RecoversKnownDoa2D) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kSpacing = 0.5;
  constexpr double kWavelength = 1.0;
  constexpr double kTruthAz = 0.12;
  constexpr double kTruthEl = 0.05;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom =
      make_planar_ura_geometry(kNy, kNz, kSpacing, kWavelength);
  const auto snap_host =
      make_steering_snapshot(geom, conv, kTruthAz, kTruthEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.5, 0.5, 101),
      holoscan::Arg("el_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("peak_interpolation") = PeakInterpolation::kParabolic);

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
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.2, 0.2, 21),
      holoscan::Arg("el_angles") = std::vector<double>{0.0});
  EXPECT_EQ(out.spectrum.shape[0], 1);
}

TEST(BeamscanEstimatorOpTest, RejectsMultiSnapshotInput) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN * 3), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN, 3}, complex_dtype<double>());
  EXPECT_THROW(
      run_beamscan<double>(
          snap_t, holoscan::Arg("array_geometry") = geom,
          holoscan::Arg("angle_convention") = AngleConvention{},
          holoscan::Arg("az_angles") = std::vector<double>{0.0},
          holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(BeamscanEstimatorOpTest, DirectionCosineMasksInvisibleRegion) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto snap_host = make_steering_snapshot(
      geom, conv, 0.0, 0.0);  // u=0,v=0
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-1.2, 1.2, 5),
      holoscan::Arg("el_angles") = linspace(-1.2, 1.2, 5));

  const double* spec = out.spectrum.as<double>();
  EXPECT_NEAR(spec[0], 0.0, 1e-6);
}

TEST(BeamscanEstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0);
  std::vector<std::complex<float>> snap_f(snap_host.size());
  for (size_t i = 0; i < snap_host.size(); ++i) {
    snap_f[i] = {static_cast<float>(snap_host[i].real()),
                 static_cast<float>(snap_host[i].imag())};
  }
  auto snap_t = to_device_tensor(snap_f, {1, kN}, complex_dtype<float>());

  const auto out = run_beamscan<float>(
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.2, 0.2, 21),
      holoscan::Arg("el_angles") = std::vector<double>{0.0});
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.05f, 0.02f);
}

TEST(BeamscanEstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN - 1), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN - 1}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  EXPECT_THROW(
      run_beamscan<double>(
          snap_t, holoscan::Arg("array_geometry") = geom,
          holoscan::Arg("angle_convention") = conv,
          holoscan::Arg("az_angles") = std::vector<double>{0.0},
          holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(BeamscanEstimatorOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  EXPECT_THROW(
      run_beamscan<double>(
          snap_t, holoscan::Arg("array_geometry") = geom,
          holoscan::Arg("angle_convention") = conv,
          holoscan::Arg("az_angles") = std::vector<double>{},
          holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

class AzGridOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<std::vector<double>> az_overrides;
  TensorMapCapture* aoa = nullptr;
  TensorMapCapture* spectrum = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<BeamscanEstimatorOp<double>>(
        "beamscan", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention,
        Arg("az_angles") = std::vector<double>{-0.1, 0.0, 0.1},
        Arg("el_angles") = el_grid);
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "az_angles",
                                                    az_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "aoa_estimates", aoa);
    holoscan::radar::testing::wire_capture_sink(this, op, "angle_spectrum",
                                                spectrum);
  }
};

TEST(BeamscanEstimatorOpTest, LiveAzGridStickyOverride) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN}, complex_dtype<double>());

  AzGridOverrideApp app;
  app.geometry = geom;
  app.convention.preset = AngleConventionPreset::kRadarBoresight;
  app.el_grid = {0.0};
  app.snapshot_frames = {
      holoscan::radar::testing::single_map(snap_t, "snapshot"),
      holoscan::radar::testing::single_map(snap_t, "snapshot")};
  app.az_overrides = {std::vector<double>{-0.1, 0.0, 0.1},
                      linspace(-0.2, 0.2, 11)};

  TensorMapCapture aoa_cap;
  TensorMapCapture spec_cap;
  app.aoa = &aoa_cap;
  app.spectrum = &spec_cap;
  app.run();

  ASSERT_EQ(spec_cap.frames.size(), 2u);
  EXPECT_EQ(spec_cap.frames[0].shape[1], 3);
  EXPECT_EQ(spec_cap.frames[1].shape[1], 11);
}

class ElGridOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<std::vector<double>> el_overrides;
  TensorMapCapture* spectrum = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<BeamscanEstimatorOp<double>>(
        "beamscan", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_overrides.front());
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "el_angles",
                                                    el_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "angle_spectrum",
                                                spectrum);
    wire_noop_output(this, op, "aoa_estimates");
  }
};

TEST(BeamscanEstimatorOpTest, LiveElGridStickyOverride) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthEl = 0.06;

  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.1, kTruthEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()},
                                 complex_dtype<double>());

  ElGridOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.3, 0.3, 31);
  app.el_overrides = {std::vector<double>{0.0}, linspace(-0.2, 0.2, 11)};
  app.snapshot_frames = {
      holoscan::radar::testing::single_map(snap_t, "snapshot"),
      holoscan::radar::testing::single_map(snap_t, "snapshot")};

  TensorMapCapture spec_cap;
  app.spectrum = &spec_cap;
  app.run();

  ASSERT_EQ(spec_cap.frames.size(), 2u);
  EXPECT_EQ(spec_cap.frames[0].shape, (std::vector<int64_t>{1, 31, 1}));
  EXPECT_EQ(spec_cap.frames[1].shape, (std::vector<int64_t>{1, 31, 11}));
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
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("el_angles") = std::vector<double>{0.0},
      holoscan::Arg("peak_interpolation") = PeakInterpolation::kParabolic);

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
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.5, 0.5, 101),
      holoscan::Arg("el_angles") = linspace(1.2, 1.7, 61),
      holoscan::Arg("peak_interpolation") = PeakInterpolation::kParabolic);

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
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.5, 0.5, 101),
      holoscan::Arg("el_angles") = linspace(-0.3, 0.3, 61),
      holoscan::Arg("peak_interpolation") = PeakInterpolation::kParabolic);

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

  const auto args = std::make_tuple(
      holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.3, 0.3, 51),
      holoscan::Arg("el_angles") = std::vector<double>{0.0},
      holoscan::Arg("peak_interpolation") = PeakInterpolation::kParabolic);

  const auto out_f = std::apply(
      [&](auto&&... a) { return run_beamscan<float>(snap_f, a...); }, args);
  const auto out_d = std::apply(
      [&](auto&&... a) { return run_beamscan<double>(snap_d, a...); }, args);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 2e-3f);
  EXPECT_NEAR(aoa_f[1], static_cast<float>(aoa_d[1]), 2e-3f);
}

TEST(BeamscanEstimatorOpTest, PeakInterpolationNoneReturnsGridCell) {
  constexpr int64_t kN = 8;
  const double kTruthAz = 0.1;  // exactly on the linspace grid below

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.2, 0.2, 41),
      holoscan::Arg("el_angles") = std::vector<double>{0.0},
      holoscan::Arg("peak_interpolation") = PeakInterpolation::kNone);

  const double* aoa = out.aoa.as<double>();
  EXPECT_DOUBLE_EQ(aoa[0], kTruthAz);
  EXPECT_DOUBLE_EQ(aoa[1], 0.0);
}

TEST(BeamscanEstimatorOpTest, ParabolicRefinesOffGridDoa) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_none = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_para = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto args = std::make_tuple(
      holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.2, 0.2, 11),
      holoscan::Arg("el_angles") = linspace(-0.1, 0.1, 3));

  const auto out_none = std::apply(
      [&](auto&&... a) {
        return run_beamscan<double>(
            snap_none, a..., holoscan::Arg("peak_interpolation") =
                           PeakInterpolation::kNone);
      },
      args);
  const auto out_para = std::apply(
      [&](auto&&... a) {
        return run_beamscan<double>(
            snap_para, a..., holoscan::Arg("peak_interpolation") =
                           PeakInterpolation::kParabolic);
      },
      args);

  const double* none_az = out_none.aoa.as<double>();
  const double* para_az = out_para.aoa.as<double>();
  EXPECT_NEAR(none_az[0], 0.16, 1e-6);
  EXPECT_NEAR(para_az[0], kTruthAz, 0.02);
  EXPECT_GT(std::abs(para_az[0] - none_az[0]), 5e-3);
}

TEST(BeamscanEstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap_t = to_device_tensor(host, {0, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = std::vector<double>{-0.1, 0.0, 0.1},
      holoscan::Arg("el_angles") = std::vector<double>{0.0});

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
  EXPECT_THROW(
      run_beamscan<double>(
          snap_t, holoscan::Arg("array_geometry") = geom,
          holoscan::Arg("angle_convention") = conv,
          holoscan::Arg("az_angles") = std::vector<double>{0.0},
          holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(BeamscanEstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(
      geom, AngleConvention{AngleConventionPreset::kRadarBoresight}, 0.0, 0.0);
  auto snap_f = snapshot_to_device<float>(snap_host, {1, kN});
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  EXPECT_THROW(
      run_beamscan<double>(
          snap_f, holoscan::Arg("array_geometry") = geom,
          holoscan::Arg("angle_convention") = conv,
          holoscan::Arg("az_angles") = std::vector<double>{0.0},
          holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
}

TEST(BeamscanEstimatorOpTest, RejectsPageableHostSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(
      geom, AngleConvention{AngleConventionPreset::kRadarBoresight}, 0.0, 0.0);
  auto snap_cpu =
      to_pageable_host_tensor(snap_host, {1, kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  EXPECT_THROW(
      run_beamscan<double>(
          snap_cpu, holoscan::Arg("array_geometry") = geom,
          holoscan::Arg("angle_convention") = conv,
          holoscan::Arg("az_angles") = std::vector<double>{0.0},
          holoscan::Arg("el_angles") = std::vector<double>{0.0}),
      std::invalid_argument);
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
  auto snap_t =
      to_device_tensor(batch, {2, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.2, 0.2, 17),
      holoscan::Arg("el_angles") = linspace(-0.1, 0.1, 9));

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
  auto snap_t =
      to_device_tensor(batch, {2, geom.num_elements()}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("az_angles") = linspace(-0.4, 0.4, 81),
      holoscan::Arg("el_angles") = linspace(-0.2, 0.2, 41),
      holoscan::Arg("peak_interpolation") = PeakInterpolation::kParabolic);

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.01);
  EXPECT_NEAR(aoa[1], kEl0, 0.01);
  EXPECT_NEAR(aoa[2], kAz1, 0.01);
  EXPECT_NEAR(aoa[3], kEl1, 0.01);
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
    auto op = make_operator<BeamscanEstimatorOp<double>>(
        "beamscan", Arg("array_geometry") = geometry_overrides.front(),
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

TEST(BeamscanEstimatorOpTest, LiveArrayGeometryStickyOverride) {
  constexpr int64_t kN = 4;
  const auto geom_a = make_ula_geometry(kN, 0.5, 1.0);
  auto geom_b = make_ula_geometry(kN, 0.5, 1.0);
  geom_b.wavelength = 0.8;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom_a, conv, 0.05, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  ArrayGeometryOverrideApp app;
  app.convention = conv;
  app.az_grid = linspace(-0.2, 0.2, 11);
  app.el_grid = {0.0};
  app.geometry_overrides = {geom_a, geom_b};
  app.snapshot_frames = {
      holoscan::radar::testing::single_map(snap_t, "snapshot"),
      holoscan::radar::testing::single_map(snap_t, "snapshot")};

  TensorMapCapture spec_cap;
  app.spectrum = &spec_cap;
  app.run();

  ASSERT_EQ(spec_cap.frames.size(), 2u);
  EXPECT_EQ(spec_cap.frames[0].shape, spec_cap.frames[1].shape);
  EXPECT_NE(spectrum_max(spec_cap.frames[0]), spectrum_max(spec_cap.frames[1]));
}

class ArrayTaperOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<WindowSpec> taper_overrides;
  TensorMapCapture* spectrum = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    WindowSpec initial = taper_overrides.empty() ? WindowSpec{}
                                                 : taper_overrides.front();
    auto op = make_operator<BeamscanEstimatorOp<double>>(
        "beamscan", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid, Arg("array_taper") = initial);
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "array_taper",
                                                    taper_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "angle_spectrum",
                                                spectrum);
    wire_noop_output(this, op, "aoa_estimates");
  }
};

TEST(BeamscanEstimatorOpTest, LiveArrayTaperStickyOverride) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.12, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  WindowSpec hann;
  hann.type = WindowType::kHann;

  ArrayTaperOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.3, 0.3, 31);
  app.el_grid = {0.0};
  app.taper_overrides = {WindowSpec{}, hann};
  app.snapshot_frames = {
      holoscan::radar::testing::single_map(snap_t, "snapshot"),
      holoscan::radar::testing::single_map(snap_t, "snapshot")};

  TensorMapCapture spec_cap;
  app.spectrum = &spec_cap;
  app.run();

  ASSERT_EQ(spec_cap.frames.size(), 2u);
  const double rect_peak = spectrum_max(spec_cap.frames[0]);
  const double hann_peak = spectrum_max(spec_cap.frames[1]);
  EXPECT_GT(rect_peak, 0.0);
  EXPECT_GT(hann_peak, 0.0);
  // Taper changes the full spectrum shape, not only the peak (coherent-gain
  // normalization can leave the peak power unchanged).
  EXPECT_NE(spec_cap.frames[0].data, spec_cap.frames[1].data);
}

class PeakInterpOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<PeakInterpolation> peak_overrides;
  TensorMapCapture* aoa = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<BeamscanEstimatorOp<double>>(
        "beamscan", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("az_angles") = az_grid,
        Arg("el_angles") = el_grid,
        Arg("peak_interpolation") = peak_overrides.front());
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(
        this, op, "peak_interpolation", peak_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "aoa_estimates", aoa);
    wire_noop_output(this, op, "angle_spectrum");
  }
};

TEST(BeamscanEstimatorOpTest, LivePeakInterpolationStickyOverride) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.17;

  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  PeakInterpOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.2, 0.2, 11);
  // Parabolic refinement requires num_el >= 3 in the reduce kernel.
  app.el_grid = linspace(-0.1, 0.1, 3);
  app.peak_overrides = {PeakInterpolation::kNone,
                        PeakInterpolation::kParabolic};
  app.snapshot_frames = {
      holoscan::radar::testing::single_map(snap_t, "snapshot"),
      holoscan::radar::testing::single_map(snap_t, "snapshot")};

  TensorMapCapture aoa_cap;
  app.aoa = &aoa_cap;
  app.run();

  ASSERT_EQ(aoa_cap.frames.size(), 2u);
  const double* none_az = aoa_cap.frames[0].as<double>();
  const double* para_az = aoa_cap.frames[1].as<double>();
  EXPECT_NEAR(none_az[0], 0.16, 1e-6);
  EXPECT_NEAR(para_az[0], kTruthAz, 0.02);
  EXPECT_GT(std::abs(para_az[0] - none_az[0]), 5e-3);
}

TEST(BeamscanEstimatorOpTest, AcceptsValidIsValidMaskOn3DSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0);
  auto snap_t =
      to_device_tensor(snap_host, {1, kN, 1}, complex_dtype<double>());
  auto valid_t = to_device_bool_tensor({1}, {1, 1});

  IsValidMaskGraphApp app;
  app.snapshot = snap_t;
  app.is_valid = valid_t;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = linspace(-0.2, 0.2, 21);
  app.el_grid = {0.0};

  TensorMapCapture aoa_cap;
  app.aoa = &aoa_cap;
  app.run();

  ASSERT_EQ(aoa_cap.frames.size(), 1u);
  ASSERT_EQ(aoa_cap.frames[0].shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = aoa_cap.frames[0].as<double>();
  EXPECT_NEAR(aoa[0], 0.05, 0.02);
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidIsValidMaskOn3DSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0);
  auto snap_t =
      to_device_tensor(snap_host, {1, kN, 1}, complex_dtype<double>());
  auto invalid_t = to_device_bool_tensor({0}, {1, 1});

  IsValidMaskGraphApp app;
  app.snapshot = snap_t;
  app.is_valid = invalid_t;
  app.geometry = geom;
  app.convention = conv;
  app.az_grid = std::vector<double>{0.0};
  app.el_grid = std::vector<double>{0.0};

  TensorMapCapture aoa_cap;
  app.aoa = &aoa_cap;
  EXPECT_THROW(app.run(), std::invalid_argument);
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
  auto mask =
      to_pageable_host_tensor<uint8_t>({1, 1}, {2, 1}, DLDataType{kDLBool, 8, 1});
  EXPECT_THROW(check_is_valid_mask(*mask, 1, 1, "test"), std::invalid_argument);
}

}  // namespace
