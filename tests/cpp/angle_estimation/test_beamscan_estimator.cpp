// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// BeamscanEstimatorOp contract tests (HSDK 5 Graph harness): known-DOA
// recovery, scan-grid / peak-interp Params coverage, optional is_valid,
// float/double agreement, and error paths.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/is_valid_mask.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/fixture_completion.hpp"
#include "utils/test_extents.hpp"
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
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_bool_tensor;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;

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

std::vector<std::complex<double>> make_steering_snapshot(const ArrayGeometry& geom,
                                                         const AngleConvention& conv, double az,
                                                         double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(conv, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> snap(static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase =
        k * (geom.element_positions[base] * dir.x + geom.element_positions[base + 1] * dir.y +
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
std::shared_ptr<holoscan::Tensor> snapshot_to_device(const std::vector<std::complex<double>>& host,
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

BeamscanParams make_params(ArrayGeometry geom, AngleConvention conv, std::vector<double> az,
                           std::vector<double> el,
                           PeakInterpolation peak = PeakInterpolation::kNone,
                           WindowSpec taper = {}) {
  BeamscanParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.az_coordinates = std::move(az);
  params.el_coordinates = std::move(el);
  params.peak_interpolation = peak;
  params.array_taper = std::move(taper);
  return params;
}

template <typename T>
holoscan::testing::TestResult run_beamscan_result(
    std::shared_ptr<holoscan::Tensor> snapshot, BeamscanParams params,
    std::optional<holoscan::radar::MaxTensorExtents> extents = std::nullopt,
    bool capture_spectrum = true) {
  using Op = BeamscanEstimatorOp<T>;
  const auto sized = extents.value_or(holoscan::radar::testing::extents_for_scan_grid(
      params.az_coordinates.size(), params.el_coordinates.size()));
  auto fixture = holoscan::testing::create_operator_test<Op>(sized);
  fixture.set_params(std::move(params));
  fixture.push(&Op::snapshot, *snapshot);
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("aoa_estimates", 1U, kMaximumBytes);
  if (capture_spectrum) {
    fixture.template capture_snapshot<holoscan::Tensor>("angle_spectrum", 1U, kMaximumBytes);
  }
  fixture.output_device("aoa_estimates", holoscan::DeviceId{0});
  fixture.output_device("angle_spectrum", holoscan::DeviceId{0});
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  completion.wait_for_output(running, "aoa_estimates", 1U);
  if (capture_spectrum) {
    completion.wait_for_output(running, "angle_spectrum", 1U);
  }
  running.request_stop();
  auto result = running.finish();
  completion.check(result);
  return result;
}

std::vector<HostTensor> beamscan_frames(const holoscan::testing::TestResult& result,
                                        const char* port) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.template snapshots<holoscan::Tensor>(port)) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

template <typename T>
BeamscanOutputs run_beamscan(std::shared_ptr<holoscan::Tensor> snapshot, BeamscanParams params) {
  const auto result = run_beamscan_result<T>(std::move(snapshot), std::move(params));
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return {holoscan::radar::testing::to_host(
              result.template snapshot<holoscan::Tensor>("aoa_estimates")),
          holoscan::radar::testing::to_host(
              result.template snapshot<holoscan::Tensor>("angle_spectrum"))};
}

template <typename T>
holoscan::testing::TestResult run_beamscan_with_mask_result(
    std::shared_ptr<holoscan::Tensor> snapshot, std::shared_ptr<holoscan::Tensor> is_valid,
    BeamscanParams params) {
  using Op = BeamscanEstimatorOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::milliseconds{100},
                                                 holoscan::testing::input("is_valid", *is_valid),
                                                 holoscan::testing::input("snapshot", *snapshot)));
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"aoa_estimates", "angle_spectrum"};
  for (const char* output : kOutputs) {
    fixture.template capture_snapshot<holoscan::Tensor>(output, 1U, kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  fixture.input_source_device("is_valid", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  for (const char* output : kOutputs) {
    completion.wait_for_output(running, output, 1U);
  }
  running.request_stop();
  auto result = running.finish();
  completion.check(result);
  return result;
}
struct LiveBeamscanTick {
  std::shared_ptr<holoscan::Tensor> snapshot;
  std::optional<ArrayGeometry> geometry;
  std::optional<std::vector<double>> az;
  std::optional<std::vector<double>> el;
  std::optional<WindowSpec> taper;
  bool malformed_geometry{false};
  std::optional<PeakInterpolation> peak_interpolation;
};

holoscan::testing::TestResult run_live_beamscan_graph(
    std::vector<LiveBeamscanTick> ticks, BeamscanParams params,
    std::optional<holoscan::radar::MaxTensorExtents> extents = std::nullopt) {
  using Op = BeamscanEstimatorOp<double>;
  auto fixture = holoscan::testing::create_operator_test<Op>(
      extents.value_or(holoscan::radar::testing::default_test_extents()));
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto& tick = ticks[index];
    const auto snapshot = holoscan::testing::input("snapshot", *tick.snapshot);
    if (tick.geometry.has_value()) {
      auto wire = holoscan::radar::to_wire(*tick.geometry);
      if (tick.malformed_geometry) {
        wire.packed_geometry = holoscan::radar::testing::to_pageable_host_tensor<double>(
            {1.0}, {1}, holoscan::radar::real_dtype<double>());
      }
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("array_geometry", std::move(wire)), snapshot));
    } else if (tick.az.has_value()) {
      auto coordinates = holoscan::radar::testing::to_pageable_host_tensor<double>(
          *tick.az, {static_cast<int64_t>(tick.az->size())}, holoscan::radar::real_dtype<double>());
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("az_coordinates", *coordinates), snapshot));
    } else if (tick.el.has_value()) {
      auto coordinates = holoscan::radar::testing::to_pageable_host_tensor<double>(
          *tick.el, {static_cast<int64_t>(tick.el->size())}, holoscan::radar::real_dtype<double>());
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("el_coordinates", *coordinates), snapshot));
    } else if (tick.taper.has_value()) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time,
          holoscan::testing::input("array_taper", holoscan::radar::to_wire(*tick.taper)),
          snapshot));
    } else if (tick.peak_interpolation.has_value()) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("peak_interpolation", *tick.peak_interpolation),
          snapshot));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(capture_time, snapshot));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  constexpr const char* kOutputs[] = {"aoa_estimates", "angle_spectrum"};
  for (const char* output : kOutputs) {
    fixture.capture_snapshot<holoscan::Tensor>(output, ticks.size(), kMaximumBytes);
    fixture.output_device(output, holoscan::DeviceId{0});
  }
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  for (const char* output : kOutputs) {
    completion.wait_for_output(running, output, ticks.size());
  }
  running.request_stop();
  auto result = running.finish();
  completion.check(result);
  return result;
}
TEST(BeamscanEstimatorOpTest, DefaultsToRadarBoresightConvention) {
  EXPECT_EQ(BeamscanParams{}.angle_convention.preset, AngleConventionPreset::kRadarBoresight);
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoa2D) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = 0.12;
  constexpr double kTruthEl = 0.05;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, kTruthEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(-0.3, 0.3, 61),
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
  auto snap_t = to_device_tensor(snap_host, {1, kN, 1}, complex_dtype<double>());

  const auto out =
      run_beamscan<double>(snap_t, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}));
  EXPECT_EQ(out.spectrum.shape[0], 1);
}

TEST(BeamscanEstimatorOpTest, MultiLookSpectrumAveragesPerLookPower) {
  constexpr int64_t kN = 6;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto first = make_steering_snapshot(geom, AngleConvention{}, -0.12, 0.0);
  const auto second = make_steering_snapshot(geom, AngleConvention{}, 0.18, 0.0);
  const auto grid = linspace(-0.3, 0.3, 31);
  const auto first_out = run_beamscan<double>(snapshot_to_device<double>(first, {1, kN}),
                                              make_params(geom, AngleConvention{}, grid, {0.0}));
  const auto second_out = run_beamscan<double>(snapshot_to_device<double>(second, {1, kN}),
                                               make_params(geom, AngleConvention{}, grid, {0.0}));

  std::vector<std::complex<double>> looks(static_cast<size_t>(kN * 2));
  for (int64_t channel = 0; channel < kN; ++channel) {
    looks[static_cast<size_t>(channel * 2)] = first[static_cast<size_t>(channel)];
    looks[static_cast<size_t>(channel * 2 + 1)] = second[static_cast<size_t>(channel)];
  }
  const auto multi_out = run_beamscan<double>(snapshot_to_device<double>(looks, {1, kN, 2}),
                                              make_params(geom, AngleConvention{}, grid, {0.0}));
  for (int64_t i = 0; i < multi_out.spectrum.size(); ++i) {
    const double expected =
        0.5 * (first_out.spectrum.as<double>()[i] + second_out.spectrum.as<double>()[i]);
    EXPECT_NEAR(multi_out.spectrum.as<double>()[i], expected, 1e-10);
  }
}

TEST(BeamscanEstimatorOpTest, MultiLookValidityMaskExcludesInvalidLook) {
  constexpr int64_t kN = 6;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto first = make_steering_snapshot(geom, AngleConvention{}, -0.12, 0.0);
  const auto second = make_steering_snapshot(geom, AngleConvention{}, 0.18, 0.0);
  std::vector<std::complex<double>> looks(static_cast<size_t>(kN * 2));
  for (int64_t channel = 0; channel < kN; ++channel) {
    looks[static_cast<size_t>(channel * 2)] = first[static_cast<size_t>(channel)];
    looks[static_cast<size_t>(channel * 2 + 1)] = second[static_cast<size_t>(channel)];
  }
  const auto grid = linspace(-0.3, 0.3, 31);
  const auto single = run_beamscan<double>(snapshot_to_device<double>(first, {1, kN}),
                                           make_params(geom, AngleConvention{}, grid, {0.0}));
  const auto masked = run_beamscan_with_mask_result<double>(
      snapshot_to_device<double>(looks, {1, kN, 2}), to_device_bool_tensor({1, 0}, {1, 2}),
      make_params(geom, AngleConvention{}, grid, {0.0}));
  ASSERT_TRUE(masked.ok()) << masked.describe_failure();
  ASSERT_EQ(beamscan_frames(masked, "angle_spectrum").size(), 1U);
  for (int64_t i = 0; i < beamscan_frames(masked, "angle_spectrum")[0].size(); ++i) {
    EXPECT_NEAR(beamscan_frames(masked, "angle_spectrum")[0].as<double>()[i],
                single.spectrum.as<double>()[i], 1e-10);
  }
}

TEST(BeamscanEstimatorOpTest, DirectionCosineMasksInvisibleRegion) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.0, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-1.0, 1.0, 5), linspace(-1.0, 1.0, 5)));

  const double* spec = out.spectrum.as<double>();
  EXPECT_NEAR(spec[0], 0.0, 1e-6);
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidScanCoordinates) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  AngleConvention direction_cosine;
  direction_cosine.preset = AngleConventionPreset::kDirectionCosine;
  auto out_of_range =
      run_beamscan_result<double>(to_device_tensor(host, {1, kN}, complex_dtype<double>()),
                                  make_params(geom, direction_cosine, {0.0}, {-1.01}));
  EXPECT_FALSE(out_of_range.diagnostics().empty());

  auto non_finite = run_beamscan_result<double>(
      to_device_tensor(host, {1, kN}, complex_dtype<double>()),
      make_params(geom, AngleConvention{}, {std::numeric_limits<double>::quiet_NaN()}, {0.0}));
  EXPECT_FALSE(non_finite.diagnostics().empty());
}

TEST(BeamscanEstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0);
  auto snap_t = snapshot_to_device<float>(snap_host, {1, kN});

  const auto out =
      run_beamscan<float>(snap_t, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}));
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
  const auto result = run_beamscan_result<double>(snap_t, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamscan_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(BeamscanEstimatorOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto result = run_beamscan_result<double>(snap_t, make_params(geom, conv, {}, {0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamscan_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(BeamscanEstimatorOpTest, PeakInterpolationConfigurableFromParams) {
  // Exercise the authoring-time Params value for peak_interpolation.
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_none = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_para = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out_none = run_beamscan<double>(
      snap_none, make_params(geom, conv, linspace(-0.2, 0.2, 11), linspace(-0.1, 0.1, 3),
                             PeakInterpolation::kNone));
  const auto out_para = run_beamscan<double>(
      snap_para, make_params(geom, conv, linspace(-0.2, 0.2, 11), linspace(-0.1, 0.1, 3),
                             PeakInterpolation::kParabolic));

  const double* none_az = out_none.aoa.as<double>();
  const double* para_az = out_para.aoa.as<double>();
  EXPECT_NEAR(none_az[0], 0.16, 1e-6);
  EXPECT_NEAR(para_az[0], kTruthAz, 0.02);
  EXPECT_GT(std::abs(para_az[0] - none_az[0]), 5e-3);
}

// Truth sits off-grid, so the interpolation mode is readable in the reported
// angle: kNone can only name the peak grid cell (0.16) while kParabolic
// refines toward the truth (0.17).
LiveBeamscanTick make_interp_tick(std::shared_ptr<holoscan::Tensor> snapshot,
                                  std::optional<PeakInterpolation> interpolation) {
  LiveBeamscanTick tick{std::move(snapshot)};
  tick.peak_interpolation = interpolation;
  return tick;
}

TEST(BeamscanEstimatorOpTest, LivePeakInterpolationAppliesForOneTick) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot =
      snapshot_to_device<double>(make_steering_snapshot(geom, conv, kTruthAz, 0.0), {1, kN});
  auto params = make_params(geom, conv, linspace(-0.2, 0.2, 11), linspace(-0.1, 0.1, 3),
                            PeakInterpolation::kNone);

  const auto out =
      run_live_beamscan_graph({make_interp_tick(snapshot, std::nullopt),
                               make_interp_tick(snapshot, PeakInterpolation::kParabolic),
                               make_interp_tick(snapshot, std::nullopt)},
                              std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamscan_frames(out, "aoa_estimates").size(), 3U);
  const double params_az = beamscan_frames(out, "aoa_estimates")[0].as<double>()[0];
  const double override_az = beamscan_frames(out, "aoa_estimates")[1].as<double>()[0];
  const double reverted_az = beamscan_frames(out, "aoa_estimates")[2].as<double>()[0];
  EXPECT_NEAR(params_az, 0.16, 1e-6);
  EXPECT_NEAR(override_az, kTruthAz, 0.02);
  EXPECT_GT(std::abs(override_az - params_az), 5e-3);
  // The third tick publishes nothing, so it is back on Params `none`.
  EXPECT_EQ(reverted_az, params_az);
}

TEST(BeamscanEstimatorOpTest, LiveAzCoordinatesAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0), {1, kN});
  auto params = make_params(geom, AngleConvention{}, {-0.1, 0.0, 0.1}, {0.0});
  const auto out = run_live_beamscan_graph(
      {{snapshot}, {snapshot, std::nullopt, linspace(-0.2, 0.2, 5)}, {snapshot}},
      std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamscan_frames(out, "angle_spectrum").size(), 3U);
  EXPECT_EQ(beamscan_frames(out, "angle_spectrum")[0].shape[1], 3);
  EXPECT_EQ(beamscan_frames(out, "angle_spectrum")[1].shape[1], 5);
  EXPECT_EQ(beamscan_frames(out, "angle_spectrum")[2].shape[1], 3);
}

TEST(BeamscanEstimatorOpTest, LiveElCoordinatesAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0), {1, kN});
  auto params = make_params(geom, AngleConvention{}, {0.0}, {0.0});
  const auto out = run_live_beamscan_graph(
      {{snapshot},
       {snapshot, std::nullopt, std::nullopt, std::vector<double>{-0.1, 0.0, 0.1}},
       {snapshot}},
      std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamscan_frames(out, "angle_spectrum").size(), 3U);
  EXPECT_EQ(beamscan_frames(out, "angle_spectrum")[0].shape[2], 1);
  EXPECT_EQ(beamscan_frames(out, "angle_spectrum")[1].shape[2], 3);
  EXPECT_EQ(beamscan_frames(out, "angle_spectrum")[2].shape[2], 1);
}

TEST(BeamscanEstimatorOpTest, LiveArrayGeometryAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto changed = make_ula_geometry(kN, 0.25, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.2, 0.0), {1, kN});
  auto params = make_params(geom, AngleConvention{}, linspace(-0.4, 0.4, 9), {0.0});
  const auto out =
      run_live_beamscan_graph({{snapshot}, {snapshot, changed}, {snapshot}}, std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamscan_frames(out, "angle_spectrum").size(), 3U);
  EXPECT_NE(beamscan_frames(out, "angle_spectrum")[0].data,
            beamscan_frames(out, "angle_spectrum")[1].data);
  EXPECT_EQ(beamscan_frames(out, "angle_spectrum")[0].data,
            beamscan_frames(out, "angle_spectrum")[2].data);
}

TEST(BeamscanEstimatorOpTest, LiveArrayTaperAppliesForOneTick) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.15, 0.0), {1, kN});
  WindowSpec hann;
  hann.type = WindowType::kHann;
  auto params = make_params(geom, AngleConvention{}, linspace(-0.4, 0.4, 17), {0.0});
  const auto out = run_live_beamscan_graph(
      {{snapshot}, {snapshot, std::nullopt, std::nullopt, std::nullopt, hann}, {snapshot}},
      std::move(params));
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(beamscan_frames(out, "angle_spectrum").size(), 3U);
  EXPECT_NE(beamscan_frames(out, "angle_spectrum")[0].data,
            beamscan_frames(out, "angle_spectrum")[1].data);
  EXPECT_EQ(beamscan_frames(out, "angle_spectrum")[0].data,
            beamscan_frames(out, "angle_spectrum")[2].data);
}

TEST(BeamscanEstimatorOpTest, LiveOverridesRejectMalformedAndBounds) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0), {1, kN});
  auto params = make_params(geom, AngleConvention{}, {0.0}, {0.0});
  const auto malformed = run_live_beamscan_graph(
      {{snapshot, geom, std::nullopt, std::nullopt, std::nullopt, true}}, params);
  ASSERT_FALSE(malformed.ok());
  EXPECT_TRUE(beamscan_frames(malformed, "aoa_estimates").empty());
  EXPECT_FALSE(malformed.diagnostics().empty());

  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_el = 2;
  const auto bounded = run_live_beamscan_graph(
      {{snapshot, std::nullopt, std::nullopt, std::vector<double>{-0.1, 0.0, 0.1}}},
      std::move(params), extents);
  ASSERT_FALSE(bounded.ok());
  EXPECT_TRUE(beamscan_frames(bounded, "aoa_estimates").empty());
  EXPECT_FALSE(bounded.diagnostics().empty());
}

TEST(BeamscanEstimatorOpTest, ArrayTaperConfigurableFromParams) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.12, 0.0);
  auto snap_rect = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_hann = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  WindowSpec hann;
  hann.type = WindowType::kHann;

  const auto out_rect =
      run_beamscan<double>(snap_rect, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}));
  const auto out_hann = run_beamscan<double>(
      snap_hann,
      make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, PeakInterpolation::kNone, hann));

  EXPECT_GT(spectrum_max(out_rect.spectrum), 0.0);
  EXPECT_GT(spectrum_max(out_hann.spectrum), 0.0);
  EXPECT_NE(out_rect.spectrum.data, out_hann.spectrum.data);
}

TEST(BeamscanEstimatorOpTest, UraArrayTaperLowersBothAxisSidelobes) {
  constexpr int64_t kNy = 8;
  constexpr int64_t kNz = 8;
  constexpr int64_t kGrid = 201;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snapshot = make_steering_snapshot(geom, conv, 0.0, 0.0);
  const auto angles = linspace(-1.0, 1.0, kGrid);

  const auto sidelobe_ratio = [&](const WindowSpec& taper, bool scan_azimuth) {
    auto snapshot_t = to_device_tensor(snapshot, {1, kNy * kNz}, complex_dtype<double>());
    const std::vector<double> az = scan_azimuth ? angles : std::vector<double>{0.0};
    const std::vector<double> el = scan_azimuth ? std::vector<double>{0.0} : angles;
    const auto output = run_beamscan<double>(
        snapshot_t, make_params(geom, conv, az, el, PeakInterpolation::kNone, taper));
    const double* spectrum = output.spectrum.as<double>();
    double peak = 0.0;
    double sidelobe = 0.0;
    for (int64_t i = 0; i < kGrid; ++i) {
      const double power = spectrum[static_cast<size_t>(i)];
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

TEST(BeamscanEstimatorOpTest, RecoversKnownDoaAzOnly1DScan) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.083;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_none = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_parabolic = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out_none = run_beamscan<double>(
      snap_none, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, PeakInterpolation::kNone));
  const auto out_parabolic = run_beamscan<double>(
      snap_parabolic,
      make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, PeakInterpolation::kParabolic));

  ASSERT_EQ(out_parabolic.spectrum.shape, (std::vector<int64_t>{1, 61, 1}));
  const double none_az = out_none.aoa.as<double>()[0];
  const double parabolic_az = out_parabolic.aoa.as<double>()[0];
  EXPECT_NEAR(none_az, 0.08, 1e-6);
  EXPECT_NEAR(parabolic_az, kTruthAz, 0.003);
  EXPECT_LT(std::abs(parabolic_az - kTruthAz), std::abs(none_az - kTruthAz));
  EXPECT_NEAR(out_parabolic.aoa.as<double>()[1], 0.0, 1e-6);
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
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, kPolarEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(1.2, 1.7, 61),
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
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, kTruthEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(-0.3, 0.3, 61),
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

  BeamscanParams params =
      make_params(geom, conv, linspace(-0.3, 0.3, 51), {0.0}, PeakInterpolation::kParabolic);

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
      snap_t, make_params(geom, conv, linspace(-0.2, 0.2, 41), {0.0}, PeakInterpolation::kNone));

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

  const auto out = run_beamscan<double>(snap_t, make_params(geom, conv, {-0.1, 0.0, 0.1}, {0.0}));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{0, 3, 1}));
}

TEST(BeamscanEstimatorOpTest, OptionalSpectrumMayBeLeftUnwired) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0), {1, kN});
  auto params = make_params(geom, AngleConvention{}, {-0.1, 0.0, 0.1}, {0.0});
  const auto extents = holoscan::radar::testing::extents_for_scan_grid(3, 1);
  const auto result = run_beamscan_result<double>(std::move(snapshot), std::move(params), extents,
                                                  /*capture_spectrum=*/false);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(beamscan_frames(result, "aoa_estimates").size(), 1U);
  EXPECT_EQ(beamscan_frames(result, "aoa_estimates")[0].shape, (std::vector<int64_t>{1, 2}));
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidSnapshotRank) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto result = run_beamscan_result<double>(snap_t, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamscan_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(BeamscanEstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(
      geom, AngleConvention{AngleConventionPreset::kRadarBoresight}, 0.0, 0.0);
  auto snap_f = snapshot_to_device<float>(snap_host, {1, kN});
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto result = run_beamscan_result<double>(snap_f, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(beamscan_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(BeamscanEstimatorOpTest, RejectsItemsBeyondCapturedExtent) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(2 * kN), {1.0, 0.0});
  auto snapshot = to_device_tensor(host, {2, kN}, complex_dtype<double>());
  auto extents = holoscan::radar::testing::extents_for_scan_grid(1, 1);
  extents.max_items = 1;
  const auto result = run_beamscan_result<double>(
      std::move(snapshot), make_params(geom, AngleConvention{}, {0.0}, {0.0}), extents);
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(beamscan_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
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
      snap_t, make_params(geom, conv, linspace(-0.2, 0.2, 17), linspace(-0.1, 0.1, 9)));

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
  auto snap_t = to_device_tensor(batch, {2, geom.num_elements()}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-0.4, 0.4, 81), linspace(-0.2, 0.2, 41),
                          PeakInterpolation::kParabolic));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.01);
  EXPECT_NEAR(aoa[1], kEl0, 0.01);
  EXPECT_NEAR(aoa[2], kAz1, 0.01);
  EXPECT_NEAR(aoa[3], kEl1, 0.01);
}

TEST(BeamscanEstimatorOpTest, AcceptsValidIsValidMaskOn3DSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0), {1, kN, 1});
  auto mask = to_device_bool_tensor({0}, {1, 1});
  const auto result = run_beamscan_with_mask_result<double>(
      std::move(snapshot), std::move(mask),
      make_params(geom, AngleConvention{}, {-0.1, 0.0, 0.1}, {0.0}));
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(beamscan_frames(result, "aoa_estimates").size(), 1U);
  ASSERT_EQ(beamscan_frames(result, "angle_spectrum").size(), 1U);
  EXPECT_TRUE(std::isnan(beamscan_frames(result, "aoa_estimates")[0].as<double>()[0]));
  EXPECT_TRUE(std::isnan(beamscan_frames(result, "aoa_estimates")[0].as<double>()[1]));
  for (int64_t i = 0; i < beamscan_frames(result, "angle_spectrum")[0].size(); ++i) {
    EXPECT_EQ(beamscan_frames(result, "angle_spectrum")[0].as<double>()[i], 0.0);
  }
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidIsValidMaskOn3DSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0), {1, kN, 1});
  auto mask = to_device_bool_tensor({1, 1}, {1, 2});
  const auto result = run_beamscan_with_mask_result<double>(
      std::move(snapshot), std::move(mask), make_params(geom, AngleConvention{}, {0.0}, {0.0}));
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(beamscan_frames(result, "aoa_estimates").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskAcceptsTrueBoolDevice) {
  auto mask = to_device_bool_tensor({1}, {1, 1});
  EXPECT_NO_THROW(check_is_valid_mask(*mask, 1, 1, "test"));
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskAcceptsFalseBoolDevice) {
  // Zero slots are legal (unused snapshot); the estimator blanks those items.
  auto mask = to_device_bool_tensor({0}, {1, 1});
  EXPECT_NO_THROW(check_is_valid_mask(*mask, 1, 1, "test"));
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskRejectsPageableHost) {
  auto mask = to_pageable_host_tensor<uint8_t>({1}, {1, 1}, DLDataType{kDLBool, 8, 1});
  EXPECT_THROW(check_is_valid_mask(*mask, 1, 1, "test"), std::invalid_argument);
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskRejectsShapeMismatch) {
  auto mask = to_pageable_host_tensor<uint8_t>({1, 1}, {2, 1}, DLDataType{kDLBool, 8, 1});
  EXPECT_THROW(check_is_valid_mask(*mask, 1, 1, "test"), std::invalid_argument);
}

}  // namespace
