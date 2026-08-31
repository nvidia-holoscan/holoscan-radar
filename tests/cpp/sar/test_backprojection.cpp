// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for BackprojectionOp (mode-1 slice plus signed-weight
// accumulation for mode 2, plus geometry_change = scroll for
// mode 4). Inputs are constructed range profiles
// (a single-bin impulse at the predicted delay), not phase history,
// so a compression bug cannot hide a formation bug. Ground truth is
// radar_common/sim/sar_phase_history_sim.hpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sim/sar_phase_history_sim.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::BackprojectionBackend;
using holoscan::radar::BackprojectionOp;
using holoscan::radar::CoordinateFrame;
using holoscan::radar::CoordinateFrameSpec;
using holoscan::radar::CoordinateFrameType;
using holoscan::radar::FormationControl;
using holoscan::radar::GeometryChange;
using holoscan::radar::RangeInterpolation;
using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::kPi;
using holoscan::radar::kSpeedOfLight;
using holoscan::radar::sim::sar_differential_range;
using holoscan::radar::sim::sar_matched_phase;
using holoscan::radar::sim::sar_predicted_peak;
using holoscan::radar::testing::complex_cube_to_device;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_uint8_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;

constexpr int64_t kBins = 64;
constexpr int64_t kPulses = 4;
constexpr double kSpacing = 0.25;
constexpr double kFreq = 9.6e9;

double centred_origin() {
  return -std::floor(static_cast<double>(kBins) / 2.0) * kSpacing;
}

CoordinateFrame scene_frame() {
  CoordinateFrameSpec spec;
  spec.frame = CoordinateFrameType::kSceneLocal;
  return CoordinateFrame(spec);
}

SampleGeometry grid_geometry() {
  SampleGeometrySpec spec;
  spec.origin = {-2.0, -2.0, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {1.0, 1.0};
  spec.counts = {5, 5};
  return SampleGeometry(spec);
}

holoscan::TensorMap control_map(FormationControl value) {
  return single_map(
      to_device_uint8_tensor({static_cast<uint8_t>(value)}, {1}),
      "control");
}

holoscan::TensorMap weight_map(const std::vector<double>& values) {
  const int64_t n = static_cast<int64_t>(values.size());
  return single_map(real_cube_to_device<float>(values, {n}),
                    "weight");
}

template <typename T>
holoscan::TensorMap profiles_impulse(
    double delta_r, double origin, std::complex<double> amplitude,
    int64_t pulses = kPulses,
    const std::vector<double>& scales = {}) {
  if (!scales.empty() &&
      static_cast<int64_t>(scales.size()) != pulses) {
    throw std::invalid_argument(
        "profiles_impulse: scales length must match pulses");
  }
  const double bin_f = (delta_r - origin) / kSpacing;
  const int64_t bin = static_cast<int64_t>(std::llround(bin_f));
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(pulses * kBins), {0.0, 0.0});
  if (bin >= 0 && bin < kBins) {
    for (int64_t p = 0; p < pulses; ++p) {
      const double scale =
          scales.empty() ? 1.0 : scales[static_cast<size_t>(p)];
      cube[static_cast<size_t>(p * kBins + bin)] =
          amplitude * scale;
    }
  }
  return single_map(
      complex_cube_to_device<T>(cube, {pulses, kBins}),
      "range_profiles");
}

template <typename T>
holoscan::TensorMap xyz_map(const std::array<double, 3>& point,
                            const char* key, int64_t pulses = kPulses) {
  std::vector<double> host(static_cast<size_t>(pulses * 3));
  for (int64_t p = 0; p < pulses; ++p) {
    host[static_cast<size_t>(p * 3 + 0)] = point[0];
    host[static_cast<size_t>(p * 3 + 1)] = point[1];
    host[static_cast<size_t>(p * 3 + 2)] = point[2];
  }
  return single_map(real_cube_to_device<T>(host, {pulses, 3}), key);
}

struct ImageCapture {
  HostTensor image;
};

template <typename T>
class BackprojectionApp : public holoscan::Application {
 public:
  BackprojectionApp(BackprojectionBackend backend, FormationControl control,
                    ImageCapture* capture, double origin, double frequency,
                    const std::array<double, 3>& target,
                    int64_t pulses = kPulses,
                    std::vector<double> scales = {})
      : backend_(backend),
        control_(control),
        capture_(capture),
        origin_(origin),
        frequency_(frequency),
        target_(target),
        pulses_(pulses),
        scales_(std::move(scales)) {}

  void compose() override {
    const auto geometry = grid_geometry();
    auto op = make_operator<BackprojectionOp<T>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", geometry),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", origin_),
        holoscan::Arg("reference_frequency", frequency_),
        holoscan::Arg("backend", backend_));

    const std::array<double, 3> apc{0.0, 0.0, 100.0};
    const std::array<double, 3> srp{0.0, 0.0, 0.0};
    const double dR = sar_differential_range(apc, srp, target_);
    const auto phase = sar_matched_phase(frequency_, dR);
    // Stored history is the conjugate of the matched term.
    const std::complex<double> amp{phase.real(), -phase.imag()};

    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles",
        {profiles_impulse<T>(dR, origin_, amp, pulses_,
                             scales_)});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions", {xyz_map<T>(apc, "positions", pulses_)});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<T>(srp, "reference_position", pulses_)});
    holoscan::radar::testing::wire_leading_override(
        this, op, "control",
        std::vector<holoscan::TensorMap>{control_map(control_)});
    holoscan::radar::testing::wire_capture_sink(
        this, op, "image", &sink_);
  }

  void take() {
    if (sink_.frames.size() != 1u) {
      throw std::runtime_error("expected one emitted image, got " +
                               std::to_string(sink_.frames.size()));
    }
    capture_->image = std::move(sink_.frames.front());
  }

 private:
  BackprojectionBackend backend_;
  FormationControl control_;
  ImageCapture* capture_;
  double origin_;
  double frequency_;
  std::array<double, 3> target_;
  int64_t pulses_;
  std::vector<double> scales_;
  TensorMapCapture sink_;
};

template <typename T>
HostTensor form_one(BackprojectionBackend backend,
                    const std::array<double, 3>& target,
                    double frequency = kFreq, int64_t pulses = kPulses,
                    std::vector<double> scales = {}) {
  const double origin =
      -std::floor(static_cast<double>(kBins) / 2.0) * kSpacing;
  ImageCapture capture;
  auto app = holoscan::make_application<BackprojectionApp<T>>(
      backend, FormationControl::kEmitAndReset, &capture, origin,
      frequency, target, pulses, std::move(scales));
  app->run();
  app->take();
  return capture.image;
}

std::pair<int64_t, int64_t> peak_index(const HostTensor& image) {
  const auto* data = image.as<std::complex<float>>();
  const bool is_f64 = image.dtype.bits == 128;
  const int64_t num_v = image.shape[image.shape.size() - 2];
  const int64_t num_u = image.shape.back();
  int64_t best_v = 0;
  int64_t best_u = 0;
  double best = -1.0;
  for (int64_t iv = 0; iv < num_v; ++iv) {
    for (int64_t iu = 0; iu < num_u; ++iu) {
      const int64_t i = iv * num_u + iu;
      const double mag =
          is_f64 ? std::abs(image.as<std::complex<double>>()[i])
                 : std::abs(data[i]);
      if (mag > best) {
        best = mag;
        best_v = iv;
        best_u = iu;
      }
    }
  }
  return {best_v, best_u};
}

std::complex<double> pixel(const HostTensor& image, int64_t iv, int64_t iu) {
  const int64_t num_u = image.shape.back();
  const int64_t i = iv * num_u + iu;
  if (image.dtype.bits == 128) {
    return image.as<std::complex<double>>()[i];
  }
  return image.as<std::complex<float>>()[i];
}

const std::array<double, 3> kTarget{0.0, 1.0, 0.0};

void expect_peak_at_target(const HostTensor& image,
                           const std::array<double, 3>& target,
                           bool check_phase, double phase_tol) {
  const auto expected = sar_predicted_peak(grid_geometry(), target);
  // Four pulses at one APC have no cross-range resolution, so argmax
  // can land on a neighbour. Assert the predicted pixel, not the
  // brightest one.
  const auto value = pixel(image, expected.first, expected.second);
  EXPECT_GT(std::abs(value), 1.0);
  if (check_phase) {
    EXPECT_NEAR(std::arg(value), 0.0, phase_tol);
  }
}

TEST(BackprojectionOp, DefaultBackendPeaksAndPhaseFloat) {
  const auto image = form_one<float>(
      holoscan::radar::kDefaultBackprojectionBackend, kTarget);
  expect_peak_at_target(image, kTarget, true, 0.25);
}

TEST(BackprojectionOp, Fp64ReferencePeaksAndPhaseFloat) {
  const auto image = form_one<float>(
      BackprojectionBackend::kNvradarFp64Reference, kTarget);
  expect_peak_at_target(image, kTarget, true, 0.15);
}

TEST(BackprojectionOp, DefaultBackendPeaksDouble) {
  const auto image = form_one<double>(
      holoscan::radar::kDefaultBackprojectionBackend, kTarget);
  expect_peak_at_target(image, kTarget, false, 0.0);
}

TEST(BackprojectionOp, KernelPulseTilesAccumulate) {
  // Host launch tile is 1024 pulses (kKernelPulseTile in the stage TU;
  // this test cannot include that CUDA header). 1300 splits 1024 + 276.
  // Identical pulses add coherently. Overwrite-not-+= would scale
  // like 276/4 instead of 1300/4.
  constexpr int64_t kTiledPulses = 1300;
  const auto four = form_one<float>(
      holoscan::radar::kDefaultBackprojectionBackend, kTarget);
  const auto tiled = form_one<float>(
      holoscan::radar::kDefaultBackprojectionBackend, kTarget, kFreq,
      kTiledPulses);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  const double mag4 =
      std::abs(pixel(four, expected.first, expected.second));
  const double magn =
      std::abs(pixel(tiled, expected.first, expected.second));
  ASSERT_GT(mag4, 1.0);
  EXPECT_NEAR(magn / mag4,
              static_cast<double>(kTiledPulses) / kPulses, 2.0);
}

TEST(BackprojectionOp, ResetWithoutEmitProducesNoImage) {
  const double origin =
      -std::floor(static_cast<double>(kBins) / 2.0) * kSpacing;
  ImageCapture capture;
  auto app = holoscan::make_application<BackprojectionApp<float>>(
      BackprojectionBackend::kNvradarFp64Reference,
      FormationControl::kResetWithoutEmit, &capture, origin, kFreq,
      kTarget);
  app->run();
  EXPECT_THROW(app->take(), std::runtime_error);
}

TEST(BackprojectionOp, PhaseLutPeaksFloat) {
  const auto image = form_one<float>(
      BackprojectionBackend::kNvradarBlockAnchoredPhaseLut, kTarget);
  expect_peak_at_target(image, kTarget, false, 0.0);
}

TEST(BackprojectionOp, ExpandedThroughputPeaksFloat) {
  const auto image = form_one<float>(
      BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput,
      kTarget);
  expect_peak_at_target(image, kTarget, false, 0.0);
}

TEST(BackprojectionOp, Fp64TrajectoryPeaksFloat) {
  const auto image = form_one<float>(
      BackprojectionBackend::kNvradarBlockAnchoredFp64Trajectory, kTarget);
  expect_peak_at_target(image, kTarget, false, 0.0);
}

TEST(BackprojectionOp, CudaFp32PeaksFloat) {
  const auto image = form_one<float>(
      BackprojectionBackend::kNvradarCudaFp32, kTarget);
  expect_peak_at_target(image, kTarget, false, 0.0);
}

TEST(BackprojectionOp, RefinedQualityPeaksFloat) {
  const auto image = form_one<float>(
      BackprojectionBackend::kNvradarBlockAnchoredRefinedQuality,
      kTarget);
  expect_peak_at_target(image, kTarget, false, 0.0);
}

TEST(BackprojectionOp, BlockAnchoredPeaksFloat) {
  const auto image = form_one<float>(
      BackprojectionBackend::kNvradarBlockAnchored, kTarget);
  expect_peak_at_target(image, kTarget, false, 0.0);
}

TEST(BackprojectionOp, CudaFp32HqPeaksFloat) {
  const auto image = form_one<float>(
      BackprojectionBackend::kNvradarCudaFp32Hq, kTarget);
  expect_peak_at_target(image, kTarget, false, 0.0);
}

#if __has_include("matx/kernels/sar_bp.cuh")
TEST(BackprojectionOp, MatxDoublePeakNearbyFloat) {
  // Even N: MatX bin origin is half a bin off until #1226. Accept a
  // 1-pixel neighbor rather than asserting phase.
  const auto image = form_one<float>(
      BackprojectionBackend::kMatxDouble, kTarget);
  const auto peak = peak_index(image);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  const int64_t dv = std::abs(peak.first - expected.first);
  const int64_t du = std::abs(peak.second - expected.second);
  EXPECT_LE(dv + du, 1);
}
#endif

TEST(BackprojectionOp, TaylorFastThrowsByName) {
  const double origin =
      -std::floor(static_cast<double>(kBins) / 2.0) * kSpacing;
  ImageCapture capture;
  auto app = holoscan::make_application<BackprojectionApp<float>>(
      BackprojectionBackend::kMatxTaylorFast,
      FormationControl::kEmitAndReset, &capture, origin, kFreq, kTarget);
  try {
    app->run();
    FAIL() << "matx_taylor_fast should throw at init";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("matx_taylor_fast"),
              std::string::npos);
  }
}

class RejectScrollWeightApp : public holoscan::Application {
 public:
  void compose() override {
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", grid_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("geometry_change", GeometryChange::kScroll));
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles",
        {profiles_impulse<float>(0.0, -8.0, {1.0, 0.0})});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions",
        {xyz_map<float>({0.0, 0.0, 100.0}, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<float>({0.0, 0.0, 0.0}, "reference_position")});
    holoscan::radar::testing::wire_data_source(
        this, op, "weight",
        {weight_map({1.0, 1.0, 1.0, 1.0})});
    holoscan::radar::testing::wire_leading_override(
        this, op, "control",
        std::vector<holoscan::TensorMap>{
            control_map(FormationControl::kEmitAndReset)});
  }
};

TEST(BackprojectionOp, ScrollRejectsConnectedWeight) {
  auto app = holoscan::make_application<RejectScrollWeightApp>();
  try {
    app->run();
    FAIL() << "scroll with weight connected should throw at init";
  } catch (const std::exception& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find("scroll"), std::string::npos);
    EXPECT_NE(msg.find("weight"), std::string::npos);
  }
}

SampleGeometry strip_geometry() {
  SampleGeometrySpec spec;
  spec.origin = {-8.0, 0.0, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {1.0, 1.0};
  spec.counts = {32, 16};
  return SampleGeometry(spec);
}

SampleGeometry translate_v(const SampleGeometry& g, int64_t k) {
  SampleGeometrySpec spec;
  spec.origin = g.origin();
  spec.axis_u = g.axis_u();
  spec.axis_v = g.axis_v();
  spec.spacing = g.spacing();
  spec.counts = g.counts();
  const double step = static_cast<double>(k) * g.spacing()[0];
  for (int i = 0; i < 3; ++i) {
    spec.origin[i] += step * g.axis_v()[i];
  }
  return SampleGeometry(spec);
}

std::vector<double> host_lattice(const SampleGeometry& g) {
  const int64_t nv = g.num_v();
  const int64_t nu = g.num_u();
  std::vector<double> xyz(static_cast<size_t>(nv * nu * 3));
  for (int64_t iv = 0; iv < nv; ++iv) {
    for (int64_t iu = 0; iu < nu; ++iu) {
      const auto p = g.sample_position(iv, iu);
      const int64_t o = (iv * nu + iu) * 3;
      xyz[static_cast<size_t>(o + 0)] = p[0];
      xyz[static_cast<size_t>(o + 1)] = p[1];
      xyz[static_cast<size_t>(o + 2)] = p[2];
    }
  }
  return xyz;
}

std::vector<double> host_entering(const SampleGeometry& g, int64_t k) {
  const int64_t nv = g.num_v();
  const int64_t nu = g.num_u();
  std::vector<double> xyz(static_cast<size_t>(k * nu * 3));
  for (int64_t r = 0; r < k; ++r) {
    for (int64_t iu = 0; iu < nu; ++iu) {
      const auto p = g.sample_position(nv - k + r, iu);
      const int64_t o = (r * nu + iu) * 3;
      xyz[static_cast<size_t>(o + 0)] = p[0];
      xyz[static_cast<size_t>(o + 1)] = p[1];
      xyz[static_cast<size_t>(o + 2)] = p[2];
    }
  }
  return xyz;
}

template <typename T>
class FormGeometryApp : public holoscan::Application {
 public:
  FormGeometryApp(SampleGeometry geometry, const std::array<double, 3>& target,
                  ImageCapture* capture)
      : geometry_(std::move(geometry)),
        target_(target),
        capture_(capture) {}

  void compose() override {
    const double origin = centred_origin();
    auto op = make_operator<BackprojectionOp<T>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", geometry_),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", origin),
        holoscan::Arg("reference_frequency", kFreq));
    const std::array<double, 3> apc{0.0, 0.0, 100.0};
    const std::array<double, 3> srp{0.0, 0.0, 0.0};
    const double dR = sar_differential_range(apc, srp, target_);
    const auto phase = sar_matched_phase(kFreq, dR);
    const std::complex<double> amp{phase.real(), -phase.imag()};
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles",
        {profiles_impulse<T>(dR, origin, amp)});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions", {xyz_map<T>(apc, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<T>(srp, "reference_position")});
    holoscan::radar::testing::wire_leading_override(
        this, op, "control",
        std::vector<holoscan::TensorMap>{
            control_map(FormationControl::kEmitAndReset)});
    holoscan::radar::testing::wire_capture_sink(this, op, "image", &sink_);
  }

  void take() {
    if (sink_.frames.size() != 1u) {
      throw std::runtime_error("expected one emitted image, got " +
                               std::to_string(sink_.frames.size()));
    }
    capture_->image = std::move(sink_.frames.front());
  }

 private:
  SampleGeometry geometry_;
  std::array<double, 3> target_;
  ImageCapture* capture_;
  TensorMapCapture sink_;
};

template <typename T>
HostTensor form_geometry(const SampleGeometry& geometry,
                         const std::array<double, 3>& target) {
  ImageCapture capture;
  auto app = holoscan::make_application<FormGeometryApp<T>>(
      geometry, target, &capture);
  app->run();
  app->take();
  return capture.image;
}

class ScrollOnceApp : public holoscan::Application {
 public:
  ScrollOnceApp(SampleGeometry geom0, SampleGeometry geom1,
                const std::array<double, 3>& target, ImageCapture* capture)
      : geom0_(std::move(geom0)),
        geom1_(std::move(geom1)),
        target_(target),
        capture_(capture) {}

  void compose() override {
    const double origin = centred_origin();
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", geom0_),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", origin),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("geometry_change", GeometryChange::kScroll));
    const std::array<double, 3> apc{0.0, 0.0, 100.0};
    const std::array<double, 3> srp{0.0, 0.0, 0.0};
    const double dR = sar_differential_range(apc, srp, target_);
    const auto phase = sar_matched_phase(kFreq, dR);
    const std::complex<double> amp{phase.real(), -phase.imag()};
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles",
        {profiles_impulse<float>(dR, origin, amp)});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions", {xyz_map<float>(apc, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<float>(srp, "reference_position")});
    holoscan::radar::testing::wire_leading_override(
        this, op, "sample_geometry", std::vector<SampleGeometry>{geom1_});
    holoscan::radar::testing::wire_leading_override(
        this, op, "control",
        std::vector<holoscan::TensorMap>{
            control_map(FormationControl::kEmitAndContinue)});
    holoscan::radar::testing::wire_capture_sink(this, op, "image", &sink_);
  }

  void take() {
    if (sink_.frames.size() != 1u) {
      throw std::runtime_error("expected one emitted image, got " +
                               std::to_string(sink_.frames.size()));
    }
    capture_->image = std::move(sink_.frames.front());
  }

 private:
  SampleGeometry geom0_;
  SampleGeometry geom1_;
  std::array<double, 3> target_;
  ImageCapture* capture_;
  TensorMapCapture sink_;
};

HostTensor scroll_once(const SampleGeometry& geom0,
                       const SampleGeometry& geom1,
                       const std::array<double, 3>& target) {
  ImageCapture capture;
  auto app = holoscan::make_application<ScrollOnceApp>(geom0, geom1, target,
                                                       &capture);
  app->run();
  app->take();
  return capture.image;
}

TEST(BackprojectionOp, ScrollWrapTargetMatchesFromScratch) {
  const auto g0 = strip_geometry();
  constexpr int64_t k = 16;
  const auto g1 = translate_v(g0, k);
  // Logical row 0 after the scroll is the wrap (physical ring_offset).
  const auto target = g1.sample_position(0, 8);
  const auto scrolled = scroll_once(g0, g1, target);
  const auto direct = form_geometry<float>(g1, target);
  const auto expected = sar_predicted_peak(g1, target);
  const auto got = pixel(scrolled, expected.first, expected.second);
  const auto ref = pixel(direct, expected.first, expected.second);
  EXPECT_EQ(expected.first, 0);
  EXPECT_GT(std::abs(got), 1.0);
  EXPECT_NEAR(std::abs(got), std::abs(ref), 0.05 * std::abs(ref));
  EXPECT_NEAR(std::arg(got), std::arg(ref), 0.2);
}

TEST(BackprojectionOp, ScrollEmittedLogicalOrder) {
  const auto g0 = strip_geometry();
  constexpr int64_t k = 16;
  const auto g1 = translate_v(g0, k);
  const auto target = g1.sample_position(0, 8);
  const auto image = scroll_once(g0, g1, target);
  const auto at_logical = pixel(image, 0, 8);
  const auto at_physical = pixel(image, 16, 8);
  EXPECT_GT(std::abs(at_logical), 1.0);
  EXPECT_LT(std::abs(at_physical), 0.25 * std::abs(at_logical));
}

class ScrollThenZeroApp : public holoscan::Application {
 public:
  explicit ScrollThenZeroApp(ImageCapture* capture)
      : capture_(capture) {}

  void compose() override {
    const auto g0 = strip_geometry();
    const auto g1 = translate_v(g0, 1);
    const double origin = centred_origin();
    const auto target = g0.sample_position(0, 8);
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", g0),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", origin),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("geometry_change", GeometryChange::kScroll));
    const std::array<double, 3> apc{0.0, 0.0, 100.0};
    const std::array<double, 3> srp{0.0, 0.0, 0.0};
    const double dR = sar_differential_range(apc, srp, target);
    const auto phase = sar_matched_phase(kFreq, dR);
    const std::complex<double> amp{phase.real(), -phase.imag()};
    const auto live = profiles_impulse<float>(dR, origin, amp);
    const auto silent = profiles_impulse<float>(dR, origin, {0.0, 0.0});
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles", {live, silent});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions",
        {xyz_map<float>(apc, "positions"),
         xyz_map<float>(apc, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<float>(srp, "reference_position"),
         xyz_map<float>(srp, "reference_position")});
    holoscan::radar::testing::wire_leading_override(
        this, op, "sample_geometry", std::vector<SampleGeometry>{g0, g1});
    holoscan::radar::testing::wire_data_source(
        this, op, "control",
        {control_map(FormationControl::kResetWithoutEmit),
         control_map(FormationControl::kEmitAndContinue)});
    holoscan::radar::testing::wire_capture_sink(this, op, "image", &sink_);
  }

  void take() {
    if (sink_.frames.size() != 1u) {
      throw std::runtime_error("expected one emitted image, got " +
                               std::to_string(sink_.frames.size()));
    }
    capture_->image = std::move(sink_.frames.front());
  }

 private:
  ImageCapture* capture_;
  TensorMapCapture sink_;
};

TEST(BackprojectionOp, ScrollEnteringRowsZeroed) {
  ImageCapture capture;
  auto app = holoscan::make_application<ScrollThenZeroApp>(&capture);
  app->run();
  app->take();
  const auto& image = capture.image;
  const int64_t nv = image.shape[image.shape.size() - 2];
  const int64_t nu = image.shape.back();
  double leading = 0.0;
  for (int64_t iu = 0; iu < nu; ++iu) {
    leading = std::max(leading, std::abs(pixel(image, nv - 1, iu)));
  }
  EXPECT_LT(leading, 1.0e-3);
}

class ScrollBadGeomApp : public holoscan::Application {
 public:
  explicit ScrollBadGeomApp(SampleGeometry next) : next_(std::move(next)) {}

  void compose() override {
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", strip_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("geometry_change", GeometryChange::kScroll));
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles",
        {profiles_impulse<float>(0.0, centred_origin(), {1.0, 0.0})});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions",
        {xyz_map<float>({0.0, 0.0, 100.0}, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<float>({0.0, 0.0, 0.0}, "reference_position")});
    holoscan::radar::testing::wire_leading_override(
        this, op, "sample_geometry", std::vector<SampleGeometry>{next_});
    holoscan::radar::testing::wire_leading_override(
        this, op, "control",
        std::vector<holoscan::TensorMap>{
            control_map(FormationControl::kEmitAndContinue)});
  }

 private:
  SampleGeometry next_;
};

TEST(BackprojectionOp, ScrollRejectsUComponent) {
  SampleGeometrySpec spec;
  spec.origin = {-7.0, 0.0, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {1.0, 1.0};
  spec.counts = {32, 16};
  auto app = holoscan::make_application<ScrollBadGeomApp>(
      SampleGeometry(spec));
  try {
    app->run();
    FAIL() << "u-component translation should throw";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("scroll"), std::string::npos);
  }
}

TEST(BackprojectionOp, ScrollRejectsFractionalRow) {
  const auto g0 = strip_geometry();
  SampleGeometrySpec spec;
  spec.origin = g0.origin();
  spec.axis_u = g0.axis_u();
  spec.axis_v = g0.axis_v();
  spec.spacing = g0.spacing();
  spec.counts = g0.counts();
  spec.origin[1] += 0.5 * g0.spacing()[0];
  auto app = holoscan::make_application<ScrollBadGeomApp>(
      SampleGeometry(spec));
  try {
    app->run();
    FAIL() << "fractional-row translation should throw";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("scroll"), std::string::npos);
    EXPECT_NE(std::string(ex.what()).find("whole"), std::string::npos);
  }
}

TEST(BackprojectionOp, ScrollRejectsKGeNumV) {
  const auto g0 = strip_geometry();
  auto app = holoscan::make_application<ScrollBadGeomApp>(
      translate_v(g0, g0.num_v()));
  try {
    app->run();
    FAIL() << "k >= num_v should throw";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("scroll"), std::string::npos);
    EXPECT_NE(std::string(ex.what()).find("num_v"), std::string::npos);
  }
}

class ScrollPointsApp : public holoscan::Application {
 public:
  ScrollPointsApp(const std::array<double, 3>& target, ImageCapture* capture)
      : target_(target), capture_(capture) {}

  void compose() override {
    const auto g0 = strip_geometry();
    constexpr int64_t k = 16;
    const auto g1 = translate_v(g0, k);
    const double origin = centred_origin();
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", origin),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("geometry_change", GeometryChange::kScroll));
    const std::array<double, 3> apc{0.0, 0.0, 100.0};
    const std::array<double, 3> srp{0.0, 0.0, 0.0};
    const double dR = sar_differential_range(apc, srp, target_);
    const auto phase = sar_matched_phase(kFreq, dR);
    const std::complex<double> amp{phase.real(), -phase.imag()};
    const auto silent =
        profiles_impulse<float>(dR, origin, {0.0, 0.0});
    const auto live = profiles_impulse<float>(dR, origin, amp);
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles", {silent, live});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions",
        {xyz_map<float>(apc, "positions"),
         xyz_map<float>(apc, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<float>(srp, "reference_position"),
         xyz_map<float>(srp, "reference_position")});
    const auto full = host_lattice(g0);
    const auto enter = host_entering(g1, k);
    holoscan::radar::testing::wire_data_source(
        this, op, "sample_points",
        {single_map(real_cube_to_device<float>(full, {32, 16, 3}),
                    "sample_points"),
         single_map(real_cube_to_device<float>(enter, {k, 16, 3}),
                    "sample_points")});
    holoscan::radar::testing::wire_data_source(
        this, op, "control",
        {control_map(FormationControl::kResetWithoutEmit),
         control_map(FormationControl::kEmitAndContinue)});
    holoscan::radar::testing::wire_capture_sink(this, op, "image", &sink_);
  }

  void take() {
    if (sink_.frames.size() != 1u) {
      throw std::runtime_error("expected one emitted image, got " +
                               std::to_string(sink_.frames.size()));
    }
    capture_->image = std::move(sink_.frames.front());
  }

 private:
  std::array<double, 3> target_;
  ImageCapture* capture_;
  TensorMapCapture sink_;
};

TEST(BackprojectionOp, ScrollPointsModeWrapMatchesGrid) {
  const auto g0 = strip_geometry();
  constexpr int64_t k = 16;
  const auto g1 = translate_v(g0, k);
  const auto target = g1.sample_position(0, 8);
  ImageCapture capture;
  auto app = holoscan::make_application<ScrollPointsApp>(target, &capture);
  app->run();
  app->take();
  const auto grid = scroll_once(g0, g1, target);
  const auto expected = sar_predicted_peak(g1, target);
  const auto pts = pixel(capture.image, expected.first, expected.second);
  const auto gpix = pixel(grid, expected.first, expected.second);
  EXPECT_GT(std::abs(pts), 1.0);
  EXPECT_NEAR(std::abs(pts), std::abs(gpix), 0.05 * std::abs(gpix));
}

class RejectNearestApp : public holoscan::Application {
 public:
  void compose() override {
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", grid_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("interpolation", RangeInterpolation::kNearest));
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles",
        {profiles_impulse<float>(0.0, -8.0, {1.0, 0.0})});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions",
        {xyz_map<float>({0.0, 0.0, 100.0}, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<float>({0.0, 0.0, 0.0}, "reference_position")});
    holoscan::radar::testing::wire_leading_override(
        this, op, "control",
        std::vector<holoscan::TensorMap>{
            control_map(FormationControl::kEmitAndReset)});
  }
};

TEST(BackprojectionOp, NearestRejected) {
  auto app = holoscan::make_application<RejectNearestApp>();
  try {
    app->run();
    FAIL() << "nearest should throw at init";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("nearest"), std::string::npos);
  }
}

class UnconnectedControlApp : public holoscan::Application {
 public:
  void compose() override {
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", grid_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("reference_frequency", kFreq));
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles",
        {profiles_impulse<float>(0.0, -8.0, {1.0, 0.0})});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions",
        {xyz_map<float>({0.0, 0.0, 100.0}, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<float>({0.0, 0.0, 0.0}, "reference_position")});
  }
};

TEST(BackprojectionOp, UnconnectedControlFailsAtInit) {
  auto app = holoscan::make_application<UnconnectedControlApp>();
  try {
    app->run();
    FAIL() << "unconnected control should throw at init";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("control"), std::string::npos);
  }
}

class TwoTickApp : public holoscan::Application {
 public:
  explicit TwoTickApp(TensorMapCapture* sink) : sink_(sink) {}

  void compose() override {
    const auto geometry = grid_geometry();
    const double origin =
        -std::floor(static_cast<double>(kBins) / 2.0) * kSpacing;
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", geometry),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", origin),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("backend",
                      BackprojectionBackend::kNvradarFp64Reference));

    const std::array<double, 3> apc{0.0, 0.0, 100.0};
    const std::array<double, 3> srp{0.0, 0.0, 0.0};
    const double dR = sar_differential_range(apc, srp, kTarget);
    const auto phase = sar_matched_phase(kFreq, dR);
    const std::complex<double> amp{phase.real(), -phase.imag()};
    const auto profiles = profiles_impulse<float>(dR, origin, amp);
    const auto pos = xyz_map<float>(apc, "positions");
    const auto ref = xyz_map<float>(srp, "reference_position");

    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles", {profiles, profiles});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions", {pos, pos});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position", {ref, ref});
    holoscan::radar::testing::wire_data_source(
        this, op, "control",
        {control_map(FormationControl::kEmitAndContinue),
         control_map(FormationControl::kEmitAndReset)});
    holoscan::radar::testing::wire_capture_sink(this, op, "image", sink_);
  }

 private:
  TensorMapCapture* sink_;
};

TEST(BackprojectionOp, TwoTicksAccumulateThenEmit) {
  TensorMapCapture sink;
  auto app = holoscan::make_application<TwoTickApp>(&sink);
  app->run();
  ASSERT_EQ(sink.frames.size(), 2u);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  const auto one_px =
      pixel(sink.frames[0], expected.first, expected.second);
  const auto two_px =
      pixel(sink.frames[1], expected.first, expected.second);
  EXPECT_GT(std::abs(one_px), 1.0);
  EXPECT_NEAR(std::abs(two_px), 2.0 * std::abs(one_px),
              0.15 * std::abs(one_px));
}

// ---------------------------------------------------------------------------
// Mode-1 landing tests. Not the full TESTING.md matrix: the path the demo
// actually runs, plus the implemented init rejections.
// ---------------------------------------------------------------------------

struct BpOptions {
  BackprojectionBackend backend{
      BackprojectionBackend::kNvradarFp64Reference};
  FormationControl control{FormationControl::kEmitAndReset};
  bool set_origin{true};
  double op_frequency{kFreq};
  double profile_frequency{kFreq};
  std::array<double, 3> target{kTarget};
  std::array<double, 3> apc{{0.0, 0.0, 100.0}};
  std::array<double, 3> srp{{0.0, 0.0, 0.0}};
  bool weight_ones{false};
  // Non-empty: drive `weight` with these values (length = kPulses).
  std::vector<double> weights;
};

class ConfigurableBpApp : public holoscan::Application {
 public:
  ConfigurableBpApp(BpOptions opt, ImageCapture* capture)
      : opt_(opt), capture_(capture) {}

  void compose() override {
    const auto geometry = grid_geometry();
    const double origin = centred_origin();
    std::shared_ptr<BackprojectionOp<float>> op;
    if (opt_.set_origin) {
      op = make_operator<BackprojectionOp<float>>(
          "bp",
          holoscan::Arg("coordinate_frame", scene_frame()),
          holoscan::Arg("sample_geometry", geometry),
          holoscan::Arg("range_bin_spacing", kSpacing),
          holoscan::Arg("range_bin_origin", origin),
          holoscan::Arg("reference_frequency", opt_.op_frequency),
          holoscan::Arg("backend", opt_.backend));
    } else {
      op = make_operator<BackprojectionOp<float>>(
          "bp",
          holoscan::Arg("coordinate_frame", scene_frame()),
          holoscan::Arg("sample_geometry", geometry),
          holoscan::Arg("range_bin_spacing", kSpacing),
          holoscan::Arg("reference_frequency", opt_.op_frequency),
          holoscan::Arg("backend", opt_.backend));
    }

    const double dR =
        sar_differential_range(opt_.apc, opt_.srp, opt_.target);
    const auto phase = sar_matched_phase(opt_.profile_frequency, dR);
    const std::complex<double> amp{phase.real(), -phase.imag()};

    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles",
        {profiles_impulse<float>(dR, origin, amp)});
    holoscan::radar::testing::wire_data_source(
        this, op, "positions", {xyz_map<float>(opt_.apc, "positions")});
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position",
        {xyz_map<float>(opt_.srp, "reference_position")});
    holoscan::radar::testing::wire_leading_override(
        this, op, "control",
        std::vector<holoscan::TensorMap>{control_map(opt_.control)});
    if (!opt_.weights.empty()) {
      holoscan::radar::testing::wire_data_source(
          this, op, "weight", {weight_map(opt_.weights)});
    } else if (opt_.weight_ones) {
      std::vector<double> ones(static_cast<size_t>(kPulses), 1.0);
      holoscan::radar::testing::wire_data_source(
          this, op, "weight", {weight_map(ones)});
    }
    holoscan::radar::testing::wire_capture_sink(
        this, op, "image", &sink_);
  }

  void take() {
    if (sink_.frames.size() != 1u) {
      throw std::runtime_error("expected one emitted image, got " +
                               std::to_string(sink_.frames.size()));
    }
    capture_->image = std::move(sink_.frames.front());
  }

 private:
  BpOptions opt_;
  ImageCapture* capture_;
  TensorMapCapture sink_;
};

HostTensor form_configured(const BpOptions& opt) {
  ImageCapture capture;
  auto app = holoscan::make_application<ConfigurableBpApp>(opt, &capture);
  app->run();
  app->take();
  return capture.image;
}

void wire_reject_inputs(
    holoscan::Fragment* app,
    const std::shared_ptr<holoscan::Operator>& op) {
  holoscan::radar::testing::wire_data_source(
      app, op, "range_profiles",
      {profiles_impulse<float>(0.0, centred_origin(), {1.0, 0.0})});
  holoscan::radar::testing::wire_data_source(
      app, op, "positions",
      {xyz_map<float>({0.0, 0.0, 100.0}, "positions")});
  holoscan::radar::testing::wire_data_source(
      app, op, "reference_position",
      {xyz_map<float>({0.0, 0.0, 0.0}, "reference_position")});
  holoscan::radar::testing::wire_leading_override(
      app, op, "control",
      std::vector<holoscan::TensorMap>{
          control_map(FormationControl::kEmitAndReset)});
}

std::string run_capturing_stderr(holoscan::Application* app) {
  setenv("HOLOSCAN_LOG_LEVEL", "WARN", 1);
  holoscan::set_log_level(holoscan::LogLevel::WARN);
  testing::internal::CaptureStderr();
  app->run();
  return testing::internal::GetCapturedStderr();
}

TEST(BackprojectionOp, DefaultOriginFormsNearerThanReference) {
  // SRP offset so the scene-centre pixel is nearer than the reference.
  // Profiles are written on the centred axis; the operator must default
  // to the same origin or the impulse falls outside the window.
  BpOptions opt;
  opt.set_origin = false;
  opt.srp = {{10.0, 0.0, 0.0}};
  opt.target = {{0.0, 0.0, 0.0}};
  const auto image = form_configured(opt);
  const auto peak = peak_index(image);
  const auto expected = sar_predicted_peak(grid_geometry(), opt.target);
  EXPECT_EQ(peak.first, expected.first);
  EXPECT_EQ(peak.second, expected.second);
  EXPECT_GT(std::abs(pixel(image, peak.first, peak.second)), 1.0);
}

TEST(BackprojectionOp, LastBatchIncludedOnEmitAndReset) {
  // Four pulses at one APC have no cross-range resolution, so two
  // ground-plane targets cannot be told apart. Empty first tick, signal
  // on the emit-and-reset tick: the closing batch is what forms the
  // image. TwoTicksAccumulateThenEmit is the complementary 2x check.
  class LastBatchApp : public holoscan::Application {
   public:
    explicit LastBatchApp(TensorMapCapture* sink) : sink_(sink) {}

    void compose() override {
      const auto geometry = grid_geometry();
      const double origin = centred_origin();
      auto op = make_operator<BackprojectionOp<float>>(
          "bp",
          holoscan::Arg("coordinate_frame", scene_frame()),
          holoscan::Arg("sample_geometry", geometry),
          holoscan::Arg("range_bin_spacing", kSpacing),
          holoscan::Arg("range_bin_origin", origin),
          holoscan::Arg("reference_frequency", kFreq),
          holoscan::Arg("backend",
                        BackprojectionBackend::kNvradarFp64Reference));

      const std::array<double, 3> apc{0.0, 0.0, 100.0};
      const std::array<double, 3> srp{0.0, 0.0, 0.0};
      const double dR = sar_differential_range(apc, srp, kTarget);
      const auto phase = sar_matched_phase(kFreq, dR);
      const std::complex<double> amp{phase.real(), -phase.imag()};
      std::vector<std::complex<double>> zeros(
          static_cast<size_t>(kPulses * kBins), {0.0, 0.0});
      const auto empty = single_map(
          complex_cube_to_device<float>(zeros, {kPulses, kBins}),
          "range_profiles");
      const auto pos = xyz_map<float>(apc, "positions");
      const auto ref = xyz_map<float>(srp, "reference_position");

      holoscan::radar::testing::wire_data_source(
          this, op, "range_profiles",
          {empty, profiles_impulse<float>(dR, origin, amp)});
      holoscan::radar::testing::wire_data_source(
          this, op, "positions", {pos, pos});
      holoscan::radar::testing::wire_data_source(
          this, op, "reference_position", {ref, ref});
      holoscan::radar::testing::wire_data_source(
          this, op, "control",
          {control_map(FormationControl::kEmitAndContinue),
           control_map(FormationControl::kEmitAndReset)});
      holoscan::radar::testing::wire_capture_sink(
          this, op, "image", sink_);
    }

   private:
    TensorMapCapture* sink_;
  };

  TensorMapCapture sink;
  auto app = holoscan::make_application<LastBatchApp>(&sink);
  app->run();
  ASSERT_EQ(sink.frames.size(), 2u);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  const auto first =
      pixel(sink.frames[0], expected.first, expected.second);
  const auto second =
      pixel(sink.frames[1], expected.first, expected.second);
  EXPECT_LT(std::abs(first), 0.25);
  EXPECT_GT(std::abs(second), 1.0);
}

TEST(BackprojectionOp, StopWarnsWhenUnemitted) {
  ImageCapture capture;
  auto app = holoscan::make_application<BackprojectionApp<float>>(
      BackprojectionBackend::kNvradarFp64Reference,
      FormationControl::kResetWithoutEmit, &capture, centred_origin(),
      kFreq, kTarget);
  const std::string log = run_capturing_stderr(app.get());
  EXPECT_NE(log.find("unemitted"), std::string::npos);
}

TEST(BackprojectionOp, StopSilentAfterEmit) {
  ImageCapture capture;
  auto app = holoscan::make_application<BackprojectionApp<float>>(
      BackprojectionBackend::kNvradarFp64Reference,
      FormationControl::kEmitAndReset, &capture, centred_origin(),
      kFreq, kTarget);
  const std::string log = run_capturing_stderr(app.get());
  EXPECT_EQ(log.find("unemitted"), std::string::npos);
}

TEST(BackprojectionOp, WrongReferenceFrequencyDecorrelates) {
  BpOptions good;
  BpOptions bad;
  bad.op_frequency = 2.0 * kFreq;
  bad.profile_frequency = kFreq;
  const auto ok = form_configured(good);
  const auto wrong = form_configured(bad);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  const auto ok_px =
      pixel(ok, expected.first, expected.second);
  const auto wrong_px =
      pixel(wrong, expected.first, expected.second);
  EXPECT_GT(std::abs(ok_px), 1.0);
  EXPECT_NEAR(std::arg(ok_px), 0.0, 0.25);
  EXPECT_GT(std::abs(wrong_px), 0.5);
  EXPECT_GT(std::abs(std::arg(wrong_px)), 1.0);
}

TEST(BackprojectionOp, UnconnectedWeightEqualsPlusOne) {
  BpOptions bare;
  BpOptions ones;
  ones.weight_ones = true;
  const auto a = form_configured(bare);
  const auto b = form_configured(ones);
  const auto peak = peak_index(a);
  EXPECT_EQ(peak, peak_index(b));
  const double mag_a = std::abs(pixel(a, peak.first, peak.second));
  const double mag_b = std::abs(pixel(b, peak.first, peak.second));
  EXPECT_NEAR(mag_b, mag_a, 0.05 * mag_a);
}

TEST(BackprojectionOp, ZeroWeightContributesNothing) {
  BpOptions opt;
  opt.weights.assign(static_cast<size_t>(kPulses), 0.0);
  const auto image = form_configured(opt);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  EXPECT_LT(std::abs(pixel(image, expected.first, expected.second)),
            0.25);
}

TEST(BackprojectionOp, FractionalWeightScalesLinearly) {
  BpOptions full;
  BpOptions half;
  half.weights.assign(static_cast<size_t>(kPulses), 0.5);
  const auto a = form_configured(full);
  const auto b = form_configured(half);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  const double mag_a =
      std::abs(pixel(a, expected.first, expected.second));
  const double mag_b =
      std::abs(pixel(b, expected.first, expected.second));
  EXPECT_GT(mag_a, 1.0);
  EXPECT_NEAR(mag_b, 0.5 * mag_a, 0.05 * mag_a);
}

// Per-tick pulse count, a uniform weight, and the control message.
struct WeightedTick {
  int64_t pulses{kPulses};
  double weight{1.0};
  FormationControl control{FormationControl::kEmitAndContinue};
  // Empty = every pulse uses the matched amplitude (scale 1).
  std::vector<double> scales{};
};

class WeightedTicksApp : public holoscan::Application {
 public:
  WeightedTicksApp(std::vector<WeightedTick> ticks,
                   TensorMapCapture* sink)
      : ticks_(std::move(ticks)), sink_(sink) {}

  void compose() override {
    const auto geometry = grid_geometry();
    const double origin = centred_origin();
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", geometry),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", origin),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("backend",
                      BackprojectionBackend::kNvradarFp64Reference));

    const std::array<double, 3> apc{0.0, 0.0, 100.0};
    const std::array<double, 3> srp{0.0, 0.0, 0.0};
    const double dR = sar_differential_range(apc, srp, kTarget);
    const auto phase = sar_matched_phase(kFreq, dR);
    const std::complex<double> amp{phase.real(), -phase.imag()};

    std::vector<holoscan::TensorMap> profiles;
    std::vector<holoscan::TensorMap> pos;
    std::vector<holoscan::TensorMap> ref;
    std::vector<holoscan::TensorMap> weights;
    std::vector<holoscan::TensorMap> control;
    profiles.reserve(ticks_.size());
    pos.reserve(ticks_.size());
    ref.reserve(ticks_.size());
    weights.reserve(ticks_.size());
    control.reserve(ticks_.size());
    for (const auto& tick : ticks_) {
      profiles.push_back(profiles_impulse<float>(
          dR, origin, amp, tick.pulses, tick.scales));
      pos.push_back(xyz_map<float>(apc, "positions", tick.pulses));
      ref.push_back(
          xyz_map<float>(srp, "reference_position", tick.pulses));
      std::vector<double> w(static_cast<size_t>(tick.pulses),
                            tick.weight);
      weights.push_back(weight_map(w));
      control.push_back(control_map(tick.control));
    }
    holoscan::radar::testing::wire_data_source(
        this, op, "range_profiles", profiles);
    holoscan::radar::testing::wire_data_source(
        this, op, "positions", pos);
    holoscan::radar::testing::wire_data_source(
        this, op, "reference_position", ref);
    holoscan::radar::testing::wire_data_source(
        this, op, "weight", weights);
    holoscan::radar::testing::wire_data_source(
        this, op, "control", control);
    holoscan::radar::testing::wire_capture_sink(
        this, op, "image", sink_);
  }

 private:
  std::vector<WeightedTick> ticks_;
  TensorMapCapture* sink_;
};

TEST(BackprojectionOp, SignedWeightCancels) {
  // +1 then the same pulses at −1 under emit-and-continue returns
  // the accumulator to empty. This is the eviction mechanism.
  TensorMapCapture sink;
  auto app = holoscan::make_application<WeightedTicksApp>(
      std::vector<WeightedTick>{
          {kPulses, 1.0, FormationControl::kEmitAndContinue},
          {kPulses, -1.0, FormationControl::kEmitAndContinue}},
      &sink);
  app->run();
  ASSERT_EQ(sink.frames.size(), 2u);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  const auto added =
      pixel(sink.frames[0], expected.first, expected.second);
  const auto cancelled =
      pixel(sink.frames[1], expected.first, expected.second);
  EXPECT_GT(std::abs(added), 1.0);
  EXPECT_LT(std::abs(cancelled), 0.05 * std::abs(added));
}

TEST(BackprojectionOp, SlidingWindowEqualsFromScratch) {
  // Eight pulses with distinct scales, then replay the first four
  // at −1. Remaining four must match a from-scratch form of scales
  // 5..8, not of the evicted 1..4 (identical pulses would not tell
  // those apart).
  const std::vector<double> all{1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0};
  const std::vector<double> evicted{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> kept{5.0, 6.0, 7.0, 8.0};
  TensorMapCapture sink;
  auto app = holoscan::make_application<WeightedTicksApp>(
      std::vector<WeightedTick>{
          {2 * kPulses, 1.0, FormationControl::kEmitAndContinue,
           all},
          {kPulses, -1.0, FormationControl::kEmitAndContinue,
           evicted}},
      &sink);
  app->run();
  ASSERT_EQ(sink.frames.size(), 2u);
  const auto eight = form_one<float>(
      BackprojectionBackend::kNvradarFp64Reference, kTarget, kFreq,
      2 * kPulses, all);
  const auto four_kept = form_one<float>(
      BackprojectionBackend::kNvradarFp64Reference, kTarget, kFreq,
      kPulses, kept);
  const auto four_evicted = form_one<float>(
      BackprojectionBackend::kNvradarFp64Reference, kTarget, kFreq,
      kPulses, evicted);
  const auto expected = sar_predicted_peak(grid_geometry(), kTarget);
  const double mag_win8 = std::abs(
      pixel(sink.frames[0], expected.first, expected.second));
  const double mag_win4 = std::abs(
      pixel(sink.frames[1], expected.first, expected.second));
  const double mag_8 =
      std::abs(pixel(eight, expected.first, expected.second));
  const double mag_kept = std::abs(
      pixel(four_kept, expected.first, expected.second));
  const double mag_gone = std::abs(
      pixel(four_evicted, expected.first, expected.second));
  EXPECT_GT(mag_8, 1.0);
  EXPECT_NEAR(mag_win8, mag_8, 0.05 * mag_8);
  EXPECT_NEAR(mag_win4, mag_kept, 0.05 * mag_kept);
  EXPECT_GT(std::abs(mag_win4 - mag_gone), 0.25 * mag_kept);
}

class RejectSincApp : public holoscan::Application {
 public:
  void compose() override {
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", grid_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("reference_frequency", kFreq),
        holoscan::Arg("interpolation", RangeInterpolation::kSinc));
    wire_reject_inputs(this, op);
  }
};

TEST(BackprojectionOp, SincRejected) {
  auto app = holoscan::make_application<RejectSincApp>();
  try {
    app->run();
    FAIL() << "sinc should throw at init";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("sinc"), std::string::npos);
  }
}

class BothGeometryModesApp : public holoscan::Application {
 public:
  void compose() override {
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", grid_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("reference_frequency", kFreq));
    std::vector<double> xyz(static_cast<size_t>(5 * 5 * 3), 0.0);
    holoscan::radar::testing::wire_data_source(
        this, op, "sample_points",
        {single_map(real_cube_to_device<float>(xyz, {5, 5, 3}),
                    "sample_points")});
    wire_reject_inputs(this, op);
  }
};

TEST(BackprojectionOp, BothGeometryModesFailAtInit) {
  auto app = holoscan::make_application<BothGeometryModesApp>();
  try {
    app->run();
    FAIL() << "both geometry modes should throw at init";
  } catch (const std::exception& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find("sample_points"), std::string::npos);
    EXPECT_NE(msg.find("sample_geometry"), std::string::npos);
  }
}

class InitialImageApp : public holoscan::Application {
 public:
  void compose() override {
    auto op = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", grid_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("reference_frequency", kFreq));
    std::vector<std::complex<double>> zeros(25, {0.0, 0.0});
    holoscan::radar::testing::wire_data_source(
        this, op, "initial_image",
        {single_map(complex_cube_to_device<float>(zeros, {5, 5}),
                    "initial_image")});
    wire_reject_inputs(this, op);
  }
};

TEST(BackprojectionOp, InitialImageRejected) {
  auto app = holoscan::make_application<InitialImageApp>();
  try {
    app->run();
    FAIL() << "initial_image should throw";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("initial_image"),
              std::string::npos);
  }
}

}  // namespace
