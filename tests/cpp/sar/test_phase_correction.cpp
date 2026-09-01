// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for PhaseCorrectionOp: add/subtract identity, broadcast across
// range, pulse-axis mismatch, required sense, float/double, and the
// formed-image sign check from TESTING.md (wrong sign doubles the
// injected error).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sim/sar_phase_history_sim.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "sar/phase_correction_op/phase_correction_op.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::BackprojectionOp;
using holoscan::radar::CoordinateFrame;
using holoscan::radar::CoordinateFrameSpec;
using holoscan::radar::CoordinateFrameType;
using holoscan::radar::FormationControl;
using holoscan::radar::PhaseCorrectionOp;
using holoscan::radar::PhaseCorrectionSense;
using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::sim::sar_differential_range;
using holoscan::radar::sim::sar_matched_phase;
using holoscan::radar::sim::sar_predicted_peak;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::complex_cube_to_device;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_device_uint8_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::test::create_operator_test;

constexpr int64_t kPulses = 8;
constexpr int64_t kBins = 16;

std::complex<double> phasor(double phi, double sign) {
  return {std::cos(sign * phi), std::sin(sign * phi)};
}

template <typename T>
std::vector<std::complex<T>> host_profiles() {
  std::vector<std::complex<T>> cube(
      static_cast<size_t>(kPulses * kBins));
  for (int64_t p = 0; p < kPulses; ++p) {
    for (int64_t r = 0; r < kBins; ++r) {
      const double re = 1.0 + static_cast<double>(r);
      const double im = 0.25 * static_cast<double>(p + 1);
      cube[static_cast<size_t>(p * kBins + r)] =
          std::complex<T>(static_cast<T>(re), static_cast<T>(im));
    }
  }
  return cube;
}

std::vector<double> host_phase() {
  std::vector<double> phi(static_cast<size_t>(kPulses));
  for (int64_t p = 0; p < kPulses; ++p) {
    phi[static_cast<size_t>(p)] = 0.35 * static_cast<double>(p);
  }
  return phi;
}

template <typename T>
HostTensor run_once(PhaseCorrectionSense sense,
                    const std::vector<std::complex<T>>& profiles,
                    const std::vector<int64_t>& pshape,
                    const std::vector<T>& phase,
                    const std::vector<int64_t>& zshape) {
  TensorMapCapture capture;
  auto harness = create_operator_test<PhaseCorrectionOp<T>>(
      holoscan::Arg("sense") = sense);
  harness->template add_input_port<holoscan::TensorMap>(
      "range_profiles",
      {single_map(to_device_tensor(profiles, pshape, complex_dtype<T>()),
                  "range_profiles")});
  harness->template add_input_port<holoscan::TensorMap>(
      "phase_correction",
      {single_map(to_device_tensor(phase, zshape, real_dtype<T>()),
                  "phase_correction")});
  harness->template add_output_port<holoscan::TensorMap>(
      "range_profiles", capture.validators());
  harness->run_test();
  if (capture.frames.size() != 1) {
    throw std::runtime_error("expected one output frame");
  }
  return std::move(capture.frames.front());
}

template <typename T>
void expect_multiply(const HostTensor& out,
                     const std::vector<std::complex<T>>& in,
                     const std::vector<T>& phi, double sign) {
  ASSERT_EQ(out.size(), static_cast<int64_t>(in.size()));
  const auto* data = out.as<std::complex<T>>();
  const int64_t bins = out.shape.back();
  const int64_t pulses = out.shape[out.shape.size() - 2];
  int64_t batch = 1;
  for (size_t i = 0; i + 2 < out.shape.size(); ++i) {
    batch *= out.shape[i];
  }
  ASSERT_EQ(static_cast<int64_t>(phi.size()), batch * pulses);
  double max_err = 0.0;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t p = 0; p < pulses; ++p) {
      const size_t pi = static_cast<size_t>(b * pulses + p);
      const auto rot =
          phasor(static_cast<double>(phi[pi]), sign);
      for (int64_t r = 0; r < bins; ++r) {
        const size_t i = pi * static_cast<size_t>(bins) +
                         static_cast<size_t>(r);
        const auto z = in[i];
        const std::complex<double> got(data[i].real(), data[i].imag());
        const std::complex<double> want =
            std::complex<double>(z.real(), z.imag()) * rot;
        max_err = std::max(max_err, std::abs(got - want));
      }
    }
  }
  EXPECT_LT(max_err, 1e-5);
}

TEST(PhaseCorrectionOp, SubtractMatchesExpNegI) {
  const auto profiles = host_profiles<float>();
  const auto phi_d = host_phase();
  std::vector<float> phi(phi_d.begin(), phi_d.end());
  const auto out = run_once<float>(PhaseCorrectionSense::kSubtract, profiles,
                                   {kPulses, kBins}, phi, {kPulses});
  expect_multiply<float>(out, profiles, phi, -1.0);
}

TEST(PhaseCorrectionOp, AddMatchesExpPosI) {
  const auto profiles = host_profiles<float>();
  const auto phi_d = host_phase();
  std::vector<float> phi(phi_d.begin(), phi_d.end());
  const auto out = run_once<float>(PhaseCorrectionSense::kAdd, profiles,
                                   {kPulses, kBins}, phi, {kPulses});
  expect_multiply<float>(out, profiles, phi, 1.0);
}

class RoundTripApp : public holoscan::Application {
 public:
  holoscan::TensorMap profiles;
  holoscan::TensorMap phase;
  TensorMapCapture* capture{nullptr};

  void compose() override {
    auto add = make_operator<PhaseCorrectionOp<float>>(
        "add", holoscan::Arg("sense") = PhaseCorrectionSense::kAdd);
    auto sub = make_operator<PhaseCorrectionOp<float>>(
        "sub",
        holoscan::Arg("sense") = PhaseCorrectionSense::kSubtract);
    wire_data_source(this, add, "range_profiles", {profiles});
    auto phase_src =
        wire_data_source(this, add, "phase_correction", {phase});
    add_flow(phase_src, sub, {{"output", "phase_correction"}});
    add_flow(add, sub, {{"range_profiles", "range_profiles"}});
    wire_capture_sink(this, sub, "range_profiles", capture);
  }
};

TEST(PhaseCorrectionOp, AddThenSubtractIsIdentity) {
  const auto host = host_profiles<float>();
  const auto phi_d = host_phase();
  std::vector<float> phi(phi_d.begin(), phi_d.end());
  TensorMapCapture capture;
  auto app = holoscan::make_application<RoundTripApp>();
  app->profiles = single_map(
      to_device_tensor(host, {kPulses, kBins}, complex_dtype<float>()),
      "range_profiles");
  app->phase = single_map(
      to_device_tensor(phi, {kPulses}, real_dtype<float>()),
      "phase_correction");
  app->capture = &capture;
  app->run();
  ASSERT_EQ(capture.frames.size(), 1u);
  const auto* data = capture.frames.front().as<std::complex<float>>();
  double max_err = 0.0;
  for (size_t i = 0; i < host.size(); ++i) {
    max_err = std::max(
        max_err, static_cast<double>(std::abs(data[i] - host[i])));
  }
  EXPECT_LT(max_err, 2e-5);
}

TEST(PhaseCorrectionOp, BroadcastsAcrossRange) {
  const auto profiles = host_profiles<float>();
  const auto phi_d = host_phase();
  std::vector<float> phi(phi_d.begin(), phi_d.end());
  const auto out = run_once<float>(PhaseCorrectionSense::kAdd, profiles,
                                   {kPulses, kBins}, phi, {kPulses});
  const auto* data = out.as<std::complex<float>>();
  // Same pulse, two bins: the applied rotation must be identical.
  const auto a = data[0 * kBins + 0];
  const auto b = data[0 * kBins + 3];
  const auto in_a = profiles[0 * kBins + 0];
  const auto in_b = profiles[0 * kBins + 3];
  const auto rot_a = a / in_a;
  const auto rot_b = b / in_b;
  EXPECT_NEAR(rot_a.real(), rot_b.real(), 1e-5);
  EXPECT_NEAR(rot_a.imag(), rot_b.imag(), 1e-5);
}

TEST(PhaseCorrectionOp, PreservesLeadingBatch) {
  constexpr int64_t kBatch = 2;
  std::vector<std::complex<float>> cube(
      static_cast<size_t>(kBatch * kPulses * kBins));
  std::vector<float> phi(static_cast<size_t>(kBatch * kPulses));
  for (int64_t b = 0; b < kBatch; ++b) {
    for (int64_t p = 0; p < kPulses; ++p) {
      phi[static_cast<size_t>(b * kPulses + p)] =
          0.2 * static_cast<float>(b + p);
      for (int64_t r = 0; r < kBins; ++r) {
        cube[static_cast<size_t>((b * kPulses + p) * kBins + r)] = {
            static_cast<float>(1 + r), static_cast<float>(b + 1)};
      }
    }
  }
  const auto out = run_once<float>(PhaseCorrectionSense::kSubtract, cube,
                                   {kBatch, kPulses, kBins}, phi,
                                   {kBatch, kPulses});
  ASSERT_EQ(out.shape.size(), 3u);
  EXPECT_EQ(out.shape[0], kBatch);
  EXPECT_EQ(out.shape[1], kPulses);
  EXPECT_EQ(out.shape[2], kBins);
  expect_multiply<float>(out, cube, phi, -1.0);
}

TEST(PhaseCorrectionOp, PulseAxisMismatchErrors) {
  const auto profiles = host_profiles<float>();
  std::vector<float> phi(static_cast<size_t>(kPulses - 1), 0.1f);
  auto harness = create_operator_test<PhaseCorrectionOp<float>>(
      holoscan::Arg("sense") = PhaseCorrectionSense::kAdd);
  harness->template add_input_port<holoscan::TensorMap>(
      "range_profiles",
      {single_map(to_device_tensor(profiles, {kPulses, kBins},
                                   complex_dtype<float>()),
                  "range_profiles")});
  harness->template add_input_port<holoscan::TensorMap>(
      "phase_correction",
      {single_map(to_device_tensor(phi, {kPulses - 1}, real_dtype<float>()),
                  "phase_correction")});
  TensorMapCapture capture;
  harness->template add_output_port<holoscan::TensorMap>(
      "range_profiles", capture.validators());
  try {
    harness->run_test();
    FAIL() << "pulse-axis mismatch should throw";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("phase_correction"),
              std::string::npos);
  }
}

TEST(PhaseCorrectionOp, SenseRequired) {
  try {
    auto harness = create_operator_test<PhaseCorrectionOp<float>>();
    const auto profiles = host_profiles<float>();
    const std::vector<float> phi(static_cast<size_t>(kPulses), 0.1f);
    harness->template add_input_port<holoscan::TensorMap>(
        "range_profiles",
        {single_map(to_device_tensor(profiles, {kPulses, kBins},
                                     complex_dtype<float>()),
                    "range_profiles")});
    harness->template add_input_port<holoscan::TensorMap>(
        "phase_correction",
        {single_map(
            to_device_tensor(phi, {kPulses}, real_dtype<float>()),
            "phase_correction")});
    TensorMapCapture capture;
    harness->template add_output_port<holoscan::TensorMap>(
        "range_profiles", capture.validators());
    harness->run_test();
    FAIL() << "omitting sense should throw";
  } catch (const std::exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("sense"), std::string::npos);
  }
}

TEST(PhaseCorrectionOp, FloatDoubleAgree) {
  const auto host_f = host_profiles<float>();
  const auto host_d = host_profiles<double>();
  const auto phi_d = host_phase();
  std::vector<float> phi_f(phi_d.begin(), phi_d.end());
  std::vector<double> phi(phi_d.begin(), phi_d.end());
  const auto out_f = run_once<float>(PhaseCorrectionSense::kAdd, host_f,
                                     {kPulses, kBins}, phi_f, {kPulses});
  const auto out_d = run_once<double>(PhaseCorrectionSense::kAdd, host_d,
                                      {kPulses, kBins}, phi, {kPulses});
  const auto* f = out_f.as<std::complex<float>>();
  const auto* d = out_d.as<std::complex<double>>();
  double max_err = 0.0;
  for (int64_t i = 0; i < out_f.size(); ++i) {
    const std::complex<double> a(f[i].real(), f[i].imag());
    max_err = std::max(max_err, std::abs(a - d[i]));
  }
  EXPECT_LT(max_err, 2e-6);
}

// --- formed-image sign (TESTING.md) ------------------------------------

constexpr int64_t kFormPulses = 16;
constexpr int64_t kFormBins = 64;
constexpr double kSpacing = 0.25;
constexpr double kFreq = 9.6e9;

double centred_origin() {
  return -std::floor(static_cast<double>(kFormBins) / 2.0) * kSpacing;
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

holoscan::TensorMap form_profiles() {
  const std::array<double, 3> apc{0.0, 0.0, 100.0};
  const std::array<double, 3> srp{0.0, 0.0, 0.0};
  const std::array<double, 3> target{0.0, 1.0, 0.0};
  const double dR = sar_differential_range(apc, srp, target);
  const auto phase = sar_matched_phase(kFreq, dR);
  const std::complex<double> amp{phase.real(), -phase.imag()};
  const double origin = centred_origin();
  const double bin_f = (dR - origin) / kSpacing;
  const int64_t bin = static_cast<int64_t>(std::llround(bin_f));
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(kFormPulses * kFormBins), {0.0, 0.0});
  if (bin >= 0 && bin < kFormBins) {
    for (int64_t p = 0; p < kFormPulses; ++p) {
      cube[static_cast<size_t>(p * kFormBins + bin)] = amp;
    }
  }
  return single_map(
      complex_cube_to_device<float>(cube, {kFormPulses, kFormBins}),
      "range_profiles");
}

holoscan::TensorMap xyz_map(const std::array<double, 3>& point,
                            const char* key) {
  std::vector<double> host(static_cast<size_t>(kFormPulses * 3));
  for (int64_t p = 0; p < kFormPulses; ++p) {
    host[static_cast<size_t>(p * 3 + 0)] = point[0];
    host[static_cast<size_t>(p * 3 + 1)] = point[1];
    host[static_cast<size_t>(p * 3 + 2)] = point[2];
  }
  return single_map(real_cube_to_device<float>(host, {kFormPulses, 3}),
                    key);
}

holoscan::TensorMap inject_phase() {
  // 0.6 rad/pulse: |sum exp(i n θ)| for n=16 is ~0.21 of coherent
  // gain at θ=0.6 and ~0.02 at 2θ. 0.4 is a trap — 16×0.4≈2π, so
  // the doubled ramp is another 2π alias and the peaks match.
  std::vector<double> phi(static_cast<size_t>(kFormPulses));
  for (int64_t p = 0; p < kFormPulses; ++p) {
    phi[static_cast<size_t>(p)] = 0.6 * static_cast<double>(p);
  }
  return single_map(real_cube_to_device<float>(phi, {kFormPulses}),
                    "phase_correction");
}

struct ImageCapture {
  HostTensor image;
};

// first_sense on the first PC; optional second_sense chains a second
// PC (add then subtract = restore, add then add = doubled error).
class FormWithPhaseApp : public holoscan::Application {
 public:
  PhaseCorrectionSense first_sense{PhaseCorrectionSense::kAdd};
  bool chain_second{false};
  PhaseCorrectionSense second_sense{PhaseCorrectionSense::kSubtract};
  ImageCapture* capture{nullptr};

  void compose() override {
    auto pc = make_operator<PhaseCorrectionOp<float>>(
        "pc", holoscan::Arg("sense") = first_sense);
    std::shared_ptr<holoscan::Operator> formed = pc;
    std::shared_ptr<PhaseCorrectionOp<float>> pc2;
    if (chain_second) {
      pc2 = make_operator<PhaseCorrectionOp<float>>(
          "pc2", holoscan::Arg("sense") = second_sense);
      add_flow(pc, pc2, {{"range_profiles", "range_profiles"}});
      formed = pc2;
    }
    auto bp = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", grid_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", centred_origin()),
        holoscan::Arg("reference_frequency", kFreq));

    const auto profiles = form_profiles();
    const auto phase = inject_phase();
    wire_data_source(this, pc, "range_profiles", {profiles});
    auto phase_src =
        wire_data_source(this, pc, "phase_correction", {phase});
    if (pc2) {
      add_flow(phase_src, pc2, {{"output", "phase_correction"}});
    }
    add_flow(formed, bp, {{"range_profiles", "range_profiles"}});
    wire_data_source(this, bp, "positions",
                     {xyz_map({0.0, 0.0, 100.0}, "positions")});
    wire_data_source(
        this, bp, "reference_position",
        {xyz_map({0.0, 0.0, 0.0}, "reference_position")});
    std::vector<uint8_t> ctrl{
        static_cast<uint8_t>(FormationControl::kEmitAndReset)};
    wire_data_source(
        this, bp, "control",
        {single_map(to_device_uint8_tensor(ctrl, {1}), "control")});
    wire_capture_sink(this, bp, "image", &sink_);
  }

  void take() {
    if (sink_.frames.empty()) {
      throw std::runtime_error("FormWithPhaseApp: no image");
    }
    capture->image = std::move(sink_.frames.front());
  }

 private:
  TensorMapCapture sink_;
};

class FormCleanApp : public holoscan::Application {
 public:
  ImageCapture* capture{nullptr};

  void compose() override {
    auto bp = make_operator<BackprojectionOp<float>>(
        "bp",
        holoscan::Arg("coordinate_frame", scene_frame()),
        holoscan::Arg("sample_geometry", grid_geometry()),
        holoscan::Arg("range_bin_spacing", kSpacing),
        holoscan::Arg("range_bin_origin", centred_origin()),
        holoscan::Arg("reference_frequency", kFreq));
    wire_data_source(this, bp, "range_profiles", {form_profiles()});
    wire_data_source(this, bp, "positions",
                     {xyz_map({0.0, 0.0, 100.0}, "positions")});
    wire_data_source(
        this, bp, "reference_position",
        {xyz_map({0.0, 0.0, 0.0}, "reference_position")});
    std::vector<uint8_t> ctrl{
        static_cast<uint8_t>(FormationControl::kEmitAndReset)};
    wire_data_source(
        this, bp, "control",
        {single_map(to_device_uint8_tensor(ctrl, {1}), "control")});
    wire_capture_sink(this, bp, "image", &sink_);
  }

  void take() {
    if (sink_.frames.empty()) {
      throw std::runtime_error("FormCleanApp: no image");
    }
    capture->image = std::move(sink_.frames.front());
  }

 private:
  TensorMapCapture sink_;
};

double peak_mag(const HostTensor& image) {
  const auto expected = sar_predicted_peak(grid_geometry(),
                                           {0.0, 1.0, 0.0});
  const int64_t num_u = image.shape.back();
  const int64_t i = expected.first * num_u + expected.second;
  if (image.dtype.bits == 128) {
    return std::abs(image.as<std::complex<double>>()[i]);
  }
  return std::abs(image.as<std::complex<float>>()[i]);
}

HostTensor run_form_clean() {
  ImageCapture cap;
  auto app = holoscan::make_application<FormCleanApp>();
  app->capture = &cap;
  app->run();
  app->take();
  return cap.image;
}

HostTensor run_form_phase(bool chain, PhaseCorrectionSense second) {
  ImageCapture cap;
  auto app = holoscan::make_application<FormWithPhaseApp>();
  app->first_sense = PhaseCorrectionSense::kAdd;
  app->chain_second = chain;
  app->second_sense = second;
  app->capture = &cap;
  app->run();
  app->take();
  return cap.image;
}

TEST(PhaseCorrectionOp, SignRestoresImpulseWrongSignDoubles) {
  const double clean = peak_mag(run_form_clean());
  const double injected = peak_mag(run_form_phase(
      false, PhaseCorrectionSense::kAdd));
  const double restored = peak_mag(run_form_phase(
      true, PhaseCorrectionSense::kSubtract));
  const double doubled = peak_mag(run_form_phase(
      true, PhaseCorrectionSense::kAdd));
  ASSERT_GT(clean, 1.0);
  EXPECT_LT(injected, 0.25 * clean);
  EXPECT_NEAR(restored, clean, 0.05 * clean);
  EXPECT_LT(doubled, injected);
}

}  // namespace
