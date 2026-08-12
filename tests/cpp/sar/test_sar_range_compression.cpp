// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for SARRangeCompressionOp.
//
// The analytic SAR phase-history simulator
// (radar_common/sim/sar_phase_history_sim.hpp) supplies the input, and the
// ground truth is the closed-form compressed profile of the same model — a
// Dirichlet kernel whose complex value at every bin is predictable. Asserting
// against it rather than against |peak| is what makes the transform direction,
// the N/2 centring and the fft_norm scaling visible: each of those three
// survives a magnitude-only test.

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

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sim/sar_phase_history_sim.hpp"
#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::FftNorm;
using holoscan::radar::kPi;
using holoscan::radar::LFMWaveformDescriptor;
using holoscan::radar::LFMWaveformSpec;
using holoscan::radar::SARRangeCompressionOp;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::sim::sar_phase_history;
using holoscan::radar::sim::sar_range_bin_spacing;
using holoscan::radar::sim::sar_range_resolution;
using holoscan::radar::sim::SarPhaseHistoryConfig;
using holoscan::radar::sim::SarPointTarget;
using holoscan::radar::testing::complex_cube_to_device;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

constexpr int64_t kPulses = 8;
constexpr int64_t kSamples = 256;

SarPhaseHistoryConfig make_config() {
  SarPhaseHistoryConfig cfg;
  cfg.num_pulses = kPulses;
  cfg.num_samples = kSamples;
  return cfg;
}

// The waveform descriptor describing the same collection the simulator
// synthesized, so the operator derives its range axis from the physics rather
// than from anything the test hands it.
LFMWaveformDescriptor to_descriptor(const SarPhaseHistoryConfig& cfg) {
  LFMWaveformSpec spec;
  spec.carrier_frequency = cfg.carrier_frequency;
  spec.bandwidth = cfg.bandwidth;
  spec.chirp_duration = 10e-6;
  spec.sample_rate = 1.2 * cfg.bandwidth;
  spec.chirp_repetition_interval = 1e-3;
  return LFMWaveformDescriptor(spec);
}

std::vector<std::complex<double>> synthesize(
    const SarPhaseHistoryConfig& cfg,
    const std::vector<SarPointTarget>& targets) {
  return sar_phase_history(cfg, targets);
}

SarPointTarget target_at(double differential_range,
                         std::complex<double> amplitude = {1.0, 0.0}) {
  SarPointTarget target;
  target.differential_range = differential_range;
  target.amplitude = amplitude;
  return target;
}

// One (batch, pulse) slice of a captured (..., num_pulses, num_bins) tensor.
template <typename T>
std::vector<std::complex<double>> slice(const HostTensor& tensor,
                                        int64_t batch, int64_t pulse) {
  const int64_t bins = tensor.shape.back();
  const int64_t pulses = tensor.shape[tensor.shape.size() - 2];
  const auto* data = tensor.as<std::complex<T>>();
  const int64_t offset = (batch * pulses + pulse) * bins;
  std::vector<std::complex<double>> out(static_cast<size_t>(bins));
  for (int64_t i = 0; i < bins; ++i) {
    out[static_cast<size_t>(i)] = std::complex<double>(data[offset + i]);
  }
  return out;
}

std::vector<double> magnitudes(
    const std::vector<std::complex<double>>& values) {
  std::vector<double> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = std::abs(values[i]);
  }
  return out;
}

int64_t argmax(const std::vector<double>& values) {
  return static_cast<int64_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

// The -3 dB main-lobe width in bins, with the two crossings interpolated. On
// a grid whose bins are a quarter of the resolution cell, rounding to whole
// bins would be a bigger error than the effect being measured.
double minus_3db_width_bins(const std::vector<double>& magnitude) {
  const int64_t peak = argmax(magnitude);
  const double half_power =
      magnitude[static_cast<size_t>(peak)] / std::sqrt(2.0);
  const auto crossing = [&](int64_t inside, int64_t outside) {
    // A lobe that runs to the array edge has no outside sample to interpolate
    // against. Current callers keep the peak well inside, but reading the
    // neighbour unguarded is an out-of-bounds access the moment nfft or the
    // target placement changes.
    if (outside < 0 || outside >= static_cast<int64_t>(magnitude.size())) {
      return static_cast<double>(inside);
    }
    const double a = magnitude[static_cast<size_t>(inside)];
    const double b = magnitude[static_cast<size_t>(outside)];
    const double fraction = (a - half_power) / (a - b);
    return static_cast<double>(inside) +
           fraction * static_cast<double>(outside - inside);
  };
  int64_t lower = peak;
  while (lower > 0 && magnitude[static_cast<size_t>(lower - 1)] > half_power) {
    --lower;
  }
  int64_t upper = peak;
  const int64_t last = static_cast<int64_t>(magnitude.size()) - 1;
  while (upper < last &&
         magnitude[static_cast<size_t>(upper + 1)] > half_power) {
    ++upper;
  }
  return crossing(upper, upper + 1) - crossing(lower, lower - 1);
}

// The closed-form compressed profile of the simulator's model, evaluated
// analytically rather than transformed.
//
// A target at differential range dR contributes s[k] = a * exp(-j*4*pi*f_k*
// dR/c). Writing f_k = f0 + k*df and using the operator's own bin spacing,
// the k-dependent factor is exp(-j*2*pi*k*n_t/N) with n_t = dR / spacing, so
// the inverse transform is a Dirichlet kernel centred on n_t:
//
//   out[n] = (a/N) * exp(-j*4*pi*f0*dR/c) * sum_k exp(j*2*pi*k*(n - n_t)/N)
//
// with the sum in closed form below. `n` is the *signed* bin index — the
// centred output index i maps to n = i - N/2, which is precisely the N/2
// convention under test.
std::complex<double> predicted_bin(const SarPhaseHistoryConfig& cfg,
                                   const SarPointTarget& target, int64_t nfft,
                                   int64_t index, FftNorm norm) {
  const double n_t = target.differential_range /
                     sar_range_bin_spacing(cfg, nfft);
  const double n = static_cast<double>(index - nfft / 2);
  const double delta = n - n_t;
  const double f0 = cfg.carrier_frequency - 0.5 * cfg.bandwidth;
  const std::complex<double> carrier = std::polar(
      1.0, -4.0 * kPi * f0 * target.differential_range / cfg.propagation_speed);

  // sum_{k=0}^{K-1} exp(j*2*pi*k*delta/N), K = num_samples (the padded tail
  // contributes nothing), evaluated as its Dirichlet closed form.
  const int64_t k_count = cfg.num_samples;
  const double arg = kPi * delta / static_cast<double>(nfft);
  std::complex<double> sum;
  if (std::abs(std::sin(arg)) < 1e-12) {
    sum = std::complex<double>(static_cast<double>(k_count), 0.0);
  } else {
    const double magnitude =
        std::sin(arg * static_cast<double>(k_count)) / std::sin(arg);
    sum = std::polar(magnitude, arg * static_cast<double>(k_count - 1));
  }

  double scale = 1.0;
  switch (norm) {
    case FftNorm::kBackward:
      scale = 1.0 / static_cast<double>(nfft);
      break;
    case FftNorm::kOrtho:
      scale = 1.0 / std::sqrt(static_cast<double>(nfft));
      break;
    case FftNorm::kForward:
      scale = 1.0;
      break;
  }
  return target.amplitude * carrier * sum * scale;
}

// The host copies of one tick's tensor outputs.
struct RunOutputs {
  HostTensor profiles;
  HostTensor bins;
};

template <typename T, typename... ArgsT>
RunOutputs run_frame(const std::vector<std::complex<double>>& history,
                     const std::vector<int64_t>& shape, ArgsT&&... args) {
  auto harness = create_operator_test<SARRangeCompressionOp<T>>(
      std::forward<ArgsT>(args)...);
  TensorMapCapture profiles;
  TensorMapCapture bins;
  harness->template add_input_port<holoscan::TensorMap>(
      "phase_history",
      {single_map(complex_cube_to_device<T>(history, shape),
                  "phase_history")});
  harness->template add_output_port<holoscan::TensorMap>(
      "range_profiles", profiles.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "range_bins", bins.validators());
  harness->template add_output_port<WindowSpec>("range_window");
  harness->run_test();
  // Throw rather than EXPECT: front() on an empty frame list is undefined
  // behaviour, and this helper returns a value so it cannot use ASSERT. GTest
  // reports the exception as a clean failure instead of crashing the binary.
  if (profiles.frames.size() != 1u) {
    throw std::runtime_error("run_frame: expected exactly 1 range_profiles "
                             "frame, got " +
                             std::to_string(profiles.frames.size()));
  }
  RunOutputs out;
  out.profiles = std::move(profiles.frames.front());
  if (!bins.frames.empty()) {
    out.bins = std::move(bins.frames.front());
  }
  return out;
}

TEST(SARRangeCompressionOpTest, PeakMapsToDifferentialRangeViaRangeBins) {
  const SarPhaseHistoryConfig cfg = make_config();
  const double spacing = sar_range_bin_spacing(cfg, kSamples);
  // Well away from the reference point: a bandwidth/(N-1) vs bandwidth/N
  // spacing error is a range *scale* error, invisible at the centre bin.
  const auto target = target_at(96.0 * spacing);

  const RunOutputs out = run_frame<float>(
      synthesize(cfg, {target}), {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));

  ASSERT_EQ(out.profiles.shape, (std::vector<int64_t>{kPulses, kSamples}));
  ASSERT_EQ(out.bins.shape, (std::vector<int64_t>{kSamples}));

  const int64_t peak = argmax(magnitudes(slice<float>(out.profiles, 0, 0)));
  const float* axis = out.bins.as<float>();
  EXPECT_NEAR(axis[peak], target.differential_range, 0.5 * spacing);

  // The axis itself, against the model's own prediction: centred, uniform,
  // spanning -(N/2)*dr through +(N/2 - 1)*dr.
  for (int64_t i = 0; i < kSamples; ++i) {
    const double expected =
        static_cast<double>(i - kSamples / 2) * spacing;
    EXPECT_NEAR(axis[i], expected, 1e-3 * spacing) << "bin " << i;
  }
  EXPECT_NEAR(axis[0], -0.5 * static_cast<double>(kSamples) * spacing,
              1e-3 * spacing);
}

TEST(SARRangeCompressionOpTest, ReferenceRangeTargetPeaksAtCentreBin) {
  const SarPhaseHistoryConfig cfg = make_config();
  const RunOutputs out = run_frame<float>(
      synthesize(cfg, {target_at(0.0)}), {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));

  const auto profile = slice<float>(out.profiles, 0, 0);
  EXPECT_EQ(argmax(magnitudes(profile)), kSamples / 2);
  // Half-bin sensitive: N/2 centring puts zero differential range exactly on
  // the peak bin, (N-1)/2 would leave it half a bin off.
  EXPECT_NEAR(out.bins.as<float>()[kSamples / 2], 0.0,
              1e-3 * sar_range_bin_spacing(cfg, kSamples));
}

TEST(SARRangeCompressionOpTest, NearerTargetLandsBelowCentreBin) {
  const SarPhaseHistoryConfig cfg = make_config();
  const double spacing = sar_range_bin_spacing(cfg, kSamples);
  const auto near = target_at(-40.0 * spacing);
  const auto far = target_at(40.0 * spacing);

  const RunOutputs near_out = run_frame<float>(
      synthesize(cfg, {near}), {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));
  const RunOutputs far_out = run_frame<float>(
      synthesize(cfg, {far}), {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));

  const int64_t near_peak =
      argmax(magnitudes(slice<float>(near_out.profiles, 0, 0)));
  const int64_t far_peak =
      argmax(magnitudes(slice<float>(far_out.profiles, 0, 0)));

  // The regression an uncentred axis produces is not a displaced peak but a
  // vanished one: the nearer target wraps to the top of the array, where
  // BackprojectionOp's bounds check drops it.
  EXPECT_LT(near_peak, kSamples / 2);
  EXPECT_GT(far_peak, kSamples / 2);
  EXPECT_EQ(near_peak, kSamples / 2 - 40);
  EXPECT_EQ(far_peak, kSamples / 2 + 40);
  EXPECT_LT(near_out.bins.as<float>()[near_peak], 0.0f);
}

TEST(SARRangeCompressionOpTest, HalfBinTargetMatchesAnalyticProfileInPhase) {
  const SarPhaseHistoryConfig cfg = make_config();
  const double spacing = sar_range_bin_spacing(cfg, kSamples);
  // A sub-bin offset: the whole profile is a Dirichlet kernel straddling two
  // bins, and its phase is what separates an N/2 centring from (N-1)/2.
  const auto target = target_at(33.5 * spacing, {0.7, -0.4});

  const RunOutputs out = run_frame<double>(
      synthesize(cfg, {target}), {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));

  const auto profile = slice<double>(out.profiles, 0, 0);
  double peak_mag = 0.0;
  for (const auto& value : profile) {
    peak_mag = std::max(peak_mag, std::abs(value));
  }
  ASSERT_GT(peak_mag, 0.0);

  for (int64_t i = 0; i < kSamples; ++i) {
    const std::complex<double> expected =
        predicted_bin(cfg, target, kSamples, i, FftNorm::kBackward);
    EXPECT_NEAR(std::abs(profile[static_cast<size_t>(i)] - expected), 0.0,
                1e-6 * peak_mag)
        << "bin " << i;
  }
}

TEST(SARRangeCompressionOpTest, NormalizationLandsOnTheInverseTransform) {
  const SarPhaseHistoryConfig cfg = make_config();
  const auto target = target_at(0.0, {0.6, 0.25});
  const auto history = synthesize(cfg, {target});

  struct Case {
    FftNorm norm;
    const char* name;
  };
  for (const Case& c : {Case{FftNorm::kBackward, "backward"},
                        Case{FftNorm::kOrtho, "ortho"},
                        Case{FftNorm::kForward, "forward"}}) {
    const RunOutputs out = run_frame<double>(
        history, {kPulses, kSamples},
        holoscan::Arg("waveform") = to_descriptor(cfg),
        holoscan::Arg("fft_norm") = c.norm);
    const auto profile = slice<double>(out.profiles, 0, 0);
    const std::complex<double> expected =
        predicted_bin(cfg, target, kSamples, kSamples / 2, c.norm);
    // backward puts the 1/N on this stage's transform, so the peak is the
    // target's own amplitude; forward leaves it unscaled at N times that.
    EXPECT_NEAR(std::abs(profile[static_cast<size_t>(kSamples / 2)] - expected),
                0.0, 1e-9 * std::abs(expected))
        << c.name;
  }
}

TEST(SARRangeCompressionOpTest, HannWindowLowersSidelobesToItsPslr) {
  const SarPhaseHistoryConfig cfg = make_config();
  const int64_t nfft = 8 * kSamples;
  const double spacing = sar_range_bin_spacing(cfg, nfft);
  // Off-bin on the un-padded grid, so rectangular sidelobes are fully
  // exposed; oversampled so the sidelobe peaks are resolved rather than
  // sampled between.
  const auto target = target_at(33.5 * 8.0 * spacing);
  const auto history = synthesize(cfg, {target});

  WindowSpec hann;
  hann.type = WindowType::kHann;

  const auto pslr_db = [&](const RunOutputs& out, int64_t main_lobe_bins) {
    const auto magnitude = magnitudes(slice<float>(out.profiles, 0, 0));
    const int64_t peak = argmax(magnitude);
    double sidelobe = 0.0;
    for (int64_t i = 0; i < nfft; ++i) {
      if (std::abs(i - peak) <= main_lobe_bins) {
        continue;
      }
      sidelobe = std::max(sidelobe, magnitude[static_cast<size_t>(i)]);
    }
    return 20.0 * std::log10(sidelobe / magnitude[static_cast<size_t>(peak)]);
  };

  const RunOutputs rect = run_frame<float>(
      history, {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg),
      holoscan::Arg("nfft") = nfft);
  const RunOutputs hann_out = run_frame<float>(
      history, {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg),
      holoscan::Arg("nfft") = nfft, holoscan::Arg("range_window") = hann);

  // Textbook peak sidelobe levels: -13.3 dB rectangular, -31.5 dB Hann. The
  // main lobe is one null-to-null width for each (2 and 4 unpadded bins,
  // times the 8x oversampling).
  EXPECT_NEAR(pslr_db(rect, 8), -13.3, 1.5);
  EXPECT_NEAR(pslr_db(hann_out, 16), -31.5, 2.0);
}

TEST(SARRangeCompressionOpTest, NfftInterpolatesWithoutAddingResolution) {
  const SarPhaseHistoryConfig cfg = make_config();
  const int64_t nfft = 4 * kSamples;
  const double base_spacing = sar_range_bin_spacing(cfg, kSamples);
  const auto target = target_at(40.0 * base_spacing);
  const auto history = synthesize(cfg, {target});

  const RunOutputs base = run_frame<float>(
      history, {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));
  const RunOutputs fine = run_frame<float>(
      history, {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg),
      holoscan::Arg("nfft") = nfft);

  ASSERT_EQ(fine.profiles.shape, (std::vector<int64_t>{kPulses, nfft}));
  ASSERT_EQ(fine.bins.shape, (std::vector<int64_t>{nfft}));
  const float* fine_axis = fine.bins.as<float>();
  const float* base_axis = base.bins.as<float>();

  // More bins over the same extent: spacing divides by 4, span is unchanged.
  //
  // The axis is float32, which is why the second tolerance is a fraction of
  // the *span* rather than of a bin, and why neither should be tightened:
  // differencing two neighbouring bins at the array edge cancels values of
  // order span/2, leaving ~1e-7 of the span as rounding error, and multiplying
  // that difference back up by nfft restores the error at that size.
  EXPECT_NEAR(fine_axis[1] - fine_axis[0],
              0.25 * (base_axis[1] - base_axis[0]),
              1e-4 * (base_axis[1] - base_axis[0]));
  const double span = static_cast<double>(kSamples) * base_spacing;
  EXPECT_NEAR(static_cast<double>(nfft) * (fine_axis[1] - fine_axis[0]), span,
              1e-4 * span);

  // Same peak location in metres, and the same -3 dB width: zero-padding
  // interpolates the impulse response, it does not sharpen it.
  const auto magnitude = magnitudes(slice<float>(fine.profiles, 0, 0));
  const int64_t peak = argmax(magnitude);
  EXPECT_NEAR(fine_axis[peak], target.differential_range,
              0.5 * (fine_axis[1] - fine_axis[0]));

  // 0.886 * c/(2B) is the -3 dB width of an unwindowed sinc: nfft moved the
  // grid under the impulse response without changing the response.
  const double width_m =
      minus_3db_width_bins(magnitude) * (fine_axis[1] - fine_axis[0]);
  EXPECT_NEAR(width_m, 0.886 * sar_range_resolution(cfg),
              0.05 * sar_range_resolution(cfg));
}

TEST(SARRangeCompressionOpTest, LeadingBatchAxesProcessedIndependently) {
  const SarPhaseHistoryConfig cfg = make_config();
  const double spacing = sar_range_bin_spacing(cfg, kSamples);
  // Two channels holding different scenes: a batch axis processed jointly
  // (or transposed) shows up as one peak in the wrong slice.
  const auto first = synthesize(cfg, {target_at(20.0 * spacing)});
  const auto second = synthesize(cfg, {target_at(-55.0 * spacing)});
  std::vector<std::complex<double>> stacked(first);
  stacked.insert(stacked.end(), second.begin(), second.end());

  const RunOutputs out = run_frame<float>(
      stacked, {2, kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));

  ASSERT_EQ(out.profiles.shape,
            (std::vector<int64_t>{2, kPulses, kSamples}));
  for (int64_t pulse = 0; pulse < kPulses; ++pulse) {
    EXPECT_EQ(argmax(magnitudes(slice<float>(out.profiles, 0, pulse))),
              kSamples / 2 + 20)
        << "pulse " << pulse;
    EXPECT_EQ(argmax(magnitudes(slice<float>(out.profiles, 1, pulse))),
              kSamples / 2 - 55)
        << "pulse " << pulse;
  }
}

TEST(SARRangeCompressionOpTest, NoRangeBinsWithoutWaveform) {
  const SarPhaseHistoryConfig cfg = make_config();
  const auto history = synthesize(cfg, {target_at(0.0)});

  auto harness = create_operator_test<SARRangeCompressionOp<float>>();
  TensorMapCapture profiles;
  harness->add_input_port<holoscan::TensorMap>(
      "phase_history",
      {single_map(complex_cube_to_device<float>(history,
                                                {kPulses, kSamples}),
                  "phase_history")});
  harness->add_output_port<holoscan::TensorMap>("range_profiles",
                                                profiles.validators());
  harness->add_output_port<holoscan::TensorMap>("range_bins");
  harness->add_output_port<WindowSpec>("range_window");
  harness->run_test();

  ASSERT_EQ(profiles.frames.size(), 1u);
  EXPECT_EQ(profiles.frames[0].shape,
            (std::vector<int64_t>{kPulses, kSamples}));
  // The axis is derived from the waveform, so without one the port stays
  // silent rather than emitting a guessed axis.
  EXPECT_EQ(harness->get_sink<holoscan::TensorMap>("range_bins")
                ->get_received_count(),
            0u);
  // The window output does not depend on the waveform.
  EXPECT_EQ(
      harness->get_sink<WindowSpec>("range_window")->get_received_count(), 1u);
}

TEST(SARRangeCompressionOpTest, NfftBelowSampleCountErrors) {
  const SarPhaseHistoryConfig cfg = make_config();
  const auto history = synthesize(cfg, {target_at(0.0)});

  auto harness = create_operator_test<SARRangeCompressionOp<float>>(
      holoscan::Arg("nfft") = static_cast<int64_t>(kSamples - 1));
  harness->add_input_port<holoscan::TensorMap>(
      "phase_history",
      {single_map(complex_cube_to_device<float>(history,
                                                {kPulses, kSamples}),
                  "phase_history")});
  harness->add_output_port<holoscan::TensorMap>("range_profiles");
  harness->add_output_port<holoscan::TensorMap>("range_bins");
  harness->add_output_port<WindowSpec>("range_window");
  EXPECT_ANY_THROW(harness->run_test());
}

TEST(SARRangeCompressionOpTest, RealInputRejected) {
  // Phase history is complex by contract; a real tensor is a wiring mistake
  // (the reader's sample_axis, say) rather than a second supported form.
  std::vector<float> real(static_cast<size_t>(kPulses * kSamples), 1.0f);
  auto tensor = holoscan::radar::testing::to_device_tensor(
      real, {kPulses, kSamples},
      DLDataType{kDLFloat, 32, 1});

  auto harness = create_operator_test<SARRangeCompressionOp<float>>();
  harness->add_input_port<holoscan::TensorMap>(
      "phase_history", {single_map(tensor, "phase_history")});
  harness->add_output_port<holoscan::TensorMap>("range_profiles");
  harness->add_output_port<holoscan::TensorMap>("range_bins");
  harness->add_output_port<WindowSpec>("range_window");
  EXPECT_ANY_THROW(harness->run_test());
}

TEST(SARRangeCompressionOpTest, FloatAndDoubleInstantiationsAgree) {
  const SarPhaseHistoryConfig cfg = make_config();
  const double spacing = sar_range_bin_spacing(cfg, kSamples);
  const auto history = synthesize(cfg, {target_at(30.0 * spacing)});

  const RunOutputs run_f = run_frame<float>(
      history, {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));
  const RunOutputs run_d = run_frame<double>(
      history, {kPulses, kSamples},
      holoscan::Arg("waveform") = to_descriptor(cfg));

  const auto mag_f = magnitudes(slice<float>(run_f.profiles, 0, 0));
  const auto mag_d = magnitudes(slice<double>(run_d.profiles, 0, 0));
  const double peak = mag_d[static_cast<size_t>(argmax(mag_d))];
  EXPECT_EQ(argmax(mag_f), argmax(mag_d));
  for (int64_t i = 0; i < kSamples; ++i) {
    EXPECT_NEAR(mag_f[static_cast<size_t>(i)], mag_d[static_cast<size_t>(i)],
                1e-5 * peak)
        << "bin " << i;
  }

  const float* axis_f = run_f.bins.as<float>();
  const double* axis_d = run_d.bins.as<double>();
  for (int64_t i = 0; i < kSamples; ++i) {
    EXPECT_NEAR(static_cast<double>(axis_f[i]), axis_d[i],
                1e-4 * std::abs(axis_d[i]) + 1e-6);
  }
}

// The descriptor edge is not a TensorMap, so it needs a sink of its own that
// records what arrives. Its input keeps the default message-available
// condition: the port is emitted on some ticks and not others, and a kNone
// input would leave this operator permanently ready — it would then spin for
// the scheduler's whole deadlock timeout rather than the run's length.
class WindowSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(WindowSinkOp)
  WindowSinkOp() = default;
  std::vector<WindowSpec>* windows = nullptr;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<WindowSpec>("range_window");
  }
  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto window = op_input.receive<WindowSpec>("range_window");
    if (window && windows != nullptr) {
      windows->push_back(window.value());
    }
  }
};

// Override values occupy the LEADING ticks (wire_leading_override), and the
// data ticks outnumber them, so a trailing tick exercises the sticky rule.
template <typename OverrideT>
class LiveOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  std::vector<OverrideT> overrides;
  std::string port;
  LFMWaveformDescriptor waveform;
  TensorMapCapture* profiles = nullptr;
  TensorMapCapture* bins = nullptr;
  std::vector<WindowSpec>* windows = nullptr;

  void compose() override {
    auto op = make_operator<SARRangeCompressionOp<float>>(
        "op_under_test", holoscan::Arg("waveform") = waveform);
    wire_data_source(this, op, "phase_history", frames);
    wire_leading_override<OverrideT>(this, op, port, overrides);
    wire_capture_sink(this, op, "range_profiles", profiles);
    wire_capture_sink(this, op, "range_bins", bins);

    auto window_sink = this->template make_operator<WindowSinkOp>(
        "window_sink");
    window_sink->windows = windows;
    this->add_flow(op, window_sink, {{"range_window", "range_window"}});
  }
};

TEST(SARRangeCompressionOpTest, LiveWindowOverrideIsStickyAndReannounces) {
  const SarPhaseHistoryConfig cfg = make_config();
  const double spacing = sar_range_bin_spacing(cfg, kSamples);
  const auto history = synthesize(cfg, {target_at(33.5 * spacing)});
  const auto frame = [&]() {
    return single_map(
        complex_cube_to_device<float>(history, {kPulses, kSamples}),
        "phase_history");
  };
  WindowSpec hann;
  hann.type = WindowType::kHann;

  TensorMapCapture profiles;
  TensorMapCapture bins;
  std::vector<WindowSpec> windows;
  auto app = holoscan::make_application<LiveOverrideApp<WindowSpec>>();
  app->frames = {frame(), frame(), frame()};
  app->port = "range_window";
  // Tick 0 Hann; tick 1 back to rectangular (a cached configuration, so no
  // rebuild); tick 2 message-less, where the rectangular override remains.
  app->overrides = {hann, WindowSpec{}};
  app->waveform = to_descriptor(cfg);
  app->profiles = &profiles;
  app->bins = &bins;
  app->windows = &windows;
  app->run();

  ASSERT_EQ(profiles.frames.size(), 3u);
  const auto tick0 = magnitudes(slice<float>(profiles.frames[0], 0, 0));
  const auto tick1 = magnitudes(slice<float>(profiles.frames[1], 0, 0));
  const auto tick2 = magnitudes(slice<float>(profiles.frames[2], 0, 0));

  const double peak = tick1[static_cast<size_t>(argmax(tick1))];
  double diff01 = 0.0;
  double diff12 = 0.0;
  for (int64_t i = 0; i < kSamples; ++i) {
    diff01 = std::max(diff01, std::abs(tick0[static_cast<size_t>(i)] -
                                       tick1[static_cast<size_t>(i)]));
    diff12 = std::max(diff12, std::abs(tick1[static_cast<size_t>(i)] -
                                       tick2[static_cast<size_t>(i)]));
  }
  EXPECT_GT(diff01, 0.01 * peak);
  EXPECT_LE(diff12, 1e-5 * peak);

  // Announced once at the first tick and again on the change, but not on the
  // message-less tick where nothing moved.
  ASSERT_EQ(windows.size(), 2u);
  EXPECT_EQ(windows[0].type, WindowType::kHann);
  EXPECT_EQ(windows[1].type, WindowType::kNone);
}

TEST(SARRangeCompressionOpTest, LiveNfftOverrideResizesOutput) {
  const SarPhaseHistoryConfig cfg = make_config();
  const auto history = synthesize(cfg, {target_at(0.0)});
  const auto frame = [&]() {
    return single_map(
        complex_cube_to_device<float>(history, {kPulses, kSamples}),
        "phase_history");
  };

  TensorMapCapture profiles;
  TensorMapCapture bins;
  std::vector<WindowSpec> windows;
  auto app = holoscan::make_application<LiveOverrideApp<int64_t>>();
  app->frames = {frame(), frame(), frame()};
  app->port = "nfft";
  app->overrides = {2 * kSamples};
  app->waveform = to_descriptor(cfg);
  app->profiles = &profiles;
  app->bins = &bins;
  app->windows = &windows;
  app->run();

  ASSERT_EQ(profiles.frames.size(), 3u);
  ASSERT_EQ(bins.frames.size(), 3u);
  // Shape-changing override: every later tick keeps the new length, and the
  // axis follows it, which is what lets downstream re-infer.
  for (size_t tick = 0; tick < 3; ++tick) {
    EXPECT_EQ(profiles.frames[tick].shape,
              (std::vector<int64_t>{kPulses, 2 * kSamples}))
        << "tick " << tick;
    EXPECT_EQ(bins.frames[tick].shape,
              (std::vector<int64_t>{2 * kSamples}));
  }
  const float* axis = bins.frames[2].as<float>();
  EXPECT_NEAR(axis[1] - axis[0],
              sar_range_bin_spacing(cfg, 2 * kSamples),
              1e-3 * sar_range_bin_spacing(cfg, 2 * kSamples));
  // The window was never overridden, so it announces exactly once.
  EXPECT_EQ(windows.size(), 1u);
}

}  // namespace
