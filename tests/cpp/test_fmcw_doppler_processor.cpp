// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for FMCWDopplerProcessorOp.
//
// Inputs are built either synthetically (a per-chirp complex exponential in
// a single range bin — a closed-form Doppler tone) or from the point-target
// simulator passed through an in-test host reference range DFT. Assertions
// run in the physical domain via the operator's own velocity_bins output,
// cross-checked against the independent bin model.
//
// Driven through the SDK-native holoscan::test::OperatorTestHarness
// (utils/sdk_harness.hpp).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/bin_model.hpp"
#include "utils/point_target_sim.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::FMCWDopplerProcessorOp;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::testing::complex_cube_to_device;
using holoscan::radar::testing::expected_doppler_bin;
using holoscan::radar::testing::expected_range_bin;
using holoscan::radar::testing::expected_velocity_axis;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_host;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

constexpr double kTestPi = 3.14159265358979323846;
constexpr double kSpeedOfLight = 299792458.0;

// Waveform shared by the velocity_bins tests (77 GHz automotive-style).
FMCWWaveformDescriptor test_waveform() {
  FMCWWaveformDescriptor waveform;
  waveform.carrier_frequency = 77e9;
  waveform.bandwidth = 150e6;
  waveform.chirp_duration = 20e-6;
  waveform.sample_rate = 12.8e6;
  waveform.chirp_repetition_interval = 25e-6;
  return waveform;
}

// Synthetic range profiles: a per-chirp complex exponential
// exp(j*2*pi*f_d*PRI*k) confined to a single range bin.
std::vector<std::complex<double>> tone_profiles(int64_t num_chirps,
                                                int64_t num_bins,
                                                int64_t range_bin,
                                                double doppler_hz,
                                                double pri) {
  std::vector<std::complex<double>> data(
      static_cast<size_t>(num_chirps * num_bins), {0.0, 0.0});
  for (int64_t k = 0; k < num_chirps; ++k) {
    const double phase =
        2.0 * kTestPi * doppler_hz * pri * static_cast<double>(k);
    data[static_cast<size_t>(k * num_bins + range_bin)] =
        std::complex<double>(std::cos(phase), std::sin(phase));
  }
  return data;
}

template <typename T>
holoscan::TensorMap profiles_frame(
    const std::vector<std::complex<double>>& data,
    const std::vector<int64_t>& shape,
    const std::string& key = "range_profiles") {
  holoscan::TensorMap map;
  map[key] = complex_cube_to_device<T>(data, shape);
  return map;
}

// Sink validators that enforce the module's payload-key convention (the
// single map entry is keyed after the port) and copy the tensor to the
// host. The capture store must outlive the harness run.
std::vector<std::function<void(const holoscan::TensorMap&)>> keyed_capture(
    TensorMapCapture* capture, const std::string& key) {
  return {[capture, key](const holoscan::TensorMap& map) {
    if (map.size() != 1 || map.begin()->first != key) {
      throw std::runtime_error(
          "expected a single-entry TensorMap keyed '" + key + "'");
    }
    capture->frames.push_back(to_host(*map.begin()->second));
  }};
}

// Run the operator once over `frame` through the SDK harness and return
// the captured host rd_maps output. velocity_bins is left unconnected
// (the port carries ConditionType::kNone).
template <typename HarnessT>
HostTensor run_rd_single(const std::shared_ptr<HarnessT>& harness,
                         holoscan::TensorMap frame) {
  TensorMapCapture rd;
  harness->template add_input_port<holoscan::TensorMap>("range_profiles",
                                                        {frame});
  harness->template add_output_port<holoscan::TensorMap>(
      "rd_maps", keyed_capture(&rd, "rd_maps"));
  harness->run_test();
  if (rd.frames.size() != 1) {
    throw std::runtime_error("expected exactly one captured rd_maps frame");
  }
  return std::move(rd.frames.front());
}

// As above, additionally capturing the velocity_bins axis (the operator
// must have a waveform set so the axis is emitted).
template <typename HarnessT>
std::pair<HostTensor, HostTensor> run_rd_velocity_single(
    const std::shared_ptr<HarnessT>& harness, holoscan::TensorMap frame) {
  TensorMapCapture rd;
  TensorMapCapture velocity;
  harness->template add_input_port<holoscan::TensorMap>("range_profiles",
                                                        {frame});
  harness->template add_output_port<holoscan::TensorMap>(
      "rd_maps", keyed_capture(&rd, "rd_maps"));
  harness->template add_output_port<holoscan::TensorMap>(
      "velocity_bins", keyed_capture(&velocity, "velocity_bins"));
  harness->run_test();
  if (rd.frames.size() != 1 || velocity.frames.size() != 1) {
    throw std::runtime_error(
        "expected exactly one captured frame per output port");
  }
  return {std::move(rd.frames.front()),
          std::move(velocity.frames.front())};
}

// |rd_maps| at one (batch, doppler, range) cell of a captured host tensor.
template <typename T>
double cell_mag(const HostTensor& tensor, int64_t batch, int64_t num_doppler,
                int64_t num_range_bins, int64_t doppler, int64_t range) {
  const std::complex<T>* data = tensor.as<std::complex<T>>();
  const std::complex<T> value =
      data[(batch * num_doppler + doppler) * num_range_bins + range];
  return std::abs(std::complex<double>(value.real(), value.imag()));
}

// Argmax along the Doppler axis at a fixed range bin.
template <typename T>
int64_t doppler_argmax(const HostTensor& tensor, int64_t batch,
                       int64_t num_doppler, int64_t num_range_bins,
                       int64_t range) {
  int64_t best = 0;
  double best_mag = -1.0;
  for (int64_t d = 0; d < num_doppler; ++d) {
    const double mag =
        cell_mag<T>(tensor, batch, num_doppler, num_range_bins, d, range);
    if (mag > best_mag) {
      best_mag = mag;
      best = d;
    }
  }
  return best;
}

// The velocity_bins output as host doubles.
template <typename T>
std::vector<double> velocity_axis_of(const HostTensor& tensor) {
  const T* data = tensor.as<T>();
  std::vector<double> axis(static_cast<size_t>(tensor.size()));
  for (int64_t i = 0; i < tensor.size(); ++i) {
    axis[static_cast<size_t>(i)] = static_cast<double>(data[i]);
  }
  return axis;
}

// Multi-tick application for the override tests: the override value
// sequence occupies the leading ticks only (TestHarnessSourceOp stops
// emitting once exhausted), so trailing data ticks exercise the
// sticky-override rule (an override remains effective
// until the next override).
template <typename V>
class LiveOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  std::string override_port;
  std::vector<V> overrides;
  std::vector<holoscan::Arg> op_args;
  TensorMapCapture* rd = nullptr;
  TensorMapCapture* velocity = nullptr;  // wired only when non-null

  void compose() override {
    auto op =
        make_operator<FMCWDopplerProcessorOp<float>>("op_under_test");
    for (const auto& arg : op_args) {
      op->add_arg(arg);
    }
    wire_data_source(this, op, "range_profiles", frames);
    wire_leading_override<V>(this, op, override_port, overrides);
    wire_capture_sink(this, op, "rd_maps", rd);
    if (velocity != nullptr) {
      wire_capture_sink(this, op, "velocity_bins", velocity);
    }
  }
};

TEST(FMCWDopplerProcessorOpTest, VelocityPeakMatchesTruthCentered) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 16;
  const int64_t range_bin = 5;
  const double radial_velocity = 5.0;  // m/s, positive = receding
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval;
  const double wavelength = waveform.wavelength();
  const double doppler_hz = -2.0 * radial_velocity / wavelength;

  // Module convention: a single-entry TensorMap is accepted regardless of
  // its key.
  const auto [rd_map, bins] = run_rd_velocity_single(
      create_operator_test<FMCWDopplerProcessorOp<float>>(
          holoscan::Arg("waveform") = waveform),
      profiles_frame<float>(
          tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
          {num_chirps, num_bins}, "profiles"));

  ASSERT_EQ(rd_map.shape, (std::vector<int64_t>{num_chirps, num_bins}));

  const int64_t peak =
      doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, range_bin);
  EXPECT_EQ(peak, expected_doppler_bin(radial_velocity, wavelength, pri,
                                       num_chirps, true));

  ASSERT_EQ(bins.shape, (std::vector<int64_t>{num_chirps}));
  const auto axis = velocity_axis_of<float>(bins);
  const auto expected =
      expected_velocity_axis(num_chirps, wavelength, pri, true);
  for (int64_t i = 0; i < num_chirps; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(axis[static_cast<size_t>(i)]),
                    static_cast<float>(expected[static_cast<size_t>(i)]))
        << "velocity_bins mismatch at bin " << i;
  }

  // The peak, read through the operator's own axis, is within one bin.
  const double bin_width =
      wavelength / 2.0 / pri / static_cast<double>(num_chirps);
  EXPECT_NEAR(axis[static_cast<size_t>(peak)], radial_velocity,
              bin_width + 1e-9);
}

TEST(FMCWDopplerProcessorOpTest, VelocityAxisUncentered) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 8;
  const int64_t range_bin = 3;
  const double radial_velocity = -4.0;  // approaching
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval;
  const double wavelength = waveform.wavelength();
  const double doppler_hz = -2.0 * radial_velocity / wavelength;

  const auto [rd_map, bins] = run_rd_velocity_single(
      create_operator_test<FMCWDopplerProcessorOp<float>>(
          holoscan::Arg("waveform") = waveform,
          holoscan::Arg("center_zero_doppler") = false),
      profiles_frame<float>(
          tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
          {num_chirps, num_bins}));

  const int64_t peak =
      doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, range_bin);
  EXPECT_EQ(peak, expected_doppler_bin(radial_velocity, wavelength, pri,
                                       num_chirps, false));

  const auto axis = velocity_axis_of<float>(bins);
  const auto expected =
      expected_velocity_axis(num_chirps, wavelength, pri, false);
  for (int64_t i = 0; i < num_chirps; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(axis[static_cast<size_t>(i)]),
                    static_cast<float>(expected[static_cast<size_t>(i)]))
        << "velocity_bins mismatch at bin " << i;
  }
  EXPECT_DOUBLE_EQ(axis[0], 0.0);  // DC at bin 0

  const double bin_width =
      wavelength / 2.0 / pri / static_cast<double>(num_chirps);
  EXPECT_NEAR(axis[static_cast<size_t>(peak)], radial_velocity,
              bin_width + 1e-9);
}

TEST(FMCWDopplerProcessorOpTest, CenterZeroDopplerPlacesDc) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const int64_t range_bin = 1;
  const auto dc_tone =
      tone_profiles(num_chirps, num_bins, range_bin, 0.0, 25e-6);

  // center_zero_doppler = true (default): DC at num_doppler / 2.
  {
    const HostTensor rd_map = run_rd_single(
        create_operator_test<FMCWDopplerProcessorOp<float>>(),
        profiles_frame<float>(dc_tone, {num_chirps, num_bins}));
    const int64_t peak =
        doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, range_bin);
    EXPECT_EQ(peak, num_chirps / 2);
    // Coherent sum of N unit samples under fft_norm = backward.
    EXPECT_NEAR(cell_mag<float>(rd_map, 0, num_chirps, num_bins, peak,
                                range_bin),
                static_cast<double>(num_chirps), 1e-3);
  }

  // center_zero_doppler = false: DC at bin 0.
  {
    const HostTensor rd_map = run_rd_single(
        create_operator_test<FMCWDopplerProcessorOp<float>>(
            holoscan::Arg("center_zero_doppler") = false),
        profiles_frame<float>(dc_tone, {num_chirps, num_bins}));
    EXPECT_EQ(doppler_argmax<float>(rd_map, 0, num_chirps, num_bins,
                                    range_bin),
              0);
  }
}

TEST(FMCWDopplerProcessorOpTest, OddChirpCountFftshiftMatchesNumpy) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 31;  // odd length exercises fftshift semantics
  const int64_t num_bins = 4;
  const int64_t range_bin = 2;
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval;
  const double wavelength = waveform.wavelength();
  // A tone exactly on unshifted bin 3: after numpy-style fftshift with odd
  // n, bin u lands at (u + floor(n/2)) % n = (3 + 15) % 31 = 18.
  const double doppler_hz = 3.0 / (pri * static_cast<double>(num_chirps));

  const auto [rd_map, bins] = run_rd_velocity_single(
      create_operator_test<FMCWDopplerProcessorOp<float>>(
          holoscan::Arg("waveform") = waveform),
      profiles_frame<float>(
          tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
          {num_chirps, num_bins}));

  const int64_t peak =
      doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, range_bin);
  EXPECT_EQ(peak, (3 + num_chirps / 2) % num_chirps);

  // DC sits at floor(n/2) = 15 on the centered odd-length axis.
  const auto axis = velocity_axis_of<float>(bins);
  EXPECT_DOUBLE_EQ(axis[static_cast<size_t>(num_chirps / 2)], 0.0);
  const auto expected =
      expected_velocity_axis(num_chirps, wavelength, pri, true);
  for (int64_t i = 0; i < num_chirps; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(axis[static_cast<size_t>(i)]),
                    static_cast<float>(expected[static_cast<size_t>(i)]))
        << "velocity_bins mismatch at bin " << i;
  }
}

TEST(FMCWDopplerProcessorOpTest, HannWindowPreservesOnBinPeak) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const int64_t range_bin = 1;
  const double pri = 25e-6;
  // Tone exactly on unshifted bin 4.
  const double doppler_hz = 4.0 / (pri * static_cast<double>(num_chirps));
  const auto tone =
      tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri);

  auto peak_magnitude = [&](const WindowSpec& window) {
    const HostTensor rd_map = run_rd_single(
        create_operator_test<FMCWDopplerProcessorOp<float>>(
            holoscan::Arg("doppler_window") = window,
            holoscan::Arg("center_zero_doppler") = false),
        profiles_frame<float>(tone, {num_chirps, num_bins}));
    const int64_t peak =
        doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, range_bin);
    EXPECT_EQ(peak, 4);
    return cell_mag<float>(rd_map, 0, num_chirps, num_bins, peak, range_bin);
  };

  WindowSpec hann;
  hann.type = WindowType::kHann;
  const double rect_peak = peak_magnitude(WindowSpec{});
  const double hann_peak = peak_magnitude(hann);

  // fft_norm = backward and coherent-gain normalization: both are the
  // coherent sum of N unit samples, within a few percent.
  EXPECT_NEAR(rect_peak, static_cast<double>(num_chirps), 1e-3);
  EXPECT_NEAR(hann_peak, rect_peak, 0.03 * rect_peak);
}

// Worst |rd_maps| leakage outside the tone's main lobe (bins 1..8) at
// `range_bin`, shared by the Hann-sidelobe tests below.
double max_doppler_sidelobe(const HostTensor& rd_map, int64_t num_chirps,
                            int64_t num_bins, int64_t range_bin) {
  double worst = 0.0;
  for (int64_t d = 0; d < num_chirps; ++d) {
    if (d >= 1 && d <= 8) {
      continue;
    }
    worst = std::max(worst, cell_mag<float>(rd_map, 0, num_chirps,
                                            num_bins, d, range_bin));
  }
  return worst;
}

TEST(FMCWDopplerProcessorOpTest, HannWindowLowersDopplerSidelobes) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const int64_t range_bin = 2;
  const double pri = 25e-6;
  // A half-bin offset tone maximizes rectangular-window leakage.
  const double doppler_hz = 4.5 / (pri * static_cast<double>(num_chirps));
  const auto tone =
      tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri);

  auto max_sidelobe = [&](const WindowSpec& window) {
    const HostTensor rd_map = run_rd_single(
        create_operator_test<FMCWDopplerProcessorOp<float>>(
            holoscan::Arg("doppler_window") = window,
            holoscan::Arg("center_zero_doppler") = false),
        profiles_frame<float>(tone, {num_chirps, num_bins}));
    // The tone sits between bins 4 and 5; exclude the main lobe (bins
    // 1..8) and take the worst remaining leakage.
    return max_doppler_sidelobe(rd_map, num_chirps, num_bins, range_bin);
  };

  WindowSpec hann;
  hann.type = WindowType::kHann;
  const double rect_sidelobe = max_sidelobe(WindowSpec{});
  const double hann_sidelobe = max_sidelobe(hann);
  EXPECT_GT(rect_sidelobe, 0.0);
  EXPECT_LT(hann_sidelobe, 0.4 * rect_sidelobe);
}

TEST(FMCWDopplerProcessorOpTest, NfftInterpolatesVelocityGrid) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t nfft = 64;
  const int64_t num_bins = 8;
  const int64_t range_bin = 6;
  const double radial_velocity = 7.0;
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval;
  const double wavelength = waveform.wavelength();
  const double doppler_hz = -2.0 * radial_velocity / wavelength;

  const auto [rd_map, bins] = run_rd_velocity_single(
      create_operator_test<FMCWDopplerProcessorOp<float>>(
          holoscan::Arg("waveform") = waveform,
          holoscan::Arg("nfft") = int64_t{nfft}),
      profiles_frame<float>(
          tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
          {num_chirps, num_bins}));

  ASSERT_EQ(rd_map.shape, (std::vector<int64_t>{nfft, num_bins}));

  const auto axis = velocity_axis_of<float>(bins);
  ASSERT_EQ(static_cast<int64_t>(axis.size()), nfft);
  const auto expected = expected_velocity_axis(nfft, wavelength, pri, true);
  for (int64_t i = 0; i < nfft; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(axis[static_cast<size_t>(i)]),
                    static_cast<float>(expected[static_cast<size_t>(i)]))
        << "velocity_bins mismatch at bin " << i;
  }

  // Bin spacing halves relative to the exact-N grid; the peak position in
  // m/s stays within the *original* resolution of the truth (zero-padding
  // interpolates, it does not resolve).
  const double original_bin_width =
      wavelength / 2.0 / pri / static_cast<double>(num_chirps);
  EXPECT_NEAR(std::abs(axis[1] - axis[0]), original_bin_width / 2.0,
              1e-5 * original_bin_width);  // float32 axis differencing
  const int64_t peak =
      doppler_argmax<float>(rd_map, 0, nfft, num_bins, range_bin);
  EXPECT_NEAR(axis[static_cast<size_t>(peak)], radial_velocity,
              original_bin_width + 1e-9);
}

TEST(FMCWDopplerProcessorOpTest, LeadingBatchAxesProcessedIndependently) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t batch = 2;
  const int64_t num_chirps = 32;
  const int64_t num_bins = 8;
  const double pri = 25e-6;
  // Slice 0: tone on unshifted bin 3 in range bin 2; slice 1: tone on
  // unshifted bin 10 in range bin 6.
  const double f0 = 3.0 / (pri * static_cast<double>(num_chirps));
  const double f1 = 10.0 / (pri * static_cast<double>(num_chirps));
  auto cube = tone_profiles(num_chirps, num_bins, 2, f0, pri);
  const auto slice1 = tone_profiles(num_chirps, num_bins, 6, f1, pri);
  cube.insert(cube.end(), slice1.begin(), slice1.end());

  const HostTensor rd_map = run_rd_single(
      create_operator_test<FMCWDopplerProcessorOp<float>>(
          holoscan::Arg("center_zero_doppler") = false),
      profiles_frame<float>(cube, {batch, num_chirps, num_bins}));

  ASSERT_EQ(rd_map.shape,
            (std::vector<int64_t>{batch, num_chirps, num_bins}));
  EXPECT_EQ(doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, 2), 3);
  EXPECT_EQ(doppler_argmax<float>(rd_map, 1, num_chirps, num_bins, 6), 10);
  // Each slice's energy stays out of the other slice's peak cell.
  EXPECT_LT(cell_mag<float>(rd_map, 0, num_chirps, num_bins, 10, 6), 1e-3);
  EXPECT_LT(cell_mag<float>(rd_map, 1, num_chirps, num_bins, 3, 2), 1e-3);
}

TEST(FMCWDopplerProcessorOpTest, LiveOverrideCenterZeroDoppler) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const int64_t range_bin = 1;
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval;
  const double wavelength = waveform.wavelength();
  auto frame = [&]() {
    return profiles_frame<float>(
        tone_profiles(num_chirps, num_bins, range_bin, 0.0, pri),
        {num_chirps, num_bins});
  };

  // Parameter default (no override): centered — DC peak at
  // num_doppler / 2.
  {
    const HostTensor rd_map = run_rd_single(
        create_operator_test<FMCWDopplerProcessorOp<float>>(
            holoscan::Arg("waveform") = waveform),
        frame());
    EXPECT_EQ(doppler_argmax<float>(rd_map, 0, num_chirps, num_bins,
                                    range_bin),
              num_chirps / 2);
  }

  // Overrides occupy leading ticks only: tick 0 overrides to uncentered,
  // tick 1 carries no message — the override is sticky.
  TensorMapCapture rd;
  TensorMapCapture velocity;
  auto app = holoscan::make_application<LiveOverrideApp<bool>>();
  app->frames = {frame(), frame()};
  app->override_port = "center_zero_doppler";
  app->overrides = {false};
  app->op_args = {holoscan::Arg("waveform") = waveform};
  app->rd = &rd;
  app->velocity = &velocity;
  app->run();

  ASSERT_EQ(rd.frames.size(), 2u);
  // Tick 0: override to uncentered — DC peak at bin 0.
  EXPECT_EQ(doppler_argmax<float>(rd.frames[0], 0, num_chirps, num_bins,
                                  range_bin),
            0);
  // Tick 1: no override message — still uncentered (sticky), and the
  // emitted velocity axis reorders to match.
  EXPECT_EQ(doppler_argmax<float>(rd.frames[1], 0, num_chirps, num_bins,
                                  range_bin),
            0);
  ASSERT_EQ(velocity.frames.size(), 2u);
  const auto axis1 = velocity_axis_of<float>(velocity.frames[1]);
  const auto expected1 =
      expected_velocity_axis(num_chirps, wavelength, pri, false);
  for (int64_t i = 0; i < num_chirps; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(axis1[static_cast<size_t>(i)]),
                    static_cast<float>(expected1[static_cast<size_t>(i)]));
  }
}

TEST(FMCWDopplerProcessorOpTest, LiveOverrideDopplerWindow) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const int64_t range_bin = 2;
  const double pri = 25e-6;
  const double doppler_hz = 4.5 / (pri * static_cast<double>(num_chirps));
  auto frame = [&]() {
    return profiles_frame<float>(
        tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
        {num_chirps, num_bins});
  };

  // Parameter default (rectangular window) as the leakage baseline.
  const HostTensor rd_rect = run_rd_single(
      create_operator_test<FMCWDopplerProcessorOp<float>>(
          holoscan::Arg("center_zero_doppler") = false),
      frame());
  const double rect_sidelobe =
      max_doppler_sidelobe(rd_rect, num_chirps, num_bins, range_bin);

  // Tick 0 overrides to Hann; tick 1 has no message (sticky).
  WindowSpec hann;
  hann.type = WindowType::kHann;
  TensorMapCapture rd;
  auto app = holoscan::make_application<LiveOverrideApp<WindowSpec>>();
  app->frames = {frame(), frame()};
  app->override_port = "doppler_window";
  app->overrides = {hann};
  app->op_args = {holoscan::Arg("center_zero_doppler") = false};
  app->rd = &rd;
  app->run();

  ASSERT_EQ(rd.frames.size(), 2u);
  EXPECT_GT(rect_sidelobe, 0.0);
  // Override applies on tick 0 and remains in effect on tick 1.
  for (size_t tick = 0; tick < rd.frames.size(); ++tick) {
    const double hann_sidelobe = max_doppler_sidelobe(
        rd.frames[tick], num_chirps, num_bins, range_bin);
    EXPECT_LT(hann_sidelobe, 0.4 * rect_sidelobe) << "tick " << tick;
  }
}

TEST(FMCWDopplerProcessorOpTest, VelocityBinsOnlyEmittedWithWaveform) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  auto frame = [&]() {
    return profiles_frame<float>(
        tone_profiles(num_chirps, num_bins, 1, 0.0, 25e-6),
        {num_chirps, num_bins});
  };

  // Without a waveform: rd_maps flows, velocity_bins never emits. The
  // velocity sink carries no validators (a validator-less sink tolerates
  // absence), so the assertion is on its received count.
  {
    TensorMapCapture rd;
    auto harness = create_operator_test<FMCWDopplerProcessorOp<float>>();
    harness->add_input_port<holoscan::TensorMap>("range_profiles",
                                                 {frame()});
    harness->add_output_port<holoscan::TensorMap>(
        "rd_maps", keyed_capture(&rd, "rd_maps"));
    harness->add_output_port<holoscan::TensorMap>("velocity_bins");
    harness->run_test();
    ASSERT_EQ(rd.frames.size(), 1u);
    auto velocity_sink =
        harness->get_sink<holoscan::TensorMap>("velocity_bins");
    ASSERT_NE(velocity_sink, nullptr);
    EXPECT_EQ(velocity_sink->get_received_count(), 0u);
  }

  // A waveform override enables the axis. The override can only
  // occupy leading ticks: tick 0 supplies the waveform (bins emitted),
  // tick 1 carries no message (sticky — bins still emitted).
  {
    TensorMapCapture rd;
    TensorMapCapture velocity;
    auto app = holoscan::make_application<
        LiveOverrideApp<FMCWWaveformDescriptor>>();
    app->frames = {frame(), frame()};
    app->override_port = "waveform";
    app->overrides = {test_waveform()};
    app->rd = &rd;
    app->velocity = &velocity;
    app->run();

    ASSERT_EQ(rd.frames.size(), 2u);
    ASSERT_EQ(velocity.frames.size(), 2u);
    EXPECT_EQ(velocity.frames[0].shape,
              (std::vector<int64_t>{num_chirps}));
    EXPECT_EQ(velocity.frames[1].shape,
              (std::vector<int64_t>{num_chirps}));
  }
}

TEST(FMCWDopplerProcessorOpTest, FloatAndDoubleAgree) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 32;
  const int64_t num_bins = 8;
  const double pri = 25e-6;
  const double doppler_hz = 6.3 / (pri * static_cast<double>(num_chirps));
  const auto tone = tone_profiles(num_chirps, num_bins, 4, doppler_hz, pri);

  auto run_precision = [&](auto precision_tag) {
    using T = decltype(precision_tag);
    return run_rd_single(
        create_operator_test<FMCWDopplerProcessorOp<T>>(),
        profiles_frame<T>(tone, {num_chirps, num_bins}));
  };
  const HostTensor rd_float = run_precision(float{});
  const HostTensor rd_double = run_precision(double{});

  ASSERT_EQ(rd_float.shape, rd_double.shape);
  for (int64_t d = 0; d < num_chirps; ++d) {
    for (int64_t r = 0; r < num_bins; ++r) {
      const double mag_f =
          cell_mag<float>(rd_float, 0, num_chirps, num_bins, d, r);
      const double mag_d =
          cell_mag<double>(rd_double, 0, num_chirps, num_bins, d, r);
      EXPECT_NEAR(mag_f, mag_d, 1e-3 * static_cast<double>(num_chirps))
          << "float/double disagree at (doppler=" << d << ", range=" << r
          << ")";
    }
  }
}

TEST(FMCWDopplerProcessorOpTest, RealInputRejected) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t num_chirps = 8;
  const int64_t num_bins = 4;
  const std::vector<double> real_cube(
      static_cast<size_t>(num_chirps * num_bins), 1.0);
  holoscan::TensorMap map;
  map["range_profiles"] =
      real_cube_to_device<float>(real_cube, {num_chirps, num_bins});

  EXPECT_THROW(
      run_rd_single(create_operator_test<FMCWDopplerProcessorOp<float>>(),
                    map),
      std::exception);
}

TEST(FMCWDopplerProcessorOpTest, MultiEntryInputRejected) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto tone = tone_profiles(8, 4, 1, 0.0, 25e-6);
  holoscan::TensorMap map;
  map["a"] = complex_cube_to_device<float>(tone, {8, 4});
  map["b"] = complex_cube_to_device<float>(tone, {8, 4});

  EXPECT_THROW(
      run_rd_single(create_operator_test<FMCWDopplerProcessorOp<float>>(),
                    map),
      std::exception);
}

TEST(FMCWDopplerProcessorOpTest, NfftSmallerThanChirpCountRejected) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto tone = tone_profiles(32, 4, 1, 0.0, 25e-6);
  EXPECT_THROW(
      run_rd_single(create_operator_test<FMCWDopplerProcessorOp<float>>(
                        holoscan::Arg("nfft") = int64_t{16}),
                    profiles_frame<float>(tone, {32, 4})),
      std::exception);
}

// Simulator path: a point target's ADC cube passed through an in-test host
// reference range DFT, then through the operator — the physical-domain
// velocity read via velocity_bins matches the simulator's ground truth.
TEST(FMCWDopplerProcessorOpTest, SimulatorTargetMapsToTruthVelocity) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  namespace rt = holoscan::radar::testing;

  rt::SimScene scene;
  scene.num_chirps = 32;
  scene.samples_per_chirp = 64;
  scene.waveform.carrier_frequency = 77e9;
  scene.waveform.bandwidth = 150e6;
  scene.waveform.chirp_duration = 20e-6;
  scene.waveform.sample_rate = 12.8e6;
  scene.waveform.chirp_repetition_interval = 25e-6;
  rt::PointTarget target;
  target.position = {20.0, 0.0, 0.0};  // 20 m down-range
  target.velocity = {6.0, 0.0, 0.0};   // 6 m/s receding
  scene.targets.push_back(target);
  const rt::SimResult sim = rt::simulate(scene);
  ASSERT_EQ(sim.truth.size(), 1u);

  // Host reference range DFT along fast time (rectangular window,
  // backward normalization) — independent of the range operator.
  const int64_t num_chirps = scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  std::vector<std::complex<double>> profiles(
      static_cast<size_t>(num_chirps * samples));
  for (int64_t k = 0; k < num_chirps; ++k) {
    for (int64_t b = 0; b < samples; ++b) {
      std::complex<double> acc{0.0, 0.0};
      for (int64_t s = 0; s < samples; ++s) {
        const double phase = -2.0 * kTestPi * static_cast<double>(b) *
                             static_cast<double>(s) /
                             static_cast<double>(samples);
        acc += sim.cube[static_cast<size_t>(k * samples + s)] *
               std::complex<double>(std::cos(phase), std::sin(phase));
      }
      profiles[static_cast<size_t>(k * samples + b)] = acc;
    }
  }

  FMCWWaveformDescriptor waveform;
  waveform.carrier_frequency = scene.waveform.carrier_frequency;
  waveform.bandwidth = scene.waveform.bandwidth;
  waveform.chirp_duration = scene.waveform.chirp_duration;
  waveform.sample_rate = scene.waveform.sample_rate;
  waveform.chirp_repetition_interval =
      scene.waveform.chirp_repetition_interval;

  const auto [rd_map, bins] = run_rd_velocity_single(
      create_operator_test<FMCWDopplerProcessorOp<float>>(
          holoscan::Arg("waveform") = waveform),
      profiles_frame<float>(profiles, {num_chirps, samples}));

  const int64_t range_bin = expected_range_bin(
      sim.truth[0].range, scene.waveform.slope(),
      scene.waveform.sample_rate, samples, kSpeedOfLight);
  const int64_t peak =
      doppler_argmax<float>(rd_map, 0, num_chirps, samples, range_bin);
  EXPECT_EQ(peak,
            expected_doppler_bin(sim.truth[0].radial_velocity,
                                 scene.waveform.wavelength(),
                                 scene.waveform.chirp_repetition_interval,
                                 num_chirps, true));

  const auto axis = velocity_axis_of<float>(bins);
  const double bin_width = scene.waveform.wavelength() / 2.0 /
                           scene.waveform.chirp_repetition_interval /
                           static_cast<double>(num_chirps);
  EXPECT_NEAR(axis[static_cast<size_t>(peak)],
              sim.truth[0].radial_velocity, bin_width + 1e-9);
}

}  // namespace
