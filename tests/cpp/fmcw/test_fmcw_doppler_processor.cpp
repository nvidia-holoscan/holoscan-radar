// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for FMCWDopplerProcessorOp (HSDK 5 Graph harness):
// Doppler-peak recovery, fftshift, Hann window, nfft interpolation,
// batch-axis handling, float/double agreement, conditional velocity_bins
// output, and error paths. Sticky [live] center_zero_doppler Graph
// mid-stream test is GTEST_SKIP'd per HSDK5_PORT.md §1.

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

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/bin_model.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/sim_reference.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::FftNorm;
using holoscan::radar::FMCWDopplerProcessorOp;
using holoscan::radar::FMCWDopplerProcessorParams;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::FMCWWaveformSpec;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::complex_dtype;
using holoscan::radar::kPi;
using holoscan::radar::kSpeedOfLight;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;
namespace sim = holoscan::radar::sim;

FMCWWaveformDescriptor test_waveform() {
  FMCWWaveformSpec spec;
  spec.carrier_frequency = 77e9;
  spec.bandwidth = 150e6;
  spec.chirp_duration = 20e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25e-6;
  return FMCWWaveformDescriptor(spec);
}

std::vector<std::complex<double>> tone_profiles(int64_t num_chirps,
                                                int64_t num_bins,
                                                int64_t range_bin,
                                                double doppler_hz,
                                                double pri) {
  std::vector<std::complex<double>> data(
      static_cast<size_t>(num_chirps * num_bins), {0.0, 0.0});
  for (int64_t k = 0; k < num_chirps; ++k) {
    const double phase =
        2.0 * kPi * doppler_hz * pri * static_cast<double>(k);
    data[static_cast<size_t>(k * num_bins + range_bin)] =
        std::complex<double>(std::cos(phase), std::sin(phase));
  }
  return data;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_input(
    const std::vector<std::complex<double>>& data,
    const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    cast[i] = std::complex<T>(static_cast<T>(data[i].real()),
                              static_cast<T>(data[i].imag()));
  }
  return to_device_tensor(cast, shape, complex_dtype<T>());
}

// Wiring lambda for 1-in / 1-out (rd_maps only, velocity_bins unwired).
template <typename Source, typename OpRef, typename Sink>
void wire_doppler_single(holoscan::Graph& graph, const Source& source,
                         OpRef& op, const Sink& sink) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->range_profiles, q);
  graph.add_flow(op->rd_maps, sink->in, q);
}

// Wiring lambda for 1-in / 2-out (rd_maps + velocity_bins).
template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_doppler_dual(holoscan::Graph& graph, const Source& source,
                       OpRef& op, const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->range_profiles, q);
  graph.add_flow(op->rd_maps, sink0->in, q);
  graph.add_flow(op->velocity_bins, sink1->in, q);
}

// Run rd_maps only (velocity_bins unwired). Returns single-frame capture.
// velocity_bins must still be device-bound even when unwired
// (IMPLICIT_DEVICE_ZERO_REJECTED).
template <typename T>
HostTensor run_rd_single(std::shared_ptr<holoscan::Tensor> input,
                         FMCWDopplerProcessorParams params = {}) {
  GraphRunResult result =
      run_single_io_operator_graph_result<FMCWDopplerProcessorOp<T>>(
          {std::move(input)}, std::move(params), "rd_maps",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            wire_doppler_single(graph, source, op, sink);
          },
          {"velocity_bins"});
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
  if (result.frames.size() != 1) {
    throw std::runtime_error(
        "expected 1 rd_maps frame, got " +
        std::to_string(result.frames.size()));
  }
  return std::move(result.frames.front());
}

// Run rd_maps + velocity_bins. Returns both frame captures.
template <typename T>
std::pair<HostTensor, HostTensor> run_rd_velocity_single(
    std::shared_ptr<holoscan::Tensor> input,
    FMCWDopplerProcessorParams params) {
  DualIOGraphRunResult result =
      run_single_in_dual_out_operator_graph_result<
          FMCWDopplerProcessorOp<T>>(
          {std::move(input)}, std::move(params), "rd_maps",
          "velocity_bins",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1) {
            wire_doppler_dual(graph, source, op, sink0, sink1);
          });
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
  if (result.frames0.size() != 1 || result.frames1.size() != 1) {
    throw std::runtime_error(
        "expected 1 frame per output, got rd=" +
        std::to_string(result.frames0.size()) + " vel=" +
        std::to_string(result.frames1.size()));
  }
  return {std::move(result.frames0.front()),
          std::move(result.frames1.front())};
}

// Run rd_maps-only variant returning GraphRunResult for error-path checks.
template <typename T>
GraphRunResult run_rd_result(std::shared_ptr<holoscan::Tensor> input,
                             FMCWDopplerProcessorParams params = {}) {
  return run_single_io_operator_graph_result<FMCWDopplerProcessorOp<T>>(
      {std::move(input)}, std::move(params), "rd_maps",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink) {
        wire_doppler_single(graph, source, op, sink);
      },
      {"velocity_bins"});
}

template <typename T>
double cell_mag(const HostTensor& tensor, int64_t batch, int64_t num_doppler,
                int64_t num_range_bins, int64_t doppler, int64_t range) {
  const std::complex<T>* data = tensor.as<std::complex<T>>();
  const std::complex<T> value =
      data[(batch * num_doppler + doppler) * num_range_bins + range];
  return std::abs(std::complex<double>(value.real(), value.imag()));
}

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

template <typename T>
std::vector<double> velocity_axis_of(const HostTensor& tensor) {
  const T* data = tensor.as<T>();
  std::vector<double> axis(static_cast<size_t>(tensor.size()));
  for (int64_t i = 0; i < tensor.size(); ++i) {
    axis[static_cast<size_t>(i)] = static_cast<double>(data[i]);
  }
  return axis;
}

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

using holoscan::radar::testing::expected_doppler_bin;
using holoscan::radar::testing::expected_range_bin;
using holoscan::radar::testing::expected_velocity_axis;

TEST(FMCWDopplerProcessorOpTest, VelocityPeakMatchesTruthCentered) {
  const int64_t num_chirps = 32;
  const int64_t num_bins = 16;
  const int64_t range_bin = 5;
  const double radial_velocity = 5.0;
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval();
  const double wavelength = waveform.wavelength();
  const double doppler_hz = -2.0 * radial_velocity / wavelength;

  FMCWDopplerProcessorParams params;
  params.waveform = waveform;
  const auto [rd_map, bins] = run_rd_velocity_single<float>(
      make_input<float>(
          tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
          {num_chirps, num_bins}),
      params);

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

  const double bin_width =
      wavelength / 2.0 / pri / static_cast<double>(num_chirps);
  EXPECT_NEAR(axis[static_cast<size_t>(peak)], radial_velocity,
              bin_width + 1e-9);
}

TEST(FMCWDopplerProcessorOpTest, VelocityAxisUncentered) {
  const int64_t num_chirps = 32;
  const int64_t num_bins = 8;
  const int64_t range_bin = 3;
  const double radial_velocity = -4.0;
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval();
  const double wavelength = waveform.wavelength();
  const double doppler_hz = -2.0 * radial_velocity / wavelength;

  FMCWDopplerProcessorParams params;
  params.waveform = waveform;
  params.center_zero_doppler = false;
  const auto [rd_map, bins] = run_rd_velocity_single<float>(
      make_input<float>(
          tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
          {num_chirps, num_bins}),
      params);

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
  EXPECT_DOUBLE_EQ(axis[0], 0.0);

  const double bin_width =
      wavelength / 2.0 / pri / static_cast<double>(num_chirps);
  EXPECT_NEAR(axis[static_cast<size_t>(peak)], radial_velocity,
              bin_width + 1e-9);
}

TEST(FMCWDopplerProcessorOpTest, CenterZeroDopplerPlacesDc) {
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const int64_t range_bin = 1;
  const auto dc_tone =
      tone_profiles(num_chirps, num_bins, range_bin, 0.0, 25e-6);

  {
    const HostTensor rd_map = run_rd_single<float>(
        make_input<float>(dc_tone, {num_chirps, num_bins}));
    const int64_t peak =
        doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, range_bin);
    EXPECT_EQ(peak, num_chirps / 2);
    EXPECT_NEAR(cell_mag<float>(rd_map, 0, num_chirps, num_bins, peak,
                                range_bin),
                static_cast<double>(num_chirps), 1e-3);
  }

  {
    FMCWDopplerProcessorParams params;
    params.center_zero_doppler = false;
    const HostTensor rd_map = run_rd_single<float>(
        make_input<float>(dc_tone, {num_chirps, num_bins}), params);
    EXPECT_EQ(doppler_argmax<float>(rd_map, 0, num_chirps, num_bins,
                                    range_bin),
              0);
  }
}

TEST(FMCWDopplerProcessorOpTest, OddChirpCountFftshiftMatchesNumpy) {
  const int64_t num_chirps = 31;
  const int64_t num_bins = 4;
  const int64_t range_bin = 2;
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval();
  const double wavelength = waveform.wavelength();
  const double doppler_hz = 3.0 / (pri * static_cast<double>(num_chirps));

  FMCWDopplerProcessorParams params;
  params.waveform = waveform;
  const auto [rd_map, bins] = run_rd_velocity_single<float>(
      make_input<float>(
          tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
          {num_chirps, num_bins}),
      params);

  const int64_t peak =
      doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, range_bin);
  EXPECT_EQ(peak, (3 + num_chirps / 2) % num_chirps);

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
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const int64_t range_bin = 1;
  const double pri = 25e-6;
  const double doppler_hz = 4.0 / (pri * static_cast<double>(num_chirps));
  const auto tone =
      tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri);

  auto peak_magnitude = [&](const WindowSpec& window) {
    FMCWDopplerProcessorParams params;
    params.doppler_window = window;
    params.center_zero_doppler = false;
    const HostTensor rd_map = run_rd_single<float>(
        make_input<float>(tone, {num_chirps, num_bins}), params);
    const int64_t peak =
        doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, range_bin);
    EXPECT_EQ(peak, 4);
    return cell_mag<float>(rd_map, 0, num_chirps, num_bins, peak, range_bin);
  };

  WindowSpec hann;
  hann.type = WindowType::kHann;
  const double rect_peak = peak_magnitude(WindowSpec{});
  const double hann_peak = peak_magnitude(hann);

  EXPECT_NEAR(rect_peak, static_cast<double>(num_chirps), 1e-3);
  EXPECT_NEAR(hann_peak, rect_peak, 0.03 * rect_peak);
}

TEST(FMCWDopplerProcessorOpTest, HannWindowLowersDopplerSidelobes) {
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const int64_t range_bin = 2;
  const double pri = 25e-6;
  const double doppler_hz = 4.5 / (pri * static_cast<double>(num_chirps));
  const auto tone =
      tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri);

  auto max_sidelobe = [&](const WindowSpec& window) {
    FMCWDopplerProcessorParams params;
    params.doppler_window = window;
    params.center_zero_doppler = false;
    const HostTensor rd_map = run_rd_single<float>(
        make_input<float>(tone, {num_chirps, num_bins}), params);
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
  const int64_t num_chirps = 32;
  const int64_t nfft = 64;
  const int64_t num_bins = 8;
  const int64_t range_bin = 6;
  const double radial_velocity = 7.0;
  const auto waveform = test_waveform();
  const double pri = waveform.chirp_repetition_interval();
  const double wavelength = waveform.wavelength();
  const double doppler_hz = -2.0 * radial_velocity / wavelength;

  FMCWDopplerProcessorParams params;
  params.waveform = waveform;
  params.nfft = nfft;
  const auto [rd_map, bins] = run_rd_velocity_single<float>(
      make_input<float>(
          tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
          {num_chirps, num_bins}),
      params);

  ASSERT_EQ(rd_map.shape, (std::vector<int64_t>{nfft, num_bins}));

  const auto axis = velocity_axis_of<float>(bins);
  ASSERT_EQ(static_cast<int64_t>(axis.size()), nfft);
  const auto expected = expected_velocity_axis(nfft, wavelength, pri, true);
  for (int64_t i = 0; i < nfft; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(axis[static_cast<size_t>(i)]),
                    static_cast<float>(expected[static_cast<size_t>(i)]))
        << "velocity_bins mismatch at bin " << i;
  }

  const double original_bin_width =
      wavelength / 2.0 / pri / static_cast<double>(num_chirps);
  EXPECT_NEAR(std::abs(axis[1] - axis[0]), original_bin_width / 2.0,
              1e-5 * original_bin_width);
  const int64_t peak =
      doppler_argmax<float>(rd_map, 0, nfft, num_bins, range_bin);
  EXPECT_NEAR(axis[static_cast<size_t>(peak)], radial_velocity,
              original_bin_width + 1e-9);
}

TEST(FMCWDopplerProcessorOpTest, NamedRxAxisProcessedIndependently) {
  // Rank-3 leading dim is the named N_rx axis (SIMO), not opaque batch.
  // default_test_extents() sets max_rx so the leading axis is admitted.
  const int64_t num_rx = 2;
  const int64_t num_chirps = 32;
  const int64_t num_bins = 8;
  const double pri = 25e-6;
  const double f0 = 3.0 / (pri * static_cast<double>(num_chirps));
  const double f1 = 10.0 / (pri * static_cast<double>(num_chirps));
  auto cube = tone_profiles(num_chirps, num_bins, 2, f0, pri);
  const auto slice1 = tone_profiles(num_chirps, num_bins, 6, f1, pri);
  cube.insert(cube.end(), slice1.begin(), slice1.end());

  FMCWDopplerProcessorParams params;
  params.center_zero_doppler = false;
  const HostTensor rd_map = run_rd_single<float>(
      make_input<float>(cube, {num_rx, num_chirps, num_bins}), params);

  ASSERT_EQ(rd_map.shape,
            (std::vector<int64_t>{num_rx, num_chirps, num_bins}));
  EXPECT_EQ(doppler_argmax<float>(rd_map, 0, num_chirps, num_bins, 2), 3);
  EXPECT_EQ(doppler_argmax<float>(rd_map, 1, num_chirps, num_bins, 6), 10);
  EXPECT_LT(cell_mag<float>(rd_map, 0, num_chirps, num_bins, 10, 6), 1e-3);
  EXPECT_LT(cell_mag<float>(rd_map, 1, num_chirps, num_bins, 3, 2), 1e-3);
}

TEST(FMCWDopplerProcessorOpTest,
     LiveOverrideCenterZeroDopplerMidStream) {
  // [EA HSDK5 ONLY] Graph mid-stream sticky override harness deferred —
  // HSDK5_PORT.md §1 (EA device/partition binding issues).
  GTEST_SKIP() << "HSDK 5 EA: live center_zero_doppler override harness "
                  "deferred (HSDK5_PORT.md §1)";
}

TEST(FMCWDopplerProcessorOpTest, LiveOverrideDopplerWindow) {
  // [EA HSDK5 ONLY] doppler_window is Params-only (HSDK5_PORT §9);
  // Graph override deferred until §9 is closed.
  GTEST_SKIP() << "HSDK 5 EA: doppler_window is Params-only "
                  "(HSDK5_PORT.md §9)";
}

TEST(FMCWDopplerProcessorOpTest, VelocityBinsOnlyEmittedWithWaveform) {
  const int64_t num_chirps = 32;
  const int64_t num_bins = 4;
  const auto tone = tone_profiles(num_chirps, num_bins, 1, 0.0, 25e-6);

  // Without a waveform: rd_maps flows, velocity_bins never emits.
  // Use single-output harness (velocity_bins unwired).
  {
    const HostTensor rd_map = run_rd_single<float>(
        make_input<float>(tone, {num_chirps, num_bins}));
    ASSERT_EQ(rd_map.shape,
              (std::vector<int64_t>{num_chirps, num_bins}));
  }

  // With a waveform: both outputs emitted.
  {
    FMCWDopplerProcessorParams params;
    params.waveform = test_waveform();
    const auto [rd_map, bins] = run_rd_velocity_single<float>(
        make_input<float>(tone, {num_chirps, num_bins}), params);
    ASSERT_EQ(rd_map.shape,
              (std::vector<int64_t>{num_chirps, num_bins}));
    EXPECT_EQ(bins.shape, (std::vector<int64_t>{num_chirps}));
  }
}

TEST(FMCWDopplerProcessorOpTest, FloatAndDoubleAgree) {
  const int64_t num_chirps = 32;
  const int64_t num_bins = 8;
  const double pri = 25e-6;
  const double doppler_hz = 6.3 / (pri * static_cast<double>(num_chirps));
  const auto tone = tone_profiles(num_chirps, num_bins, 4, doppler_hz, pri);

  auto run = [&](auto precision_tag) {
    using T = decltype(precision_tag);
    return run_rd_single<T>(make_input<T>(tone, {num_chirps, num_bins}));
  };
  const HostTensor rd_float = run(float{});
  const HostTensor rd_double = run(double{});

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

TEST(FMCWDopplerProcessorOpTest, RejectsRealInput) {
  const int64_t num_chirps = 8;
  const int64_t num_bins = 4;
  const std::vector<float> real_cube(
      static_cast<size_t>(num_chirps * num_bins), 1.0f);
  GraphRunResult result = run_rd_result<float>(
      to_device_tensor(real_cube, {num_chirps, num_bins},
                       real_dtype<float>()));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(FMCWDopplerProcessorOpTest, NfftSmallerThanChirpCountRejected) {
  const auto tone = tone_profiles(32, 4, 1, 0.0, 25e-6);
  FMCWDopplerProcessorParams params;
  params.nfft = 16;
  GraphRunResult result = run_rd_result<float>(
      make_input<float>(tone, {32, 4}), params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(FMCWDopplerProcessorOpTest, SimulatorTargetMapsToTruthVelocity) {
  sim::SimScene scene;
  scene.num_chirps = 32;
  scene.samples_per_chirp = 64;
  scene.waveform.carrier_frequency = 77e9;
  scene.waveform.bandwidth = 150e6;
  scene.waveform.chirp_duration = 20e-6;
  scene.waveform.sample_rate = 12.8e6;
  scene.waveform.chirp_repetition_interval = 25e-6;
  sim::PointTarget target;
  target.position = {20.0, 0.0, 0.0};
  target.velocity = {6.0, 0.0, 0.0};
  scene.targets.push_back(target);
  const sim::SimResult sim_result = sim::simulate(scene);
  ASSERT_EQ(sim_result.truth.size(), 1u);

  const int64_t num_chirps = scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  std::vector<std::complex<double>> profiles(
      static_cast<size_t>(num_chirps * samples));
  for (int64_t k = 0; k < num_chirps; ++k) {
    for (int64_t b = 0; b < samples; ++b) {
      std::complex<double> acc{0.0, 0.0};
      for (int64_t s = 0; s < samples; ++s) {
        const double phase = -2.0 * kPi * static_cast<double>(b) *
                             static_cast<double>(s) /
                             static_cast<double>(samples);
        acc += sim_result.cube[static_cast<size_t>(k * samples + s)] *
               std::complex<double>(std::cos(phase), std::sin(phase));
      }
      profiles[static_cast<size_t>(k * samples + b)] = acc;
    }
  }

  FMCWWaveformSpec spec;
  spec.carrier_frequency = scene.waveform.carrier_frequency;
  spec.bandwidth = scene.waveform.bandwidth;
  spec.chirp_duration = scene.waveform.chirp_duration;
  spec.sample_rate = scene.waveform.sample_rate;
  spec.chirp_repetition_interval = scene.waveform.chirp_repetition_interval;
  const FMCWWaveformDescriptor waveform(spec);

  FMCWDopplerProcessorParams params;
  params.waveform = waveform;
  const auto [rd_map, bins] = run_rd_velocity_single<float>(
      make_input<float>(profiles, {num_chirps, samples}), params);

  const int64_t range_bin = expected_range_bin(
      sim_result.truth[0].range, scene.waveform.slope(),
      scene.waveform.sample_rate, samples, kSpeedOfLight);
  const int64_t peak =
      doppler_argmax<float>(rd_map, 0, num_chirps, samples, range_bin);
  EXPECT_EQ(peak,
            expected_doppler_bin(sim_result.truth[0].radial_velocity,
                                 scene.waveform.wavelength(),
                                 scene.waveform.chirp_repetition_interval,
                                 num_chirps, true));

  const auto axis = velocity_axis_of<float>(bins);
  const double bin_width = scene.waveform.wavelength() / 2.0 /
                           scene.waveform.chirp_repetition_interval /
                           static_cast<double>(num_chirps);
  EXPECT_NEAR(axis[static_cast<size_t>(peak)],
              sim_result.truth[0].radial_velocity, bin_width + 1e-9);
}

}  // namespace
