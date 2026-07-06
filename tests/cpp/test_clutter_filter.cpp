// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for ClutterFilterOp:
// zero-Doppler nulling with moving-target preservation, the delay
// cancellers' binomial high-pass response, leading-batch-axis handling,
// method overrides, error paths, and float/double agreement.
// Driven through the SDK-native holoscan::test::OperatorTestHarness
// (utils/sdk_harness.hpp).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/point_target_sim.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ClutterFilterMethod;
using holoscan::radar::ClutterFilterOp;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;
namespace sim = holoscan::radar::testing;

constexpr double kPi = 3.14159265358979323846;
constexpr int64_t kChirps = 16;
constexpr int64_t kBins = 64;

// Per-chirp complex exponential x[k, r] = a_r * exp(j*2*pi*f*k) with a
// per-bin amplitude a_r = 1 + r/nb (so range bins are distinguishable).
// `f` is the normalized Doppler f_d*PRI in cycles/chirp.
template <typename T>
std::vector<std::complex<T>> doppler_tone(int64_t num_chirps,
                                          int64_t num_bins, double f) {
  std::vector<std::complex<T>> cube(
      static_cast<size_t>(num_chirps * num_bins));
  for (int64_t k = 0; k < num_chirps; ++k) {
    const double phase = 2.0 * kPi * f * static_cast<double>(k);
    const std::complex<double> rot{std::cos(phase), std::sin(phase)};
    for (int64_t r = 0; r < num_bins; ++r) {
      const double amp =
          1.0 + static_cast<double>(r) / static_cast<double>(num_bins);
      const std::complex<double> v = amp * rot;
      cube[static_cast<size_t>(k * num_bins + r)] =
          std::complex<T>(static_cast<T>(v.real()),
                          static_cast<T>(v.imag()));
    }
  }
  return cube;
}

// Run the operator once over a prepared frame through the SDK harness and
// return the captured host output.
template <typename T, typename HarnessT>
HostTensor run_harness_single(const std::shared_ptr<HarnessT>& harness,
                              holoscan::TensorMap frame) {
  TensorMapCapture capture;
  harness->template add_input_port<holoscan::TensorMap>("range_profiles",
                                                        {frame});
  harness->template add_output_port<holoscan::TensorMap>(
      "range_profiles", capture.validators());
  harness->run_test();
  if (capture.frames.size() != 1) {
    throw std::runtime_error("expected exactly one captured output frame");
  }
  return std::move(capture.frames.front());
}

// Run the operator once over `host` (complex<T>, full-rank `shape`) and
// return the host copy of the output. `method` unset → the parameter
// default (mean_subtraction).
template <typename T>
HostTensor run_single(const std::vector<std::complex<T>>& host,
                      const std::vector<int64_t>& shape,
                      std::optional<ClutterFilterMethod> method,
                      const std::string& key = "range_profiles") {
  auto frame =
      single_map(to_device_tensor(host, shape, complex_dtype<T>()), key);
  if (method.has_value()) {
    return run_harness_single<T>(
        create_operator_test<ClutterFilterOp<T>>(
            holoscan::Arg("method") = *method),
        std::move(frame));
  }
  return run_harness_single<T>(create_operator_test<ClutterFilterOp<T>>(),
                               std::move(frame));
}

template <typename T>
double max_abs(const HostTensor& tensor) {
  const auto* data = tensor.as<std::complex<T>>();
  double peak = 0.0;
  for (int64_t i = 0; i < tensor.size(); ++i) {
    peak = std::max(peak, static_cast<double>(std::abs(data[i])));
  }
  return peak;
}

template <typename T>
double tensor_energy(const HostTensor& tensor) {
  const auto* data = tensor.as<std::complex<T>>();
  double energy = 0.0;
  for (int64_t i = 0; i < tensor.size(); ++i) {
    energy += static_cast<double>(std::norm(data[i]));
  }
  return energy;
}

double cube_energy(const std::vector<std::complex<double>>& cube) {
  double energy = 0.0;
  for (const auto& v : cube) {
    energy += std::norm(v);
  }
  return energy;
}

// Moving target whose normalized Doppler f_d*PRI equals `f` (an integer
// number of cycles across kChirps chirps → its slow-time mean is ~0).
sim::PointTarget moving_target(const sim::SimWaveform& waveform, double f) {
  const double f_d = f / waveform.chirp_repetition_interval;
  const double v_r = -f_d * waveform.wavelength() / 2.0;
  sim::PointTarget target;
  target.position = {45.0, 0.0, 0.0};
  target.velocity = {v_r, 0.0, 0.0};
  return target;
}

// --- mean_subtraction: zero-Doppler clutter nulling ----------------------

TEST(ClutterFilterOpTest, MeanSubtractionNullsStaticTarget) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  sim::SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kBins;
  scene.targets = {sim::PointTarget{{30.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};

  const sim::SimResult result = sim::simulate(scene);
  std::vector<std::complex<float>> host(result.cube.size());
  double peak_in = 0.0;
  for (size_t i = 0; i < result.cube.size(); ++i) {
    host[i] = std::complex<float>(result.cube[i]);
    peak_in = std::max(peak_in, std::abs(result.cube[i]));
  }

  // Default method (mean_subtraction): also exercises the parameter default.
  const HostTensor out =
      run_single<float>(host, result.shape, std::nullopt);
  ASSERT_EQ(out.shape, result.shape);  // no chirp loss
  // A zero-velocity target is identical on every chirp (DC across chirps):
  // mean subtraction nulls every cell.
  EXPECT_LT(max_abs<float>(out), 1e-4 * peak_in);
}

TEST(ClutterFilterOpTest, MeanSubtractionPreservesMovingTarget) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  sim::SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kBins;
  // 4 full Doppler cycles across 16 chirps → the moving return has ~zero
  // slow-time mean and should pass through nearly untouched.
  const sim::PointTarget moving = moving_target(scene.waveform, 4.0 / 16.0);
  const sim::PointTarget stationary{{30.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  scene.targets = {moving};
  const sim::SimResult moving_only = sim::simulate(scene);
  scene.targets = {stationary, moving};
  const sim::SimResult both = sim::simulate(scene);

  std::vector<std::complex<float>> host(both.cube.size());
  for (size_t i = 0; i < both.cube.size(); ++i) {
    host[i] = std::complex<float>(both.cube[i]);
  }

  const HostTensor out = run_single<float>(
      host, both.shape, ClutterFilterMethod::kMeanSubtraction);
  ASSERT_EQ(out.shape, both.shape);
  // Static clutter removed, moving-target slow-time energy preserved
  // within a few percent.
  const double e_out = tensor_energy<float>(out);
  const double e_moving = cube_energy(moving_only.cube);
  EXPECT_NEAR(e_out / e_moving, 1.0, 0.05);
}

// --- delay cancellers: shape and binomial high-pass response --------------

TEST(ClutterFilterOpTest, SingleDelayShapeAndResponse) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const double f = 0.19;  // normalized Doppler, cycles/chirp
  const auto tone = doppler_tone<float>(kChirps, kBins, f);
  const HostTensor out = run_single<float>(
      tone, {kChirps, kBins}, ClutterFilterMethod::kSingleDelay);
  ASSERT_EQ(out.shape, (std::vector<int64_t>{kChirps - 1, kBins}));

  // out[k] = in[k]*(e^{j*2*pi*f} - 1) → per-cell gain |1 - e^{j*2*pi*f}|.
  const double expected_gain =
      std::abs(1.0 - std::polar(1.0, 2.0 * kPi * f));
  const auto* data = out.as<std::complex<float>>();
  for (int64_t k = 0; k < kChirps - 1; ++k) {
    for (int64_t r = 0; r < kBins; ++r) {
      const double amp = std::abs(tone[static_cast<size_t>(k * kBins + r)]);
      const double gain =
          std::abs(data[k * kBins + r]) / amp;
      ASSERT_NEAR(gain, expected_gain, 1e-4 * expected_gain)
          << "cell (" << k << ", " << r << ")";
    }
  }
}

TEST(ClutterFilterOpTest, DoubleDelayShapeAndResponse) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const double f = 0.19;
  const auto tone = doppler_tone<float>(kChirps, kBins, f);
  const HostTensor out = run_single<float>(
      tone, {kChirps, kBins}, ClutterFilterMethod::kDoubleDelay);
  ASSERT_EQ(out.shape, (std::vector<int64_t>{kChirps - 2, kBins}));

  // out[k] = in[k]*(1 - e^{j*2*pi*f})^2 → gain |1 - e^{j*2*pi*f}|^2.
  const double base = std::abs(1.0 - std::polar(1.0, 2.0 * kPi * f));
  const double expected_gain = base * base;
  const auto* data = out.as<std::complex<float>>();
  for (int64_t k = 0; k < kChirps - 2; ++k) {
    for (int64_t r = 0; r < kBins; ++r) {
      const double amp = std::abs(tone[static_cast<size_t>(k * kBins + r)]);
      const double gain = std::abs(data[k * kBins + r]) / amp;
      ASSERT_NEAR(gain, expected_gain, 1e-4 * expected_gain)
          << "cell (" << k << ", " << r << ")";
    }
  }
}

TEST(ClutterFilterOpTest, DelayCancellersNotchDC) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto dc = doppler_tone<float>(kChirps, kBins, 0.0);
  for (const ClutterFilterMethod method :
       {ClutterFilterMethod::kSingleDelay,
        ClutterFilterMethod::kDoubleDelay}) {
    const HostTensor out =
        run_single<float>(dc, {kChirps, kBins}, method);
    EXPECT_LT(max_abs<float>(out), 1e-5)
        << "method " << holoscan::radar::to_string(method);
  }
}

// --- leading batch axes ----------------------------------------------------

TEST(ClutterFilterOpTest, LeadingBatchAxesPreservedAndIndependent) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const int64_t batch = 2;
  const int64_t nc = 8;
  const int64_t nb = 16;
  // Different data per batch slice (different tones + per-slice offsets)
  // so cross-slice leakage in the mean would be visible.
  const auto slice0 = doppler_tone<float>(nc, nb, 0.25);
  auto slice1 = doppler_tone<float>(nc, nb, 0.125);
  for (auto& v : slice1) {
    v += std::complex<float>(2.0f, -1.0f);
  }
  std::vector<std::complex<float>> host;
  host.insert(host.end(), slice0.begin(), slice0.end());
  host.insert(host.end(), slice1.begin(), slice1.end());

  // Non-default map key: single-entry maps are accepted regardless of key.
  const HostTensor out =
      run_single<float>(host, {batch, nc, nb},
                        ClutterFilterMethod::kMeanSubtraction, "adc");
  ASSERT_EQ(out.shape, (std::vector<int64_t>{batch, nc, nb}));

  // Host reference: per-(slice, bin) slow-time mean subtraction.
  const auto* data = out.as<std::complex<float>>();
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t r = 0; r < nb; ++r) {
      std::complex<double> mean{0.0, 0.0};
      for (int64_t k = 0; k < nc; ++k) {
        mean += std::complex<double>(
            host[static_cast<size_t>((b * nc + k) * nb + r)]);
      }
      mean /= static_cast<double>(nc);
      for (int64_t k = 0; k < nc; ++k) {
        const std::complex<double> expected =
            std::complex<double>(
                host[static_cast<size_t>((b * nc + k) * nb + r)]) -
            mean;
        const std::complex<double> got(
            data[(b * nc + k) * nb + r]);
        ASSERT_NEAR(std::abs(got - expected), 0.0, 1e-5)
            << "cell (" << b << ", " << k << ", " << r << ")";
      }
    }
  }

  // Chirp loss applies per slice for the delay cancellers.
  const HostTensor delayed = run_single<float>(
      host, {batch, nc, nb}, ClutterFilterMethod::kSingleDelay);
  EXPECT_EQ(delayed.shape, (std::vector<int64_t>{batch, nc - 1, nb}));
}

// --- method override ------------------------------------------------

// Data ticks outnumber the override values, so the trailing tick exercises
// the sticky-override rule (an override remains effective
// until the next override).
class LiveOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> frames;
  std::vector<ClutterFilterMethod> overrides;
  TensorMapCapture* capture = nullptr;

  void compose() override {
    auto op = make_operator<ClutterFilterOp<float>>("op_under_test");
    wire_data_source(this, op, "range_profiles", frames);
    wire_leading_override<ClutterFilterMethod>(this, op, "method",
                                               overrides);
    wire_capture_sink(this, op, "range_profiles", capture);
  }
};

TEST(ClutterFilterOpTest, LiveMethodOverrideMidStreamAndSticky) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto tone = doppler_tone<float>(kChirps, kBins, 0.19);
  auto frame = [&]() {
    return single_map(to_device_tensor(tone, {kChirps, kBins},
                                       complex_dtype<float>()),
                      "range_profiles");
  };

  TensorMapCapture capture;
  auto app = holoscan::make_application<LiveOverrideApp>();
  app->frames = {frame(), frame(), frame()};
  // Tick 0: override to single_delay. Tick 1: override back to
  // mean_subtraction. Tick 2: no message — the tick-1 override is sticky.
  app->overrides = {ClutterFilterMethod::kSingleDelay,
                    ClutterFilterMethod::kMeanSubtraction};
  app->capture = &capture;
  app->run();

  ASSERT_EQ(capture.frames.size(), 3u);
  EXPECT_EQ(capture.frames[0].shape,
            (std::vector<int64_t>{kChirps - 1, kBins}));
  EXPECT_EQ(capture.frames[1].shape,
            (std::vector<int64_t>{kChirps, kBins}));
  EXPECT_EQ(capture.frames[2].shape,
            (std::vector<int64_t>{kChirps, kBins}));
}

// --- error paths -------------------------------------------------------------

TEST(ClutterFilterOpTest, RejectsMethodNone) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto tone = doppler_tone<float>(kChirps, kBins, 0.19);
  try {
    run_single<float>(tone, {kChirps, kBins}, ClutterFilterMethod::kNone);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("none"), std::string::npos)
        << e.what();
  }
}

TEST(ClutterFilterOpTest, RejectsRealInput) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const std::vector<float> real_cube(
      static_cast<size_t>(kChirps * kBins), 1.0f);
  auto frame = single_map(
      to_device_tensor(real_cube, {kChirps, kBins}, real_dtype<float>()),
      "range_profiles");
  try {
    run_harness_single<float>(
        create_operator_test<ClutterFilterOp<float>>(), std::move(frame));
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("range_profiles"),
              std::string::npos)
        << e.what();
  }
}

TEST(ClutterFilterOpTest, RejectsTooFewChirps) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto one_chirp = doppler_tone<float>(1, kBins, 0.0);
  EXPECT_THROW(run_single<float>(one_chirp, {1, kBins},
                                 ClutterFilterMethod::kSingleDelay),
               std::invalid_argument);
  const auto two_chirps = doppler_tone<float>(2, kBins, 0.0);
  EXPECT_THROW(run_single<float>(two_chirps, {2, kBins},
                                 ClutterFilterMethod::kDoubleDelay),
               std::invalid_argument);
}

TEST(ClutterFilterOpTest, RejectsMultiEntryTensorMap) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto tone = doppler_tone<float>(kChirps, kBins, 0.19);
  holoscan::TensorMap map;
  map.insert({"a", to_device_tensor(tone, {kChirps, kBins},
                                    complex_dtype<float>())});
  map.insert({"b", to_device_tensor(tone, {kChirps, kBins},
                                    complex_dtype<float>())});
  EXPECT_THROW(run_harness_single<float>(
                   create_operator_test<ClutterFilterOp<float>>(), map),
               std::invalid_argument);
}

// --- precision agreement -----------------------------------------------------

TEST(ClutterFilterOpTest, FloatAndDoubleAgree) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  const auto tone_d = doppler_tone<double>(kChirps, kBins, 0.15);
  std::vector<std::complex<float>> tone_f(tone_d.size());
  for (size_t i = 0; i < tone_d.size(); ++i) {
    tone_f[i] = std::complex<float>(tone_d[i]);
  }

  for (const ClutterFilterMethod method :
       {ClutterFilterMethod::kMeanSubtraction,
        ClutterFilterMethod::kSingleDelay,
        ClutterFilterMethod::kDoubleDelay}) {
    const HostTensor out_f =
        run_single<float>(tone_f, {kChirps, kBins}, method);
    const HostTensor out_d =
        run_single<double>(tone_d, {kChirps, kBins}, method);
    ASSERT_EQ(out_f.shape, out_d.shape);
    const auto* f_data = out_f.as<std::complex<float>>();
    const auto* d_data = out_d.as<std::complex<double>>();
    const double scale = std::max(max_abs<double>(out_d), 1.0);
    for (int64_t i = 0; i < out_d.size(); ++i) {
      ASSERT_NEAR(std::abs(std::complex<double>(f_data[i]) - d_data[i]),
                  0.0, 1e-5 * scale)
          << "method " << holoscan::radar::to_string(method) << " index "
          << i;
    }
  }
}

}  // namespace
