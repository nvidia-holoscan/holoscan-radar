// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for ClutterFilterOp (HSDK 5 Graph harness):
// zero-Doppler nulling with moving-target preservation, the delay
// cancellers' binomial high-pass response, named RX-axis handling,
// per-tick method overrides, error paths, and float/double agreement.

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

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/sim_reference.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ClutterFilterMethod;
using holoscan::radar::ClutterFilterOp;
using holoscan::radar::ClutterFilterParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_operator_graph;
using holoscan::radar::testing::run_operator_graph_result;
using holoscan::radar::testing::run_operator_once;
using holoscan::radar::testing::to_device_tensor;
namespace sim = holoscan::radar::sim;

using holoscan::radar::kPi;
constexpr int64_t kChirps = 16;
constexpr int64_t kBins = 64;

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

template <typename T>
HostTensor run_single(const std::vector<std::complex<T>>& host,
                      const std::vector<int64_t>& shape,
                      std::optional<ClutterFilterMethod> method) {
  ClutterFilterParams params{};
  if (method.has_value()) {
    params.method = *method;
  }
  return run_operator_once<ClutterFilterOp<T>>(
      to_device_tensor(host, shape, complex_dtype<T>()), params);
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

sim::PointTarget moving_target(const sim::SimWaveform& waveform, double f) {
  const double f_d = f / waveform.chirp_repetition_interval;
  const double v_r = -f_d * waveform.wavelength() / 2.0;
  sim::PointTarget target;
  target.position = {45.0, 0.0, 0.0};
  target.velocity = {v_r, 0.0, 0.0};
  return target;
}

TEST(ClutterFilterOpTest, MeanSubtractionNullsStaticTarget) {
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

  const HostTensor out = run_single<float>(host, result.shape, std::nullopt);
  ASSERT_EQ(out.shape, result.shape);
  EXPECT_LT(max_abs<float>(out), 1e-4 * peak_in);
}

TEST(ClutterFilterOpTest, MeanSubtractionPreservesMovingTarget) {
  sim::SimScene scene;
  scene.num_chirps = kChirps;
  scene.samples_per_chirp = kBins;
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
  const double e_out = tensor_energy<float>(out);
  const double e_moving = cube_energy(moving_only.cube);
  EXPECT_NEAR(e_out / e_moving, 1.0, 0.05);
}

TEST(ClutterFilterOpTest, SingleDelayShapeAndResponse) {
  const double f = 0.19;
  const auto tone = doppler_tone<float>(kChirps, kBins, f);
  const HostTensor out = run_single<float>(
      tone, {kChirps, kBins}, ClutterFilterMethod::kSingleDelay);
  ASSERT_EQ(out.shape, (std::vector<int64_t>{kChirps - 1, kBins}));

  const double expected_gain =
      std::abs(1.0 - std::polar(1.0, 2.0 * kPi * f));
  const auto* data = out.as<std::complex<float>>();
  for (int64_t k = 0; k < kChirps - 1; ++k) {
    for (int64_t r = 0; r < kBins; ++r) {
      const double amp = std::abs(tone[static_cast<size_t>(k * kBins + r)]);
      const double gain = std::abs(data[k * kBins + r]) / amp;
      ASSERT_NEAR(gain, expected_gain, 1e-4 * expected_gain)
          << "cell (" << k << ", " << r << ")";
    }
  }
}

TEST(ClutterFilterOpTest, MethodConfigurableFromParams) {
  // Authoring Params (set_params) must configure method identically to the
  // native Arg path used under HSDK 4.x.
  const auto tone = doppler_tone<float>(kChirps, kBins, 0.19);
  const HostTensor out = run_single<float>(
      tone, {kChirps, kBins}, ClutterFilterMethod::kSingleDelay);
  ASSERT_EQ(out.shape, (std::vector<int64_t>{kChirps - 1, kBins}));
}

TEST(ClutterFilterOpTest, DoubleDelayShapeAndResponse) {
  const double f = 0.19;
  const auto tone = doppler_tone<float>(kChirps, kBins, f);
  const HostTensor out = run_single<float>(
      tone, {kChirps, kBins}, ClutterFilterMethod::kDoubleDelay);
  ASSERT_EQ(out.shape, (std::vector<int64_t>{kChirps - 2, kBins}));

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
  const auto dc = doppler_tone<float>(kChirps, kBins, 0.0);
  for (const ClutterFilterMethod method :
       {ClutterFilterMethod::kSingleDelay,
        ClutterFilterMethod::kDoubleDelay}) {
    const HostTensor out = run_single<float>(dc, {kChirps, kBins}, method);
    EXPECT_LT(max_abs<float>(out), 1e-5)
        << "method " << holoscan::radar::to_string(method);
  }
}

TEST(ClutterFilterOpTest, NamedRxAxisPreservedAndIndependent) {
  // Rank-3 leading dim is the named N_rx axis (SIMO), not opaque batch.
  const int64_t num_rx = 2;
  const int64_t nc = 8;
  const int64_t nb = 16;
  const auto slice0 = doppler_tone<float>(nc, nb, 0.25);
  auto slice1 = doppler_tone<float>(nc, nb, 0.125);
  for (auto& v : slice1) {
    v += std::complex<float>(2.0f, -1.0f);
  }
  std::vector<std::complex<float>> host;
  host.insert(host.end(), slice0.begin(), slice0.end());
  host.insert(host.end(), slice1.begin(), slice1.end());

  const HostTensor out = run_single<float>(
      host, {num_rx, nc, nb}, ClutterFilterMethod::kMeanSubtraction);
  ASSERT_EQ(out.shape, (std::vector<int64_t>{num_rx, nc, nb}));

  const auto* data = out.as<std::complex<float>>();
  for (int64_t b = 0; b < num_rx; ++b) {
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
        const std::complex<double> got(data[(b * nc + k) * nb + r]);
        ASSERT_NEAR(std::abs(got - expected), 0.0, 1e-5)
            << "cell (" << b << ", " << k << ", " << r << ")";
      }
    }
  }

  const HostTensor delayed = run_single<float>(
      host, {num_rx, nc, nb}, ClutterFilterMethod::kSingleDelay);
  EXPECT_EQ(delayed.shape, (std::vector<int64_t>{num_rx, nc - 1, nb}));
}

TEST(ClutterFilterOpTest, RejectsOpaqueLeadingBatchWhenExtentsAreSiso) {
  holoscan::radar::MaxTensorExtents extents;
  extents.max_chirps = kChirps;
  extents.max_range_bins = kBins;
  // No max_rx / max_tx → leading rank must be 0.
  const auto host = doppler_tone<float>(kChirps, kBins, 0.0);
  std::vector<std::complex<float>> stacked = host;
  stacked.insert(stacked.end(), host.begin(), host.end());
  const GraphRunResult result =
      run_operator_graph_result<ClutterFilterOp<float>>(
          {to_device_tensor(stacked,
                            std::vector<int64_t>{2, kChirps, kBins},
                            complex_dtype<float>())},
          ClutterFilterParams{},
          extents);
  EXPECT_FALSE(result.diagnostics.empty());
}

// The method decides how many chirps survive, so the output's chirp extent is
// the observable: mean_subtraction keeps all of them, single_delay drops one,
// double_delay drops two.
GraphRunResult run_method_override(
    std::vector<std::optional<ClutterFilterMethod>> overrides) {
  const auto tone = doppler_tone<float>(kChirps, kBins, 0.19);
  std::vector<std::shared_ptr<holoscan::Tensor>> cubes;
  cubes.reserve(overrides.size());
  for (std::size_t tick = 0; tick < overrides.size(); ++tick) {
    cubes.push_back(to_device_tensor(
        tone, std::vector<int64_t>{kChirps, kBins}, complex_dtype<float>()));
  }
  ClutterFilterParams params{};
  params.method = ClutterFilterMethod::kMeanSubtraction;
  return holoscan::radar::testing::
      run_exact_override_single_io_operator_graph_result<
          ClutterFilterOp<float>, ClutterFilterMethod>(
          std::move(cubes), std::move(overrides), params,
          "range_profiles_out",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out, op->range_profiles, q);
            graph.add_flow(source->override_out, op->method, q);
            graph.add_flow(op->range_profiles_out, sink->in, q);
          });
}

TEST(ClutterFilterOpTest, LiveMethodOverrideAppliesForOneTick) {
  const GraphRunResult out = run_method_override(
      {std::nullopt, ClutterFilterMethod::kSingleDelay, std::nullopt});

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 3U);
  // Params seed the first tick, the override takes effect on its own tick,
  // and the third tick is back on Params once the control plane goes quiet.
  EXPECT_EQ(out.frames[0].shape, (std::vector<int64_t>{kChirps, kBins}));
  EXPECT_EQ(out.frames[1].shape, (std::vector<int64_t>{kChirps - 1, kBins}));
  EXPECT_EQ(out.frames[2].shape, (std::vector<int64_t>{kChirps, kBins}));
}

TEST(ClutterFilterOpTest, LiveMethodRevertsToParamsAfterMiss) {
  const GraphRunResult out = run_method_override(
      {ClutterFilterMethod::kDoubleDelay, std::nullopt});

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2U);
  EXPECT_EQ(out.frames[0].shape, (std::vector<int64_t>{kChirps - 2, kBins}));
  EXPECT_EQ(out.frames[1].shape, (std::vector<int64_t>{kChirps, kBins}));
}

TEST(ClutterFilterOpTest, LiveMethodNonePassesThroughForOneTick) {
  const auto tone = doppler_tone<float>(kChirps, kBins, 0.19);
  const GraphRunResult out =
      run_method_override({ClutterFilterMethod::kNone});

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 1U);
  const auto* actual = out.frames[0].as<std::complex<float>>();
  for (std::size_t index = 0; index < tone.size(); ++index) {
    EXPECT_EQ(actual[index], tone[index]);
  }
}

template <typename T>
void expect_none_passes_through() {
  const auto tone = doppler_tone<T>(kChirps, kBins, 0.19);
  const HostTensor output =
      run_single<T>(tone, {kChirps, kBins}, ClutterFilterMethod::kNone);
  EXPECT_EQ(output.shape, (std::vector<int64_t>{kChirps, kBins}));
  const auto* actual = output.as<std::complex<T>>();
  for (std::size_t index = 0; index < tone.size(); ++index) {
    EXPECT_EQ(actual[index], tone[index]);
  }
}

TEST(ClutterFilterOpTest, MethodNonePassesThroughFloatAndDouble) {
  expect_none_passes_through<float>();
  expect_none_passes_through<double>();
}

TEST(ClutterFilterOpTest, RejectsRealInput) {
  const std::vector<float> real_cube(
      static_cast<size_t>(kChirps * kBins), 1.0f);
  const GraphRunResult result =
      run_operator_graph_result<ClutterFilterOp<float>>(
          {to_device_tensor(real_cube, {kChirps, kBins},
                            real_dtype<float>())});
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ClutterFilterOpTest, RejectsTooFewChirps) {
  const auto one_chirp = doppler_tone<float>(1, kBins, 0.0);
  const GraphRunResult single =
      run_operator_graph_result<ClutterFilterOp<float>>(
          {to_device_tensor(one_chirp, {1, kBins}, complex_dtype<float>())},
          ClutterFilterParams{.method = ClutterFilterMethod::kSingleDelay});
  EXPECT_TRUE(single.frames.empty());
  EXPECT_FALSE(single.diagnostics.empty());

  const auto two_chirps = doppler_tone<float>(2, kBins, 0.0);
  const GraphRunResult dbl =
      run_operator_graph_result<ClutterFilterOp<float>>(
          {to_device_tensor(two_chirps, {2, kBins}, complex_dtype<float>())},
          ClutterFilterParams{.method = ClutterFilterMethod::kDoubleDelay});
  EXPECT_TRUE(dbl.frames.empty());
  EXPECT_FALSE(dbl.diagnostics.empty());
}

TEST(ClutterFilterOpTest, FloatAndDoubleAgree) {
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
