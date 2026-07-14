// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for the shared radar_common library and the test simulator itself
// (the simulator is a first-class test dependency
// and its analytic outputs are cross-checked here, independently of the
// operator implementations).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/dsp/window.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/bin_model.hpp"
#include "radar_common/sim/point_target_sim.hpp"

namespace {

using holoscan::radar::ConfigCache;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::generate_window;
using holoscan::radar::window_cache_key;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
namespace sim = holoscan::radar::sim;
namespace fix = holoscan::radar::testing;  // bin-model / fixture helpers

constexpr double kPi = 3.14159265358979323846;

// Peak sidelobe level (dB below the main lobe) of a window, via a
// zero-padded host DFT of the taper.
double peak_sidelobe_db(const std::vector<double>& w, int64_t pad_factor) {
  const int64_t n = static_cast<int64_t>(w.size());
  const int64_t nfft = n * pad_factor;
  std::vector<double> mag(static_cast<size_t>(nfft));
  for (int64_t bin = 0; bin < nfft; ++bin) {
    std::complex<double> acc{0.0, 0.0};
    for (int64_t k = 0; k < n; ++k) {
      const double phase = -2.0 * kPi * static_cast<double>(bin) *
                           static_cast<double>(k) /
                           static_cast<double>(nfft);
      acc += w[static_cast<size_t>(k)] *
             std::complex<double>(std::cos(phase), std::sin(phase));
    }
    mag[static_cast<size_t>(bin)] = std::abs(acc);
  }
  const double peak = mag[0];
  // Walk right from DC past the main lobe (first local minimum), then take
  // the max of what remains (mirror-symmetric spectrum: one side suffices).
  int64_t i = 1;
  while (i < nfft / 2 && mag[static_cast<size_t>(i)] <=
                             mag[static_cast<size_t>(i - 1)]) {
    ++i;
  }
  double side = 0.0;
  for (; i < nfft / 2; ++i) {
    side = std::max(side, mag[static_cast<size_t>(i)]);
  }
  return 20.0 * std::log10(peak / side);
}

// ---- Window generation -----------------------------------------------------

TEST(WindowGeneration, RectangularIsAllOnes) {
  WindowSpec spec;  // type = kNone
  const auto w = generate_window(spec, 64);
  ASSERT_EQ(w.size(), 64u);
  for (double v : w) {
    EXPECT_DOUBLE_EQ(v, 1.0);
  }
}

TEST(WindowGeneration, HannSymmetricWithZeroEndpoints) {
  WindowSpec spec;
  spec.type = WindowType::kHann;
  spec.coherent_gain_normalization = false;
  const auto w = generate_window(spec, 65);
  EXPECT_NEAR(w.front(), 0.0, 1e-12);
  EXPECT_NEAR(w.back(), 0.0, 1e-12);
  EXPECT_NEAR(w[32], 1.0, 1e-12);  // symmetric peak at the center
  for (size_t k = 0; k < w.size(); ++k) {
    EXPECT_NEAR(w[k], w[w.size() - 1 - k], 1e-12);
  }
}

TEST(WindowGeneration, SingleSampleCosineSumWindowsPeakAtUnity) {
  // Length-1 cosine-sum windows must return the center/peak value (sum_i a[i]),
  // not the endpoint sum_i (-1)^i a[i] which collapses to ~0 and, under
  // coherent-gain normalization, would divide by a zero mean.
  for (WindowType type :
       {WindowType::kHann, WindowType::kHamming, WindowType::kBlackman,
        WindowType::kBlackmanHarris, WindowType::kFlatTop}) {
    WindowSpec spec;
    spec.type = type;
    const auto w = generate_window(spec, 1);
    ASSERT_EQ(w.size(), 1U) << holoscan::radar::to_string(type);
    EXPECT_NEAR(w[0], 1.0, 1e-12) << holoscan::radar::to_string(type);
  }
}

TEST(WindowGeneration, CoherentGainNormalizationYieldsUnitMean) {
  for (WindowType type :
       {WindowType::kHann, WindowType::kHamming, WindowType::kBlackman,
        WindowType::kBlackmanHarris, WindowType::kKaiser,
        WindowType::kChebyshev, WindowType::kTaylor, WindowType::kFlatTop,
        WindowType::kTukey}) {
    WindowSpec spec;
    spec.type = type;
    const auto w = generate_window(spec, 128);
    const double mean =
        std::accumulate(w.begin(), w.end(), 0.0) / static_cast<double>(128);
    EXPECT_NEAR(mean, 1.0, 1e-12) << holoscan::radar::to_string(type);
  }
}

TEST(WindowGeneration, SidelobeOrderingMatchesTheory) {
  // Rectangular ~13 dB < Hann ~31 dB < Blackman-Harris ~92 dB.
  WindowSpec rect;
  WindowSpec hann;
  hann.type = WindowType::kHann;
  WindowSpec bh;
  bh.type = WindowType::kBlackmanHarris;
  const double rect_db = peak_sidelobe_db(generate_window(rect, 64), 16);
  const double hann_db = peak_sidelobe_db(generate_window(hann, 64), 16);
  const double bh_db = peak_sidelobe_db(generate_window(bh, 64), 16);
  EXPECT_GT(hann_db, rect_db);
  EXPECT_GT(bh_db, hann_db);
  EXPECT_NEAR(rect_db, 13.3, 1.0);
  EXPECT_GT(hann_db, 30.0);
  EXPECT_GT(bh_db, 80.0);
}

TEST(WindowGeneration, ChebyshevHitsRequestedSidelobeLevel) {
  WindowSpec spec;
  spec.type = WindowType::kChebyshev;
  spec.sidelobe_attenuation_db = 60.0;
  spec.coherent_gain_normalization = false;
  const auto w = generate_window(spec, 64);
  const double sll = peak_sidelobe_db(w, 16);
  EXPECT_NEAR(sll, 60.0, 2.0);
}

TEST(WindowGeneration, TaylorApproximatesRequestedSidelobeLevel) {
  WindowSpec spec;
  spec.type = WindowType::kTaylor;
  spec.sidelobe_attenuation_db = 40.0;
  spec.taylor_nbar = 4;
  spec.coherent_gain_normalization = false;
  const auto w = generate_window(spec, 128);
  const double sll = peak_sidelobe_db(w, 16);
  EXPECT_GT(sll, 35.0);
  EXPECT_LT(sll, 45.0);
}

TEST(WindowGeneration, KaiserBetaOverridesAttenuation) {
  WindowSpec by_beta;
  by_beta.type = WindowType::kKaiser;
  by_beta.kaiser_beta = 8.0;
  WindowSpec by_atten;
  by_atten.type = WindowType::kKaiser;
  by_atten.sidelobe_attenuation_db = 90.0;
  const auto w_beta = generate_window(by_beta, 64);
  const auto w_atten = generate_window(by_atten, 64);
  EXPECT_NE(w_beta, w_atten);

  // With both fields configured, kaiser_beta wins: the window is identical
  // to the beta-only spec, and the attenuation is never consulted.
  WindowSpec both = by_beta;
  both.sidelobe_attenuation_db = 90.0;
  EXPECT_EQ(generate_window(both, 64), w_beta);
}

TEST(WindowGeneration, TukeyAlphaZeroIsRectangular) {
  WindowSpec spec;
  spec.type = WindowType::kTukey;
  spec.tukey_alpha = 0.0;
  spec.coherent_gain_normalization = false;
  const auto w = generate_window(spec, 64);
  for (double v : w) {
    EXPECT_DOUBLE_EQ(v, 1.0);
  }
}

TEST(WindowGeneration, CustomCoefficientsPassThrough) {
  WindowSpec spec;
  spec.type = WindowType::kCustom;
  spec.coefficients = {1.0, 2.0, 3.0, 2.0};
  spec.coherent_gain_normalization = false;
  EXPECT_EQ(generate_window(spec, 4), spec.coefficients);
  EXPECT_THROW(generate_window(spec, 5), std::invalid_argument);
}

TEST(WindowGeneration, InvalidSpecsThrow) {
  WindowSpec bad_alpha;
  bad_alpha.type = WindowType::kTukey;
  bad_alpha.tukey_alpha = 1.5;
  EXPECT_THROW(generate_window(bad_alpha, 8), std::invalid_argument);

  WindowSpec bad_nbar;
  bad_nbar.type = WindowType::kTaylor;
  bad_nbar.taylor_nbar = 0;
  EXPECT_THROW(generate_window(bad_nbar, 8), std::invalid_argument);

  EXPECT_THROW(generate_window(WindowSpec{}, 0), std::invalid_argument);

  // Non-positive (or NaN) attenuation would silently degrade Kaiser and
  // produce NaN taps for Chebyshev/Taylor (acosh of a ratio < 1).
  for (WindowType type : {WindowType::kKaiser, WindowType::kChebyshev,
                          WindowType::kTaylor}) {
    for (double bad : {0.0, -30.0,
                       std::numeric_limits<double>::quiet_NaN()}) {
      WindowSpec bad_atten;
      bad_atten.type = type;
      bad_atten.sidelobe_attenuation_db = bad;
      EXPECT_THROW(generate_window(bad_atten, 8), std::invalid_argument)
          << holoscan::radar::to_string(type) << " atten=" << bad;
    }
  }
}

// ---- Cache key / ConfigCache ----------------------------------------------

TEST(WindowCacheKey, DistinguishesOnlyConsumedFields) {
  WindowSpec hann_a;
  hann_a.type = WindowType::kHann;
  hann_a.sidelobe_attenuation_db = 30.0;
  WindowSpec hann_b = hann_a;
  hann_b.sidelobe_attenuation_db = 60.0;  // unused by Hann
  EXPECT_EQ(window_cache_key(hann_a, 64), window_cache_key(hann_b, 64));
  EXPECT_NE(window_cache_key(hann_a, 64), window_cache_key(hann_a, 128));

  WindowSpec cheb_a;
  cheb_a.type = WindowType::kChebyshev;
  cheb_a.sidelobe_attenuation_db = 30.0;
  WindowSpec cheb_b = cheb_a;
  cheb_b.sidelobe_attenuation_db = 60.0;  // consumed by Chebyshev
  EXPECT_NE(window_cache_key(cheb_a, 64), window_cache_key(cheb_b, 64));
}

TEST(ConfigCacheTest, BuildsOncePerKeyAndCaches) {
  ConfigCache<int> cache;
  int builds = 0;
  auto builder = [&builds]() { return ++builds; };
  EXPECT_EQ(cache.get_or_build("a", false, "test artifact", builder), 1);
  EXPECT_EQ(cache.get_or_build("a", true, "test artifact", builder), 1);
  EXPECT_EQ(cache.get_or_build("b", true, "test artifact", builder), 2);
  EXPECT_EQ(cache.get_or_build("a", true, "test artifact", builder), 1);
  EXPECT_EQ(builds, 2);
  EXPECT_TRUE(cache.contains("a"));
  EXPECT_FALSE(cache.contains("c"));
  EXPECT_EQ(cache.size(), 2u);
}

// ---- FMCWWaveformDescriptor -------------------------------------------------

holoscan::radar::FMCWWaveformSpec test_waveform_spec() {
  holoscan::radar::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77e9;
  spec.bandwidth = 150e6;
  spec.chirp_duration = 20e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25e-6;
  return spec;
}

TEST(WaveformDescriptor, DerivesSlopeAndBandwidthAtConstruction) {
  const FMCWWaveformDescriptor by_bandwidth(test_waveform_spec());
  EXPECT_TRUE(by_bandwidth.is_valid());
  EXPECT_DOUBLE_EQ(by_bandwidth.slope(), 150e6 / 20e-6);
  EXPECT_DOUBLE_EQ(by_bandwidth.bandwidth(), 150e6);

  auto slope_spec = test_waveform_spec();
  slope_spec.bandwidth.reset();
  slope_spec.chirp_slope = 150e6 / 20e-6;
  const FMCWWaveformDescriptor by_slope(slope_spec);
  EXPECT_DOUBLE_EQ(by_slope.slope(), 150e6 / 20e-6);
  EXPECT_DOUBLE_EQ(by_slope.bandwidth(), 150e6);
  // Bandwidth- and slope-spelled specs describe the same waveform, so the
  // canonical descriptors compare equal (config-cache semantics).
  EXPECT_EQ(by_bandwidth, by_slope);
}

TEST(WaveformDescriptor, ConstructionRejectsIncompleteOrInconsistentSpecs) {
  auto neither = test_waveform_spec();
  neither.bandwidth.reset();
  EXPECT_THROW(FMCWWaveformDescriptor{neither}, std::invalid_argument);

  auto inconsistent = test_waveform_spec();
  inconsistent.chirp_slope = 1e12;  // != bandwidth / chirp_duration
  EXPECT_THROW(FMCWWaveformDescriptor{inconsistent}, std::invalid_argument);

  for (auto field : {&holoscan::radar::FMCWWaveformSpec::carrier_frequency,
                     &holoscan::radar::FMCWWaveformSpec::chirp_duration,
                     &holoscan::radar::FMCWWaveformSpec::sample_rate,
                     &holoscan::radar::FMCWWaveformSpec::
                         chirp_repetition_interval}) {
    auto spec = test_waveform_spec();
    spec.*field = 0.0;
    EXPECT_THROW(FMCWWaveformDescriptor{spec}, std::invalid_argument);
  }
}

TEST(WaveformDescriptor, ConstructionRejectsNonFiniteValues) {
  // NaN compares false against every bound, so finiteness is enforced
  // explicitly; both spellings of the sweep and every scalar are covered.
  for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity()}) {
    for (auto field : {&holoscan::radar::FMCWWaveformSpec::carrier_frequency,
                       &holoscan::radar::FMCWWaveformSpec::chirp_duration,
                       &holoscan::radar::FMCWWaveformSpec::sample_rate,
                       &holoscan::radar::FMCWWaveformSpec::
                           chirp_repetition_interval}) {
      auto spec = test_waveform_spec();
      spec.*field = bad;
      EXPECT_THROW(FMCWWaveformDescriptor{spec}, std::invalid_argument);
    }
    auto bad_bandwidth = test_waveform_spec();
    bad_bandwidth.bandwidth = bad;
    EXPECT_THROW(FMCWWaveformDescriptor{bad_bandwidth},
                 std::invalid_argument);
    auto bad_slope = test_waveform_spec();
    bad_slope.bandwidth.reset();
    bad_slope.chirp_slope = bad;
    EXPECT_THROW(FMCWWaveformDescriptor{bad_slope}, std::invalid_argument);
  }

  // Finite inputs whose derivation overflows are rejected too.
  auto overflow = test_waveform_spec();
  overflow.bandwidth = std::numeric_limits<double>::max();
  overflow.chirp_duration = 1e-12;
  EXPECT_THROW(FMCWWaveformDescriptor{overflow}, std::invalid_argument);
}

TEST(WaveformDescriptor, WavelengthPrecomputedAndDefaultInvalid) {
  const FMCWWaveformDescriptor wf(test_waveform_spec());
  EXPECT_NEAR(wf.wavelength(), 3.8934e-3, 1e-6);
  EXPECT_NEAR(wf.wavelength(2.0 * 299792458.0), 2.0 * wf.wavelength(), 1e-9);

  const FMCWWaveformDescriptor empty;
  EXPECT_FALSE(empty.is_valid());
}

// ---- Simulator --------------------------------------------------------------

sim::SimScene single_target_scene() {
  sim::SimScene scene;
  scene.waveform.carrier_frequency = 77e9;
  scene.waveform.bandwidth = 150e6;
  scene.waveform.chirp_duration = 20e-6;
  scene.waveform.sample_rate = 12.8e6;
  scene.waveform.chirp_repetition_interval = 25e-6;
  scene.num_chirps = 64;
  scene.samples_per_chirp = 256;
  scene.targets.push_back(
      {sim::Vec3{0.0, 40.0, 0.0}, sim::Vec3{0.0, 5.0, 0.0}, {1.0, 0.0}});
  return scene;
}

TEST(PointTargetSim, GroundTruthDerivation) {
  const auto scene = single_target_scene();
  const auto result = sim::simulate(scene);
  ASSERT_EQ(result.truth.size(), 1u);
  EXPECT_DOUBLE_EQ(result.truth[0].range, 40.0);
  EXPECT_DOUBLE_EQ(result.truth[0].radial_velocity, 5.0);  // receding
  EXPECT_DOUBLE_EQ(result.truth[0].look_direction[1], 1.0);
  ASSERT_EQ(result.shape.size(), 2u);
  EXPECT_EQ(result.shape[0], scene.num_chirps);
  EXPECT_EQ(result.shape[1], scene.samples_per_chirp);
}

TEST(PointTargetSim, Deterministic) {
  auto scene = single_target_scene();
  scene.snr_db = 20.0;
  const auto a = sim::simulate(scene);
  const auto b = sim::simulate(scene);
  EXPECT_EQ(a.cube, b.cube);
  scene.seed += 1;
  const auto c = sim::simulate(scene);
  EXPECT_NE(a.cube, c.cube);
}

TEST(PointTargetSim, BeatFrequencyMatchesAnalyticModel) {
  // Host DFT of one chirp peaks at the bin the forward bin model predicts.
  const auto scene = single_target_scene();
  const auto result = sim::simulate(scene);
  const int64_t n = scene.samples_per_chirp;

  double best_mag = 0.0;
  int64_t best_bin = -1;
  for (int64_t bin = 0; bin < n; ++bin) {
    std::complex<double> acc{0.0, 0.0};
    for (int64_t s = 0; s < n; ++s) {
      const double phase = -2.0 * kPi * static_cast<double>(bin) *
                           static_cast<double>(s) / static_cast<double>(n);
      acc += result.cube[static_cast<size_t>(s)] *
             std::complex<double>(std::cos(phase), std::sin(phase));
    }
    if (std::abs(acc) > best_mag) {
      best_mag = std::abs(acc);
      best_bin = bin;
    }
  }

  const int64_t expected = fix::expected_range_bin(
      result.truth[0].range, scene.waveform.slope(),
      scene.waveform.sample_rate, n, scene.propagation_speed);
  EXPECT_NEAR(static_cast<double>(best_bin), static_cast<double>(expected),
              1.0);
}

TEST(PointTargetSim, DopplerPhaseSlopeMatchesRadialVelocity) {
  // The chirp-to-chirp phase advance at a fixed fast-time sample equals
  // 2*pi*f_d*PRI with f_d = -2*v_r/lambda (plus the small range-walk beat
  // term, negligible over one PRI at these parameters).
  const auto scene = single_target_scene();
  const auto result = sim::simulate(scene);
  const int64_t n = scene.samples_per_chirp;

  const std::complex<double> c0 = result.cube[0];
  const std::complex<double> c1 = result.cube[static_cast<size_t>(n)];
  const double measured = std::arg(c1 * std::conj(c0));

  const double lambda = scene.waveform.wavelength();
  const double f_d =
      fix::doppler_frequency(result.truth[0].radial_velocity, lambda);
  double expected =
      2.0 * kPi * f_d * scene.waveform.chirp_repetition_interval;
  // Wrap to (-pi, pi] for comparison.
  while (expected <= -kPi) {
    expected += 2.0 * kPi;
  }
  while (expected > kPi) {
    expected -= 2.0 * kPi;
  }
  EXPECT_NEAR(measured, expected, 1e-2);
}

TEST(PointTargetSim, PreMixerTimesConjReferenceIsDechirped) {
  auto scene = single_target_scene();
  const auto dechirped = sim::simulate(scene);
  scene.dechirped = false;
  const auto pre_mixer = sim::simulate(scene);
  const auto ref =
      sim::reference_chirp(scene.waveform, scene.samples_per_chirp);

  for (int64_t k = 0; k < scene.num_chirps; ++k) {
    for (int64_t s = 0; s < scene.samples_per_chirp; ++s) {
      const size_t idx =
          static_cast<size_t>(k * scene.samples_per_chirp + s);
      const std::complex<double> recovered =
          pre_mixer.cube[idx] * std::conj(ref[static_cast<size_t>(s)]);
      EXPECT_NEAR(recovered.real(), dechirped.cube[idx].real(), 1e-9);
      EXPECT_NEAR(recovered.imag(), dechirped.cube[idx].imag(), 1e-9);
    }
  }
}

// ---- Device buffer allocation -----------------------------------------------

TEST(MakeDeviceTensor, RejectsInvalidAndOverflowingShapes) {
  const auto dtype = holoscan::radar::complex_dtype<float>();
  // Zero-sized dims are valid: an empty tensor is a legitimate pipeline state
  // (e.g. zero detections → empty detection_idx / points) that operators
  // allocate-then-check. Only negative dims and int64-overflowing products
  // must throw before any allocation (a wrapped count would produce an
  // undersized buffer that full-extent views write past).
  EXPECT_NO_THROW(holoscan::radar::make_device_tensor({0, 4}, dtype));
  EXPECT_THROW(holoscan::radar::make_device_tensor({-1, 4}, dtype),
               std::invalid_argument);
  EXPECT_THROW(holoscan::radar::make_device_tensor(
                   {int64_t{1} << 32, int64_t{1} << 32}, dtype),
               std::invalid_argument);
}

// ---- YAML converters --------------------------------------------------------

TEST(YamlConverters, EnumsRoundTripAndRejectUnknownSpellings) {
  using holoscan::radar::ClutterFilterMethod;
  using holoscan::radar::FftNorm;

  for (auto norm : {FftNorm::kBackward, FftNorm::kOrtho, FftNorm::kForward}) {
    const YAML::Node node = YAML::convert<FftNorm>::encode(norm);
    EXPECT_EQ(node.as<FftNorm>(), norm);
  }
  for (auto method : {ClutterFilterMethod::kNone,
                      ClutterFilterMethod::kMeanSubtraction,
                      ClutterFilterMethod::kSingleDelay,
                      ClutterFilterMethod::kDoubleDelay}) {
    const YAML::Node node =
        YAML::convert<ClutterFilterMethod>::encode(method);
    EXPECT_EQ(node.as<ClutterFilterMethod>(), method);
  }
  for (auto type : holoscan::radar::detail::kAllWindowTypes) {
    const YAML::Node node = YAML::convert<WindowType>::encode(type);
    EXPECT_EQ(node.as<WindowType>(), type);
  }

  EXPECT_THROW(YAML::Load("hanning").as<WindowType>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load("mean-subtraction").as<ClutterFilterMethod>(),
               std::invalid_argument);
}

TEST(YamlConverters, WindowSpecMapScalarAndRoundTrip) {
  const YAML::Node map = YAML::Load(
      "{type: kaiser, kaiser_beta: 8.5, coherent_gain_normalization: false}");
  const auto spec = map.as<WindowSpec>();
  EXPECT_EQ(spec.type, WindowType::kKaiser);
  ASSERT_TRUE(spec.kaiser_beta.has_value());
  EXPECT_DOUBLE_EQ(*spec.kaiser_beta, 8.5);
  EXPECT_FALSE(spec.coherent_gain_normalization);
  // Omitted keys keep the struct defaults.
  EXPECT_DOUBLE_EQ(spec.tukey_alpha, WindowSpec{}.tukey_alpha);

  // Bare scalar is shorthand for a default spec of that type.
  const auto shorthand = YAML::Load("hann").as<WindowSpec>();
  EXPECT_EQ(shorthand.type, WindowType::kHann);
  EXPECT_EQ(shorthand.kaiser_beta, WindowSpec{}.kaiser_beta);

  // encode -> decode is identity (operator== covers every field).
  EXPECT_EQ(YAML::convert<WindowSpec>::encode(spec).as<WindowSpec>(), spec);
}

TEST(YamlConverters, WaveformDecodesBothSpellingsAndValidates) {
  const char* common =
      "carrier_frequency: 77.0e9\n"
      "chirp_duration: 20.0e-6\n"
      "sample_rate: 12.8e6\n"
      "chirp_repetition_interval: 25.0e-6\n";
  const auto by_bandwidth =
      YAML::Load(std::string(common) + "bandwidth: 150.0e6")
          .as<FMCWWaveformDescriptor>();
  const auto by_slope =
      YAML::Load(std::string(common) + "chirp_slope: 7.5e12")
          .as<FMCWWaveformDescriptor>();
  EXPECT_TRUE(by_bandwidth.is_valid());
  // Same waveform through either spelling (field-level: the derived slope
  // 150e6/20e-6 rounds one ulp away from the literal 7.5e12), and the
  // bandwidth spelling matches native construction exactly.
  EXPECT_DOUBLE_EQ(by_slope.slope(), 7.5e12);
  EXPECT_NEAR(by_bandwidth.slope(), by_slope.slope(), 2.0);  // ~1 ulp @ 7e12
  EXPECT_DOUBLE_EQ(by_bandwidth.bandwidth(), by_slope.bandwidth());
  EXPECT_EQ(by_bandwidth, FMCWWaveformDescriptor(test_waveform_spec()));

  // Construction validation applies to YAML decode: neither spelling, and
  // inconsistent double spelling, both throw.
  EXPECT_THROW(YAML::Load(common).as<FMCWWaveformDescriptor>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load(std::string(common) +
                          "bandwidth: 150.0e6\nchirp_slope: 1.0e12")
                   .as<FMCWWaveformDescriptor>(),
               std::invalid_argument);

  // encode -> decode is identity for analytic (non-reference) waveforms.
  EXPECT_EQ(YAML::convert<FMCWWaveformDescriptor>::encode(by_bandwidth)
                .as<FMCWWaveformDescriptor>(),
            by_bandwidth);
}

TEST(PointTargetSim, SimoCubeHasRxLeadingAxis) {
  auto scene = single_target_scene();
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.002, 0.0, 0.0}};
  const auto result = sim::simulate(scene);
  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 2);
  // Broadside target on the y axis: zero aperture projection, so both RX
  // see the identical signal.
  const size_t per_rx =
      static_cast<size_t>(scene.num_chirps * scene.samples_per_chirp);
  for (size_t i = 0; i < per_rx; ++i) {
    EXPECT_NEAR(result.cube[i].real(), result.cube[per_rx + i].real(),
                1e-12);
    EXPECT_NEAR(result.cube[i].imag(), result.cube[per_rx + i].imag(),
                1e-12);
  }
}

}  // namespace
