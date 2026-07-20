// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Window generation (radar_common/dsp) and window cache keys: taper shapes,
// normalization, sidelobe levels, and the cache-key field discrimination.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/window.hpp"
#include "radar_common/math/constants.hpp"

namespace {

using holoscan::radar::generate_window;
using holoscan::radar::kPi;
using holoscan::radar::window_cache_key;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;

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

// ---- Cache key -------------------------------------------------------------

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

}  // namespace
