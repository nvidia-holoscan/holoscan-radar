// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SDK glue (radar_common/sdk): the build-once config cache and the YAML
// converters for enums, window specs, and waveform descriptors.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace {

using holoscan::radar::ConfigCache;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;

holoscan::radar::FMCWWaveformSpec test_waveform_spec() {
  holoscan::radar::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77e9;
  spec.bandwidth = 150e6;
  spec.chirp_duration = 20e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25e-6;
  return spec;
}

// ---- ConfigCache -----------------------------------------------------------

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

// ---- YAML converters -------------------------------------------------------

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

}  // namespace
