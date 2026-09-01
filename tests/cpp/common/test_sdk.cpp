// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SDK glue (radar_common/sdk): the build-once config cache and the YAML
// converters for enums, window specs, waveform descriptors, and the SAR
// coordinate-frame and sample-geometry descriptors.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace {

using holoscan::radar::ConfigCache;
using holoscan::radar::CoordinateFrame;
using holoscan::radar::CoordinateFrameType;
using holoscan::radar::Ellipsoid;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::LFMWaveformDescriptor;
using holoscan::radar::SampleGeometry;
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

  using holoscan::radar::ApertureEmitMode;
  using holoscan::radar::BackprojectionBackend;
  using holoscan::radar::FormationControl;
  using holoscan::radar::GeometryChange;
  using holoscan::radar::PhaseCorrectionSense;
  using holoscan::radar::RangeInterpolation;
  for (auto value : holoscan::radar::detail::kAllFormationControls) {
    const YAML::Node node = YAML::convert<FormationControl>::encode(value);
    EXPECT_EQ(node.as<FormationControl>(), value);
  }
  for (auto value : holoscan::radar::detail::kAllRangeInterpolations) {
    const YAML::Node node =
        YAML::convert<RangeInterpolation>::encode(value);
    EXPECT_EQ(node.as<RangeInterpolation>(), value);
  }
  for (auto value : holoscan::radar::detail::kAllBackprojectionBackends) {
    const YAML::Node node =
        YAML::convert<BackprojectionBackend>::encode(value);
    EXPECT_EQ(node.as<BackprojectionBackend>(), value);
  }
  for (auto value : holoscan::radar::detail::kAllGeometryChanges) {
    const YAML::Node node = YAML::convert<GeometryChange>::encode(value);
    EXPECT_EQ(node.as<GeometryChange>(), value);
  }
  for (auto value : holoscan::radar::detail::kAllApertureEmitModes) {
    const YAML::Node node =
        YAML::convert<ApertureEmitMode>::encode(value);
    EXPECT_EQ(node.as<ApertureEmitMode>(), value);
  }
  for (auto value : holoscan::radar::detail::kAllPhaseCorrectionSenses) {
    const YAML::Node node =
        YAML::convert<PhaseCorrectionSense>::encode(value);
    EXPECT_EQ(node.as<PhaseCorrectionSense>(), value);
  }
  EXPECT_THROW(YAML::Load("mixed").as<BackprojectionBackend>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load("emit-and-reset").as<FormationControl>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load("replay").as<ApertureEmitMode>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load("plus").as<PhaseCorrectionSense>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load("unspecified").as<PhaseCorrectionSense>(),
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

TEST(YamlConverters, CoordinateFrameScalarShorthandMapAndRoundTrip) {
  // The scalar shorthand is what a simulation or unit-test config writes:
  // a scene_local frame with no geodetic anchor has nothing else to say.
  const auto shorthand = YAML::Load("scene_local").as<CoordinateFrame>();
  EXPECT_TRUE(shorthand.is_valid());
  EXPECT_EQ(shorthand.frame(), CoordinateFrameType::kSceneLocal);
  EXPECT_FALSE(shorthand.origin_llh().has_value());
  EXPECT_EQ(shorthand.ellipsoid(), Ellipsoid::kWgs84);

  const auto full = YAML::Load(
                        "frame: ecef\n"
                        "origin_llh: [33.6407, -84.4277, 313.0]\n"
                        "epoch: 1234.5\n"
                        "ellipsoid: wgs84\n")
                        .as<CoordinateFrame>();
  ASSERT_TRUE(full.origin_llh().has_value());
  EXPECT_DOUBLE_EQ((*full.origin_llh())[1], -84.4277);
  EXPECT_DOUBLE_EQ(*full.epoch(), 1234.5);
  EXPECT_EQ(YAML::convert<CoordinateFrame>::encode(full).as<CoordinateFrame>(),
            full);

  // The frame family is required, and kUnspecified is not spellable.
  EXPECT_THROW(YAML::Load("ellipsoid: wgs84").as<CoordinateFrame>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load("unspecified").as<CoordinateFrame>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load("frame: enu\norigin_llh: [0.0, 0.0]\n")
                   .as<CoordinateFrame>(),
               std::invalid_argument);
}

TEST(YamlConverters, SampleGeometryRequiresEveryFieldAndRoundTrips) {
  const char* config =
      "origin: [100.0, 200.0, 0.0]\n"
      "axis_u: [1.0, 0.0, 0.0]\n"
      "axis_v: [0.0, 1.0, 0.0]\n"
      "spacing: [0.5, 0.25]\n"
      "counts: [8, 4]\n";
  const auto geometry = YAML::Load(config).as<SampleGeometry>();
  EXPECT_TRUE(geometry.is_valid());
  EXPECT_EQ(geometry.num_v(), 8);
  EXPECT_EQ(geometry.num_u(), 4);
  EXPECT_EQ(
      YAML::convert<SampleGeometry>::encode(geometry).as<SampleGeometry>(),
      geometry);

  // There is no sensible default image lattice, so every field is required
  // and a wrong-length sequence is named rather than silently padded. Both
  // report as std::invalid_argument naming the field, not as yaml-cpp's
  // zombie-node error.
  EXPECT_THROW(YAML::Load("origin: [0.0, 0.0, 0.0]").as<SampleGeometry>(),
               std::invalid_argument);
  EXPECT_THROW(YAML::Load(
                   "origin: [100.0, 200.0, 0.0]\n"
                   "axis_u: [1.0, 0.0, 0.0]\n"
                   "axis_v: [0.0, 1.0, 0.0]\n"
                   "spacing: [0.5, 0.25]\n"
                   "counts: [8]\n")
                   .as<SampleGeometry>(),
               std::invalid_argument);
}

TEST(YamlConverters, LFMWaveformMatchesTheFMCWConverter) {
  // D24 keeps the two chirp descriptors separate but intends them to merge,
  // so the converters must agree too — the same YAML must decode the same
  // way through both.
  const char* config =
      "carrier_frequency: 9.6e9\n"
      "bandwidth: 600.0e6\n"
      "chirp_duration: 20.0e-6\n"
      "sample_rate: 750.0e6\n"
      "chirp_repetition_interval: 500.0e-6\n";
  const auto lfm = YAML::Load(config).as<LFMWaveformDescriptor>();
  const auto fmcw = YAML::Load(config).as<FMCWWaveformDescriptor>();
  EXPECT_TRUE(lfm.is_valid());
  EXPECT_DOUBLE_EQ(lfm.slope(), fmcw.slope());
  EXPECT_DOUBLE_EQ(lfm.wavelength(), fmcw.wavelength());
  EXPECT_EQ(YAML::convert<LFMWaveformDescriptor>::encode(lfm)
                .as<LFMWaveformDescriptor>(),
            lfm);

  EXPECT_THROW(YAML::Load("carrier_frequency: 9.6e9\nchirp_duration: 20.0e-6")
                   .as<LFMWaveformDescriptor>(),
               std::invalid_argument);
}

}  // namespace
