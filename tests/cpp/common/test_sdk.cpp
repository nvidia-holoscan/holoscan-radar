// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SDK glue (radar_common/sdk): the build-once config cache and the YAML
// converters for enums, window specs, and waveform descriptors.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace {

using holoscan::radar::ConfigCache;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::ShiftInvariantSubarrayPair;
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
  EXPECT_EQ(cache.get_or_build("a", "test artifact", builder), 1);
  EXPECT_EQ(cache.get_or_build("a", "test artifact", builder), 1);
  EXPECT_EQ(cache.get_or_build("b", "test artifact", builder), 2);
  // Alternating back to a seen configuration must not rebuild.
  EXPECT_EQ(cache.get_or_build("a", "test artifact", builder), 1);
  EXPECT_EQ(builds, 2);
  EXPECT_TRUE(cache.contains("a"));
  EXPECT_FALSE(cache.contains("c"));
  EXPECT_EQ(cache.size(), 2u);
}

// Captures the Holoscan log for the enclosing scope so the warn-once
// diagnostic can be asserted rather than assumed.
//
// Two things have to be neutralized. The level is forced to WARN, and
// HOLOSCAN_LOG_LEVEL is cleared first because it otherwise overrides
// set_log_level (so a developer running with a raised level would see this
// test fail for the wrong reason). The sink writes to the stderr FILE*, so
// the capture redirects the underlying descriptor rather than replacing a
// sink: that works no matter when the logger was initialized. Both the level
// and the environment are restored on scope exit.
class ScopedLogCapture {
 public:
  ScopedLogCapture() {
    if (const char* env = std::getenv("HOLOSCAN_LOG_LEVEL")) {
      saved_env_ = env;
      had_env_ = true;
      ::unsetenv("HOLOSCAN_LOG_LEVEL");
    }
    saved_level_ = holoscan::log_level();
    holoscan::set_log_level(holoscan::LogLevel::WARN);

    capture_fd_ = ::mkstemp(path_);
    std::fflush(stderr);
    saved_stderr_fd_ = ::dup(STDERR_FILENO);
    ::dup2(capture_fd_, STDERR_FILENO);
  }

  ~ScopedLogCapture() {
    std::fflush(stderr);
    ::dup2(saved_stderr_fd_, STDERR_FILENO);
    ::close(saved_stderr_fd_);
    ::close(capture_fd_);
    ::unlink(path_);
    holoscan::set_log_level(saved_level_);
    if (had_env_) {
      ::setenv("HOLOSCAN_LOG_LEVEL", saved_env_.c_str(), 1);
    }
  }

  ScopedLogCapture(const ScopedLogCapture&) = delete;
  ScopedLogCapture& operator=(const ScopedLogCapture&) = delete;

  // Everything logged in this scope so far. Read through a fresh stream so
  // the shared descriptor offset does not affect the result.
  std::string text() const {
    std::fflush(stderr);
    std::ifstream in(path_);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
  }

 private:
  char path_[32] = "/tmp/radar_log_captureXXXXXX";
  int capture_fd_ = -1;
  int saved_stderr_fd_ = -1;
  holoscan::LogLevel saved_level_ = holoscan::LogLevel::INFO;
  std::string saved_env_;
  bool had_env_ = false;
};

TEST(ConfigCacheTest, FirstBuildIsSilentThenReconfigurationWarnsOnce) {
  ConfigCache<int> cache;
  auto builder = []() { return 7; };
  ScopedLogCapture log;

  // Whether the artifact warning appears in the output this call adds.
  const auto warned_building = [&](const char* key) {
    const std::string::size_type before = log.text().size();
    cache.get_or_build(key, "test artifact", builder);
    return log.text().find("unseen test artifact configuration", before) !=
           std::string::npos;
  };

  EXPECT_FALSE(warned_building("a"));  // initial, parameter-driven build
  EXPECT_TRUE(warned_building("b"));   // reconfigured to an unseen key
  EXPECT_FALSE(warned_building("b"));  // cached: warns once per config
  EXPECT_FALSE(warned_building("a"));  // prior config still cached
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

TEST(YamlConverters, ShiftInvariantSubarrayPairRoundTrips) {
  const auto pair =
      YAML::Load("{reference_indices: [0, 1, 2], shifted_indices: [1, 2, 3]}")
          .as<ShiftInvariantSubarrayPair>();
  EXPECT_EQ(pair.reference_indices, (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(pair.shifted_indices, (std::vector<int64_t>{1, 2, 3}));
  EXPECT_EQ(YAML::convert<ShiftInvariantSubarrayPair>::encode(pair)
                .as<ShiftInvariantSubarrayPair>(),
            pair);

  EXPECT_THROW(YAML::Load("{reference_indices: [0, 1]}").
                   as<ShiftInvariantSubarrayPair>(),
               std::invalid_argument);
}

}  // namespace
