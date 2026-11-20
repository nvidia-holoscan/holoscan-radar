// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SDK glue (radar_common/sdk): the build-once config cache and the config
// converters for enums, window specs, and waveform descriptors.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/config/io.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/sdk/config_codecs.hpp"

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
  auto builder = [&builds]() {
    return ++builds;
  };
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
  auto builder = []() {
    return 7;
  };
  ScopedLogCapture log;

  // Whether the artifact warning appears in the output this call adds.
  const auto warned_building = [&](const char* key) {
    const std::string::size_type before = log.text().size();
    cache.get_or_build(key, "test artifact", builder);
    return log.text().find("unseen test artifact configuration", before) != std::string::npos;
  };

  EXPECT_FALSE(warned_building("a"));  // initial, parameter-driven build
  EXPECT_TRUE(warned_building("b"));   // reconfigured to an unseen key
  EXPECT_FALSE(warned_building("b"));  // cached: warns once per config
  EXPECT_FALSE(warned_building("a"));  // prior config still cached
}

TEST(ConfigCacheTest, SharedBudgetEvictsLeastRecentlyUsedAcrossCaches) {
  auto budget = std::make_shared<holoscan::radar::ConfigCacheBudget>();
  ConfigCache<int> a;
  ConfigCache<int> b;
  a.bind(budget);
  b.bind(budget);
  auto build = [] {
    return 9;
  };
  a.get_or_build("a", "a", build);
  const auto one_entry = budget->bytes();
  int waits = 0;
  {
    holoscan::radar::ConfigCacheBudget::Tick tick(*budget, 2 * one_entry, [&] {
      ++waits;
    });
    b.get_or_build("b", "b", build);
    a.get_or_build("a", "a", build);  // refresh A before introducing C
    b.get_or_build("c", "b", build);
    EXPECT_TRUE(a.contains("a"));
    EXPECT_FALSE(b.contains("b"));
    EXPECT_TRUE(b.contains("c"));
    EXPECT_LE(budget->bytes(), budget->limit());
  }
  EXPECT_GT(waits, 0);
}

TEST(ConfigCacheTest, ZeroBudgetPinsUncachedArtifactsUntilGpuBarrier) {
  auto budget = std::make_shared<holoscan::radar::ConfigCacheBudget>();
  ConfigCache<std::shared_ptr<int>> cache;
  cache.bind(budget);
  std::weak_ptr<int> borrowed;
  int barriers = 0;
  {
    holoscan::radar::ConfigCacheBudget::Tick tick(*budget, 0, [&] {
      EXPECT_FALSE(borrowed.expired());
      ++barriers;
    });
    const auto& first = cache.get_or_build("a", "value", [&] {
      auto value = std::make_shared<int>(42);
      borrowed = value;
      return value;
    });
    cache.get_or_build("b", "value", [] {
      return std::make_shared<int>(13);
    });
    EXPECT_EQ(*first, 42);  // another uncached miss cannot invalidate this borrow
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(budget->bytes(), 0U);
  }
  EXPECT_TRUE(borrowed.expired());
  EXPECT_EQ(barriers, 1);
}

TEST(ConfigCacheTest, HitsDoNotWaitAndKeySweepStaysBounded) {
  auto budget = std::make_shared<holoscan::radar::ConfigCacheBudget>();
  ConfigCache<int> cache;
  cache.bind(budget);
  int barriers = 0;
  {
    holoscan::radar::ConfigCacheBudget::Tick tick(*budget, 1024, [&] {
      ++barriers;
    });
    for (int i = 0; i < 100; ++i)
      cache.get_or_build("a", "value", [] {
        return 1;
      });
  }
  EXPECT_EQ(barriers, 0);
  for (int i = 0; i < 1000; ++i) {
    holoscan::radar::ConfigCacheBudget::Tick tick(*budget, 1024, [&] {
      ++barriers;
    });
    cache.get_or_build(std::to_string(i), "value", [] {
      return 1;
    });
    EXPECT_LE(budget->bytes(), 1024U);
    EXPECT_LE(cache.size(), 3U);
  }
  EXPECT_GT(barriers, 0);
}

TEST(ConfigCacheTest, LargeKeysAreChargedAndWarningsStayBounded) {
  auto budget = std::make_shared<holoscan::radar::ConfigCacheBudget>();
  ConfigCache<int> cache;
  cache.bind(budget);
  ScopedLogCapture log;
  for (int i = 0; i < 10; ++i) {
    holoscan::radar::ConfigCacheBudget::Tick tick(*budget, 1024, [] {});
    cache.get_or_build(std::string(4096, 'x') + std::to_string(i), "value", [] {
      return 1;
    });
  }
  EXPECT_EQ(cache.size(), 0U);
  const auto text = log.text();
  const auto first = text.find("using it without caching");
  ASSERT_NE(first, std::string::npos);
  EXPECT_EQ(text.find("using it without caching", first + 1), std::string::npos);
  EXPECT_LT(text.size(), 4096U);  // diagnostic never copies an attacker-sized key
}

TEST(ConfigCacheTest, FailedBuilderDoesNotChangeRetention) {
  ConfigCache<int> cache;
  cache.get_or_build("a", "value", [] {
    return 3;
  });
  EXPECT_THROW(cache.get_or_build("b", "value",
                                  []() -> int {
                                    throw std::runtime_error("builder failed");
                                  }),
               std::runtime_error);
  EXPECT_TRUE(cache.contains("a"));
  EXPECT_FALSE(cache.contains("b"));
}

// ---- Config codecs ---------------------------------------------------------

template <typename T>
T round_trip(const T& value) {
  return holoscan::config::Document::from_value(holoscan::config::encode_config(value))
      .template get<T>("");
}

holoscan::config::Document parse_yaml(const std::string& text) {
  return holoscan::config_io::parse(text, holoscan::config_io::Format::kYaml);
}

TEST(ConfigCodecs, EnumsRoundTripAndRejectUnknownSpellings) {
  using holoscan::radar::ClutterFilterMethod;
  using holoscan::radar::FftNorm;
  using holoscan::radar::MultiplexingScheme;

  for (auto norm : {FftNorm::kBackward, FftNorm::kOrtho, FftNorm::kForward}) {
    EXPECT_EQ(round_trip(norm), norm);
  }
  for (auto method : {ClutterFilterMethod::kNone, ClutterFilterMethod::kMeanSubtraction,
                      ClutterFilterMethod::kSingleDelay, ClutterFilterMethod::kDoubleDelay}) {
    EXPECT_EQ(round_trip(method), method);
  }
  for (auto type : holoscan::radar::config_detail::kWindowTypes) {
    EXPECT_EQ(round_trip(type), type);
  }
  for (auto scheme : {MultiplexingScheme::kTdm, MultiplexingScheme::kDdm}) {
    EXPECT_EQ(round_trip(scheme), scheme);
  }

  EXPECT_FALSE(parse_yaml("mean-subtraction").try_get<ClutterFilterMethod>(""));
  EXPECT_FALSE(parse_yaml("fdm").try_get<MultiplexingScheme>(""));
}

TEST(ConfigCodecs, WindowSpecRoundTrips) {
  const auto spec =
      parse_yaml("{type: kaiser, kaiser_beta: 8.5, coherent_gain_normalization: false}")
          .get<WindowSpec>("");
  // Omitted keys keep the struct defaults.
  EXPECT_DOUBLE_EQ(spec.tukey_alpha, WindowSpec{}.tukey_alpha);
  EXPECT_EQ(round_trip(spec), spec);
}

TEST(ConfigCodecs, WaveformDecodesBothSpellingsAndRoundTrips) {
  const std::string common =
      "{carrier_frequency: 77.0e9, chirp_duration: 20.0e-6, sample_rate: 12.8e6, "
      "chirp_repetition_interval: 25.0e-6, ";
  const auto by_bandwidth =
      parse_yaml(common + "bandwidth: 150.0e6}").get<FMCWWaveformDescriptor>("");
  const auto by_slope = parse_yaml(common + "chirp_slope: 7.5e12}").get<FMCWWaveformDescriptor>("");
  // Same waveform through either spelling (field-level: the derived slope
  // 150e6/20e-6 rounds one ulp away from the literal 7.5e12), and the
  // bandwidth spelling matches native construction exactly.
  EXPECT_DOUBLE_EQ(by_slope.slope(), 7.5e12);
  EXPECT_NEAR(by_bandwidth.slope(), by_slope.slope(), 2.0);  // ~1 ulp @ 7e12
  EXPECT_DOUBLE_EQ(by_bandwidth.bandwidth(), by_slope.bandwidth());
  EXPECT_EQ(by_bandwidth, FMCWWaveformDescriptor(test_waveform_spec()));

  EXPECT_FALSE(parse_yaml(common + "bandwidth: 150.0e6, chirp_slope: 1.0e12}")
                   .try_get<FMCWWaveformDescriptor>(""));
  EXPECT_EQ(round_trip(by_bandwidth), by_bandwidth);
}

TEST(ConfigCodecs, EnumsAndWindowShorthandMatchYaml) {
  auto window_doc = holoscan::config_io::try_parse("hann", holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(window_doc) << holoscan::config::format_error(window_doc.error());
  const auto window = window_doc->try_get<WindowSpec>("");
  ASSERT_TRUE(window) << holoscan::config::format_error(window.error());
  EXPECT_EQ(window->type, WindowType::kHann);

  auto norm_doc = holoscan::config_io::try_parse("ortho", holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(norm_doc);
  const auto norm = norm_doc->try_get<holoscan::radar::FftNorm>("");
  ASSERT_TRUE(norm);
  EXPECT_EQ(*norm, holoscan::radar::FftNorm::kOrtho);

  auto invalid_doc = holoscan::config_io::try_parse("hanning", holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(invalid_doc);
  EXPECT_FALSE(invalid_doc->try_get<WindowType>(""));
}

TEST(ConfigCodecs, WindowMapAndWaveformValidate) {
  auto window_doc = holoscan::config_io::try_parse(
      "{type: kaiser, kaiser_beta: 8.5, "
      "coherent_gain_normalization: false}",
      holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(window_doc);
  const auto window = window_doc->try_get<WindowSpec>("");
  ASSERT_TRUE(window);
  EXPECT_EQ(window->type, WindowType::kKaiser);
  ASSERT_TRUE(window->kaiser_beta.has_value());
  EXPECT_DOUBLE_EQ(*window->kaiser_beta, 8.5);
  EXPECT_FALSE(window->coherent_gain_normalization);

  auto waveform_doc = holoscan::config_io::try_parse(
      "{carrier_frequency: 77.0e9, bandwidth: 150.0e6, "
      "chirp_duration: 20.0e-6, sample_rate: 12.8e6, "
      "chirp_repetition_interval: 25.0e-6}",
      holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(waveform_doc);
  const auto waveform = waveform_doc->try_get<FMCWWaveformDescriptor>("");
  ASSERT_TRUE(waveform);
  EXPECT_TRUE(waveform->is_valid());

  auto invalid_waveform_doc = holoscan::config_io::try_parse(
      "{carrier_frequency: 77.0e9, chirp_duration: 20.0e-6, "
      "sample_rate: 12.8e6, chirp_repetition_interval: 25.0e-6}",
      holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(invalid_waveform_doc);
  EXPECT_FALSE(invalid_waveform_doc->try_get<FMCWWaveformDescriptor>(""));

  auto unknown_field_doc = holoscan::config_io::try_parse("{type: hann, unexpected: true}",
                                                          holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(unknown_field_doc);
  EXPECT_FALSE(unknown_field_doc->try_get<WindowSpec>(""));
}

TEST(ConfigCodecs, ShiftInvariantSubarrayPairRoundTrips) {
  const auto pair = parse_yaml("{reference_indices: [0, 1, 2], shifted_indices: [1, 2, 3]}")
                        .get<ShiftInvariantSubarrayPair>("");
  EXPECT_EQ(pair.reference_indices, (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(pair.shifted_indices, (std::vector<int64_t>{1, 2, 3}));
  EXPECT_EQ(round_trip(pair), pair);

  EXPECT_FALSE(parse_yaml("{reference_indices: [0, 1]}").try_get<ShiftInvariantSubarrayPair>(""));
}

}  // namespace
