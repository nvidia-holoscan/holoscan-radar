// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <fstream>
#include <set>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>

#include "utils/perf_driver.hpp"
#include "utils/perf_profiles.hpp"

namespace {

namespace perf = holoscan::radar::perf;

TEST(PerfProfiles, SelectsCuratedClutterScopes) {
  const auto path = std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "clutter_filter.yaml";
  const auto suite = perf::load_profile_suite(path, "clutter_filter");
  EXPECT_EQ(suite.select("minimal").size(), 3U);
  EXPECT_EQ(suite.select("limited").size(), 6U);
  EXPECT_EQ(suite.select("overrides").size(), 12U);
  EXPECT_EQ(suite.select("moderate").size(), 11U);
  EXPECT_EQ(suite.select("thorough").size(), 15U);
  EXPECT_THROW(suite.select("unknown"), std::invalid_argument);

  for (const auto profile : perf::kProfiles) {
    for (const auto& port : suite.live_ports) {
      bool found = false;
      for (const auto& c : suite.select("overrides")) {
        found |= c.profile == profile && c.override_value.has_value() &&
                 (*c.override_value)["port"].as<std::string>() == port;
      }
      EXPECT_TRUE(found) << profile << ": " << port;
    }
  }
}

TEST(PerfProfiles, EveryMigratedSuiteHasThreeBaselinesAndBoundedThorough) {
  constexpr std::array names = {"clutter_filter",
                                "mimo_deinterleave",
                                "ddm_power_fold",
                                "feature_assembly",
                                "multi_source_expansion",
                                "ddm_dense_demultiplex",
                                "doppler_notch",
                                "virtual_array_assembly",
                                "snapshot_gather",
                                "tdm_dense_compensation",
                                "fmcw_doppler_processor",
                                "fmcw_range_processor",
                                "rank_one_residual_screen",
                                "fmcw_range_doppler_processor",
                                "cluster_dbscan",
                                "ddm_sparse_demultiplex",
                                "tdm_sparse_compensation",
                                "beamscan_estimator",
                                "phase_shift_beamformer",
                                "mvdr_estimator",
                                "music_estimator",
                                "root_music_estimator",
                                "esprit_estimator",
                                "subarray_selection",
                                "detection_reduction",
                                "cfar_detector"};
  for (const char* name : names) {
    SCOPED_TRACE(name);
    const auto suite = perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / (std::string(name) + ".yaml"),
        name);
    EXPECT_EQ(suite.select("minimal").size(), 3U);
    EXPECT_GE(suite.select("limited").size(), 3U);
    EXPECT_LE(suite.select("thorough").size(), 64U);
    for (const auto profile : perf::kProfiles) {
      for (const auto& port : suite.live_ports) {
        bool covered = false;
        for (const auto& c : suite.select("overrides")) {
          covered |= c.profile == profile && c.override_value.has_value() &&
                     (*c.override_value)["port"].as<std::string>() == port;
        }
        EXPECT_TRUE(covered) << profile << ": " << port;
      }
    }
  }
}

TEST(PerfProfiles, AdapterValidatesUnselectedCasesBeforeFiltering) {
  struct Scenario {
    std::string id;
    std::string size_id;
  };
  const auto suite = perf::load_profile_suite(
      std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "clutter_filter.yaml",
      "clutter_filter");
  auto invalid = [](const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    EXPECT_EQ(shape.size(), 2U);
    if (c.id == "sixteen-channels-large")
      throw std::invalid_argument("invalid adapter parameter");
    return Scenario{c.id, c.profile};
  };
  EXPECT_THROW(perf::select_validated_cases<Scenario>(suite, "minimal", {"small"}, invalid),
               std::invalid_argument);

  std::vector<std::string> decoded;
  auto valid = [&](const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    EXPECT_EQ(shape.size(), 2U);
    decoded.push_back(c.id);
    return Scenario{c.id, c.profile};
  };
  const auto selected = perf::select_validated_cases<Scenario>(suite, "limited", {}, valid);
  EXPECT_EQ(decoded.size(), suite.cases.size());
  ASSERT_EQ(selected.size(), 6U);
  EXPECT_EQ(selected.front().id, "baseline-small");
  EXPECT_EQ(selected.back().id, "passthrough-medium");
  EXPECT_THROW(perf::select_validated_cases<Scenario>(suite, "minimal", {"unknown"}, valid),
               std::invalid_argument);
}

TEST(PerfProfiles, LivePortManifestMatchesLockedContractForMigratedOperators) {
  const std::vector<std::pair<std::string, std::set<std::string>>> expected = {
      {"clutter_filter", {"method", "doppler_window"}},
      {"mimo_deinterleave", {}},
      {"ddm_power_fold", {}},
      {"feature_assembly", {}},
      {"multi_source_expansion", {}},
      {"ddm_dense_demultiplex", {}},
      {"doppler_notch", {"center_velocity", "stop_half_width", "transition_width"}},
      {"virtual_array_assembly", {"wavelength"}},
      {"snapshot_gather", {"snapshot_window"}},
      {"tdm_dense_compensation", {"waveform"}},
      {"fmcw_doppler_processor",
       {"doppler_window", "center_zero_doppler", "waveform", "fft_norm", "nfft"}},
      {"fmcw_range_processor", {"range_window", "apply_dechirp", "waveform", "fft_norm", "nfft"}},
      {"rank_one_residual_screen", {"array_geometry"}},
      {"beamscan_estimator",
       {"array_geometry", "az_coordinates", "el_coordinates", "array_taper", "peak_interpolation"}},
      {"phase_shift_beamformer",
       {"array_geometry", "az_coordinates", "el_coordinates", "array_taper"}},
      {"mvdr_estimator",
       {"array_geometry", "az_coordinates", "el_coordinates", "spatial_smoothing",
        "diagonal_loading", "max_sources"}},
      {"music_estimator",
       {"array_geometry", "az_coordinates", "el_coordinates", "spatial_smoothing", "num_sources",
        "source_count_pfa", "max_sources"}},
      {"root_music_estimator",
       {"array_geometry", "spatial_smoothing", "num_sources", "source_count_pfa", "max_sources"}},
      {"esprit_estimator",
       {"array_geometry", "spatial_smoothing", "num_sources", "source_count_pfa", "max_sources"}},
      {"subarray_selection", {"array_geometry", "subarray"}},
      {"detection_reduction", {"representative", "score_method", "feature_weights"}},
      {"cfar_detector",
       {"os_rank", "guard_cells", "training_cells", "pfa", "alpha_lut", "threshold_scale",
        "nci_axes", "max_detections"}},
      {"cluster_dbscan", {"epsilon", "min_points", "max_clusters"}},
      {"ddm_sparse_demultiplex", {"wavelength", "snapshot_window"}},
      {"tdm_sparse_compensation", {"waveform", "snapshot_window"}},
      {"fmcw_range_doppler_processor",
       {"range_window", "doppler_window", "apply_dechirp", "clutter_filter", "waveform", "fft_norm",
        "center_zero_doppler", "nfft_range", "nfft_doppler"}},
  };
  for (const auto& [name, ports] : expected) {
    SCOPED_TRACE(name);
    const auto suite = perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / (name + ".yaml"), name);
    EXPECT_EQ(std::set<std::string>(suite.live_ports.begin(), suite.live_ports.end()), ports);
  }
}

TEST(PerfProfiles, LimitedIncludesEveryCfarBackend) {
  const auto suite = perf::load_profile_suite(
      std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "cfar_detector.yaml",
      "cfar_detector");
  std::set<std::string> backends;
  for (const auto& c : suite.select("limited"))
    backends.insert(c.parameters["variant"].as<std::string>());
  EXPECT_EQ(backends, (std::set<std::string>{"CA", "GO", "SO", "OS"}));

  const auto ddm = perf::load_profile_suite(
      std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "ddm_dense_demultiplex.yaml",
      "ddm_dense_demultiplex");
  std::set<std::string> codes;
  for (const auto& c : ddm.select("limited"))
    codes.insert(c.parameters["code"].as<std::string>());
  EXPECT_EQ(codes, (std::set<std::string>{"linear", "nonlinear"}));
}

TEST(PerfProfiles, FftPlanCasesDistinguishUnseenAndCached) {
  for (const char* name : {"fmcw_range_processor", "fmcw_doppler_processor"}) {
    SCOPED_TRACE(name);
    const auto suite = perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / (std::string(name) + ".yaml"),
        name);
    std::set<std::string> cache_modes;
    for (const auto& c : suite.select("thorough")) {
      if (c.profile == "medium" && c.override_value &&
          (*c.override_value)["port"].as<std::string>() == "nfft")
        cache_modes.insert((*c.override_value)["cache"].as<std::string>());
    }
    EXPECT_EQ(cache_modes, (std::set<std::string>{"cached", "unseen"}));
  }
}

TEST(PerfProfiles, RejectsUnknownKeyAndInvalidExtent) {
  const auto path = std::filesystem::temp_directory_path() / "radar-perf-profile-schema.yaml";
  const std::string prefix =
      "version: 1\noperator: fixture\n"
      "profiles:\n  small: {shape: [1]}\n  medium: {shape: [2]}\n"
      "  large: {shape: [3]}\n"
      "live_ports: []\ncases:\n"
      "  - {id: base-small, profile: small, scopes: [minimal], parameters: {}}\n"
      "  - {id: base-medium, profile: medium, scopes: [minimal], parameters: {}}\n"
      "  - {id: base-large, profile: large, scopes: [minimal], parameters: {}}\n";
  {
    std::ofstream out(path);
    out << prefix << "unexpected: true\n";
  }
  EXPECT_THROW(perf::load_profile_suite(path, "fixture"), std::invalid_argument);
  {
    std::ofstream out(path);
    auto invalid = prefix;
    const auto pos = invalid.find("small: {shape: [1]}");
    invalid.replace(pos, std::string("small: {shape: [1]}").size(), "small: {shape: [0]}");
    out << invalid;
  }
  EXPECT_THROW(perf::load_profile_suite(path, "fixture"), std::invalid_argument);
  std::filesystem::remove(path);
}

TEST(PerfProfiles, RejectsMissingLiveCoverageAndDuplicateIds) {
  const auto path = std::filesystem::temp_directory_path() / "radar-perf-profile-invalid.yaml";
  {
    std::ofstream out(path);
    out << "version: 1\noperator: fixture\n"
           "profiles:\n  small: {shape: [1]}\n  medium: {shape: [2]}\n"
           "  large: {shape: [3]}\n"
           "live_ports: [method]\n"
           "cases:\n"
           "  - {id: base-small, profile: small, scopes: [minimal], parameters: {}}\n"
           "  - {id: base-medium, profile: medium, scopes: [minimal], parameters: {}}\n"
           "  - {id: base-large, profile: large, scopes: [minimal], parameters: {}}\n";
  }
  EXPECT_THROW(perf::load_profile_suite(path, "fixture"), std::invalid_argument);
  {
    std::ofstream out(path, std::ios::app);
    out << "  - {id: base-small, profile: small, scopes: [limited], parameters: {}}\n";
  }
  EXPECT_THROW(perf::load_profile_suite(path, "fixture"), std::invalid_argument);
  std::filesystem::remove(path);
}

struct AdapterFixtureScenario {
  std::string id;
  std::string size_id;
  bool live{false};
};

struct AdapterFixtureTraits {
  using Scenario = AdapterFixtureScenario;
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return {id, shape, shape[0] * shape[1], id};
  }
};

perf::ResultRow base_row(const AdapterFixtureScenario& scenario) {
  perf::ResultRow row;
  row.op_name = "FixtureOp";
  row.live_port = scenario.live ? "method" : "";
  return row;
}

perf::TimingCollector run_graph(const AdapterFixtureScenario&, const perf::RunPlan& plan) {
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::time_point{};
  collector.t_first_tick = collector.t_construct + std::chrono::milliseconds{1};
  for (int frame = 0; frame < plan.total_frames; ++frame) {
    const auto start = collector.t_construct + std::chrono::milliseconds{frame + 1};
    const auto duration =
        frame == plan.override_at ? std::chrono::microseconds{200} : std::chrono::microseconds{100};
    collector.starts.push_back(start);
    collector.ends.push_back(start + duration);
  }
  return collector;
}

void decorate_row(perf::ResultRow& row, const AdapterFixtureScenario&) {
  row.notes = "typed adapter";
}

TEST(PerfProfiles, SharedAdapterOwnsSizeAndReportIdentity) {
  const auto suite = perf::load_profile_suite(
      std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "clutter_filter.yaml",
      "clutter_filter");
  const AdapterFixtureScenario scenario{"fixture-live", "small", true};
  perf::PerfReport report;
  perf::detail::record_exercised_sizes<AdapterFixtureTraits>(suite, {scenario}, &report);
  ASSERT_EQ(report.sizes.size(), 1U);
  EXPECT_EQ(report.sizes[0].id, "small");
  EXPECT_EQ(report.sizes[0].shape, suite.shapes[0]);

  perf::RunSettings settings;
  settings.warmup_frames = 1;
  settings.measured_frames = 3;
  settings.repeats = 2;
  perf::detail::run_adapter_case<AdapterFixtureTraits>(scenario, settings, &report);
  ASSERT_EQ(report.results.size(), 2U);
  EXPECT_EQ(report.results[0].scenario_id, scenario.id);
  EXPECT_EQ(report.results[1].size_id, scenario.size_id);
  EXPECT_EQ(report.results[1].live_port, "method");
  EXPECT_EQ(report.results[1].latency.samples, 6);
  EXPECT_NEAR(report.results[1].override_delta_ms, 0.1, 1e-9);
  EXPECT_EQ(report.results[1].notes, "typed adapter");
}

}  // namespace
