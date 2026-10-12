// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <fstream>
#include <set>
#include <vector>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "utils/perf_driver.hpp"
#include "utils/perf_profiles.hpp"
#include "utils/perf_yaml_recipe.hpp"
#include "utils/perf_yaml_pilot.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_yaml_standard.hpp"
#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"

namespace {

namespace perf = holoscan::radar::perf;

std::filesystem::path profile_path(std::string_view name) {
  return perf::operator_recipe_path(HOLOSCAN_RADAR_PERF_OPERATOR_ROOT, name);
}

TEST(PerfProfiles, FindsColocatedRecipeByOperatorName) {
  const auto recipe = profile_path("cfar_detector");
  EXPECT_EQ(recipe.filename(), "recipe.yaml");
  EXPECT_EQ(recipe.parent_path().filename(), "cfar_detector");
  EXPECT_EQ(recipe.parent_path().parent_path().filename(), "detection");
  EXPECT_THROW(profile_path("missing_operator"), std::invalid_argument);
}

TEST(PerfProfiles, SelectsCuratedClutterScopes) {
  const auto path = profile_path("clutter_filter");
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
    const auto suite = perf::load_profile_suite(profile_path(name), name);
    EXPECT_EQ(suite.version, 2);
    EXPECT_NO_THROW((void)perf::parse_yaml_recipe(suite));
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
  const auto suite = perf::load_profile_suite(profile_path("clutter_filter"), "clutter_filter");
  auto invalid = [](const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    EXPECT_EQ(shape.size(), 2U);
    if (c.id == "sixteen-channels-large")
      throw std::invalid_argument("invalid adapter parameter");
    return Scenario{c.id, c.profile};
  };
  try {
    (void)perf::select_validated_cases<Scenario>(suite, "minimal", {"small"}, invalid);
    FAIL() << "invalid unselected case should fail";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("sixteen-channels-large"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("line"), std::string::npos);
  }

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
  const auto live =
      perf::select_validated_scenarios<Scenario>(suite, "overrides", {"small"}, valid);
  ASSERT_EQ(live.size(), 3U);
  EXPECT_EQ(live[0].live_port, "");
  EXPECT_EQ(live[1].live_port, "method");
  EXPECT_EQ(live[2].live_port, "doppler_window");
  EXPECT_THROW(perf::select_validated_cases<Scenario>(suite, "minimal", {"unknown"}, valid),
               std::invalid_argument);
}

TEST(PerfProfiles, PilotCaseReadsYamlAxesAndRejectsMalformedValues) {
  const auto suite = perf::load_profile_suite(profile_path("ddm_power_fold"), "ddm_power_fold");
  const auto& first = suite.cases.front();
  const auto shape = suite.shape_for(first.profile);
  perf::RecipeScenario scenario;
  scenario.values = suite.axes_for(first.profile);
  EXPECT_EQ(scenario.axis("tx"), shape[0]);
  EXPECT_EQ(scenario.axis("doppler"), shape[2]);
  EXPECT_THROW(perf::pilot_scenario<perf::RecipeScenario>(suite, first, shape, 3),
               std::invalid_argument);

  const auto params = YAML::Load("{nfft: 0}");
  EXPECT_THROW(perf::require_recipe_params(params, {"nfft", "waveform"}), std::invalid_argument);
  EXPECT_EQ(perf::recipe_pair<int64_t>(YAML::Load("[1, 2]"), "window")[1], 2);
  EXPECT_THROW((void)perf::recipe_pair<int64_t>(YAML::Load("[1]"), "window"),
               std::invalid_argument);
  EXPECT_EQ(perf::scaled_recipe_axis(YAML::Load("{scale: 3}"), 128, "nfft"), 384);
  EXPECT_THROW(perf::scaled_recipe_axis(YAML::Load("{scale: 0}"), 128, "nfft"),
               std::invalid_argument);
  EXPECT_THROW(perf::scaled_recipe_axis(YAML::Load("{scale: 3}"),
                                        std::numeric_limits<int64_t>::max(), "nfft"),
               std::invalid_argument);
}

TEST(PerfProfiles, SharedPilotCodecsValidateSemanticValues) {
  const auto range_suite =
      perf::load_profile_suite(profile_path("fmcw_range_processor"), "fmcw_range_processor");
  const auto waveform = range_suite.recipe["params"]["waveform"];
  EXPECT_NO_THROW((void)perf::recipe_fmcw_waveform(waveform));
  EXPECT_THROW((void)perf::recipe_fmcw_waveform(waveform, 1.0e-6), std::invalid_argument);

  const auto beam_suite =
      perf::load_profile_suite(profile_path("beamscan_estimator"), "beamscan_estimator");
  auto geometry = YAML::Clone(beam_suite.recipe["params"]["array_geometry"]);
  EXPECT_NO_THROW((void)perf::recipe_ula_geometry(geometry, 4));
  geometry["spacing_wavelengths"] = -0.1;
  EXPECT_THROW((void)perf::recipe_ula_geometry(geometry, 4), std::invalid_argument);
  EXPECT_EQ(perf::recipe_snapshot_edge("clamp"), holoscan::radar::SnapshotEdgeMode::kClamp);
  EXPECT_THROW(perf::recipe_snapshot_edge("unknown"), std::invalid_argument);
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
    const auto suite = perf::load_profile_suite(profile_path(name), name);
    EXPECT_EQ(std::set<std::string>(suite.live_ports.begin(), suite.live_ports.end()), ports);
  }
}

TEST(PerfProfiles, LimitedIncludesEveryCfarBackend) {
  const auto suite = perf::load_profile_suite(profile_path("cfar_detector"), "cfar_detector");
  std::set<std::string> backends;
  for (const auto& c : suite.select("limited")) {
    const auto params = perf::merge_recipe_map(suite.recipe["params"], c.parameters);
    backends.insert(params["cfar_variant"].as<std::string>());
  }
  EXPECT_EQ(backends, (std::set<std::string>{"CA", "GO", "SO", "OS"}));

  const auto ddm =
      perf::load_profile_suite(profile_path("ddm_dense_demultiplex"), "ddm_dense_demultiplex");
  std::set<std::string> codes;
  for (const auto& c : ddm.select("limited")) {
    const auto params = perf::merge_recipe_map(ddm.recipe["params"], c.parameters);
    codes.insert(params["ddm_code"]["factory"].as<std::string>());
  }
  EXPECT_EQ(codes, (std::set<std::string>{"linear_exact_ddm_code", "nonlinear_phase_code"}));
}

TEST(PerfProfiles, VersionTwoRecipesKeepCuratedCasesAndNamedInputs) {
  for (const char* name : {"ddm_power_fold", "snapshot_gather", "beamscan_estimator",
                           "fmcw_range_processor", "cfar_detector"}) {
    SCOPED_TRACE(name);
    const auto suite = perf::load_profile_suite(profile_path(name), name);
    ASSERT_EQ(suite.version, 2);
    const auto recipe = perf::parse_yaml_recipe(suite);
    EXPECT_FALSE(recipe.operator_type.empty());
    EXPECT_EQ(suite.select("minimal").size(), 3U);
    for (const auto& c : suite.cases) {
      const auto values = perf::recipe_values(suite, c);
      for (const auto& input : recipe.inputs)
        EXPECT_FALSE(perf::recipe_shape(input, values).empty());
    }
  }
  const auto clutter = perf::load_profile_suite(profile_path("clutter_filter"), "clutter_filter");
  EXPECT_EQ(clutter.version, 2);
  EXPECT_NO_THROW((void)perf::parse_yaml_recipe(clutter));
}

TEST(PerfProfiles, GenericYamlScalarParamsMaterializeThroughHsdkSchema) {
  using Params = holoscan::radar::DopplerNotchParams;
  const auto values = YAML::Load("{center_velocity: 1.25, cyclic_velocity_axis: false}");
  const auto* schema = holoscan::param_schema_descriptor<Params>();
  ASSERT_NE(schema, nullptr);
  const perf::YamlParameterAssignments assignments(*schema, values);
  auto materialized = holoscan::authoring::materialize_parameters(
      holoscan::param_type_descriptor<Params>(), schema, assignments.values());
  ASSERT_TRUE(materialized) << materialized.error().message;
  const auto* params = static_cast<const Params*>(materialized->data());
  EXPECT_DOUBLE_EQ(params->center_velocity, 1.25);
  EXPECT_FALSE(params->cyclic_velocity_axis);
}

TEST(PerfProfiles, VersionTwoRecipeRejectsUnknownGeneratorAndAxis) {
  auto suite = perf::load_profile_suite(profile_path("ddm_power_fold"), "ddm_power_fold");
  suite.recipe = YAML::Clone(suite.recipe);
  suite.recipe["inputs"][0]["generator"] = "unregistered_generator";
  EXPECT_THROW(perf::parse_yaml_recipe(suite), std::invalid_argument);
  suite.recipe["inputs"][0]["generator"] = "iq_cube";
  suite.recipe["inputs"][0]["shape"][0] = "unknown_axis";
  const auto recipe = perf::parse_yaml_recipe(suite);
  EXPECT_THROW(perf::recipe_shape(recipe.inputs[0], suite.axes_for("small")),
               std::invalid_argument);
}

TEST(PerfProfiles, VersionTwoRecipeRejectsPortPayloadAndRelatedInputMismatch) {
  auto suite = perf::load_profile_suite(profile_path("snapshot_gather"), "snapshot_gather");
  auto recipe = perf::parse_yaml_recipe(suite);
  const auto inputs = recipe.inputs;
  const auto outputs = recipe.outputs;
  const auto live = recipe.live_ports;
  recipe.inputs[0].port = "wrong_port";
  EXPECT_THROW(perf::validate_recipe_manifest(recipe, inputs, outputs, live),
               std::invalid_argument);
  recipe.inputs = inputs;
  recipe.inputs[0].dtype = "float32";
  EXPECT_THROW(perf::validate_recipe_manifest(recipe, inputs, outputs, live),
               std::invalid_argument);
  recipe.inputs = inputs;
  recipe.inputs[0].shape = {"range", "doppler", "channels"};
  EXPECT_THROW(perf::validate_recipe_manifest(recipe, inputs, outputs, live),
               std::invalid_argument);
  suite.recipe = YAML::Clone(suite.recipe);
  suite.recipe["fixtures"]["two"] = 3;
  EXPECT_THROW(perf::validate_recipe_case(suite, suite.cases.front()), std::invalid_argument);
}

TEST(PerfProfiles, PilotLiveCodecDeclarationsAreValidatedBeforeSelection) {
  auto suite =
      perf::load_profile_suite(profile_path("fmcw_range_processor"), "fmcw_range_processor");
  auto recipe = perf::parse_yaml_recipe(suite);
  EXPECT_EQ(recipe.live_codecs.at("nfft"), "scaled_axis");
  EXPECT_EQ(recipe.live_axes.at("nfft"), "samples");
  const auto cached = std::find_if(suite.cases.begin(), suite.cases.end(), [](const auto& c) {
    return c.id == "nfft-cached-medium";
  });
  ASSERT_NE(cached, suite.cases.end());
  EXPECT_NO_THROW(perf::validate_recipe_live_axis(recipe, "nfft", suite.axes_for(cached->profile)));

  suite.recipe = YAML::Clone(suite.recipe);
  suite.recipe["live"]["nfft"]["axis"] = "missing_axis";
  recipe = perf::parse_yaml_recipe(suite);
  EXPECT_THROW(perf::validate_recipe_live_axis(recipe, "nfft", suite.axes_for(cached->profile)),
               std::invalid_argument);
  suite.recipe["live"]["nfft"]["axis"] = "samples";
  suite.recipe["live"]["nfft"]["codec"] = YAML::Node();
  EXPECT_THROW(perf::parse_yaml_recipe(suite), std::invalid_argument);
}

TEST(PerfProfiles, FftPlanCasesDistinguishUnseenAndCached) {
  for (const char* name : {"fmcw_range_processor", "fmcw_doppler_processor"}) {
    SCOPED_TRACE(name);
    const auto suite = perf::load_profile_suite(profile_path(name), name);
    std::set<std::string> cache_modes;
    for (const auto& c : suite.select("thorough")) {
      if (c.profile == "medium" && c.override_value &&
          (*c.override_value)["port"].as<std::string>() == "nfft")
        cache_modes.insert((*c.override_value)["cache"].as<std::string>());
    }
    EXPECT_EQ(cache_modes, (std::set<std::string>{"cached", "unseen"}));
    struct Scenario {
      std::string id;
      std::string size_id;
    };
    const auto selected = perf::select_validated_scenarios<Scenario>(
        suite, "thorough", {"medium"}, [](const perf::ProfileCase& c, const std::vector<int64_t>&) {
          return Scenario{c.id, c.profile};
        });
    int cached_count = 0;
    for (const auto& item : selected) {
      if (item.cached) {
        ++cached_count;
        EXPECT_EQ(item.live_port, "nfft");
      }
    }
    EXPECT_EQ(cached_count, 1);
  }
}

TEST(PerfProfiles, ExpandsOnlyNamedFamilyProfilesInOrder) {
  const auto path = std::filesystem::temp_directory_path() / "radar-perf-family.yaml";
  const std::string yaml =
      "version: 1\noperator: fixture\n"
      "profiles:\n  small: {shape: [1]}\n  medium: {shape: [2]}\n"
      "  large: {shape: [3]}\n"
      "live_ports: []\ncases:\n"
      "  - {id: \"baseline-{profile}\", profiles: [small, medium, large], "
      "scopes: [minimal], parameters: {mode: default}}\n";
  {
    std::ofstream out(path);
    out << yaml;
  }
  const auto suite = perf::load_profile_suite(path, "fixture");
  ASSERT_EQ(suite.cases.size(), 3U);
  EXPECT_EQ(suite.cases[0].id, "baseline-small");
  EXPECT_EQ(suite.cases[1].id, "baseline-medium");
  EXPECT_EQ(suite.cases[2].id, "baseline-large");
  EXPECT_EQ(suite.cases[1].parameters["mode"].as<std::string>(), "default");
  {
    std::ofstream out(path, std::ios::app);
    out << "  - {id: baseline-medium, profile: medium, scopes: [limited], parameters: {}}\n";
  }
  EXPECT_THROW(perf::load_profile_suite(path, "fixture"), std::invalid_argument);
  std::filesystem::remove(path);
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
  static const char* tool_name() { return "perf_fixture"; }
  static const char* tool_description() { return "fixture"; }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return {id, shape, shape[0] * shape[1], id};
  }
};

TEST(PerfProfiles, ParsesExactCaseFilterAndReverseOrder) {
  perf::Options options;
  std::string program = "perf_fixture";
  std::string flag = "--cases";
  std::string ids = "baseline-small,baseline-medium";
  std::string reverse = "--reverse-order";
  std::vector<char*> argv{program.data(), flag.data(), ids.data(), reverse.data()};
  EXPECT_TRUE(perf::detail::parse_args<AdapterFixtureTraits>(static_cast<int>(argv.size()),
                                                             argv.data(), &options));
  EXPECT_EQ(options.cases, (std::vector<std::string>{"baseline-small", "baseline-medium"}));
  EXPECT_TRUE(options.reverse_order);
}

perf::ResultRow base_row(const AdapterFixtureScenario& scenario) {
  perf::ResultRow row;
  row.op_name = "FixtureOp";
  row.live_port = scenario.live ? "method" : "";
  return row;
}

perf::TimingCollector run_graph(const AdapterFixtureScenario&, const perf::RunPlan& plan) {
  perf::TimingCollector collector;
  if (plan.has_override())
    collector.live_emissions.push_back(plan.override_at);
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
  const auto suite = perf::load_profile_suite(profile_path("clutter_filter"), "clutter_filter");
  const AdapterFixtureScenario scenario{"fixture-live", "small", true};
  const perf::SelectedScenario<AdapterFixtureScenario> selected{scenario, "method"};
  perf::PerfReport report;
  perf::detail::record_exercised_sizes<AdapterFixtureTraits>(suite, {selected}, &report);
  ASSERT_EQ(report.sizes.size(), 1U);
  EXPECT_EQ(report.sizes[0].id, "small");
  EXPECT_EQ(report.sizes[0].shape, suite.shapes[0]);

  perf::RunSettings settings;
  settings.warmup_frames = 1;
  settings.measured_frames = 3;
  settings.repeats = 2;
  perf::detail::run_adapter_case<AdapterFixtureTraits>(selected, settings, &report);
  ASSERT_FALSE(report.results.empty());
  EXPECT_EQ(report.results.back().override_context, "first_changed_configuration_after_baseline");
  ASSERT_EQ(report.results.size(), 2U);
  EXPECT_EQ(report.results[0].scenario_id, scenario.id);
  EXPECT_EQ(report.results[1].size_id, scenario.size_id);
  EXPECT_EQ(report.results[1].live_port, "method");
  EXPECT_EQ(report.results[1].latency.samples, 6);
  EXPECT_NEAR(report.results[1].override_delta_ms, 0.1, 1e-9);
  EXPECT_EQ(report.results[1].notes, "typed adapter");
  const perf::SelectedScenario<AdapterFixtureScenario> missing_live{scenario, ""};
  EXPECT_THROW(
      perf::detail::run_adapter_case<AdapterFixtureTraits>(missing_live, settings, &report),
      std::invalid_argument);
}

TEST(PerfProfiles, SchemaV2KeepsNestedDescriptorFieldsWhenCasesOverrideOneField) {
  const auto suite =
      perf::load_profile_suite(profile_path("tdm_dense_compensation"), "tdm_dense_compensation");
  const auto& changed = suite.cases[3];
  const auto merged = perf::merge_recipe_map(suite.recipe["params"], changed.parameters);
  const auto waveform = merged["waveform"];
  EXPECT_EQ(waveform["factory"].as<std::string>(), "valid_fmcw_waveform");
  EXPECT_DOUBLE_EQ(waveform["pri_seconds"].as<double>(), 0.00003);
  EXPECT_DOUBLE_EQ(waveform["carrier_frequency"].as<double>(), 77.0e9);
  EXPECT_NO_THROW((void)perf::recipe_fmcw_waveform(waveform));
}

TEST(PerfProfiles, SchemaV2DeclaresPairedAndDescriptorOutputs) {
  const auto reduction =
      perf::load_profile_suite(profile_path("detection_reduction"), "detection_reduction");
  const auto reduction_recipe = perf::parse_yaml_recipe(reduction);
  EXPECT_EQ(reduction_recipe.inputs.size(), 3U);
  EXPECT_EQ(reduction_recipe.outputs.timed, "rep_idx");
  EXPECT_EQ(reduction_recipe.outputs.timed_pair, "reduced_points");

  const auto sparse =
      perf::load_profile_suite(profile_path("ddm_sparse_demultiplex"), "ddm_sparse_demultiplex");
  const auto sparse_recipe = perf::parse_yaml_recipe(sparse);
  EXPECT_EQ(sparse_recipe.outputs.drain_codecs.at("array_geometry"), "array_geometry");
  EXPECT_EQ(sparse_recipe.inputs[1].index_doppler_divisor, "tx");
}

TEST(PerfProfiles, SchemaV2RoundsSixInputFixtureAndKeepsEmptyDetectionCase) {
  const auto expansion =
      perf::load_profile_suite(profile_path("multi_source_expansion"), "multi_source_expansion");
  const auto baseline = perf::recipe_values(expansion, expansion.cases[0]);
  EXPECT_EQ(baseline.at("selected"), 8);
  const auto full = perf::recipe_values(expansion, expansion.cases.back());
  EXPECT_EQ(full.at("selected"), 16384);
  EXPECT_EQ(perf::parse_yaml_recipe(expansion).inputs.size(), 6U);

  const auto feature =
      perf::load_profile_suite(profile_path("feature_assembly"), "feature_assembly");
  const auto empty = perf::recipe_values(feature, feature.cases[5]);
  EXPECT_EQ(empty.at("detections"), 0);
  const auto recipe = perf::parse_yaml_recipe(feature);
  EXPECT_EQ(perf::recipe_shape(recipe.inputs[0], empty), (std::vector<int64_t>{0, 2}));
}

}  // namespace
