// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <span>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radar_app/fmcw/mimo_config.hpp"
#include "radar_app/app/presentation_choice.hpp"
#include "radar_app/app/config_path.hpp"
#include "radar_common/descriptors/window_spec.hpp"

#include "app/config_fixture.hpp"

namespace {

using holoscan::radar::testing::PresetOverride;
using holoscan::config::ConfigValue;
using holoscan::radar::testing::canonical;
using holoscan::radar::testing::resolved_document;
using holoscan::radar::testing::with_value;

namespace mimo = holoscan::radar::app::mimo;
namespace radar = holoscan::radar;

TEST(FmcwMimoConfig, RejectsOverflowingSourceProductsBeforeBuildingGeometry) {
  constexpr int64_t huge = std::numeric_limits<int64_t>::max();
  for (bool tdm : {false, true}) {
    mimo::AppParams params;
    params.tx_array.num_tx = 2;
    params.tx_array.spacing = 0.5;
    params.rx_array.az_elements = 2;
    params.rx_array.el_elements = 2;
    params.rx_array.spacing = 0.5;
    params.cube.num_chirps = huge;
    params.cube.samples_per_chirp = 8;
    EXPECT_THROW(mimo::make_runtime_config(params, "overflow test", tdm), std::invalid_argument);
    params.cube.num_chirps = 8;
    params.rx_array.az_elements = huge;
    EXPECT_THROW(mimo::make_runtime_config(params, "overflow test", tdm), std::invalid_argument);
  }
}

TEST(FmcwMimoConfig, RejectsOverflowingSnapshotExtents) {
  constexpr int64_t huge = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(mimo::snapshot_count({huge}), std::invalid_argument);
  EXPECT_THROW(mimo::snapshot_count({huge / 2, 2}), std::invalid_argument);
  EXPECT_THROW(mimo::snapshot_count({-1}), std::invalid_argument);
  EXPECT_EQ(mimo::snapshot_count({2, 3}), 35);
}

std::filesystem::path config_path(const char* name) {
  return std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) / "applications/fmcw" / name;
}

void expect_smoke_matches_production(const char* production_name, const char* smoke_name) {
  auto expected = resolved_document(config_path(production_name));
  const auto smoke = resolved_document(config_path(smoke_name));
  expected = with_value(expected, "/app/run/num_frames", ConfigValue::signed_integer(2));
  expected = with_value(expected, "/app/visualization/enabled", ConfigValue::boolean(false));
  expected = with_value(expected, "/app/range_loss", ConfigValue::null());
  const ConfigValue first_target = expected.at("/app/targets/0").to_value();
  expected = with_value(expected, "/app/targets",
                        ConfigValue::array(std::span<const ConfigValue>(&first_target, 1)));
  EXPECT_EQ(canonical(expected), canonical(smoke));
}

class MutatedConfig {
 public:
  MutatedConfig(const char* source_name, const std::string& old_text, const std::string& new_text)
  : MutatedConfig(source_name,
                  std::vector<std::pair<std::string, std::string>>{{old_text, new_text}}) {}

  MutatedConfig(const char* source_name,
                const std::vector<std::pair<std::string, std::string>>& replacements) {
    std::ifstream input(config_path(source_name));
    if (!input) {
      throw std::runtime_error("cannot read source config");
    }
    std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    for (const auto& [old_text, new_text] : replacements) {
      const std::size_t position = contents.find(old_text);
      if (position == std::string::npos ||
          contents.find(old_text, position + old_text.size()) != std::string::npos) {
        throw std::runtime_error("config mutation anchor must occur once");
      }
      contents.replace(position, old_text.size(), new_text);
    }
    path_ = std::filesystem::temp_directory_path() /
            ("holoscan_radar_mimo_config_" + std::to_string(next_id_++) + ".yaml");
    std::ofstream output(path_);
    output << contents;
    if (!output) {
      throw std::runtime_error("cannot write temporary config");
    }
  }

  ~MutatedConfig() { std::filesystem::remove(path_); }

  MutatedConfig(const MutatedConfig&) = delete;
  MutatedConfig& operator=(const MutatedConfig&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  inline static int64_t next_id_{0};
  std::filesystem::path path_;
};

void expect_common_geometry_and_tail(const mimo::RuntimeConfig& config, int64_t max_az,
                                     int64_t max_clusters) {
  EXPECT_EQ(config.array_geometry.num_elements(), 32);
  EXPECT_EQ(config.extents.max_tx, 2);
  EXPECT_EQ(config.extents.max_rx, 16);
  EXPECT_EQ(config.extents.max_virtual_channels, 32);
  EXPECT_EQ(config.extents.max_channels, 32);
  EXPECT_EQ(config.extents.max_features, 7);
  EXPECT_EQ(config.extents.max_snapshots, 1);
  EXPECT_EQ(config.extents.max_az, max_az);
  EXPECT_EQ(config.extents.max_el, max_az);
  EXPECT_EQ(config.extents.max_clusters, max_clusters);
  EXPECT_EQ(config.report_index_columns, (std::vector<int64_t>{6, 5}));
}

TEST(FmcwMimoConfig, DecodesTdmDefault) {
  const auto config = mimo::load_tdm_config(config_path("mimo_tdm/config/mimo_tdm.yaml"));

  EXPECT_EQ(config.num_frames, 250);
  EXPECT_DOUBLE_EQ(config.visualization.range_doppler_width_percent, 45.0);
  EXPECT_DOUBLE_EQ(config.visualization.noise_headroom_db, 6.0);
  EXPECT_EQ(config.scene.range_loss.model, radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 100.0);
  EXPECT_EQ(config.scene_visibility.model, radar::sim::SceneVisibilityModel::kFrontHalfSpace);
  EXPECT_EQ(config.tx_schedule.num_tx, 2);
  EXPECT_EQ(config.tx_schedule.tx_order, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.deinterleave_params->tx_schedule, config.tx_schedule);
  EXPECT_EQ(config.compensation_params->tx_schedule, config.tx_schedule);
  EXPECT_EQ(config.compensation_params->tx_positions, config.tx_positions);
  EXPECT_EQ(config.compensation_params->rx_positions, config.rx_positions);
  EXPECT_EQ(config.compensation_params->overlap_policy, holoscan::radar::OverlapPolicy::kKeepAll);
  EXPECT_EQ(config.compensation_params->snapshot_window, (std::vector<int64_t>{0, 0}));
  EXPECT_EQ(config.compensation_params->doppler_edge, holoscan::radar::SnapshotEdgeMode::kWrap);
  EXPECT_EQ(config.compensation_params->range_edge, holoscan::radar::SnapshotEdgeMode::kShift);
  EXPECT_DOUBLE_EQ(config.compensation_params->waveform.chirp_repetition_interval(), 50.0e-6);
  EXPECT_DOUBLE_EQ(config.scene.waveform.chirp_repetition_interval, 25.0e-6);
  EXPECT_EQ(config.clutter_params->method, holoscan::radar::ClutterFilterMethod::kMeanSubtraction);
  EXPECT_EQ(config.clutter_params->doppler_window.type, holoscan::radar::WindowType::kHann);
  EXPECT_EQ(config.clutter_params->doppler_window, config.doppler_params->doppler_window);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.cfar_params->nci_fold, 32);
  EXPECT_EQ(config.beamscan_params->array_geometry, config.array_geometry);
  EXPECT_EQ(config.extents.max_chirps, 64);
  ASSERT_TRUE(config.extents.max_interleaved_chirps.has_value());
  EXPECT_EQ(*config.extents.max_interleaved_chirps, 128);
  EXPECT_EQ(config.extents.max_doppler_bins, 64);
  EXPECT_EQ(config.extents.max_detections, 64 * 256);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesX11PresentationSwitch) {
  EXPECT_FALSE(mimo::load_tdm_config(config_path("mimo_tdm/config/mimo_tdm.yaml"))
                   .visualization.x11_presentation.has_value());
  const MutatedConfig file("mimo_tdm/config/mimo_tdm.yaml", "display_rate_hz: 20.0",
                           "display_rate_hz: 20.0\n        x11_presentation: false");
  EXPECT_EQ(mimo::load_tdm_config(file.path()).visualization.x11_presentation, false);
}

radar::app::detail::PresentationChoice presenter_choice(const std::filesystem::path& path) {
  return {.window = true,
          .x11_available = true,
          .interactive = true,
          .configured = std::nullopt,
          .cache_file = path,
          .setup_key = "123",
          .choose_again = false,
          .now_seconds = 1700000000,
          .config_path = "custom_preset.yaml"};
}

TEST(FmcwPresentationChoice, DefaultsToThisRunOnly) {
  holoscan::radar::testing::ConfigDirectory directory;
  const auto path = directory.write("choice", "invalid\n");
  std::filesystem::remove(path);
  const auto choice = presenter_choice(path);
  std::istringstream input("n\n\n");
  std::ostringstream output;
  EXPECT_FALSE(radar::app::detail::choose_x11_presentation(choice, input, output));
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_NE(output.str().find("This run only (default)"), std::string::npos);
  EXPECT_NE(output.str().find("in \"custom_preset.yaml\"."), std::string::npos);
}

TEST(FmcwPresentationChoice, SevenDayChoiceExpiresAtBoundary) {
  holoscan::radar::testing::ConfigDirectory directory;
  auto choice = presenter_choice(directory.write("choice", "invalid\n"));
  std::istringstream input("unknown\n n \ninvalid\n2\n");
  std::ostringstream output;
  EXPECT_FALSE(radar::app::detail::choose_x11_presentation(choice, input, output));
  EXPECT_NE(output.str().find("Please choose"), std::string::npos);
  const auto saved = radar::app::detail::read_presentation(choice);
  ASSERT_TRUE(saved);
  EXPECT_EQ(saved->expires_at, choice.now_seconds + 604800);

  choice.interactive = false;
  choice.now_seconds += 604799;
  std::istringstream no_input;
  output.str("");
  EXPECT_FALSE(radar::app::detail::choose_x11_presentation(choice, no_input, output));
  EXPECT_NE(output.str().find("expires"), std::string::npos);
  EXPECT_NE(output.str().find("--choose-presenter"), std::string::npos);
  ++choice.now_seconds;
  EXPECT_FALSE(radar::app::detail::read_presentation(choice));
  EXPECT_TRUE(radar::app::detail::choose_x11_presentation(choice, no_input, output));
}

TEST(FmcwPresentationChoice, IndefiniteChoiceSurvivesAndExplicitConfigWins) {
  holoscan::radar::testing::ConfigDirectory directory;
  auto choice = presenter_choice(directory.write("choice", "invalid\n"));
  std::istringstream input("n\n3\n");
  std::ostringstream output;
  EXPECT_FALSE(radar::app::detail::choose_x11_presentation(choice, input, output));
  choice.now_seconds += 100 * 604800;
  choice.interactive = false;
  std::istringstream no_input;
  output.str("");
  EXPECT_FALSE(radar::app::detail::choose_x11_presentation(choice, no_input, output));
  EXPECT_NE(output.str().find("until changed"), std::string::npos);
  choice.configured = true;
  output.str("");
  EXPECT_TRUE(radar::app::detail::choose_x11_presentation(choice, no_input, output));
  EXPECT_TRUE(output.str().empty());
  choice.configured = false;
  choice.choose_again = true;
  EXPECT_FALSE(radar::app::detail::choose_x11_presentation(choice, no_input, output));
  EXPECT_NE(output.str().find("takes priority"), std::string::npos);
  EXPECT_TRUE(radar::app::detail::read_presentation(choice));
}

TEST(FmcwPresentationChoice, RejectsWrongSetupClockRollbackAndInvalidMetadata) {
  holoscan::radar::testing::ConfigDirectory directory;
  auto choice = presenter_choice(directory.write("choice", "invalid\n"));
  std::ostringstream output;
  radar::app::detail::remember_presentation(choice, false, true, output);
  choice.setup_key = "456";
  EXPECT_FALSE(radar::app::detail::read_presentation(choice));
  choice.setup_key = "123";
  --choice.now_seconds;
  EXPECT_FALSE(radar::app::detail::read_presentation(choice));
  ++choice.now_seconds;
  const std::vector<std::string> invalid{
      "native\n",
      "{broken",
      "{}",
      R"({"version":1,"presenter":"native","setup_key":"123","saved_at":1,"expires_at":null})",
      R"({"version":2,"presenter":"other","setup_key":"123","saved_at":1,"expires_at":null})",
      R"({"version":2,"presenter":"native","setup_key":"123","saved_at":0,"expires_at":null})",
      R"({"version":2,"presenter":"native","setup_key":"123","saved_at":1,"expires_at":0})",
      R"({"version":2,"presenter":"native","setup_key":"123","saved_at":1,"expires_at":"never"})",
      R"({"version":2,"presenter":"native","setup_key":"123","saved_at":1})"};
  for (const auto& text : invalid) {
    SCOPED_TRACE(text);
    choice.cache_file = directory.write("choice", text);
    EXPECT_FALSE(radar::app::detail::read_presentation(choice));
  }
}

TEST(FmcwPresentationChoice, ChooseAgainClearsOldPreferenceEvenForOneRunChoice) {
  holoscan::radar::testing::ConfigDirectory directory;
  auto choice = presenter_choice(directory.write("choice", "invalid\n"));
  std::ostringstream output;
  radar::app::detail::remember_presentation(choice, false, true, output);
  choice.choose_again = true;
  choice.interactive = false;
  std::istringstream input("x\n1\n");
  EXPECT_THROW(radar::app::detail::choose_x11_presentation(choice, input, output),
               std::runtime_error);
  EXPECT_TRUE(radar::app::detail::read_presentation(choice));
  choice.interactive = true;
  EXPECT_TRUE(radar::app::detail::choose_x11_presentation(choice, input, output));
  EXPECT_FALSE(std::filesystem::exists(choice.cache_file));
}

TEST(FmcwPresentationChoice, EofAndUnattendedRunsDoNotSavePreferences) {
  holoscan::radar::testing::ConfigDirectory directory;
  const auto path = directory.write("choice", "invalid\n");
  std::filesystem::remove(path);
  auto choice = presenter_choice(path);
  std::istringstream empty;
  std::ostringstream output;
  EXPECT_TRUE(radar::app::detail::choose_x11_presentation(choice, empty, output));
  std::istringstream native("n\n");
  EXPECT_FALSE(radar::app::detail::choose_x11_presentation(choice, native, output));
  choice.interactive = false;
  EXPECT_TRUE(radar::app::detail::choose_x11_presentation(choice, empty, output));
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(FmcwPresentationChoice, NonvisualAndUnavailableX11NeverPromptOrReadCache) {
  holoscan::radar::testing::ConfigDirectory directory;
  auto choice = presenter_choice(directory.write("choice", "invalid\n"));
  std::ostringstream output;
  radar::app::detail::remember_presentation(choice, false, true, output);
  choice.window = false;
  std::istringstream input("n\n");
  EXPECT_TRUE(radar::app::detail::choose_x11_presentation(choice, input, output));
  choice.window = true;
  choice.x11_available = false;
  EXPECT_TRUE(radar::app::detail::choose_x11_presentation(choice, input, output));
  EXPECT_TRUE(output.str().empty());
  EXPECT_EQ(input.peek(), 'n');
}

TEST(FmcwPresentationChoice, CacheWriteFailureKeepsSelectedPresenter) {
  holoscan::radar::testing::ConfigDirectory directory;
  const auto choice = presenter_choice(directory.write("blocking-file", "data") / "choice");
  std::istringstream input("n\n2\n");
  std::ostringstream output;
  EXPECT_FALSE(radar::app::detail::choose_x11_presentation(choice, input, output));
  EXPECT_NE(output.str().find("Could not remember"), std::string::npos);
}

TEST(FmcwPresentationChoice, ParsesChoosePresenterBeforeOrAfterConfig) {
  for (auto args : {std::vector<std::string>{"/tmp/radar", "--choose-presenter", "custom.yaml"},
                    std::vector<std::string>{"/tmp/radar", "custom.yaml", "--choose-presenter"},
                    std::vector<std::string>{"/tmp/radar", "--choose-presenter"}}) {
    std::vector<char*> argv;
    for (auto& arg : args)
      argv.push_back(arg.data());
    const auto launch =
        radar::app::parse_fmcw_launch_options(argv.size(), argv.data(), "default.yaml");
    EXPECT_TRUE(launch.choose_presenter);
    EXPECT_EQ(launch.config_path.filename(), args.size() == 3U ? "custom.yaml" : "default.yaml");
  }
}

TEST(FmcwMimoConfig, DecodesDashboardWidthAndRejectsInvalidPercentages) {
  for (const double percent : {25.0, 37.5, 70.0}) {
    const MutatedConfig file(
        "mimo_tdm/config/mimo_tdm.yaml", "display_rate_hz: 20.0",
        "display_rate_hz: 20.0\n        range_doppler_width_percent: " + std::to_string(percent));
    const auto config = mimo::load_tdm_config(file.path());
    EXPECT_DOUBLE_EQ(config.visualization.range_doppler_width_percent, percent);
  }
  for (const auto* percent : {"24.9", "70.1", ".nan", ".inf"}) {
    SCOPED_TRACE(percent);
    const MutatedConfig file(
        "mimo_tdm/config/mimo_tdm.yaml", "display_rate_hz: 20.0",
        std::string("display_rate_hz: 20.0\n        range_doppler_width_percent: ") + percent);
    EXPECT_THROW(mimo::load_tdm_config(file.path()), std::runtime_error);
  }
}

TEST(FmcwMimoConfig, DecodesBackgroundScalingAndRejectsInvalidSettings) {
  const MutatedConfig file(
      "mimo_tdm/config/mimo_tdm.yaml", "display_rate_hz: 20.0",
      "display_rate_hz: 20.0\n        noise_scaling: false\n        noise_sample_limit: 512"
      "\n        noise_range_half_width: 2\n        noise_doppler_half_width: 4"
      "\n        noise_sigma: 1.5\n        noise_headroom_db: 12.5");
  const auto config = mimo::load_tdm_config(file.path());
  EXPECT_FALSE(config.visualization.noise_scaling);
  EXPECT_EQ(config.visualization.noise_sample_limit, 512);
  EXPECT_EQ(config.visualization.noise_range_half_width, 2);
  EXPECT_EQ(config.visualization.noise_doppler_half_width, 4);
  EXPECT_DOUBLE_EQ(config.visualization.noise_sigma, 1.5);
  EXPECT_DOUBLE_EQ(config.visualization.noise_headroom_db, 12.5);
  for (const auto* setting :
       {"noise_sample_limit: 63", "noise_sample_limit: 65537", "noise_range_half_width: -1",
        "noise_doppler_half_width: 65", "noise_sigma: .nan", "noise_sigma: 10.1",
        "noise_headroom_db: -0.1", "noise_headroom_db: .nan", "noise_headroom_db: .inf"}) {
    SCOPED_TRACE(setting);
    const MutatedConfig invalid("mimo_tdm/config/mimo_tdm.yaml", "display_rate_hz: 20.0",
                                std::string("display_rate_hz: 20.0\n        ") + setting);
    EXPECT_THROW(mimo::load_tdm_config(invalid.path()), std::runtime_error);
  }
}

TEST(FmcwMimoConfig, DecodesTdmSmoke) {
  const auto config = mimo::load_tdm_config(config_path("mimo_tdm/config/mimo_tdm_smoke.yaml"));

  EXPECT_EQ(config.num_frames, 2);
  EXPECT_FALSE(config.visualize);
  EXPECT_EQ(config.tx_schedule.num_tx, 2);
  EXPECT_EQ(config.tx_schedule.tx_order, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.deinterleave_params->tx_schedule, config.tx_schedule);
  EXPECT_EQ(config.compensation_params->tx_schedule, config.tx_schedule);
  EXPECT_EQ(config.compensation_params->overlap_policy, holoscan::radar::OverlapPolicy::kKeepAll);
  EXPECT_EQ(config.compensation_params->snapshot_window, (std::vector<int64_t>{0, 0}));
  EXPECT_DOUBLE_EQ(config.scene.waveform.chirp_repetition_interval, 25.0e-6);
  EXPECT_EQ(config.clutter_params->method, holoscan::radar::ClutterFilterMethod::kMeanSubtraction);
  EXPECT_EQ(config.clutter_params->doppler_window, config.doppler_params->doppler_window);
  EXPECT_EQ(config.doppler_params->doppler_window.type, holoscan::radar::WindowType::kHann);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.cfar_params->nci_fold, 32);
  EXPECT_EQ(config.extents.max_chirps, 64);
  ASSERT_TRUE(config.extents.max_interleaved_chirps.has_value());
  EXPECT_EQ(*config.extents.max_interleaved_chirps, 128);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesLinearDdmDefault) {
  const auto config = mimo::load_ddm_config(config_path("mimo_ddm/config/mimo_ddm_linear.yaml"));

  EXPECT_FALSE(config.correlate);
  EXPECT_EQ(config.scene.range_loss.model, radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 100.0);
  EXPECT_EQ(config.scene_visibility.model, radar::sim::SceneVisibilityModel::kFrontHalfSpace);
  EXPECT_EQ(config.ddm_code.num_tx, 2);
  EXPECT_EQ(config.doppler_offsets, (std::vector<double>{0.0, 0.5}));
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_DOUBLE_EQ(config.notch_params->center_velocity, 0.0);
  EXPECT_DOUBLE_EQ(config.notch_params->stop_half_width, 0.61);
  EXPECT_DOUBLE_EQ(config.notch_params->transition_width, 1.22);
  EXPECT_TRUE(config.notch_params->cyclic_velocity_axis);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0}));
  EXPECT_EQ(config.cfar_params->nci_fold, 32);
  EXPECT_EQ(config.beamscan_params->array_geometry, config.array_geometry);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 60 * 256);
  EXPECT_FALSE(config.extents.max_interleaved_chirps.has_value());
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesLinearDdmSmoke) {
  const auto config =
      mimo::load_ddm_config(config_path("mimo_ddm/config/mimo_ddm_linear_smoke.yaml"));

  EXPECT_FALSE(config.correlate);
  EXPECT_FALSE(config.visualize);
  EXPECT_EQ(config.doppler_offsets, (std::vector<double>{0.0, 0.5}));
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_DOUBLE_EQ(config.notch_params->stop_half_width, 0.61);
  EXPECT_DOUBLE_EQ(config.notch_params->transition_width, 1.22);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 60 * 256);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesDocumentedNonlinearDdmPreset) {
  const auto config = mimo::load_ddm_config(config_path("mimo_ddm/config/mimo_ddm_nonlinear.yaml"));

  EXPECT_TRUE(config.correlate);
  EXPECT_EQ(config.scene.range_loss.model, radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 100.0);
  EXPECT_EQ(config.scene_visibility.model, radar::sim::SceneVisibilityModel::kFrontHalfSpace);
  EXPECT_TRUE(config.doppler_offsets.empty());
  EXPECT_EQ(config.ddm_code.num_tx, 2);
  EXPECT_EQ(config.ddm_code.phase_codes.size(), 2U * 128U);
  EXPECT_EQ(config.ddm_code.guard_bands, 0);
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_DOUBLE_EQ(config.notch_params->stop_half_width, 0.61);
  EXPECT_DOUBLE_EQ(config.notch_params->transition_width, 1.22);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 128 * 256);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesNonlinearDdmGeneratorAliases) {
  for (const char* generator : {"zc", "sin"}) {
    const PresetOverride preset(config_path("mimo_ddm/config/mimo_ddm_nonlinear_smoke.yaml"),
                                "ddm:\n    generate_phase_codes: " + std::string(generator));
    const auto config = mimo::load_ddm_config(preset.path());

    EXPECT_TRUE(config.correlate);
    EXPECT_EQ(config.ddm_code.num_tx, 2);
    EXPECT_EQ(config.ddm_code.phase_codes.size(), 2U * 128U);
    EXPECT_TRUE(config.ddm_code.doppler_offsets.empty());
    EXPECT_EQ(config.ddm_code.guard_bands, 0);
  }
}

TEST(FmcwMimoConfig, SmokeConfigsOnlyApplyAllowedOverrides) {
  expect_smoke_matches_production("mimo_tdm/config/mimo_tdm.yaml",
                                  "mimo_tdm/config/mimo_tdm_smoke.yaml");
  expect_smoke_matches_production("mimo_ddm/config/mimo_ddm_linear.yaml",
                                  "mimo_ddm/config/mimo_ddm_linear_smoke.yaml");
  expect_smoke_matches_production("mimo_ddm/config/mimo_ddm_nonlinear.yaml",
                                  "mimo_ddm/config/mimo_ddm_nonlinear_smoke.yaml");
}

TEST(FmcwMimoConfig, RejectsWrongLoaderSection) {
  EXPECT_THROW(mimo::load_tdm_config(config_path("mimo_ddm/config/mimo_ddm_linear.yaml")),
               std::runtime_error);
  EXPECT_THROW(mimo::load_ddm_config(config_path("mimo_tdm/config/mimo_tdm.yaml")),
               std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsAuthoredDeinterleaveSchedule) {
  const PresetOverride config(config_path("mimo_tdm/config/mimo_tdm_smoke.yaml"),
                              "mimo_deinterleave: {tx_schedule: {num_tx: 2}}\n");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsAuthoredDerivedSparseFields) {
  for (const auto* field : {"waveform: {}", "tx_schedule: {num_tx: 2}",
                            "tx_positions: [0.0, 0.0, 0.0]", "rx_positions: [0.0, 0.0, 0.0]"}) {
    const PresetOverride config(config_path("mimo_tdm/config/mimo_tdm_smoke.yaml"),
                                "tdm_sparse_compensation:\n    " + std::string(field));
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
}

TEST(FmcwMimoConfig, RejectsTdmPriMismatch) {
  const PresetOverride config(config_path("mimo_tdm/config/mimo_tdm_smoke.yaml"),
                              "app:\n    waveform:\n        chirp_repetition_interval: 51.0e-6\n");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsDdmTxMismatch) {
  const PresetOverride config(config_path("mimo_ddm/config/mimo_ddm_linear_smoke.yaml"),
                              "ddm_dense_demultiplex:\n    ddm_code:\n        num_tx: 3\n");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsDdmClutterFilter) {
  const PresetOverride config(config_path("mimo_ddm/config/mimo_ddm_linear_smoke.yaml"),
                              "clutter_filter:\n    method: mean_subtraction\n");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsMalformedSchemeSpecificConfig) {
  const PresetOverride config(config_path("mimo_ddm/config/mimo_ddm_linear.yaml"),
                              "ddm:\n    generate_phase_codes: zadoff_chu\n");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsInvalidClusteredTailShape) {
  const PresetOverride config(config_path("mimo_tdm/config/mimo_tdm_smoke.yaml"),
                              "feature_assembly:\n    features: [range, velocity, az_estimate, "
                              "el_estimate, score, doppler_bin, range_bin]\n");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, DecodesSharedRangeLossBlock) {
  const PresetOverride preset(
      config_path("mimo_tdm/config/mimo_tdm_smoke.yaml"),
      "app:\n    range_loss: {model: monostatic_free_space, reference_range: 40.0}\n");
  const auto config = mimo::load_tdm_config(preset.path());
  EXPECT_EQ(config.scene.range_loss.model, radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 40.0);
}

}  // namespace
