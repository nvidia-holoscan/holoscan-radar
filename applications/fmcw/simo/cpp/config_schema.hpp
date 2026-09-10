// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/params/config.hpp>

#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::radar::app::simo {

struct CubeConfig {
  int64_t num_chirps{0};
  int64_t samples_per_chirp{0};
};

struct RunConfig {
  int64_t num_frames{1};
  double frame_period{0.1};
  bool meter_cadence{true};
  int64_t meter_log_every{0};
};

struct VisualizationConfig {
  bool enabled{true};
  int64_t window_width{1280};
  int64_t window_height{720};
  std::string window_title{"FMCW SIMO"};
  double display_rate_hz{60.0};
  double dynamic_range_db{60.0};
  double detection_radius{0.006};
  double representative_size{0.014};
};

struct RxArrayConfig {
  int64_t az_elements{0};
  int64_t el_elements{0};
  double spacing{0.0};
};

struct TargetConfig {
  double range{0.0};
  double azimuth{0.0};
  double elevation{0.0};
  std::array<double, 3> velocity{};
  double amplitude{0.0};
};

using CubeStorage = radar::config_detail::PortableParamStorageTraits<
    CubeConfig>;  // NOLINT
using RunStorage = radar::config_detail::PortableParamStorageTraits<
    RunConfig>;  // NOLINT
using VisualizationStorage = radar::config_detail::PortableParamStorageTraits<
    VisualizationConfig>;  // NOLINT
using RxArrayStorage = radar::config_detail::PortableParamStorageTraits<
    RxArrayConfig>;  // NOLINT
using TargetStorage = radar::config_detail::PortableParamStorageTraits<
    TargetConfig>;  // NOLINT

}  // namespace holoscan::radar::app::simo

namespace holoscan {

template <>
struct ParamStorageTraits<radar::app::simo::CubeConfig>
    : radar::app::simo::CubeStorage {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::simo::RunConfig>
    : radar::app::simo::RunStorage {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::simo::VisualizationConfig>
    : radar::app::simo::VisualizationStorage {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::simo::RxArrayConfig>
    : radar::app::simo::RxArrayStorage {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::simo::TargetConfig>
    : radar::app::simo::TargetStorage {};  // NOLINT

}  // namespace holoscan

namespace holoscan::radar::app::simo {

struct AppParams {
  FMCWWaveformDescriptor waveform;
  CubeConfig cube;
  RunConfig run;
  VisualizationConfig visualization;
  RxArrayConfig rx_array;
  std::vector<TargetConfig> targets;
  std::optional<double> noise_snr_db{};
};

HOLOSCAN_PARAM_SCHEMA(
    AppParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.app.simo.AppParams", 1),
    holoscan::field<&AppParams::waveform>("waveform"),
    holoscan::field<&AppParams::cube>("cube"),
    holoscan::field<&AppParams::run>("run"),
    holoscan::field<&AppParams::visualization>("visualization"),
    holoscan::field<&AppParams::rx_array>("rx_array"),
    holoscan::field<&AppParams::targets>("targets"),
    holoscan::field<&AppParams::noise_snr_db>("noise_snr_db"));

static_assert(holoscan::param_schema_availability<AppParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in AppParams's namespace");

}  // namespace holoscan::radar::app::simo

namespace holoscan::config {

using AppParamsCodec =
    params::DefaultedParamSchemaCodec<radar::app::simo::AppParams>;  // NOLINT

template <>
struct ConfigCodec<radar::app::simo::CubeConfig>
    : ObjectCodec<radar::app::simo::CubeConfig> {  // NOLINT
  using Config = radar::app::simo::CubeConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("num_chirps", &Config::num_chirps),
      object_field("samples_per_chirp", &Config::samples_per_chirp));
};

template <>
struct ConfigCodec<radar::app::simo::RunConfig>
    : ObjectCodec<radar::app::simo::RunConfig> {  // NOLINT
  using Config = radar::app::simo::RunConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("num_frames", &Config::num_frames),
      object_field("frame_period", &Config::frame_period),
      defaulted_object_field("meter_cadence", &Config::meter_cadence),
      defaulted_object_field("meter_log_every", &Config::meter_log_every));
};

template <>
struct ConfigCodec<radar::app::simo::VisualizationConfig>
    : ObjectCodec<radar::app::simo::VisualizationConfig> {  // NOLINT
  using Config = radar::app::simo::VisualizationConfig;
  static constexpr auto fields = std::make_tuple(
      defaulted_object_field("enabled", &Config::enabled),
      defaulted_object_field("window_width", &Config::window_width),
      defaulted_object_field("window_height", &Config::window_height),
      defaulted_object_field("window_title", &Config::window_title),
      defaulted_object_field("display_rate_hz", &Config::display_rate_hz),
      defaulted_object_field("dynamic_range_db", &Config::dynamic_range_db),
      defaulted_object_field("detection_radius", &Config::detection_radius),
      defaulted_object_field("representative_size",
                             &Config::representative_size));
};

template <>
struct ConfigCodec<radar::app::simo::RxArrayConfig>
    : ObjectCodec<radar::app::simo::RxArrayConfig> {  // NOLINT
  using Config = radar::app::simo::RxArrayConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("az_elements", &Config::az_elements),
      object_field("el_elements", &Config::el_elements),
      object_field("spacing", &Config::spacing));
};

template <>
struct ConfigCodec<radar::app::simo::TargetConfig>
    : ObjectCodec<radar::app::simo::TargetConfig> {  // NOLINT
  using Config = radar::app::simo::TargetConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("range", &Config::range),
      object_field("azimuth", &Config::azimuth),
      object_field("elevation", &Config::elevation),
      object_field("velocity", &Config::velocity),
      object_field("amplitude", &Config::amplitude));
};

template <>
struct ConfigCodec<radar::app::simo::AppParams>
    : AppParamsCodec {};  // NOLINT

}  // namespace holoscan::config
