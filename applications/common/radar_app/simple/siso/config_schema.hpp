// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/params/config.hpp>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "radar_app/app/visualization_config.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::radar::app::siso {

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

struct TargetConfig {
  double range{0.0};
  double velocity{0.0};
  double amplitude{0.0};
};

struct FeatureAssemblyParams {
  std::vector<AssemblyFeature> features{};
};

using CubeStorage = radar::config_detail::PortableParamStorageTraits<
    CubeConfig>;  // NOLINT
using RunStorage = radar::config_detail::PortableParamStorageTraits<
    RunConfig>;  // NOLINT
using TargetStorage = radar::config_detail::PortableParamStorageTraits<
    TargetConfig>;  // NOLINT
using FeatureAssemblyStorage = radar::config_detail::PortableParamStorageTraits<
    FeatureAssemblyParams>;  // NOLINT

}  // namespace holoscan::radar::app::siso

namespace holoscan {

template <>
struct ParamStorageTraits<radar::app::siso::CubeConfig>
    : radar::app::siso::CubeStorage {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::siso::RunConfig>
    : radar::app::siso::RunStorage {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::siso::TargetConfig>
    : radar::app::siso::TargetStorage {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::siso::FeatureAssemblyParams>
    : radar::app::siso::FeatureAssemblyStorage {};  // NOLINT

}  // namespace holoscan

namespace holoscan::radar::app::siso {

inline ::holoscan::radar::app::VisualizationConfig
default_visualization_config() {
  ::holoscan::radar::app::VisualizationConfig config;
  config.window_title = "FMCW SISO";
  return config;
}

struct AppParams {
  FMCWWaveformDescriptor waveform;
  CubeConfig cube;
  RunConfig run;
  ::holoscan::radar::app::VisualizationConfig visualization{
      default_visualization_config()};
  std::vector<TargetConfig> targets;
  std::optional<double> noise_snr_db{};
};

HOLOSCAN_PARAM_SCHEMA(
    AppParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.app.siso.AppParams", 1),
    holoscan::field<&AppParams::waveform>("waveform"),
    holoscan::field<&AppParams::cube>("cube"),
    holoscan::field<&AppParams::run>("run"),
    holoscan::field<&AppParams::visualization>("visualization"),
    holoscan::field<&AppParams::targets>("targets"),
    holoscan::field<&AppParams::noise_snr_db>("noise_snr_db"));

static_assert(holoscan::param_schema_availability<AppParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in AppParams's namespace");

HOLOSCAN_PARAM_SCHEMA(
    FeatureAssemblyParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.app.siso.FeatureAssemblyParams", 1),
    holoscan::field<&FeatureAssemblyParams::features>("features")
        .description(
            "Ordered FeatureAssemblyOp column tokens; constructor-capture "
            "only"));

static_assert(
    holoscan::param_schema_availability<FeatureAssemblyParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in FeatureAssemblyParams's namespace");

}  // namespace holoscan::radar::app::siso

namespace holoscan::config {

using SisoAppParamsCodec =
    params::DefaultedParamSchemaCodec<radar::app::siso::AppParams>;  // NOLINT

template <>
struct ConfigCodec<radar::app::siso::CubeConfig>
    : ObjectCodec<radar::app::siso::CubeConfig> {  // NOLINT
  using Config = radar::app::siso::CubeConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("num_chirps", &Config::num_chirps),
      object_field("samples_per_chirp", &Config::samples_per_chirp));
};

template <>
struct ConfigCodec<radar::app::siso::RunConfig>
    : ObjectCodec<radar::app::siso::RunConfig> {  // NOLINT
  using Config = radar::app::siso::RunConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("num_frames", &Config::num_frames),
      object_field("frame_period", &Config::frame_period),
      defaulted_object_field("meter_cadence", &Config::meter_cadence),
      defaulted_object_field("meter_log_every", &Config::meter_log_every));
};

template <>
struct ConfigCodec<radar::app::siso::TargetConfig>
    : ObjectCodec<radar::app::siso::TargetConfig> {  // NOLINT
  using Config = radar::app::siso::TargetConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("range", &Config::range),
      object_field("velocity", &Config::velocity),
      object_field("amplitude", &Config::amplitude));
};

template <>
struct ConfigCodec<radar::app::siso::AppParams>
    : SisoAppParamsCodec {};  // NOLINT

using SisoFeatureAssemblyParamsCodec = params::DefaultedParamSchemaCodec<
    radar::app::siso::FeatureAssemblyParams>;  // NOLINT

template <>
struct ConfigCodec<radar::app::siso::FeatureAssemblyParams>
    : SisoFeatureAssemblyParamsCodec {};  // NOLINT

}  // namespace holoscan::config
