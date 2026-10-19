// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/params/config.hpp>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_app/app/range_loss_config.hpp"
#include "radar_app/app/scene_visibility_config.hpp"
#include "radar_app/app/visualization_config.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::radar::app::mimo {

struct CubeConfig {
  // TDM: chirps per TX. DDM: total coded chirps.
  int64_t num_chirps{0};
  int64_t samples_per_chirp{0};
};

struct RunConfig {
  int64_t num_frames{1};
  double frame_period{0.1};
  bool meter_cadence{true};
  int64_t meter_log_every{0};
};

struct TxArrayConfig {
  int64_t num_tx{0};
  double spacing{0.0};
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

struct TdmSimulationConfig {
  double physical_chirp_repetition_interval{0.0};
  std::vector<int64_t> tx_order{};
};

struct TdmProfileConfig {
  std::string name;
  double carrier_frequency{0.0};
  double bandwidth{0.0};
  double chirp_duration{0.0};
  double sample_rate{0.0};
  double physical_chirp_repetition_interval{0.0};
  int64_t chirps_per_tx{0};
  int64_t samples_per_chirp{0};
};

struct DdmSimulationConfig {
  // Empty uses the authored native ddm_code. Supported generated values:
  // zadoff_chu (or zc) and sin.
  std::optional<std::string> generate_phase_codes{};
};

// Constructor-capture authoring for FeatureAssemblyOp.features.
struct FeatureAssemblyParams {
  std::vector<AssemblyFeature> features{};
};

template <typename T>
using AppStorage = radar::config_detail::PortableParamStorageTraits<T>;

}  // namespace holoscan::radar::app::mimo

namespace holoscan {

template <>
struct ParamStorageTraits<radar::app::mimo::CubeConfig>
: radar::app::mimo::AppStorage<radar::app::mimo::CubeConfig> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::mimo::RunConfig>
: radar::app::mimo::AppStorage<radar::app::mimo::RunConfig> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::mimo::TxArrayConfig>
: radar::app::mimo::AppStorage<radar::app::mimo::TxArrayConfig> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::mimo::RxArrayConfig>
: radar::app::mimo::AppStorage<radar::app::mimo::RxArrayConfig> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::mimo::TargetConfig>
: radar::app::mimo::AppStorage<radar::app::mimo::TargetConfig> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::mimo::TdmSimulationConfig>
: radar::app::mimo::AppStorage<radar::app::mimo::TdmSimulationConfig> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::mimo::TdmProfileConfig>
: radar::app::mimo::AppStorage<radar::app::mimo::TdmProfileConfig> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::mimo::DdmSimulationConfig>
: radar::app::mimo::AppStorage<radar::app::mimo::DdmSimulationConfig> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::app::mimo::FeatureAssemblyParams>
: radar::app::mimo::AppStorage<radar::app::mimo::FeatureAssemblyParams> {};  // NOLINT
}  // namespace holoscan

namespace holoscan::radar::app::mimo {

inline ::holoscan::radar::app::VisualizationConfig default_visualization_config() {
  ::holoscan::radar::app::VisualizationConfig config;
  config.window_title = "FMCW MIMO";
  return config;
}

struct AppParams {
  // Optional startup selector used only by adaptive_aoa. Other MIMO apps
  // ignore it and retain their fixed estimator graph.
  std::string aoa_algorithm{"music"};
  FMCWWaveformDescriptor waveform;
  CubeConfig cube;
  RunConfig run;
  ::holoscan::radar::app::VisualizationConfig visualization{default_visualization_config()};
  std::optional<::holoscan::radar::app::RangeLossConfig> range_loss{};
  ::holoscan::radar::app::SceneVisibilityConfig scene_visibility{};
  TxArrayConfig tx_array;
  RxArrayConfig rx_array;
  std::vector<TargetConfig> targets;
  std::optional<double> noise_snr_db{};
  std::optional<TdmSimulationConfig> tdm{};
  std::vector<TdmProfileConfig> profiles{};
  std::vector<std::string> profile_sequence{};
  int64_t max_frames_in_flight{4};
  double rx_poll_period{0.001};
  int64_t simulated_receive_polls{2};
};

HOLOSCAN_PARAM_SCHEMA(
    AppParams, holoscan::param_schema("com.nvidia.holoscan.radar.app.mimo.AppParams", 1),
    holoscan::field<&AppParams::aoa_algorithm>("aoa_algorithm"),
    holoscan::field<&AppParams::waveform>("waveform"), holoscan::field<&AppParams::cube>("cube"),
    holoscan::field<&AppParams::run>("run"),
    holoscan::field<&AppParams::visualization>("visualization"),
    holoscan::field<&AppParams::range_loss>("range_loss"),
    holoscan::field<&AppParams::scene_visibility>("scene_visibility"),
    holoscan::field<&AppParams::tx_array>("tx_array"),
    holoscan::field<&AppParams::rx_array>("rx_array"),
    holoscan::field<&AppParams::targets>("targets"),
    holoscan::field<&AppParams::noise_snr_db>("noise_snr_db"),
    holoscan::field<&AppParams::tdm>("tdm"), holoscan::field<&AppParams::profiles>("profiles"),
    holoscan::field<&AppParams::profile_sequence>("profile_sequence"),
    holoscan::field<&AppParams::max_frames_in_flight>("max_frames_in_flight"),
    holoscan::field<&AppParams::rx_poll_period>("rx_poll_period"),
    holoscan::field<&AppParams::simulated_receive_polls>("simulated_receive_polls"));

static_assert(holoscan::param_schema_availability<AppParams>() ==
              holoscan::ParamSchemaAvailability::kDeclared);

HOLOSCAN_PARAM_SCHEMA(
    FeatureAssemblyParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.app.mimo.FeatureAssemblyParams", 1),
    holoscan::field<&FeatureAssemblyParams::features>("features")
        .description("Ordered FeatureAssemblyOp column tokens; constructor-capture "
                     "only"));

static_assert(holoscan::param_schema_availability<FeatureAssemblyParams>() ==
              holoscan::ParamSchemaAvailability::kDeclared);

}  // namespace holoscan::radar::app::mimo

namespace holoscan::config {

template <>
struct ConfigCodec<radar::app::mimo::CubeConfig> : ObjectCodec<radar::app::mimo::CubeConfig> {
  using Config = radar::app::mimo::CubeConfig;
  static constexpr auto fields =
      std::make_tuple(object_field("num_chirps", &Config::num_chirps),
                      object_field("samples_per_chirp", &Config::samples_per_chirp));
};

template <>
struct ConfigCodec<radar::app::mimo::RunConfig> : ObjectCodec<radar::app::mimo::RunConfig> {
  using Config = radar::app::mimo::RunConfig;
  static constexpr auto fields =
      std::make_tuple(object_field("num_frames", &Config::num_frames),
                      object_field("frame_period", &Config::frame_period),
                      defaulted_object_field("meter_cadence", &Config::meter_cadence),
                      defaulted_object_field("meter_log_every", &Config::meter_log_every));
};

template <>
struct ConfigCodec<radar::app::mimo::TxArrayConfig> : ObjectCodec<radar::app::mimo::TxArrayConfig> {
  using Config = radar::app::mimo::TxArrayConfig;
  static constexpr auto fields = std::make_tuple(object_field("num_tx", &Config::num_tx),
                                                 object_field("spacing", &Config::spacing));
};

template <>
struct ConfigCodec<radar::app::mimo::RxArrayConfig> : ObjectCodec<radar::app::mimo::RxArrayConfig> {
  using Config = radar::app::mimo::RxArrayConfig;
  static constexpr auto fields = std::make_tuple(object_field("az_elements", &Config::az_elements),
                                                 object_field("el_elements", &Config::el_elements),
                                                 object_field("spacing", &Config::spacing));
};

template <>
struct ConfigCodec<radar::app::mimo::TargetConfig> : ObjectCodec<radar::app::mimo::TargetConfig> {
  using Config = radar::app::mimo::TargetConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("range", &Config::range), object_field("azimuth", &Config::azimuth),
      object_field("elevation", &Config::elevation), object_field("velocity", &Config::velocity),
      object_field("amplitude", &Config::amplitude));
};

template <>
struct ConfigCodec<radar::app::mimo::TdmSimulationConfig>
: ObjectCodec<radar::app::mimo::TdmSimulationConfig> {
  using Config = radar::app::mimo::TdmSimulationConfig;
  static constexpr auto fields =
      std::make_tuple(defaulted_object_field("physical_chirp_repetition_interval",
                                             &Config::physical_chirp_repetition_interval),
                      defaulted_object_field("tx_order", &Config::tx_order));
};

template <>
struct ConfigCodec<radar::app::mimo::TdmProfileConfig>
: ObjectCodec<radar::app::mimo::TdmProfileConfig> {
  using Config = radar::app::mimo::TdmProfileConfig;
  static constexpr auto fields =
      std::make_tuple(object_field("name", &Config::name),
                      object_field("carrier_frequency", &Config::carrier_frequency),
                      object_field("bandwidth", &Config::bandwidth),
                      object_field("chirp_duration", &Config::chirp_duration),
                      object_field("sample_rate", &Config::sample_rate),
                      object_field("physical_chirp_repetition_interval",
                                   &Config::physical_chirp_repetition_interval),
                      object_field("chirps_per_tx", &Config::chirps_per_tx),
                      object_field("samples_per_chirp", &Config::samples_per_chirp));
};

template <>
struct ConfigCodec<radar::app::mimo::DdmSimulationConfig>
: ObjectCodec<radar::app::mimo::DdmSimulationConfig> {
  using Config = radar::app::mimo::DdmSimulationConfig;
  static constexpr auto fields = std::make_tuple(
      defaulted_object_field("generate_phase_codes", &Config::generate_phase_codes));
};

using AppParamsCodec = params::DefaultedParamSchemaCodec<radar::app::mimo::AppParams>;  // NOLINT
template <>
struct ConfigCodec<radar::app::mimo::AppParams> : AppParamsCodec {};  // NOLINT

using FeatureAssemblyParamsCodec =
    params::DefaultedParamSchemaCodec<radar::app::mimo::FeatureAssemblyParams>;  // NOLINT
template <>
struct ConfigCodec<radar::app::mimo::FeatureAssemblyParams> : FeatureAssemblyParamsCodec {
};  // NOLINT

using MimoDeinterleaveParamsCodec =
    params::DefaultedParamSchemaCodec<radar::MIMODeinterleaveParams>;  // NOLINT
template <>
struct ConfigCodec<radar::MIMODeinterleaveParams> : MimoDeinterleaveParamsCodec {};  // NOLINT

using TdmSparseCompensationParamsCodec =
    params::DefaultedParamSchemaCodec<radar::TDMSparseCompensationParams>;  // NOLINT
template <>
struct ConfigCodec<radar::TDMSparseCompensationParams> : TdmSparseCompensationParamsCodec {
};  // NOLINT

using DdmDemultiplexParamsCodec =
    params::DefaultedParamSchemaCodec<radar::DDMDenseDemultiplexParams>;  // NOLINT
template <>
struct ConfigCodec<radar::DDMDenseDemultiplexParams> : DdmDemultiplexParamsCodec {};  // NOLINT

using VirtualArrayAssemblyParamsCodec =
    params::DefaultedParamSchemaCodec<radar::VirtualArrayAssemblyParams>;  // NOLINT
template <>
struct ConfigCodec<radar::VirtualArrayAssemblyParams> : VirtualArrayAssemblyParamsCodec {
};  // NOLINT

}  // namespace holoscan::config
