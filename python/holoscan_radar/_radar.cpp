// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/error.hpp>
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "angle_estimation/beamscan_estimator_op/params_config.hpp"
#include "angle_estimation/phase_shift_beamformer_op/params_config.hpp"
#include "angle_estimation/snapshot_gather_op/params_config.hpp"
#include "clustering/cluster_dbscan_op/params_config.hpp"
#include "detection/cfar_detector_op/params_config.hpp"
#include "detection/detection_reduction_op/params_config.hpp"
#include "fmcw/clutter_filter_op/params_config.hpp"
#include "fmcw/fmcw_doppler_processor_op/params_config.hpp"
#include "fmcw/fmcw_range_processor_op/params_config.hpp"

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_op.hpp"
#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "angle_estimation/music_estimator_op/music_estimator_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "detection/multi_source_expansion_op/multi_source_expansion_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "fmcw/tdm_dense_compensation_op/tdm_dense_compensation_op.hpp"
#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "fmcw/ddm_power_fold_op/ddm_power_fold_op.hpp"
#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_op.hpp"
#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"

#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "io/cphd_reader_op/params_config.hpp"
#include "sar/aperture_manager_op/aperture_manager_op.hpp"
#include "sar/aperture_manager_op/params_config.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "sar/backprojection_op/params_config.hpp"
#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"
#include "sar/sar_range_compression_op/params_config.hpp"
#include "sar/phase_correction_op/phase_correction_op.hpp"
#include "sar/autofocus_op/autofocus_op.hpp"
#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "io/tensor_dump_op/params_config.hpp"
#include "radar_schemas/coordinate_frame_bridge.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"
#include "radar_schemas/lfm_waveform_bridge.hpp"

namespace holoscan::config {
template <>
struct ConfigCodec<radar::PhaseCorrectionParams>
: params::DefaultedParamSchemaCodec<radar::PhaseCorrectionParams> {};
template <>
struct ConfigCodec<radar::AutofocusParams>
: params::DefaultedParamSchemaCodec<radar::AutofocusParams> {};

template <>
struct ConfigCodec<radar::TDMSparseCompensationParams>
: params::DefaultedParamSchemaCodec<radar::TDMSparseCompensationParams> {};
template <>
struct ConfigCodec<radar::DDMDenseDemultiplexParams>
: params::DefaultedParamSchemaCodec<radar::DDMDenseDemultiplexParams> {};
template <>
struct ConfigCodec<radar::VirtualArrayAssemblyParams>
: params::DefaultedParamSchemaCodec<radar::VirtualArrayAssemblyParams> {};
template <>
struct ConfigCodec<radar::RankOneResidualScreenParams>
: params::DefaultedParamSchemaCodec<radar::RankOneResidualScreenParams> {};
template <>
struct ConfigCodec<radar::MUSICParams> : params::DefaultedParamSchemaCodec<radar::MUSICParams> {};
template <>
struct ConfigCodec<radar::RootMUSICParams>
: params::DefaultedParamSchemaCodec<radar::RootMUSICParams> {};
template <>
struct ConfigCodec<radar::ESPRITParams> : params::DefaultedParamSchemaCodec<radar::ESPRITParams> {};
template <>
struct ConfigCodec<radar::MVDRParams> : params::DefaultedParamSchemaCodec<radar::MVDRParams> {};
}  // namespace holoscan::config

namespace radar = holoscan::radar;
namespace authoring = holoscan::authoring;

namespace {

struct RadarAuthoringParams {
  std::string config_json;
};
HOLOSCAN_PARAM_SCHEMA(RadarAuthoringParams,
                      holoscan::param_schema("com.nvidia.holoscan.radar.AuthoringParams", 1),
                      holoscan::field<&RadarAuthoringParams::config_json>("config_json"));

radar::MaxTensorExtents decode_extents(const holoscan::config::Document& doc) {
  const auto extent_node = doc.at("/extents");
  if (extent_node.kind() != holoscan::config::ConfigValueKind::kObject) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                 "extents must be an object"};
  }
  radar::MaxTensorExtents extents;
  std::size_t decoded_count = 0U;
  if (auto found = doc.try_find("/extents/max_samples_per_chirp"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_samples_per_chirp");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_samples_per_chirp must be a positive integer");
    }
    ++decoded_count;
    extents.max_samples_per_chirp = *value;
  }
  if (auto found = doc.try_find("/extents/max_samples_per_pulse"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_samples_per_pulse");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_samples_per_pulse must be a positive integer");
    }
    ++decoded_count;
    extents.max_samples_per_pulse = *value;
  }
  if (auto found = doc.try_find("/extents/max_pulses"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_pulses");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_pulses must be a positive integer");
    }
    ++decoded_count;
    extents.max_pulses = *value;
  }
  if (auto found = doc.try_find("/extents/max_image_u"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_image_u");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_image_u must be a positive integer");
    }
    ++decoded_count;
    extents.max_image_u = *value;
  }
  if (auto found = doc.try_find("/extents/max_image_v"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_image_v");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_image_v must be a positive integer");
    }
    ++decoded_count;
    extents.max_image_v = *value;
  }
  if (auto found = doc.try_find("/extents/max_chirps"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_chirps");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_chirps must be a positive integer");
    }
    ++decoded_count;
    extents.max_chirps = *value;
  }
  if (auto found = doc.try_find("/extents/max_range_bins"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_range_bins");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_range_bins must be a positive integer");
    }
    ++decoded_count;
    extents.max_range_bins = *value;
  }
  if (auto found = doc.try_find("/extents/max_doppler_bins"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_doppler_bins");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_doppler_bins must be a positive integer");
    }
    ++decoded_count;
    extents.max_doppler_bins = *value;
  }
  if (auto found = doc.try_find("/extents/max_rx"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_rx");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_rx must be a positive integer");
    }
    ++decoded_count;
    extents.max_rx = *value;
  }
  if (auto found = doc.try_find("/extents/max_tx"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_tx");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_tx must be a positive integer");
    }
    ++decoded_count;
    extents.max_tx = *value;
  }
  if (auto found = doc.try_find("/extents/max_interleaved_chirps"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_interleaved_chirps");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_interleaved_chirps must be a positive integer");
    }
    ++decoded_count;
    extents.max_interleaved_chirps = *value;
  }
  if (auto found = doc.try_find("/extents/max_virtual_channels"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_virtual_channels");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_virtual_channels must be a positive integer");
    }
    ++decoded_count;
    extents.max_virtual_channels = *value;
  }
  if (auto found = doc.try_find("/extents/max_channels"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_channels");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_channels must be a positive integer");
    }
    ++decoded_count;
    extents.max_channels = *value;
  }
  if (auto found = doc.try_find("/extents/max_az"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_az");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_az must be a positive integer");
    }
    ++decoded_count;
    extents.max_az = *value;
  }
  if (auto found = doc.try_find("/extents/max_el"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_el");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_el must be a positive integer");
    }
    ++decoded_count;
    extents.max_el = *value;
  }
  if (auto found = doc.try_find("/extents/max_detections"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_detections");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_detections must be a positive integer");
    }
    ++decoded_count;
    extents.max_detections = *value;
  }
  if (auto found = doc.try_find("/extents/max_points"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_points");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_points must be a positive integer");
    }
    ++decoded_count;
    extents.max_points = *value;
  }
  if (auto found = doc.try_find("/extents/max_items"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_items");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_items must be a positive integer");
    }
    ++decoded_count;
    extents.max_items = *value;
  }
  if (auto found = doc.try_find("/extents/max_snapshots"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_snapshots");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_snapshots must be a positive integer");
    }
    ++decoded_count;
    extents.max_snapshots = *value;
  }
  if (auto found = doc.try_find("/extents/max_features"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_features");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_features must be a positive integer");
    }
    ++decoded_count;
    extents.max_features = *value;
  }
  if (auto found = doc.try_find("/extents/max_sources"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_sources");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_sources must be a positive integer");
    }
    ++decoded_count;
    extents.max_sources = *value;
  }
  if (auto found = doc.try_find("/extents/max_clusters"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_clusters");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_clusters must be a positive integer");
    }
    ++decoded_count;
    extents.max_clusters = *value;
  }
  if (auto found = doc.try_find("/extents/max_subarray_channels"); found && found->has_value()) {
    auto value = doc.try_get<std::int64_t>("/extents/max_subarray_channels");
    if (!value || *value <= 0) {
      throw std::invalid_argument("/extents/max_subarray_channels must be a positive integer");
    }
    ++decoded_count;
    extents.max_subarray_channels = *value;
  }
  if (decoded_count != extent_node.size()) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                 "extents contains an unknown max_* field"};
  }
  return extents;
}

radar::MaxTensorExtents representative_extents() {
  radar::MaxTensorExtents value;
  value.max_samples_per_chirp = 64;
  value.max_samples_per_pulse = 64;
  value.max_pulses = 64;
  value.max_image_u = 64;
  value.max_image_v = 64;
  value.max_chirps = 64;
  value.max_range_bins = 64;
  value.max_doppler_bins = 64;
  value.max_rx = 64;
  value.max_tx = 64;
  value.max_interleaved_chirps = 64;
  value.max_virtual_channels = 64;
  value.max_channels = 64;
  value.max_az = 64;
  value.max_el = 64;
  value.max_detections = 64;
  value.max_points = 64;
  value.max_items = 64;
  value.max_snapshots = 64;
  value.max_features = 64;
  value.max_sources = 64;
  value.max_clusters = 64;
  value.max_subarray_channels = 64;
  return value;
}

template <typename Op, typename Params = void>
struct RadarThunks : authoring::ManualOperatorRegistrationThunks<Op> {
  using Base = authoring::ManualOperatorRegistrationThunks<Op>;

  static bool unsupported_enum(const holoscan::PortRecord& record) {
    const auto& identity = record.payload_identity;
    return identity.kind == holoscan::PayloadIdentityKind::kScalar && identity.scalar.has_value() &&
           identity.scalar->kind == holoscan::ScalarIdentityKind::kLocalEnum;
  }

  static std::vector<holoscan::PortRecord> ports(const authoring::OpaqueOperatorRef& op) {
    auto result = Base::ports(op);
    std::erase_if(result, unsupported_enum);
    return result;
  }

  static authoring::OpaquePortRef port(const authoring::OpaqueOperatorRef& op,
                                       std::string_view name) {
    const auto original = Base::ports(op);
    for (const auto& record : original) {
      if (record.name == name && unsupported_enum(record)) {
        throw holoscan::RuntimeError{
            holoscan::ErrorCode::kNotSupported,
            "Radar Python Graph cannot connect local enum input port " + std::string(name)};
      }
    }
    return Base::port(op, name);
  }

  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* materialized) {
    if (materialized != nullptr &&
        (materialized->type() == nullptr ||
         materialized->type()->local_type_token !=
             holoscan::param_type_descriptor<RadarAuthoringParams>().local_type_token)) {
      throw std::invalid_argument("Radar provider parameter type mismatch");
    }
    const auto* values = materialized == nullptr
                             ? nullptr
                             : static_cast<const RadarAuthoringParams*>(materialized->data());
    const bool representative = values == nullptr || values->config_json.empty();
    radar::MaxTensorExtents extents =
        representative ? representative_extents()
                       : decode_extents(holoscan::config_io::parse(
                             values->config_json, holoscan::config_io::Format::kJson));
    if constexpr (std::is_same_v<Op, radar::FeatureAssemblyOp<float>> ||
                  std::is_same_v<Op, radar::FeatureAssemblyOp<double>>) {
      std::vector<radar::AssemblyFeature> features{radar::AssemblyFeature::kRange};
      if (!representative) {
        auto document =
            holoscan::config_io::parse(values->config_json, holoscan::config_io::Format::kJson);
        auto decoded = document.try_get<std::vector<radar::AssemblyFeature>>("/features");
        if (!decoded) {
          throw std::invalid_argument("FeatureAssemblyOp.features is required");
        }
        features = std::move(*decoded);
      }
      return RadarThunks::make_operator_ref(graph.op<Op>(name, extents, std::move(features)));
    } else {
      auto handle = [&] {
        if constexpr (std::is_constructible_v<Op, const radar::MaxTensorExtents&>) {
          return graph.op<Op>(name, extents);
        } else {
          return graph.op<Op>(name);
        }
      }();
      if constexpr (!std::is_void_v<Params>) {
        if (!representative) {
          auto document =
              holoscan::config_io::parse(values->config_json, holoscan::config_io::Format::kJson);
          auto params_node = document.try_find("/params");
          if (!params_node) {
            throw std::invalid_argument(holoscan::config::format_error(params_node.error()));
          }
          if (params_node->has_value() && params_node->value().size() != 0U) {
            if constexpr (holoscan::config::ConfigCodable<Params>) {
              auto decoded = document.try_get<Params>("/params");
              if (!decoded) {
                throw std::invalid_argument(holoscan::config::format_error(decoded.error()));
              }
              handle.set_params(std::move(*decoded));
            } else {
              auto decoded = holoscan::params::try_decode<Params>(document, "/params");
              if (!decoded) {
                throw std::invalid_argument(holoscan::params::format_error(decoded.error()));
              }
              handle.set_params(std::move(*decoded));
            }
          }
        }
      }
      return RadarThunks::make_operator_ref(handle);
    }
  }

  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> add_from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto values = authoring::materialize_parameters(
        holoscan::param_type_descriptor<RadarAuthoringParams>(),
        holoscan::param_schema_descriptor<RadarAuthoringParams>(), assignments);
    if (!values) {
      return holoscan::make_unexpected(std::move(values).error());
    }
    return RadarThunks::invoke_add([&] {
      try {
        return add_to_graph(graph, name, &*values);
      } catch (const std::invalid_argument& error) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument, error.what()};
      } catch (const holoscan::config::ConfigException& error) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument, error.what()};
      }
    });
  }

  static void set_params(const authoring::OpaqueOperatorRef&,
                         const authoring::MaterializedParameters&) {
    throw std::invalid_argument("Radar construction parameters are immutable");
  }
};

template <typename Op, typename Params = void>
void register_one(std::string_view name) {
  using Thunks = RadarThunks<Op, Params>;
  auto result = authoring::authoring_registry().register_operator({
      .name = name,
      .param_type = &holoscan::param_type_descriptor<RadarAuthoringParams>(),
      .param_schema = holoscan::param_schema_descriptor<RadarAuthoringParams>(),
      .parameter_introspection = authoring::OperatorParameterIntrospection::kConstructorReplay,
      .add_to_graph = &Thunks::add_to_graph,
      .add_to_graph_from_assignments = &Thunks::add_from_assignments,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .ports = &Thunks::ports,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
      .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
  });
  if (!result) {
    throw holoscan::RuntimeError{result.error()};
  }
}

template <typename Payload>
void register_payload(std::string_view name) {
  auto row = authoring::detail::make_registered_payload<Payload>(name);
  auto result = authoring::authoring_registry().register_payload(std::move(row));
  if (!result) {
    throw holoscan::RuntimeError{result.error()};
  }
}

void register_catalog() {
  register_one<radar::CPHDReaderOp<float, float>, radar::CphdReaderParams>(
      "holoscan.radar.CPHDReaderOpF32GeomF32");
  register_one<radar::CPHDReaderOp<float, double>, radar::CphdReaderParams>(
      "holoscan.radar.CPHDReaderOpF32GeomF64");
  register_one<radar::CPHDReaderOp<double, double>, radar::CphdReaderParams>(
      "holoscan.radar.CPHDReaderOpF64GeomF64");
  register_one<radar::ApertureManagerOp<float, float>, radar::ApertureManagerParams>(
      "holoscan.radar.ApertureManagerOpF32GeomF32");
  register_one<radar::ApertureManagerOp<float, double>, radar::ApertureManagerParams>(
      "holoscan.radar.ApertureManagerOpF32GeomF64");
  register_one<radar::ApertureManagerOp<double, double>, radar::ApertureManagerParams>(
      "holoscan.radar.ApertureManagerOpF64GeomF64");
  register_one<radar::BackprojectionOp<float, float>, radar::BackprojectionParams>(
      "holoscan.radar.BackprojectionOpF32GeomF32");
  register_one<radar::BackprojectionOp<float, double>, radar::BackprojectionParams>(
      "holoscan.radar.BackprojectionOpF32GeomF64");
  register_one<radar::BackprojectionOp<double, double>, radar::BackprojectionParams>(
      "holoscan.radar.BackprojectionOpF64GeomF64");
  register_one<radar::SARRangeCompressionOp<float>, radar::SARRangeCompressionParams>(
      "holoscan.radar.SARRangeCompressionOpF32");
  register_one<radar::SARRangeCompressionOp<double>, radar::SARRangeCompressionParams>(
      "holoscan.radar.SARRangeCompressionOpF64");
  register_one<radar::PhaseCorrectionOp<float>, radar::PhaseCorrectionParams>(
      "holoscan.radar.PhaseCorrectionOpF32");
  register_one<radar::PhaseCorrectionOp<double>, radar::PhaseCorrectionParams>(
      "holoscan.radar.PhaseCorrectionOpF64");
  register_one<radar::AutofocusOp<float>, radar::AutofocusParams>("holoscan.radar.AutofocusOpF32");
  register_one<radar::AutofocusOp<double>, radar::AutofocusParams>("holoscan.radar.AutofocusOpF64");
  register_one<radar::TensorDumpOp, radar::TensorDumpParams>("holoscan.radar.TensorDumpOp");

  register_payload<radar::WireCoordinateFrame>(
      holoscan::schema_identity<radar::WireCoordinateFrame>().canonical_name);
  register_payload<radar::WireSampleGeometry>(
      holoscan::schema_identity<radar::WireSampleGeometry>().canonical_name);
  register_payload<radar::WireArrayGeometry>(
      holoscan::schema_identity<radar::WireArrayGeometry>().canonical_name);
  register_payload<radar::WireWindowSpec>(
      holoscan::schema_identity<radar::WireWindowSpec>().canonical_name);
  register_payload<radar::wire::FMCWWaveformT>(
      holoscan::schema_identity<radar::wire::FMCWWaveformT>().canonical_name);
  register_payload<radar::CfarCellWindowOverride>(
      holoscan::ChannelTypeIdentity<radar::CfarCellWindowOverride>::name);
  register_payload<radar::SpatialSmoothingOverride>(
      holoscan::ChannelTypeIdentity<radar::SpatialSmoothingOverride>::name);
  register_payload<radar::SnapshotWindowOverride>(
      holoscan::ChannelTypeIdentity<radar::SnapshotWindowOverride>::name);
  register_payload<radar::DDMSparseWindowOverride>(
      holoscan::ChannelTypeIdentity<radar::DDMSparseWindowOverride>::name);
  register_payload<radar::TDMSparseWindowOverride>(
      holoscan::ChannelTypeIdentity<radar::TDMSparseWindowOverride>::name);
  register_one<radar::ClutterFilterOp<float>, radar::ClutterFilterParams>(
      "holoscan.radar.ClutterFilterOpF32");
  register_one<radar::ClutterFilterOp<double>, radar::ClutterFilterParams>(
      "holoscan.radar.ClutterFilterOpF64");
  register_one<radar::SnapshotGatherOp<float>, radar::SnapshotGatherParams>(
      "holoscan.radar.SnapshotGatherOpF32");
  register_one<radar::SnapshotGatherOp<double>, radar::SnapshotGatherParams>(
      "holoscan.radar.SnapshotGatherOpF64");
  register_one<radar::PhaseShiftBeamformerOp<float>, radar::PhaseShiftBeamformerParams>(
      "holoscan.radar.PhaseShiftBeamformerOpF32");
  register_one<radar::PhaseShiftBeamformerOp<double>, radar::PhaseShiftBeamformerParams>(
      "holoscan.radar.PhaseShiftBeamformerOpF64");
  register_one<radar::BeamscanEstimatorOp<float>, radar::BeamscanParams>(
      "holoscan.radar.BeamscanEstimatorOpF32");
  register_one<radar::BeamscanEstimatorOp<double>, radar::BeamscanParams>(
      "holoscan.radar.BeamscanEstimatorOpF64");
  register_one<radar::RankOneResidualScreenOp<float>, radar::RankOneResidualScreenParams>(
      "holoscan.radar.RankOneResidualScreenOpF32");
  register_one<radar::RankOneResidualScreenOp<double>, radar::RankOneResidualScreenParams>(
      "holoscan.radar.RankOneResidualScreenOpF64");
  register_one<radar::SubarraySelectionOp<float>, radar::SubarraySelectionParams>(
      "holoscan.radar.SubarraySelectionOpF32");
  register_one<radar::SubarraySelectionOp<double>, radar::SubarraySelectionParams>(
      "holoscan.radar.SubarraySelectionOpF64");
  register_one<radar::ClusterDBSCANOp<float>, radar::ClusterDBSCANParams>(
      "holoscan.radar.ClusterDBSCANOpF32");
  register_one<radar::ClusterDBSCANOp<double>, radar::ClusterDBSCANParams>(
      "holoscan.radar.ClusterDBSCANOpF64");
  register_one<radar::DetectionReductionOp<float>, radar::DetectionReductionParams>(
      "holoscan.radar.DetectionReductionOpF32");
  register_one<radar::DetectionReductionOp<double>, radar::DetectionReductionParams>(
      "holoscan.radar.DetectionReductionOpF64");
  register_one<radar::MVDREstimatorOp<float>, radar::MVDRParams>(
      "holoscan.radar.MVDREstimatorOpF32");
  register_one<radar::MVDREstimatorOp<double>, radar::MVDRParams>(
      "holoscan.radar.MVDREstimatorOpF64");
  register_one<radar::MUSICEstimatorOp<float>, radar::MUSICParams>(
      "holoscan.radar.MUSICEstimatorOpF32");
  register_one<radar::MUSICEstimatorOp<double>, radar::MUSICParams>(
      "holoscan.radar.MUSICEstimatorOpF64");
  register_one<radar::FeatureAssemblyOp<float>>("holoscan.radar.FeatureAssemblyOpF32");
  register_one<radar::FeatureAssemblyOp<double>>("holoscan.radar.FeatureAssemblyOpF64");
  register_one<radar::MultiSourceExpansionOp<float>>("holoscan.radar.MultiSourceExpansionOpF32");
  register_one<radar::MultiSourceExpansionOp<double>>("holoscan.radar.MultiSourceExpansionOpF64");
  register_one<radar::CFARDetectorOp<float>, radar::CFARParams>("holoscan.radar.CFARDetectorOpF32");
  register_one<radar::CFARDetectorOp<double>, radar::CFARParams>(
      "holoscan.radar.CFARDetectorOpF64");
  register_one<radar::RootMUSICEstimatorOp<float>, radar::RootMUSICParams>(
      "holoscan.radar.RootMUSICEstimatorOpF32");
  register_one<radar::RootMUSICEstimatorOp<double>, radar::RootMUSICParams>(
      "holoscan.radar.RootMUSICEstimatorOpF64");
  register_one<radar::ESPRITEstimatorOp<float>, radar::ESPRITParams>(
      "holoscan.radar.ESPRITEstimatorOpF32");
  register_one<radar::ESPRITEstimatorOp<double>, radar::ESPRITParams>(
      "holoscan.radar.ESPRITEstimatorOpF64");
  register_one<radar::MIMODeinterleaveOp<float>, radar::MIMODeinterleaveParams>(
      "holoscan.radar.MIMODeinterleaveOpF32");
  register_one<radar::MIMODeinterleaveOp<double>, radar::MIMODeinterleaveParams>(
      "holoscan.radar.MIMODeinterleaveOpF64");
  register_one<radar::TDMDenseCompensationOp<float>, radar::TDMDenseCompensationParams>(
      "holoscan.radar.TDMDenseCompensationOpF32");
  register_one<radar::TDMDenseCompensationOp<double>, radar::TDMDenseCompensationParams>(
      "holoscan.radar.TDMDenseCompensationOpF64");
  register_one<radar::DDMDenseDemultiplexOp<float>, radar::DDMDenseDemultiplexParams>(
      "holoscan.radar.DDMDenseDemultiplexOpF32");
  register_one<radar::DDMDenseDemultiplexOp<double>, radar::DDMDenseDemultiplexParams>(
      "holoscan.radar.DDMDenseDemultiplexOpF64");
  register_one<radar::DopplerNotchOp<float>, radar::DopplerNotchParams>(
      "holoscan.radar.DopplerNotchOpF32");
  register_one<radar::DopplerNotchOp<double>, radar::DopplerNotchParams>(
      "holoscan.radar.DopplerNotchOpF64");
  register_one<radar::DDMPowerFoldOp<float>, radar::DDMPowerFoldParams>(
      "holoscan.radar.DDMPowerFoldOpF32");
  register_one<radar::DDMPowerFoldOp<double>, radar::DDMPowerFoldParams>(
      "holoscan.radar.DDMPowerFoldOpF64");
  register_one<radar::DDMSparseDemultiplexOp<float>, radar::DDMSparseDemultiplexParams>(
      "holoscan.radar.DDMSparseDemultiplexOpF32");
  register_one<radar::DDMSparseDemultiplexOp<double>, radar::DDMSparseDemultiplexParams>(
      "holoscan.radar.DDMSparseDemultiplexOpF64");
  register_one<radar::TDMSparseCompensationOp<float>, radar::TDMSparseCompensationParams>(
      "holoscan.radar.TDMSparseCompensationOpF32");
  register_one<radar::TDMSparseCompensationOp<double>, radar::TDMSparseCompensationParams>(
      "holoscan.radar.TDMSparseCompensationOpF64");
  register_one<radar::VirtualArrayAssemblyOp<float>, radar::VirtualArrayAssemblyParams>(
      "holoscan.radar.VirtualArrayAssemblyOpF32");
  register_one<radar::VirtualArrayAssemblyOp<double>, radar::VirtualArrayAssemblyParams>(
      "holoscan.radar.VirtualArrayAssemblyOpF64");
  register_one<radar::FMCWDopplerProcessorOp<float>, radar::FMCWDopplerProcessorParams>(
      "holoscan.radar.FMCWDopplerProcessorOpF32");
  register_one<radar::FMCWDopplerProcessorOp<double>, radar::FMCWDopplerProcessorParams>(
      "holoscan.radar.FMCWDopplerProcessorOpF64");
  register_one<radar::FMCWRangeProcessorOp<float>, radar::FMCWRangeProcessorParams>(
      "holoscan.radar.FMCWRangeProcessorOpF32");
  register_one<radar::FMCWRangeProcessorOp<double>, radar::FMCWRangeProcessorParams>(
      "holoscan.radar.FMCWRangeProcessorOpF64");
  register_one<radar::FMCWRangeDopplerProcessorOp<float>, radar::FMCWRangeDopplerProcessorParams>(
      "holoscan.radar.FMCWRangeDopplerProcessorOpF32");
  register_one<radar::FMCWRangeDopplerProcessorOp<double>, radar::FMCWRangeDopplerProcessorParams>(
      "holoscan.radar.FMCWRangeDopplerProcessorOpF64");
}
}  // namespace

void register_sar_bindings(nanobind::module_& module);
void register_calibration_bindings(nanobind::module_& module);
void register_siso_bindings(nanobind::module_& module);
void register_tdm_bindings(nanobind::module_& module);
void register_viz_bindings(nanobind::module_& module);

NB_MODULE(_radar, module) {
  register_catalog();
  register_calibration_bindings(module);
  register_siso_bindings(module);
  register_tdm_bindings(module);
  register_viz_bindings(module);
  register_sar_bindings(module);
  module.doc() = "FMCW and SAR radar operator catalog for HSDK 5";
}
