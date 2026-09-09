// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// [EA HSDK5 ONLY] Manual YAML → Params loaders for Graph apps (no
// from_config()). Shared by fmcw_siso / fmcw_simo / fmcw_mimo / …; fold back
// to from_config() when Application demos return.

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "angle_estimation/music_estimator_op/music_estimator_op.hpp"
#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "fmcw/tdm_dense_compensation_op/tdm_dense_compensation_op.hpp"
#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace holoscan::radar::app {

inline CfarVariant parse_cfar_variant(const std::string& text) {
  if (text == "CA") {
    return CfarVariant::kCA;
  }
  if (text == "GO") {
    return CfarVariant::kGO;
  }
  if (text == "SO") {
    return CfarVariant::kSO;
  }
  if (text == "OS") {
    return CfarVariant::kOS;
  }
  throw std::invalid_argument("cfar_variant: unknown value '" + text +
                              "' (accepted: CA, GO, SO, OS)");
}

inline CfarPeakNeighborhood parse_peak_neighborhood(const std::string& text) {
  if (text == "cross4") {
    return CfarPeakNeighborhood::kCross4;
  }
  if (text == "square8") {
    return CfarPeakNeighborhood::kSquare8;
  }
  throw std::invalid_argument(
      "peak_neighborhood: unknown value '" + text +
      "' (accepted: cross4, square8)");
}

inline void load_range_params(const YAML::Node& node,
                              FMCWRangeProcessorParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["waveform"]) {
    p->waveform = node["waveform"].as<FMCWWaveformDescriptor>();
  }
  if (node["range_window"]) {
    p->range_window = node["range_window"].as<WindowSpec>();
  }
  if (node["apply_dechirp"]) {
    p->apply_dechirp = node["apply_dechirp"].as<bool>();
  }
  if (node["fft_norm"]) {
    p->fft_norm = node["fft_norm"].as<FftNorm>();
  }
  if (node["propagation_speed"]) {
    p->propagation_speed = node["propagation_speed"].as<double>();
  }
  if (node["nfft"]) {
    p->nfft = node["nfft"].as<int64_t>();
  }
}

inline void load_clutter_params(const YAML::Node& node,
                                ClutterFilterParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["method"]) {
    p->method = node["method"].as<ClutterFilterMethod>();
  }
}

inline void load_doppler_params(const YAML::Node& node,
                                FMCWDopplerProcessorParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["waveform"]) {
    p->waveform = node["waveform"].as<FMCWWaveformDescriptor>();
  }
  if (node["doppler_window"]) {
    p->doppler_window = node["doppler_window"].as<WindowSpec>();
  }
  if (node["center_zero_doppler"]) {
    p->center_zero_doppler = node["center_zero_doppler"].as<bool>();
  }
  if (node["fft_norm"]) {
    p->fft_norm = node["fft_norm"].as<FftNorm>();
  }
  if (node["nfft"]) {
    p->nfft = node["nfft"].as<int64_t>();
  }
}

inline void load_cfar_params(const YAML::Node& node, CFARParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["cfar_variant"]) {
    p->cfar_variant =
        parse_cfar_variant(node["cfar_variant"].as<std::string>());
  }
  if (node["os_rank"]) {
    p->os_rank = node["os_rank"].as<int64_t>();
  }
  if (node["guard_cells"]) {
    p->guard_cells = node["guard_cells"].as<std::vector<int64_t>>();
  }
  if (node["training_cells"]) {
    p->training_cells = node["training_cells"].as<std::vector<int64_t>>();
  }
  if (node["pfa"]) {
    p->pfa = node["pfa"].as<double>();
  }
  if (node["alpha_lut"]) {
    p->alpha_lut = node["alpha_lut"].as<std::vector<double>>();
  }
  if (node["threshold_scale"]) {
    p->threshold_scale = node["threshold_scale"].as<double>();
  }
  if (node["nci_axes"]) {
    p->nci_axes = node["nci_axes"].as<std::vector<int64_t>>();
  }
  if (node["preintegrated_looks"]) {
    p->preintegrated_looks =
        node["preintegrated_looks"].as<int64_t>();
  }
  if (node["nci_fold"]) {
    p->nci_fold = node["nci_fold"].as<int64_t>();
  }
  if (node["max_detections"]) {
    p->max_detections = node["max_detections"].as<int64_t>();
  }
  if (node["blocking_alpha_lut_rebuild"]) {
    p->blocking_alpha_lut_rebuild =
        node["blocking_alpha_lut_rebuild"].as<bool>();
  }
  if (node["peak_only"]) {
    p->peak_only = node["peak_only"].as<bool>();
  }
  if (node["peak_neighborhood"]) {
    p->peak_neighborhood = parse_peak_neighborhood(
        node["peak_neighborhood"].as<std::string>());
  }
}

inline void load_dbscan_params(const YAML::Node& node,
                               ClusterDBSCANParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["epsilon"]) {
    p->epsilon = node["epsilon"].as<std::vector<double>>();
  }
  if (node["min_points"]) {
    p->min_points = node["min_points"].as<int64_t>();
  }
  if (node["max_clusters"]) {
    p->max_clusters = node["max_clusters"].as<int64_t>();
  }
}

inline void load_reduction_params(const YAML::Node& node,
                                  DetectionReductionParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["representative"]) {
    p->representative = node["representative"].as<RepresentativePolicy>();
  }
  if (node["score_method"]) {
    p->score_method = node["score_method"].as<ScoreMethod>();
  }
  if (node["feature_weights"]) {
    p->feature_weights = node["feature_weights"].as<std::vector<double>>();
  }
}

inline void load_snapshot_gather_params(const YAML::Node& node,
                                        SnapshotGatherParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["snapshot_window"]) {
    p->snapshot_window = node["snapshot_window"].as<std::vector<int64_t>>();
  }
  if (node["doppler_edge"]) {
    p->doppler_edge = node["doppler_edge"].as<SnapshotEdgeMode>();
  }
  if (node["range_edge"]) {
    p->range_edge = node["range_edge"].as<SnapshotEdgeMode>();
  }
}

inline void load_beamscan_params(const YAML::Node& node, BeamscanParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["angle_convention"]) {
    p->angle_convention = node["angle_convention"].as<AngleConvention>();
  }
  if (node["az_angles"]) {
    p->az_angles = node["az_angles"].as<std::vector<double>>();
  }
  if (node["el_angles"]) {
    p->el_angles = node["el_angles"].as<std::vector<double>>();
  }
  if (node["array_taper"]) {
    p->array_taper = node["array_taper"].as<WindowSpec>();
  }
  if (node["peak_interpolation"]) {
    p->peak_interpolation = node["peak_interpolation"].as<PeakInterpolation>();
  }
  // array_geometry is set by the app from rx_array (not the YAML block).
}

inline void load_deinterleave_params(const YAML::Node& node,
                                     MIMODeinterleaveParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["tx_schedule"]) {
    p->tx_schedule = node["tx_schedule"].as<TXSchedule>();
  }
}

// Load the two dense MIMO operator parameter blocks. Modality selection is
// app-owned; neither operator carries a multiplexing switch.
inline void load_tdm_dense_compensation_params(
    const YAML::Node& node, TDMDenseCompensationParams* p) {
  if (!node || node.IsNull()) return;
  if (node["waveform"]) p->waveform = node["waveform"].as<FMCWWaveformDescriptor>();
  if (node["tx_schedule"]) p->tx_schedule = node["tx_schedule"].as<TXSchedule>();
}

inline void load_ddm_dense_demultiplex_params(
    const YAML::Node& node, DDMDenseDemultiplexParams* p) {
  if (!node || node.IsNull()) return;
  if (node["ddm_code"]) p->ddm_code = node["ddm_code"].as<DDMCodeDescriptor>();
}

inline void load_vaa_params(const YAML::Node& node,
                            VirtualArrayAssemblyParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["overlap_policy"]) {
    p->overlap_policy = node["overlap_policy"].as<OverlapPolicy>();
  }
}

inline void load_phase_shift_beamformer_params(
    const YAML::Node& node, PhaseShiftBeamformerParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["angle_convention"]) {
    p->angle_convention = node["angle_convention"].as<AngleConvention>();
  }
  if (node["az_angles"]) {
    p->az_angles = node["az_angles"].as<std::vector<double>>();
  }
  if (node["el_angles"]) {
    p->el_angles = node["el_angles"].as<std::vector<double>>();
  }
  if (node["array_taper"]) {
    p->array_taper = node["array_taper"].as<WindowSpec>();
  }
  // array_geometry is set by the app from rx_array (not the YAML block).
}

inline void load_mvdr_params(const YAML::Node& node, MVDRParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["angle_convention"]) {
    p->angle_convention = node["angle_convention"].as<AngleConvention>();
  }
  if (node["az_angles"]) {
    p->az_angles = node["az_angles"].as<std::vector<double>>();
  }
  if (node["el_angles"]) {
    p->el_angles = node["el_angles"].as<std::vector<double>>();
  }
  if (node["diagonal_loading"]) {
    p->diagonal_loading = node["diagonal_loading"].as<double>();
  }
  if (node["max_sources"]) {
    p->max_sources = node["max_sources"].as<int64_t>();
  }
  if (node["spatial_smoothing"]) {
    p->spatial_smoothing = node["spatial_smoothing"].as<std::vector<int64_t>>();
  }
  if (node["forward_backward"]) {
    p->forward_backward = node["forward_backward"].as<bool>();
  }
}

inline void load_music_params(const YAML::Node& node, MUSICParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["angle_convention"]) {
    p->angle_convention = node["angle_convention"].as<AngleConvention>();
  }
  if (node["az_angles"]) {
    p->az_angles = node["az_angles"].as<std::vector<double>>();
  }
  if (node["el_angles"]) {
    p->el_angles = node["el_angles"].as<std::vector<double>>();
  }
  if (node["num_sources"]) {
    p->num_sources = node["num_sources"].as<int64_t>();
  }
  if (node["source_count_pfa"]) {
    p->source_count_pfa = node["source_count_pfa"].as<double>();
  }
  if (node["max_sources"]) {
    p->max_sources = node["max_sources"].as<int64_t>();
  }
  if (node["spatial_smoothing"]) {
    p->spatial_smoothing = node["spatial_smoothing"].as<std::vector<int64_t>>();
  }
  if (node["forward_backward"]) {
    p->forward_backward = node["forward_backward"].as<bool>();
  }
}

inline void load_root_music_params(const YAML::Node& node,
                                   RootMUSICParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["angle_convention"]) {
    p->angle_convention = node["angle_convention"].as<AngleConvention>();
  }
  if (node["num_sources"]) {
    p->num_sources = node["num_sources"].as<int64_t>();
  }
  if (node["source_count_pfa"]) {
    p->source_count_pfa = node["source_count_pfa"].as<double>();
  }
  if (node["max_sources"]) {
    p->max_sources = node["max_sources"].as<int64_t>();
  }
  if (node["spatial_smoothing"]) {
    p->spatial_smoothing = node["spatial_smoothing"].as<std::vector<int64_t>>();
  }
  if (node["forward_backward"]) {
    p->forward_backward = node["forward_backward"].as<bool>();
  }
}

inline void load_esprit_params(const YAML::Node& node, ESPRITParams* p) {
  if (!node || node.IsNull()) {
    return;
  }
  if (node["angle_convention"]) {
    p->angle_convention = node["angle_convention"].as<AngleConvention>();
  }
  if (node["num_sources"]) {
    p->num_sources = node["num_sources"].as<int64_t>();
  }
  if (node["source_count_pfa"]) {
    p->source_count_pfa = node["source_count_pfa"].as<double>();
  }
  if (node["max_sources"]) {
    p->max_sources = node["max_sources"].as<int64_t>();
  }
  if (node["spatial_smoothing"]) {
    p->spatial_smoothing = node["spatial_smoothing"].as<std::vector<int64_t>>();
  }
  if (node["forward_backward"]) {
    p->forward_backward = node["forward_backward"].as<bool>();
  }
  // subarray_spec stays default (auto) unless YAML adds it later.
}

}  // namespace holoscan::radar::app
