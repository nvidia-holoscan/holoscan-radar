// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/sim/scene.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"

namespace holoscan::radar::app::viz {

// Common render/history item produced from either a raw detection row or a
// clustered detection centroid.
struct DetectionMarker {
  float doppler_bin{0.0F};
  float range_bin{0.0F};
  std::array<float, 4> features{};
  int color_bucket{0};
};

struct HistorySample {
  std::int64_t frame{0};
  float doppler_bin{0.0F};
  float range_bin{0.0F};
  std::array<float, 4> features{};
  int color_bucket{0};
};

struct TruthTarget {
  std::array<double, 3> position{};
  std::array<double, 3> velocity{};
  int color_bucket{0};
};

enum class DashboardMode {
  kRawDetections,
  kClusters,
};

namespace detail {

struct BackgroundStatistics {
  float mean_db;
  float stddev_db;
  std::size_t count;
};

// A clipped display-background estimate, not a thermal-noise calibration.
std::optional<BackgroundStatistics> background_statistics(const std::vector<float>& samples_db);

struct BackgroundScaleState {
  std::optional<float> floor_db;
  std::optional<std::int64_t> last_valid_ns;

  std::array<float, 2> limits(const std::optional<BackgroundStatistics>& background,
                              float peak_power, float span_db, float floor_sigma,
                              std::int64_t now_ns);
};

}  // namespace detail

struct FmcwDashboardConfig {
  DashboardMode mode{DashboardMode::kRawDetections};
  bool show_angles{true};
  float dynamic_range_db{60.0F};
  bool noise_scaling{true};
  std::int64_t noise_sample_limit{2048};
  std::int64_t noise_range_half_width{3};
  std::int64_t noise_doppler_half_width{3};
  double noise_floor_sigma{2.0};
  double range_doppler_width_percent{45.0};
  float detection_radius{0.010F};
  double frame_period{0.1};
  double display_rate_hz{60.0};
  std::int64_t history_frames{120};
  std::int64_t detection_trail_frames{12};
  std::int64_t width{1280};
  std::int64_t height{720};
  std::array<float, 2> az_bounds{};
  std::array<float, 2> el_bounds{};
  AngleConvention angle_convention;
  std::string title;
  std::string selected_profile;
  FMCWWaveformDescriptor selected_waveform;
  std::vector<TruthTarget> truth_targets;
  sim::SceneVisibility scene_visibility{};
};

class FmcwDashboardOp : public holoscan::Operator<> {
 public:
  explicit FmcwDashboardOp(FmcwDashboardConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> cfar_power{"cfar_power"};
  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Input<holoscan::Tensor> points{"points"};
  holoscan::Input<holoscan::Tensor> labels{"labels"};
  holoscan::Input<holoscan::Tensor> sizes{"sizes"};
  holoscan::Input<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Input<holoscan::Tensor> velocity_bins{"velocity_bins"};
  holoscan::Input<holoscan::Tensor> reduced_points{"reduced_points"};
  holoscan::Input<wire::FMCWWaveformT> waveform_in{"waveform"};
  holoscan::Output<holoscan::Tensor> rgba{"rgba"};
  holoscan::Output<holoscan::Tensor> text{"text"};

 private:
  FmcwDashboardConfig config_;
  std::size_t max_byte_span_{0U};
  std::shared_ptr<holoscan::Tensor> peak_power_;
  std::shared_ptr<holoscan::Tensor> annotation_;
  std::shared_ptr<holoscan::Tensor> noise_mask_;
  std::shared_ptr<holoscan::Tensor> noise_samples_;
  detail::BackgroundScaleState noise_scale_;
  holoscan::Tensor retained_input_;
  std::int64_t frame_index_{0};
  std::optional<std::int64_t> last_render_ns_;
  std::deque<HistorySample> history_;
  std::deque<HistorySample> detection_history_;
};

namespace detail {

expected<bool, Error> selected_profile_matches(
    const std::optional<wire::FMCWWaveformT>& applied_waveform, std::string_view selected_profile,
    const FMCWWaveformDescriptor& selected_waveform);

std::vector<std::uint8_t> make_heatmap_rgba_reference(const std::vector<float>& power,
                                                      const std::vector<std::int64_t>& shape,
                                                      float dynamic_range_db);

std::vector<DetectionMarker> make_cluster_markers(
    const std::vector<std::int64_t>& detection_idx, const std::vector<float>& points,
    const std::vector<std::int64_t>& labels, const std::vector<std::int64_t>& sizes,
    std::int64_t idx_cols, std::int64_t point_cols,
    const AngleConvention& angle_convention = AngleConvention{}, bool show_angles = true);

std::vector<DetectionMarker> make_raw_detection_markers(
    const std::vector<std::int64_t>& detection_idx, const std::vector<float>& points,
    std::int64_t idx_cols, std::int64_t point_cols,
    const AngleConvention& angle_convention = AngleConvention{}, bool show_angles = true);

void apply_reduced_features(std::vector<DetectionMarker>* markers,
                            const std::vector<float>& reduced_points, std::int64_t reduced_cols,
                            const AngleConvention& angle_convention = AngleConvention{},
                            bool show_angles = true);

void append_history(std::deque<HistorySample>* history, const std::vector<DetectionMarker>& markers,
                    std::int64_t frame, std::int64_t history_frames);

std::array<float, 2> rd_plot_position(float doppler_bin, float range_bin, std::int64_t rows,
                                      std::int64_t cols, bool reverse_velocity = false);

bool use_azimuth_sector(const std::array<float, 2>& az_bounds);

// Unit-panel coordinates for an aspect-fitted polar sector. Out-of-view samples
// have no position, so histories cannot draw across neighboring panels.
std::optional<std::array<float, 2>> sector_plot_position(float range_fraction, float azimuth,
                                                         const std::array<float, 2>& az_bounds);

std::array<float, 2> history_plot_position(const HistorySample& sample, std::int64_t current_frame,
                                           std::int64_t history_frames, int feature,
                                           std::array<float, 2> bounds);

std::array<float, 4> truth_features_at_time(const TruthTarget& target, double time);

bool truth_target_is_visible_at_time(const TruthTarget& target, double time,
                                     const sim::SceneVisibility& visibility);

}  // namespace detail

}  // namespace holoscan::radar::app::viz
