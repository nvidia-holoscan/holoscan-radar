// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
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

namespace holoscan::radar::app::simo::viz {

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

enum class SimoDisplayMode {
  kRawDetections,
  kClusters,
};

struct SimoDisplayConfig {
  SimoDisplayMode mode{SimoDisplayMode::kRawDetections};
  float dynamic_range_db{60.0F};
  float detection_radius{0.010F};
  double frame_period{0.1};
  std::int64_t history_frames{120};
  std::int64_t detection_trail_frames{12};
  std::int64_t width{1280};
  std::int64_t height{720};
  std::array<float, 2> az_bounds{};
  std::array<float, 2> el_bounds{};
  AngleConvention angle_convention;
  std::string title;
  std::vector<TruthTarget> truth_targets;
};

class SimoDisplayOp : public holoscan::Operator<> {
 public:
  explicit SimoDisplayOp(SimoDisplayConfig config);

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
  holoscan::Output<holoscan::Tensor> rgba{"rgba"};

 private:
  SimoDisplayConfig config_;
  std::size_t max_byte_span_{0U};
  std::shared_ptr<holoscan::Tensor> peak_power_;
  std::shared_ptr<holoscan::Tensor> annotation_;
  holoscan::Tensor retained_input_;
  std::int64_t frame_index_{0};
  std::deque<HistorySample> history_;
  std::deque<HistorySample> detection_history_;
};

namespace detail {

std::vector<std::uint8_t> make_heatmap_rgba_reference(
    const std::vector<float>& power, const std::vector<std::int64_t>& shape,
    float dynamic_range_db);

std::vector<DetectionMarker> make_cluster_markers(
    const std::vector<std::int64_t>& detection_idx,
    const std::vector<float>& points, const std::vector<std::int64_t>& labels,
    const std::vector<std::int64_t>& sizes, std::int64_t idx_cols,
    std::int64_t point_cols,
    const AngleConvention& angle_convention = AngleConvention{});

std::vector<DetectionMarker> make_raw_detection_markers(
    const std::vector<std::int64_t>& detection_idx,
    const std::vector<float>& points, std::int64_t idx_cols,
    std::int64_t point_cols,
    const AngleConvention& angle_convention = AngleConvention{});

void apply_reduced_features(std::vector<DetectionMarker>* markers,
                            const std::vector<float>& reduced_points,
                            std::int64_t reduced_cols,
                            const AngleConvention& angle_convention =
                                AngleConvention{});

void append_history(std::deque<HistorySample>* history,
                    const std::vector<DetectionMarker>& markers,
                    std::int64_t frame, std::int64_t history_frames);

std::array<float, 2> rd_plot_position(float doppler_bin, float range_bin,
                                      std::int64_t rows, std::int64_t cols);

bool use_azimuth_sector(const std::array<float, 2>& az_bounds);

std::array<float, 2> history_plot_position(
    const HistorySample& sample, std::int64_t current_frame,
    std::int64_t history_frames, int feature,
    std::array<float, 2> bounds);

std::array<float, 4> truth_features_at_time(const TruthTarget& target,
                                            double time);

}  // namespace detail

}  // namespace holoscan::radar::app::simo::viz
