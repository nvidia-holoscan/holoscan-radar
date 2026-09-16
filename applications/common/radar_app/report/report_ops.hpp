// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_app/report/cadence_meter.hpp"
#include "radar_common/sdk/tensor_io.hpp"

// HSDK 5 Graph logging / drain sinks for the FMCW example applications.
namespace holoscan::radar::app {

// Shared completion counter for Graph apps.
struct AppFrameBank {
  static void reset(int count) {
    const std::lock_guard lock(mutex);
    target = count;
    captured = 0;
  }

  static void mark_done() {
    {
      const std::lock_guard lock(mutex);
      ++captured;
    }
    ready.notify_all();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline int target{0};
  static inline int captured{0};
};

enum class DetectionReportMode {
  kRawDetections,
  kClusters,
};

// Reports raw CFAR detections or grouped detections in physical units.
class DetectionReportOp : public holoscan::Operator<> {
 public:
  struct Config {
    DetectionReportMode mode{DetectionReportMode::kRawDetections};
    // Cluster mode only: for each detection_idx coordinate, the points column
    // containing that raw integer coordinate.
    std::vector<int64_t> point_index_columns;
    std::shared_ptr<CadenceMeter> cadence;
  };

  DetectionReportOp() = default;
  explicit DetectionReportOp(Config config)
      : cadence(std::move(config.cadence)),
        mode_(config.mode),
        point_index_columns_(std::move(config.point_index_columns)) {}

  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(points).queue_depth(2U).expects_tensor(
        holoscan::radar::tensor_input(kDevice));
    spec.input(detection_idx).queue_depth(2U).expects_tensor(
        holoscan::radar::tensor_input(kDevice));
    if (mode_ == DetectionReportMode::kClusters) {
      spec.input(labels).queue_depth(2U).expects_tensor(
          holoscan::radar::tensor_input(kDevice));
      spec.input(sizes).queue_depth(2U).expects_tensor(
          holoscan::radar::tensor_input(kDevice));
    }
    spec.input(range_bins).queue_depth(2U).expects_tensor(
        holoscan::radar::tensor_input(kDevice));
    spec.input(velocity_bins)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    if (mode_ == DetectionReportMode::kClusters) {
      result
          .trigger(holoscan::OnAligned{
              .ports = holoscan::ports(points, detection_idx, labels, sizes,
                                       range_bins, velocity_bins),
              .window = std::chrono::nanoseconds{0}})
          .input(points, holoscan::Exact{})
          .input(detection_idx, holoscan::Exact{})
          .input(labels, holoscan::Exact{})
          .input(sizes, holoscan::Exact{})
          .input(range_bins, holoscan::Exact{})
          .input(velocity_bins, holoscan::Exact{});
    } else {
      result
          .trigger(holoscan::OnAligned{
              .ports = holoscan::ports(points, detection_idx, range_bins,
                                       velocity_bins),
              .window = std::chrono::nanoseconds{0}})
          .input(points, holoscan::Exact{})
          .input(detection_idx, holoscan::Exact{})
          .input(range_bins, holoscan::Exact{})
          .input(velocity_bins, holoscan::Exact{});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    constexpr const char* kWho = "DetectionReportOp";
    HOLOSCAN_TRY(auto points_t, receive_tensor(points, kWho));
    HOLOSCAN_TRY(auto detection_idx_t,
                 receive_tensor(detection_idx, kWho));
    HOLOSCAN_TRY(auto range_bins_t, receive_tensor(range_bins, kWho));
    HOLOSCAN_TRY(auto velocity_bins_t, receive_tensor(velocity_bins, kWho));

    const auto pshape = points_t.shape();
    const auto idx_shape = detection_idx_t.shape();
    if (pshape.size() != 2U || pshape[1] < 4) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "DetectionReportOp: points must have shape (N, F) with F >= 4"});
    }
    if (idx_shape.size() != 2U || idx_shape[1] < 2) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "DetectionReportOp: detection_idx must have shape (N, K) with "
          "K >= 2"});
    }
    if (range_bins_t.shape().size() != 1U ||
        velocity_bins_t.shape().size() != 1U) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "DetectionReportOp: range_bins and velocity_bins must be rank-1"});
    }
    const int64_t num_points = pshape[0];
    const int64_t num_features = pshape[1];
    const int64_t num_reports = idx_shape[0];
    const int64_t idx_cols = idx_shape[1];
    const int64_t num_range = range_bins_t.shape()[0];
    const int64_t num_doppler = velocity_bins_t.shape()[0];

    std::vector<float> pts(static_cast<size_t>(num_points * num_features));
    std::vector<int64_t> idx(static_cast<size_t>(num_reports * idx_cols));
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));
    std::vector<int64_t> lab;
    std::vector<int64_t> sz;
    holoscan::Tensor labels_t;
    holoscan::Tensor sizes_t;

    const auto require_bytes =
        [](const char* name, const holoscan::Tensor& t,
           size_t expected) -> holoscan::expected<void, holoscan::Error> {
      if (static_cast<size_t>(t.nbytes()) != expected) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            std::string("DetectionReportOp: ") + name +
                " byte size mismatch (expected " + std::to_string(expected) +
                ", got " + std::to_string(t.nbytes()) + ")"});
      }
      return {};
    };
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("points", points_t, sizeof(float) * pts.size()));
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("detection_idx", detection_idx_t,
                      sizeof(int64_t) * idx.size()));
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("range_bins", range_bins_t, sizeof(float) * ranges.size()));
    HOLOSCAN_RETURN_IF_ERROR(require_bytes(
        "velocity_bins", velocity_bins_t, sizeof(float) * velocities.size()));

    if (mode_ == DetectionReportMode::kClusters) {
      HOLOSCAN_TRY(labels_t, receive_tensor(labels, kWho));
      HOLOSCAN_TRY(sizes_t, receive_tensor(sizes, kWho));
      const auto lshape = labels_t.shape();
      const auto sshape = sizes_t.shape();
      if (lshape.size() != 1U || lshape[0] != num_points) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: labels must be (N,) row-aligned with points"});
      }
      if (sshape.size() != 1U || sshape[0] != num_reports) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: sizes and detection_idx must have one row "
            "per cluster"});
      }
      if (point_index_columns_.size() != static_cast<size_t>(idx_cols)) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: point_index_columns must have one entry per "
            "detection_idx column in cluster mode"});
      }
      std::vector<bool> seen(static_cast<size_t>(num_features), false);
      for (const int64_t column : point_index_columns_) {
        if (column < 0 || column >= num_features ||
            seen[static_cast<size_t>(column)]) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kInvalidArgument,
              "DetectionReportOp: point_index_columns entries must be unique "
              "valid points columns"});
        }
        seen[static_cast<size_t>(column)] = true;
      }
      lab.resize(static_cast<size_t>(num_points));
      sz.resize(static_cast<size_t>(num_reports));
      HOLOSCAN_RETURN_IF_ERROR(require_bytes(
          "labels", labels_t, sizeof(int64_t) * lab.size()));
      HOLOSCAN_RETURN_IF_ERROR(require_bytes(
          "sizes", sizes_t, sizeof(int64_t) * sz.size()));
    } else if (num_reports != num_points) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "DetectionReportOp: raw detection_idx must be row-aligned with "
          "points"});
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          "DetectionReportOp: device synchronize failed"});
    }
    const auto copy_d2h =
        [](const char* name, void* dst,
           const holoscan::Tensor& t) -> holoscan::expected<void, holoscan::Error> {
      if (t.nbytes() == 0) {
        return {};
      }
      if (cudaMemcpy(dst, t.data(), static_cast<size_t>(t.nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string("DetectionReportOp: ") + name + " D2H copy failed"});
      }
      return {};
    };
    HOLOSCAN_RETURN_IF_ERROR(
        copy_d2h("points", pts.data(), points_t));
    HOLOSCAN_RETURN_IF_ERROR(
        copy_d2h("detection_idx", idx.data(), detection_idx_t));
    HOLOSCAN_RETURN_IF_ERROR(copy_d2h("range_bins", ranges.data(), range_bins_t));
    HOLOSCAN_RETURN_IF_ERROR(
        copy_d2h("velocity_bins", velocities.data(), velocity_bins_t));
    if (mode_ == DetectionReportMode::kClusters) {
      HOLOSCAN_RETURN_IF_ERROR(copy_d2h("labels", lab.data(), labels_t));
      HOLOSCAN_RETURN_IF_ERROR(copy_d2h("sizes", sz.data(), sizes_t));
    }

    if (mode_ == DetectionReportMode::kClusters) {
      HOLOSCAN_RETURN_IF_ERROR(report_clusters(
          pts, idx, lab, sz, ranges, velocities, num_points, num_features,
          num_reports, idx_cols));
    } else {
      HOLOSCAN_RETURN_IF_ERROR(report_raw(
          pts, idx, ranges, velocities, num_points, num_features, idx_cols));
    }
    if (cadence) {
      cadence->on_sink_done();
    }
    AppFrameBank::mark_done();
    return {};
  }

  holoscan::Input<holoscan::Tensor> points{"points"};
  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Input<holoscan::Tensor> labels{"labels"};
  holoscan::Input<holoscan::Tensor> sizes{"sizes"};
  holoscan::Input<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Input<holoscan::Tensor> velocity_bins{"velocity_bins"};

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> report_raw(
      const std::vector<float>& pts, const std::vector<int64_t>& idx,
      const std::vector<float>& ranges, const std::vector<float>& velocities,
      int64_t num_points, int64_t num_features, int64_t idx_cols) const {
    HOLOSCAN_LOG_INFO("CFAR detections: {}", num_points);
    constexpr int64_t kMaxLogged = 16;
    const int64_t n_log = std::min(num_points, kMaxLogged);
    for (int64_t i = 0; i < n_log; ++i) {
      const size_t index_offset = static_cast<size_t>(i * idx_cols);
      const size_t point_offset = static_cast<size_t>(i * num_features);
      const int64_t doppler_bin =
          idx[index_offset + static_cast<size_t>(idx_cols - 2)];
      const int64_t range_bin =
          idx[index_offset + static_cast<size_t>(idx_cols - 1)];
      if (doppler_bin < 0 ||
          doppler_bin >= static_cast<int64_t>(velocities.size()) ||
          range_bin < 0 ||
          range_bin >= static_cast<int64_t>(ranges.size())) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: detection_idx out of map bounds"});
      }
      const float score =
          num_features > 4 ? pts[point_offset + 4U] : 0.0F;
      HOLOSCAN_LOG_INFO(
          "  [{}] score={:.2f} range bin {} ({:.2f} m), Doppler bin {} "
          "({:+.2f} m/s), az {:.3f} rad, el {:.3f} rad",
          i, score, range_bin, pts[point_offset], doppler_bin,
          pts[point_offset + 1U], pts[point_offset + 2U],
          pts[point_offset + 3U]);
    }
    if (num_points > kMaxLogged) {
      HOLOSCAN_LOG_INFO("  ... {} more detections not logged",
                        num_points - kMaxLogged);
    }
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> report_clusters(
      const std::vector<float>& pts, const std::vector<int64_t>& idx,
      const std::vector<int64_t>& lab, const std::vector<int64_t>& sz,
      const std::vector<float>& ranges, const std::vector<float>& velocities,
      int64_t num_points, int64_t num_features, int64_t num_clusters,
      int64_t idx_cols) const {
    std::vector<double> sum_range(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_velocity(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_az(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_el(static_cast<size_t>(num_clusters), 0.0);
    std::vector<int64_t> counts(static_cast<size_t>(num_clusters), 0);
    int64_t noise = 0;
    for (int64_t i = 0; i < num_points; ++i) {
      const int64_t c = lab[static_cast<size_t>(i)];
      if (c < 0) {
        ++noise;
        continue;
      }
      if (c >= num_clusters) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: label out of range for sizes"});
      }
      sum_range[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * num_features + 0)];
      sum_velocity[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * num_features + 1)];
      sum_az[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * num_features + 2)];
      sum_el[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * num_features + 3)];
      ++counts[static_cast<size_t>(c)];
    }

    HOLOSCAN_LOG_INFO("DBSCAN clusters: {} ({} points, {} noise)",
                      num_clusters, num_points, noise);
    constexpr int64_t kMaxLogged = 16;
    const int64_t n_log = std::min(num_clusters, kMaxLogged);
    for (int64_t c = 0; c < n_log; ++c) {
      const int64_t n = sz[static_cast<size_t>(c)];
      if (n <= 0 || counts[static_cast<size_t>(c)] != n) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: cluster size does not match labels"});
      }
      const double inv = 1.0 / static_cast<double>(n);
      const size_t index_offset = static_cast<size_t>(c * idx_cols);
      const int64_t rep_dd =
          idx[index_offset + static_cast<size_t>(idx_cols - 2)];
      const int64_t rep_rr =
          idx[index_offset + static_cast<size_t>(idx_cols - 1)];
      if (rep_dd < 0 ||
          rep_dd >= static_cast<int64_t>(velocities.size()) || rep_rr < 0 ||
          rep_rr >= static_cast<int64_t>(ranges.size())) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: representative detection_idx out of axis "
            "bounds"});
      }
      int64_t representative_row = -1;
      for (int64_t row = 0; row < num_points; ++row) {
        bool matches = true;
        for (int64_t column = 0; column < idx_cols; ++column) {
          const float point_index = pts[static_cast<size_t>(
              row * num_features +
              point_index_columns_[static_cast<size_t>(column)])];
          const int64_t expected =
              idx[index_offset + static_cast<size_t>(column)];
          if (!std::isfinite(point_index) ||
              point_index != static_cast<float>(expected)) {
            matches = false;
            break;
          }
        }
        if (matches) {
          representative_row = row;
          break;
        }
      }
      if (representative_row < 0) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: representative detection_idx has no matching "
            "points row"});
      }
      if (lab[static_cast<size_t>(representative_row)] != c) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: representative detection_idx does not belong "
            "to its cluster"});
      }
      const size_t rep_offset =
          static_cast<size_t>(representative_row * num_features);
      HOLOSCAN_LOG_INFO(
          "  cluster {}: {} pts, centroid range {:.2f} m, velocity "
          "{:+.2f} m/s, az {:.3f} rad, el {:.3f} rad; rep range {:.2f} m, "
          "velocity {:+.2f} m/s, az {:.3f} rad, el {:.3f} rad",
          c, n, sum_range[static_cast<size_t>(c)] * inv,
          sum_velocity[static_cast<size_t>(c)] * inv,
          sum_az[static_cast<size_t>(c)] * inv,
          sum_el[static_cast<size_t>(c)] * inv,
          ranges[static_cast<size_t>(rep_rr)],
          velocities[static_cast<size_t>(rep_dd)], pts[rep_offset + 2U],
          pts[rep_offset + 3U]);
    }
    if (num_clusters > kMaxLogged) {
      HOLOSCAN_LOG_INFO("  ... {} more clusters not logged",
                        num_clusters - kMaxLogged);
    }
    return {};
  }

  DetectionReportMode mode_{DetectionReportMode::kRawDetections};
  std::vector<int64_t> point_index_columns_;
};

// Drain an always-emitted Tensor output so the producer can keep ticking.
class NullSinkOp : public holoscan::Operator<> {
 public:
  NullSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input).queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> input{"input"};
};

}  // namespace holoscan::radar::app
