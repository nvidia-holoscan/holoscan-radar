// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
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

// Shared completion counter for Graph apps (DetectionReport / ClusterReport).
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

// Reports CFAR detections in physical units via the upstream operators'
// axis outputs. detection_score is the dense CFAR map; each detection_idx
// row indexes into it for the per-cell strength (score > 1 ⇔ detected).
class DetectionReportOp : public holoscan::Operator<> {
 public:
  struct Config {
    std::shared_ptr<CadenceMeter> cadence;
    // When true, aoa_estimates is a required OnAligned Exact input (SIMO /
    // peak_only). When false it stays registered-but-unwired for SISO —
    // wiring it without putting it in contract() yields
    // RUNTIME_OPERATOR_REPLAY_MISMATCH on EA.
    bool report_aoa{false};
  };

  DetectionReportOp() = default;
  explicit DetectionReportOp(Config config)
      : cadence(std::move(config.cadence)), report_aoa_(config.report_aoa) {}

  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(detection_idx)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(detection_score)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(range_bins).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(velocity_bins)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(aoa_estimates)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    if (report_aoa_) {
      result
          .trigger(holoscan::OnAligned{
              .ports = holoscan::ports(detection_idx, detection_score,
                                       range_bins, velocity_bins, aoa_estimates),
              .window = std::chrono::nanoseconds{0}})
          .input(detection_idx, holoscan::Exact{})
          .input(detection_score, holoscan::Exact{})
          .input(range_bins, holoscan::Exact{})
          .input(velocity_bins, holoscan::Exact{})
          .input(aoa_estimates, holoscan::Exact{});
    } else {
      result
          .trigger(holoscan::OnAligned{
              .ports = holoscan::ports(detection_idx, detection_score,
                                       range_bins, velocity_bins),
              .window = std::chrono::nanoseconds{0}})
          .input(detection_idx, holoscan::Exact{})
          .input(detection_score, holoscan::Exact{})
          .input(range_bins, holoscan::Exact{})
          .input(velocity_bins, holoscan::Exact{});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    HOLOSCAN_TRY(auto detection_idx_t,
                 receive_tensor(detection_idx, "DetectionReportOp"));
    HOLOSCAN_TRY(auto detection_score_t,
                 receive_tensor(detection_score, "DetectionReportOp"));
    HOLOSCAN_TRY(auto range_bins_t,
                 receive_tensor(range_bins, "DetectionReportOp"));
    HOLOSCAN_TRY(auto velocity_bins_t,
                 receive_tensor(velocity_bins, "DetectionReportOp"));
    holoscan::Tensor* aoa_ptr = nullptr;
    holoscan::Tensor aoa_required;
    if (report_aoa_) {
      HOLOSCAN_TRY(aoa_required,
                   receive_tensor(aoa_estimates, "DetectionReportOp"));
      aoa_ptr = &aoa_required;
    } else {
      HOLOSCAN_TRY(auto maybe_aoa, receive_optional_value(aoa_estimates));
      if (maybe_aoa.has_value()) {
        aoa_required = std::move(*maybe_aoa);
        aoa_ptr = &aoa_required;
      }
    }

    const auto idx_shape = detection_idx_t.shape();
    if (idx_shape.size() != 2 || idx_shape[1] != 2) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "DetectionReportOp: detection_idx must be (N, 2), got rank " +
              std::to_string(idx_shape.size())});
    }
    const int64_t num_detections = idx_shape[0];
    const int64_t num_range = range_bins_t.shape()[0];
    const int64_t num_doppler = velocity_bins_t.shape()[0];
    const bool have_aoa = aoa_ptr != nullptr;

    std::vector<int64_t> idx(static_cast<size_t>(num_detections * 2));
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));
    std::vector<float> scores(
        static_cast<size_t>(num_doppler * num_range));
    std::vector<float> aoa;
    {
      const auto score_shape = detection_score_t.shape();
      if (score_shape.size() != 2 || score_shape[0] != num_doppler ||
          score_shape[1] != num_range) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: detection_score must be (num_doppler, "
            "num_range) matching the axis bins"});
      }
    }
    if (have_aoa) {
      const auto aoa_shape = aoa_ptr->shape();
      if (aoa_shape.size() != 2 || aoa_shape[1] != 2 ||
          aoa_shape[0] != num_detections) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: aoa_estimates must be (N, 2) row-aligned with "
            "detection_idx"});
      }
      aoa.resize(static_cast<size_t>(num_detections * 2));
    }

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
        require_bytes("detection_idx", detection_idx_t,
                      sizeof(int64_t) * idx.size()));
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("range_bins", range_bins_t, sizeof(float) * ranges.size()));
    HOLOSCAN_RETURN_IF_ERROR(require_bytes(
        "velocity_bins", velocity_bins_t, sizeof(float) * velocities.size()));
    HOLOSCAN_RETURN_IF_ERROR(require_bytes(
        "detection_score", detection_score_t, sizeof(float) * scores.size()));
    if (have_aoa) {
      HOLOSCAN_RETURN_IF_ERROR(require_bytes(
          "aoa_estimates", *aoa_ptr, sizeof(float) * aoa.size()));
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
        copy_d2h("detection_idx", idx.data(), detection_idx_t));
    HOLOSCAN_RETURN_IF_ERROR(copy_d2h("range_bins", ranges.data(), range_bins_t));
    HOLOSCAN_RETURN_IF_ERROR(
        copy_d2h("velocity_bins", velocities.data(), velocity_bins_t));
    HOLOSCAN_RETURN_IF_ERROR(
        copy_d2h("detection_score", scores.data(), detection_score_t));
    if (have_aoa) {
      HOLOSCAN_RETURN_IF_ERROR(
          copy_d2h("aoa_estimates", aoa.data(), *aoa_ptr));
    }

    HOLOSCAN_LOG_INFO("CFAR detections: {}", num_detections);
    constexpr int64_t kMaxLogged = 16;
    const int64_t n_log = std::min(num_detections, kMaxLogged);
    for (int64_t i = 0; i < n_log; ++i) {
      const int64_t doppler_bin = idx[static_cast<size_t>(i * 2 + 0)];
      const int64_t range_bin = idx[static_cast<size_t>(i * 2 + 1)];
      if (doppler_bin < 0 || doppler_bin >= num_doppler || range_bin < 0 ||
          range_bin >= num_range) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "DetectionReportOp: detection_idx out of map bounds"});
      }
      const float rng = ranges[static_cast<size_t>(range_bin)];
      const float vel = velocities[static_cast<size_t>(doppler_bin)];
      const float score =
          scores[static_cast<size_t>(doppler_bin * num_range + range_bin)];
      const float az = have_aoa ? aoa[static_cast<size_t>(i * 2 + 0)] : 0.0f;
      const float el = have_aoa ? aoa[static_cast<size_t>(i * 2 + 1)] : 0.0f;
      if (have_aoa) {
        HOLOSCAN_LOG_INFO(
            "  [{}] score={:.2f} range bin {} ({:.2f} m), Doppler bin {} "
            "({:+.2f} m/s), az {:.3f} rad, el {:.3f} rad",
            i, score, range_bin, rng, doppler_bin, vel, az, el);
      } else {
        HOLOSCAN_LOG_INFO(
            "  [{}] score={:.2f} range bin {} ({:.2f} m), Doppler bin {} "
            "({:+.2f} m/s)",
            i, score, range_bin, rng, doppler_bin, vel);
      }
    }
    if (num_detections > kMaxLogged) {
      HOLOSCAN_LOG_INFO("  ... {} more detections not logged",
                        num_detections - kMaxLogged);
    }
    if (cadence) {
      cadence->on_sink_done();
    }
    AppFrameBank::mark_done();
    return {};
  }

  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Input<holoscan::Tensor> detection_score{"detection_score"};
  holoscan::Input<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Input<holoscan::Tensor> velocity_bins{"velocity_bins"};
  holoscan::Input<holoscan::Tensor> aoa_estimates{"aoa_estimates"};

 private:
  bool report_aoa_{false};
};

// Reports the DBSCAN clusters formed over the CFAR detections.
class ClusterReportOp : public holoscan::Operator<> {
 public:
  struct Config {
    // When true, detection_idx is a required OnAligned Exact input (needed to
    // look up rep az/el from the first four point columns + 2-column rep_idx).
    // for SISO 2-feature points — wiring without contract membership yields
    // RUNTIME_OPERATOR_REPLAY_MISMATCH on EA.
    bool report_detection_idx{false};
    // Dense 4-column rep_idx path: physical angle grids for bin→rad lookup.
    std::vector<double> az_grid;
    std::vector<double> el_grid;
    // Optional e2e meter when ClusterReport is the sole completion sink.
    std::shared_ptr<CadenceMeter> cadence;
  };

  ClusterReportOp() = default;
  explicit ClusterReportOp(Config config)
      : report_detection_idx_(config.report_detection_idx),
        az_grid_(std::move(config.az_grid)),
        el_grid_(std::move(config.el_grid)),
        cadence_(std::move(config.cadence)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(points).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(labels).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(sizes).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(rep_idx).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(range_bins).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(velocity_bins)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(detection_idx)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    if (report_detection_idx_) {
      result
          .trigger(holoscan::OnAligned{
              .ports = holoscan::ports(points, labels, sizes, rep_idx,
                                       range_bins, velocity_bins,
                                       detection_idx),
              .window = std::chrono::nanoseconds{0}})
          .input(points, holoscan::Exact{})
          .input(labels, holoscan::Exact{})
          .input(sizes, holoscan::Exact{})
          .input(rep_idx, holoscan::Exact{})
          .input(range_bins, holoscan::Exact{})
          .input(velocity_bins, holoscan::Exact{})
          .input(detection_idx, holoscan::Exact{});
    } else {
      result
          .trigger(holoscan::OnAligned{
              .ports = holoscan::ports(points, labels, sizes, rep_idx,
                                       range_bins, velocity_bins),
              .window = std::chrono::nanoseconds{0}})
          .input(points, holoscan::Exact{})
          .input(labels, holoscan::Exact{})
          .input(sizes, holoscan::Exact{})
          .input(rep_idx, holoscan::Exact{})
          .input(range_bins, holoscan::Exact{})
          .input(velocity_bins, holoscan::Exact{});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    const std::string who = "ClusterReportOp";
    HOLOSCAN_TRY(auto points_t, receive_tensor(points, who));
    HOLOSCAN_TRY(auto labels_t, receive_tensor(labels, who));
    HOLOSCAN_TRY(auto sizes_t, receive_tensor(sizes, who));
    HOLOSCAN_TRY(auto rep_idx_t, receive_tensor(rep_idx, who));
    HOLOSCAN_TRY(auto range_bins_t, receive_tensor(range_bins, who));
    HOLOSCAN_TRY(auto velocity_bins_t, receive_tensor(velocity_bins, who));
    holoscan::Tensor* det_ptr = nullptr;
    holoscan::Tensor det_required;
    if (report_detection_idx_) {
      HOLOSCAN_TRY(det_required, receive_tensor(detection_idx, who));
      det_ptr = &det_required;
    } else {
      HOLOSCAN_TRY(auto maybe_det, receive_optional_value(detection_idx));
      if (maybe_det.has_value()) {
        det_required = std::move(*maybe_det);
        det_ptr = &det_required;
      }
    }

    const auto pshape = points_t.shape();
    const int64_t num_features = pshape.size() == 2 ? pshape[1] : 0;
    if (pshape.size() != 2 || (num_features != 2 && num_features < 4)) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "ClusterReportOp: points must be (N, 2) or have at least four "
          "columns, got rank " +
              std::to_string(pshape.size())});
    }
    const bool has_angles = num_features >= 4;
    const int64_t num_points = pshape[0];
    const auto lshape = labels_t.shape();
    if (lshape.size() != 1 || lshape[0] != num_points) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "ClusterReportOp: labels must be (N,) row-aligned with points"});
    }
    const auto sshape = sizes_t.shape();
    if (sshape.size() != 1) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "ClusterReportOp: sizes must be rank-1"});
    }
    const int64_t num_clusters = sshape[0];

    const auto rshape = rep_idx_t.shape();
    if (rshape.size() != 2 || (rshape[1] != 2 && rshape[1] != 4) ||
        rshape[0] != num_clusters) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "ClusterReportOp: rep_idx must be (num_clusters, 2) or "
          "(num_clusters, 4)"});
    }
    const int64_t rep_cols = rshape[1];
    const int64_t num_az = static_cast<int64_t>(az_grid_.size());
    const int64_t num_el = static_cast<int64_t>(el_grid_.size());
    const bool use_det_idx = has_angles && rep_cols == 2;
    if (has_angles) {
      if (rep_cols == 4) {
        if (num_az == 0 || num_el == 0) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kInvalidArgument,
              "ClusterReportOp: az_grid/el_grid required for a 4-column "
              "rep_idx"});
        }
      } else if (det_ptr == nullptr) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "ClusterReportOp: detection_idx required when points has az/el "
            "columns and rep_idx is 2-column"});
      }
    } else if (rep_cols == 4) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "ClusterReportOp: 4-column rep_idx requires points beginning with "
          "(range, velocity, az, el)"});
    }
    const int64_t num_range = range_bins_t.shape()[0];
    const int64_t num_doppler = velocity_bins_t.shape()[0];

    std::vector<float> pts(static_cast<size_t>(num_points * num_features));
    std::vector<int64_t> lab(static_cast<size_t>(num_points));
    std::vector<int64_t> sz(static_cast<size_t>(num_clusters));
    std::vector<int64_t> rep(static_cast<size_t>(num_clusters * rep_cols));
    std::vector<int64_t> det_idx;
    if (use_det_idx) {
      const auto dshape = det_ptr->shape();
      if (dshape.size() != 2 || dshape[1] != 2 || dshape[0] != num_points) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "ClusterReportOp: detection_idx must be (N, 2) row-aligned with "
            "points"});
      }
      det_idx.resize(static_cast<size_t>(num_points * 2));
    }
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));

    const auto require_bytes =
        [](const char* name, const holoscan::Tensor& t,
           size_t expected) -> holoscan::expected<void, holoscan::Error> {
      if (static_cast<size_t>(t.nbytes()) != expected) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            std::string("ClusterReportOp: ") + name +
                " byte size mismatch (expected " + std::to_string(expected) +
                ", got " + std::to_string(t.nbytes()) + ")"});
      }
      return {};
    };
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("points", points_t, sizeof(float) * pts.size()));
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("labels", labels_t, sizeof(int64_t) * lab.size()));
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("sizes", sizes_t, sizeof(int64_t) * sz.size()));
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("rep_idx", rep_idx_t, sizeof(int64_t) * rep.size()));
    HOLOSCAN_RETURN_IF_ERROR(
        require_bytes("range_bins", range_bins_t, sizeof(float) * ranges.size()));
    HOLOSCAN_RETURN_IF_ERROR(require_bytes(
        "velocity_bins", velocity_bins_t, sizeof(float) * velocities.size()));

    if (cudaDeviceSynchronize() != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          "ClusterReportOp: device synchronize failed"});
    }
    const auto copy_d2h =
        [](const char* name, void* dst,
           const holoscan::Tensor& src) -> holoscan::expected<void, holoscan::Error> {
      if (src.nbytes() == 0) {
        return {};
      }
      if (cudaMemcpy(dst, src.data(), static_cast<size_t>(src.nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string("ClusterReportOp: ") + name + " D2H copy failed"});
      }
      return {};
    };
    HOLOSCAN_RETURN_IF_ERROR(copy_d2h("points", pts.data(), points_t));
    HOLOSCAN_RETURN_IF_ERROR(copy_d2h("labels", lab.data(), labels_t));
    HOLOSCAN_RETURN_IF_ERROR(copy_d2h("sizes", sz.data(), sizes_t));
    HOLOSCAN_RETURN_IF_ERROR(copy_d2h("rep_idx", rep.data(), rep_idx_t));
    HOLOSCAN_RETURN_IF_ERROR(copy_d2h("range_bins", ranges.data(), range_bins_t));
    HOLOSCAN_RETURN_IF_ERROR(
        copy_d2h("velocity_bins", velocities.data(), velocity_bins_t));
    if (use_det_idx) {
      HOLOSCAN_RETURN_IF_ERROR(
          copy_d2h("detection_idx", det_idx.data(), *det_ptr));
    }

    std::vector<double> sum_range(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_velocity(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_az;
    std::vector<double> sum_el;
    if (has_angles) {
      sum_az.assign(static_cast<size_t>(num_clusters), 0.0);
      sum_el.assign(static_cast<size_t>(num_clusters), 0.0);
    }
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
            "ClusterReportOp: label out of range for sizes"});
      }
      sum_range[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * num_features + 0)];
      sum_velocity[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * num_features + 1)];
      if (has_angles) {
        sum_az[static_cast<size_t>(c)] +=
            pts[static_cast<size_t>(i * num_features + 2)];
        sum_el[static_cast<size_t>(c)] +=
            pts[static_cast<size_t>(i * num_features + 3)];
      }
    }

    const auto rep_az_el_from_detidx = [&](int64_t rep_dd, int64_t rep_rr,
                                           double* out_az, double* out_el) {
      for (int64_t i = 0; i < num_points; ++i) {
        if (det_idx[static_cast<size_t>(i * 2 + 0)] == rep_dd &&
            det_idx[static_cast<size_t>(i * 2 + 1)] == rep_rr) {
          *out_az = pts[static_cast<size_t>(i * num_features + 2)];
          *out_el = pts[static_cast<size_t>(i * num_features + 3)];
          return;
        }
      }
      *out_az = 0.0;
      *out_el = 0.0;
    };

    HOLOSCAN_LOG_INFO("DBSCAN clusters: {} ({} points, {} noise)",
                      num_clusters, num_points, noise);
    constexpr int64_t kMaxLogged = 16;
    const int64_t n_log = std::min(num_clusters, kMaxLogged);
    for (int64_t c = 0; c < n_log; ++c) {
      const int64_t n = sz[static_cast<size_t>(c)];
      const double inv = n > 0 ? 1.0 / static_cast<double>(n) : 0.0;
      int64_t rep_dd = 0;
      int64_t rep_rr = 0;
      double rep_az = 0.0;
      double rep_el = 0.0;
      if (rep_cols == 4) {
        const int64_t az_bin = rep[static_cast<size_t>(c * 4 + 0)];
        const int64_t el_bin = rep[static_cast<size_t>(c * 4 + 1)];
        rep_dd = rep[static_cast<size_t>(c * 4 + 2)];
        rep_rr = rep[static_cast<size_t>(c * 4 + 3)];
        if (az_bin < 0 || az_bin >= num_az || el_bin < 0 || el_bin >= num_el ||
            rep_dd < 0 || rep_dd >= num_doppler || rep_rr < 0 ||
            rep_rr >= num_range) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kInvalidArgument,
              "ClusterReportOp: rep_idx out of grid/axis bounds"});
        }
        rep_az = az_grid_[static_cast<size_t>(az_bin)];
        rep_el = el_grid_[static_cast<size_t>(el_bin)];
      } else {
        rep_dd = rep[static_cast<size_t>(c * 2 + 0)];
        rep_rr = rep[static_cast<size_t>(c * 2 + 1)];
        if (rep_dd < 0 || rep_dd >= num_doppler || rep_rr < 0 ||
            rep_rr >= num_range) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kInvalidArgument,
              "ClusterReportOp: rep_idx out of axis bounds"});
        }
        if (use_det_idx) {
          rep_az_el_from_detidx(rep_dd, rep_rr, &rep_az, &rep_el);
        }
      }
      if (has_angles) {
        HOLOSCAN_LOG_INFO(
            "  cluster {}: {} pts, centroid range {:.2f} m, velocity "
            "{:+.2f} m/s, az {:.3f} rad, el {:.3f} rad; rep range {:.2f} m, "
            "velocity {:+.2f} m/s, az {:.3f} rad, el {:.3f} rad",
            c, n, sum_range[static_cast<size_t>(c)] * inv,
            sum_velocity[static_cast<size_t>(c)] * inv,
            sum_az[static_cast<size_t>(c)] * inv,
            sum_el[static_cast<size_t>(c)] * inv,
            ranges[static_cast<size_t>(rep_rr)],
            velocities[static_cast<size_t>(rep_dd)], rep_az, rep_el);
      } else {
        HOLOSCAN_LOG_INFO(
            "  cluster {}: {} pts, centroid range {:.2f} m, velocity "
            "{:+.2f} m/s; rep range {:.2f} m, velocity {:+.2f} m/s",
            c, n, sum_range[static_cast<size_t>(c)] * inv,
            sum_velocity[static_cast<size_t>(c)] * inv,
            ranges[static_cast<size_t>(rep_rr)],
            velocities[static_cast<size_t>(rep_dd)]);
      }
    }
    if (num_clusters > kMaxLogged) {
      HOLOSCAN_LOG_INFO("  ... {} more clusters not logged",
                        num_clusters - kMaxLogged);
    }
    if (cadence_) {
      cadence_->on_sink_done();
      // Dense apps without DetectionReportOp: ClusterReport is the e2e sink.
      AppFrameBank::mark_done();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> points{"points"};
  holoscan::Input<holoscan::Tensor> labels{"labels"};
  holoscan::Input<holoscan::Tensor> sizes{"sizes"};
  holoscan::Input<holoscan::Tensor> rep_idx{"rep_idx"};
  holoscan::Input<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Input<holoscan::Tensor> velocity_bins{"velocity_bins"};
  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};

 private:
  bool report_detection_idx_{false};
  std::vector<double> az_grid_;
  std::vector<double> el_grid_;
  std::shared_ptr<CadenceMeter> cadence_;
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
