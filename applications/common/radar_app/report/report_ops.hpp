// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/sdk/tensor_io.hpp"

// Reusable logging / drain sinks for the FMCW example applications. These read
// the detector / clusterer outputs (physical units via the range/velocity axis
// bins) and log them; they carry no app-specific state, so siso, simo, and
// mimo share them unchanged. Header-only: the operators' methods are defined
// in-class (implicitly inline), so several apps can include this with no ODR
// conflict.
namespace holoscan::radar::app {

// Reports CFAR detections in physical units via the upstream operators'
// axis outputs. detection_score is the dense CFAR map; each detection_idx
// row indexes into it for the per-cell strength (score > 1 ⇔ detected).
class DetectionReportOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DetectionReportOp)
  DetectionReportOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("detection_idx");
    spec.input<holoscan::TensorMap>("detection_score");
    spec.input<holoscan::TensorMap>("range_bins");
    spec.input<holoscan::TensorMap>("velocity_bins");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    auto detection_idx = single_tensor(op_input, "detection_idx");
    auto detection_score = single_tensor(op_input, "detection_score");
    auto range_bins = single_tensor(op_input, "range_bins");
    auto velocity_bins = single_tensor(op_input, "velocity_bins");

    const auto idx_shape = detection_idx->shape();
    const auto score_shape = detection_score->shape();
    if (idx_shape.size() != 2 || idx_shape[1] != 2) {
      throw std::runtime_error(
          "DetectionReportOp: detection_idx must be (N, 2), got rank " +
          std::to_string(idx_shape.size()));
    }
    if (score_shape.size() != 2) {
      throw std::runtime_error(
          "DetectionReportOp: SISO expects a rank-2 detection_score, got "
          "rank " +
          std::to_string(score_shape.size()));
    }
    const int64_t num_detections = idx_shape[0];
    const int64_t num_doppler = score_shape[0];
    const int64_t num_range = score_shape[1];

    std::vector<int64_t> idx(static_cast<size_t>(num_detections * 2));
    std::vector<float> scores(
        static_cast<size_t>(num_doppler * num_range));
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));

    const auto require_bytes =
        [](const char* name, const std::shared_ptr<holoscan::Tensor>& t,
           size_t expected) {
          if (static_cast<size_t>(t->nbytes()) != expected) {
            throw std::runtime_error(
                std::string("DetectionReportOp: ") + name +
                " byte size mismatch (expected " + std::to_string(expected) +
                ", got " + std::to_string(t->nbytes()) + ")");
          }
        };
    require_bytes("detection_idx", detection_idx,
                  sizeof(int64_t) * idx.size());
    require_bytes("detection_score", detection_score,
                  sizeof(float) * scores.size());
    require_bytes("range_bins", range_bins, sizeof(float) * ranges.size());
    require_bytes("velocity_bins", velocity_bins,
                  sizeof(float) * velocities.size());

    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: device synchronize failed");
    }
    if (cudaMemcpy(idx.data(), detection_idx->data(),
                   static_cast<size_t>(detection_idx->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: detection_idx D2H copy failed");
    }
    if (cudaMemcpy(scores.data(), detection_score->data(),
                   static_cast<size_t>(detection_score->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: detection_score D2H copy failed");
    }
    if (cudaMemcpy(ranges.data(), range_bins->data(),
                   static_cast<size_t>(range_bins->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: range_bins D2H copy failed");
    }
    if (cudaMemcpy(velocities.data(), velocity_bins->data(),
                   static_cast<size_t>(velocity_bins->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: velocity_bins D2H copy failed");
    }

    HOLOSCAN_LOG_INFO("CFAR detections: {}", num_detections);
    constexpr int64_t kMaxLogged = 16;
    const int64_t n_log = std::min(num_detections, kMaxLogged);
    for (int64_t i = 0; i < n_log; ++i) {
      const int64_t doppler_bin = idx[static_cast<size_t>(i * 2 + 0)];
      const int64_t range_bin = idx[static_cast<size_t>(i * 2 + 1)];
      if (doppler_bin < 0 || doppler_bin >= num_doppler || range_bin < 0 ||
          range_bin >= num_range) {
        throw std::runtime_error(
            "DetectionReportOp: detection_idx out of map bounds");
      }
      const float score =
          scores[static_cast<size_t>(doppler_bin * num_range + range_bin)];
      HOLOSCAN_LOG_INFO(
          "  [{}] score={:.2f} range bin {} ({:.2f} m), Doppler bin {} "
          "({:+.2f} m/s)",
          i, score, range_bin, ranges[static_cast<size_t>(range_bin)],
          doppler_bin, velocities[static_cast<size_t>(doppler_bin)]);
    }
    if (num_detections > kMaxLogged) {
      HOLOSCAN_LOG_INFO("  ... {} more detections not logged",
                        num_detections - kMaxLogged);
    }
  }

 private:
  static std::shared_ptr<holoscan::Tensor> single_tensor(
      holoscan::InputContext& op_input, const char* port) {
    auto maybe = op_input.receive<holoscan::TensorMap>(port);
    if (!maybe || maybe->size() != 1) {
      throw std::runtime_error(std::string("DetectionReportOp: expected one "
                                           "tensor on port ") +
                               port);
    }
    return maybe->begin()->second;
  }
};

// Reports the DBSCAN clusters formed over the CFAR detections. `points` is the
// FeatureAssemblyOp feature matrix ([range, velocity] for SISO; [range,
// velocity, az, el] for the angle pipelines); `labels` is the per-point cluster
// id (noise = -1, row-aligned with points) and `sizes` the per-cluster point
// count. For each cluster it logs the size, the mean centroid, and
// DetectionReductionOp's representative cell (`rep_idx`, row-aligned with
// clusters) resolved to physical units via the axis bins.
//
// `rep_idx` may be (num_clusters, 2) = (doppler_bin, range_bin) — the sparse
// SISO/SIMO path — or (num_clusters, 4) = (az_bin, el_bin, doppler_bin,
// range_bin) — the dense beamforming path. For 4-column points with a 2-column
// rep_idx, pass `detection_idx` so rep az/el can be resolved from the matching
// detection row; for a 4-column rep_idx, set `az_grid`/`el_grid` and rep az/el
// are read straight from the grids (no detection_idx needed).
class ClusterReportOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ClusterReportOp)
  ClusterReportOp() = default;

  // Angle grids (radians) for the dense 4-column rep_idx path; leave empty for
  // the sparse (2-column) path.
  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("points");
    spec.input<holoscan::TensorMap>("labels");
    spec.input<holoscan::TensorMap>("sizes");
    spec.input<holoscan::TensorMap>("rep_idx");
    spec.input<holoscan::TensorMap>("range_bins");
    spec.input<holoscan::TensorMap>("velocity_bins");
    spec.input<holoscan::TensorMap>("detection_idx")
        .condition(holoscan::ConditionType::kNone);
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const std::string who = "ClusterReportOp";
    auto points = radar::receive_single(op_input, "points", who);
    auto labels = radar::receive_single(op_input, "labels", who);
    auto sizes = radar::receive_single(op_input, "sizes", who);
    auto rep_idx = radar::receive_single(op_input, "rep_idx", who);
    auto range_bins = radar::receive_single(op_input, "range_bins", who);
    auto velocity_bins = radar::receive_single(op_input, "velocity_bins", who);
    auto detection_idx =
        radar::receive_optional(op_input, "detection_idx", who);

    const auto pshape = points->shape();
    const int64_t num_features = pshape.size() == 2 ? pshape[1] : 0;
    if (pshape.size() != 2 || (num_features != 2 && num_features != 4)) {
      throw std::runtime_error(
          "ClusterReportOp: points must be (N, 2) or (N, 4), got rank " +
          std::to_string(pshape.size()));
    }
    const int64_t num_points = pshape[0];
    const auto lshape = labels->shape();
    if (lshape.size() != 1 || lshape[0] != num_points) {
      throw std::runtime_error(
          "ClusterReportOp: labels must be (N,) row-aligned with points");
    }
    const auto sshape = sizes->shape();
    if (sshape.size() != 1) {
      throw std::runtime_error("ClusterReportOp: sizes must be rank-1");
    }
    const int64_t num_clusters = sshape[0];

    // rep_idx is (num_clusters, rep_cols): rep_cols == 2 → (doppler_bin,
    // range_bin); rep_cols == 4 → (az_bin, el_bin, doppler_bin, range_bin).
    // Row-aligned with sizes (dense cluster ids).
    const auto rshape = rep_idx->shape();
    if (rshape.size() != 2 || (rshape[1] != 2 && rshape[1] != 4) ||
        rshape[0] != num_clusters) {
      throw std::runtime_error(
          "ClusterReportOp: rep_idx must be (num_clusters, 2) or "
          "(num_clusters, 4)");
    }
    const int64_t rep_cols = rshape[1];
    const int64_t num_az = static_cast<int64_t>(az_grid.size());
    const int64_t num_el = static_cast<int64_t>(el_grid.size());
    // az/el reporting needs a source: a 4-column rep_idx (read from the grids)
    // or detection_idx (match the representative sparse cell to a point row).
    const bool use_det_idx = (num_features == 4 && rep_cols == 2);
    if (num_features == 4) {
      if (rep_cols == 4) {
        if (num_az == 0 || num_el == 0) {
          throw std::runtime_error(
              "ClusterReportOp: az_grid/el_grid required for a 4-column "
              "rep_idx");
        }
      } else if (!detection_idx) {
        throw std::runtime_error(
            "ClusterReportOp: detection_idx required when points has az/el "
            "columns and rep_idx is 2-column");
      }
    }
    const int64_t num_range = range_bins->shape()[0];
    const int64_t num_doppler = velocity_bins->shape()[0];

    std::vector<float> pts(static_cast<size_t>(num_points * num_features));
    std::vector<int64_t> lab(static_cast<size_t>(num_points));
    std::vector<int64_t> sz(static_cast<size_t>(num_clusters));
    std::vector<int64_t> rep(static_cast<size_t>(num_clusters * rep_cols));
    std::vector<int64_t> det_idx;
    if (use_det_idx) {
      const auto dshape = detection_idx->shape();
      if (dshape.size() != 2 || dshape[1] != 2 || dshape[0] != num_points) {
        throw std::runtime_error(
            "ClusterReportOp: detection_idx must be (N, 2) row-aligned with "
            "points");
      }
      det_idx.resize(static_cast<size_t>(num_points * 2));
    }
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));

    const auto require_bytes =
        [](const char* name, const std::shared_ptr<holoscan::Tensor>& t,
           size_t expected) {
          if (static_cast<size_t>(t->nbytes()) != expected) {
            throw std::runtime_error(
                std::string("ClusterReportOp: ") + name +
                " byte size mismatch (expected " + std::to_string(expected) +
                ", got " + std::to_string(t->nbytes()) + ")");
          }
        };
    require_bytes("points", points, sizeof(float) * pts.size());
    require_bytes("labels", labels, sizeof(int64_t) * lab.size());
    require_bytes("sizes", sizes, sizeof(int64_t) * sz.size());
    require_bytes("rep_idx", rep_idx, sizeof(int64_t) * rep.size());
    require_bytes("range_bins", range_bins, sizeof(float) * ranges.size());
    require_bytes("velocity_bins", velocity_bins,
                  sizeof(float) * velocities.size());

    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("ClusterReportOp: device synchronize failed");
    }
    // Guard the zero-length copies: an empty tick (no detections) yields empty
    // tensors whose data() may be null.
    const auto copy_d2h = [](const char* name, void* dst,
                             const std::shared_ptr<holoscan::Tensor>& src) {
      if (src->nbytes() == 0) {
        return;
      }
      if (cudaMemcpy(dst, src->data(), static_cast<size_t>(src->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error(std::string("ClusterReportOp: ") + name +
                                 " D2H copy failed");
      }
    };
    copy_d2h("points", pts.data(), points);
    copy_d2h("labels", lab.data(), labels);
    copy_d2h("sizes", sz.data(), sizes);
    copy_d2h("rep_idx", rep.data(), rep_idx);
    copy_d2h("range_bins", ranges.data(), range_bins);
    copy_d2h("velocity_bins", velocities.data(), velocity_bins);
    if (use_det_idx) {
      copy_d2h("detection_idx", det_idx.data(), detection_idx);
    }

    // Accumulate per-cluster centroids from the points, keyed by label; noise
    // (-1) is counted but excluded from any cluster.
    std::vector<double> sum_range(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_velocity(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_az;
    std::vector<double> sum_el;
    if (num_features == 4) {
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
        throw std::runtime_error(
            "ClusterReportOp: label out of range for sizes");
      }
      sum_range[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * num_features + 0)];
      sum_velocity[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * num_features + 1)];
      if (num_features == 4) {
        sum_az[static_cast<size_t>(c)] +=
            pts[static_cast<size_t>(i * num_features + 2)];
        sum_el[static_cast<size_t>(c)] +=
            pts[static_cast<size_t>(i * num_features + 3)];
      }
    }

    // Resolve rep az/el from detection_idx (2-column rep path): find the point
    // row whose CFAR cell matches the representative (doppler, range).
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
          throw std::runtime_error(
              "ClusterReportOp: rep_idx out of grid/axis bounds");
        }
        rep_az = az_grid[static_cast<size_t>(az_bin)];
        rep_el = el_grid[static_cast<size_t>(el_bin)];
      } else {
        rep_dd = rep[static_cast<size_t>(c * 2 + 0)];
        rep_rr = rep[static_cast<size_t>(c * 2 + 1)];
        if (rep_dd < 0 || rep_dd >= num_doppler || rep_rr < 0 ||
            rep_rr >= num_range) {
          throw std::runtime_error(
              "ClusterReportOp: rep_idx out of axis bounds");
        }
        if (use_det_idx) {
          rep_az_el_from_detidx(rep_dd, rep_rr, &rep_az, &rep_el);
        }
      }
      if (num_features == 4) {
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
  }
};

// Drain a CFAR dense output without deserializing it as a TensorMap. The
// bool detection_mask path trips Holoscan 4.3's unsupported-DLPack-bool
// error when received as a TensorMap; raw Entity receive sidesteps that.
class NullSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NullSinkOp)
  NullSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    (void)op_input.receive<holoscan::gxf::Entity>("input");
  }
};

// Drain an unused TensorMap output so its transmitter has a downstream and the
// producer can tick. Distinct from NullSinkOp, which drains a raw gxf::Entity
// (the bool detection_mask path); this one receives a normal TensorMap (e.g. a
// beamscan angle_spectrum the app does not consume).
class TensorMapDiscardOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorMapDiscardOp)
  TensorMapDiscardOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("input");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    (void)radar::receive_single(op_input, "input", "TensorMapDiscardOp");
  }
};

}  // namespace holoscan::radar::app
