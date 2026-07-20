// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-specific operators for fmcw_dense_simo, colocated with the app:
//   AnglePeakPickOp   — collapse per-beam cube detections to one peak per RD cell
//   CubeAoaLookupOp   — bridge (az_bin, el_bin) cube columns to aoa_estimates
//   TrailingIdxColumnsOp — slice dense 4-column idx down to the RD-plane 2 columns
// The dense pipeline diverges from the sparse contract in this middle stage, so
// these stay local (not generalized into radar_app).

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace radar = holoscan::radar;

using Precision = float;

// Angle peak-pick: collapse the dense cube's many per-beam detections down to
// one detection per range-Doppler cell — the (az, el) beam with the largest
// CFAR score, i.e. the beamformed-power peak, which is the target's true DOA.
//
// Why: CFAR runs independently on each beam's RD plane (nci_axes=[]), so it has
// no cross-beam view; a strong target trips the threshold in *many* beams (its
// broad mainlobe plus sidelobes), scattering raw detections across the whole
// angular FOV. Peak-picking is the dense analog of the sparse path's "one angle
// per detection" (there, SnapshotGather + a single beamscan peak). It makes the
// angle estimate a point per RD cell instead of a beam-pattern smear, so the
// boresight scatter and RD overlay are tight and clustering is unambiguous.
//
// In:  detection_idx  — int64 (N, 4): az_bin, el_bin, doppler_bin, range_bin.
// In:  detection_score — float (num_az, num_el, num_doppler, num_range): CFAR
//                        test statistic (∝ beamformed power) over the cube.
// Out: detection_idx  — int64 (M, 4): one row per detected (doppler, range)
//                        cell, with (az, el) set to the score-argmax beam.
class AnglePeakPickOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AnglePeakPickOp)
  AnglePeakPickOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("detection_idx");
    spec.input<holoscan::TensorMap>("detection_score");
    spec.output<holoscan::TensorMap>("detection_idx");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "AnglePeakPickOp";
    auto idx = radar::receive_single(op_input, "detection_idx", who);
    auto score = radar::receive_single(op_input, "detection_score", who);
    const auto ishape = idx->shape();
    if (ishape.size() != 2 || ishape[1] != 4) {
      throw std::runtime_error(
          "AnglePeakPickOp: detection_idx must be (num_points, 4)");
    }
    const auto sshape = score->shape();
    if (sshape.size() != 4) {
      throw std::runtime_error(
          "AnglePeakPickOp: detection_score must be rank-4 "
          "(num_az, num_el, num_doppler, num_range)");
    }
    const int64_t n = ishape[0];
    const int64_t na = sshape[0];
    const int64_t ne = sshape[1];
    const int64_t nd = sshape[2];
    const int64_t nr = sshape[3];

    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("AnglePeakPickOp: device synchronize failed");
    }
    std::vector<int64_t> host_idx(static_cast<size_t>(n * 4));
    if (n > 0 &&
        cudaMemcpy(host_idx.data(), idx->data(),
                   static_cast<size_t>(idx->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("AnglePeakPickOp: detection_idx D2H failed");
    }
    std::vector<Precision> host_score(static_cast<size_t>(na * ne * nd * nr));
    if (cudaMemcpy(host_score.data(), score->data(),
                   static_cast<size_t>(score->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("AnglePeakPickOp: detection_score D2H failed");
    }

    // Unique detected (doppler, range) cells, in first-seen order.
    std::vector<int64_t> seen_flag(static_cast<size_t>(nd * nr), 0);
    std::vector<std::pair<int64_t, int64_t>> cells;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t d = host_idx[static_cast<size_t>(i * 4 + 2)];
      const int64_t r = host_idx[static_cast<size_t>(i * 4 + 3)];
      if (d < 0 || d >= nd || r < 0 || r >= nr) {
        throw std::runtime_error(
            "AnglePeakPickOp: detection (doppler, range) out of score bounds");
      }
      const size_t key = static_cast<size_t>(d * nr + r);
      if (seen_flag[key] == 0) {
        seen_flag[key] = 1;
        cells.emplace_back(d, r);
      }
    }

    const int64_t m = static_cast<int64_t>(cells.size());
    auto out = radar::make_device_tensor({m, 4}, DLDataType{kDLInt, 64, 1});
    if (m > 0) {
      std::vector<int64_t> host_out(static_cast<size_t>(m * 4));
      for (int64_t c = 0; c < m; ++c) {
        const int64_t d = cells[static_cast<size_t>(c)].first;
        const int64_t r = cells[static_cast<size_t>(c)].second;
        int64_t best_az = 0, best_el = 0;
        Precision best = -std::numeric_limits<Precision>::infinity();
        for (int64_t a = 0; a < na; ++a) {
          for (int64_t e = 0; e < ne; ++e) {
            const Precision v = host_score[static_cast<size_t>(
                ((a * ne + e) * nd + d) * nr + r)];
            if (v > best) {
              best = v;
              best_az = a;
              best_el = e;
            }
          }
        }
        host_out[static_cast<size_t>(c * 4 + 0)] = best_az;
        host_out[static_cast<size_t>(c * 4 + 1)] = best_el;
        host_out[static_cast<size_t>(c * 4 + 2)] = d;
        host_out[static_cast<size_t>(c * 4 + 3)] = r;
      }
      if (cudaMemcpy(out->data(), host_out.data(),
                     sizeof(int64_t) * host_out.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("AnglePeakPickOp: detection_idx H2D failed");
      }
    }
    radar::emit_single(op_output, "detection_idx", out);
  }
};

// Dense-path bridge (approach A): maps per-detection (az_bin, el_bin) cube
// batch columns to physical angles via the scan grids and emits aoa_estimates
// (num_points, 2), feeding FeatureAssemblyOp's existing aoa_estimates port.
// This exists because FeatureAssemblyOp sources az/el only from aoa_estimates,
// not from the RDA-cube's preserved batch bins — see the operator OPEN_ITEMS.
// Kept
// an app-local add-on (plain-tensor-ports invariant), not a shared type or an
// operator contract change.
//
// Also emits row_index (num_points,), the identity [0..num_points). It is a
// no-op relabel for the detection path (drained), but when this op is fed the
// DetectionReductionOp rep_idx (one row per cluster, in cluster order) the row
// index *is* the cluster id — so the boresight panel can reuse the row_index as
// its per-marker cluster label to show one colored centroid marker per cluster.
class CubeAoaLookupOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CubeAoaLookupOp)
  CubeAoaLookupOp() = default;

  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("detection_idx");
    spec.output<holoscan::TensorMap>("aoa_estimates");
    spec.output<holoscan::TensorMap>("row_index");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "CubeAoaLookupOp";
    auto idx = radar::receive_single(op_input, "detection_idx", who);
    const auto shape = idx->shape();
    // Cube detections: (num_points, 4) = (az_bin, el_bin, doppler_bin,
    // range_bin). Require the two leading angle-batch columns.
    if (shape.size() != 2 || shape[1] < 4) {
      throw std::runtime_error(
          "CubeAoaLookupOp: detection_idx must be (num_points, >= 4) with "
          "leading (az_bin, el_bin) columns from the RDA cube");
    }
    const int64_t n = shape[0];
    const int64_t cols = shape[1];
    auto aoa =
        radar::make_device_tensor({n, 2}, radar::real_dtype<Precision>());
    auto row_index = radar::make_device_tensor({n}, DLDataType{kDLInt, 64, 1});
    if (n > 0) {
      std::vector<int64_t> host_idx(static_cast<size_t>(n * cols));
      if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("CubeAoaLookupOp: device synchronize failed");
      }
      if (cudaMemcpy(host_idx.data(), idx->data(),
                     static_cast<size_t>(idx->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("CubeAoaLookupOp: detection_idx D2H failed");
      }
      const int64_t num_az = static_cast<int64_t>(az_grid.size());
      const int64_t num_el = static_cast<int64_t>(el_grid.size());
      std::vector<Precision> host_aoa(static_cast<size_t>(n * 2));
      std::vector<int64_t> host_rows(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i) {
        const int64_t az_bin = host_idx[static_cast<size_t>(i * cols + 0)];
        const int64_t el_bin = host_idx[static_cast<size_t>(i * cols + 1)];
        if (az_bin < 0 || az_bin >= num_az || el_bin < 0 || el_bin >= num_el) {
          throw std::runtime_error(
              "CubeAoaLookupOp: detection az/el bin out of grid range");
        }
        host_aoa[static_cast<size_t>(i * 2 + 0)] =
            static_cast<Precision>(az_grid[static_cast<size_t>(az_bin)]);
        host_aoa[static_cast<size_t>(i * 2 + 1)] =
            static_cast<Precision>(el_grid[static_cast<size_t>(el_bin)]);
        host_rows[static_cast<size_t>(i)] = i;
      }
      if (cudaMemcpy(aoa->data(), host_aoa.data(),
                     sizeof(Precision) * host_aoa.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("CubeAoaLookupOp: aoa_estimates H2D failed");
      }
      if (cudaMemcpy(row_index->data(), host_rows.data(),
                     sizeof(int64_t) * host_rows.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("CubeAoaLookupOp: row_index H2D failed");
      }
    }
    radar::emit_single(op_output, "aoa_estimates", aoa);
    radar::emit_single(op_output, "row_index", row_index);
  }
};


// Slices the trailing two columns of an int64 index tensor (num_points, C>=2)
// down to (num_points, 2). The dense CFAR / reduction indices carry the cube's
// (az_bin, el_bin) batch columns ahead of (doppler_bin, range_bin); the shared
// radar_app RD-plane viz ops (ClusterOverlayOp, RollingPlotOp) expect exactly
// the sparse 2-column (doppler_bin, range_bin) form. Row order is preserved, so
// the sliced tensor stays row-aligned with cluster labels. Input/output port
// names are generic ("idx") so one op serves both detection_idx and rep_idx.
class TrailingIdxColumnsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TrailingIdxColumnsOp)
  TrailingIdxColumnsOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("idx");
    spec.output<holoscan::TensorMap>("idx");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "TrailingIdxColumnsOp";
    auto idx = radar::receive_single(op_input, "idx", who);
    const auto shape = idx->shape();
    if (shape.size() != 2 || shape[1] < 2) {
      throw std::runtime_error(
          "TrailingIdxColumnsOp: expected (num_points, >= 2) int64 idx");
    }
    const int64_t n = shape[0];
    const int64_t cols = shape[1];
    auto out = radar::make_device_tensor({n, 2}, DLDataType{kDLInt, 64, 1});
    if (n > 0) {
      std::vector<int64_t> host(static_cast<size_t>(n * cols));
      if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("TrailingIdxColumnsOp: device sync failed");
      }
      if (cudaMemcpy(host.data(), idx->data(),
                     static_cast<size_t>(idx->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("TrailingIdxColumnsOp: idx D2H failed");
      }
      std::vector<int64_t> trimmed(static_cast<size_t>(n * 2));
      for (int64_t i = 0; i < n; ++i) {
        trimmed[static_cast<size_t>(i * 2 + 0)] =
            host[static_cast<size_t>(i * cols + (cols - 2))];
        trimmed[static_cast<size_t>(i * 2 + 1)] =
            host[static_cast<size_t>(i * cols + (cols - 1))];
      }
      if (cudaMemcpy(out->data(), trimmed.data(),
                     sizeof(int64_t) * trimmed.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("TrailingIdxColumnsOp: idx H2D failed");
      }
    }
    radar::emit_single(op_output, "idx", out);
  }
};
