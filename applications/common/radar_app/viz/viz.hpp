// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

// Reusable range-Doppler visualization operators + colormap/palette helpers for
// the FMCW example applications. The three operators rasterize what the
// detection/clustering chain produces (an RD image, the per-detection cluster
// crosses, and the rolling per-cluster centroid plots) into Holoscan tensors
// HolovizOp renders; the app owns the HolovizOp layout (views, InputSpecs) in
// its compose(). Shared, app-agnostic — siso/simo/mimo reuse them unchanged.
// Header-only: operator methods are in-class (implicitly inline) and the free
// helpers are marked inline, so multiple apps can include this without ODR
// conflict.
namespace holoscan::radar::app {

// Entry count of the Holoviz color lookup table; the RD-map image values
// index into it directly.
constexpr size_t kLutSize = 256;

// Extract one RX channel from a per-channel RD map for visualization. SIMO
// rd_maps are (N_rx, num_doppler, num_range); Holoviz helpers expect rank-2
// (num_doppler, num_range). Default channel 0 is a debug display only — CFAR
// may run NCI over all RX.
class RdChannelSliceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RdChannelSliceOp)
  RdChannelSliceOp() = default;

  int64_t channel{0};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("rd_maps");
    spec.output<holoscan::TensorMap>("rd_map");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    auto rd_maps =
        radar::receive_single(op_input, "rd_maps", "RdChannelSliceOp");
    const auto& shape = rd_maps->shape();
    if (shape.size() != 3) {
      throw std::runtime_error(
          "RdChannelSliceOp: expected rank-3 (N_rx, D, R), got rank " +
          std::to_string(shape.size()));
    }
    const int64_t num_rx = shape[0];
    const int64_t num_doppler = shape[1];
    const int64_t num_range = shape[2];
    if (channel < 0 || channel >= num_rx) {
      throw std::runtime_error("RdChannelSliceOp: channel out of range");
    }

    const size_t plane = static_cast<size_t>(num_doppler * num_range);
    std::vector<std::complex<float>> host(plane);
    const auto* src = static_cast<const std::complex<float>*>(rd_maps->data());
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("RdChannelSliceOp: device synchronize failed");
    }
    if (cudaMemcpy(host.data(), src + channel * plane,
                   plane * sizeof(std::complex<float>),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("RdChannelSliceOp: D2H slice failed");
    }

    auto out = radar::make_device_tensor({num_doppler, num_range},
                                        radar::complex_dtype<float>());
    if (cudaMemcpy(out->data(), host.data(),
                   host.size() * sizeof(std::complex<float>),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("RdChannelSliceOp: H2D upload failed");
    }
    radar::emit_single(op_output, "rd_map", out);
  }
};

// Converts the complex RD map into a log-magnitude float32 image for
// Holoviz (rendered through the turbo color lookup table). Layout is
// (doppler, range): Holoviz x = range, y = Doppler.
//
// Per frame:
//   1. Divide each range column by its median |RD| — suppresses Doppler
//      pedestals from MIMO-DDM code crosstalk that otherwise paint full
//      vertical stripes under a global log stretch (detection math is
//      unchanged; this is display-only).
//   2. Convert to dB and map between that frame's median and peak onto
//      [0, kLutSize - 1] for HolovizOp's non-normalized LUT.
// Detections/clusters are drawn separately (ClusterOverlayOp).
class RdMapToImageOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RdMapToImageOp)
  RdMapToImageOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("rd_maps");
    spec.output<holoscan::TensorMap>("rd_image");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    auto maybe = op_input.receive<holoscan::TensorMap>("rd_maps");
    if (!maybe || maybe->size() != 1) {
      throw std::runtime_error(
          "RdMapToImageOp: expected one tensor on port rd_maps");
    }
    std::shared_ptr<holoscan::Tensor> rd_map = maybe->begin()->second;
    const auto shape = rd_map->shape();
    if (shape.size() != 2) {
      throw std::runtime_error(
          "RdMapToImageOp: expected rank-2 rd_map "
          "(doppler, range), got rank " +
          std::to_string(shape.size()));
    }
    const int64_t num_doppler = shape[0];
    const int64_t num_range = shape[1];

    std::vector<std::complex<float>> rd(
        static_cast<size_t>(num_doppler * num_range));
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("RdMapToImageOp: device synchronize failed");
    }
    if (cudaMemcpy(rd.data(), rd_map->data(),
                   static_cast<size_t>(rd_map->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("RdMapToImageOp: rd_map D2H copy failed");
    }

    std::vector<float> image(rd.size());
    std::vector<float> col_abs(static_cast<size_t>(num_doppler));
    std::vector<float> col_sort(static_cast<size_t>(num_doppler));
    float peak_db = -std::numeric_limits<float>::infinity();
    for (int64_t r = 0; r < num_range; ++r) {
      for (int64_t d = 0; d < num_doppler; ++d) {
        col_abs[static_cast<size_t>(d)] =
            std::abs(rd[static_cast<size_t>(d * num_range + r)]);
      }
      col_sort = col_abs;
      std::nth_element(col_sort.begin(),
                       col_sort.begin() + col_sort.size() / 2, col_sort.end());
      const float med =
          std::max(col_sort[col_sort.size() / 2], 1e-20F);
      for (int64_t d = 0; d < num_doppler; ++d) {
        const float v =
            20.0F *
            std::log10(col_abs[static_cast<size_t>(d)] / med + 1e-20F);
        image[static_cast<size_t>(d * num_range + r)] = v;
        peak_db = std::max(peak_db, v);
      }
    }
    std::vector<float> sorted(image);
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2,
                     sorted.end());
    const float floor_db = sorted[sorted.size() / 2];
    const float span_db = std::max(peak_db - floor_db, 1.0F);
    for (float& v : image) {
      v = std::clamp((v - floor_db) / span_db, 0.0F, 1.0F) *
          static_cast<float>(kLutSize - 1);
    }

    // Holoviz image layers require a rank-3 (height, width, channels)
    // tensor, so add an explicit single-channel dimension.
    auto tensor = radar::make_device_tensor({num_doppler, num_range, 1},
                                            radar::real_dtype<float>());
    if (cudaMemcpy(tensor->data(), image.data(),
                   sizeof(float) * image.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("RdMapToImageOp: H2D upload failed");
    }
    holoscan::TensorMap out;
    out.insert({"rd_map", tensor});
    op_output.emit(out, "rd_image");
  }
};

// Overlay geometry-layer names, shared by ClusterOverlayOp (as emitted tensor
// keys) and compose() (as HolovizOp InputSpec names). Detections are drawn as
// filled discs split across kNumClusterColors colored layers, one per palette
// color; a detection is assigned to layer (cluster_id % kNumClusterColors), so
// distinct clusters in a frame get distinct colors (clusters beyond the palette
// wrap). Noise points (label -1) go to their own muted layer. A single outline
// layer holds a larger black disc under every marker, giving each dot a thin
// black rim.
constexpr int kNumClusterColors = 30;
constexpr const char* kNoiseLayerName = "noise";
constexpr const char* kDetectionOutlineName = "detection_outline";
inline std::string cluster_layer_name(int color) {
  return "cluster_" + std::to_string(color);
}

// Per-cluster color bucket assigned by RANK, not a hash: sort the frame's
// clusters by representative range bin (Doppler bin as tiebreak, then cluster
// index for determinism) and hand out palette colors in that order. rep is the
// (num_clusters, 2) representative cells, columns (doppler_bin, range_bin); the
// returned vector is row-aligned with rep (bucket[label] = that cluster's
// color). Two properties this buys over hashing the cell:
//   - No collisions among coexisting clusters: the first num_clusters palette
//     entries are used, and cluster_palette() front-loads maximally-distinct
//     colors (golden-angle hues), so a handful of clusters read very differently.
//   - Temporal stability: a cluster keeps its color while its range rank holds;
//     the rank only shifts when clusters actually reorder in range or the set
//     changes — not from the size-rank reshuffles that made the DBSCAN label
//     flicker. (Not a tracker: a new nearer target or a range crossing does
//     shift the colors below it.)
// All viz ops call this on the same rep_idx, so a cluster shows the same color
// on the RD map, the rolling plots, and the boresight panel.
inline std::vector<int> rep_color_ranks(const std::vector<int64_t>& rep,
                                        int64_t num_clusters) {
  const size_t n = static_cast<size_t>(std::max<int64_t>(num_clusters, 0));
  std::vector<int> order(n);
  for (size_t i = 0; i < n; ++i) {
    order[i] = static_cast<int>(i);
  }
  std::sort(order.begin(), order.end(), [&rep](int a, int b) {
    const int64_t ra = rep[static_cast<size_t>(a) * 2 + 1];  // range
    const int64_t rb = rep[static_cast<size_t>(b) * 2 + 1];
    if (ra != rb) {
      return ra < rb;
    }
    const int64_t da = rep[static_cast<size_t>(a) * 2 + 0];  // doppler tiebreak
    const int64_t db = rep[static_cast<size_t>(b) * 2 + 0];
    if (da != db) {
      return da < db;
    }
    return a < b;  // index tiebreak → identical ordering in every op
  });
  std::vector<int> bucket(n);
  for (size_t rank = 0; rank < n; ++rank) {
    bucket[static_cast<size_t>(order[rank])] =
        static_cast<int>(rank % kNumClusterColors);
  }
  return bucket;
}

// Turns the CFAR detections and their DBSCAN cluster labels into Holoviz
// TRIANGLES geometry layers drawn over the range-Doppler image. Consumes the
// detector's detection_idx (the (doppler_bin, range_bin) cells), the matching
// per-point cluster labels, and the RD map (for its dimensions), and emits one
// filled disc (a triangle fan) per detection at the matching image location,
// bucketed by cluster color, plus a larger black disc per detection in a shared
// outline layer. Holoviz geometry is in normalized [0, 1] coordinates: x runs
// left→right across the range axis, y top→bottom across the Doppler axis — the
// same layout RdMapToImageOp gives the image, so the crosses land on their
// cells. labels are row-aligned with detection_idx (FeatureAssemblyOp keeps
// points 1:1 with the cells), so no bin↔physical inversion is needed here.
class ClusterOverlayOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ClusterOverlayOp)
  ClusterOverlayOp() = default;

  // Marker geometry, as a fraction of the image extent: filled-disc radius, the
  // extra radius of the black outline disc drawn beneath it, and the triangle-
  // fan segment count. Set by compose() from the visualization config; the
  // defaults are kept small so nearby detections stay distinct.
  float marker_radius{0.007F};
  float marker_outline{0.0022F};
  int64_t marker_segments{20};

  // Aspect ratio (width / height, in pixels) of the RD-map sub-view. The disc x
  // radius is divided by it so the markers render round despite the view
  // stretching normalized coordinates to a non-square rectangle. Set by
  // compose() (see rd_map_view_aspect); 1.0 leaves the discs uncorrected.
  float view_aspect{1.0F};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("detection_idx");
    // Optional: DBSCAN cluster ids (row-aligned with detection_idx). Present →
    // clustered mode: each detection colors by its cluster. Absent → peak_only:
    // each detection colors by its own row index.
    spec.input<holoscan::TensorMap>("labels")
        .condition(holoscan::ConditionType::kNone);
    // Optional: per-cluster representative RD cell (num_clusters, 2), row c =
    // cluster label c. When present with labels, clusters color by their rep
    // cell (stable across frames) instead of the size-rank label.
    spec.input<holoscan::TensorMap>("rep_idx")
        .condition(holoscan::ConditionType::kNone);
    spec.input<holoscan::TensorMap>("rd_maps");
    spec.output<holoscan::TensorMap>("overlay");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const std::string who = "ClusterOverlayOp";
    auto idx = radar::receive_single(op_input, "detection_idx", who);
    auto labels = radar::receive_optional(op_input, "labels", who);
    auto rep_idx = radar::receive_optional(op_input, "rep_idx", who);
    auto rd_map = radar::receive_single(op_input, "rd_maps", who);
    const bool have_labels = static_cast<bool>(labels);

    // The RD map (only its shape is needed) sizes the normalization; the
    // detection cells index the same Doppler × range grid.
    const auto map_shape = rd_map->shape();
    if (map_shape.size() != 2) {
      throw std::runtime_error(
          "ClusterOverlayOp: expected a rank-2 rd_map, got rank " +
          std::to_string(map_shape.size()));
    }
    const int64_t num_doppler = map_shape[0];
    const int64_t num_range = map_shape[1];

    // detection_idx is int64 (num_points, 2): columns (doppler_bin,
    // range_bin). An empty frame yields a 0-row matrix.
    const auto idx_shape = idx->shape();
    if (idx_shape.size() != 2 || idx_shape[1] != 2) {
      throw std::runtime_error(
          "ClusterOverlayOp: expected a (num_points, 2) detection_idx");
    }
    const int64_t num_points = idx_shape[0];
    // labels (optional) is int64 (num_points,), row-aligned with detection_idx.
    if (have_labels) {
      const auto lab_shape = labels->shape();
      if (lab_shape.size() != 1 || lab_shape[0] != num_points) {
        throw std::runtime_error(
            "ClusterOverlayOp: labels must be (num_points,) row-aligned with "
            "detection_idx");
      }
    }

    std::vector<int64_t> bins(static_cast<size_t>(num_points * 2));
    std::vector<int64_t> lab(static_cast<size_t>(num_points));
    const size_t idx_bytes = sizeof(int64_t) * bins.size();
    const size_t lab_bytes = sizeof(int64_t) * lab.size();
    if (static_cast<size_t>(idx->nbytes()) != idx_bytes) {
      throw std::runtime_error(
          "ClusterOverlayOp: detection_idx byte size mismatch");
    }
    if (have_labels &&
        static_cast<size_t>(labels->nbytes()) != lab_bytes) {
      throw std::runtime_error("ClusterOverlayOp: labels byte size mismatch");
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("ClusterOverlayOp: device synchronize failed");
    }
    if (num_points > 0) {
      if (cudaMemcpy(bins.data(), idx->data(), idx_bytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error(
            "ClusterOverlayOp: detection_idx D2H copy failed");
      }
      if (have_labels &&
          cudaMemcpy(lab.data(), labels->data(), lab_bytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("ClusterOverlayOp: labels D2H copy failed");
      }
    }

    // Optional per-cluster representative cells (num_clusters, 2): row = label.
    // When present, clusters color by their range rank (rep_color_ranks) —
    // stable and collision-free — instead of the flickery size-rank label.
    std::vector<int64_t> rep;
    int64_t num_rep = 0;
    if (rep_idx) {
      const auto rep_shape = rep_idx->shape();
      if (rep_shape.size() != 2 || rep_shape[1] != 2) {
        throw std::runtime_error(
            "ClusterOverlayOp: rep_idx must be (num_clusters, 2)");
      }
      num_rep = rep_shape[0];
      rep.resize(static_cast<size_t>(num_rep * 2));
      if (static_cast<size_t>(rep_idx->nbytes()) !=
          sizeof(int64_t) * rep.size()) {
        throw std::runtime_error("ClusterOverlayOp: rep_idx byte size mismatch");
      }
      if (num_rep > 0 &&
          cudaMemcpy(rep.data(), rep_idx->data(), sizeof(int64_t) * rep.size(),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("ClusterOverlayOp: rep_idx D2H copy failed");
      }
    }
    const bool have_rep = rep_idx && num_rep > 0;
    // Rank clusters by their rep cell → one color per cluster (see
    // rep_color_ranks); bucket[label] is that cluster's palette index.
    const std::vector<int> ranks =
        have_rep ? rep_color_ranks(rep, num_rep) : std::vector<int>();

    // Bucket detection centers into the palette layers by cluster id; the last
    // bucket is noise. Each center is (x, y) in normalized coordinates, cell
    // centers at (bin + 0.5) / count. outline_centers backs the black outline
    // layer, one larger disc under every marker.
    std::vector<std::vector<std::array<float, 2>>> centers(kNumClusterColors +
                                                           1);
    std::vector<std::array<float, 2>> outline_centers;
    outline_centers.reserve(static_cast<size_t>(num_points));
    for (int64_t i = 0; i < num_points; ++i) {
      const int64_t doppler_bin = bins[static_cast<size_t>(i * 2 + 0)];
      const int64_t range_bin = bins[static_cast<size_t>(i * 2 + 1)];
      if (doppler_bin < 0 || doppler_bin >= num_doppler || range_bin < 0 ||
          range_bin >= num_range) {
        throw std::runtime_error(
            "ClusterOverlayOp: detection_idx out of map bounds");
      }
      // Clustered (labels present): color by the cluster's range rank when rep
      // cells are available (stable, collision-free), else the size-rank label;
      // noise (-1) → gray. peak_only (no labels): color each detection by its
      // row index so individual peaks stay distinct and match the side panels.
      size_t bucket;
      if (have_labels) {
        const int64_t label = lab[static_cast<size_t>(i)];
        if (label < 0) {
          bucket = static_cast<size_t>(kNumClusterColors);
        } else if (have_rep && label < num_rep) {
          bucket = static_cast<size_t>(ranks[static_cast<size_t>(label)]);
        } else {
          bucket = static_cast<size_t>(label % kNumClusterColors);
        }
      } else {
        bucket = static_cast<size_t>(i % kNumClusterColors);
      }
      const std::array<float, 2> center = {
          (static_cast<float>(range_bin) + 0.5F) /
              static_cast<float>(num_range),
          (static_cast<float>(doppler_bin) + 0.5F) /
              static_cast<float>(num_doppler)};
      centers[bucket].push_back(center);
      outline_centers.push_back(center);
    }

    // Precompute the unit-disc rim once; each marker is a triangle fan of
    // marker_segments triangles (center + two adjacent rim points). The x radius
    // is divided by the view aspect so the discs stay round on screen.
    const int segments = marker_segments > 2 ? static_cast<int>(marker_segments)
                                             : 3;
    std::vector<float> cos_t(static_cast<size_t>(segments) + 1);
    std::vector<float> sin_t(static_cast<size_t>(segments) + 1);
    for (int k = 0; k <= segments; ++k) {
      const float a = 2.0F * static_cast<float>(M_PI) *
                      static_cast<float>(k) / static_cast<float>(segments);
      cos_t[static_cast<size_t>(k)] = std::cos(a);
      sin_t[static_cast<size_t>(k)] = std::sin(a);
    }
    const float aspect = view_aspect > 1e-6F ? view_aspect : 1.0F;

    // Emit every layer each tick (0-vertex when empty) so the fixed HolovizOp
    // InputSpec list always has its matching tensor. Holoviz TRIANGLES want a
    // (3 * num_triangles, 2) buffer of consecutive vertices.
    holoscan::TensorMap out;
    const auto emit_discs =
        [&](const std::string& name,
            const std::vector<std::array<float, 2>>& pts, float radius) {
          const float rx = radius / aspect;
          const float ry = radius;
          std::vector<float> data;
          data.reserve(pts.size() * static_cast<size_t>(segments) * 6);
          for (const auto& p : pts) {
            for (int k = 0; k < segments; ++k) {
              const size_t k0 = static_cast<size_t>(k);
              const size_t k1 = static_cast<size_t>(k + 1);
              data.push_back(p[0]);
              data.push_back(p[1]);
              data.push_back(p[0] + rx * cos_t[k0]);
              data.push_back(p[1] + ry * sin_t[k0]);
              data.push_back(p[0] + rx * cos_t[k1]);
              data.push_back(p[1] + ry * sin_t[k1]);
            }
          }
          const int64_t nverts = static_cast<int64_t>(data.size() / 2);
          auto tensor = radar::make_device_tensor({nverts, 2},
                                                  radar::real_dtype<float>());
          if (nverts > 0 && cudaMemcpy(tensor->data(), data.data(),
                                       sizeof(float) * data.size(),
                                       cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("ClusterOverlayOp: H2D upload failed");
          }
          out.insert({name, tensor});
        };

    // Outline first (larger black disc under every marker), then one colored
    // layer per cluster bucket drawn on top.
    emit_discs(kDetectionOutlineName, outline_centers,
               marker_radius + marker_outline);
    for (int c = 0; c <= kNumClusterColors; ++c) {
      emit_discs(
          c < kNumClusterColors ? cluster_layer_name(c) : kNoiseLayerName,
          centers[static_cast<size_t>(c)], marker_radius);
    }
    op_output.emit(out, "overlay");
  }
};

// kLutSize-entry RGBA lookup table for the turbo colormap, built from the
// published 5th-order polynomial fit of the Google turbo palette.
inline std::vector<std::vector<float>> make_turbo_lut() {
  constexpr float kR[6] = {0.13572138F, 4.61539260F, -42.66032258F,
                           132.13108234F, -152.94239396F, 59.28637943F};
  constexpr float kG[6] = {0.09140261F, 2.19418839F, 4.84296658F,
                           -14.18503333F, 4.27729857F, 2.82956604F};
  constexpr float kB[6] = {0.10667330F, 12.64194608F, -60.58204836F,
                           110.36276771F, -89.90310912F, 27.34824973F};
  auto poly = [](const float (&c)[6], float t) {
    float v = c[5];
    for (int i = 4; i >= 0; --i) {
      v = v * t + c[i];
    }
    return std::clamp(v, 0.0F, 1.0F);
  };
  std::vector<std::vector<float>> lut;
  lut.reserve(kLutSize);
  for (size_t i = 0; i < kLutSize; ++i) {
    const float t = static_cast<float>(i) / (kLutSize - 1);
    lut.push_back({poly(kR, t), poly(kG, t), poly(kB, t), 1.0F});
  }
  return lut;
}

// Categorical RGBA palette for the cluster-overlay disc layers, one entry per
// color (index i colors layer cluster_i); size == kNumClusterColors by
// construction. Generated rather than hand-listed so it scales to any color
// count: hues step by the golden angle (maximally spread on the wheel even for
// many entries) and saturation/value alternate across two levels each to keep
// hue-adjacent colors distinct. High value keeps every dot bright against the
// turbo image (and each carries a black outline on the RD map).
inline std::vector<std::vector<float>> cluster_palette() {
  constexpr float kGolden = 0.61803398875F;  // golden-ratio conjugate
  std::vector<std::vector<float>> lut;
  lut.reserve(static_cast<size_t>(kNumClusterColors));
  for (int i = 0; i < kNumClusterColors; ++i) {
    const float h6 = std::fmod(static_cast<float>(i) * kGolden, 1.0F) * 6.0F;
    const float s = 0.70F + 0.30F * static_cast<float>(i % 2);        // .70/1.0
    const float v = 1.0F - 0.25F * static_cast<float>((i / 2) % 2);   // 1.0/.75
    const int hi = static_cast<int>(h6) % 6;
    const float f = h6 - std::floor(h6);
    const float p = v * (1.0F - s);
    const float q = v * (1.0F - s * f);
    const float t = v * (1.0F - s * (1.0F - f));
    float r = v, g = t, b = p;  // hi == 0
    switch (hi) {
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
      default: break;
    }
    lut.push_back({r, g, b, 1.0F});
  }
  return lut;
}

// ---- Shared 2D raster plotting helpers (RGBA8 byte canvas) --------------
// A tiny software rasterizer shared by the side-panel plot operators
// (RollingPlotOp, BoresightViewOp): per-cluster marker colors, pixel
// plotting, a 3x5 bitmap font for numeric tick labels, and the constant
// title-position tensor. Canvas is row-major RGBA8 (W*H*4 bytes), origin
// top-left.
constexpr int kPlotFontScale = 2;

// RGBA byte color for a cluster bucket (palette color, or gray for noise). The
// palette is built once (static local) since this is called per marker.
inline std::array<uint8_t, 3> plot_bucket_color(int bucket) {
  static const std::vector<std::vector<float>> pal = cluster_palette();
  if (bucket >= kNumClusterColors) {
    return {150, 150, 150};  // noise
  }
  const auto& c = pal[static_cast<size_t>(bucket)];
  return {static_cast<uint8_t>(c[0] * 255.0F),
          static_cast<uint8_t>(c[1] * 255.0F),
          static_cast<uint8_t>(c[2] * 255.0F)};
}

inline void plot_put_px(std::vector<uint8_t>* img, int64_t w, int64_t h,
                        int64_t x, int64_t y,
                        const std::array<uint8_t, 3>& c) {
  if (x < 0 || x >= w || y < 0 || y >= h) {
    return;
  }
  const size_t o = static_cast<size_t>((y * w + x) * 4);
  (*img)[o + 0] = c[0];
  (*img)[o + 1] = c[1];
  (*img)[o + 2] = c[2];
  (*img)[o + 3] = 255;
}

// Filled round marker mirroring the RD-map detection dots in the software-
// rasterized side panels. With `outline` it also draws a black disc of radius
// (radius + kPlotMarkerOutline) under the colored fill, giving a uniform black
// rim (used where markers don't overlap, e.g. the boresight scatter). The
// rolling plots draw a marker per frame, so their dots overlap into a trail —
// they pass outline=false to keep the band clean rather than speckled.
constexpr int kPlotMarkerRadius = 3;
constexpr int kPlotMarkerOutline = 1;
inline void plot_draw_dot(std::vector<uint8_t>* img, int64_t w, int64_t h,
                          int64_t cx, int64_t cy,
                          const std::array<uint8_t, 3>& col,
                          bool outline = true) {
  constexpr std::array<uint8_t, 3> kBlack = {0, 0, 0};
  const int r = kPlotMarkerRadius;
  const int outer = outline ? r + kPlotMarkerOutline : r;
  for (int dy = -outer; dy <= outer; ++dy) {
    for (int dx = -outer; dx <= outer; ++dx) {
      const int rr = dx * dx + dy * dy;
      if (rr <= r * r) {
        plot_put_px(img, w, h, cx + dx, cy + dy, col);
      } else if (outline && rr <= outer * outer) {
        plot_put_px(img, w, h, cx + dx, cy + dy, kBlack);
      }
    }
  }
}

// Minimal 3x5 bitmap font for axis tick labels (digits, '-', '.'); each
// row's low 3 bits are the columns (bit 0 = left). Rendered at kPlotFontScale.
inline std::array<uint8_t, 5> plot_glyph_rows(char c) {
  switch (c) {
    case '0': return {7, 5, 5, 5, 7};
    case '1': return {2, 3, 2, 2, 7};
    case '2': return {7, 4, 7, 1, 7};
    case '3': return {7, 4, 7, 4, 7};
    case '4': return {5, 5, 7, 4, 4};
    case '5': return {7, 1, 7, 4, 7};
    case '6': return {7, 1, 7, 5, 7};
    case '7': return {7, 4, 2, 2, 2};
    case '8': return {7, 5, 7, 5, 7};
    case '9': return {7, 5, 7, 4, 7};
    case '-': return {0, 0, 7, 0, 0};
    case '.': return {0, 0, 0, 0, 2};
    default: return {0, 0, 0, 0, 0};
  }
}

inline int64_t plot_text_width(const std::string& s) {
  return static_cast<int64_t>(s.size()) * 4 * kPlotFontScale;
}

inline void plot_draw_text(std::vector<uint8_t>* img, int64_t w, int64_t h,
                           int64_t x, int64_t y, const std::string& s,
                           const std::array<uint8_t, 3>& col) {
  int64_t cx = x;
  for (char c : s) {
    const auto rows = plot_glyph_rows(c);
    for (int r = 0; r < 5; ++r) {
      for (int ci = 0; ci < 3; ++ci) {
        if ((rows[static_cast<size_t>(r)] >> ci) & 1) {
          for (int sy = 0; sy < kPlotFontScale; ++sy) {
            for (int sx = 0; sx < kPlotFontScale; ++sx) {
              plot_put_px(img, w, h, cx + ci * kPlotFontScale + sx,
                          y + r * kPlotFontScale + sy, col);
            }
          }
        }
      }
    }
    cx += 4 * kPlotFontScale;
  }
}

// Format a tick value with `dec` decimals (0 or 1); no locale / sprintf.
inline std::string plot_format_tick(float value, int dec) {
  const bool neg = value < 0.0F;
  const float av = neg ? -value : value;
  std::string mag;
  bool nonzero = false;
  if (dec <= 0) {
    const int64_t v = static_cast<int64_t>(av + 0.5F);
    mag = std::to_string(v);
    nonzero = v != 0;
  } else {
    const int64_t scaled = static_cast<int64_t>(av * 10.0F + 0.5F);
    mag = std::to_string(scaled / 10) + "." + std::to_string(scaled % 10);
    nonzero = scaled != 0;
  }
  return (neg && nonzero ? "-" : "") + mag;
}

// Constant top-left title position (x, y, size) in a layer's [0,1] space.
inline std::shared_ptr<holoscan::Tensor> plot_title_pos() {
  const std::array<float, 3> pos = {0.03F, 0.02F, 0.05F};
  auto tensor = radar::make_device_tensor({1, 3}, radar::real_dtype<float>());
  if (cudaMemcpy(tensor->data(), pos.data(), sizeof(pos),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("plot_title_pos: title position H2D failed");
  }
  return tensor;
}

// Holoviz layer names for the side plots, shared by RollingPlotOp (emitted
// tensor keys) and compose() (InputSpec names). Each plot is an RGBA image
// (the rasterized scatter) plus a TEXT layer for its title.
constexpr const char* kRangePlotLayer = "range_plot";
constexpr const char* kVelocityPlotLayer = "velocity_plot";
constexpr const char* kRangeTitleLayer = "range_title";
constexpr const char* kVelocityTitleLayer = "velocity_title";

// Rolling time-series plots of the per-cluster centroids. Keeps a scrolling
// window of the last `window` frames and, each tick, rasterizes two RGBA
// images (range vs time, velocity vs time): each cluster's centroid — the
// DetectionReductionOp representative cell (`rep_idx`) resolved to physical
// units — is a round dot colored by cluster. The Y axes are fixed to the
// physical extents of the range / velocity bins (so absolute level is a stable
// reference); X is the frame window, newest at the right. HolovizOp places the
// images in right-side sub-viewports (see compose()). Individual detections
// are shown on the RD map (ClusterOverlayOp), not here.
//
// In:  rep_idx       — int64, (num_clusters, 2): representative cell per
//                     cluster (doppler_bin, range_bin); the dot markers.
// In:  labels        — int64, (num_points,), optional: presence flags clustered
//                     mode (reps color by their RD cell, matching the RD-map /
//                     boresight markers); absent → peak_only (color by row
//                     index). The label values themselves are not read.
// In:  range_bins    — float32, (num_range,): range plot Y extent + rep lookup.
// In:  velocity_bins — float32, (num_doppler,): velocity Y extent + rep lookup.
// Out: range_plot / velocity_plot — uint8 RGBA (H, W, 4) images.
// Out: range_title / velocity_title — float32 (1, 3) TEXT positions.
class RollingPlotOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RollingPlotOp)
  RollingPlotOp() = default;

  int64_t window{120};   // set from cfg.plots.window before the graph runs
  double frame_period{0.1};  // s/frame; scales the time-axis tick labels

  // Optional fixed Y-axis extent for the velocity plot. When finite, it
  // overrides the full-axis min/max derived from velocity_bins — useful to zoom
  // in when the targets occupy a small part of the unambiguous span (e.g. slow
  // radial velocities near zero). NaN (default) keeps the full-axis extent.
  float velocity_ylim_lo{std::numeric_limits<float>::quiet_NaN()};
  float velocity_ylim_hi{std::numeric_limits<float>::quiet_NaN()};

  // Rasterized plot image size (pixels). HolovizOp scales it into its view.
  static constexpr int64_t kW = 420;
  static constexpr int64_t kH = 300;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("rep_idx");
    // Optional: per-detection DBSCAN labels. Only their presence is used, as a
    // mode flag: present → clustered (each rep's trace color keyed off its RD
    // cell, matching the RD-map / boresight markers); absent → peak_only (color
    // by rep_idx row index).
    spec.input<holoscan::TensorMap>("labels")
        .condition(holoscan::ConditionType::kNone);
    spec.input<holoscan::TensorMap>("range_bins");
    spec.input<holoscan::TensorMap>("velocity_bins");
    spec.output<holoscan::TensorMap>("plots");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const std::string who = "RollingPlotOp";
    auto rep_idx = radar::receive_single(op_input, "rep_idx", who);
    auto labels = radar::receive_optional(op_input, "labels", who);
    auto range_bins = radar::receive_single(op_input, "range_bins", who);
    auto velocity_bins = radar::receive_single(op_input, "velocity_bins", who);

    const auto rshape = rep_idx->shape();
    if (rshape.size() != 2 || rshape[1] != 2) {
      throw std::runtime_error(
          "RollingPlotOp: rep_idx must be (num_clusters, 2)");
    }
    const int64_t num_clusters = rshape[0];

    // Cache the physical bin axes once (constant per run); they fix the Y
    // extents and resolve rep_idx bins to physical range / velocity.
    if (!axis_set_) {
      const auto load_axis =
          [](const std::shared_ptr<holoscan::Tensor>& t, const char* name,
             std::vector<float>* out, float* lo, float* hi) {
            const auto sh = t->shape();
            if (sh.size() != 1 || sh[0] < 1) {
              throw std::runtime_error(std::string("RollingPlotOp: ") + name +
                                       " must be a non-empty rank-1 tensor");
            }
            out->resize(static_cast<size_t>(sh[0]));
            if (static_cast<size_t>(t->nbytes()) !=
                sizeof(float) * out->size()) {
              throw std::runtime_error(std::string("RollingPlotOp: ") + name +
                                       " must be float32");
            }
            if (cudaMemcpy(out->data(), t->data(),
                           static_cast<size_t>(t->nbytes()),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
              throw std::runtime_error(std::string("RollingPlotOp: ") + name +
                                       " D2H copy failed");
            }
            const auto mm = std::minmax_element(out->begin(), out->end());
            *lo = *mm.first;
            *hi = *mm.second;
          };
      if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("RollingPlotOp: device synchronize failed");
      }
      load_axis(range_bins, "range_bins", &range_axis_, &r_lo_, &r_hi_);
      load_axis(velocity_bins, "velocity_bins", &velocity_axis_, &v_lo_,
                &v_hi_);
      // Apply the optional fixed velocity Y extent (zoom). Bin→physical
      // lookups still use the full cached axis; only the plot window changes.
      apply_ylim(velocity_ylim_lo, velocity_ylim_hi, &v_lo_, &v_hi_,
                 "velocity");
      axis_set_ = true;
    }

    std::vector<int64_t> rep(static_cast<size_t>(num_clusters * 2));
    if (static_cast<size_t>(rep_idx->nbytes()) !=
        sizeof(int64_t) * rep.size()) {
      throw std::runtime_error("RollingPlotOp: rep_idx dtype/size mismatch");
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("RollingPlotOp: device synchronize failed");
    }
    if (num_clusters > 0 &&
        cudaMemcpy(rep.data(), rep_idx->data(),
                   static_cast<size_t>(rep_idx->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("RollingPlotOp: rep_idx D2H failed");
    }

    // Clustered (labels present): color each rep by its cluster's range rank
    // (rep_color_ranks) — the same stable, collision-free key ClusterOverlayOp /
    // BoresightViewOp use, keeping all three panels in sync. peak_only (no
    // labels): the "reps" are the raw peaks, so color by row index like the
    // overlay. labels are only a mode flag here; their values are not read.
    const bool have_labels = static_cast<bool>(labels);
    const std::vector<int> ranks =
        have_labels ? rep_color_ranks(rep, num_clusters) : std::vector<int>();

    // Append each cluster's representative cell (the dot marker, from rep_idx
    // mapped to physical units), then evict frames older than the window.
    for (int64_t c = 0; c < num_clusters; ++c) {
      const int64_t dd = rep[static_cast<size_t>(c * 2 + 0)];
      const int64_t rr = rep[static_cast<size_t>(c * 2 + 1)];
      if (dd < 0 || dd >= static_cast<int64_t>(velocity_axis_.size()) ||
          rr < 0 || rr >= static_cast<int64_t>(range_axis_.size())) {
        throw std::runtime_error("RollingPlotOp: rep_idx out of axis bounds");
      }
      const int bucket = have_labels ? ranks[static_cast<size_t>(c)]
                                     : static_cast<int>(c % kNumClusterColors);
      cen_hist_.push_back({tick_, range_axis_[static_cast<size_t>(rr)],
                           velocity_axis_[static_cast<size_t>(dd)], bucket});
    }
    const int64_t oldest = tick_ - window + 1;
    while (!cen_hist_.empty() && cen_hist_.front().t < oldest) {
      cen_hist_.pop_front();
    }

    // Rasterize both plots and emit, plus the (constant) title positions.
    holoscan::TensorMap out;
    out.insert({kRangePlotLayer, render_plot(/*use_range=*/true, oldest)});
    out.insert(
        {kVelocityPlotLayer, render_plot(/*use_range=*/false, oldest)});
    out.insert({kRangeTitleLayer, plot_title_pos()});
    out.insert({kVelocityTitleLayer, plot_title_pos()});
    op_output.emit(out, "plots");
    ++tick_;
  }

 private:
  struct Sample {
    int64_t t;
    float r;
    float v;
    int bucket;  // [0, kNumClusterColors) cluster color, or kNumClusterColors
  };

  // Override the loaded [lo, hi] extent when both limits are finite and
  // ordered; otherwise keep the full-axis extent. Throws on lo >= hi so a
  // misconfigured zoom fails loudly rather than rendering a degenerate axis.
  static void apply_ylim(float lim_lo, float lim_hi, float* lo, float* hi,
                         const char* name) {
    if (std::isnan(lim_lo) && std::isnan(lim_hi)) {
      return;
    }
    if (std::isnan(lim_lo) || std::isnan(lim_hi)) {
      throw std::runtime_error(std::string("RollingPlotOp: ") + name +
                               " ylim requires both lo and hi");
    }
    if (!(lim_lo < lim_hi)) {
      throw std::runtime_error(std::string("RollingPlotOp: ") + name +
                               " ylim lo must be < hi");
    }
    *lo = lim_lo;
    *hi = lim_hi;
  }

  // Rasterize one plot: use_range picks the range vs velocity axis. Returns a
  // (kH, kW, 4) uint8 RGBA device tensor.
  std::shared_ptr<holoscan::Tensor> render_plot(bool use_range,
                                                int64_t oldest) {
    constexpr std::array<uint8_t, 3> kBg = {20, 20, 28};
    constexpr std::array<uint8_t, 3> kGrid = {80, 80, 95};
    std::vector<uint8_t> img(static_cast<size_t>(kW * kH * 4));
    for (int64_t p = 0; p < kW * kH; ++p) {
      img[static_cast<size_t>(p * 4 + 0)] = kBg[0];
      img[static_cast<size_t>(p * 4 + 1)] = kBg[1];
      img[static_cast<size_t>(p * 4 + 2)] = kBg[2];
      img[static_cast<size_t>(p * 4 + 3)] = 255;
    }
    // Plot box; margins leave room for the Holoviz title (top), the Y tick
    // labels (left) and the X tick labels (bottom).
    constexpr int64_t kLeft = 46, kRight = kW - 12, kTop = 30,
                      kBottom = kH - 26;
    for (int64_t x = kLeft; x <= kRight; ++x) {
      plot_put_px(&img, kW, kH, x, kTop, kGrid);
      plot_put_px(&img, kW, kH, x, kBottom, kGrid);
    }
    for (int64_t y = kTop; y <= kBottom; ++y) {
      plot_put_px(&img, kW, kH, kLeft, y, kGrid);
      plot_put_px(&img, kW, kH, kRight, y, kGrid);
    }

    const float lo = use_range ? r_lo_ : v_lo_;
    const float hi = use_range ? r_hi_ : v_hi_;
    const float span = std::max(hi - lo, 1e-6F);

    // Axis tick labels (rasterized). Y ticks span the fixed value axis; X
    // ticks span the scrolling window in seconds (0 = now, negative = past).
    constexpr std::array<uint8_t, 3> kLabel = {200, 200, 210};
    constexpr int kTicks = 5;
    const int y_dec = span < 10.0F ? 1 : 0;
    for (int i = 0; i < kTicks; ++i) {
      const float f = static_cast<float>(i) / (kTicks - 1);
      const int64_t y =
          kBottom -
          static_cast<int64_t>(f * static_cast<float>(kBottom - kTop));
      for (int64_t x = kLeft + 1; x < kRight; x += 4) {  // dashed gridline
        plot_put_px(&img, kW, kH, x, y, kGrid);
      }
      for (int64_t d = 1; d <= 3; ++d) {  // tick mark
        plot_put_px(&img, kW, kH, kLeft - d, y, kLabel);
      }
      const std::string s = plot_format_tick(lo + f * span, y_dec);
      plot_draw_text(&img, kW, kH, kLeft - 4 - plot_text_width(s),
                     y - kPlotFontScale * 2, s, kLabel);
    }
    for (int i = 0; i < kTicks; ++i) {
      const float f = static_cast<float>(i) / (kTicks - 1);
      const int64_t x =
          kLeft +
          static_cast<int64_t>(f * static_cast<float>(kRight - kLeft));
      for (int64_t d = 1; d <= 3; ++d) {  // tick mark
        plot_put_px(&img, kW, kH, x, kBottom + d, kLabel);
      }
      const float secs = -(1.0F - f) * static_cast<float>(window - 1) *
                         static_cast<float>(frame_period);
      const std::string s = plot_format_tick(secs, 0);
      plot_draw_text(&img, kW, kH, x - plot_text_width(s) / 2, kBottom + 5, s,
                     kLabel);
    }

    const auto to_px = [&](int64_t t, float val, int64_t* px, int64_t* py) {
      const float xf = static_cast<float>(t - oldest) /
                       static_cast<float>(window - 1);
      const float yf = std::clamp((val - lo) / span, 0.0F, 1.0F);
      *px = kLeft +
            static_cast<int64_t>(xf * static_cast<float>(kRight - kLeft));
      *py = kBottom -
            static_cast<int64_t>(yf * static_cast<float>(kBottom - kTop));
    };

    // Cluster centroids: a round dot per sample, colored by cluster — the same
    // marker style as the RD-map overlay. No outline: samples are one-per-frame
    // and overlap into a trail, so the rim would speckle the band.
    for (const Sample& s : cen_hist_) {
      int64_t px = 0, py = 0;
      to_px(s.t, use_range ? s.r : s.v, &px, &py);
      plot_draw_dot(&img, kW, kH, px, py, plot_bucket_color(s.bucket),
                    /*outline=*/false);
    }

    auto tensor = radar::make_device_tensor({kH, kW, 4},
                                            DLDataType{kDLUInt, 8, 1});
    if (cudaMemcpy(tensor->data(), img.data(), img.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("RollingPlotOp: plot image H2D upload failed");
    }
    return tensor;
  }

  std::deque<Sample> cen_hist_;
  int64_t tick_{0};
  std::vector<float> range_axis_;     // cached range_bins (bin → physical)
  std::vector<float> velocity_axis_;  // cached velocity_bins
  float r_lo_{0}, r_hi_{1}, v_lo_{0}, v_hi_{1};
  bool axis_set_{false};
};

// Holoviz layer names for the boresight angular-scatter panel.
constexpr const char* kBoresightPlotLayer = "boresight_plot";
constexpr const char* kBoresightTitleLayer = "boresight_title";

// Boresight (angular field-of-view) scatter. Each CFAR detection is plotted at
// its beamscan (azimuth, elevation) estimate — the view looking out along the
// array boresight, so a marker's screen position is its angular sky position:
// azimuth left→right (x), elevation bottom→top (y). Markers are colored by
// DBSCAN cluster (noise gray), matching the RD-map overlay. Axes are labeled in
// degrees over fixed extents (the beamscan grid, padded when an axis is a
// single point, e.g. the default el grid), so the panel is a stable reference.
// Instantaneous (current-frame) scatter — angle barely moves frame-to-frame,
// so no rolling history is kept.
//
// In:  aoa_estimates — float32, (num_points, 2): (az, el) radians per det.
// In:  labels        — int64, (num_points,): cluster id (noise = -1),
//                     row-aligned with aoa_estimates.
// Out: boresight_plot  — uint8 RGBA (H, W, 4) image.
// Out: boresight_title — float32 (1, 3) TEXT position.
class BoresightViewOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BoresightViewOp)
  BoresightViewOp() = default;

  // Angular extents (radians), set from the beamscan grid before the run.
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};

  static constexpr int64_t kW = 420;
  static constexpr int64_t kH = 300;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("aoa_estimates");
    // Optional: cluster ids (row-aligned with aoa_estimates). Present →
    // clustered mode: markers color by their cluster. Absent → peak_only: each
    // marker colors by its own row index.
    spec.input<holoscan::TensorMap>("labels")
        .condition(holoscan::ConditionType::kNone);
    // Optional: per-cluster representative RD cell (num_clusters, 2), row c =
    // cluster label c. When present with labels, clusters color by their rep
    // cell (stable across frames) — the same key the RD map / rolling plots use.
    spec.input<holoscan::TensorMap>("rep_idx")
        .condition(holoscan::ConditionType::kNone);
    spec.output<holoscan::TensorMap>("plot");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const std::string who = "BoresightViewOp";
    auto aoa = radar::receive_single(op_input, "aoa_estimates", who);
    auto labels = radar::receive_optional(op_input, "labels", who);
    auto rep_idx = radar::receive_optional(op_input, "rep_idx", who);
    const bool have_labels = static_cast<bool>(labels);

    const auto ashape = aoa->shape();
    if (ashape.size() != 2 || ashape[1] != 2) {
      throw std::runtime_error(
          "BoresightViewOp: aoa_estimates must be (num_points, 2)");
    }
    const int64_t n = ashape[0];
    if (have_labels) {
      const auto lshape = labels->shape();
      if (lshape.size() != 1 || lshape[0] != n) {
        throw std::runtime_error(
            "BoresightViewOp: labels must be (num_points,) row-aligned with "
            "aoa_estimates");
      }
    }

    std::vector<float> ang(static_cast<size_t>(n * 2));
    std::vector<int64_t> lab(static_cast<size_t>(n));
    if (static_cast<size_t>(aoa->nbytes()) != sizeof(float) * ang.size()) {
      throw std::runtime_error(
          "BoresightViewOp: aoa_estimates must be float32");
    }
    if (have_labels &&
        static_cast<size_t>(labels->nbytes()) != sizeof(int64_t) * lab.size()) {
      throw std::runtime_error("BoresightViewOp: labels byte size mismatch");
    }
    // Optional per-cluster rep cells (num_clusters, 2): row = label.
    std::vector<int64_t> rep;
    int64_t num_rep = 0;
    if (rep_idx) {
      const auto rep_shape = rep_idx->shape();
      if (rep_shape.size() != 2 || rep_shape[1] != 2) {
        throw std::runtime_error(
            "BoresightViewOp: rep_idx must be (num_clusters, 2)");
      }
      num_rep = rep_shape[0];
      rep.resize(static_cast<size_t>(num_rep * 2));
      if (static_cast<size_t>(rep_idx->nbytes()) !=
          sizeof(int64_t) * rep.size()) {
        throw std::runtime_error("BoresightViewOp: rep_idx byte size mismatch");
      }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("BoresightViewOp: device synchronize failed");
    }
    if (n > 0) {
      if (cudaMemcpy(ang.data(), aoa->data(),
                     static_cast<size_t>(aoa->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("BoresightViewOp: aoa_estimates D2H failed");
      }
      if (have_labels &&
          cudaMemcpy(lab.data(), labels->data(),
                     static_cast<size_t>(labels->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("BoresightViewOp: labels D2H failed");
      }
    }
    if (num_rep > 0 &&
        cudaMemcpy(rep.data(), rep_idx->data(), sizeof(int64_t) * rep.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("BoresightViewOp: rep_idx D2H failed");
    }
    const bool have_rep = rep_idx && num_rep > 0;
    const std::vector<int> ranks =
        have_rep ? rep_color_ranks(rep, num_rep) : std::vector<int>();

    // Per-detection color bucket. Clustered (labels present): color by the
    // cluster's range rank when rep cells are available (stable, collision-
    // free), else the size-rank label; noise (-1) → gray. peak_only (no labels):
    // color by row index so peaks stay distinct — matching the RD map / plots.
    std::vector<int> bucket(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      int b;
      if (have_labels) {
        const int64_t label = lab[static_cast<size_t>(i)];
        if (label < 0) {
          b = kNumClusterColors;
        } else if (have_rep && label < num_rep) {
          b = ranks[static_cast<size_t>(label)];
        } else {
          b = static_cast<int>(label % kNumClusterColors);
        }
      } else {
        b = static_cast<int>(i % kNumClusterColors);
      }
      bucket[static_cast<size_t>(i)] = b;
    }

    holoscan::TensorMap out;
    out.insert({kBoresightPlotLayer, render(ang, bucket, n)});
    out.insert({kBoresightTitleLayer, plot_title_pos()});
    op_output.emit(out, "plot");
  }

 private:
  std::shared_ptr<holoscan::Tensor> render(const std::vector<float>& ang,
                                           const std::vector<int>& bucket,
                                           int64_t n) {
    constexpr std::array<uint8_t, 3> kBg = {20, 20, 28};
    constexpr std::array<uint8_t, 3> kGrid = {80, 80, 95};
    constexpr std::array<uint8_t, 3> kLabel = {200, 200, 210};
    std::vector<uint8_t> img(static_cast<size_t>(kW * kH * 4));
    for (int64_t p = 0; p < kW * kH; ++p) {
      img[static_cast<size_t>(p * 4 + 0)] = kBg[0];
      img[static_cast<size_t>(p * 4 + 1)] = kBg[1];
      img[static_cast<size_t>(p * 4 + 2)] = kBg[2];
      img[static_cast<size_t>(p * 4 + 3)] = 255;
    }
    constexpr int64_t kLeft = 46, kRight = kW - 12, kTop = 30,
                      kBottom = kH - 26;
    for (int64_t x = kLeft; x <= kRight; ++x) {
      plot_put_px(&img, kW, kH, x, kTop, kGrid);
      plot_put_px(&img, kW, kH, x, kBottom, kGrid);
    }
    for (int64_t y = kTop; y <= kBottom; ++y) {
      plot_put_px(&img, kW, kH, kLeft, y, kGrid);
      plot_put_px(&img, kW, kH, kRight, y, kGrid);
    }

    // Label the axes in degrees; markers map from radians through the same
    // fixed extents.
    constexpr double kRad2Deg = 57.29577951308232;
    const float ax_lo = static_cast<float>(az_lo * kRad2Deg);
    const float ax_hi = static_cast<float>(az_hi * kRad2Deg);
    const float ey_lo = static_cast<float>(el_lo * kRad2Deg);
    const float ey_hi = static_cast<float>(el_hi * kRad2Deg);
    const float ax_span = std::max(ax_hi - ax_lo, 1e-6F);
    const float ey_span = std::max(ey_hi - ey_lo, 1e-6F);

    constexpr int kTicks = 5;
    for (int i = 0; i < kTicks; ++i) {  // elevation (Y)
      const float f = static_cast<float>(i) / (kTicks - 1);
      const int64_t y =
          kBottom -
          static_cast<int64_t>(f * static_cast<float>(kBottom - kTop));
      for (int64_t x = kLeft + 1; x < kRight; x += 4) {
        plot_put_px(&img, kW, kH, x, y, kGrid);
      }
      for (int64_t d = 1; d <= 3; ++d) {
        plot_put_px(&img, kW, kH, kLeft - d, y, kLabel);
      }
      const std::string s = plot_format_tick(ey_lo + f * ey_span, 0);
      plot_draw_text(&img, kW, kH, kLeft - 4 - plot_text_width(s),
                     y - kPlotFontScale * 2, s, kLabel);
    }
    for (int i = 0; i < kTicks; ++i) {  // azimuth (X)
      const float f = static_cast<float>(i) / (kTicks - 1);
      const int64_t x =
          kLeft + static_cast<int64_t>(f * static_cast<float>(kRight - kLeft));
      for (int64_t d = 1; d <= 3; ++d) {
        plot_put_px(&img, kW, kH, x, kBottom + d, kLabel);
      }
      const std::string s = plot_format_tick(ax_lo + f * ax_span, 0);
      plot_draw_text(&img, kW, kH, x - plot_text_width(s) / 2, kBottom + 5, s,
                     kLabel);
    }

    // One round dot per detection at its (az, el), colored by cluster — the
    // same marker style as the RD-map overlay.
    for (int64_t i = 0; i < n; ++i) {
      const float az_deg = ang[static_cast<size_t>(i * 2 + 0)] *
                           static_cast<float>(kRad2Deg);
      const float el_deg = ang[static_cast<size_t>(i * 2 + 1)] *
                           static_cast<float>(kRad2Deg);
      const float xf = (az_deg - ax_lo) / ax_span;
      const float yf = (el_deg - ey_lo) / ey_span;
      const int64_t px =
          kLeft +
          static_cast<int64_t>(xf * static_cast<float>(kRight - kLeft));
      const int64_t py =
          kBottom -
          static_cast<int64_t>(yf * static_cast<float>(kBottom - kTop));
      plot_draw_dot(&img, kW, kH, px, py,
                    plot_bucket_color(bucket[static_cast<size_t>(i)]));
    }

    auto tensor =
        radar::make_device_tensor({kH, kW, 4}, DLDataType{kDLUInt, 8, 1});
    if (cudaMemcpy(tensor->data(), img.data(), img.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("BoresightViewOp: plot image H2D upload failed");
    }
    return tensor;
  }
};


}  // namespace holoscan::radar::app
