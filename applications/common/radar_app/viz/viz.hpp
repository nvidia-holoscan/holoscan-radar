// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "radar_app/viz/viz_kernels.hpp"
#include "radar_common/math/constants.hpp"
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

// Dynamic Holoviz layer spec + type, aliased here (near the top) because both
// the RD-map overlay (ClusterOverlayOp) and the side-panel helpers below emit
// them. The panel-primitive helpers that build these specs live further down.
using PlotInputSpec = holoscan::ops::HolovizOp::InputSpec;
using PlotInputType = holoscan::ops::HolovizOp::InputType;

// Panel placement, window-normalized (mirrors HolovizOp::InputSpec::View).
// Hoisted here because the RD-map overlay (ClusterOverlayOp) and the side-panel
// helpers below both use it.
struct PanelRect {
  float x{0.0F}, y{0.0F}, w{1.0F}, h{1.0F};
};

// A tick: fractional position along its
// axis ([0,1], 0 = lo/left/bottom) + label.
struct PanelTick {
  float frac;
  std::string label;
};

// Tick-value formatter (defined with the panel helpers below; forward-declared
// so ClusterOverlayOp, which precedes it, can label its RD-map axes).
inline std::string plot_format_tick(float value, int dec);

// Holoviz bakes its font atlas at a fixed 25 px bitmap and scales those glyphs,
// so text reads crispest at ~25 px on screen and pixelates when scaled far from
// it (badly when upscaled on a large window). These helpers therefore size text
// to a fixed pixel target via `px_height` (the on-screen pixel height of the
// layer's View), keeping it readable and at/below the atlas resolution at any
// window size. Chosen targets:
constexpr float kRdTickPx = 20.0F;    // axis tick + unit labels
constexpr float kRdTitlePx = 25.0F;   // RD-map title
constexpr float kClockPx = 23.0F;     // scenario clock

// Scenario-clock + RD-map-axes overlays (defined with the panel helpers below;
// forward-declared so ClusterOverlayOp, which precedes them, can emit them).
inline void emit_sim_clock(holoscan::TensorMap& out,
                           std::vector<PlotInputSpec>* specs, double seconds,
                           float px_height);
// Frame border + tick marks/labels + title for the RD map, drawn in the margins
// around the inset image (View = `region`; the image occupies the shared kBox
// sub-rect). xticks/yticks may be empty (frame + title still drawn). px_height
// is the region View's on-screen pixel height, for pixel-locked text sizing.
inline void emit_rd_axes(holoscan::TensorMap& out,
                         std::vector<PlotInputSpec>* specs,
                         const PanelRect& region,
                         const std::vector<PanelTick>& xticks,
                         const std::vector<PanelTick>& yticks,
                         const std::string& title, float px_height,
                         float box_aspect);

// Slice one rank-2 (num_doppler, num_range) plane out of a batched map for
// visualization — Holoviz helpers want rank-2 while the pipeline carries a
// leading batch. Rank-agnostic in both the leading rank and the dtype: the
// trailing plane is contiguous, so this is an offset plus a straight D2D copy
// of `plane` elements, whatever the element type.
//
//   rank-3 (N_ch, D, R)      — pick an RX / virtual channel: `channel`.
//   rank-N (..., D, R)       — pick any leading cell: `leading_index`, one
//                              entry per leading axis (e.g. {i_az, i_el} on the
//                              dense path's (az, el, D, R) cube).
//
// `leading_index` wins when non-empty; `channel` is the rank-3 shorthand.
//
// A single plane is one look at the data. When the detector integrates or
// collapses across the leading axes, this is NOT what it saw — prefer
// CFARDetectorOp's `cfar_power` output, which is the thresholded surface
// itself. (Slicing `cfar_power` is still a single beam/channel of it, but at
// least it is the detector's own power domain: see the dense app, where the
// beam nearest boresight is chosen deliberately.)
class RdChannelSliceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RdChannelSliceOp)
  RdChannelSliceOp() = default;

  int64_t channel{0};
  std::vector<int64_t> leading_index;  // one index per leading axis

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
    if (shape.size() < 3) {
      throw std::runtime_error(
          "RdChannelSliceOp: expected rank >= 3 (..., D, R), got rank " +
          std::to_string(shape.size()));
    }
    const size_t n_leading = shape.size() - 2;
    const int64_t num_doppler = shape[shape.size() - 2];
    const int64_t num_range = shape[shape.size() - 1];

    const std::vector<int64_t> index =
        leading_index.empty() ? std::vector<int64_t>{channel} : leading_index;
    if (index.size() != n_leading) {
      throw std::runtime_error(
          "RdChannelSliceOp: leading_index has " +
          std::to_string(index.size()) + " entries, input has " +
          std::to_string(n_leading) + " leading axes");
    }
    // Row-major fold of the leading multi-index into a plane number.
    int64_t linear = 0;
    for (size_t i = 0; i < n_leading; ++i) {
      if (index[i] < 0 || index[i] >= shape[i]) {
        throw std::runtime_error(
            "RdChannelSliceOp: leading index " + std::to_string(index[i]) +
            " out of range for axis " + std::to_string(i) + " (extent " +
            std::to_string(shape[i]) + ")");
      }
      linear = linear * shape[i] + index[i];
    }

    const DLDataType dtype = rd_maps->dtype();
    const size_t elem_size =
        static_cast<size_t>(dtype.bits) / 8u * static_cast<size_t>(dtype.lanes);
    const size_t plane_bytes =
        static_cast<size_t>(num_doppler * num_range) * elem_size;
    // Stream-scoped: receive_cuda_stream orders this op behind the producer
    // with events, so the copy needs no host-side wait at all — and no
    // device-wide barrier, which would stall every other branch of the graph.
    cudaStream_t stream = op_input.receive_cuda_stream("rd_maps");
    auto out = radar::make_device_tensor({num_doppler, num_range}, dtype);
    const auto* src = static_cast<const std::byte*>(rd_maps->data()) +
                      static_cast<size_t>(linear) * plane_bytes;
    if (cudaMemcpyAsync(out->data(), src, plane_bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      throw std::runtime_error("RdChannelSliceOp: D2D slice copy failed");
    }
    radar::emit_single(op_output, "rd_map", out);
  }
};

// Max-projects a (..., D, R) cube down ALL its leading axes, emitting the
// (D, R) plane of per-cell maxima as float32 power. Drop-in alternative to
// RdChannelSliceOp — same `rd_maps` -> `rd_map` ports — for the dense path,
// where the leading axes are a beam grid rather than physical channels.
//
// Why prefer it there: AnglePeakPickOp collapses each detected RD cell to its
// *peak* beam, so the detections drawn over the map are already a max-over-
// beams projection. Slicing one fixed beam leaves the image in a different
// projection from the overlay, and an off-boresight target ends up drawn over a
// cell the beam pattern attenuated. Max-projecting puts both in the same
// domain.
//
// The cost is a higher noise floor: the max of N beams of noise sits above the
// mean of one. For an n-beam grid of exponential-distributed power the rise is
// ~10*log10(H_n) dB (~7.6 dB at 13x13), less in practice since adjacent beams
// are correlated across the mainlobe. RdMapToImageOp's per-column median
// normalization absorbs part of it, because the median rises too.
//
// Unlike RdChannelSliceOp this is not a byte copy: it interprets values, so
// complex input is reduced on |z|^2 to stay in the same power domain as a real
// cfar_power surface. Either way the output is float32 power and
// RdMapToImageOp maps it with db_factor = 10.
class RdAngleMaxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RdAngleMaxOp)
  RdAngleMaxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("rd_maps");
    spec.output<holoscan::TensorMap>("rd_map");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    auto rd_maps = radar::receive_single(op_input, "rd_maps", "RdAngleMaxOp");
    const auto& shape = rd_maps->shape();
    if (shape.size() < 3) {
      throw std::runtime_error(
          "RdAngleMaxOp: expected rank >= 3 (..., D, R), got rank " +
          std::to_string(shape.size()));
    }
    const int64_t num_doppler = shape[shape.size() - 2];
    const int64_t num_range = shape[shape.size() - 1];
    int64_t num_planes = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
      num_planes *= shape[i];
    }

    const DLDataType dtype = rd_maps->dtype();
    const bool is_complex =
        radar::dldatatype_equals(dtype, radar::complex_dtype<float>());
    if (!is_complex &&
        !radar::dldatatype_equals(dtype, radar::real_dtype<float>())) {
      throw std::runtime_error(
          "RdAngleMaxOp: expected complex64 or float32 rd_maps, got " +
          radar::dldatatype_name(dtype));
    }

    cudaStream_t stream = op_input.receive_cuda_stream("rd_maps");
    auto out = radar::make_device_tensor({num_doppler, num_range},
                                         radar::real_dtype<float>());
    app::rd_map_max_over_planes(rd_maps->data(), is_complex, num_planes,
                                num_doppler, num_range,
                                static_cast<float*>(out->data()), stream);
    radar::emit_single(op_output, "rd_map", out);
  }
};

// Converts an RD map into a log-magnitude float32 image for Holoviz (rendered
// through the turbo color lookup table). Layout is (doppler, range): Holoviz
// x = range, y = Doppler.
//
// Accepts either dtype a producer upstream may hand it:
//   complex64 — a raw RD map (e.g. RdChannelSliceOp). Magnitude is taken here
//               and mapped as 20*log10(|x| / median).
//   float32   — a POWER surface, i.e. CFARDetectorOp's cfar_power output: the
//               exact map the detector thresholds (post-NCI integrated power,
//               or |x|^2 when nci_axes is empty). Mapped as
//               10*log10(p / median).
// The two forms are the same dB surface when p = |x|^2, since the median
// commutes with the monotonic sqrt — so the displays stay directly comparable
// and a detection overlaid on either sits at the same relative level.
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
    const int64_t count = num_doppler * num_range;

    // Amplitude (complex) vs power (real) only changes the dB factor:
    // 20*log10(|x|) == 10*log10(|x|^2).
    const DLDataType dtype = rd_map->dtype();
    const bool is_complex =
        radar::dldatatype_equals(dtype, radar::complex_dtype<float>());
    if (!is_complex &&
        !radar::dldatatype_equals(dtype, radar::real_dtype<float>())) {
      throw std::runtime_error(
          "RdMapToImageOp: expected complex64 or float32 rd_map, got " +
          radar::dldatatype_name(dtype));
    }

    // Inherit the producer's stream: receive_cuda_stream also orders this
    // operator's work behind the producer, so the whole mapping stays
    // stream-ordered and no device-wide barrier is needed.
    cudaStream_t stream = op_input.receive_cuda_stream("rd_maps");

    // Holoviz image layers require a rank-3 (height, width, channels) tensor,
    // so add an explicit single-channel dimension. This doubles as the
    // kernels' working buffer.
    auto tensor = radar::make_device_tensor({num_doppler, num_range, 1},
                                            radar::real_dtype<float>());
    ensure_scratch(count, num_range);
    rd_map_to_lut_image(rd_map->data(), is_complex, num_doppler, num_range,
                        is_complex ? 20.0F : 10.0F,
                        static_cast<float>(kLutSize - 1),
                        static_cast<float*>(tensor->data()),
                        static_cast<float*>(col_median_->data()),
                        static_cast<float*>(sorted_->data()),
                        sort_scratch_ ? sort_scratch_->data() : nullptr,
                        sort_scratch_bytes_, stream);

    holoscan::TensorMap out;
    out.insert({"rd_map", tensor});
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "rd_image");
    }
    op_output.emit(out, "rd_image");
  }

 private:
  // Persistent per-shape scratch, so the render path does no allocation beyond
  // the emitted image itself.
  void ensure_scratch(int64_t count, int64_t num_range) {
    if (scratch_count_ == count && scratch_range_ == num_range) {
      return;
    }
    col_median_ =
        radar::make_device_tensor({num_range}, radar::real_dtype<float>());
    sorted_ = radar::make_device_tensor({count}, radar::real_dtype<float>());
    sort_scratch_bytes_ = rd_lut_image_sort_scratch_bytes(count);
    sort_scratch_ =
        sort_scratch_bytes_ > 0
            ? radar::make_device_tensor(
                  {static_cast<int64_t>(sort_scratch_bytes_)},
                  DLDataType{kDLUInt, 8, 1})
            : nullptr;
    scratch_count_ = count;
    scratch_range_ = num_range;
  }

  std::shared_ptr<holoscan::Tensor> col_median_;
  std::shared_ptr<holoscan::Tensor> sorted_;
  std::shared_ptr<holoscan::Tensor> sort_scratch_;
  size_t sort_scratch_bytes_{0};
  int64_t scratch_count_{0};
  int64_t scratch_range_{0};
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
//     colors (golden-angle hues), so a handful of clusters read very
//     differently.
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
  float marker_radius{0.004F};
  float marker_outline{0.002F};
  int64_t marker_segments{20};

  // Aspect ratio (width / height, in pixels) of the RD-map sub-view. The disc x
  // radius is divided by it so the markers render round despite the view
  // stretching normalized coordinates to a non-square rectangle. Set by
  // compose() (see rd_map_view_aspect); 1.0 leaves the discs uncorrected.
  float view_aspect{1.0F};

  // Seconds per frame — turns the internal frame counter into the scenario
  // clock (current simulated time) drawn over the RD map. Matches the source's
  // per-tick scene advance, so tick N is labeled t = N * frame_period.
  double frame_period{0.1};

  // RD-map outer region (window-normalized)
  // for the axes/frame/title chrome. The
  // inset RD image + detection discs occupy the shared kBox sub-rect of this
  // region (set by the app from the layout; see holoviz_layout.hpp's image
  // view). Default full-window is a sane fallback when unset.
  PanelRect rd_region{0.0F, 0.0F, 1.0F, 1.0F};
  std::string rd_title{"Range-Doppler"};  // ASCII hyphen; the baked font atlas
                                           // lacks the en-dash glyph
  // On-screen pixel height of the RD region View (= window height, since the
  // region spans the full height). Drives pixel-locked chrome text sizing so it
  // reads crisply against Holoviz's fixed 25 px font atlas. Set by the app.
  float rd_view_px_height{600.0F};

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
    // Optional physical axis bins (constant
    // per run) for the RD-map tick labels:
    // range_bins (num_range,) → X axis, velocity_bins (num_doppler,) → Y axis.
    // Absent → the frame + title still draw, without numeric tick labels.
    spec.input<holoscan::TensorMap>("range_bins")
        .condition(holoscan::ConditionType::kNone);
    spec.input<holoscan::TensorMap>("velocity_bins")
        .condition(holoscan::ConditionType::kNone);
    spec.output<holoscan::TensorMap>("overlay");
    // RD-map chrome InputSpecs (scenario clock + axes frame/ticks/title as TEXT
    // and LINES layers). Heads the primitive-panel InputSpec chain when panels
    // are on (→ RollingPlotOp.specs_in), else feeds HolovizOp.input_specs
    // directly. The matching geometry/position
    // tensors ride the `overlay` output.
    spec.output<std::vector<PlotInputSpec>>("time_specs");
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
    // Drain the physical-bin queues every tick (upstream emits them each
    // frame); they are only *consumed* once, to build the axis ticks below.
    auto range_bins = radar::receive_optional(op_input, "range_bins", who);
    auto velocity_bins =
        radar::receive_optional(op_input, "velocity_bins", who);
    const bool have_labels = static_cast<bool>(labels);

    // One operator stream, ordered behind every producer whose *data* (not
    // just shape) this op reads back. receive_cuda_stream returns the same
    // internal stream each call and adds an event wait per port, so the copies
    // below need only a stream sync — never a device-wide barrier, which would
    // stall the simulator / range / Doppler / CFAR / estimator branches too.
    // rd_maps is omitted deliberately: only its shape is used.
    cudaStream_t stream = op_input.receive_cuda_stream("detection_idx");
    if (have_labels) {
      op_input.receive_cuda_stream("labels");
    }
    if (rep_idx) {
      op_input.receive_cuda_stream("rep_idx");
    }
    if (range_bins) {
      op_input.receive_cuda_stream("range_bins");
    }
    if (velocity_bins) {
      op_input.receive_cuda_stream("velocity_bins");
    }

    // Cache the RD-map axis extents once (constant per run) from the optional
    // physical bin tensors, then build the tick label lists. X spans the range
    // axis left→right; Y spans the velocity
    // axis, with the image's Doppler bin 0
    // at the top (so the top edge = velocity_bins[0], the bottom edge = the
    // last bin). See emit_rd_axes for the tick placement.
    if (!axes_set_) {
      if (range_bins && velocity_bins) {
        const auto ends = [&](const std::shared_ptr<holoscan::Tensor>& t,
                              const char* name, float* first, float* last) {
          const auto sh = t->shape();
          if (sh.size() != 1 || sh[0] < 1) {
            throw std::runtime_error(std::string("ClusterOverlayOp: ") + name +
                                     " must be a non-empty rank-1 tensor");
          }
          std::vector<float> host(static_cast<size_t>(sh[0]));
          if (static_cast<size_t>(t->nbytes()) != sizeof(float) * host.size()) {
            throw std::runtime_error(std::string("ClusterOverlayOp: ") + name +
                                     " must be float32");
          }
          if (cudaMemcpyAsync(host.data(), t->data(),
                              static_cast<size_t>(t->nbytes()),
                              cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
              cudaStreamSynchronize(stream) != cudaSuccess) {
            throw std::runtime_error(std::string("ClusterOverlayOp: ") + name +
                                     " D2H copy failed");
          }
          *first = host.front();
          *last = host.back();
        };
        float r0, r1, v_first, v_last;
        ends(range_bins, "range_bins", &r0, &r1);
        ends(velocity_bins, "velocity_bins", &v_first, &v_last);
        // Y: frac 0 = bottom = last bin, frac 1 = top = first bin.
        for (int i = 0; i < 5; ++i) {
          const float f = static_cast<float>(i) / 4.0F;
          xticks_.push_back({f, plot_format_tick(r0 + f * (r1 - r0), 0)});
          yticks_.push_back(
              {f, plot_format_tick(v_last + f * (v_first - v_last), 1)});
        }
        // Latch only once the ticks are built; if the bins are not wired the
        // frame + title still draw (labels stay empty) and we retry cheaply.
        axes_set_ = true;
      }
    }

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
    if (num_points > 0) {
      if (cudaMemcpyAsync(bins.data(), idx->data(), idx_bytes,
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        throw std::runtime_error(
            "ClusterOverlayOp: detection_idx D2H copy failed");
      }
      if (have_labels &&
          cudaMemcpyAsync(lab.data(), labels->data(), lab_bytes,
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        throw std::runtime_error("ClusterOverlayOp: labels D2H copy failed");
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("ClusterOverlayOp: stream sync failed");
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
        throw std::runtime_error("ClusterOverlayOp: rep_idx size mismatch");
      }
      if (num_rep > 0 &&
          (cudaMemcpyAsync(rep.data(), rep_idx->data(),
                           sizeof(int64_t) * rep.size(),
                           cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
           cudaStreamSynchronize(stream) != cudaSuccess)) {
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
    // marker_segments triangles (center +
    // two adjacent rim points). The x radius
    // is divided by the view aspect so the discs stay round on screen.
    const int segments = marker_segments > 2 ? static_cast<int>(marker_segments)
                                             : 3;
    std::vector<float> cos_t(static_cast<size_t>(segments) + 1);
    std::vector<float> sin_t(static_cast<size_t>(segments) + 1);
    for (int k = 0; k <= segments; ++k) {
      const float a = static_cast<float>(radar::kTwoPi) *
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

    // RD-map chrome: axes frame/ticks/title, then the scenario clock (current
    // simulated time = frame_index_ * frame_period). Their geometry/label
    // tensors go out with the overlay (→ receivers); the specs go out on
    // time_specs (→ input_specs, directly or via the panel chain).
    std::vector<PlotInputSpec> chrome_specs;
    emit_rd_axes(out, &chrome_specs, rd_region, xticks_, yticks_, rd_title,
                 rd_view_px_height, view_aspect);
    emit_sim_clock(out, &chrome_specs,
                   static_cast<double>(frame_index_) * frame_period,
                   rd_view_px_height);
    op_output.emit(out, "overlay");
    op_output.emit(chrome_specs, "time_specs");
    ++frame_index_;
  }

 private:
  int64_t frame_index_{0};  // ticks since run start → scenario clock
  bool axes_set_{false};    // RD-map axis ticks built once from the bin tensors
  std::vector<PanelTick> xticks_;
  std::vector<PanelTick> yticks_;
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

// ---- Holoviz-primitive panel helpers -----------------------------------
// A side panel can be drawn as native Holoviz layers (geometry + text) instead
// of a rasterized image. Geometry is in the panel's normalized [0, 1] space
// (top-left origin, y down), placed on screen by a View. The producing op emits
// both the data tensors (on `receivers`) and the matching InputSpecs (on the
// dynamic `input_specs` port), so runtime-derived tick labels and only-used
// marker colors are handled without a static per-color layer declaration.
// (PlotInputSpec / PlotInputType / PanelRect / PanelTick are declared near the
// top of this header.)

// Plot box within the panel [0, 1] (leaves margins for title / tick labels).
constexpr float kBoxL = 0.15F, kBoxR = 0.98F, kBoxT = 0.14F, kBoxB = 0.85F;

// A (rows, comps) float device tensor from host data (rows = size / comps).
inline std::shared_ptr<holoscan::Tensor> plot_geom_tensor(
    const std::vector<float>& data, int64_t comps) {
  const int64_t rows =
      comps > 0 ? static_cast<int64_t>(data.size()) / comps : 0;
  auto tensor =
      radar::make_device_tensor({rows, comps}, radar::real_dtype<float>());
  if (!data.empty() &&
      cudaMemcpy(tensor->data(), data.data(), sizeof(float) * data.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("plot_geom_tensor: H2D upload failed");
  }
  return tensor;
}

// InputSpec for one geometry/text layer of a panel, placed in `rect`.
inline PlotInputSpec plot_layer_spec(const std::string& name,
                                     PlotInputType type,
                                     const std::vector<float>& rgba,
                                     const PanelRect& rect, int priority) {
  PlotInputSpec spec(name, type);
  spec.color_ = rgba;
  spec.priority_ = priority;
  PlotInputSpec::View view;
  view.offset_x_ = rect.x;
  view.offset_y_ = rect.y;
  view.width_ = rect.w;
  view.height_ = rect.h;
  spec.views_ = {view};
  return spec;
}

// Roboto-Bold average glyph advance ≈ 0.5× the pixel height for the short
// numeric / label strings we render; used only to right-/center-anchor text
// (Holoviz text is left-anchored at the given position).
constexpr float kGlyphAdvanceRatio = 0.52F;

// Horizontal text anchor relative to the given position.
enum class TextAnchor { kLeft, kCenter, kRight };
struct AbsText {
  float x, y;  // window-normalized; y is the text top
  std::string s;
  TextAnchor anchor;
};

// Emit a TEXT layer in a FULL-WINDOW view (normalized-square) so Holoviz's
// window-aspect text correction is exact and glyphs are not stretched by a
// non-square sub-view (Holoviz corrects text for the window aspect only, then
// the layer's View re-stretches it — see geometry_layer.cpp). Positions are
// absolute window-normalized; `size_frac` is the glyph height as a fraction of
// the window height; right/center anchoring uses a width estimate derived from
// `window_aspect` (= window width / height). No-op if `items` is empty.
inline void push_abs_text(holoscan::TensorMap& out,
                          std::vector<PlotInputSpec>* specs,
                          const std::string& name,
                          const std::vector<AbsText>& items, float size_frac,
                          float window_aspect, const std::vector<float>& rgba,
                          int priority) {
  if (items.empty()) {
    return;
  }
  // Glyph advance as a fraction of window width: advance_px / W =
  // ratio * height_px / W = ratio * size_frac
  // * (H / W) = ratio*size_frac/aspect.
  const float char_w =
      kGlyphAdvanceRatio * size_frac / std::max(window_aspect, 1e-3F);
  std::vector<float> pos;
  std::vector<std::string> txt;
  for (const AbsText& it : items) {
    float x = it.x;
    const float w = char_w * static_cast<float>(it.s.size());
    if (it.anchor == TextAnchor::kCenter) {
      x -= 0.5F * w;
    } else if (it.anchor == TextAnchor::kRight) {
      x -= w;
    }
    pos.insert(pos.end(), {x, it.y, size_frac});
    txt.push_back(it.s);
  }
  out.insert({name, plot_geom_tensor(pos, 3)});
  auto s = plot_layer_spec(name, PlotInputType::TEXT, rgba,
                           PanelRect{0.0F, 0.0F, 1.0F, 1.0F}, priority);
  s.text_ = std::move(txt);
  specs->push_back(std::move(s));
}

// Append a filled triangle-fan disc (segments triangles) to a vertex list.
inline void plot_append_disc(std::vector<float>* v, float cx, float cy,
                             float rx, float ry, int segments) {
  const float two_pi = static_cast<float>(radar::kTwoPi);
  for (int k = 0; k < segments; ++k) {
    const float a0 = two_pi * static_cast<float>(k) / segments;
    const float a1 = two_pi * static_cast<float>(k + 1) / segments;
    v->push_back(cx);
    v->push_back(cy);
    v->push_back(cx + rx * std::cos(a0));
    v->push_back(cy + ry * std::sin(a0));
    v->push_back(cx + rx * std::cos(a1));
    v->push_back(cy + ry * std::sin(a1));
  }
}

// (PanelTick is declared near the top of this header.)
// A marker: data-space position ([0,1] within the plot box) + color bucket.
struct PanelMarker {
  float xf;
  float yf;
  int bucket;
};

// Emit one plot panel as native Holoviz layers into `out` (data tensors) and
// `specs` (matching InputSpecs), all placed in `rect`. Layers back-to-front:
// dark background, axis frame + gridlines + tick marks, a black outline
// under-layer + one colored disc layer per marker bucket, tick labels, title.
// Markers/ticks are in [0,1] box coordinates (y up); marker radius is the RD-
// map value rescaled by the panel height so a dot matches the RD-map dot size.
inline void emit_panel(holoscan::TensorMap& out,
                       std::vector<PlotInputSpec>* specs,
                       const std::string& prefix, const PanelRect& rect,
                       float aspect, const std::string& title,
                       const std::vector<PanelTick>& xticks,
                       const std::vector<PanelTick>& yticks,
                       const std::vector<PanelMarker>& markers,
                       float marker_radius, float marker_outline,
                       int marker_segments) {
  const int segments = marker_segments > 2 ? marker_segments : 3;
  const auto bx = [](float xf) { return kBoxL + xf * (kBoxR - kBoxL); };
  const auto by = [](float yf) { return kBoxB - yf * (kBoxB - kBoxT); };

  // Background.
  const std::vector<float> bg = {0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1};
  out.insert({prefix + "_bg", plot_geom_tensor(bg, 2)});
  specs->push_back(plot_layer_spec(prefix + "_bg", PlotInputType::TRIANGLES,
                                   {0.08F, 0.08F, 0.11F, 1.0F}, rect, 3));

  // Axis frame, gridlines at every tick, and short outward tick marks.
  {
    std::vector<float> lines;
    const auto seg = [&lines](float x0, float y0, float x1, float y1) {
      lines.insert(lines.end(), {x0, y0, x1, y1});
    };
    seg(kBoxL, kBoxT, kBoxR, kBoxT);
    seg(kBoxR, kBoxT, kBoxR, kBoxB);
    seg(kBoxR, kBoxB, kBoxL, kBoxB);
    seg(kBoxL, kBoxB, kBoxL, kBoxT);
    for (const PanelTick& t : yticks) {
      const float y = by(t.frac);
      seg(kBoxL, y, kBoxR, y);
      seg(kBoxL - 0.015F, y, kBoxL, y);
    }
    for (const PanelTick& t : xticks) {
      const float x = bx(t.frac);
      seg(x, kBoxT, x, kBoxB);
      seg(x, kBoxB, x, kBoxB + 0.02F);
    }
    out.insert({prefix + "_frame", plot_geom_tensor(lines, 2)});
    specs->push_back(plot_layer_spec(prefix + "_frame", PlotInputType::LINES,
                                     {0.42F, 0.42F, 0.52F, 1.0F}, rect, 4));
  }

  // Markers: black outline under-layer + one colored layer per bucket present.
  {
    const float ry = marker_radius / std::max(rect.h, 1e-6F);
    const float rx = ry / std::max(aspect, 1e-6F);
    const float ory =
        (marker_radius + marker_outline) / std::max(rect.h, 1e-6F);
    const float orx = ory / std::max(aspect, 1e-6F);
    std::vector<float> outline;
    std::vector<std::vector<float>> by_bucket(kNumClusterColors + 1);
    for (const PanelMarker& m : markers) {
      const float x = bx(m.xf), y = by(m.yf);
      if (marker_outline > 0.0F) {
        plot_append_disc(&outline, x, y, orx, ory, segments);
      }
      plot_append_disc(&by_bucket[static_cast<size_t>(m.bucket)], x, y, rx, ry,
                       segments);
    }
    if (!outline.empty()) {
      out.insert({prefix + "_moutline", plot_geom_tensor(outline, 2)});
      specs->push_back(plot_layer_spec(prefix + "_moutline",
                                       PlotInputType::TRIANGLES,
                                       {0.0F, 0.0F, 0.0F, 1.0F}, rect, 5));
    }
    for (int b = 0; b <= kNumClusterColors; ++b) {
      auto& verts = by_bucket[static_cast<size_t>(b)];
      if (verts.empty()) {
        continue;
      }
      const std::string name = prefix + "_m" + std::to_string(b);
      out.insert({name, plot_geom_tensor(verts, 2)});
      const auto c = plot_bucket_color(b);
      specs->push_back(plot_layer_spec(
          name, PlotInputType::TRIANGLES,
          {c[0] / 255.0F, c[1] / 255.0F, c[2] / 255.0F, 1.0F}, rect, 6));
    }
  }

  // Text (tick labels + title) is drawn in a full-window view via push_abs_text
  // so it is not stretched by this non-square panel view. Panel-local (lx, ly)
  // maps to window-absolute (rect.x + lx*rect.w, rect.y + ly*rect.h); the glyph
  // height as a window fraction is the panel-local size × rect.h.
  const float win_aspect = aspect * rect.h / std::max(rect.w, 1e-6F);
  const auto ax = [&](float lx) { return rect.x + lx * rect.w; };
  const auto ay = [&](float ly) { return rect.y + ly * rect.h; };

  // Tick labels: Y right-anchored just left of the box, X centered below it.
  {
    const float sz = 0.062F * rect.h;  // window-height fraction
    std::vector<AbsText> items;
    for (const PanelTick& t : yticks) {
      items.push_back({ax(kBoxL) - 0.006F, ay(by(t.frac)) - 0.5F * sz, t.label,
                       TextAnchor::kRight});
    }
    for (const PanelTick& t : xticks) {
      items.push_back(
          {ax(bx(t.frac)), ay(kBoxB) + 0.006F, t.label, TextAnchor::kCenter});
    }
    push_abs_text(out, specs, prefix + "_ticks", items, sz, win_aspect,
                  {0.78F, 0.78F, 0.82F, 1.0F}, 7);
  }

  // Title (top-left of the panel).
  push_abs_text(out, specs, prefix + "_title",
                {{ax(0.03F), ay(0.02F), title, TextAnchor::kLeft}},
                0.075F * rect.h, win_aspect, {1.0F, 1.0F, 1.0F, 1.0F}, 8);
}

// Append a scenario-clock overlay — the current simulated time (seconds) — to a
// dynamic InputSpec list (`specs`) and its geometry/position tensors to `out`.
// Rendered as a small dark plate + white label at the top-left of the whole
// window (a full-window view), over the RD map. Priorities sit above the panels
// (which top out at 8) so it always reads on top. The label/plate tensors ride
// the producing op's data output (→ receivers) while the specs travel the
// input_specs chain; the layer names must stay unique across the merged
// HolovizOp spec list.
inline void emit_sim_clock(holoscan::TensorMap& out,
                           std::vector<PlotInputSpec>* specs, double seconds,
                           float px_height) {
  const PanelRect full{0.0F, 0.0F, 1.0F, 1.0F};
  // Pixel-locked label size (see kClockPx)
  // so it stays crisp at any window size.
  const float sz = kClockPx / std::max(px_height, 1.0F);
  const float ty = 0.012F;  // text top, window-normalized
  // Dark plate (two triangles) sized to bracket the text, for legibility over
  // the bright turbo map.
  const float x0 = 0.006F, y0 = ty - 0.004F, x1 = 0.190F, y1 = ty + sz + 0.006F;
  const std::vector<float> bg = {x0, y0, x1, y0, x0, y1,
                                 x1, y0, x1, y1, x0, y1};
  out.insert({"sim_clock_bg", plot_geom_tensor(bg, 2)});
  specs->push_back(plot_layer_spec("sim_clock_bg", PlotInputType::TRIANGLES,
                                   {0.0F, 0.0F, 0.0F, 0.55F}, full, 9));
  // White time label; TEXT position is (x, y, size), size = fraction of the
  // view (window) height.
  out.insert({"sim_clock", plot_geom_tensor({0.014F, ty, sz}, 3)});
  auto label = plot_layer_spec("sim_clock", PlotInputType::TEXT,
                               {1.0F, 1.0F, 1.0F, 1.0F}, full, 10);
  label.text_ = {"t = " + plot_format_tick(static_cast<float>(seconds), 1) +
                 " s"};
  specs->push_back(std::move(label));
}

// RD-map axes chrome: a border frame + outward tick marks + tick labels + axis
// unit labels + a centered title, drawn in the margins around the inset RD
// image. All geometry is in the region's [0, 1] space (View = `region`), using
// the shared kBox sub-rect so the frame lands exactly on the image edges (the
// image View is that same sub-rect; see holoviz_layout.hpp). No gridlines are
// drawn — they would obscure the map. Text sizes are small fractions of the
// region height (which spans the full window), so they read like the panels'.
inline void emit_rd_axes(holoscan::TensorMap& out,
                         std::vector<PlotInputSpec>* specs,
                         const PanelRect& region,
                         const std::vector<PanelTick>& xticks,
                         const std::vector<PanelTick>& yticks,
                         const std::string& title, float px_height,
                         float box_aspect) {
  const auto bx = [](float xf) { return kBoxL + xf * (kBoxR - kBoxL); };
  const auto by = [](float yf) { return kBoxB - yf * (kBoxB - kBoxT); };
  // Text size as a fraction of the window height that renders to a fixed pixel
  // target (see kRd*Px), so it stays readable and near the 25 px font atlas.
  const float safe_h = std::max(px_height, 1.0F);
  const float tick_sz = kRdTickPx / safe_h;
  const float title_sz = kRdTitlePx / safe_h;
  // Window aspect (W/H), derived from the inset box's on-screen aspect: the box
  // is (kBoxR-kBoxL)*region.w wide and
  // (kBoxB-kBoxT)*region.h tall of the window,
  // so W/H = box_aspect * boxH_frac / boxW_frac.
  const float win_aspect = box_aspect * (kBoxB - kBoxT) * region.h /
                           std::max((kBoxR - kBoxL) * region.w, 1e-6F);
  // Panel-local (region [0,1]) → window-absolute.
  const auto ax = [&](float lx) { return region.x + lx * region.w; };
  const auto ay = [&](float ly) { return region.y + ly * region.h; };

  // Border frame + short outward tick marks (no interior gridlines). Geometry
  // stays in the region sub-view so it lands exactly on the image edges.
  {
    std::vector<float> lines;
    const auto seg = [&lines](float x0, float y0, float x1, float y1) {
      lines.insert(lines.end(), {x0, y0, x1, y1});
    };
    seg(kBoxL, kBoxT, kBoxR, kBoxT);
    seg(kBoxR, kBoxT, kBoxR, kBoxB);
    seg(kBoxR, kBoxB, kBoxL, kBoxB);
    seg(kBoxL, kBoxB, kBoxL, kBoxT);
    for (const PanelTick& t : yticks) {
      const float y = by(t.frac);
      seg(kBoxL - 0.012F, y, kBoxL, y);
    }
    for (const PanelTick& t : xticks) {
      const float x = bx(t.frac);
      seg(x, kBoxB, x, kBoxB + 0.012F);
    }
    out.insert({"rd_frame", plot_geom_tensor(lines, 2)});
    specs->push_back(plot_layer_spec("rd_frame", PlotInputType::LINES,
                                     {0.60F, 0.60F, 0.68F, 1.0F}, region, 4));
  }

  // Tick labels (full-window view, un-stretched): Y right-anchored just left of
  // the box, X centered below it.
  {
    std::vector<AbsText> items;
    for (const PanelTick& t : yticks) {
      items.push_back({ax(kBoxL) - 0.010F, ay(by(t.frac)) - 0.5F * tick_sz,
                       t.label, TextAnchor::kRight});
    }
    for (const PanelTick& t : xticks) {
      items.push_back(
          {ax(bx(t.frac)), ay(kBoxB) + 0.016F, t.label, TextAnchor::kCenter});
    }
    push_abs_text(out, specs, "rd_ticks", items, tick_sz, win_aspect,
                  {0.85F, 0.85F, 0.88F, 1.0F}, 7);
  }

  // Axis unit labels: "Range (m)" centered under the X ticks, "Velocity (m/s)"
  // left-aligned just above the box (Holoviz can't rotate text).
  push_abs_text(
      out, specs, "rd_axislabels",
      {{ax(0.5F * (kBoxL + kBoxR)), ay(kBoxB) + 0.016F + tick_sz + 0.006F,
        "Range (m)", TextAnchor::kCenter},
       {ax(kBoxL), ay(kBoxT) - tick_sz - 0.006F, "Velocity (m/s)",
        TextAnchor::kLeft}},
      tick_sz, win_aspect, {0.80F, 0.80F, 0.85F, 1.0F}, 7);

  // Title centered in the top margin.
  // Empty title → draw none, so an app that renders its own heading (e.g. one
  // carrying a live mode name) is not stacked on top of this one.
  if (!title.empty()) {
    push_abs_text(out, specs, "rd_title",
                  {{ax(0.5F * (kBoxL + kBoxR)), ay(kBoxT) - title_sz - 0.012F,
                    title, TextAnchor::kCenter}},
                  title_sz, win_aspect, {1.0F, 1.0F, 1.0F, 1.0F}, 8);
  }
}

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

  // Render the range/velocity panels with native Holoviz primitives (geometry +
  // dynamic InputSpecs on `plot_specs`) instead of raster images. Set by
  // compose() along with each panel's view rect and on-screen aspect (for round
  // markers). Off → the legacy rasterized images.
  bool primitive{false};
  PanelRect range_view{};
  PanelRect velocity_view{};
  float range_aspect{1.0F};
  float velocity_aspect{1.0F};

  // Marker geometry, shared with the RD-map overlay (set from cfg.viz). radius
  // and outline are RD-map values (fractions of the full-height RD view);
  // they are rescaled by the panel height below so a side-panel dot renders the
  // same on-screen size as an RD-map dot. The outline is drawn as one under-
  // layer beneath the colored discs, so the overlapping trail gets a clean dark
  // border rather than per-dot speckle (marker_outline = 0 disables it).
  float marker_radius{0.004F};
  float marker_outline{0.002F};
  int64_t marker_segments{20};

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
    // Primitive-mode InputSpec chain: upstream specs (e.g. the scenario clock
    // from ClusterOverlayOp) arrive on specs_in; this op prepends them to its
    // own panel specs and emits the combined list on plot_specs, fed to
    // HolovizOp's input_specs port (directly or via a boresight chain). Unused
    // in raster mode.
    spec.input<std::vector<PlotInputSpec>>("specs_in")
        .condition(holoscan::ConditionType::kNone);
    spec.output<std::vector<PlotInputSpec>>("plot_specs");
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

    // One operator stream ordered behind each producer whose data is read back
    // (see ClusterOverlayOp): a stream sync instead of a device-wide barrier.
    cudaStream_t stream = op_input.receive_cuda_stream("rep_idx");
    op_input.receive_cuda_stream("range_bins");
    op_input.receive_cuda_stream("velocity_bins");

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
          [s = stream](const std::shared_ptr<holoscan::Tensor>& t,
                       const char* name, std::vector<float>* out, float* lo,
                       float* hi) {
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
            if (cudaMemcpyAsync(out->data(), t->data(),
                                static_cast<size_t>(t->nbytes()),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess ||
                cudaStreamSynchronize(s) != cudaSuccess) {
              throw std::runtime_error(std::string("RollingPlotOp: ") + name +
                                       " D2H copy failed");
            }
            const auto mm = std::minmax_element(out->begin(), out->end());
            *lo = *mm.first;
            *hi = *mm.second;
          };
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
    if (num_clusters > 0 &&
        (cudaMemcpyAsync(rep.data(), rep_idx->data(),
                         static_cast<size_t>(rep_idx->nbytes()),
                         cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
         cudaStreamSynchronize(stream) != cudaSuccess)) {
      throw std::runtime_error("RollingPlotOp: rep_idx D2H failed");
    }

    // Clustered (labels present): color each rep by its cluster's range rank
    // (rep_color_ranks) — the same stable,
    // collision-free key ClusterOverlayOp /
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

    // Both panels as native Holoviz primitives (geometry on `plots`, specs on
    // `plot_specs`) or the legacy raster images + constant TEXT titles.
    holoscan::TensorMap out;
    if (primitive) {
      // Prepend any upstream specs (the scenario clock) so they merge into the
      // single input_specs stream ahead of this op's panel layers.
      std::vector<PlotInputSpec> specs;
      auto upstream = op_input.receive<std::vector<PlotInputSpec>>("specs_in");
      if (upstream) {
        specs = std::move(upstream.value());
      }
      render_panel_primitives(out, &specs, oldest, "range", /*use_range=*/true,
                              range_view, range_aspect,
                              "Range (m) vs time (s)");
      render_panel_primitives(out, &specs, oldest, "velocity",
                              /*use_range=*/false, velocity_view,
                              velocity_aspect, "Velocity (m/s) vs time (s)");
      op_output.emit(specs, "plot_specs");
    } else {
      out.insert({kRangePlotLayer, render_plot(/*use_range=*/true, oldest)});
      out.insert({kRangeTitleLayer, plot_title_pos()});
      out.insert(
          {kVelocityPlotLayer, render_plot(/*use_range=*/false, oldest)});
      out.insert({kVelocityTitleLayer, plot_title_pos()});
    }
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

  // Render one rolling panel (range or velocity vs time) as native Holoviz
  // primitives: build the axis ticks + marker list and hand off to emit_panel,
  // which fills `out` (data tensors) and `specs` (matching InputSpecs).
  void render_panel_primitives(holoscan::TensorMap& out,
                               std::vector<PlotInputSpec>* specs,
                               int64_t oldest,
                               const std::string& prefix, bool use_range,
                               const PanelRect& view, float aspect,
                               const std::string& title) {
    const float lo = use_range ? r_lo_ : v_lo_;
    const float hi = use_range ? r_hi_ : v_hi_;
    const float span = std::max(hi - lo, 1e-6F);
    const int y_dec = span < 10.0F ? 1 : 0;

    std::vector<PanelTick> yticks, xticks;
    for (int i = 0; i < kTicks; ++i) {
      const float f = static_cast<float>(i) / (kTicks - 1);
      yticks.push_back({f, plot_format_tick(lo + f * span, y_dec)});
      const float secs = -(1.0F - f) * static_cast<float>(window - 1) *
                         static_cast<float>(frame_period);
      xticks.push_back({f, plot_format_tick(secs, 0)});
    }

    std::vector<PanelMarker> markers;
    markers.reserve(cen_hist_.size());
    for (const Sample& s : cen_hist_) {
      const float xf = static_cast<float>(s.t - oldest) /
                       static_cast<float>(window - 1);
      const float yf =
          std::clamp(((use_range ? s.r : s.v) - lo) / span, 0.0F, 1.0F);
      markers.push_back({xf, yf, s.bucket});
    }

    emit_panel(out, specs, prefix, view, aspect, title, xticks, yticks, markers,
               marker_radius, marker_outline,
               static_cast<int>(marker_segments));
  }

  static constexpr int kTicks = 5;

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

  // Render the panel with native Holoviz primitives (geometry on `plot`, specs
  // chained through `specs_in` → `specs_out`) instead of a raster image. Set by
  // compose() with the panel view rect, on-screen aspect, and marker geometry.
  bool primitive{false};
  PanelRect view{};
  float aspect{1.0F};
  float marker_radius{0.004F};
  float marker_outline{0.002F};
  int64_t marker_segments{20};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("aoa_estimates");
    // Optional: cluster ids (row-aligned with aoa_estimates). Present →
    // clustered mode: markers color by their cluster. Absent → peak_only: each
    // marker colors by its own row index.
    spec.input<holoscan::TensorMap>("labels")
        .condition(holoscan::ConditionType::kNone);
    // Optional: per-cluster representative RD cell (num_clusters, 2), row c =
    // cluster label c. When present with labels, clusters color by their rep
    // cell (stable across frames) — the same
    // key the RD map / rolling plots use.
    spec.input<holoscan::TensorMap>("rep_idx")
        .condition(holoscan::ConditionType::kNone);
    spec.output<holoscan::TensorMap>("plot");
    // Primitive-mode InputSpec chain: upstream panel specs (e.g. the rolling
    // plots) arrive on specs_in; this op appends its own and emits the combined
    // list on specs_out → HolovizOp's input_specs port.
    spec.input<std::vector<PlotInputSpec>>("specs_in")
        .condition(holoscan::ConditionType::kNone);
    spec.output<std::vector<PlotInputSpec>>("specs_out");
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

    // One operator stream ordered behind each producer whose data is read back
    // (see ClusterOverlayOp): a stream sync instead of a device-wide barrier.
    cudaStream_t stream = op_input.receive_cuda_stream("aoa_estimates");
    if (have_labels) {
      op_input.receive_cuda_stream("labels");
    }
    if (rep_idx) {
      op_input.receive_cuda_stream("rep_idx");
    }

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
    if (n > 0) {
      if (cudaMemcpyAsync(ang.data(), aoa->data(),
                          static_cast<size_t>(aoa->nbytes()),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        throw std::runtime_error("BoresightViewOp: aoa_estimates D2H failed");
      }
      if (have_labels &&
          cudaMemcpyAsync(lab.data(), labels->data(),
                          static_cast<size_t>(labels->nbytes()),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        throw std::runtime_error("BoresightViewOp: labels D2H failed");
      }
    }
    if (num_rep > 0 &&
        cudaMemcpyAsync(rep.data(), rep_idx->data(),
                        sizeof(int64_t) * rep.size(), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
      throw std::runtime_error("BoresightViewOp: rep_idx D2H failed");
    }
    // One wait covering every copy above.
    if ((n > 0 || num_rep > 0) &&
        cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("BoresightViewOp: stream sync failed");
    }
    const bool have_rep = rep_idx && num_rep > 0;
    const std::vector<int> ranks =
        have_rep ? rep_color_ranks(rep, num_rep) : std::vector<int>();

    // Per-detection color bucket. Clustered (labels present): color by the
    // cluster's range rank when rep cells are available (stable, collision-
    // free), else the size-rank label; noise
    // (-1) → gray. peak_only (no labels):
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
    if (primitive) {
      // Angular extents in degrees; markers map az→x, el→y over the fixed span.
      constexpr double kRad2Deg = 57.29577951308232;
      const float ax_lo = static_cast<float>(az_lo * kRad2Deg);
      const float ax_hi = static_cast<float>(az_hi * kRad2Deg);
      const float ey_lo = static_cast<float>(el_lo * kRad2Deg);
      const float ey_hi = static_cast<float>(el_hi * kRad2Deg);
      const float ax_span = std::max(ax_hi - ax_lo, 1e-6F);
      const float ey_span = std::max(ey_hi - ey_lo, 1e-6F);
      std::vector<PanelTick> xticks, yticks;
      for (int i = 0; i < 5; ++i) {
        const float f = static_cast<float>(i) / 4.0F;
        xticks.push_back({f, plot_format_tick(ax_lo + f * ax_span, 0)});
        yticks.push_back({f, plot_format_tick(ey_lo + f * ey_span, 0)});
      }
      std::vector<PanelMarker> markers;
      markers.reserve(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i) {
        const size_t ai = static_cast<size_t>(i * 2);
        const float xf =
            (static_cast<float>(ang[ai] * kRad2Deg) - ax_lo) / ax_span;
        const float yf =
            (static_cast<float>(ang[ai + 1] * kRad2Deg) - ey_lo) / ey_span;
        if (xf < 0.0F || xf > 1.0F || yf < 0.0F || yf > 1.0F) {
          continue;  // outside the field of view
        }
        markers.push_back({xf, yf, bucket[static_cast<size_t>(i)]});
      }
      std::vector<PlotInputSpec> specs;
      auto upstream = op_input.receive<std::vector<PlotInputSpec>>("specs_in");
      if (upstream) {
        specs = std::move(upstream.value());
      }
      emit_panel(out, &specs, "boresight", view, aspect,
                 "Boresight az/el (deg)",
                 xticks, yticks, markers, marker_radius, marker_outline,
                 static_cast<int>(marker_segments));
      op_output.emit(specs, "specs_out");
    } else {
      out.insert({kBoresightPlotLayer, render(ang, bucket, n)});
      out.insert({kBoresightTitleLayer, plot_title_pos()});
    }
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
