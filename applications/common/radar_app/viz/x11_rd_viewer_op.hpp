// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// [EA HSDK5 ONLY] Interim live RD-map window via Xlib when Holoviz is absent.
// Presents a scaled turbo range-Doppler image in a configurable-size window
// with the information overlays the Holoviz demo drew: framed axes with
// physical range / velocity ticks, CFAR detections as dots colored by DBSCAN
// cluster (noise gray) with their reduction representative marked, a cluster
// legend, a scenario clock / detection counters / active-variant name, and
// optional side panels:
// rolling range-vs-time and velocity-vs-time scatters plus an angular
// boresight (az/el) scatter for the angle-estimating apps.
//
// Overlay inputs are wired per config: on EA a registered-but-unwired port
// must stay out of contract(), and a wired port must be in it
// (TEMPORAL_INPUT_UNCONNECTED / RUNTIME_OPERATOR_REPLAY_MISMATCH otherwise —
// HSDK5_PORT.md §1), so the contract is built from the same flags the app
// wires. EA-quality: presents on the Holoscan compute thread.

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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

#include "radar_app/viz/colormap.hpp"
#include "radar_app/viz/overlay_style.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app {

struct X11RdViewerConfig {
  int width{1280};
  int height{720};
  // Drawn with XDrawString (8-bit Latin-1): keep titles ASCII, or non-ASCII
  // punctuation renders as mojibake.
  std::string title{"Range-Doppler"};
  // Wire detection_idx + range_bins + velocity_bins (detection dots + physical
  // axis ticks). The final two index columns are (doppler, range), so sparse
  // (N, 2) and dense (N, 4) detections are both accepted. Without it the
  // viewer shows the raster and bin-index ticks.
  bool overlay_detections{false};
  // Additionally wire labels + rep_idx (per-cluster colors, representative
  // markers, legend). Requires overlay_detections; off in CFAR peak_only mode,
  // where there is no clustering stage.
  bool overlay_clusters{false};
  // Rolling range-vs-time / velocity-vs-time panels right of the RD map.
  bool plots_enabled{false};
  std::int64_t plot_window{120};
  double frame_period{0.1};
  // Boresight (angular field-of-view) scatter panel: each detection at its
  // (az, el) estimate. Requires the aoa_estimates port to be wired, rows
  // aligned with detection_idx. The final two columns are used, admitting
  // either (N, 2) estimator output or (N, 4) physical feature points. Angular
  // extents are radians (the estimator's scan grid), labeled in degrees.
  bool overlay_boresight{false};
  double az_lo{-0.35};
  double az_hi{0.35};
  double el_lo{-0.1};
  double el_hi{0.1};
  // Detection dot radius / outline ring, as a fraction of the RD map width.
  float marker_radius{0.006F};
  float marker_outline{0.002F};
  // Named-variant label in the header, for apps that switch between algorithm
  // variants at run time (fmcw_aoa_estimators). Requires the label_idx port to
  // be wired: an int64 (1,) index into label_names.
  bool overlay_label{false};
  std::string label_caption{"variant"};
  std::vector<std::string> label_names{};
};

class X11RdViewerOp : public holoscan::Operator<> {
 public:
  X11RdViewerOp() = default;
  explicit X11RdViewerOp(X11RdViewerConfig config) : cfg_(std::move(config)) {
    if (cfg_.overlay_clusters && !cfg_.overlay_detections) {
      throw std::invalid_argument(
          "X11RdViewerOp: overlay_clusters requires overlay_detections");
    }
  }

  ~X11RdViewerOp() override { shutdown(); }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(rd_image).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(detection_idx)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(range_bins).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(velocity_bins)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(labels).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(rep_idx).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(aoa_estimates)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(label_idx).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    std::vector<holoscan::PortRef> refs;
    refs.emplace_back(rd_image);
    if (cfg_.overlay_detections) {
      refs.emplace_back(detection_idx);
      refs.emplace_back(range_bins);
      refs.emplace_back(velocity_bins);
    }
    if (cfg_.overlay_clusters) {
      refs.emplace_back(labels);
      refs.emplace_back(rep_idx);
    }
    if (cfg_.overlay_boresight) {
      refs.emplace_back(aoa_estimates);
    }
    if (cfg_.overlay_label) {
      refs.emplace_back(label_idx);
    }
    if (refs.size() < 2U) {
      result.trigger(holoscan::OnEach{rd_image});
      return result;
    }
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::move(refs)},
        .window = std::chrono::nanoseconds{0}});
    result.input(rd_image, holoscan::Exact{});
    if (cfg_.overlay_detections) {
      result.input(detection_idx, holoscan::Exact{});
      result.input(range_bins, holoscan::Exact{});
      result.input(velocity_bins, holoscan::Exact{});
    }
    if (cfg_.overlay_clusters) {
      result.input(labels, holoscan::Exact{});
      result.input(rep_idx, holoscan::Exact{});
    }
    if (cfg_.overlay_boresight) {
      result.input(aoa_estimates, holoscan::Exact{});
    }
    if (cfg_.overlay_label) {
      result.input(label_idx, holoscan::Exact{});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (closed_) {
      return {};
    }
    const char* who = "X11RdViewerOp";

    HOLOSCAN_TRY(auto image, receive_tensor(rd_image, who));
    const auto& shape = image.shape();
    if (shape.size() != 3 || shape[2] != 1) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "X11RdViewerOp: expected (doppler, range, 1) float image"});
    }
    if (!dldatatype_equals(image.dtype(), real_dtype<float>())) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "X11RdViewerOp: expected float32 LUT indices"});
    }
    num_doppler_ = shape[0];
    num_range_ = shape[1];
    if (num_doppler_ <= 0 || num_range_ <= 0) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "X11RdViewerOp: empty image extents"});
    }

    cudaStream_t stream = context.cuda_stream().get();
    const auto copy_trailing_pair =
        [&](const char* name, const holoscan::Tensor& tensor,
            auto* destination) -> holoscan::expected<void, holoscan::Error> {
      using Value = typename std::remove_reference_t<
          decltype(*destination)>::value_type;
      const auto shape = tensor.shape();
      if (shape.size() != 2 || shape[1] < 2) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            std::string("X11RdViewerOp: ") + name + " must be (N, K>=2)"});
      }
      const std::size_t rows = static_cast<std::size_t>(shape[0]);
      const std::size_t columns = static_cast<std::size_t>(shape[1]);
      std::vector<Value> full(rows * columns);
      HOLOSCAN_RETURN_IF_ERROR(copy_d2h(
          name, full.data(), tensor, full.size() * sizeof(Value), stream));
      destination->resize(rows * 2);
      for (std::size_t row = 0; row < rows; ++row) {
        (*destination)[row * 2] = full[row * columns + columns - 2];
        (*destination)[row * 2 + 1] =
            full[row * columns + columns - 1];
      }
      return {};
    };
    host_lut_.resize(static_cast<std::size_t>(num_doppler_ * num_range_));
    HOLOSCAN_RETURN_IF_ERROR(copy_d2h("rd_image", host_lut_.data(), image,
                                      host_lut_.size() * sizeof(float),
                                      stream));

    detections_.clear();
    labels_.clear();
    rep_.clear();
    ranges_.clear();
    velocities_.clear();
    if (cfg_.overlay_detections) {
      HOLOSCAN_TRY(auto idx_t, receive_tensor(detection_idx, who));
      HOLOSCAN_TRY(auto range_t, receive_tensor(range_bins, who));
      HOLOSCAN_TRY(auto velocity_t, receive_tensor(velocity_bins, who));
      const auto idx_shape = idx_t.shape();
      ranges_.resize(static_cast<std::size_t>(range_t.shape()[0]));
      velocities_.resize(static_cast<std::size_t>(velocity_t.shape()[0]));
      HOLOSCAN_RETURN_IF_ERROR(
          copy_trailing_pair("detection_idx", idx_t, &detections_));
      HOLOSCAN_RETURN_IF_ERROR(
          copy_d2h("range_bins", ranges_.data(), range_t,
                   ranges_.size() * sizeof(float), stream));
      HOLOSCAN_RETURN_IF_ERROR(
          copy_d2h("velocity_bins", velocities_.data(), velocity_t,
                   velocities_.size() * sizeof(float), stream));
    }
    if (cfg_.overlay_clusters) {
      HOLOSCAN_TRY(auto labels_t, receive_tensor(labels, who));
      HOLOSCAN_TRY(auto rep_t, receive_tensor(rep_idx, who));
      labels_.resize(static_cast<std::size_t>(labels_t.shape()[0]));
      HOLOSCAN_RETURN_IF_ERROR(
          copy_d2h("labels", labels_.data(), labels_t,
                   labels_.size() * sizeof(std::int64_t), stream));
      HOLOSCAN_RETURN_IF_ERROR(
          copy_trailing_pair("rep_idx", rep_t, &rep_));
    }
    angles_.clear();
    if (cfg_.overlay_boresight) {
      HOLOSCAN_TRY(auto aoa_t, receive_tensor(aoa_estimates, who));
      const auto aoa_shape = aoa_t.shape();
      HOLOSCAN_RETURN_IF_ERROR(
          copy_trailing_pair("aoa_estimates", aoa_t, &angles_));
      if (cfg_.overlay_detections &&
          aoa_shape[0] != static_cast<std::int64_t>(detections_.size() / 2)) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            "X11RdViewerOp: aoa_estimates rows must match detection_idx"});
      }
    }
    if (cfg_.overlay_label) {
      HOLOSCAN_TRY(auto label_t, receive_tensor(label_idx, who));
      HOLOSCAN_RETURN_IF_ERROR(copy_d2h("label_idx", &label_index_, label_t,
                                        sizeof(std::int64_t), stream));
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          "X11RdViewerOp: cudaStreamSynchronize failed"});
    }

    try {
      if (!ensure_window()) {
        closed_ = true;
        return {};
      }
      if (cfg_.overlay_clusters) {
        const auto clusters = static_cast<std::int64_t>(rep_.size() / 2);
        rank_order_ = overlay_rep_rank_order(rep_, clusters);
        buckets_ = overlay_rep_color_ranks(rep_, clusters);
      } else {
        rank_order_.clear();
        buckets_.clear();
      }
      record_history();
      render();
      pump_events();
      ++frame_index_;
      ++presented_count_;
    } catch (const std::exception& e) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("X11RdViewerOp: ") + e.what()});
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> rd_image{"rd_image"};
  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Input<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Input<holoscan::Tensor> velocity_bins{"velocity_bins"};
  holoscan::Input<holoscan::Tensor> labels{"labels"};
  holoscan::Input<holoscan::Tensor> rep_idx{"rep_idx"};
  holoscan::Input<holoscan::Tensor> aoa_estimates{"aoa_estimates"};
  holoscan::Input<holoscan::Tensor> label_idx{"label_idx"};

  // Successful present count for unit smoke (not gated on DISPLAY probe).
  static std::int64_t presented_count() { return presented_count_; }
  static void reset_presented_count() { presented_count_ = 0; }

 private:
  struct Rect {
    int x{0};
    int y{0};
    int w{0};
    int h{0};
  };

  struct PlotPoint {
    float range{0.0F};
    float velocity{0.0F};
    int bucket{-1};
  };

  struct PlotFrame {
    double t{0.0};
    std::vector<PlotPoint> points;  // every detection
    std::vector<PlotPoint> reps;    // per-cluster reduction representative
  };

  enum class Anchor { kLeft, kCenter, kRight };

  static holoscan::expected<void, holoscan::Error> copy_d2h(
      const char* name, void* dst, const holoscan::Tensor& src,
      std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0 || src.nbytes() == 0) {
      return {};
    }
    if (static_cast<std::size_t>(src.nbytes()) != bytes) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string("X11RdViewerOp: ") + name + " byte size mismatch"});
    }
    if (cudaMemcpyAsync(dst, src.data(), bytes, cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("X11RdViewerOp: ") + name + " D2H copy failed"});
    }
    return {};
  }

  static std::uint32_t pack(OverlayRgb color) {
    // TrueColor 24/32-bit visual packing (LSB blue on little-endian X11).
    return static_cast<std::uint32_t>(color.b) |
           (static_cast<std::uint32_t>(color.g) << 8) |
           (static_cast<std::uint32_t>(color.r) << 16);
  }

  bool ensure_window() {
    if (display_ != nullptr) {
      return true;
    }
    static bool threads_inited = false;
    if (!threads_inited) {
      XInitThreads();
      threads_inited = true;
    }
    display_ = XOpenDisplay(nullptr);
    if (display_ == nullptr) {
      HOLOSCAN_LOG_WARN(
          "X11RdViewerOp: XOpenDisplay failed; disabling viewer");
      return false;
    }

    const int screen = DefaultScreen(display_);
    visual_ = DefaultVisual(display_, screen);
    depth_ = DefaultDepth(display_, screen);
    win_w_ = std::max(cfg_.width, 480);
    win_h_ = std::max(cfg_.height, 320);
    window_ = XCreateSimpleWindow(
        display_, RootWindow(display_, screen), 0, 0,
        static_cast<unsigned>(win_w_), static_cast<unsigned>(win_h_), 0,
        BlackPixel(display_, screen), pack(kBackground));
    XStoreName(display_, window_, cfg_.title.c_str());
    XSelectInput(display_, window_,
                 StructureNotifyMask | ExposureMask | KeyPressMask);
    delete_atom_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display_, window_, &delete_atom_, 1);
    XMapWindow(display_, window_);
    XFlush(display_);

    gc_ = XCreateGC(display_, window_, 0, nullptr);
    for (const char* name : {"9x15", "8x13", "fixed"}) {
      font_ = XLoadQueryFont(display_, name);
      if (font_ != nullptr) {
        break;
      }
    }
    if (font_ != nullptr) {
      XSetFont(display_, gc_, font_->fid);
    } else {
      HOLOSCAN_LOG_WARN(
          "X11RdViewerOp: no X11 font available; drawing without text");
    }
    if (turbo_pixels_.empty()) {
      for (const auto& rgba : make_turbo_lut()) {
        const auto to_byte = [](float channel) {
          return static_cast<std::uint8_t>(
              std::clamp(channel, 0.0F, 1.0F) * 255.0F + 0.5F);
        };
        turbo_pixels_.push_back(pack(OverlayRgb{
            to_byte(rgba[0]), to_byte(rgba[1]), to_byte(rgba[2])}));
      }
    }
    if (palette_.empty()) {
      palette_ = overlay_cluster_palette();
    }
    ensure_backbuffer();
    return true;
  }

  void ensure_backbuffer() {
    if (back_ != 0 && back_w_ == win_w_ && back_h_ == win_h_) {
      return;
    }
    if (back_ != 0) {
      XFreePixmap(display_, back_);
    }
    back_ = XCreatePixmap(display_, window_, static_cast<unsigned>(win_w_),
                          static_cast<unsigned>(win_h_),
                          static_cast<unsigned>(depth_));
    back_w_ = win_w_;
    back_h_ = win_h_;
  }

  void ensure_raster(int width, int height) {
    if (ximage_ != nullptr && raster_w_ == width && raster_h_ == height) {
      return;
    }
    destroy_raster();
    pixels_.assign(static_cast<std::size_t>(width) *
                       static_cast<std::size_t>(height),
                   0U);
    ximage_ = XCreateImage(display_, visual_, static_cast<unsigned>(depth_),
                           ZPixmap, 0,
                           reinterpret_cast<char*>(pixels_.data()),
                           static_cast<unsigned>(width),
                           static_cast<unsigned>(height), 32, 0);
    if (ximage_ == nullptr) {
      throw std::runtime_error("XCreateImage failed");
    }
    raster_w_ = width;
    raster_h_ = height;
  }

  // Layout: RD map left (axes in the margins), optional stacked plot panels
  // right. Margins hold the tick labels, the header, and the legend strip.
  int panel_count() const {
    return (cfg_.plots_enabled ? 2 : 0) + (cfg_.overlay_boresight ? 1 : 0);
  }

  Rect rd_rect() const {
    const int content_w = win_w_ - kMarginLeft - kMarginRight;
    const int rd_w = panel_count() > 0
                         ? static_cast<int>(static_cast<float>(content_w) *
                                            0.58F) - kPanelGap / 2
                         : content_w;
    return Rect{kMarginLeft, kMarginTop, std::max(rd_w, 64),
                std::max(win_h_ - kMarginTop - kMarginBottom, 64)};
  }

  // Panels stack top-to-bottom in the right column, splitting the RD height.
  Rect panel_rect(int index) const {
    const Rect rd = rd_rect();
    const int count = std::max(panel_count(), 1);
    const int x = rd.x + rd.w + kPanelGap;
    const int w = std::max(win_w_ - kMarginRight - x, 64);
    const int h = std::max((rd.h - (count - 1) * kPanelSplit) / count, 48);
    return Rect{x, rd.y + index * (h + kPanelSplit), w, h};
  }

  void set_color(OverlayRgb color) {
    XSetForeground(display_, gc_, pack(color));
  }

  int text_width(const std::string& text) const {
    if (font_ == nullptr) {
      return 0;
    }
    return XTextWidth(font_, text.c_str(), static_cast<int>(text.size()));
  }

  void draw_text(int x, int y, const std::string& text, OverlayRgb color,
                 Anchor anchor = Anchor::kLeft) {
    if (font_ == nullptr || text.empty()) {
      return;
    }
    int left = x;
    if (anchor == Anchor::kCenter) {
      left = x - text_width(text) / 2;
    } else if (anchor == Anchor::kRight) {
      left = x - text_width(text);
    }
    set_color(color);
    XDrawString(display_, back_, gc_, left, y, text.c_str(),
                static_cast<int>(text.size()));
  }

  void fill_rect(const Rect& r, OverlayRgb color) {
    set_color(color);
    XFillRectangle(display_, back_, gc_, r.x, r.y,
                   static_cast<unsigned>(r.w), static_cast<unsigned>(r.h));
  }

  void frame_rect(const Rect& r, OverlayRgb color) {
    set_color(color);
    XDrawRectangle(display_, back_, gc_, r.x, r.y,
                   static_cast<unsigned>(r.w), static_cast<unsigned>(r.h));
  }

  void fill_disc(int cx, int cy, int radius, OverlayRgb color) {
    if (radius <= 0) {
      return;
    }
    set_color(color);
    XFillArc(display_, back_, gc_, cx - radius, cy - radius,
             static_cast<unsigned>(2 * radius),
             static_cast<unsigned>(2 * radius), 0, 360 * 64);
  }

  // Physical value at a fractional position along a bin axis (empty bins →
  // the bin index itself, so the chrome still reads without the axis tensors).
  static double axis_value(const std::vector<float>& bins, double fraction,
                           std::int64_t num_bins) {
    const double index = fraction * static_cast<double>(num_bins - 1);
    if (bins.empty()) {
      return index;
    }
    const std::size_t i = static_cast<std::size_t>(
        std::clamp<double>(std::lround(index), 0.0,
                           static_cast<double>(bins.size() - 1)));
    return bins[i];
  }

  void record_history() {
    if (!cfg_.plots_enabled || !cfg_.overlay_detections) {
      return;
    }
    PlotFrame frame;
    frame.t = static_cast<double>(frame_index_) * cfg_.frame_period;
    const std::size_t n = detections_.size() / 2;
    frame.points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::int64_t dd = detections_[i * 2 + 0];
      const std::int64_t rr = detections_[i * 2 + 1];
      if (dd < 0 || rr < 0 ||
          static_cast<std::size_t>(rr) >= ranges_.size() ||
          static_cast<std::size_t>(dd) >= velocities_.size()) {
        continue;
      }
      frame.points.push_back(
          PlotPoint{ranges_[static_cast<std::size_t>(rr)],
                    velocities_[static_cast<std::size_t>(dd)],
                    bucket_for_point(i)});
    }
    const std::size_t clusters = rep_.size() / 2;
    frame.reps.reserve(clusters);
    for (std::size_t c = 0; c < clusters; ++c) {
      const std::int64_t dd = rep_[c * 2 + 0];
      const std::int64_t rr = rep_[c * 2 + 1];
      if (dd < 0 || rr < 0 ||
          static_cast<std::size_t>(rr) >= ranges_.size() ||
          static_cast<std::size_t>(dd) >= velocities_.size()) {
        continue;
      }
      frame.reps.push_back(PlotPoint{
          ranges_[static_cast<std::size_t>(rr)],
          velocities_[static_cast<std::size_t>(dd)],
          buckets_.empty() ? -1 : buckets_[c % buckets_.size()]});
    }
    history_.push_back(std::move(frame));
    while (static_cast<std::int64_t>(history_.size()) > cfg_.plot_window) {
      history_.pop_front();
    }
  }

  // Palette bucket for detection row i: its cluster's range rank, or -1 for
  // noise / an unclustered run (colored gray).
  int bucket_for_point(std::size_t i) const {
    if (!cfg_.overlay_clusters || i >= labels_.size()) {
      return -1;
    }
    const std::int64_t label = labels_[i];
    if (label < 0 || static_cast<std::size_t>(label) >= buckets_.size()) {
      return -1;
    }
    return buckets_[static_cast<std::size_t>(label)];
  }

  OverlayRgb bucket_color(int bucket) const {
    if (bucket < 0 || palette_.empty()) {
      return kOverlayNoiseColor;
    }
    return palette_[static_cast<std::size_t>(bucket) % palette_.size()];
  }

  void render() {
    fill_rect(Rect{0, 0, win_w_, win_h_}, kBackground);
    const Rect rd = rd_rect();
    draw_rd_raster(rd);
    draw_rd_axes(rd);
    draw_detections(rd);
    draw_header();
    draw_legend(rd);
    int panel = 0;
    if (cfg_.plots_enabled) {
      draw_panel(panel_rect(panel++), true);
      draw_panel(panel_rect(panel++), false);
    }
    if (cfg_.overlay_boresight) {
      draw_boresight(panel_rect(panel));
    }
    XCopyArea(display_, back_, window_, gc_, 0, 0,
              static_cast<unsigned>(win_w_), static_cast<unsigned>(win_h_), 0,
              0);
    XFlush(display_);
  }

  void draw_rd_raster(const Rect& rd) {
    ensure_raster(rd.w, rd.h);
    // Nearest-neighbor upscale. Row 0 is Doppler bin 0 at the top, matching
    // RdMapToImageOp's (doppler, range) layout and X11's top-left origin.
    for (int py = 0; py < rd.h; ++py) {
      const std::int64_t d = std::min<std::int64_t>(
          num_doppler_ - 1,
          static_cast<std::int64_t>(py) * num_doppler_ / rd.h);
      std::uint32_t* row =
          pixels_.data() + static_cast<std::size_t>(py) *
                               static_cast<std::size_t>(rd.w);
      for (int px = 0; px < rd.w; ++px) {
        const std::int64_t r = std::min<std::int64_t>(
            num_range_ - 1,
            static_cast<std::int64_t>(px) * num_range_ / rd.w);
        const float value =
            host_lut_[static_cast<std::size_t>(d * num_range_ + r)];
        const int lut_index = std::clamp(static_cast<int>(std::lround(value)),
                                         0, static_cast<int>(kLutSize) - 1);
        row[px] = turbo_pixels_[static_cast<std::size_t>(lut_index)];
      }
    }
    XPutImage(display_, back_, gc_, ximage_, 0, 0, rd.x, rd.y,
              static_cast<unsigned>(rd.w), static_cast<unsigned>(rd.h));
  }

  void draw_rd_axes(const Rect& rd) {
    frame_rect(rd, kChrome);
    for (int i = 0; i <= kRdXTicks; ++i) {
      const double f = static_cast<double>(i) / kRdXTicks;
      const int x = rd.x + static_cast<int>(f * rd.w);
      set_color(kChrome);
      XDrawLine(display_, back_, gc_, x, rd.y + rd.h, x, rd.y + rd.h + 5);
      draw_text(x, rd.y + rd.h + 19,
                overlay_format(axis_value(ranges_, f, num_range_), 0), kLabel,
                Anchor::kCenter);
    }
    for (int i = 0; i <= kRdYTicks; ++i) {
      const double f = static_cast<double>(i) / kRdYTicks;
      const int y = rd.y + static_cast<int>(f * rd.h);
      set_color(kChrome);
      XDrawLine(display_, back_, gc_, rd.x - 5, y, rd.x, y);
      draw_text(rd.x - 8, y + 4,
                overlay_format_signed(axis_value(velocities_, f, num_doppler_),
                                      1),
                kLabel, Anchor::kRight);
    }
    draw_text(rd.x + rd.w / 2, rd.y + rd.h + 36,
              ranges_.empty() ? "range bin" : "range (m)", kLabel,
              Anchor::kCenter);
    draw_text(rd.x, rd.y - 8,
              velocities_.empty() ? "velocity bin (down)" : "velocity (m/s)",
              kLabel);
  }

  void draw_detections(const Rect& rd) {
    if (!cfg_.overlay_detections) {
      return;
    }
    // Pixel floors: a 1-2 px XFillArc leaves gaps, so a marker scaled down by
    // a small window would read as a ring around the cell it covers.
    const int radius = std::max(
        3, static_cast<int>(cfg_.marker_radius * static_cast<float>(rd.w)));
    const int outline = std::max(
        2, static_cast<int>(cfg_.marker_outline * static_cast<float>(rd.w)));
    const std::size_t n = detections_.size() / 2;
    for (std::size_t i = 0; i < n; ++i) {
      const std::int64_t dd = detections_[i * 2 + 0];
      const std::int64_t rr = detections_[i * 2 + 1];
      if (dd < 0 || dd >= num_doppler_ || rr < 0 || rr >= num_range_) {
        continue;
      }
      const int cx = cell_x(rd, rr);
      const int cy = cell_y(rd, dd);
      fill_disc(cx, cy, radius + outline, kOutline);
      fill_disc(cx, cy, radius, bucket_color(bucket_for_point(i)));
    }
    // Reduction representative per cluster: a crosshair in the cluster color.
    const std::size_t clusters = rep_.size() / 2;
    XSetLineAttributes(display_, gc_, 2, LineSolid, CapButt, JoinMiter);
    for (std::size_t c = 0; c < clusters; ++c) {
      const std::int64_t dd = rep_[c * 2 + 0];
      const std::int64_t rr = rep_[c * 2 + 1];
      if (dd < 0 || dd >= num_doppler_ || rr < 0 || rr >= num_range_) {
        continue;
      }
      const int cx = cell_x(rd, rr);
      const int cy = cell_y(rd, dd);
      const int arm = radius + outline + 6;
      set_color(kOutline);
      XDrawArc(display_, back_, gc_, cx - arm, cy - arm,
               static_cast<unsigned>(2 * arm), static_cast<unsigned>(2 * arm),
               0, 360 * 64);
      set_color(bucket_color(
          buckets_.empty() ? -1 : buckets_[c % buckets_.size()]));
      XDrawLine(display_, back_, gc_, cx - arm, cy, cx + arm, cy);
      XDrawLine(display_, back_, gc_, cx, cy - arm, cx, cy + arm);
    }
    XSetLineAttributes(display_, gc_, 1, LineSolid, CapButt, JoinMiter);
  }

  int cell_x(const Rect& rd, std::int64_t range_bin) const {
    return rd.x + static_cast<int>((static_cast<double>(range_bin) + 0.5) *
                                   rd.w / static_cast<double>(num_range_));
  }

  int cell_y(const Rect& rd, std::int64_t doppler_bin) const {
    return rd.y + static_cast<int>((static_cast<double>(doppler_bin) + 0.5) *
                                   rd.h / static_cast<double>(num_doppler_));
  }

  // Right-aligned counters over a monospace font: pad every field so a digit
  // gained or lost does not shift the whole line sideways each frame.
  static std::string pad_left(const std::string& text, int width) {
    const int fill = width - static_cast<int>(text.size());
    return fill > 0 ? std::string(static_cast<std::size_t>(fill), ' ') + text
                    : text;
  }

  static std::string pad_right(const std::string& text, int width) {
    const int fill = width - static_cast<int>(text.size());
    return fill > 0 ? text + std::string(static_cast<std::size_t>(fill), ' ')
                    : text;
  }

  // Name of the variant the producer reported this frame; padded to the widest
  // configured name so the rest of the right-aligned status stays put.
  std::string label_field() const {
    const auto count = static_cast<std::int64_t>(cfg_.label_names.size());
    const std::string name =
        (label_index_ >= 0 && label_index_ < count)
            ? cfg_.label_names[static_cast<std::size_t>(label_index_)]
            : std::string{"?"};
    std::size_t width = 1U;
    for (const auto& candidate : cfg_.label_names) {
      width = std::max(width, candidate.size());
    }
    return cfg_.label_caption + " " + pad_right(name, static_cast<int>(width));
  }

  void draw_header() {
    draw_text(kMarginLeft, 20, cfg_.title, kTitle);
    std::string status;
    if (cfg_.overlay_label && !cfg_.label_names.empty()) {
      status += label_field() + "   ";
    }
    status +=
        "t = " +
        overlay_format(static_cast<double>(frame_index_) * cfg_.frame_period,
                       1, 7) +
        " s";
    if (cfg_.overlay_detections) {
      status += "   detections " +
                pad_left(std::to_string(detections_.size() / 2), 4);
    }
    if (cfg_.overlay_clusters) {
      status += "   clusters " + pad_left(std::to_string(rep_.size() / 2), 3);
    }
    draw_text(win_w_ - kMarginRight, 20, status, kLabel, Anchor::kRight);
  }

  // Color chip + physical readout per cluster, along the bottom margin.
  //
  // Two things keep the strip from flickering. Entries run in range-rank order
  // (rank_order_), not DBSCAN label order: the label is size-ranked, so a
  // label-ordered strip swaps its entries whenever two clusters trade point
  // counts (measured at ~23% of frames on the SISO scene). And every entry
  // occupies a fixed-width slot with space-padded numbers, so a value crossing
  // a power of ten cannot shift the entries beside it.
  void draw_legend(const Rect& rd) {
    if (!cfg_.overlay_clusters || rank_order_.empty()) {
      return;
    }
    const int y = rd.y + rd.h + 52;
    if (y + 6 > win_h_) {
      return;
    }
    const int slot_w = 16 + text_width(kLegendTemplate) + 20;
    if (slot_w <= 0) {
      return;
    }
    const int slots = std::max((rd.w + 20) / slot_w, 1);
    for (std::size_t rank = 0; rank < rank_order_.size(); ++rank) {
      if (static_cast<int>(rank) >= slots) {
        break;  // single strip: drop the overflow rather than wrap
      }
      const std::size_t c = static_cast<std::size_t>(rank_order_[rank]);
      const std::int64_t dd = rep_[c * 2 + 0];
      const std::int64_t rr = rep_[c * 2 + 1];
      if (dd < 0 || rr < 0 ||
          static_cast<std::size_t>(rr) >= ranges_.size() ||
          static_cast<std::size_t>(dd) >= velocities_.size()) {
        continue;
      }
      const int x = rd.x + static_cast<int>(rank) * slot_w;
      fill_rect(Rect{x, y - 8, 10, 10},
                bucket_color(buckets_.empty() ? -1 : buckets_[c]));
      draw_text(x + 16, y,
                overlay_format(ranges_[static_cast<std::size_t>(rr)], 1, 6) +
                    " m  " +
                    overlay_format_signed(
                        velocities_[static_cast<std::size_t>(dd)], 1, 5) +
                    " m/s",
                kLabel);
    }
  }

  // Rolling scatter: range (top panel) or velocity (bottom) versus scenario
  // time, one dot per detection colored by its cluster.
  void draw_panel(const Rect& box, bool range_panel) {
    fill_rect(box, kPanelBackground);
    frame_rect(box, kChrome);
    const std::vector<float>& bins = range_panel ? ranges_ : velocities_;
    double lo = 0.0;
    double hi = 1.0;
    if (!bins.empty()) {
      lo = *std::min_element(bins.begin(), bins.end());
      hi = *std::max_element(bins.begin(), bins.end());
    }
    if (hi - lo < 1e-6) {
      hi = lo + 1.0;
    }
    const double span_s =
        static_cast<double>(cfg_.plot_window) * cfg_.frame_period;
    const double t_now =
        static_cast<double>(frame_index_) * cfg_.frame_period;
    const double t0 = std::max(0.0, t_now - span_s);

    for (int i = 1; i < kPanelYTicks; ++i) {
      const double f = static_cast<double>(i) / kPanelYTicks;
      const int y = box.y + static_cast<int>(f * box.h);
      set_color(kGrid);
      XDrawLine(display_, back_, gc_, box.x + 1, y, box.x + box.w - 1, y);
    }
    for (int i = 0; i <= kPanelYTicks; ++i) {
      const double f = static_cast<double>(i) / kPanelYTicks;
      const double value = hi - f * (hi - lo);
      const int y = box.y + static_cast<int>(f * box.h);
      draw_text(box.x - 6, y + 4,
                range_panel ? overlay_format(value, 0)
                            : overlay_format_signed(value, 1),
                kLabel, Anchor::kRight);
    }
    for (int i = 0; i <= 2; ++i) {
      const double f = static_cast<double>(i) / 2.0;
      const int x = box.x + static_cast<int>(f * box.w);
      draw_text(x, box.y + box.h + 16, overlay_format(t0 + f * span_s, 1),
                kLabel, Anchor::kCenter);
    }
    // Title above the frame: inside it, a trace riding the top of the axis
    // draws straight through the text.
    draw_text(box.x, box.y - 6,
              range_panel ? "range (m) vs t (s)" : "velocity (m/s) vs t (s)",
              kTitle);

    const auto plot_y = [&](double value) {
      return box.y + box.h -
             static_cast<int>((value - lo) / (hi - lo) * box.h);
    };
    for (const PlotFrame& frame : history_) {
      if (frame.t < t0) {
        continue;
      }
      const int x =
          box.x + static_cast<int>((frame.t - t0) / span_s * box.w);
      if (x < box.x || x > box.x + box.w) {
        continue;
      }
      for (const PlotPoint& point : frame.points) {
        const int y = plot_y(range_panel ? point.range : point.velocity);
        if (y < box.y || y > box.y + box.h) {
          continue;
        }
        fill_disc(x, y, 2, bucket_color(point.bucket));
      }
      // Each cluster's representative as a '+' over its detection cloud —
      // the distilled per-cluster track.
      for (const PlotPoint& point : frame.reps) {
        const int y = plot_y(range_panel ? point.range : point.velocity);
        if (y < box.y + 3 || y > box.y + box.h - 3) {
          continue;
        }
        set_color(bucket_color(point.bucket));
        XDrawLine(display_, back_, gc_, std::max(x - 3, box.x + 1), y,
                  std::min(x + 3, box.x + box.w - 1), y);
        XDrawLine(display_, back_, gc_, x, y - 3, x, y + 3);
      }
    }
  }

  // Angular field of view: every detection at its (az, el) estimate, azimuth
  // left->right and elevation bottom->top, over the estimator's fixed scan
  // extents. Instantaneous — angle barely moves frame-to-frame, so unlike the
  // range/velocity panels this one keeps no history.
  void draw_boresight(const Rect& box) {
    fill_rect(box, kPanelBackground);
    frame_rect(box, kChrome);
    constexpr double kRad2Deg = 57.29577951308232;
    const double ax_lo = cfg_.az_lo * kRad2Deg;
    const double ey_lo = cfg_.el_lo * kRad2Deg;
    const double ax_span = std::max(cfg_.az_hi * kRad2Deg - ax_lo, 1e-6);
    const double ey_span = std::max(cfg_.el_hi * kRad2Deg - ey_lo, 1e-6);

    for (int i = 1; i < kPanelYTicks; ++i) {
      const double f = static_cast<double>(i) / kPanelYTicks;
      const int y = box.y + static_cast<int>(f * box.h);
      const int x = box.x + static_cast<int>(f * box.w);
      set_color(kGrid);
      XDrawLine(display_, back_, gc_, box.x + 1, y, box.x + box.w - 1, y);
      XDrawLine(display_, back_, gc_, x, box.y + 1, x, box.y + box.h - 1);
    }
    for (int i = 0; i <= kPanelYTicks; ++i) {
      const double f = static_cast<double>(i) / kPanelYTicks;
      draw_text(box.x - 6, box.y + static_cast<int>(f * box.h) + 4,
                overlay_format_signed(ey_lo + (1.0 - f) * ey_span, 0), kLabel,
                Anchor::kRight);
      draw_text(box.x + static_cast<int>(f * box.w), box.y + box.h + 16,
                overlay_format_signed(ax_lo + f * ax_span, 0), kLabel,
                Anchor::kCenter);
    }
    draw_text(box.x, box.y - 6, "boresight az/el (deg)", kTitle);

    const int radius = std::max(
        3, static_cast<int>(cfg_.marker_radius * static_cast<float>(box.w)));
    const std::size_t n = angles_.size() / 2;
    for (std::size_t i = 0; i < n; ++i) {
      const double xf =
          (static_cast<double>(angles_[i * 2 + 0]) * kRad2Deg - ax_lo) /
          ax_span;
      const double yf =
          (static_cast<double>(angles_[i * 2 + 1]) * kRad2Deg - ey_lo) /
          ey_span;
      if (xf < 0.0 || xf > 1.0 || yf < 0.0 || yf > 1.0) {
        continue;  // outside the field of view
      }
      const int cx = box.x + static_cast<int>(xf * box.w);
      const int cy = box.y + box.h - static_cast<int>(yf * box.h);
      fill_disc(cx, cy, radius + 1, kOutline);
      fill_disc(cx, cy, radius, bucket_color(bucket_for_point(i)));
    }
  }

  void pump_events() {
    while (display_ != nullptr && XPending(display_) > 0) {
      XEvent event;
      XNextEvent(display_, &event);
      if (event.type == ClientMessage &&
          static_cast<Atom>(event.xclient.data.l[0]) == delete_atom_) {
        HOLOSCAN_LOG_INFO(
            "X11RdViewerOp: window closed; continuing without the viewer");
        closed_ = true;
        shutdown();
        return;
      }
      if (event.type == DestroyNotify) {
        closed_ = true;
        shutdown();
        return;
      }
      if (event.type == ConfigureNotify) {
        const int width = event.xconfigure.width;
        const int height = event.xconfigure.height;
        if (width >= 320 && height >= 240 &&
            (width != win_w_ || height != win_h_)) {
          win_w_ = width;
          win_h_ = height;
          ensure_backbuffer();
          destroy_raster();  // RD rect changed; rebuilt on the next frame
        }
      }
      if (event.type == Expose && back_ != 0) {
        XCopyArea(display_, back_, window_, gc_, 0, 0,
                  static_cast<unsigned>(win_w_),
                  static_cast<unsigned>(win_h_), 0, 0);
        XFlush(display_);
      }
    }
  }

  void destroy_raster() {
    if (ximage_ != nullptr) {
      ximage_->data = nullptr;  // pixels_ owns the buffer
      XDestroyImage(ximage_);
      ximage_ = nullptr;
    }
    pixels_.clear();
    raster_w_ = 0;
    raster_h_ = 0;
  }

  void shutdown() {
    if (display_ == nullptr) {
      return;
    }
    destroy_raster();
    if (font_ != nullptr) {
      XFreeFont(display_, font_);
      font_ = nullptr;
    }
    if (back_ != 0) {
      XFreePixmap(display_, back_);
      back_ = 0;
    }
    if (gc_ != nullptr) {
      XFreeGC(display_, gc_);
      gc_ = nullptr;
    }
    if (window_ != 0) {
      XDestroyWindow(display_, window_);
      window_ = 0;
    }
    XCloseDisplay(display_);
    display_ = nullptr;
  }

  static constexpr OverlayRgb kBackground{18U, 18U, 24U};
  static constexpr OverlayRgb kPanelBackground{28U, 28U, 36U};
  static constexpr OverlayRgb kChrome{170U, 170U, 180U};
  static constexpr OverlayRgb kGrid{70U, 70U, 82U};
  static constexpr OverlayRgb kLabel{215U, 215U, 225U};
  static constexpr OverlayRgb kTitle{245U, 245U, 250U};
  static constexpr OverlayRgb kOutline{0U, 0U, 0U};
  static constexpr int kMarginLeft{62};
  static constexpr int kMarginRight{16};
  static constexpr int kMarginTop{46};
  static constexpr int kMarginBottom{66};
  static constexpr int kPanelGap{56};
  static constexpr int kPanelSplit{34};
  // Widest legend entry, sized to the padded readout fields it holds.
  static constexpr const char* kLegendTemplate{"9999.9 m  +99.9 m/s"};
  static constexpr int kRdXTicks{6};
  static constexpr int kRdYTicks{4};
  static constexpr int kPanelYTicks{4};

  X11RdViewerConfig cfg_{};
  Display* display_{nullptr};
  Window window_{0};
  Pixmap back_{0};
  Visual* visual_{nullptr};
  GC gc_{nullptr};
  XFontStruct* font_{nullptr};
  XImage* ximage_{nullptr};
  Atom delete_atom_{0};
  int depth_{0};
  int win_w_{0};
  int win_h_{0};
  int back_w_{0};
  int back_h_{0};
  int raster_w_{0};
  int raster_h_{0};
  bool closed_{false};
  std::int64_t frame_index_{0};
  static inline std::int64_t presented_count_{0};
  std::int64_t num_doppler_{0};
  std::int64_t num_range_{0};
  std::vector<float> host_lut_;
  std::vector<std::uint32_t> pixels_;
  std::vector<std::uint32_t> turbo_pixels_;
  std::vector<OverlayRgb> palette_;
  std::vector<std::int64_t> detections_;
  std::vector<std::int64_t> labels_;
  std::vector<std::int64_t> rep_;
  std::vector<int> buckets_;     // cluster row -> palette color
  std::vector<int> rank_order_;  // range rank -> cluster row
  std::vector<float> ranges_;
  std::vector<float> velocities_;
  std::vector<float> angles_;  // (az, el) radians per detection row
  std::int64_t label_index_{-1};
  std::deque<PlotFrame> history_;
};

}  // namespace holoscan::radar::app
