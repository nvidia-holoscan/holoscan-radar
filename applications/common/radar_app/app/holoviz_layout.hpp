// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Declarative Holoviz layout for the example apps. The FMCW RD-map
// path (~90 lines of InputSpec/View: turbo LUT, one CROSSES layer per
// cluster color + noise, optional right-column panels) is owned here
// and driven by HolovizLayout. SAR formation uses a separate factory
// (`make_image_holoviz_op`) for a greyscale COLOR_LUT image plus a
// formed-size COLOR overlay and an APC-look compass; it does not
// inherit CommonRunConfig or the RD-map layout.
//
// This deliberately builds only DATA (+ a factory that makes the HolovizOp)
// — it does NOT call add_flow. HolovizOp is already the operator; which tensors
// feed its multi-receiver `receivers` port is app-specific graph topology, so
// each app keeps its own flow wiring in compose() and hands this the layout.
//
// Header-only: inline functions.

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "radar_app/viz/viz.hpp"

namespace holoscan::radar::app {

// One right-column panel: an RGBA image layer (a software-rasterized plot) with
// a TEXT title drawn on top, both occupying the same sub-viewport.
struct RightPanel {
  const char* image_layer;
  const char* title_layer;
  std::string title;
  float offset_y;  // top of the panel, window-relative [0, 1]
  float height;    // panel height, window-relative
};

// A full RD-map window: the turbo-mapped |RD| image + cluster CROSSES on the
// left, and an optional stack of right-column panels. `plots_enabled == false`
// (or no panels) renders the map full-window.
struct HolovizLayout {
  std::string window_title;
  uint32_t width{800};
  uint32_t height{600};
  bool headless{false};
  bool plots_enabled{true};
  float left_width{0.60F};  // RD-map column width when panels are shown
  std::vector<RightPanel> right_panels;
};

// Build the HolovizOp `tensors` InputSpec vector for a layout. Layer 0 is the
// turbo LUT RD image; the cluster/noise CROSSES layers draw on top (named to
// match ClusterOverlayOp's emitted tensors). With panels enabled the map +
// crosses share the left viewport and each panel adds an image + title layer.
inline std::vector<holoscan::ops::HolovizOp::InputSpec>
build_holoviz_input_specs(const HolovizLayout& layout) {
  using InputSpec = holoscan::ops::HolovizOp::InputSpec;
  using InputType = holoscan::ops::HolovizOp::InputType;

  std::vector<InputSpec> input_specs;
  input_specs.emplace_back("rd_map", InputType::COLOR_LUT);

  const auto palette = cluster_palette();
  if (static_cast<int>(palette.size()) != kNumClusterColors) {
    throw std::runtime_error(
        "build_holoviz_input_specs: cluster palette size must equal "
        "kNumClusterColors");
  }
  const auto add_crosses = [&](const std::string& name,
                               const std::vector<float>& color) {
    InputSpec spec(name, InputType::CROSSES);
    spec.color_ = color;
    spec.line_width_ = 2.0F;
    spec.priority_ = 1;
    input_specs.push_back(spec);
  };
  for (int c = 0; c < kNumClusterColors; ++c) {
    add_crosses(cluster_layer_name(c), palette[static_cast<size_t>(c)]);
  }
  add_crosses(kNoiseLayerName, {0.6F, 0.6F, 0.6F, 1.0F});

  // A View maps a layer's [0, 1] content into an (offset, size) rectangle.
  const auto make_view = [](float ox, float oy, float w, float h) {
    InputSpec::View v;
    v.offset_x_ = ox;
    v.offset_y_ = oy;
    v.width_ = w;
    v.height_ = h;
    return v;
  };

  if (layout.plots_enabled && !layout.right_panels.empty()) {
    // Shrink the RD map + crosses into the left viewport so the crosses stay
    // aligned to the image.
    const auto left = make_view(0.0F, 0.0F, layout.left_width, 1.0F);
    for (InputSpec& spec : input_specs) {
      spec.views_ = {left};
    }
    // Right column starts just past the map with a small gutter, and runs to a
    // matching right margin.
    const float right_x = layout.left_width + 0.03F;
    const float right_w = 1.0F - right_x - 0.02F;
    for (const RightPanel& panel : layout.right_panels) {
      InputSpec image(panel.image_layer, InputType::COLOR);
      image.views_ = {
          make_view(right_x, panel.offset_y, right_w, panel.height)};
      input_specs.push_back(image);
      InputSpec text(panel.title_layer, InputType::TEXT);
      text.text_ = {panel.title};
      text.color_ = {1.0F, 1.0F, 1.0F, 1.0F};
      text.priority_ = 2;
      text.views_ = {make_view(right_x, panel.offset_y, right_w, panel.height)};
      input_specs.push_back(text);
    }
  }
  return input_specs;
}

// Construct the HolovizOp from a layout: the built input specs, the shared
// turbo LUT, and the window title/size/headless flag. The app wires its own
// flows into the returned operator's `receivers` port.
inline std::shared_ptr<holoscan::ops::HolovizOp> make_rd_holoviz_op(
    holoscan::Fragment* fragment, const std::string& name,
    const HolovizLayout& layout) {
  return fragment->make_operator<holoscan::ops::HolovizOp>(
      name, holoscan::Arg("tensors", build_holoviz_input_specs(layout)),
      holoscan::Arg("color_lut", make_turbo_lut()),
      holoscan::Arg("window_title", layout.window_title),
      holoscan::Arg("width", layout.width),
      holoscan::Arg("height", layout.height),
      holoscan::Arg("headless", layout.headless));
}

// Single COLOR_LUT image, full window — the SAR formation path (one
// formed image, no cluster overlays). `layer_name` is both the
// Holoviz InputSpec name and the TensorMap key the rasterizer emits.
// R8_UINT indices are PNG codes 0–255. Do not set PASS_THROUGH:
// this GLFW surface only offers SrgbNonlinear (`No framebuffer
// format found which supports the requirements pass_through`).
// Inverse-sRGB on the LUT overshot (too dark vs the PNG).
// Optional `on_window_size`: same as make_color_holoviz_op
// (place the window after GLFW maps it).
inline std::shared_ptr<holoscan::ops::HolovizOp> make_image_holoviz_op(
    holoscan::Fragment* fragment, const std::string& name,
    const HolovizLayout& layout, const std::string& layer_name,
    const std::vector<std::vector<float>>& color_lut,
    holoscan::ops::HolovizOp::WindowSizeCallbackFunction
        on_window_size = {}) {
  using InputSpec = holoscan::ops::HolovizOp::InputSpec;
  using InputType = holoscan::ops::HolovizOp::InputType;
  using ImageFormat = holoscan::ops::HolovizOp::ImageFormat;
  std::vector<InputSpec> specs;
  InputSpec spec(layer_name, InputType::COLOR_LUT);
  spec.image_format_ = ImageFormat::R8_UINT;
  specs.push_back(spec);
  // Scene name / formed size / BP timing. Views are overlay-canvas
  // pixels over the window so a landscape strip does not stretch
  // the glyphs (0.55 x 0.15 of a 1920x201 window did).
  InputSpec label("size_label", InputType::COLOR);
  label.image_format_ = ImageFormat::R8G8B8A8_UNORM;
  label.priority_ = 1;
  const float win_w =
      static_cast<float>(std::max(layout.width, uint32_t{1}));
  const float win_h =
      static_cast<float>(std::max(layout.height, uint32_t{1}));
  constexpr float kPad = 0.015F;
  InputSpec::View banner;
  banner.offset_x_ = kPad;
  banner.offset_y_ = kPad;
  banner.width_ = static_cast<float>(kSarOverlayWidth) / win_w;
  banner.height_ = static_cast<float>(kSarOverlayHeight) / win_h;
  label.views_ = {banner};
  specs.push_back(label);
  // N-up compass (true ENU). The formed-image chevron is raster
  // u/v, drawn on the COLOR_LUT layer. Bottom-right so it does
  // not cover the size_label banner. Square in window pixels.
  InputSpec look("apc_look", InputType::COLOR);
  look.image_format_ = ImageFormat::R8G8B8A8_UNORM;
  look.priority_ = 1;
  const float cpx = static_cast<float>(kSarCompassSide);
  InputSpec::View compass;
  compass.width_ = cpx / win_w;
  compass.height_ = cpx / win_h;
  compass.offset_x_ = 1.0F - kPad - compass.width_;
  compass.offset_y_ = 1.0F - kPad - compass.height_;
  look.views_ = {compass};
  specs.push_back(look);
  holoscan::ArgList args;
  args.add(holoscan::Arg("tensors", specs));
  args.add(holoscan::Arg("color_lut", color_lut));
  args.add(holoscan::Arg("window_title", layout.window_title));
  args.add(holoscan::Arg("width", layout.width));
  args.add(holoscan::Arg("height", layout.height));
  args.add(holoscan::Arg("headless", layout.headless));
  if (on_window_size) {
    args.add(holoscan::Arg("window_size_callback",
                           std::move(on_window_size)));
  }
  return fragment->make_operator<holoscan::ops::HolovizOp>(
      name, args);
}

// Full-window RGBA image (no LUT). Optional `on_window_size`:
// HolovizOp has no window-position parameter; a caller can use the
// first size callback to place the window after GLFW maps it.
inline std::shared_ptr<holoscan::ops::HolovizOp> make_color_holoviz_op(
    holoscan::Fragment* fragment, const std::string& name,
    const HolovizLayout& layout, const std::string& layer_name,
    holoscan::ops::HolovizOp::WindowSizeCallbackFunction
        on_window_size = {}) {
  using InputSpec = holoscan::ops::HolovizOp::InputSpec;
  using InputType = holoscan::ops::HolovizOp::InputType;
  using ImageFormat = holoscan::ops::HolovizOp::ImageFormat;
  std::vector<InputSpec> specs;
  InputSpec spec(layer_name, InputType::COLOR);
  spec.image_format_ = ImageFormat::R8G8B8A8_UNORM;
  specs.push_back(spec);
  holoscan::ArgList args;
  args.add(holoscan::Arg("tensors", specs));
  args.add(holoscan::Arg("window_title", layout.window_title));
  args.add(holoscan::Arg("width", layout.width));
  args.add(holoscan::Arg("height", layout.height));
  args.add(holoscan::Arg("headless", layout.headless));
  if (on_window_size) {
    args.add(holoscan::Arg("window_size_callback",
                           std::move(on_window_size)));
  }
  return fragment->make_operator<holoscan::ops::HolovizOp>(name, args);
}

// Preset: two stacked right-column panels — range vs time and velocity vs time
// (the SISO layout).
inline std::vector<RightPanel> range_velocity_panels() {
  return {
      {kRangePlotLayer, kRangeTitleLayer, "Range (m) vs time (s)", 0.04F,
       0.44F},
      {kVelocityPlotLayer, kVelocityTitleLayer, "Velocity (m/s) vs time (s)",
       0.52F, 0.44F},
  };
}

// Preset: three stacked right-column panels — range vs time, velocity vs time,
// and the boresight az/el scatter (the SIMO/dense/MIMO layout).
inline std::vector<RightPanel> range_velocity_boresight_panels() {
  return {
      {kRangePlotLayer, kRangeTitleLayer, "Range (m) vs time (s)", 0.02F,
       0.30F},
      {kVelocityPlotLayer, kVelocityTitleLayer, "Velocity (m/s) vs time (s)",
       0.35F, 0.30F},
      {kBoresightPlotLayer, kBoresightTitleLayer, "Boresight az/el (deg)",
       0.68F, 0.30F},
  };
}

}  // namespace holoscan::radar::app
