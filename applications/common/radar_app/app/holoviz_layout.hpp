// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Declarative Holoviz layout for the FMCW example apps. The ~90-line
// InputSpec/View construction that every app copy-pasted (the turbo LUT RD-map
// layer, the detection disc layers — one per cluster color + noise + outline —
// and the right-column image+title panels) is owned here and driven by a small
// HolovizLayout struct.
//
// This deliberately builds only DATA (+ a factory that makes the HolovizOp)
// — it does NOT call add_flow. HolovizOp is already the operator; which tensors
// feed its multi-receiver `receivers` port is app-specific graph topology, so
// each app keeps its own flow wiring in compose() and hands this the layout.
//
// Header-only: inline functions.

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
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

// A full RD-map window: the turbo-mapped |RD| image + cluster discs on the
// left, and an optional stack of right-column panels. `plots_enabled == false`
// (or no panels) renders the map full-window.
struct HolovizLayout {
  std::string window_title;
  uint32_t width{800};
  uint32_t height{600};
  bool headless{false};
  bool vsync{true};  // FIFO present mode when true (resize-robust); see config
  bool plots_enabled{true};
  float left_width{0.60F};  // RD-map column width when panels are shown
  std::vector<RightPanel> right_panels;
};

// Build the HolovizOp `tensors` InputSpec vector for a layout. Layer 0 is the
// turbo LUT RD image; the detection disc layers draw on top (named to match
// ClusterOverlayOp's emitted tensors). With panels enabled the map + discs
// share the left viewport and each panel adds an image + title layer.
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
  // Detections render as filled triangle-fan discs (see ClusterOverlayOp): a
  // single black outline layer of larger discs sits at priority 1, under one
  // colored layer per cluster palette color plus a muted noise layer at
  // priority 2 — both over the RD image (COLOR_LUT, default priority 0).
  InputSpec outline(kDetectionOutlineName, InputType::TRIANGLES);
  outline.color_ = {0.0F, 0.0F, 0.0F, 1.0F};
  outline.priority_ = 1;
  input_specs.push_back(outline);
  const auto add_discs = [&](const std::string& name,
                             const std::vector<float>& color) {
    InputSpec spec(name, InputType::TRIANGLES);
    spec.color_ = color;
    spec.priority_ = 2;
    input_specs.push_back(spec);
  };
  for (int c = 0; c < kNumClusterColors; ++c) {
    add_discs(cluster_layer_name(c), palette[static_cast<size_t>(c)]);
  }
  add_discs(kNoiseLayerName, {0.6F, 0.6F, 0.6F, 1.0F});

  // A View maps a layer's [0, 1] content into an (offset, size) rectangle.
  const auto make_view = [](float ox, float oy, float w, float h) {
    InputSpec::View v;
    v.offset_x_ = ox;
    v.offset_y_ = oy;
    v.width_ = w;
    v.height_ = h;
    return v;
  };

  if (layout.plots_enabled) {
    // Shrink the RD map + discs into the left viewport (the side panels — static
    // right_panels here, or dynamic primitive panels emitted by the plot ops —
    // occupy the right column).
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

// On-screen aspect ratio (width / height in pixels) of the RD-map sub-view: the
// full window when the map is full-bleed, else the left column shared with the
// side panels. ClusterOverlayOp divides its disc x radius by this so the
// detection markers render round rather than stretched.
inline float rd_map_view_aspect(const HolovizLayout& layout) {
  const bool paneled = layout.plots_enabled && !layout.right_panels.empty();
  const float view_w = (paneled ? layout.left_width : 1.0F) *
                       static_cast<float>(layout.width);
  return view_w / static_cast<float>(layout.height);
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
      holoscan::Arg("vsync", layout.vsync),
      holoscan::Arg("headless", layout.headless));
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
