// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mode-1 SAR formation: one collection, one image, dumped as .npy
// and shown in Holoviz (visualize: true). The default YAML forms
// the Atlanta Umbra collect; pass mode01_smoke.yaml for the
// generated fixture.
//
//   CPHDReaderOp → SARRangeCompressionOp → BackprojectionOp
//                                        → TensorDumpOp
//                                        [→ SarImageToLutOp
//                                            → HoldLastImageOp
//                                            → HolovizOp]
//   reader.positions / reference_position bypass range compression.
//   StreamBoundaryControlOp (custom_ops.hpp) watches the reader's
//   D22 announcement and issues emit-and-reset on that tick.
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/sar_formation/cpp/sar_formation \
//            [mode01.yaml] <collection.cphd>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "io/cphd_reader_op/collection_paths.hpp"
#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "radar_app/app/holoviz_layout.hpp"
#include "radar_app/viz/display.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"
#include "config.hpp"       // NOLINT(build/include_subdir)
#include "custom_ops.hpp"   // NOLINT(build/include_subdir)

namespace radar = holoscan::radar;
using holoscan::radar::app::HolovizLayout;
using holoscan::radar::app::gui_available;
using holoscan::radar::app::make_grayscale_lut;
using holoscan::radar::app::make_image_holoviz_op;

class SarFormationApp : public holoscan::Application {
 public:
  AppConfig cfg;

  void compose() override {
    const auto& origin = cfg.sample_geometry.origin();
    const auto& axis_v = cfg.sample_geometry.axis_v();
    const auto& axis_u = cfg.sample_geometry.axis_u();
    HOLOSCAN_LOG_INFO(
        "sar_formation: {}  lattice {}x{}  f0={:.6e} Hz  dump {}",
        cfg.cphd_path, cfg.sample_geometry.num_v(),
        cfg.sample_geometry.num_u(), cfg.reference_frequency,
        cfg.dump_path);
    HOLOSCAN_LOG_INFO(
        "sar_formation lattice origin [{:.3f}, {:.3f}, {:.3f}]",
        origin[0], origin[1], origin[2]);
    HOLOSCAN_LOG_INFO(
        "sar_formation axis_v [{:.6f}, {:.6f}, {:.6f}]",
        axis_v[0], axis_v[1], axis_v[2]);
    HOLOSCAN_LOG_INFO(
        "sar_formation axis_u [{:.6f}, {:.6f}, {:.6f}]",
        axis_u[0], axis_u[1], axis_u[2]);

    // Spaceborne. Atlanta's lattice is scene ENU (near the origin)
    // but the APC is ~770 km out; float32 ULP there is ~9 cm.
    // The graph stays double. TESTING.md: fp32 is degraded at
    // spaceborne ranges.
    HOLOSCAN_LOG_INFO("sar_formation precision double");
    HOLOSCAN_LOG_INFO(
        "sar_formation range_bin_spacing {:.9f} m (CPHD SCSS; "
        "range_bins unconnected)",
        cfg.range_bin_spacing);
    auto reader = make_operator<radar::CPHDReaderOp<double>>(
        "cphd_reader",
        from_config("cphd_reader"),
        holoscan::Arg("path") =
            radar::CollectionPaths(cfg.cphd_path));

    auto range = make_operator<radar::SARRangeCompressionOp<double>>(
        "range_compression", from_config("range_compression"));

    // Lattice and stored-sample-N/2 frequency: sidecar when present
    // (generated fixtures), otherwise the app-owned YAML keys
    // (Atlanta and any other public CPHD).
    auto bp = make_operator<radar::BackprojectionOp<double>>(
        "backprojection",
        from_config("backprojection"),
        holoscan::Arg("sample_geometry") = cfg.sample_geometry,
        holoscan::Arg("reference_frequency") =
            cfg.reference_frequency,
        holoscan::Arg("range_bin_spacing") = cfg.range_bin_spacing);

    auto control = make_operator<StreamBoundaryControlOp>("control");

    auto dump = make_operator<radar::TensorDumpOp>(
        "tensor_dump", holoscan::Arg("path") = cfg.dump_path);

    // Fan-out of phase_history is the control operator's reachability
    // edge: metadata arrives only through receive().
    add_flow(reader, range, {{"phase_history", "phase_history"}});
    add_flow(reader, control, {{"phase_history", "announce"}});
    if (cfg.transform_to_scene_enu) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: ECEF→ENU at IARP for NVRadar grid");
      auto to_enu = make_operator<EcefToSceneEnuOp>(
          "ecef_to_enu",
          holoscan::Arg("origin_ecef") = std::vector<double>{
              cfg.enu_origin_ecef[0], cfg.enu_origin_ecef[1],
              cfg.enu_origin_ecef[2]},
          holoscan::Arg("quantize_phase_center_f32") =
              cfg.quantize_phase_center_f32);
      add_flow(reader, to_enu, {{"positions", "positions"}});
      add_flow(reader, to_enu,
               {{"reference_position", "reference_position"}});
      add_flow(to_enu, bp, {{"positions", "positions"}});
      add_flow(to_enu, bp,
               {{"reference_position", "reference_position"}});
    } else {
      add_flow(reader, bp, {{"positions", "positions"}});
      add_flow(reader, bp,
               {{"reference_position", "reference_position"}});
    }
    if (!cfg.nvradar_vph_path.empty()) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: NVRadar vph override {}",
          cfg.nvradar_vph_path);
      auto vph = make_operator<NvradarVphOverrideOp>(
          "nvradar_vph",
          holoscan::Arg("path") = cfg.nvradar_vph_path);
      add_flow(range, vph, {{"range_profiles", "range_profiles"}});
      add_flow(vph, bp, {{"range_profiles", "range_profiles"}});
    } else {
      add_flow(range, bp, {{"range_profiles", "range_profiles"}});
    }
    add_flow(control, bp, {{"control", "control"}});
    add_flow(bp, dump, {{"image", "tensor"}});

    if (cfg.visualize) {
      auto to_lut = make_operator<SarImageToLutOp>("sar_image_to_lut");
      auto hold = make_operator<HoldLastImageOp>(
          "hold_image",
          make_condition<holoscan::PeriodicCondition>(
              "hold_period", static_cast<int64_t>(100000000)));
      hold->hold_s = cfg.visualize_hold_s;

      HolovizLayout layout;
      layout.window_title = "SAR formation";
      layout.width = 1024U;
      layout.height = 1024U;
      layout.headless = cfg.headless;
      auto visualizer = make_image_holoviz_op(
          this, "holoviz", layout, "sar_image",
          make_grayscale_lut());

      add_flow(bp, to_lut, {{"image", "image"}});
      add_flow(to_lut, hold, {{"sar_image", "image"}});
      add_flow(hold, visualizer, {{"image", "receivers"}});
    }
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<SarFormationApp>();
  try {
    const CliPaths cli = parse_cli(argc, argv, "mode01.yaml");
    app->cfg = load_config(cli.config, cli.cphd_override);
    if (app->cfg.visualize && !app->cfg.headless &&
        !app->cfg.skip_display_check && !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "sar_formation: visualize is enabled but no usable "
          "display was detected (Holoviz could not initialize a "
          "window); running without the visualization window");
      app->cfg.visualize = false;
    }
    // Same file feeds from_config() for the operator blocks.
    app->config(cli.config.string());
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("sar_formation: {}", e.what());
    return 1;
  }
  return 0;
}
