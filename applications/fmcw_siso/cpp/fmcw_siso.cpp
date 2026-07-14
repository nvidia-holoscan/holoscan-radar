// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SISO FMCW pipeline example: a synthetic ADC source feeding the split
// signal-processing chain
//
//   AdcSourceOp -> FMCWRangeProcessorOp -> ClutterFilterOp
//               -> FMCWDopplerProcessorOp -> CFARDetectorOp
//                                        -> DetectionReportOp
//                                        -> FeatureAssemblyOp
//                                             -> ClusterDBSCANOp
//                                                  -> DetectionReductionOp
//                                                       -> ClusterReportOp
//                                         [-> RdMapToImageOp   -> HolovizOp]
//                                         [-> ClusterOverlayOp -> HolovizOp]
//                                         [-> RollingPlotOp    -> HolovizOp]
//                                              (RD map + cluster crosses left;
//                                               range/velocity-vs-time right)
//
// The source synthesizes one static clutter return and one moving target
// with the shared analytic point-target simulator
// (radar_common/sim/point_target_sim.hpp), emitting one cube (CPI) per tick
// and advancing the targets through space between ticks. CFAR detects on the
// range-Doppler map; the sink reports each detection in physical units via
// the operators' range_bins / velocity_bins axis outputs, so the moving
// target's range is seen to sweep over the run. In parallel, FeatureAssemblyOp
// gathers a [range, velocity] feature matrix from the CFAR cells (same axis
// outputs, physical units) and ClusterDBSCANOp density-clusters it, so the
// adjacent cells lit by one target collapse into a single cluster;
// ClusterReportOp logs each cluster's size and centroid.
//
// All configuration lives in fmcw_siso.yaml (staged next to the binary;
// override with a path as the first CLI argument): the per-operator blocks
// (range_processor, clutter_filter, doppler_processor, cfar_detector,
// cluster_dbscan, reduction) all flow through Holoscan's native
// config()/from_config() path. Only the synthetic source, run cadence, and
// visualization toggles are parsed into AppConfig; FeatureAssemblyOp's fixed
// [range, velocity] layout is applied as an explicit Arg (native-only param).
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_siso/cpp/fmcw_siso [config.yaml]

#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/viz/display.hpp"
#include "radar_app/viz/viz.hpp"

namespace radar = holoscan::radar;

// The reusable report + visualization operators (DetectionReportOp,
// ClusterReportOp, NullSinkOp, RdMapToImageOp, ClusterOverlayOp,
// RollingPlotOp), the turbo LUT / cluster palette / overlay + plot layer-name
// helpers, and the gui_available() display probe all live in the shared
// radar_app library; this app only wires them into its graph.
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

namespace {

// Rolling time-series plots drawn beside the RD map (range and velocity vs
// time). Only consulted when visualize is on; the plots share the Holoviz
// window via sub-viewports.
struct PlotsConfig {
  bool enable{true};
  int64_t window{120};  // frames retained in the scrolling window
};

struct AppConfig {
  radar::sim::SimScene scene;  // waveform, cube dims, targets, noise
  PlotsConfig plots;
  int64_t num_frames{1};
  double frame_period{0.1};  // s, between emitted cubes (and target motion)
  bool visualize{false};     // live RD-map window (requires a GUI host)
  // Skip the startup display probe: saves its GLFW + Vulkan init/teardown,
  // but on a host without a usable display the failure then surfaces inside
  // the running graph and aborts the process instead of downgrading.
  bool skip_display_check{false};
  // Render Holoviz offscreen (no window). Lets the visual pipeline run on a
  // display-less host (CI, headless dev boxes); the startup display probe is
  // skipped since offscreen rendering needs no display server.
  bool headless{false};
};

// Load and validate the app-owned YAML sections.
AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;

  // Decode + validate the shared waveform anchor through the same
  // YAML::convert<> the operators use, then mirror it into the simulator's
  // decoupled SimWaveform. The descriptor derives bandwidth at
  // construction, so a bandwidth- or slope-spelled config maps to the same
  // beat rate.
  const auto waveform =
      root["waveform"].as<radar::FMCWWaveformDescriptor>();
  cfg.scene.waveform.carrier_frequency = waveform.carrier_frequency();
  cfg.scene.waveform.bandwidth = waveform.bandwidth();
  cfg.scene.waveform.chirp_duration = waveform.chirp_duration();
  cfg.scene.waveform.sample_rate = waveform.sample_rate();
  cfg.scene.waveform.chirp_repetition_interval =
      waveform.chirp_repetition_interval();

  cfg.num_frames = root["run"]["num_frames"].as<int64_t>();
  cfg.frame_period = root["run"]["frame_period"].as<double>();
  cfg.scene.num_chirps = root["cube"]["num_chirps"].as<int64_t>();
  cfg.scene.samples_per_chirp = root["cube"]["samples_per_chirp"].as<int64_t>();

  // Fail here with the offending key rather than mid-graph: these values
  // reach cudaMalloc sizing (cube dims) and the Count/Periodic conditions
  // (run cadence), where bad values surface as cryptic CUDA/GXF errors.
  const auto require_positive = [](const char* key, auto value) {
    if (value <= 0) {
      throw std::runtime_error(std::string(key) + " must be > 0, got " +
                               std::to_string(value));
    }
  };
  require_positive("run.num_frames", cfg.num_frames);
  require_positive("run.frame_period", cfg.frame_period);
  require_positive("cube.num_chirps", cfg.scene.num_chirps);
  require_positive("cube.samples_per_chirp", cfg.scene.samples_per_chirp);

  // frame_period feeds a PeriodicCondition in nanoseconds (see compose()).
  // require_positive rejects <= 0, but NaN slips through (NaN <= 0 is false)
  // and a positive sub-nanosecond period would round to a zero (spinning)
  // interval, while a huge one overflows the int64 cast. Require a finite
  // value whose nanosecond count lands in [1, INT64_MAX].
  if (!std::isfinite(cfg.frame_period)) {
    throw std::runtime_error("run.frame_period must be finite");
  }
  const double frame_period_ns = cfg.frame_period * 1e9;
  if (frame_period_ns < 1.0 ||
      frame_period_ns >
          static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "run.frame_period must be within [1 ns, INT64_MAX ns], got " +
        std::to_string(cfg.frame_period) + " s");
  }

  for (const YAML::Node& t : root["targets"]) {
    const double range = t["range"].as<double>();
    const double velocity = t["velocity"].as<double>();
    const double amplitude = t["amplitude"].as<double>();
    cfg.scene.targets.push_back({radar::sim::Vec3{range, 0.0, 0.0},
                                 radar::sim::Vec3{velocity, 0.0, 0.0},
                                 {amplitude, 0.0}});
  }

  const YAML::Node noise = root["noise_snr_db"];
  if (noise && !noise.IsNull()) {
    cfg.scene.snr_db = noise.as<double>();
  }

  const YAML::Node visualize = root["visualize"];
  if (visualize && !visualize.IsNull()) {
    cfg.visualize = visualize.as<bool>();
  }
  const YAML::Node skip_check = root["skip_display_check"];
  if (skip_check && !skip_check.IsNull()) {
    cfg.skip_display_check = skip_check.as<bool>();
  }
  const YAML::Node headless = root["headless"];
  if (headless && !headless.IsNull()) {
    cfg.headless = headless.as<bool>();
  }
  const YAML::Node plots = root["plots"];
  if (plots && !plots.IsNull()) {
    if (plots["enable"]) {
      cfg.plots.enable = plots["enable"].as<bool>();
    }
    if (plots["window"]) {
      cfg.plots.window = plots["window"].as<int64_t>();
    }
  }
  if (cfg.plots.window <= 1) {
    throw std::runtime_error("plots.window must be > 1, got " +
                             std::to_string(cfg.plots.window));
  }

  // The cfar_detector, cluster_dbscan, and reduction operator blocks are NOT
  // parsed here — they flow through Holoscan's from_config() in compose(), and
  // each operator validates its own parameters at initialize(). Only the
  // synthetic source, run cadence, and visualization toggles above are
  // app-owned.
  return cfg;
}

// Advance each target's base position by one frame period. The simulator
// already models chirp-to-chirp motion within a CPI; this carries that motion
// across CPIs so successive frames see the target at a new range.
void advance_scene(radar::sim::SimScene* scene, double dt) {
  for (radar::sim::PointTarget& target : scene->targets) {
    for (int axis = 0; axis < 3; ++axis) {
      target.position[axis] += target.velocity[axis] * dt;
    }
  }
}

}  // namespace

// Emits one synthetic complex-IF ADC cube per tick, advancing the targets
// through space by one frame period between ticks. `scene` and `frame_period`
// are set from the parsed configuration before the graph runs.
class AdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdcSourceOp)
  AdcSourceOp() = default;

  radar::sim::SimScene scene;
  double frame_period{0.1};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    // The simulator computes in double precision; cast to the float ADC
    // sample type before upload.
    const radar::sim::SimResult sim = radar::sim::simulate(scene);
    // SISO app: the allocation below assumes the sim's single-RX rank-2
    // cube. A multi-RX scene yields rank-3 — allocate from sim.shape when
    // lifting this source for SIMO/MIMO apps.
    if (sim.shape.size() != 2) {
      throw std::runtime_error(
          "AdcSourceOp: SISO source expects a rank-2 cube, got rank " +
          std::to_string(sim.shape.size()));
    }
    std::vector<std::complex<float>> cube(sim.cube.size());
    for (size_t i = 0; i < sim.cube.size(); ++i) {
      cube[i] = std::complex<float>(static_cast<float>(sim.cube[i].real()),
                                    static_cast<float>(sim.cube[i].imag()));
    }

    auto tensor = radar::make_device_tensor(
        {scene.num_chirps, scene.samples_per_chirp},
        radar::complex_dtype<float>());
    if (cudaMemcpy(tensor->data(), cube.data(),
                   sizeof(std::complex<float>) * cube.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("AdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");

    // Step the scene forward for the next frame.
    advance_scene(&scene, frame_period);
  }
};

class FmcwSisoApp : public holoscan::Application {
 public:
  AppConfig cfg;  // set before run()

  void compose() override {
    // Emit cfg.num_frames cubes, one every cfg.frame_period seconds, so the
    // targets are seen moving through space over the run.
    auto source = make_operator<AdcSourceOp>(
        "adc_source",
        make_condition<holoscan::CountCondition>("frame_count",
                                                 cfg.num_frames),
        make_condition<holoscan::PeriodicCondition>(
            "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
    source->scene = cfg.scene;
    source->frame_period = cfg.frame_period;

    // Operator parameters come straight from the same-named YAML blocks;
    // the yaml_converters.hpp specializations decode the custom types.
    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", from_config("doppler_processor"));
    auto cfar = make_operator<radar::CFARDetectorOp<float>>(
        "cfar_detector", from_config("cfar_detector"));
    auto sink = make_operator<DetectionReportOp>("detection_report");
    // Every CFAR output needs a receiver or the operator never ticks.
    auto mask_sink = make_operator<NullSinkOp>("detection_mask_sink");
    auto noise_sink = make_operator<NullSinkOp>("noise_estimate_sink");

    // Clustering branch: assemble a [range, velocity] feature matrix from the
    // CFAR cells (physical units, via the range/Doppler axis outputs) and
    // density-cluster it. `features` is fixed in code — the parameter is
    // native-only by operator contract, and this app's downstream report /
    // overlays assume this exact 2-column layout. The DBSCAN parameters flow
    // natively from the cluster_dbscan YAML block.
    const std::vector<radar::AssemblyFeature> feature_columns{
        radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity};
    auto feature_assembly = make_operator<radar::FeatureAssemblyOp<float>>(
        "feature_assembly", holoscan::Arg("features") = feature_columns);
    auto cluster = make_operator<radar::ClusterDBSCANOp<float>>(
        "cluster_dbscan", from_config("cluster_dbscan"));
    auto cluster_report = make_operator<ClusterReportOp>("cluster_report");
    // Reduce each cluster to one representative cell. The reduction block
    // (representative / score_method / feature_weights) flows natively; set
    // feature_weights ~ 1/epsilon in the YAML so the medoid distance matches
    // the DBSCAN metric (else range's larger magnitude dominates velocity).
    auto reduction = make_operator<radar::DetectionReductionOp<float>>(
        "detection_reduction", from_config("reduction"));

    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
    add_flow(doppler, cfar, {{"rd_maps", "rd_maps"}});
    add_flow(cfar, sink, {{"detection_idx", "detection_idx"}});
    add_flow(cfar, sink, {{"detection_score", "detection_score"}});
    add_flow(range, sink, {{"range_bins", "range_bins"}});
    add_flow(doppler, sink, {{"velocity_bins", "velocity_bins"}});
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    // CFAR cells → feature matrix → DBSCAN → report. The range/Doppler axis
    // outputs feed FeatureAssemblyOp so the columns land in physical units;
    // its `points` output fans out to both the clusterer and the reporter
    // (the reporter reads it for centroids).
    add_flow(cfar, feature_assembly, {{"detection_idx", "detection_idx"}});
    add_flow(range, feature_assembly, {{"range_bins", "range_bins"}});
    add_flow(doppler, feature_assembly, {{"velocity_bins", "velocity_bins"}});
    add_flow(feature_assembly, cluster, {{"points", "points"}});
    add_flow(feature_assembly, cluster_report, {{"points", "points"}});
    add_flow(cluster, cluster_report, {{"labels", "labels"}});
    add_flow(cluster, cluster_report, {{"sizes", "sizes"}});

    // Reduction: one representative cell per cluster (medoid = most-central
    // real detection). rep_idx feeds the report (logged in physical units, so
    // the report also needs the axis bins) and, when visualizing, the GUI.
    add_flow(cluster, reduction, {{"labels", "labels"}});
    add_flow(cfar, reduction, {{"detection_idx", "detection_idx"}});
    add_flow(feature_assembly, reduction, {{"points", "points"}});
    add_flow(reduction, cluster_report, {{"rep_idx", "rep_idx"}});
    add_flow(range, cluster_report, {{"range_bins", "range_bins"}});
    add_flow(doppler, cluster_report, {{"velocity_bins", "velocity_bins"}});

    // Optional live RD-map window (cfg.visualize; main() has already
    // downgraded it with a warning on GUI-less hosts). Turbo-mapped |RD|
    // (range = x, Doppler = y) with the CFAR detections overlaid as crosses
    // colored by DBSCAN cluster (noise in gray). With cfg.plots.enable, the
    // map shares the window's left viewport and two rolling time-series plots
    // (range and velocity vs time) stack on the right.
    if (cfg.visualize) {
      using InputSpec = holoscan::ops::HolovizOp::InputSpec;
      using InputType = holoscan::ops::HolovizOp::InputType;
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      // Layer 0 is the turbo LUT image; the cluster/noise CROSSES layers draw
      // on top (higher priority renders last). One CROSSES InputSpec per
      // palette color, named to match ClusterOverlayOp's emitted tensors.
      std::vector<InputSpec> input_specs;
      input_specs.emplace_back("rd_map", InputType::COLOR_LUT);
      const auto palette = cluster_palette();
      if (static_cast<int>(palette.size()) != kNumClusterColors) {
        throw std::runtime_error(
            "fmcw_siso: cluster palette size must equal kNumClusterColors");
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

      // A View maps a layer's [0,1] content into an (offset, size) rectangle
      // of the window. With plots enabled, the RD map + overlay share the left
      // viewport so the crosses stay aligned to the shrunk image.
      const auto make_view = [](float ox, float oy, float w, float h) {
        InputSpec::View v;
        v.offset_x_ = ox;
        v.offset_y_ = oy;
        v.width_ = w;
        v.height_ = h;
        return v;
      };
      std::shared_ptr<RollingPlotOp> plot;
      if (cfg.plots.enable) {
        const auto left = make_view(0.0F, 0.0F, 0.60F, 1.0F);
        for (InputSpec& spec : input_specs) {
          spec.views_ = {left};
        }
        // Right column: each plot is an RGBA image (rasterized scatter) with a
        // TEXT title rendered on top of it.
        const auto add_plot = [&](const char* img_name, const char* title_name,
                                  const std::string& title, float oy) {
          InputSpec image(img_name, InputType::COLOR);
          image.views_ = {make_view(0.63F, oy, 0.35F, 0.44F)};
          input_specs.push_back(image);
          InputSpec text(title_name, InputType::TEXT);
          text.text_ = {title};
          text.color_ = {1.0F, 1.0F, 1.0F, 1.0F};
          text.priority_ = 2;
          text.views_ = {make_view(0.63F, oy, 0.35F, 0.44F)};
          input_specs.push_back(text);
        };
        add_plot(kRangePlotLayer, kRangeTitleLayer, "Range (m) vs time (s)",
                 0.04F);
        add_plot(kVelocityPlotLayer, kVelocityTitleLayer,
                 "Velocity (m/s) vs time (s)", 0.52F);

        plot = make_operator<RollingPlotOp>("rolling_plot");
        plot->window = cfg.plots.window;
        plot->frame_period = cfg.frame_period;
      }

      auto visualizer = make_operator<holoscan::ops::HolovizOp>(
          "holoviz", holoscan::Arg("tensors", input_specs),
          holoscan::Arg("color_lut", make_turbo_lut()),
          holoscan::Arg("window_title",
                        std::string("FMCW SISO - Range-Doppler + clusters")),
          holoscan::Arg("width", cfg.plots.enable ? 1200U : 800U),
          holoscan::Arg("height", 600U),
          holoscan::Arg("headless", cfg.headless));

      add_flow(doppler, to_image, {{"rd_maps", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      // The overlay needs the detector's cells, the cluster labels, and the
      // map dimensions; its per-color layers fan into the same receiver.
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cluster, overlay, {{"labels", "labels"}});
      add_flow(doppler, overlay, {{"rd_maps", "rd_maps"}});
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});

      // Rolling range/velocity plots: the per-cluster representative
      // (centroid) from reduction, plus the axis bins that fix each plot's Y
      // extent and resolve the representative bins to physical units.
      if (plot) {
        add_flow(reduction, plot, {{"rep_idx", "rep_idx"}});
        add_flow(range, plot, {{"range_bins", "range_bins"}});
        add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
        add_flow(plot, visualizer, {{"plots", "receivers"}});
      }
    }
  }
};

// Resolve the config path: the first CLI argument if given, otherwise
// fmcw_siso.yaml staged next to the executable.
std::filesystem::path resolve_config_path(int argc, char** argv) {
  if (argc > 1) {
    return argv[1];
  }
  return std::filesystem::path(argv[0]).parent_path() / "fmcw_siso.yaml";
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<FmcwSisoApp>();
  try {
    const std::filesystem::path config_path =
        resolve_config_path(argc, argv);
    app->cfg = load_config(config_path);
    // Headless rendering needs no display server, so the probe is moot there.
    if (app->cfg.visualize && !app->cfg.headless &&
        !app->cfg.skip_display_check && !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "fmcw_siso: visualize is enabled but no usable display was "
          "detected (Holoviz could not initialize a window); running "
          "without the RD-map window");
      app->cfg.visualize = false;
    }
    // Same file feeds the native config path: compose() pulls the
    // per-operator blocks via from_config().
    app->config(config_path.string());
    // run() is inside the try: a graph failure (e.g. an operator rejecting an
    // invalid parameter mid-tick) is rethrown out of run(); catching it here
    // exits cleanly with the message instead of terminating on an uncaught
    // exception, which the runtime reports as a process abort.
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("fmcw_siso: {}", e.what());
    return 1;
  }
  return 0;
}
