// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SIMO FMCW sparse angle-then-cluster pipeline:
//
//   SimoAdcSourceOp -> Range -> Clutter -> Doppler -> CFAR (NCI over RX)
//     -> SnapshotGather -> BeamscanEstimator -> FeatureAssembly [r,v,az,el]
//     -> ClusterDBSCAN -> DetectionReduction -> ClusterReportOp
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_simo/cpp/fmcw_simo [config.yaml]

#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
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

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
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
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

namespace {

struct PlotsConfig {
  bool enable{true};
  int64_t window{120};
};

struct AppConfig {
  radar::sim::SimScene scene;
  radar::ArrayGeometry array_geometry;
  PlotsConfig plots;
  int64_t num_frames{1};
  double frame_period{0.1};
  bool visualize{false};
  bool skip_display_check{false};
  bool headless{false};
  // Beamscan angular grid extents (radians), for the boresight view axes.
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};
};

// Uniform rectangular array (URA) in the y-z plane, normal to boresight (+x).
// The y aperture resolves azimuth and the z aperture resolves elevation, so a
// URA measures both (a single line array — along any axis or diagonal — has
// only 1D aperture and cannot separate the two). Element (iy, iz) sits at
// (0, iy * spacing, iz * spacing); rows vary elevation, columns azimuth.
radar::ArrayGeometry make_ura_geometry(int64_t az_elements, int64_t el_elements,
                                       double spacing_wl, double wavelength) {
  radar::ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = radar::ArrayGeometryType::kUra;
  geom.element_positions.reserve(
      static_cast<size_t>(az_elements * el_elements * 3));
  for (int64_t iz = 0; iz < el_elements; ++iz) {
    for (int64_t iy = 0; iy < az_elements; ++iy) {
      geom.element_positions.push_back(0.0);
      geom.element_positions.push_back(static_cast<double>(iy) * spacing_wl *
                                       wavelength);
      geom.element_positions.push_back(static_cast<double>(iz) * spacing_wl *
                                       wavelength);
    }
  }
  geom.validate();
  return geom;
}

void advance_scene(radar::sim::SimScene* scene, double dt) {
  for (radar::sim::PointTarget& target : scene->targets) {
    for (int axis = 0; axis < 3; ++axis) {
      target.position[axis] += target.velocity[axis] * dt;
    }
  }
}

AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;

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

  const YAML::Node rx = root["rx_array"];
  if (!rx) {
    throw std::runtime_error("rx_array section is required for SIMO");
  }
  // URA element counts along the azimuth (y) and elevation (z) axes.
  const int64_t az_elements = rx["az_elements"].as<int64_t>();
  const int64_t el_elements = rx["el_elements"].as<int64_t>();
  // `spacing` is the element pitch as a fraction of the carrier wavelength
  // (e.g. 0.5 = half-wavelength). The simulator computes the per-RX spatial
  // phase against the real carrier wavelength (lambda = c / f_c), so the
  // element positions and the beamscan geometry must both be in real meters at
  // that wavelength — a free-standing "wavelength" knob would desynchronize the
  // array from the signal model and alias the DOA.
  const double spacing = rx["spacing"].as<double>();
  const double wavelength = cfg.scene.waveform.wavelength();
  if (az_elements < 2 || el_elements < 2) {
    throw std::runtime_error(
        "SIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2 to "
        "observe both azimuth and elevation");
  }
  cfg.array_geometry =
      make_ura_geometry(az_elements, el_elements, spacing, wavelength);
  cfg.scene.rx_positions.clear();
  cfg.scene.rx_positions.reserve(
      static_cast<size_t>(az_elements * el_elements));
  for (int64_t iz = 0; iz < el_elements; ++iz) {
    for (int64_t iy = 0; iy < az_elements; ++iy) {
      cfg.scene.rx_positions.push_back(
          radar::sim::Vec3{0.0, static_cast<double>(iy) * spacing * wavelength,
                           static_cast<double>(iz) * spacing * wavelength});
    }
  }

  // Initial position is placed by (range, azimuth, elevation) in the
  // array-local frame: dir = (cos el cos az, cos el sin az, sin el),
  // position = range * dir. Velocity is an explicit (vx, vy, vz) vector in the
  // same frame (x = boresight, y = azimuth aperture, z = elevation aperture),
  // so its y/z components change the target's azimuth/elevation over time — the
  // targets move on all three axes, not just radially.
  for (const YAML::Node& t : root["targets"]) {
    const double range = t["range"].as<double>();
    const double azimuth = t["azimuth"].as<double>();
    const double elevation = t["elevation"].as<double>();
    const double amplitude = t["amplitude"].as<double>();
    const YAML::Node vel = t["velocity"];
    if (!vel || !vel.IsSequence() || vel.size() != 3) {
      throw std::runtime_error(
          "fmcw_simo: each target's velocity must be a 3-vector [vx, vy, vz] "
          "in m/s (array frame)");
    }
    const radar::sim::Vec3 dir{std::cos(elevation) * std::cos(azimuth),
                               std::cos(elevation) * std::sin(azimuth),
                               std::sin(elevation)};
    cfg.scene.targets.push_back(
        {radar::sim::Vec3{range * dir[0], range * dir[1], range * dir[2]},
         radar::sim::Vec3{vel[0].as<double>(), vel[1].as<double>(),
                          vel[2].as<double>()},
         {amplitude, 0.0}});
  }

  const YAML::Node noise = root["noise_snr_db"];
  if (noise && !noise.IsNull()) {
    cfg.scene.snr_db = noise.as<double>();
  }
  if (root["visualize"]) {
    cfg.visualize = root["visualize"].as<bool>();
  }
  if (root["skip_display_check"]) {
    cfg.skip_display_check = root["skip_display_check"].as<bool>();
  }
  if (root["headless"]) {
    cfg.headless = root["headless"].as<bool>();
  }
  if (const YAML::Node plots = root["plots"]) {
    if (plots["enable"]) {
      cfg.plots.enable = plots["enable"].as<bool>();
    }
    if (plots["window"]) {
      cfg.plots.window = plots["window"].as<int64_t>();
    }
  }
  if (cfg.plots.window <= 1) {
    throw std::runtime_error("plots.window must be > 1");
  }

  // Angular extents for the boresight view come from the beamscan grid, so the
  // axes match what the estimator can resolve. Pad a degenerate (single-value)
  // axis — the default el grid is one point — so it renders a visible span.
  const auto grid_extent = [](const YAML::Node& grid, double def_lo,
                              double def_hi, double* lo, double* hi) {
    if (grid && grid.IsSequence() && grid.size() > 0) {
      double mn = std::numeric_limits<double>::infinity();
      double mx = -std::numeric_limits<double>::infinity();
      for (const YAML::Node& v : grid) {
        const double x = v.as<double>();
        mn = std::min(mn, x);
        mx = std::max(mx, x);
      }
      *lo = mn;
      *hi = mx;
    } else {
      *lo = def_lo;
      *hi = def_hi;
    }
    if (*hi - *lo < 1e-6) {
      const double mid = 0.5 * (*lo + *hi);
      *lo = mid - 0.1;
      *hi = mid + 0.1;
    }
  };
  const YAML::Node be = root["beamscan_estimator"];
  grid_extent(be ? be["az_angles"] : YAML::Node(), -0.35, 0.35, &cfg.az_lo,
              &cfg.az_hi);
  grid_extent(be ? be["el_angles"] : YAML::Node(), -0.1, 0.1, &cfg.el_lo,
              &cfg.el_hi);
  return cfg;
}

}  // namespace

class SimoAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimoAdcSourceOp)
  SimoAdcSourceOp() = default;

  radar::sim::SimScene scene;
  double frame_period{0.1};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const radar::sim::SimResult sim = radar::sim::simulate(scene);
    if (sim.shape.size() != 3) {
      throw std::runtime_error(
          "SimoAdcSourceOp: expected rank-3 (N_rx, chirps, samples), "
          "got rank " + std::to_string(sim.shape.size()));
    }
    if (sim.shape[0] < 2) {
      throw std::runtime_error("SimoAdcSourceOp: SIMO requires N_rx >= 2");
    }

    std::vector<std::complex<float>> cube(sim.cube.size());
    for (size_t i = 0; i < sim.cube.size(); ++i) {
      cube[i] = std::complex<float>(static_cast<float>(sim.cube[i].real()),
                                    static_cast<float>(sim.cube[i].imag()));
    }

    auto tensor =
        radar::make_device_tensor(sim.shape, radar::complex_dtype<float>());
    if (cudaMemcpy(tensor->data(), cube.data(),
                   sizeof(std::complex<float>) * cube.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("SimoAdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    advance_scene(&scene, frame_period);
  }
};

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

class FmcwSimoApp : public holoscan::Application {
 public:
  AppConfig cfg;

  void compose() override {
    auto source = make_operator<SimoAdcSourceOp>(
        "adc_source",
        make_condition<holoscan::CountCondition>("frame_count",
                                                 cfg.num_frames),
        make_condition<holoscan::PeriodicCondition>(
            "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
    source->scene = cfg.scene;
    source->frame_period = cfg.frame_period;

    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", from_config("doppler_processor"));
    auto cfar = make_operator<radar::CFARDetectorOp<float>>(
        "cfar_detector", from_config("cfar_detector"));
    auto gather = make_operator<radar::SnapshotGatherOp<float>>(
        "snapshot_gather", from_config("snapshot_gather"));
    auto beamscan = make_operator<radar::BeamscanEstimatorOp<float>>(
        "beamscan_estimator", from_config("beamscan_estimator"),
        holoscan::Arg("array_geometry") = cfg.array_geometry);

    auto det_report = make_operator<DetectionReportOp>("detection_report");
    auto mask_sink = make_operator<NullSinkOp>("detection_mask_sink");
    auto noise_sink = make_operator<NullSinkOp>("noise_estimate_sink");
    auto spectrum_sink =
        make_operator<TensorMapDiscardOp>("angle_spectrum_discard");

    const std::vector<radar::AssemblyFeature> feature_columns{
        radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity,
        radar::AssemblyFeature::kAz, radar::AssemblyFeature::kEl};
    auto feature_assembly = make_operator<radar::FeatureAssemblyOp<float>>(
        "feature_assembly", holoscan::Arg("features") = feature_columns);
    auto cluster = make_operator<radar::ClusterDBSCANOp<float>>(
        "cluster_dbscan", from_config("cluster_dbscan"));
    auto cluster_report = make_operator<ClusterReportOp>("cluster_report");
    auto reduction = make_operator<radar::DetectionReductionOp<float>>(
        "detection_reduction", from_config("reduction"));

    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
    add_flow(doppler, cfar, {{"rd_maps", "rd_maps"}});

    add_flow(doppler, gather, {{"rd_maps", "rd_maps"}});
    add_flow(cfar, gather, {{"detection_idx", "detection_idx"}});
    add_flow(gather, beamscan, {{"snapshot", "snapshot"},
                                {"is_valid", "is_valid"}});
    add_flow(beamscan, feature_assembly,
             {{"aoa_estimates", "aoa_estimates"}});
    add_flow(beamscan, spectrum_sink, {{"angle_spectrum", "input"}});

    add_flow(cfar, det_report, {{"detection_idx", "detection_idx"},
                               {"detection_score", "detection_score"}});
    add_flow(range, det_report, {{"range_bins", "range_bins"}});
    add_flow(doppler, det_report, {{"velocity_bins", "velocity_bins"}});
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    add_flow(cfar, feature_assembly, {{"detection_idx", "detection_idx"}});
    add_flow(range, feature_assembly, {{"range_bins", "range_bins"}});
    add_flow(doppler, feature_assembly, {{"velocity_bins", "velocity_bins"}});
    add_flow(feature_assembly, cluster, {{"points", "points"}});
    add_flow(feature_assembly, cluster_report, {{"points", "points"}});
    add_flow(cluster, cluster_report, {{"labels", "labels"},
                                       {"sizes", "sizes"}});
    add_flow(cluster, reduction, {{"labels", "labels"}});
    add_flow(cfar, reduction, {{"detection_idx", "detection_idx"}});
    add_flow(feature_assembly, reduction, {{"points", "points"}});
    add_flow(reduction, cluster_report, {{"rep_idx", "rep_idx"}});
    add_flow(cfar, cluster_report, {{"detection_idx", "detection_idx"}});
    add_flow(range, cluster_report, {{"range_bins", "range_bins"}});
    add_flow(doppler, cluster_report, {{"velocity_bins", "velocity_bins"}});

    if (cfg.visualize) {
      using InputSpec = holoscan::ops::HolovizOp::InputSpec;
      using InputType = holoscan::ops::HolovizOp::InputType;
      auto ch_slice = make_operator<RdChannelSliceOp>("rd_channel_slice");
      ch_slice->channel = 0;
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      std::vector<InputSpec> input_specs;
      input_specs.emplace_back("rd_map", InputType::COLOR_LUT);
      const auto palette = cluster_palette();
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

      const auto make_view = [](float ox, float oy, float w, float h) {
        InputSpec::View v;
        v.offset_x_ = ox;
        v.offset_y_ = oy;
        v.width_ = w;
        v.height_ = h;
        return v;
      };
      std::shared_ptr<RollingPlotOp> plot;
      std::shared_ptr<BoresightViewOp> boresight;
      if (cfg.plots.enable) {
        const auto left = make_view(0.0F, 0.0F, 0.60F, 1.0F);
        for (InputSpec& spec : input_specs) {
          spec.views_ = {left};
        }
        // Three stacked panels fill the right column: range vs time, velocity
        // vs time, and the boresight az/el scatter.
        const auto add_plot = [&](const char* img_name, const char* title_name,
                                  const std::string& title, float oy,
                                  float h) {
          InputSpec image(img_name, InputType::COLOR);
          image.views_ = {make_view(0.63F, oy, 0.35F, h)};
          input_specs.push_back(image);
          InputSpec text(title_name, InputType::TEXT);
          text.text_ = {title};
          text.color_ = {1.0F, 1.0F, 1.0F, 1.0F};
          text.priority_ = 2;
          text.views_ = {make_view(0.63F, oy, 0.35F, h)};
          input_specs.push_back(text);
        };
        constexpr float kPanelH = 0.30F;
        add_plot(kRangePlotLayer, kRangeTitleLayer, "Range (m) vs time (s)",
                 0.02F, kPanelH);
        add_plot(kVelocityPlotLayer, kVelocityTitleLayer,
                 "Velocity (m/s) vs time (s)", 0.35F, kPanelH);
        add_plot(kBoresightPlotLayer, kBoresightTitleLayer,
                 "Boresight az/el (deg)", 0.68F, kPanelH);
        plot = make_operator<RollingPlotOp>("rolling_plot");
        plot->window = cfg.plots.window;
        plot->frame_period = cfg.frame_period;
        boresight = make_operator<BoresightViewOp>("boresight_view");
        boresight->az_lo = cfg.az_lo;
        boresight->az_hi = cfg.az_hi;
        boresight->el_lo = cfg.el_lo;
        boresight->el_hi = cfg.el_hi;
      }

      auto visualizer = make_operator<holoscan::ops::HolovizOp>(
          "holoviz", holoscan::Arg("tensors", input_specs),
          holoscan::Arg("color_lut", make_turbo_lut()),
          holoscan::Arg("window_title",
                        std::string("FMCW SIMO - RD (ch0) + clusters")),
          holoscan::Arg("width", cfg.plots.enable ? 1200U : 800U),
          holoscan::Arg("height", cfg.plots.enable ? 760U : 600U),
          holoscan::Arg("headless", cfg.headless));

      add_flow(doppler, ch_slice, {{"rd_maps", "rd_maps"}});
      add_flow(ch_slice, to_image, {{"rd_map", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cluster, overlay, {{"labels", "labels"}});
      add_flow(ch_slice, overlay, {{"rd_map", "rd_maps"}});
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
      if (plot) {
        add_flow(reduction, plot, {{"rep_idx", "rep_idx"}});
        add_flow(range, plot, {{"range_bins", "range_bins"}});
        add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
        add_flow(plot, visualizer, {{"plots", "receivers"}});
      }
      if (boresight) {
        add_flow(beamscan, boresight, {{"aoa_estimates", "aoa_estimates"}});
        add_flow(cluster, boresight, {{"labels", "labels"}});
        add_flow(boresight, visualizer, {{"plot", "receivers"}});
      }
    }
  }
};

std::filesystem::path resolve_config_path(int argc, char** argv) {
  if (argc > 1) {
    return argv[1];
  }
  return std::filesystem::path(argv[0]).parent_path() / "fmcw_simo.yaml";
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<FmcwSimoApp>();
  try {
    const std::filesystem::path config_path =
        resolve_config_path(argc, argv);
    app->cfg = load_config(config_path);
    if (app->cfg.visualize && !app->cfg.headless &&
        !app->cfg.skip_display_check && !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "fmcw_simo: visualize enabled but no display; running without GUI");
      app->cfg.visualize = false;
    }
    app->config(config_path.string());
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("fmcw_simo: {}", e.what());
    return 1;
  }
  return 0;
}
