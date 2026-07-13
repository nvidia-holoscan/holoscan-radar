// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SISO FMCW pipeline example: a synthetic ADC source feeding the split
// signal-processing chain
//
//   AdcSourceOp -> FMCWRangeProcessorOp -> ClutterFilterOp
//               -> FMCWDopplerProcessorOp -> PeakReportOp
//                                         [-> RdMapToImageOp -> HolovizOp]
//
// The source synthesizes one static clutter return and one moving target
// with the shared analytic point-target simulator
// (radar_common/sim/point_target_sim.hpp), emitting one cube (CPI) per tick
// and advancing the targets through space between ticks. The sink locates the
// range-Doppler peak per frame and reports it in physical units via the
// operators' range_bins / velocity_bins axis outputs, so the moving target's
// range is seen to sweep over the run.
//
// All configuration lives in fmcw_siso.yaml (staged next to the binary;
// override with a path as the first CLI argument): the per-operator blocks
// flow through Holoscan's native config()/from_config() path, and the
// cube/run/target sections drive the synthetic source.
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_siso/cpp/fmcw_siso [config.yaml]

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

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace radar = holoscan::radar;

namespace {

// Entry count of the Holoviz color lookup table; the RD-map image values
// index into it directly.
constexpr size_t kLutSize = 256;

// App-owned configuration: everything the synthetic source and run cadence
// need (see fmcw_siso.yaml). Operator parameters are NOT parsed here — the
// per-operator YAML blocks flow through Holoscan's from_config() in
// compose(), converted by radar_common/sdk/yaml_converters.hpp.
struct AppConfig {
  radar::sim::SimScene scene;  // waveform, cube dims, targets, noise
  int64_t num_frames{1};
  double frame_period{0.1};  // s, between emitted cubes (and target motion)
  bool visualize{false};     // live RD-map window (requires a GUI host)
  // Skip the startup display probe: saves its GLFW + Vulkan init/teardown,
  // but on a host without a usable display the failure then surfaces inside
  // the running graph and aborts the process instead of downgrading.
  bool skip_display_check{false};
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
  return cfg;
}

// Holoviz needs a reachable display server and a Vulkan surface; checking
// DISPLAY/WAYLAND_DISPLAY is not enough (the variable may be set with no
// server behind it). Probe by briefly initializing the Holoviz module — the
// same GLFW + Vulkan stack HolovizOp uses — on the main thread, where the
// failure is a catchable exception. Inside the running graph the same
// failure is thrown from a GXF worker thread and terminates the process,
// which is exactly what this probe exists to avoid.
bool gui_available() {
  namespace viz = holoscan::viz;
  try {
    viz::Init(64, 64, "fmcw_siso display probe");
    viz::Shutdown();
    return true;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_DEBUG("fmcw_siso: display probe failed: {}", e.what());
    return false;
  }
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

// Locates the range-Doppler peak and reports it in physical units via the
// upstream operators' axis outputs.
class PeakReportOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PeakReportOp)
  PeakReportOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("rd_maps");
    spec.input<holoscan::TensorMap>("range_bins");
    spec.input<holoscan::TensorMap>("velocity_bins");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    auto rd_map = single_tensor(op_input, "rd_maps");
    auto range_bins = single_tensor(op_input, "range_bins");
    auto velocity_bins = single_tensor(op_input, "velocity_bins");

    const auto shape = rd_map->shape();
    // SISO app: the host buffer below is sized from the trailing two axes
    // while the copy uses nbytes(); a batched (rank > 2) map would overflow
    // it. Index within the trailing slab when lifting this sink.
    if (shape.size() != 2) {
      throw std::runtime_error(
          "PeakReportOp: SISO sink expects a rank-2 rd_map, got rank " +
          std::to_string(shape.size()));
    }
    const int64_t num_doppler = shape[shape.size() - 2];
    const int64_t num_range = shape[shape.size() - 1];

    std::vector<std::complex<float>> rd(
        static_cast<size_t>(num_doppler * num_range));
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));
    cudaDeviceSynchronize();
    cudaMemcpy(rd.data(), rd_map->data(),
               static_cast<size_t>(rd_map->nbytes()),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ranges.data(), range_bins->data(),
               static_cast<size_t>(range_bins->nbytes()),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities.data(), velocity_bins->data(),
               static_cast<size_t>(velocity_bins->nbytes()),
               cudaMemcpyDeviceToHost);

    // The zero-Doppler column is suppressed by the clutter filter; report
    // the strongest remaining (moving-target) cell.
    size_t peak = 0;
    float peak_mag = 0.0F;
    for (size_t i = 0; i < rd.size(); ++i) {
      const float mag = std::abs(rd[i]);
      if (mag > peak_mag) {
        peak_mag = mag;
        peak = i;
      }
    }
    const size_t doppler_bin = peak / static_cast<size_t>(num_range);
    const size_t range_bin = peak % static_cast<size_t>(num_range);
    HOLOSCAN_LOG_INFO(
        "Peak |RD| = {:.1f} at range bin {} ({:.2f} m), Doppler bin {} "
        "({:+.2f} m/s)",
        peak_mag, range_bin, ranges[range_bin], doppler_bin,
        velocities[doppler_bin]);
  }

 private:
  static std::shared_ptr<holoscan::Tensor> single_tensor(
      holoscan::InputContext& op_input, const char* port) {
    auto maybe = op_input.receive<holoscan::TensorMap>(port);
    if (!maybe || maybe->size() != 1) {
      throw std::runtime_error(std::string("PeakReportOp: expected one "
                                           "tensor on port ") + port);
    }
    return maybe->begin()->second;
  }
};

// Converts the complex RD map into a log-magnitude float32 image for
// Holoviz (rendered through the turbo color lookup table). Per frame:
// |RD| in dB, mapped between that frame's measured noise floor and its
// peak onto [0, kLutSize - 1] — HolovizOp's LUT is non-normalized, so the
// pixel value is used as an integer index into the table.
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
          "RdMapToImageOp: SISO app expects a rank-2 rd_map, got rank " +
          std::to_string(shape.size()));
    }
    const int64_t num_doppler = shape[0];
    const int64_t num_range = shape[1];

    std::vector<std::complex<float>> rd(
        static_cast<size_t>(num_doppler * num_range));
    cudaDeviceSynchronize();
    if (cudaMemcpy(rd.data(), rd_map->data(),
                   static_cast<size_t>(rd_map->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("RdMapToImageOp: D2H copy failed");
    }

    // Log magnitude, normalized to [0, 1] between this frame's noise floor
    // and peak. The floor is estimated as the median dB value (the map is
    // overwhelmingly noise cells, so the median tracks the noise floor and
    // is insensitive to the few target/clutter cells). Per-frame
    // normalization keeps the target visible as absolute levels drift.
    std::vector<float> image(rd.size());
    float peak_db = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < rd.size(); ++i) {
      image[i] = 20.0F * std::log10(std::abs(rd[i]) + 1e-20F);
      peak_db = std::max(peak_db, image[i]);
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

// kLutSize-entry RGBA lookup table for the turbo colormap, built from the
// published 5th-order polynomial fit of the Google turbo palette.
std::vector<std::vector<float>> make_turbo_lut() {
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
    auto sink = make_operator<PeakReportOp>("peak_report");

    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
    add_flow(doppler, sink, {{"rd_maps", "rd_maps"}});
    add_flow(range, sink, {{"range_bins", "range_bins"}});
    add_flow(doppler, sink, {{"velocity_bins", "velocity_bins"}});

    // Optional live RD-map window (cfg.visualize; main() has already
    // downgraded it with a warning on GUI-less hosts). The converter's
    // single-channel float32 tensor is colored via the turbo LUT.
    if (cfg.visualize) {
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      std::vector<holoscan::ops::HolovizOp::InputSpec> input_specs;
      input_specs.emplace_back("rd_map",
                               holoscan::ops::HolovizOp::InputType::COLOR_LUT);
      auto visualizer = make_operator<holoscan::ops::HolovizOp>(
          "holoviz", holoscan::Arg("tensors", input_specs),
          holoscan::Arg("color_lut", make_turbo_lut()),
          holoscan::Arg("window_title",
                        std::string("FMCW SISO - Range-Doppler")),
          holoscan::Arg("width", 800U), holoscan::Arg("height", 600U));
      add_flow(doppler, to_image, {{"rd_maps", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
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
    if (app->cfg.visualize && !app->cfg.skip_display_check &&
        !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "fmcw_siso: visualize is enabled but no usable display was "
          "detected (Holoviz could not initialize a window); running "
          "without the RD-map window");
      app->cfg.visualize = false;
    }
    // Same file feeds the native config path: compose() pulls the
    // per-operator blocks via from_config().
    app->config(config_path.string());
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("fmcw_siso: {}", e.what());
    return 1;
  }
  app->run();
  return 0;
}
