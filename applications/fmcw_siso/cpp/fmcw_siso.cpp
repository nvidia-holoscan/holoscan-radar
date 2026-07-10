// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SISO FMCW pipeline example: a synthetic ADC source feeding the split
// signal-processing chain
//
//   AdcSourceOp -> FMCWRangeProcessorOp -> ClutterFilterOp
//               -> FMCWDopplerProcessorOp -> PeakReportOp
//
// The source synthesizes one static clutter return and one moving target
// with the shared analytic point-target simulator
// (radar_common/sim/point_target_sim.hpp), emitting one cube (CPI) per tick
// and advancing the targets through space between ticks. The sink locates the
// range-Doppler peak per frame and reports it in physical units via the
// operators' range_bins / velocity_bins axis outputs, so the moving target's
// range is seen to sweep over the run.
//
// Waveform, cube dimensions, run cadence, targets, and clutter method are
// read from fmcw_siso.yaml (staged next to the binary; override with a path
// as the first CLI argument).
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_siso/cpp/fmcw_siso [config.yaml]

#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace radar = holoscan::radar;

namespace {

// Parsed application configuration (see fmcw_siso.yaml).
struct AppConfig {
  radar::FMCWWaveformDescriptor waveform;
  radar::ClutterFilterMethod clutter_method{
      radar::ClutterFilterMethod::kMeanSubtraction};
  radar::sim::SimScene scene;  // waveform, cube dims, targets, noise
  int64_t num_frames{1};
  double frame_period{0.1};  // s, between emitted cubes (and target motion)
};

radar::ClutterFilterMethod parse_clutter_method(const std::string& name) {
  using radar::ClutterFilterMethod;
  for (auto method : {ClutterFilterMethod::kMeanSubtraction,
                      ClutterFilterMethod::kSingleDelay,
                      ClutterFilterMethod::kDoubleDelay}) {
    if (name == radar::to_string(method)) {
      return method;
    }
  }
  throw std::runtime_error("unknown clutter_filter.method '" + name + "'");
}

// Load and validate the YAML configuration.
AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;

  const YAML::Node wf = root["waveform"];
  cfg.waveform.carrier_frequency = wf["carrier_frequency"].as<double>();
  cfg.waveform.bandwidth = wf["bandwidth"].as<double>();
  cfg.waveform.chirp_duration = wf["chirp_duration"].as<double>();
  cfg.waveform.sample_rate = wf["sample_rate"].as<double>();
  cfg.waveform.chirp_repetition_interval =
      wf["chirp_repetition_interval"].as<double>();

  cfg.clutter_method =
      parse_clutter_method(root["clutter_filter"]["method"].as<std::string>());

  cfg.num_frames = root["run"]["num_frames"].as<int64_t>();
  cfg.frame_period = root["run"]["frame_period"].as<double>();

  // Mirror the waveform into the simulator's decoupled SimWaveform. slope() is
  // the source of truth, so a bandwidth- or slope-spelled descriptor both map
  // to the same beat rate.
  cfg.scene.waveform.carrier_frequency = cfg.waveform.carrier_frequency;
  cfg.scene.waveform.bandwidth =
      cfg.waveform.slope() * cfg.waveform.chirp_duration;
  cfg.scene.waveform.chirp_duration = cfg.waveform.chirp_duration;
  cfg.scene.waveform.sample_rate = cfg.waveform.sample_rate;
  cfg.scene.waveform.chirp_repetition_interval =
      cfg.waveform.chirp_repetition_interval;
  cfg.scene.num_chirps = root["cube"]["num_chirps"].as<int64_t>();
  cfg.scene.samples_per_chirp = root["cube"]["samples_per_chirp"].as<int64_t>();

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

class FmcwSisoApp : public holoscan::Application {
 public:
  AppConfig cfg;  // set before run()

  void compose() override {
    using holoscan::Arg;

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

    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", Arg("waveform") = cfg.waveform);
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter", Arg("method") = cfg.clutter_method);
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", Arg("waveform") = cfg.waveform);
    auto sink = make_operator<PeakReportOp>("peak_report");

    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
    add_flow(doppler, sink, {{"rd_maps", "rd_maps"}});
    add_flow(range, sink, {{"range_bins", "range_bins"}});
    add_flow(doppler, sink, {{"velocity_bins", "velocity_bins"}});
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
    app->cfg = load_config(resolve_config_path(argc, argv));
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("fmcw_siso: {}", e.what());
    return 1;
  }
  app->run();
  return 0;
}
