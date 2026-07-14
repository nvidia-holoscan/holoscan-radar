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
//                                                  -> ClusterReportOp
//                                         [-> RdMapToImageOp   -> HolovizOp]
//                                         [-> ClusterOverlayOp -> HolovizOp]
//                                              (crosses colored by cluster)
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
// for the signal-processing ops flow through Holoscan's native
// config()/from_config() path; the CFAR block is parsed into AppConfig (no
// YAML::convert for CfarVariant in radar_common yet) and applied as Args.
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

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
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

namespace radar = holoscan::radar;

namespace {

// Entry count of the Holoviz color lookup table; the RD-map image values
// index into it directly.
constexpr size_t kLutSize = 256;

// App-owned configuration: everything the synthetic source and run cadence
// need (see fmcw_siso.yaml). Operator parameters for the signal-processing
// ops are NOT parsed here — those YAML blocks flow through Holoscan's
// from_config() in compose(). CFAR parameters are parsed here because
// CfarVariant has no radar_common YAML::convert yet.
struct CfarConfig {
  radar::CfarVariant variant{radar::CfarVariant::kCA};
  std::vector<int64_t> guard_cells{1, 1};
  std::vector<int64_t> training_cells{2, 2};
  double pfa{1e-4};
  int64_t os_rank{0};
};

// Clustering-stage configuration: the FeatureAssemblyOp feature columns plus
// the ClusterDBSCANOp parameters. Like CfarConfig, this is app-parsed rather
// than routed through from_config() — AssemblyFeature has no radar_common
// YAML::convert, so the whole stage is kept as one app-validated block and
// applied as Args in compose(). The defaults assemble a [range, velocity]
// matrix (physical units) and cluster it with a per-axis neighborhood.
struct ClusteringConfig {
  std::vector<radar::AssemblyFeature> features{
      radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity};
  std::vector<double> epsilon{3.0, 2.0};  // per-feature radius: m, m/s
  int64_t min_points{2};                  // <= 0 → 2*num_features
  int64_t max_clusters{0};                // <= 0 → uncapped
};

struct AppConfig {
  radar::sim::SimScene scene;  // waveform, cube dims, targets, noise
  CfarConfig cfar;
  ClusteringConfig clustering;
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

  // CFARDetectorOp parameters (app-parsed; applied as Args in compose()).
  const YAML::Node cfar = root["cfar_detector"];
  if (cfar && !cfar.IsNull()) {
    if (cfar["cfar_variant"]) {
      const std::string spelling = cfar["cfar_variant"].as<std::string>();
      bool matched = false;
      for (radar::CfarVariant v : {radar::CfarVariant::kCA,
                                   radar::CfarVariant::kGO,
                                   radar::CfarVariant::kSO,
                                   radar::CfarVariant::kOS}) {
        if (spelling == radar::to_string(v)) {
          cfg.cfar.variant = v;
          matched = true;
          break;
        }
      }
      if (!matched) {
        throw std::runtime_error(
            "cfar_detector.cfar_variant: unknown value '" + spelling +
            "' (accepted: CA, GO, SO, OS)");
      }
    }
    if (cfar["guard_cells"]) {
      cfg.cfar.guard_cells = cfar["guard_cells"].as<std::vector<int64_t>>();
    }
    if (cfar["training_cells"]) {
      cfg.cfar.training_cells =
          cfar["training_cells"].as<std::vector<int64_t>>();
    }
    if (cfar["pfa"]) {
      cfg.cfar.pfa = cfar["pfa"].as<double>();
    }
    if (cfar["os_rank"]) {
      cfg.cfar.os_rank = cfar["os_rank"].as<int64_t>();
    }
  }
  if (cfg.cfar.guard_cells.size() != 2 || cfg.cfar.training_cells.size() != 2) {
    throw std::runtime_error(
        "cfar_detector.guard_cells and training_cells must each have "
        "exactly 2 entries (doppler, range)");
  }
  if (!(cfg.cfar.pfa > 0.0 && cfg.cfar.pfa < 1.0)) {
    throw std::runtime_error("cfar_detector.pfa must be in (0, 1)");
  }

  // Clustering stage (app-parsed; applied as Args in compose()). All keys are
  // optional — the ClusteringConfig defaults hold when a key or the whole
  // block is omitted.
  const YAML::Node clustering = root["clustering"];
  if (clustering && !clustering.IsNull()) {
    if (clustering["features"]) {
      std::vector<radar::AssemblyFeature> features;
      for (const YAML::Node& f : clustering["features"]) {
        const std::string spelling = f.as<std::string>();
        bool matched = false;
        for (radar::AssemblyFeature v :
             {radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity,
              radar::AssemblyFeature::kScore, radar::AssemblyFeature::kAz,
              radar::AssemblyFeature::kEl, radar::AssemblyFeature::kDopplerBin,
              radar::AssemblyFeature::kRangeBin}) {
          if (spelling == radar::to_string(v)) {
            features.push_back(v);
            matched = true;
            break;
          }
        }
        if (!matched) {
          throw std::runtime_error(
              "clustering.features: unknown token '" + spelling +
              "' (accepted: range, velocity, score, az, el, doppler_bin, "
              "range_bin)");
        }
      }
      if (features.empty()) {
        throw std::runtime_error(
            "clustering.features must list at least one feature");
      }
      cfg.clustering.features = features;
    }
    if (clustering["epsilon"]) {
      cfg.clustering.epsilon =
          clustering["epsilon"].as<std::vector<double>>();
    }
    if (clustering["min_points"]) {
      cfg.clustering.min_points = clustering["min_points"].as<int64_t>();
    }
    if (clustering["max_clusters"]) {
      cfg.clustering.max_clusters = clustering["max_clusters"].as<int64_t>();
    }
  }
  // epsilon reaches ClusterDBSCANOp, which rejects a bad value mid-graph;
  // validate here so the offending key surfaces at startup instead. Length
  // must be 1 (isotropic) or match the feature count (per-axis), and every
  // value finite and > 0.
  if (cfg.clustering.epsilon.empty()) {
    throw std::runtime_error("clustering.epsilon must be non-empty");
  }
  if (cfg.clustering.epsilon.size() != 1 &&
      cfg.clustering.epsilon.size() != cfg.clustering.features.size()) {
    throw std::runtime_error(
        "clustering.epsilon must have length 1 or match the number of "
        "features (" +
        std::to_string(cfg.clustering.features.size()) + ")");
  }
  for (const double e : cfg.clustering.epsilon) {
    if (!(std::isfinite(e) && e > 0.0)) {
      throw std::runtime_error(
          "clustering.epsilon values must be finite and > 0");
    }
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

// Reports CFAR detections in physical units via the upstream operators'
// axis outputs. detection_score is the dense CFAR map; each detection_idx
// row indexes into it for the per-cell strength (score > 1 ⇔ detected).
class DetectionReportOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DetectionReportOp)
  DetectionReportOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("detection_idx");
    spec.input<holoscan::TensorMap>("detection_score");
    spec.input<holoscan::TensorMap>("range_bins");
    spec.input<holoscan::TensorMap>("velocity_bins");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    auto detection_idx = single_tensor(op_input, "detection_idx");
    auto detection_score = single_tensor(op_input, "detection_score");
    auto range_bins = single_tensor(op_input, "range_bins");
    auto velocity_bins = single_tensor(op_input, "velocity_bins");

    const auto idx_shape = detection_idx->shape();
    const auto score_shape = detection_score->shape();
    if (idx_shape.size() != 2 || idx_shape[1] != 2) {
      throw std::runtime_error(
          "DetectionReportOp: detection_idx must be (N, 2), got rank " +
          std::to_string(idx_shape.size()));
    }
    if (score_shape.size() != 2) {
      throw std::runtime_error(
          "DetectionReportOp: SISO expects a rank-2 detection_score, got "
          "rank " +
          std::to_string(score_shape.size()));
    }
    const int64_t num_detections = idx_shape[0];
    const int64_t num_doppler = score_shape[0];
    const int64_t num_range = score_shape[1];

    std::vector<int64_t> idx(static_cast<size_t>(num_detections * 2));
    std::vector<float> scores(
        static_cast<size_t>(num_doppler * num_range));
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));

    const auto require_bytes =
        [](const char* name, const std::shared_ptr<holoscan::Tensor>& t,
           size_t expected) {
          if (static_cast<size_t>(t->nbytes()) != expected) {
            throw std::runtime_error(
                std::string("DetectionReportOp: ") + name +
                " byte size mismatch (expected " + std::to_string(expected) +
                ", got " + std::to_string(t->nbytes()) + ")");
          }
        };
    require_bytes("detection_idx", detection_idx,
                  sizeof(int64_t) * idx.size());
    require_bytes("detection_score", detection_score,
                  sizeof(float) * scores.size());
    require_bytes("range_bins", range_bins, sizeof(float) * ranges.size());
    require_bytes("velocity_bins", velocity_bins,
                  sizeof(float) * velocities.size());

    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: device synchronize failed");
    }
    if (cudaMemcpy(idx.data(), detection_idx->data(),
                   static_cast<size_t>(detection_idx->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: detection_idx D2H copy failed");
    }
    if (cudaMemcpy(scores.data(), detection_score->data(),
                   static_cast<size_t>(detection_score->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: detection_score D2H copy failed");
    }
    if (cudaMemcpy(ranges.data(), range_bins->data(),
                   static_cast<size_t>(range_bins->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: range_bins D2H copy failed");
    }
    if (cudaMemcpy(velocities.data(), velocity_bins->data(),
                   static_cast<size_t>(velocity_bins->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionReportOp: velocity_bins D2H copy failed");
    }

    HOLOSCAN_LOG_INFO("CFAR detections: {}", num_detections);
    constexpr int64_t kMaxLogged = 16;
    const int64_t n_log = std::min(num_detections, kMaxLogged);
    for (int64_t i = 0; i < n_log; ++i) {
      const int64_t doppler_bin = idx[static_cast<size_t>(i * 2 + 0)];
      const int64_t range_bin = idx[static_cast<size_t>(i * 2 + 1)];
      if (doppler_bin < 0 || doppler_bin >= num_doppler || range_bin < 0 ||
          range_bin >= num_range) {
        throw std::runtime_error(
            "DetectionReportOp: detection_idx out of map bounds");
      }
      const float score =
          scores[static_cast<size_t>(doppler_bin * num_range + range_bin)];
      HOLOSCAN_LOG_INFO(
          "  [{}] score={:.2f} range bin {} ({:.2f} m), Doppler bin {} "
          "({:+.2f} m/s)",
          i, score, range_bin, ranges[static_cast<size_t>(range_bin)],
          doppler_bin, velocities[static_cast<size_t>(doppler_bin)]);
    }
    if (num_detections > kMaxLogged) {
      HOLOSCAN_LOG_INFO("  ... {} more detections not logged",
                        num_detections - kMaxLogged);
    }
  }

 private:
  static std::shared_ptr<holoscan::Tensor> single_tensor(
      holoscan::InputContext& op_input, const char* port) {
    auto maybe = op_input.receive<holoscan::TensorMap>(port);
    if (!maybe || maybe->size() != 1) {
      throw std::runtime_error(std::string("DetectionReportOp: expected one "
                                           "tensor on port ") +
                               port);
    }
    return maybe->begin()->second;
  }
};

// Reports the DBSCAN clusters formed over the CFAR detections. `points` is the
// FeatureAssemblyOp feature matrix (this app assembles [range (m),
// velocity (m/s)]); `labels` is the per-point cluster id (noise = -1,
// row-aligned with points) and `sizes` the per-cluster point count. Logs the
// cluster count, the noise-point count, and each cluster's size and centroid
// in physical units — so the adjacent CFAR cells lit by one target read as a
// single (range, velocity) estimate.
class ClusterReportOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ClusterReportOp)
  ClusterReportOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("points");
    spec.input<holoscan::TensorMap>("labels");
    spec.input<holoscan::TensorMap>("sizes");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const std::string who = "ClusterReportOp";
    auto points = radar::receive_single(op_input, "points", who);
    auto labels = radar::receive_single(op_input, "labels", who);
    auto sizes = radar::receive_single(op_input, "sizes", who);

    const auto pshape = points->shape();
    // This app fixes the FeatureAssemblyOp columns to [range, velocity]; the
    // centroid reporting below assumes that 2-column layout.
    if (pshape.size() != 2 || pshape[1] != 2) {
      throw std::runtime_error(
          "ClusterReportOp: points must be (N, 2) [range, velocity], got "
          "rank " +
          std::to_string(pshape.size()));
    }
    const int64_t num_points = pshape[0];
    const auto lshape = labels->shape();
    if (lshape.size() != 1 || lshape[0] != num_points) {
      throw std::runtime_error(
          "ClusterReportOp: labels must be (N,) row-aligned with points");
    }
    const auto sshape = sizes->shape();
    if (sshape.size() != 1) {
      throw std::runtime_error("ClusterReportOp: sizes must be rank-1");
    }
    const int64_t num_clusters = sshape[0];

    std::vector<float> pts(static_cast<size_t>(num_points * 2));
    std::vector<int64_t> lab(static_cast<size_t>(num_points));
    std::vector<int64_t> sz(static_cast<size_t>(num_clusters));

    const auto require_bytes =
        [](const char* name, const std::shared_ptr<holoscan::Tensor>& t,
           size_t expected) {
          if (static_cast<size_t>(t->nbytes()) != expected) {
            throw std::runtime_error(
                std::string("ClusterReportOp: ") + name +
                " byte size mismatch (expected " + std::to_string(expected) +
                ", got " + std::to_string(t->nbytes()) + ")");
          }
        };
    require_bytes("points", points, sizeof(float) * pts.size());
    require_bytes("labels", labels, sizeof(int64_t) * lab.size());
    require_bytes("sizes", sizes, sizeof(int64_t) * sz.size());

    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("ClusterReportOp: device synchronize failed");
    }
    // Guard the zero-length copies: an empty tick (no detections) yields empty
    // tensors whose data() may be null.
    const auto copy_d2h = [](const char* name, void* dst,
                             const std::shared_ptr<holoscan::Tensor>& src) {
      if (src->nbytes() == 0) {
        return;
      }
      if (cudaMemcpy(dst, src->data(), static_cast<size_t>(src->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error(std::string("ClusterReportOp: ") + name +
                                 " D2H copy failed");
      }
    };
    copy_d2h("points", pts.data(), points);
    copy_d2h("labels", lab.data(), labels);
    copy_d2h("sizes", sz.data(), sizes);

    // Accumulate per-cluster centroids from the points, keyed by label; noise
    // (-1) is counted but excluded from any cluster.
    std::vector<double> sum_range(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_velocity(static_cast<size_t>(num_clusters), 0.0);
    int64_t noise = 0;
    for (int64_t i = 0; i < num_points; ++i) {
      const int64_t c = lab[static_cast<size_t>(i)];
      if (c < 0) {
        ++noise;
        continue;
      }
      if (c >= num_clusters) {
        throw std::runtime_error(
            "ClusterReportOp: label out of range for sizes");
      }
      sum_range[static_cast<size_t>(c)] += pts[static_cast<size_t>(i * 2 + 0)];
      sum_velocity[static_cast<size_t>(c)] +=
          pts[static_cast<size_t>(i * 2 + 1)];
    }

    HOLOSCAN_LOG_INFO("DBSCAN clusters: {} ({} points, {} noise)",
                      num_clusters, num_points, noise);
    constexpr int64_t kMaxLogged = 16;
    const int64_t n_log = std::min(num_clusters, kMaxLogged);
    for (int64_t c = 0; c < n_log; ++c) {
      const int64_t n = sz[static_cast<size_t>(c)];
      const double inv = n > 0 ? 1.0 / static_cast<double>(n) : 0.0;
      HOLOSCAN_LOG_INFO(
          "  cluster {}: {} pts, centroid range {:.2f} m, velocity "
          "{:+.2f} m/s",
          c, n, sum_range[static_cast<size_t>(c)] * inv,
          sum_velocity[static_cast<size_t>(c)] * inv);
    }
    if (num_clusters > kMaxLogged) {
      HOLOSCAN_LOG_INFO("  ... {} more clusters not logged",
                        num_clusters - kMaxLogged);
    }
  }
};

// Drain a CFAR dense output without deserializing it as a TensorMap. The
// bool detection_mask path trips Holoscan 4.3's unsupported-DLPack-bool
// error when received as a TensorMap; raw Entity receive sidesteps that.
class NullSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NullSinkOp)
  NullSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    (void)op_input.receive<holoscan::gxf::Entity>("input");
  }
};

// Converts the complex RD map into a log-magnitude float32 image for
// Holoviz (rendered through the turbo color lookup table). Per frame:
// |RD| in dB, mapped between that frame's measured noise floor and its
// peak onto [0, kLutSize - 1] — HolovizOp's LUT is non-normalized, so the
// pixel value is used as an integer index into the table. Detections and
// their clusters are drawn separately, as a colored crosses geometry layer
// on top of this image (see ClusterOverlayOp), so the image stays a clean
// turbo map.
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
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("RdMapToImageOp: device synchronize failed");
    }
    if (cudaMemcpy(rd.data(), rd_map->data(),
                   static_cast<size_t>(rd_map->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("RdMapToImageOp: rd_map D2H copy failed");
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

// Overlay geometry-layer names, shared by ClusterOverlayOp (as emitted tensor
// keys) and compose() (as HolovizOp InputSpec names). Detections are split
// across kNumClusterColors CROSSES layers, one per palette color; a detection
// is assigned to layer (cluster_id % kNumClusterColors), so distinct clusters
// in a frame get distinct colors (clusters beyond the palette wrap). Noise
// points (label -1) go to their own muted layer.
constexpr int kNumClusterColors = 8;
constexpr const char* kNoiseLayerName = "noise";
inline std::string cluster_layer_name(int color) {
  return "cluster_" + std::to_string(color);
}

// Turns the CFAR detections and their DBSCAN cluster labels into Holoviz
// CROSSES geometry layers drawn over the range-Doppler image. Consumes the
// detector's detection_idx (the (doppler_bin, range_bin) cells), the matching
// per-point cluster labels, and the RD map (for its dimensions), and emits one
// cross per detection at the matching image location, bucketed by cluster
// color. Holoviz geometry is in normalized [0, 1] coordinates: x runs
// left→right across the range axis, y top→bottom across the Doppler axis — the
// same layout RdMapToImageOp gives the image, so the crosses land on their
// cells. labels are row-aligned with detection_idx (FeatureAssemblyOp keeps
// points 1:1 with the cells), so no bin↔physical inversion is needed here.
class ClusterOverlayOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ClusterOverlayOp)
  ClusterOverlayOp() = default;

  // Half-size of each cross, as a fraction of the image extent.
  static constexpr float kCrossSize = 0.02F;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("detection_idx");
    spec.input<holoscan::TensorMap>("labels");
    spec.input<holoscan::TensorMap>("rd_maps");
    spec.output<holoscan::TensorMap>("overlay");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const std::string who = "ClusterOverlayOp";
    auto idx = radar::receive_single(op_input, "detection_idx", who);
    auto labels = radar::receive_single(op_input, "labels", who);
    auto rd_map = radar::receive_single(op_input, "rd_maps", who);

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
    // labels is int64 (num_points,), row-aligned with detection_idx.
    const auto lab_shape = labels->shape();
    if (lab_shape.size() != 1 || lab_shape[0] != num_points) {
      throw std::runtime_error(
          "ClusterOverlayOp: labels must be (num_points,) row-aligned with "
          "detection_idx");
    }

    std::vector<int64_t> bins(static_cast<size_t>(num_points * 2));
    std::vector<int64_t> lab(static_cast<size_t>(num_points));
    const size_t idx_bytes = sizeof(int64_t) * bins.size();
    const size_t lab_bytes = sizeof(int64_t) * lab.size();
    if (static_cast<size_t>(idx->nbytes()) != idx_bytes) {
      throw std::runtime_error(
          "ClusterOverlayOp: detection_idx byte size mismatch");
    }
    if (static_cast<size_t>(labels->nbytes()) != lab_bytes) {
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
      if (cudaMemcpy(lab.data(), labels->data(), lab_bytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("ClusterOverlayOp: labels D2H copy failed");
      }
    }

    // Bucket detections into the palette layers by cluster id; the last bucket
    // is noise. Each entry is a cross (x, y, size) in normalized coordinates,
    // cell centers at (bin + 0.5) / count.
    std::vector<std::vector<float>> buckets(kNumClusterColors + 1);
    for (int64_t i = 0; i < num_points; ++i) {
      const int64_t doppler_bin = bins[static_cast<size_t>(i * 2 + 0)];
      const int64_t range_bin = bins[static_cast<size_t>(i * 2 + 1)];
      if (doppler_bin < 0 || doppler_bin >= num_doppler || range_bin < 0 ||
          range_bin >= num_range) {
        throw std::runtime_error(
            "ClusterOverlayOp: detection_idx out of map bounds");
      }
      const int64_t label = lab[static_cast<size_t>(i)];
      const size_t bucket =
          label >= 0 ? static_cast<size_t>(label % kNumClusterColors)
                     : static_cast<size_t>(kNumClusterColors);
      buckets[bucket].push_back((static_cast<float>(range_bin) + 0.5F) /
                                static_cast<float>(num_range));
      buckets[bucket].push_back((static_cast<float>(doppler_bin) + 0.5F) /
                                static_cast<float>(num_doppler));
      buckets[bucket].push_back(kCrossSize);
    }

    // Emit every layer each tick (0-row when empty) so the fixed HolovizOp
    // InputSpec list always has its matching tensor.
    constexpr int64_t kCrossComponents = 3;
    holoscan::TensorMap out;
    for (int c = 0; c <= kNumClusterColors; ++c) {
      const std::string name =
          c < kNumClusterColors ? cluster_layer_name(c) : kNoiseLayerName;
      const int64_t n =
          static_cast<int64_t>(buckets[static_cast<size_t>(c)].size()) /
          kCrossComponents;
      auto tensor = radar::make_device_tensor({n, kCrossComponents},
                                              radar::real_dtype<float>());
      if (n > 0 &&
          cudaMemcpy(tensor->data(), buckets[static_cast<size_t>(c)].data(),
                     sizeof(float) * buckets[static_cast<size_t>(c)].size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("ClusterOverlayOp: H2D upload failed");
      }
      out.insert({name, tensor});
    }
    op_output.emit(out, "overlay");
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

// Categorical RGBA palette for the cluster-overlay CROSSES layers, one entry
// per color (index i colors layer cluster_i). Chosen to read against the
// turbo image; size must equal kNumClusterColors.
std::vector<std::vector<float>> cluster_palette() {
  return {
      {1.0F, 0.0F, 1.0F, 1.0F},  // magenta
      {1.0F, 1.0F, 1.0F, 1.0F},  // white
      {0.0F, 1.0F, 1.0F, 1.0F},  // cyan
      {1.0F, 0.5F, 0.0F, 1.0F},  // orange
      {0.6F, 0.2F, 1.0F, 1.0F},  // violet
      {1.0F, 0.9F, 0.1F, 1.0F},  // yellow
      {1.0F, 0.3F, 0.6F, 1.0F},  // pink
      {0.2F, 1.0F, 0.4F, 1.0F},  // green
  };
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
    // CFAR args are applied explicitly: CfarVariant has no YAML::convert in
    // radar_common yet (parsed into AppConfig.cfar in load_config()).
    auto cfar = make_operator<radar::CFARDetectorOp<float>>(
        "cfar_detector",
        holoscan::Arg("cfar_variant") = cfg.cfar.variant,
        holoscan::Arg("guard_cells") = cfg.cfar.guard_cells,
        holoscan::Arg("training_cells") = cfg.cfar.training_cells,
        holoscan::Arg("pfa") = cfg.cfar.pfa,
        holoscan::Arg("os_rank") = cfg.cfar.os_rank);
    auto sink = make_operator<DetectionReportOp>("detection_report");
    // Every CFAR output needs a receiver or the operator never ticks.
    auto mask_sink = make_operator<NullSinkOp>("detection_mask_sink");
    auto noise_sink = make_operator<NullSinkOp>("noise_estimate_sink");

    // Clustering branch: assemble a [range, velocity] feature matrix from the
    // CFAR cells (physical units, via the range/Doppler axis outputs) and
    // density-cluster it. `features` is applied as an Arg (AssemblyFeature has
    // no YAML::convert, mirroring CFAR); the DBSCAN parameters likewise come
    // from the app-parsed clustering block.
    auto feature_assembly = make_operator<radar::FeatureAssemblyOp<float>>(
        "feature_assembly",
        holoscan::Arg("features") = cfg.clustering.features);
    auto cluster = make_operator<radar::ClusterDBSCANOp<float>>(
        "cluster_dbscan",
        holoscan::Arg("epsilon") = cfg.clustering.epsilon,
        holoscan::Arg("min_points") = cfg.clustering.min_points,
        holoscan::Arg("max_clusters") = cfg.clustering.max_clusters);
    auto cluster_report = make_operator<ClusterReportOp>("cluster_report");

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

    // Optional live RD-map window (cfg.visualize; main() has already
    // downgraded it with a warning on GUI-less hosts). Turbo-mapped |RD|
    // (range = x, Doppler = y) with the CFAR detections overlaid as crosses
    // colored by DBSCAN cluster (noise in gray).
    if (cfg.visualize) {
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      // Layer 0 is the turbo LUT image; the cluster/noise CROSSES layers draw
      // on top (higher priority renders last). One CROSSES InputSpec per
      // palette color, named to match ClusterOverlayOp's emitted tensors.
      std::vector<holoscan::ops::HolovizOp::InputSpec> input_specs;
      input_specs.emplace_back("rd_map",
                               holoscan::ops::HolovizOp::InputType::COLOR_LUT);
      const auto palette = cluster_palette();
      if (static_cast<int>(palette.size()) != kNumClusterColors) {
        throw std::runtime_error(
            "fmcw_siso: cluster palette size must equal kNumClusterColors");
      }
      const auto add_crosses = [&](const std::string& name,
                                   const std::vector<float>& color) {
        holoscan::ops::HolovizOp::InputSpec spec(
            name, holoscan::ops::HolovizOp::InputType::CROSSES);
        spec.color_ = color;
        spec.line_width_ = 2.0F;
        spec.priority_ = 1;
        input_specs.push_back(spec);
      };
      for (int c = 0; c < kNumClusterColors; ++c) {
        add_crosses(cluster_layer_name(c), palette[static_cast<size_t>(c)]);
      }
      add_crosses(kNoiseLayerName, {0.6F, 0.6F, 0.6F, 1.0F});

      auto visualizer = make_operator<holoscan::ops::HolovizOp>(
          "holoviz", holoscan::Arg("tensors", input_specs),
          holoscan::Arg("color_lut", make_turbo_lut()),
          holoscan::Arg("window_title",
                        std::string("FMCW SISO - Range-Doppler + clusters")),
          holoscan::Arg("width", 800U), holoscan::Arg("height", 600U),
          holoscan::Arg("headless", cfg.headless));

      add_flow(doppler, to_image, {{"rd_maps", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      // The overlay needs the detector's cells, the cluster labels, and the
      // map dimensions; its per-color layers fan into the same receiver.
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cluster, overlay, {{"labels", "labels"}});
      add_flow(doppler, overlay, {{"rd_maps", "rd_maps"}});
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
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
