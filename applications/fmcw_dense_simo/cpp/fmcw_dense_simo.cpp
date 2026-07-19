// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SIMO FMCW *dense* angle-then-cluster pipeline (headless-capable, with an
// optional Holoviz display):
//
//   SimoAdcSourceOp -> Range -> Clutter -> Doppler
//     -> PhaseShiftBeamformerOp (rd_maps -> rda_cube, angle on every RD cell)
//     -> CFARDetectorOp (on the RDA cube; nci_axes=[] -> az/el ride as batch)
//     -> AnglePeakPickOp (keep the peak (az,el) beam per detected RD cell)
//     -> CubeAoaLookupOp (bridge: detection az/el bins -> aoa_estimates)
//     -> FeatureAssembly [range, velocity, az, el] -> ClusterDBSCAN
//     -> DetectionReduction -> DenseClusterReportOp (logs clusters)
//
// This is a lean demo exercising PhaseShiftBeamformerOp end-to-end; it is kept
// separate from fmcw_simo (the sparse-path demo) because the dense path
// diverges in the detect+angle middle *and* the reporting layer (4-column
// detection_idx / rank-4 detection_score vs. the sparse 2-column layout the
// shared radar_app report ops assume). See the operator's OPEN_ITEMS for the
// FeatureAssembly az/el contract gap that the CubeAoaLookupOp bridge works
// around.
//
// The display reuses the app-agnostic radar_app/viz operators unchanged: the
// RD-map image + rolling range/velocity plots + boresight az/el scatter. The
// Doppler op still emits per-channel rd_maps (N_rx, D, R) exactly like the
// sparse path, so those ops apply directly; the only mismatch is that the dense
// detection_idx / rep_idx carry the (az_bin, el_bin) batch columns ahead of
// (doppler_bin, range_bin). A small TrailingIdxColumnsOp slices the trailing
// two columns so ClusterOverlayOp / RollingPlotOp see the sparse 2-column form
// they expect — no changes to the shared ops.
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_dense_simo/cpp/fmcw_dense_simo [config.yaml]

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
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
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
#include "radar_app/report/cadence_meter.hpp"
#include "radar_app/viz/display.hpp"
#include "radar_app/viz/viz.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

namespace {

using Precision = float;

struct PlotsConfig {
  bool enable{true};
  int64_t window{120};
  float velocity_ylim_lo{std::numeric_limits<float>::quiet_NaN()};
  float velocity_ylim_hi{std::numeric_limits<float>::quiet_NaN()};
};

struct AppConfig {
  radar::sim::SimScene scene;
  radar::ArrayGeometry array_geometry;
  std::vector<double> az_angles;
  std::vector<double> el_angles;
  PlotsConfig plots;
  int64_t num_frames{1};
  double frame_period{0.1};
  bool meter_cadence{true};
  int64_t meter_log_every{0};
  bool visualize{false};
  bool skip_display_check{false};
  bool headless{false};
};

// Uniform rectangular array (URA) in the y-z plane, normal to boresight (+x):
// element (iy, iz) at (0, iy*spacing, iz*spacing). y resolves azimuth, z
// elevation. Matches the fmcw_simo geometry so the two demos share a scene.
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

  const auto waveform = root["waveform"].as<radar::FMCWWaveformDescriptor>();
  cfg.scene.waveform.carrier_frequency = waveform.carrier_frequency();
  cfg.scene.waveform.bandwidth = waveform.bandwidth();
  cfg.scene.waveform.chirp_duration = waveform.chirp_duration();
  cfg.scene.waveform.sample_rate = waveform.sample_rate();
  cfg.scene.waveform.chirp_repetition_interval =
      waveform.chirp_repetition_interval();

  cfg.num_frames = root["run"]["num_frames"].as<int64_t>();
  cfg.frame_period = root["run"]["frame_period"].as<double>();
  if (root["run"]["meter_cadence"]) {
    cfg.meter_cadence = root["run"]["meter_cadence"].as<bool>();
  }
  if (root["run"]["meter_log_every"]) {
    cfg.meter_log_every = root["run"]["meter_log_every"].as<int64_t>();
  }
  if (cfg.meter_log_every < 0) {
    throw std::runtime_error("run.meter_log_every must be >= 0");
  }
  cfg.scene.num_chirps = root["cube"]["num_chirps"].as<int64_t>();
  cfg.scene.samples_per_chirp = root["cube"]["samples_per_chirp"].as<int64_t>();

  const YAML::Node rx = root["rx_array"];
  if (!rx) {
    throw std::runtime_error("rx_array section is required for SIMO");
  }
  const int64_t az_elements = rx["az_elements"].as<int64_t>();
  const int64_t el_elements = rx["el_elements"].as<int64_t>();
  const double spacing = rx["spacing"].as<double>();
  const double wavelength = cfg.scene.waveform.wavelength();
  if (az_elements < 2 || el_elements < 2) {
    throw std::runtime_error(
        "SIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2");
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

  // The dense report and the az/el lookup bridge both need the beam grid, so
  // parse it here (the beamformer itself reads the same block via from_config).
  const YAML::Node beam = root["phase_shift_beamformer"];
  if (!beam || !beam["az_angles"] || !beam["el_angles"]) {
    throw std::runtime_error(
        "phase_shift_beamformer.az_angles and el_angles are required");
  }
  for (const YAML::Node& v : beam["az_angles"]) {
    cfg.az_angles.push_back(v.as<double>());
  }
  for (const YAML::Node& v : beam["el_angles"]) {
    cfg.el_angles.push_back(v.as<double>());
  }

  for (const YAML::Node& t : root["targets"]) {
    const double range = t["range"].as<double>();
    const double azimuth = t["azimuth"].as<double>();
    const double elevation = t["elevation"].as<double>();
    const double amplitude = t["amplitude"].as<double>();
    const YAML::Node vel = t["velocity"];
    if (!vel || !vel.IsSequence() || vel.size() != 3) {
      throw std::runtime_error(
          "fmcw_dense_simo: each target velocity must be a 3-vector [vx, vy, "
          "vz] m/s (array frame)");
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
    if (plots["velocity_ylim"] && plots["velocity_ylim"].IsSequence() &&
        plots["velocity_ylim"].size() == 2) {
      cfg.plots.velocity_ylim_lo = plots["velocity_ylim"][0].as<float>();
      cfg.plots.velocity_ylim_hi = plots["velocity_ylim"][1].as<float>();
    }
  }
  if (cfg.plots.window <= 1) {
    throw std::runtime_error("plots.window must be > 1");
  }
  return cfg;
}

}  // namespace

// Emits a simulated SIMO ADC cube (N_rx, num_chirps, samples) per tick and
// advances the scene. Identical to fmcw_simo's source (kept local to keep the
// dense demo self-contained).
class SimoAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimoAdcSourceOp)
  SimoAdcSourceOp() = default;

  radar::sim::SimScene scene;
  double frame_period{0.1};
  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const radar::sim::SimResult sim = radar::sim::simulate(scene);
    if (sim.shape.size() != 3 || sim.shape[0] < 2) {
      throw std::runtime_error(
          "SimoAdcSourceOp: expected rank-3 (N_rx>=2, chirps, samples)");
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
    radar::emit_single(op_output, "adc_cube", tensor);
    if (cadence) {
      cadence->on_source_emit();
    }
    advance_scene(&scene, frame_period);
  }
};

// Angle peak-pick: collapse the dense cube's many per-beam detections down to
// one detection per range-Doppler cell — the (az, el) beam with the largest
// CFAR score, i.e. the beamformed-power peak, which is the target's true DOA.
//
// Why: CFAR runs independently on each beam's RD plane (nci_axes=[]), so it has
// no cross-beam view; a strong target trips the threshold in *many* beams (its
// broad mainlobe plus sidelobes), scattering raw detections across the whole
// angular FOV. Peak-picking is the dense analog of the sparse path's "one angle
// per detection" (there, SnapshotGather + a single beamscan peak). It makes the
// angle estimate a point per RD cell instead of a beam-pattern smear, so the
// boresight scatter and RD overlay are tight and clustering is unambiguous.
//
// In:  detection_idx  — int64 (N, 4): az_bin, el_bin, doppler_bin, range_bin.
// In:  detection_score — float (num_az, num_el, num_doppler, num_range): CFAR
//                        test statistic (∝ beamformed power) over the cube.
// Out: detection_idx  — int64 (M, 4): one row per detected (doppler, range)
//                        cell, with (az, el) set to the score-argmax beam.
class AnglePeakPickOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AnglePeakPickOp)
  AnglePeakPickOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("detection_idx");
    spec.input<holoscan::TensorMap>("detection_score");
    spec.output<holoscan::TensorMap>("detection_idx");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "AnglePeakPickOp";
    auto idx = radar::receive_single(op_input, "detection_idx", who);
    auto score = radar::receive_single(op_input, "detection_score", who);
    const auto ishape = idx->shape();
    if (ishape.size() != 2 || ishape[1] != 4) {
      throw std::runtime_error(
          "AnglePeakPickOp: detection_idx must be (num_points, 4)");
    }
    const auto sshape = score->shape();
    if (sshape.size() != 4) {
      throw std::runtime_error(
          "AnglePeakPickOp: detection_score must be rank-4 "
          "(num_az, num_el, num_doppler, num_range)");
    }
    const int64_t n = ishape[0];
    const int64_t na = sshape[0];
    const int64_t ne = sshape[1];
    const int64_t nd = sshape[2];
    const int64_t nr = sshape[3];

    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("AnglePeakPickOp: device synchronize failed");
    }
    std::vector<int64_t> host_idx(static_cast<size_t>(n * 4));
    if (n > 0 &&
        cudaMemcpy(host_idx.data(), idx->data(),
                   static_cast<size_t>(idx->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("AnglePeakPickOp: detection_idx D2H failed");
    }
    std::vector<Precision> host_score(static_cast<size_t>(na * ne * nd * nr));
    if (cudaMemcpy(host_score.data(), score->data(),
                   static_cast<size_t>(score->nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("AnglePeakPickOp: detection_score D2H failed");
    }

    // Unique detected (doppler, range) cells, in first-seen order.
    std::vector<int64_t> seen_flag(static_cast<size_t>(nd * nr), 0);
    std::vector<std::pair<int64_t, int64_t>> cells;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t d = host_idx[static_cast<size_t>(i * 4 + 2)];
      const int64_t r = host_idx[static_cast<size_t>(i * 4 + 3)];
      if (d < 0 || d >= nd || r < 0 || r >= nr) {
        throw std::runtime_error(
            "AnglePeakPickOp: detection (doppler, range) out of score bounds");
      }
      const size_t key = static_cast<size_t>(d * nr + r);
      if (seen_flag[key] == 0) {
        seen_flag[key] = 1;
        cells.emplace_back(d, r);
      }
    }

    const int64_t m = static_cast<int64_t>(cells.size());
    auto out = radar::make_device_tensor({m, 4}, DLDataType{kDLInt, 64, 1});
    if (m > 0) {
      std::vector<int64_t> host_out(static_cast<size_t>(m * 4));
      for (int64_t c = 0; c < m; ++c) {
        const int64_t d = cells[static_cast<size_t>(c)].first;
        const int64_t r = cells[static_cast<size_t>(c)].second;
        int64_t best_az = 0, best_el = 0;
        Precision best = -std::numeric_limits<Precision>::infinity();
        for (int64_t a = 0; a < na; ++a) {
          for (int64_t e = 0; e < ne; ++e) {
            const Precision v = host_score[static_cast<size_t>(
                ((a * ne + e) * nd + d) * nr + r)];
            if (v > best) {
              best = v;
              best_az = a;
              best_el = e;
            }
          }
        }
        host_out[static_cast<size_t>(c * 4 + 0)] = best_az;
        host_out[static_cast<size_t>(c * 4 + 1)] = best_el;
        host_out[static_cast<size_t>(c * 4 + 2)] = d;
        host_out[static_cast<size_t>(c * 4 + 3)] = r;
      }
      if (cudaMemcpy(out->data(), host_out.data(),
                     sizeof(int64_t) * host_out.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("AnglePeakPickOp: detection_idx H2D failed");
      }
    }
    radar::emit_single(op_output, "detection_idx", out);
  }
};

// Dense-path bridge (approach A): maps per-detection (az_bin, el_bin) cube
// batch columns to physical angles via the scan grids and emits aoa_estimates
// (num_points, 2), feeding FeatureAssemblyOp's existing aoa_estimates port.
// This exists because FeatureAssemblyOp sources az/el only from aoa_estimates,
// not from the RDA-cube's preserved batch bins — see the operator OPEN_ITEMS.
// Kept
// an app-local add-on (plain-tensor-ports invariant), not a shared type or an
// operator contract change.
//
// Also emits row_index (num_points,), the identity [0..num_points). It is a
// no-op relabel for the detection path (drained), but when this op is fed the
// DetectionReductionOp rep_idx (one row per cluster, in cluster order) the row
// index *is* the cluster id — so the boresight panel can reuse the row_index as
// its per-marker cluster label to show one colored centroid marker per cluster.
class CubeAoaLookupOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CubeAoaLookupOp)
  CubeAoaLookupOp() = default;

  std::vector<double> az_grid;
  std::vector<double> el_grid;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("detection_idx");
    spec.output<holoscan::TensorMap>("aoa_estimates");
    spec.output<holoscan::TensorMap>("row_index");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "CubeAoaLookupOp";
    auto idx = radar::receive_single(op_input, "detection_idx", who);
    const auto shape = idx->shape();
    // Cube detections: (num_points, 4) = (az_bin, el_bin, doppler_bin,
    // range_bin). Require the two leading angle-batch columns.
    if (shape.size() != 2 || shape[1] < 4) {
      throw std::runtime_error(
          "CubeAoaLookupOp: detection_idx must be (num_points, >= 4) with "
          "leading (az_bin, el_bin) columns from the RDA cube");
    }
    const int64_t n = shape[0];
    const int64_t cols = shape[1];
    auto aoa =
        radar::make_device_tensor({n, 2}, radar::real_dtype<Precision>());
    auto row_index = radar::make_device_tensor({n}, DLDataType{kDLInt, 64, 1});
    if (n > 0) {
      std::vector<int64_t> host_idx(static_cast<size_t>(n * cols));
      if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("CubeAoaLookupOp: device synchronize failed");
      }
      if (cudaMemcpy(host_idx.data(), idx->data(),
                     static_cast<size_t>(idx->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("CubeAoaLookupOp: detection_idx D2H failed");
      }
      const int64_t num_az = static_cast<int64_t>(az_grid.size());
      const int64_t num_el = static_cast<int64_t>(el_grid.size());
      std::vector<Precision> host_aoa(static_cast<size_t>(n * 2));
      std::vector<int64_t> host_rows(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i) {
        const int64_t az_bin = host_idx[static_cast<size_t>(i * cols + 0)];
        const int64_t el_bin = host_idx[static_cast<size_t>(i * cols + 1)];
        if (az_bin < 0 || az_bin >= num_az || el_bin < 0 || el_bin >= num_el) {
          throw std::runtime_error(
              "CubeAoaLookupOp: detection az/el bin out of grid range");
        }
        host_aoa[static_cast<size_t>(i * 2 + 0)] =
            static_cast<Precision>(az_grid[static_cast<size_t>(az_bin)]);
        host_aoa[static_cast<size_t>(i * 2 + 1)] =
            static_cast<Precision>(el_grid[static_cast<size_t>(el_bin)]);
        host_rows[static_cast<size_t>(i)] = i;
      }
      if (cudaMemcpy(aoa->data(), host_aoa.data(),
                     sizeof(Precision) * host_aoa.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("CubeAoaLookupOp: aoa_estimates H2D failed");
      }
      if (cudaMemcpy(row_index->data(), host_rows.data(),
                     sizeof(int64_t) * host_rows.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("CubeAoaLookupOp: row_index H2D failed");
      }
    }
    radar::emit_single(op_output, "aoa_estimates", aoa);
    radar::emit_single(op_output, "row_index", row_index);
  }
};

// Dense-aware cluster log. Mirrors radar_app's ClusterReportOp but handles the
// 4-column dense layout: points = (range, velocity, az, el), rep_idx =
// (az_bin, el_bin, doppler_bin, range_bin). Kept local because the shared op
// hard-codes the sparse 2-column detection_idx / rep_idx.
class DenseClusterReportOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DenseClusterReportOp)
  DenseClusterReportOp() = default;

  std::vector<double> az_grid;
  std::vector<double> el_grid;
  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("points");
    spec.input<holoscan::TensorMap>("labels");
    spec.input<holoscan::TensorMap>("sizes");
    spec.input<holoscan::TensorMap>("rep_idx");
    spec.input<holoscan::TensorMap>("range_bins");
    spec.input<holoscan::TensorMap>("velocity_bins");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "DenseClusterReportOp";
    auto points = radar::receive_single(op_input, "points", who);
    auto labels = radar::receive_single(op_input, "labels", who);
    auto sizes = radar::receive_single(op_input, "sizes", who);
    auto rep_idx = radar::receive_single(op_input, "rep_idx", who);
    auto range_bins = radar::receive_single(op_input, "range_bins", who);
    auto velocity_bins = radar::receive_single(op_input, "velocity_bins", who);

    const auto pshape = points->shape();
    if (pshape.size() != 2 || pshape[1] != 4) {
      throw std::runtime_error(
          "DenseClusterReportOp: points must be (N, 4) [range, vel, az, el]");
    }
    const int64_t num_points = pshape[0];
    const int64_t num_clusters = sizes->shape()[0];
    const auto rshape = rep_idx->shape();
    if (rshape.size() != 2 || rshape[1] != 4 || rshape[0] != num_clusters) {
      throw std::runtime_error(
          "DenseClusterReportOp: rep_idx must be (num_clusters, 4)");
    }
    const int64_t num_range = range_bins->shape()[0];
    const int64_t num_doppler = velocity_bins->shape()[0];

    std::vector<float> pts(static_cast<size_t>(num_points * 4));
    std::vector<int64_t> lab(static_cast<size_t>(num_points));
    std::vector<int64_t> sz(static_cast<size_t>(num_clusters));
    std::vector<int64_t> rep(static_cast<size_t>(num_clusters * 4));
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));

    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("DenseClusterReportOp: device sync failed");
    }
    const auto copy_d2h = [](void* dst,
                             const std::shared_ptr<holoscan::Tensor>& src) {
      if (src->nbytes() == 0) {
        return;
      }
      if (cudaMemcpy(dst, src->data(), static_cast<size_t>(src->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("DenseClusterReportOp: D2H copy failed");
      }
    };
    copy_d2h(pts.data(), points);
    copy_d2h(lab.data(), labels);
    copy_d2h(sz.data(), sizes);
    copy_d2h(rep.data(), rep_idx);
    copy_d2h(ranges.data(), range_bins);
    copy_d2h(velocities.data(), velocity_bins);

    std::vector<double> sum_r(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_v(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_az(static_cast<size_t>(num_clusters), 0.0);
    std::vector<double> sum_el(static_cast<size_t>(num_clusters), 0.0);
    int64_t noise = 0;
    for (int64_t i = 0; i < num_points; ++i) {
      const int64_t c = lab[static_cast<size_t>(i)];
      if (c < 0) {
        ++noise;
        continue;
      }
      sum_r[static_cast<size_t>(c)] += pts[static_cast<size_t>(i * 4 + 0)];
      sum_v[static_cast<size_t>(c)] += pts[static_cast<size_t>(i * 4 + 1)];
      sum_az[static_cast<size_t>(c)] += pts[static_cast<size_t>(i * 4 + 2)];
      sum_el[static_cast<size_t>(c)] += pts[static_cast<size_t>(i * 4 + 3)];
    }

    const int64_t num_az = static_cast<int64_t>(az_grid.size());
    const int64_t num_el = static_cast<int64_t>(el_grid.size());
    HOLOSCAN_LOG_INFO("dense: {} detections -> {} clusters ({} noise)",
                      num_points, num_clusters, noise);
    for (int64_t c = 0; c < num_clusters; ++c) {
      const int64_t n = sz[static_cast<size_t>(c)];
      const double inv = n > 0 ? 1.0 / static_cast<double>(n) : 0.0;
      const int64_t az_bin = rep[static_cast<size_t>(c * 4 + 0)];
      const int64_t el_bin = rep[static_cast<size_t>(c * 4 + 1)];
      const int64_t dd = rep[static_cast<size_t>(c * 4 + 2)];
      const int64_t rr = rep[static_cast<size_t>(c * 4 + 3)];
      if (az_bin < 0 || az_bin >= num_az || el_bin < 0 || el_bin >= num_el ||
          dd < 0 || dd >= num_doppler || rr < 0 || rr >= num_range) {
        throw std::runtime_error(
            "DenseClusterReportOp: rep_idx out of grid/axis bounds");
      }
      HOLOSCAN_LOG_INFO(
          "  cluster {}: {} pts, centroid range {:.2f} m, vel {:+.2f} m/s, "
          "az {:.3f} rad, el {:.3f} rad; rep range {:.2f} m, vel {:+.2f} m/s, "
          "az {:.3f} rad, el {:.3f} rad",
          c, n, sum_r[static_cast<size_t>(c)] * inv,
          sum_v[static_cast<size_t>(c)] * inv,
          sum_az[static_cast<size_t>(c)] * inv,
          sum_el[static_cast<size_t>(c)] * inv,
          ranges[static_cast<size_t>(rr)], velocities[static_cast<size_t>(dd)],
          az_grid[static_cast<size_t>(az_bin)],
          el_grid[static_cast<size_t>(el_bin)]);
    }
    if (cadence) {
      cadence->on_sink_done();
    }
  }
};

// Drains an unused output (raw Entity receive) so its transmitter has a
// downstream and the producer can tick. CFAR always emits detection_mask /
// detection_score / noise_estimate; the dense demo consumes none of them.
// Entity receive also sidesteps Holoscan 4.3's bool-DLPack path for the mask.
class EntityDrainOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(EntityDrainOp)
  EntityDrainOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    (void)op_input.receive<holoscan::gxf::Entity>("input");
  }
};

// Slices the trailing two columns of an int64 index tensor (num_points, C>=2)
// down to (num_points, 2). The dense CFAR / reduction indices carry the cube's
// (az_bin, el_bin) batch columns ahead of (doppler_bin, range_bin); the shared
// radar_app RD-plane viz ops (ClusterOverlayOp, RollingPlotOp) expect exactly
// the sparse 2-column (doppler_bin, range_bin) form. Row order is preserved, so
// the sliced tensor stays row-aligned with cluster labels. Input/output port
// names are generic ("idx") so one op serves both detection_idx and rep_idx.
class TrailingIdxColumnsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TrailingIdxColumnsOp)
  TrailingIdxColumnsOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("idx");
    spec.output<holoscan::TensorMap>("idx");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "TrailingIdxColumnsOp";
    auto idx = radar::receive_single(op_input, "idx", who);
    const auto shape = idx->shape();
    if (shape.size() != 2 || shape[1] < 2) {
      throw std::runtime_error(
          "TrailingIdxColumnsOp: expected (num_points, >= 2) int64 idx");
    }
    const int64_t n = shape[0];
    const int64_t cols = shape[1];
    auto out = radar::make_device_tensor({n, 2}, DLDataType{kDLInt, 64, 1});
    if (n > 0) {
      std::vector<int64_t> host(static_cast<size_t>(n * cols));
      if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("TrailingIdxColumnsOp: device sync failed");
      }
      if (cudaMemcpy(host.data(), idx->data(),
                     static_cast<size_t>(idx->nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("TrailingIdxColumnsOp: idx D2H failed");
      }
      std::vector<int64_t> trimmed(static_cast<size_t>(n * 2));
      for (int64_t i = 0; i < n; ++i) {
        trimmed[static_cast<size_t>(i * 2 + 0)] =
            host[static_cast<size_t>(i * cols + (cols - 2))];
        trimmed[static_cast<size_t>(i * 2 + 1)] =
            host[static_cast<size_t>(i * cols + (cols - 1))];
      }
      if (cudaMemcpy(out->data(), trimmed.data(),
                     sizeof(int64_t) * trimmed.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("TrailingIdxColumnsOp: idx H2D failed");
      }
    }
    radar::emit_single(op_output, "idx", out);
  }
};

class FmcwDenseSimoApp : public holoscan::Application {
 public:
  AppConfig cfg;
  std::shared_ptr<CadenceMeter> cadence;

  void compose() override {
    using holoscan::Arg;
    auto source = make_operator<SimoAdcSourceOp>(
        "adc_source",
        make_condition<holoscan::CountCondition>("frame_count",
                                                 cfg.num_frames),
        make_condition<holoscan::PeriodicCondition>(
            "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
    source->scene = cfg.scene;
    source->frame_period = cfg.frame_period;

    auto range = make_operator<radar::FMCWRangeProcessorOp<Precision>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<Precision>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<Precision>>(
        "doppler_processor", from_config("doppler_processor"));
    auto beamformer = make_operator<radar::PhaseShiftBeamformerOp<Precision>>(
        "phase_shift_beamformer", from_config("phase_shift_beamformer"),
        Arg("array_geometry") = cfg.array_geometry);
    auto cfar = make_operator<radar::CFARDetectorOp<Precision>>(
        "cfar_detector", from_config("cfar_detector"));

    auto peak_pick = make_operator<AnglePeakPickOp>("angle_peak_pick");
    auto aoa_lookup = make_operator<CubeAoaLookupOp>("cube_aoa_lookup");
    aoa_lookup->az_grid = cfg.az_angles;
    aoa_lookup->el_grid = cfg.el_angles;

    const std::vector<radar::AssemblyFeature> feature_columns{
        radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity,
        radar::AssemblyFeature::kAz, radar::AssemblyFeature::kEl};
    auto feature_assembly = make_operator<radar::FeatureAssemblyOp<Precision>>(
        "feature_assembly", Arg("features") = feature_columns);
    auto cluster = make_operator<radar::ClusterDBSCANOp<Precision>>(
        "cluster_dbscan", from_config("cluster_dbscan"));
    auto reduction = make_operator<radar::DetectionReductionOp<Precision>>(
        "detection_reduction", from_config("reduction"));
    auto report = make_operator<DenseClusterReportOp>("dense_cluster_report");
    report->az_grid = cfg.az_angles;
    report->el_grid = cfg.el_angles;

    if (cfg.meter_cadence) {
      cadence = std::make_shared<CadenceMeter>();
      cadence->frame_period_s = cfg.frame_period;
      cadence->log_every = cfg.meter_log_every;
      source->cadence = cadence;
      report->cadence = cadence;
    }

    // Front end.
    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});

    // Dense angle path: beamform every RD cell, detect on the cube.
    add_flow(doppler, beamformer, {{"rd_maps", "rd_maps"}});
    add_flow(beamformer, cfar, {{"rda_cube", "rd_maps"}});

    // Collapse the per-beam cube detections to one peak beam per RD cell; this
    // peaked detection_idx drives the whole downstream (angle, features,
    // clustering, reduction). CFAR's detection_score feeds the peak-pick; its
    // bool mask / noise outputs are unused, so drain them so CFAR ticks.
    add_flow(cfar, peak_pick, {{"detection_idx", "detection_idx"},
                               {"detection_score", "detection_score"}});
    auto mask_sink = make_operator<EntityDrainOp>("detection_mask_sink");
    auto noise_sink = make_operator<EntityDrainOp>("noise_estimate_sink");
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    // Bridge the peak beam's (az_bin, el_bin) columns to angle values. The
    // per-detection row_index is unused here (only the rep instance in the
    // display uses it as a cluster label), so drain it to keep the op ticking.
    add_flow(peak_pick, aoa_lookup, {{"detection_idx", "detection_idx"}});
    add_flow(aoa_lookup, feature_assembly,
             {{"aoa_estimates", "aoa_estimates"}});
    auto row_index_sink = make_operator<EntityDrainOp>("aoa_row_index_sink");
    add_flow(aoa_lookup, row_index_sink, {{"row_index", "input"}});

    // Assemble [range, velocity, az, el] and cluster.
    add_flow(peak_pick, feature_assembly, {{"detection_idx", "detection_idx"}});
    add_flow(range, feature_assembly, {{"range_bins", "range_bins"}});
    add_flow(doppler, feature_assembly, {{"velocity_bins", "velocity_bins"}});
    add_flow(feature_assembly, cluster, {{"points", "points"}});

    add_flow(cluster, reduction, {{"labels", "labels"}});
    add_flow(peak_pick, reduction, {{"detection_idx", "detection_idx"}});
    add_flow(feature_assembly, reduction, {{"points", "points"}});

    add_flow(feature_assembly, report, {{"points", "points"}});
    add_flow(cluster, report, {{"labels", "labels"}, {"sizes", "sizes"}});
    add_flow(reduction, report, {{"rep_idx", "rep_idx"}});
    add_flow(range, report, {{"range_bins", "range_bins"}});
    add_flow(doppler, report, {{"velocity_bins", "velocity_bins"}});

    if (cfg.visualize) {
      compose_display(doppler, peak_pick, cluster, reduction, range);
    }
  }

 private:
  // Optional Holoviz display, reusing the app-agnostic radar_app/viz ops. The
  // RD map (channel-0 slice) fills the left; the right column stacks range-,
  // velocity-vs-time rolling plots and the boresight az/el scatter. The dense
  // 4-column detection_idx / rep_idx are sliced to their trailing
  // (doppler_bin, range_bin) columns for the RD-plane overlay/plot ops.
  void compose_display(
      const std::shared_ptr<radar::FMCWDopplerProcessorOp<Precision>>& doppler,
      const std::shared_ptr<AnglePeakPickOp>& detections,
      const std::shared_ptr<radar::ClusterDBSCANOp<Precision>>& cluster,
      const std::shared_ptr<radar::DetectionReductionOp<Precision>>& reduction,
      const std::shared_ptr<radar::FMCWRangeProcessorOp<Precision>>& range) {
    using holoscan::Arg;
    using InputSpec = holoscan::ops::HolovizOp::InputSpec;
    using InputType = holoscan::ops::HolovizOp::InputType;

    auto ch_slice = make_operator<RdChannelSliceOp>("rd_channel_slice");
    ch_slice->channel = 0;
    auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
    auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");
    // Peaked detection_idx (M, 4) -> (M, 2) trailing (doppler_bin, range_bin).
    auto det_idx_rd = make_operator<TrailingIdxColumnsOp>("det_idx_rd_slice");

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
    std::shared_ptr<TrailingIdxColumnsOp> rep_idx_rd;
    std::shared_ptr<CubeAoaLookupOp> rep_aoa;
    if (cfg.plots.enable) {
      const auto left = make_view(0.0F, 0.0F, 0.60F, 1.0F);
      for (InputSpec& spec : input_specs) {
        spec.views_ = {left};
      }
      const auto add_plot = [&](const char* img_name, const char* title_name,
                                const std::string& title, float oy, float h) {
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
      plot->velocity_ylim_lo = cfg.plots.velocity_ylim_lo;
      plot->velocity_ylim_hi = cfg.plots.velocity_ylim_hi;
      rep_idx_rd = make_operator<TrailingIdxColumnsOp>("rep_idx_rd_slice");
      rep_aoa = make_operator<CubeAoaLookupOp>("cube_aoa_lookup_rep");
      rep_aoa->az_grid = cfg.az_angles;
      rep_aoa->el_grid = cfg.el_angles;
      boresight = make_operator<BoresightViewOp>("boresight_view");
      double az_lo = 0.0, az_hi = 0.0, el_lo = 0.0, el_hi = 0.0;
      grid_extent(cfg.az_angles, &az_lo, &az_hi);
      grid_extent(cfg.el_angles, &el_lo, &el_hi);
      boresight->az_lo = az_lo;
      boresight->az_hi = az_hi;
      boresight->el_lo = el_lo;
      boresight->el_hi = el_hi;
    }

    auto visualizer = make_operator<holoscan::ops::HolovizOp>(
        "holoviz", Arg("tensors", input_specs),
        Arg("color_lut", make_turbo_lut()),
        Arg("window_title",
            std::string("FMCW SIMO dense - RD (ch0) + clusters")),
        Arg("width", cfg.plots.enable ? 1200U : 800U),
        Arg("height", cfg.plots.enable ? 760U : 600U),
        Arg("headless", cfg.headless));

    add_flow(doppler, ch_slice, {{"rd_maps", "rd_maps"}});
    add_flow(ch_slice, to_image, {{"rd_map", "rd_maps"}});
    add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
    add_flow(detections, det_idx_rd, {{"detection_idx", "idx"}});
    add_flow(det_idx_rd, overlay, {{"idx", "detection_idx"}});
    add_flow(cluster, overlay, {{"labels", "labels"}});
    add_flow(ch_slice, overlay, {{"rd_map", "rd_maps"}});
    add_flow(overlay, visualizer, {{"overlay", "receivers"}});
    if (plot) {
      add_flow(reduction, rep_idx_rd, {{"rep_idx", "idx"}});
      add_flow(rep_idx_rd, plot, {{"idx", "rep_idx"}});
      add_flow(range, plot, {{"range_bins", "range_bins"}});
      add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
      add_flow(plot, visualizer, {{"plots", "receivers"}});
    }
    if (boresight) {
      // Centroids-only: plot the per-cluster representative (medoid) angle, not
      // the raw per-beam detections. rep_idx is one row per cluster (cluster
      // order), so the rep lookup's row_index is the cluster id used to color
      // each marker consistently with the RD-plane overlay.
      add_flow(reduction, rep_aoa, {{"rep_idx", "detection_idx"}});
      add_flow(rep_aoa, boresight, {{"aoa_estimates", "aoa_estimates"},
                                    {"row_index", "labels"}});
      add_flow(boresight, visualizer, {{"plot", "receivers"}});
    }
  }

  // Min/max of a scan grid, padding a degenerate (single-value) axis so it
  // renders a visible span in the boresight view.
  static void grid_extent(const std::vector<double>& grid, double* lo,
                          double* hi) {
    if (grid.empty()) {
      *lo = -0.35;
      *hi = 0.35;
    } else {
      *lo = *std::min_element(grid.begin(), grid.end());
      *hi = *std::max_element(grid.begin(), grid.end());
    }
    if (*hi - *lo < 1e-6) {
      const double mid = 0.5 * (*lo + *hi);
      *lo = mid - 0.1;
      *hi = mid + 0.1;
    }
  }
};

namespace {

std::filesystem::path resolve_config_path(int argc, char** argv) {
  if (argc > 1) {
    return argv[1];
  }
  return std::filesystem::path(argv[0]).parent_path() / "fmcw_dense_simo.yaml";
}

}  // namespace

int main(int argc, char** argv) {
  auto app = holoscan::make_application<FmcwDenseSimoApp>();
  try {
    const std::filesystem::path config_path = resolve_config_path(argc, argv);
    app->cfg = load_config(config_path);
    if (app->cfg.visualize && !app->cfg.headless &&
        !app->cfg.skip_display_check && !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "fmcw_dense_simo: visualize enabled but no display; running without "
          "GUI");
      app->cfg.visualize = false;
    }
    app->config(config_path.string());
    app->run();
    if (app->cadence) {
      app->cadence->log_summary();
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("fmcw_dense_simo: {}", e.what());
    return 1;
  }
  return 0;
}
