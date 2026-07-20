// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// MIMO FMCW sparse angle-then-cluster pipeline (TDM or DDM via YAML):
//
// TDM: MimoTdmAdcSourceOp -> Deinterleave -> Range -> Clutter -> Doppler
//        -> Compensation -> VirtualArrayAssemblyOp -> ...
// DDM: MimoDdmAdcSourceOp -> Range -> Doppler -> Compensation -> Clutter
//        -> VirtualArrayAssemblyOp -> ...
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_mimo/cpp/fmcw_mimo [config.yaml]

#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <memory>
#include <optional>
#include <random>
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
#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"
#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_app/app/config.hpp"
#include "radar_app/app/driver.hpp"
#include "radar_app/app/holoviz_layout.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/viz/display.hpp"
#include "radar_app/viz/viz.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

namespace {

struct MimoScene {
  std::vector<radar::sim::PointTarget> targets;
  radar::sim::SimWaveform waveform{};
  int64_t num_tx{2};
  int64_t num_chirps{64};
  int64_t samples_per_chirp{256};
  std::vector<radar::sim::Vec3> tx_positions;
  std::vector<radar::sim::Vec3> rx_positions;
  double propagation_speed{radar::sim::kSimSpeedOfLight};
  std::optional<double> snr_db{};
  // Fixed RNG seed. The ADC sources reseed with this on every frame, so the
  // additive-noise realization is deliberately identical frame to frame: the
  // demo trades frozen-speckle realism for reproducibility (deterministic
  // clusters and screenshots). Accepted 2026-07-16 as fine for a demo; a
  // frame-varying scene run should fold the frame index into the seed.
  uint64_t seed{0x5eed5eedULL};
};

struct AppConfig : CommonRunConfig {
  radar::MultiplexingScheme scheme{radar::MultiplexingScheme::kTdm};
  MimoScene scene;
  radar::TXSchedule tx_schedule;
  radar::DDMCodeDescriptor ddm_code;
  std::vector<double> ddm_doppler_offsets;
  radar::ArrayGeometry array_geometry;
  std::vector<double> tx_positions_flat;
  std::vector<double> rx_positions_flat;
  double wavelength{0.0};
  // Optional fixed velocity-plot Y extent (m/s); NaN keeps the full-axis span.
  float velocity_ylim_lo{std::numeric_limits<float>::quiet_NaN()};
  float velocity_ylim_hi{std::numeric_limits<float>::quiet_NaN()};
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};
  radar::FMCWWaveformDescriptor report_waveform;
  int64_t report_num_doppler{0};
};

radar::sim::Vec3 scale_position(double iy, double iz, double spacing_wl,
                                double wavelength) {
  return radar::sim::Vec3{
      0.0, iy * spacing_wl * wavelength, iz * spacing_wl * wavelength};
}

std::vector<double> flatten_positions(
    const std::vector<radar::sim::Vec3>& positions) {
  std::vector<double> flat;
  flat.reserve(positions.size() * 3);
  for (const radar::sim::Vec3& p : positions) {
    flat.push_back(p[0]);
    flat.push_back(p[1]);
    flat.push_back(p[2]);
  }
  return flat;
}

std::vector<double> uniform_ddm_offsets(int64_t num_tx) {
  std::vector<double> offsets(static_cast<size_t>(num_tx));
  for (int64_t m = 0; m < num_tx; ++m) {
    offsets[static_cast<size_t>(m)] =
        static_cast<double>(m) / static_cast<double>(num_tx);
  }
  return offsets;
}

std::vector<double> resolve_ddm_offsets(const radar::DDMCodeDescriptor& code) {
  code.validate();
  if (!code.doppler_offsets.empty()) {
    return code.doppler_offsets;
  }
  return uniform_ddm_offsets(code.num_tx);
}

radar::sim::SimResult simulate_mimo_tdm_interleaved(
    const MimoScene& scene, const radar::TXSchedule& schedule) {
  if (scene.tx_positions.empty()) {
    throw std::invalid_argument(
        "simulate_mimo_tdm_interleaved: need >= 1 TX element");
  }
  if (scene.rx_positions.empty()) {
    throw std::invalid_argument(
        "simulate_mimo_tdm_interleaved: need >= 1 RX element");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = scene.num_tx;
  const int64_t slow_time = num_tx * scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  const double physical_pri = scene.waveform.chirp_repetition_interval;
  const double fs = scene.waveform.sample_rate;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double c = scene.propagation_speed;

  radar::TXSchedule sched = schedule;
  sched.num_tx = num_tx;
  const std::vector<int64_t> pattern = sched.expand_chirp_tx_index(slow_time);

  radar::sim::SimResult result;
  result.shape = {num_rx, slow_time, samples};
  result.cube.assign(static_cast<size_t>(num_rx * slow_time * samples),
                     {0.0, 0.0});

  for (const radar::sim::PointTarget& target : scene.targets) {
    const radar::sim::TargetTruth truth = radar::sim::derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t k = 0; k < slow_time; ++k) {
      const int64_t tx =
          pattern[static_cast<size_t>(k)];
      if (tx < 0 || tx >= num_tx) {
        throw std::runtime_error("simulate_mimo_tdm_interleaved: bad TX index");
      }
      const radar::sim::Vec3& tx_pos =
          scene.tx_positions[static_cast<size_t>(tx)];
      const double range_k =
          truth.range +
          truth.radial_velocity * static_cast<double>(k) * physical_pri;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        const double aperture =
            radar::sim::dot(tx_pos, truth.look_direction) +
            radar::sim::dot(scene.rx_positions[static_cast<size_t>(rx)],
                            truth.look_direction);
        const double tau = (2.0 * range_k - aperture) / c;
        const double beat = slope * tau;
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          const double phase =
              2.0 * radar::kPi * (beat * t) - 2.0 * radar::kPi * f_c * tau;
          const std::complex<double> sample =
              target.amplitude *
              std::complex<double>(std::cos(phase), std::sin(phase));
          const size_t idx =
              static_cast<size_t>((rx * slow_time + k) * samples + s);
          result.cube[idx] += sample;
        }
      }
    }
  }

  if (scene.snr_db.has_value()) {
    const double noise_power = std::pow(10.0, -(*scene.snr_db) / 10.0);
    const double sigma = std::sqrt(noise_power / 2.0);
    std::mt19937_64 rng(scene.seed);
    std::normal_distribution<double> gauss(0.0, sigma);
    for (auto& v : result.cube) {
      v += std::complex<double>(gauss(rng), gauss(rng));
    }
  }
  return result;
}

radar::sim::SimResult simulate_mimo_ddm_coded(
    const MimoScene& scene, const std::vector<double>& offsets) {
  if (scene.tx_positions.empty() || scene.rx_positions.empty()) {
    throw std::invalid_argument("simulate_mimo_ddm_coded: need TX and RX");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = scene.num_tx;
  const int64_t num_chirps = scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  const double pri = scene.waveform.chirp_repetition_interval;
  const double fs = scene.waveform.sample_rate;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double c = scene.propagation_speed;

  if (static_cast<int64_t>(offsets.size()) != num_tx) {
    throw std::invalid_argument(
        "simulate_mimo_ddm_coded: offsets length must equal num_tx");
  }

  radar::sim::SimResult result;
  result.shape = {num_rx, num_chirps, samples};
  result.cube.assign(static_cast<size_t>(num_rx * num_chirps * samples),
                     {0.0, 0.0});

  for (const radar::sim::PointTarget& target : scene.targets) {
    const radar::sim::TargetTruth truth = radar::sim::derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t k = 0; k < num_chirps; ++k) {
      const double range_k =
          truth.range +
          truth.radial_velocity * static_cast<double>(k) * pri;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          std::complex<double> acc{0.0, 0.0};
          for (int64_t tx = 0; tx < num_tx; ++tx) {
            const radar::sim::Vec3& tx_pos =
                scene.tx_positions[static_cast<size_t>(tx)];
            const double aperture =
                radar::sim::dot(tx_pos, truth.look_direction) +
                radar::sim::dot(scene.rx_positions[static_cast<size_t>(rx)],
                                truth.look_direction);
            const double tau = (2.0 * range_k - aperture) / c;
            const double beat = slope * tau;
            const double code_phase =
                2.0 * radar::kPi * offsets[static_cast<size_t>(tx)] *
                static_cast<double>(k);
            const double phase =
                2.0 * radar::kPi * (beat * t) - 2.0 * radar::kPi * f_c * tau +
                code_phase;
            acc += target.amplitude *
                   std::complex<double>(std::cos(phase), std::sin(phase));
          }
          const size_t idx =
              static_cast<size_t>((rx * num_chirps + k) * samples + s);
          result.cube[idx] += acc;
        }
      }
    }
  }

  if (scene.snr_db.has_value()) {
    const double noise_power = std::pow(10.0, -(*scene.snr_db) / 10.0);
    const double sigma = std::sqrt(noise_power / 2.0);
    std::mt19937_64 rng(scene.seed);
    std::normal_distribution<double> gauss(0.0, sigma);
    for (auto& v : result.cube) {
      v += std::complex<double>(gauss(rng), gauss(rng));
    }
  }
  return result;
}

void load_scene_geometry(AppConfig* cfg, const YAML::Node& root,
                         double wavelength) {
  const YAML::Node tx = root["tx_array"];
  const double tx_spacing = tx["spacing"].as<double>();
  cfg->scene.tx_positions.clear();
  cfg->scene.tx_positions.reserve(static_cast<size_t>(cfg->scene.num_tx));
  for (int64_t itx = 0; itx < cfg->scene.num_tx; ++itx) {
    cfg->scene.tx_positions.push_back(radar::sim::Vec3{
        0.0, static_cast<double>(itx) * tx_spacing * wavelength, 0.0});
  }

  const YAML::Node rx = root["rx_array"];
  if (!rx) {
    throw std::runtime_error("rx_array section is required for MIMO");
  }
  const int64_t az_elements = rx["az_elements"].as<int64_t>();
  const int64_t el_elements = rx["el_elements"].as<int64_t>();
  const double rx_spacing = rx["spacing"].as<double>();
  if (az_elements < 2 || el_elements < 2) {
    throw std::runtime_error(
        "MIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2");
  }
  cfg->scene.rx_positions.clear();
  cfg->scene.rx_positions.reserve(
      static_cast<size_t>(az_elements * el_elements));
  for (int64_t iz = 0; iz < el_elements; ++iz) {
    for (int64_t iy = 0; iy < az_elements; ++iy) {
      cfg->scene.rx_positions.push_back(
          scale_position(static_cast<double>(iy), static_cast<double>(iz),
                         rx_spacing, wavelength));
    }
  }

  cfg->array_geometry = radar::sim::make_virtual_geometry(
      cfg->scene.tx_positions, cfg->scene.rx_positions, wavelength);
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
  cfg.report_waveform = waveform;

  const YAML::Node mcc = root["mimo_coherent_compensation"];
  if (!mcc || !mcc["multiplexing_scheme"]) {
    throw std::runtime_error(
        "fmcw_mimo: mimo_coherent_compensation.multiplexing_scheme is "
        "required");
  }
  cfg.scheme = mcc["multiplexing_scheme"].as<radar::MultiplexingScheme>();
  if (cfg.scheme != radar::MultiplexingScheme::kTdm &&
      cfg.scheme != radar::MultiplexingScheme::kDdm) {
    throw std::runtime_error(
        "fmcw_mimo: multiplexing_scheme must be tdm or ddm");
  }

  parse_run(root, &cfg);
  cfg.scene.samples_per_chirp = root["cube"]["samples_per_chirp"].as<int64_t>();

  const YAML::Node tx = root["tx_array"];
  if (!tx) {
    throw std::runtime_error("tx_array section is required for MIMO");
  }
  cfg.scene.num_tx = tx["num_tx"].as<int64_t>();
  if (cfg.scene.num_tx < 1) {
    throw std::runtime_error("tx_array.num_tx must be >= 1");
  }

  const double wavelength = cfg.scene.waveform.wavelength();
  cfg.wavelength = wavelength;
  load_scene_geometry(&cfg, root, wavelength);

  if (cfg.scheme == radar::MultiplexingScheme::kDdm) {
    cfg.scene.waveform.chirp_repetition_interval =
        waveform.chirp_repetition_interval();
    if (root["cube"]["num_chirps"]) {
      cfg.scene.num_chirps = root["cube"]["num_chirps"].as<int64_t>();
    } else {
      cfg.scene.num_chirps = root["cube"]["chirps_per_tx"].as<int64_t>();
    }
    if (!mcc["ddm_code"]) {
      throw std::runtime_error(
          "fmcw_mimo: mimo_coherent_compensation.ddm_code is required for DDM");
    }
    cfg.ddm_code = mcc["ddm_code"].as<radar::DDMCodeDescriptor>();
    if (cfg.ddm_code.num_tx != cfg.scene.num_tx) {
      throw std::runtime_error(
          "fmcw_mimo: ddm_code.num_tx must match tx_array.num_tx");
    }
    cfg.ddm_doppler_offsets = resolve_ddm_offsets(cfg.ddm_code);
    if (cfg.scene.num_chirps % cfg.scene.num_tx != 0) {
      throw std::runtime_error(
          "fmcw_mimo: cube.num_chirps must be divisible by tx_array.num_tx "
          "for DDM");
    }
    const int64_t band = cfg.scene.num_chirps / cfg.scene.num_tx;
    cfg.report_num_doppler = band - cfg.ddm_code.guard_bands;
    if (cfg.report_num_doppler <= 0) {
      throw std::runtime_error(
          "fmcw_mimo: DDM guard_bands leaves no Doppler bins per TX");
    }
  } else {
    const YAML::Node sim = root["sim"];
    if (!sim || !sim["physical_chirp_repetition_interval"]) {
      throw std::runtime_error(
          "fmcw_mimo: sim.physical_chirp_repetition_interval is required "
          "for TDM");
    }
    cfg.scene.waveform.chirp_repetition_interval =
        sim["physical_chirp_repetition_interval"].as<double>();
    cfg.scene.num_chirps = root["cube"]["chirps_per_tx"].as<int64_t>();
    if (!root["mimo_deinterleave"]) {
      throw std::runtime_error(
          "fmcw_mimo: mimo_deinterleave section is required for TDM");
    }
    cfg.tx_schedule = root["mimo_deinterleave"]["tx_schedule"]
                          .as<radar::TXSchedule>();
    if (cfg.tx_schedule.num_tx != cfg.scene.num_tx) {
      throw std::runtime_error(
          "mimo_deinterleave.tx_schedule.num_tx must match tx_array.num_tx");
    }
    cfg.report_num_doppler = cfg.scene.num_chirps;
  }

  cfg.tx_positions_flat = flatten_positions(cfg.scene.tx_positions);
  cfg.rx_positions_flat = flatten_positions(cfg.scene.rx_positions);

  parse_targets_range_az_el(root, &cfg.scene, "fmcw_mimo");
  parse_noise(root, &cfg.scene);
  parse_viz_toggles(root, &cfg);
  parse_plots(root, &cfg);

  // Optional fixed velocity-plot Y extent (dense/MIMO plot knob).
  if (const YAML::Node plots = root["plots"]; plots && !plots.IsNull()) {
    if (const YAML::Node ylim = plots["velocity_ylim"]) {
      if (!ylim.IsSequence() || ylim.size() != 2) {
        throw std::runtime_error("plots.velocity_ylim must be a [lo, hi] pair");
      }
      cfg.velocity_ylim_lo = ylim[0].as<float>();
      cfg.velocity_ylim_hi = ylim[1].as<float>();
    }
  }

  // Boresight view axes mirror the beamscan estimator's angle grid (READ-ONLY
  // MIRROR of an operator block; the estimator owns az_angles/el_angles via
  // from_config()).
  const YAML::Node be = root["beamscan_estimator"];
  grid_extent(extract_grid(be ? be["az_angles"] : YAML::Node()), -0.35, 0.35,
              &cfg.az_lo, &cfg.az_hi);
  grid_extent(extract_grid(be ? be["el_angles"] : YAML::Node()), -0.1, 0.1,
              &cfg.el_lo, &cfg.el_hi);
  return cfg;
}

// Post-DDM-demux velocity axis for reporting: CFAR / feature assembly run on
// VirtualArrayAssemblyOp rd_maps whose Doppler width is num_doppler_per_tx,
// not the pre-demux FMCWDopplerProcessorOp axis.
class DdmReportVelocityBinsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DdmReportVelocityBinsOp)
  DdmReportVelocityBinsOp() = default;

  radar::FMCWWaveformDescriptor waveform;
  int64_t num_doppler{0};
  // The DDM demux keeps the full slow-time FFT bin resolution PRF/N (N = the
  // pre-demux Doppler length = num_chirps) but retains only W = N/num_tx minus
  // guard bins, re-centered. So the demuxed velocity axis has W bins spaced at
  // PRF/N (NOT PRF/W); using PRF/W would inflate velocities by ~num_tx and
  // fragment each target across Doppler in clustering.
  double bin_spacing_hz{0.0};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("sync");
    spec.output<holoscan::TensorMap>("velocity_bins");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    (void)radar::receive_single(op_input, "sync", "DdmReportVelocityBinsOp");
    // The axis is constant per run; build and upload it once, then re-emit the
    // cached device tensor each tick (the "sync" input just paces the emit).
    if (!velocity_bins_) {
      velocity_bins_ = build_axis();
    }
    holoscan::TensorMap out;
    out.insert({"velocity_bins", velocity_bins_});
    op_output.emit(out, "velocity_bins");
  }

 private:
  std::shared_ptr<holoscan::Tensor> build_axis() const {
    if (num_doppler <= 0) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: num_doppler must be positive");
    }
    if (!waveform.is_valid()) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: waveform descriptor is invalid");
    }
    if (bin_spacing_hz <= 0.0) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: bin_spacing_hz must be > 0");
    }
    const double wavelength = waveform.wavelength();
    // Center-zero axis: bin i maps to Doppler frequency (i - W/2) * df.
    std::vector<float> axis(static_cast<size_t>(num_doppler));
    for (int64_t i = 0; i < num_doppler; ++i) {
      const double f =
          static_cast<double>(i - num_doppler / 2) * bin_spacing_hz;
      axis[static_cast<size_t>(i)] =
          static_cast<float>(-f * wavelength / 2.0);
    }
    auto tensor = radar::make_device_tensor({num_doppler},
                                            radar::real_dtype<float>());
    if (cudaMemcpy(tensor->data(), axis.data(), sizeof(float) * axis.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: velocity_bins H2D upload failed");
    }
    return tensor;
  }

  std::shared_ptr<holoscan::Tensor> velocity_bins_;
};

class MimoTdmAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MimoTdmAdcSourceOp)
  MimoTdmAdcSourceOp() = default;

  MimoScene scene;
  radar::TXSchedule tx_schedule;
  double frame_period{0.1};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const radar::sim::SimResult sim =
        simulate_mimo_tdm_interleaved(scene, tx_schedule);
    if (sim.shape.size() != 3) {
      throw std::runtime_error(
          "MimoTdmAdcSourceOp: expected rank-3 (N_rx, M*L, samples)");
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
      throw std::runtime_error("MimoTdmAdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    radar::sim::advance_scene(&scene, frame_period);
  }
};

class MimoDdmAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MimoDdmAdcSourceOp)
  MimoDdmAdcSourceOp() = default;

  MimoScene scene;
  std::vector<double> doppler_offsets;
  double frame_period{0.1};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const radar::sim::SimResult sim =
        simulate_mimo_ddm_coded(scene, doppler_offsets);
    if (sim.shape.size() != 3) {
      throw std::runtime_error(
          "MimoDdmAdcSourceOp: expected rank-3 (N_rx, num_chirps, samples)");
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
      throw std::runtime_error("MimoDdmAdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    radar::sim::advance_scene(&scene, frame_period);
  }
};

}  // namespace

class FmcwMimoApp : public holoscan::Application {
 public:
  AppConfig cfg;

  void compose() override {
    const bool ddm = cfg.scheme == radar::MultiplexingScheme::kDdm;
    const std::string window_title =
        ddm ? "FMCW MIMO DDM - virtual RD + clusters"
            : "FMCW MIMO TDM - virtual RD + clusters";

    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", from_config("doppler_processor"));
    auto compensation =
        make_operator<radar::MIMOCoherentCompensationOp<float>>(
            "mimo_coherent_compensation",
            from_config("mimo_coherent_compensation"));
    auto vaa = make_operator<radar::VirtualArrayAssemblyOp<float>>(
        "virtual_array_assembly", from_config("virtual_array_assembly"),
        holoscan::Arg("tx_positions") = cfg.tx_positions_flat,
        holoscan::Arg("rx_positions") = cfg.rx_positions_flat,
        holoscan::Arg("wavelength") = cfg.wavelength);

    if (ddm) {
      auto source = make_operator<MimoDdmAdcSourceOp>(
          "adc_source",
          make_condition<holoscan::CountCondition>("frame_count",
                                                   cfg.num_frames),
          make_condition<holoscan::PeriodicCondition>(
              "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
      source->scene = cfg.scene;
      source->doppler_offsets = cfg.ddm_doppler_offsets;
      source->frame_period = cfg.frame_period;

      add_flow(source, range, {{"adc_cube", "adc_cube"}});
      add_flow(range, doppler, {{"range_profiles", "range_profiles"}});
      add_flow(doppler, compensation, {{"rd_maps", "rd_maps"}});
      add_flow(compensation, clutter, {{"rd_maps", "range_profiles"}});
      add_flow(clutter, vaa, {{"range_profiles", "rd_maps"}});
    } else {
      auto source = make_operator<MimoTdmAdcSourceOp>(
          "adc_source",
          make_condition<holoscan::CountCondition>("frame_count",
                                                   cfg.num_frames),
          make_condition<holoscan::PeriodicCondition>(
              "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
      source->scene = cfg.scene;
      source->tx_schedule = cfg.tx_schedule;
      source->frame_period = cfg.frame_period;

      auto deinterleave = make_operator<radar::MIMODeinterleaveOp<float>>(
          "mimo_deinterleave", from_config("mimo_deinterleave"));

      add_flow(source, deinterleave, {{"adc_cube", "adc_cube"}});
      add_flow(deinterleave, range, {{"adc_cube", "adc_cube"}});
      add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
      add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
      add_flow(doppler, compensation, {{"rd_maps", "rd_maps"}});
      add_flow(compensation, vaa, {{"rd_maps", "rd_maps"}});
    }

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

    std::shared_ptr<DdmReportVelocityBinsOp> ddm_velocity_bins;
    if (ddm) {
      ddm_velocity_bins =
          make_operator<DdmReportVelocityBinsOp>("ddm_velocity_bins");
      ddm_velocity_bins->waveform = cfg.report_waveform;
      ddm_velocity_bins->num_doppler = cfg.report_num_doppler;
      ddm_velocity_bins->bin_spacing_hz =
          (1.0 / cfg.report_waveform.chirp_repetition_interval()) /
          static_cast<double>(cfg.scene.num_chirps);
      add_flow(range, ddm_velocity_bins, {{"range_bins", "sync"}});
    }

    add_flow(vaa, cfar, {{"rd_maps", "rd_maps"}});
    add_flow(vaa, gather, {{"rd_maps", "rd_maps"}});
    add_flow(cfar, gather, {{"detection_idx", "detection_idx"}});
    add_flow(gather, beamscan, {{"snapshot", "snapshot"},
                                {"is_valid", "is_valid"}});
    add_flow(vaa, beamscan, {{"array_geometry", "array_geometry"}});
    add_flow(beamscan, feature_assembly,
             {{"aoa_estimates", "aoa_estimates"}});
    add_flow(beamscan, spectrum_sink, {{"angle_spectrum", "input"}});

    add_flow(cfar, det_report, {{"detection_idx", "detection_idx"},
                                  {"detection_score", "detection_score"}});
    add_flow(range, det_report, {{"range_bins", "range_bins"}});
    if (ddm) {
      add_flow(ddm_velocity_bins, det_report,
               {{"velocity_bins", "velocity_bins"}});
    } else {
      add_flow(doppler, det_report, {{"velocity_bins", "velocity_bins"}});
    }
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    add_flow(cfar, feature_assembly, {{"detection_idx", "detection_idx"}});
    add_flow(range, feature_assembly, {{"range_bins", "range_bins"}});
    if (ddm) {
      add_flow(ddm_velocity_bins, feature_assembly,
               {{"velocity_bins", "velocity_bins"}});
    } else {
      add_flow(doppler, feature_assembly,
               {{"velocity_bins", "velocity_bins"}});
    }
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
    if (ddm) {
      add_flow(ddm_velocity_bins, cluster_report,
               {{"velocity_bins", "velocity_bins"}});
    } else {
      add_flow(doppler, cluster_report,
               {{"velocity_bins", "velocity_bins"}});
    }

    if (cfg.visualize) {
      auto ch_slice = make_operator<RdChannelSliceOp>("rd_channel_slice");
      ch_slice->channel = 0;
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      HolovizLayout layout;
      layout.window_title = window_title;
      layout.width = cfg.plots.enable ? 1200U : 800U;
      layout.height = cfg.plots.enable ? 760U : 600U;
      layout.headless = cfg.headless;
      layout.plots_enabled = cfg.plots.enable;
      if (cfg.plots.enable) {
        layout.right_panels = range_velocity_boresight_panels();
      }
      auto visualizer = make_rd_holoviz_op(this, "holoviz", layout);

      std::shared_ptr<RollingPlotOp> plot;
      std::shared_ptr<BoresightViewOp> boresight;
      if (cfg.plots.enable) {
        plot = make_operator<RollingPlotOp>("rolling_plot");
        plot->window = cfg.plots.window;
        plot->frame_period = cfg.frame_period;
        plot->velocity_ylim_lo = cfg.velocity_ylim_lo;
        plot->velocity_ylim_hi = cfg.velocity_ylim_hi;
        boresight = make_operator<BoresightViewOp>("boresight_view");
        boresight->az_lo = cfg.az_lo;
        boresight->az_hi = cfg.az_hi;
        boresight->el_lo = cfg.el_lo;
        boresight->el_hi = cfg.el_hi;
      }

      add_flow(vaa, ch_slice, {{"rd_maps", "rd_maps"}});
      add_flow(ch_slice, to_image, {{"rd_map", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cluster, overlay, {{"labels", "labels"}});
      add_flow(ch_slice, overlay, {{"rd_map", "rd_maps"}});
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
      if (plot) {
        add_flow(reduction, plot, {{"rep_idx", "rep_idx"}});
        add_flow(range, plot, {{"range_bins", "range_bins"}});
        if (ddm) {
          add_flow(ddm_velocity_bins, plot,
                   {{"velocity_bins", "velocity_bins"}});
        } else {
          add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
        }
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

int main(int argc, char** argv) {
  return run_fmcw_app<FmcwMimoApp>(argc, argv, "fmcw_mimo", "fmcw_mimo.yaml",
                                   load_config);
}
