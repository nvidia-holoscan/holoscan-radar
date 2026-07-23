// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared parsing of the app-owned YAML sections for the FMCW example apps.
//
// IMPORTANT — the config boundary: these helpers parse ONLY app-owned sections
// (waveform, run, cube, targets, noise_snr_db, plots, the visualize/headless/
// skip_display_check toggles). The signal-processing / detection / clustering /
// reduction operators take their parameters natively through Holoscan's
// from_config() (backed by the YAML::convert<> specializations in
// radar_common), so their blocks are NEVER read here. An app may separately
// read an operator block purely to *mirror* a value into a viz/geometry grid
// (e.g. the beamscan angle grids for the boresight axes) — that stays in the
// app and must be commented as a read-only mirror, not moved here.
//
// Header-only: every helper is inline / a template, so multiple app TUs can
// include this with no ODR conflict.

#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/sim/point_target_sim.hpp"

namespace holoscan::radar::app {

// Run cadence + visualization toggles shared by every app. Each app embeds this
// as a base class of AppConfig and adds its scene + app-specific fields;
// run_fmcw_app() (driver.hpp) reads the visualize/headless/skip_display_check
// members through the derived cfg.
struct CommonRunConfig {
  int64_t num_frames{1};
  double frame_period{0.1};  // s, between emitted cubes (and target motion)
  // Wall-clock CadenceMeter (source inter-emit + DetectionReport e2e).
  bool meter_cadence{true};
  int64_t meter_log_every{0};  // 0 = warn-only + end summary; N = INFO every N
  bool visualize{false};     // live Holoviz window (requires a GUI host)
  bool skip_display_check{false};  // skip the startup display probe
  bool headless{false};            // render offscreen (no window / display)
  // CFAR peak-grouping mode. App-owned top-level key, forwarded to
  // CFARDetectorOp via an explicit Arg("peak_only"). When true the CFAR peaks
  // are one-per-target, so the app drops the DBSCAN clustering + reduction
  // stages and reports the CFAR detections directly (see each app's compose()).
  bool peak_only{false};
  struct PlotsConfig {
    bool enable{true};
    int64_t window{120};  // frames retained in the scrolling window
  } plots;
};

// Throw with the offending key rather than failing mid-graph: these values feed
// cudaMalloc sizing and the Count/Periodic conditions, where bad values surface
// as cryptic CUDA/GXF errors.
template <class T>
inline void require_positive(const char* key, T value) {
  if (value <= 0) {
    throw std::runtime_error(std::string(key) + " must be > 0, got " +
                             std::to_string(value));
  }
}

// frame_period feeds a PeriodicCondition in nanoseconds. require_positive
// rejects <= 0, but NaN slips through (NaN <= 0 is false), a positive
// sub-nanosecond period rounds to a zero (spinning) interval, and a huge one
// overflows the int64 cast. Require a finite value whose nanosecond count lands
// in [1, INT64_MAX].
inline void validate_frame_period(double seconds) {
  if (!std::isfinite(seconds)) {
    throw std::runtime_error("run.frame_period must be finite");
  }
  const double ns = seconds * 1e9;
  if (ns < 1.0 ||
      ns > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "run.frame_period must be within [1 ns, INT64_MAX ns], got " +
        std::to_string(seconds) + " s");
  }
}

// Decode the shared waveform anchor through the same YAML::convert<> the
// operators use, then mirror it into the simulator's decoupled SimWaveform. The
// descriptor derives bandwidth at construction, so a bandwidth- or
// slope-spelled config maps to the same beat rate. `Scene` is any struct with
// a `.waveform` of type sim::SimWaveform (SimScene or an app's own scene).
template <class Scene>
inline void parse_waveform_into_scene(const YAML::Node& root, Scene* scene) {
  const auto waveform = root["waveform"].as<FMCWWaveformDescriptor>();
  scene->waveform.carrier_frequency = waveform.carrier_frequency();
  scene->waveform.bandwidth = waveform.bandwidth();
  scene->waveform.chirp_duration = waveform.chirp_duration();
  scene->waveform.sample_rate = waveform.sample_rate();
  scene->waveform.chirp_repetition_interval =
      waveform.chirp_repetition_interval();
}

// run.{num_frames, frame_period, meter_*} + validation.
inline void parse_run(const YAML::Node& root, CommonRunConfig* cfg) {
  cfg->num_frames = root["run"]["num_frames"].as<int64_t>();
  cfg->frame_period = root["run"]["frame_period"].as<double>();
  require_positive("run.num_frames", cfg->num_frames);
  require_positive("run.frame_period", cfg->frame_period);
  validate_frame_period(cfg->frame_period);
  if (root["run"]["meter_cadence"]) {
    cfg->meter_cadence = root["run"]["meter_cadence"].as<bool>();
  }
  if (root["run"]["meter_log_every"]) {
    cfg->meter_log_every = root["run"]["meter_log_every"].as<int64_t>();
  }
  if (cfg->meter_log_every < 0) {
    throw std::runtime_error("run.meter_log_every must be >= 0");
  }
}

// cube.{num_chirps, samples_per_chirp} + validation. `Scene` is any struct with
// int64 `.num_chirps` / `.samples_per_chirp` (SimScene). MIMO parses its own
// cube (chirps_per_tx vs num_chirps) and does not use this.
template <class Scene>
inline void parse_cube(const YAML::Node& root, Scene* scene) {
  scene->num_chirps = root["cube"]["num_chirps"].as<int64_t>();
  scene->samples_per_chirp = root["cube"]["samples_per_chirp"].as<int64_t>();
  require_positive("cube.num_chirps", scene->num_chirps);
  require_positive("cube.samples_per_chirp", scene->samples_per_chirp);
}

// Convenience for the standard apps (siso/simo/dense): waveform + run + cube.
template <class Scene>
inline void parse_run_and_cube(const YAML::Node& root, CommonRunConfig* cfg,
                               Scene* scene) {
  parse_waveform_into_scene(root, scene);
  parse_run(root, cfg);
  parse_cube(root, scene);
}

// noise_snr_db → scene->snr_db (std::optional<double>). Unset leaves it empty
// (noise-free). `Scene` is any struct with a `.snr_db` optional.
template <class Scene>
inline void parse_noise(const YAML::Node& root, Scene* scene) {
  const YAML::Node noise = root["noise_snr_db"];
  if (noise && !noise.IsNull()) {
    scene->snr_db = noise.as<double>();
  }
}

inline void parse_viz_toggles(const YAML::Node& root, CommonRunConfig* cfg) {
  if (const YAML::Node v = root["visualize"]; v && !v.IsNull()) {
    cfg->visualize = v.as<bool>();
  }
  if (const YAML::Node v = root["skip_display_check"]; v && !v.IsNull()) {
    cfg->skip_display_check = v.as<bool>();
  }
  if (const YAML::Node v = root["headless"]; v && !v.IsNull()) {
    cfg->headless = v.as<bool>();
  }
}

// App-owned top-level `peak_only` flag (default false). The app forwards it to
// CFARDetectorOp via Arg("peak_only") AND uses it to gate the clustering
// subgraph, so it is an app config key (not read from the cfar_detector
// operator block, per the config-boundary note above).
inline void parse_peak_only(const YAML::Node& root, CommonRunConfig* cfg) {
  if (const YAML::Node v = root["peak_only"]; v && !v.IsNull()) {
    cfg->peak_only = v.as<bool>();
  }
}

// plots.{enable, window}. Apps with extra plot knobs (e.g. velocity_ylim) read
// those from the same block themselves.
inline void parse_plots(const YAML::Node& root, CommonRunConfig* cfg) {
  if (const YAML::Node plots = root["plots"]; plots && !plots.IsNull()) {
    if (plots["enable"]) {
      cfg->plots.enable = plots["enable"].as<bool>();
    }
    if (plots["window"]) {
      cfg->plots.window = plots["window"].as<int64_t>();
    }
  }
  if (cfg->plots.window <= 1) {
    throw std::runtime_error("plots.window must be > 1, got " +
                             std::to_string(cfg->plots.window));
  }
}

// 1-D targets (SISO): (range, velocity) along boresight, amplitude real.
template <class Scene>
inline void parse_targets_range1d(const YAML::Node& root, Scene* scene) {
  for (const YAML::Node& t : root["targets"]) {
    const double range = t["range"].as<double>();
    const double velocity = t["velocity"].as<double>();
    const double amplitude = t["amplitude"].as<double>();
    scene->targets.push_back(sim::PointTarget{sim::Vec3{range, 0.0, 0.0},
                                              sim::Vec3{velocity, 0.0, 0.0},
                                              {amplitude, 0.0}});
  }
}

// 3-D targets (SIMO/dense/MIMO): placed by (range, azimuth, elevation) in the
// array-local frame — dir = (cos el cos az, cos el sin az, sin el),
// position = range * dir — with an explicit (vx, vy, vz) velocity vector, so
// targets move on all three axes, not just radially.
template <class Scene>
inline void parse_targets_range_az_el(const YAML::Node& root, Scene* scene,
                                      const char* who) {
  for (const YAML::Node& t : root["targets"]) {
    const double range = t["range"].as<double>();
    const double azimuth = t["azimuth"].as<double>();
    const double elevation = t["elevation"].as<double>();
    const double amplitude = t["amplitude"].as<double>();
    const YAML::Node vel = t["velocity"];
    if (!vel || !vel.IsSequence() || vel.size() != 3) {
      throw std::runtime_error(
          std::string(who) +
          ": each target's velocity must be a 3-vector [vx, vy, vz] in m/s "
          "(array frame)");
    }
    const sim::Vec3 dir{std::cos(elevation) * std::cos(azimuth),
                        std::cos(elevation) * std::sin(azimuth),
                        std::sin(elevation)};
    scene->targets.push_back(sim::PointTarget{
        sim::Vec3{range * dir[0], range * dir[1], range * dir[2]},
        sim::Vec3{vel[0].as<double>(), vel[1].as<double>(),
                  vel[2].as<double>()},
        {amplitude, 0.0}});
  }
}

// Convert a YAML sequence of scalars to a vector<double> (empty if absent).
inline std::vector<double> extract_grid(const YAML::Node& seq) {
  std::vector<double> grid;
  if (seq && seq.IsSequence()) {
    grid.reserve(seq.size());
    for (const YAML::Node& v : seq) {
      grid.push_back(v.as<double>());
    }
  }
  return grid;
}

// Derive an axis extent [lo, hi] from an angle grid, falling back to
// [def_lo, def_hi] when the grid is empty and padding a degenerate
// (single-value) axis so it renders a visible span.
inline void grid_extent(const std::vector<double>& grid, double def_lo,
                        double def_hi, double* lo, double* hi) {
  if (!grid.empty()) {
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    for (double x : grid) {
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
}

}  // namespace holoscan::radar::app
