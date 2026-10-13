// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Point-target FMCW scene model — the shared, STL-only data types and geometry
// for the analytic simulator: the waveform/target/scene structs, the derived
// per-target ground truth (range / radial velocity / look direction), and the
// per-frame target mover. Consumed by the
// runtime GPU simulator (simulator.hpp),
// the MIMO GPU core (simulator.hpp), the source operators, and config parsing.
//
// The closed-form CPU signal model itself — simulate() / reference_chirp() /
// to_real_if() — is the analytic ground truth used only by tests, and lives in
// tests/cpp/utils/sim_reference.hpp. The runtime path synthesizes on the device
// (simulator.hpp), validated against that
// CPU reference to tolerance. This header
// stays STL-only (no operator or MatX includes) so the scene model and the
// reference derived from it remain independent of the code under test.
//
// Modeling assumptions: stop-and-hop (target frozen within a chirp, advances
// between chirps), ego motion folded into target relative velocity by the
// caller, far-field plane wave, monostatic, and single-slope sweeps. The
// configured waveform and excitation repeat cyclically: propagation delay is
// folded by the physical CRI for fast-time beat generation, while the full
// delay remains in carrier/spatial phase. A magnitude-only analog IF response
// precedes sampling.
//
// Sign conventions (must match the operators' locked semantics):
//  - Dechirped sample: s(k, t) = A * exp(+j*2*pi*f_b(k)*t - j*2*pi*f_c*tau_k)
//    with tau_k = q*CRI + delta and beat frequency f_b(k) = slope * delta,
//    so a forward FFT along
//    fast-time peaks at the positive beat frequency (bin 0 = zero range).
//  - The slow-time carrier phase -2*pi*f_c*tau_k advances by
//    2*pi*f_d*PRI per chirp with f_d = -2*v_r/lambda, so radial velocity
//    v_r (positive = receding) satisfies the contract v = -f_d*lambda/2.
//  - Baseband reference chirp: r(t) = exp(+j*pi*slope*t^2). The pre-mixer
//    output is s(k, t) * r(t); hardware/software dechirp is multiplication
//    by conj(r). Operators synthesizing a dechirp reference must use this
//    same convention.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/math/constants.hpp"

namespace holoscan::radar::sim {

inline constexpr double kSimSpeedOfLight = kSpeedOfLight;

using Vec3 = std::array<double, 3>;

inline double dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline double norm(const Vec3& a) {
  return std::sqrt(dot(a, a));
}

struct PointTarget {
  Vec3 position{};  // m, array-local Cartesian frame
  Vec3 velocity{};  // m/s, relative (ego motion folded)
  std::complex<double> amplitude{1.0, 0.0};
};

// Analytic waveform fields (mirrors FMCWWaveformDescriptor).
struct SimWaveform {
  double carrier_frequency{77e9};           // Hz
  double bandwidth{150e6};                  // Hz (slope = bandwidth/duration)
  double chirp_duration{20e-6};             // s
  double sample_rate{12.8e6};               // Hz
  double chirp_repetition_interval{25e-6};  // s (PRI)

  double slope() const { return bandwidth / chirp_duration; }
  double wavelength(double c = kSimSpeedOfLight) const { return c / carrier_frequency; }
};

enum class IfSamplingMode {
  kComplexIq,
  kReal,
};

enum class AnalogIfResponseType {
  kRaisedCosine,
  kNone,
};

struct AnalogIfConfig {
  IfSamplingMode sampling_mode{IfSamplingMode::kComplexIq};
  AnalogIfResponseType response{AnalogIfResponseType::kRaisedCosine};
  double passband_fraction{0.9};
  double stopband_fraction{1.0};
};

enum class RangeLossModel {
  kNone,
  kMonostaticFreeSpace,
};

struct RangeLossConfig {
  RangeLossModel model{RangeLossModel::kNone};
  double reference_range{1.0};  // m
};

enum class SceneVisibilityModel {
  kFrontHalfSpace,
  kFrustum,
  kAll,
};

// Binary geometric coverage in the radar-local Cartesian frame. This is an
// application scene-selection policy, not an antenna-gain model. The analytic
// solvers remain isotropic and apply only to the targets supplied to them.
struct SceneVisibility {
  SceneVisibilityModel model{SceneVisibilityModel::kFrontHalfSpace};
  Vec3 boresight_direction{1.0, 0.0, 0.0};
  Vec3 up_direction{0.0, 0.0, 1.0};
  std::optional<double> horizontal_fov{};  // full angle, radians
  std::optional<double> vertical_fov{};    // full angle, radians
};

struct SimScene {
  std::vector<PointTarget> targets;
  SimWaveform waveform{};
  int64_t num_chirps{64};
  int64_t samples_per_chirp{256};
  // Element positions (m, array-local frame). simulate() models a single TX
  // (tx_positions must hold exactly one element); the MIMO simulators
  // (simulator.hpp::simulate_mimo_gpu) consume the full TX set for TDM/DDM.
  std::vector<Vec3> tx_positions{Vec3{0.0, 0.0, 0.0}};
  std::vector<Vec3> rx_positions{Vec3{0.0, 0.0, 0.0}};
  double propagation_speed{kSimSpeedOfLight};
  // Magnitude-only analog IF response applied to each clean target tone before
  // sampling. The default is complex-IQ with a 10% anti-alias transition band.
  AnalogIfConfig analog_if{};
  // Optional relative monostatic propagation law. Target amplitude is the
  // received-voltage coefficient at reference_range when enabled.
  RangeLossConfig range_loss{};
  // Additive complex white Gaussian noise, relative to a unit-amplitude
  // target (SNR in dB). Unset → noise-free.
  std::optional<double> snr_db{};
  uint64_t seed{0x5eed5eedULL};
  // true → dechirped (IF) cube; false → pre-mixer cube (for apply_dechirp
  // tests): pre_mixer(k, t) = dechirped(k, t) * reference(t).
  bool dechirped{true};
};

struct SceneVisibilityBasis {
  Vec3 forward{};
  Vec3 horizontal{};
  Vec3 vertical{};
};

inline Vec3 normalize_direction(const Vec3& direction, const char* field) {
  const double length = norm(direction);
  if (!std::isfinite(length) || length <= 0.0) {
    throw std::invalid_argument(std::string("scene visibility ") + field +
                                " must be finite and nonzero");
  }
  return Vec3{direction[0] / length, direction[1] / length, direction[2] / length};
}

inline SceneVisibilityBasis visibility_basis(const SceneVisibility& visibility) {
  const Vec3 forward = normalize_direction(visibility.boresight_direction, "boresight_direction");
  const double up_projection = dot(visibility.up_direction, forward);
  const Vec3 orthogonal_up{visibility.up_direction[0] - up_projection * forward[0],
                           visibility.up_direction[1] - up_projection * forward[1],
                           visibility.up_direction[2] - up_projection * forward[2]};
  const Vec3 vertical = normalize_direction(orthogonal_up, "up_direction");
  return SceneVisibilityBasis{
      .forward = forward,
      .horizontal = cross(vertical, forward),
      .vertical = vertical,
  };
}

inline void validate_scene_visibility(const SceneVisibility& visibility) {
  if (visibility.model == SceneVisibilityModel::kAll) {
    return;
  }
  (void)normalize_direction(visibility.boresight_direction, "boresight_direction");
  if (visibility.model != SceneVisibilityModel::kFrustum) {
    return;
  }
  (void)visibility_basis(visibility);
  const auto valid_fov = [](const std::optional<double>& value) {
    return value.has_value() && std::isfinite(*value) && *value > 0.0 && *value < kPi;
  };
  if (!valid_fov(visibility.horizontal_fov) || !valid_fov(visibility.vertical_fov)) {
    throw std::invalid_argument(
        "frustum visibility requires horizontal_fov and vertical_fov in "
        "the open interval (0, pi)");
  }
}

inline bool target_is_visible(const Vec3& position, const SceneVisibility& visibility) {
  if (visibility.model == SceneVisibilityModel::kAll) {
    return true;
  }
  const Vec3 forward_direction =
      normalize_direction(visibility.boresight_direction, "boresight_direction");
  const double forward = dot(position, forward_direction);
  if (!(forward > 0.0)) {
    return false;
  }
  if (visibility.model == SceneVisibilityModel::kFrontHalfSpace) {
    return true;
  }
  validate_scene_visibility(visibility);
  const SceneVisibilityBasis basis = visibility_basis(visibility);
  const double horizontal_limit = forward * std::tan(*visibility.horizontal_fov * 0.5);
  const double vertical_limit = forward * std::tan(*visibility.vertical_fov * 0.5);
  const double tolerance = 1.0e-12 * std::max(1.0, forward);
  return std::abs(dot(position, basis.horizontal)) <= horizontal_limit + tolerance &&
         std::abs(dot(position, basis.vertical)) <= vertical_limit + tolerance;
}

inline SimScene visible_scene(const SimScene& scene, const SceneVisibility& visibility) {
  validate_scene_visibility(visibility);
  SimScene result = scene;
  result.targets.erase(std::remove_if(result.targets.begin(), result.targets.end(),
                                      [&](const PointTarget& target) {
                                        return !target_is_visible(target.position, visibility);
                                      }),
                       result.targets.end());
  return result;
}

// Per-target derived physical ground truth (tests assert in the
// physical domain, not on bin indices).
struct TargetTruth {
  double range{};            // m
  double radial_velocity{};  // m/s, positive = receding
  Vec3 look_direction{};     // unit vector, position / |position|
};

struct SimResult {
  // Complex IF cube, compact row-major:
  //   num_rx == 1 → (num_chirps, samples_per_chirp)
  //   num_rx  > 1 → (num_rx, num_chirps, samples_per_chirp)
  std::vector<std::complex<double>> cube;
  std::vector<int64_t> shape;
  std::vector<TargetTruth> truth;
  IfSamplingMode sampling_mode{IfSamplingMode::kComplexIq};
};

inline TargetTruth derive_truth(const PointTarget& target) {
  TargetTruth truth;
  truth.range = norm(target.position);
  if (truth.range <= 0.0) {
    throw std::invalid_argument(
        "derive_truth: target at the origin has no defined look "
        "direction");
  }
  truth.look_direction = {target.position[0] / truth.range, target.position[1] / truth.range,
                          target.position[2] / truth.range};
  truth.radial_velocity = dot(target.velocity, truth.look_direction);
  return truth;
}

// Advance each target's base position by one frame period (dt seconds). The
// simulator models chirp-to-chirp motion within a CPI; this carries that motion
// across CPIs so successive frames see the target at a new position. Templated
// on the scene type: any struct exposing a `.targets` container of PointTarget
// works — SimScene here, or an application's own multi-TX scene struct. This is
// the default mover (see SceneMover).
template <class Scene>
inline void advance_scene(Scene* scene, double dt) {
  for (PointTarget& target : scene->targets) {
    for (int axis = 0; axis < 3; ++axis) {
      target.position[axis] += target.velocity[axis] * dt;
    }
  }
}

// Per-frame scene-evolution strategy (the
// mover / target-source seam): given the
// scene and the frame period, mutate it for the next frame. The default
// (linear_mover) moves each target along its velocity; a custom mover can
// implement orbits/maneuvers, or clear-and-regenerate scene.targets entirely (a
// procedural "scatterer source"). The
// source operators call the configured mover
// each tick, so the motion model is swappable without touching the solver.
using SceneMover = std::function<void(SimScene&, double dt)>;

// The default constant-velocity mover, as a SceneMover.
inline SceneMover linear_mover() {
  return [](SimScene& scene, double dt) {
    advance_scene(&scene, dt);
  };
}

}  // namespace holoscan::radar::sim
