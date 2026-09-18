// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Point-target simulator self-checks (radar_common/sim). The simulator is a
// first-class test dependency; its analytic outputs (ground truth, beat
// frequency, Doppler phase, dechirp convention, SIMO layout) are cross-checked
// here independently of the operator implementations.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "utils/additives_reference.hpp"
#include "utils/mimo_reference.hpp"
#include "utils/sim_reference.hpp"
#include "utils/bin_model.hpp"

namespace {

using holoscan::radar::kPi;
namespace sim = holoscan::radar::sim;
namespace rd = holoscan::radar;  // descriptor types (TXSchedule, DDMCode...)
namespace fix = holoscan::radar::testing;  // bin-model / fixture helpers

sim::SimScene single_target_scene() {
  sim::SimScene scene;
  scene.waveform.carrier_frequency = 77e9;
  scene.waveform.bandwidth = 150e6;
  scene.waveform.chirp_duration = 20e-6;
  scene.waveform.sample_rate = 12.8e6;
  scene.waveform.chirp_repetition_interval = 25e-6;
  scene.num_chirps = 64;
  scene.samples_per_chirp = 256;
  scene.targets.push_back(
      {sim::Vec3{0.0, 40.0, 0.0}, sim::Vec3{0.0, 5.0, 0.0}, {1.0, 0.0}});
  return scene;
}

sim::SimScene scene_for_beat(double beat_frequency, int64_t ambiguity_order = 0) {
  auto scene = single_target_scene();
  scene.num_chirps = 2;
  scene.targets.clear();
  const double residual_delay = beat_frequency / scene.waveform.slope();
  const double delay = residual_delay +
                       static_cast<double>(ambiguity_order) *
                           scene.waveform.chirp_repetition_interval;
  const double range = 0.5 * scene.propagation_speed * delay;
  scene.targets.push_back(
      {sim::Vec3{range, 0.0, 0.0}, sim::Vec3{}, {1.0, 0.0}});
  return scene;
}

TEST(PointTargetSim, GroundTruthDerivation) {
  const auto scene = single_target_scene();
  const auto result = sim::simulate(scene);
  ASSERT_EQ(result.truth.size(), 1u);
  EXPECT_DOUBLE_EQ(result.truth[0].range, 40.0);
  EXPECT_DOUBLE_EQ(result.truth[0].radial_velocity, 5.0);  // receding
  EXPECT_DOUBLE_EQ(result.truth[0].look_direction[1], 1.0);
  ASSERT_EQ(result.shape.size(), 2u);
  EXPECT_EQ(result.shape[0], scene.num_chirps);
  EXPECT_EQ(result.shape[1], scene.samples_per_chirp);
}

TEST(PointTargetSim, Deterministic) {
  auto scene = single_target_scene();
  scene.snr_db = 20.0;
  // simulate() is a pure function of the
  // scene; the noise additive is seeded, so
  // the composed (solver + add_thermal_noise) output is bit-identical for a
  // fixed seed and differs when the seed changes.
  auto a = sim::simulate(scene);
  sim::add_thermal_noise(a.cube, scene.snr_db, scene.seed);
  auto b = sim::simulate(scene);
  sim::add_thermal_noise(b.cube, scene.snr_db, scene.seed);
  EXPECT_EQ(a.cube, b.cube);
  auto c = sim::simulate(scene);
  sim::add_thermal_noise(c.cube, scene.snr_db, scene.seed + 1);
  EXPECT_NE(a.cube, c.cube);
}

TEST(PointTargetSim, BeatFrequencyMatchesAnalyticModel) {
  // Host DFT of one chirp peaks at the bin the forward bin model predicts.
  const auto scene = single_target_scene();
  const auto result = sim::simulate(scene);
  const int64_t n = scene.samples_per_chirp;

  double best_mag = 0.0;
  int64_t best_bin = -1;
  for (int64_t bin = 0; bin < n; ++bin) {
    std::complex<double> acc{0.0, 0.0};
    for (int64_t s = 0; s < n; ++s) {
      const double phase = -2.0 * kPi * static_cast<double>(bin) *
                           static_cast<double>(s) / static_cast<double>(n);
      acc += result.cube[static_cast<size_t>(s)] *
             std::complex<double>(std::cos(phase), std::sin(phase));
    }
    if (std::abs(acc) > best_mag) {
      best_mag = std::abs(acc);
      best_bin = bin;
    }
  }

  const int64_t expected = fix::expected_range_bin(
      result.truth[0].range, scene.waveform.slope(),
      scene.waveform.sample_rate, n, scene.propagation_speed);
  EXPECT_NEAR(static_cast<double>(best_bin), static_cast<double>(expected),
              1.0);
}

TEST(PointTargetSim, DopplerPhaseSlopeMatchesRadialVelocity) {
  // The chirp-to-chirp phase advance at a fixed fast-time sample equals
  // 2*pi*f_d*PRI with f_d = -2*v_r/lambda (plus the small range-walk beat
  // term, negligible over one PRI at these parameters).
  const auto scene = single_target_scene();
  const auto result = sim::simulate(scene);
  const int64_t n = scene.samples_per_chirp;

  const std::complex<double> c0 = result.cube[0];
  const std::complex<double> c1 = result.cube[static_cast<size_t>(n)];
  const double measured = std::arg(c1 * std::conj(c0));

  const double lambda = scene.waveform.wavelength();
  const double f_d =
      fix::doppler_frequency(result.truth[0].radial_velocity, lambda);
  double expected =
      2.0 * kPi * f_d * scene.waveform.chirp_repetition_interval;
  // Wrap to (-pi, pi] for comparison.
  while (expected <= -kPi) {
    expected += 2.0 * kPi;
  }
  while (expected > kPi) {
    expected -= 2.0 * kPi;
  }
  EXPECT_NEAR(measured, expected, 1e-2);
}

TEST(PointTargetSim, PreMixerTimesConjReferenceIsDechirped) {
  auto scene = single_target_scene();
  const auto dechirped = sim::simulate(scene);
  scene.dechirped = false;
  const auto pre_mixer = sim::simulate(scene);
  const auto ref =
      sim::reference_chirp(scene.waveform, scene.samples_per_chirp);

  for (int64_t k = 0; k < scene.num_chirps; ++k) {
    for (int64_t s = 0; s < scene.samples_per_chirp; ++s) {
      const size_t idx =
          static_cast<size_t>(k * scene.samples_per_chirp + s);
      const std::complex<double> recovered =
          pre_mixer.cube[idx] * std::conj(ref[static_cast<size_t>(s)]);
      EXPECT_NEAR(recovered.real(), dechirped.cube[idx].real(), 1e-9);
      EXPECT_NEAR(recovered.imag(), dechirped.cube[idx].imag(), 1e-9);
    }
  }
}

TEST(PointTargetSim, SimoCubeHasRxLeadingAxis) {
  auto scene = single_target_scene();
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.002, 0.0, 0.0}};
  const auto result = sim::simulate(scene);
  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 2);
  // Broadside target on the y axis: zero aperture projection, so both RX
  // see the identical signal.
  const size_t per_rx =
      static_cast<size_t>(scene.num_chirps * scene.samples_per_chirp);
  for (size_t i = 0; i < per_rx; ++i) {
    EXPECT_NEAR(result.cube[i].real(), result.cube[per_rx + i].real(),
                1e-12);
    EXPECT_NEAR(result.cube[i].imag(), result.cube[per_rx + i].imag(),
                1e-12);
  }
}

// --- SceneMover seam (pluggable per-frame target evolution) ---

TEST(SceneMover, LinearMoverMatchesAdvanceScene) {
  auto a = single_target_scene();
  auto b = single_target_scene();
  const sim::SceneMover mover = sim::linear_mover();
  for (int i = 0; i < 5; ++i) {
    mover(a, 0.1);
    sim::advance_scene(&b, 0.1);
  }
  EXPECT_EQ(a.targets[0].position, b.targets[0].position);
}

TEST(SceneMover, CustomMoverIsHonored) {
  // A custom mover can move targets non-linearly or regenerate them entirely;
  // here it pins every target to a fixed range each frame.
  auto scene = single_target_scene();
  const sim::SceneMover pin = [](sim::SimScene& s, double) {
    for (auto& t : s.targets) {
      t.position = sim::Vec3{0.0, 25.0, 0.0};
    }
  };
  pin(scene, 0.1);
  EXPECT_DOUBLE_EQ(scene.targets[0].position[1], 25.0);
}

// --- MIMO simulators (mimo_reference.hpp) ---

// Two-TX, two-RX scene sharing the single-target waveform/target.
sim::SimScene mimo_scene() {
  auto scene = single_target_scene();
  const double lambda = scene.waveform.wavelength();
  scene.tx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.0, 2.0 * lambda, 0.0}};
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.0, 0.5 * lambda, 0.0}};
  return scene;
}

// Element-wise complex compare. The MIMO simulators and simulate() share the
// same arithmetic in the single-TX case, but they are separate functions, so
// compiler FP contraction/reordering can differ by a few ULP — compare with a
// tight tolerance rather than bit-exact.
void expect_cubes_near(const std::vector<std::complex<double>>& a,
                       const std::vector<std::complex<double>>& b, double tol) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), tol);
    EXPECT_NEAR(a[i].imag(), b[i].imag(), tol);
  }
}

TEST(MimoSim, TdmCubeShape) {
  const auto scene = mimo_scene();
  rd::TXSchedule schedule;  // round-robin; num_tx is set from the scene
  const auto result = sim::simulate_mimo_tdm_interleaved(scene, schedule);
  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 2);                      // num_rx
  EXPECT_EQ(result.shape[1], 2 * scene.num_chirps);   // num_tx * num_chirps
  EXPECT_EQ(result.shape[2], scene.samples_per_chirp);
}

TEST(MimoSim, TdmSingleTxMatchesSimulate) {
  // num_tx == 1 with the identity schedule reduces to the single-TX model, so
  // the interleaved cube equals simulate() (ties the generalization to base).
  const auto scene = single_target_scene();  // default: 1 TX, 1 RX
  const auto mimo = sim::simulate_mimo_tdm_interleaved(scene, rd::TXSchedule{});
  const auto base = sim::simulate(scene);
  expect_cubes_near(mimo.cube, base.cube, 1e-12);
}

TEST(MimoSim, DdmCubeShape) {
  const auto scene = mimo_scene();
  const auto result =
      sim::simulate_mimo_ddm_coded(scene, sim::uniform_ddm_offsets(2));
  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 2);
  EXPECT_EQ(result.shape[1], scene.num_chirps);
  EXPECT_EQ(result.shape[2], scene.samples_per_chirp);
}

TEST(MimoSim, DdmSingleTxNoCodeMatchesSimulate) {
  const auto scene = single_target_scene();  // 1 TX, 1 RX
  const auto mimo = sim::simulate_mimo_ddm_coded(scene, {0.0});
  const auto base = sim::simulate(scene);
  expect_cubes_near(mimo.cube, base.cube, 1e-12);
}

TEST(MimoSim, DdmCodePhaseNullsOddChirps) {
  // Two co-located TX with offsets {0, 1/2}: TX1 carries code phase pi*k, so
  // each chirp sums (1 + (-1)^k) — even chirps add coherently (2x a single TX),
  // odd chirps cancel. This is the defining property of the DDM Doppler code.
  auto scene = single_target_scene();
  scene.tx_positions = {sim::Vec3{0.0, 0.0, 0.0}, sim::Vec3{0.0, 0.0, 0.0}};
  const auto ddm = sim::simulate_mimo_ddm_coded(scene, {0.0, 0.5});
  const auto base = sim::simulate(single_target_scene());  // one TX at origin
  const int64_t samples = scene.samples_per_chirp;
  for (int64_t k = 0; k < scene.num_chirps; ++k) {
    for (int64_t s = 0; s < samples; ++s) {
      const size_t idx = static_cast<size_t>(k * samples + s);
      if (k % 2 == 1) {
        EXPECT_NEAR(std::abs(ddm.cube[idx]), 0.0, 1e-9);
      } else {
        EXPECT_NEAR(std::abs(ddm.cube[idx]),
                    2.0 * std::abs(base.cube[idx]), 1e-9);
      }
    }
  }
}

TEST(MimoSim, TdmDeterministic) {
  auto scene = mimo_scene();
  scene.snr_db = 20.0;
  const rd::TXSchedule sched;
  // Noise is a seeded additive applied on top of the clean solver response.
  auto a = sim::simulate_mimo_tdm_interleaved(scene, sched);
  sim::add_thermal_noise(a.cube, scene.snr_db, scene.seed);
  auto b = sim::simulate_mimo_tdm_interleaved(scene, sched);
  sim::add_thermal_noise(b.cube, scene.snr_db, scene.seed);
  EXPECT_EQ(a.cube, b.cube);
  auto c = sim::simulate_mimo_tdm_interleaved(scene, sched);
  sim::add_thermal_noise(c.cube, scene.snr_db, scene.seed + 1);
  EXPECT_NE(a.cube, c.cube);
}

TEST(MimoSim, ResolveDdmOffsets) {
  rd::DDMCodeDescriptor uniform;
  uniform.num_tx = 2;  // no doppler_offsets -> uniform code
  const auto u = sim::resolve_ddm_offsets(uniform);
  ASSERT_EQ(u.size(), 2u);
  EXPECT_DOUBLE_EQ(u[0], 0.0);
  EXPECT_DOUBLE_EQ(u[1], 0.5);

  rd::DDMCodeDescriptor explicit_code;
  explicit_code.num_tx = 3;
  explicit_code.doppler_offsets = {0.1, 0.4, 0.7};
  EXPECT_EQ(sim::resolve_ddm_offsets(explicit_code),
            (std::vector<double>{0.1, 0.4, 0.7}));
}

TEST(PointTargetSim, AnalogIfRaisedCosineHasExpectedGain) {
  const double fs = single_target_scene().waveform.sample_rate;
  const auto magnitude = [](sim::SimScene scene) {
    return std::abs(sim::simulate(scene).cube.front());
  };

  EXPECT_NEAR(magnitude(scene_for_beat(0.5 * fs)), 1.0, 1e-12);
  EXPECT_NEAR(magnitude(scene_for_beat(0.925 * fs)),
              0.5 * (1.0 + std::cos(0.25 * kPi)), 1e-12);
  EXPECT_NEAR(magnitude(scene_for_beat(0.95 * fs)), 0.5, 1e-12);
  EXPECT_NEAR(magnitude(scene_for_beat(0.975 * fs)),
              0.5 * (1.0 + std::cos(0.75 * kPi)), 1e-12);
  EXPECT_NEAR(magnitude(scene_for_beat(fs)), 0.0, 1e-12);
}

TEST(PointTargetSim, ExplicitUnfilteredModeRetainsAdcAliasingTone) {
  const double fs = single_target_scene().waveform.sample_rate;
  auto filtered = scene_for_beat(1.05 * fs);
  auto unfiltered = filtered;
  unfiltered.analog_if.response = sim::AnalogIfResponseType::kNone;

  EXPECT_NEAR(std::abs(sim::simulate(filtered).cube.front()), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(sim::simulate(unfiltered).cube.front()), 1.0, 1e-12);
  const auto result = sim::simulate(unfiltered);
  const std::complex<double> step = result.cube[1] * std::conj(result.cube[0]);
  EXPECT_NEAR(std::arg(step), 2.0 * kPi * 0.05, 1e-9);
}

TEST(PointTargetSim, CriDelayedTargetReappearsAtResidualRange) {
  const double beat = 0.2 * single_target_scene().waveform.sample_rate;
  const auto principal = sim::simulate(scene_for_beat(beat));
  const auto ambiguous = sim::simulate(scene_for_beat(beat, 1));

  ASSERT_EQ(principal.cube.size(), ambiguous.cube.size());
  for (size_t i = 0; i < principal.cube.size(); ++i) {
    EXPECT_NEAR(std::abs(principal.cube[i]), std::abs(ambiguous.cube[i]),
                1e-12);
  }
}

TEST(PointTargetSim, ExactCriDelayLandsAtZeroBeat) {
  const auto result = sim::simulate(scene_for_beat(0.0, 1));
  for (int64_t s = 1; s < single_target_scene().samples_per_chirp; ++s) {
    EXPECT_NEAR(std::abs(result.cube[static_cast<size_t>(s)] - result.cube[0]),
                0.0, 1e-9);
  }
}

TEST(PointTargetSim, RealIfUsesHalfRateBoundaryAndConversionIsExplicit) {
  const double fs = single_target_scene().waveform.sample_rate;
  auto complex_scene = scene_for_beat(0.75 * fs);
  auto real_scene = complex_scene;
  real_scene.analog_if.sampling_mode = sim::IfSamplingMode::kReal;

  const auto complex_result = sim::simulate(complex_scene);
  const auto real_result = sim::simulate(real_scene);
  EXPECT_NEAR(std::abs(complex_result.cube.front()), 1.0, 1e-12);
  EXPECT_NEAR(std::abs(real_result.cube.front()), 0.0, 1e-12);
  EXPECT_THROW(sim::to_real_if(complex_result), std::invalid_argument);
  EXPECT_EQ(sim::to_real_if(real_result).size(), real_result.cube.size());
}

TEST(PointTargetSim, ValidatesIfFractionsAndWaveformTiming) {
  auto scene = single_target_scene();
  scene.analog_if.passband_fraction = 1.0;
  EXPECT_THROW(sim::simulate(scene), std::invalid_argument);
  scene = single_target_scene();
  scene.analog_if.stopband_fraction = 1.1;
  EXPECT_THROW(sim::simulate(scene), std::invalid_argument);
  scene = single_target_scene();
  scene.samples_per_chirp += 1;
  EXPECT_THROW(sim::simulate(scene), std::invalid_argument);
  scene = single_target_scene();
  scene.waveform.chirp_repetition_interval = 19e-6;
  EXPECT_THROW(sim::simulate(scene), std::invalid_argument);
}

TEST(PointTargetSim, MovingTargetWrapsAcrossCriBoundary) {
  auto scene = single_target_scene();
  scene.num_chirps = 2;
  scene.analog_if.response = sim::AnalogIfResponseType::kNone;
  const double cri = scene.waveform.chirp_repetition_interval;
  const double initial_delay = cri - 0.1e-6;
  const double delay_step = 0.2e-6;
  const double range = 0.5 * scene.propagation_speed * initial_delay;
  const double velocity =
      0.5 * scene.propagation_speed * delay_step / cri;
  scene.targets = {{sim::Vec3{range, 0.0, 0.0},
                    sim::Vec3{velocity, 0.0, 0.0}, {1.0, 0.0}}};

  const auto result = sim::simulate(scene);
  const size_t n = static_cast<size_t>(scene.samples_per_chirp);
  const std::complex<double> wrapped_step =
      result.cube[n + 1] * std::conj(result.cube[n]);
  const double expected =
      2.0 * kPi * scene.waveform.slope() * 0.1e-6 / scene.waveform.sample_rate;
  EXPECT_NEAR(std::arg(wrapped_step), std::remainder(expected, 2.0 * kPi), 1e-8);
}

TEST(MimoSim, TdmCriAmbiguitySelectsDelayedTxSlot) {
  auto scene = mimo_scene();
  scene.num_chirps = 2;
  const double ambiguity_range =
      0.5 * scene.propagation_speed *
      scene.waveform.chirp_repetition_interval;
  scene.targets = {{sim::Vec3{0.0, ambiguity_range + 40.0, 0.0},
                    sim::Vec3{}, {1.0, 0.0}}};
  const auto tdm =
      sim::simulate_mimo_tdm_interleaved(scene, rd::TXSchedule{});

  for (int64_t receive_slot = 0; receive_slot < 2; ++receive_slot) {
    auto single = scene;
    single.num_chirps = 1;
    const int64_t delayed_tx = receive_slot == 0 ? 1 : 0;
    single.tx_positions = {
        scene.tx_positions[static_cast<size_t>(delayed_tx)]};
    const auto expected = sim::simulate(single);
    for (int64_t rx = 0; rx < 2; ++rx) {
      const size_t actual_idx = static_cast<size_t>(
          (rx * 2 * scene.num_chirps + receive_slot) *
          scene.samples_per_chirp);
      const size_t expected_idx =
          static_cast<size_t>(rx * scene.samples_per_chirp);
      EXPECT_NEAR(std::abs(tdm.cube[actual_idx] - expected.cube[expected_idx]),
                  0.0, 1e-9);
    }
  }
}

TEST(MimoSim, DdmCriAmbiguitySelectsDelayedOffsetAndPhaseCode) {
  auto scene = single_target_scene();
  scene.num_chirps = 4;
  scene.tx_positions = {sim::Vec3{}, sim::Vec3{}};
  const double ambiguity_range =
      0.5 * scene.propagation_speed *
      scene.waveform.chirp_repetition_interval;
  scene.targets = {{sim::Vec3{ambiguity_range + 40.0, 0.0, 0.0},
                    sim::Vec3{}, {1.0, 0.0}}};
  auto single = scene;
  single.tx_positions = {sim::Vec3{}};
  const auto base = sim::simulate(single);

  const auto offset = sim::simulate_mimo_ddm_coded(scene, {0.0, 0.25});
  const std::complex<double> expected_sum{1.0, -1.0};
  EXPECT_NEAR(std::abs(offset.cube[0] / base.cube[0] - expected_sum), 0.0,
              1e-9);

  const std::vector<std::complex<double>> codes = {
      {1.0, 0.0}, {1.0, 0.0},  {1.0, 0.0}, {1.0, 0.0},
      {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
  const auto phase = sim::simulate_mimo_ddm_phase_coded(scene, codes);
  EXPECT_NEAR(std::abs(phase.cube[0] / base.cube[0] - expected_sum), 0.0,
              1e-9);
}

}  // namespace
