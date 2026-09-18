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

// --- Application scene visibility seam ---

TEST(SceneVisibility, FrontHalfSpaceFiltersWithoutMutatingScene) {
  sim::SimScene scene;
  scene.targets = {
      {sim::Vec3{10.0, 0.0, 0.0}, sim::Vec3{}, {1.0, 0.0}},
      {sim::Vec3{-10.0, 0.0, 0.0}, sim::Vec3{}, {2.0, 0.0}},
      {sim::Vec3{0.0, 10.0, 0.0}, sim::Vec3{}, {3.0, 0.0}},
  };
  scene.analog_if.response = sim::AnalogIfResponseType::kNone;
  scene.range_loss.model = sim::RangeLossModel::kMonostaticFreeSpace;
  scene.range_loss.reference_range = 17.0;
  scene.snr_db = 23.0;
  scene.seed = 1234U;

  const sim::SimScene visible = sim::visible_scene(scene, {});
  ASSERT_EQ(visible.targets.size(), 1U);
  EXPECT_EQ(visible.targets[0].position, (sim::Vec3{10.0, 0.0, 0.0}));
  EXPECT_EQ(scene.targets.size(), 3U);
  EXPECT_EQ(visible.analog_if.response, sim::AnalogIfResponseType::kNone);
  EXPECT_EQ(visible.range_loss.model,
            sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(visible.range_loss.reference_range, 17.0);
  EXPECT_EQ(visible.snr_db, scene.snr_db);
  EXPECT_EQ(visible.seed, scene.seed);
}

TEST(SceneVisibility, AllAndRotatedBoresightAreSupported) {
  sim::SimScene scene;
  scene.targets = {
      {sim::Vec3{-10.0, 0.0, 0.0}, sim::Vec3{}, {1.0, 0.0}},
      {sim::Vec3{0.0, 10.0, 0.0}, sim::Vec3{}, {1.0, 0.0}},
  };
  sim::SceneVisibility all;
  all.model = sim::SceneVisibilityModel::kAll;
  EXPECT_EQ(sim::visible_scene(scene, all).targets.size(), 2U);

  sim::SceneVisibility rotated;
  rotated.boresight_direction = sim::Vec3{0.0, 2.0, 0.0};
  const sim::SimScene visible = sim::visible_scene(scene, rotated);
  ASSERT_EQ(visible.targets.size(), 1U);
  EXPECT_EQ(visible.targets[0].position, (sim::Vec3{0.0, 10.0, 0.0}));

  rotated.boresight_direction = sim::Vec3{0.0, 0.0, 1.0};
  EXPECT_TRUE(sim::target_is_visible({0.0, 0.0, 10.0}, rotated));
}

TEST(SceneVisibility, RectangularFrustumUsesHorizontalAndVerticalFov) {
  sim::SceneVisibility frustum;
  frustum.model = sim::SceneVisibilityModel::kFrustum;
  frustum.horizontal_fov = kPi / 2.0;
  frustum.vertical_fov = kPi / 2.0;

  EXPECT_TRUE(sim::target_is_visible({10.0, 9.0, 9.0}, frustum));
  EXPECT_TRUE(sim::target_is_visible({10.0, 10.0, 10.0}, frustum));
  EXPECT_FALSE(sim::target_is_visible({10.0, 11.0, 0.0}, frustum));
  EXPECT_FALSE(sim::target_is_visible({10.0, 0.0, 11.0}, frustum));
  EXPECT_FALSE(sim::target_is_visible({-10.0, 0.0, 0.0}, frustum));

  frustum.vertical_fov.reset();
  EXPECT_THROW(sim::validate_scene_visibility(frustum),
               std::invalid_argument);
  frustum.vertical_fov = kPi / 2.0;
  frustum.up_direction = frustum.boresight_direction;
  EXPECT_THROW(sim::validate_scene_visibility(frustum),
               std::invalid_argument);
}

TEST(SceneVisibility, AuthoritativeTargetCanExitAndReenter) {
  sim::SimScene scene;
  scene.targets = {{sim::Vec3{1.0, 0.0, 0.0},
                    sim::Vec3{-2.0, 0.0, 0.0}, {1.0, 0.0}}};
  const sim::SceneVisibility visibility;
  EXPECT_EQ(sim::visible_scene(scene, visibility).targets.size(), 1U);
  sim::advance_scene(&scene, 1.0);
  EXPECT_TRUE(sim::visible_scene(scene, visibility).targets.empty());
  scene.targets[0].velocity[0] = 2.0;
  sim::advance_scene(&scene, 1.0);
  EXPECT_EQ(sim::visible_scene(scene, visibility).targets.size(), 1U);
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

void expect_pre_mixer_recovers_dechirped(const sim::SimResult& dechirped,
                                          const sim::SimResult& pre_mixer,
                                          const sim::SimWaveform& waveform,
                                          int64_t samples) {
  ASSERT_EQ(pre_mixer.cube.size(), dechirped.cube.size());
  const auto reference = sim::reference_chirp(waveform, samples);
  for (size_t i = 0; i < pre_mixer.cube.size(); ++i) {
    const size_t sample = i % static_cast<size_t>(samples);
    const auto recovered = pre_mixer.cube[i] * std::conj(reference[sample]);
    EXPECT_NEAR(recovered.real(), dechirped.cube[i].real(), 1e-9);
    EXPECT_NEAR(recovered.imag(), dechirped.cube[i].imag(), 1e-9);
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

TEST(MimoSim, PreMixerRecoversAllDechirpedMimoReferences) {
  auto scene = mimo_scene();
  scene.num_chirps = 4;
  scene.samples_per_chirp = 32;
  const auto tdm =
      sim::simulate_mimo_tdm_interleaved(scene, rd::TXSchedule{});
  const auto offsets = sim::uniform_ddm_offsets(2);
  const auto ddm = sim::simulate_mimo_ddm_coded(scene, offsets);
  std::vector<std::complex<double>> codes(8);
  for (int64_t tx = 0; tx < 2; ++tx) {
    for (int64_t k = 0; k < scene.num_chirps; ++k) {
      const double phase = 0.3 * static_cast<double>(tx) +
                           0.2 * static_cast<double>(k);
      codes[static_cast<size_t>(tx * scene.num_chirps + k)] =
          {std::cos(phase), std::sin(phase)};
    }
  }
  const auto phase_coded =
      sim::simulate_mimo_ddm_phase_coded(scene, codes);

  scene.dechirped = false;
  const auto tdm_pre =
      sim::simulate_mimo_tdm_interleaved(scene, rd::TXSchedule{});
  const auto ddm_pre = sim::simulate_mimo_ddm_coded(scene, offsets);
  const auto phase_pre = sim::simulate_mimo_ddm_phase_coded(scene, codes);
  expect_pre_mixer_recovers_dechirped(
      tdm, tdm_pre, scene.waveform, scene.samples_per_chirp);
  expect_pre_mixer_recovers_dechirped(
      ddm, ddm_pre, scene.waveform, scene.samples_per_chirp);
  expect_pre_mixer_recovers_dechirped(
      phase_coded, phase_pre, scene.waveform, scene.samples_per_chirp);
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

TEST(PointTargetSim, RangeLossUsesInverseSquareVoltageGain) {
  auto scene = single_target_scene();
  scene.analog_if.response = sim::AnalogIfResponseType::kNone;
  scene.targets[0].velocity = sim::Vec3{};
  const auto none = sim::simulate(scene);

  scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
  const auto at_reference = sim::simulate(scene);
  expect_cubes_near(none.cube, at_reference.cube, 1e-12);

  scene.targets[0].position = sim::Vec3{0.0, 80.0, 0.0};
  scene.targets[0].amplitude = {0.0, 1.0};
  const auto doubled = sim::simulate(scene);
  scene.range_loss.model = sim::RangeLossModel::kNone;
  const auto doubled_none = sim::simulate(scene);
  const auto ratio = doubled.cube.front() / doubled_none.cube.front();
  EXPECT_NEAR(std::abs(ratio), 0.25, 1e-12);
  EXPECT_NEAR(std::arg(ratio), 0.0, 1e-12);
}

TEST(PointTargetSim, RangeLossTracksMovingCenterRange) {
  auto scene = single_target_scene();
  scene.num_chirps = 2;
  scene.samples_per_chirp = 1;
  scene.analog_if.response = sim::AnalogIfResponseType::kNone;
  scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
  scene.targets = {{sim::Vec3{40.0, 0.0, 0.0},
                    sim::Vec3{40.0 / scene.waveform.chirp_repetition_interval,
                              0.0, 0.0},
                    {1.0, 0.0}}};

  const auto result = sim::simulate(scene);
  EXPECT_NEAR(std::abs(result.cube[1]) / std::abs(result.cube[0]), 0.25,
              1e-12);
}

TEST(PointTargetSim, CriFoldingRetainsFullRangeLoss) {
  auto near_scene = single_target_scene();
  near_scene.num_chirps = 1;
  near_scene.analog_if.response = sim::AnalogIfResponseType::kNone;
  near_scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
  near_scene.targets = {
      {sim::Vec3{40.0, 0.0, 0.0}, sim::Vec3{}, {1.0, 0.0}}};

  auto far_scene = near_scene;
  const double ambiguity_range =
      0.5 * far_scene.propagation_speed *
      far_scene.waveform.chirp_repetition_interval;
  far_scene.targets[0].position =
      sim::Vec3{ambiguity_range + 40.0, 0.0, 0.0};

  const auto near_result = sim::simulate(near_scene);
  const auto far_result = sim::simulate(far_scene);
  const double expected =
      std::pow(40.0 / (ambiguity_range + 40.0), 2);
  EXPECT_NEAR(std::abs(far_result.cube.front()) /
                  std::abs(near_result.cube.front()),
              expected, 1e-12);
  EXPECT_NEAR(20.0 * std::log10(expected), -79.0513, 1e-4);
}

TEST(PointTargetSim, RangeLossMatchesPreMixerAndSimoChannels) {
  auto scene = single_target_scene();
  scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 20.0};
  scene.rx_positions = {sim::Vec3{}, sim::Vec3{0.002, 0.0, 0.0}};
  const auto dechirped = sim::simulate(scene);
  scene.dechirped = false;
  const auto pre_mixer = sim::simulate(scene);
  expect_pre_mixer_recovers_dechirped(
      dechirped, pre_mixer, scene.waveform, scene.samples_per_chirp);

  const size_t per_rx =
      static_cast<size_t>(scene.num_chirps * scene.samples_per_chirp);
  EXPECT_NEAR(std::abs(dechirped.cube[0]),
              std::abs(dechirped.cube[per_rx]), 1e-12);
}

TEST(PointTargetSim, RangeLossChangesSignalButNotNoisePower) {
  auto near_scene = single_target_scene();
  near_scene.targets[0].velocity = sim::Vec3{};
  near_scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
  auto far_scene = near_scene;
  far_scene.targets[0].position = sim::Vec3{0.0, 80.0, 0.0};

  const auto near_result = sim::simulate(near_scene);
  const auto far_result = sim::simulate(far_scene);
  EXPECT_NEAR(std::norm(near_result.cube.front()) /
                  std::norm(far_result.cube.front()),
              16.0, 1e-10);

  std::vector<std::complex<double>> near_noise(near_result.cube.size());
  std::vector<std::complex<double>> far_noise(far_result.cube.size());
  sim::add_thermal_noise(near_noise, 20.0, near_scene.seed);
  sim::add_thermal_noise(far_noise, 20.0, far_scene.seed);
  EXPECT_EQ(near_noise, far_noise);
}

TEST(PointTargetSim, RangeLossRejectsInvalidRanges) {
  auto scene = single_target_scene();
  scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 0.0};
  EXPECT_THROW(sim::simulate(scene), std::invalid_argument);

  scene.range_loss.reference_range = 40.0;
  scene.num_chirps = 2;
  scene.targets[0].velocity =
      sim::Vec3{0.0, -80.0 / scene.waveform.chirp_repetition_interval, 0.0};
  EXPECT_THROW(sim::simulate(scene), std::invalid_argument);
}

TEST(MimoSim, RangeLossScalesTdmAndBothDdmReferences) {
  auto scene = mimo_scene();
  scene.num_chirps = 4;
  scene.samples_per_chirp = 16;
  scene.targets[0].velocity = sim::Vec3{};
  scene.analog_if.response = sim::AnalogIfResponseType::kNone;
  const auto tdm_none =
      sim::simulate_mimo_tdm_interleaved(scene, rd::TXSchedule{});
  const auto offsets = sim::uniform_ddm_offsets(2);
  const auto ddm_none = sim::simulate_mimo_ddm_coded(scene, offsets);
  std::vector<std::complex<double>> codes(8, {1.0, 0.0});
  const auto phase_none =
      sim::simulate_mimo_ddm_phase_coded(scene, codes);

  scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 20.0};
  const auto tdm_loss =
      sim::simulate_mimo_tdm_interleaved(scene, rd::TXSchedule{});
  const auto ddm_loss = sim::simulate_mimo_ddm_coded(scene, offsets);
  const auto phase_loss =
      sim::simulate_mimo_ddm_phase_coded(scene, codes);

  const auto expect_quarter = [](const sim::SimResult& loss,
                                 const sim::SimResult& none) {
    ASSERT_EQ(loss.cube.size(), none.cube.size());
    for (size_t i = 0; i < loss.cube.size(); ++i) {
      if (std::abs(none.cube[i]) > 1e-10) {
        EXPECT_NEAR(std::abs(loss.cube[i] / none.cube[i]), 0.25, 1e-10);
      }
    }
  };
  expect_quarter(tdm_loss, tdm_none);
  expect_quarter(ddm_loss, ddm_none);
  expect_quarter(phase_loss, phase_none);
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

TEST(PointTargetSim, FullCriDelayRemainsInCarrierPhase) {
  constexpr int64_t ambiguity_order = 3;
  const double beat = 0.2 * single_target_scene().waveform.sample_rate;
  auto principal_scene = scene_for_beat(beat);
  principal_scene.waveform.carrier_frequency += 1234.0;
  auto ambiguous_scene = scene_for_beat(beat, ambiguity_order);
  ambiguous_scene.waveform.carrier_frequency =
      principal_scene.waveform.carrier_frequency;

  const auto principal = sim::simulate(principal_scene);
  const auto ambiguous = sim::simulate(ambiguous_scene);
  const std::complex<double> ratio =
      ambiguous.cube.front() / principal.cube.front();
  const double phase =
      -2.0 * kPi * principal_scene.waveform.carrier_frequency *
      static_cast<double>(ambiguity_order) *
      principal_scene.waveform.chirp_repetition_interval;
  const std::complex<double> expected{std::cos(phase), std::sin(phase)};
  EXPECT_NEAR(std::abs(ratio - expected), 0.0, 1e-6);
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

TEST(MimoSim, TdmMultiPeriodCriAmbiguitySelectsDelayedTxSlot) {
  auto scene = mimo_scene();
  scene.num_chirps = 2;
  const int64_t ambiguity_order = 5;
  const double ambiguity_range =
      0.5 * scene.propagation_speed *
      scene.waveform.chirp_repetition_interval *
      static_cast<double>(ambiguity_order);
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

TEST(MimoSim, DdmMultiPeriodCriAmbiguitySelectsDelayedCode) {
  auto scene = single_target_scene();
  scene.num_chirps = 4;
  scene.tx_positions = {sim::Vec3{}, sim::Vec3{}};
  const int64_t ambiguity_order = 5;
  const double ambiguity_range =
      0.5 * scene.propagation_speed *
      scene.waveform.chirp_repetition_interval *
      static_cast<double>(ambiguity_order);
  scene.targets = {{sim::Vec3{ambiguity_range + 40.0, 0.0, 0.0},
                    sim::Vec3{}, {1.0, 0.0}}};
  auto single = scene;
  single.tx_positions = {sim::Vec3{}};
  const auto base = sim::simulate(single);

  const auto offset = sim::simulate_mimo_ddm_coded(scene, {0.0, 0.25});
  const std::complex<double> expected_sum{1.0, -1.0};
  EXPECT_NEAR(std::abs(offset.cube[0] / base.cube[0] - expected_sum), 0.0,
               1e-8);

  const std::vector<std::complex<double>> codes = {
      {1.0, 0.0}, {1.0, 0.0},  {1.0, 0.0}, {1.0, 0.0},
      {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
  const auto phase = sim::simulate_mimo_ddm_phase_coded(scene, codes);
  EXPECT_NEAR(std::abs(phase.cube[0] / base.cube[0] - expected_sum), 0.0,
               1e-8);
}

}  // namespace
