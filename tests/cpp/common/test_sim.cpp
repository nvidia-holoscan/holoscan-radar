// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sim/mimo_sim.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "radar_common/sim/sar_phase_history_sim.hpp"
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
  const auto a = sim::simulate(scene);
  const auto b = sim::simulate(scene);
  EXPECT_EQ(a.cube, b.cube);
  scene.seed += 1;
  const auto c = sim::simulate(scene);
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

// --- MIMO simulators (mimo_sim.hpp) ---

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
  const auto a = sim::simulate_mimo_tdm_interleaved(scene, sched);
  const auto b = sim::simulate_mimo_tdm_interleaved(scene, sched);
  EXPECT_EQ(a.cube, b.cube);
  scene.seed += 1;
  const auto c = sim::simulate_mimo_tdm_interleaved(scene, sched);
  EXPECT_NE(a.cube, c.cube);
}

// The SAR phase-history model's own tautology guard: the phase sign and the
// frequency axis are two statements of one convention, and nothing else in the
// chain notices when they disagree. Compressing one pulse on the host and
// asking where the target landed costs microseconds and catches a flipped
// sign, a missing factor of two, and a reversed frequency axis — the same
// check tools/sar_sim/sar_sim/signal.py runs on the Python side.
TEST(SarPhaseHistorySim, CompressesToThePredictedDelayBin) {
  sim::SarPhaseHistoryConfig cfg;
  cfg.num_pulses = 1;
  cfg.num_samples = 64;

  const double spacing = sim::sar_range_bin_spacing(cfg, cfg.num_samples);
  sim::SarPointTarget target;
  target.differential_range = 9.0 * spacing;
  const auto signal = sim::sar_phase_history(cfg, {target});
  ASSERT_EQ(signal.size(), static_cast<size_t>(cfg.num_samples));

  // A brute-force inverse DFT, exp(+2*pi*i*k*n/N): the positive-exponent
  // kernel the model's phase sign pairs with. Its conjugate would put the
  // target at the negative of its true delay, which is the mirrored image
  // this check exists to catch.
  const int64_t n = cfg.num_samples;
  int64_t peak = 0;
  double peak_magnitude = -1.0;
  for (int64_t bin = 0; bin < n; ++bin) {
    std::complex<double> sum;
    for (int64_t k = 0; k < n; ++k) {
      const double phase = 2.0 * kPi * static_cast<double>(k * bin) /
                           static_cast<double>(n);
      sum += signal[static_cast<size_t>(k)] *
             std::complex<double>(std::cos(phase), std::sin(phase));
    }
    if (std::abs(sum) > peak_magnitude) {
      peak_magnitude = std::abs(sum);
      peak = bin;
    }
  }
  // Unshifted, so a farther target sits at a positive bin index near zero
  // rather than wrapping to the top of the array.
  EXPECT_EQ(peak, 9);
  EXPECT_NEAR(peak_magnitude, static_cast<double>(n), 1e-6 * n);

  // A nearer target is the mirror case, and wraps.
  sim::SarPointTarget nearer;
  nearer.differential_range = -9.0 * spacing;
  const auto near_signal = sim::sar_phase_history(cfg, {nearer});
  std::complex<double> sum;
  for (int64_t k = 0; k < n; ++k) {
    const double phase = 2.0 * kPi * static_cast<double>(k * (n - 9)) /
                         static_cast<double>(n);
    sum += near_signal[static_cast<size_t>(k)] *
           std::complex<double>(std::cos(phase), std::sin(phase));
  }
  EXPECT_NEAR(std::abs(sum), static_cast<double>(n), 1e-6 * n);
}

TEST(SarPhaseHistorySim, FrequencyAxisSpansTheBandFirstSampleToLast) {
  sim::SarPhaseHistoryConfig cfg;
  cfg.num_samples = 8;
  const auto freqs = sim::sar_frequency_axis(cfg);
  ASSERT_EQ(freqs.size(), 8u);
  // N samples divide the band into N-1 steps, and the band is centred on the
  // carrier: the CPHD FxBand / SCSS reading, not bandwidth/N.
  // Not DOUBLE_EQ: differencing frequencies ~1e10 apart spends the low bits,
  // and the convention under test is a part in N, not a part in 1e16.
  const double tolerance = 1e-6;
  EXPECT_NEAR(freqs.front(), cfg.carrier_frequency - 0.5 * cfg.bandwidth,
              tolerance);
  EXPECT_NEAR(freqs.back(), cfg.carrier_frequency + 0.5 * cfg.bandwidth,
              tolerance);
  EXPECT_NEAR(freqs[1] - freqs[0], cfg.bandwidth / 7.0, tolerance);
}

TEST(SarPhaseHistorySim, PerPulseRangeOverridesTheConstant) {
  sim::SarPhaseHistoryConfig cfg;
  cfg.num_pulses = 3;
  cfg.num_samples = 4;
  sim::SarPointTarget target;
  target.differential_range = 100.0;  // ignored when the per-pulse list is set
  target.differential_range_per_pulse = {0.0, 0.0, 5.0};

  const auto signal = sim::sar_phase_history(cfg, {target});
  // Zero differential range leaves the samples at the reflectivity itself.
  for (int64_t k = 0; k < cfg.num_samples; ++k) {
    EXPECT_NEAR(std::abs(signal[static_cast<size_t>(k)] - 1.0), 0.0, 1e-12);
  }
  EXPECT_GT(std::abs(signal[static_cast<size_t>(2 * cfg.num_samples)] - 1.0),
            1e-6);

  sim::SarPointTarget mismatched;
  mismatched.differential_range_per_pulse = {0.0, 1.0};
  EXPECT_THROW(sim::sar_phase_history(cfg, {mismatched}),
               std::invalid_argument);
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

}  // namespace
