// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/autofocus_op/autofocus_op.hpp"
#include "sar/autofocus_phase_support.hpp"
#include "utils/fixture_completion.hpp"
#include "utils/tensor_fixtures.hpp"

// Operator-scope tests. The estimate's journey onto a differently formed
// product -- transfer through PhaseCorrectionOp, and the whole Figure 2
// retain/reform path -- is an application concern and is tested with the app,
// which owns the resampling and aux-lattice operators those cases need.

namespace {
using holoscan::radar::sar_test::kPi;
using holoscan::radar::sar_test::remove_affine;
using holoscan::radar::sar_test::to_bin_order;

using holoscan::radar::AutofocusAzimuthAxis;
using holoscan::radar::AutofocusOp;
using holoscan::radar::AutofocusParams;
using holoscan::radar::AutofocusTermination;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::autofocus_refused;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

constexpr int64_t kAz = 64;
constexpr int64_t kRg = 32;
constexpr int64_t kStrong = 24;

// A quadratic plus a ramp and an offset, in *pulse* order -- the domain a
// phase error physically lives in and the one `phase_correction` is indexed
// by. This used to be built against the raw bin index instead, which put a
// 7.56 rad jump between adjacent pulses 31 and 32 (0.125 rad median
// elsewhere): a phase error that teleports most of a wavelength between
// consecutive pulses. The estimator reproduced that jump, because it
// integrated the gradient up the bin axis straight through the point where
// pulse adjacency breaks, so the test passed by pinning the bug. See
// autofocus_stage.hpp integrate_detrend_kernel.
std::vector<double> injected_phi_pulse() {
  std::vector<double> phi(static_cast<size_t>(kAz));
  const double half = 0.5 * static_cast<double>(kAz);
  for (int64_t p = 0; p < kAz; ++p) {
    const double x = (static_cast<double>(p) - half) / half;
    phi[static_cast<size_t>(p)] = 2.5 * x * x + 3.8 * x + 0.4;
  }
  return phi;
}

// The same error in bin order, ready to inject on an azimuth spectrum.
std::vector<double> injected_phi() {
  auto phi = injected_phi_pulse();
  phi = to_bin_order(phi);
  return phi;
}

// The same magnitude as an even function and as an odd one about the dwell
// centre. PGA has no reason to prefer either, and until 2026-10-29 it
// recovered only the even part.
std::vector<double> sinusoidal_phi(double amplitude, double cycles, bool odd, int64_t n_az = kAz) {
  std::vector<double> phi(static_cast<size_t>(n_az));
  const double two_pi = 2.0 * kPi;
  for (int64_t k = 0; k < n_az; ++k) {
    const double freq = (k < n_az / 2) ? static_cast<double>(k) : static_cast<double>(k - n_az);
    const double turns = cycles * freq / static_cast<double>(n_az);
    phi[static_cast<size_t>(k)] =
        amplitude * (odd ? std::sin(two_pi * turns) : std::cos(two_pi * turns));
  }
  return phi;
}

double phase_rms(const std::vector<double>& a, const std::vector<double>& b) {
  double acc = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(a.size()));
}

// Thin wrapper over the shared builder so the many call sites below keep the
// file's default extents. The model itself lives in the shared header.
template <typename T>
std::vector<std::complex<T>> make_image(const std::vector<double>& phi, int64_t n_strong,
                                        bool az_is_u, int64_t n_az = kAz, int64_t n_rg = kRg) {
  return holoscan::radar::sar_test::make_image<T>(phi, n_strong, az_is_u, n_az, n_rg);
}

struct AfOut {
  // `ran_ok` is the operator's own verdict and `observed` is whether the
  // captured ports produced anything. `ok` is the conjunction, which is what
  // the positive tests want -- but a negative test must assert on `ran_ok`:
  // asserting `!ok` also passes when the operator ran perfectly and simply
  // emitted nothing, which is the failure mode those tests exist to catch.
  bool ran_ok{false};
  bool observed{false};
  bool ok{false};
  std::string failure;
  HostTensor image;
  HostTensor phase;
  HostTensor sharpness;
  HostTensor termination;
  HostTensor iterations;
};

MaxTensorExtents extents_for(const std::vector<int64_t>& shape) {
  MaxTensorExtents extents;
  extents.max_image_v = std::max<int64_t>(shape.size() > 0 ? shape[0] : 1, 1);
  extents.max_image_u = std::max<int64_t>(shape.size() > 1 ? shape[1] : 1, 1);
  return extents;
}

template <typename T>
AfOut run_af(AutofocusAzimuthAxis axis, const std::vector<std::complex<T>>& img,
             const std::vector<int64_t>& shape, int64_t num_iterations = 4,
             double window_fraction = 0.5, int64_t min_scatterers = 16,
             double max_correction = std::numeric_limits<double>::infinity()) {
  using Op = AutofocusOp<T>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents_for(shape));
  fixture.set_params(AutofocusParams{.azimuth_axis = axis,
                                     .num_iterations = num_iterations,
                                     .window_fraction = window_fraction,
                                     .min_scatterers = min_scatterers,
                                     .max_correction = max_correction});
  auto input = to_device_tensor(img, shape, complex_dtype<T>());
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::nanoseconds{1},
                                                 holoscan::testing::input("image", *input)));
  const int64_t n_az_out = std::max<int64_t>(shape[0], shape.size() > 1 ? shape[1] : 1);
  const std::size_t vector_bytes =
      static_cast<std::size_t>(n_az_out + num_iterations + 2) * sizeof(T);
  fixture.template capture_snapshot<holoscan::Tensor>("image_out", 1U, input->nbytes());
  fixture.template capture_snapshot<holoscan::Tensor>("phase_correction", 1U, vector_bytes);
  fixture.template capture_snapshot<holoscan::Tensor>("sharpness", 1U, vector_bytes);
  fixture.template capture_snapshot<holoscan::Tensor>("termination", 1U, sizeof(std::int32_t));
  fixture.template capture_snapshot<holoscan::Tensor>("iterations", 1U, sizeof(std::int32_t));
  fixture.input_source_device("image", holoscan::DeviceId{0});
  fixture.output_device("image_out", holoscan::DeviceId{0});
  fixture.output_device("phase_correction", holoscan::DeviceId{0});
  fixture.output_device("sharpness", holoscan::DeviceId{0});
  fixture.output_device("termination", holoscan::DeviceId{0});
  fixture.output_device("iterations", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  // Wait on every captured port rather than on whichever is emitted last.
  // Observation order is not emit order: the two int32 status ports are
  // observed *before* the larger vectors despite being emitted after them, so
  // waiting on the final emit stopped the run with sharpness unrecorded.
  bool observed = true;
  for (const char* port : {"phase_correction", "sharpness", "termination", "iterations"}) {
    observed = completion.wait_for_output(running, port, 1U) && observed;
  }
  running.request_stop();
  const auto result = running.finish();

  AfOut out;
  out.failure = result.describe_failure();
  out.ran_ok = result.ok();
  out.observed = observed;
  out.ok = out.ran_ok && observed;
  if (!out.ok) {
    return out;
  }
  completion.check(result);
  out.image = to_host(result.template snapshot<holoscan::Tensor>("image_out"));
  out.phase = to_host(result.template snapshot<holoscan::Tensor>("phase_correction"));
  out.sharpness = to_host(result.template snapshot<holoscan::Tensor>("sharpness"));
  out.termination = to_host(result.template snapshot<holoscan::Tensor>("termination"));
  out.iterations = to_host(result.template snapshot<holoscan::Tensor>("iterations"));
  return out;
}

AutofocusTermination termination_of(const AfOut& out) {
  return static_cast<AutofocusTermination>(out.termination.as<std::int32_t>()[0]);
}

int64_t iterations_of(const AfOut& out) {
  return out.iterations.as<std::int32_t>()[0];
}

std::vector<double> phase_as_double(const HostTensor& t) {
  std::vector<double> out(static_cast<size_t>(t.size()));
  if (t.dtype.bits == 64) {
    const auto* p = t.as<double>();
    for (int64_t i = 0; i < t.size(); ++i) {
      out[static_cast<size_t>(i)] = p[i];
    }
  } else {
    const auto* p = t.as<float>();
    for (int64_t i = 0; i < t.size(); ++i) {
      out[static_cast<size_t>(i)] = static_cast<double>(p[i]);
    }
  }
  return out;
}

int host_strong_count(const std::vector<std::complex<float>>& img, int64_t n_v, int64_t n_u,
                      bool az_is_u) {
  int n_strong = 0;
  const int64_t n_range = az_is_u ? n_v : n_u;
  const int64_t n_az = az_is_u ? n_u : n_v;
  for (int64_t r = 0; r < n_range; ++r) {
    double best = -1.0;
    double sum = 0.0;
    for (int64_t a = 0; a < n_az; ++a) {
      const std::complex<float> z = az_is_u ? img[static_cast<size_t>(r * n_az + a)]
                                            : img[static_cast<size_t>(a * n_range + r)];
      const double p = static_cast<double>(z.real() * z.real() + z.imag() * z.imag());
      sum += p;
      if (p > best) {
        best = p;
      }
    }
    const double mean = sum / static_cast<double>(n_az);
    if (best > 10.0 * mean) {
      ++n_strong;
    }
  }
  return n_strong;
}

// Folding with `std::max` would drop a NaN silently -- `std::max(a, b)` returns
// `a` when `b` is NaN -- so an all-NaN vector would report a peak of 0.0 and
// *satisfy* the `EXPECT_LT(peak_abs(...), 1e-6)` pass-through assertions below
// instead of failing them. Report the non-finite value instead of hiding it.
double peak_abs(const std::vector<double>& v) {
  double peak = 0.0;
  for (double x : v) {
    if (!std::isfinite(x)) {
      ADD_FAILURE() << "peak_abs: non-finite phase sample (" << x << ")";
      return std::numeric_limits<double>::quiet_NaN();
    }
    peak = std::max(peak, std::abs(x));
  }
  return peak;
}

TEST(AutofocusOp, RecoversQuadraticUpToAffine) {
  const auto phi = injected_phi();
  const auto img = make_image<float>(phi, kStrong, true);
  EXPECT_GE(host_strong_count(img, kRg, kAz, true), 16)
      << "host image itself lacks strong scatterers";
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz});
  ASSERT_TRUE(out.ok) << out.failure;
  ASSERT_EQ(out.phase.size(), kAz);
  auto est = phase_as_double(out.phase);
  auto want = injected_phi_pulse();
  remove_affine(&est);
  remove_affine(&want);
  EXPECT_LT(phase_rms(est, want), 0.25);
}

// The estimator must not care whether the error is even or odd about the dwell
// centre. It used to: integrating the phase gradient up the azimuth-FFT bin
// axis ran straight through the bin pair where pulse adjacency breaks, which
// handed the second half of the aperture a constant taken across the wrap and
// emitted the two halves offset from each other. An even error barely notices
// -- which is why every sweep of quadratics looked healthy -- and an odd one is
// destroyed. On the Kourou collect the same 1-cycle sinusoid shifted a quarter
// cycle scored 0.160 rad even against 3.818 rad odd; both are ~0.18 rad now.
TEST(AutofocusOp, RecoversOddAndEvenErrorsAlike) {
  const double amplitude = 2.0;
  const int64_t n_az = 1024;
  double worst = 0.0;
  std::array<double, 2> rms{};
  for (int which = 0; which < 2; ++which) {
    const bool odd = which == 1;
    // Built and asserted in *pulse* order, which is the domain the error
    // physically lives in and the domain the port is indexed by. The injection
    // is the only thing that moves to bin order, through the same inverse the
    // operator's own mapping defines -- so a wrong mapping cannot cancel
    // itself out of the comparison the way it does when the assertion is made
    // in bin order.
    auto want = sinusoidal_phi(amplitude, 1.0, odd, n_az);
    auto phi = want;
    phi = to_bin_order(phi);
    const auto img = make_image<float>(phi, kStrong, true, n_az);
    const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, n_az});
    ASSERT_TRUE(out.ok) << (odd ? "odd: " : "even: ") << out.failure;
    auto est = phase_as_double(out.phase);
    remove_affine(&est);
    remove_affine(&want);
    rms[static_cast<size_t>(which)] = phase_rms(est, want);
    worst = std::max(worst, rms[static_cast<size_t>(which)]);
  }
  // Both inside the pi/8 acceptance target of SAR/TESTING.md, and within a
  // factor of two of each other: the point is the absence of a symmetry
  // preference, which a pass/fail on the odd case alone would not pin.
  EXPECT_LT(worst, 0.3927) << "even " << rms[0] << " rad, odd " << rms[1] << " rad";
  EXPECT_LT(std::max(rms[0], rms[1]), 2.0 * std::min(rms[0], rms[1]) + 0.05)
      << "even " << rms[0] << " rad, odd " << rms[1] << " rad";
}

// A pure quadratic, with no ramp or offset for `remove_affine` to strip. The
// canonical test above injects both, and the sub-bin remainder the integer
// circshift cannot take out costs it ~0.2 rad -- enough to mask how accurate
// the estimator is, and enough to mask a regression in it.
std::vector<double> pure_quadratic(double amplitude, int64_t n_az = kAz) {
  std::vector<double> phi(static_cast<size_t>(n_az));
  const double mid = 0.5 * static_cast<double>(n_az - 1);
  for (int64_t k = 0; k < n_az; ++k) {
    const double x = (static_cast<double>(k) - mid) / std::max(mid, 1.0);
    phi[static_cast<size_t>(k)] = amplitude * x * x;
  }
  return phi;
}

// `pure_quadratic` is built in *bin* index and is even about the dwell centre
// once converted: bin 0 carries the peak and the two aperture edges meet at
// zero. So the three tests that use it agree on one symmetry class, and the
// odd half of the bin-to-pulse mapping -- the half that produced both of the
// faults this file guards -- is exercised only by the sinusoid pair above, at
// their much looser bound. This is the odd counterpart at the tight bound: a
// cubic across the dwell in pulse order, which survives `remove_affine`
// because only its constant and linear terms are blind to PGA.
std::vector<double> odd_cubic(double amplitude, int64_t n_az = kAz) {
  std::vector<double> phi(static_cast<size_t>(n_az));
  const double half = 0.5 * static_cast<double>(n_az);
  for (int64_t p = 0; p < n_az; ++p) {
    const double x = (static_cast<double>(p) - half) / half;
    phi[static_cast<size_t>(p)] = amplitude * x * x * x;
  }
  return holoscan::radar::sar_test::to_bin_order(phi);
}

// The odd counterpart to `pure_quadratic`. That one is written in bin index
// and reads as even about the dwell centre once converted -- bin 0 is the
// centre and the two halves mirror -- and so does every other shape the
// accuracy tests below inject. So all of their bounds were measured on the
// one symmetry class the integration seam cannot perturb: a step at the dwell
// centre is small for an even error and is the whole error for an odd one.
// A cubic in pulse index is odd about the centre and survives `remove_affine`.
std::vector<double> pure_cubic(double amplitude, int64_t n_az = kAz) {
  std::vector<double> phi(static_cast<size_t>(n_az));
  const double half = 0.5 * static_cast<double>(n_az);
  for (int64_t p = 0; p < n_az; ++p) {
    const double x = (static_cast<double>(p) - half) / half;
    phi[static_cast<size_t>(p)] = amplitude * x * x * x;
  }
  return holoscan::radar::sar_test::to_bin_order(phi);
}

// Both sides are detrended in *pulse* order, which is the domain PGA's blind
// affine term lives in. This used to strip the estimate's affine part in pulse
// order and the injection's in bin order, which are different operations; they
// cancel only when the injected error is symmetric about the dwell centre, and
// every caller below happens to pass exactly that.
double residual_rms(const HostTensor& phase, const std::vector<double>& injected) {
  auto est = phase_as_double(phase);
  auto want = holoscan::radar::sar_test::to_pulse_order(injected);
  remove_affine(&est);
  remove_affine(&want);
  return phase_rms(est, want);
}

// At the full azimuth extent the estimator is not merely inside the acceptance
// target, it is exact to five decimal places. Pinning that is what makes the
// window-width loss below legible as a defect rather than as how PGA behaves.
TEST(AutofocusOp, PureQuadraticIsRecoveredExactlyAtFullWindow) {
  for (double amplitude : {2.5, 5.0}) {
    const auto phi = pure_quadratic(amplitude);
    const auto img = make_image<float>(phi, kStrong, true);
    const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, 1.0);
    ASSERT_TRUE(out.ok) << out.failure;
    EXPECT_LT(residual_rms(out.phase, phi), 0.01) << "amplitude " << amplitude;
  }
}

// SAR/TESTING.md asks for the residual's *shape*, not only its RMS: an
// estimate can carry the right total magnitude and still place it on the
// wrong pulses, and an RMS averages that away. Sampling the dwell centre, the
// quarter points and both edges localizes a sign inversion partway out or a
// flattening at the edges to the bin-to-pulse mapping, which is what an RMS
// cannot do. Positions are read in **pulse** order, as TESTING.md requires --
// reading them in bin order once inverted the diagnosis of the window fault
// and sent it after the wrong cause. The cubic is the stimulus because an
// error even about the centre reads the same forwards and backwards, so it
// cannot catch a reversal of the azimuth axis.
TEST(AutofocusOp, ResidualShapeMatchesTruthAcrossTheDwell) {
  const auto phi = pure_cubic(5.0);
  const auto img = make_image<float>(phi, kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, 1.0);
  ASSERT_TRUE(out.ok) << out.failure;

  auto est = phase_as_double(out.phase);
  auto want = holoscan::radar::sar_test::to_pulse_order(phi);
  ASSERT_EQ(est.size(), want.size());
  remove_affine(&est);
  remove_affine(&want);

  // Both edges, both quarter points, and the centre. The edges are where a
  // window that is too narrow flattens the estimate and where the integration
  // constant lands if it is taken at the wrong end.
  const int64_t last = kAz - 1;
  const std::array<std::pair<const char*, int64_t>, 5> probes{{{"leading edge", 0},
                                                               {"first quarter", kAz / 4},
                                                               {"dwell centre", kAz / 2},
                                                               {"third quarter", 3 * kAz / 4},
                                                               {"trailing edge", last}}};
  // Absolute, not relative: the truth passes through zero at the centre and a
  // relative bound there would be unsatisfiable. 0.39 rad is the acceptance
  // target of SAR/TESTING.md, applied pointwise rather than in aggregate.
  for (const auto& [label, k] : probes) {
    const auto i = static_cast<size_t>(k);
    EXPECT_NEAR(est[i], want[i], 0.3927) << label << " (pulse " << k << "): estimate " << est[i]
                                         << " rad, truth " << want[i] << " rad";
  }

  // The shape probes above are blind to a global sign flip at the two edges
  // alone, so pin the swing as well: an estimate that is correct pointwise but
  // half the amplitude would pass every bound above on a shallow stimulus.
  EXPECT_GT(peak_abs(est), 0.9 * peak_abs(want))
      << "estimate peak-to-peak " << peak_abs(est) << " rad against truth " << peak_abs(want);
}

// The odd-error leg of the test above, and the one that actually exercises
// the bin-to-pulse seam. Held to the same bound: there is no reason for PGA
// to recover an odd error less exactly than an even one at the full window,
// so a split between the two legs is the signature of an integration or
// mapping fault rather than of estimator accuracy.
TEST(AutofocusOp, PureCubicIsRecoveredExactlyAtFullWindow) {
  for (double amplitude : {2.5, 5.0}) {
    const auto phi = pure_cubic(amplitude);
    const auto img = make_image<float>(phi, kStrong, true);
    const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, 1.0);
    ASSERT_TRUE(out.ok) << out.failure;
    EXPECT_LT(residual_rms(out.phase, phi), 0.01) << "amplitude " << amplitude;
  }
}

// The odd counterpart of the test above, at the same bound. An integration
// that anchors anywhere but DC, or that runs up the bin axis, splits the
// aperture's two halves by a step at the dwell centre -- which is nearly free
// on an even error and is the whole error on an odd one, so the quadratic
// above cannot see it.
TEST(AutofocusOp, OddCubicIsRecoveredExactlyAtFullWindow) {
  for (double amplitude : {2.5, 5.0}) {
    const auto phi = odd_cubic(amplitude);
    const auto img = make_image<float>(phi, kStrong, true);
    const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, 1.0);
    ASSERT_TRUE(out.ok) << out.failure;
    // The estimate must carry the error, not merely sit near zero: a refusal
    // emits zeros, so the truth has to retain real swing after detrending for
    // the bound below to mean anything. Detrending `x^3` over [-1, 1) removes
    // its best-fit `0.6x`, leaving `x^3 - 0.6x` with a peak of 0.4 at the
    // aperture edges; 0.3 is that with margin for the discrete grid.
    auto truth = holoscan::radar::sar_test::to_pulse_order(phi);
    remove_affine(&truth);
    EXPECT_GT(peak_abs(truth), 0.3 * amplitude) << "stimulus is degenerate";
    EXPECT_LT(residual_rms(out.phase, phi), 0.01) << "amplitude " << amplitude;
  }
}

// Narrowing the window costs accuracy, because fewer image samples carry less
// of the phase history, and what it costs is *swing*: the estimate keeps its
// shape and under-reports the amplitude, retaining 100%, 100%, 96% and 86% of
// the true peak-to-peak at window-to-azimuth ratios of 1:1, 1:2, 1:4 and 1:8.
// Every ratio stays inside the 0.39 rad acceptance target of SAR/TESTING.md on
// a 5 rad error. The bounds are the measured behaviour, not the target;
// tighten them as the estimator improves and never loosen them.
//
// The 1:8 bound is also the regression guard on the transform length. The
// window narrows the image support, and the FFT behind the LUE has to stay at
// the full azimuth extent regardless: packing the window into a `width`-long
// transform instead decimates the phase history by n_az/width, so each LUE
// difference spans that many bins and its small-angle form saturates. That is
// what this test measured at 0.830 rad before 2026-10-27, and it is a 5.5x
// regression on this bound.
TEST(AutofocusOp, NarrowWindowCostsSwingAndStaysInsideTheTarget) {
  const auto phi = pure_quadratic(5.0);
  const auto img = make_image<float>(phi, kStrong, true);
  auto truth = phi;
  remove_affine(&truth);
  const double want_peak = peak_abs(truth);
  double previous = 0.0;
  // Measured 0.0020, 0.0089, 0.0363, 0.1501 rad.
  const double bounds[] = {0.005, 0.02, 0.06, 0.25};
  const double swings[] = {0.99, 0.98, 0.93, 0.80};
  int i = 0;
  for (double window_fraction : {1.0, 0.5, 0.25, 0.125}) {
    const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, window_fraction);
    ASSERT_TRUE(out.ok) << out.failure;
    const double rms = residual_rms(out.phase, phi);
    EXPECT_GT(rms, previous) << "window_fraction " << window_fraction
                             << ": the residual is expected to grow as the window narrows; "
                                "if this now falls, the estimator improved -- retighten the bound";
    EXPECT_LT(rms, bounds[i]) << "window_fraction " << window_fraction << ": residual " << rms
                              << " rad regressed past its measured value";
    auto est = phase_as_double(out.phase);
    remove_affine(&est);
    EXPECT_GT(peak_abs(est) / want_peak, swings[i])
        << "window_fraction " << window_fraction << ": the estimate lost more swing than measured";
    previous = rms;
    ++i;
  }
  EXPECT_LT(previous, 0.393) << "the narrowest window no longer meets the acceptance target";
}

// Accuracy over azimuth extent, at a fixed absolute window width -- the way
// the autofocus application sizes `window_fraction`, which targets ~64 columns
// rather than a fraction of the aperture.
//
// **On a clean synthetic scene the estimator is flat in extent and essentially
// exact, and that is the point of pinning it.** On Kourou the affine-removed
// residual against injected truth grows with aperture length and is not
// monotonic -- 0.149 / 0.283 / 0.692 / 0.428 / 0.332 rad at 8000 / 12000 /
// 16000 / 20000 / 23375 pulses against a 0.39 rad target -- and the estimate
// over-reports swing by up to 31%. None of that reproduces here, at 0.002 to
// 0.004 rad and swing within 2% across a 16x range of extents. So the real
// trend is **not** a property of extent alone: it needs something this scene
// does not have (real clutter, range bins of varying scatterer quality, an
// error that is not a pure quadratic, or the oversampled polar auxiliary).
// That is why the bounds below are tight. They cannot catch the real-data
// defect -- only the collect can, and the recipe for it is in
// `operators/sar/autofocus_op/README.md` -- but they do catch the
// estimator acquiring an extent dependence on a clean scene, which is the
// nearest thing CI can observe and would be the first sign the cause is
// simpler than it looks.
//
// Bounds are measured behaviour, not the acceptance target; tighten them as
// the estimator improves and never loosen them.
TEST(AutofocusOp, AccuracyIsFlatOverAzimuthExtentOnACleanScene) {
  constexpr int64_t kWindowColumns = 64;
  // Measured 0.0020, 0.0040, 0.0043 rad and swing 1.003, 0.991, 0.984. The
  // extents stop at 1024 because make_image injects through an O(n^2) host
  // DFT; 16x of extent is enough to show a dependence if one appears.
  const double bounds[] = {0.004, 0.006, 0.006};
  const double swing_floor[] = {0.99, 0.98, 0.97};
  int i = 0;
  for (int64_t n_az : {64, 256, 1024}) {
    const auto phi = pure_quadratic(5.0, n_az);
    const auto img = make_image<float>(phi, kStrong, true, n_az);
    auto truth = phi;
    remove_affine(&truth);
    const double window_fraction =
        std::min(1.0, static_cast<double>(kWindowColumns) / static_cast<double>(n_az));
    const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, n_az}, 4, window_fraction);
    ASSERT_TRUE(out.ok) << "n_az " << n_az << ": " << out.failure;
    ASSERT_EQ(out.phase.size(), n_az);
    const double rms = residual_rms(out.phase, phi);
    auto est = phase_as_double(out.phase);
    remove_affine(&est);
    const double swing = peak_abs(est) / peak_abs(truth);
    EXPECT_LT(rms, bounds[i]) << "n_az " << n_az << ": residual " << rms
                              << " rad regressed past its measured value";
    EXPECT_GT(swing, swing_floor[i]) << "n_az " << n_az << ": swing " << swing
                                     << " -- the estimate lost more amplitude than measured";
    EXPECT_LT(swing, 1.02) << "n_az " << n_az << ": swing " << swing
                           << " -- over-reporting swing is the dominant error on a long real "
                              "aperture, and it must not start here";
    ++i;
  }
}

// Every kernel in the stage is one element per thread with no grid-stride
// loop, so a launch grid that does not cover the image silently leaves its tail
// unwritten -- and the tail of a freshly pooled output is zeros. This image is
// deliberately just past 65535 blocks of 256 threads, the clamp that used to be
// applied: on a 2478 x 8000 auxiliary it dropped the last 380 range rows out of
// image_out and hid them from the peak finder too.
TEST(AutofocusOp, LargeImageIsNotTruncatedByTheLaunchGrid) {
  constexpr int64_t kWide = 64;
  constexpr int64_t kTall = (65535 * 256) / kWide + 64;
  // One bright pixel per row over a low floor, so every row clears the
  // peak-to-mean gate and the real PGA path runs. The pass-through is a plain
  // cudaMemcpy and would not exercise a launch grid at all.
  std::vector<std::complex<float>> img(static_cast<size_t>(kTall * kWide), {0.01F, 0.F});
  for (int64_t r = 0; r < kTall; ++r) {
    img[static_cast<size_t>(r * kWide + (r % kWide))] = {1.F, 0.F};
  }
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kTall, kWide}, 1);
  ASSERT_TRUE(out.ok) << out.failure;
  ASSERT_EQ(out.image.size(), kTall * kWide);
  const auto* got = out.image.as<std::complex<float>>();
  int64_t zero_rows = 0;
  // A non-finite row has to be counted separately: a NaN row sum is not 0.0, so
  // testing only for zeros would report a fully covered grid on an all-NaN
  // image -- the same blindness that let a NaN product pass this suite once.
  int64_t nonfinite_rows = 0;
  for (int64_t r = 0; r < kTall; ++r) {
    double row = 0.0;
    for (int64_t c = 0; c < kWide; ++c) {
      row += std::abs(got[r * kWide + c]);
    }
    if (!std::isfinite(row)) {
      ++nonfinite_rows;
    } else if (row == 0.0) {
      ++zero_rows;
    }
  }
  EXPECT_EQ(zero_rows, 0) << "image_out left " << zero_rows << " of " << kTall
                          << " rows unwritten; the launch grid does not cover the image";
  EXPECT_EQ(nonfinite_rows, 0) << "image_out has " << nonfinite_rows << " of " << kTall
                               << " non-finite rows";
}

TEST(AutofocusOp, AzimuthAxisV) {
  const auto phi = injected_phi();
  const auto img = make_image<float>(phi, kStrong, false);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kV, img, {kAz, kRg});
  ASSERT_TRUE(out.ok) << out.failure;
  ASSERT_EQ(out.phase.size(), kAz);
  auto est = phase_as_double(out.phase);
  auto want = injected_phi_pulse();
  remove_affine(&est);
  remove_affine(&want);
  EXPECT_LT(phase_rms(est, want), 0.25);
}

TEST(AutofocusOp, WrongAxisDoesNotRecover) {
  const auto phi = injected_phi();
  const auto img = make_image<float>(phi, kStrong, true);
  const auto good = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz});
  const auto bad = run_af<float>(AutofocusAzimuthAxis::kV, img, {kRg, kAz});
  ASSERT_TRUE(good.ok) << good.failure;
  ASSERT_TRUE(bad.ok) << bad.failure;
  // Declared axis is v (length kRg), so the estimate cannot match the kAz-long
  // injected pulse phase -- D20's declared-not-checked hazard. Along range the
  // 10x peak-to-mean gate finds no scatterers, so this run passes through;
  // either way the image is not recovered.
  ASSERT_EQ(bad.phase.size(), kRg);
  const auto sh_good = phase_as_double(good.sharpness);
  const auto sh_bad = phase_as_double(bad.sharpness);
  ASSERT_FALSE(sh_good.empty());
  ASSERT_FALSE(sh_bad.empty());
  EXPECT_LT(sh_bad.back(), 0.9 * sh_good.back());
}

TEST(AutofocusOp, MinScatterersPassesThrough) {
  const auto phi = injected_phi();
  const auto img = make_image<float>(phi, 4, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, 0.5, 16);
  ASSERT_TRUE(out.ok) << out.failure;
  ASSERT_EQ(out.phase.size(), kAz);
  EXPECT_LT(peak_abs(phase_as_double(out.phase)), 1e-6);
  const auto* got = out.image.as<std::complex<float>>();
  double max_err = 0.0;
  for (size_t i = 0; i < img.size(); ++i) {
    max_err = std::max(max_err, static_cast<double>(std::abs(got[i] - img[i])));
  }
  EXPECT_LT(max_err, 1e-5);
  EXPECT_EQ(termination_of(out), AutofocusTermination::kRefusedScatterers);
  EXPECT_EQ(iterations_of(out), 0);
}

TEST(AutofocusOp, MaxCorrectionPassesThrough) {
  const auto phi = injected_phi();
  const auto img = make_image<float>(phi, kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, 0.5, 16, 0.05);
  ASSERT_TRUE(out.ok) << out.failure;
  EXPECT_LT(peak_abs(phase_as_double(out.phase)), 1e-6);
  // Pass-through is the behaviour under test, so check the image as well as
  // the phase -- a zero `phase_correction` alongside a *corrected* image
  // would satisfy the line above and still be wrong.
  const auto* got = out.image.as<std::complex<float>>();
  double max_err = 0.0;
  for (size_t i = 0; i < img.size(); ++i) {
    max_err = std::max(max_err, static_cast<double>(std::abs(got[i] - img[i])));
  }
  EXPECT_LT(max_err, 1e-5) << "image_out differs from the input on a refused tick";
  EXPECT_EQ(termination_of(out), AutofocusTermination::kRefusedMaxCorrection);
  // Iterations ran and were accepted before the ceiling rejected the total, but
  // none of them survive into phase_correction, so the count reports zero
  // rather than the work done.
  EXPECT_EQ(iterations_of(out), 0);
}

// The reason these ports exist: on every pre-existing port a refusal and a
// clean no-op image are the same tick -- zero phase_correction, flat sharpness
// -- so a consumer could not tell "I declined" from "there was nothing to do".
TEST(AutofocusOp, TerminationSeparatesRefusalFromAnUneventfulRun) {
  const std::vector<double> zero;
  const auto clean = make_image<float>(zero, kStrong, true);
  const auto ok = run_af<float>(AutofocusAzimuthAxis::kU, clean, {kRg, kAz});
  ASSERT_TRUE(ok.ok) << ok.failure;

  const auto starved = make_image<float>(injected_phi(), 4, true);
  const auto refused = run_af<float>(AutofocusAzimuthAxis::kU, starved, {kRg, kAz}, 4, 0.5, 16);
  ASSERT_TRUE(refused.ok) << refused.failure;

  // Indistinguishable on the old ports...
  EXPECT_LT(peak_abs(phase_as_double(refused.phase)), 1e-6);
  const auto refused_sharp = phase_as_double(refused.sharpness);
  EXPECT_NEAR(refused_sharp.back(), refused_sharp.front(), 1e-3 * refused_sharp.front());
  // ...and distinguishable on the new ones.
  EXPECT_EQ(termination_of(refused), AutofocusTermination::kRefusedScatterers);
  EXPECT_TRUE(autofocus_refused(termination_of(refused)));
  EXPECT_FALSE(autofocus_refused(termination_of(ok)));
}

// the SAR operator READMEs states the AutofocusTermination values are contract:
// append, never renumber. They leave the operator as a bare int32 with no
// accompanying name, so a consumer decoding them has nothing but the number,
// and a reordering of the enum would change their meaning silently.
static_assert(static_cast<int>(AutofocusTermination::kConverged) == 0);
static_assert(static_cast<int>(AutofocusTermination::kBudgetExhausted) == 1);
static_assert(static_cast<int>(AutofocusTermination::kRefusedSharpness) == 2);
static_assert(static_cast<int>(AutofocusTermination::kRefusedScatterers) == 3);
static_assert(static_cast<int>(AutofocusTermination::kRefusedMaxCorrection) == 4);
static_assert(static_cast<int>(AutofocusTermination::kStoppedScatterers) == 5);

TEST(AutofocusOp, IterationsCountsOnlyAcceptedIterations) {
  const auto img = make_image<float>(injected_phi(), kStrong, true);
  // A budget far beyond what this image needs: the guard stops the loop early,
  // so the count must be the accepted iterations and not the budget.
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 32);
  ASSERT_TRUE(out.ok) << out.failure;
  EXPECT_EQ(termination_of(out), AutofocusTermination::kConverged);
  EXPECT_GE(iterations_of(out), 1);
  EXPECT_LT(iterations_of(out), 32);

  // A budget of one cannot stop early, so it reports exhaustion instead.
  const auto capped = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 1);
  ASSERT_TRUE(capped.ok) << capped.failure;
  EXPECT_EQ(termination_of(capped), AutofocusTermination::kBudgetExhausted);
  EXPECT_EQ(iterations_of(capped), 1);
}

TEST(AutofocusOp, WellFocusedStaysSharp) {
  const std::vector<double> zero;
  const auto img = make_image<float>(zero, kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz});
  ASSERT_TRUE(out.ok) << out.failure;
  const auto* got = out.image.as<std::complex<float>>();
  double max_err = 0.0;
  for (size_t i = 0; i < img.size(); ++i) {
    max_err = std::max(max_err, static_cast<double>(std::abs(got[i] - img[i])));
  }
  EXPECT_LT(max_err, 0.05);
  const auto sharp = phase_as_double(out.sharpness);
  ASSERT_EQ(sharp.size(), 5U);
  const double span = std::abs(sharp.back() - sharp.front());
  EXPECT_LT(span, 0.05 * std::max(sharp.front(), 1e-6));
}

TEST(AutofocusOp, SharpnessImprovesWhenDefocused) {
  const auto phi = injected_phi();
  const auto img = make_image<float>(phi, kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz});
  ASSERT_TRUE(out.ok) << out.failure;
  const auto sharp = phase_as_double(out.sharpness);
  ASSERT_EQ(sharp.size(), 5U);
  EXPECT_GT(sharp.back(), sharp.front());
  // Strictly non-decreasing, which is what TESTING.md asks for and what the
  // operator's own accept/reject guard enforces: an iteration that lowers
  // sharpness is discarded and the previous value is restored. The 2% slack
  // this used to allow let a trace lose ~8% across four iterations and still
  // pass a test named for monotonic improvement.
  for (size_t i = 1; i < sharp.size(); ++i) {
    EXPECT_GE(sharp[i], sharp[i - 1]) << "sharpness dropped at iteration " << i;
  }
}

TEST(AutofocusOp, AzimuthAxisRequired) {
  const auto img = make_image<float>(injected_phi(), kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kUnspecified, img, {kRg, kAz});
  ASSERT_FALSE(out.ran_ok) << "the operator did not reject this configuration";
  EXPECT_NE(out.failure.find("azimuth_axis"), std::string::npos) << out.failure;
}

TEST(AutofocusOp, RejectsNonMatrixImage) {
  const auto phi = injected_phi();
  const auto slice = make_image<float>(phi, kStrong, true);
  std::vector<std::complex<float>> cube(static_cast<size_t>(2 * kRg * kAz));
  std::copy(slice.begin(), slice.end(), cube.begin());
  std::copy(slice.begin(), slice.end(), cube.begin() + kRg * kAz);
  // (num_v, num_u) only: a leading batch axis is an application concern, not
  // an opaque module axis this operator silently loops over.
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, cube, {2, kRg, kAz});
  ASSERT_FALSE(out.ran_ok) << "the operator did not reject this configuration";
  EXPECT_NE(out.failure.find("num_v"), std::string::npos) << out.failure;
}

TEST(AutofocusOp, RejectsIterationsBeyondPoolBound) {
  const auto img = make_image<float>(injected_phi(), kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz},
                                 holoscan::radar::kMaxAutofocusIterations + 1);
  ASSERT_FALSE(out.ran_ok) << "the operator did not reject this configuration";
  EXPECT_NE(out.failure.find("num_iterations"), std::string::npos) << out.failure;
}

// +inf is the uncapped default, but -inf used to pass validation too, and the
// stage skips a non-finite ceiling, so the guard silently never applied.
TEST(AutofocusOp, RejectsNegativeInfiniteMaxCorrection) {
  const auto img = make_image<float>(injected_phi(), kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, 0.5, 16,
                                 -std::numeric_limits<double>::infinity());
  ASSERT_FALSE(out.ran_ok) << "the operator did not reject this configuration";
  EXPECT_NE(out.failure.find("max_correction"), std::string::npos) << out.failure;
}

TEST(AutofocusOp, FloatDoubleAgree) {
  const auto phi = injected_phi();
  const auto img_f = make_image<float>(phi, kStrong, true);
  const auto img_d = make_image<double>(phi, kStrong, true);
  const auto out_f = run_af<float>(AutofocusAzimuthAxis::kU, img_f, {kRg, kAz});
  const auto out_d = run_af<double>(AutofocusAzimuthAxis::kU, img_d, {kRg, kAz});
  ASSERT_TRUE(out_f.ok) << out_f.failure;
  ASSERT_TRUE(out_d.ok) << out_d.failure;
  auto pf = phase_as_double(out_f.phase);
  auto pd = phase_as_double(out_d.phase);
  remove_affine(&pf);
  remove_affine(&pd);
  EXPECT_LT(phase_rms(pf, pd), 0.05);
  // Agreeing with each other is not the same as being right: two pass-throughs
  // agree perfectly. Score both against the injected truth, which is also the
  // only coverage anywhere that the double instantiation recovers a phase at
  // all -- run_af<double> appears nowhere else in this file.
  auto want = injected_phi_pulse();
  remove_affine(&want);
  EXPECT_LT(phase_rms(pf, want), 0.25) << "float estimate does not recover the injected phase";
  EXPECT_LT(phase_rms(pd, want), 0.25) << "double estimate does not recover the injected phase";
}

// The [live] convention is per-tick with no latch: the override holds for the
// tick it arrives on, and the next tick falls back to Params. Asserting the
// second tick equals a Params-only run is what distinguishes that from a
// sticky override.
TEST(AutofocusOp, LiveWindowFractionAppliesForOneTick) {
  using Op = AutofocusOp<float>;
  const auto img = make_image<float>(injected_phi(), kStrong, true);
  const std::vector<int64_t> shape{kRg, kAz};
  auto fixture = holoscan::testing::create_operator_test<Op>(extents_for(shape));
  fixture.set_params(AutofocusParams{.azimuth_axis = AutofocusAzimuthAxis::kU});
  auto frame0 = to_device_tensor(img, shape, complex_dtype<float>());
  auto frame1 = to_device_tensor(img, shape, complex_dtype<float>());
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::milliseconds{100},
                                                 holoscan::testing::input("window_fraction", 0.25),
                                                 holoscan::testing::input("image", *frame0)));
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::milliseconds{200},
                                                 holoscan::testing::input("image", *frame1)));
  fixture.capture_snapshot<holoscan::Tensor>("phase_correction", 2U, kAz * sizeof(float));
  fixture.capture_snapshot<holoscan::Tensor>("image_out", 2U, frame0->nbytes());
  fixture.capture_snapshot<holoscan::Tensor>(
      "sharpness", 2U,
      static_cast<std::size_t>(AutofocusParams{}.num_iterations + 2) * sizeof(float));
  fixture.input_source_device("image", holoscan::DeviceId{0});
  fixture.output_device("image_out", holoscan::DeviceId{0});
  fixture.output_device("phase_correction", holoscan::DeviceId{0});
  fixture.output_device("sharpness", holoscan::DeviceId{0});
  // Routed but not snapshotted: this test is about the [live] override, and the
  // fixture requires every declared output to have a device.
  fixture.output_device("termination", holoscan::DeviceId{0});
  fixture.output_device("iterations", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running =
      fixture.run_async(completion.run_options(holoscan::testing::FrameRelease::kManual));
  const auto released0 = running.release_frame(completion.remaining());
  const bool observed0 = completion.wait_for_output(running, "sharpness", 1U);
  const auto released1 = running.release_frame(completion.remaining());
  const bool observed1 = completion.wait_for_output(running, "sharpness", 2U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(released0) << released0.error().message;
  ASSERT_TRUE(released1) << released1.error().message;
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_TRUE(observed0);
  ASSERT_TRUE(observed1);

  const auto captured = result.snapshots<holoscan::Tensor>("phase_correction");
  ASSERT_EQ(captured.size(), 2U);
  const auto overridden = phase_as_double(to_host(captured[0]));
  const auto fell_back = phase_as_double(to_host(captured[1]));
  ASSERT_EQ(overridden.size(), static_cast<size_t>(kAz));
  ASSERT_EQ(fell_back.size(), static_cast<size_t>(kAz));

  // The reference has to be a genuinely Params-only run, so it takes the
  // AutofocusParams defaults rather than run_af's own (which differ: run_af
  // defaults to 4 iterations, the operator to 8).
  const AutofocusParams defaults{};
  const auto reference =
      run_af<float>(AutofocusAzimuthAxis::kU, img, shape, defaults.num_iterations,
                    defaults.window_fraction, defaults.min_scatterers, defaults.max_correction);
  ASSERT_TRUE(reference.ok) << reference.failure;
  const auto params_only = phase_as_double(reference.phase);
  // Tick 2 fell back to Params.
  EXPECT_LT(phase_rms(fell_back, params_only), 1e-6);
  // ...and tick 1 did not. Without this the test passes just as happily when
  // the override is wired but ignored entirely, which is the failure it exists
  // to catch. A different window is a different estimate; asserting only that
  // they are not bit-identical keeps the claim robust to retuning.
  EXPECT_GT(phase_rms(overridden, params_only), 1e-6)
      << "window_fraction=0.25 produced the same estimate as the Params 0.5: "
         "the [live] override had no effect";
}
}  // namespace
