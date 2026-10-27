// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/autofocus_op/autofocus_op.hpp"
#include "utils/fixture_completion.hpp"
#include "utils/tensor_fixtures.hpp"

// Operator-scope tests. The estimate's journey onto a differently formed
// product -- transfer through PhaseCorrectionOp, and the whole Figure 2
// retain/reform path -- is an application concern and is tested with the app,
// which owns the resampling and aux-lattice operators those cases need.

namespace {

using holoscan::radar::AutofocusAzimuthAxis;
using holoscan::radar::AutofocusOp;
using holoscan::radar::AutofocusParams;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

constexpr int64_t kAz = 64;
constexpr int64_t kRg = 32;
constexpr int64_t kStrong = 24;

const double kPi = std::acos(-1.0);

void host_dft(std::vector<std::complex<double>>* x, bool inverse) {
  const int64_t n = static_cast<int64_t>(x->size());
  std::vector<std::complex<double>> out(static_cast<size_t>(n));
  const double sign = inverse ? 1.0 : -1.0;
  const double scale = inverse ? (1.0 / static_cast<double>(n)) : 1.0;
  for (int64_t k = 0; k < n; ++k) {
    std::complex<double> s{0.0, 0.0};
    for (int64_t t = 0; t < n; ++t) {
      const double ang = sign * 2.0 * kPi * static_cast<double>(t * k) / static_cast<double>(n);
      s += (*x)[static_cast<size_t>(t)] * std::complex<double>(std::cos(ang), std::sin(ang));
    }
    out[static_cast<size_t>(k)] = s * scale;
  }
  x->swap(out);
}

std::vector<double> injected_phi() {
  std::vector<double> phi(static_cast<size_t>(kAz));
  const double mid = 0.5 * static_cast<double>(kAz - 1);
  for (int64_t k = 0; k < kAz; ++k) {
    const double x = (static_cast<double>(k) - mid) / std::max(mid, 1.0);
    phi[static_cast<size_t>(k)] = 2.5 * x * x + 0.12 * static_cast<double>(k) + 0.4;
  }
  return phi;
}

// PGA is blind to a constant and a linear ramp: they shift the image rather
// than blur it, so only the residual is recoverable and comparisons strip both.
void remove_affine(std::vector<double>* phi) {
  const int64_t n = static_cast<int64_t>(phi->size());
  if (n < 2) {
    return;
  }
  double sum = 0.0;
  double sumk = 0.0;
  for (int64_t k = 0; k < n; ++k) {
    sum += (*phi)[static_cast<size_t>(k)];
    sumk += static_cast<double>(k) * (*phi)[static_cast<size_t>(k)];
  }
  const double xmean = 0.5 * static_cast<double>(n - 1);
  const double denom =
      static_cast<double>(n) * (static_cast<double>(n) * static_cast<double>(n) - 1.0) / 12.0;
  const double slope = (sumk - xmean * sum) / denom;
  const double intercept = sum / static_cast<double>(n) - slope * xmean;
  for (int64_t k = 0; k < n; ++k) {
    (*phi)[static_cast<size_t>(k)] -= intercept + slope * static_cast<double>(k);
  }
}

double phase_rms(const std::vector<double>& a, const std::vector<double>& b) {
  double acc = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(a.size()));
}

// A range-Doppler stand-in: point scatterers in n_strong rows, clutter in the
// rest, with phi injected on the azimuth spectrum of every row -- one
// scene-common error, which is the space-invariance PGA assumes.
template <typename T>
std::vector<std::complex<T>> make_image(const std::vector<double>& phi, int64_t n_strong,
                                        bool az_is_u) {
  const int64_t n_v = az_is_u ? kRg : kAz;
  const int64_t n_u = az_is_u ? kAz : kRg;
  std::vector<std::complex<T>> img(static_cast<size_t>(n_v * n_u),
                                   {static_cast<T>(0), static_cast<T>(0)});
  for (int64_t r = 0; r < kRg; ++r) {
    std::vector<std::complex<double>> row(static_cast<size_t>(kAz), {0.0, 0.0});
    if (r < n_strong) {
      row[static_cast<size_t>(kAz / 2)] = {1.0, 0.0};
    } else {
      for (int64_t a = 0; a < kAz; ++a) {
        row[static_cast<size_t>(a)] = {0.01, 0.0};
      }
    }
    host_dft(&row, false);
    for (int64_t k = 0; k < kAz; ++k) {
      const double p = phi.empty() ? 0.0 : phi[static_cast<size_t>(k)];
      row[static_cast<size_t>(k)] *= std::complex<double>(std::cos(p), std::sin(p));
    }
    host_dft(&row, true);
    for (int64_t a = 0; a < kAz; ++a) {
      const auto z = row[static_cast<size_t>(a)];
      const std::complex<T> c(static_cast<T>(z.real()), static_cast<T>(z.imag()));
      if (az_is_u) {
        img[static_cast<size_t>(r * kAz + a)] = c;
      } else {
        img[static_cast<size_t>(a * kRg + r)] = c;
      }
    }
  }
  return img;
}

struct AfOut {
  bool ok{false};
  std::string failure;
  HostTensor image;
  HostTensor phase;
  HostTensor sharpness;
};

MaxTensorExtents extents_for(const std::vector<int64_t>& shape) {
  MaxTensorExtents extents;
  extents.max_image_v = std::max<int64_t>(shape.size() > 0 ? shape[0] : 1, kAz);
  extents.max_image_u = std::max<int64_t>(shape.size() > 1 ? shape[1] : 1, kAz);
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
  const std::size_t vector_bytes = static_cast<std::size_t>(kAz + num_iterations + 2) * sizeof(T);
  fixture.template capture_snapshot<holoscan::Tensor>("image_out", 1U, input->nbytes());
  fixture.template capture_snapshot<holoscan::Tensor>("phase_correction", 1U, vector_bytes);
  fixture.template capture_snapshot<holoscan::Tensor>("sharpness", 1U, vector_bytes);
  fixture.input_source_device("image", holoscan::DeviceId{0});
  fixture.output_device("image_out", holoscan::DeviceId{0});
  fixture.output_device("phase_correction", holoscan::DeviceId{0});
  fixture.output_device("sharpness", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  // sharpness is emitted last, so observing it means all three ports have
  // landed; waiting on image_out instead races the stop against the vectors.
  const bool observed = completion.wait_for_output(running, "sharpness", 1U);
  running.request_stop();
  const auto result = running.finish();

  AfOut out;
  out.failure = result.describe_failure();
  out.ok = result.ok() && observed;
  if (!out.ok) {
    return out;
  }
  completion.check(result);
  out.image = to_host(result.template snapshot<holoscan::Tensor>("image_out"));
  out.phase = to_host(result.template snapshot<holoscan::Tensor>("phase_correction"));
  out.sharpness = to_host(result.template snapshot<holoscan::Tensor>("sharpness"));
  return out;
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

// make_image is IFFT-formed (RD). The emitted port is pulse-signed for a
// backprojected lattice (Figure 2), which conjugates that stand-in. Flip
// before comparing to an injected pulse phase.
void rd_standin_flip(std::vector<double>* phi) {
  for (double& p : *phi) {
    p = -p;
  }
}

double peak_abs(const std::vector<double>& v) {
  double peak = 0.0;
  for (double x : v) {
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
  auto want = phi;
  remove_affine(&est);
  remove_affine(&want);
  rd_standin_flip(&est);
  EXPECT_LT(phase_rms(est, want), 0.25);
}

// A pure quadratic, with no ramp or offset for `remove_affine` to strip. The
// canonical test above injects both, and the sub-bin remainder the integer
// circshift cannot take out costs it ~0.2 rad -- enough to mask how accurate
// the estimator is, and enough to mask a regression in it.
std::vector<double> pure_quadratic(double amplitude) {
  std::vector<double> phi(static_cast<size_t>(kAz));
  const double mid = 0.5 * static_cast<double>(kAz - 1);
  for (int64_t k = 0; k < kAz; ++k) {
    const double x = (static_cast<double>(k) - mid) / std::max(mid, 1.0);
    phi[static_cast<size_t>(k)] = amplitude * x * x;
  }
  return phi;
}

double residual_rms(const HostTensor& phase, const std::vector<double>& injected) {
  auto est = phase_as_double(phase);
  auto want = injected;
  remove_affine(&est);
  remove_affine(&want);
  rd_standin_flip(&est);
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

// Narrowing the window costs accuracy at the aperture edges, where the phase
// gradient is largest, while the dwell centre stays right -- the estimate keeps
// its total swing and redistributes it. The bound here is the measured
// behaviour, not the target: at a 1:8 window the residual is already past the
// 0.39 rad acceptance target of SAR/TESTING.md on a 5 rad error. Tighten these
// as the estimator improves; they must never loosen.
TEST(AutofocusOp, NarrowWindowLosesTheApertureEdges) {
  const auto phi = pure_quadratic(5.0);
  const auto img = make_image<float>(phi, kStrong, true);
  double previous = 0.0;
  for (double window_fraction : {1.0, 0.5, 0.25, 0.125}) {
    const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, window_fraction);
    ASSERT_TRUE(out.ok) << out.failure;
    const double rms = residual_rms(out.phase, phi);
    EXPECT_GT(rms, previous) << "window_fraction " << window_fraction
                             << ": the residual is expected to grow as the window narrows; "
                                "if this now falls, the estimator improved -- retighten the bound";
    previous = rms;
  }
  EXPECT_LT(previous, 1.0) << "the 1:8 window residual regressed beyond its measured 0.83 rad";
}

// Every kernel in the stage is one element per thread with no grid-stride
// loop, so a launch grid that does not cover the image silently leaves its tail
// unwritten -- and the tail of a freshly pooled output is zeros. This image is
// deliberately just past 65535 blocks of 256 threads, the clamp that used to be
// applied: on a 2478 x 8000 auxiliary it dropped the last 380 range rows out of
// image_out and hid them from the peak finder too. min_scatterers forces the
// pass-through, which exercises the same copy without paying for a formation.
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
  for (int64_t r = 0; r < kTall; ++r) {
    double row = 0.0;
    for (int64_t c = 0; c < kWide; ++c) {
      row += std::abs(got[r * kWide + c]);
    }
    if (row == 0.0) {
      ++zero_rows;
    }
  }
  EXPECT_EQ(zero_rows, 0) << "image_out left " << zero_rows << " of " << kTall
                          << " rows unwritten; the launch grid does not cover the image";
}

TEST(AutofocusOp, AzimuthAxisV) {
  const auto phi = injected_phi();
  const auto img = make_image<float>(phi, kStrong, false);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kV, img, {kAz, kRg});
  ASSERT_TRUE(out.ok) << out.failure;
  ASSERT_EQ(out.phase.size(), kAz);
  auto est = phase_as_double(out.phase);
  auto want = phi;
  remove_affine(&est);
  remove_affine(&want);
  rd_standin_flip(&est);
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
}

TEST(AutofocusOp, MaxCorrectionPassesThrough) {
  const auto phi = injected_phi();
  const auto img = make_image<float>(phi, kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz}, 4, 0.5, 16, 0.05);
  ASSERT_TRUE(out.ok) << out.failure;
  EXPECT_LT(peak_abs(phase_as_double(out.phase)), 1e-6);
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
  for (size_t i = 1; i < sharp.size(); ++i) {
    EXPECT_GE(sharp[i], 0.98 * sharp[i - 1]);
  }
}

TEST(AutofocusOp, AzimuthAxisRequired) {
  const auto img = make_image<float>(injected_phi(), kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kUnspecified, img, {kRg, kAz});
  ASSERT_FALSE(out.ok);
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
  ASSERT_FALSE(out.ok);
  EXPECT_NE(out.failure.find("num_v"), std::string::npos) << out.failure;
}

TEST(AutofocusOp, RejectsIterationsBeyondPoolBound) {
  const auto img = make_image<float>(injected_phi(), kStrong, true);
  const auto out = run_af<float>(AutofocusAzimuthAxis::kU, img, {kRg, kAz},
                                 holoscan::radar::kMaxAutofocusIterations + 1);
  ASSERT_FALSE(out.ok);
  EXPECT_NE(out.failure.find("num_iterations"), std::string::npos) << out.failure;
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
  fixture.capture_snapshot<holoscan::Tensor>("sharpness", 2U, 8U * sizeof(float));
  fixture.input_source_device("image", holoscan::DeviceId{0});
  fixture.output_device("image_out", holoscan::DeviceId{0});
  fixture.output_device("phase_correction", holoscan::DeviceId{0});
  fixture.output_device("sharpness", holoscan::DeviceId{0});
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

  const auto reference = run_af<float>(AutofocusAzimuthAxis::kU, img, shape, 4, 0.5);
  ASSERT_TRUE(reference.ok) << reference.failure;
  EXPECT_LT(phase_rms(fell_back, phase_as_double(reference.phase)), 1e-6);
}
}  // namespace
