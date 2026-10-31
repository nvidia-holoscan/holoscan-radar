// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Phase-vector helpers shared by the AutofocusOp unit tests
// (`sar/test_autofocus.cpp`) and the Figure 2 graph tests
// (`integration/test_sar_autofocus_graph.cpp`).
//
// These live in one place because the two suites previously carried their own
// copies and `to_bin_order` drifted between them: the integration copy still
// inverted a bare fftshift after the operator gained an axis reversal on
// 2026-10-30, and because the injected truth was even about the dwell centre
// the mismatch was invisible. A whole suite agreed with a stale convention.
// The same argument the CMake file already makes for the app's azimuth
// resampler applies here -- asserting against a copy is not asserting.

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace holoscan::radar::sar_test {

inline constexpr double kPi = 3.14159265358979323846;

// Deliberately the naive O(n^2) transform. It is the reference the injected
// truth is built from, so being obviously correct matters more here than being
// fast, and at the extents these suites use it is not the bottleneck.
inline void host_dft(std::vector<std::complex<double>>* x, bool inverse) {
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

// A range-Doppler stand-in: point scatterers in `n_strong` rows, clutter in the
// rest, with `phi` injected on the azimuth spectrum of every row -- one
// scene-common error, which is the space-invariance PGA assumes.
template <typename T>
std::vector<std::complex<T>> make_image(const std::vector<double>& phi, int64_t n_strong,
                                        bool az_is_u, int64_t n_az, int64_t n_rg) {
  const int64_t n_v = az_is_u ? n_rg : n_az;
  const int64_t n_u = az_is_u ? n_az : n_rg;
  std::vector<std::complex<T>> img(static_cast<size_t>(n_v * n_u),
                                   {static_cast<T>(0), static_cast<T>(0)});
  for (int64_t r = 0; r < n_rg; ++r) {
    std::vector<std::complex<double>> row(static_cast<size_t>(n_az), {0.0, 0.0});
    if (r < n_strong) {
      row[static_cast<size_t>(n_az / 2)] = {1.0, 0.0};
    } else {
      for (int64_t a = 0; a < n_az; ++a) {
        row[static_cast<size_t>(a)] = {0.01, 0.0};
      }
    }
    host_dft(&row, false);
    for (int64_t k = 0; k < n_az; ++k) {
      const double p = phi.empty() ? 0.0 : phi[static_cast<size_t>(k)];
      row[static_cast<size_t>(k)] *= std::complex<double>(std::cos(p), std::sin(p));
    }
    host_dft(&row, true);
    for (int64_t a = 0; a < n_az; ++a) {
      const auto z = row[static_cast<size_t>(a)];
      const std::complex<T> c(static_cast<T>(z.real()), static_cast<T>(z.imag()));
      img[static_cast<size_t>(az_is_u ? (r * n_az + a) : (a * n_rg + r))] = c;
    }
  }
  return img;
}

// Inverse of the operator's bin-to-pulse mapping, so an emitted pulse-indexed
// estimate can be compared against a bin-indexed injection. The operator emits
// pulse p from bin (n - 1 - p + n/2) mod n -- an fftshift *and* an axis
// reversal, azimuth being the Fourier conjugate of slow time -- so bin k came
// from pulse (n - 1 + n/2 - k) mod n.
inline std::vector<double> to_bin_order(const std::vector<double>& phi) {
  const int64_t n = static_cast<int64_t>(phi.size());
  std::vector<double> out(static_cast<size_t>(n));
  for (int64_t k = 0; k < n; ++k) {
    const int64_t p = ((n - 1 + (n + 1) / 2 - k) % n + n) % n;
    out[static_cast<size_t>(k)] = phi[static_cast<size_t>(p)];
  }
  return out;
}

// PGA is blind to a constant and a linear ramp: they shift the image rather
// than blur it, so only the residual is recoverable and comparisons strip both.
inline void remove_affine(std::vector<double>* phi) {
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

}  // namespace holoscan::radar::sar_test
