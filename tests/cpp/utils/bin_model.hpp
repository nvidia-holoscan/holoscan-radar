// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Forward bin model: expected bin indices and
// physical axes for a given processing configuration. Tests primarily
// assert in the physical domain via each operator's own range_bins /
// velocity_bins outputs; this helper exists where an exact-bin assertion
// is wanted, and as the independent reference those axis outputs are
// checked against.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace holoscan::radar::testing {

// ---- Range axis -----------------------------------------------------------

// Beat frequency (Hz) of a target at `range_m` for sweep slope (Hz/s) and
// propagation speed c: f_b = slope * 2R / c.
inline double beat_frequency(double range_m, double slope, double c) {
  return slope * 2.0 * range_m / c;
}

// Range represented by an unshifted range-FFT bin (bin 0 = zero range):
// range(i) = c * (i * fs / nfft) / (2 * slope). This mirrors the locked
// definition of the operators' range_bins output.
inline double range_of_bin(int64_t bin, double slope, double sample_rate,
                           int64_t nfft, double c) {
  const double f = static_cast<double>(bin) * sample_rate /
                   static_cast<double>(nfft);
  return c * f / (2.0 * slope);
}

// The (rounded) bin a target at `range_m` lands in.
inline int64_t expected_range_bin(double range_m, double slope,
                                  double sample_rate, int64_t nfft,
                                  double c) {
  const double f_b = beat_frequency(range_m, slope, c);
  return static_cast<int64_t>(std::llround(f_b * static_cast<double>(nfft) /
                                           sample_rate));
}

// The full expected range axis for num_bins bins (complex IF → num_bins =
// nfft; real IF → num_bins = nfft/2 + 1; both use the same per-bin rule).
inline std::vector<double> expected_range_axis(int64_t num_bins, double slope,
                                               double sample_rate,
                                               int64_t nfft, double c) {
  std::vector<double> axis(static_cast<size_t>(num_bins));
  for (int64_t i = 0; i < num_bins; ++i) {
    axis[static_cast<size_t>(i)] = range_of_bin(i, slope, sample_rate, nfft,
                                                c);
  }
  return axis;
}

// ---- Doppler / velocity axis ----------------------------------------------

// Doppler frequency (Hz) of radial velocity v_r (positive = receding):
// f_d = -2 * v_r / lambda (the contract's v = -f_d * lambda / 2 inverted).
inline double doppler_frequency(double radial_velocity, double wavelength) {
  return -2.0 * radial_velocity / wavelength;
}

// The Doppler-FFT bin index (0..num_doppler-1) a Doppler frequency lands
// in, on the unshifted axis (DC at bin 0), folding aliases into range.
inline int64_t unshifted_doppler_bin(double f_d, double pri,
                                     int64_t num_doppler) {
  const double n = static_cast<double>(num_doppler);
  double bin = std::round(f_d * pri * n);
  bin = std::fmod(bin, n);
  if (bin < 0.0) {
    bin += n;
  }
  return static_cast<int64_t>(bin);
}

// Apply numpy.fft.fftshift index mapping: shifted[i] = unshifted[(i + n/2)
// mod n] — i.e. an unshifted bin u appears at (u + floor(n/2)) mod n... the
// inverse: position of unshifted bin u after fftshift.
inline int64_t fftshift_position(int64_t unshifted_bin, int64_t n) {
  return (unshifted_bin + n / 2) % n;
}

// The bin a target at radial velocity v_r lands in, matching the
// operators' center_zero_doppler handling.
inline int64_t expected_doppler_bin(double radial_velocity,
                                    double wavelength, double pri,
                                    int64_t num_doppler,
                                    bool center_zero_doppler) {
  const double f_d = doppler_frequency(radial_velocity, wavelength);
  const int64_t unshifted = unshifted_doppler_bin(f_d, pri, num_doppler);
  return center_zero_doppler ? fftshift_position(unshifted, num_doppler)
                             : unshifted;
}

// The expected velocity axis (m/s per Doppler bin), mirroring the locked
// definition of the operators' velocity_bins output: per-bin Doppler
// frequency on the (possibly centered) axis, then v = -f_d * lambda / 2.
inline std::vector<double> expected_velocity_axis(int64_t num_doppler,
                                                  double wavelength,
                                                  double pri,
                                                  bool center_zero_doppler) {
  std::vector<double> axis(static_cast<size_t>(num_doppler));
  const double prf = 1.0 / pri;
  for (int64_t i = 0; i < num_doppler; ++i) {
    // Unshifted axis: bins [0, n) span [0, prf) with wraparound; the
    // signed frequency of bin i is i/n*prf folded into [-prf/2, prf/2).
    int64_t idx = i;
    if (center_zero_doppler) {
      // Position i on the shifted axis corresponds to unshifted bin
      // (i - n/2) mod n.
      idx = (i - num_doppler / 2 + num_doppler) % num_doppler;
    }
    double f = static_cast<double>(idx) / static_cast<double>(num_doppler) *
               prf;
    if (f >= prf / 2.0) {
      f -= prf;
    }
    axis[static_cast<size_t>(i)] = -f * wavelength / 2.0;
  }
  return axis;
}

}  // namespace holoscan::radar::testing
