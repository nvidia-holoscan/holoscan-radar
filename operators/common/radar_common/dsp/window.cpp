// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/window.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::radar {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Generalized cosine window: sum_i (-1)^i a[i] cos(2*pi*i*n / (N-1)).
std::vector<double> cosine_sum_window(int64_t n,
                                      const std::vector<double>& a) {
  std::vector<double> w(n, a[0]);
  if (n == 1) {
    // Single-sample window: the peak value, sum_i (-1)^i a[i].
    double v = 0.0;
    double sign = 1.0;
    for (double c : a) {
      v += sign * c;
      sign = -sign;
    }
    w[0] = v;
    return w;
  }
  for (int64_t k = 0; k < n; ++k) {
    double v = 0.0;
    double sign = 1.0;
    for (size_t i = 0; i < a.size(); ++i) {
      v += sign * a[i] *
           std::cos(2.0 * kPi * static_cast<double>(i) *
                    static_cast<double>(k) / static_cast<double>(n - 1));
      sign = -sign;
    }
    w[k] = v;
  }
  return w;
}

std::vector<double> tukey_window(int64_t n, double alpha) {
  if (alpha < 0.0 || alpha > 1.0) {
    throw std::invalid_argument("WindowSpec: tukey_alpha must be in [0, 1]");
  }
  std::vector<double> w(n, 1.0);
  if (n == 1 || alpha == 0.0) {
    return w;
  }
  const double taper = alpha * static_cast<double>(n - 1) / 2.0;
  for (int64_t k = 0; k < n; ++k) {
    const double x = static_cast<double>(k);
    const double m = static_cast<double>(n - 1);
    if (x < taper) {
      w[k] = 0.5 * (1.0 + std::cos(kPi * (x / taper - 1.0)));
    } else if (x > m - taper) {
      w[k] = 0.5 * (1.0 + std::cos(kPi * ((x - m + taper) / taper)));
    }
  }
  return w;
}

// Modified Bessel function of the first kind, order zero (power series).
double bessel_i0(double x) {
  const double half_x = x / 2.0;
  double term = 1.0;
  double sum = 1.0;
  for (int k = 1; k < 200; ++k) {
    term *= (half_x / static_cast<double>(k)) *
            (half_x / static_cast<double>(k));
    sum += term;
    if (term < 1e-18 * sum) {
      break;
    }
  }
  return sum;
}

// Kaiser beta from the target sidelobe attenuation (Kaiser's empirical
// formula; attenuation in positive dB).
double kaiser_beta_from_attenuation(double atten_db) {
  // !(x > 0) also rejects NaN, which every comparison below would pass.
  if (!(atten_db > 0.0)) {
    throw std::invalid_argument(
        "WindowSpec: sidelobe_attenuation_db must be > 0 for the kaiser "
        "window (when kaiser_beta is unset)");
  }
  if (atten_db > 50.0) {
    return 0.1102 * (atten_db - 8.7);
  }
  if (atten_db > 21.0) {
    return 0.5842 * std::pow(atten_db - 21.0, 0.4) +
           0.07886 * (atten_db - 21.0);
  }
  return 0.0;
}

std::vector<double> kaiser_window(int64_t n, double beta) {
  std::vector<double> w(n, 1.0);
  if (n == 1) {
    return w;
  }
  const double denom = bessel_i0(beta);
  const double m = static_cast<double>(n - 1);
  for (int64_t k = 0; k < n; ++k) {
    const double r = 2.0 * static_cast<double>(k) / m - 1.0;
    w[k] = bessel_i0(beta * std::sqrt(1.0 - r * r)) / denom;
  }
  return w;
}

// Dolph-Chebyshev window with the given sidelobe attenuation (positive dB),
// via the standard frequency-sampling construction (as in
// scipy.signal.windows.chebwin).
std::vector<double> chebyshev_window(int64_t n, double atten_db) {
  if (!(atten_db > 0.0)) {
    throw std::invalid_argument(
        "WindowSpec: sidelobe_attenuation_db must be > 0 for the chebyshev "
        "window");
  }
  if (n == 1) {
    return {1.0};
  }
  const double order = static_cast<double>(n - 1);
  const double ripple_ratio = std::pow(10.0, atten_db / 20.0);
  const double beta = std::cosh(std::acosh(ripple_ratio) / order);

  // Chebyshev polynomial T_order evaluated at beta*cos(pi*k/n).
  std::vector<double> p(n);
  for (int64_t k = 0; k < n; ++k) {
    const double x = beta * std::cos(kPi * static_cast<double>(k) /
                                     static_cast<double>(n));
    if (x > 1.0) {
      p[k] = std::cosh(order * std::acosh(x));
    } else if (x < -1.0) {
      const double c = std::cosh(order * std::acosh(-x));
      p[k] = ((n - 1) % 2 == 0) ? c : -c;
    } else {
      p[k] = std::cos(order * std::acos(x));
    }
  }

  // Symmetric inverse DFT with a (n-1)/2 sample shift: the imaginary parts
  // of the even-symmetric spectrum cancel, and the half-integer shift for
  // even n centers the peak between the two middle samples. Matches
  // scipy.signal.windows.chebwin for both parities.
  std::vector<double> w(n, 0.0);
  const double center = static_cast<double>(n - 1) / 2.0;
  for (int64_t k = 0; k < n; ++k) {
    double v = 0.0;
    for (int64_t j = 0; j < n; ++j) {
      v += p[j] * std::cos(2.0 * kPi * static_cast<double>(j) *
                           (static_cast<double>(k) - center) /
                           static_cast<double>(n));
    }
    w[k] = v;
  }

  // Normalize to unit peak.
  const double peak = *std::max_element(w.begin(), w.end());
  for (double& v : w) {
    v /= peak;
  }
  return w;
}

// Taylor window (as in scipy.signal.windows.taylor): nbar nearly-constant
// sidelobes at -atten_db dB.
std::vector<double> taylor_window(int64_t n, double atten_db, int nbar) {
  if (!(atten_db > 0.0)) {
    throw std::invalid_argument(
        "WindowSpec: sidelobe_attenuation_db must be > 0 for the taylor "
        "window");
  }
  if (nbar < 1) {
    throw std::invalid_argument("WindowSpec: taylor_nbar must be >= 1");
  }
  if (n == 1) {
    return {1.0};
  }
  const double b = std::pow(10.0, atten_db / 20.0);
  const double a = std::acosh(b) / kPi;
  const double s2 = static_cast<double>(nbar) * static_cast<double>(nbar) /
                    (a * a + (static_cast<double>(nbar) - 0.5) *
                                 (static_cast<double>(nbar) - 0.5));

  std::vector<double> fm(static_cast<size_t>(nbar) - 1, 0.0);
  for (int mi = 1; mi <= nbar - 1; ++mi) {
    const double m = static_cast<double>(mi);
    double numerator = 1.0;
    double denominator = 1.0;
    for (int i = 1; i <= nbar - 1; ++i) {
      const double di = static_cast<double>(i);
      numerator *= 1.0 - (m * m) / (s2 * (a * a + (di - 0.5) * (di - 0.5)));
      if (i != mi) {
        denominator *= 1.0 - (m * m) / (di * di);
      }
    }
    // F_m carries the (-1)^(m+1) alternation: positive for odd m.
    const double sign = (mi % 2 == 1) ? 1.0 : -1.0;
    fm[static_cast<size_t>(mi) - 1] = sign * numerator / (2.0 * denominator);
  }

  std::vector<double> w(n, 1.0);
  for (int64_t k = 0; k < n; ++k) {
    const double xi =
        (static_cast<double>(k) - static_cast<double>(n) / 2.0 + 0.5) /
        static_cast<double>(n);
    double v = 1.0;
    for (int mi = 1; mi <= nbar - 1; ++mi) {
      v += 2.0 * fm[static_cast<size_t>(mi) - 1] *
           std::cos(2.0 * kPi * static_cast<double>(mi) * xi);
    }
    w[k] = v;
  }

  // Normalize to unit peak (scipy's norm=True normalizes the center value).
  double peak = 0.0;
  for (double v : w) {
    peak = std::max(peak, std::abs(v));
  }
  for (double& v : w) {
    v /= peak;
  }
  return w;
}

}  // namespace

std::vector<double> generate_window(const WindowSpec& spec, int64_t length) {
  if (length < 1) {
    throw std::invalid_argument("generate_window: length must be >= 1");
  }

  std::vector<double> w;
  switch (spec.type) {
    case WindowType::kNone:
      w.assign(static_cast<size_t>(length), 1.0);
      break;
    case WindowType::kHann:
      w = cosine_sum_window(length, {0.5, 0.5});
      break;
    case WindowType::kHamming:
      w = cosine_sum_window(length, {0.54, 0.46});
      break;
    case WindowType::kBlackman:
      w = cosine_sum_window(length, {0.42, 0.5, 0.08});
      break;
    case WindowType::kBlackmanHarris:
      w = cosine_sum_window(length, {0.35875, 0.48829, 0.14128, 0.01168});
      break;
    case WindowType::kFlatTop:
      w = cosine_sum_window(length, {0.21557895, 0.41663158, 0.277263158,
                                     0.083578947, 0.006947368});
      break;
    case WindowType::kTukey:
      w = tukey_window(length, spec.tukey_alpha);
      break;
    case WindowType::kKaiser: {
      const double beta =
          spec.kaiser_beta.has_value()
              ? *spec.kaiser_beta
              : kaiser_beta_from_attenuation(spec.sidelobe_attenuation_db);
      w = kaiser_window(length, beta);
      break;
    }
    case WindowType::kChebyshev:
      w = chebyshev_window(length, spec.sidelobe_attenuation_db);
      break;
    case WindowType::kTaylor:
      w = taylor_window(length, spec.sidelobe_attenuation_db,
                        spec.taylor_nbar);
      break;
    case WindowType::kCustom:
      if (static_cast<int64_t>(spec.coefficients.size()) != length) {
        throw std::invalid_argument(
            "WindowSpec: custom coefficients length does not match the "
            "window length");
      }
      w = spec.coefficients;
      break;
  }

  if (spec.coherent_gain_normalization) {
    const double mean =
        std::accumulate(w.begin(), w.end(), 0.0) / static_cast<double>(length);
    if (mean == 0.0) {
      throw std::invalid_argument(
          "WindowSpec: window mean is zero; cannot apply coherent-gain "
          "normalization");
    }
    for (double& v : w) {
      v /= mean;
    }
  }
  return w;
}

std::string window_cache_key(const WindowSpec& spec, int64_t length) {
  std::ostringstream key;
  key.precision(17);
  key << to_string(spec.type) << ':' << length << ':'
      << (spec.coherent_gain_normalization ? 1 : 0);
  switch (spec.type) {
    case WindowType::kKaiser:
      key << ":beta=";
      if (spec.kaiser_beta.has_value()) {
        key << *spec.kaiser_beta;
      } else {
        key << "sll:" << spec.sidelobe_attenuation_db;
      }
      break;
    case WindowType::kChebyshev:
      key << ":sll=" << spec.sidelobe_attenuation_db;
      break;
    case WindowType::kTaylor:
      key << ":sll=" << spec.sidelobe_attenuation_db
          << ":nbar=" << spec.taylor_nbar;
      break;
    case WindowType::kTukey:
      key << ":alpha=" << spec.tukey_alpha;
      break;
    case WindowType::kCustom: {
      // Hash the coefficients (FNV-1a) so distinct custom windows get
      // distinct entries without embedding the full vector in the key.
      uint64_t h = 1469598103934665603ULL;
      for (double c : spec.coefficients) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(c));
        std::memcpy(&bits, &c, sizeof(bits));
        for (int b = 0; b < 8; ++b) {
          h ^= (bits >> (8 * b)) & 0xFFULL;
          h *= 1099511628211ULL;
        }
      }
      key << ":hash=" << h << ":len=" << spec.coefficients.size();
      break;
    }
    default:
      break;
  }
  return key.str();
}

}  // namespace holoscan::radar
