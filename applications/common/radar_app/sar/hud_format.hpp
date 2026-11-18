// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>

namespace holoscan::radar::app::sar {

inline std::string format_seconds(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0)
    return "0ms";
  if (seconds < 1.0) {
    return std::to_string(static_cast<int>(std::llround(seconds * 1000.0))) + "ms";
  }
  const int scaled = static_cast<int>(std::llround(seconds * 10.0));
  return std::to_string(scaled / 10) + "." + std::to_string(scaled % 10) + "s";
}

inline std::string format_lag_line(double behind_s, double target_s = 0.0, double wall_s = 0.0) {
  if (!std::isfinite(behind_s))
    return {};
  const std::string clocks =
      "  " + format_seconds(target_s) + " target / " + format_seconds(wall_s) + " wall";
  if (std::abs(behind_s) < 0.05)
    return "on time" + clocks;
  if (behind_s > 0.0) {
    const int scaled = static_cast<int>(std::llround(behind_s * 10.0));
    return "+" + std::to_string(scaled / 10) + "." + std::to_string(scaled % 10) + "s behind" +
           clocks;
  }
  return format_seconds(-behind_s) + " ahead" + clocks;
}

inline std::string format_form_latency(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0)
    return {};
  return format_seconds(seconds) + " after last pulse";
}

inline std::string format_sci(double value) {
  if (!std::isfinite(value) || value <= 0.0)
    return "0";
  int exp = static_cast<int>(std::floor(std::log10(value)));
  double mantissa = value / std::pow(10.0, static_cast<double>(exp));
  int scaled = static_cast<int>(std::llround(mantissa * 100.0));
  if (scaled >= 1000) {
    scaled = 100;
    ++exp;
  }
  const int whole = scaled / 100;
  const int frac = scaled % 100;
  std::string out = std::to_string(whole) + ".";
  if (frac < 10)
    out += '0';
  out += std::to_string(frac);
  out += 'e';
  if (exp < 0) {
    out += '-';
    exp = -exp;
  }
  out += std::to_string(exp);
  return out;
}

// Display-only conversion of the report's rounded measurement. Preserve the
// console format and measured value; never resample a later BP diagnostic.
inline std::string format_display_rate_line(std::string_view line) {
  constexpr std::string_view suffix = " px*pls/s";
  if (!line.ends_with(suffix))
    return std::string(line);
  const auto end = line.size() - suffix.size();
  if (end == 0)
    return std::string(line);
  const auto start = line.rfind(' ', end - 1);
  if (start == std::string_view::npos)
    return std::string(line);
  double rate = 0;
  const auto parsed = std::from_chars(line.data() + start + 1, line.data() + end, rate);
  if (parsed.ec != std::errc() || parsed.ptr != line.data() + end || !std::isfinite(rate) ||
      rate < 0)
    return std::string(line);
  constexpr std::string_view prefixes[] = {"", "k", "M", "G", "T"};
  int unit = 0;
  while (rate >= 1000 && unit < 4) {
    rate /= 1000;
    ++unit;
  }
  // Round before selecting the final prefix, so 999.99 M never displays as 1000 M.
  rate = std::round(rate * 10) / 10;
  if (rate >= 1000 && unit < 4) {
    rate /= 1000;
    ++unit;
  }
  char digits[48]{};
  const auto formatted =
      std::to_chars(digits, digits + sizeof(digits), rate, std::chars_format::fixed, 1);
  if (formatted.ec != std::errc())
    return std::string(line);
  return std::string(line.substr(0, start + 1)) + std::string(digits, formatted.ptr) + " " +
         std::string(prefixes[unit]) + "px*pulse/s";
}

// Same rate line the on-image HUD draws. Prefix the BP lattice when it
// differs from the displayed tensor (a mosaic/parent strip).
inline std::string format_bp_rate_line(double compute_s, int64_t pulses, int64_t pixels,
                                       int64_t bp_num_v, int64_t bp_num_u, int64_t display_v,
                                       int64_t display_u) {
  if (!(compute_s > 0.0) || pulses <= 0 || pixels <= 0)
    return {};
  const double rate = static_cast<double>(pixels) * static_cast<double>(pulses) / compute_s;
  std::string line = format_seconds(compute_s) + "  " + format_sci(rate) + " px*pls/s";
  if (bp_num_v > 0 && bp_num_u > 0 &&
      (display_v <= 0 || display_u <= 0 || bp_num_v != display_v || bp_num_u != display_u)) {
    line = std::to_string(bp_num_v) + "x" + std::to_string(bp_num_u) + "  " + line;
  }
  return line;
}

}  // namespace holoscan::radar::app::sar
