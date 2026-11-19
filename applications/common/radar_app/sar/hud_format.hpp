// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#include "radar_common/sar/formation_hud.hpp"

namespace holoscan::radar::app::sar {

inline std::string format_autofocus_status(const radar::AutofocusHudStatus& status,
                                           bool image_ready, bool input_complete) {
  using Outcome = radar::AutofocusHudOutcome;
  switch (status.outcome) {
    case Outcome::kDisabled:
      return {};
    case Outcome::kPending:
      return input_complete ? "Autofocus running" : "Preparing autofocus";
    case Outcome::kAccepted:
      return image_ready ? "Correction accepted" : "AF accepted; forming image";
    case Outcome::kDeclined:
      return image_ready ? "Correction declined" : "AF declined; forming image";
    case Outcome::kUnchanged:
      return "No correction applied";
    case Outcome::kOracle:
      return "Oracle correction";
    case Outcome::kUnavailable:
      return "AF status unavailable";
  }
  return "AF status unavailable";
}

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
  return "Final-batch latency " + format_seconds(seconds);
}

// The image card uses one fixed giga-unit and the same accumulated GPU work
// time for both metrics. Display dimensions may describe a decimated mosaic;
// callers supply the measured BP pixel/pulse counts, not that display extent.
inline std::string format_bp_giga_work(int64_t pulses, int64_t pixels) {
  if (pulses <= 0 || pixels <= 0)
    return "--";
  const double work = static_cast<double>(pixels) * static_cast<double>(pulses) / 1e9;
  char digits[48]{};
  const auto formatted =
      std::to_chars(digits, digits + sizeof(digits), work, std::chars_format::fixed, 1);
  return formatted.ec == std::errc() ? std::string(digits, formatted.ptr) : "--";
}

inline std::string format_bp_giga_throughput(double compute_s, int64_t pulses, int64_t pixels) {
  if (!std::isfinite(compute_s) || compute_s <= 0 || pulses <= 0 || pixels <= 0)
    return "--";
  const double rate = static_cast<double>(pixels) * static_cast<double>(pulses) / compute_s / 1e9;
  if (!std::isfinite(rate))
    return "--";
  char digits[48]{};
  const auto formatted =
      std::to_chars(digits, digits + sizeof(digits), rate, std::chars_format::fixed, 1);
  return formatted.ec == std::errc() ? std::string(digits, formatted.ptr) : "--";
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

// The source interval stays in original radar time. Scale only this display
// budget, using the rate captured with the image rather than the live reader.
inline std::string format_playback_budget(std::optional<double> radar_s,
                                          std::optional<double> rate) {
  if (!rate || !std::isfinite(*rate) || *rate < 0)
    return "-- @--";
  if (*rate == 0)
    return "-- unpaced";
  char digits[48]{};
  const auto formatted =
      std::to_chars(digits, digits + sizeof(digits), *rate, std::chars_format::general);
  if (formatted.ec != std::errc())
    return "-- @--";
  std::string budget = "--";
  if (radar_s && std::isfinite(*radar_s) && *radar_s > 0) {
    const double paced_s = *radar_s / *rate;
    if (std::isfinite(paced_s) && paced_s > 0) {
      // Very slow playback can exceed format_seconds' integer tick range.
      budget = paced_s < 1e6 ? format_seconds(paced_s) : format_sci(paced_s) + "s";
    }
  }
  return budget + " @" + std::string(digits, formatted.ptr) + "x";
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
