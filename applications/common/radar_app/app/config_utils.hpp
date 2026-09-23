// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/config/io.hpp>
#include <holoscan/core/parameter_input.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/error.hpp>

#include "radar_common/math/uniform_grid.hpp"

namespace holoscan::radar::app::config {

template <typename T>
inline void require_positive(std::string_view key, T value) {
  if (value <= 0) {
    throw std::runtime_error(std::string(key) + " must be > 0, got " + std::to_string(value));
  }
}

inline void require_finite_positive(std::string_view key, double value) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(std::string(key) + " must be a positive finite value");
  }
}

inline void validate_frame_period(double seconds) {
  if (!std::isfinite(seconds)) {
    throw std::runtime_error("app.run.frame_period must be finite");
  }
  const double nanoseconds = seconds * 1e9;
  if (nanoseconds < 1.0 || nanoseconds > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error("app.run.frame_period must be within [1 ns, INT64_MAX ns], got " +
                             std::to_string(seconds) + " s");
  }
}

template <typename Params>
inline holoscan::PreparedParams<Params> decode_params(const holoscan::config::Document& document,
                                                      std::string_view path,
                                                      Params baseline = Params{}) {
  auto decoded = holoscan::params::try_decode<Params>(document, path, std::move(baseline));
  if (!decoded) {
    throw std::runtime_error(std::string(path) + ": " +
                             holoscan::params::format_error(decoded.error()));
  }
  return std::move(*decoded);
}

inline void reject_authored(const holoscan::config::Document& document, std::string_view path) {
  auto value = document.try_find(path);
  if (!value) {
    throw std::runtime_error(holoscan::config::format_error(value.error()));
  }
  if (value->has_value()) {
    throw std::runtime_error(std::string(path) + " is app-derived and must not be authored");
  }
}

template <typename Params>
inline holoscan::config::Document inject_derived_params(holoscan::config::Document document,
                                                        std::string_view path, Params params) {
  auto replaced = document.try_replace(path, std::move(params));
  if (!replaced) {
    throw std::runtime_error(std::string(path) + ": cannot inject app-derived Params: " +
                             holoscan::config::format_error(replaced.error()));
  }
  return std::move(*replaced);
}

inline holoscan::config::Document load_document(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  auto loaded = holoscan::config_io::try_load(path, holoscan::config_io::Format::kYaml);
  if (!loaded) {
    throw std::runtime_error(holoscan::config::format_error(loaded.error()));
  }
  return std::move(*loaded);
}

template <typename Enum>
inline holoscan::config::Document normalize_enum_field(holoscan::config::Document document,
                                                       std::string_view path) {
  auto found = document.try_find(path);
  if (!found) {
    throw std::runtime_error(holoscan::config::format_error(found.error()));
  }
  if (!found->has_value()) {
    return document;
  }
  auto decoded = document.try_get<Enum>(path);
  if (!decoded) {
    throw std::runtime_error(holoscan::config::format_error(decoded.error()));
  }
  using Underlying = std::underlying_type_t<Enum>;
  auto replaced = document.try_replace(path, static_cast<Underlying>(*decoded));
  if (!replaced) {
    throw std::runtime_error(holoscan::config::format_error(replaced.error()));
  }
  return std::move(*replaced);
}

inline holoscan::config::Document normalize_scan_coordinate_field(
    holoscan::config::Document document, std::string_view path) {
  auto found = document.try_find(path);
  if (!found) {
    throw std::runtime_error(holoscan::config::format_error(found.error()));
  }
  if (!found->has_value() || found->value().kind() == holoscan::config::ConfigValueKind::kArray) {
    return document;
  }
  const auto view = found->value();
  if (view.kind() != holoscan::config::ConfigValueKind::kObject || view.size() != 3U) {
    throw std::runtime_error(std::string(path) +
                             " must be an explicit array or a {min, max, step} mapping");
  }
  const auto decode_member = [&](std::string_view member) {
    const std::string member_path = std::string(path) + "/" + std::string(member);
    auto decoded = document.try_get<double>(member_path);
    if (!decoded) {
      throw std::runtime_error(holoscan::config::format_error(decoded.error()));
    }
    return *decoded;
  };

  std::vector<double> grid;
  try {
    grid =
        make_uniform_grid(decode_member("min"), decode_member("max"), decode_member("step"), path);
  } catch (const std::invalid_argument& error) {
    throw std::runtime_error(error.what());
  }
  auto replaced = document.try_replace(path, std::move(grid));
  if (!replaced) {
    throw std::runtime_error(holoscan::config::format_error(replaced.error()));
  }
  return std::move(*replaced);
}

}  // namespace holoscan::radar::app::config
