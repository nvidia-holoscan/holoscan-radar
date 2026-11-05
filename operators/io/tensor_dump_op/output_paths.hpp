// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <string>
#include <system_error>

namespace holoscan::radar::dump_paths {

// Offer a host command only for outputs inside a development checkout. The
// relative path survives the CLI's host/container repository mount mapping.
// Installed packages and files outside that mount retain the normal save log.
inline std::filesystem::path checkout_relative_path(const std::string& path) {
  std::error_code ec;
  auto root = std::filesystem::current_path(ec);
  if (ec)
    return {};
  const auto absolute = std::filesystem::absolute(path, ec).lexically_normal();
  if (ec)
    return {};
  constexpr const char* viewer = "tools/sar_data/view_image.py";
  for (;;) {
    if (std::filesystem::is_regular_file(root / viewer, ec))
      break;
    const auto parent = root.parent_path();
    if (parent == root || parent.empty())
      return {};
    root = parent;
  }
  const auto relative = absolute.lexically_relative(root);
  if (relative.empty() || *relative.begin() == "..")
    return {};
  return relative;
}

inline std::filesystem::path display_path(const std::string& path) {
  const auto relative = checkout_relative_path(path);
  return relative.empty() ? std::filesystem::absolute(path).lexically_normal() : relative;
}

inline std::string quote_path(const std::filesystem::path& path) {
  std::string quoted = "'";
  for (const char c : path.generic_string()) {
    if (c == '\'')
      quoted += "'\\''";
    else
      quoted.push_back(c);
  }
  return quoted + "'";
}

inline std::string image_view_command(const std::string& path) {
  const auto relative = checkout_relative_path(path);
  if (relative.empty())
    return {};
  return std::string("uv run tools/sar_data/view_image.py ") + quote_path(relative);
}

}  // namespace holoscan::radar::dump_paths
