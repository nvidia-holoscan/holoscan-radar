// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <string_view>

#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include <holoscan/config/document.hpp>

#include "radar_app/app/config_utils.hpp"

namespace holoscan::radar::app::config {

// Normalize each layer before inheritance: a child's legacy alias must beat a
// parent's common spelling, while a common spelling in the same file wins.
inline holoscan::config::Document normalize_visualization_aliases(
    holoscan::config::Document document) {
  using holoscan::config::ConfigValue;
  using holoscan::config::Edit;
  constexpr std::array<std::pair<std::string_view, std::string_view>, 5> aliases{
      {{"/visualize", "enabled"},
       {"/headless", "headless"},
       {"/skip_display_check", "skip_display_check"},
       {"/visualize_size", "max_side"},
       {"/visualize_hold_s", "final_image_hold_s"}}};
  for (const auto& [alias, field] : aliases) {
    const auto value = document.find(alias);
    if (!value)
      continue;
    if (!document.find("/app"))
      document = document.patched(Edit::add("/app", ConfigValue::object({})));
    if (!document.find("/app/visualization"))
      document = document.patched(Edit::add("/app/visualization", ConfigValue::object({})));
    const std::string target = "/app/visualization/" + std::string(field);
    if (!document.find(target)) {
      document = document.patched(Edit::add(target, value->to_value()));
    } else if (holoscan::config::Document::from_value(document.find(target)->to_value())
                   .digest()
                   .semantic !=
               holoscan::config::Document::from_value(value->to_value()).digest().semantic) {
      HOLOSCAN_LOG_WARN("GUI: {} overrides legacy {} in the same configuration", target, alias);
    }
  }
  return document;
}

namespace detail {

inline holoscan::config::Document load_inherited_document_impl(
    const std::filesystem::path& path, std::vector<std::filesystem::path> chain) {
  try {
    const auto identity = std::filesystem::weakly_canonical(path);
    if (std::find(chain.begin(), chain.end(), identity) != chain.end()) {
      throw std::runtime_error("extends cycle");
    }
    if (chain.size() >= 32U) {
      throw std::runtime_error("extends chain exceeds 32 files");
    }
    chain.push_back(identity);
    auto document = normalize_visualization_aliases(load_document(path));
    if (document.root().kind() != holoscan::config::ConfigValueKind::kObject) {
      throw std::runtime_error("configuration root must be a mapping");
    }
    const auto base = document.find("/extends");
    if (!base) {
      return document;
    }
    if (base->kind() != holoscan::config::ConfigValueKind::kString) {
      throw std::runtime_error("extends must name one nonempty file");
    }
    const auto name = document.get<std::string>("/extends");
    if (name.empty()) {
      throw std::runtime_error("extends must name one nonempty file");
    }
    auto inherited = load_inherited_document_impl(path.parent_path() / name, std::move(chain));
    document = document.patched(holoscan::config::Edit::remove("/extends"));
    return holoscan::config::Document::from_value(holoscan::config::merge_config_values(
        inherited.root().to_value(), document.root().to_value()));
  } catch (const std::exception& error) {
    throw std::runtime_error("config " + path.string() + ": " + error.what());
  }
}

}  // namespace detail

// Shared by FMCW and SAR, including operator parameter and sizing reads.
inline holoscan::config::Document load_inherited_document(const std::filesystem::path& path) {
  return detail::load_inherited_document_impl(path, {});
}

}  // namespace holoscan::radar::app::config
