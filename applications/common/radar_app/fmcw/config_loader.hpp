// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/config/document.hpp>

#include "radar_app/app/config_utils.hpp"

namespace holoscan::radar::app::config {
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
    auto document = load_document(path);
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

// FMCW-only opt-in: SAR retains the ordinary single-document loader.
inline holoscan::config::Document load_inherited_document(const std::filesystem::path& path) {
  return detail::load_inherited_document_impl(path, {});
}

}  // namespace holoscan::radar::app::config
