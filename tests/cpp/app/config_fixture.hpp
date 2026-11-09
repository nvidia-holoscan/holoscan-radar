// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>  // NOLINT(build/c++17)
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <holoscan/config/io.hpp>

#include "radar_app/fmcw/config_loader.hpp"

namespace holoscan::radar::testing {

class ConfigDirectory {
 public:
  ConfigDirectory()
  : root_(std::filesystem::temp_directory_path() /
          ("radar_config_" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    std::filesystem::create_directories(root_);
  }
  ~ConfigDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }
  ConfigDirectory(const ConfigDirectory&) = delete;
  ConfigDirectory& operator=(const ConfigDirectory&) = delete;

  std::filesystem::path write(const std::string& name, const std::string& text) const {
    const auto path = root_ / name;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << text;
    if (!output)
      throw std::runtime_error("cannot write test config");
    return path;
  }

 private:
  std::filesystem::path root_;
};

class PresetOverride {
 public:
  PresetOverride(const std::filesystem::path& base, const std::string& patch)
  : path_(directory_.write(
        "override.yaml", "extends: " + std::filesystem::absolute(base).string() + "\n" + patch)) {}
  const std::filesystem::path& path() const { return path_; }

 private:
  ConfigDirectory directory_;
  std::filesystem::path path_;
};

// The document with every `extends:` applied.
inline holoscan::config::Document resolved_document(const std::filesystem::path& path) {
  return app::config::load_inherited_document(path);
}

// `document` with `value` at `path`, adding the key when it is absent.
inline holoscan::config::Document with_value(const holoscan::config::Document& document,
                                             std::string_view path,
                                             holoscan::config::ConfigValue value) {
  using holoscan::config::Edit;
  return document.patched(document.find(path) ? Edit::replace(path, std::move(value))
                                              : Edit::add(path, std::move(value)));
}

// Order-independent text for comparing two documents.
inline std::string canonical(const holoscan::config::Document& document) {
  return holoscan::config_io::serialize(document, holoscan::config_io::Format::kJson);
}

// Flow-style YAML for one value, for building preset override patches.
inline std::string flow_yaml(holoscan::config::ConfigView value) {
  using Kind = holoscan::config::ConfigValueKind;
  switch (value.kind()) {
    case Kind::kNull:
      return "null";
    case Kind::kBool:
      return value.as_bool() ? "true" : "false";
    case Kind::kSignedInteger:
      return std::to_string(value.as_signed_integer());
    case Kind::kUnsignedInteger:
      return std::to_string(value.as_unsigned_integer());
    case Kind::kFloating: {
      std::string text = std::format("{}", value.as_floating());
      if (text.find_first_of(".en") == std::string::npos)
        text += ".0";
      return text;
    }
    case Kind::kString: {
      std::string text = "\"";
      for (const char c : value.as_string()) {
        if (c == '"' || c == '\\')
          text += '\\';
        text += c;
      }
      return text + "\"";
    }
    case Kind::kArray: {
      std::string text = "[";
      for (std::size_t i = 0; i < value.size(); ++i)
        text += (i ? ", " : "") + flow_yaml(value.element(i));
      return text + "]";
    }
    case Kind::kObject: {
      std::string text = "{";
      for (std::size_t i = 0; i < value.size(); ++i)
        text += (i ? ", " : "") + std::string(value.key(i)) + ": " + flow_yaml(value.element(i));
      return text + "}";
    }
  }
  throw std::runtime_error("invalid config value");
}

}  // namespace holoscan::radar::testing
