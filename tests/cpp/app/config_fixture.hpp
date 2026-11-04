// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstddef>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

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

inline YAML::Node yaml_value(holoscan::config::ConfigView value) {
  using Kind = holoscan::config::ConfigValueKind;
  switch (value.kind()) {
    case Kind::kNull:
      return YAML::Node(YAML::NodeType::Null);
    case Kind::kBool:
      return YAML::Node(value.as_bool());
    case Kind::kSignedInteger:
      return YAML::Node(value.as_signed_integer());
    case Kind::kUnsignedInteger:
      return YAML::Node(value.as_unsigned_integer());
    case Kind::kFloating:
      return YAML::Node(value.as_floating());
    case Kind::kString:
      return YAML::Node(std::string(value.as_string()));
    case Kind::kArray: {
      YAML::Node node(YAML::NodeType::Sequence);
      for (std::size_t i = 0; i < value.size(); ++i)
        node.push_back(yaml_value(value.element(i)));
      return node;
    }
    case Kind::kObject: {
      YAML::Node node(YAML::NodeType::Map);
      for (std::size_t i = 0; i < value.size(); ++i)
        node[std::string(value.key(i))] = yaml_value(value.element(i));
      return node;
    }
  }
  throw std::runtime_error("invalid config value");
}

inline YAML::Node resolved_yaml(const std::filesystem::path& path) {
  return yaml_value(app::config::load_inherited_document(path).root());
}

}  // namespace holoscan::radar::testing
