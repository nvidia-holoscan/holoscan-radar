// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <cstdint>

#include <array>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <initializer_list>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace holoscan::radar::perf {

inline constexpr std::array<std::string_view, 5> kScopes = {
    "minimal", "limited", "overrides", "moderate",
    "thorough"};  // NOLINT(whitespace/indent_namespace)
inline constexpr std::array<std::string_view, 3> kProfiles = {"small", "medium", "large"};

inline bool known_scope(std::string_view scope) {
  for (const auto known : kScopes) {
    if (scope == known)
      return true;
  }
  return false;
}

inline bool known_profile(std::string_view profile) {
  for (const auto known : kProfiles) {
    if (profile == known)
      return true;
  }
  return false;
}

inline void require_keys(const YAML::Node& map, std::initializer_list<std::string_view> allowed,
                         std::string_view where) {
  if (!map.IsMap())
    throw std::invalid_argument(std::string(where) + " must be a map");
  std::set<std::string> seen;
  for (const auto& entry : map) {
    const auto key = entry.first.as<std::string>();
    if (!seen.insert(key).second)
      throw std::invalid_argument(std::string(where) + ": duplicate key " + key);
    bool found = false;
    for (const auto name : allowed)
      found |= key == name;
    if (!found)
      throw std::invalid_argument(std::string(where) + ": unknown key " + key);
  }
}

struct ProfileCase {
  std::string id;
  std::string profile;
  std::vector<std::string> scopes;
  YAML::Node parameters;
  std::optional<YAML::Node> override_value;
};

struct ProfileSuite {
  std::array<std::vector<int64_t>, 3> shapes;
  std::vector<std::string> live_ports;
  std::vector<ProfileCase> cases;

  [[nodiscard]] std::vector<ProfileCase> select(std::string_view scope) const {
    if (!known_scope(scope))
      throw std::invalid_argument("unknown performance scope");
    std::vector<ProfileCase> result;
    for (const auto& c : cases) {
      bool include = false;
      for (const auto& tag : c.scopes) {
        include |= tag == "minimal" || tag == scope || (tag == "limited" && scope != "minimal") ||
                   (tag == "overrides" && scope == "thorough") ||
                   (tag == "moderate" && scope == "thorough");
      }
      if (include)
        result.push_back(c);
    }
    return result;
  }
};

inline ProfileSuite load_profile_suite(const std::filesystem::path& path,
                                       const std::string& expected_operator) {
  const YAML::Node root = YAML::LoadFile(path.string());
  require_keys(root, {"version", "operator", "profiles", "live_ports", "cases"}, "suite");
  for (const char* key : {"version", "operator", "profiles", "live_ports", "cases"}) {
    if (!root[key])
      throw std::invalid_argument("suite missing required key: " + std::string(key));
  }
  if (root["version"].as<int>() != 1 || root["operator"].as<std::string>() != expected_operator) {
    throw std::invalid_argument("profile suite version or operator mismatch: " + path.string());
  }
  ProfileSuite result;
  const auto profiles = root["profiles"];
  require_keys(profiles, {"small", "medium", "large"}, "profiles");
  for (size_t i = 0; i < kProfiles.size(); ++i) {
    const YAML::Node profile = profiles[std::string(kProfiles[i])];
    require_keys(profile, {"shape"}, "profile");
    if (!profile["shape"])
      throw std::invalid_argument("profile missing shape");
    const YAML::Node shape = profile["shape"];
    if (!shape.IsSequence() || shape.size() == 0) {
      throw std::invalid_argument("profile shape must be a nonempty sequence");
    }
    for (const auto& extent : shape) {
      const int64_t value = extent.as<int64_t>();
      if (value <= 0)
        throw std::invalid_argument("profile extents must be positive");
      result.shapes[i].push_back(value);
    }
  }
  const YAML::Node lives = root["live_ports"];
  if (!lives.IsSequence())
    throw std::invalid_argument("live_ports must be a sequence");
  std::set<std::string> live_set;
  for (const auto& value : lives) {
    const auto port = value.as<std::string>();
    if (port.empty())
      throw std::invalid_argument("live port cannot be empty");
    if (!live_set.insert(port).second)
      throw std::invalid_argument("duplicate live port: " + port);
    result.live_ports.push_back(port);
  }
  const YAML::Node cases = root["cases"];
  if (!cases.IsSequence())
    throw std::invalid_argument("cases must be a sequence");
  std::set<std::string> ids;
  for (const auto& item : cases) {
    require_keys(item, {"id", "profile", "scopes", "parameters", "override"}, "case");
    for (const char* key : {"id", "profile", "scopes", "parameters"}) {
      if (!item[key])
        throw std::invalid_argument("case missing required key: " + std::string(key));
    }
    ProfileCase c;
    c.id = item["id"].as<std::string>();
    if (c.id.empty())
      throw std::invalid_argument("case ID cannot be empty");
    c.profile = item["profile"].as<std::string>();
    if (!ids.insert(c.id).second)
      throw std::invalid_argument("duplicate case: " + c.id);
    if (!known_profile(c.profile))
      throw std::invalid_argument("unknown profile: " + c.profile);
    const YAML::Node scopes = item["scopes"];
    if (!scopes.IsSequence() || scopes.size() == 0) {
      throw std::invalid_argument("case scopes must be a nonempty sequence");
    }
    std::set<std::string> scope_set;
    for (const auto& scope : scopes) {
      const auto name = scope.as<std::string>();
      if (!known_scope(name))
        throw std::invalid_argument("unknown scope: " + name);
      if (!scope_set.insert(name).second)
        throw std::invalid_argument("duplicate scope tag: " + name);
      c.scopes.push_back(name);
    }
    c.parameters = item["parameters"];
    if (!c.parameters.IsMap())
      throw std::invalid_argument("case parameters must be a map");
    if (item["override"]) {
      c.override_value = item["override"];
      require_keys(*c.override_value, {"port", "value", "cache"}, "override");
      if ((*c.override_value)["cache"]) {
        const auto cache = (*c.override_value)["cache"].as<std::string>();
        if (cache != "unseen" && cache != "cached")
          throw std::invalid_argument("override cache must be unseen or cached");
      }
      if (!(*c.override_value)["port"] || !(*c.override_value)["value"])
        throw std::invalid_argument("override requires port and value");
      const auto port = (*c.override_value)["port"].as<std::string>();
      if (!live_set.contains(port))
        throw std::invalid_argument("unknown live port: " + port);
    }
    result.cases.push_back(std::move(c));
  }
  for (const auto profile : kProfiles) {
    bool baseline = false;
    for (const auto& c : result.select("minimal")) {
      baseline |= c.profile == profile && !c.override_value.has_value();
    }
    if (!baseline)
      throw std::invalid_argument("minimal lacks baseline for " + std::string(profile));
    for (const auto& port : result.live_ports) {
      bool covered = false;
      for (const auto& c : result.select("overrides")) {
        covered |= c.profile == profile && c.override_value.has_value() &&
                   (*c.override_value)["port"].as<std::string>() == port;
      }
      if (!covered) {
        throw std::invalid_argument("overrides lacks " + port + " at " + std::string(profile));
      }
    }
  }
  return result;
}

}  // namespace holoscan::radar::perf
