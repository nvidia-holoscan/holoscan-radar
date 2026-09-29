// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>

#include <array>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <initializer_list>
#include <map>
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
  YAML::Node fixture;
  std::optional<YAML::Node> override_value;
  int source_line{0};
};

struct ProfileSuite {
  int version{1};
  std::array<std::vector<int64_t>, 3> shapes;
  std::vector<std::string> axes;
  YAML::Node recipe;
  std::vector<std::string> live_ports;
  std::vector<ProfileCase> cases;

  [[nodiscard]] const std::vector<int64_t>& shape_for(std::string_view profile) const {
    for (size_t i = 0; i < kProfiles.size(); ++i)
      if (kProfiles[i] == profile)
        return shapes[i];
    throw std::invalid_argument("unknown profile: " + std::string(profile));
  }

  [[nodiscard]] std::map<std::string, int64_t> axes_for(std::string_view profile) const {
    const auto& shape = shape_for(profile);
    if (axes.size() != shape.size())
      throw std::invalid_argument("profile axes and shape have different lengths");
    std::map<std::string, int64_t> result;
    for (size_t i = 0; i < axes.size(); ++i)
      result.emplace(axes[i], shape[i]);
    return result;
  }

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
  require_keys(root, {"version", "operator", "axes", "recipe", "profiles", "live_ports", "cases"},
               "suite");
  for (const char* key : {"version", "operator", "profiles", "live_ports", "cases"}) {
    if (!root[key])
      throw std::invalid_argument("suite missing required key: " + std::string(key));
  }
  const int version = root["version"].as<int>();
  if ((version != 1 && version != 2) || root["operator"].as<std::string>() != expected_operator) {
    throw std::invalid_argument("profile suite version or operator mismatch: " + path.string());
  }
  ProfileSuite result;
  result.version = version;
  if (version == 2) {
    if (!root["axes"] || !root["recipe"])
      throw std::invalid_argument("version 2 suite requires axes and recipe");
    const auto axes = root["axes"];
    if (!axes.IsSequence() || axes.size() == 0)
      throw std::invalid_argument("axes must be a nonempty sequence");
    std::set<std::string> seen_axes;
    for (const auto& axis : axes) {
      const auto name = axis.as<std::string>();
      if (name.empty() || !seen_axes.insert(name).second)
        throw std::invalid_argument("empty or duplicate profile axis: " + name);
      result.axes.push_back(name);
    }
    result.recipe = root["recipe"];
    require_keys(result.recipe,
                 {"operator_type", "inputs", "constructor", "params", "outputs", "live", "fixtures",
                  "extension"},
                 "recipe");
    for (const char* key : {"operator_type", "inputs", "constructor", "params", "outputs"}) {
      if (!result.recipe[key])
        throw std::invalid_argument("recipe missing required key: " + std::string(key));
    }
  } else if (root["axes"] || root["recipe"]) {
    throw std::invalid_argument("version 1 suite cannot declare axes or recipe");
  }
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
    if (version == 2 && result.shapes[i].size() != result.axes.size())
      throw std::invalid_argument("profile shape rank differs from axes");
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
    if (version == 2) {
      require_keys(item,
                   {"id", "profile", "profiles", "scopes", "parameters", "fixture", "override"},
                   "case");
    } else {
      require_keys(item, {"id", "profile", "profiles", "scopes", "parameters", "override"}, "case");
    }
    for (const char* key : {"id", "scopes", "parameters"}) {
      if (!item[key])
        throw std::invalid_argument("case missing required key: " + std::string(key));
    }
    if (static_cast<bool>(item["profile"]) == static_cast<bool>(item["profiles"]))
      throw std::invalid_argument("case requires exactly one of profile or profiles");
    ProfileCase c;
    c.id = item["id"].as<std::string>();
    if (c.id.empty())
      throw std::invalid_argument("case ID cannot be empty");
    c.source_line = item.Mark().line + 1;
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
    if (item["fixture"]) {
      c.fixture = item["fixture"];
      if (!c.fixture.IsMap())
        throw std::invalid_argument("case fixture must be a map");
    }
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
    std::vector<std::string> profiles;
    if (item["profile"]) {
      profiles.push_back(item["profile"].as<std::string>());
      if (c.id.find("{profile}") != std::string::npos)
        throw std::invalid_argument("single case ID cannot contain {profile}: " + c.id);
    } else {
      const auto family = item["profiles"];
      if (!family.IsSequence() || family.size() == 0)
        throw std::invalid_argument("case profiles must be a nonempty sequence");
      const auto marker = c.id.find("{profile}");
      if (marker == std::string::npos || c.id.find("{profile}", marker + 9) != std::string::npos)
        throw std::invalid_argument("case family ID requires one {profile}: " + c.id);
      std::set<std::string> seen_profiles;
      for (const auto& profile : family) {
        const auto name = profile.as<std::string>();
        if (!seen_profiles.insert(name).second)
          throw std::invalid_argument("duplicate family profile: " + name);
        profiles.push_back(name);
      }
    }
    for (const auto& profile : profiles) {
      if (!known_profile(profile))
        throw std::invalid_argument("unknown profile: " + profile);
      ProfileCase expanded = c;
      expanded.profile = profile;
      const auto marker = expanded.id.find("{profile}");
      if (marker != std::string::npos)
        expanded.id.replace(marker, 9, profile);
      if (!ids.insert(expanded.id).second)
        throw std::invalid_argument("duplicate case: " + expanded.id);
      result.cases.push_back(std::move(expanded));
    }
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

template <typename Scenario>
struct SelectedScenario {
  Scenario scenario;
  std::string live_port;
  bool cached{false};
};

template <typename Scenario, typename Parse>
std::vector<SelectedScenario<Scenario>> select_validated_scenarios(
    const ProfileSuite& suite, std::string_view scope, const std::vector<std::string>& sizes,
    Parse&& parse) {
  std::set<std::string> selected;
  for (const auto& c : suite.select(scope))
    selected.insert(c.id);
  std::vector<SelectedScenario<Scenario>> result;
  for (const auto& c : suite.cases) {
    try {
      Scenario scenario = parse(c, suite.shape_for(c.profile));
      if (scenario.id != c.id || scenario.size_id != c.profile)
        throw std::invalid_argument("adapter changed YAML case identity");
      if (selected.contains(c.id) &&
          (sizes.empty() || std::find(sizes.begin(), sizes.end(), c.profile) != sizes.end())) {
        const std::string port =
            c.override_value ? (*c.override_value)["port"].as<std::string>() : std::string{};
        const bool cached = c.override_value && (*c.override_value)["cache"] &&
                            (*c.override_value)["cache"].as<std::string>() == "cached";
        result.push_back({std::move(scenario), port, cached});
      }
    } catch (const std::exception& e) {
      throw std::invalid_argument("case '" + c.id + "' at line " + std::to_string(c.source_line) +
                                  ": " + e.what());
    }
  }
  if (result.empty())
    throw std::invalid_argument("no YAML cases match requested sizes");
  return result;
}

// Decode every case before applying scope and size filters. Operators provide
// only the typed conversion and contract-specific validation callback.
template <typename Scenario, typename Parse>
std::vector<Scenario> select_validated_cases(const ProfileSuite& suite, std::string_view scope,
                                             const std::vector<std::string>& sizes, Parse&& parse) {
  auto selected =
      select_validated_scenarios<Scenario>(suite, scope, sizes, std::forward<Parse>(parse));
  std::vector<Scenario> result;
  for (auto& item : selected)
    result.push_back(std::move(item.scenario));
  return result;
}

}  // namespace holoscan::radar::perf
