// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

#include <holoscan/logger/logger.hpp>

namespace holoscan::radar {

// Cache for configuration-derived artifacts (window coefficient tensors,
// dechirp references, seen FFT-plan keys), implementing the
// warn-once-and-cache rule of the module's parameter override convention
// (see operators/common/README.md):
//
//  - the initial, parameter-driven build is silent;
//  - a override to an unseen configuration builds the artifact and
//    warns once (pass warn_on_build = true for override-triggered lookups);
//  - previously-seen configurations are returned from cache so alternating
//    overrides do not thrash.
template <typename Value>
class ConfigCache {
 public:
  // Returns the cached value for `key`, building it with `builder()` on a
  // miss. `what` names the artifact in the warning message.
  template <typename Builder>
  const Value& get_or_build(const std::string& key, bool warn_on_build,
                            const char* what, Builder&& builder) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    if (warn_on_build) {
      HOLOSCAN_LOG_WARN(
          "override selected an unseen {} configuration; building it "
          "now and caching for reuse (key: {})", what, key);
    }
    auto [entry, inserted] = cache_.emplace(key, builder());
    return entry->second;
  }

  bool contains(const std::string& key) const {
    return cache_.find(key) != cache_.end();
  }

  size_t size() const { return cache_.size(); }

 private:
  std::unordered_map<std::string, Value> cache_;
};

}  // namespace holoscan::radar
