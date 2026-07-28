// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace holoscan::radar {

// Ordered, corresponding element indices for two identical displaced
// subarrays. The displacement is derived from ArrayGeometry so it cannot
// disagree with the element positions. Empty vectors request automatic
// maximum-overlap ULA derivation.
struct ShiftInvariantSubarrayPair {
  std::vector<int64_t> reference_indices{};
  std::vector<int64_t> shifted_indices{};

  bool is_auto() const {
    return reference_indices.empty() && shifted_indices.empty();
  }

  void validate() const {
    if (is_auto()) {
      return;
    }
    if (reference_indices.empty() || shifted_indices.empty()) {
      throw std::invalid_argument(
          "ShiftInvariantSubarrayPair: reference_indices and "
          "shifted_indices must both be empty (auto) or both be populated");
    }
    if (reference_indices.size() != shifted_indices.size()) {
      throw std::invalid_argument(
          "ShiftInvariantSubarrayPair: index vectors must have equal length");
    }
    std::unordered_set<int64_t> reference_seen;
    std::unordered_set<int64_t> shifted_seen;
    for (size_t index = 0; index < reference_indices.size(); ++index) {
      const int64_t reference = reference_indices[index];
      const int64_t shifted = shifted_indices[index];
      if (!reference_seen.insert(reference).second ||
          !shifted_seen.insert(shifted).second) {
        throw std::invalid_argument(
            "ShiftInvariantSubarrayPair: indices within each subarray must "
            "be unique");
      }
    }
  }

  void validate(int64_t num_elements) const {
    validate();
    if (is_auto()) {
      return;
    }
    if (num_elements <= 0) {
      throw std::invalid_argument(
          "ShiftInvariantSubarrayPair: num_elements must be positive");
    }
    for (size_t index = 0; index < reference_indices.size(); ++index) {
      const int64_t reference = reference_indices[index];
      const int64_t shifted = shifted_indices[index];
      if (reference < 0 || reference >= num_elements || shifted < 0 ||
          shifted >= num_elements) {
        throw std::invalid_argument(
            "ShiftInvariantSubarrayPair: index out of array_geometry range");
      }
    }
  }

  friend bool operator==(const ShiftInvariantSubarrayPair& a,
                         const ShiftInvariantSubarrayPair& b) {
    return a.reference_indices == b.reference_indices &&
           a.shifted_indices == b.shifted_indices;
  }
  friend bool operator!=(const ShiftInvariantSubarrayPair& a,
                         const ShiftInvariantSubarrayPair& b) {
    return !(a == b);
  }
};

}  // namespace holoscan::radar
