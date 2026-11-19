// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/tensor/max_byte_span.hpp"

namespace holoscan::radar {

// Host generation and MatX indexing both use signed 64-bit element counts.
// Check the bytes and vector capacity too, before allocating or entering a
// generation loop. These are representability limits, not resource budgets.
template <typename T>
inline int64_t checked_element_count(std::initializer_list<int64_t> dims, const char* who) {
  const std::size_t count = product_extents(dims, who);
  (void)byte_span_elems(count, sizeof(T), who);
  if (count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) ||
      count > std::vector<T>().max_size()) {
    throw std::invalid_argument(std::string(who) + ": element count exceeds the supported range");
  }
  return static_cast<int64_t>(count);
}

}  // namespace holoscan::radar
