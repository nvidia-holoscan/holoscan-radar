// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "radar_common/descriptors/max_tensor_extents.hpp"

namespace holoscan::radar {

// Named leading axes for FMCW cubes: none, N_rx, or (M_tx, N_rx).
// Opaque leading batch above that is rejected (MAX_TENSOR_EXTENTS Part B).
inline std::size_t named_tx_rx_leading_rank(const MaxTensorExtents& extents) {
  if (extents.max_tx.has_value()) {
    return 2U;
  }
  if (extents.max_rx.has_value() || extents.max_channels.has_value()) {
    return 1U;
  }
  return 0U;
}

inline std::int64_t extent_or_one(const std::optional<std::int64_t>& field) {
  return field.value_or(1);
}

inline std::int64_t extent_rx_or_channels(const MaxTensorExtents& extents) {
  if (extents.max_rx.has_value()) {
    return *extents.max_rx;
  }
  if (extents.max_channels.has_value()) {
    return *extents.max_channels;
  }
  return 1;
}

// Pool product for named (M_tx, N_rx) leading axes. Setting max_tx admits a
// rank-2 leading layout, so max_rx/max_channels must also be explicit —
// defaulting the missing axis to 1 silently undersizes the pool.
inline void require_tx_rx_pool_extents(const MaxTensorExtents& extents,
                                       const char* who, std::int64_t* max_tx,
                                       std::int64_t* max_rx) {
  if (extents.max_tx.has_value()) {
    *max_tx = require_extent(extents.max_tx, "max_tx", who);
    if (!extents.max_rx.has_value() && !extents.max_channels.has_value()) {
      throw std::invalid_argument(
          std::string(who) +
          ": max_tx requires max_rx or max_channels so the output pool "
          "matches the admitted (M_tx, N_rx) leading rank");
    }
    *max_rx = extent_rx_or_channels(extents);
  } else {
    *max_tx = 1;
    *max_rx = extent_rx_or_channels(extents);
  }
  if (*max_rx <= 0 || *max_tx <= 0) {
    throw std::invalid_argument(
        std::string(who) +
        ": max_rx/max_tx extents must be positive when set");
  }
}

inline std::string unexpected_leading_message(const char* who,
                                              std::size_t got,
                                              std::size_t max_leading) {
  return std::string(who) +
         ": unexpected leading axes (opaque batch is not supported); got "
         "leading rank " +
         std::to_string(got) + ", max named leading " +
         std::to_string(max_leading);
}

}  // namespace holoscan::radar
