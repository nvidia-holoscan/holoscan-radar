// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <optional>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar {

// HSDK 5 mapping of the module's sticky [live] override convention
// (OPERATOR_API.md).
//
// Contract: LatestWithin admits a fresh override when present; on_missing
// SafeDefault lets the activation proceed when the port is unconnected or
// silent (ReuseLast alone would suppress compute() before the first sample,
// which breaks Params-seeded defaults). Sticky retention across silent ticks
// is held in an operator-owned std::optional updated by
// resolve_sticky_override — not experimental runtime_mutable (scalars-only).
//
// LatestWithin still needs a positive max_age; use a century-scale bound so a
// queued override is not aged out mid-session.

inline constexpr std::chrono::nanoseconds kStickyOverrideMaxAge =
    std::chrono::years{100};

template <typename T>
void add_sticky_override_to_contract(Contract& contract, const Input<T>& port) {
  contract.input(port, LatestWithin{.max_age = kStickyOverrideMaxAge})
      .on_missing(port, SafeDefault{});
}

// Resolve the effective value for this activation: update `sticky` when a
// fresh sample arrives, then return sticky (or `seed` when never overridden).
template <typename T>
expected<T, Error> resolve_sticky_override(Input<T>& port, const T& seed,
                                           std::optional<T>& sticky) {
  // Unconnected / silent override ports must not fail the activation.
  auto maybe = receive_optional_value(port);
  if (!maybe) {
    if (sticky.has_value()) {
      return *sticky;
    }
    return seed;
  }
  if (maybe->has_value()) {
    sticky = std::move(*maybe).value();
  }
  if (sticky.has_value()) {
    return *sticky;
  }
  return seed;
}

}  // namespace holoscan::radar
