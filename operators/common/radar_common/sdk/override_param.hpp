// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar {

// HSDK 5 mapping of the module's `[live]` override convention
// (OPERATOR_API.md).
//
// Enabled ports join the data trigger with Exact{} so the override's
// capture_time must match the cube. on_missing SafeDefault{} lets the
// activation proceed when this tick has no match; never treat the
// default-constructed payload as a value (nfft == 0 is a legal seed).
// Stickiness vs revert-to-Params is compute-side (sticky_* bool).
//
// POD [live] ports join via maybe_add_live_override_to_contract (Exact +
// SafeDefault, gated by expect_*). resolve_sticky_override /
// add_sticky_override_to_contract remain for the LatestWithin repro test.

inline constexpr auto kStickyOverrideMaxAge = std::chrono::years{100};

template <typename T>
void add_live_override_to_contract(Contract& contract, const Input<T>& port) {
  contract.input(port, Exact{}).on_missing(port, SafeDefault{});
}

template <typename T>
void maybe_add_live_override_to_contract(Contract& contract,
                                         const Input<T>& port, bool expect) {
  if (expect) {
    add_live_override_to_contract(contract, port);
  }
}

// Resolve the effective value for an enabled Exact override port.
// Missing sample: sticky keeps the last match (else seed); otherwise
// always returns seed. A present sample is latched only when sticky.
template <typename T>
expected<T, Error> resolve_live_override(Input<T>& port, const T& seed,
                                         std::optional<T>& sticky,
                                         bool keep_sticky) {
  HOLOSCAN_TRY(auto maybe, receive_optional_value(port));
  if (maybe.has_value()) {
    if (keep_sticky) {
      sticky = *maybe;
    }
    return std::move(*maybe);
  }
  if (keep_sticky && sticky.has_value()) {
    return *sticky;
  }
  return seed;
}

template <typename T>
expected<T, Error> resolve_expected_override(
    bool expect, Input<T>& port, const T& seed, std::optional<T>& sticky,
    bool keep_sticky) {
  if (!expect) {
    return seed;
  }
  return resolve_live_override(port, seed, sticky, keep_sticky);
}

// Resolve a wire override through a validating converter. The converter runs
// only for a fresh sample, and sticky state owns the converted descriptor so
// generated wire storage never leaks into operator state.
template <typename Wire, typename Converted, typename Converter>
expected<Converted, Error> resolve_live_override_via(
    Input<Wire>& port, const Converted& seed,
    std::optional<Converted>& sticky, bool keep_sticky,
    Converter&& converter) {
  HOLOSCAN_TRY(auto maybe, receive_optional_value(port));
  if (maybe.has_value()) {
    HOLOSCAN_TRY(
        auto converted,
        std::invoke(std::forward<Converter>(converter), *maybe));
    if (keep_sticky) {
      sticky = converted;
    }
    return converted;
  }
  if (keep_sticky && sticky.has_value()) {
    return *sticky;
  }
  return seed;
}

template <typename Wire, typename Converted, typename Converter>
expected<Converted, Error> resolve_expected_override_via(
    bool expect, Input<Wire>& port, const Converted& seed,
    std::optional<Converted>& sticky, bool keep_sticky,
    Converter&& converter) {
  if (!expect) {
    return seed;
  }
  return resolve_live_override_via(port, seed, sticky, keep_sticky,
                                   std::forward<Converter>(converter));
}

// Deprecated LatestWithin mapping for operators not yet on Exact+expect.
template <typename T>
void add_sticky_override_to_contract(Contract& contract, const Input<T>& port) {
  contract.input(port, LatestWithin{.max_age = kStickyOverrideMaxAge})
      .on_missing(port, SafeDefault{});
}

template <typename T>
expected<T, Error> resolve_sticky_override(Input<T>& port, const T& seed,
                                           std::optional<T>& sticky) {
  HOLOSCAN_TRY(auto maybe, receive_optional_value(port));
  if (maybe.has_value()) {
    sticky = std::move(*maybe);
  }
  if (sticky.has_value()) {
    return *sticky;
  }
  return seed;
}

}  // namespace holoscan::radar
