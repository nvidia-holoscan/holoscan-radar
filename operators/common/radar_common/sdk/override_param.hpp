// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
// A `[live]` port joins the data trigger with Exact{}, so an override must
// carry the trigger sample's capture_time; on_missing SafeDefault{} lets the
// activation proceed on a tick with no match. The entry is unconditional —
// wiring the port is what enables the override, and an unwired port simply
// never matches.
//
// An override applies to **one tick**. A matched sample is the effective value
// for that activation only; any tick without a match falls back to the Params
// value. Nothing is carried forward, so an application that wants an override
// to hold publishes it on every tick — which is what Exact{} asks for anyway,
// since the sequencer stamps the override and the data from one transmit time.
//
// Never treat a default-constructed SafeDefault payload as a value: `nfft == 0`
// is a legal seed, so absence is signalled by the optional, not by the payload.

template <typename T>
void add_live_override_to_contract(Contract& contract, const Input<T>& port) {
  contract.input(port, Exact{}).on_missing(port, SafeDefault{});
}

// Effective value for a `[live]` port this tick: the matched sample, or the
// Params seed when this tick has none.
template <typename T>
expected<T, Error> resolve_live_override(Input<T>& port, const T& seed) {
  HOLOSCAN_TRY(auto maybe, receive_optional_value(port));
  if (maybe.has_value()) {
    return std::move(*maybe);
  }
  return seed;
}

// Same, through a validating converter, for a port whose physical payload is a
// wire schema rather than the logical type. The converter runs only on a
// matched sample, so generated wire storage never reaches operator state.
template <typename Wire, typename Converted, typename Converter>
expected<Converted, Error> resolve_live_override_via(Input<Wire>& port,
                                                     const Converted& seed,
                                                     Converter&& converter) {
  HOLOSCAN_TRY(auto maybe, receive_optional_value(port));
  if (!maybe.has_value()) {
    return seed;
  }
  return std::invoke(std::forward<Converter>(converter), *maybe);
}

}  // namespace holoscan::radar
