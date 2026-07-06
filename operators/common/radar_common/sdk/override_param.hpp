// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>

#include <holoscan/holoscan.hpp>

namespace holoscan::radar {

// Helpers implementing the module's parameter override convention (see
// operators/common/README.md): an overridable parameter may
// additionally be overridden in-flight via a same-named non-blocking input
// port. Overrides are sticky — the parameter seeds the initial effective
// value and an arriving override remains effective until the next override
// (callers resolve against the previous effective value each tick).

// Declare the non-blocking override input port for an overridable
// parameter. Call from setup(), after the spec.param() declaration of the
// same name.
template <typename T>
void add_override_port(holoscan::OperatorSpec& spec, const char* name) {
  spec.input<T>(name).condition(holoscan::ConditionType::kNone);
}

// Receive an override if one arrived this tick. Returns std::nullopt when
// the port is unconnected or carried no message, in which case the current
// effective value stands.
template <typename T>
std::optional<T> receive_override(holoscan::InputContext& op_input,
                                  const char* name) {
  auto maybe = op_input.receive<T>(name);
  if (maybe) {
    return std::optional<T>(std::move(maybe.value()));
  }
  return std::nullopt;
}

// Resolve the effective value of an overridable parameter for this tick:
// the override when one arrived, else `current` (the previous effective
// value, seeded from the parameter on the first tick). `changed` is set
// when an override arrived and differs from `current` (used to trigger the
// warn-once-and-cache path for costly rebuilds).
template <typename T>
T resolve_override_param(holoscan::InputContext& op_input, const char* name,
                         const T& current, bool* changed) {
  auto override_value = receive_override<T>(op_input, name);
  if (override_value.has_value()) {
    if (changed != nullptr) {
      *changed = !(*override_value == current);
    }
    return std::move(*override_value);
  }
  if (changed != nullptr) {
    *changed = false;
  }
  return current;
}

}  // namespace holoscan::radar
