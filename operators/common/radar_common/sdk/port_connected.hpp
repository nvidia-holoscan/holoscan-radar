// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include <holoscan/holoscan.hpp>

// Build-time wiring check: whether a named input port has an incoming
// edge (docs/planning/SAR/OPERATOR_API.md → *Build-time wiring validation*).
// Detects that an edge exists, not that it is the right edge.
//
// Operator is not enable_shared_from_this, so this finds `op` by name
// among the fragment graph's nodes and unions each predecessor's port
// map. A missing fragment (operator not yet composed) is treated as
// unconnected.

namespace holoscan::radar {

inline bool input_port_connected(holoscan::Operator* op,
                                 std::string_view port) {
  if (op == nullptr) {
    return false;
  }
  // fragment() / graph() are non-const on 4.3 (no const overloads).
  holoscan::Fragment* fragment = op->fragment();
  if (fragment == nullptr) {
    return false;
  }
  auto& graph = fragment->graph();
  std::shared_ptr<holoscan::Operator> self;
  for (const auto& node : graph.get_nodes()) {
    if (node && node->name() == op->name()) {
      self = node;
      break;
    }
  }
  if (!self) {
    return false;
  }
  const std::string port_name(port);
  for (const auto& pred : graph.get_previous_nodes(self)) {
    // 4.3: optional<shared_ptr<unordered_map<src, set<dst>>>>
    const auto maybe_map = graph.get_port_map(pred, self);
    if (!maybe_map || !*maybe_map) {
      continue;
    }
    for (const auto& entry : **maybe_map) {
      const auto& dest_ports = entry.second;
      if (dest_ports.find(port_name) != dest_ports.end()) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace holoscan::radar
