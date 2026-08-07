// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/holoscan.hpp>

namespace holoscan::radar {

// Single-entry TensorMap I/O helpers implementing the module convention that a
// port carries exactly one tensor and the port name — not the map key — is the
// contract (see operators/common/README.md). Each operator passes its own name
// as `who` so error messages stay specific.

// Receive the single tensor on a required input port. Throws when no message
// arrived, when the map does not hold exactly one entry, or when that entry is
// a null tensor. The returned pointer is always non-null.
inline std::shared_ptr<holoscan::Tensor> receive_single(
    holoscan::InputContext& op_input, const char* port,
    const std::string& who) {
  auto maybe = op_input.receive<holoscan::TensorMap>(port);
  if (!maybe) {
    throw std::runtime_error(
        who + ": no message received on input port '" + port + "'");
  }
  holoscan::TensorMap& map = maybe.value();
  if (map.size() != 1) {
    throw std::invalid_argument(
        who + ": input port '" + port +
        "' expects a single-entry TensorMap, got " +
        std::to_string(map.size()) + " entries");
  }
  std::shared_ptr<holoscan::Tensor> tensor = map.begin()->second;
  if (!tensor) {
    throw std::runtime_error(
        who + ": received a null tensor on input port '" + port + "'");
  }
  return tensor;
}

// Receive the single tensor on an optional (non-blocking) input port. Returns
// nullptr when the port is unconnected or carried no tensor this tick; throws
// only when a message arrived with more than one entry. Callers guard the
// result with `if (tensor)`.
inline std::shared_ptr<holoscan::Tensor> receive_optional(
    holoscan::InputContext& op_input, const char* port,
    const std::string& who) {
  auto maybe = op_input.receive<holoscan::TensorMap>(port);
  if (!maybe) {
    return nullptr;
  }
  holoscan::TensorMap& map = maybe.value();
  if (map.empty()) {
    return nullptr;
  }
  if (map.size() != 1) {
    throw std::invalid_argument(
        who + ": input port '" + port +
        "' expects a single-entry TensorMap, got " +
        std::to_string(map.size()) + " entries");
  }
  return map.begin()->second;
}

// Emit a single tensor on an output port as a one-entry TensorMap keyed by the
// port name.
inline void emit_single(holoscan::OutputContext& op_output, const char* port,
                        std::shared_ptr<holoscan::Tensor> tensor) {
  holoscan::TensorMap map;
  map.insert({port, std::move(tensor)});
  op_output.emit(map, port);
}

}  // namespace holoscan::radar
