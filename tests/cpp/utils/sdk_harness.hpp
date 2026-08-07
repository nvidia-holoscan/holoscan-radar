// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Radar-flavored conveniences over the Holoscan SDK's native operator test
// harness (holoscan/test/test_harness.hpp):
//
//  - holoscan::test::OperatorTestHarness / create_operator_test drive one
//    operator inside a scheduled mini-application, with typed
//    TestHarnessSourceOp feeds per input port and TestHarnessSinkOp
//    validators per output port. TensorMapCapture below plugs into those
//    validators to copy device TensorMap outputs to the host.
//
//  - The harness requires every input port's data sequence to have the
//    same length, so override tests that need message-less ticks
//    (sticky-override semantics) instead compose the same SDK source/sink
//    operators directly with a shorter CountCondition on the override
//    source: wire_leading_override() below.
//
// (holoscan::ops::OperatorRunner was evaluated and rejected: it requires a
// running application's executor — the SDK's own tests only use it for
// operator-in-operator composition.)

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "utils/tensor_fixtures.hpp"

namespace holoscan::radar::testing {

// Single-entry TensorMap (the module's port payload convention).
inline holoscan::TensorMap single_map(std::shared_ptr<holoscan::Tensor> tensor,
                                      const std::string& key) {
  holoscan::TensorMap map;
  map.insert({key, std::move(tensor)});
  return map;
}

// Capture store for TensorMap output ports: the validator copies each
// received single-entry map to the host. Pass validators() to
// OperatorTestHarness::add_output_port<holoscan::TensorMap>() or to a
// TestHarnessSinkOp directly. The store must outlive the harness run.
struct TensorMapCapture {
  std::vector<HostTensor> frames;  // one per received message

  std::vector<std::function<void(const holoscan::TensorMap&)>> validators() {
    return {[this](const holoscan::TensorMap& map) {
      if (map.size() != 1) {
        throw std::runtime_error(
            "TensorMapCapture: expected a single-entry TensorMap");
      }
      frames.push_back(to_host(*map.begin()->second));
    }};
  }
};

// Wire a override source that emits `values` on the leading ticks
// only (its CountCondition is the value count), then stays silent while
// the data source keeps ticking — the building block for sticky-override
// tests. Call from an Application's compose() after creating the operator
// under test.
template <typename V>
void wire_leading_override(holoscan::Fragment* app,
                           const std::shared_ptr<holoscan::Operator>& op,
                           const std::string& port, std::vector<V> values) {
  auto source =
      app->make_operator<holoscan::test::TestHarnessSourceOp<V>>(
          port + "_source",
          app->make_condition<holoscan::CountCondition>(
              static_cast<int64_t>(values.size())));
  source->set_test_data(std::move(values));
  app->add_flow(source, op, {{"output", port}});
}

// Wire a TensorMap data source emitting one frame per tick (CountCondition
// = frame count).
inline std::shared_ptr<holoscan::Operator> wire_data_source(
    holoscan::Fragment* app, const std::shared_ptr<holoscan::Operator>& op,
    const std::string& port, std::vector<holoscan::TensorMap> frames) {
  auto source = app->make_operator<
      holoscan::test::TestHarnessSourceOp<holoscan::TensorMap>>(
      port + "_source",
      app->make_condition<holoscan::CountCondition>(
          static_cast<int64_t>(frames.size())));
  source->set_test_data(std::move(frames));
  app->add_flow(source, op, {{"output", port}});
  return source;
}

// Wire a capturing sink onto an output port.
inline void wire_capture_sink(holoscan::Fragment* app,
                              const std::shared_ptr<holoscan::Operator>& op,
                              const std::string& port,
                              TensorMapCapture* capture) {
  auto sink = app->make_operator<
      holoscan::test::TestHarnessSinkOp<holoscan::TensorMap>>(port + "_sink");
  sink->set_validators(capture->validators());
  app->add_flow(op, sink, {{port, "input"}});
}

}  // namespace holoscan::radar::testing
