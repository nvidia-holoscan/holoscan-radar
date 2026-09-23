// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <mutex>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::radar::testing {

template <typename Payload, typename Tag = void>
struct WireCaptureBank {
  static void reset() {
    const std::lock_guard lock(mutex);
    payloads.clear();
    capture_times.clear();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<Payload> payloads;
  static inline std::vector<TimePoint> capture_times;
};

template <typename Payload, typename Tag = void>
class WireCaptureOp final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(8U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{in});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override {
    auto sample = in.receive_data();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    {
      const std::lock_guard lock(WireCaptureBank<Payload, Tag>::mutex);
      WireCaptureBank<Payload, Tag>::payloads.push_back(std::move(*sample));
      WireCaptureBank<Payload, Tag>::capture_times.push_back(
          context.activation_time());
    }
    WireCaptureBank<Payload, Tag>::ready.notify_all();
    return {};
  }

  Input<Payload> in{"in"};
};

}  // namespace holoscan::radar::testing
