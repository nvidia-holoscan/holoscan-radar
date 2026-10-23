// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// emit_required_tensor: fail-fast names a full queue; wait retries until
// a slow cross-partition consumer drains a depth-1 kBlocking queue.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "radar_common/sdk/tensor_io.hpp"

namespace {

using holoscan::radar::emit_required_tensor;
using holoscan::radar::explain_fail_fast_emit;
using holoscan::radar::pooled_tensor_output;
using holoscan::radar::prepare_device_tensor;
using holoscan::radar::receive_tensor;
using holoscan::radar::tensor_input;

constexpr auto kTick = std::chrono::milliseconds{1};
constexpr auto kSinkHold = std::chrono::milliseconds{50};
constexpr int kTicks = 8;

struct Counters {
  std::atomic<int> received{0};
};

class RequiredEmitOp final : public holoscan::Operator<> {
 public:
  explicit RequiredEmitOp(bool wait) : wait_(wait) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation device{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt};
    spec.output(out).produces_tensor(pooled_tensor_output(device, 1U, 16U));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick})
        .complete_after(holoscan::ComputeCalls{.limit = static_cast<uint32_t>(kTicks)});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::vector<int64_t> shape{2};
    return emit_required_tensor(out, context, "RequiredEmitOp.out", wait_, [&] {
      return prepare_device_tensor(
          out, context, shape, DLDataType{kDLFloat, 32, 1},
          [&](void*, std::size_t) -> holoscan::expected<void, holoscan::Error> {
            return {};
          });
    });
  }

  holoscan::Output<holoscan::Tensor> out{"out"};

 private:
  bool wait_{false};
};

class SlowSinkOp final : public holoscan::Operator<> {
 public:
  explicit SlowSinkOp(std::shared_ptr<Counters> counters) : counters_(std::move(counters)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation device{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt};
    spec.input(in).queue_depth(1U).expects_tensor(tensor_input(device));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    HOLOSCAN_TRY(auto tensor, receive_tensor(in, "SlowSinkOp"));
    (void)tensor;
    counters_->received.fetch_add(1);
    std::this_thread::sleep_for(kSinkHold);
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};

 private:
  std::shared_ptr<Counters> counters_;
};

struct CancellationState {
  std::atomic<bool> retrying{false};
  std::atomic<bool> retired{false};
};

// Model a reader already inside its cooperative queue/pool retry when another
// partition fails. The retry must observe cancellation without an external stop.
class RetryingPeerOp final : public holoscan::Operator<> {
 public:
  explicit RetryingPeerOp(std::shared_ptr<CancellationState> state) : state_(std::move(state)) {}

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    (void)holoscan::radar::retry_transient(
        context, "RetryingPeerOp", [&]() -> holoscan::expected<void, holoscan::Error> {
          state_->retrying.store(true);
          return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kBackpressured});
        });
    state_->retired.store(true);
    return {};
  }

 private:
  std::shared_ptr<CancellationState> state_;
};

class FailingEmitOp final : public holoscan::Operator<> {
 public:
  FailingEmitOp(std::shared_ptr<CancellationState> state, holoscan::ErrorCode code)
  : state_(std::move(state)), code_(code) {}

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (!state_->retrying.load())
      return {};
    return holoscan::radar::emit_required_result(
        context, "FailingEmitOp.out", false, [&]() -> holoscan::expected<void, holoscan::Error> {
          return holoscan::make_unexpected(holoscan::Error{code_});
        });
  }

 private:
  std::shared_ptr<CancellationState> state_;
  holoscan::ErrorCode code_;
};

struct RunOutcome {
  holoscan::RunTermination termination{holoscan::RunTermination::kFailed};
  std::string diagnostics;
  int received{0};
};

RunOutcome run_backpressure_graph(bool wait) {
  auto counters = std::make_shared<Counters>();
  holoscan::Graph graph{wait ? "emit-required-wait" : "emit-required-fail-fast"};
  auto producer = graph.op<RequiredEmitOp>("producer", wait);
  auto sink = graph.op<SlowSinkOp>("sink", counters);
  const holoscan::ConnectionOptions queue{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                          .queue_depth = 1U};
  graph.add_flow(producer->out, sink->in, queue);
  graph.partition("producer").add(producer);
  graph.partition("consumer").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  const holoscan::ThreadPlacement lane{};
  options.deployment.bind_thread_placement("producer", lane)
      .bind_thread_placement("consumer", lane);
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "producer", .output_port = "out", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  RunOutcome outcome;
  if (!plan.ok()) {
    outcome.diagnostics = std::string(plan.json());
    return outcome;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline =
      std::chrono::steady_clock::now() + (wait ? std::chrono::seconds{5} : std::chrono::seconds{2});
  while (std::chrono::steady_clock::now() < deadline) {
    if (wait && counters->received.load() >= kTicks)
      break;
    if (!wait && !session.diagnostics().empty())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }
  session.request_stop();
  session.wait();
  outcome.termination = session.termination();
  outcome.received = counters->received.load();
  for (const auto& diagnostic : session.diagnostics())
    outcome.diagnostics += holoscan::format_diagnostic(diagnostic, plan.graph());
  return outcome;
}

TEST(EmitRequired, FailFastMessageNamesFullQueue) {
  const auto backpressured =
      explain_fail_fast_emit("RequiredEmitOp.out", holoscan::ErrorCode::kBackpressured);
  const std::string queue_full = holoscan::runtime_error_message(backpressured);
  EXPECT_EQ(backpressured.code, holoscan::ErrorCode::kBackpressured);
  EXPECT_NE(queue_full.find("output queue is full"), std::string::npos);
  EXPECT_EQ(queue_full.find("wait_for_downstream"), std::string::npos);

  const auto exhausted =
      explain_fail_fast_emit("RequiredEmitOp.out", holoscan::ErrorCode::kResourceExhausted);
  const std::string no_buffer = holoscan::runtime_error_message(exhausted);
  EXPECT_EQ(exhausted.code, holoscan::ErrorCode::kResourceExhausted);
  EXPECT_NE(no_buffer.find("no free tensor buffer"), std::string::npos);
  EXPECT_EQ(no_buffer.find("wait_for_downstream"), std::string::npos);
}

TEST(EmitRequired, FailFastAbortsWhenConsumerIsBehind) {
  const RunOutcome outcome = run_backpressure_graph(false);
  EXPECT_EQ(outcome.termination, holoscan::RunTermination::kFailed) << outcome.diagnostics;
  EXPECT_NE(outcome.diagnostics.find("output queue is full"), std::string::npos)
      << outcome.diagnostics;
  EXPECT_LT(outcome.received, kTicks);
}

TEST(EmitRequired, FailFastCancelsRetryingPeerAndPreservesFailure) {
  for (const auto code :
       {holoscan::ErrorCode::kBackpressured, holoscan::ErrorCode::kResourceExhausted}) {
    SCOPED_TRACE(holoscan::error_code_name(code));
    auto state = std::make_shared<CancellationState>();
    holoscan::Graph graph{"emit-required-cancellation"};
    auto peer = graph.op<RetryingPeerOp>("retrying_peer", state);
    auto failing = graph.op<FailingEmitOp>("failing_emit", state, code);
    graph.partition("peer").add(peer);
    graph.partition("failure").add(failing);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    holoscan::CompileOptions options;
    const holoscan::ThreadPlacement lane{};
    options.deployment.bind_thread_placement("peer", lane).bind_thread_placement("failure", lane);
    const auto plan = holoscan::compile(graph, std::move(options));
    ASSERT_TRUE(plan.ok()) << plan.json();
    auto session = holoscan::run_async(plan);
    const bool finished = session.wait_for(std::chrono::seconds{2});
    // Bound a regression's cleanup without letting the fallback satisfy the assertion.
    if (!finished)
      session.request_stop();
    session.wait();
    EXPECT_TRUE(finished) << "Fatal emit failed to cancel a peer's retry loop";
    EXPECT_TRUE(state->retired.load());
    EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
    bool preserved = false;
    const auto expected =
        holoscan::runtime_error_message(explain_fail_fast_emit("FailingEmitOp.out", code));
    for (const auto& diagnostic : session.diagnostics()) {
      const auto formatted = holoscan::format_diagnostic(diagnostic, plan.graph());
      preserved = preserved || formatted.find(expected) != std::string::npos;
    }
    EXPECT_TRUE(preserved) << "Original fatal emit diagnostic was lost";
  }
}

TEST(EmitRequired, WaitRetriesUntilConsumerDrains) {
  const RunOutcome outcome = run_backpressure_graph(true);
  EXPECT_TRUE(outcome.diagnostics.empty()) << outcome.diagnostics;
  EXPECT_EQ(outcome.received, kTicks);
}

}  // namespace
