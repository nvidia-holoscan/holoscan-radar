// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal Holoscan Graph reproducer for HSDK5_PORT.md §1:
// mixing a bind_tensor_output_device Tensor path with a mid-run scalar
// override Input fails on Holoscan 5 EA (compile:
// PARTITION_MANUAL_AUTHORING_REQUIRED; or runtime:
// RUNTIME_OPERATOR_REPLAY_MISMATCH / RUNTIME_PORT_BINDING_UNAVAILABLE /
// RUNTIME_EXECUTION_LANE_PLAN_INVALID).
//
// Not part of the default `unit` label — run explicitly:
//   ctest --test-dir build -L ea_repro --output-on-failure
// or:
//   ./build/tests/cpp/test_live_override_repro
//
// Control case (tensor path only, override port unwired) is expected to pass.
// Dual-OnClock / co-emit / sticky-contract cases document the EA failure for
// Holoscan maintainers (full plan JSON under /tmp/live_override_repro_*.json).

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace {

using holoscan::radar::add_sticky_override_to_contract;
using holoscan::radar::emit_device_tensor;
using holoscan::radar::make_device_tensor;
using holoscan::radar::real_dtype;
using holoscan::radar::receive_tensor;
using holoscan::radar::resolve_sticky_override;

inline constexpr auto kTick = std::chrono::milliseconds{5};
inline constexpr auto kDeadline = std::chrono::seconds{5};
inline constexpr auto kPoll = std::chrono::milliseconds{25};

struct Bank {
  static void reset(int n) {
    const std::lock_guard lock(mutex);
    remaining = n;
    captured = 0;
  }
  static inline std::mutex mutex;
  static inline int remaining{0};
  static inline int captured{0};
};

// Tiny device Tensor source: one float32 scalar per tick.
class DeviceTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out, "out")
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 1U, .max_byte_span = sizeof(float)});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    {
      const std::lock_guard lock(Bank::mutex);
      if (Bank::remaining <= 0) {
        return {};
      }
      --Bank::remaining;
    }
    auto host = std::make_shared<float>(1.0F);
    auto tensor = make_device_tensor({1}, real_dtype<float>());
    auto mut = tensor->mutable_data();
    if (!mut) {
      return holoscan::make_unexpected(std::move(mut).error());
    }
    if (cudaMemcpy(mut.value(), host.get(), sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "H2D failed"});
    }
    return emit_device_tensor(
        out, context, tensor->shape(), tensor->dtype(),
        [&](void* dst, std::size_t nbytes) -> holoscan::expected<void, holoscan::Error> {
          if (cudaMemcpyAsync(dst, tensor->data(), nbytes,
                              cudaMemcpyDeviceToDevice,
                              context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(
                holoscan::Error{holoscan::ErrorCode::kFailure,
                                "D2D failed"});
          }
          return {};
        });
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
};

// Separate OnClock source for a mid-run scalar override.
class ScalarOverrideSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // Emit every tick so stickiness is not required for the repro.
    return out.emit(
        int64_t{7},
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<int64_t> out{"out"};
};

// One source that co-emits device Tensor + scalar on the same OnClock tick
// (second wiring pattern tried during the EA probe).
class CoEmitSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(tensor_out, "tensor_out")
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 1U, .max_byte_span = sizeof(float)});
    spec.output(scalar_out, "scalar_out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    {
      const std::lock_guard lock(Bank::mutex);
      if (Bank::remaining <= 0) {
        return {};
      }
      --Bank::remaining;
    }
    auto tensor = make_device_tensor({1}, real_dtype<float>());
    float one = 1.0F;
    auto mut = tensor->mutable_data();
    if (!mut) {
      return holoscan::make_unexpected(std::move(mut).error());
    }
    if (cudaMemcpy(mut.value(), &one, sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "H2D failed"});
    }
    HOLOSCAN_RETURN_IF_ERROR(emit_device_tensor(
        tensor_out, context, tensor->shape(), tensor->dtype(),
        [&](void* dst,
            std::size_t nbytes) -> holoscan::expected<void, holoscan::Error> {
          if (cudaMemcpyAsync(dst, tensor->data(), nbytes,
                              cudaMemcpyDeviceToDevice,
                              context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(
                holoscan::Error{holoscan::ErrorCode::kFailure,
                                "D2D failed"});
          }
          return {};
        }));
    return scalar_out.emit(
        int64_t{7},
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::Tensor> tensor_out{"tensor_out"};
  holoscan::Output<int64_t> scalar_out{"scalar_out"};
};

// Pass-through op: OnEach on the Tensor; sticky optional scalar override.
class PassThroughWithOverrideOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(in, "in").queue_depth(2U).expects_tensor(kDevice);
    spec.output(out, "out")
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 1U, .max_byte_span = sizeof(float)});
    spec.input(scale, "scale").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    // Match production EA workaround (ClutterFilterOp): trigger on the data
    // tensor only. Including the sticky override in the temporal contract
    // makes an unwired graph fail TEMPORAL_INPUT_UNCONNECTED; including it
    // when wired is the pattern Holoscan maintainers should also evaluate.
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    HOLOSCAN_TRY(auto input, receive_tensor(in, "PassThroughWithOverrideOp"));
    HOLOSCAN_TRY(auto scale_value,
                 resolve_sticky_override(scale, int64_t{1}, sticky_scale_));
    (void)scale_value;
    return emit_device_tensor(
        out, context, input.shape(), input.dtype(),
        [&](void* dst,
            std::size_t nbytes) -> holoscan::expected<void, holoscan::Error> {
          if (cudaMemcpyAsync(dst, input.data(), nbytes,
                              cudaMemcpyDeviceToDevice,
                              context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(
                holoscan::Error{holoscan::ErrorCode::kFailure,
                                "D2D failed"});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
  holoscan::Output<holoscan::Tensor> out{"out"};
  holoscan::Input<int64_t> scale{"scale"};

 private:
  std::optional<int64_t> sticky_scale_;
};

// Same as PassThroughWithOverrideOp but puts the sticky override into the
// temporal contract (the pattern we want once EA admits it).
class PassThroughWithStickyContractOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(in, "in").queue_depth(2U).expects_tensor(kDevice);
    spec.output(out, "out")
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 1U, .max_byte_span = sizeof(float)});
    spec.input(scale, "scale").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    add_sticky_override_to_contract(result, scale);
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    HOLOSCAN_TRY(auto input, receive_tensor(in, "PassThroughWithStickyContractOp"));
    HOLOSCAN_TRY(auto scale_value,
                 resolve_sticky_override(scale, int64_t{1}, sticky_scale_));
    (void)scale_value;
    return emit_device_tensor(
        out, context, input.shape(), input.dtype(),
        [&](void* dst,
            std::size_t nbytes) -> holoscan::expected<void, holoscan::Error> {
          if (cudaMemcpyAsync(dst, input.data(), nbytes,
                              cudaMemcpyDeviceToDevice,
                              context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(
                holoscan::Error{holoscan::ErrorCode::kFailure,
                                "D2D failed"});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
  holoscan::Output<holoscan::Tensor> out{"out"};
  holoscan::Input<int64_t> scale{"scale"};

 private:
  std::optional<int64_t> sticky_scale_;
};

class CaptureSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(8U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = in.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    {
      const std::lock_guard lock(Bank::mutex);
      ++Bank::captured;
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

struct ReproResult {
  bool plan_ok{false};
  std::string plan_json;
  std::vector<std::string> diagnostics;
  int captured{0};
};

void bind_device(holoscan::CompileOptions& options, const char* op_path,
                 const char* port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = op_path,
          .output_port = port,
          .device = holoscan::DeviceId{0},
      });
}

ReproResult run_until(holoscan::Graph& graph, holoscan::CompileOptions options,
                      int expected_frames) {
  ReproResult result;
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  result.plan_ok = plan.ok();
  result.plan_json = std::string(plan.json());
  if (!plan.ok()) {
    return result;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + kDeadline;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      const std::lock_guard lock(Bank::mutex);
      if (Bank::captured >= expected_frames) {
        break;
      }
    }
    if (!session.diagnostics().empty()) {
      break;
    }
    std::this_thread::sleep_for(kPoll);
  }
  session.request_stop();
  session.wait();
  for (const auto& d : session.diagnostics()) {
    result.diagnostics.emplace_back(d.token);
  }
  {
    const std::lock_guard lock(Bank::mutex);
    result.captured = Bank::captured;
  }
  return result;
}

bool has_known_ea_token(const ReproResult& r) {
  const char* known[] = {
      "RUNTIME_PORT_BINDING_UNAVAILABLE",
      "RUNTIME_EXECUTION_LANE_PLAN_INVALID",
      "PARTITION_MANUAL_AUTHORING_REQUIRED",
      "RUNTIME_OPERATOR_REPLAY_MISMATCH",
  };
  for (const auto& t : r.diagnostics) {
    for (const char* k : known) {
      if (t == k) {
        return true;
      }
    }
  }
  for (const char* k : known) {
    if (r.plan_json.find(k) != std::string::npos) {
      return true;
    }
  }
  // Compile rejection counts only when a known EA diagnostic token appears
  // in the structured diagnostics or plan JSON — not any bare compile fail.
  return false;
}

std::string summarize(const ReproResult& r) {
  std::string msg = "plan_ok=";
  msg += r.plan_ok ? "true" : "false";
  msg += " captured=";
  msg += std::to_string(r.captured);
  msg += " diagnostics=";
  for (const auto& t : r.diagnostics) {
    msg += " ";
    msg += t;
  }
  // Extract SCREAMING_SNAKE tokens that look like Holoscan diagnostics.
  std::string tokens;
  for (std::size_t i = 0; i < r.plan_json.size();) {
    if (std::isupper(static_cast<unsigned char>(r.plan_json[i]))) {
      std::size_t j = i;
      while (j < r.plan_json.size() &&
             (std::isalnum(static_cast<unsigned char>(r.plan_json[j])) ||
              r.plan_json[j] == '_')) {
        ++j;
      }
      if (j - i >= 12) {
        const std::string tok = r.plan_json.substr(i, j - i);
        if (tok.find("RUNTIME_") == 0 || tok.find("PARTITION_") == 0 ||
            tok.find("TEMPORAL_") == 0 || tok.find("IMPLICIT_") == 0) {
          tokens += " ";
          tokens += tok;
        }
      }
      i = j;
    } else {
      ++i;
    }
  }
  if (!tokens.empty()) {
    msg += " plan_tokens=";
    msg += tokens;
  }
  return msg;
}

void print_repro(const char* name, const ReproResult& r) {
  std::cout << "[LiveOverrideRepro] " << name << ": " << summarize(r)
            << std::endl;
  if (!r.plan_ok || !r.diagnostics.empty()) {
    const std::string path =
        std::string("/tmp/live_override_repro_") + name + ".json";
    std::ofstream out(path);
    out << r.plan_json;
    std::cout << "[LiveOverrideRepro] wrote full plan JSON to " << path
              << std::endl;
  }
}

}  // namespace

// Control: device Tensor path only (override Input declared but unwired).
TEST(LiveOverrideRepro, TensorPathAloneWorks) {
  constexpr int kFrames = 3;
  Bank::reset(kFrames);

  holoscan::Graph graph{"live-override-control"};
  const auto src = graph.op<DeviceTensorSourceOp>("source");
  const auto op = graph.op<PassThroughWithOverrideOp>("op");
  const auto sink = graph.op<CaptureSinkOp>("sink");
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(src->out, op->in, q);
  graph.add_flow(op->out, sink->in, q);

  holoscan::CompileOptions options;
  bind_device(options, "source", "out");
  bind_device(options, "op", "out");

  const ReproResult r = run_until(graph, std::move(options), kFrames);
  print_repro("TensorPathAloneWorks", r);
  EXPECT_TRUE(r.plan_ok) << summarize(r);
  EXPECT_TRUE(r.diagnostics.empty()) << summarize(r);
  EXPECT_EQ(r.captured, kFrames) << summarize(r);
}

// Repro A: separate OnClock device Tensor source + OnClock scalar override.
TEST(LiveOverrideRepro, DualOnClockDeviceTensorPlusScalar) {
  constexpr int kFrames = 3;
  Bank::reset(kFrames);

  holoscan::Graph graph{"live-override-dual-clock"};
  const auto tensor_src = graph.op<DeviceTensorSourceOp>("source");
  const auto scalar_src = graph.op<ScalarOverrideSourceOp>("override");
  const auto op = graph.op<PassThroughWithOverrideOp>("op");
  const auto sink = graph.op<CaptureSinkOp>("sink");
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(tensor_src->out, op->in, q);
  graph.add_flow(scalar_src->out, op->scale, q);
  graph.add_flow(op->out, sink->in, q);

  holoscan::CompileOptions options;
  bind_device(options, "source", "out");
  bind_device(options, "op", "out");

  const ReproResult r = run_until(graph, std::move(options), kFrames);
  print_repro("DualOnClockDeviceTensorPlusScalar", r);
  // Today on EA this is the failing pattern. Accept either a clean run (if a
  // newer SDK fixed it) or the documented binding/lane tokens.
  if (r.plan_ok && r.diagnostics.empty() && r.captured == kFrames) {
    SUCCEED() << "dual-OnClock live override path works on this SDK build";
    return;
  }
  EXPECT_TRUE(has_known_ea_token(r))
      << "unexpected failure mode (wanted known EA tokens): " << summarize(r);
}

// Repro B: one OnClock source co-emits Tensor + scalar.
TEST(LiveOverrideRepro, CoEmitDeviceTensorAndScalar) {
  constexpr int kFrames = 3;
  Bank::reset(kFrames);

  holoscan::Graph graph{"live-override-coemit"};
  const auto src = graph.op<CoEmitSourceOp>("source");
  const auto op = graph.op<PassThroughWithOverrideOp>("op");
  const auto sink = graph.op<CaptureSinkOp>("sink");
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(src->tensor_out, op->in, q);
  graph.add_flow(src->scalar_out, op->scale, q);
  graph.add_flow(op->out, sink->in, q);

  holoscan::CompileOptions options;
  bind_device(options, "source", "tensor_out");
  bind_device(options, "op", "out");

  const ReproResult r = run_until(graph, std::move(options), kFrames);
  print_repro("CoEmitDeviceTensorAndScalar", r);
  if (r.plan_ok && r.diagnostics.empty() && r.captured == kFrames) {
    SUCCEED() << "co-emit live override path works on this SDK build";
    return;
  }
  EXPECT_TRUE(has_known_ea_token(r))
      << "unexpected failure mode (wanted known EA tokens): " << summarize(r);
}

// Repro C: dual OnClock + sticky override in the temporal contract (intended).
TEST(LiveOverrideRepro, DualOnClockWithStickyContract) {
  constexpr int kFrames = 3;
  Bank::reset(kFrames);

  holoscan::Graph graph{"live-override-sticky-contract"};
  const auto tensor_src = graph.op<DeviceTensorSourceOp>("source");
  const auto scalar_src = graph.op<ScalarOverrideSourceOp>("override");
  const auto op = graph.op<PassThroughWithStickyContractOp>("op");
  const auto sink = graph.op<CaptureSinkOp>("sink");
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(tensor_src->out, op->in, q);
  graph.add_flow(scalar_src->out, op->scale, q);
  graph.add_flow(op->out, sink->in, q);

  holoscan::CompileOptions options;
  bind_device(options, "source", "out");
  bind_device(options, "op", "out");

  const ReproResult r = run_until(graph, std::move(options), kFrames);
  print_repro("DualOnClockWithStickyContract", r);
  if (r.plan_ok && r.diagnostics.empty() && r.captured == kFrames) {
    SUCCEED() << "sticky-contract live override path works on this SDK build";
    return;
  }
  EXPECT_TRUE(has_known_ea_token(r))
      << "unexpected failure mode (wanted known EA tokens): " << summarize(r);
}
