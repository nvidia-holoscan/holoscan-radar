// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared HSDK 5 Graph source / sink / runner for performance drivers.
// [EA HSDK5 ONLY] Replaces HSDK 4.x Application helpers in perf_source.hpp /
// perf_sink.hpp. Mid-session sticky [live] override timing remains deferred
// until docs/planning/HSDK5_PORT.md §1.
//
// Pattern (matches landed perf_clutter_filter):
//   - PerfTensorSourceOp emits one preallocated device Tensor per OnClock tick
//     and records start timestamps into TimingCollector.
//   - PerfTerminalSinkOp drains the timed output, cudaDeviceSynchronize()'s,
//     and records end timestamps.
//   - PerfNullSinkOp drains always-emitted optional outs without timing.
//   - run_perf_graph() compiles + runs until `total_frames` ends are captured.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "utils/perf_app.hpp"

namespace holoscan::radar::perf {

inline constexpr auto kPerfTick = std::chrono::milliseconds{1};
inline constexpr auto kPerfDeadline = std::chrono::seconds{60};
inline constexpr auto kPerfPoll = std::chrono::milliseconds{25};
inline constexpr std::size_t kPerfMaxByteSpan = 256ULL << 20;

struct PerfFrameBank {
  static void reset(std::shared_ptr<holoscan::Tensor> frame, int count,
                    TimingCollector* collector) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frame);
    input_frame_b = nullptr;
    input_frame_c = nullptr;
    frames_remaining = count;
    captured = 0;
    collector_ = collector;
  }

  static void reset_dual(std::shared_ptr<holoscan::Tensor> frame_a,
                         std::shared_ptr<holoscan::Tensor> frame_b, int count,
                         TimingCollector* collector) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frame_a);
    input_frame_b = std::move(frame_b);
    input_frame_c = nullptr;
    frames_remaining = count;
    captured = 0;
    collector_ = collector;
  }

  static void reset_triple(std::shared_ptr<holoscan::Tensor> frame_a,
                           std::shared_ptr<holoscan::Tensor> frame_b,
                           std::shared_ptr<holoscan::Tensor> frame_c, int count,
                           TimingCollector* collector) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frame_a);
    input_frame_b = std::move(frame_b);
    input_frame_c = std::move(frame_c);
    frames_remaining = count;
    captured = 0;
    collector_ = collector;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::shared_ptr<holoscan::Tensor> input_frame;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_b;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_c;
  static inline int frames_remaining{0};
  static inline int captured{0};
  static inline TimingCollector* collector_{nullptr};
};

inline void bind_tensor_out(holoscan::CompileOptions& options,
                            const char* operator_path,
                            const char* output_port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = operator_path,
          .output_port = output_port,
          .device = holoscan::DeviceId{0},
      });
}

namespace detail {

inline holoscan::expected<void, holoscan::Error> emit_device_copy(
    holoscan::Output<holoscan::Tensor>& out,
    holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame) {
  const auto shape = frame->shape();
  const DLDataType dtype = frame->dtype();
  auto loan = out.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(),
                      cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        "PerfTensorSourceOp: cudaMemcpyAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(
      std::move(*loan),
      holoscan::EmitOptions{.capture_time = context.activation_time()});
}

inline void record_start_locked() {
  if (PerfFrameBank::collector_ != nullptr) {
    if (!PerfFrameBank::collector_->first_tick_set) {
      PerfFrameBank::collector_->t_first_tick = Clock::now();
      PerfFrameBank::collector_->first_tick_set = true;
    }
    PerfFrameBank::collector_->starts.push_back(Clock::now());
  }
}

}  // namespace detail

// Single-output device Tensor source for 1-in graphs.
class PerfTensorSourceOp final : public holoscan::Operator<> {
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
            .max_rank = 8U, .max_byte_span = kPerfMaxByteSpan});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      frame = PerfFrameBank::input_frame;
      detail::record_start_locked();
    }
    return detail::emit_device_copy(out, context, frame);
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
};

// Dual-output source: both tensors emit on the same OnClock tick (needed for
// OnAligned Exact consumers such as SnapshotGatherOp).
class PerfDualTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    holoscan::TensorAllocationBounds bounds{.max_rank = 8U,
                                            .max_byte_span = kPerfMaxByteSpan};
    spec.output(out_a, "out_a")
        .produces_tensor(kDevice)
        .tensor_allocation(bounds);
    spec.output(out_b, "out_b")
        .produces_tensor(kDevice)
        .tensor_allocation(bounds);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> a;
    std::shared_ptr<holoscan::Tensor> b;
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr ||
          PerfFrameBank::input_frame_b == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      a = PerfFrameBank::input_frame;
      b = PerfFrameBank::input_frame_b;
      detail::record_start_locked();
    }
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_copy(out_a, context, a));
    return detail::emit_device_copy(out_b, context, b);
  }

  holoscan::Output<holoscan::Tensor> out_a{"out_a"};
  holoscan::Output<holoscan::Tensor> out_b{"out_b"};
};

// Triple-output source for DetectionReductionOp (labels / idx / points).
class PerfTripleTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    holoscan::TensorAllocationBounds bounds{.max_rank = 8U,
                                            .max_byte_span = kPerfMaxByteSpan};
    spec.output(out_a, "out_a")
        .produces_tensor(kDevice)
        .tensor_allocation(bounds);
    spec.output(out_b, "out_b")
        .produces_tensor(kDevice)
        .tensor_allocation(bounds);
    spec.output(out_c, "out_c")
        .produces_tensor(kDevice)
        .tensor_allocation(bounds);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> a;
    std::shared_ptr<holoscan::Tensor> b;
    std::shared_ptr<holoscan::Tensor> c;
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr ||
          PerfFrameBank::input_frame_b == nullptr ||
          PerfFrameBank::input_frame_c == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      a = PerfFrameBank::input_frame;
      b = PerfFrameBank::input_frame_b;
      c = PerfFrameBank::input_frame_c;
      detail::record_start_locked();
    }
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_copy(out_a, context, a));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_copy(out_b, context, b));
    return detail::emit_device_copy(out_c, context, c);
  }

  holoscan::Output<holoscan::Tensor> out_a{"out_a"};
  holoscan::Output<holoscan::Tensor> out_b{"out_b"};
  holoscan::Output<holoscan::Tensor> out_c{"out_c"};
};

// Timed terminal sink: syncs the device and records end timestamps.
class PerfTerminalSinkOp final : public holoscan::Operator<> {
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
    cudaDeviceSynchronize();
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::collector_ != nullptr) {
        PerfFrameBank::collector_->ends.push_back(Clock::now());
      }
      ++PerfFrameBank::captured;
    }
    PerfFrameBank::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

// Untimed drain for always-emitted optional outs (keeps the producer ticking).
class PerfNullSinkOp final : public holoscan::Operator<> {
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
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

// BuildFn signature:
//   void(holoscan::Graph& graph,
//        const auto& src,   // shared_ptr-like op handle from graph.op
//        holoscan::CompileOptions& options)
// Caller adds the operator under test, sinks, flows, and bind_tensor_out for
// every device-tensor producer besides "src"/"out" (already bound).
template <typename BuildFn>
TimingCollector run_perf_graph(std::shared_ptr<holoscan::Tensor> input,
                               int total_frames, const char* graph_name,
                               BuildFn build) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset(std::move(input), total_frames, &collector);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfTensorSourceOp>("src");
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out");
  build(graph, src, options);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    throw std::runtime_error(std::string(graph_name) +
                             " perf compile failed: " +
                             std::string(plan.json()));
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  {
    const auto deadline = std::chrono::steady_clock::now() + kPerfDeadline;
    std::unique_lock lock(PerfFrameBank::mutex);
    while (PerfFrameBank::captured < total_frames &&
           session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      PerfFrameBank::ready.wait_for(lock, kPerfPoll);
    }
  }
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string msg = std::string(graph_name) + " perf run diagnostics:";
    for (const auto& d : session.diagnostics()) {
      msg += " ";
      msg += std::string(d.token);
    }
    throw std::runtime_error(msg);
  }
  if (collector.ends.size() < static_cast<size_t>(total_frames)) {
    throw std::runtime_error(
        std::string(graph_name) + " perf: expected " +
        std::to_string(total_frames) + " frames, got " +
        std::to_string(collector.ends.size()));
  }
  return collector;
}

template <typename BuildFn>
TimingCollector run_perf_graph_dual(std::shared_ptr<holoscan::Tensor> input_a,
                                    std::shared_ptr<holoscan::Tensor> input_b,
                                    int total_frames, const char* graph_name,
                                    BuildFn build) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_dual(std::move(input_a), std::move(input_b),
                            total_frames, &collector);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfDualTensorSourceOp>("src");
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out_a");
  bind_tensor_out(options, "src", "out_b");
  build(graph, src, options);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    throw std::runtime_error(std::string(graph_name) +
                             " perf compile failed: " +
                             std::string(plan.json()));
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  {
    const auto deadline = std::chrono::steady_clock::now() + kPerfDeadline;
    std::unique_lock lock(PerfFrameBank::mutex);
    while (PerfFrameBank::captured < total_frames &&
           session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      PerfFrameBank::ready.wait_for(lock, kPerfPoll);
    }
  }
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string msg = std::string(graph_name) + " perf run diagnostics:";
    for (const auto& d : session.diagnostics()) {
      msg += " ";
      msg += std::string(d.token);
    }
    throw std::runtime_error(msg);
  }
  if (collector.ends.size() < static_cast<size_t>(total_frames)) {
    throw std::runtime_error(
        std::string(graph_name) + " perf: expected " +
        std::to_string(total_frames) + " frames, got " +
        std::to_string(collector.ends.size()));
  }
  return collector;
}

template <typename BuildFn>
TimingCollector run_perf_graph_triple(
    std::shared_ptr<holoscan::Tensor> input_a,
    std::shared_ptr<holoscan::Tensor> input_b,
    std::shared_ptr<holoscan::Tensor> input_c, int total_frames,
    const char* graph_name, BuildFn build) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_triple(std::move(input_a), std::move(input_b),
                              std::move(input_c), total_frames, &collector);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfTripleTensorSourceOp>("src");
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out_a");
  bind_tensor_out(options, "src", "out_b");
  bind_tensor_out(options, "src", "out_c");
  build(graph, src, options);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    throw std::runtime_error(std::string(graph_name) +
                             " perf compile failed: " +
                             std::string(plan.json()));
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  {
    const auto deadline = std::chrono::steady_clock::now() + kPerfDeadline;
    std::unique_lock lock(PerfFrameBank::mutex);
    while (PerfFrameBank::captured < total_frames &&
           session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      PerfFrameBank::ready.wait_for(lock, kPerfPoll);
    }
  }
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string msg = std::string(graph_name) + " perf run diagnostics:";
    for (const auto& d : session.diagnostics()) {
      msg += " ";
      msg += std::string(d.token);
    }
    throw std::runtime_error(msg);
  }
  if (collector.ends.size() < static_cast<size_t>(total_frames)) {
    throw std::runtime_error(
        std::string(graph_name) + " perf: expected " +
        std::to_string(total_frames) + " frames, got " +
        std::to_string(collector.ends.size()));
  }
  return collector;
}

}  // namespace holoscan::radar::perf
