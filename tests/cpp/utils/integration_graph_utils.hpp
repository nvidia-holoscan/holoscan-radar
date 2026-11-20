// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Finite CUDA Tensor sources and collectors for whole-Graph integration tests.
// Single-operator suites must use holoscan::testing instead of this utility.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
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
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "utils/tensor_fixtures.hpp"

namespace holoscan::radar::testing {

namespace integration {

inline constexpr auto kHarnessTick = std::chrono::milliseconds{5};
// How long a runner waits for the expected captures (or a diagnostic), and how
// often it re-checks while waiting.
inline constexpr auto kHarnessDeadline = std::chrono::seconds{15};
inline constexpr auto kHarnessPoll = std::chrono::milliseconds{25};

struct FrameBank {
  static void reset() {
    const std::lock_guard lock(mutex);
    input_frames.clear();
    captured.clear();
    input_index = 0;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<std::shared_ptr<holoscan::Tensor>> input_frames;
  static inline std::vector<HostTensor> captured;
  static inline std::size_t input_index{0};
};

// Second input/output bank for 2-in / 2-out Graph tests. Captures for the
// second output still use FrameBankB; dual inputs are published by one
// DualTensorSourceOp (same OnClock activation) so OnAligned Exact graphs
// compile without PARTITION_MANUAL_AUTHORING_REQUIRED.
struct FrameBankB {
  static void reset() {
    const std::lock_guard lock(mutex);
    input_frames.clear();
    captured.clear();
    input_index = 0;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<std::shared_ptr<holoscan::Tensor>> input_frames;
  static inline std::vector<HostTensor> captured;
  static inline std::size_t input_index{0};
};

// Third input bank (TripleTensorSourceOp) + capture bank for 1-in / N-out
// graphs (e.g. MVDREstimatorOp's five always-emitted Tensor outputs).
struct FrameBankC {
  static void reset() {
    const std::lock_guard lock(mutex);
    input_frames.clear();
    captured.clear();
    input_index = 0;
  }
  static inline std::mutex mutex;
  static inline std::vector<std::shared_ptr<holoscan::Tensor>> input_frames;
  static inline std::vector<HostTensor> captured;
  static inline std::size_t input_index{0};
};

struct FrameBankD {
  static void reset() {
    const std::lock_guard lock(mutex);
    captured.clear();
  }
  static inline std::mutex mutex;
  static inline std::vector<HostTensor> captured;
};

struct FrameBankE {
  static void reset() {
    const std::lock_guard lock(mutex);
    captured.clear();
  }
  static inline std::mutex mutex;
  static inline std::vector<HostTensor> captured;
};

// Emit one frame from `bank` onto `out` using the runtime stream. Shared by
// the single- and dual-output harness sources.
inline holoscan::expected<void, holoscan::Error> emit_bank_frame(
    holoscan::Output<holoscan::Tensor>& out, holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame, const char* who) {
  const auto shape = frame->shape();
  const DLDataType dtype = frame->dtype();
  auto loan = out.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  cudaStream_t stream = context.cuda_stream().get();
  // Zero-byte loans (e.g. empty detection_idx) have null data pointers.
  if (writer->byte_size() > 0 && cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(),
                                                 cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, std::string(who) + ": cudaMemcpyAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(std::move(*loan),
                  holoscan::EmitOptions{.capture_time = context.activation_time()});
}

class TensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out, "out")
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, 256ULL * 1024ULL * 1024ULL));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    {
      const std::lock_guard lock(FrameBank::mutex);
      if (FrameBank::input_index >= FrameBank::input_frames.size()) {
        return {};
      }
      frame = std::move(FrameBank::input_frames[FrameBank::input_index++]);
    }
    return emit_bank_frame(out, context, frame, "TensorSourceOp");
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
};

// Co-emits one device Tensor and an optional override payload with the same
// explicit capture time. A missing payload exercises Exact + SafeDefault:
// that tick falls back to the operator's Params value.
template <typename Payload>
struct OverrideBank {
  static inline std::vector<std::optional<Payload>> payloads;
};

template <typename Payload>
class TensorOverrideSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out, "out")
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, 256ULL * 1024ULL * 1024ULL));
    spec.output(override_out, "override_out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    std::optional<Payload> payload;
    {
      const std::lock_guard lock(FrameBank::mutex);
      if (FrameBank::input_index >= FrameBank::input_frames.size()) {
        return {};
      }
      const std::size_t index = FrameBank::input_index++;
      frame = std::move(FrameBank::input_frames[index]);
      if (index < OverrideBank<Payload>::payloads.size()) {
        payload = std::move(OverrideBank<Payload>::payloads[index]);
      }
    }
    if (payload.has_value()) {
      auto override_emitted = override_out.emit(
          std::move(*payload), holoscan::EmitOptions{.capture_time = context.activation_time()});
      if (!override_emitted) {
        return override_emitted;
      }
    }
    auto tensor_emitted = emit_bank_frame(out, context, frame, "TensorOverrideSourceOp");
    if (!tensor_emitted) {
      return tensor_emitted;
    }
    return {};
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
  holoscan::Output<Payload> override_out{"override_out"};
};

// One OnClock activation publishes both input tensors with the same
// capture_time. OnAligned{.window = 0ns} + Exact requires the heads to share a
// capture time exactly, and two independent OnClock sources tick at different
// times, so a single co-emitting source is what makes the join fire here (see
// /opt/nvidia/holoscan/examples/temporal_contracts/on_aligned/). Co-emission is
// not the only way to get equal times in general — a data-triggered operator
// inherits its input's capture time, so a production diamond feeding both ports
// from one upstream frame aligns without co-emitting (see
// docs/ARCHITECTURE.md, "Temporal coherence and publication order"). Two separate OnClock sources
// additionally fail compile with PARTITION_MANUAL_AUTHORING_REQUIRED on EA when combined with
// device Tensor binding.
class DualTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    const holoscan::TensorBounds bounds{.max_rank = 8U,
                                        .max_byte_span = 256ULL * 1024ULL * 1024ULL};
    spec.output(out0, "out0")
        .produces_tensor(holoscan::radar::pooled_tensor_output(kDevice, bounds));
    spec.output(out1, "out1")
        .produces_tensor(holoscan::radar::pooled_tensor_output(kDevice, bounds));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame0;
    std::shared_ptr<holoscan::Tensor> frame1;
    {
      const std::lock_guard lock0(FrameBank::mutex);
      const std::lock_guard lock1(FrameBankB::mutex);
      if (FrameBank::input_index >= FrameBank::input_frames.size() ||
          FrameBankB::input_index >= FrameBankB::input_frames.size()) {
        return {};
      }
      frame0 = std::move(FrameBank::input_frames[FrameBank::input_index++]);
      frame1 = std::move(FrameBankB::input_frames[FrameBankB::input_index++]);
    }
    auto first = emit_bank_frame(out0, context, frame0, "DualTensorSourceOp.out0");
    if (!first) {
      return first;
    }
    return emit_bank_frame(out1, context, frame1, "DualTensorSourceOp.out1");
  }

  holoscan::Output<holoscan::Tensor> out0{"out0"};
  holoscan::Output<holoscan::Tensor> out1{"out1"};
};

// Co-emits two aligned device Tensors and an optional Exact override.
template <typename Payload>
class DualTensorOverrideSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    const holoscan::TensorBounds bounds{.max_rank = 8U,
                                        .max_byte_span = 256ULL * 1024ULL * 1024ULL};
    spec.output(out0, "out0")
        .produces_tensor(holoscan::radar::pooled_tensor_output(kDevice, bounds));
    spec.output(out1, "out1")
        .produces_tensor(holoscan::radar::pooled_tensor_output(kDevice, bounds));
    spec.output(override_out, "override_out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame0;
    std::shared_ptr<holoscan::Tensor> frame1;
    std::optional<Payload> payload;
    {
      const std::lock_guard lock0(FrameBank::mutex);
      const std::lock_guard lock1(FrameBankB::mutex);
      if (FrameBank::input_index >= FrameBank::input_frames.size() ||
          FrameBankB::input_index >= FrameBankB::input_frames.size()) {
        return {};
      }
      const std::size_t index = FrameBank::input_index++;
      frame0 = std::move(FrameBank::input_frames[index]);
      frame1 = std::move(FrameBankB::input_frames[FrameBankB::input_index++]);
      if (index < OverrideBank<Payload>::payloads.size()) {
        payload = std::move(OverrideBank<Payload>::payloads[index]);
      }
    }
    if (payload.has_value()) {
      auto emitted = override_out.emit(
          std::move(*payload), holoscan::EmitOptions{.capture_time = context.activation_time()});
      if (!emitted) {
        return emitted;
      }
    }
    auto first = emit_bank_frame(out0, context, frame0, "DualTensorOverrideSourceOp.out0");
    if (!first) {
      return first;
    }
    return emit_bank_frame(out1, context, frame1, "DualTensorOverrideSourceOp.out1");
  }

  holoscan::Output<holoscan::Tensor> out0{"out0"};
  holoscan::Output<holoscan::Tensor> out1{"out1"};
  holoscan::Output<Payload> override_out{"override_out"};
};

// Three tensors on one OnClock activation (labels + detection_idx + points).
// OptionalThirdFirst publishes out2 before the trigger-forming required inputs,
// allowing out2 to exercise Exact instead of racing the required-input join.
template <bool OptionalThirdFirst = false>
class TripleTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    const holoscan::TensorBounds bounds{.max_rank = 8U,
                                        .max_byte_span = 256ULL * 1024ULL * 1024ULL};
    spec.output(out0, "out0")
        .produces_tensor(holoscan::radar::pooled_tensor_output(kDevice, bounds));
    spec.output(out1, "out1")
        .produces_tensor(holoscan::radar::pooled_tensor_output(kDevice, bounds));
    spec.output(out2, "out2")
        .produces_tensor(holoscan::radar::pooled_tensor_output(kDevice, bounds));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame0;
    std::shared_ptr<holoscan::Tensor> frame1;
    std::shared_ptr<holoscan::Tensor> frame2;
    {
      const std::lock_guard lock0(FrameBank::mutex);
      const std::lock_guard lock1(FrameBankB::mutex);
      const std::lock_guard lock2(FrameBankC::mutex);
      if (FrameBank::input_index >= FrameBank::input_frames.size() ||
          FrameBankB::input_index >= FrameBankB::input_frames.size() ||
          FrameBankC::input_index >= FrameBankC::input_frames.size()) {
        return {};
      }
      frame0 = std::move(FrameBank::input_frames[FrameBank::input_index++]);
      frame1 = std::move(FrameBankB::input_frames[FrameBankB::input_index++]);
      frame2 = std::move(FrameBankC::input_frames[FrameBankC::input_index++]);
    }
    if constexpr (OptionalThirdFirst) {
      auto optional = emit_bank_frame(out2, context, frame2, "TripleTensorSourceOp.out2");
      if (!optional) {
        return optional;
      }
      auto first = emit_bank_frame(out0, context, frame0, "TripleTensorSourceOp.out0");
      if (!first) {
        return first;
      }
      return emit_bank_frame(out1, context, frame1, "TripleTensorSourceOp.out1");
    } else {
      auto first = emit_bank_frame(out0, context, frame0, "TripleTensorSourceOp.out0");
      if (!first) {
        return first;
      }
      auto second = emit_bank_frame(out1, context, frame1, "TripleTensorSourceOp.out1");
      if (!second) {
        return second;
      }
      return emit_bank_frame(out2, context, frame2, "TripleTensorSourceOp.out2");
    }
  }

  holoscan::Output<holoscan::Tensor> out0{"out0"};
  holoscan::Output<holoscan::Tensor> out1{"out1"};
  holoscan::Output<holoscan::Tensor> out2{"out2"};
};

class TensorCaptureOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(8U); }

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
    HostTensor host = to_host(*sample);
    {
      const std::lock_guard lock(FrameBank::mutex);
      FrameBank::captured.push_back(std::move(host));
    }
    FrameBank::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

class TensorCaptureBOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(8U); }

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
    HostTensor host = to_host(*sample);
    {
      const std::lock_guard lock(FrameBankB::mutex);
      FrameBankB::captured.push_back(std::move(host));
    }
    FrameBankB::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

class DualTensorCaptureOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in0, "in0").queue_depth(8U);
    spec.input(in1, "in1").queue_depth(8U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::ports(in0, in1),
                                       .window = std::chrono::nanoseconds{0}});
    result.input(in0, holoscan::Exact{}).input(in1, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample0 = in0.receive_data();
    if (!sample0) {
      return holoscan::make_unexpected(std::move(sample0).error());
    }
    auto sample1 = in1.receive_data();
    if (!sample1) {
      return holoscan::make_unexpected(std::move(sample1).error());
    }
    HostTensor host0 = to_host(*sample0);
    HostTensor host1 = to_host(*sample1);
    {
      const std::lock_guard lock0(FrameBank::mutex);
      FrameBank::captured.push_back(std::move(host0));
    }
    {
      const std::lock_guard lock1(FrameBankB::mutex);
      FrameBankB::captured.push_back(std::move(host1));
    }
    FrameBank::ready.notify_all();
    FrameBankB::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> in0{"in0"};
  holoscan::Input<holoscan::Tensor> in1{"in1"};
};

class TensorCaptureCOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(8U); }

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
    HostTensor host = to_host(*sample);
    {
      const std::lock_guard lock(FrameBankC::mutex);
      FrameBankC::captured.push_back(std::move(host));
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

class TensorCaptureDOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(8U); }

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
    HostTensor host = to_host(*sample);
    {
      const std::lock_guard lock(FrameBankD::mutex);
      FrameBankD::captured.push_back(std::move(host));
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

class TensorCaptureEOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(8U); }

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
    HostTensor host = to_host(*sample);
    {
      const std::lock_guard lock(FrameBankE::mutex);
      FrameBankE::captured.push_back(std::move(host));
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

inline void bind_single_io_device_pools(holoscan::CompileOptions& options, const char* out_port) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "out",
      .device = holoscan::DeviceId{0},
  });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "op",
      .output_port = out_port,
      .device = holoscan::DeviceId{0},
  });
}

inline void bind_device_pools(holoscan::CompileOptions& options) {
  bind_single_io_device_pools(options, "range_profiles_out");
}

inline void bind_dual_io_device_pools(holoscan::CompileOptions& options, const char* out0_port,
                                      const char* out1_port) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "out0",
      .device = holoscan::DeviceId{0},
  });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "out1",
      .device = holoscan::DeviceId{0},
  });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "op",
      .output_port = out0_port,
      .device = holoscan::DeviceId{0},
  });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "op",
      .output_port = out1_port,
      .device = holoscan::DeviceId{0},
  });
}

// 1-in / 2-out device binds (TensorSourceOp "out" + two op outputs).
inline void bind_single_in_dual_out_device_pools(holoscan::CompileOptions& options,
                                                 const char* out0_port, const char* out1_port) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "out",
      .device = holoscan::DeviceId{0},
  });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "op",
      .output_port = out0_port,
      .device = holoscan::DeviceId{0},
  });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "op",
      .output_port = out1_port,
      .device = holoscan::DeviceId{0},
  });
}

inline void bind_op_device_output(holoscan::CompileOptions& options, const char* out_port) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "op",
      .output_port = out_port,
      .device = holoscan::DeviceId{0},
  });
}

// 2-in / 1-out device binds (DualTensorSourceOp out0/out1 + one op output).
inline void bind_dual_in_single_out_device_pools(holoscan::CompileOptions& options,
                                                 const char* out_port) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "out0",
      .device = holoscan::DeviceId{0},
  });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "out1",
      .device = holoscan::DeviceId{0},
  });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "op",
      .output_port = out_port,
      .device = holoscan::DeviceId{0},
  });
}

// 3-in / 1-out device binds (TripleTensorSourceOp out0/out1/out2 + one op
// output).
inline void bind_triple_in_single_out_device_pools(holoscan::CompileOptions& options,
                                                   const char* out_port) {
  bind_dual_in_single_out_device_pools(options, out_port);
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "out2",
      .device = holoscan::DeviceId{0},
  });
}

}  // namespace integration
}  // namespace holoscan::radar::testing
