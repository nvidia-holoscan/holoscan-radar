// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal HSDK 5 Graph harness for ported operator C++ tests.
// [EA HSDK5 ONLY] — starts from the ClutterFilter pilot; DualIO helpers cover
// 2-in / 2-out graphs (SnapshotGatherOp). Every device-tensor producer must
// be bind_tensor_output_device()'d.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
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

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "utils/tensor_fixtures.hpp"

namespace holoscan::radar::testing {

namespace detail {

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
    holoscan::Output<holoscan::Tensor>& out,
    holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame, const char* who) {
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
  // Zero-byte loans (e.g. empty detection_idx) have null data pointers.
  if (writer->byte_size() > 0 &&
      cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(),
                      cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string(who) + ": cudaMemcpyAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(
      std::move(*loan),
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
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, 256ULL * 1024ULL * 1024ULL));
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
      frame = FrameBank::input_frames[FrameBank::input_index++];
    }
    return emit_bank_frame(out, context, frame, "TensorSourceOp");
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
};

// Co-emits one device Tensor and an optional override payload with the same
// explicit capture time. A missing payload exercises Exact + SafeDefault;
// subsequent missing payloads exercise compute-side sticky state.
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
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, 256ULL * 1024ULL * 1024ULL));
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
      frame = FrameBank::input_frames[index];
      if (index < OverrideBank<Payload>::payloads.size()) {
        payload = std::move(OverrideBank<Payload>::payloads[index]);
      }
    }
    if (payload.has_value()) {
      auto override_emitted = override_out.emit(
          std::move(*payload),
          holoscan::EmitOptions{.capture_time = context.activation_time()});
      if (!override_emitted) {
        return override_emitted;
      }
    }
    auto tensor_emitted =
        emit_bank_frame(out, context, frame, "TensorOverrideSourceOp");
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
// docs/planning/OPERATOR_DEVELOPMENT.md, "Sample time propagation and aligned
// joins"). Two separate OnClock sources additionally fail compile with
// PARTITION_MANUAL_AUTHORING_REQUIRED on EA when combined with device Tensor
// binding.
class DualTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    const holoscan::TensorBounds bounds{
        .max_rank = 8U, .max_byte_span = 256ULL * 1024ULL * 1024ULL};
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
      frame0 = FrameBank::input_frames[FrameBank::input_index++];
      frame1 = FrameBankB::input_frames[FrameBankB::input_index++];
    }
    auto first = emit_bank_frame(out0, context, frame0,
                                 "DualTensorSourceOp.out0");
    if (!first) {
      return first;
    }
    return emit_bank_frame(out1, context, frame1, "DualTensorSourceOp.out1");
  }

  holoscan::Output<holoscan::Tensor> out0{"out0"};
  holoscan::Output<holoscan::Tensor> out1{"out1"};
};

// Three tensors on one OnClock activation (labels + detection_idx + points).
class TripleTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    const holoscan::TensorBounds bounds{
        .max_rank = 8U, .max_byte_span = 256ULL * 1024ULL * 1024ULL};
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
      frame0 = FrameBank::input_frames[FrameBank::input_index++];
      frame1 = FrameBankB::input_frames[FrameBankB::input_index++];
      frame2 = FrameBankC::input_frames[FrameBankC::input_index++];
    }
    auto first = emit_bank_frame(out0, context, frame0,
                                 "TripleTensorSourceOp.out0");
    if (!first) {
      return first;
    }
    auto second = emit_bank_frame(out1, context, frame1,
                                  "TripleTensorSourceOp.out1");
    if (!second) {
      return second;
    }
    return emit_bank_frame(out2, context, frame2,
                           "TripleTensorSourceOp.out2");
  }

  holoscan::Output<holoscan::Tensor> out0{"out0"};
  holoscan::Output<holoscan::Tensor> out1{"out1"};
  holoscan::Output<holoscan::Tensor> out2{"out2"};
};

class TensorCaptureOp final : public holoscan::Operator<> {
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

class TensorCaptureCOp final : public holoscan::Operator<> {
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
    HostTensor host = to_host(*sample);
    {
      const std::lock_guard lock(FrameBankE::mutex);
      FrameBankE::captured.push_back(std::move(host));
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

inline void bind_single_io_device_pools(holoscan::CompileOptions& options,
                                        const char* out_port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "out",
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "op",
          .output_port = out_port,
          .device = holoscan::DeviceId{0},
      });
}

inline void bind_device_pools(holoscan::CompileOptions& options) {
  bind_single_io_device_pools(options, "range_profiles_out");
}

inline void bind_dual_io_device_pools(
    holoscan::CompileOptions& options,
    const char* out0_port, const char* out1_port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "out0",
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "out1",
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "op",
          .output_port = out0_port,
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "op",
          .output_port = out1_port,
          .device = holoscan::DeviceId{0},
      });
}

// 1-in / 2-out device binds (TensorSourceOp "out" + two op outputs).
inline void bind_single_in_dual_out_device_pools(
    holoscan::CompileOptions& options, const char* out0_port,
    const char* out1_port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "out",
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "op",
          .output_port = out0_port,
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "op",
          .output_port = out1_port,
          .device = holoscan::DeviceId{0},
      });
}

inline void bind_op_device_output(holoscan::CompileOptions& options,
                                  const char* out_port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "op",
          .output_port = out_port,
          .device = holoscan::DeviceId{0},
      });
}

// 2-in / 1-out device binds (DualTensorSourceOp out0/out1 + one op output).
inline void bind_dual_in_single_out_device_pools(
    holoscan::CompileOptions& options, const char* out_port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "out0",
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "out1",
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "op",
          .output_port = out_port,
          .device = holoscan::DeviceId{0},
      });
}

// 3-in / 1-out device binds (TripleTensorSourceOp out0/out1/out2 + one op
// output).
inline void bind_triple_in_single_out_device_pools(
    holoscan::CompileOptions& options, const char* out_port) {
  bind_dual_in_single_out_device_pools(options, out_port);
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "out2",
          .device = holoscan::DeviceId{0},
      });
}

}  // namespace detail

// Unit-test MaxTensorExtents: large enough for Graph suites, small enough
// that fat ops (e.g. CFAR with max_tx×max_rx×D×R map pools) do not OOM at
// cold materialization. Callers with larger scenarios pass explicit extents.
inline MaxTensorExtents default_test_extents() {
  MaxTensorExtents extents;
  extents.max_samples_per_chirp = 512;
  extents.max_chirps = 256;
  extents.max_range_bins = 512;
  extents.max_doppler_bins = 256;
  extents.max_rx = 8;
  extents.max_tx = 4;
  extents.max_interleaved_chirps = 256;
  extents.max_virtual_channels = 64;
  extents.max_channels = 64;
  extents.max_az = 64;
  extents.max_el = 32;
  extents.max_detections = 4096;
  extents.max_points = 4096;
  extents.max_items = 4096;
  extents.max_snapshots = 16;
  extents.max_features = 16;
  extents.max_sources = 8;
  extents.max_clusters = 256;
  extents.max_subarray_channels = 32;
  return extents;
}

// The angle operators reject a scan grid wider than the captured max_az /
// max_el, and those two extents also multiply into the output pool span, so
// grid-scanning suites size them to the grid under test rather than paying
// for the default caps.
inline MaxTensorExtents extents_for_scan_grid(std::size_t num_az,
                                              std::size_t num_el) {
  MaxTensorExtents extents = default_test_extents();
  extents.max_az = std::max<std::int64_t>(
      1, static_cast<std::int64_t>(num_az));
  extents.max_el = std::max<std::int64_t>(
      1, static_cast<std::int64_t>(num_el));
  return extents;
}

template <typename OpT>
auto emplace_harness_op(holoscan::Graph& graph,
                        const MaxTensorExtents& extents) {
  if constexpr (std::is_constructible_v<OpT, const MaxTensorExtents&>) {
    return graph.op<OpT>("op", extents);
  } else {
    return graph.op<OpT>("op");
  }
}

struct GraphRunResult {
  std::vector<HostTensor> frames;
  bool plan_ok{true};
  std::string plan_json;
  holoscan::RunTermination termination{holoscan::RunTermination::kStopped};
  std::vector<std::string> diagnostics;
};

struct DualIOGraphRunResult {
  std::vector<HostTensor> frames0;
  std::vector<HostTensor> frames1;
  bool plan_ok{true};
  std::string plan_json;
  holoscan::RunTermination termination{holoscan::RunTermination::kStopped};
  std::vector<std::string> diagnostics;
};

// 1-in / 3-out Graph result (e.g. FMCWRangeDopplerProcessorOp: rd_maps,
// range_bins, velocity_bins — all emitted / bound in tests).
struct ThreeOutGraphRunResult {
  std::vector<HostTensor> frames0;
  std::vector<HostTensor> frames1;
  std::vector<HostTensor> frames2;
  bool plan_ok{true};
  std::string plan_json;
  holoscan::RunTermination termination{holoscan::RunTermination::kStopped};
  std::vector<std::string> diagnostics;
};

// 1-in / 4-out Graph result (RootMUSIC / ESPRIT: aoa, doa_multi, source_count,
// eigenvalues — always emitted / bound in tests; no angle_spectrum).
struct FourOutGraphRunResult {
  std::vector<HostTensor> frames0;
  std::vector<HostTensor> frames1;
  std::vector<HostTensor> frames2;
  std::vector<HostTensor> frames3;
  bool plan_ok{true};
  std::string plan_json;
  holoscan::RunTermination termination{holoscan::RunTermination::kStopped};
  std::vector<std::string> diagnostics;
};

// 1-in / 5-out Graph result (MVDREstimatorOp: aoa, spectrum, doa_multi,
// source_count, eigenvalues — all always emitted / bound in tests).
struct FiveOutGraphRunResult {
  std::vector<HostTensor> frames0;
  std::vector<HostTensor> frames1;
  std::vector<HostTensor> frames2;
  std::vector<HostTensor> frames3;
  std::vector<HostTensor> frames4;
  bool plan_ok{true};
  std::string plan_json;
  holoscan::RunTermination termination{holoscan::RunTermination::kStopped};
  std::vector<std::string> diagnostics;
};

// 1-in / 1-out Graph runner. `WireFn` receives (graph, source, op, sink) and
// must add_flow source->out → op input and op output → sink->in. Device binds
// use source/out plus op/`out_port`. `extra_op_out_ports` binds additional
// device Tensor outputs that are declared but unwired (e.g. optional range_bins)
// — unbound device producers fail compile with IMPLICIT_DEVICE_ZERO_REJECTED.
template <typename OpT, typename ParamsT, typename WireFn>
GraphRunResult run_single_io_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs, ParamsT params,
    const char* out_port, WireFn&& wire,
    std::initializer_list<const char*> extra_op_out_ports = {},
    const MaxTensorExtents& extents = default_test_extents()) {
  GraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBank::input_frames = std::move(inputs);
  const std::size_t expected = detail::FrameBank::input_frames.size();

  holoscan::Graph graph{"single-io-harness"};
  const auto source = graph.op<detail::TensorSourceOp>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink = graph.op<detail::TensorCaptureOp>("sink");
  wire(graph, source, op, sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_single_io_device_pools(options, out_port);
  for (const char* port : extra_op_out_ports) {
    detail::bind_op_device_output(options, port);
  }

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      // A run diagnostic raises no FrameBank notification, so poll for it
      // instead of waiting once on the full deadline.
      const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      std::unique_lock lock(detail::FrameBank::mutex);
      while (detail::FrameBank::captured.size() < expected &&
             session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        detail::FrameBank::ready.wait_for(lock, detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames = std::move(detail::FrameBank::captured);
  }
  return result;
}

// 1-in / optional Exact override / 1-out runner. `payloads` is positional
// against `inputs`: a nullopt (or short list) leaves that tick's override
// unpublished, which is how a miss and the sticky latch get exercised.
template <typename OpT, typename Payload, typename ParamsT, typename WireFn>
GraphRunResult run_exact_override_single_io_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs,
    std::vector<std::optional<Payload>> payloads, ParamsT params,
    const char* out_port, WireFn&& wire,
    const MaxTensorExtents& extents = default_test_extents()) {
  GraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBank::input_frames = std::move(inputs);
  detail::OverrideBank<Payload>::payloads = std::move(payloads);
  const std::size_t expected = detail::FrameBank::input_frames.size();

  holoscan::Graph graph{"exact-override-single-io-harness"};
  const auto source =
      graph.op<detail::TensorOverrideSourceOp<Payload>>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink = graph.op<detail::TensorCaptureOp>("sink");
  wire(graph, source, op, sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_single_io_device_pools(options, out_port);

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline =
          std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      std::unique_lock lock(detail::FrameBank::mutex);
      while (detail::FrameBank::captured.size() < expected &&
             session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        detail::FrameBank::ready.wait_for(lock, detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames = std::move(detail::FrameBank::captured);
  }
  return result;
}

template <typename OpT>
GraphRunResult run_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs,
    ClutterFilterParams params = {},
    const MaxTensorExtents& extents = default_test_extents()) {
  return run_single_io_operator_graph_result<OpT>(
      std::move(inputs), std::move(params), "range_profiles_out",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink) {
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(source->out, op->range_profiles, q);
        graph.add_flow(op->range_profiles_out, sink->in, q);
      },
      {},
      extents);
}

template <typename OpT>
std::vector<HostTensor> run_operator_graph(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs,
    ClutterFilterParams params = {},
    const MaxTensorExtents& extents = default_test_extents()) {
  const std::size_t expected = inputs.size();
  GraphRunResult result = run_operator_graph_result<OpT>(
      std::move(inputs), std::move(params), extents);
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames.size() != expected) {
    throw std::runtime_error(
        "expected " + std::to_string(expected) + " captured frames, got " +
        std::to_string(result.frames.size()));
  }
  return std::move(result.frames);
}

template <typename OpT>
HostTensor run_operator_once(
    std::shared_ptr<holoscan::Tensor> input,
    ClutterFilterParams params = {},
    const MaxTensorExtents& extents = default_test_extents()) {
  auto frames = run_operator_graph<OpT>(
      {std::move(input)}, std::move(params), extents);
  if (frames.size() != 1U) {
    throw std::runtime_error(
        "expected exactly one captured output frame, got " +
        std::to_string(frames.size()));
  }
  return std::move(frames.front());
}

// 1-in / 2-out Graph runner (e.g. BeamscanEstimatorOp). `WireFn` receives
// (graph, source, op, sink0, sink1) and must add_flow source->out plus both
// op outputs. Device binds use source/out and the named op output ports.
// `extra_op_out_ports` binds additional device Tensor outputs that are
// declared but unwired (e.g. optional is_valid_out) — unbound device
// producers still need bind_tensor_output_device or compile fails with
// IMPLICIT_DEVICE_ZERO_REJECTED.
template <typename OpT, typename ParamsT, typename WireFn>
DualIOGraphRunResult run_single_in_dual_out_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs, ParamsT params,
    const char* out0_port, const char* out1_port, WireFn&& wire,
    std::initializer_list<const char*> extra_op_out_ports = {},
    const MaxTensorExtents& extents = default_test_extents()) {
  DualIOGraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBank::input_frames = std::move(inputs);
  const std::size_t expected = detail::FrameBank::input_frames.size();

  holoscan::Graph graph{"single-in-dual-out-harness"};
  const auto source = graph.op<detail::TensorSourceOp>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink0 = graph.op<detail::TensorCaptureOp>("sink0");
  const auto sink1 = graph.op<detail::TensorCaptureBOp>("sink1");
  wire(graph, source, op, sink0, sink1);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_single_in_dual_out_device_pools(options, out0_port, out1_port);
  for (const char* port : extra_op_out_ports) {
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{
            .operator_path = "op",
            .output_port = port,
            .device = holoscan::DeviceId{0},
        });
  }

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      while (session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        std::size_t n0 = 0;
        std::size_t n1 = 0;
        {
          const std::lock_guard lock0(detail::FrameBank::mutex);
          n0 = detail::FrameBank::captured.size();
        }
        {
          const std::lock_guard lock1(detail::FrameBankB::mutex);
          n1 = detail::FrameBankB::captured.size();
        }
        if (n0 >= expected && n1 >= expected) {
          break;
        }
        std::this_thread::sleep_for(detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames0 = std::move(detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankB::mutex);
    result.frames1 = std::move(detail::FrameBankB::captured);
  }
  return result;
}

// 1-in / 3-out Graph runner (e.g. FMCWRangeDopplerProcessorOp). `WireFn`
// receives (graph, source, op, sink0, sink1, sink2) and must add_flow
// source->out plus all three op outputs. Device binds cover source/out
// and every named op port. `extra_op_out_ports` binds additional device
// Tensor outputs that are declared but unwired.
template <typename OpT, typename ParamsT, typename WireFn>
ThreeOutGraphRunResult run_single_in_three_out_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs, ParamsT params,
    const char* out0_port, const char* out1_port, const char* out2_port,
    WireFn&& wire,
    std::initializer_list<const char*> extra_op_out_ports = {},
    const MaxTensorExtents& extents = default_test_extents()) {
  ThreeOutGraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBankC::reset();
  detail::FrameBank::input_frames = std::move(inputs);
  const std::size_t expected = detail::FrameBank::input_frames.size();

  holoscan::Graph graph{"single-in-three-out-harness"};
  const auto source = graph.op<detail::TensorSourceOp>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink0 = graph.op<detail::TensorCaptureOp>("sink0");
  const auto sink1 = graph.op<detail::TensorCaptureBOp>("sink1");
  const auto sink2 = graph.op<detail::TensorCaptureCOp>("sink2");
  wire(graph, source, op, sink0, sink1, sink2);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_single_in_dual_out_device_pools(options, out0_port, out1_port);
  detail::bind_op_device_output(options, out2_port);
  for (const char* port : extra_op_out_ports) {
    detail::bind_op_device_output(options, port);
  }

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline =
          std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      while (session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        std::size_t n0 = 0;
        std::size_t n1 = 0;
        std::size_t n2 = 0;
        {
          const std::lock_guard lock0(detail::FrameBank::mutex);
          n0 = detail::FrameBank::captured.size();
        }
        {
          const std::lock_guard lock1(detail::FrameBankB::mutex);
          n1 = detail::FrameBankB::captured.size();
        }
        {
          const std::lock_guard lock2(detail::FrameBankC::mutex);
          n2 = detail::FrameBankC::captured.size();
        }
        if (n0 >= expected && n1 >= expected && n2 >= expected) {
          break;
        }
        std::this_thread::sleep_for(detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames0 = std::move(detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankB::mutex);
    result.frames1 = std::move(detail::FrameBankB::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankC::mutex);
    result.frames2 = std::move(detail::FrameBankC::captured);
  }
  return result;
}

// 1-in / 4-out Graph runner (RootMUSICEstimatorOp / ESPRITEstimatorOp).
// `WireFn` receives (graph, source, op, sink0..sink3).
template <typename OpT, typename ParamsT, typename WireFn>
FourOutGraphRunResult run_single_in_four_out_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs, ParamsT params,
    const char* out0_port, const char* out1_port, const char* out2_port,
    const char* out3_port, WireFn&& wire,
    const MaxTensorExtents& extents = default_test_extents()) {
  FourOutGraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBankC::reset();
  detail::FrameBankD::reset();
  detail::FrameBank::input_frames = std::move(inputs);
  const std::size_t expected = detail::FrameBank::input_frames.size();

  holoscan::Graph graph{"single-in-four-out-harness"};
  const auto source = graph.op<detail::TensorSourceOp>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink0 = graph.op<detail::TensorCaptureOp>("sink0");
  const auto sink1 = graph.op<detail::TensorCaptureBOp>("sink1");
  const auto sink2 = graph.op<detail::TensorCaptureCOp>("sink2");
  const auto sink3 = graph.op<detail::TensorCaptureDOp>("sink3");
  wire(graph, source, op, sink0, sink1, sink2, sink3);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_single_in_dual_out_device_pools(options, out0_port, out1_port);
  detail::bind_op_device_output(options, out2_port);
  detail::bind_op_device_output(options, out3_port);

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      while (session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        std::size_t n0 = 0;
        std::size_t n1 = 0;
        std::size_t n2 = 0;
        std::size_t n3 = 0;
        {
          const std::lock_guard lock0(detail::FrameBank::mutex);
          n0 = detail::FrameBank::captured.size();
        }
        {
          const std::lock_guard lock1(detail::FrameBankB::mutex);
          n1 = detail::FrameBankB::captured.size();
        }
        {
          const std::lock_guard lock2(detail::FrameBankC::mutex);
          n2 = detail::FrameBankC::captured.size();
        }
        {
          const std::lock_guard lock3(detail::FrameBankD::mutex);
          n3 = detail::FrameBankD::captured.size();
        }
        if (n0 >= expected && n1 >= expected && n2 >= expected &&
            n3 >= expected) {
          break;
        }
        std::this_thread::sleep_for(detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames0 = std::move(detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankB::mutex);
    result.frames1 = std::move(detail::FrameBankB::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankC::mutex);
    result.frames2 = std::move(detail::FrameBankC::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankD::mutex);
    result.frames3 = std::move(detail::FrameBankD::captured);
  }
  return result;
}

// 1-in / optional Exact override / 4-out Graph runner. The source co-emits
// each Tensor and present override payload at one explicit capture time.
template <typename OpT, typename Payload, typename ParamsT, typename WireFn>
FourOutGraphRunResult run_exact_override_four_out_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs,
    std::vector<std::optional<Payload>> payloads, ParamsT params,
    const char* out0_port, const char* out1_port, const char* out2_port,
    const char* out3_port, WireFn&& wire,
    const MaxTensorExtents& extents = default_test_extents()) {
  FourOutGraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBankC::reset();
  detail::FrameBankD::reset();
  const std::size_t expected = inputs.size();
  detail::FrameBank::input_frames = std::move(inputs);
  detail::OverrideBank<Payload>::payloads = std::move(payloads);

  holoscan::Graph graph{"exact-override-four-out-harness"};
  const auto source =
      graph.op<detail::TensorOverrideSourceOp<Payload>>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink0 = graph.op<detail::TensorCaptureOp>("sink0");
  const auto sink1 = graph.op<detail::TensorCaptureBOp>("sink1");
  const auto sink2 = graph.op<detail::TensorCaptureCOp>("sink2");
  const auto sink3 = graph.op<detail::TensorCaptureDOp>("sink3");
  wire(graph, source, op, sink0, sink1, sink2, sink3);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_single_in_dual_out_device_pools(options, out0_port, out1_port);
  detail::bind_op_device_output(options, out2_port);
  detail::bind_op_device_output(options, out3_port);

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline =
          std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      while (session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        std::size_t n0 = 0;
        std::size_t n1 = 0;
        std::size_t n2 = 0;
        std::size_t n3 = 0;
        {
          const std::lock_guard lock0(detail::FrameBank::mutex);
          n0 = detail::FrameBank::captured.size();
        }
        {
          const std::lock_guard lock1(detail::FrameBankB::mutex);
          n1 = detail::FrameBankB::captured.size();
        }
        {
          const std::lock_guard lock2(detail::FrameBankC::mutex);
          n2 = detail::FrameBankC::captured.size();
        }
        {
          const std::lock_guard lock3(detail::FrameBankD::mutex);
          n3 = detail::FrameBankD::captured.size();
        }
        if (n0 >= expected && n1 >= expected && n2 >= expected &&
            n3 >= expected) {
          break;
        }
        std::this_thread::sleep_for(detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& diagnostic : session.diagnostics()) {
      result.diagnostics.push_back(std::string(diagnostic.token));
    }
  } catch (const holoscan::RuntimeError& error) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + error.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames0 = std::move(detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankB::mutex);
    result.frames1 = std::move(detail::FrameBankB::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankC::mutex);
    result.frames2 = std::move(detail::FrameBankC::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankD::mutex);
    result.frames3 = std::move(detail::FrameBankD::captured);
  }
  return result;
}

// 1-in / 5-out Graph runner (MVDREstimatorOp). `WireFn` receives
// (graph, source, op, sink0..sink4) and must add_flow source->out plus all
// five op outputs. Device binds cover source/out and every named op port.
template <typename OpT, typename ParamsT, typename WireFn>
FiveOutGraphRunResult run_single_in_five_out_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs, ParamsT params,
    const char* out0_port, const char* out1_port, const char* out2_port,
    const char* out3_port, const char* out4_port, WireFn&& wire,
    const MaxTensorExtents& extents = default_test_extents()) {
  FiveOutGraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBankC::reset();
  detail::FrameBankD::reset();
  detail::FrameBankE::reset();
  detail::FrameBank::input_frames = std::move(inputs);
  const std::size_t expected = detail::FrameBank::input_frames.size();

  holoscan::Graph graph{"single-in-five-out-harness"};
  const auto source = graph.op<detail::TensorSourceOp>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink0 = graph.op<detail::TensorCaptureOp>("sink0");
  const auto sink1 = graph.op<detail::TensorCaptureBOp>("sink1");
  const auto sink2 = graph.op<detail::TensorCaptureCOp>("sink2");
  const auto sink3 = graph.op<detail::TensorCaptureDOp>("sink3");
  const auto sink4 = graph.op<detail::TensorCaptureEOp>("sink4");
  wire(graph, source, op, sink0, sink1, sink2, sink3, sink4);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_single_in_dual_out_device_pools(options, out0_port, out1_port);
  detail::bind_op_device_output(options, out2_port);
  detail::bind_op_device_output(options, out3_port);
  detail::bind_op_device_output(options, out4_port);

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      while (session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        std::size_t n0 = 0;
        std::size_t n1 = 0;
        std::size_t n2 = 0;
        std::size_t n3 = 0;
        std::size_t n4 = 0;
        {
          const std::lock_guard lock0(detail::FrameBank::mutex);
          n0 = detail::FrameBank::captured.size();
        }
        {
          const std::lock_guard lock1(detail::FrameBankB::mutex);
          n1 = detail::FrameBankB::captured.size();
        }
        {
          const std::lock_guard lock2(detail::FrameBankC::mutex);
          n2 = detail::FrameBankC::captured.size();
        }
        {
          const std::lock_guard lock3(detail::FrameBankD::mutex);
          n3 = detail::FrameBankD::captured.size();
        }
        {
          const std::lock_guard lock4(detail::FrameBankE::mutex);
          n4 = detail::FrameBankE::captured.size();
        }
        if (n0 >= expected && n1 >= expected && n2 >= expected &&
            n3 >= expected && n4 >= expected) {
          break;
        }
        std::this_thread::sleep_for(detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames0 = std::move(detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankB::mutex);
    result.frames1 = std::move(detail::FrameBankB::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankC::mutex);
    result.frames2 = std::move(detail::FrameBankC::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankD::mutex);
    result.frames3 = std::move(detail::FrameBankD::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankE::mutex);
    result.frames4 = std::move(detail::FrameBankE::captured);
  }
  return result;
}

// 2-in / 2-out Graph runner (e.g. SnapshotGatherOp). `WireFn` receives
// (graph, source, op, sink0, sink1) and must add_flow source->out0/out1 plus
// both op outputs. Device binds use the named ports on "source" and "op".
template <typename OpT, typename ParamsT, typename WireFn>
DualIOGraphRunResult run_dual_io_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs0,
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs1, ParamsT params,
    const char* out0_port, const char* out1_port, WireFn&& wire,
    const MaxTensorExtents& extents = default_test_extents()) {
  DualIOGraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBank::input_frames = std::move(inputs0);
  detail::FrameBankB::input_frames = std::move(inputs1);
  const std::size_t expected = detail::FrameBank::input_frames.size();
  if (detail::FrameBankB::input_frames.size() != expected) {
    result.plan_ok = false;
    result.plan_json = "dual-io harness: input bank sizes differ";
    return result;
  }

  holoscan::Graph graph{"dual-io-harness"};
  const auto source = graph.op<detail::DualTensorSourceOp>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink0 = graph.op<detail::TensorCaptureOp>("sink0");
  const auto sink1 = graph.op<detail::TensorCaptureBOp>("sink1");
  wire(graph, source, op, sink0, sink1);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_dual_io_device_pools(options, out0_port, out1_port);

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      while (session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        std::size_t n0 = 0;
        std::size_t n1 = 0;
        {
          const std::lock_guard lock0(detail::FrameBank::mutex);
          n0 = detail::FrameBank::captured.size();
        }
        {
          const std::lock_guard lock1(detail::FrameBankB::mutex);
          n1 = detail::FrameBankB::captured.size();
        }
        if (n0 >= expected && n1 >= expected) {
          break;
        }
        std::this_thread::sleep_for(detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames0 = std::move(detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(detail::FrameBankB::mutex);
    result.frames1 = std::move(detail::FrameBankB::captured);
  }
  return result;
}

// 2-in / 1-out Graph runner (e.g. DetectionReductionOp without points).
// `WireFn` receives (graph, source, op, sink) and must add_flow source
// out0/out1 → op inputs and op output → sink->in.
template <typename OpT, typename ParamsT, typename WireFn>
GraphRunResult run_dual_in_single_out_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs0,
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs1, ParamsT params,
    const char* out_port, WireFn&& wire,
    const MaxTensorExtents& extents = default_test_extents()) {
  GraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBank::input_frames = std::move(inputs0);
  detail::FrameBankB::input_frames = std::move(inputs1);
  const std::size_t expected = detail::FrameBank::input_frames.size();
  if (detail::FrameBankB::input_frames.size() != expected) {
    result.plan_ok = false;
    result.plan_json = "dual-in single-out harness: input bank sizes differ";
    return result;
  }

  holoscan::Graph graph{"dual-in-single-out-harness"};
  const auto source = graph.op<detail::DualTensorSourceOp>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink = graph.op<detail::TensorCaptureOp>("sink");
  wire(graph, source, op, sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_dual_in_single_out_device_pools(options, out_port);

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      std::unique_lock lock(detail::FrameBank::mutex);
      while (detail::FrameBank::captured.size() < expected &&
             session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        detail::FrameBank::ready.wait_for(lock, detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames = std::move(detail::FrameBank::captured);
  }
  return result;
}

// 3-in / 1-out Graph runner (e.g. DetectionReductionOp with points).
// `WireFn` receives (graph, source, op, sink); source is TripleTensorSourceOp.
template <typename OpT, typename ParamsT, typename WireFn>
GraphRunResult run_triple_in_single_out_operator_graph_result(
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs0,
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs1,
    std::vector<std::shared_ptr<holoscan::Tensor>> inputs2, ParamsT params,
    const char* out_port, WireFn&& wire,
    const MaxTensorExtents& extents = default_test_extents()) {
  GraphRunResult result;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBankC::reset();
  detail::FrameBank::input_frames = std::move(inputs0);
  detail::FrameBankB::input_frames = std::move(inputs1);
  detail::FrameBankC::input_frames = std::move(inputs2);
  const std::size_t expected = detail::FrameBank::input_frames.size();
  if (detail::FrameBankB::input_frames.size() != expected ||
      detail::FrameBankC::input_frames.size() != expected) {
    result.plan_ok = false;
    result.plan_json = "triple-in single-out harness: input bank sizes differ";
    return result;
  }

  holoscan::Graph graph{"triple-in-single-out-harness"};
  const auto source = graph.op<detail::TripleTensorSourceOp>("source");
  auto op = emplace_harness_op<OpT>(graph, extents);
  op.set_params(std::move(params));
  const auto sink = graph.op<detail::TensorCaptureOp>("sink");
  wire(graph, source, op, sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  detail::bind_triple_in_single_out_device_pools(options, out_port);

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    {
      const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
      std::unique_lock lock(detail::FrameBank::mutex);
      while (detail::FrameBank::captured.size() < expected &&
             session.diagnostics().empty() &&
             std::chrono::steady_clock::now() < deadline) {
        detail::FrameBank::ready.wait_for(lock, detail::kHarnessPoll);
      }
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& d : session.diagnostics()) {
      result.diagnostics.push_back(std::string(d.token));
    }
  } catch (const holoscan::RuntimeError& e) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + e.what();
    return result;
  }
  {
    const std::lock_guard lock(detail::FrameBank::mutex);
    result.frames = std::move(detail::FrameBank::captured);
  }
  return result;
}

}  // namespace holoscan::radar::testing
