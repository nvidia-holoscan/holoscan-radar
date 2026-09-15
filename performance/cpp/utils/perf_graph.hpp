// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared HSDK 5 Graph source / sink / runner for performance drivers.
// [EA HSDK5 ONLY] Replaces HSDK 4.x Application helpers in perf_source.hpp /
// perf_sink.hpp. Mid-session POD [live] overrides use a source that co-emits
// the override and Tensor on one clock so the graph stays plannable. EA
// replay requires a sideband sample every tick, so the source emits the
// Params-equivalent value until the measurement frame, then the override.
//
// Pattern:
//   - PerfTensorSourceOp emits one preallocated device Tensor per OnClock tick
//     and records start timestamps into TimingCollector.
//   - PerfTerminalSinkOp drains the timed output, cudaDeviceSynchronize()'s,
//     and records end timestamps.
//   - PerfNullSinkOp drains always-emitted optional outs without timing.
//   - run_perf_graph() compiles + runs until `total_frames` ends are captured.
//   - run_perf_graph*_with_override() uses the co-emit source; the driver
//     must wire override_out and put src/op/sinks in one partition.

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

#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "utils/perf_app.hpp"

namespace holoscan::radar::perf {

inline constexpr auto kPerfTick = std::chrono::milliseconds{1};
// Runs are bounded by lack of progress, not by total wall time: per-frame cost
// spans milliseconds (beamformer) to seconds (Root-MUSIC at the xl rung) and
// scales with the host GPU, so any fixed budget either fails fast hardware's
// large rungs or hides a hang on slow hardware. Callers cap total runtime with
// --stop-after-s.
inline constexpr auto kPerfStall = std::chrono::seconds{60};
inline constexpr auto kPerfPoll = std::chrono::milliseconds{25};

// How PerfTensorSourceOp publishes each tick. Drivers pass this into
// run_perf_graph* — default remains kCopy (stage once, D2D into the loan).
// kFillInPlace / kZeroCopy are diagnostic: fill currently zero-fills the loan;
// zero-copy republish is rejected on this EA (HSDK5_PORT.md §3).
enum class PerfPublishMode {
  kCopy,
  kFillInPlace,
  kZeroCopy,
};

// Effectively unbounded: the perf override source publishes once and the
// measurement frames that follow must all see that value.
inline constexpr auto kPerfOverrideMaxAge = std::chrono::years{100};

// Perf-only operator adapter. Production joins a `[live]` override with Exact{}
// (OPERATOR_API.md). Do not wrap an operator whose production contract already
// declares that selector on the same port — a second selector is
// TEMPORAL_CONTRACT_INVALID. PerfTensorOverrideSourceOp publishes the
// sideband every tick, so Exact matches (perf_clutter_filter). Historical POD
// drivers still use this adapter to restack LatestWithin for a hold; prefer
// Exact when the source already re-publishes. MemberPorts are pointers to
// public Input<T> members on OperatorT.
template <typename OperatorT, auto... MemberPorts>
class PerfLiveContractOp final : public OperatorT {
 public:
  using OperatorT::OperatorT;

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result = OperatorT::contract();
    ((result
          .input(this->*MemberPorts,
                 holoscan::LatestWithin{.max_age = kPerfOverrideMaxAge})
          .on_missing(this->*MemberPorts, holoscan::SafeDefault{})),
     ...);
    return result;
  }
};

struct PerfFrameBank {
  static void reset(std::shared_ptr<holoscan::Tensor> frame, int count,
                    TimingCollector* collector,
                    PerfPublishMode publish = PerfPublishMode::kCopy) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frame);
    input_frame_b = nullptr;
    input_frame_c = nullptr;
    frames_remaining = count;
    captured = 0;
    collector_ = collector;
    publish_mode = publish;
  }

  static void reset_dual(std::shared_ptr<holoscan::Tensor> frame_a,
                         std::shared_ptr<holoscan::Tensor> frame_b, int count,
                         TimingCollector* collector,
                         PerfPublishMode publish = PerfPublishMode::kCopy) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frame_a);
    input_frame_b = std::move(frame_b);
    input_frame_c = nullptr;
    frames_remaining = count;
    captured = 0;
    collector_ = collector;
    publish_mode = publish;
  }

  static void reset_triple(std::shared_ptr<holoscan::Tensor> frame_a,
                           std::shared_ptr<holoscan::Tensor> frame_b,
                           std::shared_ptr<holoscan::Tensor> frame_c, int count,
                           TimingCollector* collector,
                           PerfPublishMode publish = PerfPublishMode::kCopy) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frame_a);
    input_frame_b = std::move(frame_b);
    input_frame_c = std::move(frame_c);
    frames_remaining = count;
    captured = 0;
    collector_ = collector;
    publish_mode = publish;
  }

  // Output spans for the source ports. The runner stages the exact frames
  // before compile(), so each port sizes its pool to the payload it will emit
  // instead of a worst-case constant — a pool costs slot_count x span, and the
  // ladders reach 512 MiB per frame.
  static std::size_t span_a() {
    const std::lock_guard lock(mutex);
    return staged_span(input_frame);
  }

  static std::size_t span_b() {
    const std::lock_guard lock(mutex);
    return staged_span(input_frame_b);
  }

  static std::size_t span_c() {
    const std::lock_guard lock(mutex);
    return staged_span(input_frame_c);
  }

  static std::size_t staged_span(
      const std::shared_ptr<holoscan::Tensor>& frame) {
    if (frame == nullptr || frame->nbytes() <= 0) {
      throw std::runtime_error(
          "PerfFrameBank: source frame must be staged before compile()");
    }
    return static_cast<std::size_t>(frame->nbytes());
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::shared_ptr<holoscan::Tensor> input_frame;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_b;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_c;
  static inline int frames_remaining{0};
  static inline int captured{0};
  static inline TimingCollector* collector_{nullptr};
  static inline PerfPublishMode publish_mode{PerfPublishMode::kCopy};
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

// Zero-copy republish of a staged device Tensor (shared handle, no D2D) is
// rejected under Graph + bind_tensor_output_device with
// RUNTIME_PORT_BINDING_UNAVAILABLE (HSDK5_PORT.md §3). Default path is the
// pool loan + D2D copy. Drivers select via PerfPublishMode on run_perf_graph*.

inline holoscan::expected<void, holoscan::Error> emit_device_zerocopy(
    holoscan::Output<holoscan::Tensor>& out,
    holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame) {
  if (frame == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "PerfTensorSourceOp: null frame"});
  }
  // Tensor copy retains a claim on the same backing allocation; emit() moves
  // that handle into Runtime ownership without a device memcpy.
  return out.emit(
      holoscan::Tensor(*frame),
      holoscan::EmitOptions{.capture_time = context.activation_time()});
}

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

// Diagnostic in-place producer: zero-fills the bound loan. Useful for VAA-
// class ops whose kernels are value-agnostic; not a general substitute for
// the staged deterministic payload (ESPRIT / CFAR / DBSCAN are data-dependent).
inline holoscan::expected<void, holoscan::Error> emit_device_fill_in_place(
    holoscan::Output<holoscan::Tensor>& out,
    holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame) {
  auto loan = out.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = frame->shape(),
                                  .dtype = frame->dtype()});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (cudaMemsetAsync(writer->data(), 0, writer->byte_size(), stream) !=
      cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        "PerfTensorSourceOp: cudaMemsetAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(
      std::move(*loan),
      holoscan::EmitOptions{.capture_time = context.activation_time()});
}

inline holoscan::expected<void, holoscan::Error> emit_device_frame(
    holoscan::Output<holoscan::Tensor>& out,
    holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame,
    PerfPublishMode mode) {
  switch (mode) {
    case PerfPublishMode::kZeroCopy:
      return emit_device_zerocopy(out, context, frame);
    case PerfPublishMode::kFillInPlace:
      return emit_device_fill_in_place(out, context, frame);
    case PerfPublishMode::kCopy:
      return emit_device_copy(out, context, frame);
  }
  return emit_device_copy(out, context, frame);
}

inline std::string stall_message(const char* graph_name, std::size_t captured,
                                 int total_frames) {
  return std::string(graph_name) + " perf: stalled after " +
         std::to_string(captured) + " of " + std::to_string(total_frames) +
         " frames (no progress for " +
         std::to_string(kPerfStall.count()) + "s)";
}

// Waits until every frame is captured, a run diagnostic appears, or the run
// stops making progress for kPerfStall. A short stall window is safe even for
// slow scenarios because it resets on every captured frame; only a graph that
// produces nothing at all trips it. Callers detect the stall from the frame
// count they collected.
inline void wait_for_frames(holoscan::RunSession& session, int total_frames) {
  auto last_progress = std::chrono::steady_clock::now();
  int seen = 0;
  std::unique_lock lock(PerfFrameBank::mutex);
  while (PerfFrameBank::captured < total_frames &&
         session.diagnostics().empty()) {
    PerfFrameBank::ready.wait_for(lock, kPerfPoll);
    const auto now = std::chrono::steady_clock::now();
    if (PerfFrameBank::captured != seen) {
      seen = PerfFrameBank::captured;
      last_progress = now;
    } else if (now - last_progress >= kPerfStall) {
      return;
    }
  }
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
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out, "out")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_a()));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    PerfPublishMode mode = PerfPublishMode::kCopy;
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      frame = PerfFrameBank::input_frame;
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    return detail::emit_device_frame(out, context, frame, mode);
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
};

// Tensor source with a scalar/POD sideband on the same clock. It publishes
// the configured initial value before `override_at`, then the override.
// Publishing both outputs every activation satisfies replay. Callers must
// put source, operator, and sinks in one explicit partition (independent
// OnClock roots otherwise fail compile). Enum sidebands bind since the
// kLocalEnum fix (HSDK5_PORT.md §12). ClutterFilterOp.method uses this
// path for override_ms; other enum lives are not yet wired (§8).
template <typename Value>
class PerfTensorOverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfTensorOverrideSourceOp(int64_t override_at, Value initial_value,
                             Value override_value)
      : override_at_(override_at),
        initial_value_(std::move(initial_value)),
        override_value_(std::move(override_value)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out, "out")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_a()));
    spec.output(override_out, "override_out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    PerfPublishMode mode = PerfPublishMode::kCopy;
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      frame = PerfFrameBank::input_frame;
      mode = PerfFrameBank::publish_mode;
    }
    const Value& value =
        frame_index_ < override_at_ ? initial_value_ : override_value_;
    HOLOSCAN_RETURN_IF_ERROR(override_out.emit(
        value,
        holoscan::EmitOptions{.capture_time = context.activation_time()}));
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      detail::record_start_locked();
    }
    ++frame_index_;
    return detail::emit_device_frame(out, context, frame, mode);
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
  holoscan::Output<Value> override_out{"override_out"};

 private:
  int64_t override_at_{-1};
  Value initial_value_;
  Value override_value_;
  int64_t frame_index_{0};
};

// Dual-output source: both tensors emit on the same OnClock tick (needed for
// OnAligned Exact consumers such as SnapshotGatherOp).
class PerfDualTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out_a, "out_a")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_a()));
    spec.output(out_b, "out_b")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_b()));
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
    PerfPublishMode mode = PerfPublishMode::kCopy;
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
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(out_a, context, a, mode));
    return detail::emit_device_frame(out_b, context, b, mode);
  }

  holoscan::Output<holoscan::Tensor> out_a{"out_a"};
  holoscan::Output<holoscan::Tensor> out_b{"out_b"};
};

template <typename Value>
class PerfDualTensorOverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfDualTensorOverrideSourceOp(int64_t override_at, Value initial_value,
                                 Value override_value)
      : override_at_(override_at),
        initial_value_(std::move(initial_value)),
        override_value_(std::move(override_value)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out_a, "out_a")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_a()));
    spec.output(out_b, "out_b")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_b()));
    spec.output(override_out, "override_out");
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
    PerfPublishMode mode = PerfPublishMode::kCopy;
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
      mode = PerfFrameBank::publish_mode;
    }
    const Value& value =
        frame_index_ < override_at_ ? initial_value_ : override_value_;
    HOLOSCAN_RETURN_IF_ERROR(override_out.emit(
        value,
        holoscan::EmitOptions{.capture_time = context.activation_time()}));
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      detail::record_start_locked();
    }
    ++frame_index_;
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(out_a, context, a, mode));
    return detail::emit_device_frame(out_b, context, b, mode);
  }

  holoscan::Output<holoscan::Tensor> out_a{"out_a"};
  holoscan::Output<holoscan::Tensor> out_b{"out_b"};
  holoscan::Output<Value> override_out{"override_out"};

 private:
  int64_t override_at_{-1};
  Value initial_value_;
  Value override_value_;
  int64_t frame_index_{0};
};

// Triple-output source for DetectionReductionOp (labels / idx / points).
class PerfTripleTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out_a, "out_a")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_a()));
    spec.output(out_b, "out_b")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_b()));
    spec.output(out_c, "out_c")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, PerfFrameBank::span_c()));
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
    PerfPublishMode mode = PerfPublishMode::kCopy;
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
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(out_a, context, a, mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(out_b, context, b, mode));
    return detail::emit_device_frame(out_c, context, c, mode);
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

// Timed terminal sink for two exact-aligned outputs from one operator.
class PerfDualTerminalSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in0, "in0").queue_depth(8U);
    spec.input(in1, "in1").queue_depth(8U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::ports(in0, in1),
        .window = std::chrono::nanoseconds{0}});
    result.input(in0, holoscan::Exact{})
        .input(in1, holoscan::Exact{});
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

  holoscan::Input<holoscan::Tensor> in0{"in0"};
  holoscan::Input<holoscan::Tensor> in1{"in1"};
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

// Untimed drain for a generated schema payload.
template <typename Payload>
class PerfPayloadSinkOp final : public holoscan::Operator<> {
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

  holoscan::Input<Payload> in{"in"};
};

// BuildFn signature:
//   void(holoscan::Graph& graph,
//        const auto& src,   // shared_ptr-like op handle from graph.op
//        holoscan::CompileOptions& options)
// Caller adds the operator under test, sinks, flows, and bind_tensor_out for
// every device-tensor producer besides "src"/"out" (already bound).
template <typename BuildFn>
TimingCollector run_perf_graph(
    std::shared_ptr<holoscan::Tensor> input, int total_frames,
    const char* graph_name, BuildFn build,
    PerfPublishMode publish = PerfPublishMode::kCopy) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset(std::move(input), total_frames, &collector, publish);

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
  detail::wait_for_frames(session, total_frames);
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
    throw std::runtime_error(detail::stall_message(
        graph_name, collector.ends.size(), total_frames));
  }
  return collector;
}

// BuildFn receives a co-emitting source with `.out` and `.override_out`. It
// must wire both and author one exact partition containing every graph op.
template <typename Value, typename BuildFn>
TimingCollector run_perf_graph_with_override(
    std::shared_ptr<holoscan::Tensor> input, int total_frames,
    int64_t override_at, Value initial_value, Value override_value,
    const char* graph_name, BuildFn build,
    PerfPublishMode publish = PerfPublishMode::kCopy) {
  if (override_at < 0 || override_at >= total_frames) {
    throw std::invalid_argument(
        "run_perf_graph_with_override: override_at outside frame plan");
  }
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset(std::move(input), total_frames, &collector, publish);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfTensorOverrideSourceOp<Value>>(
      "src", override_at, std::move(initial_value),
      std::move(override_value));
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
  detail::wait_for_frames(session, total_frames);
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
    throw std::runtime_error(detail::stall_message(
        graph_name, collector.ends.size(), total_frames));
  }
  return collector;
}

template <typename BuildFn>
TimingCollector run_perf_graph_dual(
    std::shared_ptr<holoscan::Tensor> input_a,
    std::shared_ptr<holoscan::Tensor> input_b, int total_frames,
    const char* graph_name, BuildFn build,
    PerfPublishMode publish = PerfPublishMode::kCopy) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_dual(std::move(input_a), std::move(input_b),
                            total_frames, &collector, publish);

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
  detail::wait_for_frames(session, total_frames);
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
    throw std::runtime_error(detail::stall_message(
        graph_name, collector.ends.size(), total_frames));
  }
  return collector;
}

template <typename Value, typename BuildFn>
TimingCollector run_perf_graph_dual_with_override(
    std::shared_ptr<holoscan::Tensor> input_a,
    std::shared_ptr<holoscan::Tensor> input_b, int total_frames,
    int64_t override_at, Value initial_value, Value override_value,
    const char* graph_name, BuildFn build,
    PerfPublishMode publish = PerfPublishMode::kCopy) {
  if (override_at < 0 || override_at >= total_frames) {
    throw std::invalid_argument(
        "run_perf_graph_dual_with_override: override_at outside frame plan");
  }
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_dual(std::move(input_a), std::move(input_b),
                            total_frames, &collector, publish);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfDualTensorOverrideSourceOp<Value>>(
      "src", override_at, std::move(initial_value),
      std::move(override_value));
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
  detail::wait_for_frames(session, total_frames);
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
    throw std::runtime_error(detail::stall_message(
        graph_name, collector.ends.size(), total_frames));
  }
  return collector;
}

template <typename BuildFn>
TimingCollector run_perf_graph_triple(
    std::shared_ptr<holoscan::Tensor> input_a,
    std::shared_ptr<holoscan::Tensor> input_b,
    std::shared_ptr<holoscan::Tensor> input_c, int total_frames,
    const char* graph_name, BuildFn build,
    PerfPublishMode publish = PerfPublishMode::kCopy) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_triple(std::move(input_a), std::move(input_b),
                              std::move(input_c), total_frames, &collector,
                              publish);

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
  detail::wait_for_frames(session, total_frames);
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
    throw std::runtime_error(detail::stall_message(
        graph_name, collector.ends.size(), total_frames));
  }
  return collector;
}

}  // namespace holoscan::radar::perf
