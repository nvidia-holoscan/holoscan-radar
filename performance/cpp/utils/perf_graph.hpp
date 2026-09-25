// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared HSDK 5 Graph source / sink / runner for performance drivers.
// Mid-session [live] overrides use a source that publishes the sideband and
// Tensor on one clock, once on the changed frame. The next frame uses Params.
//
// Pattern:
//   - PerfTensorSourceOp emits one device Tensor at a time and records start
//     timestamps into TimingCollector.
//   - PerfTerminalSinkOp drains the timed output, cudaDeviceSynchronize()'s,
//     and records end timestamps.
//   - PerfNullSinkOp drains always-emitted optional outs without timing.
//   - run_perf_graph() compiles + runs until `total_frames` ends are captured.
//   - run_perf_graph*_with_override() can emit one Exact sideband on the
//     changed frame; the driver wires override_out in one partition.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <complex>
#include <type_traits>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
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

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/perf_app.hpp"

namespace holoscan::radar::perf {

// The sink acknowledgement holds the source to one frame in flight. A short
// clock period keeps source cadence from capping operator throughput.
inline constexpr auto kPerfTick = std::chrono::microseconds{1};
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
// zero-copy support depends on the active Graph output binding.
enum class PerfPublishMode {
  kCopy,
  kFillInPlace,
  kZeroCopy,
};

struct PerfFrameBank {
  static void reset(std::shared_ptr<holoscan::Tensor> frame, int count, TimingCollector* collector,
                    PerfPublishMode publish = PerfPublishMode::kCopy) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frame);
    input_frame_b = nullptr;
    input_frame_c = nullptr;
    input_frame_d = nullptr;
    input_frame_e = nullptr;
    input_frame_f = nullptr;
    frames_remaining = count;
    captured = 0;
    in_flight = 0;
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
    input_frame_d = nullptr;
    input_frame_e = nullptr;
    input_frame_f = nullptr;
    frames_remaining = count;
    captured = 0;
    in_flight = 0;
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
    input_frame_d = nullptr;
    input_frame_e = nullptr;
    input_frame_f = nullptr;
    frames_remaining = count;
    captured = 0;
    in_flight = 0;
    collector_ = collector;
    publish_mode = publish;
  }

  static void reset_six(std::array<std::shared_ptr<holoscan::Tensor>, 6> frames, int count,
                        TimingCollector* collector,
                        PerfPublishMode publish = PerfPublishMode::kCopy) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frames[0]);
    input_frame_b = std::move(frames[1]);
    input_frame_c = std::move(frames[2]);
    input_frame_d = std::move(frames[3]);
    input_frame_e = std::move(frames[4]);
    input_frame_f = std::move(frames[5]);
    frames_remaining = count;
    captured = 0;
    in_flight = 0;
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

  static std::size_t span_d() {
    const std::lock_guard lock(mutex);
    return staged_span(input_frame_d);
  }
  static std::size_t span_e() {
    const std::lock_guard lock(mutex);
    return staged_span(input_frame_e);
  }
  static std::size_t span_f() {
    const std::lock_guard lock(mutex);
    return staged_span(input_frame_f);
  }

  static std::size_t staged_span(const std::shared_ptr<holoscan::Tensor>& frame) {
    if (frame == nullptr || frame->nbytes() <= 0) {
      throw std::runtime_error("PerfFrameBank: source frame must be staged before compile()");
    }
    return static_cast<std::size_t>(frame->nbytes());
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::shared_ptr<holoscan::Tensor> input_frame;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_b;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_c;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_d;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_e;
  static inline std::shared_ptr<holoscan::Tensor> input_frame_f;
  static inline int frames_remaining{0};
  static inline int captured{0};
  static inline int in_flight{0};
  static inline TimingCollector* collector_{nullptr};
  static inline PerfPublishMode publish_mode{PerfPublishMode::kCopy};
};

inline void bind_tensor_out(holoscan::CompileOptions& options, const char* operator_path,
                            const char* output_port) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = operator_path,
      .output_port = output_port,
      .device = holoscan::DeviceId{0},
  });
}

namespace detail {

// Zero-copy republish of a staged device Tensor (shared handle, no D2D) is
// optional under Graph output binding. The default path uses a pool loan plus
// D2D copy; drivers select the publication mode through PerfPublishMode.

inline holoscan::expected<void, holoscan::Error> emit_device_zerocopy(
    holoscan::Output<holoscan::Tensor>& out, holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame) {
  if (frame == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "PerfTensorSourceOp: null frame"});
  }
  // Tensor copy retains a claim on the same backing allocation; emit() moves
  // that handle into Runtime ownership without a device memcpy.
  return out.emit(holoscan::Tensor(*frame),
                  holoscan::EmitOptions{.capture_time = context.activation_time()});
}

inline holoscan::expected<void, holoscan::Error> emit_device_copy(
    holoscan::Output<holoscan::Tensor>& out, holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame) {
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
  if (cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(), cudaMemcpyDeviceToDevice,
                      stream) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure,
                                                     "PerfTensorSourceOp: cudaMemcpyAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(std::move(*loan),
                  holoscan::EmitOptions{.capture_time = context.activation_time()});
}

// Diagnostic in-place producer: zero-fills the bound loan. Useful for VAA-
// class ops whose kernels are value-agnostic; not a general substitute for
// the staged deterministic payload (ESPRIT / CFAR / DBSCAN are data-dependent).
inline holoscan::expected<void, holoscan::Error> emit_device_fill_in_place(
    holoscan::Output<holoscan::Tensor>& out, holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame) {
  auto loan = out.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = frame->shape(), .dtype = frame->dtype()});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (cudaMemsetAsync(writer->data(), 0, writer->byte_size(), stream) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure,
                                                     "PerfTensorSourceOp: cudaMemsetAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(std::move(*loan),
                  holoscan::EmitOptions{.capture_time = context.activation_time()});
}

inline holoscan::expected<void, holoscan::Error> emit_device_frame(
    holoscan::Output<holoscan::Tensor>& out, holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame, PerfPublishMode mode) {
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

inline std::string stall_message(const char* graph_name, std::size_t captured, int total_frames) {
  return std::string(graph_name) + " perf: stalled after " + std::to_string(captured) + " of " +
         std::to_string(total_frames) + " frames (no progress for " +
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
  while (PerfFrameBank::captured < total_frames && session.diagnostics().empty()) {
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

inline void record_live_emission(int64_t frame_index) {
  const std::lock_guard lock(PerfFrameBank::mutex);
  if (PerfFrameBank::collector_ != nullptr)
    PerfFrameBank::collector_->live_emissions.push_back(frame_index);
}

inline void record_start_locked() {
  if (PerfFrameBank::collector_ != nullptr) {
    if (!PerfFrameBank::collector_->first_tick_set) {
      PerfFrameBank::collector_->t_first_tick = Clock::now();
      PerfFrameBank::collector_->first_tick_set = true;
    }
    PerfFrameBank::collector_->starts.push_back(Clock::now());
    ++PerfFrameBank::in_flight;
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
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_a()));
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
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0 ||
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
// the configured initial value on ordinary frames or, in one-shot mode,
// publishes only the changed value at `override_at`. Callers must
// put source, operator, and sinks in one explicit partition (independent
// OnClock roots otherwise fail compile). ClutterFilterOp.method uses this path
// for override_ms; other enum-valued drivers can reuse the same source.
template <typename Value>
class PerfTensorOverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfTensorOverrideSourceOp(int64_t override_at, Value initial_value, Value override_value,
                             bool one_shot, bool prime_changed)
  : override_at_(override_at),
    initial_value_(std::move(initial_value)),
    override_value_(std::move(override_value)),
    one_shot_(one_shot),
    prime_changed_(prime_changed) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out, "out")
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_a()));
    if constexpr (std::is_same_v<Value, WireFMCWWaveform>) {
      constexpr holoscan::TensorRepresentation kWaveform{
          .memory_kind = holoscan::MemoryKind::kHost,
          .dtype = DLDataType{kDLComplex, 128U, 1U},
          .rank = 1U,
      };
      const auto reference_bytes = [](const WireFMCWWaveform& waveform) {
        return waveform.reference_samples
                   ? static_cast<std::size_t>(waveform.reference_samples->nbytes())
                   : 0U;
      };
      const std::size_t max_bytes =
          std::max({sizeof(std::complex<double>), reference_bytes(initial_value_),
                    reference_bytes(override_value_)});
      spec.output(override_out, "override_out")
          .max_emits_per_compute(1U)
          .produces_tensor(holoscan::radar::pooled_tensor_output(kWaveform, 1U, max_bytes));
    } else {
      spec.output(override_out, "override_out");
    }
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
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      frame = PerfFrameBank::input_frame;
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    if (!one_shot_ || frame_index_ == override_at_ || (prime_changed_ && frame_index_ == 0)) {
      const Value& value = (frame_index_ == override_at_ || (prime_changed_ && frame_index_ == 0))
                               ? override_value_
                               : initial_value_;
      if constexpr (std::is_same_v<Value, WireFMCWWaveform>) {
        auto reference = wire_bridge_detail::read_host_vector<std::complex<double>>(
            value.reference_samples, DLDataType{kDLComplex, 128U, 1U}, "reference_samples");
        if (!reference)
          return holoscan::make_unexpected(std::move(reference).error());
        const std::array<int64_t, 1> shape{static_cast<int64_t>(reference->size())};
        auto loan = override_out.allocate_tensor(
            holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLComplex, 128U, 1U}});
        if (!loan)
          return holoscan::make_unexpected(std::move(loan).error());
        auto written = loan->write_host_and_commit(
            [&reference](const holoscan::TensorOutputWriteGuard& writer) {
              std::copy(reference->begin(), reference->end(),
                        writer.data_as<std::complex<double>>());
            });
        if (!written)
          return holoscan::make_unexpected(std::move(written).error());
        HOLOSCAN_RETURN_IF_ERROR(override_out.emit_tensor(
            std::move(*loan), Value(value),
            holoscan::EmitOptions{.capture_time = context.activation_time()}));
      } else {
        HOLOSCAN_RETURN_IF_ERROR(override_out.emit(
            value, holoscan::EmitOptions{.capture_time = context.activation_time()}));
      }
      detail::record_live_emission(frame_index_);
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
  bool one_shot_{false};
  bool prime_changed_{false};
  int64_t frame_index_{0};
};

// Descriptor override counterpart to PerfTensorOverrideSourceOp. WindowSpec
// rides a host Tensor-backed wire payload, so a plain local-value output is
// insufficient for HSDK 5 port binding.
class PerfTensorWindowOverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfTensorWindowOverrideSourceOp(int64_t override_at, WindowSpec initial_value,
                                   WindowSpec override_value, bool one_shot)
  : override_at_(override_at),
    initial_value_(std::move(initial_value)),
    override_value_(std::move(override_value)),
    one_shot_(one_shot) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{.memory_kind =
                                                         holoscan::MemoryKind::kCudaDevice};
    constexpr holoscan::TensorRepresentation kWindow{.memory_kind = holoscan::MemoryKind::kHost,
                                                     .dtype = DLDataType{kDLFloat, 64U, 1U},
                                                     .rank = 1U};
    spec.output(out).produces_tensor(
        holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_a()));
    spec.output(override_out)
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kWindow, 1U, 4096U * sizeof(double)));
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
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      frame = PerfFrameBank::input_frame;
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    if (!one_shot_ || frame_index_ == override_at_) {
      const WindowSpec& value = frame_index_ == override_at_ ? override_value_ : initial_value_;
      const std::array<int64_t, 1> shape{static_cast<int64_t>(value.coefficients.size())};
      auto loan = override_out.allocate_tensor(
          holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLFloat, 64U, 1U}});
      if (!loan) {
        return holoscan::make_unexpected(std::move(loan).error());
      }
      const std::span<const double> coefficients{value.coefficients};
      auto written = loan->write_host_and_commit(
          [coefficients](const holoscan::TensorOutputWriteGuard& writer) {
            std::copy(coefficients.begin(), coefficients.end(), writer.data_as<double>());
          });
      if (!written) {
        return holoscan::make_unexpected(std::move(written).error());
      }
      HOLOSCAN_RETURN_IF_ERROR(override_out.emit_tensor(
          std::move(*loan), holoscan::radar::to_wire(value),
          holoscan::EmitOptions{.capture_time = context.activation_time()}));
      detail::record_live_emission(frame_index_);
    }
    ++frame_index_;
    return detail::emit_device_frame(out, context, frame, mode);
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
  holoscan::Output<wire::WindowSpecT> override_out{"override_out"};

 private:
  int64_t override_at_{-1};
  WindowSpec initial_value_;
  WindowSpec override_value_;
  bool one_shot_{false};
  int64_t frame_index_{0};
};

// One-frame pageable-host vector sideband for live Tensor-valued parameters.
template <typename Element>
class PerfTensorVectorOverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfTensorVectorOverrideSourceOp(int64_t override_at, std::vector<Element> initial,
                                   std::vector<Element> changed, bool one_shot)
  : override_at_(override_at),
    initial_(std::move(initial)),
    changed_(std::move(changed)),
    one_shot_(one_shot) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr DLDataType kDtype = std::is_same_v<Element, double> ? DLDataType{kDLFloat, 64U, 1U}
                                                                  : DLDataType{kDLInt, 64U, 1U};
    constexpr holoscan::TensorRepresentation kDevice{.memory_kind =
                                                         holoscan::MemoryKind::kCudaDevice};
    constexpr holoscan::TensorRepresentation kHost{
        .memory_kind = holoscan::MemoryKind::kHost, .dtype = kDtype, .rank = 1U};
    spec.output(out).produces_tensor(
        holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_a()));
    const std::size_t bytes = std::max(initial_.size(), changed_.size()) * sizeof(Element);
    spec.output(override_out)
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kHost, 1U, std::max(bytes, sizeof(Element))));
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
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr)
        return {};
      --PerfFrameBank::frames_remaining;
      frame = PerfFrameBank::input_frame;
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    if (!one_shot_ || frame_index_ == override_at_) {
      constexpr DLDataType kDtype = std::is_same_v<Element, double> ? DLDataType{kDLFloat, 64U, 1U}
                                                                    : DLDataType{kDLInt, 64U, 1U};
      const auto& values = frame_index_ == override_at_ ? changed_ : initial_;
      const std::array<int64_t, 1> shape{static_cast<int64_t>(values.size())};
      auto loan = override_out.allocate_tensor(
          holoscan::TensorLoanRequest{.shape = shape, .dtype = kDtype});
      if (!loan)
        return holoscan::make_unexpected(std::move(loan).error());
      auto written =
          loan->write_host_and_commit([&values](const holoscan::TensorOutputWriteGuard& writer) {
            std::copy(values.begin(), values.end(), writer.data_as<Element>());
          });
      if (!written)
        return holoscan::make_unexpected(std::move(written).error());
      HOLOSCAN_RETURN_IF_ERROR(override_out.emit(
          std::move(*loan), holoscan::EmitOptions{.capture_time = context.activation_time()}));
      detail::record_live_emission(frame_index_);
    }
    ++frame_index_;
    return detail::emit_device_frame(out, context, frame, mode);
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
  holoscan::Output<holoscan::Tensor> override_out{"override_out"};

 private:
  int64_t override_at_{-1};
  std::vector<Element> initial_;
  std::vector<Element> changed_;
  bool one_shot_{false};
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
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_a()));
    spec.output(out_b, "out_b")
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_b()));
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
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr || PerfFrameBank::input_frame_b == nullptr) {
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
  PerfDualTensorOverrideSourceOp(int64_t override_at, Value initial_value, Value override_value,
                                 bool one_shot)
  : override_at_(override_at),
    initial_value_(std::move(initial_value)),
    override_value_(std::move(override_value)),
    one_shot_(one_shot) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out_a, "out_a")
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_a()));
    spec.output(out_b, "out_b")
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_b()));
    if constexpr (std::is_same_v<Value, WireFMCWWaveform>) {
      constexpr holoscan::TensorRepresentation kWaveform{
          .memory_kind = holoscan::MemoryKind::kHost,
          .dtype = DLDataType{kDLComplex, 128U, 1U},
          .rank = 1U,
      };
      const auto reference_bytes = [](const WireFMCWWaveform& waveform) {
        return waveform.reference_samples
                   ? static_cast<std::size_t>(waveform.reference_samples->nbytes())
                   : 0U;
      };
      const std::size_t max_bytes =
          std::max({sizeof(std::complex<double>), reference_bytes(initial_value_),
                    reference_bytes(override_value_)});
      spec.output(override_out, "override_out")
          .max_emits_per_compute(1U)
          .produces_tensor(holoscan::radar::pooled_tensor_output(kWaveform, 1U, max_bytes));
    } else {
      spec.output(override_out, "override_out");
    }
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
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr || PerfFrameBank::input_frame_b == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      a = PerfFrameBank::input_frame;
      b = PerfFrameBank::input_frame_b;
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    if (!one_shot_ || frame_index_ == override_at_) {
      const Value& value = frame_index_ == override_at_ ? override_value_ : initial_value_;
      if constexpr (std::is_same_v<Value, WireFMCWWaveform>) {
        auto reference = wire_bridge_detail::read_host_vector<std::complex<double>>(
            value.reference_samples, DLDataType{kDLComplex, 128U, 1U}, "reference_samples");
        if (!reference)
          return holoscan::make_unexpected(std::move(reference).error());
        const std::array<int64_t, 1> shape{static_cast<int64_t>(reference->size())};
        auto loan = override_out.allocate_tensor(
            holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLComplex, 128U, 1U}});
        if (!loan)
          return holoscan::make_unexpected(std::move(loan).error());
        auto written = loan->write_host_and_commit(
            [&reference](const holoscan::TensorOutputWriteGuard& writer) {
              std::copy(reference->begin(), reference->end(),
                        writer.data_as<std::complex<double>>());
            });
        if (!written)
          return holoscan::make_unexpected(std::move(written).error());
        HOLOSCAN_RETURN_IF_ERROR(override_out.emit_tensor(
            std::move(*loan), Value(value),
            holoscan::EmitOptions{.capture_time = context.activation_time()}));
      } else {
        HOLOSCAN_RETURN_IF_ERROR(override_out.emit(
            value, holoscan::EmitOptions{.capture_time = context.activation_time()}));
      }
      detail::record_live_emission(frame_index_);
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
  bool one_shot_{false};
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
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_a()));
    spec.output(out_b, "out_b")
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_b()));
    spec.output(out_c, "out_c")
        .produces_tensor(
            holoscan::radar::pooled_tensor_output(kDevice, 8U, PerfFrameBank::span_c()));
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
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr || PerfFrameBank::input_frame_b == nullptr ||
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
// MultiSourceExpansionOp has six data inputs. Keep them on the same clock and
// reuse the same one-frame-in-flight timing discipline as the smaller sources.
class PerfSixTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation device{.memory_kind =
                                                        holoscan::MemoryKind::kCudaDevice};
    spec.output(a).produces_tensor(
        radar::pooled_tensor_output(device, 8U, PerfFrameBank::span_a()));
    spec.output(b).produces_tensor(
        radar::pooled_tensor_output(device, 8U, PerfFrameBank::span_b()));
    spec.output(c).produces_tensor(
        radar::pooled_tensor_output(device, 8U, PerfFrameBank::span_c()));
    spec.output(d).produces_tensor(
        radar::pooled_tensor_output(device, 8U, PerfFrameBank::span_d()));
    spec.output(e).produces_tensor(
        radar::pooled_tensor_output(device, 8U, PerfFrameBank::span_e()));
    spec.output(f).produces_tensor(
        radar::pooled_tensor_output(device, 8U, PerfFrameBank::span_f()));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::array<std::shared_ptr<holoscan::Tensor>, 6> frames;
    PerfPublishMode mode = PerfPublishMode::kCopy;
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0)
        return {};
      --PerfFrameBank::frames_remaining;
      frames = {PerfFrameBank::input_frame,   PerfFrameBank::input_frame_b,
                PerfFrameBank::input_frame_c, PerfFrameBank::input_frame_d,
                PerfFrameBank::input_frame_e, PerfFrameBank::input_frame_f};
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(a, context, frames[0], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(b, context, frames[1], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(c, context, frames[2], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(d, context, frames[3], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(e, context, frames[4], mode));
    return detail::emit_device_frame(f, context, frames[5], mode);
  }

  holoscan::Output<holoscan::Tensor> a{"a"};
  holoscan::Output<holoscan::Tensor> b{"b"};
  holoscan::Output<holoscan::Tensor> c{"c"};
  holoscan::Output<holoscan::Tensor> d{"d"};
  holoscan::Output<holoscan::Tensor> e{"e"};
  holoscan::Output<holoscan::Tensor> f{"f"};
};

// Shared source for adapters that have several differently typed live ports.
// Sidebands declares the output ports and converts the selected YAML value;
// this source owns the clock, frame publication, and one-frame transition.
template <typename Sidebands, auto... DataPorts>
class PerfLiveSourceOp final : public holoscan::Operator<>, public Sidebands {
 public:
  static_assert(sizeof...(DataPorts) >= 1 && sizeof...(DataPorts) <= 3);

  template <typename... Args>
  explicit PerfLiveSourceOp(int64_t override_at, Args&&... args)
  : Sidebands(std::forward<Args>(args)...), override_at_(override_at) {}

  void setup(holoscan::OperatorSpec& spec) override { Sidebands::setup(spec); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::array<std::shared_ptr<holoscan::Tensor>, sizeof...(DataPorts)> frames;
    PerfPublishMode mode = PerfPublishMode::kCopy;
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr ||
          (frames.size() > 1 && PerfFrameBank::input_frame_b == nullptr) ||
          (frames.size() > 2 && PerfFrameBank::input_frame_c == nullptr))
        return {};
      --PerfFrameBank::frames_remaining;
      frames[0] = PerfFrameBank::input_frame;
      if constexpr (sizeof...(DataPorts) > 1)
        frames[1] = PerfFrameBank::input_frame_b;
      if constexpr (sizeof...(DataPorts) > 2)
        frames[2] = PerfFrameBank::input_frame_c;
      mode = PerfFrameBank::publish_mode;
      detail::record_start_locked();
    }
    if (frame_index_ == override_at_) {
      HOLOSCAN_RETURN_IF_ERROR(Sidebands::emit_override(context));
      detail::record_live_emission(frame_index_);
    }
    ++frame_index_;
    return emit_data<0, DataPorts...>(context, frames, mode);
  }

 private:
  template <size_t Index, auto Port, auto... Rest>
  holoscan::expected<void, holoscan::Error> emit_data(
      holoscan::ExecutionContext& context,
      const std::array<std::shared_ptr<holoscan::Tensor>, sizeof...(DataPorts)>& frames,
      PerfPublishMode mode) {
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(this->*Port, context, frames[Index], mode));
    if constexpr (sizeof...(Rest) > 0)
      return emit_data<Index + 1, Rest...>(context, frames, mode);
    return {};
  }

  int64_t override_at_{-1};
  int64_t frame_index_{0};
};

class PerfTerminalSinkOp final : public holoscan::Operator<> {
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
    cudaDeviceSynchronize();
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::collector_ != nullptr) {
        PerfFrameBank::collector_->ends.push_back(Clock::now());
      }
      ++PerfFrameBank::captured;
      --PerfFrameBank::in_flight;
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
    cudaDeviceSynchronize();
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::collector_ != nullptr) {
        PerfFrameBank::collector_->ends.push_back(Clock::now());
      }
      ++PerfFrameBank::captured;
      --PerfFrameBank::in_flight;
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
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

// Untimed drain for a generated schema payload.
template <typename Payload>
class PerfPayloadSinkOp final : public holoscan::Operator<> {
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
// Compile, run, and verify a staged source -> operator -> sink Graph.
inline void execute_perf_graph(holoscan::Graph& graph, holoscan::CompileOptions options,
                               int total_frames, const char* graph_name,
                               TimingCollector* collector) {
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok())
    throw std::runtime_error(std::string(graph_name) +
                             " perf compile failed: " + std::string(plan.json()));
  holoscan::RunSession session = holoscan::run_async(plan);
  detail::wait_for_frames(session, total_frames);
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string message = std::string(graph_name) + " perf run diagnostics:";
    for (const auto& diagnostic : session.diagnostics()) {
      message += " ";
      message += std::string(diagnostic.token);
    }
    throw std::runtime_error(message);
  }
  if (collector->ends.size() < static_cast<size_t>(total_frames))
    throw std::runtime_error(
        detail::stall_message(graph_name, collector->ends.size(), total_frames));
}

template <typename BuildFn>
TimingCollector run_perf_graph(std::shared_ptr<holoscan::Tensor> input, int total_frames,
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
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

// BuildFn receives a co-emitting source with `.out` and `.override_out`. It
// must wire both and author one exact partition containing every graph op.
template <typename Value, typename BuildFn>
TimingCollector run_perf_graph_with_override(std::shared_ptr<holoscan::Tensor> input,
                                             int total_frames, int64_t override_at,
                                             Value initial_value, Value override_value,
                                             const char* graph_name, BuildFn build,
                                             PerfPublishMode publish = PerfPublishMode::kCopy,
                                             bool one_shot = true, bool prime_changed = false) {
  if (override_at < 0 || override_at >= total_frames) {
    throw std::invalid_argument("run_perf_graph_with_override: override_at outside frame plan");
  }
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset(std::move(input), total_frames, &collector, publish);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfTensorOverrideSourceOp<Value>>(
      "src", override_at, std::move(initial_value), std::move(override_value), one_shot,
      prime_changed);
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out");
  build(graph, src, options);
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

template <typename BuildFn>
TimingCollector run_perf_graph_with_window_override(
    std::shared_ptr<holoscan::Tensor> input, int total_frames, int64_t override_at,
    WindowSpec initial_value, WindowSpec override_value, const char* graph_name, BuildFn build,
    PerfPublishMode publish = PerfPublishMode::kCopy, bool one_shot = true) {
  if (override_at < 0 || override_at >= total_frames) {
    throw std::invalid_argument(
        "run_perf_graph_with_window_override: override_at outside frame plan");
  }
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset(std::move(input), total_frames, &collector, publish);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfTensorWindowOverrideSourceOp>(
      "src", override_at, std::move(initial_value), std::move(override_value), one_shot);
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out");
  build(graph, src, options);
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

template <typename Element, typename BuildFn>
TimingCollector run_perf_graph_with_vector_override(
    std::shared_ptr<holoscan::Tensor> input, int total_frames, int64_t override_at,
    std::vector<Element> initial_value, std::vector<Element> override_value, const char* graph_name,
    BuildFn build, PerfPublishMode publish = PerfPublishMode::kCopy, bool one_shot = true) {
  if (override_at < 0 || override_at >= total_frames) {
    throw std::invalid_argument(
        "run_perf_graph_with_vector_override: override_at outside frame plan");
  }
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset(std::move(input), total_frames, &collector, publish);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfTensorVectorOverrideSourceOp<Element>>(
      "src", override_at, std::move(initial_value), std::move(override_value), one_shot);
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out");
  build(graph, src, options);
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

template <typename BuildFn>
TimingCollector run_perf_graph_dual(std::shared_ptr<holoscan::Tensor> input_a,
                                    std::shared_ptr<holoscan::Tensor> input_b, int total_frames,
                                    const char* graph_name, BuildFn build,
                                    PerfPublishMode publish = PerfPublishMode::kCopy) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_dual(std::move(input_a), std::move(input_b), total_frames, &collector,
                            publish);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfDualTensorSourceOp>("src");
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out_a");
  bind_tensor_out(options, "src", "out_b");
  build(graph, src, options);
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

template <typename Value, typename BuildFn>
TimingCollector run_perf_graph_dual_with_override(std::shared_ptr<holoscan::Tensor> input_a,
                                                  std::shared_ptr<holoscan::Tensor> input_b,
                                                  int total_frames, int64_t override_at,
                                                  Value initial_value, Value override_value,
                                                  const char* graph_name, BuildFn build,
                                                  PerfPublishMode publish = PerfPublishMode::kCopy,
                                                  bool one_shot = true) {
  if (override_at < 0 || override_at >= total_frames) {
    throw std::invalid_argument(
        "run_perf_graph_dual_with_override: override_at outside frame plan");
  }
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_dual(std::move(input_a), std::move(input_b), total_frames, &collector,
                            publish);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfDualTensorOverrideSourceOp<Value>>(
      "src", override_at, std::move(initial_value), std::move(override_value), one_shot);
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out_a");
  bind_tensor_out(options, "src", "out_b");
  build(graph, src, options);
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

template <typename BuildFn>
TimingCollector run_perf_graph_triple(std::shared_ptr<holoscan::Tensor> input_a,
                                      std::shared_ptr<holoscan::Tensor> input_b,
                                      std::shared_ptr<holoscan::Tensor> input_c, int total_frames,
                                      const char* graph_name, BuildFn build,
                                      PerfPublishMode publish = PerfPublishMode::kCopy) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_triple(std::move(input_a), std::move(input_b), std::move(input_c),
                              total_frames, &collector, publish);

  holoscan::Graph graph{graph_name};
  const auto src = graph.op<PerfTripleTensorSourceOp>("src");
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out_a");
  bind_tensor_out(options, "src", "out_b");
  bind_tensor_out(options, "src", "out_c");
  build(graph, src, options);
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

template <typename SourceOp, typename BuildFn>
TimingCollector run_perf_graph_triple_with_override(std::shared_ptr<holoscan::Tensor> input_a,
                                                    std::shared_ptr<holoscan::Tensor> input_b,
                                                    std::shared_ptr<holoscan::Tensor> input_c,
                                                    int total_frames, int64_t override_at,
                                                    std::string port, int64_t features,
                                                    const char* graph_name, BuildFn build) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset_triple(std::move(input_a), std::move(input_b), std::move(input_c),
                              total_frames, &collector);
  holoscan::Graph graph{graph_name};
  const auto src = graph.op<SourceOp>("src", override_at, std::move(port), features);
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out_a");
  bind_tensor_out(options, "src", "out_b");
  bind_tensor_out(options, "src", "out_c");
  build(graph, src, options);
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

template <typename SourceOp, typename BuildFn, typename Scenario>
TimingCollector run_perf_graph_with_source(std::shared_ptr<holoscan::Tensor> input,
                                           int total_frames, int64_t override_at, Scenario scenario,
                                           const char* graph_name, BuildFn build) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  PerfFrameBank::reset(std::move(input), total_frames, &collector);
  holoscan::Graph graph{graph_name};
  const auto src = graph.op<SourceOp>("src", override_at, std::move(scenario));
  holoscan::CompileOptions options;
  bind_tensor_out(options, "src", "out");
  build(graph, src, options);
  execute_perf_graph(graph, std::move(options), total_frames, graph_name, &collector);
  return collector;
}

}  // namespace holoscan::radar::perf
