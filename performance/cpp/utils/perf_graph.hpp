// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared HSDK 5 Graph source / sink / runner for performance drivers.
// Mid-session [live] overrides use a source that publishes the sideband and
// Tensor on one clock, once on the changed frame. The next frame uses Params.
//
// Pattern:
//   - PerfDataSourceOp emits staged device Tensors and records start
//     timestamps into TimingCollector.
//   - PerfTerminalSinkOp drains the timed output, cudaDeviceSynchronize()'s,
//     and records end timestamps.
//   - PerfNullSinkOp drains always-emitted optional outs without timing.
//   - run_typed_recipe() compiles and runs until `total_frames` ends are captured.
//   - The recipe-selected source emits one Exact sideband on the changed frame.

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
#include <optional>
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

// How a staged source publishes each tick. Recipe runners pass this mode;
// default remains kCopy (stage once, D2D into the loan).
// kFillInPlace / kZeroCopy are diagnostic: fill currently zero-fills the loan;
// zero-copy support depends on the active Graph output binding.
enum class PerfPublishMode {
  kCopy,
  kFillInPlace,
  kZeroCopy,
};

struct PerfFrameBank {
  template <std::size_t N>
  static void reset(std::array<std::shared_ptr<holoscan::Tensor>, N> frames, int count,
                    TimingCollector* collector, PerfPublishMode publish = PerfPublishMode::kCopy) {
    static_assert(N == 1 || N == 2 || N == 3 || N == 6);
    const std::lock_guard lock(mutex);
    input_frames.fill(nullptr);
    for (std::size_t i = 0; i < N; ++i)
      input_frames[i] = std::move(frames[i]);
    frames_remaining = count;
    captured = 0;
    in_flight = 0;
    collector_ = collector;
    publish_mode = publish;
  }

  // Stage exact frame sizes before compile; pool cost is slots times span.
  static std::size_t span(std::size_t index) {
    const std::lock_guard lock(mutex);
    return staged_span(input_frames.at(index));
  }

  static std::size_t staged_span(const std::shared_ptr<holoscan::Tensor>& frame) {
    if (frame == nullptr || frame->nbytes() <= 0) {
      throw std::runtime_error("PerfFrameBank: source frame must be staged before compile()");
    }
    return static_cast<std::size_t>(frame->nbytes());
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::array<std::shared_ptr<holoscan::Tensor>, 6> input_frames{};
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
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "PerfDataSourceOp: null frame"});
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
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "PerfDataSourceOp: cudaMemcpyAsync failed"});
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
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "PerfDataSourceOp: cudaMemsetAsync failed"});
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

template <size_t Count>
struct StagedFrames {
  std::array<std::shared_ptr<holoscan::Tensor>, Count> frames;
  PerfPublishMode mode{PerfPublishMode::kCopy};
};

// One source tick claims the staged inputs and records one start timestamp.
// Every source variant shares this pacing and timestamp boundary.
template <size_t Count>
std::optional<StagedFrames<Count>> take_staged_frames() {
  static_assert(Count >= 1 && Count <= 6);
  const std::lock_guard lock(PerfFrameBank::mutex);
  if (PerfFrameBank::in_flight != 0 || PerfFrameBank::frames_remaining <= 0)
    return std::nullopt;
  StagedFrames<Count> result;
  for (size_t i = 0; i < Count; ++i) {
    if (PerfFrameBank::input_frames[i] == nullptr)
      return std::nullopt;
    result.frames[i] = PerfFrameBank::input_frames[i];
  }
  --PerfFrameBank::frames_remaining;
  result.mode = PerfFrameBank::publish_mode;
  record_start_locked();
  return result;
}

template <typename Value>
void setup_live_output(holoscan::OperatorSpec& spec, holoscan::Output<Value>& output,
                       const Value& initial, const Value& changed) {
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
    const std::size_t max_bytes = std::max(
        {sizeof(std::complex<double>), reference_bytes(initial), reference_bytes(changed)});
    spec.output(output, "override_out")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(kWaveform, 1U, max_bytes));
  } else {
    spec.output(output, "override_out");
  }
}

template <typename Value>
holoscan::expected<void, holoscan::Error> emit_live_value(holoscan::Output<Value>& output,
                                                          holoscan::ExecutionContext& context,
                                                          const Value& value) {
  if constexpr (std::is_same_v<Value, WireFMCWWaveform>) {
    auto reference = wire_bridge_detail::read_host_vector<std::complex<double>>(
        value.reference_samples, DLDataType{kDLComplex, 128U, 1U}, "reference_samples");
    if (!reference)
      return holoscan::make_unexpected(std::move(reference).error());
    const std::array<int64_t, 1> shape{static_cast<int64_t>(reference->size())};
    auto loan = output.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLComplex, 128U, 1U}});
    if (!loan)
      return holoscan::make_unexpected(std::move(loan).error());
    auto written =
        loan->write_host_and_commit([&reference](const holoscan::TensorOutputWriteGuard& writer) {
          std::copy(reference->begin(), reference->end(), writer.data_as<std::complex<double>>());
        });
    if (!written)
      return holoscan::make_unexpected(std::move(written).error());
    return output.emit_tensor(std::move(*loan), Value(value),
                              holoscan::EmitOptions{.capture_time = context.activation_time()});
  } else {
    return output.emit(value, holoscan::EmitOptions{.capture_time = context.activation_time()});
  }
}

}  // namespace detail

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

}  // namespace holoscan::radar::perf
