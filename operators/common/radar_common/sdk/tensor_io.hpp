// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_common/sdk/cooperative_sleep.hpp"

namespace holoscan::radar {

// HSDK 5 Tensor port helpers. Ports carry a plain holoscan::Tensor (no
// TensorMap). Receive/emit return expected<>; callers propagate with
// HOLOSCAN_TRY. The runtime compute stream is context.cuda_stream().

// Port specs. A TensorRepresentation states the facts a port constrains
// (memory kind / dtype / rank); a consumer constrains only those, while a
// producer that wants Runtime-owned loans also sizes its pool.

inline TensorInputSpec tensor_input(const TensorRepresentation& representation) {
  return TensorInputSpec{.representation = representation};
}

// A producer that hands on storage it does not own (no loans allocated).
inline TensorOutputSpec tensor_output(const TensorRepresentation& representation) {
  return TensorOutputSpec{.representation = representation};
}

inline TensorOutputSpec pooled_tensor_output(const TensorRepresentation& representation,
                                             const TensorBounds& bounds) {
  return TensorOutputSpec{.representation = representation,
                          .bounds = bounds,
                          .storage = TensorOutputStorage::kRuntimePool};
}

inline TensorOutputSpec pooled_tensor_output(const TensorRepresentation& representation,
                                             std::uint32_t max_rank, std::uint64_t max_byte_span) {
  return pooled_tensor_output(representation,
                              TensorBounds{.max_rank = max_rank, .max_byte_span = max_byte_span});
}

inline expected<Tensor, Error> receive_tensor(Input<Tensor>& port, const std::string& who) {
  auto sample = port.receive_data();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  // An empty tensor legitimately carries no allocation; only a sized tensor
  // without storage is malformed.
  if (sample->nbytes() > 0 && sample->data() == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              who + ": received a null tensor on input port '" + std::string(port.name()) + "'"});
  }
  return std::move(*sample);
}

// True for the receive errors that mean "this activation has no value here"
// (port unconnected / omitted from the contract, nothing selected yet). Every
// other code is a real transport or materialization failure and must surface —
// silently substituting a default would hide miswired graphs.
inline bool is_absent_input_error(ErrorCode code) {
  return code == ErrorCode::kInputUndeclared || code == ErrorCode::kInputEmpty ||
         code == ErrorCode::kInputNotSelected || code == ErrorCode::kNotReady;
}

// Optional override receive: success with a value when the port has
// a selected/reused sample; nullopt on SafeDefault / absent / never-seen.
template <typename T>
expected<std::optional<T>, Error> receive_optional_value(Input<T>& port) {
  if (port.is_missing()) {
    return std::optional<T>{};
  }
  auto sample = port.receive_data();
  if (!sample) {
    if (is_absent_input_error(sample.error().code)) {
      return std::optional<T>{};
    }
    return make_unexpected(std::move(sample).error());
  }
  return std::optional<T>{std::move(*sample)};
}

// Allocate a device Tensor loan, run `fill` on the runtime compute stream,
// and commit — but do **not** emit. Use this when a tick publishes several
// outputs: prepare every loan first so a fill/commit failure cannot leave a
// sibling port already published, then emit the prepared loans together.
// `fill` must enqueue only stream-ordered device work (no host sync).
template <typename FillFn>
expected<TensorOutputLoan, Error> prepare_device_tensor(Output<Tensor>& port,
                                                        ExecutionContext& context,
                                                        std::span<const int64_t> shape,
                                                        DLDataType dtype, FillFn&& fill) {
  auto loan = port.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }

  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }

  HOLOSCAN_TRY(std::forward<FillFn>(fill)(writer->data(), writer->byte_size()));

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return std::move(*loan);
}

// Stamping activation_time() propagates the frame's time rather than replacing
// it: for a data-triggered operator HSDK 5 sets activation time to the
// triggering sample's capture_time. Aligned fusion operators are the exception
// (activation time is the newest aligned head) — see
// docs/planning/OPERATOR_DEVELOPMENT.md, "Sample time propagation and aligned
// joins".
//
// [[nodiscard]] mirrors Output::emit, which carries it: a dropped result is a
// silently lost output. HOLOSCAN_TRY is right for a required output, where a
// failed emit really is an error.
//
// An *optional* output — one the application may leave unwired — must never
// use HOLOSCAN_TRY: publishing to an output with no consumer fails, and
// propagating that would fail the whole activation and drop the tick's other
// outputs as collateral. Per OPERATOR_API.md, "Optional output convention",
// such a port takes one of two dispositions: `always` discards the result here
// with an explicit (void), and `latched` goes through LatchedOutput below.
[[nodiscard]] inline expected<void, Error> emit_prepared_tensor(Output<Tensor>& port,
                                                                ExecutionContext& context,
                                                                TensorOutputLoan loan) {
  return port.emit(std::move(loan), EmitOptions{.capture_time = context.activation_time()});
}

// Emit a schema payload whose DescriptorTensor storage is a prepared Tensor
// loan. This is the schema-valued counterpart of emit_prepared_tensor above.
template <typename Payload>
[[nodiscard]] expected<void, Error> emit_prepared_schema_tensor(Output<Payload>& port,
                                                                ExecutionContext& context,
                                                                TensorOutputLoan loan,
                                                                Payload payload) {
  return port.emit_tensor(std::move(loan), std::move(payload),
                          EmitOptions{.capture_time = context.activation_time()});
}

// A `latched` optional output (OPERATOR_API.md, "Optional output
// convention"). An application may leave an optional output unwired, but
// nothing exposes an output's wiring to the operator, so it cannot ask —
// it learns only by emitting and failing. This records that answer once and
// gates every later tick on it.
//
// Only ErrorCode::kFailure latches. That is what an unconsumed output
// reports; any other code (kResourceExhausted, kBackpressured, ...) is a
// genuine fault that says nothing about wiring, and disabling the port for a
// whole run over one transient fault would turn a lost frame into a lost run.
// Those are ignored for the tick, leaving the port live.
//
// Latching is sound because wiring is fixed for a run: the plan is compiled
// before the first activation and cannot be rewired while it runs, so an emit
// that failed for want of a consumer will fail the same way forever.
class LatchedOutput {
 public:
  // False once the port is known unconsumed. Guard the work performed *solely*
  // to produce this output on it — allocation, fill kernels, commit — not just
  // the emit; skipping that work is the point.
  [[nodiscard]] bool active() const noexcept { return !disabled_; }

  // Emit and absorb. Never fails the activation: a declined optional output
  // must not take the tick's other outputs down with it.
  void emit(Output<Tensor>& port, ExecutionContext& context, TensorOutputLoan loan,
            std::string_view op_and_port) {
    const auto emitted = emit_prepared_tensor(port, context, std::move(loan));
    absorb_emit_result(emitted, op_and_port);
  }

  template <typename Payload>
  void emit_schema_tensor(Output<Payload>& port, ExecutionContext& context, TensorOutputLoan loan,
                          Payload payload, std::string_view op_and_port) {
    const auto emitted =
        emit_prepared_schema_tensor(port, context, std::move(loan), std::move(payload));
    absorb_emit_result(emitted, op_and_port);
  }

 private:
  void absorb_emit_result(const expected<void, Error>& emitted, std::string_view op_and_port) {
    if (emitted) {
      return;
    }
    if (emitted.error().code != ErrorCode::kFailure) {
      return;
    }
    HOLOSCAN_LOG_WARN(
        "{}: emit failed because no consumer is wired. Skipping the work that "
        "produces this optional output for the rest of the run; wire the port "
        "if you need it.",
        op_and_port);
    disabled_ = true;
  }

  bool disabled_{false};
};

// Allocate a device Tensor loan, run `fill` with a writable device pointer on
// the runtime compute stream, commit, and emit. `fill` must enqueue only
// stream-ordered device work (no host sync). Prefer prepare_device_tensor +
// emit_prepared_tensor when a tick publishes more than one output.
template <typename FillFn>
[[nodiscard]] expected<void, Error> emit_device_tensor(Output<Tensor>& port,
                                                       ExecutionContext& context,
                                                       std::span<const int64_t> shape,
                                                       DLDataType dtype, FillFn&& fill) {
  HOLOSCAN_TRY(auto loan,
               prepare_device_tensor(port, context, shape, dtype, std::forward<FillFn>(fill)));
  return emit_prepared_tensor(port, context, std::move(loan));
}

// Byte span for a compact row-major tensor of the given shape. Matches
// make_device_tensor: reject negative dims and int64 / size_t overflow
// rather than wrapping to an undersized allocation size.
inline std::size_t tensor_byte_span(std::span<const int64_t> shape, DLDataType dtype) {
  int64_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0 || (dim != 0 && count > std::numeric_limits<int64_t>::max() / dim)) {
      throw std::invalid_argument("tensor_byte_span: invalid or overflowing shape");
    }
    count *= dim;
  }
  const std::size_t bytes_per_element =
      ((static_cast<std::size_t>(dtype.bits) + 7) / 8) * static_cast<std::size_t>(dtype.lanes);
  if (bytes_per_element == 0 || static_cast<std::size_t>(count) >
                                    std::numeric_limits<std::size_t>::max() / bytes_per_element) {
    throw std::invalid_argument("tensor_byte_span: invalid or overflowing shape");
  }
  return static_cast<std::size_t>(count) * bytes_per_element;
}

// CPHDReaderOp is the only library operator that retries a full queue
// (unmetered playback_rate 0, or wait_for_downstream). Other required emits
// fail-fast through emit_required_tensor(..., false). A failed Tensor emit
// consumes the loan, so a wait retry must re-prepare, not re-emit.

template <typename Fn>
auto retry_transient(ExecutionContext& context, std::string_view what, Fn&& fn) -> decltype(fn()) {
  bool warned = false;
  for (;;) {
    auto result = fn();
    if (result)
      return result;
    const Error err = result.error();
    if ((err.code != ErrorCode::kResourceExhausted && err.code != ErrorCode::kBackpressured) ||
        context.cancellation().stop_requested()) {
      return result;
    }
    if (!warned) {
      HOLOSCAN_LOG_WARN("{}: waiting for tensor pool/queue due to pipeline backpressure ({})", what,
                        holoscan::error_code_name(err.code));
      warned = true;
    }
    cooperative_sleep(context, 0.01);
  }
}

// Fail-fast (wait false): keep the SDK code, replace the message so a
// RUNTIME_OPERATOR_CALLBACK_FAILED names the port. Log it too: the runtime
// wrapper can bury the Error. Only CPHDReaderOp retries (unmetered or
// wait_for_downstream).
inline Error explain_fail_fast_emit(std::string_view what, ErrorCode code) {
  const char* detail = code == ErrorCode::kBackpressured
                           ? "next stage is behind; output queue is full"
                           : "no free tensor buffer";
  return Error{code, std::string(what) + ": " + detail + ". The run is stopping."};
}

template <typename Fn>
auto emit_required_result(ExecutionContext& context, std::string_view what, bool wait,
                          Fn&& once) -> decltype(once()) {
  if (wait)
    return retry_transient(context, what, std::forward<Fn>(once));
  auto result = once();
  if (result)
    return result;
  const ErrorCode code = result.error().code;
  if (code != ErrorCode::kResourceExhausted && code != ErrorCode::kBackpressured)
    return result;
  Error explained = explain_fail_fast_emit(what, code);
  HOLOSCAN_LOG_ERROR("{}", holoscan::runtime_error_message(explained));
  return make_unexpected(std::move(explained));
}

template <typename PrepareFn>
[[nodiscard]] expected<void, Error> emit_required_tensor(Output<Tensor>& port,
                                                         ExecutionContext& context,
                                                         std::string_view what, bool wait,
                                                         PrepareFn&& prepare,
                                                         std::optional<EmitOptions> options = {}) {
  auto once = [&]() -> expected<void, Error> {
    HOLOSCAN_TRY(auto loan, prepare());
    const EmitOptions opts =
        options.value_or(EmitOptions{.capture_time = context.activation_time()});
    return port.emit(std::move(loan), opts);
  };
  return emit_required_result(context, what, wait, once);
}

template <typename Payload, typename PrepareFn>
[[nodiscard]] expected<void, Error> emit_required_schema_tensor(
    Output<Payload>& port, ExecutionContext& context, std::string_view what, bool wait,
    Payload payload, PrepareFn&& prepare, std::optional<EmitOptions> options = {}) {
  auto once = [&]() -> expected<void, Error> {
    HOLOSCAN_TRY(auto loan, prepare());
    const EmitOptions opts =
        options.value_or(EmitOptions{.capture_time = context.activation_time()});
    return port.emit_tensor(std::move(loan), payload, opts);
  };
  return emit_required_result(context, what, wait, once);
}

template <typename Port, typename Value>
[[nodiscard]] expected<void, Error> emit_required_value(Port& port, ExecutionContext& context,
                                                        std::string_view what, bool wait,
                                                        Value value, EmitOptions options) {
  auto once = [&]() -> expected<void, Error> {
    return port.emit(value, options);
  };
  return emit_required_result(context, what, wait, once);
}

}  // namespace holoscan::radar
