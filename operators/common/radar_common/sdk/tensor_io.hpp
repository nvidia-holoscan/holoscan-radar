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

namespace holoscan::radar {

// HSDK 5 Tensor port helpers. Ports carry a plain holoscan::Tensor (no
// TensorMap). Receive/emit return expected<>; callers propagate with
// HOLOSCAN_TRY. The runtime compute stream is context.cuda_stream().

// Port specs. A TensorRepresentation states the facts a port constrains
// (memory kind / dtype / rank); a consumer constrains only those, while a
// producer that wants Runtime-owned loans also sizes its pool.

inline TensorInputSpec tensor_input(
    const TensorRepresentation& representation) {
  return TensorInputSpec{.representation = representation};
}

// A producer that hands on storage it does not own (no loans allocated).
inline TensorOutputSpec tensor_output(
    const TensorRepresentation& representation) {
  return TensorOutputSpec{.representation = representation};
}

inline TensorOutputSpec pooled_tensor_output(
    const TensorRepresentation& representation, const TensorBounds& bounds) {
  return TensorOutputSpec{.representation = representation,
                          .bounds = bounds,
                          .storage = TensorOutputStorage::kRuntimePool};
}

inline TensorOutputSpec pooled_tensor_output(
    const TensorRepresentation& representation, std::uint32_t max_rank,
    std::uint64_t max_byte_span) {
  return pooled_tensor_output(
      representation,
      TensorBounds{.max_rank = max_rank, .max_byte_span = max_byte_span});
}

inline expected<Tensor, Error> receive_tensor(Input<Tensor>& port,
                                              const std::string& who) {
  auto sample = port.receive_data();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  // An empty tensor legitimately carries no allocation; only a sized tensor
  // without storage is malformed.
  if (sample->nbytes() > 0 && sample->data() == nullptr) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        who + ": received a null tensor on input port '" +
            std::string(port.name()) + "'"});
  }
  return std::move(*sample);
}

// True for the receive errors that mean "this activation has no value here"
// (port unconnected / omitted from the contract, nothing selected yet). Every
// other code is a real transport or materialization failure and must surface —
// silently substituting a default would hide miswired graphs.
inline bool is_absent_input_error(ErrorCode code) {
  return code == ErrorCode::kInputUndeclared ||
         code == ErrorCode::kInputEmpty ||
         code == ErrorCode::kInputNotSelected || code == ErrorCode::kNotReady;
}

// Optional sticky-override receive: success with a value when the port has
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
expected<TensorOutputLoan, Error> prepare_device_tensor(
    Output<Tensor>& port, ExecutionContext& context,
    std::span<const int64_t> shape, DLDataType dtype, FillFn&& fill) {
  auto loan = port.allocate_tensor(
      TensorLoanRequest{.shape = shape, .dtype = dtype});
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
inline expected<void, Error> emit_prepared_tensor(
    Output<Tensor>& port, ExecutionContext& context, TensorOutputLoan loan) {
  return port.emit(std::move(loan),
                   EmitOptions{.capture_time = context.activation_time()});
}

// Allocate a device Tensor loan, run `fill` with a writable device pointer on
// the runtime compute stream, commit, and emit. `fill` must enqueue only
// stream-ordered device work (no host sync). Prefer prepare_device_tensor +
// emit_prepared_tensor when a tick publishes more than one output.
template <typename FillFn>
expected<void, Error> emit_device_tensor(Output<Tensor>& port,
                                         ExecutionContext& context,
                                         std::span<const int64_t> shape,
                                         DLDataType dtype, FillFn&& fill) {
  HOLOSCAN_TRY(auto loan,
               prepare_device_tensor(port, context, shape, dtype,
                                    std::forward<FillFn>(fill)));
  return emit_prepared_tensor(port, context, std::move(loan));
}

// Byte span for a compact row-major tensor of the given shape. Matches
// make_device_tensor: reject negative dims and int64 / size_t overflow
// rather than wrapping to an undersized allocation size.
inline std::size_t tensor_byte_span(std::span<const int64_t> shape,
                                    DLDataType dtype) {
  int64_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0 ||
        (dim != 0 && count > std::numeric_limits<int64_t>::max() / dim)) {
      throw std::invalid_argument(
          "tensor_byte_span: invalid or overflowing shape");
    }
    count *= dim;
  }
  const std::size_t bytes_per_element =
      ((static_cast<std::size_t>(dtype.bits) + 7) / 8) *
      static_cast<std::size_t>(dtype.lanes);
  if (bytes_per_element == 0 ||
      static_cast<std::size_t>(count) >
          std::numeric_limits<std::size_t>::max() / bytes_per_element) {
    throw std::invalid_argument(
        "tensor_byte_span: invalid or overflowing shape");
  }
  return static_cast<std::size_t>(count) * bytes_per_element;
}

}  // namespace holoscan::radar
