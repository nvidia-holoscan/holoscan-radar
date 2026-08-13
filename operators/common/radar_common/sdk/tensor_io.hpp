// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace holoscan::radar {

// HSDK 5 Tensor port helpers. Ports carry a plain holoscan::Tensor (no
// TensorMap). Receive/emit return expected<>; callers propagate with
// HOLOSCAN_TRY. The runtime compute stream is context.cuda_stream().

inline expected<Tensor, Error> receive_tensor(Input<Tensor>& port,
                                              const std::string& who) {
  auto sample = port.receive_data();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  if (sample->data() == nullptr) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        who + ": received a null tensor on input port '" +
            std::string(port.name()) + "'"});
  }
  return std::move(*sample);
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
    if (sample.error().code == ErrorCode::kNotReady ||
        sample.error().code == ErrorCode::kReceiveError) {
      return std::optional<T>{};
    }
    return make_unexpected(std::move(sample).error());
  }
  return std::optional<T>{std::move(*sample)};
}

// Allocate a device Tensor loan, run `fill` with a writable device pointer on
// the runtime compute stream, commit, and emit. `fill` must enqueue only
// stream-ordered device work (no host sync).
template <typename FillFn>
expected<void, Error> emit_device_tensor(Output<Tensor>& port,
                                         ExecutionContext& context,
                                         std::span<const int64_t> shape,
                                         DLDataType dtype, FillFn&& fill) {
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
  return port.emit(std::move(*loan),
                   EmitOptions{.capture_time = context.activation_time()});
}

// Convenience: byte span for a compact row-major tensor of the given shape.
inline std::size_t tensor_byte_span(std::span<const int64_t> shape,
                                    DLDataType dtype) {
  std::size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) {
      return 0;
    }
    count *= static_cast<std::size_t>(dim);
  }
  const std::size_t bytes_per_element =
      ((static_cast<std::size_t>(dtype.bits) + 7) / 8) *
      static_cast<std::size_t>(dtype.lanes);
  return count * bytes_per_element;
}

}  // namespace holoscan::radar
