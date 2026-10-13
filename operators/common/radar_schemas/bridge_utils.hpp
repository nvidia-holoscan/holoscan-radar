// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::radar::wire_bridge_detail {

struct AggregateInitializer {
  template <typename T>
  constexpr operator T() const noexcept;  // NOLINT(runtime/explicit)
};

template <typename T, std::size_t... Indices>
consteval bool aggregate_initializable(std::index_sequence<Indices...>) {
  return requires { T{(static_cast<void>(Indices), AggregateInitializer{})...}; };
}

template <typename T, std::size_t Count = 0>
consteval std::size_t aggregate_field_count() {
  static_assert(std::is_aggregate_v<T>);
  static_assert(Count < 32, "aggregate field-count tripwire limit exceeded");
  if constexpr (aggregate_initializable<T>(std::make_index_sequence<Count + 1>{})) {
    return aggregate_field_count<T, Count + 1>();
  } else {
    return Count;
  }
}

template <typename T>
struct HostTensorMemory final : TensorMemoryReference {
  explicit HostTensorMemory(std::span<const T> source) : values(source.begin(), source.end()) {}

  std::vector<T> values;
};

template <typename T>
std::shared_ptr<Tensor> make_host_tensor(std::span<const T> source, DLDataType dtype) {
  static_assert(std::is_trivially_copyable_v<T>);
  auto memory = std::make_shared<HostTensorMemory<T>>(source);
  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference = memory;
  context->dl_shape = {static_cast<std::int64_t>(source.size())};
  context->tensor.dl_tensor.data = memory->values.empty() ? nullptr : memory->values.data();
  context->tensor.dl_tensor.device = DLDevice{kDLCPU, 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = dtype;
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0;
  return std::make_shared<Tensor>(std::move(context));
}

template <typename T>
std::shared_ptr<Tensor> make_host_matrix(std::span<const T> source, std::int64_t rows,
                                         std::int64_t columns, DLDataType dtype) {
  if (rows < 0 || columns <= 0 ||
      static_cast<std::uint64_t>(rows) > static_cast<std::uint64_t>(SIZE_MAX) ||
      static_cast<std::uint64_t>(columns) > static_cast<std::uint64_t>(SIZE_MAX) ||
      static_cast<std::size_t>(rows) > SIZE_MAX / static_cast<std::size_t>(columns) ||
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns) != source.size()) {
    throw std::invalid_argument("host Tensor matrix shape is invalid");
  }
  static_assert(std::is_trivially_copyable_v<T>);
  auto memory = std::make_shared<HostTensorMemory<T>>(source);
  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference = memory;
  context->dl_shape = {rows, columns};
  context->tensor.dl_tensor.data = memory->values.empty() ? nullptr : memory->values.data();
  context->tensor.dl_tensor.device = DLDevice{kDLCPU, 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = dtype;
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0;
  return std::make_shared<Tensor>(std::move(context));
}

inline bool dtype_equals(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

inline unexpected<Error> invalid_wire_tensor(const char* field, const std::string& reason) {
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument, std::string("wire field '") + field + "': " + reason});
}

template <typename T>
expected<std::vector<T>, Error> read_host_vector(const std::shared_ptr<Tensor>& tensor,
                                                 DLDataType expected_dtype, const char* field) {
  if (tensor == nullptr) {
    return invalid_wire_tensor(field, "Tensor pointer must not be null");
  }
  if (tensor->shape().size() != 1U) {
    return invalid_wire_tensor(field, "expected rank 1");
  }
  if (tensor->device().device_type != kDLCPU) {
    return invalid_wire_tensor(field, "expected pageable host memory");
  }
  if (!tensor->is_contiguous()) {
    return invalid_wire_tensor(field, "expected contiguous storage");
  }
  if (!dtype_equals(tensor->dtype(), expected_dtype)) {
    return invalid_wire_tensor(field, "unexpected dtype");
  }
  if (tensor->nbytes() == 0) {
    return std::vector<T>{};
  }
  const auto extent = tensor->shape()[0];
  if (extent < 0 ||
      static_cast<std::uint64_t>(extent) > static_cast<std::uint64_t>(SIZE_MAX / sizeof(T))) {
    return invalid_wire_tensor(field, "invalid extent");
  }
  const std::size_t expected_bytes = static_cast<std::size_t>(extent) * sizeof(T);
  if (static_cast<std::size_t>(tensor->nbytes()) != expected_bytes || tensor->data() == nullptr) {
    return invalid_wire_tensor(field, "byte size is inconsistent with shape");
  }
  std::vector<T> result(static_cast<std::size_t>(extent));
  std::memcpy(result.data(), tensor->data(), expected_bytes);
  return result;
}

template <typename T>
expected<std::vector<T>, Error> read_host_matrix(const std::shared_ptr<Tensor>& tensor,
                                                 DLDataType expected_dtype,
                                                 std::int64_t expected_columns, const char* field) {
  if (tensor == nullptr) {
    return invalid_wire_tensor(field, "Tensor pointer must not be null");
  }
  if (tensor->shape().size() != 2U || tensor->shape()[1] != expected_columns) {
    return invalid_wire_tensor(field, "unexpected matrix shape");
  }
  if (tensor->device().device_type != kDLCPU) {
    return invalid_wire_tensor(field, "expected pageable host memory");
  }
  if (!tensor->is_contiguous()) {
    return invalid_wire_tensor(field, "expected contiguous storage");
  }
  if (!dtype_equals(tensor->dtype(), expected_dtype)) {
    return invalid_wire_tensor(field, "unexpected dtype");
  }
  const auto rows = tensor->shape()[0];
  if (rows < 0 || expected_columns <= 0 ||
      static_cast<std::uint64_t>(rows) > static_cast<std::uint64_t>(SIZE_MAX) ||
      static_cast<std::size_t>(rows) > SIZE_MAX / static_cast<std::size_t>(expected_columns) ||
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(expected_columns) >
          SIZE_MAX / sizeof(T)) {
    return invalid_wire_tensor(field, "invalid extent");
  }
  const std::size_t count =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(expected_columns);
  const std::size_t expected_bytes = count * sizeof(T);
  if (static_cast<std::size_t>(tensor->nbytes()) != expected_bytes ||
      (expected_bytes != 0U && tensor->data() == nullptr)) {
    return invalid_wire_tensor(field, "byte size is inconsistent with shape");
  }
  std::vector<T> result(count);
  if (expected_bytes != 0U) {
    std::memcpy(result.data(), tensor->data(), expected_bytes);
  }
  return result;
}

}  // namespace holoscan::radar::wire_bridge_detail
