// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::radar {

inline bool config_dtype_equals(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits &&
         lhs.lanes == rhs.lanes;
}

inline unexpected<Error> invalid_config_tensor(const char* field,
                                               const std::string& reason) {
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument,
            std::string("live config field '") + field + "': " + reason});
}

template <typename T>
expected<std::vector<T>, Error> read_host_config_vector(
    const Tensor& tensor, DLDataType expected_dtype,
    std::int64_t max_extent, const char* field) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (tensor.shape().size() != 1U) {
    return invalid_config_tensor(field, "expected rank 1");
  }
  if (tensor.device().device_type != kDLCPU) {
    return invalid_config_tensor(field, "expected pageable host memory");
  }
  if (!tensor.is_contiguous()) {
    return invalid_config_tensor(field, "expected contiguous storage");
  }
  if (!config_dtype_equals(tensor.dtype(), expected_dtype)) {
    return invalid_config_tensor(field, "unexpected dtype");
  }
  const std::int64_t extent = tensor.shape()[0];
  if (extent < 0 || max_extent < 0 || extent > max_extent ||
      static_cast<std::uint64_t>(extent) >
          static_cast<std::uint64_t>(SIZE_MAX / sizeof(T))) {
    return invalid_config_tensor(field, "extent exceeds configured maximum");
  }
  const std::size_t expected_bytes =
      static_cast<std::size_t>(extent) * sizeof(T);
  if (static_cast<std::size_t>(tensor.nbytes()) != expected_bytes ||
      (expected_bytes != 0U && tensor.data() == nullptr)) {
    return invalid_config_tensor(field,
                                 "byte size is inconsistent with shape");
  }
  std::vector<T> result(static_cast<std::size_t>(extent));
  if (expected_bytes != 0U) {
    std::memcpy(result.data(), tensor.data(), expected_bytes);
  }
  return result;
}

}  // namespace holoscan::radar
