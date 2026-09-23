// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/extents_layout.hpp"

namespace holoscan::radar {

// DLPack dtype descriptors for the two precision families the operators
// support: complex64/complex128 data with float32/float64 axis outputs,
// fixed per instance by template precision.
template <typename T>
constexpr DLDataType real_dtype() {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "radar operators support float and double precision only");
  return DLDataType{kDLFloat, sizeof(T) == 4 ? uint8_t{32} : uint8_t{64}, 1};
}

template <typename T>
constexpr DLDataType complex_dtype() {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "radar operators support float and double precision only");
  return DLDataType{kDLComplex, sizeof(T) == 4 ? uint8_t{64} : uint8_t{128}, 1};
}

constexpr bool dldatatype_equals(DLDataType a, DLDataType b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

// Human-readable dtype for error messages, e.g. "complex64".
std::string dldatatype_name(DLDataType dtype);

// A rank-agnostic view of a batched matrix tensor (..., rows, cols): the
// leading batch axes are collapsed into a single `batch` dimension
// (leading `...` axes are opaque, preserved batch dims).
struct BatchedShape {
  std::vector<int64_t> leading;  // original leading axes (may be empty)
  int64_t batch{1};              // product of the leading axes (>= 1)
  int64_t rows{0};               // second-to-last axis
  int64_t cols{0};               // last axis
};

// Validate and collapse a tensor of rank >= 2 into a BatchedShape. Throws
// std::invalid_argument when the rank is < 2 or the layout is not compact
// row-major (strided inputs are rejected in this wave).
BatchedShape collapse_batch(const holoscan::Tensor& tensor, const char* port_name);

// collapse_batch plus MaxTensorExtents named TX/RX bounds, for compute()
// shells (opaque leading batch and batch > max_tx*max_rx are errors).
inline expected<BatchedShape, Error> collapse_named_tx_rx_batch(const Tensor& input,
                                                                const MaxTensorExtents& extents,
                                                                const char* port_name,
                                                                const char* who) {
  BatchedShape batched;
  try {
    batched = collapse_batch(input, port_name);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }

  const std::size_t max_leading = named_tx_rx_leading_rank(extents);
  if (batched.leading.size() > max_leading) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              unexpected_leading_message(who, batched.leading.size(), max_leading)});
  }
  const std::int64_t max_rx = extent_rx_or_channels(extents);
  const std::int64_t max_tx = extent_or_one(extents.max_tx);
  if (batched.batch > max_tx * max_rx) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(who) + ": leading product " +
                                               std::to_string(batched.batch) +
                                               " exceeds max_tx*max_rx from MaxTensorExtents"});
  }
  return batched;
}

// The full-rank output shape for a batched result: `leading` axes followed
// by (rows, cols).
std::vector<int64_t> full_rank_shape(const BatchedShape& shape, int64_t rows, int64_t cols);

// Whether the tensor's memory is directly accessible from CUDA kernels
// (device, managed, or pinned-host memory).
bool is_cuda_accessible(const holoscan::Tensor& tensor);

// Throw std::invalid_argument (naming the operator and the port) unless the
// tensor is CUDA-accessible. Module-wide input requirement: pageable-host
// input is rejected; ingest operators upload before the processing chain
// (STATUS.md decision 2026-07-10).
void require_cuda_accessible(const holoscan::Tensor& tensor, const char* port_name,
                             const std::string& who);

// Throw std::invalid_argument (naming the operator and the port) unless the
// tensor is contiguous (compact row-major). Strided inputs are rejected by the
// operators that index raw device pointers directly.
void require_contiguous(const holoscan::Tensor& tensor, const char* port_name,
                        const std::string& who);

// Throw std::invalid_argument (naming the operator, the port, the received
// dtype, and the expected dtype) unless the tensor's dtype matches `expected`.
void require_dtype(const holoscan::Tensor& tensor, DLDataType expected, const char* port_name,
                   const std::string& who);

// Classification of an operator input against its instantiation precision.
enum class InputPrecision {
  kReal,     // float32/float64 matching T (real-IF front end)
  kComplex,  // complex64/complex128 matching T
};

// Classify the input dtype for a precision-T operator instance. Throws
// std::invalid_argument (naming the port, the received dtype, and the
// instantiation) on any other dtype, or when `allow_real` is false and the
// dtype is real.
template <typename T>
InputPrecision classify_input_dtype(const holoscan::Tensor& tensor, const char* port_name,
                                    bool allow_real);

extern template InputPrecision classify_input_dtype<float>(const holoscan::Tensor&, const char*,
                                                           bool);
extern template InputPrecision classify_input_dtype<double>(const holoscan::Tensor&, const char*,
                                                            bool);

// Complex-only input: classify_input_dtype with allow_real=false, discarding
// the enum. Same error strings. Use this when the operator has no real-IF
// path; keep classify_input_dtype when the caller needs kReal vs kComplex.
template <typename T>
inline void require_complex(const holoscan::Tensor& tensor, const char* port_name) {
  (void)classify_input_dtype<T>(tensor, port_name, /*allow_real=*/false);
}

}  // namespace holoscan::radar
