// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace holoscan::radar {

// Multiply positive extents into an element count; throw on overflow / <=0.
inline std::size_t product_extents(std::initializer_list<std::int64_t> dims, const char* who) {
  std::size_t product = 1;
  for (std::int64_t dim : dims) {
    if (dim <= 0) {
      throw std::invalid_argument(std::string(who) + ": extent dimensions must be positive");
    }
    const auto udim = static_cast<std::size_t>(dim);
    if (product > (static_cast<std::size_t>(-1) / udim)) {
      throw std::invalid_argument(std::string(who) + ": extent product overflows size_t");
    }
    product *= udim;
  }
  return product;
}

template <typename T>
constexpr std::size_t complex_element_bytes() {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "radar complex tensors use float or double precision");
  return sizeof(T) * 2U;
}

template <typename T>
constexpr std::size_t real_element_bytes() {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "radar real tensors use float or double precision");
  return sizeof(T);
}

inline std::size_t byte_span_elems(std::size_t n_elems, std::size_t elem_bytes, const char* who) {
  if (n_elems == 0U) {
    throw std::invalid_argument(std::string(who) + ": byte span needs a positive element count");
  }
  if (elem_bytes == 0U || n_elems > (static_cast<std::size_t>(-1) / elem_bytes)) {
    throw std::invalid_argument(std::string(who) + ": byte span overflows size_t");
  }
  return n_elems * elem_bytes;
}

// Fat range-profile / RD-map style cubes: product of named axes × complex T.
template <typename T>
inline std::size_t byte_span_complex_cube(std::initializer_list<std::int64_t> dims,
                                          const char* who) {
  return byte_span_elems(product_extents(dims, who), complex_element_bytes<T>(), who);
}

template <typename T>
inline std::size_t byte_span_real_cube(std::initializer_list<std::int64_t> dims, const char* who) {
  return byte_span_elems(product_extents(dims, who), real_element_bytes<T>(), who);
}

}  // namespace holoscan::radar
