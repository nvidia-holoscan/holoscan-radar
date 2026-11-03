// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

namespace holoscan::radar::perf {

#ifdef RADAR_PERF_DOUBLE
using Scalar = double;
inline constexpr bool kDoublePrecision = true;
inline constexpr const char* kPrecision = "double";
#else
using Scalar = float;
inline constexpr bool kDoublePrecision = false;
inline constexpr const char* kPrecision = "float";
#endif

// Recipes declare the single-precision member of a compile-time dtype pair.
// The separate double executable selects its paired representation.
inline std::string precision_dtype(std::string_view dtype) {
  if constexpr (kDoublePrecision) {
    if (dtype == "float32")
      return "float64";
    if (dtype == "complex64")
      return "complex128";
  }
  return std::string(dtype);
}

inline std::string precision_operator(std::string_view name) {
  std::string result(name);
  if constexpr (kDoublePrecision) {
    if (result.ends_with(".float"))
      result.replace(result.size() - 6, 6, ".double");
  }
  return result;
}

}  // namespace holoscan::radar::perf
