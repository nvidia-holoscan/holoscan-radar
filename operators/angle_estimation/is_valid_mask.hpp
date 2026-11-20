// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::detail {

// Validates optional is_valid mask transport (dtype bool/uint8, shape
// (num_items, num_snapshots), CUDA-accessible + contiguous). Does **not**
// reject zero slots — those mean "unused snapshot" per the operator README and
// are honored by the estimator (blank outputs when K_i == 0).
inline void check_is_valid_mask(const holoscan::Tensor& is_valid, int64_t num_items,
                                int64_t num_snapshots, const char* who) {
  const DLDataType dtype = is_valid.dtype();
  const bool is_bool = dtype.code == kDLBool && dtype.bits == 8 && dtype.lanes == 1;
  const bool is_u8 = dtype.code == kDLUInt && dtype.bits == 8 && dtype.lanes == 1;
  if (!is_bool && !is_u8) {
    throw std::invalid_argument(std::string(who) + ": is_valid must have dtype bool or uint8");
  }
  if (is_valid.shape().size() != 2 || is_valid.shape()[0] != num_items ||
      is_valid.shape()[1] != num_snapshots) {
    throw std::invalid_argument(std::string(who) +
                                ": is_valid must have shape (num_items, num_snapshots) matching "
                                "snapshot");
  }
  require_cuda_accessible(is_valid, "is_valid", who);
  require_contiguous(is_valid, "is_valid", who);
}

}  // namespace holoscan::radar::detail
