// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>

namespace holoscan::radar::detail {

// Validates optional is_valid mask transport (dtype bool/uint8, shape
// (num_items, num_snapshots), CUDA-accessible + contiguous). Does **not**
// reject zero slots — those mean "unused snapshot" per OPERATOR_API.md and
// are honored by the estimator (blank outputs when K_i == 0).
void check_is_valid_mask(const holoscan::Tensor& is_valid, int64_t num_items,
                         int64_t num_snapshots, const char* who);

}  // namespace holoscan::radar::detail
