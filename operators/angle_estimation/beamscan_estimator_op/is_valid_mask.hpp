// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/core/domain/tensor.hpp>

#include <cstdint>

namespace holoscan::radar::detail {

// Validates optional SnapshotGather is_valid mask on a 3-D snapshot input.
// Accepts kDLBool or kDLUInt (8-bit) element masks. Device-resident masks
// are copied to the host before inspection.
void check_is_valid_mask(const holoscan::Tensor& is_valid, int64_t num_items,
                         int64_t num_snapshots, const char* who);

}  // namespace holoscan::radar::detail
