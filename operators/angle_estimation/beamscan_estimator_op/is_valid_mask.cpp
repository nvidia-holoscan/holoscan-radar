// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/beamscan_estimator_op/is_valid_mask.hpp"

#include <stdexcept>
#include <string>

#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::detail {

void check_is_valid_mask(const holoscan::Tensor& is_valid, int64_t num_items,
                         int64_t num_snapshots, const char* who) {
  // Shape / dtype only. Zero slots are legal: OPERATOR_API.md says is_valid
  // marks which snapshot slots to use (clamp/ragged); the estimator must
  // restrict itself to marked slots, not fail the activation.
  const DLDataType dtype = is_valid.dtype();
  const bool is_bool =
      dtype.code == kDLBool && dtype.bits == 8 && dtype.lanes == 1;
  const bool is_u8 =
      dtype.code == kDLUInt && dtype.bits == 8 && dtype.lanes == 1;
  if (!is_bool && !is_u8) {
    throw std::invalid_argument(
        std::string(who) +
        ": is_valid must have dtype bool or uint8");
  }
  if (is_valid.shape().size() != 2 ||
      is_valid.shape()[0] != num_items ||
      is_valid.shape()[1] != num_snapshots) {
    throw std::invalid_argument(
        std::string(who) +
        ": is_valid must have shape (num_items, num_snapshots) matching "
        "snapshot");
  }
  require_cuda_accessible(is_valid, "is_valid", who);
  require_contiguous(is_valid, "is_valid", who);
}

}  // namespace holoscan::radar::detail
