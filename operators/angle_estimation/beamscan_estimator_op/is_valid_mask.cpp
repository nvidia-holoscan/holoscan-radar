// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/beamscan_estimator_op/is_valid_mask.hpp"

#include <cuda_runtime_api.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::detail {

void check_is_valid_mask(const holoscan::Tensor& is_valid, int64_t num_items,
                         int64_t num_snapshots, const char* who) {
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

  const int64_t mask_elems = num_items * num_snapshots;
  std::vector<uint8_t> host_mask(static_cast<size_t>(mask_elems));
  if (mask_elems > 0) {
    const cudaMemcpyKind kind =
        is_cuda_accessible(is_valid) ? cudaMemcpyDeviceToHost
                                     : cudaMemcpyHostToHost;
    if (cudaMemcpy(host_mask.data(), is_valid.data(),
                   static_cast<size_t>(mask_elems) * sizeof(uint8_t),
                   kind) != cudaSuccess) {
      throw std::runtime_error(
          std::string(who) + ": is_valid mask copy failed");
    }
  }

  for (int64_t i = 0; i < num_items; ++i) {
    if (host_mask[static_cast<size_t>(i)] == 0) {
      throw std::invalid_argument(
          std::string(who) +
          ": is_valid marks an invalid snapshot slot");
    }
  }
}

}  // namespace holoscan::radar::detail
