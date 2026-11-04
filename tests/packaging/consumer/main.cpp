// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <holoscan/radar/fmcw/fmcw_range_processor_op.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"

int main() {
  holoscan::radar::MaxTensorExtents extents{};
  extents.max_chirps = 4;
  extents.max_samples_per_chirp = 16;
  extents.max_range_bins = 16;
  holoscan::radar::FMCWRangeProcessorOp<float> single(extents);
  holoscan::radar::FMCWRangeProcessorOp<double> dual(extents);
  return 0;
}
