// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op_impl.cuh"

namespace holoscan::radar {

template class DetectionReductionOp<float>;
template class DetectionReductionOp<double>;

}  // namespace holoscan::radar
