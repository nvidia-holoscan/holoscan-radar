// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CUDA-TU stage for DetectionReductionOp. Kernels live in the public
// holoscan::radar_common_reduce library (radar_common/reduce/); this header
// pulls in their launch wrappers for the operator .cu. A thin stage is
// intentional — the contract delivery model is "thin op over a shared
// library primitive," not operator-local kernels in _stage.hpp.

#pragma once

#include "radar_common/reduce/group_by_label.hpp"
