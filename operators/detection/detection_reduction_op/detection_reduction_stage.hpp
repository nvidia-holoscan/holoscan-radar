// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CUDA-TU stage for DetectionReductionOp — includes public group-by-label
// launch wrappers + template custom-peak kernels (definitions of non-template
// kernels are linked from group_by_label.cu).

#pragma once

#include "radar_common/reduce/group_by_label.hpp"
