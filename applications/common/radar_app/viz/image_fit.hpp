// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace holoscan::radar::app::viz {
cudaError_t fit_image(const void* source, void* target, int width, int height, int window_width,
                      int window_height, cudaStream_t stream);
}  // namespace holoscan::radar::app::viz
