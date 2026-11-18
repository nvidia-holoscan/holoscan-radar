// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace holoscan::radar::app::viz {
cudaError_t fit_image(const void* source, void* target, int width, int height, int window_width,
                      int window_height, cudaStream_t stream);
cudaError_t compose_docked_panels(const void* image, const void* footer, void* target, int width,
                                  int image_height, int footer_height, int window_width,
                                  int window_height, bool fit, cudaStream_t stream,
                                  int status_width = 0);
}  // namespace holoscan::radar::app::viz
