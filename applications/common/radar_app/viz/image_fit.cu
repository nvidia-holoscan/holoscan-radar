// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_app/viz/image_fit.hpp"

#include <algorithm>
#include <cmath>

namespace holoscan::radar::app::viz {
namespace {
__global__ void fit_kernel(const uchar4* source, uchar4* target, int width, int height,
                           float content_width, float content_height) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height)
    return;
  const float x = index % width + 0.5F, y = index / width + 0.5F;
  const float sx = (x - (width - content_width) * 0.5F) * width / content_width - 0.5F;
  const float sy = (y - (height - content_height) * 0.5F) * height / content_height - 0.5F;
  if (sx < -0.5F || sy < -0.5F || sx >= width - 0.5F || sy >= height - 0.5F) {
    target[index] = make_uchar4(0, 0, 0, 255);
    return;
  }
  const int ix = min(width - 1, max(0, static_cast<int>(floorf(sx + 0.5F))));
  const int iy = min(height - 1, max(0, static_cast<int>(floorf(sy + 0.5F))));
  target[index] = source[iy * width + ix];
}
}  // namespace
cudaError_t fit_image(const void* source, void* target, int width, int height, int window_width,
                      int window_height, cudaStream_t stream) {
  const double scale = std::min(static_cast<double>(window_width) / width,
                                static_cast<double>(window_height) / height);
  // Compensate for the SDK window's final stretch. The content then has equal
  // physical X/Y scale at the actual window extent, with bounded black borders.
  const float cw = static_cast<float>(width * scale * width / window_width);
  const float ch = static_cast<float>(height * scale * height / window_height);
  fit_kernel<<<(width * height + 255) / 256, 256, 0, stream>>>(
      static_cast<const uchar4*>(source), static_cast<uchar4*>(target), width, height, cw, ch);
  return cudaPeekAtLastError();
}
}  // namespace holoscan::radar::app::viz

namespace holoscan::radar::app::viz {
namespace {
__device__ uchar4 sample_panel(const uchar4* source, int width, int height, float x, float y) {
  if (!source || x < 0 || y < 0 || x >= width || y >= height)
    return make_uchar4(0, 0, 0, 255);
  return source[min(height - 1, static_cast<int>(y)) * width + min(width - 1, static_cast<int>(x))];
}
__global__ void dock_kernel(const uchar4* image, const uchar4* footer, uchar4* target, int width,
                            int image_height, int footer_height, int window_width,
                            int window_height, bool fit) {
  const int height = image_height + footer_height;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width * height)
    return;
  const float x = (index % width + 0.5F) * window_width / width;
  const float y = (index / width + 0.5F) * window_height / height;
  const int fh = min(footer_height, max(1, window_height / 2));
  const int ih = max(1, window_height - fh);
  if (y >= ih) {
    target[index] = sample_panel(footer, width, footer_height, x * width / window_width,
                                 (y - ih) * footer_height / fh);
  } else {
    const float scale =
        fminf(static_cast<float>(window_width) / width, static_cast<float>(ih) / image_height);
    const float sx =
        fit ? (x - (window_width - width * scale) * 0.5F) / scale : x * width / window_width;
    const float sy = fit ? (y - (ih - image_height * scale) * 0.5F) / scale : y * image_height / ih;
    target[index] = sample_panel(image, width, image_height, sx, sy);
  }
}
}  // namespace
cudaError_t compose_docked_panels(const void* image, const void* footer, void* target, int width,
                                  int image_height, int footer_height, int window_width,
                                  int window_height, bool fit, cudaStream_t stream) {
  dock_kernel<<<(width * (image_height + footer_height) + 255) / 256, 256, 0, stream>>>(
      static_cast<const uchar4*>(image), static_cast<const uchar4*>(footer),
      static_cast<uchar4*>(target), width, image_height, footer_height, window_width, window_height,
      fit);
  return cudaGetLastError();
}
}  // namespace holoscan::radar::app::viz
