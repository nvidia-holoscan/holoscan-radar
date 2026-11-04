// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <optional>
#include <stdexcept>
#include <string>

#include <holoscan/core/memory_kind.hpp>

namespace holoscan::radar::app {

// Discover renderer placement once before Graph construction, retaining the
// caller's compute device. Explicit image transfers join different render GPUs.
int select_application_device(bool window, bool cuda_sink);

// FMCW uses the SDK's offscreen renderer and finished-image X11 transport when
// enabled and an X11 display is available; otherwise use SDK presentation.
bool fmcw_uses_x11_presentation(bool enabled);
bool resolve_fmcw_x11_presentation(bool window, std::optional<bool> configured,
                                   bool choose_again = false, const std::string& config_path = {});
int select_fmcw_application_device(bool window, bool cuda_sink, bool x11_presentation);

// Window adapter found by the most recent startup selection on this thread.
int application_display_device();
int application_composer_device();

inline holoscan::DeviceId application_device() {
  int ordinal = -1;
  const auto status = cudaGetDevice(&ordinal);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA device query failed: ") +
                             cudaGetErrorString(status));
  }
  return holoscan::DeviceId{ordinal};
}

}  // namespace holoscan::radar::app
