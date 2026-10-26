// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include <holoscan/core/memory_kind.hpp>

namespace holoscan::radar::app {

// Discover renderer placement once before Graph construction, retaining the
// caller's compute device. Explicit image transfers join different render GPUs.
int select_application_device(bool window, bool cuda_sink);

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
