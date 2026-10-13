// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace holoscan::radar {

// Throw std::runtime_error (naming the operator and the failing site) when a
// CUDA error is pending. Call after a kernel launch or async API sequence to
// surface sticky launch/config errors at a defined point. `who` is the
// operator name; `where` labels the launch site, e.g. "assemble kernel".
inline void check_cuda(const char* where, const std::string& who) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(who + ": " + where + ": " + cudaGetErrorString(err));
  }
}

}  // namespace holoscan::radar
