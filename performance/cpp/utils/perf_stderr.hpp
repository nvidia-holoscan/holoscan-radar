// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Suppresses known, expected Holoscan SDK 4.3 stderr noise during perf runs.
// CFAR emits a bool detection_mask; the SDK logs an ERROR when bool tensors
// round-trip through certain code paths (pending HSDK bool/DLPack support).
// Filtering is line-based and matches only that message so other ERROR logs
// still reach stderr.

#pragma once

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>

namespace holoscan::radar::perf {

inline bool is_hsdk_bool_dlpack_noise(const std::string& line) {
  return line.find(
             "Unsupported DLPack data type (code: 6, bits: 8, lanes: 1)") !=
         std::string::npos;
}

// Run `fn` with stderr filtered through a pipe that drops the known bool noise.
inline void suppress_known_hsdk_noise_during(const std::function<void()>& fn) {
  int pipe_fd[2];
  if (pipe(pipe_fd) != 0) {
    fn();
    return;
  }

  const int saved_stderr = dup(STDERR_FILENO);
  if (saved_stderr < 0) {
    close(pipe_fd[0]);
    close(pipe_fd[1]);
    fn();
    return;
  }

  dup2(pipe_fd[1], STDERR_FILENO);
  close(pipe_fd[1]);

  std::thread reader([saved_stderr, pipe_read = pipe_fd[0]]() {
    std::string pending;
    char chunk[512];
    ssize_t n = 0;
    while ((n = read(pipe_read, chunk, sizeof(chunk))) > 0) {
      pending.append(chunk, static_cast<size_t>(n));
      size_t newline = 0;
      while ((newline = pending.find('\n')) != std::string::npos) {
        const std::string line = pending.substr(0, newline + 1);
        pending.erase(0, newline + 1);
        if (!is_hsdk_bool_dlpack_noise(line)) {
          (void)write(saved_stderr, line.data(), line.size());
        }
      }
    }
    if (!pending.empty() && !is_hsdk_bool_dlpack_noise(pending)) {
      (void)write(saved_stderr, pending.data(), pending.size());
    }
    close(pipe_read);
  });

  fn();
  fflush(stderr);
  dup2(saved_stderr, STDERR_FILENO);
  close(saved_stderr);
  reader.join();
}

}  // namespace holoscan::radar::perf
