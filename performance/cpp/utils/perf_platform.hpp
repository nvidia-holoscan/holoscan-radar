// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Best-effort provenance + platform capture for the performance report
// header. Every field degrades to "unknown" rather than aborting a run, so a
// benchmark still produces a usable report on an unfamiliar host.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>

#include <holoscan/version_config.hpp>

namespace holoscan::radar::perf {

struct Provenance {
  std::string git_commit{"unknown"};
  bool git_dirty{false};
  std::string timestamp_utc{"unknown"};
};

struct Platform {
  std::string gpu_name{"unknown"};
  std::string gpu_driver{"unknown"};
  std::string cuda_runtime{"unknown"};
  std::string holoscan_version{"unknown"};
  std::string cpu_model{"unknown"};
  std::string os_name{"unknown"};
};

namespace detail {

// Run a shell command and return its trimmed stdout, or "" on failure.
inline std::string run_capture(const char* cmd) {
  std::array<char, 256> buf{};
  std::string out;
  FILE* pipe = popen(cmd, "r");
  if (pipe == nullptr) {
    return "";
  }
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) !=
         nullptr) {
    out += buf.data();
  }
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' ||
                          out.back() == ' ')) {
    out.pop_back();
  }
  return out;
}

inline std::string first_matching_line(const std::string& path,
                                       const std::string& key) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind(key, 0) == 0) {
      const auto colon = line.find(':');
      if (colon != std::string::npos) {
        std::string value = line.substr(colon + 1);
        const auto start = value.find_first_not_of(" \t");
        return start == std::string::npos ? "" : value.substr(start);
      }
    }
  }
  return "";
}

}  // namespace detail

inline Provenance capture_provenance() {
  Provenance p;
  const std::string commit = detail::run_capture(
      "git rev-parse --short HEAD 2>/dev/null");
  if (!commit.empty()) {
    p.git_commit = commit;
  }
  // Non-empty porcelain output means the working tree has local changes.
  p.git_dirty = !detail::run_capture(
                     "git status --porcelain 2>/dev/null")
                     .empty();

  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::array<char, 32> ts{};
  std::tm tm_utc{};
  gmtime_r(&t, &tm_utc);
  if (std::strftime(ts.data(), ts.size(), "%Y-%m-%dT%H:%M:%SZ", &tm_utc) > 0) {
    p.timestamp_utc = ts.data();
  }
  return p;
}

inline Platform capture_platform() {
  Platform p;

  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) == cudaSuccess &&
      cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    p.gpu_name = prop.name;
  }

  int driver = 0;
  if (cudaDriverGetVersion(&driver) == cudaSuccess && driver > 0) {
    p.gpu_driver = std::to_string(driver / 1000) + "." +
                   std::to_string((driver % 1000) / 10);
  }
  int runtime = 0;
  if (cudaRuntimeGetVersion(&runtime) == cudaSuccess && runtime > 0) {
    p.cuda_runtime = std::to_string(runtime / 1000) + "." +
                     std::to_string((runtime % 1000) / 10);
  }

  p.holoscan_version = std::to_string(HOLOSCAN_VERSION_MAJOR) + "." +
                       std::to_string(HOLOSCAN_VERSION_MINOR) + "." +
                       std::to_string(HOLOSCAN_VERSION_PATCH);

  const std::string cpu = detail::first_matching_line("/proc/cpuinfo",
                                                      "model name");
  if (!cpu.empty()) {
    p.cpu_model = cpu;
  }

  const std::string os = detail::run_capture(
      "uname -sr 2>/dev/null");
  if (!os.empty()) {
    p.os_name = os;
  }
  return p;
}

}  // namespace holoscan::radar::perf
