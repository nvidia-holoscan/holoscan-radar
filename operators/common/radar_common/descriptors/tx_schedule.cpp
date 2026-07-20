// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/descriptors/tx_schedule.hpp"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace holoscan::radar {

namespace {

void require_positive(const char* name, int64_t value) {
  if (value <= 0) {
    throw std::invalid_argument(std::string("TXSchedule: ") + name +
                                " must be positive");
  }
}

bool is_uniform_repeating_pattern(const std::vector<int64_t>& pattern,
                                  int64_t num_tx) {
  if (static_cast<int64_t>(pattern.size()) % num_tx != 0) {
    return false;
  }
  for (size_t k = static_cast<size_t>(num_tx); k < pattern.size(); ++k) {
    if (pattern[k] != pattern[k % static_cast<size_t>(num_tx)]) {
      return false;
    }
  }
  return true;
}

void validate_tx_indices(const std::vector<int64_t>& pattern, int64_t num_tx) {
  for (int64_t tx : pattern) {
    if (tx < 0 || tx >= num_tx) {
      throw std::invalid_argument(
          "TXSchedule: chirp TX index out of range [0, num_tx)");
    }
  }
}

}  // namespace

void TXSchedule::validate() const {
  require_positive("num_tx", num_tx);
  if (!tx_order.empty()) {
    if (static_cast<int64_t>(tx_order.size()) != num_tx) {
      throw std::invalid_argument(
          "TXSchedule: tx_order length must equal num_tx");
    }
    std::vector<bool> seen(static_cast<size_t>(num_tx), false);
    for (int64_t tx : tx_order) {
      if (tx < 0 || tx >= num_tx) {
        throw std::invalid_argument(
            "TXSchedule: tx_order entry out of range [0, num_tx)");
      }
      seen[static_cast<size_t>(tx)] = true;
    }
    for (bool hit : seen) {
      if (!hit) {
        throw std::invalid_argument(
            "TXSchedule: tx_order must be a permutation of [0, num_tx)");
      }
    }
  }
}

std::vector<int64_t> TXSchedule::expand_chirp_tx_index(
    int64_t slow_time) const {
  validate();
  if (slow_time <= 0 || slow_time % num_tx != 0) {
    throw std::invalid_argument(
        "TXSchedule: slow_time must be a positive multiple of num_tx");
  }

  std::vector<int64_t> pattern;
  if (!chirp_tx_index.empty()) {
    if (static_cast<int64_t>(chirp_tx_index.size()) != slow_time) {
      throw std::invalid_argument(
          "TXSchedule: chirp_tx_index length must equal slow_time");
    }
    pattern = chirp_tx_index;
  } else {
    const std::vector<int64_t> cycle =
        tx_order.empty()
            ? [&]() {
                std::vector<int64_t> rr(static_cast<size_t>(num_tx));
                for (int64_t m = 0; m < num_tx; ++m) {
                  rr[static_cast<size_t>(m)] = m;
                }
                return rr;
              }()
            : tx_order;
    pattern.reserve(static_cast<size_t>(slow_time));
    for (int64_t k = 0; k < slow_time; ++k) {
      pattern.push_back(cycle[static_cast<size_t>(k % num_tx)]);
    }
  }

  validate_tx_indices(pattern, num_tx);
  if (!is_uniform_repeating_pattern(pattern, num_tx)) {
    throw std::invalid_argument(
        "TXSchedule: non-uniform / non-repeating schedule is rejected for "
        "first-wave TDM");
  }

  const int64_t chirps_per_tx = slow_time / num_tx;
  std::vector<int64_t> counts(static_cast<size_t>(num_tx), 0);
  for (int64_t tx : pattern) {
    ++counts[static_cast<size_t>(tx)];
  }
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    if (counts[static_cast<size_t>(tx)] != chirps_per_tx) {
      throw std::invalid_argument(
          "TXSchedule: each TX must appear exactly chirps_per_tx times");
    }
  }
  return pattern;
}

std::vector<int64_t> TXSchedule::build_gather_map(int64_t slow_time) const {
  const std::vector<int64_t> pattern = expand_chirp_tx_index(slow_time);
  const int64_t chirps_per_tx = slow_time / num_tx;
  std::vector<int64_t> gather(static_cast<size_t>(num_tx * chirps_per_tx), -1);
  std::vector<int64_t> seen(static_cast<size_t>(num_tx), 0);
  for (int64_t k = 0; k < slow_time; ++k) {
    const int64_t tx = pattern[static_cast<size_t>(k)];
    const int64_t l = seen[static_cast<size_t>(tx)]++;
    gather[static_cast<size_t>(tx * chirps_per_tx + l)] = k;
  }
  for (int64_t idx : gather) {
    if (idx < 0) {
      throw std::logic_error("TXSchedule: internal gather map build failed");
    }
  }
  return gather;
}

}  // namespace holoscan::radar
