// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace holoscan::radar {

// TDM slow-time firing pattern for MIMODeinterleaveOp (gather) and
// TDMDenseCompensationOp / DDMDenseDemultiplexOp (per-TX firing-slot index). Describes how an
// interleaved (M·L) chirp axis splits into explicit (M_tx, L) axes.
//
// v1 scope: uniform spacing — the length-M firing cycle repeats identically
// L times with equal per-TX counts. Custom `tx_order` permutations of
// [0, …, M−1] are allowed. Non-uniform / staggered schedules (varying TX
// dwell or count) are rejected (expand_chirp_tx_index throws), because TDM
// compensation models a fixed PRI/M inter-TX offset. Staggered TDM is a
// deferred future enhancement (STATUS.md → Roadmap; settled as L12).
struct TXSchedule {
  int64_t num_tx{0};
  // TX firing order within one repetition cycle; repeats L times. Empty →
  // round-robin [0, 1, …, M−1].
  std::vector<int64_t> tx_order{};
  // Explicit per-chirp TX index (length M·L). When set, overrides tx_order and
  // must still be a uniform repeating pattern (first-wave TDM requirement).
  std::vector<int64_t> chirp_tx_index{};

  void validate() const;

  // Expand to a length-`slow_time` TX index per interleaved chirp. `slow_time`
  // must equal num_tx * chirps_per_tx. Throws on non-uniform schedules.
  std::vector<int64_t> expand_chirp_tx_index(int64_t slow_time) const;

  // For each output slot (tx, l) with linear index tx * L + l, the interleaved
  // slow-time index to gather from.
  std::vector<int64_t> build_gather_map(int64_t slow_time) const;

  // Firing slot in the length-M cycle for each TX id: result[tx] = s where
  // the cycle fires TX `tx` in slot `s`. Empty default schedule (num_tx == 0
  // and both order vectors empty) returns empty — callers treat that as
  // identity (slot == TX id). Non-empty schedules must pass validate() /
  // expand_chirp_tx_index (uniform repeating cycle).
  std::vector<int64_t> firing_slot_by_tx() const;

  friend bool operator==(const TXSchedule& a, const TXSchedule& b) {
    return a.num_tx == b.num_tx && a.tx_order == b.tx_order && a.chirp_tx_index == b.chirp_tx_index;
  }
  friend bool operator!=(const TXSchedule& a, const TXSchedule& b) { return !(a == b); }
};

}  // namespace holoscan::radar
