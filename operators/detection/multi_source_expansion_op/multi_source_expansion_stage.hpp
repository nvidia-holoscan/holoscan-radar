// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cmath>

namespace holoscan::radar::detail {

// CUDA atomicMin/atomicAdd require this exact 64-bit overload type.
using ExpansionAtomicWord = unsigned long long;  // NOLINT(runtime/int)

enum class ExpansionError : int32_t {
  kNone = 0,
  kSelectedOrder = 1,
  kSourceCount = 2,
  kNonFiniteDoa = 3,
};

template <typename T>
__global__ void build_expansion_plan_kernel(const int64_t* selected, int64_t num_selected,
                                            const T* resolved, const T* multi,
                                            const int32_t* counts, int64_t num_items,
                                            int64_t max_sources, uint32_t* row_counts,
                                            int64_t* selected_ordinal, int64_t* total,
                                            int32_t* error) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  *error = static_cast<int32_t>(ExpansionError::kNone);
  for (int64_t i = 0; i < num_items; ++i) {
    row_counts[i] = 1;
    selected_ordinal[i] = -1;
  }
  int64_t previous = -1;
  int64_t output_count = num_items;
  for (int64_t s = 0; s < num_selected; ++s) {
    const int64_t item = selected[s];
    if (item <= previous || item < 0 || item >= num_items) {
      *error = static_cast<int32_t>(ExpansionError::kSelectedOrder);
      return;
    }
    previous = item;
    selected_ordinal[item] = s;
    const int32_t count = counts[s];
    if (count < 0 || count > max_sources) {
      *error = static_cast<int32_t>(ExpansionError::kSourceCount);
      return;
    }
    if (count >= 2) {
      if (!isfinite(static_cast<double>(resolved[s * 2])) ||
          !isfinite(static_cast<double>(resolved[s * 2 + 1]))) {
        *error = static_cast<int32_t>(ExpansionError::kNonFiniteDoa);
        return;
      }
      for (int32_t k = 1; k < count; ++k) {
        const int64_t base = (s * max_sources + k) * 2;
        if (!isfinite(static_cast<double>(multi[base])) ||
            !isfinite(static_cast<double>(multi[base + 1]))) {
          *error = static_cast<int32_t>(ExpansionError::kNonFiniteDoa);
          return;
        }
      }
      row_counts[item] = static_cast<uint32_t>(count);
      output_count += count - 1;
    }
  }
  *total = output_count;
}

// Separate initialization from the validation grid so no block can reset a
// partial total/error recorded by another block.
static __global__ void initialize_parallel_expansion_plan_kernel(int64_t num_items, int64_t* total,
                                                                 ExpansionAtomicWord* first_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *total = num_items;
    *first_error = ~0ULL;
  }
}

template <typename T>
__global__ void validate_parallel_expansion_plan_kernel(const int64_t* selected,
                                                        int64_t num_selected, const T* resolved,
                                                        const T* multi, const int32_t* counts,
                                                        int64_t num_items, int64_t max_sources,
                                                        int64_t* total,
                                                        ExpansionAtomicWord* first_error) {
  const int64_t ordinal = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ordinal >= num_selected)
    return;
  const int64_t item = selected[ordinal];
  const int64_t previous = ordinal == 0 ? -1 : selected[ordinal - 1];
  ExpansionError error = ExpansionError::kNone;
  int32_t count = 0;
  // Match serial within-row precedence without indexing scratch by an invalid
  // or duplicate item. The minimum packed key then preserves first-row errors.
  if (item <= previous || item < 0 || item >= num_items) {
    error = ExpansionError::kSelectedOrder;
  } else {
    count = counts[ordinal];
    if (count < 0 || count > max_sources) {
      error = ExpansionError::kSourceCount;
    } else if (count >= 2) {
      if (!isfinite(static_cast<double>(resolved[ordinal * 2])) ||
          !isfinite(static_cast<double>(resolved[ordinal * 2 + 1]))) {
        error = ExpansionError::kNonFiniteDoa;
      } else {
        for (int32_t source = 1; source < count; ++source) {
          const int64_t base = (ordinal * max_sources + source) * 2;
          if (!isfinite(static_cast<double>(multi[base])) ||
              !isfinite(static_cast<double>(multi[base + 1]))) {
            error = ExpansionError::kNonFiniteDoa;
            break;
          }
        }
      }
    }
  }
  if (error != ExpansionError::kNone) {
    const auto key =
        (static_cast<ExpansionAtomicWord>(ordinal) << 2U) | static_cast<ExpansionAtomicWord>(error);
    atomicMin(first_error, key);
  } else if (count >= 2) {
    atomicAdd(reinterpret_cast<ExpansionAtomicWord*>(total),
              static_cast<ExpansionAtomicWord>(count - 1));
  }
}

// Only launch after successful validation. Sorted unique selections make a
// per-item lower_bound race-free and assign every scratch element exactly once.
static __global__ void materialize_parallel_expansion_plan_kernel(
    const int64_t* selected, int64_t num_selected, const int32_t* counts, int64_t num_items,
    uint32_t* row_counts, int64_t* selected_ordinal) {
  const int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= num_items)
    return;
  int64_t lower = 0;
  int64_t upper = num_selected;
  while (lower < upper) {
    const int64_t middle = lower + (upper - lower) / 2;
    if (selected[middle] < item) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  const bool found = lower < num_selected && selected[lower] == item;
  selected_ordinal[item] = found ? lower : -1;
  const int32_t count = found ? counts[lower] : 0;
  row_counts[item] = count >= 2 ? static_cast<uint32_t>(count) : 1U;
}

template <typename T>
__global__ void scatter_expanded_rows_kernel(const int64_t* detection_idx, const T* baseline,
                                             const int64_t* selected_ordinal, const T* resolved,
                                             const T* multi, const int32_t* counts,
                                             const uint32_t* offsets, int64_t num_items,
                                             int64_t idx_cols, int64_t max_sources,
                                             int64_t* idx_out, T* aoa_out) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items)
    return;
  const int64_t selected = selected_ordinal[item];
  const int32_t count = selected >= 0 ? counts[selected] : 0;
  const int64_t copies = count >= 2 ? count : 1;
  const int64_t output = static_cast<int64_t>(offsets[item]);
  for (int64_t k = threadIdx.x; k < copies * idx_cols; k += blockDim.x) {
    idx_out[output * idx_cols + k] = detection_idx[item * idx_cols + (k % idx_cols)];
  }
  if (threadIdx.x == 0) {
    if (count >= 2) {
      aoa_out[output * 2] = resolved[selected * 2];
      aoa_out[output * 2 + 1] = resolved[selected * 2 + 1];
      for (int32_t k = 1; k < count; ++k) {
        const int64_t src = (selected * max_sources + k) * 2;
        aoa_out[(output + k) * 2] = multi[src];
        aoa_out[(output + k) * 2 + 1] = multi[src + 1];
      }
    } else {
      aoa_out[output * 2] = baseline[item * 2];
      aoa_out[output * 2 + 1] = baseline[item * 2 + 1];
    }
  }
}

}  // namespace holoscan::radar::detail
