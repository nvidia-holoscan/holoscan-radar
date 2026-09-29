// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

// One contiguous pulse slice of a retained (or incoming) batch. `first`
// and `count` select along the pulse axis (second-to-last); the three
// tensors share that axis. Holoscan-free in spirit — the mechanical
// ring + concat — but the handles are holoscan::Tensor because that is
// what the operators already pass.
struct PulseGroup {
  std::shared_ptr<holoscan::Tensor> profiles;
  std::shared_ptr<holoscan::Tensor> positions;
  std::shared_ptr<holoscan::Tensor> reference;
  int64_t first{0};
  int64_t count{0};
};

inline int64_t pulse_group_total(const std::vector<PulseGroup>& groups) {
  int64_t total = 0;
  for (const PulseGroup& group : groups) {
    total += group.count;
  }
  return total;
}

inline size_t dtype_bytes(DLDataType dtype) {
  return ((static_cast<size_t>(dtype.bits) + 7) / 8) * static_cast<size_t>(dtype.lanes);
}

// Copy `count` pulses from src[src_first:] into dst[dst_first:] along
// the pulse axis of a compact (..., pulses, cols) tensor.
inline void copy_pulse_rows(const holoscan::Tensor& src, int64_t src_first, holoscan::Tensor& dst,
                            int64_t dst_first, int64_t count, cudaStream_t stream,
                            const char* who) {
  if (count <= 0) {
    return;
  }
  const BatchedShape s = collapse_batch(src, "aperture_concat");
  const BatchedShape d = collapse_batch(dst, "aperture_concat");
  if (s.cols != d.cols || s.batch != d.batch) {
    throw std::invalid_argument(std::string(who) + ": pulse concat shape mismatch");
  }
  if (src_first < 0 || count < 0 || src_first + count > s.rows || dst_first < 0 ||
      dst_first + count > d.rows) {
    throw std::invalid_argument(std::string(who) + ": pulse concat slice out of range");
  }
  const size_t pulse_bytes = static_cast<size_t>(s.cols) * dtype_bytes(src.dtype());
  const auto* src_bytes = static_cast<const char*>(src.data());
  auto* dst_bytes = static_cast<char*>(const_cast<void*>(dst.data()));
  for (int64_t b = 0; b < s.batch; ++b) {
    const char* sp = src_bytes + static_cast<size_t>(b * s.rows + src_first) * pulse_bytes;
    char* dp = dst_bytes + static_cast<size_t>(b * d.rows + dst_first) * pulse_bytes;
    if (cudaMemcpyAsync(dp, sp, pulse_bytes * static_cast<size_t>(count), cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
      throw std::runtime_error(std::string(who) + ": pulse concat copy failed");
    }
  }
}

// Concatenate groups along the pulse axis. A single full-batch group
// with no companions is returned in place (zero copy).
inline PulseGroup concat_pulse_groups(const std::vector<PulseGroup>& groups, cudaStream_t stream,
                                      const char* who) {
  PulseGroup out;
  const int64_t total = pulse_group_total(groups);
  if (total <= 0 || groups.empty()) {
    return out;
  }
  if (groups.size() == 1 && groups[0].first == 0) {
    const BatchedShape shape = collapse_batch(*groups[0].profiles, "range_profiles");
    if (groups[0].count == shape.rows) {
      return groups[0];
    }
  }
  const holoscan::Tensor& p0 = *groups[0].profiles;
  const holoscan::Tensor& x0 = *groups[0].positions;
  const holoscan::Tensor& r0 = *groups[0].reference;
  const BatchedShape cube = collapse_batch(p0, "range_profiles");
  const BatchedShape pos = collapse_batch(x0, "positions");
  const BatchedShape ref = collapse_batch(r0, "reference_position");
  out.profiles = make_device_tensor(full_rank_shape(cube, total, cube.cols), p0.dtype());
  out.positions = make_device_tensor(full_rank_shape(pos, total, pos.cols), x0.dtype());
  out.reference = make_device_tensor(full_rank_shape(ref, total, ref.cols), r0.dtype());
  out.first = 0;
  out.count = total;
  int64_t cursor = 0;
  for (const PulseGroup& group : groups) {
    if (group.count <= 0) {
      continue;
    }
    copy_pulse_rows(*group.profiles, group.first, *out.profiles, cursor, group.count, stream, who);
    copy_pulse_rows(*group.positions, group.first, *out.positions, cursor, group.count, stream,
                    who);
    copy_pulse_rows(*group.reference, group.first, *out.reference, cursor, group.count, stream,
                    who);
    cursor += group.count;
  }
  return out;
}

inline std::vector<int64_t> weight_shape_for(const BatchedShape& cube, int64_t pulses) {
  std::vector<int64_t> shape = cube.leading;
  shape.push_back(pulses);
  return shape;
}

// Host-side ring of pulse batches. Capacity 0 means "do not retain".
class ApertureBuffer {
 public:
  explicit ApertureBuffer(int64_t capacity_pulses) : capacity_(capacity_pulses) {
    if (capacity_pulses < 0) {
      throw std::invalid_argument("ApertureBuffer: max_aperture_pulses must be >= 0");
    }
  }

  int64_t capacity() const { return capacity_; }
  int64_t occupancy() const { return occupancy_; }
  bool retaining() const { return capacity_ > 0; }

  void admit(PulseGroup group) {
    if (capacity_ == 0 || group.count <= 0) {
      return;
    }
    occupancy_ += group.count;
    batches_.push_back(std::move(group));
  }

  std::vector<PulseGroup> evict(int64_t count) {
    std::vector<PulseGroup> out;
    int64_t remain = count;
    while (remain > 0 && !batches_.empty()) {
      PulseGroup& front = batches_.front();
      const int64_t take = remain < front.count ? remain : front.count;
      PulseGroup slice = front;
      slice.count = take;
      out.push_back(slice);
      front.first += take;
      front.count -= take;
      occupancy_ -= take;
      remain -= take;
      if (front.count == 0) {
        batches_.pop_front();
      }
    }
    return out;
  }

  std::vector<PulseGroup> evict_for_admission(int64_t incoming) {
    if (capacity_ == 0) {
      return {};
    }
    const int64_t overflow = occupancy_ + incoming - capacity_;
    if (overflow <= 0) {
      return {};
    }
    return evict(overflow);
  }

  void clear() {
    batches_.clear();
    occupancy_ = 0;
  }

  std::vector<PulseGroup> retained() const {
    std::vector<PulseGroup> out;
    out.assign(batches_.begin(), batches_.end());
    return out;
  }

 private:
  int64_t capacity_{0};
  int64_t occupancy_{0};
  std::deque<PulseGroup> batches_;
};

}  // namespace holoscan::radar
