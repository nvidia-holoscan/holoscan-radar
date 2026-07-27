// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// Gathers per-detection channel snapshots from per-channel RD maps for the
// sparse angle path. Contract reference: ports, shapes, dtypes, parameters.
//
// In:  rd_maps        — complex64/complex128, (..., num_channels, num_doppler,
//                       num_range). Channel axis immediately before the
//                       (doppler, range) plane.
// In:  detection_idx  — int64, (num_detections, 2 [+ n_batch]). Columns are
//                       (…preserved_batch, doppler_bin, range_bin).
// Out: snapshot       — complex64/complex128,
//                       (num_detections, num_channels, num_snapshots).
// Out: is_valid       — uint8 (0/1), (num_detections, num_snapshots). Per-
//                       detection mask of usable snapshot slots (ragged under
//                       clamp); 1 = valid. uint8 rather than bool so the mask
//                       is NumPy-visible on HSDK 4.3 (kDLBool has no NumPy
//                       typestr on the 4.3 message path). Consumers accept bool
//                       or uint8.
//
// Precision is fixed at compile time (T = float | double).
template <typename T>
class SnapshotGatherOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "SnapshotGatherOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SnapshotGatherOp)

  SnapshotGatherOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::vector<int64_t>> snapshot_window_;  // (doppler, range) half
  Parameter<SnapshotEdgeMode> doppler_edge_;
  Parameter<SnapshotEdgeMode> range_edge_;

  std::vector<int64_t> effective_snapshot_window_{0, 0};
  bool seeded_{false};

  std::shared_ptr<holoscan::Tensor> retained_rd_maps_;
  std::shared_ptr<holoscan::Tensor> retained_detection_idx_;
  std::shared_ptr<holoscan::Tensor> retained_leading_dims_;
};

}  // namespace holoscan::radar
