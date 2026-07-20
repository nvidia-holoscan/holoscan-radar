// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/dsp/subarray_selection.hpp"

namespace holoscan::radar {

// Selects a uniform ULA/URA sub-array from a (possibly holey) MIMO virtual
// array and re-emits a reduced (snapshot, geometry) pair — team-blocked item
// S3. Sits between SnapshotGatherOp and the gridless estimators (Root-MUSIC /
// ESPRIT, which require a uniform geometry) or in front of MVDR/MUSIC spatial
// smoothing on a holey virtual array.
//
// The selection depends only on the geometry (+ optional explicit indices), so
// it is planned once and cached; per tick only the channel gather runs (the S1
// gather with a single sub-array, no forward-backward).
//
// In:  snapshot       — complex64/complex128, (num_items, N_virtual) or
//                       (num_items, N_virtual, num_snapshots). The holey
//                       virtual-array snapshot from SnapshotGatherOp.
// In:  is_valid       — optional bool/uint8, (num_items, num_snapshots). Passed
//                       through unchanged (channel selection leaves it intact).
// Out: snapshot       — complex64/complex128, (num_items, L[, num_snapshots]).
//                       The selected-channel snapshot (L = sub-array size).
// Out: array_geometry — ArrayGeometry. The reduced ULA/URA the selected
//                       channels form, for the estimator's [live] override.
// Out: is_valid       — optional; the input mask, forwarded.
//
// Precision is fixed at compile time (T = float | double).
template <typename T>
class SubarraySelectionOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "SubarraySelectionOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SubarraySelectionOp)

  SubarraySelectionOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<ArrayGeometry> array_geometry_;
  // Explicit sub-array channel indices; empty → auto (largest filled ULA/URA).
  // [live].
  Parameter<std::vector<int64_t>> subarray_;
  // Minimum acceptable auto-selected sub-array size (error below it).
  Parameter<int64_t> min_subarray_size_;

  ArrayGeometry effective_geometry_{};
  std::vector<int64_t> effective_subarray_{};
  int64_t effective_min_{2};
  bool seeded_{false};

  // Selection plan + uploaded channel-index table, rebuilt only when the
  // geometry / selection config changes (keyed by sel_key_).
  SubarraySelection selection_{};
  std::string sel_key_{};
  std::shared_ptr<holoscan::Tensor> d_channel_index_{};
};

}  // namespace holoscan::radar
