// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

namespace detail {
struct SubarraySelectionState;
}  // namespace detail

// Selects a uniform ULA/URA sub-array from a (possibly holey) MIMO virtual
// array and re-emits a reduced (snapshot, geometry) pair — S3. Sits between
// SnapshotGatherOp and the gridless estimators (Root-MUSIC / ESPRIT) or in
// front of MVDR/MUSIC spatial smoothing on a holey virtual array.
//
// The selection depends only on the geometry (+ optional explicit indices), so
// it is planned once and cached; per tick only the channel gather runs (the S1
// gather with a single sub-array, no forward-backward).
//
// In:  snapshot       — complex64/complex128, (num_items, N_virtual) or
//                       (num_items, N_virtual, num_snapshots).
// In:  is_valid       — optional bool/uint8, (num_items, num_snapshots).
//                       Passed through unchanged when present.
// Out: snapshot_out   — complex64/complex128, (num_items, L[, num_snapshots]).
//                       [EA HSDK5 ONLY] unique in/out port names (was
//                       "snapshot").
// Out: array_geometry — optional/latched float64 Tensor, shape (L + 1, 3):
//                       packed reduced
//                       ArrayGeometry (see array_geometry_tensor.hpp).
//                       [EA HSDK5 ONLY] Output<ArrayGeometry> is not a
//                       PublishedPayload (owns vector/optional); serialized
//                       Tensor interim — HSDK5_PORT.md §10.
// Out: is_valid_out   — optional; the input mask, forwarded.
//                       [EA HSDK5 ONLY] renamed from "is_valid".
//
// Precision is fixed at compile time (T = float | double).
//
struct SubarraySelectionParams {
  ArrayGeometry array_geometry{};
  // Explicit sub-array channel indices; empty → auto (largest filled ULA/URA).
  std::vector<int64_t> subarray{};
  // Minimum acceptable auto-selected sub-array size (error below it).
  int64_t min_subarray_size{2};
};

HOLOSCAN_PARAM_SCHEMA(
    SubarraySelectionParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.SubarraySelectionParams", 1),
    holoscan::field<&SubarraySelectionParams::array_geometry>("array_geometry")
        .description(
            "Holey virtual-array element positions and wavelength; "
            "num_elements must match snapshot channels"),
    holoscan::field<&SubarraySelectionParams::subarray>("subarray")
        .description(
            "Explicit channel indices (empty = auto largest filled "
            "ULA/URA)"),
    holoscan::field<&SubarraySelectionParams::min_subarray_size>(
        "min_subarray_size")
        .description(
            "Auto-selection errors below this many channels"));

static_assert(
    holoscan::param_schema_availability<SubarraySelectionParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in SubarraySelectionParams's namespace");

template <typename T>
class SubarraySelectionOp
    : public holoscan::Operator<SubarraySelectionParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "SubarraySelectionOp supports float and double precision");

 public:
  explicit SubarraySelectionOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> snapshot{"snapshot"};
  // Optional; unconnected graphs omit it from the temporal contract.
  Input<Tensor> is_valid{"is_valid"};
  // [live] Exact overrides. The geometry input uses a distinct physical name
  // because HSDK 5 requires unique input/output port names.
  Input<WireArrayGeometry> array_geometry_in{"array_geometry_in"};
  Input<Tensor> subarray{"subarray"};
  // [EA HSDK5 ONLY] unique in/out names — see OPERATOR_API.md / HSDK5_PORT §2.
  Output<Tensor> snapshot_out{"snapshot_out"};
  // Packed ArrayGeometry (float64 Tensor) — HSDK5_PORT §10.
  Output<Tensor> array_geometry{"array_geometry"};
  Output<Tensor> is_valid_out{"is_valid_out"};

 private:
  // Keep prior activation inputs alive while asynchronous copies/gathers read.
  Tensor retained_snapshot_;
  Tensor retained_is_valid_;
  // Independent optional/latched outputs (OPERATOR_API.md).
  LatchedOutput array_geometry_out_;
  LatchedOutput is_valid_out_;
  std::shared_ptr<detail::SubarraySelectionState> state_;

  std::size_t snapshot_max_byte_span_{0U};
  std::size_t geom_max_byte_span_{0U};
  std::size_t is_valid_max_byte_span_{0U};
  std::int64_t subarray_max_extent_{0};
  std::int64_t max_items_{0};
  std::int64_t max_snapshots_{0};
};

extern template class SubarraySelectionOp<float>;
extern template class SubarraySelectionOp<double>;

}  // namespace holoscan::radar
