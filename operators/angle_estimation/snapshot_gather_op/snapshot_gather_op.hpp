// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/channel/pod_type.hpp>
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

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::radar {

// Fixed-layout live payload for snapshot_window. Params retain the public
// std::vector<int64_t> representation; the graph port uses this bounded POD.
struct SnapshotWindowOverride {
  int64_t doppler{0};
  int64_t range{0};
};

static_assert(sizeof(SnapshotWindowOverride) == 16U);
static_assert(alignof(SnapshotWindowOverride) == 8U);
static_assert(std::is_standard_layout_v<SnapshotWindowOverride>);
static_assert(std::is_trivially_copyable_v<SnapshotWindowOverride>);

}  // namespace holoscan::radar

namespace holoscan {

template <>
struct ChannelTypeIdentity<radar::SnapshotWindowOverride> {
  static constexpr std::string_view name = "com.nvidia.holoscan.radar.SnapshotWindowOverride";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<radar::SnapshotWindowOverride> {
  static constexpr auto fields =
      pod_fields(pod_field("doppler", &radar::SnapshotWindowOverride::doppler,
                           offsetof(radar::SnapshotWindowOverride, doppler)),
                 pod_field("range", &radar::SnapshotWindowOverride::range,
                           offsetof(radar::SnapshotWindowOverride, range)));
};

}  // namespace holoscan

namespace holoscan::radar {

namespace detail {
struct SnapshotGatherState;
}  // namespace detail

// Gathers per-detection channel snapshots from per-channel RD maps for the
// sparse angle path. This header is the contract reference for the
// operator's ports, shapes, dtypes, and parameters; module-wide
// conventions live in operators/common/README.md.
//
// In:  rd_maps        — complex64/complex128, (..., num_channels, num_doppler,
//                       num_range). Channel axis immediately before the
//                       (doppler, range) plane.
// In:  detection_idx  — int64, (num_detections, 2 [+ n_batch]). Columns are
//                       (…preserved_batch, doppler_bin, range_bin).
// Out: snapshot       — complex64/complex128,
//                       (num_detections, num_channels, num_snapshots).
// Out: is_valid       — optional/always uint8 (0/1),
//                       (num_detections, num_snapshots). Per-detection mask
//                       of usable snapshot slots (ragged under clamp); 1 =
//                       valid. uint8 rather than bool so the mask is
//                       NumPy-visible (kDLBool has no NumPy typestr on the
//                       Holoscan message path). Consumers accept bool or
//                       uint8.
//
// Precision is fixed at compile time by the template instantiation.

struct SnapshotGatherParams {
  std::vector<int64_t> snapshot_window{0, 0};
  SnapshotEdgeMode doppler_edge{SnapshotEdgeMode::kWrap};
  SnapshotEdgeMode range_edge{SnapshotEdgeMode::kShift};
};

HOLOSCAN_PARAM_SCHEMA(
    SnapshotGatherParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.SnapshotGatherParams", 1),
    holoscan::field<&SnapshotGatherParams::snapshot_window>("snapshot_window")
        .description("Per-axis (doppler, range) neighborhood half-size around each "
                     "detection; empty→{0,0}, size-1→isotropic"),
    holoscan::field<&SnapshotGatherParams::doppler_edge>("doppler_edge")
        .description("Neighborhood behavior at the Doppler-axis edges "
                     "{wrap, clamp, shift}"),
    holoscan::field<&SnapshotGatherParams::range_edge>("range_edge")
        .description("Neighborhood behavior at the range-axis edges "
                     "{wrap, clamp, shift}"));

static_assert(holoscan::param_schema_availability<SnapshotGatherParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in SnapshotGatherParams's "
              "namespace");

template <typename T>
class SnapshotGatherOp : public holoscan::Operator<SnapshotGatherParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "SnapshotGatherOp supports float and double precision");

 public:
  explicit SnapshotGatherOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  Input<Tensor> detection_idx{"detection_idx"};
  Output<Tensor> snapshot{"snapshot"};
  Output<Tensor> is_valid{"is_valid"};
  // [live] Exact override of params().snapshot_window; the entry is
  // unconditional and wiring the port enables it (the operator README).
  // Fixed-layout port payload; see SnapshotWindowOverride.
  // Edge modes are static Params only (no live ports).
  Input<SnapshotWindowOverride> snapshot_window{"snapshot_window"};

 private:
  // Keep the previous activation's inputs alive across ticks so async device
  // reads cannot outlive the buffers.
  Tensor retained_rd_maps_;
  Tensor retained_detection_idx_;
  std::shared_ptr<detail::SnapshotGatherState> state_;

  std::int64_t max_detections_{0};
  std::int64_t max_channels_{0};
  std::int64_t max_snapshots_{0};
  std::size_t snapshot_max_byte_span_{0U};
  std::size_t is_valid_max_byte_span_{0U};
};

extern template class SnapshotGatherOp<float>;
extern template class SnapshotGatherOp<double>;

}  // namespace holoscan::radar
