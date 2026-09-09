// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"

namespace holoscan::radar {

struct TDMSparseWindowOverride {
  int64_t doppler{0};
  int64_t range{0};
};

static_assert(sizeof(TDMSparseWindowOverride) == 16U);
static_assert(alignof(TDMSparseWindowOverride) == 8U);
static_assert(std::is_standard_layout_v<TDMSparseWindowOverride>);
static_assert(std::is_trivially_copyable_v<TDMSparseWindowOverride>);

}  // namespace holoscan::radar

namespace holoscan {

template <>
struct ChannelTypeIdentity<radar::TDMSparseWindowOverride> {
  static constexpr std::string_view name =
      "com.nvidia.holoscan.radar.TDMSparseWindowOverride";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<radar::TDMSparseWindowOverride> {
  static constexpr auto fields = pod_fields(
      pod_field("doppler", &radar::TDMSparseWindowOverride::doppler,
                offsetof(radar::TDMSparseWindowOverride, doppler)),
      pod_field("range", &radar::TDMSparseWindowOverride::range,
                offsetof(radar::TDMSparseWindowOverride, range)));
};

}  // namespace holoscan

namespace holoscan::radar {

struct TDMSparseCompensationParams {
  FMCWWaveformDescriptor waveform{};
  TXSchedule tx_schedule{};
  std::vector<double> tx_positions{};
  std::vector<double> rx_positions{};
  OverlapPolicy overlap_policy{OverlapPolicy::kDedup};
  std::vector<int64_t> snapshot_window{0, 0};
  SnapshotEdgeMode doppler_edge{SnapshotEdgeMode::kWrap};
  SnapshotEdgeMode range_edge{SnapshotEdgeMode::kShift};
};

HOLOSCAN_PARAM_SCHEMA(
    TDMSparseCompensationParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.TDMSparseCompensationParams", 1),
    holoscan::field<&TDMSparseCompensationParams::waveform>("waveform")
        .description("TDM PRI and carrier frequency. [live]."),
    holoscan::field<&TDMSparseCompensationParams::tx_schedule>("tx_schedule")
        .description("Static uniform TDM firing schedule."),
    holoscan::field<&TDMSparseCompensationParams::tx_positions>("tx_positions")
        .description("TX positions (m), row-major (num_tx, 3). Static."),
    holoscan::field<&TDMSparseCompensationParams::rx_positions>("rx_positions")
        .description("RX positions (m), row-major (num_rx, 3). Static."),
    holoscan::field<&TDMSparseCompensationParams::overlap_policy>(
        "overlap_policy")
        .description("Virtual overlap policy: dedup, average, or keep_all."),
    holoscan::field<&TDMSparseCompensationParams::snapshot_window>(
        "snapshot_window")
        .description("RD neighborhood half-size (doppler, range). [live]."),
    holoscan::field<&TDMSparseCompensationParams::doppler_edge>(
        "doppler_edge")
        .description("Doppler window edge policy."),
    holoscan::field<&TDMSparseCompensationParams::range_edge>("range_edge")
        .description("Range window edge policy."));

template <typename T>
class TDMSparseCompensationOp
    : public holoscan::Operator<TDMSparseCompensationParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "TDMSparseCompensationOp supports float and double only");

 public:
  explicit TDMSparseCompensationOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  Input<Tensor> detection_idx{"detection_idx"};
  Input<wire::FMCWWaveformT> waveform_in{"waveform"};
  Input<TDMSparseWindowOverride> snapshot_window_in{"snapshot_window"};

  Output<Tensor> snapshot{"snapshot"};
  Output<Tensor> is_valid{"is_valid"};
  Output<WireArrayGeometry> array_geometry{"array_geometry"};

 private:
  struct CachedCorrection {
    std::shared_ptr<Tensor> device_table;
  };

  expected<void, Error> ensure_virtual_tables(
      const detail::VirtualAssemblyPlan& plan,
      const std::vector<double>& tx_positions,
      const std::vector<double>& rx_positions, cudaStream_t stream);

  ConfigCache<CachedCorrection> correction_cache_{};
  Tensor retained_rd_maps_;
  Tensor retained_detection_idx_;
  std::shared_ptr<Tensor> src_channel_;
  std::shared_ptr<Tensor> member_offsets_;
  std::shared_ptr<Tensor> member_indices_;
  std::shared_ptr<Tensor> error_flag_;
  std::vector<double> cached_tx_positions_;
  std::vector<double> cached_rx_positions_;
  OverlapPolicy cached_policy_{OverlapPolicy::kDedup};
  int64_t cached_num_tx_{-1};
  int64_t cached_num_rx_{-1};
  std::size_t src_capacity_{0};
  std::size_t offsets_capacity_{0};
  std::size_t members_capacity_{0};
  MaxTensorExtents extents_;
  std::size_t snapshot_max_byte_span_{0};
  std::size_t is_valid_max_byte_span_{0};
  std::size_t geometry_max_byte_span_{0};
};

extern template class TDMSparseCompensationOp<float>;
extern template class TDMSparseCompensationOp<double>;

}  // namespace holoscan::radar
