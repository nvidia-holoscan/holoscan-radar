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

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/dsp/virtual_array.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

struct DDMSparseWindowOverride {
  int64_t doppler{0};
  int64_t range{0};
};

static_assert(sizeof(DDMSparseWindowOverride) == 16U);
static_assert(std::is_standard_layout_v<DDMSparseWindowOverride>);
static_assert(std::is_trivially_copyable_v<DDMSparseWindowOverride>);

}  // namespace holoscan::radar

namespace holoscan {

template <>
struct ChannelTypeIdentity<radar::DDMSparseWindowOverride> {
  static constexpr std::string_view name =
      "com.nvidia.holoscan.radar.DDMSparseWindowOverride";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<radar::DDMSparseWindowOverride> {
  static constexpr auto fields = pod_fields(
      pod_field("doppler", &radar::DDMSparseWindowOverride::doppler,
                offsetof(radar::DDMSparseWindowOverride, doppler)),
      pod_field("range", &radar::DDMSparseWindowOverride::range,
                offsetof(radar::DDMSparseWindowOverride, range)));
};

}  // namespace holoscan

namespace holoscan::radar {

struct DDMSparseDemultiplexParams {
  DDMCodeDescriptor ddm_code{};
  std::vector<double> tx_positions{};
  std::vector<double> rx_positions{};
  OverlapPolicy overlap_policy{OverlapPolicy::kDedup};
  double wavelength{0.0};
  std::vector<int64_t> snapshot_window{0, 0};
  SnapshotEdgeMode doppler_edge{SnapshotEdgeMode::kWrap};
  SnapshotEdgeMode range_edge{SnapshotEdgeMode::kShift};
};

HOLOSCAN_PARAM_SCHEMA(
    DDMSparseDemultiplexParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.DDMSparseDemultiplexParams", 1),
    holoscan::field<&DDMSparseDemultiplexParams::ddm_code>("ddm_code"),
    holoscan::field<&DDMSparseDemultiplexParams::tx_positions>("tx_positions"),
    holoscan::field<&DDMSparseDemultiplexParams::rx_positions>("rx_positions"),
    holoscan::field<&DDMSparseDemultiplexParams::overlap_policy>(
        "overlap_policy"),
    holoscan::field<&DDMSparseDemultiplexParams::wavelength>("wavelength")
        .description("Operating wavelength in meters. [live]."),
    holoscan::field<&DDMSparseDemultiplexParams::snapshot_window>(
        "snapshot_window")
        .description("Folded-RD neighborhood half-size. [live]."),
    holoscan::field<&DDMSparseDemultiplexParams::doppler_edge>("doppler_edge"),
    holoscan::field<&DDMSparseDemultiplexParams::range_edge>("range_edge"));

template <typename T>
class DDMSparseDemultiplexOp
    : public holoscan::Operator<DDMSparseDemultiplexParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

 public:
  explicit DDMSparseDemultiplexOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  Input<Tensor> detection_idx{"detection_idx"};
  Input<Tensor> fold_idx{"fold_idx"};
  Input<double> wavelength_in{"wavelength"};
  Input<DDMSparseWindowOverride> snapshot_window_in{"snapshot_window"};
  Output<Tensor> snapshot{"snapshot"};
  Output<Tensor> is_valid{"is_valid"};
  Output<WireArrayGeometry> array_geometry{"array_geometry"};

 private:
  struct CachedDdmTables {
    std::shared_ptr<Tensor> demux;
    std::shared_ptr<Tensor> tx_by_band;
    std::shared_ptr<Tensor> initial_phase;
  };

  expected<void, Error> ensure_virtual_tables(
      const detail::VirtualAssemblyPlan& plan,
      const std::vector<double>& tx_positions,
      const std::vector<double>& rx_positions, cudaStream_t stream);

  ConfigCache<CachedDdmTables> ddm_cache_{};
  Tensor retained_rd_maps_;
  Tensor retained_detection_idx_;
  Tensor retained_fold_idx_;
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

extern template class DDMSparseDemultiplexOp<float>;
extern template class DDMSparseDemultiplexOp<double>;

}  // namespace holoscan::radar
