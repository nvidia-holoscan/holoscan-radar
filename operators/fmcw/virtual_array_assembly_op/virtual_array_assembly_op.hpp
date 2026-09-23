// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

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

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

// MIMO marshaling: flatten separated (tx, rx) RD maps into a virtual channel
// axis and emit the matching ArrayGeometry in the shared wire schema.
// Contract reference: ports, shapes, dtypes, parameters.
//
// In:  rd_maps — complex64/complex128, (M_tx, N_rx, num_doppler,
//               num_range_bins). No opaque leading batch.
// Out: rd_maps_out — same dtype, (num_virtual, num_doppler,
//               num_range_bins).
//                unique in/out port names required.
// Out: array_geometry — WireArrayGeometry carrying the virtual element
//               positions, wavelength, and optional geometry type.
//
// Precision is fixed at compile time by the template instantiation.
// Fat output pool sizing requires MaxTensorExtents at graph.op() (ctor).

struct VirtualArrayAssemblyParams {
  std::vector<double> tx_positions{};
  std::vector<double> rx_positions{};
  double wavelength{0.0};
  OverlapPolicy overlap_policy{OverlapPolicy::kDedup};
};

HOLOSCAN_PARAM_SCHEMA(
    VirtualArrayAssemblyParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.VirtualArrayAssemblyParams", 1),
    holoscan::field<&VirtualArrayAssemblyParams::tx_positions>("tx_positions")
        .description(
            "TX element positions (m), row-major (num_tx, 3). Static."),
    holoscan::field<&VirtualArrayAssemblyParams::rx_positions>("rx_positions")
        .description(
            "RX element positions (m), row-major (num_rx, 3). Static."),
    holoscan::field<&VirtualArrayAssemblyParams::wavelength>("wavelength")
        .description(
            "Operating wavelength (m) for the emitted array_geometry. "
            "Seeds the [live] Exact override."),
    holoscan::field<&VirtualArrayAssemblyParams::overlap_policy>(
        "overlap_policy")
        .description(
            "How coincident virtual elements are handled: dedup, average, "
            "or keep_all. Static."));

static_assert(
    holoscan::param_schema_availability<VirtualArrayAssemblyParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in VirtualArrayAssemblyParams's "
    "namespace");

template <typename T>
class VirtualArrayAssemblyOp
    : public holoscan::Operator<VirtualArrayAssemblyParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "VirtualArrayAssemblyOp supports float and double precision");

 public:
  // Required: sizes the rd_maps_out and array_geometry descriptor-Tensor
  // pools at setup(), and the private index tables.
  explicit VirtualArrayAssemblyOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  // HSDK 5 requires unique port names within an operator (inputs and
  // outputs share one namespace). OPERATOR_API.md's logical output channel
  // remains "rd_maps"; the physical port name is rd_maps_out.
  //  revisit naming once the full port is settled.
  Output<Tensor> rd_maps_out{"rd_maps_out"};
  Output<WireArrayGeometry> array_geometry{"array_geometry"};

  // [live] Exact override (OPERATOR_API.md).
  Input<double> wavelength{"wavelength"};

 private:
  expected<void, Error> ensure_device_index_tables(
      const detail::VirtualAssemblyStageConfig& stage,
      const detail::VirtualAssemblyPlan& plan, cudaStream_t stream);

  // Keep the previous activation's input alive across ticks so async
  // device reads cannot outlive the buffer.
  Tensor retained_input_;

  // Index-table cache: ctor-sized device tables, rebuilt on a new
  // (tx_positions, rx_positions, overlap_policy, M_tx, N_rx) key.
  // Not ConfigCache — wavelength is live but does not rebuild the gather.
  int64_t cached_num_tx_{-1};
  int64_t cached_num_rx_{-1};
  OverlapPolicy cached_policy_{OverlapPolicy::kDedup};
  std::vector<double> cached_tx_positions_;
  std::vector<double> cached_rx_positions_;
  std::shared_ptr<Tensor> src_channel_;
  std::size_t src_channel_capacity_{0};
  std::shared_ptr<Tensor> member_offsets_;
  std::size_t member_offsets_capacity_{0};
  std::shared_ptr<Tensor> member_indices_;
  std::size_t member_indices_capacity_{0};

  MaxTensorExtents extents_;
  std::uint32_t max_rank_{3U};
  std::size_t max_byte_span_{0U};
  std::size_t geom_byte_span_{0U};
};

extern template class VirtualArrayAssemblyOp<float>;
extern template class VirtualArrayAssemblyOp<double>;

}  // namespace holoscan::radar
