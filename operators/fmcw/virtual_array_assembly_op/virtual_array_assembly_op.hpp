// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
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
#include <holoscan/core/tensor_output_loan.hpp>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"
#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// MIMO marshaling: flatten separated (tx, rx) RD maps into a virtual channel
// axis and emit the matching ArrayGeometry as a packed float64 Tensor.
// Contract reference: ports, shapes, dtypes, parameters.
//
// In:  rd_maps — complex64/complex128, (..., M_tx, N_rx, num_doppler,
//               num_range_bins).
// Out: rd_maps_out — same dtype, (..., num_virtual, num_doppler,
//               num_range_bins).
//               [EA HSDK5 ONLY] unique in/out port names required.
// Out: array_geometry — float64 Tensor, shape (num_virtual + 1, 3): packed
//               ArrayGeometry (see array_geometry_tensor.hpp).
//               [EA HSDK5 ONLY] Output<ArrayGeometry> is not a
//               PublishedPayload; serialized Tensor interim — HSDK5_PORT §10.
//
// Precision is fixed at compile time by the template instantiation.

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
            "Seeds the sticky [live] override."),
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

inline constexpr std::uint32_t kVirtualArrayAssemblyMaxRank = 8U;
// 256 MiB — rd_maps_out data plane
inline constexpr std::size_t kVirtualArrayAssemblyMaxByteSpan = 256ULL << 20;
// Packed ArrayGeometry is (num_virtual + 1, 3) float64 — tiny vs the cube.
inline constexpr std::size_t kVirtualArrayAssemblyGeomMaxByteSpan = 64ULL << 10;

template <typename T>
class VirtualArrayAssemblyOp
    : public holoscan::Operator<VirtualArrayAssemblyParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "VirtualArrayAssemblyOp supports float and double precision");

 public:
  VirtualArrayAssemblyOp() = default;
  ~VirtualArrayAssemblyOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  // [EA HSDK5 ONLY] unique in/out port names — see OPERATOR_API.md /
  // HSDK5_PORT §2.
  Output<Tensor> rd_maps_out{"rd_maps_out"};
  // Packed ArrayGeometry (float64 Tensor) — HSDK5_PORT §10.
  Output<Tensor> array_geometry{"array_geometry"};

  // [live] sticky scalar — HSDK5_PORT §1. Unconnected is legal; compute()
  // seeds from Params.
  Input<double> wavelength{"wavelength"};

 private:
  void ensure_device_plan(const detail::VirtualAssemblyPlan& plan,
                          cudaStream_t stream);

  Tensor retained_input_;
  std::optional<double> sticky_wavelength_;

  std::vector<double> cached_tx_positions_;
  std::vector<double> cached_rx_positions_;
  OverlapPolicy cached_policy_{OverlapPolicy::kDedup};
  int64_t cached_num_tx_{-1};
  int64_t cached_num_rx_{-1};
  detail::VirtualAssemblyPlan cached_plan_{};

  int64_t* d_src_channel_{nullptr};
  size_t d_src_capacity_{0};
  int64_t* d_member_offsets_{nullptr};
  size_t d_member_offsets_capacity_{0};
  int64_t* d_member_indices_{nullptr};
  size_t d_member_indices_capacity_{0};
};

extern template class VirtualArrayAssemblyOp<float>;
extern template class VirtualArrayAssemblyOp<double>;

}  // namespace holoscan::radar
