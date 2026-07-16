// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// MIMO marshaling: flatten separated (tx, rx) RD maps into a virtual channel
// axis and emit the matching ArrayGeometry. Contract reference: ports, shapes,
// dtypes, parameters.
//
// In:  rd_maps — complex64/complex128, (..., M_tx, N_rx, num_doppler,
//               num_range_bins).
// Out: rd_maps — same dtype, (..., num_virtual, num_doppler, num_range_bins).
// Out: array_geometry — runtime virtual element_positions + wavelength.
template <typename T>
class VirtualArrayAssemblyOp : public holoscan::Operator {
 public:
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "VirtualArrayAssemblyOp supports float and double only");

  HOLOSCAN_OPERATOR_FORWARD_ARGS(VirtualArrayAssemblyOp)

  VirtualArrayAssemblyOp() = default;
  ~VirtualArrayAssemblyOp() override;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  void ensure_device_plan(const detail::VirtualAssemblyPlan& plan,
                          cudaStream_t stream);

  Parameter<std::vector<double>> tx_positions_;
  Parameter<std::vector<double>> rx_positions_;
  Parameter<double> wavelength_;
  Parameter<OverlapPolicy> overlap_policy_;

  double effective_wavelength_{0.0};

  std::shared_ptr<holoscan::Tensor> retained_input_;

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

}  // namespace holoscan::radar
