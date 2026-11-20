// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

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

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"

namespace holoscan::radar {

// TDM front-end marshaling: split interleaved slow-time into an explicit TX
// axis. Contract reference: ports, shapes, dtypes, parameters.
//
// In:  adc_cube — complex64/complex128 or float32/float64 (real IF),
//               (N_rx, M·L, samples_per_chirp). No opaque leading batch.
// Out: adc_cube_out — same dtype,
//               (M_tx, N_rx, L, samples_per_chirp).
//                unique in/out port names required.
//
// Precision is fixed at compile time by the template instantiation.
// Fat output pool sizing requires MaxTensorExtents at graph.op() (ctor).

struct MIMODeinterleaveParams {
  TXSchedule tx_schedule{};
};

HOLOSCAN_PARAM_SCHEMA(
    MIMODeinterleaveParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.MIMODeinterleaveParams", 1),
    holoscan::field<&MIMODeinterleaveParams::tx_schedule>("tx_schedule")
        .description("TXSchedule describing how the interleaved slow-time axis "
                     "splits into (M_tx, L). Required; static for the instance."));

static_assert(holoscan::param_schema_availability<MIMODeinterleaveParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in MIMODeinterleaveParams's "
              "namespace");

template <typename T>
class MIMODeinterleaveOp : public holoscan::Operator<MIMODeinterleaveParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "MIMODeinterleaveOp supports float and double precision");

 public:
  // Required: sizes the adc_cube_out Tensor pool at setup().
  explicit MIMODeinterleaveOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> adc_cube{"adc_cube"};
  // HSDK 5 requires unique port names within an operator (inputs and
  // outputs share one namespace). the operator README's logical output channel
  // remains "adc_cube"; the physical port name is adc_cube_out.
  //  revisit naming once the full port is settled.
  Output<Tensor> adc_cube_out{"adc_cube_out"};

 private:
  expected<void, Error> ensure_device_gather_map(int64_t slow_time, const TXSchedule& schedule,
                                                 cudaStream_t stream);

  // Keep the previous activation's input alive across ticks so async
  // device reads cannot outlive the buffer.
  Tensor retained_input_;

  // Gather-map cache: ctor-sized device table, rebuilt on a new
  // (slow_time, schedule) pair. Not ConfigCache — there are no lives.
  int64_t cached_slow_time_{-1};
  TXSchedule cached_schedule_{};
  std::shared_ptr<Tensor> gather_map_;
  std::size_t gather_map_capacity_{0};

  MaxTensorExtents extents_;
  std::uint32_t max_rank_{4U};
  std::size_t max_byte_span_{0U};
};

extern template class MIMODeinterleaveOp<float>;
extern template class MIMODeinterleaveOp<double>;

}  // namespace holoscan::radar
