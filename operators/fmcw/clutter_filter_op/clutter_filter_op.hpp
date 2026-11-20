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

#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

namespace holoscan::radar {

namespace detail {
template <typename T>
struct ClutterStageState;  // kernel-side caches; defined in clutter_stage.hpp
}  // namespace detail

// Static-clutter / zero-Doppler suppression along slow-time; sits between
// the range and Doppler stages. This header is the contract reference for
// the operator's ports, shapes, dtypes, and parameters; module-wide
// conventions live in operators/common/README.md.
//
// In:  range_profiles — complex64/complex128,
//                       (num_chirps, num_range_bins) or with named leading
//                       N_rx / (M_tx, N_rx) axes (no opaque leading batch).
// Out: range_profiles_out — complex64/complex128,
//                       same leading + (num_chirps_out, num_range_bins), where
//                       num_chirps_out = num_chirps (mean_subtraction),
//                       num_chirps - 1 (single_delay), or
//                       num_chirps - 2 (double_delay).
//                        unique in/out port names required.
//
// mean_subtraction uses a uniform mean when doppler_window is none and a
// window-weighted mean otherwise. The chirp count is unchanged. Other methods
// ignore doppler_window.
//
// Precision is fixed at compile time by the template instantiation.
// Fat output pool sizing requires MaxTensorExtents at graph.op() (ctor).

struct ClutterFilterParams {
  ClutterFilterMethod method{ClutterFilterMethod::kMeanSubtraction};
  WindowSpec doppler_window{};
  std::uint64_t cache_budget_bytes{67108864ULL};
};

HOLOSCAN_PARAM_SCHEMA(
    ClutterFilterParams, holoscan::param_schema("com.nvidia.holoscan.radar.ClutterFilterParams", 3),
    holoscan::field<&ClutterFilterParams::cache_budget_bytes>("cache_budget_bytes")
        .description("Retained operator cache bytes; 0 disables retention. "
                     "Oversized artifacts run uncached with a warning."),
    holoscan::field<&ClutterFilterParams::method>("method").description(
        "Slow-time clutter suppression {none, mean_subtraction, single_delay, "
        "double_delay}: none is pass-through; mean_subtraction nulls "
        "zero-Doppler with no chirp loss; the delay cancellers are 1st/2nd-"
        "order MTI high-pass filters and drop 1-2 chirps."),
    holoscan::field<&ClutterFilterParams::doppler_window>("doppler_window")
        .description("WindowSpec weighting the slow-time mean for mean_subtraction. "
                     "Default none uses a uniform mean. Ignored by none and the delay "
                     "cancellers."));

static_assert(holoscan::param_schema_availability<ClutterFilterParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in ClutterFilterParams's "
              "namespace");

template <typename T>
class ClutterFilterOp : public holoscan::Operator<ClutterFilterParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "ClutterFilterOp supports float and double precision");

 public:
  // Required: sizes the range_profiles_out Tensor pool at setup().
  explicit ClutterFilterOp(const MaxTensorExtents& extents);
  ~ClutterFilterOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> range_profiles{"range_profiles"};
  // HSDK 5 requires unique port names within an operator (inputs and
  // outputs share one namespace). the operator README's logical output channel
  // remains "range_profiles"; the physical port name is range_profiles_out.
  //  revisit naming once the full port is settled.
  Output<Tensor> range_profiles_out{"range_profiles_out"};

  // [live] Exact override (the operator README).
  Input<ClutterFilterMethod> method{"method"};
  Input<wire::WindowSpecT> doppler_window_in{"doppler_window"};

 private:
  std::shared_ptr<ConfigCacheBudget> cache_budget_{std::make_shared<ConfigCacheBudget>()};

  // Keep the previous activation's input alive across ticks so async device
  // reads cannot outlive the buffer.
  Tensor retained_input_;

  // Kernel-side window coefficient cache, allocated in the constructor.
  std::shared_ptr<detail::ClutterStageState<T>> state_;

  MaxTensorExtents extents_;
  std::uint32_t max_rank_{2U};
  std::size_t max_byte_span_{0U};
};

extern template class ClutterFilterOp<float>;
extern template class ClutterFilterOp<double>;

}  // namespace holoscan::radar
