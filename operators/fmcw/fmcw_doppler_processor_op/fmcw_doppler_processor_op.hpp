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
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

namespace holoscan::radar {

namespace detail {
template <typename T>
struct DopplerStageState;  // kernel-side caches; defined in doppler_stage.hpp
}  // namespace detail

// FMCW Doppler compression: windowed FFT along slow-time. This header is
// the contract reference for the operator's ports, shapes, dtypes, and
// parameters; module-wide conventions live in operators/common/README.md.
//
// In:  range_profiles — complex64/complex128,
//                       (num_chirps, num_range_bins) or with named leading
//                       N_rx / (M_tx, N_rx).
// Out: rd_maps        — complex64/complex128,
//                       same leading + (num_doppler, num_range_bins);
//                       zero-Doppler centered by default.
// Out: velocity_bins  — optional, float32/float64, (num_doppler,);
//                       emitted only when a waveform descriptor is set.
//
// Precision is fixed at compile time by the template instantiation
// (T = float → complex64/float32, T = double → complex128/float64).
// Fat outs require MaxTensorExtents at graph.op() (ctor).

struct FMCWDopplerProcessorParams {
  WindowSpec doppler_window{};
  bool center_zero_doppler{true};
  // Default-constructed = none/absent; is_valid() → false.
  FMCWWaveformDescriptor waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  // 0 = exact (no zero-padding); ≥ num_chirps when non-zero.
  int64_t nfft{0};
};

HOLOSCAN_PARAM_SCHEMA(
    FMCWDopplerProcessorParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.FMCWDopplerProcessorParams", 1),
    holoscan::field<&FMCWDopplerProcessorParams::doppler_window>("doppler_window")
        .description("WindowSpec applied along the chirp (slow-time) axis before "
                     "the Doppler FFT (coherent-gain normalized). Default: none "
                     "(rectangular)."),
    holoscan::field<&FMCWDopplerProcessorParams::center_zero_doppler>("center_zero_doppler")
        .description("fftshift the Doppler axis so zero-Doppler is centered "
                     "(conventional RD layout); else DC at bin 0."),
    holoscan::field<&FMCWDopplerProcessorParams::waveform>("waveform")
        .description("Source for the velocity_bins axis (carrier -> wavelength, "
                     "chirp repetition interval -> PRF). Required only when "
                     "velocity_bins is used. Default: empty (no velocity axis)."),
    holoscan::field<&FMCWDopplerProcessorParams::fft_norm>("fft_norm")
        .description("FFT normalization convention {backward, ortho, forward}; "
                     "backward keeps coherent-integration gain visible."),
    holoscan::field<&FMCWDopplerProcessorParams::nfft>("nfft").description(
        "Slow-time FFT length >= num_chirps; larger interpolates "
        "the velocity grid over the same span. 0 = num_chirps."));

static_assert(holoscan::param_schema_availability<FMCWDopplerProcessorParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in "
              "FMCWDopplerProcessorParams's namespace");

template <typename T>
class FMCWDopplerProcessorOp : public holoscan::Operator<FMCWDopplerProcessorParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FMCWDopplerProcessorOp supports float and double precision");

 public:
  explicit FMCWDopplerProcessorOp(const MaxTensorExtents& extents);
  ~FMCWDopplerProcessorOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> range_profiles{"range_profiles"};
  Output<Tensor> rd_maps{"rd_maps"};
  Output<Tensor> velocity_bins{"velocity_bins"};

  // [live] Exact overrides (OPERATOR_API.md).
  Input<wire::WindowSpecT> doppler_window_in{"doppler_window"};
  Input<bool> center_zero_doppler_in{"center_zero_doppler"};
  Input<wire::FMCWWaveformT> waveform_in{"waveform"};
  Input<FftNorm> fft_norm_in{"fft_norm"};
  Input<int64_t> nfft_in{"nfft"};

 private:
  Tensor retained_input_;

  // velocity_bins is optional/latched (OPERATOR_API.md).
  LatchedOutput velocity_bins_out_;

  // Kernel-side caches (window coefficients, seen FFT lengths); allocated
  // in the ctor.
  std::shared_ptr<detail::DopplerStageState<T>> state_;

  MaxTensorExtents extents_;
  std::uint32_t max_rank_{2U};
  std::size_t max_byte_span_{0U};
  std::size_t velocity_bins_byte_span_{0U};
};

// Explicit instantiations live in fmcw_doppler_processor_op.cu.
extern template class FMCWDopplerProcessorOp<float>;
extern template class FMCWDopplerProcessorOp<double>;

}  // namespace holoscan::radar
