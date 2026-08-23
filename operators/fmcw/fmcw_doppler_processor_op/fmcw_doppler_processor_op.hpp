// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
//                       (..., num_chirps, num_range_bins).
// Out: rd_maps        — complex64/complex128,
//                       (..., num_doppler, num_range_bins);
//                       zero-Doppler centered by default.
// Out: velocity_bins  — optional, float32/float64, (num_doppler,);
//                       emitted only when a waveform descriptor is set.
//
// Precision is fixed at compile time by the template instantiation.

struct FMCWDopplerProcessorParams {
  WindowSpec doppler_window{};
  bool center_zero_doppler{true};
  FMCWWaveformDescriptor waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  int64_t nfft{0};
  bool expect_doppler_window{false};
  bool expect_center_zero_doppler{false};
  bool expect_waveform{false};
  bool expect_fft_norm{false};
  bool expect_nfft{false};
  bool sticky_doppler_window{true};
  bool sticky_center_zero_doppler{true};
  bool sticky_waveform{true};
  bool sticky_fft_norm{true};
  bool sticky_nfft{true};
};

HOLOSCAN_PARAM_SCHEMA(
    FMCWDopplerProcessorParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.FMCWDopplerProcessorParams", 1),
    holoscan::field<&FMCWDopplerProcessorParams::doppler_window>(
        "doppler_window")
        .description(
            "WindowSpec applied along the chirp (slow-time) axis before "
            "the Doppler FFT (coherent-gain normalized). Default: none "
            "(rectangular)."),
    holoscan::field<&FMCWDopplerProcessorParams::center_zero_doppler>(
        "center_zero_doppler")
        .description(
            "fftshift the Doppler axis so zero-Doppler is centered "
            "(conventional RD layout); else DC at bin 0."),
    holoscan::field<&FMCWDopplerProcessorParams::waveform>("waveform")
        .description(
            "Source for the velocity_bins axis (carrier -> wavelength, "
            "chirp repetition interval -> PRF). Required only when "
            "velocity_bins is used. Default: empty (no velocity axis)."),
    holoscan::field<&FMCWDopplerProcessorParams::fft_norm>("fft_norm")
        .description(
            "FFT normalization convention {backward, ortho, forward}; "
            "backward keeps coherent-integration gain visible."),
    holoscan::field<&FMCWDopplerProcessorParams::nfft>("nfft")
        .description(
            "Slow-time FFT length >= num_chirps; larger interpolates "
            "the velocity grid over the same span. 0 = num_chirps."),
    holoscan::field<&FMCWDopplerProcessorParams::expect_doppler_window>(
        "expect_doppler_window")
        .description(
            "Set when the application wires the doppler_window override."),
    holoscan::field<&FMCWDopplerProcessorParams::expect_center_zero_doppler>(
        "expect_center_zero_doppler")
        .description(
            "Set when the application wires the center_zero_doppler "
            "override."),
    holoscan::field<&FMCWDopplerProcessorParams::expect_waveform>(
        "expect_waveform")
        .description("Set when the application wires the waveform override."),
    holoscan::field<&FMCWDopplerProcessorParams::expect_fft_norm>(
        "expect_fft_norm")
        .description(
            "Set when the application wires the fft_norm override."),
    holoscan::field<&FMCWDopplerProcessorParams::expect_nfft>("expect_nfft")
        .description("Set when the application wires the nfft override."),
    holoscan::field<&FMCWDopplerProcessorParams::sticky_doppler_window>(
        "sticky_doppler_window")
        .description(
            "Keep the last Exact doppler_window match on a miss. False "
            "reverts to Params. Ignored unless expect_doppler_window."),
    holoscan::field<&FMCWDopplerProcessorParams::sticky_center_zero_doppler>(
        "sticky_center_zero_doppler")
        .description(
            "Keep the last Exact center_zero_doppler match on a miss. "
            "Ignored unless expect_center_zero_doppler."),
    holoscan::field<&FMCWDopplerProcessorParams::sticky_waveform>(
        "sticky_waveform")
        .description(
            "Keep the last Exact waveform match on a miss. False reverts "
            "to Params. Ignored unless expect_waveform."),
    holoscan::field<&FMCWDopplerProcessorParams::sticky_fft_norm>(
        "sticky_fft_norm")
        .description(
            "Keep the last Exact fft_norm match on a miss. Ignored "
            "unless expect_fft_norm."),
    holoscan::field<&FMCWDopplerProcessorParams::sticky_nfft>("sticky_nfft")
        .description(
            "Keep the last Exact nfft match on a miss. Ignored unless "
            "expect_nfft."));

static_assert(
    holoscan::param_schema_availability<FMCWDopplerProcessorParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in "
    "FMCWDopplerProcessorParams's namespace");


template <typename T>
class FMCWDopplerProcessorOp
    : public holoscan::Operator<FMCWDopplerProcessorParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FMCWDopplerProcessorOp supports float and double precision");

 public:
  explicit FMCWDopplerProcessorOp(
      const MaxTensorExtents& extents);
  ~FMCWDopplerProcessorOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> range_profiles{"range_profiles"};
  Output<Tensor> rd_maps{"rd_maps"};
  Output<Tensor> velocity_bins{"velocity_bins"};

  // [live] Exact overrides gated by expect_* (OPERATOR_API.md).
  Input<wire::WindowSpecT> doppler_window_in{"doppler_window"};
  Input<bool> center_zero_doppler_in{"center_zero_doppler"};
  Input<wire::FMCWWaveformT> waveform_in{"waveform"};
  Input<FftNorm> fft_norm_in{"fft_norm"};
  Input<int64_t> nfft_in{"nfft"};

 private:
  // Kernel-side caches (window coefficients, seen FFT lengths); allocated
  // on first compute().
  std::shared_ptr<detail::DopplerStageState<T>> state_;

  // Stream-ordered lifetime guard for the previous tick's input.
  Tensor retained_input_;

  // Sticky override state.
  std::optional<WindowSpec> sticky_doppler_window_;
  std::optional<bool> sticky_center_zero_doppler_;
  std::optional<FMCWWaveformDescriptor> sticky_waveform_;
  std::optional<FftNorm> sticky_fft_norm_;
  std::optional<int64_t> sticky_nfft_;

  MaxTensorExtents extents_;
  std::uint32_t max_rank_{2U};
  std::size_t max_byte_span_{0U};
  std::size_t velocity_bins_byte_span_{0U};
};

// Explicit instantiations live in fmcw_doppler_processor_op.cu.
extern template class FMCWDopplerProcessorOp<float>;
extern template class FMCWDopplerProcessorOp<double>;

}  // namespace holoscan::radar
