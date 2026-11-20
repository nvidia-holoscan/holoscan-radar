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
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/lfm_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

namespace holoscan::radar {
namespace detail {
template <typename T>
struct RangeCompressionState;
}  // namespace detail

struct SARRangeCompressionParams {
  WindowSpec range_window{};
  LFMWaveformDescriptor waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  int64_t nfft{0};  // 0 = observed num_samples.
  double propagation_speed{kDefaultPropagationSpeed};
  std::uint64_t cache_budget_bytes{2147483648ULL};
};

HOLOSCAN_PARAM_SCHEMA(
    SARRangeCompressionParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.SARRangeCompressionParams", 4),
    holoscan::field<&SARRangeCompressionParams::cache_budget_bytes>("cache_budget_bytes")
        .description("Retained operator cache bytes; 0 disables retention. "
                     "Oversized artifacts run uncached with a warning."),
    holoscan::field<&SARRangeCompressionParams::range_window>("range_window"),
    holoscan::field<&SARRangeCompressionParams::waveform>("waveform"),
    holoscan::field<&SARRangeCompressionParams::fft_norm>("fft_norm"),
    holoscan::field<&SARRangeCompressionParams::nfft>("nfft"),
    holoscan::field<&SARRangeCompressionParams::propagation_speed>("propagation_speed"));

template <typename T>
class SARRangeCompressionOp : public holoscan::Operator<SARRangeCompressionParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "SARRangeCompressionOp supports float and double precision");

 public:
  explicit SARRangeCompressionOp(const MaxTensorExtents& extents);
  ~SARRangeCompressionOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> phase_history{"phase_history"};
  Output<Tensor> range_profiles{"range_profiles"};
  Output<Tensor> range_bins{"range_bins"};
  Output<WireWindowSpec> range_window_out{"range_window_out"};

  Input<WireWindowSpec> range_window_in{"range_window"};
  Input<WireLFMWaveform> waveform_in{"waveform"};
  Input<FftNorm> fft_norm_in{"fft_norm"};
  Input<int64_t> nfft_in{"nfft"};
  Input<double> propagation_speed_in{"propagation_speed"};

 private:
  std::shared_ptr<ConfigCacheBudget> cache_budget_{std::make_shared<ConfigCacheBudget>()};

  std::shared_ptr<detail::RangeCompressionState<T>> state_;
  Tensor retained_input_;
  LatchedOutput range_bins_out_;
  LatchedOutput range_window_out_;
  MaxTensorExtents extents_;
  std::size_t max_byte_span_{0U};
  std::size_t bins_byte_span_{0U};
  std::size_t window_byte_span_{0U};
};

extern template class SARRangeCompressionOp<float>;
extern template class SARRangeCompressionOp<double>;

}  // namespace holoscan::radar
