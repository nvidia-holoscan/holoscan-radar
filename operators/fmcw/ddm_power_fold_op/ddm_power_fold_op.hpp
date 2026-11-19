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

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/config_cache.hpp"

namespace holoscan::radar {

struct DDMPowerFoldParams {
  DDMCodeDescriptor ddm_code{};
  std::uint64_t cache_budget_bytes{67108864ULL};
};

HOLOSCAN_PARAM_SCHEMA(DDMPowerFoldParams,
                      holoscan::param_schema("com.nvidia.holoscan.radar.DDMPowerFoldParams", 2),
                      holoscan::field<&DDMPowerFoldParams::cache_budget_bytes>("cache_budget_bytes")
                          .description("Retained operator cache bytes; 0 disables retention. "
                                       "Oversized artifacts run uncached with a warning."),
                      holoscan::field<&DDMPowerFoldParams::ddm_code>("ddm_code")
                          .description("Required static linear exact-bin DDM layout."));

template <typename T>
class DDMPowerFoldOp : public holoscan::Operator<DDMPowerFoldParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

 public:
  explicit DDMPowerFoldOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  Output<Tensor> power_map{"power_map"};

 private:
  std::shared_ptr<ConfigCacheBudget> cache_budget_{std::make_shared<ConfigCacheBudget>()};

  struct CachedTable {
    std::shared_ptr<Tensor> device_table;
    std::size_t cache_bytes() const {
      return cache_size_add(0U, cache_payload_bytes(device_table));
    }
  };

  Tensor retained_input_;
  ConfigCache<CachedTable> table_cache_{};
  MaxTensorExtents extents_;
  std::size_t max_byte_span_{0U};
};

extern template class DDMPowerFoldOp<float>;
extern template class DDMPowerFoldOp<double>;

}  // namespace holoscan::radar
