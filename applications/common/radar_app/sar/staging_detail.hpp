// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"

namespace holoscan::radar::app::sar {
namespace detail {
inline expected<void, Error> emit_geometry(Output<WireSampleGeometry>& output,
                                           ExecutionContext& context,
                                           const SampleGeometry& geometry, TimePoint capture_time,
                                           std::string_view what = "sample_geometry") {
  return emit_required_schema_tensor(
      output, context, what, false, to_wire(geometry),
      [&]() -> expected<TensorOutputLoan, Error> {
        auto loan = output.allocate_tensor(TensorLoanRequest{
            .shape = std::vector<int64_t>{11}, .dtype = DLDataType{kDLFloat, 64U, 1U}});
        if (!loan)
          return make_unexpected(std::move(loan).error());
        const auto packed = pack_sample_geometry(geometry);
        auto written = loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
          std::copy(packed.begin(), packed.end(), writer.data_as<double>());
        });
        if (!written)
          return make_unexpected(std::move(written).error());
        return std::move(*loan);
      },
      EmitOptions{.capture_time = capture_time});
}

inline constexpr TensorRepresentation kDevice{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = std::nullopt,
    .rank = std::nullopt};  // NOLINT(whitespace/indent_namespace)
inline constexpr TensorRepresentation kHostGeometry{
    .memory_kind = MemoryKind::kHost,
    .dtype = DLDataType{kDLFloat, 64U, 1U},
    .rank = 1U};  // NOLINT(whitespace/indent_namespace)
}  // namespace detail

}  // namespace holoscan::radar::app::sar
