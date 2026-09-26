// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

namespace holoscan::radar::perf {
namespace hr = holoscan::radar;

// Common pool declarations for typed angle-source ports. Each adapter keeps
// the port names and wires the matching operator input explicitly.
inline constexpr std::size_t kHostPayloadSpan = 4096U * sizeof(double);

template <typename Output>
void setup_device_input(holoscan::OperatorSpec& spec, Output& output, std::size_t span) {
  constexpr holoscan::TensorRepresentation representation{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  spec.output(output).produces_tensor(hr::pooled_tensor_output(representation, 8U, span));
}

template <typename Output>
void setup_host_payload(holoscan::OperatorSpec& spec, Output& output, std::size_t rank,
                        std::size_t slots, DLDataType dtype = DLDataType{kDLFloat, 64U, 1U},
                        std::size_t span = kHostPayloadSpan) {
  const holoscan::TensorRepresentation representation{
      .memory_kind = holoscan::MemoryKind::kHost,
      .dtype = dtype,
      .rank = rank,
  };
  spec.output(output).max_emits_per_compute(1U).produces_tensor(
      hr::pooled_tensor_output(representation, slots, span));
}

template <typename Value>
void setup_scalar_payload(holoscan::OperatorSpec& spec, holoscan::Output<Value>& output) {
  spec.output(output).max_emits_per_compute(1U);
}

template <typename Value>
holoscan::expected<void, holoscan::Error> emit_scalar_payload(holoscan::Output<Value>& output,
                                                              Value value,
                                                              holoscan::TimePoint time) {
  return output.emit(std::move(value), holoscan::EmitOptions{.capture_time = time});
}

inline std::vector<double> linspace(double start, double stop, int64_t n) {
  std::vector<double> out(static_cast<size_t>(n));
  if (n == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(n - 1);
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = start + step * static_cast<double>(i);
  }
  return out;
}

inline hr::ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing,
                                           double wavelength) {
  hr::ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = hr::ArrayGeometryType::kUla;
  geom.element_positions.reserve(static_cast<size_t>(num_elements * 3));
  for (int64_t m = 0; m < num_elements; ++m) {
    geom.element_positions.push_back(0.0);
    geom.element_positions.push_back(static_cast<double>(m) * spacing);
    geom.element_positions.push_back(0.0);
  }
  return geom;
}

template <typename Element, typename Fill>
holoscan::expected<void, holoscan::Error> emit_host_vector_generated(
    holoscan::Output<holoscan::Tensor>& output, int64_t count, Fill&& fill,
    holoscan::TimePoint capture_time) {
  static_assert(std::is_same_v<Element, double> || std::is_same_v<Element, int64_t>);
  const std::array<int64_t, 1> shape{count};
  constexpr DLDataType dtype =
      std::is_same_v<Element, double> ? DLDataType{kDLFloat, 64U, 1U} : DLDataType{kDLInt, 64U, 1U};
  auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan)
    return holoscan::make_unexpected(std::move(loan).error());
  auto written = loan->write_host_and_commit(
      [fill = std::forward<Fill>(fill)](const holoscan::TensorOutputWriteGuard& writer) {
        fill(writer.data_as<Element>());
      });
  if (!written)
    return holoscan::make_unexpected(std::move(written).error());
  return output.emit(std::move(*loan), holoscan::EmitOptions{.capture_time = capture_time});
}

template <typename Element>
holoscan::expected<void, holoscan::Error> emit_host_vector(
    holoscan::Output<holoscan::Tensor>& output, std::span<const Element> values,
    holoscan::TimePoint capture_time) {
  return emit_host_vector_generated<Element>(
      output, static_cast<int64_t>(values.size()),
      [values](Element* data) {
        std::copy(values.begin(), values.end(), data);
      },
      capture_time);
}

template <typename Element>
holoscan::expected<void, holoscan::Error> emit_host_vector(
    holoscan::Output<holoscan::Tensor>& output, const std::vector<Element>& values,
    holoscan::TimePoint capture_time) {
  return emit_host_vector(output, std::span<const Element>{values}, capture_time);
}

template <typename Wire>
holoscan::expected<void, holoscan::Error> emit_wire_doubles(holoscan::Output<Wire>& output,
                                                            Wire wire,
                                                            std::span<const double> values,
                                                            std::span<const int64_t> shape,
                                                            holoscan::TimePoint capture_time) {
  constexpr DLDataType dtype{kDLFloat, 64U, 1U};
  auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written =
      loan->write_host_and_commit([values](const holoscan::TensorOutputWriteGuard& writer) {
        std::copy(values.begin(), values.end(), writer.data_as<double>());
      });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return output.emit_tensor(std::move(*loan), std::move(wire),
                            holoscan::EmitOptions{.capture_time = capture_time});
}

inline holoscan::expected<void, holoscan::Error> emit_geometry_payload(
    holoscan::Output<hr::WireArrayGeometry>& output, const hr::ArrayGeometry& geometry,
    holoscan::TimePoint time) {
  const auto packed = hr::pack_array_geometry(geometry);
  const std::array<int64_t, 2> shape{geometry.num_elements() + 1, 3};
  return emit_wire_doubles(output, hr::to_wire(geometry), packed, shape, time);
}

inline holoscan::expected<void, holoscan::Error> emit_window_payload(
    holoscan::Output<hr::WireWindowSpec>& output, const hr::WindowSpec& window,
    holoscan::TimePoint time) {
  const std::array<int64_t, 1> shape{static_cast<int64_t>(window.coefficients.size())};
  return emit_wire_doubles(output, hr::to_wire(window), window.coefficients, shape, time);
}

inline hr::SpatialSmoothingOverride smoothing_wire(const std::vector<int64_t>& smoothing) {
  hr::SpatialSmoothingOverride wire;
  wire.count = static_cast<int64_t>(smoothing.size());
  if (!smoothing.empty()) {
    wire.first = smoothing[0];
  }
  if (smoothing.size() > 1U) {
    wire.second = smoothing[1];
  }
  return wire;
}

inline holoscan::expected<void, holoscan::Error> emit_smoothing_payload(
    holoscan::Output<hr::SpatialSmoothingOverride>& output, const std::vector<int64_t>& smoothing,
    holoscan::TimePoint time) {
  return emit_scalar_payload(output, smoothing_wire(smoothing), time);
}

}  // namespace holoscan::radar::perf
