// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <flatbuffers/reflection_generated.h>
#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>

#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

namespace holoscan::radar {
namespace {

static_assert(wire_bridge_detail::aggregate_field_count<WindowSpec>() == 7);
static_assert(wire_bridge_detail::aggregate_field_count<FMCWWaveformSpec>() == 7);
static_assert(wire_bridge_detail::aggregate_field_count<ArrayGeometry>() == 3);

TEST(WireArrayGeometry, PackedLayoutRoundTrips) {
  // clang-format off
  const ArrayGeometry source{
      .element_positions = {
          0.0, 0.0, 0.0,
          0.0, 0.001, 0.0,
          0.0, 0.002, 0.0,
      },
      .wavelength = 0.0039,
      .geometry_type = ArrayGeometryType::kUla,
  };
  // clang-format on
  const WireArrayGeometry wire_value = to_wire(source);
  ASSERT_NE(wire_value.packed_geometry, nullptr);
  ASSERT_EQ(wire_value.packed_geometry->shape().size(), 2U);
  EXPECT_EQ(wire_value.packed_geometry->shape()[0], 4);
  EXPECT_EQ(wire_value.packed_geometry->shape()[1], 3);

  const auto result = from_wire(wire_value);
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  EXPECT_EQ(*result, source);
}

TEST(WireArrayGeometry, RejectsWrongTensorRank) {
  WireArrayGeometry wire_value;
  const std::vector<double> packed(12U, 0.0);
  wire_value.packed_geometry = wire_bridge_detail::make_host_tensor<double>(
      std::span<const double>{packed}, DLDataType{kDLFloat, 64U, 1U});
  const auto result = from_wire(wire_value);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

TEST(WireArrayGeometry, UnsetGeometryTypeRoundTrips) {
  const ArrayGeometry source{
      .element_positions = {0.0, 0.0, 0.0},
      .wavelength = 0.0039,
  };
  const WireArrayGeometry wire_value = to_wire(source);
  EXPECT_FALSE(wire_value.has_geometry_type);
  const auto result = from_wire(wire_value);
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  EXPECT_EQ(*result, source);
}

void expect_window_round_trip(const WindowSpec& source) {
  WireWindowSpec wire_value = to_wire(source);
  ASSERT_NE(wire_value.coefficients, nullptr);
  const auto result = from_wire(wire_value);
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  EXPECT_EQ(*result, source);
}

FMCWWaveformSpec baseline_waveform() {
  return FMCWWaveformSpec{
      .carrier_frequency = 77.0e9,
      .bandwidth = 1.0e9,
      .chirp_duration = 50.0e-6,
      .sample_rate = 20.0e6,
      .chirp_repetition_interval = 60.0e-6,
  };
}

void expect_waveform_round_trip(const FMCWWaveformSpec& source) {
  const FMCWWaveformDescriptor expected{source};
  WireFMCWWaveform wire_value = to_wire(source);
  ASSERT_NE(wire_value.reference_samples, nullptr);
  const auto result = from_wire(wire_value);
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  EXPECT_EQ(*result, expected);
}

TEST(WireWindowSpec, EverySourceFieldRoundTrips) {
  WindowSpec value;
  value.type = WindowType::kHann;
  expect_window_round_trip(value);

  value = WindowSpec{};
  value.sidelobe_attenuation_db = 57.0;
  expect_window_round_trip(value);

  value = WindowSpec{};
  value.kaiser_beta = -1.0;
  expect_window_round_trip(value);

  value = WindowSpec{};
  value.taylor_nbar = 6;
  expect_window_round_trip(value);

  value = WindowSpec{};
  value.tukey_alpha = 0.25;
  expect_window_round_trip(value);

  value = WindowSpec{};
  value.coefficients = {0.25, 0.5, 0.25};
  expect_window_round_trip(value);

  value = WindowSpec{};
  value.coherent_gain_normalization = false;
  expect_window_round_trip(value);
}

TEST(WireWindowSpec, EmptyCoefficientsUsePresentZeroLengthTensor) {
  const WireWindowSpec wire_value = to_wire(WindowSpec{});
  ASSERT_NE(wire_value.coefficients, nullptr);
  EXPECT_EQ(wire_value.coefficients->nbytes(), 0);
  expect_window_round_trip(WindowSpec{});
}

TEST(WireFMCWWaveform, EverySourceFieldRoundTrips) {
  auto value = baseline_waveform();
  value.carrier_frequency = 79.0e9;
  expect_waveform_round_trip(value);

  value = baseline_waveform();
  value.bandwidth = 800.0e6;
  expect_waveform_round_trip(value);

  value = baseline_waveform();
  value.bandwidth = std::nullopt;
  value.chirp_slope = 18.0e12;
  expect_waveform_round_trip(value);

  value = baseline_waveform();
  value.chirp_duration = 55.0e-6;
  expect_waveform_round_trip(value);

  value = baseline_waveform();
  value.sample_rate = 25.0e6;
  expect_waveform_round_trip(value);

  value = baseline_waveform();
  value.chirp_repetition_interval = 75.0e-6;
  expect_waveform_round_trip(value);

  value = baseline_waveform();
  value.reference_samples = {{1.0, -2.0}, {3.5, 4.25}, {-1.0, 0.5}};
  expect_waveform_round_trip(value);
}

TEST(WireFMCWWaveform, EmptyReferenceUsesPresentZeroLengthTensor) {
  const WireFMCWWaveform wire_value = to_wire(baseline_waveform());
  ASSERT_NE(wire_value.reference_samples, nullptr);
  EXPECT_EQ(wire_value.reference_samples->nbytes(), 0);
  expect_waveform_round_trip(baseline_waveform());
}

TEST(WireFMCWWaveform, RejectsInvalidAndInconsistentMetadata) {
  WireFMCWWaveform invalid = to_wire(baseline_waveform());
  invalid.chirp_duration = 0.0;
  auto result = from_wire(invalid);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);

  WireFMCWWaveform inconsistent = to_wire(baseline_waveform());
  inconsistent.has_chirp_slope = true;
  inconsistent.chirp_slope = 1.0;
  result = from_wire(inconsistent);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

struct ReflectedField {
  std::string_view name;
  std::uint16_t id;
  reflection::BaseType base_type;
};

template <typename BinarySchema>
void expect_reflected_fields(std::string_view root_name,
                             const std::vector<ReflectedField>& expected_fields) {
  const reflection::Schema* schema = reflection::GetSchema(BinarySchema::data());
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->root_table(), nullptr);
  ASSERT_NE(schema->root_table()->name(), nullptr);
  EXPECT_EQ(schema->root_table()->name()->string_view(), root_name);

  const auto* fields = schema->root_table()->fields();
  ASSERT_NE(fields, nullptr);
  ASSERT_EQ(fields->size(), expected_fields.size());
  for (const ReflectedField& expected : expected_fields) {
    const reflection::Field* found = nullptr;
    for (const reflection::Field* field : *fields) {
      if (field->name()->string_view() == expected.name) {
        found = field;
        break;
      }
    }
    ASSERT_NE(found, nullptr) << expected.name;
    EXPECT_EQ(found->id(), expected.id) << expected.name;
    ASSERT_NE(found->type(), nullptr);
    EXPECT_EQ(found->type()->base_type(), expected.base_type) << expected.name;
  }
}

TEST(WireSchemas, EmbeddedReflectionMatchesWindowSpecContract) {
  expect_reflected_fields<WireWindowSpecBinarySchema>(
      "holoscan.radar.wire.WindowSpec",
      {
          {"type", 0U, reflection::BaseType::UByte},
          {"sidelobe_attenuation_db", 1U, reflection::BaseType::Double},
          {"kaiser_beta", 2U, reflection::BaseType::Double},
          {"has_kaiser_beta", 3U, reflection::BaseType::Bool},
          {"taylor_nbar", 4U, reflection::BaseType::Int},
          {"tukey_alpha", 5U, reflection::BaseType::Double},
          {"coefficients", 6U, reflection::BaseType::Obj},
          {"coherent_gain_normalization", 7U, reflection::BaseType::Bool},
      });
}

TEST(WireSchemas, EmbeddedReflectionMatchesArrayGeometryContract) {
  expect_reflected_fields<WireArrayGeometryBinarySchema>(
      "holoscan.radar.wire.ArrayGeometry",
      {
          {"packed_geometry", 0U, reflection::BaseType::Obj},
          {"has_geometry_type", 1U, reflection::BaseType::Bool},
          {"geometry_type", 2U, reflection::BaseType::UByte},
      });
}

TEST(WireSchemas, EmbeddedReflectionMatchesWaveformContract) {
  expect_reflected_fields<WireFMCWWaveformBinarySchema>(
      "holoscan.radar.wire.FMCWWaveform",
      {
          {"carrier_frequency", 0U, reflection::BaseType::Double},
          {"bandwidth", 1U, reflection::BaseType::Double},
          {"has_bandwidth", 2U, reflection::BaseType::Bool},
          {"chirp_slope", 3U, reflection::BaseType::Double},
          {"has_chirp_slope", 4U, reflection::BaseType::Bool},
          {"chirp_duration", 5U, reflection::BaseType::Double},
          {"sample_rate", 6U, reflection::BaseType::Double},
          {"chirp_repetition_interval", 7U, reflection::BaseType::Double},
          {"reference_samples", 8U, reflection::BaseType::Obj},
      });
}

}  // namespace
}  // namespace holoscan::radar
