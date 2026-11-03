// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/graph_records.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/testing/schema_snapshot.hpp>
#include <holoscan/testing/tensor_snapshot.hpp>

#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_authoring_registry.hpp"
#include "utils/perf_precision.hpp"
#include "utils/perf_report.hpp"

namespace holoscan::radar::perf {
struct NativeCaptureState {
  inline static std::ostream* stream{nullptr};
  inline static std::string case_id;
  inline static bool first{true};
  inline static std::vector<std::string> semantic_parameters;
};

inline std::string snapshot_base64(const std::vector<std::byte>& bytes) {
  constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  for (std::size_t i = 0; i < bytes.size(); i += 3U) {
    const auto a = std::to_integer<unsigned>(bytes[i]);
    const auto b = i + 1U < bytes.size() ? std::to_integer<unsigned>(bytes[i + 1U]) : 0U;
    const auto c = i + 2U < bytes.size() ? std::to_integer<unsigned>(bytes[i + 2U]) : 0U;
    result += alphabet[a >> 2U];
    result += alphabet[((a & 3U) << 4U) | (b >> 4U)];
    result += i + 1U < bytes.size() ? alphabet[((b & 15U) << 2U) | (c >> 6U)] : '=';
    result += i + 2U < bytes.size() ? alphabet[c & 63U] : '=';
  }
  return result;
}

inline void snapshot_coordinate(std::ostream& out, const holoscan::SampleMetadata& metadata) {
  const auto time = metadata.capture_time();
  if (!time.valid())
    throw std::runtime_error("native output has no capture coordinate");
  out << "\"sequence\":" << metadata.sequence << ",\"capture_time_ns\":" << time.timestamp_ns
      << ",\"clock_id\":" << time.clock_id.value;
}

inline void snapshot_tensor_content(std::ostream& out,
                                    const holoscan::testing::TensorSnapshot& snapshot) {
  const auto dtype = snapshot.dtype();
  std::string name;
  switch (dtype.code) {
    case kDLFloat:
      name = "float";
      break;
    case kDLComplex:
      name = "complex";
      break;
    case kDLInt:
      name = "int";
      break;
    case kDLUInt:
      name = "uint";
      break;
    case kDLBool:
      name = "bool";
      break;
    default:
      throw std::runtime_error("unsupported native output dtype");
  }
  if (dtype.code != kDLBool)
    name += std::to_string(dtype.bits);
  const auto device = snapshot.source_device().device_type;
  const auto memory = device == kDLCUDA          ? "device"
                      : device == kDLCUDAHost    ? "pinned_host"
                      : device == kDLCUDAManaged ? "managed"
                                                 : "host";
  out << "\"memory_kind\":\"" << memory << "\",\"shape\":" << detail::shape_json(snapshot.shape())
      << ",\"dtype\":\"" << name << "\",\"data_base64\":\"" << snapshot_base64(snapshot.bytes())
      << '"';
}

template <typename Schema>
void snapshot_schema(std::ostream& out, const holoscan::testing::SchemaSnapshot<Schema>& snapshot) {
  const auto metadata = snapshot.metadata();
  out << "{\"kind\":\"descriptor\",";
  snapshot_coordinate(out, snapshot.sample_metadata());
  out << ",\"value\":{\"metadata\":{";
  if constexpr (std::is_same_v<Schema, WireArrayGeometry>) {
    out << "\"has_geometry_type\":" << metadata.has_geometry_type
        << ",\"geometry_type\":" << static_cast<unsigned>(metadata.geometry_type);
  } else {
    static_assert(sizeof(Schema) == 0, "add canonical descriptor metadata export");
  }
  out << "},\"tensors\":[";
  const auto& fields = holoscan::DescriptorTensorSchemaNativeTraits<Schema>::tensor_field_ids;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const holoscan::TensorFieldSelector<Schema> field{fields[i], static_cast<uint16_t>(i)};
    out << (i ? "," : "") << "{\"field_id\":" << fields[i]
        << ",\"present\":" << snapshot.has_tensor(field);
    if (snapshot.has_tensor(field)) {
      out << ',';
      snapshot_tensor_content(out, snapshot.tensor(field));
    }
    out << '}';
  }
  out << "]}}";
}

template <typename Schema>
void snapshot_schema_port(std::ostream& out, const holoscan::testing::TestResult& result,
                          const std::string& port) {
  bool first = true;
  for (const auto& snapshot : result.snapshots<Schema>(port)) {
    out << (std::exchange(first, false) ? "" : ",");
    snapshot_schema(out, snapshot);
  }
}

inline holoscan::Tensor native_live_payload(const std::vector<double>& value) {
  return *wire_bridge_detail::make_host_tensor<double>(value, DLDataType{kDLFloat, 64U, 1U});
}
inline holoscan::Tensor native_live_payload(const std::vector<int64_t>& value) {
  return *wire_bridge_detail::make_host_tensor<int64_t>(value, DLDataType{kDLInt, 64U, 1U});
}
inline WireArrayGeometry native_live_payload(const ArrayGeometry& value) {
  return to_wire(value);
}
inline WireWindowSpec native_live_payload(const WindowSpec& value) {
  return to_wire(value);
}
template <typename Value>
Value native_live_payload(const Value& value) {
  return value;
}

template <typename T>
void stage_scaled_native_input(holoscan::Tensor& destination, const holoscan::Tensor& source,
                               std::size_t frame) {
  std::vector<T> values(static_cast<std::size_t>(source.nbytes()) / sizeof(T));
  if (cudaMemcpy(values.data(), source.data(), source.nbytes(), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    throw std::runtime_error("cannot read native fixture input");
  const T scale = static_cast<T>(1.0 + static_cast<double>(frame) * 0.125);
  for (auto& value : values)
    value *= scale;
  auto writable = destination.mutable_data();
  if (!writable || cudaMemcpy(writable.value(), values.data(), source.nbytes(),
                              cudaMemcpyHostToDevice) != cudaSuccess)
    throw std::runtime_error("cannot stage varying native fixture input");
}

template <typename Op, std::size_t N, typename Outputs, typename LiveStage>
TimingCollector capture_native_recipe(
    const std::array<std::shared_ptr<holoscan::Tensor>, N>& inputs, std::string_view registry,
    const MaxTensorExtents& extents, const typename Op::Params& params,
    const std::array<std::string, N>& ports, const Outputs& outputs, LiveStage stage_live) {
  constexpr std::size_t frames = 3U;
  const auto factory = [=](holoscan::Graph& graph) {
    return PerfRegisteredOperator<Op>::construct(graph, registry, "op", extents, params);
  };
  holoscan::Graph inspection{"native-recipe-inspection"};
  const auto records = factory(inspection).ports();
  holoscan::testing::NativeOperatorTest fixture(factory);
  std::vector<std::string> wired;
  for (const auto& port : records) {
    if (port.direction != holoscan::PortDirection::kOutput)
      continue;
    const bool tensor_output = holoscan::authoring::payload_identities_equal(
        port.payload_identity,
        holoscan::detail::operator_spec_payload_identity<holoscan::Tensor>());
    // SDK admission requires bindings even for intentionally unwired pools.
    if (tensor_output)
      fixture.output_device(port.name, holoscan::DeviceId{0});
    if (std::find(outputs.unwired.begin(), outputs.unwired.end(), port.name) !=
        outputs.unwired.end())
      continue;
    std::size_t bytes = 0U;
    if (port.setup.tensor_contract && port.setup.tensor_contract->max_byte_span)
      bytes = static_cast<std::size_t>(*port.setup.tensor_contract->max_byte_span);
    for (const auto& field : port.setup.tensor_fields) {
      if (!field.tensor_contract || !field.tensor_contract->max_byte_span)
        throw std::runtime_error("native descriptor field lacks an authored byte bound: " +
                                 port.name);
      bytes += static_cast<std::size_t>(*field.tensor_contract->max_byte_span);
    }
    if (bytes == 0U)
      throw std::runtime_error("native output lacks an authored byte bound: " + port.name);
    if (tensor_output) {
      fixture.declare_capture(port.name,
                              holoscan::testing::snapshot_capture<holoscan::Tensor>(frames, bytes));
    } else if (holoscan::authoring::payload_identities_equal(
                   port.payload_identity,
                   holoscan::detail::operator_spec_payload_identity<WireArrayGeometry>())) {
      fixture.declare_capture(
          port.name, holoscan::testing::snapshot_capture<WireArrayGeometry>(frames, bytes));
    } else {
      throw std::runtime_error("unsupported native output payload: " + port.name);
    }
    wired.push_back(port.name);
  }
  for (std::size_t i = 0; i < N; ++i)
    fixture.input_source_device(ports[i], holoscan::DeviceId{0});
  // Native fixture inputs use SDK readiness-aware storage. Vary floating
  // data each frame to expose stale output reuse; indices and masks stay valid.
  std::size_t waveform_frames = 0U;
  std::size_t mask_frames = 0U;
  std::ostringstream semantic;
  if (!NativeCaptureState::semantic_parameters.empty())
    semantic << ",\"semantic_inputs\":[";
  for (std::size_t frame = 0; frame < frames; ++frame) {
    std::array<holoscan::Tensor, N> staged_inputs;
    for (std::size_t i = 0; i < N; ++i) {
      auto allocated = holoscan::make_tensor(inputs[i]->shape(), inputs[i]->dtype(), std::nullopt,
                                             holoscan::MemoryStorageType::kDevice,
                                             std::make_shared<holoscan::UnboundedAllocator>());
      if (!allocated)
        throw std::runtime_error(allocated.error().message);
      staged_inputs[i] = std::move(*allocated);
      if (inputs[i]->nbytes() == 0)
        continue;
      const auto dtype = inputs[i]->dtype();
      if (frame > 0U && (dtype.code == kDLFloat || dtype.code == kDLComplex)) {
        const auto scalar_bits = dtype.code == kDLComplex ? dtype.bits / 2U : dtype.bits;
        if (scalar_bits == 32U)
          stage_scaled_native_input<float>(staged_inputs[i], *inputs[i], frame);
        else if (scalar_bits == 64U)
          stage_scaled_native_input<double>(staged_inputs[i], *inputs[i], frame);
        else
          throw std::runtime_error("unsupported native fixture scalar precision");
      } else {
        auto writable = staged_inputs[i].mutable_data();
        if (ports[i] == "is_valid" && dtype.bits == 8U &&
            (dtype.code == kDLBool || dtype.code == kDLUInt)) {
          std::vector<uint8_t> mask(static_cast<std::size_t>(inputs[i]->nbytes()));
          for (std::size_t index = 0; index < mask.size(); ++index)
            mask[index] = frame == 0U ? 1U : (frame == 1U ? 0U : index % 2U);
          if (!writable || cudaMemcpy(writable.value(), mask.data(), mask.size(),
                                      cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("cannot stage varying native validity mask");
        } else if (!writable || cudaMemcpy(writable.value(), inputs[i]->data(), inputs[i]->nbytes(),
                                           cudaMemcpyDeviceToDevice) != cudaSuccess) {
          throw std::runtime_error("cannot stage native fixture input");
        }
      }
    }
    if (!NativeCaptureState::semantic_parameters.empty()) {
      semantic << (frame ? "," : "")
               << "{\"parameters\":" << NativeCaptureState::semantic_parameters.at(frame)
               << ",\"inputs\":{";
      for (std::size_t i = 0; i < N; ++i) {
        std::vector<std::byte> bytes(static_cast<std::size_t>(staged_inputs[i].nbytes()));
        if (!bytes.empty() && cudaMemcpy(bytes.data(), staged_inputs[i].data(), bytes.size(),
                                         cudaMemcpyDeviceToHost) != cudaSuccess)
          throw std::runtime_error("cannot capture semantic input");
        const auto dtype = staged_inputs[i].dtype();
        semantic << (i ? "," : "") << '\"' << detail::json_escape(ports[i])
                 << "\":{\"shape\":" << detail::shape_json(staged_inputs[i].shape())
                 << ",\"dtype\":\"" << (dtype.code == kDLFloat ? "float" : "int")
                 << static_cast<unsigned>(dtype.bits) << "\",\"data_base64\":\""
                 << snapshot_base64(bytes) << "\"}";
      }
      semantic << "}}";
    }
    holoscan::testing::TestFrame staged;
    for (std::size_t i = 0; i < N; ++i) {
      auto entry =
          holoscan::testing::frame_at(std::chrono::milliseconds{frame + 1U},
                                      holoscan::testing::input(ports[i], staged_inputs[i]));
      staged.fields.push_back(std::move(entry.fields.front()));
    }
    stage_live(staged, frame);
    if constexpr (requires { params.min_subarray_size; }) {
      const bool matched_mask =
          std::any_of(staged.fields.begin(), staged.fields.end(), [](const auto& field) {
            return field.name == "is_valid";
          });
      mask_frames += matched_mask ? 1U : 0U;
    }
    if constexpr (requires { params.waveform.is_valid(); }) {
      const bool matched_waveform =
          std::any_of(staged.fields.begin(), staged.fields.end(), [](const auto& field) {
            return field.name == "waveform";
          });
      waveform_frames += params.waveform.is_valid() || matched_waveform ? 1U : 0U;
    }
    fixture.push_frame(staged);
  }
  if (!NativeCaptureState::semantic_parameters.empty())
    semantic << "]";
  auto running = fixture.run_async({.expect_stop = true});
  for (const auto& port : wired) {
    std::size_t expected = frames;
    // SubarraySelection forwards a mask only on ticks with a matching mask input.
    if constexpr (requires { params.min_subarray_size; }) {
      if (port == "is_valid_out")
        expected = mask_frames;
    }
    if constexpr (requires { params.waveform.is_valid(); }) {
      if (port == "range_bins" || port == "velocity_bins")
        expected = waveform_frames;
    }
    if (expected != 0U && !running.wait_for_output(port, expected, std::chrono::seconds{20})) {
      running.request_stop();
      const auto failed = running.finish();
      throw std::runtime_error("native recipe output incomplete on " + port + ": " +
                               failed.describe_failure());
    }
  }
  if (!wired.empty())
    running.request_stop();
  const auto result = running.finish();
  if (!result.ok() || result.termination() != holoscan::RunTermination::kStopped)
    throw std::runtime_error("native recipe failed: " + result.describe_failure());
  auto& out = *NativeCaptureState::stream;
  out << (std::exchange(NativeCaptureState::first, false) ? "" : ",") << std::setprecision(17)
      << "{\"id\":\"" << detail::json_escape(NativeCaptureState::case_id) << "\",\"precision\":\""
      << kPrecision << "\",\"termination\":\"stopped\",\"diagnostics\":[";
  bool first_diagnostic = true;
  for (const auto& diagnostic : result.diagnostics()) {
    out << (std::exchange(first_diagnostic, false) ? "" : ",") << "{\"id\":" << diagnostic.id.value
        << ",\"severity\":" << static_cast<unsigned>(diagnostic.severity) << ",\"token\":\""
        << detail::json_escape(diagnostic.token) << "\",\"required\":\""
        << detail::json_escape(diagnostic.required) << "\",\"provided\":\""
        << detail::json_escape(diagnostic.provided) << "\"}";
  }
  out << "]" << semantic.str() << ",\"coverage_gaps\":[],\"outputs\":{";
  bool first = true;
  for (const auto& port : records) {
    if (port.direction != holoscan::PortDirection::kOutput)
      continue;
    const bool connected = std::find(wired.begin(), wired.end(), port.name) != wired.end();
    out << (std::exchange(first, false) ? "" : ",") << '"' << detail::json_escape(port.name)
        << "\":{\"wired\":" << (connected ? "true" : "false") << ",\"emissions\":[";
    if (connected) {
      if (holoscan::authoring::payload_identities_equal(
              port.payload_identity,
              holoscan::detail::operator_spec_payload_identity<holoscan::Tensor>())) {
        bool first_emission = true;
        for (const auto& snapshot : result.snapshots<holoscan::Tensor>(port.name)) {
          out << (std::exchange(first_emission, false) ? "" : ",") << "{\"kind\":\"tensor\",";
          snapshot_coordinate(out, snapshot.sample_metadata());
          out << ',';
          snapshot_tensor_content(out, snapshot);
          out << '}';
        }
      } else if (holoscan::authoring::payload_identities_equal(
                     port.payload_identity,
                     holoscan::detail::operator_spec_payload_identity<WireArrayGeometry>())) {
        snapshot_schema_port<WireArrayGeometry>(out, result, port.name);
      }
    }
    out << "]}";
  }
  out << '}';
  out << '}';
  return {};
}
}  // namespace holoscan::radar::perf
