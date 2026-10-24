// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Native SDK snapshots own values and observed output sample metadata. Capture
// remains separate from the timed performance Graph.
#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

#include <bit>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/time/time_point.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/perf_report.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::ClutterFilterMethod;
using holoscan::radar::ClutterFilterOp;
using holoscan::radar::ClutterFilterParams;
using holoscan::radar::FMCWRangeDopplerProcessorOp;
using holoscan::radar::FMCWRangeDopplerProcessorParams;
using holoscan::radar::FMCWRangeProcessorOp;
using holoscan::radar::FMCWRangeProcessorParams;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::FMCWWaveformSpec;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::perf::detail::json_escape;

constexpr std::size_t kFrames = 3U;
constexpr auto kTimeout = std::chrono::seconds{10};

std::string base64(const std::vector<std::byte>& bytes) {
  constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  for (std::size_t index = 0; index < bytes.size(); index += 3U) {
    const auto a = std::to_integer<unsigned>(bytes[index]);
    const auto b = index + 1U < bytes.size() ? std::to_integer<unsigned>(bytes[index + 1U]) : 0U;
    const auto c = index + 2U < bytes.size() ? std::to_integer<unsigned>(bytes[index + 2U]) : 0U;
    result += kAlphabet[a >> 2U];
    result += kAlphabet[((a & 3U) << 4U) | (b >> 4U)];
    result += index + 1U < bytes.size() ? kAlphabet[((b & 15U) << 2U) | (c >> 6U)] : '=';
    result += index + 2U < bytes.size() ? kAlphabet[c & 63U] : '=';
  }
  return result;
}

MaxTensorExtents bounds(int64_t chirps, int64_t samples, int64_t channels = 1) {
  MaxTensorExtents extents;
  extents.max_chirps = chirps;
  extents.max_samples_per_chirp = samples;
  extents.max_range_bins = samples * 2;
  extents.max_rx = channels;
  extents.max_tx = 1;
  return extents;
}

template <typename T>
holoscan::Tensor input_tensor(int64_t chirps, int64_t samples, int frame, int64_t channels = 1,
                              bool real_input = false) {
  const auto count = static_cast<std::size_t>(channels * chirps * samples);
  std::vector<int64_t> shape{chirps, samples};
  if (channels > 1) {
    shape.insert(shape.begin(), channels);
  }
  if (real_input) {
    std::vector<T> data(count);
    for (std::size_t index = 0; index < count; ++index) {
      data[index] = static_cast<T>((index % 17U) + frame) - static_cast<T>(index % 5U);
    }
    return *holoscan::radar::testing::to_device_tensor(data, shape,
                                                       holoscan::radar::real_dtype<T>());
  }
  std::vector<std::complex<T>> data(count);
  for (std::size_t index = 0; index < count; ++index) {
    data[index] = {static_cast<T>((index % 17U) + frame), static_cast<T>(index % 7U)};
  }
  return *holoscan::radar::testing::to_device_tensor(data, shape,
                                                     holoscan::radar::complex_dtype<T>());
}

FMCWWaveformDescriptor fixture_waveform(double bandwidth = 150e6) {
  FMCWWaveformSpec spec;
  spec.carrier_frequency = 77e9;
  spec.bandwidth = bandwidth;
  spec.chirp_duration = 20e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25e-6;
  return FMCWWaveformDescriptor(spec);
}

WindowSpec hann_window() {
  WindowSpec window;
  window.type = WindowType::kHann;
  return window;
}

// One-tick Exact override pushed with frame 1 only.
enum class Live { kNone, kNfft, kWindow, kWaveform };

// Native range-stage case: Params, input form, one optional live override,
// and optional-output wiring. Covers both the split and fused operators.
struct StageCase {
  std::string id;
  int64_t chirps;
  int64_t samples;
  int64_t channels{1};
  bool real_input{false};
  bool hann{false};
  bool dechirp{false};
  bool padded{false};
  Live live{Live::kNone};
  bool optional_wired{true};
  bool waveform_present{true};
};

template <typename T, typename Fixture>
void push_stage_frames(Fixture& fixture, const StageCase& spec, const std::string& nfft_port) {
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    const auto tensor = input_tensor<T>(spec.chirps, spec.samples, static_cast<int>(frame),
                                        spec.channels, spec.real_input);
    const auto stamp = std::chrono::milliseconds{static_cast<int64_t>(frame + 1U)};
    const auto cube = holoscan::testing::input("adc_cube", tensor);
    if (frame != 1U || spec.live == Live::kNone) {
      fixture.push_frame(holoscan::testing::frame_at(stamp, cube));
    } else if (spec.live == Live::kNfft) {
      fixture.push_frame(holoscan::testing::frame_at(
          stamp, holoscan::testing::input(nfft_port, int64_t{spec.samples * 2}), cube));
    } else if (spec.live == Live::kWindow) {
      fixture.push_frame(holoscan::testing::frame_at(
          stamp, holoscan::testing::input("range_window", holoscan::radar::to_wire(hann_window())),
          cube));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(
          stamp,
          holoscan::testing::input("waveform", holoscan::radar::to_wire(fixture_waveform(75e6))),
          cube));
    }
  }
}

template <typename Fixture>
holoscan::testing::TestResult finish(Fixture& fixture, const std::vector<std::string>& ports) {
  auto running = fixture.run_async({.expect_stop = true});
  bool observed = true;
  for (const auto& port : ports) {
    observed = running.wait_for_output(port, kFrames, kTimeout) && observed;
  }
  running.request_stop();
  auto result = running.finish();
  if (!observed || !result.ok() || result.termination() != holoscan::RunTermination::kStopped) {
    throw std::runtime_error("verification fixture failed: " + result.describe_failure());
  }
  return result;
}

template <typename T>
void write_port(std::ostream& stream, const holoscan::testing::TestResult& result,
                const std::string& port, bool wired) {
  stream << '"' << json_escape(port) << "\":{\"wired\":" << (wired ? "true" : "false")
         << ",\"emissions\":[";
  if (wired) {
    const auto snapshots = result.snapshots<holoscan::Tensor>(port);
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
      const auto host = holoscan::radar::testing::to_host(snapshots[index]);
      const auto& metadata = snapshots[index].sample_metadata();
      const auto coordinate = metadata.capture_time();
      if (!coordinate.valid()) {
        throw std::runtime_error("verification output is missing its capture coordinate: " + port);
      }
      if (snapshots[index].source_device().device_type != kDLCUDA) {
        throw std::runtime_error("verification output is not device memory: " + port);
      }
      const bool complex = host.dtype.code == kDLComplex;
      const std::string dtype =
          std::string(complex ? "complex" : "float") + std::to_string(host.dtype.bits);
      stream << (index ? "," : "") << "{\"kind\":\"tensor\",\"sequence\":" << metadata.sequence
             << ",\"capture_time_ns\":" << coordinate.timestamp_ns
             << ",\"clock_id\":" << coordinate.clock_id.value
             << ",\"memory_kind\":\"device\",\"shape\":"
             << holoscan::radar::perf::detail::shape_json(host.shape) << ",\"dtype\":\"" << dtype
             << "\",\"data_base64\":\"" << base64(host.data) << "\"}";
    }
  }
  stream << "]}";
}

void write_diagnostics(std::ostream& stream, const holoscan::testing::TestResult& result) {
  stream << '[';
  bool first = true;
  for (const auto& diagnostic : result.diagnostics()) {
    // Exclude graph-generation and plan-local IDs; retain semantic SDK evidence.
    stream << (std::exchange(first, false) ? "" : ",") << "{\"id\":" << diagnostic.id.value
           << ",\"severity\":" << static_cast<unsigned>(diagnostic.severity) << ",\"token\":\""
           << json_escape(diagnostic.token) << "\",\"required\":\""
           << json_escape(diagnostic.required) << "\",\"provided\":\""
           << json_escape(diagnostic.provided) << "\",\"port\":\""
           << json_escape(diagnostic.detail ? diagnostic.detail->port_name : "") << "\"}";
  }
  stream << ']';
}

template <typename T>
void clutter_case(std::ostream& stream, int64_t chirps, int64_t samples, const std::string& id,
                  bool live) {
  using Op = ClutterFilterOp<T>;
  auto fixture = holoscan::testing::create_operator_test<Op>(bounds(chirps, samples));
  fixture.set_params(ClutterFilterParams{.method = ClutterFilterMethod::kNone});
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    auto tensor = input_tensor<T>(chirps, samples, static_cast<int>(frame));
    const auto stamp = std::chrono::milliseconds{static_cast<int64_t>(frame + 1U)};
    if (live && frame == 1U) {
      fixture.push_frame(holoscan::testing::frame_at(
          stamp, holoscan::testing::input("method", ClutterFilterMethod::kMeanSubtraction),
          holoscan::testing::input("range_profiles", tensor)));
    } else {
      fixture.push_frame(
          holoscan::testing::frame_at(stamp, holoscan::testing::input("range_profiles", tensor)));
    }
  }
  const auto maximum_bytes = static_cast<std::size_t>(chirps * samples) * sizeof(std::complex<T>);
  fixture.template capture_snapshot<holoscan::Tensor>("range_profiles_out", kFrames, maximum_bytes);
  fixture.input_source_device("range_profiles", holoscan::DeviceId{0});
  fixture.output_device("range_profiles_out", holoscan::DeviceId{0});
  const auto result = finish(fixture, {"range_profiles_out"});
  stream << "{\"id\":\"" << id << "\",\"precision\":\""
         << (std::is_same_v<T, float> ? "float" : "double")
         << "\",\"termination\":\"stopped\",\"diagnostics\":";
  write_diagnostics(stream, result);
  stream << ",\"coverage_gaps\":[],\"outputs\":{";
  write_port<T>(stream, result, "range_profiles_out", true);
  stream << "}}";
}

template <typename T>
void write_case_header(std::ostream& stream, const std::string& id,
                       const holoscan::testing::TestResult& result) {
  stream << "{\"id\":\"" << id << "\",\"precision\":\""
         << (std::is_same_v<T, float> ? "float" : "double")
         << "\",\"termination\":\"stopped\",\"diagnostics\":";
  write_diagnostics(stream, result);
  stream << ",\"coverage_gaps\":[],\"outputs\":{";
}

std::size_t stage_capture_bytes(const StageCase& spec, std::size_t element_bytes) {
  return static_cast<std::size_t>(spec.channels * spec.chirps * spec.samples * 2) * element_bytes;
}

template <typename T>
void range_case(std::ostream& stream, const StageCase& spec) {
  using Op = FMCWRangeProcessorOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(bounds(spec.chirps, spec.samples, spec.channels));
  FMCWRangeProcessorParams params;
  if (spec.waveform_present) {
    params.waveform = fixture_waveform();
  }
  if (spec.hann) {
    params.range_window = hann_window();
  }
  params.apply_dechirp = spec.dechirp;
  params.nfft = spec.padded ? spec.samples * 2 : 0;
  fixture.set_params(std::move(params));
  push_stage_frames<T>(fixture, spec, "nfft");
  const auto maximum_bytes = stage_capture_bytes(spec, sizeof(std::complex<T>));
  fixture.template capture_snapshot<holoscan::Tensor>("range_profiles", kFrames, maximum_bytes);
  fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
  fixture.output_device("range_profiles", holoscan::DeviceId{0});
  fixture.output_device("range_bins", holoscan::DeviceId{0});
  std::vector<std::string> wait_ports{"range_profiles"};
  if (spec.optional_wired) {
    fixture.template capture_snapshot<holoscan::Tensor>("range_bins", kFrames, maximum_bytes);
    if (spec.waveform_present) {
      wait_ports.emplace_back("range_bins");
    }
  }
  const auto result = finish(fixture, wait_ports);
  write_case_header<T>(stream, spec.id, result);
  write_port<T>(stream, result, "range_profiles", true);
  stream << ',';
  write_port<T>(stream, result, "range_bins", spec.optional_wired);
  stream << "}}";
}

// Shared-stage consumer: same range-stage inputs, inline Doppler defaults.
template <typename T>
void fused_case(std::ostream& stream, const StageCase& spec) {
  using Op = FMCWRangeDopplerProcessorOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(bounds(spec.chirps, spec.samples, spec.channels));
  FMCWRangeDopplerProcessorParams params;
  if (spec.waveform_present) {
    params.waveform = fixture_waveform();
  }
  if (spec.hann) {
    params.range_window = hann_window();
  }
  params.apply_dechirp = spec.dechirp;
  params.nfft_range = spec.padded ? spec.samples * 2 : 0;
  fixture.set_params(std::move(params));
  push_stage_frames<T>(fixture, spec, "nfft_range");
  const auto maximum_bytes = stage_capture_bytes(spec, sizeof(std::complex<T>));
  fixture.template capture_snapshot<holoscan::Tensor>("rd_maps", kFrames, maximum_bytes);
  fixture.input_source_device("adc_cube", holoscan::DeviceId{0});
  for (const char* port : {"rd_maps", "range_bins", "velocity_bins"}) {
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  std::vector<std::string> wait_ports{"rd_maps"};
  if (spec.optional_wired) {
    for (const char* port : {"range_bins", "velocity_bins"}) {
      fixture.template capture_snapshot<holoscan::Tensor>(port, kFrames, maximum_bytes);
      if (spec.waveform_present) {
        wait_ports.emplace_back(port);
      }
    }
  }
  const auto result = finish(fixture, wait_ports);
  write_case_header<T>(stream, spec.id, result);
  write_port<T>(stream, result, "rd_maps", true);
  stream << ',';
  write_port<T>(stream, result, "range_bins", spec.optional_wired);
  stream << ',';
  write_port<T>(stream, result, "velocity_bins", spec.optional_wired);
  stream << "}}";
}

// Range-stage inventory shared by the split and fused fixtures. The original
// v1 IDs keep their meaning; the rest cover window, dechirp, padding, real-IF,
// named-channel, and live window/waveform paths.
std::vector<StageCase> stage_cases(const YAML::Node& profiles) {
  std::vector<StageCase> cases;
  for (const auto& name : {"small", "medium", "large"}) {
    const auto shape = profiles[name]["shape"].template as<std::vector<int64_t>>();
    if (shape.size() != 2U || shape[0] <= 0 || shape[1] <= 0) {
      throw std::invalid_argument("verification requires two positive named profile axes");
    }
    const std::string size = name;
    cases.push_back({.id = size + "-baseline", .chirps = shape[0], .samples = shape[1]});
    cases.push_back({.id = size + "-live-match-miss",
                     .chirps = shape[0],
                     .samples = shape[1],
                     .live = Live::kNfft});
    cases.push_back({.id = size + "-hann", .chirps = shape[0], .samples = shape[1], .hann = true});
    cases.push_back({.id = size + "-padded-hann",
                     .chirps = shape[0],
                     .samples = shape[1],
                     .hann = true,
                     .padded = true});
  }
  const auto small = profiles["small"]["shape"].as<std::vector<int64_t>>();
  const auto medium = profiles["medium"]["shape"].as<std::vector<int64_t>>();
  const int64_t c = medium[0];
  const int64_t s = medium[1];
  cases.push_back({.id = "medium-padded", .chirps = c, .samples = s, .padded = true});
  cases.push_back({.id = "medium-dechirp", .chirps = c, .samples = s, .dechirp = true});
  cases.push_back(
      {.id = "medium-dechirp-hann", .chirps = c, .samples = s, .hann = true, .dechirp = true});
  cases.push_back({.id = "medium-dechirp-padded-hann",
                   .chirps = c,
                   .samples = s,
                   .hann = true,
                   .dechirp = true,
                   .padded = true});
  cases.push_back({.id = "medium-real-if", .chirps = c, .samples = s, .real_input = true});
  cases.push_back({.id = "medium-real-if-padded-hann",
                   .chirps = c,
                   .samples = s,
                   .real_input = true,
                   .hann = true,
                   .padded = true});
  cases.push_back(
      {.id = "medium-four-channels-hann", .chirps = c, .samples = s, .channels = 4, .hann = true});
  cases.push_back({.id = "medium-live-window", .chirps = c, .samples = s, .live = Live::kWindow});
  cases.push_back(
      {.id = "medium-live-waveform", .chirps = c, .samples = s, .live = Live::kWaveform});
  cases.push_back({.id = "medium-live-waveform-dechirp",
                   .chirps = c,
                   .samples = s,
                   .dechirp = true,
                   .live = Live::kWaveform});
  cases.push_back(
      {.id = "optional-unwired", .chirps = small[0], .samples = small[1], .optional_wired = false});
  cases.push_back({.id = "optional-suppressed",
                   .chirps = small[0],
                   .samples = small[1],
                   .waveform_present = false});
  return cases;
}

template <typename T>
void write_cases(std::ostream& stream, const std::string& op, const YAML::Node& profiles,
                 bool* first) {
  const auto separator = [&] {
    stream << (*first ? "" : ",");
    *first = false;
  };
  if (op == "clutter_filter") {
    for (const auto& name : {"small", "medium", "large"}) {
      const auto shape = profiles[name]["shape"].template as<std::vector<int64_t>>();
      if (shape.size() != 2U || shape[0] <= 0 || shape[1] <= 0) {
        throw std::invalid_argument("verification requires two positive named profile axes");
      }
      for (bool live : {false, true}) {
        separator();
        const std::string id = std::string(name) + (live ? "-live-match-miss" : "-baseline");
        clutter_case<T>(stream, shape[0], shape[1], id, live);
      }
    }
    return;
  }
  const bool fused = op == "fmcw_range_doppler_processor";
  for (const auto& spec : stage_cases(profiles)) {
    separator();
    if (fused) {
      fused_case<T>(stream, spec);
    } else {
      range_case<T>(stream, spec);
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument("usage: verify_operator <operator> <recipe.yaml> <output.json>");
    }
    if (std::endian::native != std::endian::little) {
      throw std::runtime_error("snapshot export currently requires little-endian storage");
    }
    const std::string op = argv[1];
    if (op != "clutter_filter" && op != "fmcw_range_processor" &&
        op != "fmcw_range_doppler_processor") {
      throw std::invalid_argument("no native verification fixture registered for " + op);
    }
    const auto recipe = YAML::LoadFile(argv[2]);
    if (recipe["operator"].as<std::string>() != op) {
      throw std::invalid_argument("recipe/operator mismatch");
    }
    std::ofstream stream(argv[3]);
    if (!stream) {
      throw std::runtime_error("cannot open snapshot export");
    }
    stream << "{\"schema_version\":1,\"byte_order\":\"little\",\"operator\":\"" << op
           << "\",\"fixture_id\":\"native-v2-" << op << "\",\"cases\":[";
    bool first = true;
    write_cases<float>(stream, op, recipe["profiles"], &first);
    write_cases<double>(stream, op, recipe["profiles"], &first);
    stream << "]}\n";
    if (!stream) {
      throw std::runtime_error("snapshot export write failed");
    }
    std::cerr << "Value snapshots and observed output coordinates exported.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
