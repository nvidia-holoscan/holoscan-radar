// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for MVDREstimatorOp — adaptive (Capon)
// sparse-path DOA. Per item it forms a snapshot covariance, diagonal-loads
// and inverts it (batched), then scans the (az, el) grid for the Capon
// spectrum. Sweeps detection count, channel count, snapshot count, and
// scan-grid size. Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.
//
//  Graph-based rewrite of the HSDK 4.x Application harness.
// Measures each live override independently through the production Exact
// contract.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <set>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

struct SizeSpec {
  std::string id;
  int64_t channels;
  int64_t detections;
  int64_t snapshots;
  int64_t num_az;
  int64_t num_el;
  std::string role;
};

struct MvdrScenario {
  enum class OverrideKind {
    kNone,
    kArrayGeometry,
    kAzAngles,
    kElAngles,
    kSpatialSmoothing,
    kDiagonalLoading,
    kMaxSources,
  };

  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t snapshots{0};
  int64_t num_az{0};
  int64_t num_el{0};
  OverrideKind override_kind{OverrideKind::kNone};
  double static_loading{-1.0};
  int64_t static_max_sources{1};
};

const char* override_name(MvdrScenario::OverrideKind kind) {
  switch (kind) {
    case MvdrScenario::OverrideKind::kNone:
      return "";
    case MvdrScenario::OverrideKind::kArrayGeometry:
      return "array_geometry";
    case MvdrScenario::OverrideKind::kAzAngles:
      return "az_coordinates";
    case MvdrScenario::OverrideKind::kElAngles:
      return "el_coordinates";
    case MvdrScenario::OverrideKind::kSpatialSmoothing:
      return "spatial_smoothing";
    case MvdrScenario::OverrideKind::kDiagonalLoading:
      return "diagonal_loading";
    case MvdrScenario::OverrideKind::kMaxSources:
      return "max_sources";
  }
  return "unknown";
}

const std::vector<SizeSpec>& size_ladder() {
  // Covariance + inverse scratch is O(detections · channels²); the ladder
  // keeps that bounded rather than chasing wide dense-array sizes.
  static const std::vector<SizeSpec> ladder = {
      {"xs", 4, 64, 16, 21, 11, "small array; fast smoke"},
      {"sm", 8, 128, 32, 41, 21, "compact SIMO array"},
      {"md", 16, 256, 48, 61, 31, "common dev size"},
      {"lg", 32, 512, 64, 81, 41, "wide aperture, many looks"},
      {"xl", 32, 1024, 96, 101, 51, "upper characterization bound"}};
  return ladder;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& s : size_ladder()) {
    if (s.id == id) {
      return &s;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* s = find_size(id);
  if (s == nullptr) {
    std::cerr << "internal error: unknown size id '" << id << "'\n";
    std::abort();
  }
  return *s;
}

std::vector<double> linspace(double start, double stop, int64_t n) {
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

hr::ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing, double wavelength) {
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

holoscan::expected<void, holoscan::Error> emit_host_vector(
    holoscan::Output<holoscan::Tensor>& output, std::span<const double> values,
    holoscan::TimePoint capture_time) {
  const std::array<int64_t, 1> shape{static_cast<int64_t>(values.size())};
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
  return output.emit(std::move(*loan), holoscan::EmitOptions{.capture_time = capture_time});
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

hr::SpatialSmoothingOverride smoothing_wire(const std::vector<int64_t>& smoothing) {
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

class PerfMVDROverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfMVDROverrideSourceOp(int64_t override_at, hr::MVDRParams initial, hr::MVDRParams override,
                           MvdrScenario::OverrideKind kind)
  : override_at_(override_at),
    initial_(std::move(initial)),
    override_(std::move(override)),
    kind_(kind) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    constexpr holoscan::TensorRepresentation kHostVector{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 1U,
    };
    constexpr holoscan::TensorRepresentation kHostMatrix{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 2U,
    };
    constexpr std::size_t kHostPayloadSpan = 4096U * sizeof(double);
    spec.output(snapshot).produces_tensor(
        hr::pooled_tensor_output(kDevice, 8U, perf::PerfFrameBank::span_a()));
    spec.output(array_geometry)
        .max_emits_per_compute(1U)
        .produces_tensor(hr::pooled_tensor_output(kHostMatrix, 2U, kHostPayloadSpan));
    spec.output(az_coordinates)
        .max_emits_per_compute(1U)
        .produces_tensor(hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(el_coordinates)
        .max_emits_per_compute(1U)
        .produces_tensor(hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(spatial_smoothing).max_emits_per_compute(1U);
    spec.output(diagonal_loading).max_emits_per_compute(1U);
    spec.output(max_sources).max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = perf::kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    perf::PerfPublishMode mode = perf::PerfPublishMode::kCopy;
    {
      const std::lock_guard lock(perf::PerfFrameBank::mutex);
      if (perf::PerfFrameBank::in_flight != 0 || perf::PerfFrameBank::frames_remaining <= 0 ||
          perf::PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --perf::PerfFrameBank::frames_remaining;
      frame = perf::PerfFrameBank::input_frame;
      mode = perf::PerfFrameBank::publish_mode;
      perf::detail::record_start_locked();
    }

    if (frame_index_ == override_at_) {
      const holoscan::TimePoint time = context.activation_time();
      holoscan::expected<void, holoscan::Error> emitted;
      switch (kind_) {
        case MvdrScenario::OverrideKind::kNone:
          break;
        case MvdrScenario::OverrideKind::kArrayGeometry: {
          const auto packed = hr::pack_array_geometry(override_.array_geometry);
          const std::array<int64_t, 2> shape{override_.array_geometry.num_elements() + 1, 3};
          emitted = emit_wire_doubles(array_geometry, hr::to_wire(override_.array_geometry), packed,
                                      shape, time);
          break;
        }
        case MvdrScenario::OverrideKind::kAzAngles: {
          emitted = emit_host_vector(az_coordinates, override_.az_coordinates, time);
          break;
        }
        case MvdrScenario::OverrideKind::kElAngles: {
          emitted = emit_host_vector(el_coordinates, override_.el_coordinates, time);
          break;
        }
        case MvdrScenario::OverrideKind::kSpatialSmoothing: {
          emitted = spatial_smoothing.emit(smoothing_wire(override_.spatial_smoothing),
                                           holoscan::EmitOptions{.capture_time = time});
          break;
        }
        case MvdrScenario::OverrideKind::kDiagonalLoading: {
          emitted = diagonal_loading.emit(override_.diagonal_loading,
                                          holoscan::EmitOptions{.capture_time = time});
          break;
        }
        case MvdrScenario::OverrideKind::kMaxSources: {
          emitted =
              max_sources.emit(override_.max_sources, holoscan::EmitOptions{.capture_time = time});
          break;
        }
      }
      if (!emitted)
        return emitted;
    }
    ++frame_index_;
    return perf::detail::emit_device_frame(snapshot, context, frame, mode);
  }

  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Output<hr::WireArrayGeometry> array_geometry{"array_geometry"};
  holoscan::Output<holoscan::Tensor> az_coordinates{"az_coordinates"};
  holoscan::Output<holoscan::Tensor> el_coordinates{"el_coordinates"};
  holoscan::Output<hr::SpatialSmoothingOverride> spatial_smoothing{"spatial_smoothing"};
  holoscan::Output<double> diagonal_loading{"diagonal_loading"};
  holoscan::Output<int64_t> max_sources{"max_sources"};

 private:
  int64_t override_at_{-1};
  hr::MVDRParams initial_;
  hr::MVDRParams override_;
  MvdrScenario::OverrideKind kind_{MvdrScenario::OverrideKind::kNone};
  int64_t frame_index_{0};
};

perf::TimingCollector run_mvdr_graph(const MvdrScenario& scen, const perf::RunPlan& plan) {
  auto snap =
      perf::make_iq_cube_device<PerfPrecision>({scen.detections, scen.channels, scen.snapshots});
  constexpr double kInitialLoading = -1.0;
  constexpr double kOverrideLoading = 1e-2;
  constexpr int64_t kInitialMaxSources = 4;
  constexpr int64_t kOverrideMaxSources = 3;
  hr::MVDRParams initial;
  initial.array_geometry = make_ula_geometry(scen.channels, 0.5, 1.0);
  initial.angle_convention.preset = hr::AngleConventionPreset::kRadarBoresight;
  initial.az_coordinates = linspace(-0.5, 0.5, scen.num_az);
  initial.el_coordinates = linspace(-0.3, 0.3, scen.num_el);
  initial.diagonal_loading = scen.static_loading;
  initial.max_sources = scen.static_max_sources;
  hr::MVDRParams override = initial;
  switch (scen.override_kind) {
    case MvdrScenario::OverrideKind::kNone:
      break;
    case MvdrScenario::OverrideKind::kArrayGeometry:
      override.array_geometry = make_ula_geometry(scen.channels, 0.45, 1.0);
      break;
    case MvdrScenario::OverrideKind::kAzAngles:
      override.az_coordinates = linspace(-0.45, 0.55, scen.num_az);
      break;
    case MvdrScenario::OverrideKind::kElAngles:
      override.el_coordinates = linspace(-0.25, 0.35, scen.num_el);
      break;
    case MvdrScenario::OverrideKind::kSpatialSmoothing:
      override.spatial_smoothing = {2};
      break;
    case MvdrScenario::OverrideKind::kDiagonalLoading:
      override.diagonal_loading = kOverrideLoading;
      break;
    case MvdrScenario::OverrideKind::kMaxSources:
      override.max_sources = kOverrideMaxSources;
      break;
  }

  hr::MaxTensorExtents extents;
  extents.max_items = scen.detections;
  extents.max_az = scen.num_az;
  extents.max_el = scen.num_el;
  extents.max_sources = kInitialMaxSources;
  extents.max_channels = scen.channels;
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::now();
  perf::PerfFrameBank::reset(std::move(snap), plan.total_frames, &collector);

  holoscan::Graph graph{"mvdr-perf"};
  const auto source = graph.op<PerfMVDROverrideSourceOp>("src", plan.override_at, initial, override,
                                                         scen.override_kind);
  auto op = graph.op<hr::MVDREstimatorOp<PerfPrecision>>("mvdr", extents);
  op.set_params(std::move(initial));
  const auto sink = graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
  const auto spectrum = graph.op<perf::PerfNullSinkOp>("spectrum_drain");
  const auto doa = graph.op<perf::PerfNullSinkOp>("doa_drain");
  const auto count = graph.op<perf::PerfNullSinkOp>("count_drain");
  const auto eigen = graph.op<perf::PerfNullSinkOp>("eig_drain");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};

  graph.add_flow(source->snapshot, op->snapshot, q);
  if (scen.override_kind == MvdrScenario::OverrideKind::kArrayGeometry)
    graph.add_flow(source->array_geometry, op->array_geometry, q);
  if (scen.override_kind == MvdrScenario::OverrideKind::kAzAngles)
    graph.add_flow(source->az_coordinates, op->az_coordinates, q);
  if (scen.override_kind == MvdrScenario::OverrideKind::kElAngles)
    graph.add_flow(source->el_coordinates, op->el_coordinates, q);
  if (scen.override_kind == MvdrScenario::OverrideKind::kSpatialSmoothing)
    graph.add_flow(source->spatial_smoothing, op->spatial_smoothing, q);
  if (scen.override_kind == MvdrScenario::OverrideKind::kDiagonalLoading)
    graph.add_flow(source->diagonal_loading, op->diagonal_loading, q);
  if (scen.override_kind == MvdrScenario::OverrideKind::kMaxSources)
    graph.add_flow(source->max_sources, op->max_sources, q);
  graph.add_flow(op->aoa_estimates, sink->in, q);
  graph.add_flow(op->angle_spectrum, spectrum->in, q);
  graph.add_flow(op->doa_multi, doa->in, q);
  graph.add_flow(op->source_count, count->in, q);
  graph.add_flow(op->eigenvalues, eigen->in, q);

  holoscan::CompileOptions options;
  perf::bind_tensor_out(options, "src", "snapshot");
  perf::bind_tensor_out(options, "mvdr", "aoa_estimates");
  perf::bind_tensor_out(options, "mvdr", "angle_spectrum");
  perf::bind_tensor_out(options, "mvdr", "doa_multi");
  perf::bind_tensor_out(options, "mvdr", "source_count");
  perf::bind_tensor_out(options, "mvdr", "eigenvalues");
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan compiled = holoscan::compile(graph, std::move(options));
  if (!compiled.ok()) {
    throw std::runtime_error("mvdr-perf compile failed: " + std::string(compiled.json()));
  }
  holoscan::RunSession session = holoscan::run_async(compiled);
  perf::detail::wait_for_frames(session, plan.total_frames);
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string message = "mvdr-perf run diagnostics:";
    for (const auto& diagnostic : session.diagnostics()) {
      message += " ";
      message += std::string(diagnostic.token);
    }
    throw std::runtime_error(message);
  }
  if (collector.ends.size() < static_cast<std::size_t>(plan.total_frames)) {
    throw std::runtime_error(
        perf::detail::stall_message("mvdr-perf", collector.ends.size(), plan.total_frames));
  }
  return collector;
}

void fill_throughput(perf::ResultRow* row, const MvdrScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  row->cells_per_second = n / sec;
  const double beams = static_cast<double>(scen.num_az) * static_cast<double>(scen.num_el);
  row->slices_per_second = n * beams / sec;
}

perf::ResultRow base_row(const MvdrScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "MVDREstimatorOp";
  row.variant = std::string("capon_live_") + override_name(scen.override_kind);
  row.live_port = override_name(scen.override_kind);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.detections, scen.channels, scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.notes = std::string("live_override=") + override_name(scen.override_kind);
  return row;
}

MvdrScenario make_scenario(const SizeSpec& s,
                           MvdrScenario::OverrideKind kind = MvdrScenario::OverrideKind::kNone) {
  MvdrScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "mvdr_" << s.id << "_c" << s.channels << "_n" << s.detections << "_s" << s.snapshots << "_g"
     << s.num_az << "x" << s.num_el << "_live_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

void append_override_scenarios(std::vector<MvdrScenario>* result, const SizeSpec& size) {
  for (const auto kind :
       {MvdrScenario::OverrideKind::kArrayGeometry, MvdrScenario::OverrideKind::kAzAngles,
        MvdrScenario::OverrideKind::kElAngles, MvdrScenario::OverrideKind::kSpatialSmoothing,
        MvdrScenario::OverrideKind::kDiagonalLoading, MvdrScenario::OverrideKind::kMaxSources}) {
    result->push_back(make_scenario(size, kind));
  }
}

struct MvdrTraits : perf::PerfDriverDefaults {
  using Scenario = MvdrScenario;

  static const char* tool_name() { return "perf_mvdr_estimator"; }
  static const char* tool_description() {
    return "MVDREstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() { return "Default size sweep with all six live overrides"; }
  static const char* sizes_subset_help() { return "Size sweep across an explicit subset"; }
  static const char* full_help() { return "Full size ladder with all six live overrides"; }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "mvdr_estimator.yaml",
        "mvdr_estimator");
  }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 5)
        throw std::invalid_argument("mvdr_estimator profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape, shape[0] * shape[1] * shape[2],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }
  static std::vector<Scenario> build_profile_matrix(const std::string& scope) {
    const auto profiles = suite();
    std::set<std::string> selected_ids;
    for (const auto& c : profiles.select(scope))
      selected_ids.insert(c.id);
    std::vector<Scenario> result;
    for (const auto& c : profiles.cases) {
      size_t i = 0;
      for (; i < perf::kProfiles.size(); ++i)
        if (c.profile == perf::kProfiles[i])
          break;
      const auto& shape = profiles.shapes.at(i);
      if (shape.size() != 5 || shape[0] < 4 || shape[1] < 1 || shape[2] < 2 || shape[3] < 1 ||
          shape[4] < 1)
        throw std::invalid_argument("mvdr_estimator invalid shape");
      const SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], shape[4], c.profile};
      auto scenario = make_scenario(size);
      scenario.id = c.id;
      perf::require_keys(c.parameters, {"diagonal_loading", "max_sources"}, "parameters");
      scenario.static_loading = c.parameters["diagonal_loading"].as<double>();
      if (c.parameters["max_sources"])
        scenario.static_max_sources = c.parameters["max_sources"].as<int64_t>();
      if (scenario.static_max_sources < 1 || scenario.static_max_sources > 4 ||
          scenario.static_loading < -1.0)
        throw std::invalid_argument("MVDR invalid loading");
      if (c.override_value) {
        const auto port = (*c.override_value)["port"].as<std::string>();
        const auto value = (*c.override_value)["value"].as<std::string>();
        if (port == "array_geometry" && value == "spacing_045")
          scenario.override_kind = MvdrScenario::OverrideKind::kArrayGeometry;
        else if (port == "az_coordinates" && value == "shift_005")
          scenario.override_kind = MvdrScenario::OverrideKind::kAzAngles;
        else if (port == "el_coordinates" && value == "shift_005")
          scenario.override_kind = MvdrScenario::OverrideKind::kElAngles;
        else if (port == "spatial_smoothing" && value == "two")
          scenario.override_kind = MvdrScenario::OverrideKind::kSpatialSmoothing;
        else if (port == "diagonal_loading" && value == "0.01")
          scenario.override_kind = MvdrScenario::OverrideKind::kDiagonalLoading;
        else if (port == "max_sources" && value == "3")
          scenario.override_kind = MvdrScenario::OverrideKind::kMaxSources;
        else
          throw std::invalid_argument("mvdr_estimator invalid live override");
      }
      if (selected_ids.contains(c.id))
        result.push_back(std::move(scenario));
    }
    return result;
  }

  static void run_scenario(const MvdrScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(
        rs, /*has_override=*/scen.override_kind != MvdrScenario::OverrideKind::kNone);
    const auto samples =
        perf::collect_scenario_samples(rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_mvdr_graph(scen, p);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes = "override_ms isolates inline " + std::string(override_name(scen.override_kind)) +
                "; cells/s is detections/s; Capon scan dominates per tick";
    if (row.live_port.empty())
      row.notes = "default parameters; no live override";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs"), MvdrScenario::OverrideKind::kMaxSources),
                   warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<MvdrTraits>(argc, argv);
}
