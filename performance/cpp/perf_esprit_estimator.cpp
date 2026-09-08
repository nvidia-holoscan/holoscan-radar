// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ESPRITEstimatorOp. Measures the shared
// covariance/Hermitian-eig/EFT front-end plus the device LS rotation solve.
// Measurement only: no timing assertions.
//
// [EA HSDK5 ONLY] Graph-based rewrite of the HSDK 4.x Application harness.
// Measures each live override independently through the production Exact
// contract. subarray_spec is static.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
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

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
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
  std::string role;
};

struct Scenario {
  enum class OverrideKind {
    kArrayGeometry,
    kSpatialSmoothing,
    kNumSources,
    kSourceCountPfa,
    kMaxSources,
  };

  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t snapshots{0};
  bool is_ura{false};
  int64_t ny{0};
  int64_t nz{0};
  OverrideKind override_kind{OverrideKind::kMaxSources};
};

const char* override_name(Scenario::OverrideKind kind) {
  switch (kind) {
    case Scenario::OverrideKind::kArrayGeometry:
      return "array_geometry";
    case Scenario::OverrideKind::kSpatialSmoothing:
      return "spatial_smoothing";
    case Scenario::OverrideKind::kNumSources:
      return "num_sources";
    case Scenario::OverrideKind::kSourceCountPfa:
      return "source_count_pfa";
    case Scenario::OverrideKind::kMaxSources:
      return "max_sources";
  }
  return "unknown";
}
const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> sizes = {
      {"xs", 4, 64, 16, "small array; fast smoke"},
      {"sm", 8, 128, 32, "compact SIMO array"},
      {"md", 16, 256, 48, "common dev size"},
      {"lg", 32, 512, 64, "wide aperture, many looks"},
      {"xl", 32, 1024, 96, "upper characterization bound"}};
  return sizes;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& size : size_ladder()) {
    if (size.id == id) {
      return &size;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* size = find_size(id);
  if (size == nullptr) {
    std::cerr << "unknown size '" << id << "'\n";
    std::abort();
  }
  return *size;
}

hr::ArrayGeometry make_ula(int64_t channels, double spacing) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = hr::ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < channels; ++channel) {
    geometry.element_positions.insert(
        geometry.element_positions.end(),
        {0.0, spacing * static_cast<double>(channel), 0.0});
  }
  return geometry;
}

hr::ArrayGeometry make_ura(int64_t ny, int64_t nz, double spacing) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = hr::ArrayGeometryType::kUra;
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      geometry.element_positions.insert(
          geometry.element_positions.end(),
          {0.0, spacing * static_cast<double>(iy),
           spacing * static_cast<double>(iz)});
    }
  }
  return geometry;
}

hr::ArrayGeometry scenario_geometry(const Scenario& s, double spacing) {
  if (s.is_ura) {
    return make_ura(s.ny, s.nz, spacing);
  }
  return make_ula(s.channels, spacing);
}

Scenario make_scenario(const SizeSpec& size, Scenario::OverrideKind kind) {
  Scenario scenario;
  scenario.size_id = size.id;
  scenario.channels = size.channels;
  scenario.detections = size.detections;
  scenario.snapshots = size.snapshots;
  scenario.override_kind = kind;
  std::ostringstream id;
  id << "esprit_" << size.id << "_c" << size.channels << "_n"
     << size.detections << "_s" << size.snapshots << "_live_"
     << override_name(kind);
  scenario.id = id.str();
  return scenario;
}

Scenario make_ura_scenario(const char* size_id, int64_t ny, int64_t nz,
                           int64_t detections, int64_t snapshots,
                           Scenario::OverrideKind kind) {
  Scenario scenario;
  scenario.size_id = size_id;
  scenario.is_ura = true;
  scenario.ny = ny;
  scenario.nz = nz;
  scenario.channels = ny * nz;
  scenario.detections = detections;
  scenario.snapshots = snapshots;
  scenario.override_kind = kind;
  std::ostringstream id;
  id << "esprit_ura_" << ny << "x" << nz << "_n" << detections
     << "_s" << snapshots << "_live_" << override_name(kind);
  scenario.id = id.str();
  return scenario;
}

template <typename Wire>
holoscan::expected<void, holoscan::Error> emit_wire_doubles(
    holoscan::Output<Wire>& output, Wire wire,
    std::span<const double> values, std::span<const int64_t> shape,
    holoscan::TimePoint capture_time) {
  constexpr DLDataType dtype{kDLFloat, 64U, 1U};
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit(
      [values](const holoscan::TensorOutputWriteGuard& writer) {
        std::copy(values.begin(), values.end(), writer.data_as<double>());
      });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return output.emit_tensor(
      std::move(*loan), std::move(wire),
      holoscan::EmitOptions{.capture_time = capture_time});
}

hr::SpatialSmoothingOverride smoothing_wire(
    const std::vector<int64_t>& smoothing) {
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

class PerfESPRITOverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfESPRITOverrideSourceOp(int64_t override_at,
                            hr::ESPRITParams initial,
                            hr::ESPRITParams override)
      : override_at_(override_at),
        initial_(std::move(initial)),
        override_(std::move(override)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    constexpr holoscan::TensorRepresentation kHostMatrix{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 2U,
    };
    constexpr std::size_t kHostPayloadSpan = 4096U * sizeof(double);
    spec.output(snapshot)
        .produces_tensor(hr::pooled_tensor_output(
            kDevice, 8U, perf::PerfFrameBank::span_a()));
    spec.output(array_geometry)
        .max_emits_per_compute(1U)
        .produces_tensor(
            hr::pooled_tensor_output(kHostMatrix, 2U, kHostPayloadSpan));
    spec.output(spatial_smoothing).max_emits_per_compute(1U);
    spec.output(num_sources).max_emits_per_compute(1U);
    spec.output(source_count_pfa).max_emits_per_compute(1U);
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
      if (perf::PerfFrameBank::frames_remaining <= 0 ||
          perf::PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --perf::PerfFrameBank::frames_remaining;
      frame = perf::PerfFrameBank::input_frame;
      mode = perf::PerfFrameBank::publish_mode;
    }

    const auto& values =
        frame_index_ < override_at_ ? initial_ : override_;
    const holoscan::TimePoint time = context.activation_time();
    const auto packed = hr::pack_array_geometry(values.array_geometry);
    auto geometry_wire = hr::to_wire(values.array_geometry);
    const std::array<int64_t, 2> geometry_shape{
        values.array_geometry.num_elements() + 1, 3};
    auto emitted = emit_wire_doubles(
        array_geometry, std::move(geometry_wire), packed, geometry_shape,
        time);
    if (!emitted) {
      return emitted;
    }
    emitted = spatial_smoothing.emit(
        smoothing_wire(values.spatial_smoothing),
        holoscan::EmitOptions{.capture_time = time});
    if (!emitted) {
      return emitted;
    }
    emitted = num_sources.emit(
        values.num_sources,
        holoscan::EmitOptions{.capture_time = time});
    if (!emitted) {
      return emitted;
    }
    emitted = source_count_pfa.emit(
        values.source_count_pfa,
        holoscan::EmitOptions{.capture_time = time});
    if (!emitted) {
      return emitted;
    }
    emitted = max_sources.emit(
        values.max_sources,
        holoscan::EmitOptions{.capture_time = time});
    if (!emitted) {
      return emitted;
    }
    {
      const std::lock_guard lock(perf::PerfFrameBank::mutex);
      perf::detail::record_start_locked();
    }
    ++frame_index_;
    return perf::detail::emit_device_frame(snapshot, context, frame, mode);
  }

  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Output<hr::WireArrayGeometry>
      array_geometry{"array_geometry"};
  holoscan::Output<hr::SpatialSmoothingOverride>
      spatial_smoothing{"spatial_smoothing"};
  holoscan::Output<int64_t> num_sources{"num_sources"};
  holoscan::Output<double> source_count_pfa{"source_count_pfa"};
  holoscan::Output<int64_t> max_sources{"max_sources"};

 private:
  int64_t override_at_{-1};
  hr::ESPRITParams initial_;
  hr::ESPRITParams override_;
  int64_t frame_index_{0};
};

perf::TimingCollector run_esprit_graph(const Scenario& scen,
                                       const perf::RunPlan& plan) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(
      {scen.detections, scen.channels, scen.snapshots});
  constexpr int64_t kInitialMaxSources = 4;
  constexpr int64_t kOverrideMaxSources = 3;
  hr::ESPRITParams initial;
  initial.array_geometry = scenario_geometry(scen, 0.5);
  initial.angle_convention.preset =
      hr::AngleConventionPreset::kRadarBoresight;
  initial.num_sources =
      scen.override_kind == Scenario::OverrideKind::kSourceCountPfa
          ? -1
          : 1;
  initial.source_count_pfa = 1e-4;
  initial.max_sources = kInitialMaxSources;
  hr::ESPRITParams override = initial;
  switch (scen.override_kind) {
    case Scenario::OverrideKind::kArrayGeometry:
      override.array_geometry = scenario_geometry(scen, 0.45);
      break;
    case Scenario::OverrideKind::kSpatialSmoothing:
      override.spatial_smoothing =
          scen.is_ura ? std::vector<int64_t>{2, 2}
                      : std::vector<int64_t>{2};
      break;
    case Scenario::OverrideKind::kNumSources:
      override.num_sources = -1;
      break;
    case Scenario::OverrideKind::kSourceCountPfa:
      override.source_count_pfa = 1e-2;
      break;
    case Scenario::OverrideKind::kMaxSources:
      override.max_sources = kOverrideMaxSources;
      break;
  }

  hr::MaxTensorExtents extents;
  extents.max_items = scen.detections;
  extents.max_sources = kInitialMaxSources;
  extents.max_channels = scen.channels;
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::now();
  perf::PerfFrameBank::reset(
      std::move(cube), plan.total_frames, &collector);

  holoscan::Graph graph{"esprit-perf"};
  const auto source = graph.op<PerfESPRITOverrideSourceOp>(
      "src", plan.override_at, initial, override);
  auto op =
      graph.op<hr::ESPRITEstimatorOp<PerfPrecision>>("esprit", extents);
  op.set_params(std::move(initial));
  const auto aoa_sink =
      graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
  const auto doa_drain =
      graph.op<perf::PerfNullSinkOp>("doa_drain");
  const auto sc_drain =
      graph.op<perf::PerfNullSinkOp>("sc_drain");
  const auto eig_drain =
      graph.op<perf::PerfNullSinkOp>("eig_drain");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};

  graph.add_flow(source->snapshot, op->snapshot, q);
  graph.add_flow(source->array_geometry, op->array_geometry, q);
  graph.add_flow(source->spatial_smoothing, op->spatial_smoothing, q);
  graph.add_flow(source->num_sources, op->num_sources, q);
  graph.add_flow(source->source_count_pfa, op->source_count_pfa, q);
  graph.add_flow(source->max_sources, op->max_sources, q);
  graph.add_flow(op->aoa_estimates, aoa_sink->in, q);
  graph.add_flow(op->doa_multi, doa_drain->in, q);
  graph.add_flow(op->source_count, sc_drain->in, q);
  graph.add_flow(op->eigenvalues, eig_drain->in, q);

  holoscan::CompileOptions options;
  perf::bind_tensor_out(options, "src", "snapshot");
  perf::bind_tensor_out(options, "esprit", "aoa_estimates");
  perf::bind_tensor_out(options, "esprit", "doa_multi");
  perf::bind_tensor_out(options, "esprit", "source_count");
  perf::bind_tensor_out(options, "esprit", "eigenvalues");
  graph.set_default_clock(
      graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan compiled =
      holoscan::compile(graph, std::move(options));
  if (!compiled.ok()) {
    throw std::runtime_error(
        "esprit-perf compile failed: " + std::string(compiled.json()));
  }
  holoscan::RunSession session = holoscan::run_async(compiled);
  perf::detail::wait_for_frames(session, plan.total_frames);
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string message = "esprit-perf run diagnostics:";
    for (const auto& diagnostic : session.diagnostics()) {
      message += " ";
      message += std::string(diagnostic.token);
    }
    throw std::runtime_error(message);
  }
  if (collector.ends.size() <
      static_cast<std::size_t>(plan.total_frames)) {
    throw std::runtime_error(perf::detail::stall_message(
        "esprit-perf", collector.ends.size(), plan.total_frames));
  }
  return collector;
}

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "ESPRITEstimatorOp";
  row.variant =
      std::string("esprit_ls_live_") +
      override_name(scenario.override_kind);
  row.size_id = scenario.size_id;
  row.input_shape = {scenario.detections, scenario.channels,
                     scenario.snapshots};
  row.out_batch = scenario.detections;
  row.k = scenario.channels;
  row.notes = std::string("live_override=") +
              override_name(scenario.override_kind) +
              "; subarray_spec is static";
  return row;
}

void append_override_scenarios(std::vector<Scenario>* result,
                               const SizeSpec& size) {
  for (const auto kind : {
           Scenario::OverrideKind::kArrayGeometry,
           Scenario::OverrideKind::kSpatialSmoothing,
           Scenario::OverrideKind::kNumSources,
           Scenario::OverrideKind::kSourceCountPfa,
           Scenario::OverrideKind::kMaxSources}) {
    result->push_back(make_scenario(size, kind));
  }
}

void append_ura_override_scenarios(std::vector<Scenario>* result,
                                   const char* size_id, int64_t ny,
                                   int64_t nz, int64_t detections,
                                   int64_t snapshots) {
  for (const auto kind : {
           Scenario::OverrideKind::kArrayGeometry,
           Scenario::OverrideKind::kSpatialSmoothing,
           Scenario::OverrideKind::kNumSources,
           Scenario::OverrideKind::kSourceCountPfa,
           Scenario::OverrideKind::kMaxSources}) {
    result->push_back(make_ura_scenario(
        size_id, ny, nz, detections, snapshots, kind));
  }
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;

  static const char* tool_name() { return "perf_esprit_estimator"; }
  static const char* tool_description() {
    return "ESPRITEstimatorOp performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Default ESPRIT size sweep with all five live overrides";
  }
  static const char* sizes_subset_help() {
    return "ESPRIT sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full ESPRIT size ladder with all five live overrides";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    for (const auto& size : size_ladder()) {
      perf::SizeDef definition;
      definition.id = size.id;
      definition.shape = {size.channels, size.detections,
                          size.snapshots};
      definition.cells = size.detections;
      definition.role = size.role;
      result.push_back(definition);
    }
    return result;
  }

  static std::vector<Scenario> build_default_matrix() {
    std::vector<Scenario> result;
    for (const char* id : {"xs", "sm", "md"}) {
      append_override_scenarios(&result, must_find_size(id));
    }
    append_ura_override_scenarios(
        &result, "ura_sm", 6, 6, 128, 96);
    return result;
  }

  static std::vector<Scenario> build_size_sweep(
      const std::vector<std::string>& ids) {
    std::vector<Scenario> result;
    for (const auto& id : ids) {
      if (const SizeSpec* size = find_size(id)) {
        append_override_scenarios(&result, *size);
      } else {
        std::cerr << "warning: unknown size id '" << id
                  << "', skipping\n";
      }
    }
    return result;
  }

  static std::vector<Scenario> build_full_matrix() {
    std::vector<Scenario> result;
    for (const auto& size : size_ladder()) {
      append_override_scenarios(&result, size);
    }
    append_ura_override_scenarios(
        &result, "ura_sm", 6, 6, 128, 96);
    append_ura_override_scenarios(
        &result, "ura_lg", 8, 8, 256, 128);
    return result;
  }

  static void run_scenario(const Scenario& scenario,
                           const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const auto plan =
        perf::make_run_plan(settings, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        settings, plan, [&](int, const perf::RunPlan& run_plan) {
          return run_esprit_graph(scenario, run_plan);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    perf::ResultRow row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    if (row.latency.median_ms > 0.0) {
      row.cells_per_second =
          scenario.detections / (row.latency.median_ms / 1000.0);
    }
    row.notes =
        "override_ms isolates inline " +
        std::string(override_name(scenario.override_kind)) +
        "; steady num_sources is fixed at one except the "
        "source_count_pfa scenario, which uses auto; subarray_spec is static";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(
          make_scenario(must_find_size("xs"),
                        Scenario::OverrideKind::kMaxSources),
          settings, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
