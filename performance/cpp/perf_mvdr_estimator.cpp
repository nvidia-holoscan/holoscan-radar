// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for MVDREstimatorOp — adaptive (Capon)
// sparse-path DOA. Per item it forms a snapshot covariance, diagonal-loads
// and inverts it (batched), then scans the (az, el) grid for the Capon
// spectrum. Sweeps detection count, channel count, snapshot count, and
// scan-grid size. Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.
//
// [EA HSDK5 ONLY] Graph-based rewrite of the HSDK 4.x Application harness.
// Measures each live override independently through the production Exact
// contract.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
  OverrideKind override_kind{OverrideKind::kMaxSources};
};

const char* override_name(MvdrScenario::OverrideKind kind) {
  switch (kind) {
    case MvdrScenario::OverrideKind::kArrayGeometry:
      return "array_geometry";
    case MvdrScenario::OverrideKind::kAzAngles:
      return "az_angles";
    case MvdrScenario::OverrideKind::kElAngles:
      return "el_angles";
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
      {"lg", 32, 512, 64, 81, 41,
       "wide aperture, many looks"},
      {"xl", 32, 1024, 96, 101, 51,
       "upper characterization bound"}};
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
    std::cerr << "internal error: unknown size id '"
              << id << "'\n";
    std::abort();
  }
  return *s;
}

std::vector<double> linspace(double start, double stop,
                             int64_t n) {
  std::vector<double> out(static_cast<size_t>(n));
  if (n == 1) {
    out[0] = start;
    return out;
  }
  const double step =
      (stop - start) / static_cast<double>(n - 1);
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] =
        start + step * static_cast<double>(i);
  }
  return out;
}

hr::ArrayGeometry make_ula_geometry(int64_t num_elements,
                                    double spacing,
                                    double wavelength) {
  hr::ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = hr::ArrayGeometryType::kUla;
  geom.element_positions.reserve(
      static_cast<size_t>(num_elements * 3));
  for (int64_t m = 0; m < num_elements; ++m) {
    geom.element_positions.push_back(0.0);
    geom.element_positions.push_back(
        static_cast<double>(m) * spacing);
    geom.element_positions.push_back(0.0);
  }
  return geom;
}

holoscan::expected<void, holoscan::Error> emit_host_vector(
    holoscan::Output<holoscan::Tensor>& output,
    std::span<const double> values, holoscan::TimePoint capture_time) {
  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(values.size())};
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
  return output.emit(
      std::move(*loan),
      holoscan::EmitOptions{.capture_time = capture_time});
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

class PerfMVDROverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfMVDROverrideSourceOp(int64_t override_at, hr::MVDRParams initial,
                           hr::MVDRParams override)
      : override_at_(override_at),
        initial_(std::move(initial)),
        override_(std::move(override)) {}

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
    spec.output(snapshot)
        .produces_tensor(hr::pooled_tensor_output(
            kDevice, 8U, perf::PerfFrameBank::span_a()));
    spec.output(array_geometry)
        .max_emits_per_compute(1U)
        .produces_tensor(
            hr::pooled_tensor_output(kHostMatrix, 2U, kHostPayloadSpan));
    spec.output(az_angles)
        .max_emits_per_compute(1U)
        .produces_tensor(
            hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(el_angles)
        .max_emits_per_compute(1U)
        .produces_tensor(
            hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
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
    emitted = emit_host_vector(az_angles, values.az_angles, time);
    if (!emitted) {
      return emitted;
    }
    emitted = emit_host_vector(el_angles, values.el_angles, time);
    if (!emitted) {
      return emitted;
    }
    emitted = spatial_smoothing.emit(
        smoothing_wire(values.spatial_smoothing),
        holoscan::EmitOptions{.capture_time = time});
    if (!emitted) {
      return emitted;
    }
    emitted = diagonal_loading.emit(
        values.diagonal_loading,
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
  holoscan::Output<holoscan::Tensor> az_angles{"az_angles"};
  holoscan::Output<holoscan::Tensor> el_angles{"el_angles"};
  holoscan::Output<hr::SpatialSmoothingOverride>
      spatial_smoothing{"spatial_smoothing"};
  holoscan::Output<double> diagonal_loading{"diagonal_loading"};
  holoscan::Output<int64_t> max_sources{"max_sources"};

 private:
  int64_t override_at_{-1};
  hr::MVDRParams initial_;
  hr::MVDRParams override_;
  int64_t frame_index_{0};
};

perf::TimingCollector run_mvdr_graph(
    const MvdrScenario& scen, const perf::RunPlan& plan) {
  auto snap = perf::make_iq_cube_device<PerfPrecision>(
      {scen.detections, scen.channels, scen.snapshots});
  constexpr double kInitialLoading = -1.0;
  constexpr double kOverrideLoading = 1e-2;
  constexpr int64_t kInitialMaxSources = 4;
  constexpr int64_t kOverrideMaxSources = 3;
  hr::MVDRParams initial;
  initial.array_geometry =
      make_ula_geometry(scen.channels, 0.5, 1.0);
  initial.angle_convention.preset =
      hr::AngleConventionPreset::kRadarBoresight;
  initial.az_angles = linspace(-0.5, 0.5, scen.num_az);
  initial.el_angles = linspace(-0.3, 0.3, scen.num_el);
  initial.diagonal_loading = kInitialLoading;
  initial.max_sources = kInitialMaxSources;
  hr::MVDRParams override = initial;
  switch (scen.override_kind) {
    case MvdrScenario::OverrideKind::kArrayGeometry:
      override.array_geometry =
          make_ula_geometry(scen.channels, 0.45, 1.0);
      break;
    case MvdrScenario::OverrideKind::kAzAngles:
      override.az_angles = linspace(-0.45, 0.55, scen.num_az);
      break;
    case MvdrScenario::OverrideKind::kElAngles:
      override.el_angles = linspace(-0.25, 0.35, scen.num_el);
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
  perf::PerfFrameBank::reset(
      std::move(snap), plan.total_frames, &collector);

  holoscan::Graph graph{"mvdr-perf"};
  const auto source = graph.op<PerfMVDROverrideSourceOp>(
      "src", plan.override_at, initial, override);
  auto op =
      graph.op<hr::MVDREstimatorOp<PerfPrecision>>("mvdr", extents);
  op.set_params(std::move(initial));
  const auto sink =
      graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
  const auto spectrum =
      graph.op<perf::PerfNullSinkOp>("spectrum_drain");
  const auto doa = graph.op<perf::PerfNullSinkOp>("doa_drain");
  const auto count = graph.op<perf::PerfNullSinkOp>("count_drain");
  const auto eigen = graph.op<perf::PerfNullSinkOp>("eig_drain");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};

  graph.add_flow(source->snapshot, op->snapshot, q);
  graph.add_flow(source->array_geometry, op->array_geometry, q);
  graph.add_flow(source->az_angles, op->az_angles, q);
  graph.add_flow(source->el_angles, op->el_angles, q);
  graph.add_flow(source->spatial_smoothing, op->spatial_smoothing, q);
  graph.add_flow(source->diagonal_loading, op->diagonal_loading, q);
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
  graph.set_default_clock(
      graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan compiled =
      holoscan::compile(graph, std::move(options));
  if (!compiled.ok()) {
    throw std::runtime_error(
        "mvdr-perf compile failed: " + std::string(compiled.json()));
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
  if (collector.ends.size() <
      static_cast<std::size_t>(plan.total_frames)) {
    throw std::runtime_error(perf::detail::stall_message(
        "mvdr-perf", collector.ends.size(), plan.total_frames));
  }
  return collector;
}

void fill_throughput(perf::ResultRow* row,
                     const MvdrScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  row->cells_per_second = n / sec;
  const double beams =
      static_cast<double>(scen.num_az) *
      static_cast<double>(scen.num_el);
  row->slices_per_second = n * beams / sec;
}

perf::ResultRow base_row(const MvdrScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "MVDREstimatorOp";
  row.variant =
      std::string("capon_live_") +
      override_name(scen.override_kind);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.detections, scen.channels,
                     scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.notes = std::string("live_override=") +
              override_name(scen.override_kind);
  return row;
}

MvdrScenario make_scenario(
    const SizeSpec& s, MvdrScenario::OverrideKind kind) {
  MvdrScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "mvdr_" << s.id << "_c" << s.channels
     << "_n" << s.detections << "_s" << s.snapshots
     << "_g" << s.num_az << "x" << s.num_el
     << "_live_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

void append_override_scenarios(std::vector<MvdrScenario>* result,
                               const SizeSpec& size) {
  for (const auto kind : {
           MvdrScenario::OverrideKind::kArrayGeometry,
           MvdrScenario::OverrideKind::kAzAngles,
           MvdrScenario::OverrideKind::kElAngles,
           MvdrScenario::OverrideKind::kSpatialSmoothing,
           MvdrScenario::OverrideKind::kDiagonalLoading,
           MvdrScenario::OverrideKind::kMaxSources}) {
    result->push_back(make_scenario(size, kind));
  }
}

struct MvdrTraits : perf::PerfDriverDefaults {
  using Scenario = MvdrScenario;

  static const char* tool_name() {
    return "perf_mvdr_estimator";
  }
  static const char* tool_description() {
    return "MVDREstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Default size sweep with all six live overrides";
  }
  static const char* sizes_subset_help() {
    return "Size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full size ladder with all six live overrides";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_az, s.num_el};
      def.cells = s.num_az * s.num_el;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<MvdrScenario> build_default_matrix() {
    std::vector<MvdrScenario> out;
    for (const char* sid : {"xs", "sm", "md"}) {
      append_override_scenarios(
          &out, must_find_size(sid));
    }
    return out;
  }

  static std::vector<MvdrScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<MvdrScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '"
                  << sid << "', skipping\n";
        continue;
      }
      append_override_scenarios(&out, *s);
    }
    return out;
  }

  static std::vector<MvdrScenario> build_full_matrix() {
    std::vector<MvdrScenario> out;
    for (const auto& s : size_ladder()) {
      append_override_scenarios(&out, s);
    }
    return out;
  }

  static void run_scenario(const MvdrScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_mvdr_graph(scen, p);
        });

    perf::append_setup_row(
        base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "override_ms isolates inline " +
        std::string(override_name(scen.override_kind)) +
        "; cells/s is detections/s; Capon scan dominates per tick";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(
          make_scenario(
              must_find_size("xs"),
              MvdrScenario::OverrideKind::kMaxSources),
          warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<MvdrTraits>(
      argc, argv);
}
