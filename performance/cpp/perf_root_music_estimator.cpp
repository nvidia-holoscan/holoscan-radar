// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for RootMUSICEstimatorOp — subspace,
// gridless (polynomial rooting) sparse-path DOA on a ULA. Per item it
// forms a snapshot covariance, eigendecomposes it, resolves a model order,
// builds the noise-subspace projector, then roots the
// conjugate-reciprocal polynomial. Sweeps detection count, channel count,
// and snapshot count (no scan grid). Measurement tool only: no timing
// assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.
//
//  Graph-based rewrite of the HSDK 4.x Application harness.
// Measures each live override independently through the production Exact
// contract.
//
// num_sources is fixed (=1) across the steady matrix rather than auto:
// this keeps the model order — and hence the noise-subspace projector's
// cost — deterministic across repeats, isolating the
// covariance/eig/projector/root path from the (cheap,
// separately-kernelled) EFT source-count decision.
//
// Smoke test (parent wires in performance/cpp/CMakeLists.txt):
//   add_executable(perf_root_music_estimator
//       perf_root_music_estimator.cpp)
//   target_link_libraries(perf_root_music_estimator PRIVATE
//       radar_perf_utils holoscan::root_music_estimator_op)
//   add_test(NAME perf_root_music_estimator_smoke
//       COMMAND perf_root_music_estimator --sizes xs --warmup 1
//           --frames 2 --quiet
//           --output
//           ${CMAKE_CURRENT_BINARY_DIR}/
//               perf_root_music_estimator_smoke.json)
//   set_tests_properties(perf_root_music_estimator_smoke
//       PROPERTIES LABELS "performance")

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

#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
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

struct RootMusicScenario {
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
  std::optional<int64_t> fixed_model_order;
};

const char* override_name(RootMusicScenario::OverrideKind kind) {
  switch (kind) {
    case RootMusicScenario::OverrideKind::kArrayGeometry:
      return "array_geometry";
    case RootMusicScenario::OverrideKind::kSpatialSmoothing:
      return "spatial_smoothing";
    case RootMusicScenario::OverrideKind::kNumSources:
      return "num_sources";
    case RootMusicScenario::OverrideKind::kSourceCountPfa:
      return "source_count_pfa";
    case RootMusicScenario::OverrideKind::kMaxSources:
      return "max_sources";
  }
  return "unknown";
}

const std::vector<SizeSpec>& size_ladder() {
  // Same channel/detection/snapshot counts as MVDR; covariance + eig +
  // projector + rooting scratch is O(detections · channels²).
  static const std::vector<SizeSpec> ladder = {
      {"xs", 4, 64, 16, "small array; fast smoke"},
      {"sm", 8, 128, 32, "compact SIMO array"},
      {"md", 16, 256, 48, "common dev size"},
      {"lg", 32, 512, 64,
       "wide aperture, many looks"},
      {"xl", 32, 1024, 96,
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

hr::ArrayGeometry make_ura_geometry(int64_t ny, int64_t nz,
                                    double spacing,
                                    double wavelength) {
  hr::ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = hr::ArrayGeometryType::kUra;
  geom.element_positions.reserve(
      static_cast<size_t>(ny * nz * 3));
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      geom.element_positions.push_back(0.0);
      geom.element_positions.push_back(
          static_cast<double>(iy) * spacing);
      geom.element_positions.push_back(
          static_cast<double>(iz) * spacing);
    }
  }
  return geom;
}

hr::ArrayGeometry scenario_geometry(
    const RootMusicScenario& scen, double spacing,
    double wavelength) {
  if (scen.is_ura) {
    return make_ura_geometry(scen.ny, scen.nz, spacing,
                             wavelength);
  }
  return make_ula_geometry(scen.channels, spacing,
                           wavelength);
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

class PerfRootMUSICOverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfRootMUSICOverrideSourceOp(
      int64_t override_at, hr::RootMUSICParams initial,
      hr::RootMUSICParams override)
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
  hr::RootMUSICParams initial_;
  hr::RootMUSICParams override_;
  int64_t frame_index_{0};
};

perf::TimingCollector run_root_music_graph(
    const RootMusicScenario& scen, const perf::RunPlan& plan) {
  auto snap = perf::make_iq_cube_device<PerfPrecision>(
      {scen.detections, scen.channels, scen.snapshots});
  constexpr int64_t kInitialMaxSources = 4;
  constexpr int64_t kOverrideMaxSources = 3;
  hr::RootMUSICParams initial;
  initial.array_geometry = scenario_geometry(scen, 0.5, 1.0);
  initial.angle_convention.preset =
      hr::AngleConventionPreset::kRadarBoresight;
  initial.num_sources = scen.fixed_model_order.value_or(
      scen.override_kind ==
              RootMusicScenario::OverrideKind::kSourceCountPfa
          ? -1
          : 1);
  initial.source_count_pfa = 1e-4;
  initial.max_sources = kInitialMaxSources;
  hr::RootMUSICParams override = initial;
  switch (scen.override_kind) {
    case RootMusicScenario::OverrideKind::kArrayGeometry:
      override.array_geometry = scenario_geometry(scen, 0.45, 1.0);
      break;
    case RootMusicScenario::OverrideKind::kSpatialSmoothing:
      override.spatial_smoothing =
          scen.is_ura ? std::vector<int64_t>{2, 2}
                      : std::vector<int64_t>{2};
      break;
    case RootMusicScenario::OverrideKind::kNumSources:
      override.num_sources = -1;
      break;
    case RootMusicScenario::OverrideKind::kSourceCountPfa:
      override.source_count_pfa = 1e-2;
      break;
    case RootMusicScenario::OverrideKind::kMaxSources:
      override.max_sources = kOverrideMaxSources;
      break;
  }
  if (scen.fixed_model_order.has_value()) {
    override = initial;
  }

  hr::MaxTensorExtents extents;
  extents.max_items = scen.detections;
  extents.max_sources = kInitialMaxSources;
  extents.max_channels = scen.channels;
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::now();
  perf::PerfFrameBank::reset(
      std::move(snap), plan.total_frames, &collector);

  holoscan::Graph graph{"root-music-perf"};
  const auto source = graph.op<PerfRootMUSICOverrideSourceOp>(
      "src", plan.override_at, initial, override);
  auto op = graph.op<hr::RootMUSICEstimatorOp<PerfPrecision>>(
      "root_music", extents);
  op.set_params(std::move(initial));
  const auto sink =
      graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
  const auto doa = graph.op<perf::PerfNullSinkOp>("doa_drain");
  const auto count = graph.op<perf::PerfNullSinkOp>("count_drain");
  const auto eigen = graph.op<perf::PerfNullSinkOp>("eig_drain");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};

  graph.add_flow(source->snapshot, op->snapshot, q);
  graph.add_flow(source->array_geometry, op->array_geometry, q);
  graph.add_flow(source->spatial_smoothing, op->spatial_smoothing, q);
  graph.add_flow(source->num_sources, op->num_sources, q);
  graph.add_flow(source->source_count_pfa, op->source_count_pfa, q);
  graph.add_flow(source->max_sources, op->max_sources, q);
  graph.add_flow(op->aoa_estimates, sink->in, q);
  graph.add_flow(op->doa_multi, doa->in, q);
  graph.add_flow(op->source_count, count->in, q);
  graph.add_flow(op->eigenvalues, eigen->in, q);

  holoscan::CompileOptions options;
  perf::bind_tensor_out(options, "src", "snapshot");
  perf::bind_tensor_out(options, "root_music", "aoa_estimates");
  perf::bind_tensor_out(options, "root_music", "doa_multi");
  perf::bind_tensor_out(options, "root_music", "source_count");
  perf::bind_tensor_out(options, "root_music", "eigenvalues");
  graph.set_default_clock(
      graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan compiled =
      holoscan::compile(graph, std::move(options));
  if (!compiled.ok()) {
    throw std::runtime_error(
        "root-music-perf compile failed: " +
        std::string(compiled.json()));
  }
  holoscan::RunSession session = holoscan::run_async(compiled);
  perf::detail::wait_for_frames(session, plan.total_frames);
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string message = "root-music-perf run diagnostics:";
    for (const auto& diagnostic : session.diagnostics()) {
      message += " ";
      message += std::string(diagnostic.token);
    }
    throw std::runtime_error(message);
  }
  if (collector.ends.size() <
      static_cast<std::size_t>(plan.total_frames)) {
    throw std::runtime_error(perf::detail::stall_message(
        "root-music-perf", collector.ends.size(), plan.total_frames));
  }
  return collector;
}

void fill_throughput(perf::ResultRow* row,
                     const RootMusicScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  row->cells_per_second =
      static_cast<double>(scen.detections) / sec;
}

perf::ResultRow base_row(const RootMusicScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "RootMUSICEstimatorOp";
  row.variant = scen.fixed_model_order.has_value()
                    ? "root_music_model_order_" +
                          std::to_string(*scen.fixed_model_order)
                    : std::string("root_music_live_") +
                          override_name(scen.override_kind);
  row.size_id = scen.size_id;
  row.input_shape = {scen.detections, scen.channels,
                     scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.notes = scen.fixed_model_order.has_value()
                  ? "fixed_model_order=" +
                        std::to_string(*scen.fixed_model_order)
                  : std::string("live_override=") +
                        override_name(scen.override_kind);
  return row;
}

RootMusicScenario make_scenario(
    const SizeSpec& s, RootMusicScenario::OverrideKind kind) {
  RootMusicScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "root_music_" << s.id << "_c" << s.channels
     << "_n" << s.detections << "_s" << s.snapshots
     << "_live_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

RootMusicScenario make_ura_scenario(const char* size_id,
                                    int64_t ny,
                                    int64_t nz,
                                    int64_t detections,
                                    int64_t snapshots,
                                    RootMusicScenario::OverrideKind kind) {
  RootMusicScenario sc;
  sc.size_id = size_id;
  sc.is_ura = true;
  sc.ny = ny;
  sc.nz = nz;
  sc.channels = ny * nz;
  sc.detections = detections;
  sc.snapshots = snapshots;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "root_music_ura_" << ny << "x" << nz
     << "_n" << detections << "_s" << snapshots
     << "_live_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

RootMusicScenario make_model_order_scenario(
    const SizeSpec& size, int64_t model_order) {
  RootMusicScenario scenario = make_scenario(
      size, RootMusicScenario::OverrideKind::kMaxSources);
  scenario.fixed_model_order = model_order;
  std::ostringstream id;
  id << "root_music_" << size.id << "_c" << size.channels
     << "_n" << size.detections << "_s" << size.snapshots
     << "_model_order_" << model_order;
  scenario.id = id.str();
  return scenario;
}

RootMusicScenario make_ura_model_order_scenario(
    const char* size_id, int64_t ny, int64_t nz,
    int64_t detections, int64_t snapshots, int64_t model_order) {
  RootMusicScenario scenario = make_ura_scenario(
      size_id, ny, nz, detections, snapshots,
      RootMusicScenario::OverrideKind::kMaxSources);
  scenario.fixed_model_order = model_order;
  std::ostringstream id;
  id << "root_music_ura_" << ny << "x" << nz
     << "_n" << detections << "_s" << snapshots
     << "_model_order_" << model_order;
  scenario.id = id.str();
  return scenario;
}

void append_override_scenarios(std::vector<RootMusicScenario>* result,
                               const SizeSpec& size) {
  for (const auto kind : {
           RootMusicScenario::OverrideKind::kArrayGeometry,
           RootMusicScenario::OverrideKind::kSpatialSmoothing,
           RootMusicScenario::OverrideKind::kNumSources,
           RootMusicScenario::OverrideKind::kSourceCountPfa,
           RootMusicScenario::OverrideKind::kMaxSources}) {
    result->push_back(make_scenario(size, kind));
  }
}

void append_ura_override_scenarios(
    std::vector<RootMusicScenario>* result, const char* size_id,
    int64_t ny, int64_t nz, int64_t detections, int64_t snapshots) {
  for (const auto kind : {
           RootMusicScenario::OverrideKind::kArrayGeometry,
           RootMusicScenario::OverrideKind::kSpatialSmoothing,
           RootMusicScenario::OverrideKind::kNumSources,
           RootMusicScenario::OverrideKind::kSourceCountPfa,
           RootMusicScenario::OverrideKind::kMaxSources}) {
    result->push_back(make_ura_scenario(
        size_id, ny, nz, detections, snapshots, kind));
  }
}

void append_model_order_scenarios(
    std::vector<RootMusicScenario>* result, const SizeSpec& size) {
  const int64_t upper = std::min<int64_t>(4, size.channels - 1);
  for (const int64_t order : {int64_t{1}, int64_t{2}, upper}) {
    if (order <= 0 ||
        std::any_of(
            result->begin(), result->end(),
            [&](const RootMusicScenario& scenario) {
              return scenario.size_id == size.id &&
                     scenario.fixed_model_order == order;
            })) {
      continue;
    }
    result->push_back(make_model_order_scenario(size, order));
  }
}

void append_ura_model_order_scenarios(
    std::vector<RootMusicScenario>* result, const char* size_id,
    int64_t ny, int64_t nz, int64_t detections, int64_t snapshots) {
  const int64_t channels = ny * nz;
  const int64_t upper = std::min<int64_t>(4, channels - 1);
  for (const int64_t order : {int64_t{1}, int64_t{2}, upper}) {
    if (order <= 0 ||
        std::any_of(
            result->begin(), result->end(),
            [&](const RootMusicScenario& scenario) {
              return scenario.size_id == size_id &&
                     scenario.fixed_model_order == order;
            })) {
      continue;
    }
    result->push_back(make_ura_model_order_scenario(
        size_id, ny, nz, detections, snapshots, order));
  }
}

void append_scenarios(std::vector<RootMusicScenario>* result,
                      const SizeSpec& size) {
  append_override_scenarios(result, size);
  append_model_order_scenarios(result, size);
}

void append_ura_scenarios(
    std::vector<RootMusicScenario>* result, const char* size_id,
    int64_t ny, int64_t nz, int64_t detections, int64_t snapshots) {
  append_ura_override_scenarios(
      result, size_id, ny, nz, detections, snapshots);
  append_ura_model_order_scenarios(
      result, size_id, ny, nz, detections, snapshots);
}

struct RootMusicTraits : perf::PerfDriverDefaults {
  using Scenario = RootMusicScenario;

  static const char* tool_name() {
    return "perf_root_music_estimator";
  }
  static const char* tool_description() {
    return "RootMUSICEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Default sizes with five live overrides and model-order sweep";
  }
  static const char* sizes_subset_help() {
    return "Size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full sizes with five live overrides and model-order sweep";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.channels, s.detections,
                   s.snapshots};
      def.cells = s.detections;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<RootMusicScenario>
  build_default_matrix() {
    std::vector<RootMusicScenario> out;
    for (const char* sid : {"xs", "sm", "md"}) {
      append_scenarios(&out, must_find_size(sid));
    }
    append_ura_scenarios(
        &out, "ura_sm", 6, 6, 128, 96);
    return out;
  }

  static std::vector<RootMusicScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<RootMusicScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '"
                  << sid << "', skipping\n";
        continue;
      }
      append_scenarios(&out, *s);
    }
    return out;
  }

  static std::vector<RootMusicScenario>
  build_full_matrix() {
    std::vector<RootMusicScenario> out;
    for (const auto& s : size_ladder()) {
      append_scenarios(&out, s);
    }
    append_ura_scenarios(
        &out, "ura_sm", 6, 6, 128, 96);
    append_ura_scenarios(
        &out, "ura_lg", 8, 8, 256, 128);
    return out;
  }

  static void run_scenario(
      const RootMusicScenario& scen,
      const perf::RunSettings& rs,
      perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(
            rs, /*has_override=*/!scen.fixed_model_order.has_value());
    const auto samples = perf::collect_scenario_samples(
        rs, plan,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_root_music_graph(scen, p);
        });

    perf::append_setup_row(
        base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    if (scen.fixed_model_order.has_value()) {
      row.notes =
          "fixed num_sources=" +
          std::to_string(*scen.fixed_model_order) +
          " model-order characterization; no inline override; "
          "cells/s is detections/s";
    } else {
      row.notes =
          "override_ms isolates inline " +
          std::string(override_name(scen.override_kind)) +
          "; steady num_sources is fixed at one except the "
          "source_count_pfa scenario, which uses auto; "
          "cells/s is detections/s";
    }
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
              RootMusicScenario::OverrideKind::kMaxSources),
          warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<RootMusicTraits>(
      argc, argv);
}
