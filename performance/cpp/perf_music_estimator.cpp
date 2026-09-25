// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for MUSICEstimatorOp — subspace (MUSIC)
// sparse-path DOA. Per item it forms a snapshot covariance,
// eigendecomposes it, resolves a model order, builds the noise-subspace
// projector, then scans the (az, el) grid for the MUSIC pseudospectrum.
// Sweeps detection count, channel count, snapshot count, and scan-grid
// size. Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.
//
//  Graph-based rewrite of the HSDK 4.x Application harness.
// Measures each live override independently through the production Exact
// contract.
//
// num_sources is fixed (=1) across the matrix rather than auto: this keeps
// the model order — and hence the noise-subspace projector's cost —
// deterministic across repeats, isolating the
// covariance/eig/projector/scan path from the (cheap, separately-kernelled)
// EFT source-count decision.

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

#include "angle_estimation/music_estimator_op/music_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_payloads.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using perf::linspace;
using perf::make_ula_geometry;
using perf::emit_host_vector;
using perf::emit_wire_doubles;
using perf::smoothing_wire;

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

struct MusicScenario {
  enum class OverrideKind {
    kNone,
    kArrayGeometry,
    kAzAngles,
    kElAngles,
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
  int64_t num_az{0};
  int64_t num_el{0};
  OverrideKind override_kind{OverrideKind::kNone};
  std::optional<int64_t> fixed_model_order;
};

const char* override_name(MusicScenario::OverrideKind kind) {
  switch (kind) {
    case MusicScenario::OverrideKind::kNone:
      return "";
    case MusicScenario::OverrideKind::kArrayGeometry:
      return "array_geometry";
    case MusicScenario::OverrideKind::kAzAngles:
      return "az_coordinates";
    case MusicScenario::OverrideKind::kElAngles:
      return "el_coordinates";
    case MusicScenario::OverrideKind::kSpatialSmoothing:
      return "spatial_smoothing";
    case MusicScenario::OverrideKind::kNumSources:
      return "num_sources";
    case MusicScenario::OverrideKind::kSourceCountPfa:
      return "source_count_pfa";
    case MusicScenario::OverrideKind::kMaxSources:
      return "max_sources";
  }
  return "unknown";
}

struct LivePorts {
 public:
  LivePorts(hr::MUSICParams override, MusicScenario::OverrideKind kind)
  : override_(std::move(override)), kind_(kind) {}

  void setup(holoscan::OperatorSpec& spec) {
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
    spec.output(num_sources).max_emits_per_compute(1U);
    spec.output(source_count_pfa).max_emits_per_compute(1U);
    spec.output(max_sources).max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_override(
      holoscan::ExecutionContext& context) {
    const holoscan::TimePoint time = context.activation_time();
    holoscan::expected<void, holoscan::Error> emitted;
    switch (kind_) {
      case MusicScenario::OverrideKind::kNone:
        break;
      case MusicScenario::OverrideKind::kArrayGeometry: {
        const auto packed = hr::pack_array_geometry(override_.array_geometry);
        const std::array<int64_t, 2> shape{override_.array_geometry.num_elements() + 1, 3};
        emitted = emit_wire_doubles(array_geometry, hr::to_wire(override_.array_geometry), packed,
                                    shape, time);
        break;
      }
      case MusicScenario::OverrideKind::kAzAngles: {
        emitted = emit_host_vector(az_coordinates, override_.az_coordinates, time);
        break;
      }
      case MusicScenario::OverrideKind::kElAngles: {
        emitted = emit_host_vector(el_coordinates, override_.el_coordinates, time);
        break;
      }
      case MusicScenario::OverrideKind::kSpatialSmoothing: {
        emitted = spatial_smoothing.emit(smoothing_wire(override_.spatial_smoothing),
                                         holoscan::EmitOptions{.capture_time = time});
        break;
      }
      case MusicScenario::OverrideKind::kNumSources: {
        emitted =
            num_sources.emit(override_.num_sources, holoscan::EmitOptions{.capture_time = time});
        break;
      }
      case MusicScenario::OverrideKind::kSourceCountPfa: {
        emitted = source_count_pfa.emit(override_.source_count_pfa,
                                        holoscan::EmitOptions{.capture_time = time});
        break;
      }
      case MusicScenario::OverrideKind::kMaxSources: {
        emitted =
            max_sources.emit(override_.max_sources, holoscan::EmitOptions{.capture_time = time});
        break;
      }
    }
    if (!emitted)
      return emitted;
    return {};
  }

  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Output<hr::WireArrayGeometry> array_geometry{"array_geometry"};
  holoscan::Output<holoscan::Tensor> az_coordinates{"az_coordinates"};
  holoscan::Output<holoscan::Tensor> el_coordinates{"el_coordinates"};
  holoscan::Output<hr::SpatialSmoothingOverride> spatial_smoothing{"spatial_smoothing"};
  holoscan::Output<int64_t> num_sources{"num_sources"};
  holoscan::Output<double> source_count_pfa{"source_count_pfa"};
  holoscan::Output<int64_t> max_sources{"max_sources"};

 private:
  hr::MUSICParams override_;
  MusicScenario::OverrideKind kind_{MusicScenario::OverrideKind::kNone};
};

using LiveSource = perf::PerfLiveSourceOp<LivePorts, &LivePorts::snapshot>;

perf::TimingCollector run_music_graph(const MusicScenario& scen, const perf::RunPlan& plan) {
  auto snap =
      perf::make_iq_cube_device<PerfPrecision>({scen.detections, scen.channels, scen.snapshots});
  constexpr int64_t kInitialMaxSources = 4;
  constexpr int64_t kOverrideMaxSources = 3;
  hr::MUSICParams initial;
  initial.array_geometry = make_ula_geometry(scen.channels, 0.5, 1.0);
  initial.angle_convention.preset = hr::AngleConventionPreset::kRadarBoresight;
  initial.az_coordinates = linspace(-0.5, 0.5, scen.num_az);
  initial.el_coordinates = linspace(-0.3, 0.3, scen.num_el);
  initial.num_sources = scen.override_kind == MusicScenario::OverrideKind::kSourceCountPfa
                            ? -1
                            : scen.fixed_model_order.value_or(-1);
  initial.source_count_pfa = 1e-2;
  initial.max_sources = kInitialMaxSources;
  hr::MUSICParams override = initial;
  switch (scen.override_kind) {
    case MusicScenario::OverrideKind::kNone:
      break;
    case MusicScenario::OverrideKind::kArrayGeometry:
      override.array_geometry = make_ula_geometry(scen.channels, 0.45, 1.0);
      break;
    case MusicScenario::OverrideKind::kAzAngles:
      override.az_coordinates = linspace(-0.45, 0.55, scen.num_az);
      break;
    case MusicScenario::OverrideKind::kElAngles:
      override.el_coordinates = linspace(-0.25, 0.35, scen.num_el);
      break;
    case MusicScenario::OverrideKind::kSpatialSmoothing:
      override.spatial_smoothing = {2};
      break;
    case MusicScenario::OverrideKind::kNumSources:
      override.num_sources = 2;
      break;
    case MusicScenario::OverrideKind::kSourceCountPfa:
      override.source_count_pfa = 1e-3;
      break;
    case MusicScenario::OverrideKind::kMaxSources:
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

  holoscan::Graph graph{"music-perf"};
  const auto source = graph.op<LiveSource>("src", plan.override_at, override, scen.override_kind);
  auto op = graph.op<hr::MUSICEstimatorOp<PerfPrecision>>("music", extents);
  op.set_params(std::move(initial));
  const auto sink = graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
  const auto spectrum = graph.op<perf::PerfNullSinkOp>("spectrum_drain");
  const auto doa = graph.op<perf::PerfNullSinkOp>("doa_drain");
  const auto count = graph.op<perf::PerfNullSinkOp>("count_drain");
  const auto eigen = graph.op<perf::PerfNullSinkOp>("eig_drain");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->snapshot, op->snapshot, q);
  if (scen.override_kind == MusicScenario::OverrideKind::kArrayGeometry)
    graph.add_flow(source->array_geometry, op->array_geometry, q);
  if (scen.override_kind == MusicScenario::OverrideKind::kAzAngles)
    graph.add_flow(source->az_coordinates, op->az_coordinates, q);
  if (scen.override_kind == MusicScenario::OverrideKind::kElAngles)
    graph.add_flow(source->el_coordinates, op->el_coordinates, q);
  if (scen.override_kind == MusicScenario::OverrideKind::kSpatialSmoothing)
    graph.add_flow(source->spatial_smoothing, op->spatial_smoothing, q);
  if (scen.override_kind == MusicScenario::OverrideKind::kNumSources)
    graph.add_flow(source->num_sources, op->num_sources, q);
  if (scen.override_kind == MusicScenario::OverrideKind::kSourceCountPfa)
    graph.add_flow(source->source_count_pfa, op->source_count_pfa, q);
  if (scen.override_kind == MusicScenario::OverrideKind::kMaxSources)
    graph.add_flow(source->max_sources, op->max_sources, q);
  graph.add_flow(op->aoa_estimates, sink->in, q);
  graph.add_flow(op->angle_spectrum, spectrum->in, q);
  graph.add_flow(op->doa_multi, doa->in, q);
  graph.add_flow(op->source_count, count->in, q);
  graph.add_flow(op->eigenvalues, eigen->in, q);

  holoscan::CompileOptions options;
  perf::bind_tensor_out(options, "src", "snapshot");
  perf::bind_tensor_out(options, "music", "aoa_estimates");
  perf::bind_tensor_out(options, "music", "angle_spectrum");
  perf::bind_tensor_out(options, "music", "doa_multi");
  perf::bind_tensor_out(options, "music", "source_count");
  perf::bind_tensor_out(options, "music", "eigenvalues");
  perf::execute_perf_graph(graph, std::move(options), plan.total_frames, "music-perf", &collector);
  return collector;
}

void fill_throughput(perf::ResultRow* row, const MusicScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  row->cells_per_second = n / sec;
  const double beams = static_cast<double>(scen.num_az) * static_cast<double>(scen.num_el);
  row->slices_per_second = n * beams / sec;
}

perf::ResultRow base_row(const MusicScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "MUSICEstimatorOp";
  row.variant =
      scen.override_kind == MusicScenario::OverrideKind::kNone && scen.fixed_model_order.has_value()
          ? "music_model_order_" + std::to_string(*scen.fixed_model_order)
          : std::string("music_live_") + override_name(scen.override_kind);
  row.live_port = override_name(scen.override_kind);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.detections, scen.channels, scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.notes =
      scen.override_kind == MusicScenario::OverrideKind::kNone && scen.fixed_model_order.has_value()
          ? "fixed_model_order=" + std::to_string(*scen.fixed_model_order)
          : std::string("live_override=") + override_name(scen.override_kind);
  return row;
}

MusicScenario make_scenario(const SizeSpec& s,
                            MusicScenario::OverrideKind kind = MusicScenario::OverrideKind::kNone) {
  MusicScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "music_" << s.id << "_c" << s.channels << "_n" << s.detections << "_s" << s.snapshots
     << "_g" << s.num_az << "x" << s.num_el << "_live_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

struct MusicTraits : perf::PerfDriverDefaults {
  using Scenario = MusicScenario;

  static const char* tool_name() { return "perf_music_estimator"; }
  static const char* tool_description() {
    return "MUSICEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "music_estimator.yaml",
        "music_estimator");
  }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 5)
        throw std::invalid_argument("music_estimator profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape, shape[0] * shape[1] * shape[2],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 5 || shape[0] < 4 || shape[1] < 1 || shape[2] < 2 || shape[3] < 1 ||
        shape[4] < 1)
      throw std::invalid_argument("music_estimator invalid shape");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], shape[4], c.profile};
    auto scenario = make_scenario(size);
    scenario.id = c.id;
    perf::require_keys(c.parameters, {"num_sources"}, "parameters");
    const auto model_order = c.parameters["num_sources"].as<std::string>();
    if (model_order == "auto") {
      scenario.fixed_model_order.reset();
    } else {
      const int64_t value = std::stoll(model_order);
      if (value < 1 || value > 3)
        throw std::invalid_argument("invalid model order");
      scenario.fixed_model_order = value;
    }
    if (c.override_value) {
      const auto port = (*c.override_value)["port"].as<std::string>();
      const auto value = (*c.override_value)["value"].as<std::string>();
      if (port == "array_geometry" && value == "spacing_045")
        scenario.override_kind = MusicScenario::OverrideKind::kArrayGeometry;
      else if (port == "az_coordinates" && value == "shift_005")
        scenario.override_kind = MusicScenario::OverrideKind::kAzAngles;
      else if (port == "el_coordinates" && value == "shift_005")
        scenario.override_kind = MusicScenario::OverrideKind::kElAngles;
      else if (port == "spatial_smoothing" && value == "two")
        scenario.override_kind = MusicScenario::OverrideKind::kSpatialSmoothing;
      else if (port == "num_sources" && value == "2")
        scenario.override_kind = MusicScenario::OverrideKind::kNumSources;
      else if (port == "source_count_pfa" && value == "0.001")
        scenario.override_kind = MusicScenario::OverrideKind::kSourceCountPfa;
      else if (port == "max_sources" && value == "3")
        scenario.override_kind = MusicScenario::OverrideKind::kMaxSources;
      else
        throw std::invalid_argument("music_estimator invalid live override");
    }
    return scenario;
  }

  static void run_scenario(const MusicScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    perf::measure_scenario(
        rs, scen.override_kind != MusicScenario::OverrideKind::kNone,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_music_graph(scen, p);
        },
        base_row(scen), report,
        [&](perf::ResultRow& row) {
          fill_throughput(&row, scen, row.latency.median_ms);
          if (scen.override_kind == MusicScenario::OverrideKind::kNone) {
            row.notes =
                "Params num_sources=" +
                (scen.fixed_model_order ? std::to_string(*scen.fixed_model_order) : "auto") +
                "; no inline override; cells/s is detections/s";
          } else {
            row.notes =
                "override_ms isolates inline " + std::string(override_name(scen.override_kind)) +
                "; Params num_sources=" + std::to_string(scen.fixed_model_order.value_or(-1)) +
                "; cells/s is detections/s";
          }
        });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<MusicTraits>(argc, argv);
}
