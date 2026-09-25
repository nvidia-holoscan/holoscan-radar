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
#include "utils/perf_payloads.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using perf::make_ula_geometry;
using perf::emit_wire_doubles;
using perf::smoothing_wire;

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
    kNone,
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
  OverrideKind override_kind{OverrideKind::kNone};
  std::optional<int64_t> fixed_model_order;
};

const char* override_name(RootMusicScenario::OverrideKind kind) {
  switch (kind) {
    case RootMusicScenario::OverrideKind::kNone:
      return "";
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

hr::ArrayGeometry make_ura_geometry(int64_t ny, int64_t nz, double spacing, double wavelength) {
  hr::ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = hr::ArrayGeometryType::kUra;
  geom.element_positions.reserve(static_cast<size_t>(ny * nz * 3));
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      geom.element_positions.push_back(0.0);
      geom.element_positions.push_back(static_cast<double>(iy) * spacing);
      geom.element_positions.push_back(static_cast<double>(iz) * spacing);
    }
  }
  return geom;
}

hr::ArrayGeometry scenario_geometry(const RootMusicScenario& scen, double spacing,
                                    double wavelength) {
  if (scen.is_ura) {
    return make_ura_geometry(scen.ny, scen.nz, spacing, wavelength);
  }
  return make_ula_geometry(scen.channels, spacing, wavelength);
}

struct LivePorts {
 public:
  LivePorts(hr::RootMUSICParams override, RootMusicScenario::OverrideKind kind)
  : override_(std::move(override)), kind_(kind) {}

  void setup(holoscan::OperatorSpec& spec) {
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
    spec.output(snapshot).produces_tensor(
        hr::pooled_tensor_output(kDevice, 8U, perf::PerfFrameBank::span_a()));
    spec.output(array_geometry)
        .max_emits_per_compute(1U)
        .produces_tensor(hr::pooled_tensor_output(kHostMatrix, 2U, kHostPayloadSpan));
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
      case RootMusicScenario::OverrideKind::kNone:
        break;
      case RootMusicScenario::OverrideKind::kArrayGeometry: {
        const auto packed = hr::pack_array_geometry(override_.array_geometry);
        const std::array<int64_t, 2> shape{override_.array_geometry.num_elements() + 1, 3};
        emitted = emit_wire_doubles(array_geometry, hr::to_wire(override_.array_geometry), packed,
                                    shape, time);
        break;
      }
      case RootMusicScenario::OverrideKind::kSpatialSmoothing: {
        emitted = spatial_smoothing.emit(smoothing_wire(override_.spatial_smoothing),
                                         holoscan::EmitOptions{.capture_time = time});
        break;
      }
      case RootMusicScenario::OverrideKind::kNumSources: {
        emitted =
            num_sources.emit(override_.num_sources, holoscan::EmitOptions{.capture_time = time});
        break;
      }
      case RootMusicScenario::OverrideKind::kSourceCountPfa: {
        emitted = source_count_pfa.emit(override_.source_count_pfa,
                                        holoscan::EmitOptions{.capture_time = time});
        break;
      }
      case RootMusicScenario::OverrideKind::kMaxSources: {
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
  holoscan::Output<hr::SpatialSmoothingOverride> spatial_smoothing{"spatial_smoothing"};
  holoscan::Output<int64_t> num_sources{"num_sources"};
  holoscan::Output<double> source_count_pfa{"source_count_pfa"};
  holoscan::Output<int64_t> max_sources{"max_sources"};

 private:
  hr::RootMUSICParams override_;
  RootMusicScenario::OverrideKind kind_{RootMusicScenario::OverrideKind::kNone};
};

using LiveSource = perf::PerfLiveSourceOp<LivePorts, &LivePorts::snapshot>;

perf::TimingCollector run_root_music_graph(const RootMusicScenario& scen,
                                           const perf::RunPlan& plan) {
  auto snap =
      perf::make_iq_cube_device<PerfPrecision>({scen.detections, scen.channels, scen.snapshots});
  constexpr int64_t kInitialMaxSources = 4;
  constexpr int64_t kOverrideMaxSources = 3;
  hr::RootMUSICParams initial;
  initial.array_geometry = scenario_geometry(scen, 0.5, 1.0);
  initial.angle_convention.preset = hr::AngleConventionPreset::kRadarBoresight;
  initial.num_sources = scen.override_kind == RootMusicScenario::OverrideKind::kSourceCountPfa
                            ? -1
                            : scen.fixed_model_order.value_or(-1);
  initial.source_count_pfa = 1e-2;
  initial.max_sources = kInitialMaxSources;
  hr::RootMUSICParams override = initial;
  switch (scen.override_kind) {
    case RootMusicScenario::OverrideKind::kNone:
      break;
    case RootMusicScenario::OverrideKind::kArrayGeometry:
      override.array_geometry = scenario_geometry(scen, 0.45, 1.0);
      break;
    case RootMusicScenario::OverrideKind::kSpatialSmoothing:
      override.spatial_smoothing =
          scen.is_ura ? std::vector<int64_t>{2, 2} : std::vector<int64_t>{2};
      break;
    case RootMusicScenario::OverrideKind::kNumSources:
      override.num_sources = 2;
      break;
    case RootMusicScenario::OverrideKind::kSourceCountPfa:
      override.source_count_pfa = 1e-3;
      break;
    case RootMusicScenario::OverrideKind::kMaxSources:
      override.max_sources = kOverrideMaxSources;
      break;
  }

  hr::MaxTensorExtents extents;
  extents.max_items = scen.detections;
  extents.max_sources = kInitialMaxSources;
  extents.max_channels = scen.channels;
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::now();
  perf::PerfFrameBank::reset(std::move(snap), plan.total_frames, &collector);

  holoscan::Graph graph{"root-music-perf"};
  const auto source = graph.op<LiveSource>("src", plan.override_at, override, scen.override_kind);
  auto op = graph.op<hr::RootMUSICEstimatorOp<PerfPrecision>>("root_music", extents);
  op.set_params(std::move(initial));
  const auto sink = graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
  const auto doa = graph.op<perf::PerfNullSinkOp>("doa_drain");
  const auto count = graph.op<perf::PerfNullSinkOp>("count_drain");
  const auto eigen = graph.op<perf::PerfNullSinkOp>("eig_drain");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};

  graph.add_flow(source->snapshot, op->snapshot, q);
  if (scen.override_kind == RootMusicScenario::OverrideKind::kArrayGeometry)
    graph.add_flow(source->array_geometry, op->array_geometry, q);
  if (scen.override_kind == RootMusicScenario::OverrideKind::kSpatialSmoothing)
    graph.add_flow(source->spatial_smoothing, op->spatial_smoothing, q);
  if (scen.override_kind == RootMusicScenario::OverrideKind::kNumSources)
    graph.add_flow(source->num_sources, op->num_sources, q);
  if (scen.override_kind == RootMusicScenario::OverrideKind::kSourceCountPfa)
    graph.add_flow(source->source_count_pfa, op->source_count_pfa, q);
  if (scen.override_kind == RootMusicScenario::OverrideKind::kMaxSources)
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
  perf::execute_perf_graph(graph, std::move(options), plan.total_frames, "root-music-perf",
                           &collector);
  return collector;
}

void fill_throughput(perf::ResultRow* row, const RootMusicScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  row->cells_per_second = static_cast<double>(scen.detections) / sec;
}

perf::ResultRow base_row(const RootMusicScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "RootMUSICEstimatorOp";
  row.variant = scen.override_kind == RootMusicScenario::OverrideKind::kNone &&
                        scen.fixed_model_order.has_value()
                    ? "root_music_model_order_" + std::to_string(*scen.fixed_model_order)
                    : std::string("root_music_live_") + override_name(scen.override_kind);
  row.live_port = override_name(scen.override_kind);
  row.size_id = scen.size_id;
  row.input_shape = {scen.detections, scen.channels, scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.notes = scen.override_kind == RootMusicScenario::OverrideKind::kNone &&
                      scen.fixed_model_order.has_value()
                  ? "fixed_model_order=" + std::to_string(*scen.fixed_model_order)
                  : std::string("live_override=") + override_name(scen.override_kind);
  return row;
}

RootMusicScenario make_scenario(const SizeSpec& s, RootMusicScenario::OverrideKind kind =
                                                       RootMusicScenario::OverrideKind::kNone) {
  RootMusicScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "root_music_" << s.id << "_c" << s.channels << "_n" << s.detections << "_s" << s.snapshots
     << "_live_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

struct RootMusicTraits : perf::PerfDriverDefaults {
  using Scenario = RootMusicScenario;

  static const char* tool_name() { return "perf_root_music_estimator"; }
  static const char* tool_description() {
    return "RootMUSICEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "root_music_estimator.yaml",
        "root_music_estimator");
  }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 3)
        throw std::invalid_argument("root_music_estimator profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape, shape[0] * shape[1] * shape[2],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 3 || shape[0] < 4 || shape[1] < 1 || shape[2] < 2)
      throw std::invalid_argument("root_music_estimator invalid shape");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], c.profile};
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
        scenario.override_kind = RootMusicScenario::OverrideKind::kArrayGeometry;
      else if (port == "spatial_smoothing" && value == "two")
        scenario.override_kind = RootMusicScenario::OverrideKind::kSpatialSmoothing;
      else if (port == "num_sources" && value == "2")
        scenario.override_kind = RootMusicScenario::OverrideKind::kNumSources;
      else if (port == "source_count_pfa" && value == "0.001")
        scenario.override_kind = RootMusicScenario::OverrideKind::kSourceCountPfa;
      else if (port == "max_sources" && value == "3")
        scenario.override_kind = RootMusicScenario::OverrideKind::kMaxSources;
      else
        throw std::invalid_argument("root_music_estimator invalid live override");
    }
    return scenario;
  }

  static void run_scenario(const RootMusicScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    perf::measure_scenario(
        rs, scen.override_kind != RootMusicScenario::OverrideKind::kNone,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_root_music_graph(scen, p);
        },
        base_row(scen), report,
        [&](perf::ResultRow& row) {
          fill_throughput(&row, scen, row.latency.median_ms);
          if (scen.override_kind == RootMusicScenario::OverrideKind::kNone) {
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
  return holoscan::radar::perf::perf_main<RootMusicTraits>(argc, argv);
}
