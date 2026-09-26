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
#include <set>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
#include "utils/perf_recipe.hpp"
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

perf::TimingCollector run_graph(const RootMusicScenario& scen, const perf::RunPlan& plan) {
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
  const auto configure = [&](holoscan::Graph& graph, const auto& source,
                             holoscan::CompileOptions& options, auto connect) {
    auto op = graph.op<hr::RootMUSICEstimatorOp<PerfPrecision>>("root_music", extents);
    op.set_params(std::move(initial));
    const holoscan::ConnectionOptions q{.queue_depth = 8U};
    graph.add_flow(source->out, op->snapshot, q);
    connect(graph, source, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "root_music", 8U);
    outputs.timed(op->aoa_estimates, "aoa_estimates");
    outputs.drain(op->doa_multi, "doa_multi");
    outputs.drain(op->source_count, "source_count");
    outputs.drain(op->eigenvalues, "eigenvalues");
    outputs.finish(source, op);
  };
  if (scen.override_kind == RootMusicScenario::OverrideKind::kNone) {
    return perf::run_data_recipe(std::array{std::move(snap)}, plan, "root-music-perf",
                                 [&](auto& graph, const auto& source, auto& options) {
                                   configure(graph, source, options,
                                             [](auto&, const auto&, const auto&, const auto&) {});
                                 });
  }
  const auto cases = std::make_tuple(
      perf::live_option("array_geometry",
                        [&] {
                          return perf::selected_live_case(
                              "array_geometry", override.array_geometry,
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->array_geometry, q);
                              });
                        }),
      perf::live_option("spatial_smoothing",
                        [&] {
                          return perf::selected_live_case(
                              "spatial_smoothing", perf::smoothing_wire(override.spatial_smoothing),
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->spatial_smoothing, q);
                              });
                        }),
      perf::live_option("num_sources",
                        [&] {
                          return perf::selected_live_case(
                              "num_sources", override.num_sources,
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->num_sources, q);
                              });
                        }),
      perf::live_option("source_count_pfa",
                        [&] {
                          return perf::selected_live_case(
                              "source_count_pfa", override.source_count_pfa,
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->source_count_pfa, q);
                              });
                        }),
      perf::live_option("max_sources", [&] {
        return perf::selected_live_case(
            "max_sources", override.max_sources,
            [](auto& graph, const auto& source, const auto& op, const auto& q) {
              graph.add_flow(source->override_out, op->max_sources, q);
            });
      }));
  return perf::dispatch_live_case(
      override_name(scen.override_kind), cases, [&](const auto& selected) {
        return perf::run_selected_live_recipe(std::array{std::move(snap)}, plan, "root-music-perf",
                                              selected, configure);
      });
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
  row.op_name = "RootMUSICEstimatorOp";
  row.variant = scen.override_kind == RootMusicScenario::OverrideKind::kNone &&
                        scen.fixed_model_order.has_value()
                    ? "root_music_model_order_" + std::to_string(*scen.fixed_model_order)
                    : std::string("root_music_live_") + override_name(scen.override_kind);
  row.live_port = override_name(scen.override_kind);
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

void decorate_row(perf::ResultRow& row, const RootMusicScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  if (scen.override_kind == RootMusicScenario::OverrideKind::kNone) {
    row.notes = "Params num_sources=" +
                (scen.fixed_model_order ? std::to_string(*scen.fixed_model_order) : "auto") +
                "; no inline override; cells/s is detections/s";
  } else {
    row.notes = "override_ms isolates inline " + std::string(override_name(scen.override_kind)) +
                "; Params num_sources=" + std::to_string(scen.fixed_model_order.value_or(-1)) +
                "; cells/s is detections/s";
  }
}

struct RootMusicTraits : perf::PerfDriverDefaults {
  using Scenario = RootMusicScenario;

  static const char* tool_name() { return "perf_root_music_estimator"; }
  static const char* tool_description() {
    return "RootMUSICEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 3)
      throw std::invalid_argument("root_music_estimator profile rank mismatch");
    return {id, shape, shape[0] * shape[1] * shape[2], id};
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
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<RootMusicTraits>(argc, argv);
}
