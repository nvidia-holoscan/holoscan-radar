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
#include "utils/perf_recipe.hpp"
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

perf::TimingCollector run_graph(const MusicScenario& scen, const perf::RunPlan& plan) {
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
  const auto configure = [&](holoscan::Graph& graph, const auto& source,
                             holoscan::CompileOptions& options, auto connect) {
    auto op = graph.op<hr::MUSICEstimatorOp<PerfPrecision>>("music", extents);
    op.set_params(std::move(initial));
    const holoscan::ConnectionOptions q{.queue_depth = 8U};
    graph.add_flow(source->out, op->snapshot, q);
    connect(graph, source, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "music", 8U);
    outputs.timed(op->aoa_estimates, "aoa_estimates");
    outputs.drain(op->angle_spectrum, "angle_spectrum");
    outputs.drain(op->doa_multi, "doa_multi");
    outputs.drain(op->source_count, "source_count");
    outputs.drain(op->eigenvalues, "eigenvalues");
    outputs.finish(source, op);
  };
  if (scen.override_kind == MusicScenario::OverrideKind::kNone) {
    return perf::run_data_recipe(std::array{std::move(snap)}, plan, "music-perf",
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
      perf::live_option("az_coordinates",
                        [&] {
                          return perf::selected_live_case(
                              "az_coordinates", override.az_coordinates,
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->az_coordinates, q);
                              });
                        }),
      perf::live_option("el_coordinates",
                        [&] {
                          return perf::selected_live_case(
                              "el_coordinates", override.el_coordinates,
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->el_coordinates, q);
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
        return perf::run_selected_live_recipe(std::array{std::move(snap)}, plan, "music-perf",
                                              selected, configure);
      });
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
  row.op_name = "MUSICEstimatorOp";
  row.variant =
      scen.override_kind == MusicScenario::OverrideKind::kNone && scen.fixed_model_order.has_value()
          ? "music_model_order_" + std::to_string(*scen.fixed_model_order)
          : std::string("music_live_") + override_name(scen.override_kind);
  row.live_port = override_name(scen.override_kind);
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

void decorate_row(perf::ResultRow& row, const MusicScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  if (scen.override_kind == MusicScenario::OverrideKind::kNone) {
    row.notes = "Params num_sources=" +
                (scen.fixed_model_order ? std::to_string(*scen.fixed_model_order) : "auto") +
                "; no inline override; cells/s is detections/s";
  } else {
    row.notes = "override_ms isolates inline " + std::string(override_name(scen.override_kind)) +
                "; Params num_sources=" + std::to_string(scen.fixed_model_order.value_or(-1)) +
                "; cells/s is detections/s";
  }
}

struct MusicTraits : perf::PerfDriverDefaults {
  using Scenario = MusicScenario;

  static const char* tool_name() { return "perf_music_estimator"; }
  static const char* tool_description() {
    return "MUSICEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 5)
      throw std::invalid_argument("music_estimator profile rank mismatch");
    return {id, shape, shape[0] * shape[1] * shape[2], id};
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
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<MusicTraits>(argc, argv);
}
