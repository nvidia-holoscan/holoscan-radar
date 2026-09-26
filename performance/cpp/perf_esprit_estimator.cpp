// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ESPRITEstimatorOp. Measures the shared
// covariance/Hermitian-eig/EFT front-end plus the device LS rotation solve.
// Measurement only: no timing assertions.
//
//  Graph-based rewrite of the HSDK 4.x Application harness.
// Measures each live override independently through the production Exact
// contract. subarray_spec is static.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <iostream>
#include <memory>
#include <mutex>
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
#include "utils/perf_payloads.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

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

struct Scenario {
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
  int64_t static_model_order{-1};
};

const char* override_name(Scenario::OverrideKind kind) {
  switch (kind) {
    case Scenario::OverrideKind::kNone:
      return "";
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

hr::ArrayGeometry make_ula(int64_t channels, double spacing) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = hr::ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < channels; ++channel) {
    geometry.element_positions.insert(geometry.element_positions.end(),
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
          {0.0, spacing * static_cast<double>(iy), spacing * static_cast<double>(iz)});
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

Scenario make_scenario(const SizeSpec& size,
                       Scenario::OverrideKind kind = Scenario::OverrideKind::kNone) {
  Scenario scenario;
  scenario.size_id = size.id;
  scenario.channels = size.channels;
  scenario.detections = size.detections;
  scenario.snapshots = size.snapshots;
  scenario.override_kind = kind;
  std::ostringstream id;
  id << "esprit_" << size.id << "_c" << size.channels << "_n" << size.detections << "_s"
     << size.snapshots << "_live_" << override_name(kind);
  scenario.id = id.str();
  return scenario;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  auto cube =
      perf::make_iq_cube_device<PerfPrecision>({scen.detections, scen.channels, scen.snapshots});
  constexpr int64_t kInitialMaxSources = 4;
  constexpr int64_t kOverrideMaxSources = 3;
  hr::ESPRITParams initial;
  initial.array_geometry = scenario_geometry(scen, 0.5);
  initial.angle_convention.preset = hr::AngleConventionPreset::kRadarBoresight;
  initial.num_sources =
      scen.override_kind == Scenario::OverrideKind::kSourceCountPfa ? -1 : scen.static_model_order;
  initial.source_count_pfa = 1e-2;
  initial.max_sources = kInitialMaxSources;
  hr::ESPRITParams override = initial;
  switch (scen.override_kind) {
    case Scenario::OverrideKind::kNone:
      break;
    case Scenario::OverrideKind::kArrayGeometry:
      override.array_geometry = scenario_geometry(scen, 0.45);
      break;
    case Scenario::OverrideKind::kSpatialSmoothing:
      override.spatial_smoothing =
          scen.is_ura ? std::vector<int64_t>{2, 2} : std::vector<int64_t>{2};
      break;
    case Scenario::OverrideKind::kNumSources:
      override.num_sources = 2;
      break;
    case Scenario::OverrideKind::kSourceCountPfa:
      override.source_count_pfa = 1e-3;
      break;
    case Scenario::OverrideKind::kMaxSources:
      override.max_sources = kOverrideMaxSources;
      break;
  }

  hr::MaxTensorExtents extents;
  extents.max_items = scen.detections;
  extents.max_sources = kInitialMaxSources;
  extents.max_channels = scen.channels;
  const auto configure = [&](holoscan::Graph& graph, const auto& source,
                             holoscan::CompileOptions& options, auto connect) {
    auto op = graph.op<hr::ESPRITEstimatorOp<PerfPrecision>>("esprit", extents);
    op.set_params(std::move(initial));
    const holoscan::ConnectionOptions q{.queue_depth = 8U};
    graph.add_flow(source->out, op->snapshot, q);
    connect(graph, source, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "esprit", 8U);
    outputs.timed(op->aoa_estimates, "aoa_estimates");
    outputs.drain(op->doa_multi, "doa_multi");
    outputs.drain(op->source_count, "source_count");
    outputs.drain(op->eigenvalues, "eigenvalues");
    outputs.finish(source, op);
  };
  if (scen.override_kind == Scenario::OverrideKind::kNone) {
    return perf::run_data_recipe(std::array{std::move(cube)}, plan, "esprit-perf",
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
        return perf::run_selected_live_recipe(std::array{std::move(cube)}, plan, "esprit-perf",
                                              selected, configure);
      });
}

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.op_name = "ESPRITEstimatorOp";
  row.variant = std::string("esprit_ls_live_") + override_name(scenario.override_kind);
  row.live_port = override_name(scenario.override_kind);
  row.input_shape = {scenario.detections, scenario.channels, scenario.snapshots};
  row.out_batch = scenario.detections;
  row.k = scenario.channels;
  row.notes = std::string("live_override=") + override_name(scenario.override_kind) +
              "; subarray_spec is static";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scenario) {
  if (row.latency.median_ms > 0.0) {
    row.cells_per_second = scenario.detections / (row.latency.median_ms / 1000.0);
  }
  row.notes = "override_ms isolates inline " + std::string(override_name(scenario.override_kind)) +
              "; steady num_sources is fixed at one except the "
              "source_count_pfa scenario, which uses auto; subarray_spec is static";
  if (row.live_port.empty())
    row.notes = "default parameters; no live override";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;

  static const char* tool_name() { return "perf_esprit_estimator"; }
  static const char* tool_description() {
    return "ESPRITEstimatorOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 3)
      throw std::invalid_argument("esprit_estimator profile rank mismatch");
    return {id, shape, shape[0] * shape[1] * shape[2], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 3 || shape[0] < 4 || shape[1] < 1 || shape[2] < 2)
      throw std::invalid_argument("esprit_estimator invalid shape");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], c.profile};
    auto scenario = make_scenario(size);
    scenario.id = c.id;
    perf::require_keys(c.parameters, {"num_sources"}, "parameters");
    const auto model_order = c.parameters["num_sources"].as<std::string>();
    if (model_order == "auto") {
      scenario.static_model_order = -1;
    } else {
      const int64_t value = std::stoll(model_order);
      if (value < 1 || value > 3)
        throw std::invalid_argument("invalid model order");
      scenario.static_model_order = value;
    }
    if (c.override_value) {
      const auto port = (*c.override_value)["port"].as<std::string>();
      const auto value = (*c.override_value)["value"].as<std::string>();
      if (port == "array_geometry" && value == "spacing_045")
        scenario.override_kind = Scenario::OverrideKind::kArrayGeometry;
      else if (port == "spatial_smoothing" && value == "two")
        scenario.override_kind = Scenario::OverrideKind::kSpatialSmoothing;
      else if (port == "num_sources" && value == "2")
        scenario.override_kind = Scenario::OverrideKind::kNumSources;
      else if (port == "source_count_pfa" && value == "0.001")
        scenario.override_kind = Scenario::OverrideKind::kSourceCountPfa;
      else if (port == "max_sources" && value == "3")
        scenario.override_kind = Scenario::OverrideKind::kMaxSources;
      else
        throw std::invalid_argument("esprit_estimator invalid live override");
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
