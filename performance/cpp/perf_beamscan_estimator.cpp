// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for BeamscanEstimatorOp — conventional
// sparse-path Bartlett DOA on a scan grid. Sweeps detection count, channel
// count, snapshot count, and scan-grid size. Measurement tool only: no timing
// assertions.
// See docs/planning/OPERATOR_DEVELOPMENT.md.
//
//  Graph-based rewrite of the HSDK 4.x Application harness.

#include <algorithm>
#include <array>
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

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
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

struct BeamScenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t snapshots{1};
  int64_t num_az{0};
  int64_t num_el{0};
  enum class OverrideKind {
    kNone,
    kArrayGeometry,
    kAzAngles,
    kElAngles,
    kArrayTaper,
    kPeakInterpolation,
  } override_kind{OverrideKind::kNone};
  hr::WindowType static_taper{hr::WindowType::kNone};
};

const char* override_name(BeamScenario::OverrideKind kind) {
  switch (kind) {
    case BeamScenario::OverrideKind::kNone:
      return "";
    case BeamScenario::OverrideKind::kArrayGeometry:
      return "array_geometry";
    case BeamScenario::OverrideKind::kAzAngles:
      return "az_coordinates";
    case BeamScenario::OverrideKind::kElAngles:
      return "el_coordinates";
    case BeamScenario::OverrideKind::kArrayTaper:
      return "array_taper";
    case BeamScenario::OverrideKind::kPeakInterpolation:
      return "peak_interpolation";
  }
  return "unknown";
}

perf::TimingCollector run_graph(const BeamScenario& scen, const perf::RunPlan& plan) {
  auto snapshot =
      perf::make_iq_cube_device<PerfPrecision>({scen.detections, scen.channels, scen.snapshots});
  hr::BeamscanParams initial;
  initial.array_geometry = make_ula_geometry(scen.channels, 0.5, 1.0);
  initial.az_coordinates = linspace(-0.5, 0.5, scen.num_az);
  initial.el_coordinates = linspace(-0.3, 0.3, scen.num_el);
  initial.peak_interpolation = hr::PeakInterpolation::kNone;
  initial.array_taper.type = scen.static_taper;
  hr::BeamscanParams override = initial;
  switch (scen.override_kind) {
    case BeamScenario::OverrideKind::kNone:
      break;
    case BeamScenario::OverrideKind::kArrayGeometry:
      override.array_geometry = make_ula_geometry(scen.channels, 0.45, 1.0);
      break;
    case BeamScenario::OverrideKind::kAzAngles:
      override.az_coordinates = linspace(-0.45, 0.55, scen.num_az);
      break;
    case BeamScenario::OverrideKind::kElAngles:
      override.el_coordinates = linspace(-0.25, 0.35, scen.num_el);
      break;
    case BeamScenario::OverrideKind::kArrayTaper:
      override.array_taper.type = hr::WindowType::kHann;
      break;
    case BeamScenario::OverrideKind::kPeakInterpolation:
      override.peak_interpolation = hr::PeakInterpolation::kParabolic;
      break;
  }
  hr::MaxTensorExtents extents;
  extents.max_items = scen.detections;
  extents.max_snapshots = scen.snapshots;
  extents.max_az = scen.num_az;
  extents.max_el = scen.num_el;
  const auto configure = [&](holoscan::Graph& graph, const auto& source,
                             holoscan::CompileOptions& options, auto connect) {
    auto op = graph.op<hr::BeamscanEstimatorOp<PerfPrecision>>("beamscan", extents);
    op.set_params(std::move(initial));
    const holoscan::ConnectionOptions q{.queue_depth = 8U};
    graph.add_flow(source->out, op->snapshot, q);
    connect(graph, source, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "beamscan", 8U);
    outputs.timed(op->aoa_estimates, "aoa_estimates");
    outputs.drain(op->angle_spectrum, "angle_spectrum");
    outputs.finish(source, op);
  };
  if (scen.override_kind == BeamScenario::OverrideKind::kNone) {
    return perf::run_data_recipe(std::array{std::move(snapshot)}, plan, "beamscan-perf",
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
      perf::live_option("array_taper",
                        [&] {
                          return perf::selected_live_case(
                              "array_taper", override.array_taper,
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->array_taper, q);
                              });
                        }),
      perf::live_option("peak_interpolation", [&] {
        return perf::selected_live_case(
            "peak_interpolation", override.peak_interpolation,
            [](auto& graph, const auto& source, const auto& op, const auto& q) {
              graph.add_flow(source->override_out, op->peak_interpolation, q);
            });
      }));
  return perf::dispatch_live_case(
      override_name(scen.override_kind), cases, [&](const auto& selected) {
        return perf::run_selected_live_recipe(std::array{std::move(snapshot)}, plan,
                                              "beamscan-perf", selected, configure);
      });
}

void fill_throughput(perf::ResultRow* row, const BeamScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  const double beams = static_cast<double>(scen.num_az) * static_cast<double>(scen.num_el);
  const double ch = static_cast<double>(scen.channels);
  row->cells_per_second = n / sec;
  row->slices_per_second = n * beams / sec;

  const double csz = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double snap_bytes = n * ch * static_cast<double>(scen.snapshots) * csz;
  const double weight_bytes = beams * ch * csz;
  const double spec_bytes = n * beams * static_cast<double>(sizeof(PerfPrecision));
  const double aoa_bytes = n * 2.0 * static_cast<double>(sizeof(PerfPrecision));
  row->estimated_effective_gbps = (snap_bytes + weight_bytes + spec_bytes + aoa_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const BeamScenario& scen) {
  perf::ResultRow row;
  row.op_name = "BeamscanEstimatorOp";
  row.variant = "bartlett";
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.detections, scen.channels, scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.live_port = override_name(scen.override_kind);
  return row;
}

BeamScenario make_scenario(const SizeSpec& s,
                           BeamScenario::OverrideKind kind = BeamScenario::OverrideKind::kNone) {
  BeamScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "beamscan_" << s.id << "_c" << s.channels << "_n" << s.detections << "_g" << s.num_az << "x"
     << s.num_el << "_s" << s.snapshots << "_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

void decorate_row(perf::ResultRow& row, const BeamScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes =
      "cold_ms is first-frame latency; median/p90 "
      "are post-warmup steady state; override_ms "
      "isolates " +
      std::string(override_name(scen.override_kind)) + "; cells/s is detections/s";
  if (row.live_port.empty())
    row.notes = "default parameters; no live override";
}

struct BeamTraits : perf::PerfDriverDefaults {
  using Scenario = BeamScenario;

  static const char* tool_name() { return "perf_beamscan_estimator"; }
  static const char* tool_description() {
    return "BeamscanEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 5)
      throw std::invalid_argument("Beamscan profile rank mismatch");
    return {id, shape, shape[0] * shape[1] * shape[2], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 5 || shape[0] < 2 || shape[1] < 1 || shape[2] < 1 || shape[3] < 1 ||
        shape[4] < 1)
      throw std::invalid_argument("Beamscan invalid shape");
    perf::require_keys(c.parameters, {"array_taper"}, "parameters");
    const auto taper = c.parameters["array_taper"].as<std::string>();
    if (taper != "none" && taper != "hann")
      throw std::invalid_argument("Beamscan invalid taper");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], shape[4], c.profile};
    auto scenario = make_scenario(size);
    scenario.id = c.id;
    scenario.static_taper = taper == "hann" ? hr::WindowType::kHann : hr::WindowType::kNone;
    if (c.override_value) {
      const auto port = (*c.override_value)["port"].as<std::string>();
      const auto value = (*c.override_value)["value"].as<std::string>();
      if (port == "array_geometry" && value == "spacing_045")
        scenario.override_kind = Scenario::OverrideKind::kArrayGeometry;
      else if (port == "az_coordinates" && value == "shift_005")
        scenario.override_kind = Scenario::OverrideKind::kAzAngles;
      else if (port == "el_coordinates" && value == "shift_005")
        scenario.override_kind = Scenario::OverrideKind::kElAngles;
      else if (port == "array_taper" && value == "hann")
        scenario.override_kind = Scenario::OverrideKind::kArrayTaper;
      else if (port == "peak_interpolation" && value == "parabolic")
        scenario.override_kind = Scenario::OverrideKind::kPeakInterpolation;
      else
        throw std::invalid_argument("Beamscan invalid live override");
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<BeamTraits>(argc, argv);
}
