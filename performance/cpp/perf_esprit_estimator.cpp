// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ESPRITEstimatorOp. Measures the shared
// covariance/Hermitian-eig/EFT front-end plus the device LS rotation solve.
// Measurement only: no timing assertions.
//
// [EA HSDK5 ONLY] Graph-based rewrite of the HSDK 4.x Application harness.
// Mid-session [live] override measurement is deferred until
// docs/planning/HSDK5_PORT.md §1 (EA partition / device binding). Params
// still select array_geometry, num_sources, etc. under test.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
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
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t snapshots{0};
  bool is_ura{false};
  int64_t ny{0};
  int64_t nz{0};
};

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

Scenario make_scenario(const SizeSpec& size) {
  Scenario scenario;
  scenario.size_id = size.id;
  scenario.channels = size.channels;
  scenario.detections = size.detections;
  scenario.snapshots = size.snapshots;
  std::ostringstream id;
  id << "esprit_" << size.id << "_c" << size.channels << "_n"
     << size.detections << "_s" << size.snapshots;
  scenario.id = id.str();
  return scenario;
}

Scenario make_ura_scenario(const char* size_id, int64_t ny, int64_t nz,
                           int64_t detections, int64_t snapshots) {
  Scenario scenario;
  scenario.size_id = size_id;
  scenario.is_ura = true;
  scenario.ny = ny;
  scenario.nz = nz;
  scenario.channels = ny * nz;
  scenario.detections = detections;
  scenario.snapshots = snapshots;
  std::ostringstream id;
  id << "esprit_ura_" << ny << "x" << nz << "_n" << detections
     << "_s" << snapshots;
  scenario.id = id.str();
  return scenario;
}

perf::TimingCollector run_esprit_graph(const Scenario& scen,
                                       int total_frames) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(
      {scen.detections, scen.channels, scen.snapshots});
  return perf::run_perf_graph(
      std::move(cube), total_frames, "esprit-perf",
      [&](holoscan::Graph& graph, const auto& src,
          holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_items = scen.detections;
        extents.max_sources = 4;
        extents.max_channels = scen.channels;
        auto op =
            graph.op<hr::ESPRITEstimatorOp<PerfPrecision>>(
                "esprit", extents);
        hr::ESPRITParams params;
        params.array_geometry = scenario_geometry(scen, 0.5);
        params.angle_convention.preset =
            hr::AngleConventionPreset::kRadarBoresight;
        params.num_sources = 1;
        params.source_count_pfa = 1e-4;
        params.max_sources = 4;
        op.set_params(std::move(params));

        const auto aoa_sink =
            graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
        const auto doa_drain =
            graph.op<perf::PerfNullSinkOp>("doa_drain");
        const auto sc_drain =
            graph.op<perf::PerfNullSinkOp>("sc_drain");
        const auto eig_drain =
            graph.op<perf::PerfNullSinkOp>("eig_drain");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};

        graph.add_flow(src->out, op->snapshot, q);
        graph.add_flow(op->aoa_estimates, aoa_sink->in, q);
        graph.add_flow(op->doa_multi, doa_drain->in, q);
        graph.add_flow(op->source_count, sc_drain->in, q);
        graph.add_flow(op->eigenvalues, eig_drain->in, q);

        perf::bind_tensor_out(options, "esprit", "aoa_estimates");
        perf::bind_tensor_out(options, "esprit", "doa_multi");
        perf::bind_tensor_out(options, "esprit", "source_count");
        perf::bind_tensor_out(options, "esprit", "eigenvalues");
      });
}

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "ESPRITEstimatorOp";
  row.variant = "esprit_ls";
  row.size_id = scenario.size_id;
  row.input_shape = {scenario.detections, scenario.channels,
                     scenario.snapshots};
  row.out_batch = scenario.detections;
  row.k = scenario.channels;
  row.notes = "live_override=deferred_ea";
  return row;
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;

  static const char* tool_name() { return "perf_esprit_estimator"; }
  static const char* tool_description() {
    return "ESPRITEstimatorOp performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Default ESPRIT size sweep";
  }
  static const char* sizes_subset_help() {
    return "ESPRIT sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full ESPRIT size ladder; live-override timing deferred on "
           "HSDK 5 EA (HSDK5_PORT.md §1)";
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
    return {make_scenario(must_find_size("xs")),
            make_scenario(must_find_size("sm")),
            make_scenario(must_find_size("md")),
            make_ura_scenario("ura_sm", 6, 6, 128, 96)};
  }

  static std::vector<Scenario> build_size_sweep(
      const std::vector<std::string>& ids) {
    std::vector<Scenario> result;
    for (const auto& id : ids) {
      if (const SizeSpec* size = find_size(id)) {
        result.push_back(make_scenario(*size));
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
      result.push_back(make_scenario(size));
    }
    result.push_back(make_ura_scenario("ura_sm", 6, 6, 128, 96));
    result.push_back(
        make_ura_scenario("ura_lg", 8, 8, 256, 128));
    return result;
  }

  static void run_scenario(const Scenario& scenario,
                           const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const auto plan =
        perf::make_run_plan(settings, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        settings, plan, [&](int, const perf::RunPlan& run_plan) {
          return run_esprit_graph(scenario, run_plan.total_frames);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    perf::ResultRow row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    if (row.latency.median_ms > 0.0) {
      row.cells_per_second =
          scenario.detections / (row.latency.median_ms / 1000.0);
    }
    row.notes =
        "fixed one-source steady path; live bundle deferred "
        "(HSDK5_PORT.md §1)";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs")), settings,
                   &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
