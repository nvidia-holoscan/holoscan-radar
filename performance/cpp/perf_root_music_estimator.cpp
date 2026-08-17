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
// [EA HSDK5 ONLY] Graph-based rewrite of the HSDK 4.x Application harness.
// Mid-session [live] override measurement is deferred until
// docs/planning/HSDK5_PORT.md §1 (EA partition / device binding). Params
// still select the array geometry under test.
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

#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
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

struct RootMusicScenario {
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

perf::TimingCollector run_root_music_graph(
    const RootMusicScenario& scen, int total_frames) {
  auto snap = perf::make_iq_cube_device<PerfPrecision>(
      {scen.detections, scen.channels, scen.snapshots});
  return perf::run_perf_graph(
      std::move(snap), total_frames, "root-music-perf",
      [&](holoscan::Graph& graph, const auto& src,
          holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_items = scen.detections;
        extents.max_sources = 4;
        extents.max_channels = scen.channels;
        auto op =
            graph.op<hr::RootMUSICEstimatorOp<PerfPrecision>>(
                "root_music", extents);
        hr::RootMUSICParams params;
        params.array_geometry =
            scenario_geometry(scen, 0.5, 1.0);
        params.angle_convention.preset =
            hr::AngleConventionPreset::kRadarBoresight;
        params.num_sources = 1;
        params.source_count_pfa = 1e-4;
        params.max_sources = 4;
        op.set_params(std::move(params));

        const auto sink =
            graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
        const auto ns1 =
            graph.op<perf::PerfNullSinkOp>("doa_drain");
        const auto ns2 =
            graph.op<perf::PerfNullSinkOp>("count_drain");
        const auto ns3 =
            graph.op<perf::PerfNullSinkOp>("eig_drain");
        const holoscan::ConnectionOptions q{
            .queue_depth = 4U};

        graph.add_flow(src->out, op->snapshot, q);
        graph.add_flow(
            op->aoa_estimates, sink->in, q);
        graph.add_flow(op->doa_multi, ns1->in, q);
        graph.add_flow(
            op->source_count, ns2->in, q);
        graph.add_flow(op->eigenvalues, ns3->in, q);

        perf::bind_tensor_out(
            options, "root_music", "aoa_estimates");
        perf::bind_tensor_out(
            options, "root_music", "doa_multi");
        perf::bind_tensor_out(
            options, "root_music", "source_count");
        perf::bind_tensor_out(
            options, "root_music", "eigenvalues");
      });
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
  row.variant = "root_music";
  row.size_id = scen.size_id;
  row.input_shape = {scen.detections, scen.channels,
                     scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  return row;
}

RootMusicScenario make_scenario(const SizeSpec& s) {
  RootMusicScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  std::ostringstream id;
  id << "root_music_" << s.id << "_c" << s.channels
     << "_n" << s.detections << "_s" << s.snapshots
     << "_compute";
  sc.id = id.str();
  return sc;
}

RootMusicScenario make_ura_scenario(const char* size_id,
                                    int64_t ny,
                                    int64_t nz,
                                    int64_t detections,
                                    int64_t snapshots) {
  RootMusicScenario sc;
  sc.size_id = size_id;
  sc.is_ura = true;
  sc.ny = ny;
  sc.nz = nz;
  sc.channels = ny * nz;
  sc.detections = detections;
  sc.snapshots = snapshots;
  std::ostringstream id;
  id << "root_music_ura_" << ny << "x" << nz
     << "_n" << detections << "_s" << snapshots;
  sc.id = id.str();
  return sc;
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
    return "Default size sweep (no live override)";
  }
  static const char* sizes_subset_help() {
    return "Size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full size ladder; live-override timing "
           "deferred on HSDK 5 EA (HSDK5_PORT.md §1)";
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
      out.push_back(
          make_scenario(must_find_size(sid)));
    }
    out.push_back(
        make_ura_scenario("ura_sm", 6, 6, 128, 96));
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
      out.push_back(make_scenario(*s));
    }
    return out;
  }

  static std::vector<RootMusicScenario>
  build_full_matrix() {
    std::vector<RootMusicScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    out.push_back(
        make_ura_scenario("ura_sm", 6, 6, 128, 96));
    out.push_back(
        make_ura_scenario("ura_lg", 8, 8, 256, 128));
    return out;
  }

  static void run_scenario(
      const RootMusicScenario& scen,
      const perf::RunSettings& rs,
      perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        rs, plan,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_root_music_graph(
              scen, p.total_frames);
        });

    perf::append_setup_row(
        base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "num_sources fixed=1 (deterministic model "
        "order); cold_ms is first-frame latency; "
        "median/p90 are post-warmup steady state; live "
        "override timing deferred (HSDK5_PORT.md §1); "
        "cells/s is detections/s";
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
          make_scenario(must_find_size("xs")),
          warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<RootMusicTraits>(
      argc, argv);
}
