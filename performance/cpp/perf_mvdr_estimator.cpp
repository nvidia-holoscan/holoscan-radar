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
// The source exercises the production Exact diagonal_loading contract.

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

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
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
  int64_t num_az;
  int64_t num_el;
  std::string role;
};

struct MvdrScenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t snapshots{0};
  int64_t num_az{0};
  int64_t num_el{0};
};

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

perf::TimingCollector run_mvdr_graph(
    const MvdrScenario& scen, const perf::RunPlan& plan) {
  auto snap = perf::make_iq_cube_device<PerfPrecision>(
      {scen.detections, scen.channels, scen.snapshots});
  constexpr double kInitialLoading = -1.0;
  constexpr double kOverrideLoading = 1e-2;
  return perf::run_perf_graph_with_override(
      std::move(snap), plan.total_frames, plan.override_at, kInitialLoading,
      kOverrideLoading, "mvdr-perf",
      [&](holoscan::Graph& graph, const auto& src,
          holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_items = scen.detections;
        extents.max_az = scen.num_az;
        extents.max_el = scen.num_el;
        extents.max_sources = 4;
        extents.max_channels = scen.channels;
        auto op =
            graph.op<hr::MVDREstimatorOp<PerfPrecision>>("mvdr", extents);
        hr::MVDRParams params;
        params.array_geometry =
            make_ula_geometry(scen.channels, 0.5, 1.0);
        params.angle_convention.preset =
            hr::AngleConventionPreset::kRadarBoresight;
        params.az_angles =
            linspace(-0.5, 0.5, scen.num_az);
        params.el_angles =
            linspace(-0.3, 0.3, scen.num_el);
        params.diagonal_loading = kInitialLoading;
        op.set_params(std::move(params));

        const auto sink =
            graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
        const auto ns1 =
            graph.op<perf::PerfNullSinkOp>("spectrum_drain");
        const auto ns2 =
            graph.op<perf::PerfNullSinkOp>("doa_drain");
        const auto ns3 =
            graph.op<perf::PerfNullSinkOp>("count_drain");
        const auto ns4 =
            graph.op<perf::PerfNullSinkOp>("eig_drain");
        const holoscan::ConnectionOptions q{
            .queue_depth = 4U};

        graph.add_flow(src->out, op->snapshot, q);
        graph.add_flow(
            src->override_out, op->diagonal_loading, q);
        graph.add_flow(op->aoa_estimates, sink->in, q);
        graph.add_flow(
            op->angle_spectrum, ns1->in, q);
        graph.add_flow(op->doa_multi, ns2->in, q);
        graph.add_flow(op->source_count, ns3->in, q);
        graph.add_flow(op->eigenvalues, ns4->in, q);
        perf::bind_tensor_out(
            options, "mvdr", "aoa_estimates");
        perf::bind_tensor_out(
            options, "mvdr", "angle_spectrum");
        perf::bind_tensor_out(
            options, "mvdr", "doa_multi");
        perf::bind_tensor_out(
            options, "mvdr", "source_count");
        perf::bind_tensor_out(
            options, "mvdr", "eigenvalues");
      });
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
  row.variant = "capon";
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.detections, scen.channels,
                     scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  return row;
}

MvdrScenario make_scenario(const SizeSpec& s) {
  MvdrScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  std::ostringstream id;
  id << "mvdr_" << s.id << "_c" << s.channels
     << "_n" << s.detections << "_s" << s.snapshots
     << "_g" << s.num_az << "x" << s.num_el
     << "_compute";
  sc.id = id.str();
  return sc;
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
    return "Default size sweep with diagonal-loading override timing";
  }
  static const char* sizes_subset_help() {
    return "Size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full size ladder with diagonal-loading override timing";
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
      out.push_back(
          make_scenario(must_find_size(sid)));
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
      out.push_back(make_scenario(*s));
    }
    return out;
  }

  static std::vector<MvdrScenario> build_full_matrix() {
    std::vector<MvdrScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
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
        "cold_ms is first-frame latency; median/p90 "
        "are post-warmup steady state; override_ms is the peak after an "
        "inline diagonal-loading change; cells/s "
        "is detections/s; Capon scan dominates per "
        "tick";
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
  return holoscan::radar::perf::perf_main<MvdrTraits>(
      argc, argv);
}
