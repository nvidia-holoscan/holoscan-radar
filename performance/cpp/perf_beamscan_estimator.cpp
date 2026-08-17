// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for BeamscanEstimatorOp — conventional
// sparse-path Bartlett DOA on a scan grid. Sweeps detection count, channel
// count, and scan-grid size. Measurement tool only: no timing assertions.
// See docs/planning/OPERATOR_DEVELOPMENT.md.
//
// [EA HSDK5 ONLY] Graph-based rewrite of the HSDK 4.x Application harness.
// Mid-session [live] override measurement is deferred until
// docs/planning/HSDK5_PORT.md §1 (EA partition / device binding). Params
// still select the scan grid under test. Optional is_valid input omitted
// (Graph optional mask deferred §1).

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

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
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
  int64_t num_az;
  int64_t num_el;
  std::string role;
};

struct BeamScenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t num_az{0};
  int64_t num_el{0};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 4, 64, 21, 11, "small array; fast smoke"},
      {"sm", 8, 256, 41, 21, "compact SIMO array"},
      {"md", 16, 1024, 61, 31, "common dev size"},
      {"lg", 32, 4096, 81, 41, "dense virtual array"},
      {"xl", 64, 16384, 101, 51,
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

perf::TimingCollector run_beamscan_graph(
    const BeamScenario& scen, int total_frames) {
  auto snap = perf::make_iq_cube_device<PerfPrecision>(
      {scen.detections, scen.channels});
  hr::MaxTensorExtents extents;
  extents.max_items = scen.detections;
  extents.max_az = scen.num_az;
  extents.max_el = scen.num_el;
  return perf::run_perf_graph(
      std::move(snap), total_frames, "beamscan-perf",
      [&](holoscan::Graph& graph, const auto& src,
          holoscan::CompileOptions& options) {
        auto op =
            graph.op<hr::BeamscanEstimatorOp<PerfPrecision>>(
                "beamscan", extents);
        hr::BeamscanParams params;
        params.array_geometry =
            make_ula_geometry(scen.channels, 0.5, 1.0);
        params.angle_convention.preset =
            hr::AngleConventionPreset::kRadarBoresight;
        params.az_angles =
            linspace(-0.5, 0.5, scen.num_az);
        params.el_angles =
            linspace(-0.3, 0.3, scen.num_el);
        params.peak_interpolation =
            hr::PeakInterpolation::kNone;
        op.set_params(std::move(params));
        const auto sink =
            graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
        const auto null =
            graph.op<perf::PerfNullSinkOp>("spectrum_drain");
        const holoscan::ConnectionOptions q{
            .queue_depth = 4U};
        graph.add_flow(src->out, op->snapshot, q);
        graph.add_flow(op->aoa_estimates, sink->in, q);
        graph.add_flow(
            op->angle_spectrum, null->in, q);
        perf::bind_tensor_out(
            options, "beamscan", "aoa_estimates");
        perf::bind_tensor_out(
            options, "beamscan", "angle_spectrum");
      });
}

void fill_throughput(perf::ResultRow* row,
                     const BeamScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  const double beams =
      static_cast<double>(scen.num_az) *
      static_cast<double>(scen.num_el);
  const double ch = static_cast<double>(scen.channels);
  row->cells_per_second = n / sec;
  row->slices_per_second = n * beams / sec;

  const double csz =
      2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double snap_bytes = n * ch * csz;
  const double weight_bytes = beams * ch * csz;
  const double spec_bytes =
      n * beams *
      static_cast<double>(sizeof(PerfPrecision));
  const double aoa_bytes =
      n * 2.0 *
      static_cast<double>(sizeof(PerfPrecision));
  row->estimated_effective_gbps =
      (snap_bytes + weight_bytes + spec_bytes +
       aoa_bytes) /
      sec / 1e9;
}

perf::ResultRow base_row(const BeamScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "BeamscanEstimatorOp";
  row.variant = "bartlett";
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.detections, scen.channels};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  return row;
}

BeamScenario make_scenario(const SizeSpec& s) {
  BeamScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  std::ostringstream id;
  id << "beamscan_" << s.id << "_c" << s.channels
     << "_n" << s.detections << "_g" << s.num_az
     << "x" << s.num_el << "_compute";
  sc.id = id.str();
  return sc;
}

struct BeamTraits : perf::PerfDriverDefaults {
  using Scenario = BeamScenario;

  static const char* tool_name() {
    return "perf_beamscan_estimator";
  }
  static const char* tool_description() {
    return "BeamscanEstimatorOp performance "
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
      def.shape = {s.num_az, s.num_el};
      def.cells = s.num_az * s.num_el;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<BeamScenario> build_default_matrix() {
    std::vector<BeamScenario> out;
    for (const char* sid : {"xs", "sm", "md"}) {
      out.push_back(
          make_scenario(must_find_size(sid)));
    }
    return out;
  }

  static std::vector<BeamScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<BeamScenario> out;
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

  static std::vector<BeamScenario> build_full_matrix() {
    std::vector<BeamScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    return out;
  }

  static void run_scenario(const BeamScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        rs, plan,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_beamscan_graph(
              scen, p.total_frames);
        });

    perf::append_setup_row(
        base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 "
        "are post-warmup steady state; live override "
        "timing deferred (HSDK5_PORT.md §1); cells/s "
        "is detections/s";
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
  return holoscan::radar::perf::perf_main<BeamTraits>(
      argc, argv);
}
