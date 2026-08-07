// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for RootMUSICEstimatorOp — subspace,
// gridless (polynomial rooting) sparse-path DOA on a ULA. Per item it forms
// a snapshot covariance, eigendecomposes it (batched matx::eig, EigenMode::
// VECTOR), resolves a model order, builds the noise-subspace projector, then
// roots the conjugate-reciprocal polynomial via one companion-matrix
// cusolverDnXgeev call per item. Sweeps detection count, channel count, and
// snapshot count (no scan grid). Inlines
// a mid-session bundle of [live] overrides: array_geometry (slightly
// different spacing), num_sources (fixed 1 → auto/EFT), source_count_pfa,
// max_sources, and spatial_smoothing ([2] on the ULA). Measurement tool only:
// no timing assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.
//
// num_sources is fixed (=1) across the steady matrix rather than auto: this
// keeps the model order — and hence the noise-subspace projector's cost —
// deterministic across repeats, isolating the covariance/eig/projector/root
// path from the (cheap, separately-kernelled) EFT source-count decision.
//
// Smoke test (parent wires in performance/cpp/CMakeLists.txt):
//   add_executable(perf_root_music_estimator perf_root_music_estimator.cpp)
//   target_link_libraries(perf_root_music_estimator PRIVATE
//       radar_perf_utils holoscan::root_music_estimator_op)
//   add_test(NAME perf_root_music_estimator_smoke
//       COMMAND perf_root_music_estimator --sizes xs --warmup 1 --frames 2
//           --quiet
//           --output
//           ${CMAKE_CURRENT_BINARY_DIR}/perf_root_music_estimator_smoke.json)
//   set_tests_properties(perf_root_music_estimator_smoke PROPERTIES
//       LABELS "performance")

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_source.hpp"
#include "utils/perf_stats.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // v1 characterizes single precision.

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
  int64_t override_at{0};
  bool is_ura{false};  // URA exercises the native 2-D az+el rooting path.
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
      {"lg", 32, 512, 64, "wide aperture, many looks"},
      {"xl", 32, 1024, 96, "upper characterization bound"}};
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
    std::cerr << "internal error: unknown size id '" << id << "'\n";
    std::abort();
  }
  return *s;
}

hr::ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing,
                                    double wavelength) {
  hr::ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = hr::ArrayGeometryType::kUla;
  geom.element_positions.reserve(static_cast<size_t>(num_elements * 3));
  for (int64_t m = 0; m < num_elements; ++m) {
    geom.element_positions.push_back(0.0);
    geom.element_positions.push_back(static_cast<double>(m) * spacing);
    geom.element_positions.push_back(0.0);
  }
  return geom;
}

hr::ArrayGeometry make_ura_geometry(int64_t ny, int64_t nz, double spacing,
                                    double wavelength) {
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

hr::ArrayGeometry scenario_geometry(const RootMusicScenario& scen,
                                    double spacing, double wavelength) {
  if (scen.is_ura) {
    return make_ura_geometry(scen.ny, scen.nz, spacing, wavelength);
  }
  return make_ula_geometry(scen.channels, spacing, wavelength);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_snapshots(int64_t detections,
                                                 int64_t channels,
                                                 int64_t snapshots) {
  return perf::make_iq_cube_device<T>({detections, channels, snapshots});
}

class RootMusicPerfSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RootMusicPerfSourceOp)
  RootMusicPerfSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("snapshot");
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    const auto now = perf::Clock::now();
    if (collector_ != nullptr) {
      if (!collector_->first_tick_set) {
        collector_->t_first_tick = now;
        collector_->first_tick_set = true;
      }
      collector_->starts.push_back(now);
    }
    op_output.emit(snapshot_, "snapshot");
  }

  void configure(perf::TimingCollector* collector,
                 holoscan::TensorMap snapshot) {
    collector_ = collector;
    snapshot_ = std::move(snapshot);
  }

 private:
  perf::TimingCollector* collector_{nullptr};
  holoscan::TensorMap snapshot_;
};

template <typename T>
class RootMusicPerfApp : public holoscan::Application {
 public:
  const RootMusicScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap snapshot;
  hr::ArrayGeometry geometry;
  hr::AngleConvention convention;
  int total_frames{25};

  void compose() override {
    auto op = make_operator<hr::RootMUSICEstimatorOp<T>>(
        "root_music", holoscan::Arg("array_geometry") = geometry,
        holoscan::Arg("angle_convention") = convention,
        holoscan::Arg("num_sources") = int64_t{1},
        holoscan::Arg("source_count_pfa") = 1e-4,
        holoscan::Arg("max_sources") = int64_t{4});

    auto src = make_operator<RootMusicPerfSourceOp>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, snapshot);
    add_flow(src, op, {{"snapshot", "snapshot"}});

    auto aoa_sink = make_operator<perf::PerfTerminalSinkOp>("aoa_sink");
    aoa_sink->configure(collector);
    add_flow(op, aoa_sink, {{"aoa_estimates", "input"}});

    auto doa_drain = make_operator<perf::PerfNullSinkOp>("doa_multi_drain");
    add_flow(op, doa_drain, {{"doa_multi", "input"}});
    auto count_drain =
        make_operator<perf::PerfNullSinkOp>("source_count_drain");
    add_flow(op, count_drain, {{"source_count", "input"}});
    auto eig_drain = make_operator<perf::PerfNullSinkOp>("eigenvalues_drain");
    add_flow(op, eig_drain, {{"eigenvalues", "input"}});

    auto geometry_ovr =
        make_operator<perf::PerfOverrideSourceOp<hr::ArrayGeometry>>(
            "geometry_ovr",
            make_condition<holoscan::CountCondition>(total_frames));
    geometry_ovr->configure(
        scen->override_at,
        scenario_geometry(*scen, 0.55, geometry.wavelength));
    add_flow(geometry_ovr, op, {{"output", "array_geometry"}});

    auto sources_ovr =
        make_operator<perf::PerfOverrideSourceOp<int64_t>>(
            "sources_ovr",
            make_condition<holoscan::CountCondition>(total_frames));
    sources_ovr->configure(scen->override_at, int64_t{-1});
    add_flow(sources_ovr, op, {{"output", "num_sources"}});

    auto pfa_ovr = make_operator<perf::PerfOverrideSourceOp<double>>(
        "pfa_ovr", make_condition<holoscan::CountCondition>(total_frames));
    pfa_ovr->configure(scen->override_at, 0.5);
    add_flow(pfa_ovr, op, {{"output", "source_count_pfa"}});

    auto max_sources_ovr =
        make_operator<perf::PerfOverrideSourceOp<int64_t>>(
            "max_sources_ovr",
            make_condition<holoscan::CountCondition>(total_frames));
    max_sources_ovr->configure(scen->override_at, int64_t{3});
    add_flow(max_sources_ovr, op, {{"output", "max_sources"}});

    // Spatial smoothing collapses the URA's (iz, iy) lattice out of the
    // separable order the native 2-D path derives its invariances from, so it
    // is exercised only on the ULA (1-D) scenarios.
    if (!scen->is_ura) {
      auto smoothing_ovr =
          make_operator<perf::PerfOverrideSourceOp<std::vector<int64_t>>>(
              "smoothing_ovr",
              make_condition<holoscan::CountCondition>(total_frames));
      smoothing_ovr->configure(scen->override_at, std::vector<int64_t>{2});
      add_flow(smoothing_ovr, op, {{"output", "spatial_smoothing"}});
    }
  }
};

void fill_throughput(perf::ResultRow* row, const RootMusicScenario& scen,
                     double median_ms) {
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
  row.variant = "root_music";
  row.size_id = scen.size_id;
  row.input_shape = {scen.detections, scen.channels, scen.snapshots};
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
  id << "root_music_" << s.id << "_c" << s.channels << "_n" << s.detections
     << "_s" << s.snapshots;
  sc.id = id.str();
  return sc;
}

RootMusicScenario make_ura_scenario(const char* size_id, int64_t ny, int64_t nz,
                                    int64_t detections, int64_t snapshots) {
  RootMusicScenario sc;
  sc.size_id = size_id;
  sc.is_ura = true;
  sc.ny = ny;
  sc.nz = nz;
  sc.channels = ny * nz;
  sc.detections = detections;
  sc.snapshots = snapshots;
  std::ostringstream id;
  id << "root_music_ura_" << ny << "x" << nz << "_n" << detections << "_s"
     << snapshots;
  sc.id = id.str();
  return sc;
}

struct RootMusicTraits : perf::PerfDriverDefaults {
  using Scenario = RootMusicScenario;

  static const char* tool_name() { return "perf_root_music_estimator"; }
  static const char* tool_description() {
    return "RootMUSICEstimatorOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Default size sweep (inline live-override bundle)";
  }
  static const char* sizes_subset_help() {
    return "Size sweep across an explicit subset";
  }
  static const char* full_help() { return "Full size ladder"; }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.channels, s.detections, s.snapshots};
      def.cells = s.detections;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<RootMusicScenario> build_default_matrix() {
    std::vector<RootMusicScenario> out;
    for (const char* sid : {"xs", "sm", "md"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    // Native 2-D az+el path: a compact filled URA (per-axis rooting + pairing).
    out.push_back(make_ura_scenario("ura_sm", 6, 6, 128, 96));
    return out;
  }

  static std::vector<RootMusicScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<RootMusicScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(*s));
    }
    return out;
  }

  static std::vector<RootMusicScenario> build_full_matrix() {
    std::vector<RootMusicScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    // URA 2-D scenarios: per-axis rooting doubles the per-item geev calls, and
    // channel counts grow as ny * nz — characterize, don't gate.
    out.push_back(make_ura_scenario("ura_sm", 6, 6, 128, 96));
    out.push_back(make_ura_scenario("ura_lg", 8, 8, 256, 128));
    return out;
  }

  static void run_scenario(const RootMusicScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          RootMusicScenario run_scen = scen;
          run_scen.override_at = p.override_at;

          perf::TimingCollector collector;
          auto snap_t = make_snapshots<PerfPrecision>(
              scen.detections, scen.channels, scen.snapshots);

          collector.t_construct = perf::Clock::now();
          auto app =
              holoscan::make_application<RootMusicPerfApp<PerfPrecision>>();
          app->scen = &run_scen;
          app->collector = &collector;
          app->snapshot = perf::single_map(std::move(snap_t), "snapshot");
          app->geometry = scenario_geometry(scen, 0.5, 1.0);
          app->convention.preset = hr::AngleConventionPreset::kRadarBoresight;
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "num_sources fixed=1 (deterministic model order, isolates the "
        "covariance/eig(VECTOR)/projector/root path from EFT); cold_ms is "
        "first-frame latency; median/p90 are pre-override steady state; "
        "override_ms is peak latency over the live bundle (geometry spacing, "
        "num_sources fixed->auto+EFT, source_count_pfa, max_sources, "
        "spatial_smoothing); cells/s is detections/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([&] {
      perf::PerfReport discarded;
      perf::RunSettings warm_rs;
      warm_rs.warmup_frames = 0;
      warm_rs.measured_frames = 2;
      warm_rs.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs")), warm_rs, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<RootMusicTraits>(argc, argv);
}
