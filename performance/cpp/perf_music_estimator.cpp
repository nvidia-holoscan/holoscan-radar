// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for MUSICEstimatorOp — subspace (MUSIC)
// sparse-path DOA. Per item it forms a snapshot covariance, eigendecomposes
// it (batched matx::eig, EigenMode::VECTOR — full eigenvector basis, unlike
// MVDR's NO_VECTOR), resolves a model order, builds the noise-subspace
// projector, then scans the (az, el) grid for the MUSIC pseudospectrum.
// Sweeps detection count, channel count, snapshot count, and scan-grid size,
// and inlines a mid-session [live] az_angles override (finer grid → steering
// rebuild), mirroring perf_mvdr_estimator.cpp. Measurement tool only: no
// timing assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.
//
// num_sources is fixed (=1) across the matrix rather than auto: this keeps
// the model order — and hence the noise-subspace projector's cost —
// deterministic across repeats, isolating the covariance/eig/projector/scan
// path from the (cheap, separately-kernelled) EFT source-count decision.

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

#include "angle_estimation/music_estimator_op/music_estimator_op.hpp"
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
  int64_t num_az;
  int64_t num_el;
  std::string role;
};

struct MusicScenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t snapshots{0};
  int64_t num_az{0};
  int64_t num_el{0};
  int64_t override_at{0};
  int64_t override_num_az{0};
};

const std::vector<SizeSpec>& size_ladder() {
  // Same ladder as MVDR (covariance + eig + projector scratch is
  // O(detections · channels²), same shape family).
  static const std::vector<SizeSpec> ladder = {
      {"xs", 4, 64, 16, 21, 11, "small array; fast smoke"},
      {"sm", 8, 128, 32, 41, 21, "compact SIMO array"},
      {"md", 16, 256, 48, 61, 31, "common dev size"},
      {"lg", 32, 512, 64, 81, 41, "wide aperture, many looks"},
      {"xl", 32, 1024, 96, 101, 51, "upper characterization bound"}};
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

std::vector<double> linspace(double start, double stop, int64_t n) {
  std::vector<double> out(static_cast<size_t>(n));
  if (n == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(n - 1);
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = start + step * static_cast<double>(i);
  }
  return out;
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

template <typename T>
std::shared_ptr<holoscan::Tensor> make_snapshots(int64_t detections,
                                                 int64_t channels,
                                                 int64_t snapshots) {
  return perf::make_iq_cube_device<T>({detections, channels, snapshots});
}

class MusicPerfSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MusicPerfSourceOp)
  MusicPerfSourceOp() = default;

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
class MusicPerfApp : public holoscan::Application {
 public:
  const MusicScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap snapshot;
  hr::ArrayGeometry geometry;
  hr::AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;
  int total_frames{25};

  void compose() override {
    auto op = make_operator<hr::MUSICEstimatorOp<T>>(
        "music", holoscan::Arg("array_geometry") = geometry,
        holoscan::Arg("angle_convention") = convention,
        holoscan::Arg("az_angles") = az_grid,
        holoscan::Arg("el_angles") = el_grid,
        holoscan::Arg("num_sources") = int64_t{1});

    auto src = make_operator<MusicPerfSourceOp>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, snapshot);
    add_flow(src, op, {{"snapshot", "snapshot"}});

    auto aoa_sink = make_operator<perf::PerfTerminalSinkOp>("aoa_sink");
    aoa_sink->configure(collector);
    add_flow(op, aoa_sink, {{"aoa_estimates", "input"}});

    auto spec_drain = make_operator<perf::PerfNullSinkOp>("spectrum_drain");
    add_flow(op, spec_drain, {{"angle_spectrum", "input"}});
    auto doa_drain = make_operator<perf::PerfNullSinkOp>("doa_multi_drain");
    add_flow(op, doa_drain, {{"doa_multi", "input"}});
    auto count_drain =
        make_operator<perf::PerfNullSinkOp>("source_count_drain");
    add_flow(op, count_drain, {{"source_count", "input"}});
    auto eig_drain = make_operator<perf::PerfNullSinkOp>("eigenvalues_drain");
    add_flow(op, eig_drain, {{"eigenvalues", "input"}});

    auto ovr = make_operator<perf::PerfOverrideSourceOp<std::vector<double>>>(
        "ovr", make_condition<holoscan::CountCondition>(total_frames));
    ovr->configure(scen->override_at,
                   linspace(-0.5, 0.5, scen->override_num_az));
    add_flow(ovr, op, {{"output", "az_angles"}});
  }
};

void fill_throughput(perf::ResultRow* row, const MusicScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  row->cells_per_second = n / sec;
  const double beams =
      static_cast<double>(scen.num_az) * static_cast<double>(scen.num_el);
  row->slices_per_second = n * beams / sec;
}

perf::ResultRow base_row(const MusicScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "MUSICEstimatorOp";
  row.variant = "music";
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.detections, scen.channels, scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  return row;
}

MusicScenario make_scenario(const SizeSpec& s) {
  MusicScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  sc.override_num_az = s.num_az + 20;
  std::ostringstream id;
  id << "music_" << s.id << "_c" << s.channels << "_n" << s.detections << "_s"
     << s.snapshots << "_g" << s.num_az << "x" << s.num_el;
  sc.id = id.str();
  return sc;
}

struct MusicTraits : perf::PerfDriverDefaults {
  using Scenario = MusicScenario;

  static const char* tool_name() { return "perf_music_estimator"; }
  static const char* tool_description() {
    return "MUSICEstimatorOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Default size sweep (inline az_angles override)";
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
      def.shape = {s.num_az, s.num_el};
      def.cells = s.num_az * s.num_el;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<MusicScenario> build_default_matrix() {
    std::vector<MusicScenario> out;
    for (const char* sid : {"xs", "sm", "md"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    return out;
  }

  static std::vector<MusicScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<MusicScenario> out;
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

  static std::vector<MusicScenario> build_full_matrix() {
    std::vector<MusicScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    return out;
  }

  static void run_scenario(const MusicScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          MusicScenario run_scen = scen;
          run_scen.override_at = p.override_at;

          perf::TimingCollector collector;
          auto snap_t = make_snapshots<PerfPrecision>(
              scen.detections, scen.channels, scen.snapshots);

          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<MusicPerfApp<PerfPrecision>>();
          app->scen = &run_scen;
          app->collector = &collector;
          app->snapshot = perf::single_map(std::move(snap_t), "snapshot");
          app->geometry = make_ula_geometry(scen.channels, 0.5, 1.0);
          app->convention.preset = hr::AngleConventionPreset::kRadarBoresight;
          app->az_grid = linspace(-0.5, 0.5, scen.num_az);
          app->el_grid = linspace(-0.3, 0.3, scen.num_el);
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
        "covariance/eig(VECTOR)/projector/scan path from EFT); cold_ms is "
        "first-frame latency; median/p90 are pre-override steady state; "
        "override_ms is peak latency over the live-az_angles (finer grid) "
        "window; cells/s is detections/s";
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
  return holoscan::radar::perf::perf_main<MusicTraits>(argc, argv);
}
