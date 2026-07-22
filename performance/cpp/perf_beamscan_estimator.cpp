// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for BeamscanEstimatorOp — conventional
// sparse-path Bartlett DOA on a scan grid. Sweeps detection count, channel
// count, and scan-grid size, and inlines a mid-session [live] az_angles
// override (finer grid → steering-weight rebuild). Measurement tool only: no
// timing assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
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
  int64_t override_at{0};
  int64_t override_num_az{0};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 4, 64, 21, 11, "small array; fast smoke"},
      {"sm", 8, 256, 41, 21, "compact SIMO array"},
      {"md", 16, 1024, 61, 31, "common dev size"},
      {"lg", 32, 4096, 81, 41, "dense virtual array"},
      {"xl", 64, 16384, 101, 51, "upper characterization bound"}};
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
std::shared_ptr<holoscan::Tensor> make_device_from_host(
    const std::vector<T>& host, const std::vector<int64_t>& shape,
    DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    if (cudaMemcpy(tensor->data(), host.data(), sizeof(T) * host.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "perf_beamscan_estimator: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_snapshots(int64_t detections,
                                                 int64_t channels) {
  return perf::make_iq_cube_device<T>({detections, channels});
}

class BeamscanPerfSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BeamscanPerfSourceOp)
  BeamscanPerfSourceOp() = default;

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
class BeamscanPerfApp : public holoscan::Application {
 public:
  const BeamScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap snapshot;
  hr::ArrayGeometry geometry;
  hr::AngleConvention convention;
  std::vector<double> az_grid;
  std::vector<double> el_grid;
  int total_frames{25};
  bool checksum{false};  // wire the byte-equivalence checksum sink

  void compose() override {
    auto op = make_operator<hr::BeamscanEstimatorOp<T>>(
        "beamscan", holoscan::Arg("array_geometry") = geometry,
        holoscan::Arg("angle_convention") = convention,
        holoscan::Arg("az_angles") = az_grid,
        holoscan::Arg("el_angles") = el_grid,
        holoscan::Arg("peak_interpolation") = hr::PeakInterpolation::kNone);

    auto src = make_operator<BeamscanPerfSourceOp>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, snapshot);
    add_flow(src, op, {{"snapshot", "snapshot"}});

    auto aoa_sink = make_operator<perf::PerfTerminalSinkOp>("aoa_sink");
    aoa_sink->configure(collector);
    add_flow(op, aoa_sink, {{"aoa_estimates", "input"}});

    // Byte-equivalence gate: fan the primary output out to a checksum sink too
    // (prints one output_checksum line on the first, pre-override frame).
    if (checksum) {
      auto cs = make_operator<perf::PerfChecksumSinkOp>("checksum_sink");
      add_flow(op, cs, {{"aoa_estimates", "input"}});
    }

    auto spec_drain = make_operator<perf::PerfNullSinkOp>("spectrum_drain");
    add_flow(op, spec_drain, {{"angle_spectrum", "input"}});

    auto ovr = make_operator<perf::PerfOverrideSourceOp<std::vector<double>>>(
        "ovr", make_condition<holoscan::CountCondition>(total_frames));
    ovr->configure(scen->override_at,
                   linspace(-0.5, 0.5, scen->override_num_az));
    add_flow(ovr, op, {{"output", "az_angles"}});
  }
};

void fill_throughput(perf::ResultRow* row, const BeamScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  const double beams =
      static_cast<double>(scen.num_az) * static_cast<double>(scen.num_el);
  const double ch = static_cast<double>(scen.channels);
  row->cells_per_second = n / sec;
  row->slices_per_second = n * beams / sec;

  const double complex_sz = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double snap_bytes = n * ch * complex_sz;
  const double weight_bytes = beams * ch * complex_sz;
  const double spec_bytes =
      n * beams * static_cast<double>(sizeof(PerfPrecision));
  const double aoa_bytes = n * 2.0 * static_cast<double>(sizeof(PerfPrecision));
  row->estimated_effective_gbps =
      (snap_bytes + weight_bytes + spec_bytes + aoa_bytes) / sec / 1e9;
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
  sc.override_num_az = s.num_az + 20;
  std::ostringstream id;
  id << "beamscan_" << s.id << "_c" << s.channels << "_n" << s.detections
     << "_g" << s.num_az << "x" << s.num_el;
  sc.id = id.str();
  return sc;
}

struct BeamTraits : perf::PerfDriverDefaults {
  using Scenario = BeamScenario;

  static const char* tool_name() { return "perf_beamscan_estimator"; }
  static const char* tool_description() {
    return "BeamscanEstimatorOp performance characterization";
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

  static std::vector<BeamScenario> build_default_matrix() {
    std::vector<BeamScenario> out;
    for (const char* sid : {"xs", "sm", "md"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    return out;
  }

  static std::vector<BeamScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<BeamScenario> out;
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
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          BeamScenario run_scen = scen;
          run_scen.override_at = p.override_at;

          perf::TimingCollector collector;
          auto snap_t = make_snapshots<PerfPrecision>(scen.detections,
                                                    scen.channels);

          collector.t_construct = perf::Clock::now();
          auto app =
              holoscan::make_application<BeamscanPerfApp<PerfPrecision>>();
          app->scen = &run_scen;
          app->collector = &collector;
          app->snapshot =
              perf::single_map(std::move(snap_t), "snapshot");
          app->geometry = make_ula_geometry(scen.channels, 0.5, 1.0);
          app->convention.preset = hr::AngleConventionPreset::kRadarBoresight;
          app->az_grid = linspace(-0.5, 0.5, scen.num_az);
          app->el_grid = linspace(-0.3, 0.3, scen.num_el);
          app->total_frames = p.total_frames;
          app->checksum = rs.checksum;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are pre-override steady "
        "state; override_ms is peak latency over the live-az_angles (finer "
        "grid) window; cells/s is detections/s";
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
  return holoscan::radar::perf::perf_main<BeamTraits>(argc, argv);
}
