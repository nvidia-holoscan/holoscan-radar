// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for SnapshotGatherOp — the sparse-path bridge
// that gathers a per-detection channel-vector neighborhood out of per-channel
// RD maps. Sweeps channel count, RD-map size, detection count, and snapshot
// window, and inlines a mid-session [live] snapshot_window override (the
// interesting cost: the window grows, so the output re-shapes and the gather
// re-plans). Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.

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

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
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

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

struct SizeSpec {
  std::string id;
  int64_t channels;
  int64_t doppler;
  int64_t range;
  int64_t detections;
  std::string role;
};

// A snapshot window is a (doppler, range) half-extent; the gather produces
// (2*wd + 1) * (2*wr + 1) snapshots per channel per detection.
struct Window {
  int64_t wd{0};
  int64_t wr{0};
};

struct SnapScenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t doppler{0};
  int64_t range{0};
  int64_t detections{0};
  Window window;
  // Filled per run by run_scenario() for inline live-override measurement: the
  // window grows one cell on each axis, forcing an output re-shape + re-plan.
  int64_t override_at{0};
  Window override_window;
};

int64_t snapshots_for(const Window& w) {
  return (2 * w.wd + 1) * (2 * w.wr + 1);
}

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 4, 64, 256, 64, "small map; fast smoke"},
      {"sm", 8, 128, 512, 256, "compact SIMO array"},
      {"md", 16, 256, 512, 1024, "common dev size"},
      {"lg", 32, 512, 1024, 4096, "dense virtual array"},
      {"xl", 64, 512, 2048, 16384, "upper characterization bound"}};
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

std::string window_tag(const Window& w) {
  std::ostringstream os;
  os << "w" << w.wd << "x" << w.wr;
  return os.str();
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_from_host(
    const std::vector<T>& host, const std::vector<int64_t>& shape,
    DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    if (cudaMemcpy(tensor->data(), host.data(), sizeof(T) * host.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("perf_snapshot_gather: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

// Per-channel RD maps: (num_channels, num_doppler, num_range) complex. The
// deterministic IQ payload gives every cell distinct, reproducible content.
template <typename T>
std::shared_ptr<holoscan::Tensor> make_rd_maps(int64_t channels,
                                               int64_t doppler,
                                               int64_t range) {
  return perf::make_iq_cube_device<T>({channels, doppler, range});
}

// Deterministic (doppler_bin, range_bin) detection grid across the RD plane
// (wraps if detections > cells), so the gather does non-trivial, reproducible
// work every frame.
std::shared_ptr<holoscan::Tensor> make_detection_idx(int64_t detections,
                                                     int64_t doppler,
                                                     int64_t range) {
  std::vector<int64_t> host(
      static_cast<size_t>(std::max<int64_t>(detections, 0) * 2), 0);
  const int64_t cells = std::max<int64_t>(doppler * range, 1);
  for (int64_t i = 0; i < detections; ++i) {
    const int64_t cell = (i * 997) % cells;
    host[static_cast<size_t>(i * 2)] = cell / range;
    host[static_cast<size_t>(i * 2 + 1)] = cell % range;
  }
  return make_device_from_host(host, {detections, 2}, kInt64Dtype);
}

// Multi-port source: emits the same preallocated TensorMaps every tick so
// measured latency is dominated by SnapshotGatherOp, not H2D.
class SnapGatherPerfSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SnapGatherPerfSourceOp)
  SnapGatherPerfSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("rd_maps");
    spec.output<holoscan::TensorMap>("detection_idx");
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
    op_output.emit(rd_maps_, "rd_maps");
    op_output.emit(idx_, "detection_idx");
  }

  void configure(perf::TimingCollector* collector, holoscan::TensorMap rd_maps,
                 holoscan::TensorMap idx) {
    collector_ = collector;
    rd_maps_ = std::move(rd_maps);
    idx_ = std::move(idx);
  }

 private:
  perf::TimingCollector* collector_{nullptr};
  holoscan::TensorMap rd_maps_;
  holoscan::TensorMap idx_;
};

template <typename T>
class SnapGatherPerfApp : public holoscan::Application {
 public:
  const SnapScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap rd_maps;
  holoscan::TensorMap idx;
  int total_frames{25};

  void compose() override {
    auto op = make_operator<hr::SnapshotGatherOp<T>>(
        "gather",
        holoscan::Arg("snapshot_window") =
            std::vector<int64_t>{scen->window.wd, scen->window.wr},
        holoscan::Arg("doppler_edge") = hr::SnapshotEdgeMode::kWrap,
        holoscan::Arg("range_edge") = hr::SnapshotEdgeMode::kShift);

    auto src = make_operator<SnapGatherPerfSourceOp>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, rd_maps, idx);
    add_flow(src, op, {{"rd_maps", "rd_maps"}});
    add_flow(src, op, {{"detection_idx", "detection_idx"}});

    // Sparse output goes to the timed terminal sink; is_valid (kDLBool, same
    // transport caveat as CFAR detection_mask) drains through a null sink.
    auto term = make_operator<perf::PerfTerminalSinkOp>("snapshot_sink");
    term->configure(collector);
    add_flow(op, term, {{"snapshot", "input"}});

    auto valid_drain = make_operator<perf::PerfNullSinkOp>("valid_drain");
    add_flow(op, valid_drain, {{"is_valid", "input"}});

    auto ovr = make_operator<perf::PerfOverrideSourceOp<std::vector<int64_t>>>(
        "ovr", make_condition<holoscan::CountCondition>(total_frames));
    ovr->configure(scen->override_at, std::vector<int64_t>{
                                          scen->override_window.wd,
                                          scen->override_window.wr});
    add_flow(ovr, op, {{"output", "snapshot_window"}});
  }
};

void fill_throughput(perf::ResultRow* row, const SnapScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  const double snaps = static_cast<double>(snapshots_for(scen.window));
  const double gathered = n * static_cast<double>(scen.channels) * snaps;

  row->cells_per_second = n / sec;              // detections/s (sparse axis)
  row->slices_per_second = n * snaps / sec;     // gathered snapshots/s

  // Effective traffic: read each gathered complex sample from the RD maps,
  // write it to the snapshot matrix, write the bool validity flag, and read
  // the detection index rows.
  const double complex_sz = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double idx_bytes = n * 2.0 * static_cast<double>(sizeof(int64_t));
  const double valid_bytes = n * snaps;  // bool (1 byte/elem)
  row->estimated_effective_gbps =
      (2.0 * gathered * complex_sz + valid_bytes + idx_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const SnapScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "SnapshotGatherOp";
  row.variant = window_tag(scen.window);
  row.size_id = scen.size_id;
  row.map_shape = {scen.doppler, scen.range};
  row.input_shape = {scen.detections, 2};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  return row;
}

SnapScenario make_scenario(const SizeSpec& s, Window w) {
  SnapScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.doppler = s.doppler;
  sc.range = s.range;
  sc.detections = s.detections;
  sc.window = w;
  std::ostringstream id;
  id << "snap_gather_" << s.id << "_c" << s.channels << "_n" << s.detections
     << "_" << window_tag(w);
  sc.id = id.str();
  return sc;
}

// ---------------------------------------------------------------------------
// Traits: the operator-specific surface the shared driver (perf_driver.hpp)
// plugs into. SnapshotGatherOp has a [live] snapshot_window override port
// (exercised inline in run_scenario) and benefits from a process warmup, so it
// overrides process_warmup; it takes the default no-op post_run and the
// default (no) failure isolation.
// ---------------------------------------------------------------------------
struct SnapTraits : perf::PerfDriverDefaults {
  using Scenario = SnapScenario;

  static const char* tool_name() { return "perf_snapshot_gather"; }
  static const char* tool_description() {
    return "SnapshotGatherOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Window w1x1 size sweep (inline snapshot_window override)";
  }
  static const char* sizes_subset_help() {
    return "Window w1x1 size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (sizes x windows {0x0, 1x1, 2x2})";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.doppler, s.range};
      def.cells = s.doppler * s.range;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<SnapScenario> build_default_matrix() {
    std::vector<SnapScenario> out;
    const SizeSpec& md = must_find_size("md");
    for (const Window w : {Window{0, 0}, Window{1, 1}, Window{2, 2}}) {
      out.push_back(make_scenario(md, w));
    }
    for (const char* sid : {"xs", "sm", "md", "lg"}) {
      out.push_back(make_scenario(must_find_size(sid), Window{1, 1}));
    }
    return out;
  }

  static std::vector<SnapScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<SnapScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(*s, Window{1, 1}));
    }
    return out;
  }

  static std::vector<SnapScenario> build_full_matrix() {
    std::vector<SnapScenario> out;
    for (const auto& s : size_ladder()) {
      for (const Window w : {Window{0, 0}, Window{1, 1}, Window{2, 2}}) {
        out.push_back(make_scenario(s, w));
      }
    }
    return out;
  }

  // Run one scenario (all repeats). Each run measures setup, pre-override
  // steady-state compute, and an inline [live] snapshot_window spike (window
  // grows one cell per axis → output re-shape + gather re-plan) in one session.
  static void run_scenario(const SnapScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          SnapScenario run_scen = scen;
          run_scen.override_at = p.override_at;
          run_scen.override_window = Window{scen.window.wd + 1,
                                            scen.window.wr + 1};

          perf::TimingCollector collector;
          auto rd_maps_t =
              make_rd_maps<PerfPrecision>(scen.channels, scen.doppler,
                                          scen.range);
          auto idx_t =
              make_detection_idx(scen.detections, scen.doppler, scen.range);

          auto rd_maps = perf::single_map(std::move(rd_maps_t), "rd_maps");
          auto idx = perf::single_map(std::move(idx_t), "detection_idx");

          collector.t_construct = perf::Clock::now();
          auto app =
              holoscan::make_application<SnapGatherPerfApp<PerfPrecision>>();
          app->scen = &run_scen;
          app->collector = &collector;
          app->rd_maps = std::move(rd_maps);
          app->idx = std::move(idx);
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are pre-override steady "
        "state; override_ms is peak latency over the 3-frame live-"
        "snapshot_window (window +1/axis) window; cells/s is detections/s";
    report->results.push_back(row);
  }

  // Pay CUDA + Holoscan first-Application costs (and first-touch the gather
  // kernel) so scenario #1 cold_ms is comparable to later rows.
  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([&] {
      perf::PerfReport discarded;
      perf::RunSettings warm_rs;
      warm_rs.warmup_frames = 0;
      warm_rs.measured_frames = 2;
      warm_rs.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs"), Window{1, 1}), warm_rs,
                   &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<SnapTraits>(argc, argv);
}
