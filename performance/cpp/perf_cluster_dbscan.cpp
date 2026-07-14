// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ClusterDBSCANOp — brute-force O(N²·F)
// DBSCAN over a (num_points, num_features) feature matrix. Every compute run
// inlines a mid-session [live] override (epsilon / min_points / max_clusters).
// The workload is compact, well-separated blobs so the connected-components
// loop converges in a few iterations; N and F both drive cost (FeatureAssembly
// layouts are typically F∈{2,3,5}). Measurement tool only: no timing
// assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.

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

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
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

using PerfPrecision = float;

struct SizeSpec {
  std::string id;
  int64_t num_points;
  int64_t num_features;
  int64_t num_clusters;
  std::string role;
};

enum class OverrideKind {
  kEpsilon,
  kMinPoints,
  kMaxClusters,
};

struct DbscanScenario {
  std::string id;
  std::string size_id;
  int64_t num_points{0};
  int64_t num_features{2};
  int64_t num_clusters{1};
  std::vector<double> epsilon{0.5};
  int64_t min_points{0};
  int64_t max_clusters{0};
  // Filled per run for the inline live-override measurement.
  OverrideKind override_kind{OverrideKind::kEpsilon};
  int64_t override_at{0};
  std::vector<double> override_epsilon{0.6};
  int64_t override_min_points{0};
  int64_t override_max_clusters{0};
};

// O(N² · F) sweeps. N spans compact→dense detection clouds; F spans the
// FeatureAssembly layouts that actually feed this op (2 = range+vel,
// 3 = +score pre-angle, 5 = +az+el post-angle). Pairwise distance cost grows
// with F, so the ladder is intentionally (N, F)-diverse, not N-only.
const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 256, 2, 8, "small; fast smoke (R+V)"},
      {"sm", 1024, 2, 32, "compact 2-feature (R+V)"},
      {"sm3", 1024, 3, 32, "compact 3-feature (R+V+score)"},
      {"md", 2048, 2, 64, "mid-scale 2-feature"},
      {"md3", 2048, 3, 64, "mid-scale 3-feature (pre-angle)"},
      {"md5", 2048, 5, 64, "mid-scale 5-feature (post-angle)"},
      {"lg", 4096, 2, 128, "dense 2-feature cloud"},
      {"lg5", 4096, 5, 128, "dense 5-feature cloud"},
      {"xl", 8192, 2, 256, "upper N bound (O(N²), 2-feature)"},
      {"xl5", 8192, 5, 256, "upper N×F bound (O(N²F))"}};
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

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_from_host(
    const std::vector<T>& host, const std::vector<int64_t>& shape) {
  auto tensor = hr::make_device_tensor(shape, hr::real_dtype<T>());
  if (!host.empty()) {
    if (cudaMemcpy(tensor->data(), host.data(), sizeof(T) * host.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("perf_cluster_dbscan: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

// Deterministic compact blobs: cluster c centered at (c·100, 0, …) with a small
// (±0.1) per-axis jitter so within-blob points are eps-neighbors and separate
// blobs are not. No RNG (keeps runs reproducible).
template <typename T>
std::shared_ptr<holoscan::Tensor> make_points(int64_t num_points,
                                              int64_t num_features,
                                              int64_t num_clusters) {
  const int64_t nclus = std::max<int64_t>(num_clusters, 1);
  const double spacing = 100.0;
  std::vector<T> host(static_cast<size_t>(
      std::max<int64_t>(num_points, 0) * std::max<int64_t>(num_features, 0)));
  for (int64_t i = 0; i < num_points; ++i) {
    const int64_t c = i % nclus;
    for (int64_t f = 0; f < num_features; ++f) {
      const double base = (f == 0) ? static_cast<double>(c) * spacing : 0.0;
      const double jitter =
          static_cast<double>(((i * 131 + f * 17) % 21) - 10) / 100.0;
      host[static_cast<size_t>(i * num_features + f)] =
          static_cast<T>(base + jitter);
    }
  }
  return make_device_from_host(host, {num_points, num_features});
}

class DbscanPerfSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DbscanPerfSourceOp)
  DbscanPerfSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("points");
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
    op_output.emit(points_, "points");
  }

  void configure(perf::TimingCollector* collector, holoscan::TensorMap points) {
    collector_ = collector;
    points_ = std::move(points);
  }

 private:
  perf::TimingCollector* collector_{nullptr};
  holoscan::TensorMap points_;
};

template <typename T>
class DbscanPerfApp : public holoscan::Application {
 public:
  const DbscanScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap points;
  int total_frames{25};

  void compose() override {
    auto op = make_operator<hr::ClusterDBSCANOp<T>>(
        "dbscan", holoscan::Arg("epsilon") = scen->epsilon,
        holoscan::Arg("min_points") = scen->min_points,
        holoscan::Arg("max_clusters") = scen->max_clusters);

    auto src = make_operator<DbscanPerfSourceOp>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, points);
    add_flow(src, op, {{"points", "points"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("labels_sink");
    term->configure(collector);
    add_flow(op, term, {{"labels", "input"}});

    // sizes must be wired for the op to tick; it is not the timed result.
    auto sizes_drain = make_operator<perf::PerfNullSinkOp>("sizes_drain");
    add_flow(op, sizes_drain, {{"sizes", "input"}});

    wire_override(op);
  }

 private:
  void wire_override(const std::shared_ptr<holoscan::Operator>& op) {
    switch (scen->override_kind) {
      case OverrideKind::kEpsilon: {
        auto s = make_operator<perf::PerfOverrideSourceOp<std::vector<double>>>(
            "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_epsilon);
        add_flow(s, op, {{"output", "epsilon"}});
        break;
      }
      case OverrideKind::kMinPoints: {
        auto s = make_operator<perf::PerfOverrideSourceOp<int64_t>>(
            "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_min_points);
        add_flow(s, op, {{"output", "min_points"}});
        break;
      }
      case OverrideKind::kMaxClusters: {
        auto s = make_operator<perf::PerfOverrideSourceOp<int64_t>>(
            "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_max_clusters);
        add_flow(s, op, {{"output", "max_clusters"}});
        break;
      }
    }
  }
};

std::string override_tag(OverrideKind kind) {
  switch (kind) {
    case OverrideKind::kEpsilon: return "eps";
    case OverrideKind::kMinPoints: return "minpts";
    case OverrideKind::kMaxClusters: return "maxclus";
  }
  return "unk";
}

std::string override_param_name(OverrideKind kind) {
  switch (kind) {
    case OverrideKind::kEpsilon: return "epsilon";
    case OverrideKind::kMinPoints: return "min_points";
    case OverrideKind::kMaxClusters: return "max_clusters";
  }
  return "unknown";
}

void fill_throughput(perf::ResultRow* row, const DbscanScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.num_points);
  row->cells_per_second = n / sec;  // points/s
  row->slices_per_second =
      static_cast<double>(std::max<int64_t>(scen.num_clusters, 1)) / sec;

  const double tsz = sizeof(PerfPrecision);
  const double points_bytes = n * static_cast<double>(scen.num_features) * tsz;
  const double labels_bytes = n * static_cast<double>(sizeof(int64_t));
  row->estimated_effective_gbps =
      (points_bytes + labels_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const DbscanScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "ClusterDBSCANOp";
  row.variant = "dbscan";
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_points, scen.num_features};
  row.input_shape = {scen.num_points, scen.num_features};
  row.out_batch = scen.num_clusters;
  row.k = scen.num_features;
  row.notes = "live_override=" + override_param_name(scen.override_kind);
  return row;
}

DbscanScenario make_scenario(const SizeSpec& s,
                             OverrideKind ovr = OverrideKind::kEpsilon) {
  DbscanScenario sc;
  sc.size_id = s.id;
  sc.num_points = s.num_points;
  sc.num_features = s.num_features;
  sc.num_clusters = s.num_clusters;
  sc.epsilon = {0.5};
  sc.min_points = 2 * s.num_features;
  sc.max_clusters = 0;
  sc.override_kind = ovr;
  std::ostringstream id;
  id << "dbscan_" << s.id << "_n" << s.num_points << "_f" << s.num_features
     << "_" << override_tag(ovr);
  sc.id = id.str();
  return sc;
}

// ---------------------------------------------------------------------------
// Traits: the operator-specific surface the shared driver (perf_driver.hpp)
// plugs into. ClusterDBSCANOp has [live] override ports (exercised inline in
// run_scenario) and pays a one-time CUDA/Holoscan first-touch, so it overrides
// process_warmup; it takes the default no-op post_run and no failure isolation.
// ---------------------------------------------------------------------------
struct DbscanTraits : perf::PerfDriverDefaults {
  using Scenario = DbscanScenario;

  static const char* tool_name() { return "perf_cluster_dbscan"; }
  static const char* tool_description() {
    return "ClusterDBSCANOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Size sweep with an inline epsilon override";
  }
  static const char* sizes_subset_help() {
    return "Size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (sizes x {epsilon, min_points, max_clusters} "
           "overrides)";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_points, s.num_features};
      def.cells = s.num_points;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<DbscanScenario> build_default_matrix() {
    std::vector<DbscanScenario> out;
    // N ladder at F=2, plus mid-scale F=3/5 and dense F=5 so the default
    // report covers both axes without the full --sizes all matrix.
    for (const char* sid :
         {"xs", "sm", "sm3", "md", "md3", "md5", "lg", "lg5"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    // Distinct live params beyond the default epsilon override.
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(md, OverrideKind::kMinPoints));
    out.push_back(make_scenario(md, OverrideKind::kMaxClusters));
    return out;
  }

  static std::vector<DbscanScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<DbscanScenario> out;
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

  static std::vector<DbscanScenario> build_full_matrix() {
    std::vector<DbscanScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s, OverrideKind::kEpsilon));
      out.push_back(make_scenario(s, OverrideKind::kMinPoints));
      out.push_back(make_scenario(s, OverrideKind::kMaxClusters));
    }
    return out;
  }

  // Run one scenario (all repeats). Each run measures setup, pre-override
  // steady-state compute, and an inline [live] override spike in one session.
  static void run_scenario(const DbscanScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          DbscanScenario run_scen = scen;
          run_scen.override_at = p.override_at;
          switch (run_scen.override_kind) {
            case OverrideKind::kEpsilon:
              run_scen.override_epsilon = {0.6};  // re-resolve vector override
              break;
            case OverrideKind::kMinPoints:
              run_scen.override_min_points = run_scen.min_points + 1;
              break;
            case OverrideKind::kMaxClusters:
              // Cap below the blob count to exercise the drop-surplus path.
              run_scen.override_max_clusters =
                  std::max<int64_t>(1, run_scen.num_clusters / 2);
              break;
          }

          perf::TimingCollector collector;
          auto points_t = make_points<PerfPrecision>(
              scen.num_points, scen.num_features, scen.num_clusters);
          auto points = perf::single_map(std::move(points_t), "points");

          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<DbscanPerfApp<PerfPrecision>>();
          app->scen = &run_scen;
          app->collector = &collector;
          app->points = std::move(points);
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are pre-override "
        "steady state; override_ms is peak latency over the 3-frame "
        "live-" +
        override_param_name(scen.override_kind) +
        " window; cells/s is points/s";
    report->results.push_back(row);
  }

  // Pay CUDA + Holoscan first-Application costs so scenario #1 cold_ms is
  // comparable to later rows.
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
  return holoscan::radar::perf::perf_main<DbscanTraits>(argc, argv);
}
