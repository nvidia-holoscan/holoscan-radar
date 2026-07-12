// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for DetectionReductionOp — group-by-label
// reduction over (labels, detection_idx, points). Every compute run inlines a
// mid-session [live] override (representative / feature_weights /
// score_method). Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_source.hpp"
#include "utils/perf_stats.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

struct SizeSpec {
  std::string id;
  int64_t num_points;
  int64_t num_clusters;
  int64_t num_features;
  std::string role;
};

enum class OverrideKind {
  kRepresentative,
  kFeatureWeights,
  kScoreMethod,
};

struct RedScenario {
  std::string id;
  std::string size_id;
  hr::RepresentativePolicy policy{hr::RepresentativePolicy::kPeak};
  hr::ScoreMethod score_method{hr::ScoreMethod::kWeightedSum};
  int64_t num_points{0};
  int64_t num_clusters{0};
  int64_t num_features{2};
  // Filled per run by run_scenario() for inline live-override measurement.
  OverrideKind override_kind{OverrideKind::kRepresentative};
  int64_t override_at{0};
  hr::RepresentativePolicy override_policy{hr::RepresentativePolicy::kFirst};
  hr::ScoreMethod override_score_method{hr::ScoreMethod::kWeightedL2};
  std::vector<double> override_weights;
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 256, 32, 2, "small; fast smoke"},
      {"sm", 1024, 64, 2, "compact detection set"},
      {"md", 4096, 128, 2, "common mid-scale"},
      {"lg", 16384, 256, 2, "dense cluster-then-angle"},
      {"xl", 65536, 512, 2, "upper characterization bound"}};
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

std::string policy_tag(hr::RepresentativePolicy p) {
  return std::string(hr::to_string(p));
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
          "perf_detection_reduction: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

// Deterministic labels: ~1/8 noise, remaining round-robin across clusters.
std::shared_ptr<holoscan::Tensor> make_labels(int64_t num_points,
                                              int64_t num_clusters) {
  std::vector<int64_t> host(
      static_cast<size_t>(std::max<int64_t>(num_points, 0)));
  const int64_t nclus = std::max<int64_t>(num_clusters, 1);
  for (int64_t i = 0; i < num_points; ++i) {
    host[static_cast<size_t>(i)] =
        (i % 8 == 0) ? -1 : (i % nclus);
  }
  return make_device_from_host(host, {num_points}, kInt64Dtype);
}

std::shared_ptr<holoscan::Tensor> make_detection_idx(int64_t num_points) {
  std::vector<int64_t> host(
      static_cast<size_t>(std::max<int64_t>(num_points, 0) * 2));
  for (int64_t i = 0; i < num_points; ++i) {
    host[static_cast<size_t>(i * 2)] = i % 1024;
    host[static_cast<size_t>(i * 2 + 1)] = (i * 3) % 2048;
  }
  return make_device_from_host(host, {num_points, 2}, kInt64Dtype);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_points(int64_t num_points,
                                              int64_t num_features) {
  std::vector<T> host(static_cast<size_t>(
      std::max<int64_t>(num_points, 0) * std::max<int64_t>(num_features, 0)));
  for (int64_t i = 0; i < num_points; ++i) {
    for (int64_t f = 0; f < num_features; ++f) {
      host[static_cast<size_t>(i * num_features + f)] =
          static_cast<T>((i * 17 + f * 3) % 100) * static_cast<T>(0.01);
    }
  }
  return make_device_from_host(host, {num_points, num_features},
                               hr::real_dtype<T>());
}

class ReductionPerfSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ReductionPerfSourceOp)
  ReductionPerfSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("labels");
    spec.output<holoscan::TensorMap>("detection_idx");
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
    op_output.emit(labels_, "labels");
    op_output.emit(idx_, "detection_idx");
    op_output.emit(points_, "points");
  }

  void configure(perf::TimingCollector* collector, holoscan::TensorMap labels,
                 holoscan::TensorMap idx, holoscan::TensorMap points) {
    collector_ = collector;
    labels_ = std::move(labels);
    idx_ = std::move(idx);
    points_ = std::move(points);
  }

 private:
  perf::TimingCollector* collector_{nullptr};
  holoscan::TensorMap labels_;
  holoscan::TensorMap idx_;
  holoscan::TensorMap points_;
};

template <typename T>
class ReductionPerfApp : public holoscan::Application {
 public:
  const RedScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap labels;
  holoscan::TensorMap idx;
  holoscan::TensorMap points;
  int total_frames{25};

  void compose() override {
    // Always supply feature_weights + wire points so a mid-run override to
    // peak/medoid (or a weights/score_method flip) is legal without rewiring.
    std::vector<double> weights(static_cast<size_t>(scen->num_features), 1.0);
    auto op = make_operator<hr::DetectionReductionOp<T>>(
        "reduction",
        holoscan::Arg("representative") = scen->policy,
        holoscan::Arg("score_method") = scen->score_method,
        holoscan::Arg("feature_weights") = weights);

    auto src = make_operator<ReductionPerfSourceOp>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, labels, idx, points);
    add_flow(src, op, {{"labels", "labels"}});
    add_flow(src, op, {{"detection_idx", "detection_idx"}});
    add_flow(src, op, {{"points", "points"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("rep_sink");
    term->configure(collector);
    add_flow(op, term, {{"rep_idx", "input"}});

    wire_override(op);
  }

 private:
  void wire_override(const std::shared_ptr<holoscan::Operator>& op) {
    switch (scen->override_kind) {
      case OverrideKind::kRepresentative: {
        auto s = make_operator<
            perf::PerfOverrideSourceOp<hr::RepresentativePolicy>>(
            "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_policy);
        add_flow(s, op, {{"output", "representative"}});
        break;
      }
      case OverrideKind::kFeatureWeights: {
        auto s = make_operator<
            perf::PerfOverrideSourceOp<std::vector<double>>>(
            "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_weights);
        add_flow(s, op, {{"output", "feature_weights"}});
        break;
      }
      case OverrideKind::kScoreMethod: {
        auto s =
            make_operator<perf::PerfOverrideSourceOp<hr::ScoreMethod>>(
                "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_score_method);
        add_flow(s, op, {{"output", "score_method"}});
        break;
      }
    }
  }
};

void fill_throughput(perf::ResultRow* row, const RedScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.num_points);
  row->cells_per_second = n / sec;
  row->slices_per_second =
      static_cast<double>(std::max<int64_t>(scen.num_clusters, 1)) / sec;

  const double tsz = sizeof(PerfPrecision);
  const double labels_bytes = n * static_cast<double>(sizeof(int64_t));
  const double idx_bytes = n * 2.0 * static_cast<double>(sizeof(int64_t));
  // Points stay wired for every scenario (override may switch to peak/medoid).
  const double points_bytes =
      n * static_cast<double>(scen.num_features) * tsz;
  const double rep_bytes =
      static_cast<double>(scen.num_clusters) * 2.0 *
      static_cast<double>(sizeof(int64_t));
  row->estimated_effective_gbps =
      (labels_bytes + idx_bytes + points_bytes + rep_bytes) / sec / 1e9;
}

std::string override_tag(OverrideKind kind) {
  switch (kind) {
    case OverrideKind::kRepresentative: return "rep";
    case OverrideKind::kFeatureWeights: return "wts";
    case OverrideKind::kScoreMethod: return "score";
  }
  return "unk";
}

std::string override_param_name(OverrideKind kind) {
  switch (kind) {
    case OverrideKind::kRepresentative: return "representative";
    case OverrideKind::kFeatureWeights: return "feature_weights";
    case OverrideKind::kScoreMethod: return "score_method";
  }
  return "unknown";
}

hr::RepresentativePolicy flip_policy(hr::RepresentativePolicy p) {
  // Prefer a flip that changes the kernel path (first ↔ peak/medoid).
  if (p == hr::RepresentativePolicy::kFirst) {
    return hr::RepresentativePolicy::kPeak;
  }
  return hr::RepresentativePolicy::kFirst;
}

perf::ResultRow base_row(const RedScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "DetectionReductionOp";
  row.variant = policy_tag(scen.policy);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_points, scen.num_features};
  row.input_shape = {scen.num_points, 2};
  row.out_batch = scen.num_clusters;
  row.k = scen.num_features;
  row.notes = "live_override=" + override_param_name(scen.override_kind);
  return row;
}

// Run one scenario (all repeats). Each run measures setup, pre-override
// steady-state compute, and an inline [live] override spike in one session.
void run_scenario(const RedScenario& scen, const perf::RunSettings& rs,
                  perf::PerfReport* report) {
  constexpr int kOverrideSpikeFrames = 3;
  const int64_t override_at =
      rs.warmup_frames + std::max(2, rs.measured_frames / 2);
  const int total =
      std::max(rs.warmup_frames + rs.measured_frames,
               static_cast<int>(override_at) + kOverrideSpikeFrames);

  std::vector<double> setup_samples;
  std::vector<double> cold_samples;
  std::vector<double> steady_samples;
  double override_ms_best = std::numeric_limits<double>::quiet_NaN();

  for (int rep = 0; rep < rs.repeats; ++rep) {
    RedScenario run_scen = scen;
    run_scen.override_at = override_at;
    if (run_scen.override_kind == OverrideKind::kRepresentative) {
      run_scen.override_policy = flip_policy(run_scen.policy);
    } else if (run_scen.override_kind == OverrideKind::kFeatureWeights) {
      run_scen.override_weights.assign(
          static_cast<size_t>(run_scen.num_features), 0.0);
      if (!run_scen.override_weights.empty()) {
        run_scen.override_weights[0] = 1.0;
      }
    } else if (run_scen.override_kind == OverrideKind::kScoreMethod) {
      run_scen.override_score_method = hr::ScoreMethod::kWeightedL2;
    }

    perf::TimingCollector collector;
    auto labels_t = make_labels(scen.num_points, scen.num_clusters);
    auto idx_t = make_detection_idx(scen.num_points);
    auto points_t =
        make_points<PerfPrecision>(scen.num_points, scen.num_features);

    auto labels = perf::single_map(std::move(labels_t), "labels");
    auto idx = perf::single_map(std::move(idx_t), "detection_idx");
    auto points = perf::single_map(std::move(points_t), "points");

    collector.t_construct = perf::Clock::now();
    auto app = holoscan::make_application<ReductionPerfApp<PerfPrecision>>();
    app->scen = &run_scen;
    app->collector = &collector;
    app->labels = std::move(labels);
    app->idx = std::move(idx);
    app->points = std::move(points);
    app->total_frames = total;
    app->run();

    setup_samples.push_back(collector.setup_ms());
    const auto lat = collector.frame_latencies_ms();
    if (!lat.empty()) {
      cold_samples.push_back(lat.front());
    }
    for (size_t i = static_cast<size_t>(rs.warmup_frames); i < lat.size();
         ++i) {
      const int64_t frame = static_cast<int64_t>(i);
      if (frame < override_at) {
        steady_samples.push_back(lat[i]);
      } else if (frame < override_at + kOverrideSpikeFrames) {
        const double v = lat[i];
        if (std::isnan(override_ms_best) || v > override_ms_best) {
          override_ms_best = v;
        }
      }
    }
  }

  {
    perf::ResultRow row = base_row(scen);
    row.bucket = "setup";
    row.setup_ms = perf::summarize(setup_samples).median_ms;
    report->results.push_back(row);
  }
  {
    perf::ResultRow row = base_row(scen);
    row.bucket = "compute";
    row.cold_ms = perf::summarize(cold_samples).median_ms;
    row.latency = perf::summarize(steady_samples);
    row.override_ms = override_ms_best;
    if (!std::isnan(override_ms_best) && !std::isnan(row.latency.median_ms)) {
      row.override_delta_ms = override_ms_best - row.latency.median_ms;
    }
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are pre-override "
        "steady state; override_ms is peak latency over the 3-frame "
        "live-" +
        override_param_name(scen.override_kind) +
        " window; cells/s is points/s";
    report->results.push_back(row);
  }
}

RedScenario make_scenario(hr::RepresentativePolicy policy, const SizeSpec& s,
                          OverrideKind ovr = OverrideKind::kRepresentative) {
  RedScenario sc;
  sc.policy = policy;
  sc.size_id = s.id;
  sc.num_points = s.num_points;
  sc.num_clusters = s.num_clusters;
  sc.num_features = s.num_features;
  sc.override_kind = ovr;
  // Weights / score_method overrides need a points-consuming policy.
  if (ovr == OverrideKind::kFeatureWeights ||
      ovr == OverrideKind::kScoreMethod) {
    sc.policy = hr::RepresentativePolicy::kPeak;
  }
  std::ostringstream id;
  // Keep ids short so the shared text-report scenario column (40) aligns.
  id << "red_" << policy_tag(sc.policy) << "_" << s.id << "_n" << s.num_points
     << "_" << override_tag(ovr);
  sc.id = id.str();
  return sc;
}

std::vector<RedScenario> build_default_matrix() {
  std::vector<RedScenario> out;
  const SizeSpec& md = must_find_size("md");
  for (auto p : {hr::RepresentativePolicy::kFirst,
                 hr::RepresentativePolicy::kPeak,
                 hr::RepresentativePolicy::kMedoid}) {
    out.push_back(make_scenario(p, md));
  }
  for (const char* sid : {"xs", "sm", "md", "lg"}) {
    out.push_back(make_scenario(hr::RepresentativePolicy::kPeak,
                                must_find_size(sid)));
  }
  // Distinct live params beyond the default representative flip.
  out.push_back(make_scenario(hr::RepresentativePolicy::kPeak, md,
                              OverrideKind::kFeatureWeights));
  out.push_back(make_scenario(hr::RepresentativePolicy::kPeak, md,
                              OverrideKind::kScoreMethod));
  return out;
}

std::vector<RedScenario> build_size_sweep(
    const std::vector<std::string>& sizes) {
  std::vector<RedScenario> out;
  for (const auto& sid : sizes) {
    const SizeSpec* s = find_size(sid);
    if (s == nullptr) {
      std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
      continue;
    }
    out.push_back(make_scenario(hr::RepresentativePolicy::kPeak, *s));
  }
  return out;
}

std::vector<RedScenario> build_full_matrix() {
  std::vector<RedScenario> out;
  for (const auto& s : size_ladder()) {
    for (auto p : {hr::RepresentativePolicy::kFirst,
                   hr::RepresentativePolicy::kPeak,
                   hr::RepresentativePolicy::kMedoid}) {
      out.push_back(make_scenario(p, s));
    }
    out.push_back(make_scenario(hr::RepresentativePolicy::kPeak, s,
                                OverrideKind::kFeatureWeights));
    out.push_back(make_scenario(hr::RepresentativePolicy::kPeak, s,
                                OverrideKind::kScoreMethod));
  }
  return out;
}

struct Options {
  std::string output;
  std::string text;
  bool quiet{false};
  bool full{false};
  std::vector<std::string> sizes;
  bool sizes_all{false};
  double stop_after_s{0.0};
  perf::RunSettings rs;
};

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

void print_help() {
  std::cerr
      << "Usage: perf_detection_reduction [options]\n"
      << "  --output PATH    Write JSON report\n"
      << "  --text PATH      Write text report\n"
      << "  --quiet          Suppress stdout text report\n"
      << "  --warmup N       Warmup frames (default from RunSettings)\n"
      << "  --frames N       Measured frames\n"
      << "  --repeats N      Scenario repeats\n"
      << "  --sizes all      Peak-policy size sweep (inline representative "
         "override)\n"
      << "  --sizes a,b,c    Peak-policy size sweep across a subset\n"
      << "  --full           Full matrix (policies × sizes + "
         "feature_weights/score_method overrides)\n"
      << "  --stop-after-s S Stop after a scenario exceeds S seconds\n"
      << "  --help           Show this help\n";
}

bool parse_args(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "error: " << flag << " requires an argument\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      print_help();
      std::exit(0);
    } else if (a == "--output") {
      opt->output = next("--output");
    } else if (a == "--text") {
      opt->text = next("--text");
    } else if (a == "--quiet") {
      opt->quiet = true;
    } else if (a == "--full") {
      opt->full = true;
    } else if (a == "--warmup") {
      opt->rs.warmup_frames = std::stoi(next("--warmup"));
    } else if (a == "--frames") {
      opt->rs.measured_frames = std::stoi(next("--frames"));
    } else if (a == "--repeats") {
      opt->rs.repeats = std::stoi(next("--repeats"));
    } else if (a == "--stop-after-s") {
      opt->stop_after_s = std::stod(next("--stop-after-s"));
      if (opt->stop_after_s <= 0.0) {
        std::cerr << "error: --stop-after-s must be positive\n";
        return false;
      }
    } else if (a == "--sizes") {
      const std::string v = next("--sizes");
      if (v == "all") {
        opt->sizes_all = true;
      } else {
        opt->sizes = split_csv(v);
      }
    } else {
      std::cerr << "error: unknown argument '" << a << "'\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse_args(argc, argv, &opt)) {
    return 2;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "error: no CUDA device available; DetectionReductionOp "
                 "requires a GPU\n";
    return 1;
  }

  if (std::getenv("HOLOSCAN_LOG_LEVEL") == nullptr) {
    setenv("HOLOSCAN_LOG_LEVEL", "ERROR", 0);
  }

  std::vector<RedScenario> scenarios;
  if (opt.full) {
    scenarios = build_full_matrix();
  } else if (opt.sizes_all) {
    std::vector<std::string> all;
    for (const auto& s : size_ladder()) {
      all.push_back(s.id);
    }
    scenarios = build_size_sweep(all);
  } else if (!opt.sizes.empty()) {
    scenarios = build_size_sweep(opt.sizes);
  } else {
    scenarios = build_default_matrix();
  }

  perf::PerfReport report;
  report.provenance = perf::capture_provenance();
  report.platform = perf::capture_platform();
  report.run_settings = opt.rs;
  report.run_settings.stop_after_s = opt.stop_after_s;
  report.run_settings.scenarios_planned = static_cast<int>(scenarios.size());

  {
    std::vector<std::string> used;
    for (const auto& sc : scenarios) {
      if (std::find(used.begin(), used.end(), sc.size_id) == used.end()) {
        used.push_back(sc.size_id);
      }
    }
    for (const auto& s : size_ladder()) {
      if (std::find(used.begin(), used.end(), s.id) != used.end()) {
        perf::SizeDef def;
        def.id = s.id;
        def.shape = {s.num_points, s.num_features};
        def.cells = s.num_points;
        def.role = s.role;
        report.sizes.push_back(def);
      }
    }
  }

  // Pay CUDA + Holoscan first-Application costs (and first-touch each policy
  // kernel) so scenario #1 cold_ms is comparable to later rows.
  perf::run_process_warmup([&] {
    perf::PerfReport discarded;
    perf::RunSettings warm_rs;
    warm_rs.warmup_frames = 0;
    warm_rs.measured_frames = 2;
    warm_rs.repeats = 1;
    const SizeSpec& xs = must_find_size("xs");
    for (auto p : {hr::RepresentativePolicy::kFirst,
                   hr::RepresentativePolicy::kPeak,
                   hr::RepresentativePolicy::kMedoid}) {
      run_scenario(make_scenario(p, xs), warm_rs, &discarded);
    }
  });

  const size_t n = scenarios.size();
  bool stopped_early = false;
  for (size_t i = 0; i < n; ++i) {
    std::cerr << "[" << (i + 1) << "/" << n << "] " << scenarios[i].id
              << std::flush;
    const auto t0 = perf::Clock::now();
    run_scenario(scenarios[i], opt.rs, &report);
    const double elapsed_s = perf::to_ms(perf::Clock::now() - t0) / 1000.0;
    std::cerr << "  (" << elapsed_s << " s)\n";
    report.run_settings.scenarios_completed = static_cast<int>(i + 1);
    if (opt.stop_after_s > 0.0 && elapsed_s > opt.stop_after_s) {
      std::cerr << "stopping early: scenario exceeded --stop-after-s "
                << opt.stop_after_s << " s\n";
      stopped_early = true;
      break;
    }
  }
  report.run_settings.stopped_early = stopped_early;

  if (!opt.output.empty()) {
    std::ofstream f(opt.output);
    if (!f) {
      std::cerr << "error: cannot open --output file '" << opt.output << "'\n";
      return 1;
    }
    perf::write_json(f, report);
    std::cerr << "wrote JSON report: " << opt.output << "\n";
  }
  if (!opt.text.empty()) {
    std::ofstream f(opt.text);
    if (!f) {
      std::cerr << "error: cannot open --text file '" << opt.text << "'\n";
      return 1;
    }
    perf::write_text(f, report);
    std::cerr << "wrote text report: " << opt.text << "\n";
  }
  if (!opt.quiet) {
    perf::write_text(std::cout, report);
  }
  return 0;
}
