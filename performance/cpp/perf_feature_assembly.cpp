// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FeatureAssemblyOp — second consumer of the
// reusable performance/ harness. Sweeps detection count, score-map size, and
// feature packs (bins / +score / full clustering columns). Measurement tool
// only: no timing assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_stats.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // v1 characterizes single precision.

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

struct SizeSpec {
  std::string id;
  int64_t doppler;
  int64_t range;
  std::string role;
};

enum class FeaturePack { kBins, kScore, kFull };

struct FeatScenario {
  std::string id;
  std::string size_id;
  FeaturePack pack{FeaturePack::kScore};
  int64_t num_detections{0};
  int64_t doppler{0};
  int64_t range{0};
  int64_t score_batch{1};  // leading dim on detection_score when > 1
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 64, 256, "small map; fast smoke"},
      {"sm", 128, 512, "compact sensor"},
      {"md", 256, 512, "common dev size"},
      {"lg", 512, 1024, "moderate production-scale"},
      {"xlw", 1024, 2048, "range-heavy moderate (v1 upper bound)"}};
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

std::string pack_tag(FeaturePack pack) {
  switch (pack) {
    case FeaturePack::kBins: return "bins";
    case FeaturePack::kScore: return "score";
    case FeaturePack::kFull: return "full";
  }
  return "unknown";
}

std::vector<hr::AssemblyFeature> pack_features(FeaturePack pack) {
  switch (pack) {
    case FeaturePack::kBins:
      return {hr::AssemblyFeature::kRange, hr::AssemblyFeature::kVelocity};
    case FeaturePack::kScore:
      return {hr::AssemblyFeature::kRange, hr::AssemblyFeature::kVelocity,
              hr::AssemblyFeature::kScore};
    case FeaturePack::kFull:
      return {hr::AssemblyFeature::kRange, hr::AssemblyFeature::kVelocity,
              hr::AssemblyFeature::kScore, hr::AssemblyFeature::kAz,
              hr::AssemblyFeature::kEl};
  }
  return {};
}

bool pack_needs_score(FeaturePack pack) {
  return pack == FeaturePack::kScore || pack == FeaturePack::kFull;
}

bool pack_needs_aoa(FeaturePack pack) { return pack == FeaturePack::kFull; }

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_from_host(
    const std::vector<T>& host, const std::vector<int64_t>& shape,
    DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    if (cudaMemcpy(tensor->data(), host.data(), sizeof(T) * host.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "perf_feature_assembly: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

// Deterministic detection grid across the DxR plane (wraps if N > cells).
std::shared_ptr<holoscan::Tensor> make_detection_idx(int64_t num_detections,
                                                     int64_t doppler,
                                                     int64_t range) {
  std::vector<int64_t> host(
      static_cast<size_t>(std::max<int64_t>(num_detections, 0) * 2), 0);
  const int64_t cells = std::max<int64_t>(doppler * range, 1);
  for (int64_t i = 0; i < num_detections; ++i) {
    const int64_t cell = (i * 997) % cells;
    host[static_cast<size_t>(i * 2)] = cell / range;
    host[static_cast<size_t>(i * 2 + 1)] = cell % range;
  }
  return make_device_from_host(host, {num_detections, 2}, kInt64Dtype);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_axis_bins(int64_t n) {
  std::vector<T> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    host[static_cast<size_t>(i)] = static_cast<T>(i);
  }
  return make_device_from_host(host, {n}, hr::real_dtype<T>());
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_score_map(
    const std::vector<int64_t>& shape) {
  return perf::make_power_map_device<T>(shape);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_aoa(int64_t num_detections) {
  std::vector<T> host(
      static_cast<size_t>(std::max<int64_t>(num_detections, 0) * 2));
  for (int64_t i = 0; i < num_detections; ++i) {
    host[static_cast<size_t>(i * 2)] = static_cast<T>(0.01) * static_cast<T>(i);
    host[static_cast<size_t>(i * 2 + 1)] =
        static_cast<T>(-0.02) * static_cast<T>(i);
  }
  return make_device_from_host(host, {num_detections, 2}, hr::real_dtype<T>());
}

// Multi-port source: emits the same preallocated TensorMaps every tick so
// measured latency is dominated by FeatureAssemblyOp, not H2D.
class FeatAsmPerfSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FeatAsmPerfSourceOp)
  FeatAsmPerfSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    // Ports are always declared: Holoscan calls setup() before compose() can
    // configure() emit flags. Unused outputs simply stay unwired.
    spec.output<holoscan::TensorMap>("detection_idx");
    spec.output<holoscan::TensorMap>("range_bins");
    spec.output<holoscan::TensorMap>("velocity_bins");
    spec.output<holoscan::TensorMap>("detection_score");
    spec.output<holoscan::TensorMap>("aoa_estimates");
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
    op_output.emit(idx_, "detection_idx");
    op_output.emit(range_bins_, "range_bins");
    op_output.emit(velocity_bins_, "velocity_bins");
    op_output.emit(score_, "detection_score");
    op_output.emit(aoa_, "aoa_estimates");
  }

  void configure(perf::TimingCollector* collector, holoscan::TensorMap idx,
                 holoscan::TensorMap range_bins,
                 holoscan::TensorMap velocity_bins, holoscan::TensorMap score,
                 holoscan::TensorMap aoa) {
    collector_ = collector;
    idx_ = std::move(idx);
    range_bins_ = std::move(range_bins);
    velocity_bins_ = std::move(velocity_bins);
    score_ = std::move(score);
    aoa_ = std::move(aoa);
  }

 private:
  perf::TimingCollector* collector_{nullptr};
  holoscan::TensorMap idx_;
  holoscan::TensorMap range_bins_;
  holoscan::TensorMap velocity_bins_;
  holoscan::TensorMap score_;
  holoscan::TensorMap aoa_;
};

template <typename T>
class FeatAsmPerfApp : public holoscan::Application {
 public:
  const FeatScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap idx;
  holoscan::TensorMap range_bins;
  holoscan::TensorMap velocity_bins;
  holoscan::TensorMap score;
  holoscan::TensorMap aoa;
  int total_frames{25};

  void compose() override {
    const auto features = pack_features(scen->pack);
    auto op = make_operator<hr::FeatureAssemblyOp<T>>(
        "feat", holoscan::Arg("features") = features);

    auto src = make_operator<FeatAsmPerfSourceOp>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    // Always emit and wire every optional input: Holoscan refuses to tick a
    // source with an unconnected output port. FeatureAssembly ignores unused
    // optionals when the feature pack does not require them.
    src->configure(collector, idx, range_bins, velocity_bins, score, aoa);
    add_flow(src, op, {{"detection_idx", "detection_idx"}});
    add_flow(src, op, {{"range_bins", "range_bins"}});
    add_flow(src, op, {{"velocity_bins", "velocity_bins"}});
    add_flow(src, op, {{"detection_score", "detection_score"}});
    add_flow(src, op, {{"aoa_estimates", "aoa_estimates"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("points_sink");
    term->configure(collector);
    add_flow(op, term, {{"points", "input"}});
  }
};

void fill_throughput(perf::ResultRow* row, const FeatScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.num_detections);
  const double cells = static_cast<double>(scen.doppler) *
                       static_cast<double>(scen.range) *
                       static_cast<double>(scen.score_batch);
  row->cells_per_second = n / sec;  // detections/s (sparse axis)
  row->slices_per_second =
      static_cast<double>(std::max<int64_t>(scen.score_batch, 1)) / sec;

  const double tsz = sizeof(PerfPrecision);
  const double idx_bytes = n * 2.0 * static_cast<double>(sizeof(int64_t));
  const double bins_bytes =
      static_cast<double>(scen.doppler + scen.range) * tsz;
  const double score_bytes =
      pack_needs_score(scen.pack) ? cells * tsz : 0.0;
  const double aoa_bytes =
      pack_needs_aoa(scen.pack) ? n * 2.0 * tsz : 0.0;
  const double points_bytes =
      n * static_cast<double>(pack_features(scen.pack).size()) * tsz;
  row->estimated_effective_gbps =
      (idx_bytes + bins_bytes + score_bytes + aoa_bytes + points_bytes) / sec /
      1e9;
}

perf::ResultRow base_row(const FeatScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "FeatureAssemblyOp";
  row.variant = pack_tag(scen.pack);
  row.size_id = scen.size_id;
  row.map_shape = {scen.doppler, scen.range};
  row.input_shape = {scen.num_detections, 2};
  row.out_batch = scen.num_detections;
  row.k = static_cast<int64_t>(pack_features(scen.pack).size());
  if (scen.score_batch > 1) {
    row.notes = "score_batch=" + std::to_string(scen.score_batch);
  }
  return row;
}

void run_scenario(const FeatScenario& scen, const perf::RunSettings& rs,
                  perf::PerfReport* report) {
  const int total = rs.warmup_frames + rs.measured_frames;

  std::vector<double> setup_samples;
  std::vector<double> cold_samples;
  std::vector<double> steady_samples;

  for (int rep = 0; rep < rs.repeats; ++rep) {
    perf::TimingCollector collector;

    auto idx_t =
        make_detection_idx(scen.num_detections, scen.doppler, scen.range);
    auto range_t = make_axis_bins<PerfPrecision>(scen.range);
    auto vel_t = make_axis_bins<PerfPrecision>(scen.doppler);
    std::vector<int64_t> score_shape;
    if (scen.score_batch > 1) {
      score_shape = {scen.score_batch, scen.doppler, scen.range};
    } else {
      score_shape = {scen.doppler, scen.range};
    }
    auto score_t = make_score_map<PerfPrecision>(score_shape);
    auto aoa_t = make_aoa<PerfPrecision>(scen.num_detections);

    auto idx = perf::single_map(std::move(idx_t), "detection_idx");
    auto range_bins = perf::single_map(std::move(range_t), "range_bins");
    auto velocity_bins = perf::single_map(std::move(vel_t), "velocity_bins");
    auto score = perf::single_map(std::move(score_t), "detection_score");
    auto aoa = perf::single_map(std::move(aoa_t), "aoa_estimates");

    collector.t_construct = perf::Clock::now();
    auto app = holoscan::make_application<FeatAsmPerfApp<PerfPrecision>>();
    app->scen = &scen;
    app->collector = &collector;
    app->idx = std::move(idx);
    app->range_bins = std::move(range_bins);
    app->velocity_bins = std::move(velocity_bins);
    app->score = std::move(score);
    app->aoa = std::move(aoa);
    app->total_frames = total;
    app->run();

    setup_samples.push_back(collector.setup_ms());
    const auto lat = collector.frame_latencies_ms();
    if (!lat.empty()) {
      cold_samples.push_back(lat.front());
    }
    for (size_t i = static_cast<size_t>(rs.warmup_frames); i < lat.size();
         ++i) {
      steady_samples.push_back(lat[i]);
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
    fill_throughput(&row, scen, row.latency.median_ms);
    if (row.notes.empty()) {
      row.notes =
          "cold_ms is first-frame latency; median/p90 are post-warmup "
          "steady state; cells/s is detections/s";
    } else {
      row.notes +=
          "; cold_ms first-frame; median/p90 steady; cells/s detections/s";
    }
    report->results.push_back(row);
  }
}

FeatScenario make_scenario(FeaturePack pack, const SizeSpec& s,
                           int64_t num_detections, int64_t score_batch = 1) {
  FeatScenario sc;
  sc.pack = pack;
  sc.size_id = s.id;
  sc.doppler = s.doppler;
  sc.range = s.range;
  sc.num_detections = num_detections;
  sc.score_batch = score_batch;
  std::ostringstream id;
  id << "feat_asm_" << pack_tag(pack) << "_" << s.id << "_n" << num_detections;
  if (score_batch > 1) {
    id << "_b" << score_batch;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

std::vector<FeatScenario> build_default_matrix() {
  std::vector<FeatScenario> out;
  const SizeSpec& md = must_find_size("md");
  const std::vector<int64_t> counts = {64, 1024, 16384};
  const std::vector<FeaturePack> packs = {
      FeaturePack::kBins, FeaturePack::kScore, FeaturePack::kFull};
  for (int64_t n : counts) {
    for (auto pack : packs) {
      out.push_back(make_scenario(pack, md, n));
    }
  }
  for (const char* sid : {"xs", "md", "lg", "xlw"}) {
    const SizeSpec& s = must_find_size(sid);
    out.push_back(make_scenario(FeaturePack::kScore, s, 1024));
  }
  out.push_back(make_scenario(FeaturePack::kScore, md, 1024, 4));
  out.push_back(make_scenario(FeaturePack::kBins, md, 0));
  return out;
}

std::vector<FeatScenario> build_size_sweep(
    const std::vector<std::string>& sizes) {
  std::vector<FeatScenario> out;
  for (const auto& sid : sizes) {
    const SizeSpec* s = find_size(sid);
    if (s == nullptr) {
      std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
      continue;
    }
    out.push_back(make_scenario(FeaturePack::kScore, *s, 1024));
  }
  return out;
}

std::vector<FeatScenario> build_full_matrix() {
  std::vector<FeatScenario> out;
  const std::vector<int64_t> counts = {0, 64, 256, 1024, 4096, 16384, 65536};
  const std::vector<FeaturePack> packs = {
      FeaturePack::kBins, FeaturePack::kScore, FeaturePack::kFull};
  for (const auto& s : size_ladder()) {
    for (int64_t n : counts) {
      for (auto pack : packs) {
        out.push_back(make_scenario(pack, s, n));
      }
    }
  }
  const SizeSpec& md = must_find_size("md");
  for (int64_t b : {4, 16}) {
    out.push_back(make_scenario(FeaturePack::kScore, md, 1024, b));
    out.push_back(make_scenario(FeaturePack::kFull, md, 4096, b));
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
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.push_back(tok);
    }
  }
  return out;
}

void print_help() {
  std::cout <<
      "perf_feature_assembly — FeatureAssemblyOp performance "
      "characterization\n\n"
      "Usage: perf_feature_assembly [options]\n\n"
      "  --output <path>   Write machine-readable JSON report\n"
      "  --text <path>     Also write the text summary to a file\n"
      "  --quiet           Suppress the stdout text summary\n"
      "  --warmup N        Warmup frames excluded from stats (default 5)\n"
      "  --frames N        Measured frames per scenario (default 20)\n"
      "  --repeats N       Repeat each scenario N times, pooled (default 1)\n"
      "  --sizes all       Score-pack size sweep across the full ladder\n"
      "  --sizes a,b,c     Score-pack size sweep across an explicit subset\n"
      "  --full            Full matrix (packs × sizes × detection counts)\n"
      "  --stop-after-s S  Stop after first scenario whose wall time exceeds "
      "S seconds\n"
      "  --help            Show this help\n";
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
    std::cerr << "error: no CUDA device available; FeatureAssemblyOp requires "
                 "a GPU\n";
    return 1;
  }

  if (std::getenv("HOLOSCAN_LOG_LEVEL") == nullptr) {
    setenv("HOLOSCAN_LOG_LEVEL", "ERROR", 0);
  }

  std::vector<FeatScenario> scenarios;
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
        def.shape = {s.doppler, s.range};
        def.cells = s.doppler * s.range;
        def.role = s.role;
        report.sizes.push_back(def);
      }
    }
  }

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
