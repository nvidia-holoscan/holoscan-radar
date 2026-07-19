// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// End-to-end Holoscan graph comparison: split
//   range → [clutter] → Doppler
// vs fused
//   FMCWRangeDopplerProcessorOp
//
// Measurement tool only — no timing assertions and no keep/kill verdict.
// Summing the three per-operator perf tools overstates split cost (extra
// Holoscan nodes / intermediate tensors are missing from a naive sum); this
// tool times both graphs with identical I/O so the team can fill the
// decision table in
// operators/fmcw/fmcw_range_doppler_processor_op/OPEN_ITEMS.md §1.
//
// See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_source.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

enum class GraphKind { kSplit, kFused };

struct SizeSpec {
  std::string id;
  int64_t num_chirps;
  int64_t samples_per_chirp;
  std::string role;
};

struct CompareScenario {
  std::string id;
  std::string size_id;
  std::string pair_id;  // shared by split/fused pair for post_run join
  int64_t num_chirps{0};
  int64_t samples_per_chirp{0};
  int64_t channels{1};
  hr::ClutterFilterMethod clutter{hr::ClutterFilterMethod::kNone};
  GraphKind graph{GraphKind::kFused};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 16, 128, "small cube; fast smoke"},
      {"sm", 32, 256, "compact sensor"},
      {"md", 64, 512, "common dev size"},
      {"lg", 128, 1024, "moderate production-scale"},
      {"xl", 256, 2048, "range-heavy moderate (v1 upper bound)"}};
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

std::string clutter_tag(hr::ClutterFilterMethod m) {
  switch (m) {
    case hr::ClutterFilterMethod::kNone: return "noclt";
    case hr::ClutterFilterMethod::kMeanSubtraction: return "msub";
    case hr::ClutterFilterMethod::kSingleDelay: return "sdel";
    case hr::ClutterFilterMethod::kDoubleDelay: return "ddel";
  }
  return "unk";
}

std::string graph_tag(GraphKind g) {
  return g == GraphKind::kSplit ? "split" : "fused";
}

hr::FMCWWaveformDescriptor make_waveform() {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25.0e-6;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<int64_t> cube_shape(const CompareScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.samples_per_chirp};
  }
  return {scen.num_chirps, scen.samples_per_chirp};
}

// ---------------------------------------------------------------------------
// Graphs (no [live] overrides — steady e2e comparison).
// ---------------------------------------------------------------------------
template <typename T>
class FusedCompareApp : public holoscan::Application {
 public:
  const CompareScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  hr::FMCWWaveformDescriptor waveform;
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::FMCWRangeDopplerProcessorOp<T>>(
        "fused", Arg("waveform") = waveform,
        Arg("clutter_filter") = scen->clutter);

    auto src = make_operator<perf::PerfTensorSourceOp<perf::AdcCubePort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"adc_cube", "adc_cube"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("rd_sink");
    term->configure(collector);
    add_flow(op, term, {{"rd_maps", "input"}});

    for (const char* port : {"range_bins", "velocity_bins"}) {
      auto ns =
          make_operator<perf::PerfNullSinkOp>(std::string(port) + "_sink");
      add_flow(op, ns, {{port, "input"}});
    }
  }
};

template <typename T>
class SplitCompareApp : public holoscan::Application {
 public:
  const CompareScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  hr::FMCWWaveformDescriptor waveform;
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto range = make_operator<hr::FMCWRangeProcessorOp<T>>(
        "range", Arg("waveform") = waveform);
    auto doppler = make_operator<hr::FMCWDopplerProcessorOp<T>>(
        "doppler", Arg("waveform") = waveform);

    auto src = make_operator<perf::PerfTensorSourceOp<perf::AdcCubePort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, range, {{"adc_cube", "adc_cube"}});

    if (scen->clutter == hr::ClutterFilterMethod::kNone) {
      add_flow(range, doppler, {{"range_profiles", "range_profiles"}});
    } else {
      auto clutter = make_operator<hr::ClutterFilterOp<T>>(
          "clutter", Arg("method") = scen->clutter);
      add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
      add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
    }

    auto term = make_operator<perf::PerfTerminalSinkOp>("rd_sink");
    term->configure(collector);
    add_flow(doppler, term, {{"rd_maps", "input"}});

    auto range_bins_sink =
        make_operator<perf::PerfNullSinkOp>("range_bins_sink");
    add_flow(range, range_bins_sink, {{"range_bins", "input"}});
    auto vel_bins_sink =
        make_operator<perf::PerfNullSinkOp>("velocity_bins_sink");
    add_flow(doppler, vel_bins_sink, {{"velocity_bins", "input"}});
  }
};

void fill_throughput(perf::ResultRow* row, const CompareScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double chirps = static_cast<double>(scen.num_chirps);
  const double n = static_cast<double>(scen.samples_per_chirp);
  row->cells_per_second = batch * chirps * n / sec;
  row->slices_per_second = batch / sec;
  const double csz = sizeof(std::complex<PerfPrecision>);
  const double bytes = 2.0 * batch * chirps * n * csz;
  row->estimated_effective_gbps = bytes / sec / 1e9;
}

perf::ResultRow base_row(const CompareScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "FMCWRangeDoppler_split_vs_fused";
  row.variant = clutter_tag(scen.clutter) + "_" + graph_tag(scen.graph);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_chirps, scen.samples_per_chirp};
  row.input_shape = cube_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.samples_per_chirp;
  row.calibration_pair = scen.pair_id;
  row.calibration_mode = graph_tag(scen.graph);
  row.notes =
      "e2e Holoscan graph; no live override; pair_id joins split/fused rows";
  return row;
}

CompareScenario make_scenario(hr::ClutterFilterMethod clutter,
                              const SizeSpec& s, GraphKind graph,
                              int64_t channels = 1) {
  CompareScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.samples_per_chirp = s.samples_per_chirp;
  sc.channels = channels;
  sc.clutter = clutter;
  sc.graph = graph;
  std::ostringstream pair;
  pair << "svf_" << clutter_tag(clutter) << "_" << s.id;
  if (channels > 1) {
    pair << "_ch" << channels;
  }
  sc.pair_id = pair.str();
  sc.id = sc.pair_id + "_" + graph_tag(graph);
  return sc;
}

void append_pair(std::vector<CompareScenario>* out,
                 hr::ClutterFilterMethod clutter, const SizeSpec& s,
                 int64_t channels = 1) {
  out->push_back(make_scenario(clutter, s, GraphKind::kSplit, channels));
  out->push_back(make_scenario(clutter, s, GraphKind::kFused, channels));
}

void print_comparison_table(const perf::PerfReport& report) {
  struct PairStats {
    std::string pair_id;
    std::string size_id;
    std::string clutter;
    double split_ms{0.0};
    double fused_ms{0.0};
    bool have_split{false};
    bool have_fused{false};
  };
  std::vector<PairStats> pairs;
  auto find_or_add = [&](const std::string& pair_id) -> PairStats& {
    for (auto& p : pairs) {
      if (p.pair_id == pair_id) {
        return p;
      }
    }
    pairs.push_back(PairStats{});
    pairs.back().pair_id = pair_id;
    return pairs.back();
  };

  for (const auto& row : report.results) {
    if (row.bucket != "compute" || row.calibration_pair.empty()) {
      continue;
    }
    auto& p = find_or_add(row.calibration_pair);
    p.size_id = row.size_id;
    const auto us = row.variant.find('_');
    p.clutter = us == std::string::npos ? row.variant : row.variant.substr(0, us);
    if (row.calibration_mode == "split") {
      p.split_ms = row.latency.median_ms;
      p.have_split = true;
    } else if (row.calibration_mode == "fused") {
      p.fused_ms = row.latency.median_ms;
      p.have_fused = true;
    }
  }

  std::cout << "\n=== split vs fused comparison (median_ms, team decision aid) "
               "===\n";
  std::cout << "No keep/kill verdict is encoded. Fill OPEN_ITEMS §1 from this "
               "table.\n";
  std::cout << std::left << std::setw(28) << "pair_id" << std::setw(8)
            << "size" << std::setw(8) << "clutter" << std::right << std::setw(12)
            << "split_ms" << std::setw(12) << "fused_ms" << std::setw(12)
            << "delta_ms" << std::setw(12) << "pct_saved" << std::setw(10)
            << "speedup\n";
  for (const auto& p : pairs) {
    if (!p.have_split || !p.have_fused) {
      continue;
    }
    const double delta = p.split_ms - p.fused_ms;
    const double pct =
        p.split_ms > 0.0 ? 100.0 * (1.0 - p.fused_ms / p.split_ms) : 0.0;
    const double speedup = p.fused_ms > 0.0 ? p.split_ms / p.fused_ms : 0.0;
    std::cout << std::left << std::setw(28) << p.pair_id << std::setw(8)
              << p.size_id << std::setw(8) << p.clutter << std::right
              << std::fixed << std::setprecision(3) << std::setw(12)
              << p.split_ms << std::setw(12) << p.fused_ms << std::setw(12)
              << delta << std::setw(12) << pct << std::setw(10) << speedup
              << "\n";
  }
  std::cout << "(delta_ms = split − fused; pct_saved = 100·(1 − fused/split); "
               "speedup = split/fused)\n";
  std::cout << "Note: do not compare against a naive sum of "
               "perf_fmcw_range + clutter + doppler medians — that omits "
               "inter-op Holoscan cost on the split path.\n\n";
}

struct CompareTraits : perf::PerfDriverDefaults {
  using Scenario = CompareScenario;

  static const char* tool_name() { return "perf_fmcw_split_vs_fused"; }
  static const char* tool_description() {
    return "End-to-end split (range→[clutter]→Doppler) vs fused "
           "FMCWRangeDopplerProcessorOp comparison (no keep/kill verdict)";
  }
  static const char* sizes_all_help() {
    return "noclt+msub pairs across the full size ladder";
  }
  static const char* sizes_subset_help() {
    return "noclt+msub pairs for an explicit size subset";
  }
  static const char* full_help() {
    return "Full matrix (noclt/msub/sdel × sizes × channel batch)";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_chirps, s.samples_per_chirp};
      def.cells = s.num_chirps * s.samples_per_chirp;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<CompareScenario> build_default_matrix() {
    std::vector<CompareScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      append_pair(&out, hr::ClutterFilterMethod::kNone, must_find_size(sid));
      append_pair(&out, hr::ClutterFilterMethod::kMeanSubtraction,
                  must_find_size(sid));
    }
    append_pair(&out, hr::ClutterFilterMethod::kNone, must_find_size("md"), 4);
    return out;
  }

  static std::vector<CompareScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<CompareScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      append_pair(&out, hr::ClutterFilterMethod::kNone, *s);
      append_pair(&out, hr::ClutterFilterMethod::kMeanSubtraction, *s);
    }
    return out;
  }

  static std::vector<CompareScenario> build_full_matrix() {
    std::vector<CompareScenario> out;
    for (const auto& s : size_ladder()) {
      for (auto c : {hr::ClutterFilterMethod::kNone,
                     hr::ClutterFilterMethod::kMeanSubtraction,
                     hr::ClutterFilterMethod::kSingleDelay}) {
        append_pair(&out, c, s);
      }
    }
    for (const char* sid : {"md", "lg"}) {
      for (int64_t ch : {4, 16}) {
        append_pair(&out, hr::ClutterFilterMethod::kNone, must_find_size(sid),
                    ch);
      }
    }
    return out;
  }

  static void run_scenario(const CompareScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen)),
              "adc_cube");

          collector.t_construct = perf::Clock::now();
          if (scen.graph == GraphKind::kFused) {
            auto app =
                holoscan::make_application<FusedCompareApp<PerfPrecision>>();
            app->scen = &scen;
            app->collector = &collector;
            app->frame = std::move(cube);
            app->waveform = make_waveform();
            app->total_frames = p.total_frames;
            app->run();
          } else {
            auto app =
                holoscan::make_application<SplitCompareApp<PerfPrecision>>();
            app->scen = &scen;
            app->collector = &collector;
            app->frame = std::move(cube);
            app->waveform = make_waveform();
            app->total_frames = p.total_frames;
            app->run();
          }
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);
    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(hr::ClutterFilterMethod::kNone,
                                 must_find_size("xs"), GraphKind::kFused),
                   warm, &discarded);
      run_scenario(make_scenario(hr::ClutterFilterMethod::kNone,
                                 must_find_size("xs"), GraphKind::kSplit),
                   warm, &discarded);
    });
  }

  static void post_run(perf::PerfReport* report) {
    if (report != nullptr) {
      print_comparison_table(*report);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CompareTraits>(argc, argv);
}
