// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// End-to-end Holoscan graph comparison: split
//   range → [clutter] → Doppler
// vs fused
//   FMCWRangeDopplerProcessorOp
//
//  Graph-based rewrite of the HSDK 4.x Application
// harness.  No [live] overrides — steady e2e comparison only.
// Measurement tool only — no timing assertions and no keep/kill
// verdict.  Summing the three per-operator perf tools overstates split
// cost (extra Holoscan nodes / intermediate tensors are missing from a
// naive sum); this tool times both graphs with identical I/O so the
// team can fill the decision in CLARAHOLOS-3417.
//
// See docs/planning/OPERATOR_DEVELOPMENT.md.

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

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
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
  hr::WindowType doppler_window{hr::WindowType::kNone};
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
    case hr::ClutterFilterMethod::kNone:
      return "noclt";
    case hr::ClutterFilterMethod::kMeanSubtraction:
      return "msub";
    case hr::ClutterFilterMethod::kSingleDelay:
      return "sdel";
    case hr::ClutterFilterMethod::kDoubleDelay:
      return "ddel";
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

// -------------------------------------------------------------------
// Fused graph: src → FMCWRangeDopplerProcessorOp → sinks.
//
// Topology:
//   src/out → fused/adc_cube
//   fused/rd_maps → out_sink/in   (timed)
//   fused/range_bins → rb_sink/in (null)
//   fused/velocity_bins → vb_sink/in (null)
//
// Binds: fused/{rd_maps, range_bins, velocity_bins}
// -------------------------------------------------------------------
perf::TimingCollector run_fused_compare_graph(const CompareScenario& scen, int total_frames) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen));
  return perf::run_perf_graph(
      std::move(cube), total_frames, "svf-fused",
      [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents fused_extents;
        fused_extents.max_chirps = scen.num_chirps;
        fused_extents.max_range_bins = scen.samples_per_chirp;
        if (scen.channels > 1) {
          fused_extents.max_rx = scen.channels;
        }
        auto op = graph.op<hr::FMCWRangeDopplerProcessorOp<PerfPrecision>>("fused", fused_extents);
        hr::FMCWRangeDopplerProcessorParams params;
        params.waveform = make_waveform();
        params.clutter_filter = scen.clutter;
        params.doppler_window.type = scen.doppler_window;
        op.set_params(std::move(params));
        const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
        const auto rb = graph.op<perf::PerfNullSinkOp>("rb_sink");
        const auto vb = graph.op<perf::PerfNullSinkOp>("vb_sink");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(src->out, op->adc_cube, q);
        graph.add_flow(op->rd_maps, sink->in, q);
        graph.add_flow(op->range_bins, rb->in, q);
        graph.add_flow(op->velocity_bins, vb->in, q);
        perf::bind_tensor_out(options, "fused", "rd_maps");
        perf::bind_tensor_out(options, "fused", "range_bins");
        perf::bind_tensor_out(options, "fused", "velocity_bins");
      });
}

// -------------------------------------------------------------------
// Split graph: src → range → [clutter →] doppler → sinks.
//
// Topology (with clutter):
//   src/out → range/adc_cube
//   range/range_profiles → clutter/range_profiles
//   clutter/range_profiles_out → doppler/range_profiles
//   doppler/rd_maps → out_sink/in     (timed)
//   range/range_bins → rb_sink/in     (null)
//   doppler/velocity_bins → vb_sink/in (null)
//
// Topology (without clutter):
//   src/out → range/adc_cube
//   range/range_profiles → doppler/range_profiles
//   doppler/rd_maps → out_sink/in     (timed)
//   range/range_bins → rb_sink/in     (null)
//   doppler/velocity_bins → vb_sink/in (null)
//
// Binds: range/{range_profiles, range_bins},
//        [clutter/range_profiles_out],
//        doppler/{rd_maps, velocity_bins}
// -------------------------------------------------------------------
perf::TimingCollector run_split_compare_graph(const CompareScenario& scen, int total_frames) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen));
  return perf::run_perf_graph(
      std::move(cube), total_frames, "svf-split",
      [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents range_extents;
        range_extents.max_chirps = scen.num_chirps;
        range_extents.max_range_bins = scen.samples_per_chirp;
        if (scen.channels > 1) {
          range_extents.max_rx = scen.channels;
        }
        auto range = graph.op<hr::FMCWRangeProcessorOp<PerfPrecision>>("range", range_extents);
        hr::FMCWRangeProcessorParams rp;
        rp.waveform = make_waveform();
        range.set_params(std::move(rp));

        hr::MaxTensorExtents doppler_extents;
        doppler_extents.max_chirps = scen.num_chirps;
        doppler_extents.max_range_bins = scen.samples_per_chirp;
        if (scen.channels > 1) {
          doppler_extents.max_rx = scen.channels;
        }
        auto doppler =
            graph.op<hr::FMCWDopplerProcessorOp<PerfPrecision>>("doppler", doppler_extents);
        hr::FMCWDopplerProcessorParams dp;
        dp.waveform = make_waveform();
        dp.doppler_window.type = scen.doppler_window;
        doppler.set_params(std::move(dp));

        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(src->out, range->adc_cube, q);

        if (scen.clutter != hr::ClutterFilterMethod::kNone) {
          hr::MaxTensorExtents extents;
          extents.max_chirps = scen.num_chirps;
          extents.max_range_bins = scen.samples_per_chirp;
          if (scen.channels > 1) {
            extents.max_rx = scen.channels;
          }
          auto clt = graph.op<hr::ClutterFilterOp<PerfPrecision>>("clutter", extents);
          hr::ClutterFilterParams cp;
          cp.method = scen.clutter;
          if (scen.clutter == hr::ClutterFilterMethod::kMeanSubtraction) {
            cp.doppler_window.type = scen.doppler_window;
          }
          clt.set_params(std::move(cp));
          graph.add_flow(range->range_profiles, clt->range_profiles, q);
          graph.add_flow(clt->range_profiles_out, doppler->range_profiles, q);
          perf::bind_tensor_out(options, "clutter", "range_profiles_out");
        } else {
          graph.add_flow(range->range_profiles, doppler->range_profiles, q);
        }

        const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
        const auto rb = graph.op<perf::PerfNullSinkOp>("rb_sink");
        const auto vb = graph.op<perf::PerfNullSinkOp>("vb_sink");
        graph.add_flow(doppler->rd_maps, sink->in, q);
        graph.add_flow(range->range_bins, rb->in, q);
        graph.add_flow(doppler->velocity_bins, vb->in, q);
        perf::bind_tensor_out(options, "range", "range_profiles");
        perf::bind_tensor_out(options, "range", "range_bins");
        perf::bind_tensor_out(options, "doppler", "rd_maps");
        perf::bind_tensor_out(options, "doppler", "velocity_bins");
      });
}

void fill_throughput(perf::ResultRow* row, const CompareScenario& scen, double median_ms) {
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
  row.variant = clutter_tag(scen.clutter) + "_" + std::string(hr::to_string(scen.doppler_window)) +
                "_" + graph_tag(scen.graph);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_chirps, scen.samples_per_chirp};
  row.input_shape = cube_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.samples_per_chirp;
  row.calibration_pair = scen.pair_id;
  row.calibration_mode = graph_tag(scen.graph);
  row.notes =
      "e2e Holoscan graph; no live override; "
      "pair_id joins split/fused rows";
  return row;
}

CompareScenario make_scenario(hr::ClutterFilterMethod clutter, const SizeSpec& s, GraphKind graph,
                              int64_t channels = 1,
                              hr::WindowType doppler_window = hr::WindowType::kNone) {
  CompareScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.samples_per_chirp = s.samples_per_chirp;
  sc.channels = channels;
  sc.clutter = clutter;
  sc.doppler_window = doppler_window;
  sc.graph = graph;
  std::ostringstream pair;
  pair << "svf_" << clutter_tag(clutter) << "_" << hr::to_string(doppler_window) << "_" << s.id;
  if (channels > 1) {
    pair << "_ch" << channels;
  }
  sc.pair_id = pair.str();
  sc.id = sc.pair_id + "_" + graph_tag(graph);
  return sc;
}

void append_pair(std::vector<CompareScenario>* out, hr::ClutterFilterMethod clutter,
                 const SizeSpec& s, int64_t channels = 1,
                 hr::WindowType doppler_window = hr::WindowType::kNone) {
  out->push_back(make_scenario(clutter, s, GraphKind::kSplit, channels, doppler_window));
  out->push_back(make_scenario(clutter, s, GraphKind::kFused, channels, doppler_window));
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
    if (us == std::string::npos) {
      p.clutter = row.variant;
    } else {
      p.clutter = row.variant.substr(0, us);
    }
    if (row.calibration_mode == "split") {
      p.split_ms = row.latency.median_ms;
      p.have_split = true;
    } else if (row.calibration_mode == "fused") {
      p.fused_ms = row.latency.median_ms;
      p.have_fused = true;
    }
  }

  std::cout << "\n=== split vs fused comparison "
               "(median_ms, team decision aid) ===\n";
  std::cout << "No keep/kill verdict is encoded. Attach results to "
               "CLARAHOLOS-3417.\n";
  std::cout << std::left << std::setw(28) << "pair_id" << std::setw(8) << "size" << std::setw(8)
            << "clutter" << std::right << std::setw(12) << "split_ms" << std::setw(12) << "fused_ms"
            << std::setw(12) << "delta_ms" << std::setw(12) << "pct_saved" << std::setw(10)
            << "speedup\n";
  for (const auto& p : pairs) {
    if (!p.have_split || !p.have_fused) {
      continue;
    }
    const double delta = p.split_ms - p.fused_ms;
    const double pct = p.split_ms > 0.0 ? 100.0 * (1.0 - p.fused_ms / p.split_ms) : 0.0;
    const double speedup = p.fused_ms > 0.0 ? p.split_ms / p.fused_ms : 0.0;
    std::cout << std::left << std::setw(28) << p.pair_id << std::setw(8) << p.size_id
              << std::setw(8) << p.clutter << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << p.split_ms << std::setw(12) << p.fused_ms << std::setw(12)
              << delta << std::setw(12) << pct << std::setw(10) << speedup << "\n";
  }
  std::cout << "(delta_ms = split - fused; "
               "pct_saved = 100*(1 - fused/split); "
               "speedup = split/fused)\n";
  std::cout << "Note: do not compare against a naive sum "
               "of perf_fmcw_range + clutter + doppler "
               "medians — that omits inter-op Holoscan "
               "cost on the split path.\n\n";
}

struct CompareTraits : perf::PerfDriverDefaults {
  using Scenario = CompareScenario;

  static const char* tool_name() { return "perf_fmcw_split_vs_fused"; }
  static const char* tool_description() {
    return "End-to-end split vs fused comparison "
           "(HSDK 5 Graph; no keep/kill verdict)";
  }
  static const char* sizes_all_help() {
    return "noclt+msub pairs across the full size "
           "ladder";
  }
  static const char* sizes_subset_help() {
    return "noclt+msub pairs for an explicit size "
           "subset";
  }
  static const char* full_help() {
    return "Full matrix (noclt/msub/sdel x sizes x "
           "channel batch)";
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
      append_pair(&out, hr::ClutterFilterMethod::kMeanSubtraction, must_find_size(sid));
    }
    append_pair(&out, hr::ClutterFilterMethod::kNone, must_find_size("md"), 4);
    append_pair(&out, hr::ClutterFilterMethod::kMeanSubtraction, must_find_size("md"), 1,
                hr::WindowType::kHann);
    return out;
  }

  static std::vector<CompareScenario> build_size_sweep(const std::vector<std::string>& sizes) {
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
      for (auto c : {hr::ClutterFilterMethod::kNone, hr::ClutterFilterMethod::kMeanSubtraction,
                     hr::ClutterFilterMethod::kSingleDelay}) {
        append_pair(&out, c, s);
      }
      append_pair(&out, hr::ClutterFilterMethod::kMeanSubtraction, s, 1, hr::WindowType::kHann);
    }
    for (const char* sid : {"md", "lg"}) {
      for (int64_t ch : {4, 16}) {
        append_pair(&out, hr::ClutterFilterMethod::kNone, must_find_size(sid), ch);
      }
    }
    return out;
  }

  static void run_scenario(const CompareScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples =
        perf::collect_scenario_samples(rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          if (scen.graph == GraphKind::kFused) {
            return run_fused_compare_graph(scen, p.total_frames);
          }
          return run_split_compare_graph(scen, p.total_frames);
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
      run_scenario(
          make_scenario(hr::ClutterFilterMethod::kNone, must_find_size("xs"), GraphKind::kFused),
          warm, &discarded);
      run_scenario(
          make_scenario(hr::ClutterFilterMethod::kNone, must_find_size("xs"), GraphKind::kSplit),
          warm, &discarded);
    });
  }

  static void post_run(perf::PerfReport* report, bool quiet) {
    if (report != nullptr && !quiet) {
      print_comparison_table(*report);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CompareTraits>(argc, argv);
}
