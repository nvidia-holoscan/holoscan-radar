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

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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
#include "radar_common/descriptors/window_spec.hpp"
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
      {"md", 128, 512, "common dev size"},
      {"lg", 256, 2048, "moderate production-scale"},
      {"xl", 1024, 4096, "large production-scale (upper bound)"}};
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
// Live-override + sustained run modes.
// ---------------------------------------------------------------------------
enum class LiveParam { kNone, kNfft, kWindow, kWaveform };

LiveParam to_live_param(const std::string& s) {
  if (s == "nfft") return LiveParam::kNfft;
  if (s == "window") return LiveParam::kWindow;
  if (s == "waveform") return LiveParam::kWaveform;
  return LiveParam::kNone;
}

std::string live_param_tag(LiveParam p) {
  switch (p) {
    case LiveParam::kNfft: return "nfft";
    case LiveParam::kWindow: return "window";
    case LiveParam::kWaveform: return "waveform";
    case LiveParam::kNone: return "none";
  }
  return "none";
}

// A modified waveform for the live-swap experiment: 1.5x bandwidth, so the
// descriptor compares != the constructed default and forces the operator's
// warn-once axis/reference rebuild on the override tick.
hr::FMCWWaveformDescriptor make_waveform_alt() {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 225.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25.0e-6;
  return hr::FMCWWaveformDescriptor(spec);
}

// Prebuilt [live] override values for one scenario, shared by both graphs so
// the split/fused comparison stays apples-to-apples. nfft carries distinct
// per-axis values (2x the input dim); window/waveform share one taper /
// descriptor across the two stages. One-shot uses index 0; rotate cycles the
// whole vector.
struct LiveConfig {
  LiveParam param{LiveParam::kNone};
  bool rotate{false};
  bool sustained{false};
  int64_t override_at{0};
  int rotate_period{4};
  std::vector<int64_t> nfft_range;
  std::vector<int64_t> nfft_doppler;
  std::vector<hr::WindowSpec> range_window;
  std::vector<hr::WindowSpec> doppler_window;
  std::vector<hr::FMCWWaveformDescriptor> waveform;
};

LiveConfig build_live_config(const CompareScenario& scen,
                             const perf::RunSettings& rs) {
  LiveConfig live;
  live.param = to_live_param(rs.live_param);
  live.rotate = rs.live_rotate;
  live.sustained = rs.sustained;
  live.rotate_period = rs.rotate_period;
  const int64_t r = scen.samples_per_chirp;
  const int64_t d = scen.num_chirps;
  switch (live.param) {
    case LiveParam::kNfft:
      // 2x the input dim — valid (nfft >= dim) and re-keys the output geometry
      // pool. Rotate toggles back to the 1x default.
      live.nfft_range = live.rotate ? std::vector<int64_t>{2 * r, r}
                                    : std::vector<int64_t>{2 * r};
      live.nfft_doppler = live.rotate ? std::vector<int64_t>{2 * d, d}
                                      : std::vector<int64_t>{2 * d};
      break;
    case LiveParam::kWindow: {
      hr::WindowSpec hann;
      hann.type = hr::WindowType::kHann;
      const hr::WindowSpec rect;  // default rectangular (kNone)
      live.range_window = live.rotate
                              ? std::vector<hr::WindowSpec>{hann, rect}
                              : std::vector<hr::WindowSpec>{hann};
      live.doppler_window = live.range_window;
      break;
    }
    case LiveParam::kWaveform: {
      const auto alt = make_waveform_alt();
      const auto base = make_waveform();
      live.waveform =
          live.rotate ? std::vector<hr::FMCWWaveformDescriptor>{alt, base}
                      : std::vector<hr::FMCWWaveformDescriptor>{alt};
      break;
    }
    case LiveParam::kNone:
      break;
  }
  return live;
}

// ---------------------------------------------------------------------------
// Graphs. A default (kNone, non-sustained) LiveConfig reproduces the original
// static steady comparison; live/sustained modes add override sources / swap
// the timed sink.
// ---------------------------------------------------------------------------
template <typename T>
class FusedCompareApp : public holoscan::Application {
 public:
  const CompareScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  hr::FMCWWaveformDescriptor waveform;
  int total_frames{25};
  LiveConfig live;

  // Wire one override source (one-shot or rotating) into `target`'s [live]
  // port. Duplicated in SplitCompareApp because the two apps are unrelated
  // Application subclasses (both need the protected make_operator/add_flow).
  template <typename V>
  void wire_override(const std::shared_ptr<holoscan::Operator>& target,
                     const char* port, const std::vector<V>& values,
                     const std::string& name) {
    if (values.empty()) {
      return;
    }
    if (live.rotate) {
      auto s = make_operator<perf::PerfRotatingOverrideSourceOp<V>>(
          name + "_ovr",
          make_condition<holoscan::CountCondition>(total_frames));
      s->configure(values, live.rotate_period, live.override_at);
      add_flow(s, target, {{"output", port}});
    } else {
      auto s = make_operator<perf::PerfOverrideSourceOp<V>>(
          name + "_ovr",
          make_condition<holoscan::CountCondition>(total_frames));
      s->configure(live.override_at, values.front());
      add_flow(s, target, {{"output", port}});
    }
  }

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::FMCWRangeDopplerProcessorOp<T>>(
        "fused", Arg("waveform") = waveform,
        Arg("clutter_filter") = scen->clutter);

    auto src = make_operator<perf::PerfTensorSourceOp<perf::AdcCubePort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"adc_cube", "adc_cube"}});

    // Timed rd_maps sink: device-syncing terminal in latency/live modes; a
    // non-syncing null sink in sustained mode so frames pipeline.
    if (live.sustained) {
      auto ns = make_operator<perf::PerfNullSinkOp>("rd_sink");
      add_flow(op, ns, {{"rd_maps", "input"}});
    } else {
      auto term = make_operator<perf::PerfTerminalSinkOp>("rd_sink");
      term->configure(collector);
      add_flow(op, term, {{"rd_maps", "input"}});
    }

    for (const char* port : {"range_bins", "velocity_bins"}) {
      auto ns =
          make_operator<perf::PerfNullSinkOp>(std::string(port) + "_sink");
      add_flow(op, ns, {{port, "input"}});
    }

    switch (live.param) {
      case LiveParam::kNfft:
        wire_override(op, "nfft_range", live.nfft_range, "nfft_range");
        wire_override(op, "nfft_doppler", live.nfft_doppler, "nfft_doppler");
        break;
      case LiveParam::kWindow:
        wire_override(op, "range_window", live.range_window, "range_window");
        wire_override(op, "doppler_window", live.doppler_window,
                      "doppler_window");
        break;
      case LiveParam::kWaveform:
        wire_override(op, "waveform", live.waveform, "waveform");
        break;
      case LiveParam::kNone:
        break;
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
  LiveConfig live;

  // See FusedCompareApp::wire_override (duplicated — unrelated subclasses).
  template <typename V>
  void wire_override(const std::shared_ptr<holoscan::Operator>& target,
                     const char* port, const std::vector<V>& values,
                     const std::string& name) {
    if (values.empty()) {
      return;
    }
    if (live.rotate) {
      auto s = make_operator<perf::PerfRotatingOverrideSourceOp<V>>(
          name + "_ovr",
          make_condition<holoscan::CountCondition>(total_frames));
      s->configure(values, live.rotate_period, live.override_at);
      add_flow(s, target, {{"output", port}});
    } else {
      auto s = make_operator<perf::PerfOverrideSourceOp<V>>(
          name + "_ovr",
          make_condition<holoscan::CountCondition>(total_frames));
      s->configure(live.override_at, values.front());
      add_flow(s, target, {{"output", port}});
    }
  }

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

    // Timed rd_maps sink: syncing terminal (latency/live) vs non-syncing null
    // (sustained, so the split chain pipelines across its ops).
    if (live.sustained) {
      auto ns = make_operator<perf::PerfNullSinkOp>("rd_sink");
      add_flow(doppler, ns, {{"rd_maps", "input"}});
    } else {
      auto term = make_operator<perf::PerfTerminalSinkOp>("rd_sink");
      term->configure(collector);
      add_flow(doppler, term, {{"rd_maps", "input"}});
    }

    auto range_bins_sink =
        make_operator<perf::PerfNullSinkOp>("range_bins_sink");
    add_flow(range, range_bins_sink, {{"range_bins", "input"}});
    auto vel_bins_sink =
        make_operator<perf::PerfNullSinkOp>("velocity_bins_sink");
    add_flow(doppler, vel_bins_sink, {{"velocity_bins", "input"}});

    // [live] override ports differ per graph: the split stages own their own
    // nfft / window / waveform ports (waveform fanned to both stages).
    switch (live.param) {
      case LiveParam::kNfft:
        wire_override(range, "nfft", live.nfft_range, "nfft_range");
        wire_override(doppler, "nfft", live.nfft_doppler, "nfft_doppler");
        break;
      case LiveParam::kWindow:
        wire_override(range, "range_window", live.range_window, "range_window");
        wire_override(doppler, "doppler_window", live.doppler_window,
                      "doppler_window");
        break;
      case LiveParam::kWaveform:
        wire_override(range, "waveform", live.waveform, "wf_range");
        wire_override(doppler, "waveform", live.waveform, "wf_doppler");
        break;
      case LiveParam::kNone:
        break;
    }
  }
};

// Build and run one graph session for `scen` in the given mode, populating
// `collector`. Shared by the steady / live / sustained runners so the
// fused-vs-split branch lives in one place.
void run_graph_session(const CompareScenario& scen,
                       perf::TimingCollector* collector, const LiveConfig& live,
                       int total_frames) {
  auto cube = perf::single_map(
      perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen)), "adc_cube");
  collector->t_construct = perf::Clock::now();
  if (scen.graph == GraphKind::kFused) {
    auto app = holoscan::make_application<FusedCompareApp<PerfPrecision>>();
    app->scen = &scen;
    app->collector = collector;
    app->frame = std::move(cube);
    app->waveform = make_waveform();
    app->total_frames = total_frames;
    app->live = live;
    app->run();
  } else {
    auto app = holoscan::make_application<SplitCompareApp<PerfPrecision>>();
    app->scen = &scen;
    app->collector = collector;
    app->frame = std::move(cube);
    app->waveform = make_waveform();
    app->total_frames = total_frames;
    app->live = live;
    app->run();
  }
}

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

  std::cout << "\n=== split vs fused comparison (median_ms, team decision aid) "
               "===\n";
  std::cout << "No keep/kill verdict is encoded. Fill OPEN_ITEMS §1 from this "
               "table.\n";
  std::cout << std::left << std::setw(28) << "pair_id" << std::setw(8)
            << "size" << std::setw(8) << "clutter" << std::right
            << std::setw(12) << "split_ms" << std::setw(12) << "fused_ms"
            << std::setw(12) << "delta_ms" << std::setw(12) << "pct_saved"
            << std::setw(10)
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

// Parse the leading clutter tag out of a "clutter_graph" variant.
std::string clutter_of_variant(const std::string& variant) {
  const auto us = variant.find('_');
  return us == std::string::npos ? variant : variant.substr(0, us);
}

// Live-override table: per pair, the steady median and the override-tick cost
// (spike for one-shot; worst rebuild tick for rotate) for both graphs.
void print_live_table(const perf::PerfReport& report) {
  struct PairStats {
    std::string pair_id;
    std::string size_id;
    std::string clutter;
    double s_steady{0.0}, f_steady{0.0};
    double s_ovr{0.0}, f_ovr{0.0};
    bool have_split{false}, have_fused{false};
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
    p.clutter = clutter_of_variant(row.variant);
    if (row.calibration_mode == "split") {
      p.s_steady = row.latency.median_ms;
      p.s_ovr = row.override_ms;
      p.have_split = true;
    } else if (row.calibration_mode == "fused") {
      p.f_steady = row.latency.median_ms;
      p.f_ovr = row.override_ms;
      p.have_fused = true;
    }
  }

  const std::string mode =
      report.run_settings.live_rotate
          ? "rotate p=" + std::to_string(report.run_settings.rotate_period)
          : "one-shot";
  std::cout << "\n=== split vs fused LIVE override (" << mode << ", param="
            << report.run_settings.live_param << ") ===\n";
  std::cout << "steady_ms = post-warmup steady median; ovr_ms = "
            << (report.run_settings.live_rotate ? "worst rotate tick"
                                                : "override-tick spike")
            << "; ovr_d = ovr − steady.\n";
  std::cout << std::left << std::setw(28) << "pair_id" << std::setw(7) << "size"
            << std::setw(7) << "clutter" << std::right << std::setw(11)
            << "s_steady" << std::setw(11) << "f_steady" << std::setw(11)
            << "s_ovr" << std::setw(11) << "f_ovr" << std::setw(11) << "s_ovr_d"
            << std::setw(11) << "f_ovr_d\n";
  for (const auto& p : pairs) {
    if (!p.have_split || !p.have_fused) {
      continue;
    }
    std::cout << std::left << std::setw(28) << p.pair_id << std::setw(7)
              << p.size_id << std::setw(7) << p.clutter << std::right
              << std::fixed << std::setprecision(3) << std::setw(11)
              << p.s_steady << std::setw(11) << p.f_steady << std::setw(11)
              << p.s_ovr << std::setw(11) << p.f_ovr << std::setw(11)
              << (p.s_ovr - p.s_steady) << std::setw(11)
              << (p.f_ovr - p.f_steady) << "\n";
  }
  std::cout << "(ovr_d is the reconfiguration cost paid on the change tick; a "
               "small ovr_d means the warn-once plan/coeff cache is "
               "working)\n\n";
}

// Sustained-throughput table: true pipelined frames/sec for each graph (single
// end sync, so the split chain overlaps its stages), plus the fused/split
// ratio. Contrast these fps against the latency table's 1000/median.
void print_sustained_table(const perf::PerfReport& report) {
  struct PairStats {
    std::string pair_id, size_id, clutter;
    double s_fps{0.0}, f_fps{0.0};
    bool have_split{false}, have_fused{false};
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
    p.clutter = clutter_of_variant(row.variant);
    if (row.calibration_mode == "split") {
      p.s_fps = row.slices_per_second;
      p.have_split = true;
    } else if (row.calibration_mode == "fused") {
      p.f_fps = row.slices_per_second;
      p.have_fused = true;
    }
  }

  std::cout << "\n=== split vs fused SUSTAINED throughput (pipelined; sync "
               "once) ===\n";
  std::cout << "fps = frames/sec over the post-warmup window; fused/split > 1 "
               "means fused is faster.\n";
  std::cout << std::left << std::setw(28) << "pair_id" << std::setw(7) << "size"
            << std::setw(8) << "clutter" << std::right << std::setw(13)
            << "split_fps" << std::setw(13) << "fused_fps" << std::setw(13)
            << "fused/split\n";
  for (const auto& p : pairs) {
    if (!p.have_split || !p.have_fused) {
      continue;
    }
    const double ratio = p.s_fps > 0.0 ? p.f_fps / p.s_fps : 0.0;
    std::cout << std::left << std::setw(28) << p.pair_id << std::setw(7)
              << p.size_id << std::setw(8) << p.clutter << std::right
              << std::fixed << std::setprecision(1) << std::setw(13) << p.s_fps
              << std::setw(13) << p.f_fps << std::setprecision(3)
              << std::setw(13) << ratio << "\n";
  }
  std::cout << "(no per-frame sync, so the split ops may overlap; but on a "
               "single GPU they share the device, so sustained fps tends to "
               "track 1000/median rather than exceed it — compare to confirm "
               "per workload)\n\n";
}

struct CompareTraits : perf::PerfDriverDefaults {
  using Scenario = CompareScenario;

  // Opt into the --live / --sustained run modes (rejected on other tools).
  static constexpr bool has_live_modes = true;

  // An invalid live param (or an operator rejecting one) throws; isolate it so
  // one bad scenario cannot abort the whole report.
  static bool isolate_failures() { return true; }

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
    if (rs.sustained) {
      run_sustained(scen, rs, report);
      return;
    }
    if (!rs.live_param.empty()) {
      run_live(scen, rs, report);
      return;
    }
    // Default static steady comparison (a kNone LiveConfig wires no override
    // sources and keeps the device-syncing terminal sink).
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const LiveConfig none;
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          run_graph_session(scen, &collector, none, p.total_frames);
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);
    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    report->results.push_back(row);
  }

  // One-shot live override: reuse the shared override-spike bucketing (steady
  // median + peak on the change tick).
  static void run_live(const CompareScenario& scen, const perf::RunSettings& rs,
                       perf::PerfReport* report) {
    if (rs.live_rotate) {
      run_live_rotate(scen, rs, report);
      return;
    }
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    LiveConfig live = build_live_config(scen, rs);
    live.override_at = plan.override_at;
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          run_graph_session(scen, &collector, live, p.total_frames);
          return collector;
        });
    perf::append_setup_row(base_row(scen), samples, report);
    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes = "live one-shot: " + live_param_tag(live.param);
    report->results.push_back(row);
  }

  // Rotating override: the single-window bucketer does not fit continuous
  // thrash, so take every post-warmup latency — median is the thrash-steady,
  // max is the worst rebuild tick.
  static void run_live_rotate(const CompareScenario& scen,
                              const perf::RunSettings& rs,
                              perf::PerfReport* report) {
    LiveConfig live = build_live_config(scen, rs);
    live.override_at = rs.warmup_frames;  // rotate once warm
    const int total = rs.warmup_frames + rs.measured_frames;
    std::vector<double> steady;
    std::vector<double> setup_ms;
    double worst = std::numeric_limits<double>::quiet_NaN();
    for (int rep = 0; rep < rs.repeats; ++rep) {
      perf::TimingCollector collector;
      run_graph_session(scen, &collector, live, total);
      setup_ms.push_back(collector.setup_ms());
      const auto lat = collector.frame_latencies_ms();
      for (size_t i = static_cast<size_t>(rs.warmup_frames); i < lat.size();
           ++i) {
        steady.push_back(lat[i]);
        if (std::isnan(worst) || lat[i] > worst) {
          worst = lat[i];
        }
      }
    }
    perf::ResultRow setup_row = base_row(scen);
    setup_row.bucket = "setup";
    setup_row.setup_ms = perf::summarize(setup_ms).median_ms;
    report->results.push_back(setup_row);

    perf::ResultRow row = base_row(scen);
    row.bucket = "compute";
    row.latency = perf::summarize(steady);
    row.override_ms = worst;
    if (!std::isnan(worst) && !std::isnan(row.latency.median_ms)) {
      row.override_delta_ms = worst - row.latency.median_ms;
    }
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes = "live rotate p=" + std::to_string(rs.rotate_period) + ": " +
                live_param_tag(live.param);
    report->results.push_back(row);
  }

  // Sustained throughput: run N frames with a single end sync (no per-tick
  // barrier), so the split chain pipelines across its ops. The reported
  // median_ms is amortized per frame; slices_per_second is true frames/sec.
  static void run_sustained(const CompareScenario& scen,
                            const perf::RunSettings& rs,
                            perf::PerfReport* report) {
    LiveConfig live;
    live.sustained = true;
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const int measured = plan.total_frames - rs.warmup_frames;
    std::vector<double> amort;  // per-rep amortized ms/frame
    std::vector<double> setup_ms;
    for (int rep = 0; rep < rs.repeats; ++rep) {
      perf::TimingCollector collector;
      run_graph_session(scen, &collector, live, plan.total_frames);
      cudaDeviceSynchronize();
      const auto t1 = perf::Clock::now();
      setup_ms.push_back(collector.setup_ms());
      if (measured > 0 &&
          collector.starts.size() > static_cast<size_t>(rs.warmup_frames)) {
        const double wall_ms = perf::to_ms(
            t1 - collector.starts[static_cast<size_t>(rs.warmup_frames)]);
        amort.push_back(wall_ms / measured);
      }
    }
    perf::ResultRow setup_row = base_row(scen);
    setup_row.bucket = "setup";
    setup_row.setup_ms = perf::summarize(setup_ms).median_ms;
    report->results.push_back(setup_row);

    perf::ResultRow row = base_row(scen);
    row.bucket = "compute";
    row.latency = perf::summarize(amort);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes = "sustained (pipeline; sync once); median_ms amortized/frame";
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
    if (report == nullptr) {
      return;
    }
    if (report->run_settings.sustained) {
      print_sustained_table(*report);
    } else if (!report->run_settings.live_param.empty()) {
      print_live_table(*report);
    } else {
      print_comparison_table(*report);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CompareTraits>(argc, argv);
}
