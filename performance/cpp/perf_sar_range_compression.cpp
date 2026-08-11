// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for SARRangeCompressionOp — windowed, optionally
// zero-padded inverse transform along fast-time, followed by the centring
// shift. Sweeps the (pulses x samples) size ladder, the range window
// (rectangular / Hann), zero-padding, and a multi-channel batch.
//
// Three things this harness is here to price, none of them the transform.
//
// **The centring fftshift is a second pass over the whole cube**, so it is
// bandwidth the FMCW range stage does not spend, and the padded scenarios say
// what that costs at 2x and 4x the transform length.
//
// **The range axis is optional**, and a caller who leaves BackprojectionOp to
// carry range_bin_spacing never supplies a waveform here. The `_noaxis`
// variants are that configuration, and differencing them against their
// axis-emitting twin prices the axis path on its own.
//
// **nfft is the only override expected to cost real time**, so it gets its own
// measurement rather than the compute table's single spike frame. Four
// transitions are separated, because they rebuild different things: an unseen
// length at 2x and at 4x (cuFFT plan + shift scratch + a larger output), an
// unseen *prime* length (Bluestein rather than a radix decomposition -- and the
// contract names a large-prime sample count as the reason nfft exists), and a
// return to a length this operator instance already ran, which should rebuild
// nothing and is the only direct check that the caching claim holds. Each is
// sampled once per graph session over several sessions, with the first session
// reported apart from the rest: the cuFFT plan cache is process-global, so only
// session 0 pays a genuinely cold build.
//
// Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_source.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // complex64, the CPHD ingest precision.

struct PhaseHistoryPort {
  static constexpr const char* kName = "phase_history";
};

struct SizeSpec {
  std::string id;
  int64_t num_pulses;
  int64_t num_samples;
  std::string role;
};

// Which [live] parameter the inlined mid-session override moves, and for nfft
// which *transition* — the four differ in what actually gets rebuilt, which is
// the whole question this harness exists to answer for nfft.
enum class OverrideKind {
  kWindow,      // rebuilds the window coefficients
  kNfftMiss2x,  // unseen length: cuFFT plan + shift scratch + larger output
  kNfftMiss4x,  // same, twice the length again -- is the cost length-dependent?
  kNfftPrime,   // unseen *and* not a fast length: cuFFT's Bluestein path
  kNfftReturn,  // back to a length this instance already ran: everything cached
};

bool is_nfft_override(OverrideKind kind) {
  return kind != OverrideKind::kWindow;
}

// nfft transitions are measured by the probe path (one sample per session,
// first session separated) rather than by the compute row's single spike frame.
bool is_probe(OverrideKind kind) { return is_nfft_override(kind); }

struct CompressionScenario {
  std::string id;
  std::string size_id;
  int64_t num_pulses{0};
  int64_t num_samples{0};
  int64_t channels{1};  // leading batch axis (a second polarization, say)
  int64_t pad_factor{1};  // nfft = pad_factor * num_samples
  hr::WindowType window{hr::WindowType::kNone};
  OverrideKind override_kind{OverrideKind::kWindow};
  // Whether a waveform descriptor is supplied, and so whether the range_bins
  // axis is emitted. Off is not an exotic configuration: a caller who lets
  // BackprojectionOp carry range_bin_spacing never wires a waveform here.
  bool emit_axis{true};
};

// Powers of two so cuFFT takes its fast path. Pulse counts are an aperture
// slab per tick rather than a whole collection: BackprojectionOp accumulates
// across ticks, so this stage sees the reader's batch.
const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 16, 512, "small slab; fast smoke"},
      {"sm", 32, 1024, "compact aperture slab"},
      {"md", 64, 2048, "common dev size"},
      {"lg", 128, 4096, "moderate production slab"},
      {"xl", 256, 8192, "range-heavy slab (v1 upper bound)"}};
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

std::string window_tag(hr::WindowType w) {
  return w == hr::WindowType::kHann ? "hann" : "rect";
}

hr::WindowSpec window_of(hr::WindowType type) {
  hr::WindowSpec w;
  w.type = type;
  return w;
}

// Flip to the other window so a live override always forces a rebuild.
hr::WindowType flip_window(hr::WindowType w) {
  return w == hr::WindowType::kHann ? hr::WindowType::kNone
                                    : hr::WindowType::kHann;
}

// A fixed, valid descriptor so range_bins is emitted every tick. The values
// only need to pass validation; they do not affect timing.
hr::LFMWaveformDescriptor make_waveform() {
  hr::LFMWaveformSpec spec;
  spec.carrier_frequency = 9.6e9;
  spec.bandwidth = 600.0e6;
  spec.chirp_duration = 10.0e-6;
  spec.sample_rate = 720.0e6;
  spec.chirp_repetition_interval = 1.0e-3;
  return hr::LFMWaveformDescriptor(spec);
}

std::vector<int64_t> cube_shape(const CompressionScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_pulses, scen.num_samples};
  }
  return {scen.num_pulses, scen.num_samples};
}

int64_t nfft_of(const CompressionScenario& scen) {
  return scen.pad_factor * scen.num_samples;
}

bool is_prime(int64_t n) {
  if (n < 2) {
    return false;
  }
  for (int64_t d = 2; d * d <= n; ++d) {
    if (n % d == 0) {
      return false;
    }
  }
  return true;
}

// Smallest prime >= n. cuFFT has no radix decomposition for a prime length, so
// it falls back to Bluestein -- a different algorithm, not a slower plan of the
// same one. This is the length a caller reaches nfft for in the first place
// (the contract calls it "the escape hatch for a large-prime num_samples"), so
// it is worth its own transition rather than being assumed to behave like 2x.
int64_t next_prime(int64_t n) {
  while (!is_prime(n)) {
    ++n;
  }
  return n;
}

const std::vector<OverrideKind>& nfft_transitions() {
  static const std::vector<OverrideKind> kinds = {
      OverrideKind::kNfftMiss2x, OverrideKind::kNfftMiss4x,
      OverrideKind::kNfftPrime, OverrideKind::kNfftReturn};
  return kinds;
}

// The length the override moves *to*. kNfftReturn goes up first and comes back,
// so its target is the original length; the schedule below encodes that.
//
// The prime target moves with the session index, and the others do not. A
// power-of-two plan is cheap enough that reusing one length across sessions
// costs nothing to interpret, but the first prime measured 350x the cost of a
// power of two, and that number means two very different things depending on
// whether it is per-length or once-per-process. Handing every session a prime
// it has never seen answers that directly: if the cost is a Bluestein plan, it
// recurs; if it is cuFFT loading Bluestein machinery once, it does not.
int64_t override_target_nfft(const CompressionScenario& scen, int session = 0) {
  const int64_t base = nfft_of(scen);
  switch (scen.override_kind) {
    case OverrideKind::kNfftMiss4x:
      return 4 * base;
    case OverrideKind::kNfftPrime:
      return next_prime(2 * base + 1 + 64 * session);
    case OverrideKind::kNfftReturn:
      return base;
    default:
      return 2 * base;
  }
}

// The descriptor output is not a tensor, so the shared gxf::Entity null sink
// cannot drain it. Default (message-available) input condition: the port emits
// on the first tick and on change, and a kNone input would leave this sink
// permanently ready, spinning the scheduler for the whole run.
class WindowSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(WindowSinkOp)
  WindowSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<hr::WindowSpec>("range_window");
  }
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    (void)op_input.receive<hr::WindowSpec>("range_window");
  }
};

// Emits a scheduled sequence of override values -- (tick, value) pairs -- so a
// scenario can drive the operator through more than one reconfiguration in a
// single session. The shared PerfOverrideSourceOp fires exactly once, which
// cannot express "leave the cached length and come back to it".
template <typename V>
class ScheduledOverrideSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ScheduledOverrideSourceOp)
  ScheduledOverrideSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<V>("output");
  }
  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    for (const auto& [tick, value] : schedule_) {
      if (tick == tick_) {
        op_output.emit(value, "output");
        break;
      }
    }
    ++tick_;
  }
  void configure(std::vector<std::pair<int64_t, V>> schedule) {
    schedule_ = std::move(schedule);
  }

 private:
  int64_t tick_{0};
  std::vector<std::pair<int64_t, V>> schedule_;
};

// ---------------------------------------------------------------------------
// Mini-application: phase_history source -> compression -> terminal (profiles)
// + null (bins) + window sinks, plus one [live] override source.
// ---------------------------------------------------------------------------
template <typename T>
class CompressionPerfApp : public holoscan::Application {
 public:
  const CompressionScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  hr::LFMWaveformDescriptor waveform;
  int64_t override_at{0};
  // Non-empty for the nfft probe path, which needs two reconfigurations in one
  // session to reach a cached length.
  std::vector<std::pair<int64_t, int64_t>> nfft_schedule;
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto op = scen->emit_axis
                  ? make_operator<hr::SARRangeCompressionOp<T>>(
                        "compression", Arg("waveform") = waveform,
                        Arg("range_window") = window_of(scen->window),
                        Arg("nfft") = nfft_of(*scen))
                  : make_operator<hr::SARRangeCompressionOp<T>>(
                        "compression",
                        Arg("range_window") = window_of(scen->window),
                        Arg("nfft") = nfft_of(*scen));

    auto src = make_operator<perf::PerfTensorSourceOp<PhaseHistoryPort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"phase_history", "phase_history"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("profiles_sink");
    term->configure(collector);
    add_flow(op, term, {{"range_profiles", "input"}});

    // Without a waveform the port never emits, so wiring a sink to it would
    // leave that sink waiting on a message that never comes.
    if (scen->emit_axis) {
      auto bins = make_operator<perf::PerfNullSinkOp>("bins_sink");
      add_flow(op, bins, {{"range_bins", "input"}});
    }

    auto windows = make_operator<WindowSinkOp>("window_sink");
    add_flow(op, windows, {{"range_window", "range_window"}});

    if (!nfft_schedule.empty()) {
      auto ovr = make_operator<ScheduledOverrideSourceOp<int64_t>>(
          "ovr", make_condition<holoscan::CountCondition>(total_frames));
      ovr->configure(nfft_schedule);
      add_flow(ovr, op, {{"output", "nfft"}});
    } else {
      auto ovr = make_operator<perf::PerfOverrideSourceOp<hr::WindowSpec>>(
          "ovr", make_condition<holoscan::CountCondition>(total_frames));
      ovr->configure(override_at, window_of(flip_window(scen->window)));
      add_flow(ovr, op, {{"output", "range_window"}});
    }
  }
};

void fill_throughput(perf::ResultRow* row, const CompressionScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double pulses = static_cast<double>(scen.num_pulses);
  const double bins = static_cast<double>(nfft_of(scen));
  row->cells_per_second = batch * pulses * bins / sec;  // output bins/s
  row->slices_per_second = batch * pulses / sec;        // range transforms/s

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double in_bytes = batch * pulses * static_cast<double>(scen.num_samples)
                          * csz;
  // Read once, written to scratch, read back by the shift, written out: the
  // centring pass is why the output side is counted twice.
  const double out_bytes = 3.0 * batch * pulses * bins * csz +
                           bins * sizeof(PerfPrecision);
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

std::string transition_tag(OverrideKind kind) {
  switch (kind) {
    case OverrideKind::kNfftMiss2x:
      return "miss_2x";
    case OverrideKind::kNfftMiss4x:
      return "miss_4x";
    case OverrideKind::kNfftPrime:
      return "prime";
    case OverrideKind::kNfftReturn:
      return "hit_return";
    default:
      return "window";
  }
}

std::string variant_of(const CompressionScenario& scen) {
  std::ostringstream variant;
  variant << window_tag(scen.window);
  if (scen.pad_factor > 1) {
    variant << "_pad" << scen.pad_factor;
  }
  if (!scen.emit_axis) {
    variant << "_noaxis";
  }
  if (is_nfft_override(scen.override_kind)) {
    variant << "_nfft_" << transition_tag(scen.override_kind);
  }
  return variant.str();
}

perf::ResultRow base_row(const CompressionScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "SARRangeCompressionOp";
  row.variant = variant_of(scen);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_pulses, scen.num_samples};
  row.input_shape = cube_shape(scen);
  row.out_batch = scen.channels;
  row.k = nfft_of(scen);  // transform length
  return row;
}

CompressionScenario make_scenario(
    hr::WindowType window, const SizeSpec& s, int64_t channels = 1,
    int64_t pad_factor = 1,
    OverrideKind override_kind = OverrideKind::kWindow,
    bool emit_axis = true) {
  CompressionScenario sc;
  sc.size_id = s.id;
  sc.num_pulses = s.num_pulses;
  sc.num_samples = s.num_samples;
  sc.channels = channels;
  sc.pad_factor = pad_factor;
  sc.window = window;
  sc.override_kind = override_kind;
  sc.emit_axis = emit_axis;
  std::ostringstream id;
  id << "compress_" << variant_of(sc) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << (is_probe(override_kind) ? "_probe" : "_compute");
  sc.id = id.str();
  return sc;
}

// Frame layout for a probe session. The override lands at `first_at`; a return
// transition adds a second one two ticks later, so the length it returns to has
// been left and is genuinely being re-entered from the cache. Steady windows
// bracket the whole thing.
struct ProbePlan {
  int64_t first_at{0};
  int64_t measured_at{0};  // the tick whose latency is the probe's sample
  int64_t steady_before_end{0};
  int64_t steady_after_begin{0};
  int total_frames{0};
};

ProbePlan make_probe_plan(const perf::RunSettings& rs,
                          const CompressionScenario& scen) {
  ProbePlan p;
  const int before = std::max(3, rs.measured_frames / 2);
  p.steady_before_end = rs.warmup_frames + before;
  p.first_at = p.steady_before_end;
  p.measured_at = scen.override_kind == OverrideKind::kNfftReturn
                      ? p.first_at + 2
                      : p.first_at;
  p.steady_after_begin = p.measured_at + 1;
  p.total_frames = static_cast<int>(p.steady_after_begin) + 3;
  return p;
}

std::vector<std::pair<int64_t, int64_t>> nfft_schedule_of(
    const CompressionScenario& scen, const ProbePlan& plan, int session) {
  const int64_t base = nfft_of(scen);
  if (scen.override_kind == OverrideKind::kNfftReturn) {
    return {{plan.first_at, 2 * base}, {plan.measured_at, base}};
  }
  return {{plan.measured_at, override_target_nfft(scen, session)}};
}

// ---------------------------------------------------------------------------
// Traits: the operator-specific surface the shared driver plugs into.
// ---------------------------------------------------------------------------
struct CompressionTraits : perf::PerfDriverDefaults {
  using Scenario = CompressionScenario;

  static const char* tool_name() { return "perf_sar_range_compression"; }
  static const char* tool_description() {
    return "SARRangeCompressionOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Rectangular-window, unpadded size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Rectangular-window, unpadded size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (windows x sizes x zero-padding x channel batch), "
           "plus the axis-off variants and the four nfft override probes";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_pulses, s.num_samples};
      def.cells = s.num_pulses * s.num_samples;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<CompressionScenario> build_default_matrix() {
    std::vector<CompressionScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(hr::WindowType::kNone, must_find_size(sid)));
    }
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(hr::WindowType::kHann, md));
    // The axes this operator has that the FMCW range stage does not: what
    // zero-padding costs, and what emitting the range axis costs.
    out.push_back(make_scenario(hr::WindowType::kNone, md, 1, 2));
    out.push_back(make_scenario(hr::WindowType::kNone, md, 4));
    out.push_back(make_scenario(hr::WindowType::kNone, md, 1, 1,
                                OverrideKind::kWindow, /*emit_axis=*/false));
    for (auto kind : nfft_transitions()) {
      out.push_back(make_scenario(hr::WindowType::kNone, md, 1, 1, kind));
    }
    return out;
  }

  static std::vector<CompressionScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<CompressionScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(hr::WindowType::kNone, *s));
    }
    return out;
  }

  static std::vector<CompressionScenario> build_full_matrix() {
    std::vector<CompressionScenario> out;
    for (const auto& s : size_ladder()) {
      for (auto w : {hr::WindowType::kNone, hr::WindowType::kHann}) {
        out.push_back(make_scenario(w, s));
      }
      // xl at 4x is 32k-point transforms over a 256-pulse slab; the ladder's
      // upper end is padded at 2x only so the matrix stays runnable.
      out.push_back(make_scenario(hr::WindowType::kNone, s, 1, 2));
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      out.push_back(make_scenario(hr::WindowType::kNone, s, 1, 4));
      for (int64_t ch : {4, 16}) {
        out.push_back(make_scenario(hr::WindowType::kNone, s, ch));
      }
      out.push_back(make_scenario(hr::WindowType::kNone, s, 1, 1,
                                  OverrideKind::kWindow, /*emit_axis=*/false));
      for (auto kind : nfft_transitions()) {
        out.push_back(make_scenario(hr::WindowType::kNone, s, 1, 1, kind));
      }
    }
    return out;
  }

  // One graph session for a probe scenario, returning its frame latencies.
  static std::vector<double> run_probe_session(
      const CompressionScenario& scen, const ProbePlan& plan, int session) {
    perf::TimingCollector collector;
    auto cube = perf::single_map(
        perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen)),
        "phase_history");

    collector.t_construct = perf::Clock::now();
    auto app = holoscan::make_application<CompressionPerfApp<PerfPrecision>>();
    app->scen = &scen;
    app->collector = &collector;
    app->frame = std::move(cube);
    app->waveform = make_waveform();
    app->nfft_schedule = nfft_schedule_of(scen, plan, session);
    app->total_frames = plan.total_frames;
    app->run();
    return collector.frame_latencies_ms();
  }

  // The nfft path: sample the override tick once per session across several
  // sessions, keeping the first apart from the rest. Everything process-global
  // -- the cuFFT plan cache above all -- is cold only in session 0, so pooling
  // that sample with the others would report an average nobody pays.
  static void run_probe_scenario(const CompressionScenario& scen,
                                 const perf::RunSettings& rs,
                                 perf::PerfReport* report) {
    const ProbePlan plan = make_probe_plan(rs, scen);
    const int sessions = std::max(rs.repeats, 5);

    double first = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> later;
    std::vector<double> before;
    std::vector<double> after;
    for (int session = 0; session < sessions; ++session) {
      const auto lat = run_probe_session(scen, plan, session);
      if (static_cast<int64_t>(lat.size()) <= plan.measured_at) {
        continue;
      }
      const double sample = lat[static_cast<size_t>(plan.measured_at)];
      if (session == 0) {
        first = sample;
      } else {
        later.push_back(sample);
      }
      for (int64_t i = rs.warmup_frames; i < plan.steady_before_end; ++i) {
        before.push_back(lat[static_cast<size_t>(i)]);
      }
      for (size_t i = static_cast<size_t>(plan.steady_after_begin);
           i < lat.size(); ++i) {
        after.push_back(lat[i]);
      }
    }

    perf::OverrideProbeRow row;
    row.scenario_id = scen.id;
    row.op_name = "SARRangeCompressionOp";
    row.size_id = scen.size_id;
    row.parameter = "nfft";
    row.transition = transition_tag(scen.override_kind);
    std::ostringstream from_to;
    if (scen.override_kind == OverrideKind::kNfftReturn) {
      from_to << 2 * nfft_of(scen) << " -> " << nfft_of(scen);
    } else if (scen.override_kind == OverrideKind::kNfftPrime) {
      // A different prime each session, so name the range rather than a value.
      from_to << nfft_of(scen) << " -> " << override_target_nfft(scen, 0)
              << "+";
    } else {
      from_to << nfft_of(scen) << " -> " << override_target_nfft(scen);
    }
    row.from_to = from_to.str();
    row.first_ms = first;
    row.later = perf::summarize(later);
    row.steady_before_ms = perf::summarize(before).median_ms;
    row.steady_after_ms = perf::summarize(after).median_ms;
    switch (scen.override_kind) {
      case OverrideKind::kNfftReturn:
        row.notes =
            "cached: plan, scratch and output shape all already exist";
        break;
      case OverrideKind::kNfftPrime:
        row.notes =
            "a different unseen prime every session, so med_ms is a "
            "Bluestein plan build and not a cache hit";
        break;
      default:
        row.notes = "plan build + scratch alloc + larger output";
        break;
    }
    report->override_probe.push_back(std::move(row));
  }

  static void run_scenario(const CompressionScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    if (is_probe(scen.override_kind)) {
      run_probe_scenario(scen, rs, report);
      return;
    }
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          // Preallocate the input before the setup clock so the H2D upload is
          // not billed as operator setup.
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen)),
              "phase_history");

          collector.t_construct = perf::Clock::now();
          auto app =
              holoscan::make_application<CompressionPerfApp<PerfPrecision>>();
          app->scen = &scen;
          app->collector = &collector;
          app->frame = std::move(cube);
          app->waveform = make_waveform();
          app->override_at = p.override_at;
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes = scen.emit_axis
                    ? "live_override=range_window (coefficient rebuild)"
                    : "live_override=range_window (coefficient rebuild); no "
                      "waveform, so range_bins is not emitted";
    report->results.push_back(row);
  }

  // Pay CUDA + Holoscan first-Application (and first cuFFT plan) costs on a
  // discarded xs run so scenario #1 cold_ms is not inflated by process init.
  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(hr::WindowType::kNone, must_find_size("xs")),
                   warm, &discarded);
    });
  }

  static void post_run(perf::PerfReport* report) {
    report->footnotes.push_back(
        "cells/s counts output range bins per second, so a padded scenario "
        "produces pad_factor times as many of them from the same input.");
    report->footnotes.push_back(
        "GB/s counts the output side three times -- written by the transform, "
        "read by the centring shift, written again -- because the fftshift is "
        "a separate full pass over the cube rather than part of the "
        "transform.");
    report->footnotes.push_back(
        "_noaxis scenarios supply no waveform, so range_bins is never emitted. "
        "Difference one against its axis-emitting twin at the same size to "
        "price the axis path, which rebuilds the axis on the host and copies "
        "it to the device every tick.");
    report->footnotes.push_back(
        "In the override probe, first_ms is session 0 and med_ms is the "
        "sessions after it. They differ because cuFFT's plan cache is "
        "process-global and outlives a session: first_ms is the cost of a plan "
        "nobody has built, med_ms the cost of a fresh operator instance "
        "reaching a plan the process already has. A long-running application "
        "pays first_ms once per distinct length and med_ms never.");
    report->footnotes.push_back(
        "hit_return leaves the configured length for double it and comes back. "
        "Nothing is rebuilt on the way back -- the plan, the shift scratch and "
        "the output shape all already exist -- so its med_ms against a miss "
        "row is the value of the caches, and against before_ms is what a "
        "cached override costs over no override at all.");
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CompressionTraits>(argc, argv);
}
