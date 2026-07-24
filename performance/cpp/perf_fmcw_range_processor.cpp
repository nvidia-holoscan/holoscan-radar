// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FMCWRangeProcessorOp — windowed FFT along
// fast-time. Sweeps the (chirps x samples) size ladder, the range window
// (rectangular / Hann), and a multi-channel batch. Every compute run inlines a
// mid-session [live] override so the reconfiguration cost is measured. The
// default/--full matrices exercise the range_window coefficient rebuild
// (warn-once-and-cache); --sweep-overrides walks every [live] param in turn
// (range_window, apply_dechirp, waveform, fft_norm, propagation_speed, nfft)
// and reports a per-param override_delta_ms so the report contrasts negligible
// sticky flips against rebuild-dominated reconfiguration. Measurement tool
// only: no timing assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

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

using PerfPrecision = float;  // v1 characterizes single precision (complex64).

// The [live] override under measurement in a scenario. One scenario exercises
// exactly one param so its override_delta_ms isolates that param's
// reconfiguration cost. A mix of cheap sticky flips (fft_norm,
// propagation_speed, apply_dechirp) and costly rebuilds (range_window
// coefficients, waveform dechirp reference, nfft cuFFT plan).
enum class OverrideKind {
  kRangeWindow,       // WindowSpec — costly: window-coefficient rebuild
  kApplyDechirp,      // bool — sticky flip that enables the dechirp path
  kWaveform,          // FMCWWaveformDescriptor — costly dechirp-ref rebuild
  kFftNorm,           // FftNorm — cheap sticky flip (scale only)
  kPropagationSpeed,  // double — cheap sticky flip (range axis only)
  kNfft,              // int64_t — costly: cuFFT plan rebuild (zero-pad)
};

const std::vector<OverrideKind>& all_override_kinds() {
  static const std::vector<OverrideKind> kinds = {
      OverrideKind::kRangeWindow,      OverrideKind::kApplyDechirp,
      OverrideKind::kWaveform,         OverrideKind::kFftNorm,
      OverrideKind::kPropagationSpeed, OverrideKind::kNfft};
  return kinds;
}

std::string override_tag(OverrideKind k) {
  switch (k) {
    case OverrideKind::kRangeWindow: return "range_window";
    case OverrideKind::kApplyDechirp: return "apply_dechirp";
    case OverrideKind::kWaveform: return "waveform";
    case OverrideKind::kFftNorm: return "fft_norm";
    case OverrideKind::kPropagationSpeed: return "propagation_speed";
    case OverrideKind::kNfft: return "nfft";
  }
  return "unknown";
}

struct SizeSpec {
  std::string id;
  int64_t num_chirps;
  int64_t samples_per_chirp;  // fast-time FFT length (natural nfft)
  std::string role;
};

struct RangeScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t samples_per_chirp{0};
  int64_t channels{1};  // leading batch axis (e.g. rx elements)
  hr::WindowType window{hr::WindowType::kNone};  // build-time range window
  bool apply_dechirp{false};  // build-time dechirp (on for the waveform sweep)

  // Live-override under test. Only the forcing value matching override_kind is
  // applied; the rest are derived inline in wire_override().
  OverrideKind override_kind{OverrideKind::kRangeWindow};
  hr::WindowSpec override_window;  // forcing value for kRangeWindow
  int64_t override_at{0};
};

// FFT-friendly powers of two so cuFFT takes its fast path; the ladder spans a
// smoke size up to a moderate production cube.
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

std::string window_tag(hr::WindowType w) {
  return w == hr::WindowType::kHann ? "hann" : "rect";
}

hr::WindowSpec window_of(hr::WindowType type) {
  hr::WindowSpec w;
  w.type = type;
  return w;
}

// Flip to the other window so a live override always forces a coefficient
// rebuild (rect <-> Hann).
hr::WindowType flip_window(hr::WindowType w) {
  return w == hr::WindowType::kHann ? hr::WindowType::kNone
                                    : hr::WindowType::kHann;
}

// A fixed, valid waveform descriptor so range_bins is emitted every tick. The
// exact values do not affect timing; they only need to pass descriptor
// validation (all finite and > 0, bandwidth defined).
hr::FMCWWaveformDescriptor make_waveform() {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25.0e-6;
  return hr::FMCWWaveformDescriptor(spec);
}

// A different valid waveform (double the bandwidth) so a live waveform override
// always differs from the build-time descriptor and forces the rebuild path.
hr::FMCWWaveformDescriptor make_override_waveform() {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 300.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25.0e-6;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<int64_t> cube_shape(const RangeScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.samples_per_chirp};
  }
  return {scen.num_chirps, scen.samples_per_chirp};
}

// ---------------------------------------------------------------------------
// Mini-application: adc_cube source -> range op -> terminal (profiles) + null
// (bins) sinks + one [live] override source selected by scen->override_kind.
// ---------------------------------------------------------------------------
template <typename T>
class RangePerfApp : public holoscan::Application {
 public:
  const RangeScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;  // adc_cube
  hr::FMCWWaveformDescriptor waveform;
  int64_t override_at{0};
  int total_frames{25};
  bool checksum{false};  // wire the byte-equivalence checksum sink

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::FMCWRangeProcessorOp<T>>(
        "range", Arg("waveform") = waveform,
        Arg("range_window") = window_of(scen->window),
        Arg("apply_dechirp") = scen->apply_dechirp);

    auto src = make_operator<perf::PerfTensorSourceOp<perf::AdcCubePort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"adc_cube", "adc_cube"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("profiles_sink");
    term->configure(collector);
    add_flow(op, term, {{"range_profiles", "input"}});

    // range_bins is optional but emitted (waveform is set); drain it so the
    // operator ticks and the axis path is exercised.
    auto ns = make_operator<perf::PerfNullSinkOp>("bins_sink");
    add_flow(op, ns, {{"range_bins", "input"}});

    // Byte-equivalence gate: fan range_profiles out to a checksum sink too (it
    // prints one output_checksum line on the first, pre-override frame).
    if (checksum) {
      auto cs = make_operator<perf::PerfChecksumSinkOp>("checksum_sink");
      add_flow(op, cs, {{"range_profiles", "input"}});
    }

    wire_override(op);
  }

 private:
  // Instantiate the one correctly-typed override source for scen->override_kind
  // and wire it to the matching [live] port, mirroring CFAR's wire_override.
  // The source emits a single forcing value on tick override_at that always
  // differs from the effective value, so the operator's override path fires.
  void wire_override(const std::shared_ptr<holoscan::Operator>& op) {
    using holoscan::CountCondition;
    switch (scen->override_kind) {
      case OverrideKind::kRangeWindow: {
        auto s = make_operator<perf::PerfOverrideSourceOp<hr::WindowSpec>>(
            "ovr", make_condition<CountCondition>(total_frames));
        s->configure(override_at, scen->override_window);
        add_flow(s, op, {{"output", "range_window"}});
        break;
      }
      case OverrideKind::kApplyDechirp: {
        auto s = make_operator<perf::PerfOverrideSourceOp<bool>>(
            "ovr", make_condition<CountCondition>(total_frames));
        s->configure(override_at, !scen->apply_dechirp);
        add_flow(s, op, {{"output", "apply_dechirp"}});
        break;
      }
      case OverrideKind::kWaveform: {
        auto s = make_operator<
            perf::PerfOverrideSourceOp<hr::FMCWWaveformDescriptor>>(
            "ovr", make_condition<CountCondition>(total_frames));
        s->configure(override_at, make_override_waveform());
        add_flow(s, op, {{"output", "waveform"}});
        break;
      }
      case OverrideKind::kFftNorm: {
        auto s = make_operator<perf::PerfOverrideSourceOp<hr::FftNorm>>(
            "ovr", make_condition<CountCondition>(total_frames));
        s->configure(override_at, hr::FftNorm::kOrtho);
        add_flow(s, op, {{"output", "fft_norm"}});
        break;
      }
      case OverrideKind::kPropagationSpeed: {
        auto s = make_operator<perf::PerfOverrideSourceOp<double>>(
            "ovr", make_condition<CountCondition>(total_frames));
        s->configure(override_at, hr::kDefaultPropagationSpeed * 0.5);
        add_flow(s, op, {{"output", "propagation_speed"}});
        break;
      }
      case OverrideKind::kNfft: {
        auto s = make_operator<perf::PerfOverrideSourceOp<int64_t>>(
            "ovr", make_condition<CountCondition>(total_frames));
        s->configure(override_at, 2 * scen->samples_per_chirp);
        add_flow(s, op, {{"output", "nfft"}});
        break;
      }
    }
  }
};

void fill_throughput(perf::ResultRow* row, const RangeScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double chirps = static_cast<double>(scen.num_chirps);
  const double n = static_cast<double>(scen.samples_per_chirp);
  row->cells_per_second = batch * chirps * n / sec;  // complex bins/s
  row->slices_per_second = batch * chirps / sec;     // range FFTs/s

  const double csz = sizeof(std::complex<PerfPrecision>);  // complex sample
  const double in_bytes = batch * chirps * n * csz;
  const double out_bytes = batch * chirps * n * csz +
                           n * sizeof(PerfPrecision);  // profiles + bins
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const RangeScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "FMCWRangeProcessorOp";
  row.variant = window_tag(scen.window);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_chirps, scen.samples_per_chirp};
  row.input_shape = cube_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.samples_per_chirp;  // transform length
  row.notes = "live_override=" + override_tag(scen.override_kind);
  return row;
}

std::string override_notes(OverrideKind kind) {
  return "cold_ms is first-frame latency including the initial window/FFT-plan "
         "build; median/p90 are pre-override steady state; override_ms is peak "
         "latency over the post-override window and override_delta_ms isolates "
         "the reconfiguration cost of the live " +
         override_tag(kind) + " param; cells/s is complex range bins/s";
}

RangeScenario make_scenario(hr::WindowType window, const SizeSpec& s,
                            int64_t channels = 1) {
  RangeScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.samples_per_chirp = s.samples_per_chirp;
  sc.channels = channels;
  sc.window = window;
  sc.override_kind = OverrideKind::kRangeWindow;
  sc.override_window = window_of(flip_window(window));
  std::ostringstream id;
  id << "range_" << window_tag(window) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

// One override-sweep scenario: a rect-window single-channel base exercising the
// single [live] param `kind`. The waveform sweep builds with dechirp on so a
// waveform change rebuilds the dechirp reference (the costly path); the input
// cube is complex, which dechirp requires.
RangeScenario make_override_scenario(OverrideKind kind, const SizeSpec& s) {
  RangeScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.samples_per_chirp = s.samples_per_chirp;
  sc.channels = 1;
  sc.window = hr::WindowType::kNone;
  sc.override_kind = kind;
  sc.override_window = window_of(hr::WindowType::kHann);
  sc.apply_dechirp = (kind == OverrideKind::kWaveform);
  std::ostringstream id;
  id << "range_ovr_" << override_tag(kind) << "_" << s.id << "_compute";
  sc.id = id.str();
  return sc;
}

// ---------------------------------------------------------------------------
// Traits: the operator-specific surface the shared driver plugs into.
// FMCWRangeProcessorOp has six [live] override ports; the default/--full
// matrices exercise range_window inline, and --sweep-overrides walks every
// param (build_override_sweep). It needs no post-run summary or stderr filter.
// ---------------------------------------------------------------------------
struct RangeTraits : perf::PerfDriverDefaults {
  using Scenario = RangeScenario;

  static const char* tool_name() { return "perf_fmcw_range_processor"; }
  static const char* tool_description() {
    return "FMCWRangeProcessorOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Rectangular-window size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Rectangular-window size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (windows x sizes x channel batch)";
  }
  static const char* override_sweep_help() {
    return "Per-param [live] override sweep (one scenario per param); "
           "--sizes selects sizes (default sm,md,xl)";
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

  static std::vector<RangeScenario> build_default_matrix() {
    std::vector<RangeScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(hr::WindowType::kNone, must_find_size(sid)));
    }
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(hr::WindowType::kHann, md));
    out.push_back(make_scenario(hr::WindowType::kNone, md, 4));
    return out;
  }

  static std::vector<RangeScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<RangeScenario> out;
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

  static std::vector<RangeScenario> build_variant_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<RangeScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      for (auto w : {hr::WindowType::kNone, hr::WindowType::kHann}) {
        out.push_back(make_scenario(w, *s));
      }
    }
    return out;
  }

  static std::vector<RangeScenario> build_full_matrix() {
    std::vector<RangeScenario> out;
    for (const auto& s : size_ladder()) {
      for (auto w : {hr::WindowType::kNone, hr::WindowType::kHann}) {
        out.push_back(make_scenario(w, s));
      }
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      for (int64_t ch : {4, 16}) {
        out.push_back(make_scenario(hr::WindowType::kNone, s, ch));
      }
    }
    return out;
  }

  // --sweep-overrides: one scenario per [live] param at each requested size
  // (default small/medium/large). Directly answers "cost of changing a
  // parameter at runtime, after initialization" per param.
  static std::vector<RangeScenario> build_override_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<std::string> use = sizes;
    if (use.empty()) {
      use = {"sm", "md", "xl"};
    }
    std::vector<RangeScenario> out;
    for (const auto& sid : use) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      for (auto kind : all_override_kinds()) {
        out.push_back(make_override_scenario(kind, *s));
      }
    }
    return out;
  }

  static void run_scenario(const RangeScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          // Preallocate the input before the setup clock so the H2D upload is
          // not billed as operator setup. Complex IQ so the dechirp path (used
          // by the waveform / apply_dechirp overrides) is valid.
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen)),
              "adc_cube");

          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<RangePerfApp<PerfPrecision>>();
          app->scen = &scen;
          app->collector = &collector;
          app->frame = std::move(cube);
          app->waveform = make_waveform();
          app->override_at = p.override_at;
          app->total_frames = p.total_frames;
          app->checksum = rs.checksum;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes = override_notes(scen.override_kind);
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
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<RangeTraits>(argc, argv);
}
