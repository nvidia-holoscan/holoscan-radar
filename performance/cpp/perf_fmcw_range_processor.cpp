// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FMCWRangeProcessorOp — windowed FFT along
// fast-time. Sweeps the (chirps x samples) size ladder, the range window
// (rectangular / Hann), and a multi-channel batch. Every compute run inlines a
// mid-session [live] range_window override so the window-coefficient rebuild
// (warn-once-and-cache) cost is measured. Measurement tool only: no timing
// assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.

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
};

// FFT-friendly powers of two so cuFFT takes its fast path; the ladder spans a
// smoke size up to a moderate production cube.
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

std::vector<int64_t> cube_shape(const RangeScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.samples_per_chirp};
  }
  return {scen.num_chirps, scen.samples_per_chirp};
}

// ---------------------------------------------------------------------------
// Mini-application: adc_cube source -> range op -> terminal (profiles) + null
// (bins) sinks + a [live] range_window override source.
// ---------------------------------------------------------------------------
template <typename T>
class RangePerfApp : public holoscan::Application {
 public:
  const RangeScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;  // adc_cube
  hr::FMCWWaveformDescriptor waveform;
  hr::WindowSpec override_window;
  int64_t override_at{0};
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::FMCWRangeProcessorOp<T>>(
        "range", Arg("waveform") = waveform,
        Arg("range_window") = window_of(scen->window));

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

    auto ovr = make_operator<perf::PerfOverrideSourceOp<hr::WindowSpec>>(
        "ovr", make_condition<holoscan::CountCondition>(total_frames));
    ovr->configure(override_at, override_window);
    add_flow(ovr, op, {{"output", "range_window"}});
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
  row.notes = "live_override=range_window";
  return row;
}

RangeScenario make_scenario(hr::WindowType window, const SizeSpec& s,
                            int64_t channels = 1) {
  RangeScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.samples_per_chirp = s.samples_per_chirp;
  sc.channels = channels;
  sc.window = window;
  std::ostringstream id;
  id << "range_" << window_tag(window) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

// ---------------------------------------------------------------------------
// Traits: the operator-specific surface the shared driver plugs into.
// FMCWRangeProcessorOp has [live] override ports (range_window exercised inline
// in run_scenario) and needs no process warmup or post-run summary, so it takes
// those defaults. It does not emit bool tensors, so no stderr filtering.
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

  static void run_scenario(const RangeScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          // Preallocate the input before the setup clock so the H2D upload is
          // not billed as operator setup.
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen)),
              "adc_cube");

          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<RangePerfApp<PerfPrecision>>();
          app->scen = &scen;
          app->collector = &collector;
          app->frame = std::move(cube);
          app->waveform = make_waveform();
          app->override_window = window_of(flip_window(scen.window));
          app->override_at = p.override_at;
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency including initial window/FFT plan "
        "build; median/p90 are pre-override steady state; override_ms is peak "
        "latency over the 3-frame live range_window rebuild window; cells/s is "
        "complex range bins/s";
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
