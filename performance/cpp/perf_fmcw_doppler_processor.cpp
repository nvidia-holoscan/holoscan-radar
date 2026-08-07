// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FMCWDopplerProcessorOp — windowed FFT along
// slow-time (chirps). Sweeps the (chirps x range_bins) size ladder, the
// Doppler window (rectangular / Hann), and a multi-channel batch. Every compute
// run inlines a mid-session [live] doppler_window override so the
// window-coefficient rebuild cost is measured. Measurement tool only: no timing
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

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
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
  int64_t num_chirps;     // slow-time FFT length (natural nfft -> num_doppler)
  int64_t num_range_bins;
  std::string role;
};

struct DopplerScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t num_range_bins{0};
  int64_t channels{1};  // leading batch axis (e.g. rx elements)
  hr::WindowType window{hr::WindowType::kNone};  // build-time Doppler window
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 16, 128, "small map; fast smoke"},
      {"sm", 32, 256, "compact sensor / few chirps"},
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

hr::WindowType flip_window(hr::WindowType w) {
  return w == hr::WindowType::kHann ? hr::WindowType::kNone
                                    : hr::WindowType::kHann;
}

// Fixed valid waveform so velocity_bins is emitted every tick. Values only need
// to pass descriptor validation; they do not affect timing.
hr::FMCWWaveformDescriptor make_waveform() {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25.0e-6;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<int64_t> profiles_shape(const DopplerScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.num_range_bins};
  }
  return {scen.num_chirps, scen.num_range_bins};
}

// ---------------------------------------------------------------------------
// Mini-application: range_profiles source -> Doppler op -> terminal (rd_maps) +
// null (velocity_bins) sinks + a [live] doppler_window override source.
// ---------------------------------------------------------------------------
template <typename T>
class DopplerPerfApp : public holoscan::Application {
 public:
  const DopplerScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;  // range_profiles
  hr::FMCWWaveformDescriptor waveform;
  hr::WindowSpec override_window;
  int64_t override_at{0};
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::FMCWDopplerProcessorOp<T>>(
        "doppler", Arg("waveform") = waveform,
        Arg("doppler_window") = window_of(scen->window));

    auto src = make_operator<perf::PerfTensorSourceOp<perf::RangeProfilesPort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"range_profiles", "range_profiles"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("rd_sink");
    term->configure(collector);
    add_flow(op, term, {{"rd_maps", "input"}});

    auto ns = make_operator<perf::PerfNullSinkOp>("vel_sink");
    add_flow(op, ns, {{"velocity_bins", "input"}});

    auto ovr = make_operator<perf::PerfOverrideSourceOp<hr::WindowSpec>>(
        "ovr", make_condition<holoscan::CountCondition>(total_frames));
    ovr->configure(override_at, override_window);
    add_flow(ovr, op, {{"output", "doppler_window"}});
  }
};

void fill_throughput(perf::ResultRow* row, const DopplerScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double doppler = static_cast<double>(scen.num_chirps);
  const double bins = static_cast<double>(scen.num_range_bins);
  row->cells_per_second = batch * doppler * bins / sec;  // complex cells/s
  row->slices_per_second = batch * bins / sec;  // Doppler FFTs/s (one per bin)

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double in_bytes = batch * doppler * bins * csz;
  const double out_bytes = batch * doppler * bins * csz +
                           doppler * sizeof(PerfPrecision);  // maps + vel_bins
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const DopplerScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "FMCWDopplerProcessorOp";
  row.variant = window_tag(scen.window);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_chirps, scen.num_range_bins};
  row.input_shape = profiles_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.num_chirps;  // transform length
  row.notes = "live_override=doppler_window";
  return row;
}

DopplerScenario make_scenario(hr::WindowType window, const SizeSpec& s,
                              int64_t channels = 1) {
  DopplerScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.num_range_bins = s.num_range_bins;
  sc.channels = channels;
  sc.window = window;
  std::ostringstream id;
  id << "doppler_" << window_tag(window) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

// ---------------------------------------------------------------------------
// Traits. FMCWDopplerProcessorOp has a [live] doppler_window override exercised
// inline in run_scenario; it needs no process warmup or post-run summary, and
// emits no bool tensors, so it takes the driver defaults for those.
// ---------------------------------------------------------------------------
struct DopplerTraits : perf::PerfDriverDefaults {
  using Scenario = DopplerScenario;

  static const char* tool_name() { return "perf_fmcw_doppler_processor"; }
  static const char* tool_description() {
    return "FMCWDopplerProcessorOp performance characterization";
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
      def.shape = {s.num_chirps, s.num_range_bins};
      def.cells = s.num_chirps * s.num_range_bins;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<DopplerScenario> build_default_matrix() {
    std::vector<DopplerScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(hr::WindowType::kNone, must_find_size(sid)));
    }
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(hr::WindowType::kHann, md));
    out.push_back(make_scenario(hr::WindowType::kNone, md, 4));
    return out;
  }

  static std::vector<DopplerScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<DopplerScenario> out;
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

  static std::vector<DopplerScenario> build_full_matrix() {
    std::vector<DopplerScenario> out;
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

  static void run_scenario(const DopplerScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(profiles_shape(scen)),
              "range_profiles");

          collector.t_construct = perf::Clock::now();
          auto app =
              holoscan::make_application<DopplerPerfApp<PerfPrecision>>();
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
        "latency over the 3-frame live doppler_window rebuild window; cells/s "
        "is complex rd-map cells/s";
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
  return holoscan::radar::perf::perf_main<DopplerTraits>(argc, argv);
}
