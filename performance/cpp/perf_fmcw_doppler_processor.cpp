// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FMCWDopplerProcessorOp — windowed FFT along
// slow-time (chirps). Sweeps the (chirps x range_bins) size ladder, the
// Doppler window (rectangular / Hann), and a multi-channel batch.
//
// This driver measures Params-selected doppler_window configurations; it does not
// report mid-session live-override latency. Measurement tool only: no timing
// assertions.
// See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // v1 characterizes single precision (complex64).

struct SizeSpec {
  std::string id;
  int64_t num_chirps;  // slow-time FFT length
  int64_t num_range_bins;
  std::string role;
};

struct DopplerScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t num_range_bins{0};
  int64_t channels{1};  // leading batch axis (e.g. rx elements)
  hr::WindowType window{hr::WindowType::kNone};
  bool center_zero{true};
  hr::FftNorm fft_norm{hr::FftNorm::kBackward};
  int64_t nfft{0};
  double pri{25.0e-6};
  std::string live_port;
  std::string override_value;
  bool cached_override{false};
};

std::string window_tag(hr::WindowType w) {
  return w == hr::WindowType::kHann ? "hann" : "rect";
}

hr::WindowSpec window_of(hr::WindowType type) {
  hr::WindowSpec w;
  w.type = type;
  return w;
}

// Fixed valid waveform so velocity_bins is emitted every tick. Values only need
// to pass descriptor validation; they do not affect timing.
hr::FMCWWaveformDescriptor make_waveform(double pri = 25.0e-6) {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = pri;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<int64_t> profiles_shape(const DopplerScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.num_range_bins};
  }
  return {scen.num_chirps, scen.num_range_bins};
}

// ---------------------------------------------------------------------------
// Graph: range_profiles source -> Doppler op -> terminal (rd_maps) + null
// (velocity_bins). velocity_bins is always emitted because we set a valid
// waveform.
// ---------------------------------------------------------------------------
hr::FftNorm norm_of(const std::string& value) {
  if (value == "backward")
    return hr::FftNorm::kBackward;
  if (value == "ortho")
    return hr::FftNorm::kOrtho;
  if (value == "forward")
    return hr::FftNorm::kForward;
  throw std::invalid_argument("unknown FFT normalization: " + value);
}

hr::WindowType window_type_of(const std::string& value) {
  if (value == "none")
    return hr::WindowType::kNone;
  if (value == "hann")
    return hr::WindowType::kHann;
  if (value == "hamming")
    return hr::WindowType::kHamming;
  throw std::invalid_argument("unknown Doppler window: " + value);
}

perf::TimingCollector run_graph(const DopplerScenario& scen, const perf::RunPlan& plan) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(profiles_shape(scen));
  const auto configure = [&](holoscan::Graph& graph, const auto& src,
                             holoscan::CompileOptions& options, auto connect) {
    hr::MaxTensorExtents extents;
    extents.max_chirps = std::max<int64_t>(
        scen.live_port == "nfft" ? scen.num_chirps * 2 : scen.num_chirps, scen.nfft);
    extents.max_range_bins = scen.num_range_bins;
    if (scen.channels > 1)
      extents.max_rx = scen.channels;
    auto op = graph.op<hr::FMCWDopplerProcessorOp<PerfPrecision>>("doppler", extents);
    hr::FMCWDopplerProcessorParams params;
    params.waveform = make_waveform(scen.pri);
    params.doppler_window = window_of(scen.window);
    params.center_zero_doppler = scen.center_zero;
    params.fft_norm = scen.fft_norm;
    params.nfft = scen.nfft;
    op.set_params(std::move(params));
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(src->out, op->range_profiles, q);
    connect(graph, src, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "doppler");
    outputs.timed(op->rd_maps, "rd_maps");
    outputs.drain(op->velocity_bins, "velocity_bins");
    outputs.finish(src, op);
  };
  if (scen.live_port.empty()) {
    return perf::run_data_recipe(std::array{std::move(cube)}, plan, "doppler-perf",
                                 [&](auto& graph, const auto& src, auto& options) {
                                   configure(graph, src, options,
                                             [](auto&, const auto&, const auto&, const auto&) {});
                                 });
  }
  const auto cases = std::make_tuple(
      perf::live_option("doppler_window",
                        [&] {
                          return perf::live_case(
                              "doppler_window", window_of(scen.window),
                              window_of(window_type_of(scen.override_value)),
                              [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                graph.add_flow(src->override_out, op->doppler_window_in, q);
                              });
                        }),
      perf::live_option("center_zero_doppler",
                        [&] {
                          return perf::live_case(
                              "center_zero_doppler", scen.center_zero,
                              scen.override_value == "true",
                              [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                graph.add_flow(src->override_out, op->center_zero_doppler_in, q);
                              });
                        }),
      perf::live_option("waveform",
                        [&] {
                          return perf::live_case(
                              "waveform", hr::to_wire(make_waveform(scen.pri)),
                              hr::to_wire(make_waveform(std::stod(scen.override_value))),
                              [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                graph.add_flow(src->override_out, op->waveform_in, q);
                              });
                        }),
      perf::live_option("fft_norm",
                        [&] {
                          return perf::live_case(
                              "fft_norm", scen.fft_norm, norm_of(scen.override_value),
                              [](auto& graph, const auto& src, const auto& op, const auto& q) {
                                graph.add_flow(src->override_out, op->fft_norm_in, q);
                              });
                        }),
      perf::live_option("nfft", [&] {
        return perf::live_case(
            "nfft", scen.nfft, int64_t{2 * scen.num_chirps},
            [](auto& graph, const auto& src, const auto& op, const auto& q) {
              graph.add_flow(src->override_out, op->nfft_in, q);
            },
            scen.cached_override);
      }));
  return perf::dispatch_live_case(scen.live_port, cases, [&](const auto& selected) {
    return perf::run_selected_live_recipe(std::array{std::move(cube)}, plan, "doppler-perf",
                                          selected, configure);
  });
}

void fill_throughput(perf::ResultRow* row, const DopplerScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double doppler = static_cast<double>(scen.num_chirps);
  const double bins = static_cast<double>(scen.num_range_bins);
  row->cells_per_second = batch * doppler * bins / sec;
  row->slices_per_second = batch * bins / sec;

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double in_bytes = batch * doppler * bins * csz;
  const double out_bytes = batch * doppler * bins * csz + doppler * sizeof(PerfPrecision);
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const DopplerScenario& scen) {
  perf::ResultRow row;
  row.op_name = "FMCWDopplerProcessorOp";
  row.variant = window_tag(scen.window);
  perf::set_input_and_trailing_map_shape(row, profiles_shape(scen));
  row.out_batch = scen.channels;
  row.k = scen.num_chirps;
  row.live_port = scen.live_port;
  return row;
}

DopplerScenario make_scenario(hr::WindowType window, const SizeSpec& s, int64_t channels = 1) {
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
// Traits: the operator-specific surface used by the shared driver.
// ---------------------------------------------------------------------------
void validate_run_settings(const DopplerScenario& scen, const perf::RunSettings& rs) {
  if (scen.cached_override && rs.warmup_frames < 1)
    throw std::invalid_argument("cached transition requires at least one warmup frame");
}

void decorate_row(perf::ResultRow& row, const DopplerScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes = "end-to-end Graph latency; cells/s is complex RD-map cells/s";
  if (!scen.live_port.empty())
    row.notes += "; one-frame " + scen.live_port + " change";
}

struct DopplerTraits : perf::PerfDriverDefaults {
  using Scenario = DopplerScenario;

  static const char* tool_name() { return "perf_fmcw_doppler_processor"; }
  static const char* tool_description() {
    return "FMCWDopplerProcessorOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 2)
      throw std::invalid_argument("FMCWDoppler profile rank mismatch");
    return {id, shape, shape[0] * shape[1], id};
  }
  static DopplerScenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 2 || shape[0] < 2)
      throw std::invalid_argument("FMCWDoppler invalid profile shape");
    perf::require_keys(
        c.parameters,
        {"doppler_window", "center_zero_doppler", "fft_norm", "nfft", "pri", "channels"},
        "parameters");
    if (!c.parameters["doppler_window"] || !c.parameters["fft_norm"])
      throw std::invalid_argument("FMCWDoppler needs window and FFT norm");
    const auto window = window_type_of(c.parameters["doppler_window"].as<std::string>());
    const SizeSpec size{c.profile, shape[0], shape[1], c.profile};
    auto scenario = make_scenario(window, size);
    scenario.id = c.id;
    scenario.size_id = c.profile;
    scenario.fft_norm = norm_of(c.parameters["fft_norm"].as<std::string>());
    if (c.parameters["center_zero_doppler"])
      scenario.center_zero = c.parameters["center_zero_doppler"].as<bool>();
    if (c.parameters["channels"])
      scenario.channels = c.parameters["channels"].as<int64_t>();
    if (c.parameters["nfft"])
      scenario.nfft = c.parameters["nfft"].as<int64_t>();
    if (c.parameters["pri"])
      scenario.pri = c.parameters["pri"].as<double>();
    if (scenario.channels <= 0 || (scenario.nfft != 0 && scenario.nfft < shape[0]) ||
        scenario.nfft > 2 * shape[0] || scenario.pri <= 20.0e-6)
      throw std::invalid_argument("FMCWDoppler incompatible parameters");
    if (c.override_value) {
      scenario.live_port = (*c.override_value)["port"].as<std::string>();
      scenario.override_value = (*c.override_value)["value"].as<std::string>();
      if ((*c.override_value)["cache"]) {
        const auto cache = (*c.override_value)["cache"].as<std::string>();
        if (scenario.live_port != "nfft")
          throw std::invalid_argument("cached transition only applies to nfft");
        scenario.cached_override = cache == "cached";
      }
      if (scenario.live_port == "doppler_window") {
        (void)window_type_of(scenario.override_value);
      } else if (scenario.live_port == "center_zero_doppler") {
        if (scenario.override_value != "true" && scenario.override_value != "false")
          throw std::invalid_argument("invalid center_zero_doppler override");
      } else if (scenario.live_port == "waveform") {
        if (std::stod(scenario.override_value) <= 20.0e-6)
          throw std::invalid_argument("invalid waveform PRI override");
      } else if (scenario.live_port == "fft_norm") {
        (void)norm_of(scenario.override_value);
      } else if (scenario.live_port == "nfft") {
        if (scenario.override_value != "double")
          throw std::invalid_argument("nfft override must be double");
      } else {
        throw std::invalid_argument("unknown FMCWDoppler live port");
      }
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<DopplerTraits>(argc, argv);
}
