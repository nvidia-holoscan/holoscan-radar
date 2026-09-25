// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FMCWRangeDopplerProcessorOp — fused
// range + optional inline clutter + Doppler in one graph node.  Sweeps
// the (chirps x samples) size ladder, the inline clutter stage (off /
// mean subtraction), and a multi-channel batch.
//
// Profiles each live input in isolation as a one-frame Exact override.
// Measurement only; no timing assertions.
//
// This tool characterizes the fused operator as an individual Graph node.
// See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // v1 characterizes single precision (complex64).

struct SizeSpec {
  std::string id;
  int64_t num_chirps;
  int64_t samples_per_chirp;
  std::string role;
};

struct FusedScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t samples_per_chirp{0};
  int64_t channels{1};
  hr::ClutterFilterMethod clutter{hr::ClutterFilterMethod::kNone};
  hr::WindowType doppler_window{hr::WindowType::kNone};
  hr::WindowType range_window{hr::WindowType::kNone};
  bool apply_dechirp{false};
  bool center_zero{true};
  hr::FftNorm fft_norm{hr::FftNorm::kBackward};
  int64_t nfft_range{0};
  int64_t nfft_doppler{0};
  double pri{25.0e-6};
  std::string live_port;
  std::string override_value;
};

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

hr::FMCWWaveformDescriptor make_waveform(double pri = 25.0e-6) {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = pri;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<int64_t> cube_shape(const FusedScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.samples_per_chirp};
  }
  return {scen.num_chirps, scen.samples_per_chirp};
}

// -------------------------------------------------------------------
// Graph: adc_cube source → fused op → terminal (rd_maps) + null
// (range_bins, velocity_bins).  Both axis outputs are always emitted
// here because a valid waveform is set.
// -------------------------------------------------------------------
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
  throw std::invalid_argument("unknown fused window: " + value);
}
hr::ClutterFilterMethod clutter_of(const std::string& value) {
  if (value == "none")
    return hr::ClutterFilterMethod::kNone;
  if (value == "mean_subtraction")
    return hr::ClutterFilterMethod::kMeanSubtraction;
  if (value == "single_delay")
    return hr::ClutterFilterMethod::kSingleDelay;
  if (value == "double_delay")
    return hr::ClutterFilterMethod::kDoubleDelay;
  throw std::invalid_argument("unknown fused clutter mode: " + value);
}

perf::TimingCollector run_graph(const FusedScenario& scen, const perf::RunPlan& plan) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(cube_shape(scen));
  const auto configure = [&](holoscan::Graph& graph, const auto& src,
                             holoscan::CompileOptions& options, auto connect) {
    hr::MaxTensorExtents extents;
    extents.max_chirps =
        std::max<int64_t>(scen.live_port == "nfft_doppler" ? scen.num_chirps * 2 : scen.num_chirps,
                          scen.nfft_doppler);
    extents.max_range_bins = std::max<int64_t>(
        scen.live_port == "nfft_range" ? scen.samples_per_chirp * 2 : scen.samples_per_chirp,
        scen.nfft_range);
    if (scen.channels > 1)
      extents.max_rx = scen.channels;
    auto op = graph.op<hr::FMCWRangeDopplerProcessorOp<PerfPrecision>>("fused", extents);
    hr::FMCWRangeDopplerProcessorParams params;
    params.waveform = make_waveform(scen.pri);
    params.clutter_filter = scen.clutter;
    params.range_window.type = scen.range_window;
    params.doppler_window.type = scen.doppler_window;
    params.apply_dechirp = scen.apply_dechirp;
    params.fft_norm = scen.fft_norm;
    params.center_zero_doppler = scen.center_zero;
    params.nfft_range = scen.nfft_range;
    params.nfft_doppler = scen.nfft_doppler;
    op.set_params(std::move(params));
    const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
    const auto rb = graph.op<perf::PerfNullSinkOp>("rb_sink");
    const auto vb = graph.op<perf::PerfNullSinkOp>("vb_sink");
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(src->out, op->adc_cube, q);
    connect(graph, src, op, q);
    graph.add_flow(op->rd_maps, sink->in, q);
    graph.add_flow(op->range_bins, rb->in, q);
    graph.add_flow(op->velocity_bins, vb->in, q);
    perf::bind_tensor_out(options, "fused", "rd_maps");
    perf::bind_tensor_out(options, "fused", "range_bins");
    perf::bind_tensor_out(options, "fused", "velocity_bins");
  };
  if (scen.live_port == "range_window") {
    return perf::run_perf_graph_with_window_override(
        std::move(cube), plan.total_frames, plan.override_at,
        hr::WindowSpec{.type = scen.range_window},
        hr::WindowSpec{.type = window_type_of(scen.override_value)}, "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->range_window_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "doppler_window") {
    return perf::run_perf_graph_with_window_override(
        std::move(cube), plan.total_frames, plan.override_at,
        hr::WindowSpec{.type = scen.doppler_window},
        hr::WindowSpec{.type = window_type_of(scen.override_value)}, "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->doppler_window_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "apply_dechirp") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.apply_dechirp,
        scen.override_value == "true", "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->apply_dechirp_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "clutter_filter") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.clutter,
        clutter_of(scen.override_value), "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->clutter_filter_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "waveform") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, hr::to_wire(make_waveform(scen.pri)),
        hr::to_wire(make_waveform(std::stod(scen.override_value))), "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->waveform_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "fft_norm") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.fft_norm,
        norm_of(scen.override_value), "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->fft_norm_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "center_zero_doppler") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.center_zero,
        scen.override_value == "true", "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->center_zero_doppler_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "nfft_range") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.nfft_range,
        int64_t{2 * scen.samples_per_chirp}, "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->nfft_range_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scen.live_port == "nfft_doppler") {
    return perf::run_perf_graph_with_override(
        std::move(cube), plan.total_frames, plan.override_at, scen.nfft_doppler,
        int64_t{2 * scen.num_chirps}, "rdproc-perf",
        [&](auto& graph, const auto& src, auto& options) {
          configure(graph, src, options,
                    [](auto& g, const auto& source, const auto& op, const auto& q) {
                      g.add_flow(source->override_out, op->nfft_doppler_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  return perf::run_perf_graph(std::move(cube), plan.total_frames, "rdproc-perf",
                              [&](auto& graph, const auto& src, auto& options) {
                                configure(graph, src, options,
                                          [](auto&, const auto&, const auto&, const auto&) {});
                              });
}

void fill_throughput(perf::ResultRow* row, const FusedScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double chirps = static_cast<double>(scen.num_chirps);
  const double n = static_cast<double>(scen.samples_per_chirp);
  row->cells_per_second = batch * chirps * n / sec;  // complex rd-map cells/s
  row->slices_per_second = batch / sec;              // full cubes/s

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double in_bytes = batch * chirps * n * csz;
  const double out_bytes = batch * chirps * n * csz + (chirps + n) * sizeof(PerfPrecision);
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const FusedScenario& scen) {
  perf::ResultRow row;
  row.op_name = "FMCWRangeDopplerProcessorOp";
  row.variant = clutter_tag(scen.clutter) + "_" + std::string(hr::to_string(scen.doppler_window));
  row.map_shape = {scen.num_chirps, scen.samples_per_chirp};
  row.input_shape = cube_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.samples_per_chirp;
  row.live_port = scen.live_port;
  return row;
}

FusedScenario make_scenario(hr::ClutterFilterMethod clutter, const SizeSpec& s,
                            int64_t channels = 1,
                            hr::WindowType doppler_window = hr::WindowType::kNone) {
  FusedScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.samples_per_chirp = s.samples_per_chirp;
  sc.channels = channels;
  sc.clutter = clutter;
  sc.doppler_window = doppler_window;
  std::ostringstream id;
  id << "rdproc_" << clutter_tag(clutter) << "_" << hr::to_string(doppler_window) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

// -------------------------------------------------------------------
// Traits: the operator-specific surface used by the shared driver.
// -------------------------------------------------------------------
void decorate_row(perf::ResultRow& row, const FusedScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes = "end-to-end Graph latency; cells/s is complex RD-map cells/s";
  if (!scen.live_port.empty())
    row.notes += "; one-frame " + scen.live_port + " change";
}

struct FusedTraits : perf::PerfDriverDefaults {
  using Scenario = FusedScenario;

  static const char* tool_name() { return "perf_fmcw_range_doppler_processor"; }
  static const char* tool_description() {
    return "FMCWRangeDopplerProcessorOp (fused) "
           "performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 2)
      throw std::invalid_argument("FMCWRangeDoppler profile rank mismatch");
    return {id, shape, shape[0] * shape[1], id};
  }
  static FusedScenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 2 || shape[0] < 4 || shape[1] < 2)
      throw std::invalid_argument("FMCWRangeDoppler invalid profile shape");
    perf::require_keys(
        c.parameters,
        {"clutter_filter", "range_window", "doppler_window", "apply_dechirp", "fft_norm",
         "center_zero_doppler", "nfft_range", "nfft_doppler", "pri", "channels"},
        "parameters");
    for (const char* key : {"clutter_filter", "range_window", "doppler_window", "fft_norm"}) {
      if (!c.parameters[key])
        throw std::invalid_argument("FMCWRangeDoppler missing parameter");
    }
    const SizeSpec size{c.profile, shape[0], shape[1], c.profile};
    auto scenario =
        make_scenario(clutter_of(c.parameters["clutter_filter"].as<std::string>()), size);
    scenario.id = c.id;
    scenario.size_id = c.profile;
    scenario.range_window = window_type_of(c.parameters["range_window"].as<std::string>());
    scenario.doppler_window = window_type_of(c.parameters["doppler_window"].as<std::string>());
    scenario.fft_norm = norm_of(c.parameters["fft_norm"].as<std::string>());
    if (c.parameters["apply_dechirp"])
      scenario.apply_dechirp = c.parameters["apply_dechirp"].as<bool>();
    if (c.parameters["center_zero_doppler"])
      scenario.center_zero = c.parameters["center_zero_doppler"].as<bool>();
    if (c.parameters["channels"])
      scenario.channels = c.parameters["channels"].as<int64_t>();
    if (c.parameters["nfft_range"])
      scenario.nfft_range = c.parameters["nfft_range"].as<int64_t>();
    if (c.parameters["nfft_doppler"])
      scenario.nfft_doppler = c.parameters["nfft_doppler"].as<int64_t>();
    if (c.parameters["pri"])
      scenario.pri = c.parameters["pri"].as<double>();
    if (scenario.channels <= 0 || scenario.pri <= 20.0e-6 ||
        (scenario.nfft_range != 0 && scenario.nfft_range < shape[1]) ||
        (scenario.nfft_doppler != 0 && scenario.nfft_doppler < shape[0] - 2) ||
        scenario.nfft_range > 2 * shape[1] || scenario.nfft_doppler > 2 * shape[0])
      throw std::invalid_argument("FMCWRangeDoppler incompatible parameters");
    if (c.override_value) {
      scenario.live_port = (*c.override_value)["port"].as<std::string>();
      scenario.override_value = (*c.override_value)["value"].as<std::string>();
      const auto& port = scenario.live_port;
      const auto& value = scenario.override_value;
      if (port == "range_window" || port == "doppler_window") {
        (void)window_type_of(value);
      } else if (port == "clutter_filter") {
        (void)clutter_of(value);
      } else if (port == "fft_norm") {
        (void)norm_of(value);
      } else if (port == "apply_dechirp" || port == "center_zero_doppler") {
        if (value != "true" && value != "false")
          throw std::invalid_argument("invalid fused bool override");
      } else if (port == "waveform") {
        if (std::stod(value) <= 20.0e-6)
          throw std::invalid_argument("invalid fused waveform PRI");
      } else if (port == "nfft_range" || port == "nfft_doppler") {
        if (value != "double")
          throw std::invalid_argument("fused nfft override must be double");
      } else {
        throw std::invalid_argument("unknown fused live port");
      }
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<FusedTraits>(argc, argv);
}
