// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Typed range-FFT semantics for the shared YAML recipe driver.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_registered_recipe.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_yaml_pilot.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace perf_fmcw_range_processor_adapter {

using PerfPrecision = float;  // v1 characterizes single precision (complex64).

struct RangeScenario : perf::RecipeScenario {
  hr::WindowType window{hr::WindowType::kNone};
  bool apply_dechirp{false};
  hr::FftNorm fft_norm{hr::FftNorm::kBackward};
  int64_t nfft{0};
};

// FFT-friendly powers of two so cuFFT takes its fast path; the ladder spans a
// smoke size up to a moderate production cube.

std::string window_tag(hr::WindowType w) {
  return w == hr::WindowType::kHann ? "hann" : "rect";
}

perf::TimingCollector run_graph(const RangeScenario& scen, const perf::RunPlan& plan) {
  hr::MaxTensorExtents extents;
  extents.max_chirps = scen.axis("chirps");
  const auto changed_nfft =
      scen.live_port == "nfft"
          ? perf::scaled_recipe_axis(scen.live_value, scen.axis("samples"), "nfft")
          : scen.axis("samples");
  extents.max_range_bins = std::max({scen.axis("samples"), scen.nfft, changed_nfft});
  if (scen.axis("channels") > 1)
    extents.max_rx = scen.axis("channels");
  hr::FMCWRangeProcessorParams params;
  params.waveform = perf::recipe_fmcw_waveform(scen.params["waveform"]);
  params.range_window = perf::recipe_window(scen.window);
  params.apply_dechirp = scen.apply_dechirp;
  params.fft_norm = scen.fft_norm;
  params.nfft = scen.nfft;
  using Op = hr::FMCWRangeProcessorOp<PerfPrecision>;
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec != "enum" || scen.live_port != "fft_norm")
          throw std::invalid_argument("unsupported range FFT live codec: " + codec);
        const auto changed = perf::recipe_fft_norm(scen.live_value.as<std::string>());
        return perf::run_recipe_live_case_with_connector<Op, 1>(
            scen, plan, extents, params, changed, changed,
            [](auto& graph, const auto& source, const auto& op, const auto& queue) {
              graph.add_flow(source->override_out, op->fft_norm_in, queue);
            });
      });
}

void fill_throughput(perf::ResultRow* row, const RangeScenario& scen) {
  const double batch = static_cast<double>(scen.axis("channels"));
  const double chirps = static_cast<double>(scen.axis("chirps"));
  const double n = static_cast<double>(scen.axis("samples"));

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double in_bytes = batch * chirps * n * csz;
  const double out_bytes = batch * chirps * n * csz + n * sizeof(PerfPrecision);
  perf::set_recipe_throughput(*row, batch * chirps * n, batch * chirps, in_bytes + out_bytes);
}

perf::ResultRow base_row(const RangeScenario& scen) {
  perf::ResultRow row;
  row.op_name = "FMCWRangeProcessorOp";
  row.variant = window_tag(scen.window);
  perf::set_input_and_trailing_map_shape(row,
                                         perf::recipe_shape(scen.recipe.inputs.at(0), scen.values));
  row.out_batch = scen.axis("channels");
  row.k = scen.axis("samples");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const RangeScenario& scen) {
  fill_throughput(&row, scen);
  row.notes =
      "cold_ms is first-frame latency including initial "
      "window/FFT plan build; median/p90 are post-warmup "
      "steady state; one-frame live changes are timed; "
      "cells/s is complex range bins/s";
}

struct RangeTraits : perf::PerfDriverDefaults {
  using Scenario = RangeScenario;

  static const char* tool_name() { return "perf_fmcw_range_processor"; }
  static const char* tool_description() {
    return "FMCWRangeProcessorOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 2, 2);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<hr::FMCWRangeProcessorOp<PerfPrecision>>(
        suite, "radar.FMCWRangeProcessorOp.float", "range_fft_bounds",
        {{"adc_cube",
          {"channels", "chirps", "samples"},
          "complex64",
          "device",
          "iq_cube",
          {"channels"}}},
        {"range_profiles", {"range_bins"}, {}, 4U},
        {{"range_window", "range_window"},
         {"apply_dechirp", "apply_dechirp"},
         {"waveform", "waveform"},
         {"fft_norm", "fft_norm"},
         {"nfft", "nfft"}},
        {{"range_window", "window_spec"},
         {"apply_dechirp", "bool"},
         {"waveform", "waveform_descriptor"},
         {"fft_norm", "enum"},
         {"nfft", "scaled_axis"}});
  }
  static RangeScenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                                  const perf::ProfileSuite& suite) {
    auto scenario = perf::pilot_scenario<RangeScenario>(suite, c, shape, 2);
    if (shape[1] < 2)
      throw std::invalid_argument("FMCWRange invalid profile shape");
    const auto& params = scenario.params;
    perf::require_recipe_params(params,
                                {"range_window", "apply_dechirp", "fft_norm", "nfft", "waveform"});
    const auto window = perf::recipe_window_type(params["range_window"]["type"].as<std::string>());
    scenario.window = window;
    scenario.fft_norm = perf::recipe_fft_norm(params["fft_norm"].as<std::string>());
    scenario.apply_dechirp = params["apply_dechirp"].as<bool>();
    scenario.nfft = params["nfft"].as<int64_t>();
    (void)perf::recipe_fmcw_waveform(params["waveform"]);
    if (scenario.nfft != 0 && scenario.nfft < shape[1])
      throw std::invalid_argument("FMCWRange nfft cannot be shorter than input samples");
    if (c.override_value) {
      const auto value = scenario.live_value;
      if (scenario.live_port == "range_window") {
        (void)perf::recipe_window_type(value["type"].as<std::string>());
      } else if (scenario.live_port == "apply_dechirp") {
        (void)value.as<bool>();
      } else if (scenario.live_port == "waveform") {
        (void)perf::recipe_fmcw_waveform(params["waveform"], value["pri_seconds"].as<double>());
      } else if (scenario.live_port == "fft_norm") {
        (void)perf::recipe_fft_norm(value.as<std::string>());
      } else if (scenario.live_port == "nfft") {
        (void)perf::scaled_recipe_axis(value, scenario.axis("samples"), "nfft");
      } else {
        throw std::invalid_argument("unknown FMCWRange live port");
      }
      if ((*c.override_value)["cache"] && scenario.live_port != "nfft")
        throw std::invalid_argument("cached transition only applies to nfft");
    }
    return scenario;
  }
};

}  // namespace perf_fmcw_range_processor_adapter

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<perf_fmcw_range_processor_adapter::RangeTraits>(argc,
                                                                                          argv);
}
