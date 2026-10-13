// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::FMCWDopplerProcessorOp<float>;
struct Scenario : perf::RecipeScenario {};

hr::FMCWDopplerProcessorParams parameters(const Scenario& scen) {
  hr::FMCWDopplerProcessorParams params;
  params.waveform = perf::recipe_fmcw_waveform(scen.params["waveform"]);
  params.doppler_window = perf::recipe_window(
      perf::recipe_window_type(scen.params["doppler_window"]["type"].as<std::string>()));
  params.center_zero_doppler = scen.params["center_zero_doppler"].as<bool>();
  params.fft_norm = perf::recipe_fft_norm(scen.params["fft_norm"].as<std::string>());
  params.nfft = scen.params["nfft"].as<int64_t>();
  return params;
}

hr::MaxTensorExtents bounds(const Scenario& scen, const hr::FMCWDopplerProcessorParams& params) {
  hr::MaxTensorExtents extents;
  extents.max_chirps = std::max<int64_t>(scen.axis("chirps"), params.nfft);
  if (scen.live_port == "nfft")
    extents.max_chirps =
        std::max<int64_t>(*extents.max_chirps,
                          perf::scaled_recipe_axis(scen.live_value, scen.axis("chirps"), "nfft"));
  extents.max_range_bins = scen.axis("range_bins");
  if (scen.axis("channels") > 1)
    extents.max_rx = scen.axis("channels");
  return extents;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  const auto params = parameters(scen);
  const auto extents = bounds(scen, params);
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec != "fft_norm_enum")
          throw std::invalid_argument("unknown Doppler FFT live codec: " + codec);
        const auto value = perf::recipe_fft_norm(scen.live_value.as<std::string>());
        return perf::run_recipe_live_case_with_connector<Op, 1>(
            scen, plan, extents, params, value, value,
            [](auto& graph, const auto& source, const auto& op, const auto& queue) {
              graph.add_flow(source->override_out, op->fft_norm_in, queue);
            });
      });
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "FMCWDopplerProcessorOp";
  row.variant = scen.params["doppler_window"]["type"].as<std::string>() == "hann" ? "hann" : "rect";
  perf::set_input_and_trailing_map_shape(row,
                                         perf::recipe_shape(scen.recipe.inputs.at(0), scen.values));
  row.out_batch = scen.axis("channels");
  row.k = scen.axis("chirps");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double batch = static_cast<double>(scen.axis("channels"));
  const double chirps = static_cast<double>(scen.axis("chirps"));
  const double bins = static_cast<double>(scen.axis("range_bins"));
  const double cells = batch * chirps * bins;
  const double bytes = 2.0 * cells * sizeof(std::complex<float>) + chirps * sizeof(float);
  perf::set_recipe_throughput(row, cells, batch * bins, bytes);
  row.notes = "end-to-end Graph latency; cells/s is complex RD-map cells/s";
  if (!scen.live_port.empty())
    row.notes += "; one-frame " + scen.live_port + " change";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_fmcw_doppler_processor"; }
  static const char* tool_description() {
    return "FMCWDopplerProcessorOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 2, 2);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(suite, "radar.FMCWDopplerProcessorOp.float",
                                         "doppler_fft_bounds",
                                         {{"range_profiles",
                                           {"channels", "chirps", "range_bins"},
                                           "complex64",
                                           "device",
                                           "iq_cube",
                                           {"channels"}}},
                                         {"rd_maps", {"velocity_bins"}, {}, 4U},
                                         {{"doppler_window", "doppler_window"},
                                          {"center_zero_doppler", "center_zero_doppler"},
                                          {"waveform", "waveform"},
                                          {"fft_norm", "fft_norm"},
                                          {"nfft", "nfft"}},
                                         {{"doppler_window", "window_spec"},
                                          {"center_zero_doppler", "bool"},
                                          {"waveform", "waveform_descriptor"},
                                          {"fft_norm", "fft_norm_enum"},
                                          {"nfft", "scaled_axis"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 2);
    if (scen.axis("chirps") < 2 || scen.axis("channels") < 1)
      throw std::invalid_argument("FMCWDoppler invalid profile shape");
    perf::require_recipe_params(
        scen.params, {"waveform", "doppler_window", "center_zero_doppler", "fft_norm", "nfft"});
    const auto params = parameters(scen);
    if ((params.nfft != 0 && params.nfft < scen.axis("chirps")) ||
        params.nfft > 2 * scen.axis("chirps"))
      throw std::invalid_argument("FMCWDoppler incompatible nfft");
    (void)bounds(scen, params);
    if (scen.live_port == "waveform")
      (void)perf::recipe_fmcw_waveform(scen.params["waveform"],
                                       scen.live_value["pri_seconds"].as<double>());
    if (scen.live_port == "fft_norm")
      (void)perf::recipe_fft_norm(scen.live_value.as<std::string>());
    if (scen.cached_override && scen.live_port != "nfft")
      throw std::invalid_argument("cached transition only applies to nfft");
    return scen;
  }
  static void validate_run_settings(const Scenario& scen, const perf::RunSettings& rs) {
    if (scen.cached_override && rs.warmup_frames < 1)
      throw std::invalid_argument("cached transition requires at least one warmup frame");
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
