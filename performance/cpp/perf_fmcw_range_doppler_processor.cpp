// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::FMCWRangeDopplerProcessorOp<float>;
struct Scenario : perf::RecipeScenario {};

hr::ClutterFilterMethod clutter(std::string_view value) {
  if (value == "none")
    return hr::ClutterFilterMethod::kNone;
  if (value == "mean_subtraction")
    return hr::ClutterFilterMethod::kMeanSubtraction;
  if (value == "single_delay")
    return hr::ClutterFilterMethod::kSingleDelay;
  if (value == "double_delay")
    return hr::ClutterFilterMethod::kDoubleDelay;
  throw std::invalid_argument("unknown fused clutter filter");
}

std::string clutter_tag(hr::ClutterFilterMethod value) {
  switch (value) {
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

hr::FMCWRangeDopplerProcessorParams parameters(const Scenario& scen) {
  hr::FMCWRangeDopplerProcessorParams params;
  params.waveform = perf::recipe_fmcw_waveform(scen.params["waveform"]);
  params.clutter_filter = clutter(scen.params["clutter_filter"].as<std::string>());
  params.range_window = perf::recipe_window(
      perf::recipe_window_type(scen.params["range_window"]["type"].as<std::string>()));
  params.doppler_window = perf::recipe_window(
      perf::recipe_window_type(scen.params["doppler_window"]["type"].as<std::string>()));
  params.apply_dechirp = scen.params["apply_dechirp"].as<bool>();
  params.fft_norm = perf::recipe_fft_norm(scen.params["fft_norm"].as<std::string>());
  params.center_zero_doppler = scen.params["center_zero_doppler"].as<bool>();
  params.nfft_range = scen.params["nfft_range"].as<int64_t>();
  params.nfft_doppler = scen.params["nfft_doppler"].as<int64_t>();
  return params;
}

hr::MaxTensorExtents bounds(const Scenario& scen,
                            const hr::FMCWRangeDopplerProcessorParams& params) {
  hr::MaxTensorExtents extents;
  extents.max_chirps = std::max<int64_t>(scen.axis("chirps"), params.nfft_doppler);
  extents.max_range_bins = std::max<int64_t>(scen.axis("samples"), params.nfft_range);
  if (scen.live_port == "nfft_doppler")
    extents.max_chirps = std::max<int64_t>(
        *extents.max_chirps,
        perf::scaled_recipe_axis(scen.live_value, scen.axis("chirps"), "nfft_doppler"));
  if (scen.live_port == "nfft_range")
    extents.max_range_bins = std::max<int64_t>(
        *extents.max_range_bins,
        perf::scaled_recipe_axis(scen.live_value, scen.axis("samples"), "nfft_range"));
  if (scen.axis("channels") > 1)
    extents.max_rx = scen.axis("channels");
  return extents;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  const auto params = parameters(scen);
  const auto extents = bounds(scen, params);
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec == "clutter_enum") {
          const auto value = clutter(scen.live_value.as<std::string>());
          return perf::run_recipe_live_case_with_connector<Op, 1>(
              scen, plan, extents, params, value, value,
              [](auto& graph, const auto& source, const auto& op, const auto& queue) {
                graph.add_flow(source->override_out, op->clutter_filter_in, queue);
              });
        }
        if (codec == "fft_norm_enum") {
          const auto value = perf::recipe_fft_norm(scen.live_value.as<std::string>());
          return perf::run_recipe_live_case_with_connector<Op, 1>(
              scen, plan, extents, params, value, value,
              [](auto& graph, const auto& source, const auto& op, const auto& queue) {
                graph.add_flow(source->override_out, op->fft_norm_in, queue);
              });
        }
        throw std::invalid_argument("unknown fused range–Doppler live codec: " + codec);
      });
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "FMCWRangeDopplerProcessorOp";
  row.variant = clutter_tag(clutter(scen.params["clutter_filter"].as<std::string>())) + "_" +
                std::string(hr::to_string(perf::recipe_window_type(
                    scen.params["doppler_window"]["type"].as<std::string>())));
  perf::set_input_and_trailing_map_shape(row,
                                         perf::recipe_shape(scen.recipe.inputs.at(0), scen.values));
  row.out_batch = scen.axis("channels");
  row.k = scen.axis("samples");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double batch = static_cast<double>(scen.axis("channels"));
  const double chirps = static_cast<double>(scen.axis("chirps"));
  const double samples = static_cast<double>(scen.axis("samples"));
  const double cells = batch * chirps * samples;
  const double bytes =
      2.0 * cells * sizeof(std::complex<float>) + (chirps + samples) * sizeof(float);
  perf::set_recipe_throughput(row, cells, batch, bytes);
  row.notes = "end-to-end Graph latency; cells/s is complex RD-map cells/s";
  if (!scen.live_port.empty())
    row.notes += "; one-frame " + scen.live_port + " change";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_fmcw_range_doppler_processor"; }
  static const char* tool_description() {
    return "FMCWRangeDopplerProcessorOp (fused) performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 2, 2);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(suite, "radar.FMCWRangeDopplerProcessorOp.float",
                                         "fused_fft_bounds",
                                         {{"adc_cube",
                                           {"channels", "chirps", "samples"},
                                           "complex64",
                                           "device",
                                           "iq_cube",
                                           {"channels"}}},
                                         {"rd_maps", {"range_bins", "velocity_bins"}, {}, 4U},
                                         {{"range_window", "range_window"},
                                          {"doppler_window", "doppler_window"},
                                          {"apply_dechirp", "apply_dechirp"},
                                          {"clutter_filter", "clutter_filter"},
                                          {"waveform", "waveform"},
                                          {"fft_norm", "fft_norm"},
                                          {"center_zero_doppler", "center_zero_doppler"},
                                          {"nfft_range", "nfft_range"},
                                          {"nfft_doppler", "nfft_doppler"}},
                                         {{"range_window", "window_spec"},
                                          {"doppler_window", "window_spec"},
                                          {"apply_dechirp", "bool"},
                                          {"clutter_filter", "clutter_enum"},
                                          {"waveform", "waveform_descriptor"},
                                          {"fft_norm", "fft_norm_enum"},
                                          {"center_zero_doppler", "bool"},
                                          {"nfft_range", "scaled_axis"},
                                          {"nfft_doppler", "scaled_axis"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 2);
    if (scen.axis("chirps") < 4 || scen.axis("samples") < 2 || scen.axis("channels") < 1)
      throw std::invalid_argument("FMCWRangeDoppler invalid profile shape");
    perf::require_recipe_params(scen.params, {"waveform", "clutter_filter", "range_window",
                                              "doppler_window", "apply_dechirp", "fft_norm",
                                              "center_zero_doppler", "nfft_range", "nfft_doppler"});
    const auto params = parameters(scen);
    if ((params.nfft_range != 0 && params.nfft_range < scen.axis("samples")) ||
        (params.nfft_doppler != 0 && params.nfft_doppler < scen.axis("chirps") - 2) ||
        params.nfft_range > 2 * scen.axis("samples") ||
        params.nfft_doppler > 2 * scen.axis("chirps"))
      throw std::invalid_argument("FMCWRangeDoppler incompatible FFT lengths");
    (void)bounds(scen, params);
    if (scen.live_port == "waveform")
      (void)perf::recipe_fmcw_waveform(scen.params["waveform"],
                                       scen.live_value["pri_seconds"].as<double>());
    if (scen.live_port == "clutter_filter")
      (void)clutter(scen.live_value.as<std::string>());
    if (scen.live_port == "fft_norm")
      (void)perf::recipe_fft_norm(scen.live_value.as<std::string>());
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
