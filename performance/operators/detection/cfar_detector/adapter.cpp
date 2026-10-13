// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for CFARDetectorOp — runs a matrix of scenarios
// (size ladder, batch tiers, CFAR variants, footprints, NCI, peak modes) in
// independent Graph sessions and emits one report. Measurement tool only: no
// timing assertions.
//
// Graph-based harness with one-frame Exact live overrides and Params reversion.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <set>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_payloads.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_registered_recipe.hpp"
#include "utils/perf_yaml_pilot.hpp"
#include "utils/perf_runner.hpp"
#include "utils/perf_stats.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace perf_cfar_detector_adapter {

using PerfPrecision = float;

struct Footprint {
  std::vector<int64_t> guard;
  std::vector<int64_t> training;
};

struct CfarScenario : perf::RecipeScenario {
  hr::CfarVariant variant;
  std::vector<int64_t> input_shape;
  Footprint footprint;
  std::vector<int64_t> nci_axes;
  int64_t os_rank{0};
  double pfa{1e-4};
  std::string calibration_pair;
  bool use_alpha_lut{false};
  std::vector<double> alpha_lut;
  hr::CfarPeakMode peak_mode{hr::CfarPeakMode::kNone};
  std::vector<int64_t> peak_radius{1};
  double threshold_scale{1.0};
};

int64_t interior_training(const Footprint& fp) {
  const int64_t hd = fp.guard[0] + fp.training[0];
  const int64_t hr_val = fp.guard[1] + fp.training[1];
  const int64_t window = (2 * hd + 1) * (2 * hr_val + 1);
  const int64_t guard_box = (2 * fp.guard[0] + 1) * (2 * fp.guard[1] + 1);
  return window - guard_box;
}

std::string variant_tag(hr::CfarVariant v) {
  return std::string(hr::to_string(v));
}

void batch_and_k(const std::vector<int64_t>& shape, const std::vector<int64_t>& nci_axes,
                 int64_t* out_batch, int64_t* k) {
  *out_batch = 1;
  *k = 1;
  const int64_t lead = static_cast<int64_t>(shape.size()) - 2;
  for (int64_t axis = 0; axis < lead; ++axis) {
    const bool integrated = std::find(nci_axes.begin(), nci_axes.end(), axis) != nci_axes.end();
    if (integrated) {
      *k *= shape[static_cast<size_t>(axis)];
    } else {
      *out_batch *= shape[static_cast<size_t>(axis)];
    }
  }
}

std::vector<double> precompute_alpha_lut(const CfarScenario& scen) {
  const size_t n = scen.input_shape.size();
  const int nd = static_cast<int>(scen.input_shape[n - 2]);
  const int nr = static_cast<int>(scen.input_shape[n - 1]);
  const int gd = static_cast<int>(scen.footprint.guard[0]);
  const int gr = static_cast<int>(scen.footprint.guard[1]);
  const int td = static_cast<int>(scen.footprint.training[0]);
  const int tr = static_cast<int>(scen.footprint.training[1]);
  int64_t out_batch = 1;
  int64_t k = 1;
  batch_and_k(scen.input_shape, scen.nci_axes, &out_batch, &k);
  (void)out_batch;
  switch (scen.variant) {
    case hr::CfarVariant::kCA:
      return hr::detail::build_ca_alpha_lut(scen.pfa, k, nd, nr, gd, gr, td, tr);
    case hr::CfarVariant::kGO:
      return hr::detail::build_go_alpha_lut(scen.pfa, k, nd, nr, gd, gr, td, tr);
    case hr::CfarVariant::kSO:
      return hr::detail::build_so_alpha_lut(scen.pfa, k, nd, nr, gd, gr, td, tr);
    case hr::CfarVariant::kOS:
      return hr::detail::build_os_alpha_lut(scen.pfa, k, scen.os_rank, nd, nr, gd, gr, td, tr);
  }
  return {};
}

perf::TimingCollector run_graph(const CfarScenario& scenario, const perf::RunPlan& plan) {
  CfarScenario prepared = scenario;
  if (prepared.use_alpha_lut && prepared.alpha_lut.empty())
    prepared.alpha_lut = precompute_alpha_lut(prepared);
  const auto& scen = prepared;
  hr::MaxTensorExtents extents;
  extents.max_doppler_bins = scen.input_shape[scen.input_shape.size() - 2];
  extents.max_range_bins = scen.input_shape[scen.input_shape.size() - 1];
  int64_t out_batch = 1, k = 1;
  batch_and_k(scen.input_shape, scen.nci_axes, &out_batch, &k);
  if (scen.input_shape.size() == 3U) {
    extents.max_rx = scen.input_shape[0];
  } else if (scen.input_shape.size() == 4U) {
    extents.max_tx = scen.input_shape[0];
    extents.max_rx = scen.input_shape[1];
  }
  extents.max_detections =
      out_batch * extents.max_doppler_bins.value() * extents.max_range_bins.value();
  hr::CFARParams params;
  params.cfar_variant = scen.variant;
  params.guard_cells = scen.footprint.guard;
  params.training_cells = scen.footprint.training;
  params.pfa = scen.pfa;
  params.threshold_scale = scen.threshold_scale;
  params.nci_axes = scen.nci_axes;
  params.os_rank = scen.os_rank;
  params.max_detections = 0;
  params.alpha_lut = scen.alpha_lut;
  params.peak_mode = scen.peak_mode;
  params.peak_radius = scen.peak_radius;
  using Op = hr::CFARDetectorOp<PerfPrecision>;
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        const auto emit = [&](const auto& changed) {
          return perf::run_recipe_live_case<Op, 1>(scen, plan, extents, params, changed, changed);
        };
        if (codec == "rank_decrement")
          return emit(
              std::max<int64_t>(0, scen.os_rank - scen.live_value["decrement"].as<int64_t>()));
        if (codec == "scaled_alpha_lut") {
          auto changed = precompute_alpha_lut(scen);
          const auto factor = scen.live_value["factor"].as<double>();
          for (auto& alpha : changed)
            if (alpha > 0.0)
              alpha *= factor;
          return emit(changed);
        }
        throw std::invalid_argument("unsupported CFAR live codec: " + codec);
      });
}

void fill_throughput(perf::ResultRow* row) {
  const int64_t D = row->map_shape[0];
  const int64_t R = row->map_shape[1];
  const double cells =
      static_cast<double>(row->out_batch) * static_cast<double>(D) * static_cast<double>(R);

  const double tsz = sizeof(PerfPrecision);
  const double in_bytes = cells * static_cast<double>(row->k) * tsz;
  const double out_bytes = cells * (2.0 * tsz + 1.0);
  perf::set_recipe_throughput(*row, cells, static_cast<double>(row->out_batch),
                              in_bytes + out_bytes);
}

perf::ResultRow base_row(const CfarScenario& scen) {
  perf::ResultRow row;
  row.variant = variant_tag(scen.variant);
  perf::set_input_and_trailing_map_shape(row, scen.input_shape);
  row.guard = scen.footprint.guard;
  row.training = scen.footprint.training;
  int64_t out_batch = 1, k = 1;
  batch_and_k(scen.input_shape, scen.nci_axes, &out_batch, &k);
  row.out_batch = out_batch;
  row.k = k;
  row.calibration_pair = scen.calibration_pair;
  row.calibration_mode = scen.use_alpha_lut ? "alpha_lut" : "pfa";
  row.live_port = scen.live_port;
  return row;
}

// --- typed scenario construction from YAML cases ---

struct CalibrationPairSpec {
  std::string slow_id;
  std::string pair_id;
};

const std::vector<CalibrationPairSpec>& calibration_pair_specs() {
  static const std::vector<CalibrationPairSpec> specs = {
      {"cfar_GO_xlw_small_k1_compute", "GO_xlw_k1"}, {"cfar_SO_xlw_small_k1_compute", "SO_xlw_k1"},
      {"cfar_GO_md_small_k8_compute", "GO_md_k8"},   {"cfar_SO_md_small_k8_compute", "SO_md_k8"},
      {"cfar_OS_md_small_k8_compute", "OS_md_k8"},   {"cfar_OS_lg_small_k8_compute", "OS_lg_k8"},
  };
  return specs;
}

void build_alpha_calibration_summary(perf::PerfReport* report) {
  std::unordered_map<std::string, perf::AlphaCalibrationRow> rows;
  for (const auto& spec : calibration_pair_specs()) {
    perf::AlphaCalibrationRow row;
    row.pair_id = spec.pair_id;
    rows.emplace(spec.pair_id, std::move(row));
  }
  for (const auto& row : report->results) {
    if (row.bucket != "compute" || row.calibration_pair.empty()) {
      continue;
    }
    auto it = rows.find(row.calibration_pair);
    if (it == rows.end()) {
      continue;
    }
    perf::AlphaCalibrationRow& out = it->second;
    out.variant = row.variant;
    out.size_id = row.size_id;
    out.k = row.k;
    if (row.calibration_mode == "pfa") {
      out.slow_cold_ms = row.cold_ms;
    } else if (row.calibration_mode == "alpha_lut") {
      out.fast_cold_ms = row.cold_ms;
    }
  }
  for (auto& [_, row] : rows) {
    if (perf::detail::is_num(row.slow_cold_ms) && perf::detail::is_num(row.fast_cold_ms) &&
        row.fast_cold_ms > 0.0) {
      row.cold_speedup = row.slow_cold_ms / row.fast_cold_ms;
    }
    if (perf::detail::is_num(row.slow_cold_ms) || perf::detail::is_num(row.fast_cold_ms)) {
      report->alpha_calibration.push_back(row);
    }
  }
  std::sort(report->alpha_calibration.begin(), report->alpha_calibration.end(),
            [](const perf::AlphaCalibrationRow& a, const perf::AlphaCalibrationRow& b) {
              return a.pair_id < b.pair_id;
            });
}

void decorate_row(perf::ResultRow& row, const CfarScenario& scen) {
  fill_throughput(&row);
  if (scen.use_alpha_lut) {
    row.notes = "cold_ms includes initial alpha_lut H2D; override_ms is the changed frame for " +
                scen.live_port;
  } else {
    row.notes = "cold_ms includes initial LUT calibration; override_ms is the changed frame for " +
                scen.live_port;
  }
  if (scen.peak_mode != hr::CfarPeakMode::kNone) {
    row.notes += "; peak_mode=" + std::string(hr::to_string(scen.peak_mode));
    if (scen.peak_mode == hr::CfarPeakMode::kRangeDoppler) {
      row.notes += " (radius=" + std::to_string(scen.peak_radius.front()) + ")";
    }
  }
  if (row.live_port.empty())
    row.notes = "default parameters; no live override";
}

struct CfarTraits : perf::PerfDriverDefaults {
  using Scenario = CfarScenario;

  static const char* tool_name() { return "perf_cfar_detector"; }
  static const char* tool_description() {
    return "CFARDetectorOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 2, 2);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    const auto recipe = perf::validate_pilot_suite<hr::CFARDetectorOp<PerfPrecision>>(
        suite, "radar.CFARDetectorOp.float", "cfar_bounds",
        {{"rd_maps",
          {"channels", "doppler", "range"},
          "float32",
          "device",
          "power_map",
          {"channels"}}},
        {"detection_idx",
         {"detection_mask", "detection_score", "noise_estimate", "cfar_power"},
         {},
         4U},
        {{"os_rank", "os_rank"},
         {"guard_cells", "guard_cells"},
         {"training_cells", "training_cells"},
         {"pfa", "pfa"},
         {"alpha_lut", "alpha_lut"},
         {"threshold_scale", "threshold_scale"},
         {"nci_axes", "nci_axes"},
         {"max_detections", "max_detections"}},
        {{"os_rank", "rank_decrement"},
         {"guard_cells", "cfar_cell_window"},
         {"training_cells", "cfar_cell_window"},
         {"pfa", "double"},
         {"alpha_lut", "scaled_alpha_lut"},
         {"threshold_scale", "double"},
         {"nci_axes", "host_int64_vector"},
         {"max_detections", "int64"}});
    std::set<std::string> backends;
    for (const auto& c : suite.select("limited")) {
      const auto params = perf::merge_recipe_map(recipe.params, c.parameters);
      backends.insert(params["cfar_variant"].as<std::string>());
    }
    if (backends != std::set<std::string>{"CA", "GO", "SO", "OS"})
      throw std::invalid_argument("CFAR Limited must cover CA, GO, SO, and OS backends");
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scenario = perf::pilot_scenario<CfarScenario>(suite, c, shape, 2);
    if (shape[0] < 16 || shape[1] < 16)
      throw std::invalid_argument("CFAR invalid shape");
    const auto& params = scenario.params;
    perf::require_recipe_params(
        params,
        {"cfar_variant", "guard_cells", "training_cells", "pfa", "peak_mode", "threshold_scale",
         "os_rank", "nci_axes", "alpha_lut", "max_detections"},
        {"cfar_variant", "guard_cells", "training_cells", "pfa", "peak_mode", "threshold_scale"});
    const auto variant = params["cfar_variant"].as<std::string>();
    const auto channels = scenario.values.at("channels");
    const auto peak = params["peak_mode"].as<std::string>();
    const auto guard_pair = perf::recipe_pair<int64_t>(params["guard_cells"], "guard_cells");
    const auto training_pair =
        perf::recipe_pair<int64_t>(params["training_cells"], "training_cells");
    const std::vector<int64_t> guard(guard_pair.begin(), guard_pair.end());
    const std::vector<int64_t> training(training_pair.begin(), training_pair.end());
    if ((variant != "CA" && variant != "GO" && variant != "SO" && variant != "OS") ||
        (peak != "none" && peak != "range_doppler"))
      throw std::invalid_argument("CFAR invalid parameter choice");
    const auto v = variant == "GO"   ? hr::CfarVariant::kGO
                   : variant == "SO" ? hr::CfarVariant::kSO
                   : variant == "OS" ? hr::CfarVariant::kOS
                                     : hr::CfarVariant::kCA;
    const Footprint fp{guard, training};
    for (std::size_t axis = 0; axis < 2; ++axis)
      if (guard[axis] < 0 || training[axis] <= 0 || guard[axis] > shape[axis] ||
          training[axis] > shape[axis] || 2 * (guard[axis] + training[axis]) + 1 > shape[axis])
        throw std::invalid_argument("CFAR footprint exceeds profile plane");
    scenario.variant = v;
    scenario.footprint = fp;
    scenario.input_shape =
        channels > 1 ? std::vector<int64_t>{channels, shape[0], shape[1]} : shape;
    scenario.pfa = params["pfa"].as<double>();
    scenario.threshold_scale = params["threshold_scale"].as<double>();
    scenario.peak_mode =
        peak == "range_doppler" ? hr::CfarPeakMode::kRangeDoppler : hr::CfarPeakMode::kNone;
    if (v == hr::CfarVariant::kOS) {
      const auto rank = params["os_rank"];
      if (!rank || rank["factory"].as<std::string>() != "three_quarter_training_rank")
        throw std::invalid_argument("OS CFAR needs a valid rank factory");
      scenario.os_rank = (interior_training(fp) * 3) / 4;
    }
    if (!std::isfinite(scenario.pfa) || scenario.pfa <= 0.0 || scenario.pfa >= 1.0 ||
        !std::isfinite(scenario.threshold_scale) || scenario.threshold_scale <= 0.0)
      throw std::invalid_argument("CFAR invalid pfa or threshold scale");
    if (scenario.live_port == "os_rank") {
      const auto decrement = scenario.live_value["decrement"].as<int64_t>();
      if (v != hr::CfarVariant::kOS || decrement < 0 || decrement >= scenario.os_rank)
        throw std::invalid_argument("CFAR invalid live OS rank");
    } else if (scenario.live_port == "guard_cells" || scenario.live_port == "training_cells") {
      const auto pair = perf::recipe_pair<int64_t>(scenario.live_value, scenario.live_port);
      const auto& other = scenario.live_port == "guard_cells" ? training_pair : guard_pair;
      for (std::size_t axis = 0; axis < 2; ++axis) {
        const auto g = scenario.live_port == "guard_cells" ? pair[axis] : other[axis];
        const auto t = scenario.live_port == "training_cells" ? pair[axis] : other[axis];
        if (g < 0 || t <= 0 || g > shape[axis] || t > shape[axis] || 2 * (g + t) + 1 > shape[axis])
          throw std::invalid_argument("CFAR live footprint exceeds profile plane");
      }
    } else if (scenario.live_port == "pfa") {
      const auto value = scenario.live_value.as<double>();
      if (!std::isfinite(value) || value <= 0.0 || value >= 1.0)
        throw std::invalid_argument("CFAR live pfa must be between zero and one");
    } else if (scenario.live_port == "alpha_lut") {
      const auto factor = scenario.live_value["factor"].as<double>();
      if (scenario.live_value["factory"].as<std::string>() != "scaled_alpha_lut" ||
          !std::isfinite(factor) || factor <= 0.0)
        throw std::invalid_argument("CFAR invalid live alpha LUT factory or factor");
    } else if (scenario.live_port == "threshold_scale") {
      const auto value = scenario.live_value.as<double>();
      if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument("CFAR live threshold scale must be positive");
    } else if (scenario.live_port == "nci_axes") {
      const auto axes = scenario.live_value.as<std::vector<int64_t>>();
      if (axes.size() > (channels > 1 ? 1U : 0U) ||
          std::any_of(axes.begin(), axes.end(), [](int64_t axis) {
            return axis != 0;
          }))
        throw std::invalid_argument("CFAR live NCI axes must name a leading input axis");
    } else if (scenario.live_port == "max_detections") {
      if (scenario.live_value.as<int64_t>() <= 0)
        throw std::invalid_argument("CFAR live detection cap must be positive");
    } else if (!scenario.live_port.empty()) {
      throw std::invalid_argument("unknown CFAR live port");
    }
    return scenario;
  }

  static void post_run(perf::PerfReport* report, bool /*quiet*/) {
    build_alpha_calibration_summary(report);
  }
};

}  // namespace perf_cfar_detector_adapter

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<perf_cfar_detector_adapter::CfarTraits>(argc, argv);
}
