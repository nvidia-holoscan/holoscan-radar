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
#include "utils/perf_runner.hpp"
#include "utils/perf_stats.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

struct Footprint {
  std::string name;
  std::vector<int64_t> guard;
  std::vector<int64_t> training;
};

struct SizeSpec {
  std::string id;
  int64_t doppler;
  int64_t range;
  std::string role;
};

struct CfarScenario {
  std::string id;
  std::string size_id;
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
  std::string live_port;
  double threshold_scale{1.0};
};

Footprint footprint_small() {
  return {"small", {1, 1}, {2, 2}};
}
Footprint footprint_mid() {
  return {"mid", {2, 2}, {4, 4}};
}
Footprint footprint_large() {
  return {"large", {2, 2}, {8, 8}};
}

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

struct LivePorts {
 public:
  explicit LivePorts(CfarScenario scenario) : scenario_(std::move(scenario)) {
    if (scenario_.live_port == "alpha_lut") {
      changed_alpha_lut_ = precompute_alpha_lut(scenario_);
      for (auto& value : changed_alpha_lut_)
        if (value > 0.0)
          value *= 1.1;
    }
  }
  void setup(holoscan::OperatorSpec& spec) {
    perf::setup_device_input(spec, out, perf::PerfFrameBank::span_a());
    perf::setup_scalar_payload(spec, os_rank);
    perf::setup_scalar_payload(spec, guard_cells);
    perf::setup_scalar_payload(spec, training_cells);
    perf::setup_scalar_payload(spec, pfa);
    perf::setup_host_payload(spec, alpha_lut, 1U, 1U);
    perf::setup_scalar_payload(spec, threshold_scale);
    perf::setup_host_payload(spec, nci_axes, 1U, 1U, DLDataType{kDLInt, 64U, 1U},
                             4U * sizeof(int64_t));
    perf::setup_scalar_payload(spec, max_detections);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_override(
      holoscan::ExecutionContext& context) {
    const auto time = context.activation_time();
    holoscan::expected<void, holoscan::Error> emitted;
    if (scenario_.live_port == "os_rank")
      emitted = perf::emit_scalar_payload(os_rank,
                                          scenario_.os_rank > 0 ? scenario_.os_rank - 1 : 0, time);
    else if (scenario_.live_port == "guard_cells")
      emitted = perf::emit_scalar_payload(guard_cells, hr::CfarCellWindowOverride{0, 0}, time);
    else if (scenario_.live_port == "training_cells")
      emitted = perf::emit_scalar_payload(training_cells, hr::CfarCellWindowOverride{3, 3}, time);
    else if (scenario_.live_port == "pfa")
      emitted = perf::emit_scalar_payload(pfa, 1e-3, time);
    else if (scenario_.live_port == "alpha_lut")
      emitted = perf::emit_host_vector(alpha_lut, changed_alpha_lut_, time);
    else if (scenario_.live_port == "threshold_scale")
      emitted = perf::emit_scalar_payload(threshold_scale, 1.1, time);
    else if (scenario_.live_port == "nci_axes")
      emitted = perf::emit_host_vector(nci_axes, std::vector<int64_t>{0}, time);
    else if (scenario_.live_port == "max_detections")
      emitted = perf::emit_scalar_payload(max_detections, int64_t{100}, time);
    if (!emitted)
      return emitted;
    return {};
  }
  holoscan::Output<holoscan::Tensor> out{"out"};
  holoscan::Output<int64_t> os_rank{"os_rank"};
  holoscan::Output<hr::CfarCellWindowOverride> guard_cells{"guard_cells"};
  holoscan::Output<hr::CfarCellWindowOverride> training_cells{"training_cells"};
  holoscan::Output<double> pfa{"pfa"};
  holoscan::Output<holoscan::Tensor> alpha_lut{"alpha_lut"};
  holoscan::Output<double> threshold_scale{"threshold_scale"};
  holoscan::Output<holoscan::Tensor> nci_axes{"nci_axes"};
  holoscan::Output<int64_t> max_detections{"max_detections"};

 private:
  CfarScenario scenario_;
  std::vector<double> changed_alpha_lut_;
};

using LiveSource = perf::PerfLiveSourceOp<LivePorts, &LivePorts::out>;

perf::TimingCollector run_graph(const CfarScenario& scenario, const perf::RunPlan& plan) {
  CfarScenario prepared = scenario;
  if (prepared.use_alpha_lut && prepared.alpha_lut.empty()) {
    prepared.alpha_lut = precompute_alpha_lut(prepared);
  }
  const CfarScenario& scen = prepared;
  auto input = perf::make_power_map_device<PerfPrecision>(scen.input_shape);
  using BaseOp = hr::CFARDetectorOp<PerfPrecision>;
  const auto build = [&](holoscan::Graph& graph, const auto& src,
                         holoscan::CompileOptions& options) {
    hr::MaxTensorExtents extents;
    extents.max_doppler_bins = scen.input_shape[scen.input_shape.size() - 2];
    extents.max_range_bins = scen.input_shape[scen.input_shape.size() - 1];
    int64_t out_batch = 1, k = 1;
    batch_and_k(scen.input_shape, scen.nci_axes, &out_batch, &k);
    // Input leading rank (before NCI) must be admitted even when NCI
    // collapses it to a single plane (out_batch == 1).
    if (scen.input_shape.size() == 3U) {
      extents.max_rx = scen.input_shape[0];
    } else if (scen.input_shape.size() == 4U) {
      extents.max_tx = scen.input_shape[0];
      extents.max_rx = scen.input_shape[1];
    }
    extents.max_detections =
        out_batch * extents.max_doppler_bins.value() * extents.max_range_bins.value();
    auto op = graph.op<BaseOp>("cfar", extents);
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
    op.set_params(std::move(params));

    const auto idx_sink = graph.op<perf::PerfTerminalSinkOp>("idx_sink");
    const auto mask_drain = graph.op<perf::PerfNullSinkOp>("mask_drain");
    const auto score_drain = graph.op<perf::PerfNullSinkOp>("score_drain");
    const auto noise_drain = graph.op<perf::PerfNullSinkOp>("noise_drain");
    const auto power_drain = graph.op<perf::PerfNullSinkOp>("power_drain");
    const holoscan::ConnectionOptions q{.queue_depth = 4U};

    graph.add_flow(src->out, op->rd_maps, q);
    if constexpr (requires { src->os_rank; }) {
      if (scen.live_port == "os_rank")
        graph.add_flow(src->os_rank, op->os_rank, q);
      if (scen.live_port == "guard_cells")
        graph.add_flow(src->guard_cells, op->guard_cells, q);
      if (scen.live_port == "training_cells")
        graph.add_flow(src->training_cells, op->training_cells, q);
      if (scen.live_port == "pfa")
        graph.add_flow(src->pfa, op->pfa, q);
      if (scen.live_port == "alpha_lut")
        graph.add_flow(src->alpha_lut, op->alpha_lut, q);
      if (scen.live_port == "threshold_scale")
        graph.add_flow(src->threshold_scale, op->threshold_scale, q);
      if (scen.live_port == "nci_axes")
        graph.add_flow(src->nci_axes, op->nci_axes, q);
      if (scen.live_port == "max_detections")
        graph.add_flow(src->max_detections, op->max_detections, q);
    }
    graph.add_flow(op->detection_idx, idx_sink->in, q);
    graph.add_flow(op->detection_mask, mask_drain->in, q);
    graph.add_flow(op->detection_score, score_drain->in, q);
    graph.add_flow(op->noise_estimate, noise_drain->in, q);
    graph.add_flow(op->cfar_power, power_drain->in, q);
    graph.partition("perf")
        .add(src)
        .add(op)
        .add(idx_sink)
        .add(mask_drain)
        .add(score_drain)
        .add(noise_drain)
        .add(power_drain);

    perf::bind_tensor_out(options, "cfar", "detection_idx");
    perf::bind_tensor_out(options, "cfar", "detection_mask");
    perf::bind_tensor_out(options, "cfar", "detection_score");
    perf::bind_tensor_out(options, "cfar", "noise_estimate");
    perf::bind_tensor_out(options, "cfar", "cfar_power");
  };
  if (scen.live_port.empty())
    return perf::run_perf_graph(std::move(input), plan.total_frames, "cfar-perf", build);
  return perf::run_perf_graph_with_source<LiveSource>(std::move(input), plan.total_frames,
                                                      plan.override_at, scen, "cfar-perf", build);
}

void fill_throughput(perf::ResultRow* row, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const int64_t D = row->map_shape[0];
  const int64_t R = row->map_shape[1];
  const double cells =
      static_cast<double>(row->out_batch) * static_cast<double>(D) * static_cast<double>(R);
  row->cells_per_second = cells / sec;
  row->slices_per_second = static_cast<double>(row->out_batch) / sec;

  const double tsz = sizeof(PerfPrecision);
  const double in_bytes = cells * static_cast<double>(row->k) * tsz;
  const double out_bytes = cells * (2.0 * tsz + 1.0);
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const CfarScenario& scen) {
  perf::ResultRow row;
  row.variant = variant_tag(scen.variant);
  const size_t n = scen.input_shape.size();
  row.map_shape = {scen.input_shape[n - 2], scen.input_shape[n - 1]};
  row.input_shape = scen.input_shape;
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

std::vector<int64_t> plane_shape(const SizeSpec& s) {
  return {s.doppler, s.range};
}

CfarScenario make_compute_scenario(hr::CfarVariant v, const SizeSpec& s, const Footprint& fp,
                                   int64_t out_batch) {
  CfarScenario sc;
  sc.variant = v;
  sc.size_id = s.id;
  sc.footprint = fp;
  if (out_batch > 1) {
    sc.input_shape = {out_batch, s.doppler, s.range};
  } else {
    sc.input_shape = plane_shape(s);
  }
  if (v == hr::CfarVariant::kOS) {
    sc.os_rank = (interior_training(fp) * 3) / 4;
  }
  std::ostringstream id;
  id << "cfar_" << variant_tag(v) << "_" << s.id << "_" << fp.name;
  if (out_batch > 1) {
    id << "_b" << out_batch;
  }
  id << "_k1_compute";
  sc.id = id.str();
  return sc;
}

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
  fill_throughput(&row, row.latency.median_ms);
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
    if (shape.size() != 2)
      throw std::invalid_argument("CFAR profile rank mismatch");
    return {id, shape, shape[0] * shape[1], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 2 || shape[0] < 16 || shape[1] < 16)
      throw std::invalid_argument("CFAR invalid shape");
    perf::require_keys(c.parameters,
                       {"variant", "footprint", "channels", "pfa", "peak_mode", "threshold_scale"},
                       "parameters");
    const auto variant = c.parameters["variant"].as<std::string>();
    const auto footprint = c.parameters["footprint"].as<std::string>();
    const auto channels = c.parameters["channels"].as<int64_t>();
    const auto peak = c.parameters["peak_mode"].as<std::string>();
    if ((variant != "CA" && variant != "GO" && variant != "SO" && variant != "OS") ||
        (footprint != "small" && footprint != "mid" && footprint != "large") ||
        (channels != 1 && channels != 4) || (peak != "none" && peak != "range_doppler"))
      throw std::invalid_argument("CFAR invalid parameter choice");
    const auto v = variant == "GO"   ? hr::CfarVariant::kGO
                   : variant == "SO" ? hr::CfarVariant::kSO
                   : variant == "OS" ? hr::CfarVariant::kOS
                                     : hr::CfarVariant::kCA;
    const auto fp = footprint == "mid"     ? footprint_mid()
                    : footprint == "large" ? footprint_large()
                                           : footprint_small();
    const SizeSpec size{c.profile, shape[0], shape[1], c.profile};
    auto scenario = make_compute_scenario(v, size, fp, channels);
    if (channels == 4)
      scenario.input_shape = {channels, shape[0], shape[1]};
    scenario.id = c.id;
    scenario.pfa = c.parameters["pfa"].as<double>();
    scenario.threshold_scale = c.parameters["threshold_scale"].as<double>();
    scenario.peak_mode =
        peak == "range_doppler" ? hr::CfarPeakMode::kRangeDoppler : hr::CfarPeakMode::kNone;
    if (scenario.pfa <= 0.0 || scenario.pfa >= 1.0 || scenario.threshold_scale <= 0.0)
      throw std::invalid_argument("CFAR invalid pfa or threshold scale");
    if (c.override_value) {
      scenario.live_port = (*c.override_value)["port"].as<std::string>();
      const auto value = (*c.override_value)["value"].as<std::string>();
      const std::map<std::string, std::string> allowed = {
          {"os_rank", "lower"},    {"guard_cells", "zero"},  {"training_cells", "three"},
          {"pfa", "0.001"},        {"alpha_lut", "scaled"},  {"threshold_scale", "1.1"},
          {"nci_axes", "channel"}, {"max_detections", "100"}};
      if (allowed.at(scenario.live_port) != value ||
          (scenario.live_port == "os_rank" && v != hr::CfarVariant::kOS) ||
          (scenario.live_port == "nci_axes" && channels != 4) ||
          (scenario.live_port == "alpha_lut" && v != hr::CfarVariant::kCA))
        throw std::invalid_argument("CFAR incompatible live override");
    }
    return scenario;
  }

  static void post_run(perf::PerfReport* report, bool /*quiet*/) {
    build_alpha_calibration_summary(report);
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CfarTraits>(argc, argv);
}
