// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for CFARDetectorOp — runs a matrix of scenarios
// (size ladder, batch tiers, CFAR variants, footprints, NCI, peak_only NMS) in
// one session and emits a single unified report. Measurement tool only: no
// timing assertions.
//
// [EA HSDK5 ONLY] Graph-based harness. Measures a mid-session threshold_scale
// override. The post_run calibration summary is kept for alpha_lut vs pfa
// cold-start comparison; other live ports are not timed here.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
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
  bool peak_only{false};
  hr::CfarPeakNeighborhood peak_neighborhood{
      hr::CfarPeakNeighborhood::kSquare8};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 64, 256, "small real map; fast smoke; border-heavy"},
      {"sm", 128, 512, "compact sensor / few chirps"},
      {"md", 256, 512, "common dev size"},
      {"mdw", 128, 1024, "range-heavy, same cell count as md"},
      {"lgs", 512, 512, "square mid-scale"},
      {"lg", 512, 1024, "moderate production-scale"},
      {"xl", 1024, 1024, "large moderate"},
      {"xlw", 1024, 2048, "range-heavy moderate (v1 upper bound)"}};
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

Footprint footprint_small() { return {"small", {1, 1}, {2, 2}}; }
Footprint footprint_mid() { return {"mid", {2, 2}, {4, 4}}; }
Footprint footprint_large() { return {"large", {2, 2}, {8, 8}}; }

const std::vector<hr::CfarVariant>& all_cfar_variants() {
  static const std::vector<hr::CfarVariant> variants = {
      hr::CfarVariant::kCA, hr::CfarVariant::kGO,
      hr::CfarVariant::kSO, hr::CfarVariant::kOS};
  return variants;
}

int64_t interior_training(const Footprint& fp) {
  const int64_t hd = fp.guard[0] + fp.training[0];
  const int64_t hr_val = fp.guard[1] + fp.training[1];
  const int64_t window = (2 * hd + 1) * (2 * hr_val + 1);
  const int64_t guard_box =
      (2 * fp.guard[0] + 1) * (2 * fp.guard[1] + 1);
  return window - guard_box;
}

std::string variant_tag(hr::CfarVariant v) {
  return std::string(hr::to_string(v));
}

void batch_and_k(const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& nci_axes,
                 int64_t* out_batch, int64_t* k) {
  *out_batch = 1;
  *k = 1;
  const int64_t lead = static_cast<int64_t>(shape.size()) - 2;
  for (int64_t axis = 0; axis < lead; ++axis) {
    const bool integrated =
        std::find(nci_axes.begin(), nci_axes.end(), axis) !=
        nci_axes.end();
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
      return hr::detail::build_ca_alpha_lut(
          scen.pfa, k, nd, nr, gd, gr, td, tr);
    case hr::CfarVariant::kGO:
      return hr::detail::build_go_alpha_lut(
          scen.pfa, k, nd, nr, gd, gr, td, tr);
    case hr::CfarVariant::kSO:
      return hr::detail::build_so_alpha_lut(
          scen.pfa, k, nd, nr, gd, gr, td, tr);
    case hr::CfarVariant::kOS:
      return hr::detail::build_os_alpha_lut(
          scen.pfa, k, scen.os_rank, nd, nr, gd, gr, td, tr);
  }
  return {};
}

perf::TimingCollector run_cfar_graph(const CfarScenario& scen,
                                     const perf::RunPlan& plan) {
  auto input =
      perf::make_power_map_device<PerfPrecision>(scen.input_shape);
  using BaseOp = hr::CFARDetectorOp<PerfPrecision>;
  // Production Exact{} + per-tick co-emit. Restacking LatestWithin through
  // PerfLiveContractOp would make the temporal contract invalid.
  constexpr double kInitialThresholdScale = 1.0;
  constexpr double kOverrideThresholdScale = 1.1;
  return perf::run_perf_graph_with_override(
      std::move(input), plan.total_frames, plan.override_at,
      kInitialThresholdScale, kOverrideThresholdScale, "cfar-perf",
      [&](holoscan::Graph& graph, const auto& src,
          holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_doppler_bins = scen.input_shape[
            scen.input_shape.size() - 2];
        extents.max_range_bins = scen.input_shape[
            scen.input_shape.size() - 1];
        int64_t out_batch = 1, k = 1;
        batch_and_k(scen.input_shape, scen.nci_axes,
                    &out_batch, &k);
        // Input leading rank (before NCI) must be admitted even when NCI
        // collapses it to a single plane (out_batch == 1).
        if (scen.input_shape.size() > 2) {
          extents.max_rx = scen.input_shape[0];
        }
        extents.max_detections = out_batch *
            extents.max_doppler_bins.value() *
            extents.max_range_bins.value();
        auto op = graph.op<BaseOp>("cfar", extents);
        hr::CFARParams params;
        params.cfar_variant = scen.variant;
        params.guard_cells = scen.footprint.guard;
        params.training_cells = scen.footprint.training;
        params.pfa = scen.pfa;
        params.threshold_scale = kInitialThresholdScale;
        params.nci_axes = scen.nci_axes;
        params.os_rank = scen.os_rank;
        params.max_detections = 0;
        params.alpha_lut = scen.alpha_lut;
        params.peak_only = scen.peak_only;
        params.peak_neighborhood = scen.peak_neighborhood;
        op.set_params(std::move(params));

        const auto idx_sink =
            graph.op<perf::PerfTerminalSinkOp>("idx_sink");
        const auto mask_drain =
            graph.op<perf::PerfNullSinkOp>("mask_drain");
        const auto score_drain =
            graph.op<perf::PerfNullSinkOp>("score_drain");
        const auto noise_drain =
            graph.op<perf::PerfNullSinkOp>("noise_drain");
        const auto power_drain =
            graph.op<perf::PerfNullSinkOp>("power_drain");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};

        graph.add_flow(src->out, op->rd_maps, q);
        graph.add_flow(
            src->override_out, op->threshold_scale, q);
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
      });
}

void fill_throughput(perf::ResultRow* row, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const int64_t D = row->map_shape[0];
  const int64_t R = row->map_shape[1];
  const double cells = static_cast<double>(row->out_batch) *
                       static_cast<double>(D) *
                       static_cast<double>(R);
  row->cells_per_second = cells / sec;
  row->slices_per_second =
      static_cast<double>(row->out_batch) / sec;

  const double tsz = sizeof(PerfPrecision);
  const double in_bytes =
      cells * static_cast<double>(row->k) * tsz;
  const double out_bytes = cells * (2.0 * tsz + 1.0);
  row->estimated_effective_gbps =
      (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const CfarScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.variant = variant_tag(scen.variant);
  row.size_id = scen.size_id;
  const size_t n = scen.input_shape.size();
  row.map_shape = {scen.input_shape[n - 2],
                   scen.input_shape[n - 1]};
  row.input_shape = scen.input_shape;
  row.guard = scen.footprint.guard;
  row.training = scen.footprint.training;
  int64_t out_batch = 1, k = 1;
  batch_and_k(scen.input_shape, scen.nci_axes, &out_batch, &k);
  row.out_batch = out_batch;
  row.k = k;
  row.calibration_pair = scen.calibration_pair;
  row.calibration_mode = scen.use_alpha_lut ? "alpha_lut" : "pfa";
  row.notes = "live_override=threshold_scale; other lives not timed";
  return row;
}

// --- scenario matrix construction per run tier ---

std::vector<int64_t> plane_shape(const SizeSpec& s) {
  return {s.doppler, s.range};
}

CfarScenario make_compute_scenario(hr::CfarVariant v,
                                   const SizeSpec& s,
                                   const Footprint& fp,
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

CfarScenario make_nci_scenario(hr::CfarVariant v,
                               const SizeSpec& s,
                               int64_t channels) {
  CfarScenario sc;
  sc.variant = v;
  sc.size_id = s.id;
  sc.footprint = footprint_small();
  sc.input_shape = {channels, s.doppler, s.range};
  sc.nci_axes = {0};
  if (v == hr::CfarVariant::kOS) {
    sc.os_rank = (interior_training(sc.footprint) * 3) / 4;
  }
  std::ostringstream id;
  id << "cfar_" << variant_tag(v) << "_" << s.id << "_small_k"
     << channels << "_compute";
  sc.id = id.str();
  return sc;
}

CfarScenario make_peak_scenario(hr::CfarVariant v, const SizeSpec& s,
                                const Footprint& fp,
                                hr::CfarPeakNeighborhood nb) {
  CfarScenario sc = make_compute_scenario(v, s, fp, 1);
  sc.peak_only = true;
  sc.peak_neighborhood = nb;
  const std::string suffix = "_compute";
  const std::string insert =
      std::string("_peak_") + std::string(hr::to_string(nb));
  if (sc.id.size() > suffix.size() &&
      sc.id.compare(sc.id.size() - suffix.size(), suffix.size(),
                    suffix) == 0) {
    sc.id.insert(sc.id.size() - suffix.size(), insert);
  } else {
    sc.id += insert;
  }
  return sc;
}

const std::vector<hr::CfarPeakNeighborhood>&
all_peak_neighborhoods() {
  static const std::vector<hr::CfarPeakNeighborhood> nbs = {
      hr::CfarPeakNeighborhood::kSquare8,
      hr::CfarPeakNeighborhood::kCross4};
  return nbs;
}

void append_peak_only_scenarios(std::vector<CfarScenario>* out,
                                const std::vector<std::string>& sizes,
                                bool all_variants_at_md) {
  const Footprint fp = footprint_small();
  for (const auto& sid : sizes) {
    const SizeSpec& s = must_find_size(sid);
    for (auto nb : all_peak_neighborhoods()) {
      out->push_back(
          make_peak_scenario(hr::CfarVariant::kCA, s, fp, nb));
    }
  }
  if (!all_variants_at_md) {
    return;
  }
  const SizeSpec& md = must_find_size("md");
  for (auto v : all_cfar_variants()) {
    if (v == hr::CfarVariant::kCA) {
      continue;
    }
    for (auto nb : all_peak_neighborhoods()) {
      out->push_back(make_peak_scenario(v, md, fp, nb));
    }
  }
}

struct CalibrationPairSpec {
  std::string slow_id;
  std::string pair_id;
};

const std::vector<CalibrationPairSpec>& calibration_pair_specs() {
  static const std::vector<CalibrationPairSpec> specs = {
      {"cfar_GO_xlw_small_k1_compute", "GO_xlw_k1"},
      {"cfar_SO_xlw_small_k1_compute", "SO_xlw_k1"},
      {"cfar_GO_md_small_k8_compute", "GO_md_k8"},
      {"cfar_SO_md_small_k8_compute", "SO_md_k8"},
      {"cfar_OS_md_small_k8_compute", "OS_md_k8"},
      {"cfar_OS_lg_small_k8_compute", "OS_lg_k8"},
  };
  return specs;
}

void tag_and_append_alpha_lut_pairs(
    std::vector<CfarScenario>* scenarios) {
  std::unordered_map<std::string, CfarScenario> by_id;
  for (const auto& sc : *scenarios) {
    by_id.emplace(sc.id, sc);
  }
  for (const auto& spec : calibration_pair_specs()) {
    auto it = by_id.find(spec.slow_id);
    if (it == by_id.end()) {
      continue;
    }
    for (auto& sc : *scenarios) {
      if (sc.id == spec.slow_id) {
        sc.calibration_pair = spec.pair_id;
      }
    }
    CfarScenario fast = it->second;
    fast.calibration_pair = spec.pair_id;
    fast.use_alpha_lut = true;
    fast.alpha_lut.clear();
    const std::string suffix = "_compute";
    if (fast.id.size() > suffix.size() &&
        fast.id.compare(fast.id.size() - suffix.size(),
                        suffix.size(), suffix) == 0) {
      fast.id.replace(fast.id.size() - suffix.size(), suffix.size(),
                      "_alphalut_compute");
    } else {
      fast.id += "_alphalut";
    }
    scenarios->push_back(std::move(fast));
  }
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
    if (perf::detail::is_num(row.slow_cold_ms) &&
        perf::detail::is_num(row.fast_cold_ms) &&
        row.fast_cold_ms > 0.0) {
      row.cold_speedup = row.slow_cold_ms / row.fast_cold_ms;
    }
    if (perf::detail::is_num(row.slow_cold_ms) ||
        perf::detail::is_num(row.fast_cold_ms)) {
      report->alpha_calibration.push_back(row);
    }
  }
  std::sort(report->alpha_calibration.begin(),
            report->alpha_calibration.end(),
            [](const perf::AlphaCalibrationRow& a,
               const perf::AlphaCalibrationRow& b) {
              return a.pair_id < b.pair_id;
            });
}

struct CfarTraits : perf::PerfDriverDefaults {
  using Scenario = CfarScenario;

  static const char* tool_name() { return "perf_cfar_detector"; }
  static const char* tool_description() {
    return "CFARDetectorOp performance characterization "
           "(HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "CA size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "CA size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (all variants/footprints/batch/NCI) plus the "
           "default alpha_lut calibration pairs and peak_only NMS "
           "rows; inline threshold_scale override timing";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.doppler, s.range};
      def.cells = s.doppler * s.range;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<CfarScenario> build_default_matrix() {
    std::vector<CfarScenario> out;
    const std::vector<std::string> sizes = {"xs", "md", "lg", "xlw"};
    const Footprint fp = footprint_small();
    for (const auto& sid : sizes) {
      const SizeSpec& s = must_find_size(sid);
      for (auto v : all_cfar_variants()) {
        out.push_back(make_compute_scenario(v, s, fp, 1));
      }
    }
    const SizeSpec& md = must_find_size("md");
    for (auto v : all_cfar_variants()) {
      out.push_back(make_compute_scenario(v, md, fp, 4));
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      for (auto v : all_cfar_variants()) {
        out.push_back(make_nci_scenario(v, s, 8));
      }
    }
    tag_and_append_alpha_lut_pairs(&out);
    append_peak_only_scenarios(&out, {"xs", "md", "lg", "xlw"},
                               /*all_variants_at_md=*/false);
    return out;
  }

  static std::vector<CfarScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<CfarScenario> out;
    const Footprint fp = footprint_small();
    std::vector<std::string> known;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid
                  << "', skipping\n";
        continue;
      }
      out.push_back(
          make_compute_scenario(hr::CfarVariant::kCA, *s, fp, 1));
      known.push_back(sid);
    }
    append_peak_only_scenarios(&out, known,
                               /*all_variants_at_md=*/false);
    return out;
  }

  static std::vector<CfarScenario> build_full_matrix() {
    std::vector<CfarScenario> out;
    const auto& variants = all_cfar_variants();
    const std::vector<Footprint> fps = {
        footprint_small(), footprint_mid(), footprint_large()};
    for (const auto& s : size_ladder()) {
      for (auto v : variants) {
        for (const auto& fp : fps) {
          out.push_back(make_compute_scenario(v, s, fp, 1));
        }
      }
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      for (auto v : variants) {
        out.push_back(
            make_compute_scenario(v, s, footprint_small(), 4));
        out.push_back(
            make_compute_scenario(v, s, footprint_small(), 16));
      }
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      for (auto v : variants) {
        out.push_back(make_nci_scenario(v, s, 8));
      }
    }
    tag_and_append_alpha_lut_pairs(&out);
    std::vector<std::string> all_sizes;
    for (const auto& s : size_ladder()) {
      all_sizes.push_back(s.id);
    }
    append_peak_only_scenarios(&out, all_sizes,
                               /*all_variants_at_md=*/true);
    return out;
  }

  static void run_scenario(const CfarScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          CfarScenario run_scen = scen;
          if (run_scen.use_alpha_lut && run_scen.alpha_lut.empty()) {
            run_scen.alpha_lut = precompute_alpha_lut(run_scen);
          }
          return run_cfar_graph(run_scen, p);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, row.latency.median_ms);
    if (scen.use_alpha_lut) {
      row.notes =
          "cold_ms includes initial alpha_lut H2D; median/p90 are "
          "steady state; override_ms is the peak after an inline "
          "threshold_scale change; other lives are not timed";
    } else {
      row.notes =
          "cold_ms includes initial LUT calibration; median/p90 are "
          "steady state; override_ms is the peak after an inline "
          "threshold_scale change; other lives are not timed";
    }
    if (scen.peak_only) {
      row.notes +=
          "; peak_only NMS enabled (neighborhood=" +
          std::string(hr::to_string(scen.peak_neighborhood)) + ")";
    }
    report->results.push_back(row);
  }

  static bool isolate_failures() { return true; }

  static void post_run(perf::PerfReport* report, bool /*quiet*/) {
    build_alpha_calibration_summary(report);
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CfarTraits>(argc, argv);
}
