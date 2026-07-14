// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for CFARDetectorOp — the first consumer of the
// reusable performance/ harness. Runs a matrix of scenarios (size ladder,
// batch tiers, CFAR variants, footprints, live overrides) in one session and
// emits a single unified report. See docs and utils/ for the harness design.
//
// This is a measurement tool, not a correctness test: it does not assert on
// timings. v1 goal is one stable, repeatable, platform-annotated report.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_source.hpp"
#include "utils/perf_stats.hpp"
#include "utils/perf_stderr.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // v1 characterizes single precision.

enum class OverrideKind { kNone, kPfa, kTrainingCells, kAlphaLut };

struct Footprint {
  std::string name;
  std::vector<int64_t> guard;     // (doppler, range)
  std::vector<int64_t> training;  // (doppler, range)
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
  std::vector<int64_t> input_shape;  // full shape (leading..., D, R)
  Footprint footprint;
  std::vector<int64_t> nci_axes;
  int64_t os_rank{0};
  double pfa{1e-4};
  // Filled per run by run_scenario() for inline live-override measurement.
  OverrideKind override_kind{OverrideKind::kNone};
  int64_t override_at{0};
  double override_double{0.0};
  std::vector<int64_t> override_vec;
  bool blocking_rebuild{false};
  std::vector<double> override_alpha_lut;
  std::string calibration_pair;
  bool use_alpha_lut{false};
  std::vector<double> alpha_lut;
};

// --- size ladder (docs/planning WORKFLOWS moderate application scale) -----
// Ids are kept short (<= 3 chars) so they never blow out the report columns.
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

// Lookup for compile-time-known ladder ids used in the fixed matrices. A miss
// is a programming error (typo in a hardcoded id), so it aborts rather than
// silently dereferencing a null pointer.
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
      hr::CfarVariant::kCA, hr::CfarVariant::kGO, hr::CfarVariant::kSO,
      hr::CfarVariant::kOS};
  return variants;
}

// Interior training-cell count for a footprint (for OS rank selection).
int64_t interior_training(const Footprint& fp) {
  const int64_t hd = fp.guard[0] + fp.training[0];
  const int64_t hr = fp.guard[1] + fp.training[1];
  const int64_t window = (2 * hd + 1) * (2 * hr + 1);
  const int64_t guard_box = (2 * fp.guard[0] + 1) * (2 * fp.guard[1] + 1);
  return window - guard_box;
}

std::string variant_tag(hr::CfarVariant v) {
  return std::string(hr::to_string(v));
}

// out_batch = product of leading dims not integrated; k = product of the
// integrated leading dims (nci_axes index the full shape).
void batch_and_k(const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& nci_axes, int64_t* out_batch,
                 int64_t* k) {
  *out_batch = 1;
  *k = 1;
  const int64_t lead = static_cast<int64_t>(shape.size()) - 2;
  for (int64_t axis = 0; axis < lead; ++axis) {
    const bool integrated =
        std::find(nci_axes.begin(), nci_axes.end(), axis) != nci_axes.end();
    if (integrated) {
      *k *= shape[static_cast<size_t>(axis)];
    } else {
      *out_batch *= shape[static_cast<size_t>(axis)];
    }
  }
}

std::vector<double> precompute_alpha_lut(const CfarScenario& scen) {
  const size_t n = scen.input_shape.size();
  const int num_doppler = static_cast<int>(scen.input_shape[n - 2]);
  const int num_range = static_cast<int>(scen.input_shape[n - 1]);
  const int guard_d = static_cast<int>(scen.footprint.guard[0]);
  const int guard_r = static_cast<int>(scen.footprint.guard[1]);
  const int train_d = static_cast<int>(scen.footprint.training[0]);
  const int train_r = static_cast<int>(scen.footprint.training[1]);
  int64_t out_batch = 1;
  int64_t k = 1;
  batch_and_k(scen.input_shape, scen.nci_axes, &out_batch, &k);
  (void)out_batch;
  switch (scen.variant) {
    case hr::CfarVariant::kCA:
      return hr::detail::build_ca_alpha_lut(scen.pfa, k, num_doppler, num_range,
                                          guard_d, guard_r, train_d, train_r);
    case hr::CfarVariant::kGO:
      return hr::detail::build_go_alpha_lut(scen.pfa, k, num_doppler, num_range,
                                          guard_d, guard_r, train_d, train_r);
    case hr::CfarVariant::kSO:
      return hr::detail::build_so_alpha_lut(scen.pfa, k, num_doppler, num_range,
                                          guard_d, guard_r, train_d, train_r);
    case hr::CfarVariant::kOS:
      // build_os_alpha_lut is dense for K=1 and sparse (reachable N only)
      // for K>1; operator validate allows those sparse zeros.
      return hr::detail::build_os_alpha_lut(
          scen.pfa, k, scen.os_rank, num_doppler, num_range, guard_d, guard_r,
          train_d, train_r);
  }
  return {};
}

// ---------------------------------------------------------------------------
// Mini-application: source -> CFAR -> terminal + null sinks (+ optional
// override source). Fresh instance per scenario run.
// ---------------------------------------------------------------------------
template <typename T>
class CfarPerfApp : public holoscan::Application {
 public:
  const CfarScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::CFARDetectorOp<T>>(
        "cfar", Arg("cfar_variant") = scen->variant,
        Arg("guard_cells") = scen->footprint.guard,
        Arg("training_cells") = scen->footprint.training,
        Arg("pfa") = scen->pfa, Arg("threshold_scale") = 1.0,
        Arg("nci_axes") = scen->nci_axes, Arg("os_rank") = scen->os_rank,
        Arg("max_detections") = int64_t{0},
        Arg("blocking_alpha_lut_rebuild") = scen->blocking_rebuild,
        Arg("alpha_lut") = scen->alpha_lut);

    auto src = make_operator<perf::PerfDataSourceOp>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"rd_maps", "rd_maps"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("idx_sink");
    term->configure(collector);
    add_flow(op, term, {{"detection_idx", "input"}});

    // Every output port needs a receiver or the operator never ticks. The
    // null sink drains each dense output as a raw entity (no Tensor
    // deserialization), which also sidesteps the SDK's inability to
    // round-trip the bool detection_mask through a TensorMap receive.
    for (const char* port :
         {"detection_mask", "detection_score", "noise_estimate"}) {
      auto ns =
          make_operator<perf::PerfNullSinkOp>(std::string(port) + "_sink");
      add_flow(op, ns, {{port, "input"}});
    }

    wire_override(op);
  }

 private:
  void wire_override(const std::shared_ptr<holoscan::Operator>& op) {
    switch (scen->override_kind) {
      case OverrideKind::kNone:
        return;
      case OverrideKind::kPfa: {
        auto s = make_operator<perf::PerfOverrideSourceOp<double>>(
            "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_double);
        add_flow(s, op, {{"output", "pfa"}});
        break;
      }
      case OverrideKind::kTrainingCells: {
        auto s =
            make_operator<perf::PerfOverrideSourceOp<std::vector<int64_t>>>(
                "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_vec);
        add_flow(s, op, {{"output", "training_cells"}});
        break;
      }
      case OverrideKind::kAlphaLut: {
        auto s = make_operator<perf::PerfOverrideSourceOp<std::vector<double>>>(
            "ovr", make_condition<holoscan::CountCondition>(total_frames));
        s->configure(scen->override_at, scen->override_alpha_lut);
        add_flow(s, op, {{"output", "alpha_lut"}});
        break;
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Throughput + effective-bandwidth estimates for a compute row.
// ---------------------------------------------------------------------------
void fill_throughput(perf::ResultRow* row, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const int64_t D = row->map_shape[0];
  const int64_t R = row->map_shape[1];
  const double cells = static_cast<double>(row->out_batch) *
                       static_cast<double>(D) * static_cast<double>(R);
  row->cells_per_second = cells / sec;
  row->slices_per_second = static_cast<double>(row->out_batch) / sec;

  const double tsz = sizeof(PerfPrecision);
  const double in_bytes = cells * static_cast<double>(row->k) * tsz;
  const double out_bytes = cells * (2.0 * tsz + 1.0);  // score+noise+mask byte
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const CfarScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.variant = variant_tag(scen.variant);
  row.size_id = scen.size_id;
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
  return row;
}

// ---------------------------------------------------------------------------
// Run one scenario (all repeats), append its rows to the report.
// Each run measures setup, steady-state compute, and a blocking pfa live
// override inline in a single graph session (no duplicate scenario loop).
// ---------------------------------------------------------------------------
void run_scenario(const CfarScenario& scen, const perf::RunSettings& rs,
                  perf::PerfReport* report) {
  constexpr int kOverrideSpikeFrames = 3;
  const int64_t override_at =
      rs.warmup_frames + std::max(2, rs.measured_frames / 2);
  const int total =
      std::max(rs.warmup_frames + rs.measured_frames,
               static_cast<int>(override_at) + kOverrideSpikeFrames);

  std::vector<double> setup_samples;
  std::vector<double> cold_samples;
  // Steady-state frames = the pre-override measured window only (warmup..
  // override_at), so with an inline override the effective steady sample count
  // is ~measured_frames/2, not the full --frames. This same window feeds both
  // the compute row and the live_override baseline.
  std::vector<double> steady_samples;
  double override_ms_best = std::numeric_limits<double>::quiet_NaN();

  for (int rep = 0; rep < rs.repeats; ++rep) {
    CfarScenario run_scen = scen;
    if (run_scen.use_alpha_lut && run_scen.alpha_lut.empty()) {
      run_scen.alpha_lut = precompute_alpha_lut(run_scen);
    }
    run_scen.override_at = override_at;
    if (run_scen.use_alpha_lut) {
      CfarScenario override_scen = run_scen;
      override_scen.pfa = 1e-6;
      run_scen.override_kind = OverrideKind::kAlphaLut;
      run_scen.override_alpha_lut = precompute_alpha_lut(override_scen);
      run_scen.blocking_rebuild = false;
    } else {
      run_scen.override_kind = OverrideKind::kPfa;
      run_scen.override_double = 1e-6;
      run_scen.blocking_rebuild = true;
    }

    perf::TimingCollector collector;
    // Preallocate the input before starting the setup clock so H2D upload is
    // not counted as operator setup.
    auto frame = perf::single_map(
        perf::make_power_map_device<PerfPrecision>(scen.input_shape),
        "rd_maps");

    collector.t_construct = perf::Clock::now();
    auto app = holoscan::make_application<CfarPerfApp<PerfPrecision>>();
    app->scen = &run_scen;
    app->collector = &collector;
    app->frame = std::move(frame);
    app->total_frames = total;
    perf::suppress_known_hsdk_noise_during([&] { app->run(); });

    setup_samples.push_back(collector.setup_ms());
    const auto lat = collector.frame_latencies_ms();
    if (!lat.empty()) {
      cold_samples.push_back(lat.front());
    }

    for (size_t i = static_cast<size_t>(rs.warmup_frames); i < lat.size();
         ++i) {
      const int64_t frame = static_cast<int64_t>(i);
      if (frame < override_at) {
        steady_samples.push_back(lat[i]);
      } else if (frame < override_at + kOverrideSpikeFrames) {
        const double v = lat[i];
        if (std::isnan(override_ms_best) || v > override_ms_best) {
          override_ms_best = v;
        }
      }
    }
  }

  // setup row (one per scenario, reported as median across repeats)
  {
    perf::ResultRow row = base_row(scen);
    row.bucket = "setup";
    row.setup_ms = perf::summarize(setup_samples).median_ms;
    report->results.push_back(row);
  }

  {
    perf::ResultRow row = base_row(scen);
    row.bucket = "compute";
    row.cold_ms = perf::summarize(cold_samples).median_ms;
    row.latency = perf::summarize(steady_samples);
    row.override_ms = override_ms_best;
    if (!std::isnan(override_ms_best) && !std::isnan(row.latency.median_ms)) {
      row.override_delta_ms = override_ms_best - row.latency.median_ms;
    }
    fill_throughput(&row, row.latency.median_ms);
    if (scen.use_alpha_lut) {
      row.notes =
          "cold_ms is first-frame latency including initial alpha_lut H2D; "
          "median/p90 are pre-override steady state; override_ms is peak "
          "latency over the 3-frame alpha_lut live-update window "
          "(pfa=1e-6 LUT)";
    } else {
      row.notes =
          "cold_ms is first-frame latency including initial LUT "
          "calibration; median/p90 are pre-override steady state; "
          "override_ms is peak latency over the 3-frame blocking pfa "
          "reconfig window inline in the same run";
    }
    report->results.push_back(row);
  }
}

// ---------------------------------------------------------------------------
// Scenario matrix construction per run tier.
// ---------------------------------------------------------------------------
std::vector<int64_t> plane_shape(const SizeSpec& s) {
  return {s.doppler, s.range};
}

CfarScenario make_compute_scenario(hr::CfarVariant v, const SizeSpec& s,
                                   const Footprint& fp, int64_t out_batch) {
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

CfarScenario make_nci_scenario(hr::CfarVariant v, const SizeSpec& s,
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
  id << "cfar_" << variant_tag(v) << "_" << s.id << "_small_k" << channels
     << "_compute";
  sc.id = id.str();
  return sc;
}

std::vector<CfarScenario> build_default_matrix() {
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
  return out;
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

void tag_and_append_alpha_lut_pairs(std::vector<CfarScenario>* scenarios) {
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
        fast.id.compare(fast.id.size() - suffix.size(), suffix.size(),
                        suffix) == 0) {
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
      out.slow_ovr_ms = row.override_ms;
    } else if (row.calibration_mode == "alpha_lut") {
      out.fast_cold_ms = row.cold_ms;
      out.fast_ovr_ms = row.override_ms;
    }
  }
  for (auto& [_, row] : rows) {
    if (perf::detail::is_num(row.slow_cold_ms) &&
        perf::detail::is_num(row.fast_cold_ms) && row.fast_cold_ms > 0.0) {
      row.cold_speedup = row.slow_cold_ms / row.fast_cold_ms;
    }
    if (perf::detail::is_num(row.slow_ovr_ms) &&
        perf::detail::is_num(row.fast_ovr_ms) && row.fast_ovr_ms > 0.0) {
      row.ovr_speedup = row.slow_ovr_ms / row.fast_ovr_ms;
    }
    if (perf::detail::is_num(row.slow_cold_ms) ||
        perf::detail::is_num(row.fast_cold_ms)) {
      report->alpha_calibration.push_back(row);
    }
  }
  std::sort(report->alpha_calibration.begin(), report->alpha_calibration.end(),
            [](const perf::AlphaCalibrationRow& a,
               const perf::AlphaCalibrationRow& b) {
              return a.pair_id < b.pair_id;
            });
}

std::vector<CfarScenario> build_size_sweep(
    const std::vector<std::string>& sizes) {
  std::vector<CfarScenario> out;
  const Footprint fp = footprint_small();
  for (const auto& sid : sizes) {
    const SizeSpec* s = find_size(sid);
    if (s == nullptr) {
      std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
      continue;
    }
    out.push_back(make_compute_scenario(hr::CfarVariant::kCA, *s, fp, 1));
  }
  return out;
}

std::vector<CfarScenario> build_full_matrix() {
  std::vector<CfarScenario> out;
  const auto& variants = all_cfar_variants();
  const std::vector<Footprint> fps = {footprint_small(), footprint_mid(),
                                      footprint_large()};
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
      out.push_back(make_compute_scenario(v, s, footprint_small(), 4));
      out.push_back(make_compute_scenario(v, s, footprint_small(), 16));
    }
  }
  for (const char* sid : {"md", "lg"}) {
    const SizeSpec& s = must_find_size(sid);
    for (auto v : variants) {
      out.push_back(make_nci_scenario(v, s, 8));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Options {
  std::string output;
  std::string text;
  bool quiet{false};
  bool full{false};
  std::vector<std::string> sizes;  // empty = tier default
  bool sizes_all{false};
  double stop_after_s{0.0};  // 0 = disabled
  perf::RunSettings rs;
};

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.push_back(tok);
    }
  }
  return out;
}

void print_help() {
  std::cout <<
      "perf_cfar_detector — CFARDetectorOp performance characterization\n\n"
      "Usage: perf_cfar_detector [options]\n\n"
      "  --output <path>   Write machine-readable JSON report\n"
      "  --text <path>     Also write the text summary to a file\n"
      "  --quiet           Suppress the stdout text summary\n"
      "  --warmup N        Warmup frames excluded from stats (default 5)\n"
      "  --frames N        Measured frames per scenario (default 20)\n"
      "  --repeats N       Repeat each scenario N times, pooled (default 1)\n"
      "  --sizes all       CA size sweep across the full ladder\n"
      "  --sizes a,b,c     CA size sweep across an explicit subset\n"
      "  --full            Full matrix (all variants/footprints/batch/NCI)\n"
      "                    plus the default alpha_lut calibration pairs\n"
      "  --stop-after-s S  Stop after first scenario whose wall time exceeds "
      "S seconds\n"
      "  --help            Show this help\n";
}

bool parse_args(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "error: " << flag << " requires an argument\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      print_help();
      std::exit(0);
    } else if (a == "--output") {
      opt->output = next("--output");
    } else if (a == "--text") {
      opt->text = next("--text");
    } else if (a == "--quiet") {
      opt->quiet = true;
    } else if (a == "--full") {
      opt->full = true;
    } else if (a == "--warmup") {
      opt->rs.warmup_frames = std::stoi(next("--warmup"));
    } else if (a == "--frames") {
      opt->rs.measured_frames = std::stoi(next("--frames"));
    } else if (a == "--repeats") {
      opt->rs.repeats = std::stoi(next("--repeats"));
    } else if (a == "--stop-after-s") {
      opt->stop_after_s = std::stod(next("--stop-after-s"));
      if (opt->stop_after_s <= 0.0) {
        std::cerr << "error: --stop-after-s must be positive\n";
        return false;
      }
    } else if (a == "--sizes") {
      const std::string v = next("--sizes");
      if (v == "all") {
        opt->sizes_all = true;
      } else {
        opt->sizes = split_csv(v);
      }
    } else {
      std::cerr << "error: unknown argument '" << a << "'\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse_args(argc, argv, &opt)) {
    return 2;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "error: no CUDA device available; the CFAR operator requires "
                 "a GPU\n";
    return 1;
  }

  // Reduce Holoscan log jitter unless the user configured a level. ERROR drops
  // WARN spam (e.g. the per-run stack-size advisory) that would otherwise
  // clutter progress output; the known bool TensorMap ERROR noise is
  // additionally content-filtered during app->run() (perf_stderr.hpp).
  if (std::getenv("HOLOSCAN_LOG_LEVEL") == nullptr) {
    setenv("HOLOSCAN_LOG_LEVEL", "ERROR", 0);
  }

  std::vector<CfarScenario> scenarios;
  if (opt.full) {
    scenarios = build_full_matrix();
    // Keep the curated pfa↔alpha_lut pairs so --full still reports the
    // calibration-mode comparison (slow MC vs precomputed LUT), not only pfa.
    tag_and_append_alpha_lut_pairs(&scenarios);
  } else if (opt.sizes_all) {
    std::vector<std::string> all;
    for (const auto& s : size_ladder()) {
      all.push_back(s.id);
    }
    scenarios = build_size_sweep(all);
  } else if (!opt.sizes.empty()) {
    scenarios = build_size_sweep(opt.sizes);
  } else {
    scenarios = build_default_matrix();
    tag_and_append_alpha_lut_pairs(&scenarios);
  }

  perf::PerfReport report;
  report.provenance = perf::capture_provenance();
  report.platform = perf::capture_platform();
  report.run_settings = opt.rs;
  report.run_settings.stop_after_s = opt.stop_after_s;
  report.run_settings.scenarios_planned = static_cast<int>(scenarios.size());

  // Record ladder definitions for the sizes this run actually exercises, in
  // ladder order, so the report is self-describing.
  {
    std::vector<std::string> used;
    for (const auto& sc : scenarios) {
      if (std::find(used.begin(), used.end(), sc.size_id) == used.end()) {
        used.push_back(sc.size_id);
      }
    }
    for (const auto& s : size_ladder()) {
      if (std::find(used.begin(), used.end(), s.id) != used.end()) {
        perf::SizeDef def;
        def.id = s.id;
        def.shape = {s.doppler, s.range};
        def.cells = s.doppler * s.range;
        def.role = s.role;
        report.sizes.push_back(def);
      }
    }
  }

  const size_t n = scenarios.size();
  bool stopped_early = false;
  for (size_t i = 0; i < n; ++i) {
    std::cerr << "[" << (i + 1) << "/" << n << "] " << scenarios[i].id
              << std::flush;
    const auto t0 = perf::Clock::now();
    // Isolate scenario failures: a thrown graph error (e.g. an operator
    // rejecting an invalid parameter) would otherwise propagate out of main
    // and abort the process, taking the remaining scenarios and the report
    // with it. Log and continue so one bad scenario does not lose the sweep.
    try {
      run_scenario(scenarios[i], opt.rs, &report);
    } catch (const std::exception& e) {
      std::cerr << "  FAILED: " << e.what() << "\n";
      continue;
    }
    const double elapsed_s = perf::to_ms(perf::Clock::now() - t0) / 1000.0;
    std::cerr << "  (" << elapsed_s << " s)\n";
    report.run_settings.scenarios_completed = static_cast<int>(i + 1);
    if (opt.stop_after_s > 0.0 && elapsed_s > opt.stop_after_s) {
      std::cerr << "stopping early: scenario exceeded --stop-after-s "
                << opt.stop_after_s << " s\n";
      stopped_early = true;
      break;
    }
  }
  report.run_settings.stopped_early = stopped_early;
  build_alpha_calibration_summary(&report);

  if (!opt.output.empty()) {
    std::ofstream f(opt.output);
    if (!f) {
      std::cerr << "error: cannot open --output file '" << opt.output << "'\n";
      return 1;
    }
    perf::write_json(f, report);
    std::cerr << "wrote JSON report: " << opt.output << "\n";
  }
  if (!opt.text.empty()) {
    std::ofstream f(opt.text);
    if (!f) {
      std::cerr << "error: cannot open --text file '" << opt.text << "'\n";
      return 1;
    }
    perf::write_text(f, report);
    std::cerr << "wrote text report: " << opt.text << "\n";
  }
  if (!opt.quiet) {
    perf::write_text(std::cout, report);
  }
  return 0;
}
