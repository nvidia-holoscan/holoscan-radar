// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unified performance report. One in-memory PerfReport is populated by the
// scenario runner and rendered two ways from the same object: machine-readable
// JSON (when --output is set) and a human-readable text summary (stdout by
// default, and to a file with --text). Dependency-free JSON writing keeps the
// harness free of third-party libraries.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "utils/perf_platform.hpp"
#include "utils/perf_stats.hpp"

namespace holoscan::radar::perf {

struct RunSettings {
  int warmup_frames{5};
  int measured_frames{20};
  int repeats{1};
  // Whether the caller named these on the command line. Most harnesses
  // synthesize their input and can run as many frames as asked, so they ignore
  // both. A harness reading a finite source cannot: the collection caps the
  // achievable frame count, and it needs to know whether a plan that does not
  // fit may be reduced to one that does (defaults, so fitting is a courtesy)
  // or must be honoured and the scenario skipped (pinned, so silently
  // measuring something else would misreport what was run).
  bool warmup_pinned{false};
  bool frames_pinned{false};
  std::string sync_policy{"device_sync_at_terminal_sink"};
  double stop_after_s{0.0};  // 0 = disabled; wall-clock cap per scenario
  bool stopped_early{false};
  int scenarios_planned{0};
  int scenarios_completed{0};
};

// Definition of a named size-ladder entry, so a reader knows what an id like
// "xlw" maps to without cross-referencing the source. Only the sizes actually
// exercised in a run are recorded.
struct SizeDef {
  std::string id;
  std::vector<int64_t> shape;  // CFAR plane (doppler, range)
  int64_t cells{0};
  std::string role;
};

// One row per scenario x timing bucket. NaN fields are omitted from text and
// serialized as null in JSON, so bucket-specific columns stay clean.
struct ResultRow {
  std::string scenario_id;
  std::string op_name{"CFARDetectorOp"};
  std::string bucket;  // "setup" | "compute"
  std::string variant;
  std::string precision{"float"};
  std::string size_id;
  std::vector<int64_t> map_shape;
  std::vector<int64_t> input_shape;
  int64_t out_batch{1};
  std::vector<int64_t> guard;
  std::vector<int64_t> training;
  int64_t k{1};

  double setup_ms{std::numeric_limits<double>::quiet_NaN()};
  double cold_ms{std::numeric_limits<double>::quiet_NaN()};
  LatencyStats latency;  // compute / override per-frame stats

  double override_ms{std::numeric_limits<double>::quiet_NaN()};
  double override_delta_ms{std::numeric_limits<double>::quiet_NaN()};

  double cells_per_second{std::numeric_limits<double>::quiet_NaN()};
  double slices_per_second{std::numeric_limits<double>::quiet_NaN()};
  double estimated_effective_gbps{std::numeric_limits<double>::quiet_NaN()};

  std::string status{"ok"};  // "ok" | "skipped" | "timeout"
  // Set when a row measured successfully but under conditions the reader has
  // to know about -- e.g. a frame plan reduced to fit a short collection.
  // Separate from `status`, which answers whether the scenario measured at all:
  // a consumer filtering on status == "ok" must still see these rows, and a
  // human must still be told the plan was not the one printed in the header.
  std::string qualifier;
  std::string notes;
  std::string calibration_pair;   // paired slow/fast id, empty if none
  std::string calibration_mode;   // "pfa" | "alpha_lut"
};

struct AlphaCalibrationRow {
  std::string pair_id;
  std::string variant;
  std::string size_id;
  int64_t k{1};
  double slow_cold_ms{std::numeric_limits<double>::quiet_NaN()};
  double fast_cold_ms{std::numeric_limits<double>::quiet_NaN()};
  double slow_ovr_ms{std::numeric_limits<double>::quiet_NaN()};
  double fast_ovr_ms{std::numeric_limits<double>::quiet_NaN()};
  double cold_speedup{std::numeric_limits<double>::quiet_NaN()};
  double ovr_speedup{std::numeric_limits<double>::quiet_NaN()};
};

// One measured [live]-override transition, sampled across whole graph sessions
// rather than read off a single tick.
//
// The compute table's `ovr_ms` is the worst frame in one short spike window, so
// it answers "is this override cheap or expensive" and nothing finer. When an
// override is the *point* of the measurement -- a costly rebuild a caller might
// do mid-mission -- one noisy sample is not enough, and this row is the longer
// answer: one sample per session, summarized.
//
// `first_ms` is separated from the rest on purpose. Process-global caches (the
// cuFFT plan cache above all) survive a session, so the first session pays a
// genuinely cold build and later ones do not, and pooling them would report a
// cost nobody experiences. The pair is the real range: cold build vs. warm
// process.
struct OverrideProbeRow {
  std::string scenario_id;
  std::string op_name;
  std::string size_id;
  std::string parameter;   // which [live] parameter moved
  std::string transition;  // e.g. "miss_2x", "hit_return"
  std::string from_to;     // human-readable value change
  double first_ms{std::numeric_limits<double>::quiet_NaN()};
  LatencyStats later;      // override tick, sessions after the first
  double steady_before_ms{std::numeric_limits<double>::quiet_NaN()};
  double steady_after_ms{std::numeric_limits<double>::quiet_NaN()};
  std::string notes;
};

struct PerfReport {
  Provenance provenance;
  Platform platform;
  RunSettings run_settings;
  std::vector<SizeDef> sizes;
  std::vector<ResultRow> results;
  std::vector<AlphaCalibrationRow> alpha_calibration;
  std::vector<OverrideProbeRow> override_probe;
  // Text that applies to the whole run -- how to read a column, what the
  // throughput model counts -- printed once below the tables. A harness whose
  // explanation is the same for every scenario belongs here rather than in each
  // row's notes, where N copies of one paragraph bury the per-row detail that
  // actually differs.
  std::vector<std::string> footnotes;
};

namespace detail {

inline bool is_num(double v) { return !std::isnan(v); }

inline std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      default: out += c;
    }
  }
  return out;
}

inline std::string num_or_null(double v) {
  if (!is_num(v)) {
    return "null";
  }
  std::ostringstream os;
  os << std::setprecision(6) << v;
  return os.str();
}

inline std::string shape_json(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    out += std::to_string(s[i]);
    if (i + 1 < s.size()) {
      out += ", ";
    }
  }
  out += "]";
  return out;
}

inline std::string shape_text(const std::vector<int64_t>& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    out += std::to_string(s[i]);
    if (i + 1 < s.size()) {
      out += "x";
    }
  }
  return out;
}

// Fixed-width left-justified column. Overlong values are truncated with
// "..." so subsequent columns stay aligned.
inline std::string col(const std::string& s, int width) {
  if (width <= 0) {
    return "";
  }
  if (static_cast<int>(s.size()) == width) {
    return s;
  }
  if (static_cast<int>(s.size()) < width) {
    return s + std::string(static_cast<size_t>(width) - s.size(), ' ');
  }
  if (width <= 3) {
    return s.substr(0, static_cast<size_t>(width));
  }
  return s.substr(0, static_cast<size_t>(width - 3)) + "...";
}

inline std::string fixed(double v, int prec) {
  if (!is_num(v)) {
    return "-";
  }
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << v;
  return os.str();
}

// Close a table row: a bracket tag for anything that is not a plain successful
// measurement, then the row's note on an indented continuation line.
//
// A skipped or timed-out row is otherwise indistinguishable from a failure --
// every measured column reads "-" either way -- and the reason a harness
// recorded is the whole difference between "this input cannot support this
// scenario" and "something broke". A qualified row is worse: it carries real
// numbers, so nothing about it looks unusual, and the caveat has to be printed
// or it is lost. The note lands on its own line rather than inline because the
// notes run to a sentence and would push the row past any terminal width, and
// because a run with many tagged rows stays scannable when the columns keep
// their alignment.
inline void end_row(std::ostream& os, const std::string& status,
                    const std::string& qualifier, const std::string& notes) {
  const std::string tag = status != "ok" ? status : qualifier;
  if (!tag.empty()) {
    os << " [" << tag << "]";
  }
  os << "\n";
  if (!tag.empty() && !notes.empty()) {
    os << "    " << notes << "\n";
  }
}

inline std::string sci(double v) {
  if (!is_num(v)) {
    return "-";
  }
  std::ostringstream os;
  os << std::setprecision(3) << v;
  return os.str();
}

// Minimum shape column width for the text table, with trailing padding so a
// shape that exactly fills the field (e.g. 4x4x16x128x512) does not run into
// the next numeric column.
inline int shape_column_width(const PerfReport& r, int min_width = 14) {
  int width = min_width;
  auto consider = [&](const std::vector<int64_t>& shape) {
    width = std::max(width,
                     static_cast<int>(shape_text(shape).size()) + 2);
  };
  for (const auto& s : r.sizes) {
    consider(s.shape);
  }
  for (const auto& row : r.results) {
    consider(row.input_shape);
  }
  return std::min(width, 26);
}

}  // namespace detail

inline void write_json(std::ostream& os, const PerfReport& r) {
  os << "{\n";
  os << "  \"schema_version\": 2,\n";
  os << "  \"provenance\": {\n";
  os << "    \"git_commit\": \"" << detail::json_escape(r.provenance.git_commit)
     << "\",\n";
  os << "    \"git_dirty\": " << (r.provenance.git_dirty ? "true" : "false")
     << ",\n";
  os << "    \"timestamp_utc\": \""
     << detail::json_escape(r.provenance.timestamp_utc) << "\"\n";
  os << "  },\n";
  os << "  \"platform\": {\n";
  os << "    \"gpu_name\": \"" << detail::json_escape(r.platform.gpu_name)
     << "\",\n";
  os << "    \"gpu_driver\": \"" << detail::json_escape(r.platform.gpu_driver)
     << "\",\n";
  os << "    \"cuda_runtime\": \""
     << detail::json_escape(r.platform.cuda_runtime) << "\",\n";
  os << "    \"holoscan_version\": \""
     << detail::json_escape(r.platform.holoscan_version) << "\",\n";
  os << "    \"cpu_model\": \"" << detail::json_escape(r.platform.cpu_model)
     << "\",\n";
  os << "    \"os\": \"" << detail::json_escape(r.platform.os_name) << "\"\n";
  os << "  },\n";
  os << "  \"run_settings\": {\n";
  os << "    \"warmup_frames\": " << r.run_settings.warmup_frames << ",\n";
  os << "    \"measured_frames\": " << r.run_settings.measured_frames << ",\n";
  os << "    \"repeats\": " << r.run_settings.repeats << ",\n";
  os << "    \"sync_policy\": \""
     << detail::json_escape(r.run_settings.sync_policy) << "\",\n";
  os << "    \"stop_after_s\": "
     << detail::num_or_null(r.run_settings.stop_after_s) << ",\n";
  os << "    \"stopped_early\": "
     << (r.run_settings.stopped_early ? "true" : "false") << ",\n";
  os << "    \"scenarios_planned\": " << r.run_settings.scenarios_planned
     << ",\n";
  os << "    \"scenarios_completed\": " << r.run_settings.scenarios_completed
     << "\n";
  os << "  },\n";
  os << "  \"sizes\": [\n";
  for (size_t i = 0; i < r.sizes.size(); ++i) {
    const auto& s = r.sizes[i];
    os << "    {\"id\": \"" << s.id << "\", \"shape\": "
       << detail::shape_json(s.shape) << ", \"cells\": " << s.cells
       << ", \"role\": \"" << detail::json_escape(s.role) << "\"}"
       << (i + 1 < r.sizes.size() ? "," : "") << "\n";
  }
  os << "  ],\n";
  os << "  \"results\": [\n";
  for (size_t i = 0; i < r.results.size(); ++i) {
    const auto& row = r.results[i];
    os << "    {\n";
    os << "      \"scenario_id\": \"" << detail::json_escape(row.scenario_id)
       << "\",\n";
    os << "      \"operator\": \"" << detail::json_escape(row.op_name)
       << "\",\n";
    os << "      \"bucket\": \"" << row.bucket << "\",\n";
    os << "      \"variant\": \"" << row.variant << "\",\n";
    os << "      \"precision\": \"" << row.precision << "\",\n";
    os << "      \"size_id\": \"" << row.size_id << "\",\n";
    os << "      \"map_shape\": " << detail::shape_json(row.map_shape) << ",\n";
    os << "      \"input_shape\": " << detail::shape_json(row.input_shape)
       << ",\n";
    os << "      \"out_batch\": " << row.out_batch << ",\n";
    os << "      \"guard\": " << detail::shape_json(row.guard) << ",\n";
    os << "      \"training\": " << detail::shape_json(row.training) << ",\n";
    os << "      \"k\": " << row.k << ",\n";
    os << "      \"setup_ms\": " << detail::num_or_null(row.setup_ms) << ",\n";
    os << "      \"cold_ms\": " << detail::num_or_null(row.cold_ms) << ",\n";
    os << "      \"min_ms\": " << detail::num_or_null(row.latency.min_ms)
       << ",\n";
    os << "      \"median_ms\": " << detail::num_or_null(row.latency.median_ms)
       << ",\n";
    os << "      \"p90_ms\": " << detail::num_or_null(row.latency.p90_ms)
       << ",\n";
    os << "      \"p99_ms\": " << detail::num_or_null(row.latency.p99_ms)
       << ",\n";
    os << "      \"max_ms\": " << detail::num_or_null(row.latency.max_ms)
       << ",\n";
    os << "      \"override_ms\": " << detail::num_or_null(row.override_ms)
       << ",\n";
    os << "      \"override_delta_ms\": "
       << detail::num_or_null(row.override_delta_ms) << ",\n";
    os << "      \"cells_per_second\": "
       << detail::num_or_null(row.cells_per_second) << ",\n";
    os << "      \"slices_per_second\": "
       << detail::num_or_null(row.slices_per_second) << ",\n";
    os << "      \"estimated_effective_gbps\": "
       << detail::num_or_null(row.estimated_effective_gbps) << ",\n";
    os << "      \"status\": \"" << row.status << "\",\n";
    os << "      \"qualifier\": \"" << detail::json_escape(row.qualifier)
       << "\",\n";
    os << "      \"notes\": \"" << detail::json_escape(row.notes) << "\",\n";
    os << "      \"calibration_pair\": \""
       << detail::json_escape(row.calibration_pair) << "\",\n";
    os << "      \"calibration_mode\": \""
       << detail::json_escape(row.calibration_mode) << "\"\n";
    os << "    }" << (i + 1 < r.results.size() ? "," : "") << "\n";
  }
  os << "  ],\n";
  os << "  \"alpha_calibration\": [\n";
  for (size_t i = 0; i < r.alpha_calibration.size(); ++i) {
    const auto& row = r.alpha_calibration[i];
    os << "    {\n";
    os << "      \"pair_id\": \"" << detail::json_escape(row.pair_id)
       << "\",\n";
    os << "      \"variant\": \"" << detail::json_escape(row.variant)
       << "\",\n";
    os << "      \"size_id\": \"" << row.size_id << "\",\n";
    os << "      \"k\": " << row.k << ",\n";
    os << "      \"slow_cold_ms\": " << detail::num_or_null(row.slow_cold_ms)
       << ",\n";
    os << "      \"fast_cold_ms\": " << detail::num_or_null(row.fast_cold_ms)
       << ",\n";
    os << "      \"slow_ovr_ms\": " << detail::num_or_null(row.slow_ovr_ms)
       << ",\n";
    os << "      \"fast_ovr_ms\": " << detail::num_or_null(row.fast_ovr_ms)
       << ",\n";
    os << "      \"cold_speedup\": " << detail::num_or_null(row.cold_speedup)
       << ",\n";
    os << "      \"ovr_speedup\": " << detail::num_or_null(row.ovr_speedup)
       << "\n";
    os << "    }" << (i + 1 < r.alpha_calibration.size() ? "," : "") << "\n";
  }
  os << "  ],\n";
  os << "  \"override_probe\": [\n";
  for (size_t i = 0; i < r.override_probe.size(); ++i) {
    const auto& row = r.override_probe[i];
    os << "    {\n";
    os << "      \"scenario_id\": \"" << detail::json_escape(row.scenario_id)
       << "\",\n";
    os << "      \"op_name\": \"" << detail::json_escape(row.op_name)
       << "\",\n";
    os << "      \"size_id\": \"" << row.size_id << "\",\n";
    os << "      \"parameter\": \"" << detail::json_escape(row.parameter)
       << "\",\n";
    os << "      \"transition\": \"" << detail::json_escape(row.transition)
       << "\",\n";
    os << "      \"from_to\": \"" << detail::json_escape(row.from_to)
       << "\",\n";
    os << "      \"first_ms\": " << detail::num_or_null(row.first_ms) << ",\n";
    os << "      \"later_samples\": " << row.later.samples << ",\n";
    os << "      \"later_median_ms\": "
       << detail::num_or_null(row.later.median_ms) << ",\n";
    os << "      \"later_p90_ms\": " << detail::num_or_null(row.later.p90_ms)
       << ",\n";
    os << "      \"later_max_ms\": " << detail::num_or_null(row.later.max_ms)
       << ",\n";
    os << "      \"steady_before_ms\": "
       << detail::num_or_null(row.steady_before_ms) << ",\n";
    os << "      \"steady_after_ms\": "
       << detail::num_or_null(row.steady_after_ms) << ",\n";
    os << "      \"notes\": \"" << detail::json_escape(row.notes) << "\"\n";
    os << "    }" << (i + 1 < r.override_probe.size() ? "," : "") << "\n";
  }
  os << "  ],\n";
  os << "  \"footnotes\": [\n";
  for (size_t i = 0; i < r.footnotes.size(); ++i) {
    os << "    \"" << detail::json_escape(r.footnotes[i]) << "\""
       << (i + 1 < r.footnotes.size() ? "," : "") << "\n";
  }
  os << "  ]\n";
  os << "}\n";
}

inline void write_text(std::ostream& os, const PerfReport& r) {
  os << "=== holoscan-radar performance report ===\n";
  os << "commit: " << r.provenance.git_commit
     << (r.provenance.git_dirty ? " (dirty)" : " (clean)") << "  "
     << r.provenance.timestamp_utc << "\n";
  os << "platform: " << r.platform.gpu_name << " | driver "
     << r.platform.gpu_driver << " | CUDA " << r.platform.cuda_runtime
     << " | Holoscan " << r.platform.holoscan_version << "\n";
  os << "          " << r.platform.cpu_model << " | " << r.platform.os_name
     << "\n";
  os << "run: warmup=" << r.run_settings.warmup_frames
     << " measured=" << r.run_settings.measured_frames
     << " repeats=" << r.run_settings.repeats;
  if (r.run_settings.stop_after_s > 0.0) {
    os << " stop_after_s=" << r.run_settings.stop_after_s;
  }
  os << "\n";
  if (r.run_settings.stopped_early) {
    os << "stopped early after " << r.run_settings.scenarios_completed << "/"
       << r.run_settings.scenarios_planned
       << " scenarios (last exceeded stop_after_s)\n";
  }
  os << "note: latencies are valid only on the machine/GPU they were captured "
        "on.\n";

  // --- sizes (ladder definitions used in this run) ---
  const int shape_w = detail::shape_column_width(r);
  if (!r.sizes.empty()) {
    os << "\n--- sizes ---\n";
    os << detail::col("id", 6) << detail::col("shape (DxR)", shape_w)
       << detail::col("cells", 10) << "role\n";
    for (const auto& s : r.sizes) {
      os << detail::col(s.id, 6)
         << detail::col(detail::shape_text(s.shape), shape_w)
         << detail::col(std::to_string(s.cells), 10) << s.role << "\n";
    }
  }

  // --- setup ---
  os << "\n--- setup ---\n";
  os << detail::col("scenario", 40) << detail::col("size", 6)
     << detail::col("setup_ms", 10) << "\n";
  for (const auto& row : r.results) {
    if (row.bucket != "setup") {
      continue;
    }
    os << detail::col(row.scenario_id, 40) << detail::col(row.size_id, 6)
       << detail::col(detail::fixed(row.setup_ms, 2), 10);
    detail::end_row(os, row.status, row.qualifier, row.notes);
  }

  // --- compute (steady state + inline live override) ---
  os << "\n--- compute ---\n";
  os << detail::col("scenario", 40) << detail::col("size", 6)
     << detail::col("shape", shape_w) << detail::col("cold_ms", 10)
     << detail::col("median_ms", 10) << detail::col("p90_ms", 10)
     << detail::col("ovr_ms", 10) << detail::col("ovr_d_ms", 10)
     << detail::col("cells/s", 11) << detail::col("GB/s", 8) << "\n";
  for (const auto& row : r.results) {
    if (row.bucket != "compute") {
      continue;
    }
    os << detail::col(row.scenario_id, 40) << detail::col(row.size_id, 6)
       << detail::col(detail::shape_text(row.input_shape), shape_w)
       << detail::col(detail::fixed(row.cold_ms, 3), 10)
       << detail::col(detail::fixed(row.latency.median_ms, 3), 10)
       << detail::col(detail::fixed(row.latency.p90_ms, 3), 10)
       << detail::col(detail::fixed(row.override_ms, 3), 10)
       << detail::col(detail::fixed(row.override_delta_ms, 3), 10)
       << detail::col(detail::sci(row.cells_per_second), 11)
       << detail::col(detail::fixed(row.estimated_effective_gbps, 1), 8);
    detail::end_row(os, row.status, row.qualifier, row.notes);
  }

  if (!r.alpha_calibration.empty()) {
    os << "--- alpha calibration ---\n";
    os << detail::col("pair", 14) << detail::col("variant", 6)
       << detail::col("size", 6) << detail::col("k", 4)
       << detail::col("slow_cold", 10) << detail::col("fast_cold", 10)
       << detail::col("cold_x", 8) << detail::col("slow_ovr", 10)
       << detail::col("fast_ovr", 10) << detail::col("ovr_x", 8) << "\n";
    for (const auto& row : r.alpha_calibration) {
      os << detail::col(row.pair_id, 14) << detail::col(row.variant, 6)
         << detail::col(row.size_id, 6) << detail::col(std::to_string(row.k), 4)
         << detail::col(detail::fixed(row.slow_cold_ms, 3), 10)
         << detail::col(detail::fixed(row.fast_cold_ms, 3), 10)
         << detail::col(detail::fixed(row.cold_speedup, 2), 8)
         << detail::col(detail::fixed(row.slow_ovr_ms, 3), 10)
         << detail::col(detail::fixed(row.fast_ovr_ms, 3), 10)
         << detail::col(detail::fixed(row.ovr_speedup, 2), 8) << "\n";
    }
  }

  if (!r.override_probe.empty()) {
    os << "\n--- live override probe (one sample per session) ---\n";
    os << detail::col("scenario", 34) << detail::col("param", 8)
       << detail::col("transition", 12) << detail::col("from -> to", 16)
       << detail::col("n", 4) << detail::col("first_ms", 10)
       << detail::col("med_ms", 10) << detail::col("p90_ms", 10)
       << detail::col("before_ms", 10) << detail::col("after_ms", 10) << "\n";
    for (const auto& row : r.override_probe) {
      os << detail::col(row.scenario_id, 34) << detail::col(row.parameter, 8)
         << detail::col(row.transition, 12) << detail::col(row.from_to, 16)
         // The first session counts only when it produced a sample: it is
         // dropped when the session was too short, and adding it
         // unconditionally then reports one more session than was measured.
         << detail::col(std::to_string(row.later.samples +
                                       (std::isfinite(row.first_ms) ? 1 : 0)),
                        4)
         << detail::col(detail::fixed(row.first_ms, 3), 10)
         << detail::col(detail::fixed(row.later.median_ms, 3), 10)
         << detail::col(detail::fixed(row.later.p90_ms, 3), 10)
         << detail::col(detail::fixed(row.steady_before_ms, 3), 10)
         << detail::col(detail::fixed(row.steady_after_ms, 3), 10);
      if (row.notes.empty()) {
        os << "\n";
      } else {
        os << "  " << row.notes << "\n";
      }
    }
  }

  if (!r.footnotes.empty()) {
    os << "\n--- notes ---\n";
    for (const auto& note : r.footnotes) {
      os << "  " << note << "\n";
    }
  }
  os << "\n";
}

}  // namespace holoscan::radar::perf
