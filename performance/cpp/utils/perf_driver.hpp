// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reusable driver shell for the performance harness. Every perf_<operator>
// consumer shares the same outer program: parse a common CLI, verify a GPU is
// present, quiet the SDK, pick a scenario matrix (default / --full / --sizes),
// stamp the report header, record the exercised size ladder, run the scenarios
// (progress + optional --stop-after-s cap + optional failure isolation), and
// write JSON / text / stdout. Only the scenario matrix, the graph, the input
// tensors, and the throughput model are operator-specific; those stay in the
// consumer via a Traits struct.
//
// A consumer provides a Traits type and calls perf_main<Traits>() from main().
// Traits must supply (all static):
//
//   using Scenario = <type with public `std::string id` and `std::string
//                     size_id` members>;
//   const char* tool_name();          // e.g. "perf_feature_assembly"
//   const char* tool_description();   // one-line summary for --help
//   const char* sizes_all_help();     // --sizes all description
//   const char* sizes_subset_help();  // --sizes a,b,c description
//   const char* full_help();          // --full description (may be multi-line)
//   std::vector<SizeDef> size_ladder_defs();  // full ladder, in ladder order
//   std::vector<Scenario> build_default_matrix();
//   std::vector<Scenario> build_full_matrix();
//   std::vector<Scenario> build_size_sweep(const std::vector<std::string>&);
//   void run_scenario(const Scenario&, const RunSettings&, PerfReport*);
//
// Optional hooks come from PerfDriverDefaults (inherit to get no-op defaults,
// redefine to customize):
//
//   void process_warmup(const RunSettings&);  // discarded prime run
//   void post_run(PerfReport*);               // e.g. calibration summary
//   bool isolate_failures();                  // try/catch each scenario
//
// A consumer may additionally define (detected at compile time; no base-class
// default because the return type is the consumer's Scenario):
//
//   std::vector<Scenario> build_override_sweep(
//       const std::vector<std::string>& sizes);  // one scenario per [live]
//                                                 // param, selected by
//                                                 // --sweep-overrides
//
// When present, --sweep-overrides calls it; when absent, --sweep-overrides is
// rejected for that tool.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "utils/perf_app.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"

namespace holoscan::radar::perf {

// No-op defaults for the optional Traits hooks. Inherit from this so a Traits
// struct only defines the hooks it actually needs.
struct PerfDriverDefaults {
  static void process_warmup(const RunSettings&) {}
  static void post_run(PerfReport*) {}
  static bool isolate_failures() { return false; }
};

// Common CLI shared by every consumer. Operator-specific behavior enters only
// through the scenario matrix the flags select, never through new flags.
struct Options {
  std::string output;
  std::string text;
  bool quiet{false};
  bool full{false};
  bool sweep_overrides{false};     // one scenario per [live] override param
  std::vector<std::string> sizes;  // empty = tier default
  bool sizes_all{false};
  double stop_after_s{0.0};  // 0 = disabled; per-scenario wall-clock cap
  RunSettings rs;
};

namespace detail {

// Compile-time detection of an optional Traits::build_override_sweep(sizes)
// hook. Present on consumers whose operator has [live] override ports and that
// implement the per-param sweep; absent otherwise. Lets --sweep-overrides be a
// no-compile-cost opt-in per consumer without a base-class default (which is
// impossible here — the return type is the consumer's Scenario). A consumer
// that defines build_override_sweep must also define override_sweep_help()
// (used verbatim in --help).
template <typename Traits>
auto has_override_sweep_probe(int) -> decltype(
    Traits::build_override_sweep(
        std::declval<const std::vector<std::string>&>()),
    std::true_type{});
template <typename Traits>
std::false_type has_override_sweep_probe(...);
template <typename Traits>
using has_override_sweep = decltype(has_override_sweep_probe<Traits>(0));

inline std::vector<std::string> split_csv(const std::string& s) {
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

template <typename Traits>
void print_help() {
  std::cout << Traits::tool_name() << " — " << Traits::tool_description()
            << "\n\n"
            << "Usage: " << Traits::tool_name() << " [options]\n\n"
            << "  --output <path>   Write machine-readable JSON report\n"
               "  --text <path>     Also write the text summary to a file\n"
               "  --quiet           Suppress the stdout text summary\n"
               "  --warmup N        Warmup frames excluded from stats "
               "(default 5)\n"
               "  --frames N        Measured frames per scenario (default 20)\n"
               "  --repeats N       Repeat each scenario N times, pooled "
               "(default 1)\n"
            << "  --sizes all       " << Traits::sizes_all_help() << "\n"
            << "  --sizes a,b,c     " << Traits::sizes_subset_help() << "\n"
            << "  --full            " << Traits::full_help() << "\n";
  if constexpr (has_override_sweep<Traits>::value) {
    std::cout << "  --sweep-overrides " << Traits::override_sweep_help()
              << "\n";
  }
  std::cout << "  --stop-after-s S  Stop after first scenario whose wall time "
               "exceeds S seconds\n"
               "  --help            Show this help\n";
}

template <typename Traits>
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
      print_help<Traits>();
      std::exit(0);
    } else if (a == "--output") {
      opt->output = next("--output");
    } else if (a == "--text") {
      opt->text = next("--text");
    } else if (a == "--quiet") {
      opt->quiet = true;
    } else if (a == "--full") {
      opt->full = true;
    } else if (a == "--sweep-overrides") {
      opt->sweep_overrides = true;
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

inline bool ensure_cuda_device(const char* tool_name) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "error: no CUDA device available; " << tool_name
              << " requires a GPU\n";
    return false;
  }
  return true;
}

inline void default_log_level() {
  // Reduce Holoscan log jitter unless the user configured a level. ERROR drops
  // WARN spam (e.g. per-run stack-size advisories) that would clutter progress.
  if (std::getenv("HOLOSCAN_LOG_LEVEL") == nullptr) {
    setenv("HOLOSCAN_LOG_LEVEL", "ERROR", 0);
  }
}

template <typename Traits>
std::vector<typename Traits::Scenario> select_scenarios(const Options& opt) {
  if (opt.sweep_overrides) {
    if constexpr (has_override_sweep<Traits>::value) {
      std::vector<std::string> sizes = opt.sizes;
      if (opt.sizes_all) {
        for (const auto& def : Traits::size_ladder_defs()) {
          sizes.push_back(def.id);
        }
      }
      return Traits::build_override_sweep(sizes);
    } else {
      std::cerr << "error: " << Traits::tool_name()
                << " does not implement --sweep-overrides (operator has no "
                   "[live] override params, or the sweep is not yet wired)\n";
      std::exit(2);
    }
  }
  if (opt.full) {
    return Traits::build_full_matrix();
  }
  if (opt.sizes_all) {
    std::vector<std::string> all;
    for (const auto& def : Traits::size_ladder_defs()) {
      all.push_back(def.id);
    }
    return Traits::build_size_sweep(all);
  }
  if (!opt.sizes.empty()) {
    return Traits::build_size_sweep(opt.sizes);
  }
  return Traits::build_default_matrix();
}

// Record ladder definitions for the sizes this run actually exercises, in
// ladder order, so the report is self-describing.
template <typename Traits>
void record_exercised_sizes(
    const std::vector<typename Traits::Scenario>& scenarios,
    PerfReport* report) {
  std::vector<std::string> used;
  for (const auto& sc : scenarios) {
    if (std::find(used.begin(), used.end(), sc.size_id) == used.end()) {
      used.push_back(sc.size_id);
    }
  }
  for (const auto& def : Traits::size_ladder_defs()) {
    if (std::find(used.begin(), used.end(), def.id) != used.end()) {
      report->sizes.push_back(def);
    }
  }
}

template <typename Traits>
void run_all_scenarios(
    const std::vector<typename Traits::Scenario>& scenarios, const Options& opt,
    PerfReport* report) {
  const size_t n = scenarios.size();
  bool stopped_early = false;
  for (size_t i = 0; i < n; ++i) {
    std::cerr << "[" << (i + 1) << "/" << n << "] " << scenarios[i].id
              << std::flush;
    const auto t0 = Clock::now();
    // Optional failure isolation: a thrown graph error (e.g. an operator
    // rejecting an invalid parameter) would otherwise propagate out of main
    // and abort the process, losing the remaining scenarios and the report.
    if (Traits::isolate_failures()) {
      try {
        Traits::run_scenario(scenarios[i], opt.rs, report);
      } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << "\n";
        continue;
      }
    } else {
      Traits::run_scenario(scenarios[i], opt.rs, report);
    }
    const double elapsed_s = to_ms(Clock::now() - t0) / 1000.0;
    std::cerr << "  (" << elapsed_s << " s)\n";
    report->run_settings.scenarios_completed = static_cast<int>(i + 1);
    if (opt.stop_after_s > 0.0 && elapsed_s > opt.stop_after_s) {
      std::cerr << "stopping early: scenario exceeded --stop-after-s "
                << opt.stop_after_s << " s\n";
      stopped_early = true;
      break;
    }
  }
  report->run_settings.stopped_early = stopped_early;
}

inline int write_outputs(const Options& opt, const PerfReport& report) {
  if (!opt.output.empty()) {
    std::ofstream f(opt.output);
    if (!f) {
      std::cerr << "error: cannot open --output file '" << opt.output << "'\n";
      return 1;
    }
    write_json(f, report);
    std::cerr << "wrote JSON report: " << opt.output << "\n";
  }
  if (!opt.text.empty()) {
    std::ofstream f(opt.text);
    if (!f) {
      std::cerr << "error: cannot open --text file '" << opt.text << "'\n";
      return 1;
    }
    write_text(f, report);
    std::cerr << "wrote text report: " << opt.text << "\n";
  }
  if (!opt.quiet) {
    write_text(std::cout, report);
  }
  return 0;
}

}  // namespace detail

// Program entry shared by every perf_<operator> consumer. Returns a process
// exit code; a consumer's main() is a single `return perf_main<Traits>(...)`.
template <typename Traits>
int perf_main(int argc, char** argv) {
  Options opt;
  if (!detail::parse_args<Traits>(argc, argv, &opt)) {
    return 2;
  }
  if (!detail::ensure_cuda_device(Traits::tool_name())) {
    return 1;
  }
  detail::default_log_level();

  std::vector<typename Traits::Scenario> scenarios =
      detail::select_scenarios<Traits>(opt);

  PerfReport report;
  report.provenance = capture_provenance();
  report.platform = capture_platform();
  report.run_settings = opt.rs;
  report.run_settings.stop_after_s = opt.stop_after_s;
  report.run_settings.scenarios_planned = static_cast<int>(scenarios.size());

  detail::record_exercised_sizes<Traits>(scenarios, &report);

  Traits::process_warmup(opt.rs);
  detail::run_all_scenarios<Traits>(scenarios, opt, &report);
  Traits::post_run(&report);

  return detail::write_outputs(opt, report);
}

}  // namespace holoscan::radar::perf
