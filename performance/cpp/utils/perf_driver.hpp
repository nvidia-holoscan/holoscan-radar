// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reusable driver shell for the performance harness. Every perf_<operator>
// consumer shares the same outer program: parse a common CLI, verify a GPU is
// present, quiet the SDK, select YAML scope cases,
// stamp the report header, record the exercised size profiles, run the scenarios
// (progress + optional --stop-after-s cap + optional failure isolation), and
// write JSON / text / stdout. Each Traits adapter builds valid tensors and
// Graph wiring for the scenario selected from its YAML suite.
//
// A consumer provides a Traits type and calls perf_main<Traits>() from main().
// Traits must supply (all static):
//
//   using Scenario = <type with public `std::string id` and `std::string
//                     size_id` members>;
//   const char* tool_name();          // e.g. "perf_feature_assembly"
//   const char* tool_description();   // one-line summary for --help
//   std::vector<SizeDef> size_ladder_defs();  // full ladder, in ladder order
//   Scenario parse_case(const ProfileCase&, const std::vector<int64_t>& shape);
//   void run_scenario(const Scenario&, const RunSettings&, PerfReport*);
//
// Optional hooks come from PerfDriverDefaults (inherit to get no-op defaults,
// redefine to customize):
//
//   void post_run(PerfReport*, bool quiet);   // e.g. calibration / tables

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <set>
#include <string>
#include <vector>

#include "utils/perf_app.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_profiles.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_process_warmup.hpp"

namespace holoscan::radar::perf {

// No-op defaults for the optional Traits hooks. Inherit from this so a Traits
// struct only defines the hooks it actually needs.
struct PerfDriverDefaults {
  static void post_run(PerfReport*, bool /*quiet*/) {}
};

// Common CLI shared by every consumer. Operator-specific behavior enters only
// through the scenario matrix the flags select, never through new flags.
struct Options {
  std::string output;
  std::string text;
  bool quiet{false};
  std::string scope{"limited"};
  std::vector<std::string> sizes;  // empty = tier default
  bool sizes_all{false};
  double stop_after_s{0.0};  // 0 = disabled; per-scenario wall-clock cap
  RunSettings rs;
};

namespace detail {

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
  std::cout << Traits::tool_name() << " — " << Traits::tool_description() << "\n\n"
            << "Usage: " << Traits::tool_name() << " [options]\n\n"
            << "  --output <path>   Write machine-readable JSON report\n"
               "  --text <path>     Also write the text summary to a file\n"
               "  --quiet           Suppress the stdout text summary\n"
               "  --warmup N        Warmup frames excluded from stats "
               "(default 5)\n"
               "  --frames N        Measured frames per scenario (default 50)\n"
               "  --repeats N       Repeat each scenario N times, pooled "
               "(default 3)\n"
            << "  --scope NAME      minimal|limited|overrides|moderate|thorough\n"
            << "  --sizes all       Use all YAML size profiles\n"
            << "  --sizes a,b,c     Limit to named YAML profiles\n"
            << "  --full            Alias for --scope thorough\n"
            << "  --stop-after-s S  Stop after first scenario whose wall time "
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
      opt->scope = "thorough";
    } else if (a == "--scope") {
      opt->scope = next("--scope");
      if (!known_scope(opt->scope)) {
        std::cerr << "error: unknown scope " << opt->scope << "\n";
        return false;
      }
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
  if (opt->rs.warmup_frames < 0 || opt->rs.measured_frames < 2 || opt->rs.repeats < 1) {
    std::cerr << "error: require --warmup >= 0, --frames >= 2, --repeats >= 1\n";
    return false;
  }
  return true;
}

inline bool ensure_cuda_device(const char* tool_name) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "error: no CUDA device available; " << tool_name << " requires a GPU\n";
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
ProfileSuite load_adapter_suite() {
  const std::string operator_name = std::string(Traits::tool_name()).substr(5);
  return load_profile_suite(
      std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / (operator_name + ".yaml"),
      operator_name);
}

template <typename Traits>
std::vector<typename Traits::Scenario> select_scenarios(const Options& opt) {
  const auto suite = load_adapter_suite<Traits>();
  return select_validated_cases<typename Traits::Scenario>(
      suite, opt.scope, opt.sizes_all ? std::vector<std::string>{} : opt.sizes,
      [](const ProfileCase& c, const std::vector<int64_t>& shape) {
        return Traits::parse_case(c, shape);
      });
}

// Record ladder definitions for the sizes this run actually exercises, in
// ladder order, so the report is self-describing.
template <typename Traits>
void record_exercised_sizes(const std::vector<typename Traits::Scenario>& scenarios,
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
void run_all_scenarios(const std::vector<typename Traits::Scenario>& scenarios, const Options& opt,
                       PerfReport* report) {
  const size_t n = scenarios.size();
  bool stopped_early = false;
  for (size_t i = 0; i < n; ++i) {
    std::cerr << "[" << (i + 1) << "/" << n << "] " << scenarios[i].id << std::flush;
    const auto t0 = Clock::now();
    Traits::run_scenario(scenarios[i], opt.rs, report);
    const double elapsed_s = to_ms(Clock::now() - t0) / 1000.0;
    std::cerr << "  (" << elapsed_s << " s)\n";
    report->run_settings.scenarios_completed = static_cast<int>(i + 1);
    if (opt.stop_after_s > 0.0 && elapsed_s > opt.stop_after_s) {
      std::cerr << "stopping early: scenario exceeded --stop-after-s " << opt.stop_after_s
                << " s\n";
      stopped_early = true;
      break;
    }
  }
  report->run_settings.stopped_early = stopped_early;
}

inline bool complete_report(const PerfReport& report) {
  const auto completed = static_cast<size_t>(report.run_settings.scenarios_completed);
  if (report.results.size() != 2U * completed)
    return false;
  for (size_t i = 0; i < completed; ++i) {
    const auto& setup = report.results[2U * i];
    const auto& compute = report.results[2U * i + 1U];
    if (setup.bucket != "setup" || compute.bucket != "compute" ||
        setup.scenario_id != compute.scenario_id || !std::isfinite(setup.setup_ms) ||
        !std::isfinite(compute.cold_ms) || !std::isfinite(compute.latency.median_ms) ||
        compute.latency.samples !=
            report.run_settings.measured_frames * report.run_settings.repeats ||
        !std::isfinite(compute.achieved_frames_per_second) ||
        !std::isfinite(compute.source_frames_per_second))
      return false;
    if (!compute.live_port.empty() &&
        (!std::isfinite(compute.override_ms) || !std::isfinite(compute.override_delta_ms)))
      return false;
  }
  return true;
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

  std::vector<typename Traits::Scenario> scenarios;
  try {
    scenarios = detail::select_scenarios<Traits>(opt);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }

  PerfReport report;
  report.provenance = capture_provenance();
  report.platform = capture_platform();
  report.run_settings = opt.rs;
  report.run_settings.scope = opt.scope;
  report.run_settings.stop_after_s = opt.stop_after_s;
  report.run_settings.scenarios_planned = static_cast<int>(scenarios.size());

  detail::record_exercised_sizes<Traits>(scenarios, &report);

  run_process_warmup([&] {
    RunSettings warm;
    warm.warmup_frames = 0;
    warm.measured_frames = 2;
    warm.repeats = 1;
    PerfReport discarded;
    const auto suite = detail::load_adapter_suite<Traits>();
    const auto baseline = select_validated_cases<typename Traits::Scenario>(
        suite, "minimal", {"small"}, [](const ProfileCase& c, const std::vector<int64_t>& shape) {
          return Traits::parse_case(c, shape);
        });
    Traits::run_scenario(baseline.front(), warm, &discarded);
  });
  try {
    detail::run_all_scenarios<Traits>(scenarios, opt, &report);
    Traits::post_run(&report, opt.quiet);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (!detail::complete_report(report)) {
    std::cerr << "error: incomplete performance report\n";
    return 1;
  }
  return detail::write_outputs(opt, report);
}

}  // namespace holoscan::radar::perf
