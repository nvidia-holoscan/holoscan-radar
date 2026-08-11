// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for TensorDumpOp — what inserting a debugging
// tap costs the graph it is inserted into. Three axes: tensor size, on-disk
// format (npy / raw), and where the bytes go (a real file or /dev/null, which
// separates the device-to-host copy from the filesystem). A throttled scenario
// measures the other number a caller wants: what a tick costs when every_n
// skips it. `every_n` is [live], and every written scenario overrides it
// mid-session.
//
// Two things the numbers are not. The write lands in the page cache — nothing
// here fsyncs — so this is the cost the graph pays, not time to durable
// storage. And the copy is synchronous by design: the sink has to see the
// producer's finished data, so it waits on the stream, and that wait is part
// of what a dump costs.

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/perf_source.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;
namespace fs = std::filesystem;

namespace {

using PerfPrecision = float;

struct TensorPort {
  static constexpr const char* kName = "tensor";
};

struct SizeSpec {
  std::string id;
  int64_t rows;
  int64_t cols;
  std::string role;
};

// Where the dump goes. `null` keeps the same code path -- the same copy, the
// same formatting, the same write() -- and drops the bytes at the device
// driver, so differencing it against `file` attributes the cost.
enum class Destination { kFile, kNull };

struct DumpScenario {
  std::string id;
  std::string size_id;
  int64_t rows{0};
  int64_t cols{0};
  hr::TensorDumpFormat format{hr::TensorDumpFormat::kNpy};
  Destination destination{Destination::kFile};
  int64_t every_n{1};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 64, 64, "32 KiB — a debug tap on a small intermediate"},
      {"sm", 256, 256, "512 KiB — a pulse batch"},
      {"md", 1024, 1024, "8 MiB — a range-compressed aperture slab"},
      {"lg", 2048, 1024, "16 MiB — a formed SAR image tile"},
      {"xl", 2048, 2048, "32 MiB — a large formed image"}};
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

std::vector<int64_t> tensor_shape(const DumpScenario& scen) {
  return {scen.rows, scen.cols};
}

double bytes_per_frame(const DumpScenario& scen) {
  return static_cast<double>(scen.rows) * static_cast<double>(scen.cols) *
         static_cast<double>(sizeof(std::complex<PerfPrecision>));
}

// A directory per scenario, emptied afterwards: with {index} in the pattern
// every tick is a new file, which is what a real dump does, and the harness
// should not leave gigabytes behind to prove it.
class ScratchDir {
 public:
  ScratchDir() {
    const char* base = std::getenv("HOLOSCAN_RADAR_PERF_DUMP_DIR");
    path_ = (base != nullptr ? fs::path(base) : fs::temp_directory_path()) /
            ("perf_tensor_dump_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(path_, ec);
    fs::create_directories(path_, ec);
  }
  ~ScratchDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  ScratchDir(const ScratchDir&) = delete;
  ScratchDir& operator=(const ScratchDir&) = delete;

  std::string pattern() const { return (path_ / "dump_{index}.bin").string(); }

 private:
  fs::path path_;
};

// The sink is the last operator in the graph, so nothing downstream can record
// the end of a frame; it records its own.
class TimedTensorDumpOp : public hr::TensorDumpOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TimedTensorDumpOp, hr::TensorDumpOp)
  TimedTensorDumpOp() = default;

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    hr::TensorDumpOp::compute(op_input, op_output, context);
    if (collector_ != nullptr) {
      collector_->ends.push_back(perf::Clock::now());
    }
  }

  void configure(perf::TimingCollector* collector) { collector_ = collector; }

 private:
  perf::TimingCollector* collector_{nullptr};
};

class DumpPerfApp : public holoscan::Application {
 public:
  const DumpScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  std::string path;
  int total_frames{25};
  int64_t override_at{-1};
  int64_t override_value{2};

  void compose() override {
    using holoscan::Arg;
    auto dump = make_operator<TimedTensorDumpOp>(
        "dump", Arg("path") = path, Arg("format") = scen->format,
        Arg("every_n") = scen->every_n);
    dump->configure(collector);

    auto src = make_operator<perf::PerfTensorSourceOp<TensorPort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, dump, {{"tensor", "tensor"}});

    if (override_at >= 0) {
      auto ovr = make_operator<perf::PerfOverrideSourceOp<int64_t>>(
          "every_n_override",
          make_condition<holoscan::CountCondition>(total_frames));
      ovr->configure(override_at, override_value);
      add_flow(ovr, dump, {{"output", "every_n"}});
    }
  }
};

void fill_throughput(perf::ResultRow* row, const DumpScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0 || scen.every_n != 1) {
    return;
  }
  const double sec = median_ms / 1000.0;
  row->cells_per_second =
      static_cast<double>(scen.rows) * static_cast<double>(scen.cols) / sec;
  // Each byte is moved twice on a written tick: device to host, then host to
  // the file. Counting both is what makes the number comparable with the
  // reader's ingest figure.
  row->estimated_effective_gbps = 2.0 * bytes_per_frame(scen) / sec / 1e9;
}

std::string variant_of(const DumpScenario& scen) {
  if (scen.every_n != 1) {
    return "throttled";
  }
  const char* where = scen.destination == Destination::kNull ? "null" : "file";
  return std::string(hr::to_string(scen.format)) + "_" + where;
}

perf::ResultRow base_row(const DumpScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "TensorDumpOp";
  row.variant = variant_of(scen);
  row.size_id = scen.size_id;
  row.input_shape = tensor_shape(scen);
  row.k = scen.every_n;
  return row;
}

DumpScenario make_scenario(const SizeSpec& s,
                           hr::TensorDumpFormat format =
                               hr::TensorDumpFormat::kNpy,
                           Destination destination = Destination::kFile,
                           int64_t every_n = 1) {
  DumpScenario sc;
  sc.size_id = s.id;
  sc.rows = s.rows;
  sc.cols = s.cols;
  sc.format = format;
  sc.destination = destination;
  sc.every_n = every_n;
  std::ostringstream id;
  id << "dump_" << variant_of(sc) << "_" << s.id;
  sc.id = id.str();
  return sc;
}

struct DumpTraits : perf::PerfDriverDefaults {
  using Scenario = DumpScenario;

  static const char* tool_name() { return "perf_tensor_dump"; }
  static const char* tool_description() {
    return "TensorDumpOp performance characterization (D2H + write)";
  }
  static const char* sizes_all_help() {
    return "npy-to-file dumps across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "npy-to-file dumps across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix: every size, both formats, file and /dev/null "
           "destinations, and a throttled scenario";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.rows, s.cols};
      def.cells = s.rows * s.cols;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<DumpScenario> build_default_matrix() {
    std::vector<DumpScenario> out;
    for (const char* sid : {"xs", "sm", "md", "lg"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    // The two comparisons worth having by default: what the filesystem costs
    // on top of the copy, and what a skipped tick costs.
    out.push_back(make_scenario(must_find_size("md"),
                                hr::TensorDumpFormat::kNpy,
                                Destination::kNull));
    out.push_back(make_scenario(must_find_size("md"),
                                hr::TensorDumpFormat::kNpy,
                                Destination::kFile, 1000));
    return out;
  }

  static std::vector<DumpScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<DumpScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(*s));
    }
    return out;
  }

  static std::vector<DumpScenario> build_full_matrix() {
    std::vector<DumpScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
      out.push_back(make_scenario(s, hr::TensorDumpFormat::kRaw));
      out.push_back(make_scenario(s, hr::TensorDumpFormat::kNpy,
                                  Destination::kNull));
    }
    for (const char* sid : {"sm", "xl"}) {
      out.push_back(make_scenario(must_find_size(sid),
                                  hr::TensorDumpFormat::kNpy,
                                  Destination::kFile, 1000));
    }
    return out;
  }

  static void run_scenario(const DumpScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    // A throttled scenario measures skipped ticks; overriding every_n in the
    // middle of it would change what is being measured, so the live override
    // rides on the written scenarios where its cost is visible against a
    // written tick.
    const bool has_override = scen.every_n == 1;
    const perf::RunPlan plan = perf::make_run_plan(rs, has_override);
    ScratchDir scratch;
    const std::string path = scen.destination == Destination::kNull
                                 ? std::string("/dev/null")
                                 : scratch.pattern();

    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          auto frame = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(tensor_shape(scen)),
              "tensor");

          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<DumpPerfApp>();
          app->scen = &scen;
          app->collector = &collector;
          app->frame = std::move(frame);
          app->path = path;
          app->total_frames = p.total_frames;
          app->override_at = p.override_at;
          app->override_value = 2;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    if (scen.every_n != 1) {
      // Four columns come out blank on this row, and a blank that is a
      // deliberate omission has to say so: unexplained, it reads as a
      // measurement that failed.
      std::ostringstream notes;
      row.qualifier = "throttled";
      notes << "every_n = " << scen.every_n
            << ", so every measured tick after the first is skipped -- this "
               "is the cost of leaving a throttled dump wired into a live "
               "graph, and cold_ms is the one tick that did write. cells/s "
               "and GB/s are blank because no bytes move; ovr_ms and "
               "ovr_d_ms are blank because the live override is measured on "
               "the written scenarios, where overriding every_n does not "
               "change what is being measured.";
      row.notes = notes.str();
    }
    report->results.push_back(row);
  }

  // The explanations that are the same for every row. In each row's notes they
  // would print N times and bury the one row with something specific to say --
  // and for an untagged row they would not print at all.
  static void post_run(perf::PerfReport* report) {
    report->footnotes.push_back(
        "GB/s counts each byte twice: once device-to-host, once written. "
        "cells/s counts tensor elements dumped per second.");
    report->footnotes.push_back(
        "A _file scenario writes into the page cache -- nothing here fsyncs -- "
        "so it is the cost to the graph, not time to durable storage. The "
        "matching _null scenario writes to /dev/null on the same code path, so "
        "differencing the two separates the copy from the filesystem.");
    report->footnotes.push_back(
        "every_n is overridden to 2 mid-session on every written scenario. It "
        "rebuilds and caches nothing, so ovr_d_ms is run-to-run variance "
        "around zero, of either sign, rather than a reconfiguration cost. "
        "Whether the override took effect is a unit test's question.");
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      warm.warmup_pinned = true;
      warm.frames_pinned = true;
      run_scenario(make_scenario(must_find_size("xs")), warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<DumpTraits>(argc, argv);
}
