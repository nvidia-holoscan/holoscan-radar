// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for PhaseCorrectionOp — broadcast
// per-pulse phase multiply. Sweeps (pulses x range_bins), sense, and a
// leading batch. No [live] params, so no override measurement.
// Measurement tool only: no timing assertions. See
// docs/planning/OPERATOR_DEVELOPMENT.md.

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/phase_correction_op/phase_correction_op.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_source.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

struct PhasePort {
  static constexpr const char* kName = "phase_correction";
};

struct SizeSpec {
  std::string id;
  int64_t num_pulses;
  int64_t num_range_bins;
  std::string role;
};

struct PhaseScenario {
  std::string id;
  std::string size_id;
  int64_t num_pulses{0};
  int64_t num_range_bins{0};
  int64_t channels{1};
  hr::PhaseCorrectionSense sense{hr::PhaseCorrectionSense::kSubtract};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 16, 512, "small slab; fast smoke"},
      {"sm", 32, 1024, "compact aperture slab"},
      {"md", 64, 2048, "common dev size"},
      {"lg", 128, 4096, "moderate production slab"},
      {"xl", 256, 8192, "range-heavy slab"}};
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

std::string sense_tag(hr::PhaseCorrectionSense s) {
  return s == hr::PhaseCorrectionSense::kAdd ? "add" : "sub";
}

std::vector<int64_t> profiles_shape(const PhaseScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_pulses, scen.num_range_bins};
  }
  return {scen.num_pulses, scen.num_range_bins};
}

std::vector<int64_t> phase_shape(const PhaseScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_pulses};
  }
  return {scen.num_pulses};
}

std::shared_ptr<holoscan::Tensor> make_phase_device(
    const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  std::vector<PerfPrecision> host(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    host[static_cast<size_t>(i)] =
        static_cast<PerfPrecision>(0.01 * static_cast<double>(i));
  }
  auto tensor = hr::make_device_tensor(shape, hr::real_dtype<PerfPrecision>());
  if (cudaMemcpy(tensor->data(), host.data(),
                 sizeof(PerfPrecision) * host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("make_phase_device: H2D failed");
  }
  return tensor;
}

template <typename T>
class PhasePerfApp : public holoscan::Application {
 public:
  const PhaseScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap profiles;
  holoscan::TensorMap phase;
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::PhaseCorrectionOp<T>>(
        "phase_correction", Arg("sense") = scen->sense);

    auto src = make_operator<
        perf::PerfTensorSourceOp<perf::RangeProfilesPort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, profiles);
    add_flow(src, op, {{"range_profiles", "range_profiles"}});

    auto ph = make_operator<perf::PerfTensorSourceOp<PhasePort>>(
        "phase_src",
        make_condition<holoscan::CountCondition>(total_frames));
    ph->configure(nullptr, phase);
    add_flow(ph, op, {{"phase_correction", "phase_correction"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("out_sink");
    term->configure(collector);
    add_flow(op, term, {{"range_profiles", "input"}});
  }
};

void fill_throughput(perf::ResultRow* row, const PhaseScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double pulses = static_cast<double>(scen.num_pulses);
  const double bins = static_cast<double>(scen.num_range_bins);
  row->cells_per_second = batch * pulses * bins / sec;
  const double csz = sizeof(std::complex<PerfPrecision>);
  const double bytes = 2.0 * batch * pulses * bins * csz;
  row->estimated_effective_gbps = bytes / sec / 1e9;
}

perf::ResultRow base_row(const PhaseScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "PhaseCorrectionOp";
  row.variant = sense_tag(scen.sense);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_pulses, scen.num_range_bins};
  row.input_shape = profiles_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.num_pulses;
  row.notes = "no_live_params";
  return row;
}

PhaseScenario make_scenario(hr::PhaseCorrectionSense sense,
                            const SizeSpec& s, int64_t channels = 1) {
  PhaseScenario sc;
  sc.size_id = s.id;
  sc.num_pulses = s.num_pulses;
  sc.num_range_bins = s.num_range_bins;
  sc.channels = channels;
  sc.sense = sense;
  std::ostringstream id;
  id << "phase_" << sense_tag(sense) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

struct PhaseTraits : perf::PerfDriverDefaults {
  using Scenario = PhaseScenario;

  static const char* tool_name() { return "perf_phase_correction"; }
  static const char* tool_description() {
    return "PhaseCorrectionOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "subtract size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "subtract size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (sense x sizes x channel batch)";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_pulses, s.num_range_bins};
      def.cells = s.num_pulses * s.num_range_bins;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<PhaseScenario> build_default_matrix() {
    std::vector<PhaseScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(hr::PhaseCorrectionSense::kSubtract,
                                  must_find_size(sid)));
    }
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(hr::PhaseCorrectionSense::kAdd, md));
    out.push_back(make_scenario(hr::PhaseCorrectionSense::kSubtract, md,
                                4));
    return out;
  }

  static std::vector<PhaseScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<PhaseScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(
          make_scenario(hr::PhaseCorrectionSense::kSubtract, *s));
    }
    return out;
  }

  static std::vector<PhaseScenario> build_full_matrix() {
    std::vector<PhaseScenario> out;
    const auto senses = {hr::PhaseCorrectionSense::kSubtract,
                         hr::PhaseCorrectionSense::kAdd};
    for (const auto& s : size_ladder()) {
      for (auto sense : senses) {
        out.push_back(make_scenario(sense, s));
      }
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      for (int64_t ch : {4, 16}) {
        out.push_back(make_scenario(hr::PhaseCorrectionSense::kSubtract,
                                    s, ch));
      }
    }
    return out;
  }

  static void run_scenario(const PhaseScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(
                  profiles_shape(scen)),
              "range_profiles");
          auto phi = perf::single_map(make_phase_device(phase_shape(scen)),
                                      "phase_correction");

          collector.t_construct = perf::Clock::now();
          auto app =
              holoscan::make_application<PhasePerfApp<PerfPrecision>>();
          app->scen = &scen;
          app->collector = &collector;
          app->profiles = std::move(cube);
          app->phase = std::move(phi);
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are steady; no "
        "[live] params so no override_ms; cells/s is complex cells/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(hr::PhaseCorrectionSense::kSubtract,
                                 must_find_size("xs")),
                   warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<PhaseTraits>(argc, argv);
}
