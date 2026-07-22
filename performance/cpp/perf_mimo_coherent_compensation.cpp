// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for MIMOCoherentCompensationOp — TDM per-Doppler
// inter-TX phase correction on separated RD maps. Sweeps
// (M_tx, N_rx, num_doppler, num_range_bins), a leading batch axis, and
// inlines a mid-session [live] waveform override so correction-table rebuild
// cost is measured. Measurement tool only.

#include <cuda_runtime_api.h>

#include <complex>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_platform.hpp"
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

constexpr double kDefaultPri = 25.0e-6;
constexpr double kOverridePri = 27.5e-6;

struct SizeSpec {
  std::string id;
  int64_t num_tx;
  int64_t num_rx;
  int64_t num_doppler;
  int64_t num_range_bins;
  std::string role;
};

struct CompensationScenario {
  std::string id;
  std::string size_id;
  hr::MultiplexingScheme scheme{hr::MultiplexingScheme::kTdm};
  std::string variant{"tdm"};  // tdm | ddm | ddm_nl
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t num_doppler{0};
  int64_t num_range_bins{0};
  int64_t batch{1};
  bool non_linear_ddm{false};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 2, 4, 32, 128, "small MIMO; fast smoke"},
      {"sm", 4, 8, 64, 256, "compact MIMO"},
      {"md", 4, 16, 128, 512, "common dev size"},
      {"lg", 8, 32, 256, 1024, "moderate MIMO"},
      {"xl", 8, 64, 512, 2048, "upper characterization bound"}};
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

hr::FMCWWaveformDescriptor make_waveform(double pri) {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = pri;
  return hr::FMCWWaveformDescriptor(spec);
}

hr::DDMCodeDescriptor make_ddm_code(int64_t num_tx) {
  hr::DDMCodeDescriptor code;
  code.num_tx = num_tx;
  code.doppler_offsets.resize(static_cast<size_t>(num_tx));
  for (int64_t m = 0; m < num_tx; ++m) {
    code.doppler_offsets[static_cast<size_t>(m)] =
        static_cast<double>(m) / static_cast<double>(num_tx);
  }
  code.guard_bands = 0;
  return code;
}

hr::DDMCodeDescriptor make_ddm_nl_code(int64_t num_tx, int64_t num_doppler) {
  hr::DDMCodeDescriptor code;
  code.num_tx = num_tx;
  code.guard_bands = 0;
  code.phase_codes.resize(static_cast<size_t>(num_tx * num_doppler));
  for (int64_t m = 0; m < num_tx; ++m) {
    for (int64_t k = 0; k < num_doppler; ++k) {
      const double phase =
          std::sin(0.7 * static_cast<double>(k) + static_cast<double>(m)) * 2.0;
      code.phase_codes[static_cast<size_t>(m * num_doppler + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }
  return code;
}

// TDM input carries an explicit M_tx axis; DDM / DDM-NL input folds the M TX
// into the Doppler axis, so it has no M_tx axis.
std::vector<int64_t> input_shape(const CompensationScenario& scen) {
  const bool ddm = scen.scheme == hr::MultiplexingScheme::kDdm;
  std::vector<int64_t> shape;
  if (scen.batch > 1) {
    shape.push_back(scen.batch);
  }
  if (!ddm) {
    shape.push_back(scen.num_tx);
  }
  shape.push_back(scen.num_rx);
  shape.push_back(scen.num_doppler);
  shape.push_back(scen.num_range_bins);
  return shape;
}

template <typename T>
class CompensationPerfApp : public holoscan::Application {
 public:
  const CompensationScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  hr::FMCWWaveformDescriptor waveform;
  hr::FMCWWaveformDescriptor override_waveform;
  int64_t override_at{0};
  int total_frames{25};
  bool checksum{false};  // wire the byte-equivalence checksum sink

  hr::MultiplexingScheme scheme{hr::MultiplexingScheme::kTdm};
  hr::DDMCodeDescriptor ddm_code;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::MIMOCoherentCompensationOp<T>>(
        "compensation", Arg("multiplexing_scheme") = scheme,
        Arg("waveform") = waveform, Arg("ddm_code") = ddm_code);

    auto src = make_operator<perf::PerfTensorSourceOp<perf::RdMapsPort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"rd_maps", "rd_maps"}});

    auto sink = make_operator<perf::PerfTerminalSinkOp>("sink");
    sink->configure(collector);
    add_flow(op, sink, {{"rd_maps", "input"}});

    // Byte-equivalence gate: fan the primary output out to a checksum sink too
    // (prints one output_checksum line on the first, pre-override frame).
    if (checksum) {
      auto cs = make_operator<perf::PerfChecksumSinkOp>("checksum_sink");
      add_flow(op, cs, {{"rd_maps", "input"}});
    }

    auto ovr =
        make_operator<perf::PerfOverrideSourceOp<hr::FMCWWaveformDescriptor>>(
            "ovr", make_condition<holoscan::CountCondition>(total_frames));
    ovr->configure(override_at, override_waveform);
    add_flow(ovr, op, {{"output", "waveform"}});
  }
};

void fill_throughput(perf::ResultRow* row, const CompensationScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.batch);
  // Input samples touched: TDM carries M_tx explicitly; DDM folds the M TX
  // into the (already counted) Doppler axis, so it has no M_tx factor.
  const double tx_factor =
      scen.scheme == hr::MultiplexingScheme::kDdm
          ? 1.0
          : static_cast<double>(scen.num_tx);
  const double elems = batch * tx_factor * static_cast<double>(scen.num_rx) *
                       static_cast<double>(scen.num_doppler) *
                       static_cast<double>(scen.num_range_bins);
  row->cells_per_second = elems / sec;
  const double csz = sizeof(std::complex<PerfPrecision>);
  const double bytes = elems * csz;
  row->estimated_effective_gbps = 2.0 * bytes / sec / 1e9;
}

perf::ResultRow base_row(const CompensationScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "MIMOCoherentCompensationOp";
  row.variant = scen.variant;
  row.size_id = scen.size_id;
  row.input_shape = input_shape(scen);
  row.out_batch = scen.batch;
  row.k = scen.num_doppler;
  row.notes = "live_override=waveform";
  return row;
}

CompensationScenario make_scenario(
    const SizeSpec& s,
    hr::MultiplexingScheme scheme = hr::MultiplexingScheme::kTdm,
    int64_t batch = 1, bool non_linear_ddm = false) {
  CompensationScenario sc;
  sc.size_id = s.id;
  sc.scheme = scheme;
  sc.non_linear_ddm = non_linear_ddm;
  if (scheme == hr::MultiplexingScheme::kDdm) {
    sc.variant = non_linear_ddm ? "ddm_nl" : "ddm";
  } else {
    sc.variant = "tdm";
  }
  sc.num_tx = s.num_tx;
  sc.num_rx = s.num_rx;
  sc.num_doppler = s.num_doppler;
  sc.num_range_bins = s.num_range_bins;
  sc.batch = batch;
  std::ostringstream id;
  id << "compensation_" << sc.variant << "_" << s.id;
  if (batch > 1) {
    id << "_b" << batch;
  }
  sc.id = id.str();
  return sc;
}

struct CompensationTraits : perf::PerfDriverDefaults {
  using Scenario = CompensationScenario;

  static const char* tool_name() { return "perf_mimo_coherent_compensation"; }
  static const char* tool_description() {
    return "MIMOCoherentCompensationOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "TDM + DDM (linear) + DDM-NL (correlate) size sweep across the "
           "full ladder";
  }
  static const char* sizes_subset_help() {
    return "TDM + DDM (linear) + DDM-NL size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (sizes + batch axis on md/lg + DDM-NL)";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_tx, s.num_rx, s.num_doppler, s.num_range_bins};
      def.cells = s.num_tx * s.num_rx * s.num_doppler * s.num_range_bins;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<CompensationScenario> build_default_matrix() {
    std::vector<CompensationScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    out.push_back(make_scenario(must_find_size("md"),
                                hr::MultiplexingScheme::kTdm, 4));
    // DDM gather (linear) + correlate (non-linear) across a representative
    // subset.
    for (const char* sid : {"xs", "md", "xl"}) {
      out.push_back(
          make_scenario(must_find_size(sid), hr::MultiplexingScheme::kDdm));
      out.push_back(make_scenario(must_find_size(sid),
                                  hr::MultiplexingScheme::kDdm, 1,
                                  /*non_linear_ddm=*/true));
    }
    return out;
  }

  static std::vector<CompensationScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<CompensationScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(*s));
      out.push_back(make_scenario(*s, hr::MultiplexingScheme::kDdm));
      out.push_back(make_scenario(*s, hr::MultiplexingScheme::kDdm, 1,
                                  /*non_linear_ddm=*/true));
    }
    return out;
  }

  static std::vector<CompensationScenario> build_full_matrix() {
    std::vector<CompensationScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    for (const char* sid : {"md", "lg"}) {
      out.push_back(make_scenario(must_find_size(sid),
                                  hr::MultiplexingScheme::kTdm, 4));
    }
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s, hr::MultiplexingScheme::kDdm));
      out.push_back(make_scenario(s, hr::MultiplexingScheme::kDdm, 1,
                                  /*non_linear_ddm=*/true));
    }
    return out;
  }

  static void run_scenario(const CompensationScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(input_shape(scen)),
              "rd_maps");

          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<CompensationPerfApp<
              PerfPrecision>>();
          app->scen = &scen;
          app->collector = &collector;
          app->frame = std::move(cube);
          app->scheme = scen.scheme;
          app->ddm_code =
              scen.non_linear_ddm
                  ? make_ddm_nl_code(scen.num_tx, scen.num_doppler)
                  : make_ddm_code(scen.num_tx);
          app->waveform = make_waveform(kDefaultPri);
          app->override_waveform = make_waveform(kOverridePri);
          app->override_at = p.override_at;
          app->total_frames = p.total_frames;
          app->checksum = rs.checksum;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    if (scen.non_linear_ddm) {
      row.notes =
          "DDM non-linear phase_codes conjugate-correlation demux "
          "(waveform-independent); cold_ms includes initial corr-weights "
          "build; median/p90 are steady state; [live] waveform override is "
          "inert; cells/s counts input complex samples read; output keeps "
          "full Doppler axis";
    } else if (scen.scheme == hr::MultiplexingScheme::kDdm) {
      row.notes =
          "DDM per-band Doppler gather (waveform-independent); cold_ms is "
          "first-frame latency including initial demux-table build; "
          "median/p90 are steady state; the [live] waveform override is "
          "inert for DDM; cells/s counts input complex samples read";
    } else {
      row.notes =
          "TDM per-bin complex multiply; cold_ms is first-frame latency "
          "including initial correction-table build; median/p90 are "
          "pre-override steady state; override_ms is peak latency over the "
          "3-frame live waveform rebuild window; cells/s counts complex "
          "samples touched";
    }
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs")), warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CompensationTraits>(argc, argv);
}
