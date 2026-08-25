// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for BackprojectionOp. Sweeps pulses x pixels
// and emits one row per implemented backend. Every tick is emit-and-reset.
// Each compute row times a mid-session [live] override: sample_geometry
// by default (a reset; expect ovr_ms ~= cold_ms) and, on one default-
// matrix row, reference_frequency (LUT refill). interpolation has no
// other implemented value; backend is static.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
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

struct PositionsPort {
  static constexpr const char* kName = "positions";
};
struct ReferencePort {
  static constexpr const char* kName = "reference_position";
};
struct ControlPort {
  static constexpr const char* kName = "control";
};

struct SizeSpec {
  std::string id;
  int64_t num_pulses;
  int64_t num_v;
  int64_t num_u;
  int64_t num_bins;
  std::string role;
};

enum class OverrideKind { kGeometry, kFrequency };

struct BpScenario {
  std::string id;
  std::string size_id;
  int64_t num_pulses{0};
  int64_t num_v{0};
  int64_t num_u{0};
  int64_t num_bins{0};
  hr::BackprojectionBackend backend{hr::kDefaultBackprojectionBackend};
  OverrideKind override_kind{OverrideKind::kGeometry};
  hr::SampleGeometry geometry;
  double range_origin{0.0};
  double range_spacing{0.25};
  double altitude{0.0};
  double aperture{0.0};
};

const std::vector<SizeSpec>& size_ladder() {
  // pulses = num_v, nfft = num_u. The cube is square with the image.
  static const std::vector<SizeSpec> ladder = {
      {"xs", 32, 32, 32, 32, "1k pixels; smoke"},
      {"sm", 64, 64, 64, 64, "4k pixels"},
      {"md", 128, 128, 128, 128, "16k pixels"},
      {"lg", 256, 256, 256, 256, "65k pixels"},
      {"xl", 512, 512, 512, 512, "262k pixels"},
      {"2k", 2048, 2048, 2048, 2048, "4M pixels; Houston-scale grid"},
  };
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

std::string backend_tag(hr::BackprojectionBackend b) {
  return std::string(hr::to_string(b));
}

hr::CoordinateFrame scene_frame() {
  hr::CoordinateFrameSpec spec;
  spec.frame = hr::CoordinateFrameType::kSceneLocal;
  return hr::CoordinateFrame(spec);
}

// Stand-off spotlight: lattice on z=0, SRP at the origin, APC flies a
// short x-track at altitude H. range_origin / range_spacing cover the
// analytic [min, max] differential range so every pixel-pulse interpolates
// (the previous CFAR-filler positions left R0=0 and an empty swath).
void fill_swath(BpScenario* scen) {
  const double su = 1.0;
  const double sv = 1.0;
  const double Lu = 0.5 * static_cast<double>(scen->num_u - 1) * su;
  const double Lv = 0.5 * static_cast<double>(scen->num_v - 1) * sv;
  hr::SampleGeometrySpec spec;
  spec.origin = {-Lu, -Lv, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {sv, su};
  spec.counts = {scen->num_v, scen->num_u};
  scen->geometry = hr::SampleGeometry(spec);

  const double H = 4.0 * std::hypot(Lu, Lv);
  const double A = Lv;
  const double half_A = (scen->num_pulses > 1) ? 0.5 * A : 0.0;
  // Conservative bounds: R0 >= H, farthest R = hypot(Lu+A/2, Lv, H).
  const double dR_min = H - std::hypot(half_A, H);
  const double dR_max = std::hypot(Lu + half_A, Lv, H) - H;
  const double pad = 0.01 * std::max(dR_max - dR_min, 1e-6);
  const double denom = static_cast<double>(
      std::max(scen->num_bins - 2, int64_t{1}));
  scen->altitude = H;
  scen->aperture = A;
  scen->range_origin = dR_min - pad;
  scen->range_spacing = (dR_max - dR_min + 2.0 * pad) / denom;
}

holoscan::TensorMap swath_xyz_map(const BpScenario& scen, bool reference) {
  const int64_t Np = scen.num_pulses;
  std::vector<PerfPrecision> host(static_cast<size_t>(Np * 3),
                                  static_cast<PerfPrecision>(0));
  if (!reference) {
    for (int64_t p = 0; p < Np; ++p) {
      const double t =
          (Np > 1)
              ? (static_cast<double>(p) / static_cast<double>(Np - 1) -
                 0.5) *
                    scen.aperture
              : 0.0;
      host[static_cast<size_t>(p * 3 + 0)] =
          static_cast<PerfPrecision>(t);
      host[static_cast<size_t>(p * 3 + 2)] =
          static_cast<PerfPrecision>(scen.altitude);
    }
  }
  auto tensor = holoscan::radar::make_device_tensor(
      {Np, 3}, holoscan::radar::real_dtype<PerfPrecision>());
  if (cudaMemcpy(tensor->data(), host.data(),
                 sizeof(PerfPrecision) * host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("perf_backprojection: positions H2D");
  }
  return perf::single_map(std::move(tensor),
                          reference ? "reference_position" : "positions");
}

// Same counts / spacing / axes as fill_swath, origin shifted one pixel
// along +u so the override is a real new lattice still inside the swath.
hr::SampleGeometry shifted_geometry(const BpScenario& scen) {
  const double su = 1.0;
  const double sv = 1.0;
  const double Lu = 0.5 * static_cast<double>(scen.num_u - 1) * su;
  const double Lv = 0.5 * static_cast<double>(scen.num_v - 1) * sv;
  hr::SampleGeometrySpec spec;
  spec.origin = {-Lu + su, -Lv, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {sv, su};
  spec.counts = {scen.num_v, scen.num_u};
  return hr::SampleGeometry(spec);
}

class BpPerfApp : public holoscan::Application {
 public:
  const BpScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap profiles;
  holoscan::TensorMap positions;
  holoscan::TensorMap reference;
  holoscan::TensorMap control;
  OverrideKind override_kind{OverrideKind::kGeometry};
  hr::SampleGeometry override_geometry;
  double override_frequency{9.7e9};
  int64_t override_at{0};
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::BackprojectionOp<PerfPrecision>>(
        "bp", Arg("coordinate_frame") = scene_frame(),
        Arg("sample_geometry") = scen->geometry,
        Arg("range_bin_spacing") = scen->range_spacing,
        Arg("range_bin_origin") = scen->range_origin,
        Arg("reference_frequency") = 9.6e9,
        Arg("backend") = scen->backend);

    auto src = make_operator<
        perf::PerfTensorSourceOp<perf::RangeProfilesPort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, profiles);
    add_flow(src, op, {{"range_profiles", "range_profiles"}});

    auto pos = make_operator<perf::PerfTensorSourceOp<PositionsPort>>(
        "pos", make_condition<holoscan::CountCondition>(total_frames));
    pos->configure(nullptr, positions);
    add_flow(pos, op, {{"positions", "positions"}});

    auto ref = make_operator<perf::PerfTensorSourceOp<ReferencePort>>(
        "ref", make_condition<holoscan::CountCondition>(total_frames));
    ref->configure(nullptr, reference);
    add_flow(ref, op, {{"reference_position", "reference_position"}});

    auto ctl = make_operator<perf::PerfTensorSourceOp<ControlPort>>(
        "ctl", make_condition<holoscan::CountCondition>(total_frames));
    ctl->configure(nullptr, control);
    add_flow(ctl, op, {{"control", "control"}});

    auto term = make_operator<perf::PerfTerminalSinkOp>("out_sink");
    term->configure(collector);
    add_flow(op, term, {{"image", "input"}});

    // emit-and-reset every tick makes a geometry replace legal. Frequency
    // is a LUT refill and does not need a resetting control.
    auto count = make_condition<holoscan::CountCondition>(total_frames);
    if (override_kind == OverrideKind::kGeometry) {
      auto ovr =
          make_operator<perf::PerfOverrideSourceOp<hr::SampleGeometry>>(
              "geo_ovr", count);
      ovr->configure(override_at, override_geometry);
      add_flow(ovr, op, {{"output", "sample_geometry"}});
    } else {
      auto ovr = make_operator<perf::PerfOverrideSourceOp<double>>(
          "freq_ovr", count);
      ovr->configure(override_at, override_frequency);
      add_flow(ovr, op, {{"output", "reference_frequency"}});
    }
  }
};

void fill_throughput(perf::ResultRow* row, const BpScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double pulses = static_cast<double>(scen.num_pulses);
  const double pixels = static_cast<double>(scen.num_v * scen.num_u);
  row->cells_per_second = pulses * pixels / sec;
  row->slices_per_second = pixels / sec;
  const double bytes =
      pulses * static_cast<double>(scen.num_bins) * 8.0 + pixels * 8.0;
  row->estimated_effective_gbps = bytes / sec / 1e9;
}

perf::ResultRow base_row(const BpScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "BackprojectionOp";
  row.variant = backend_tag(scen.backend);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_v, scen.num_u};
  // Printed "shape" is the cells/s product: pulses x pixels.
  row.input_shape = {scen.num_pulses, scen.num_v, scen.num_u};
  row.k = scen.num_pulses;
  if (scen.override_kind == OverrideKind::kGeometry) {
    row.notes = "live_override=sample_geometry (one-pixel +u; ~cold)";
  } else {
    row.notes = "live_override=reference_frequency (LUT refill)";
  }
  return row;
}

BpScenario make_scenario(hr::BackprojectionBackend backend,
                         const SizeSpec& s,
                         OverrideKind kind = OverrideKind::kGeometry) {
  BpScenario sc;
  sc.size_id = s.id;
  sc.num_pulses = s.num_pulses;
  sc.num_v = s.num_v;
  sc.num_u = s.num_u;
  sc.num_bins = s.num_bins;
  sc.backend = backend;
  sc.override_kind = kind;
  fill_swath(&sc);
  std::ostringstream id;
  id << "bp_" << backend_tag(backend) << "_" << s.id;
  if (kind == OverrideKind::kFrequency) {
    id << "_freq";
  }
  sc.id = id.str();
  return sc;
}

const std::vector<hr::BackprojectionBackend>& implemented_backends() {
  static const std::vector<hr::BackprojectionBackend> backends = {
      hr::kDefaultBackprojectionBackend,
      hr::BackprojectionBackend::kNvradarBlockAnchoredRefinedThroughput,
      hr::BackprojectionBackend::kNvradarFp64Reference,
      hr::BackprojectionBackend::kNvradarCudaFp32,
      hr::BackprojectionBackend::kNvradarCudaFp32Hq,
      hr::BackprojectionBackend::kNvradarCudaFp32HqFast,
      hr::BackprojectionBackend::kNvradarBlockAnchored,
      hr::BackprojectionBackend::kNvradarBlockAnchoredRefinedQuality,
      hr::BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput,
      hr::BackprojectionBackend::kNvradarBlockAnchoredFp64Trajectory,
      hr::BackprojectionBackend::kNvradarBlockAnchoredExpandedQuality,
      hr::BackprojectionBackend::kNvradarBlockAnchoredPhaseLut,
      hr::BackprojectionBackend::kNvradarBlockAnchoredRefinedQualityPhaseLut,
      hr::BackprojectionBackend::
          kNvradarBlockAnchoredExpandedThroughputPhaseLut,
      hr::BackprojectionBackend::
          kNvradarBlockAnchoredExpandedThroughputPhaseLutVec,
      hr::BackprojectionBackend::
          kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0,
      hr::BackprojectionBackend::kNvradarBlockAnchoredExpandedQualityPhaseLut,
      hr::BackprojectionBackend::kNvradarBlockAnchoredPhaseLutLegacy,
      hr::BackprojectionBackend::kNvradarBlockAnchoredFp64TrajectoryPhaseLut,
      // 4.3 MatX 0.9.4 has no matx/kernels/sar_bp.cuh; the operator
      // rejects every matx_* backend. Do not list them here or --full
      // aborts mid-sweep.
  };
  return backends;
}

struct BpTraits : perf::PerfDriverDefaults {
  using Scenario = BpScenario;

  static const char* tool_name() { return "perf_backprojection"; }
  static const char* tool_description() {
    return "BackprojectionOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Implemented-backend size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Implemented-backend size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Implemented backends x sizes";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_pulses, s.num_v, s.num_u};
      def.cells = s.num_pulses * s.num_v * s.num_u;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<BpScenario> build_default_matrix() {
    std::vector<BpScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl", "2k"}) {
      out.push_back(make_scenario(
          hr::kDefaultBackprojectionBackend, must_find_size(sid)));
    }
    out.push_back(make_scenario(
        hr::BackprojectionBackend::kNvradarFp64Reference,
        must_find_size("md")));
    // LUT refill, not a reset. Distinct from the geometry rows above.
    out.push_back(make_scenario(hr::kDefaultBackprojectionBackend,
                               must_find_size("md"),
                               OverrideKind::kFrequency));
    return out;
  }

  static std::vector<BpScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<BpScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      for (auto b : implemented_backends()) {
        out.push_back(make_scenario(b, *s));
      }
    }
    return out;
  }

  static std::vector<BpScenario> build_full_matrix() {
    std::vector<BpScenario> out;
    for (const auto& s : size_ladder()) {
      for (auto b : implemented_backends()) {
        out.push_back(make_scenario(b, s));
      }
    }
    return out;
  }

  static void run_scenario(const BpScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          auto profiles = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(
                  {scen.num_pulses, scen.num_bins}),
              "range_profiles");
          auto positions = swath_xyz_map(scen, false);
          auto reference = swath_xyz_map(scen, true);
          uint8_t ctl = static_cast<uint8_t>(
              hr::FormationControl::kEmitAndReset);
          auto ctl_tensor = holoscan::radar::make_device_tensor(
              {1}, DLDataType{kDLUInt, 8, 1});
          if (cudaMemcpy(ctl_tensor->data(), &ctl, 1,
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("perf_backprojection: control H2D");
          }
          auto control = perf::single_map(std::move(ctl_tensor), "control");

          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<BpPerfApp>();
          app->scen = &scen;
          app->collector = &collector;
          app->profiles = std::move(profiles);
          app->positions = std::move(positions);
          app->reference = std::move(reference);
          app->control = std::move(control);
          app->override_kind = scen.override_kind;
          app->override_geometry = shifted_geometry(scen);
          app->override_frequency = 9.7e9;
          app->override_at = p.override_at;
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);
    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    report->results.push_back(row);
  }

  static void post_run(perf::PerfReport* report) {
    report->footnotes.push_back(
        "cells/s is pixels * pulses per second (the backprojection "
        "work product). The shape column is (pulses, num_v, num_u).");
    report->footnotes.push_back(
        "Default and --full rows override sample_geometry (origin +1 "
        "pixel in u; same counts). That is a reset: expect ovr_ms ~= "
        "cold_ms. ovr_d_ms is reset-minus-kernel, not a sticky resume. "
        "The default-matrix *_freq row overrides reference_frequency "
        "(LUT refill only). interpolation has no other implemented "
        "value this slice.");
    report->footnotes.push_back(
        "Each size is pulses = num_v and nfft = num_u. The lattice is "
        "centred under a stand-off APC; range_origin / spacing cover "
        "the analytic dR span so every pixel interpolates.");
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(
          make_scenario(
              hr::kDefaultBackprojectionBackend,
              must_find_size("xs")),
          warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<BpTraits>(argc, argv);
}
