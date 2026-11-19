// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/graph.hpp>

#include "radar_app/report/report_ops.hpp"
#include "radar_app/sar/coordinate_ops.hpp"
#include "radar_app/sar/graph_driver.hpp"
#include "aux_points_op.hpp"        // NOLINT(build/include_subdir)
#include "config.hpp"               // NOLINT(build/include_subdir)
#include "phase_resample_op.hpp"    // NOLINT(build/include_subdir)
#include "position_error_op.hpp"    // NOLINT(build/include_subdir)
#include "sharpness_report_op.hpp"  // NOLINT(build/include_subdir)
#include "sar/autofocus_op/autofocus_op.hpp"
#include "sar/phase_correction_op/phase_correction_op.hpp"

namespace app = holoscan::radar::app;
namespace radar = holoscan::radar;

int main(int argc, char** argv) {
  auto extras = std::make_shared<AutofocusExtras>();
  auto load = [extras](const std::filesystem::path& path,
                       const std::filesystem::path& cphd_override) {
    return load_autofocus_config(path, cphd_override, extras.get());
  };
  auto build = [extras](auto precision, auto& graph, auto& options, const auto& queue,
                        const auto& cfg, auto& reader, auto& range, auto& manager, auto& bp,
                        auto& source, auto& application, auto& formation, const auto& extents,
                        const auto& hud) -> const holoscan::Output<holoscan::Tensor>* {
    // The auxiliary chain forms at the driver's precisions, not a fixed
    // double: the whole point of a float-sample run is that the replay it
    // consumes is complex64.
    using T = typename decltype(precision)::Sample;
    using TGeom = typename decltype(precision)::Geometry;
    // The driver normalizes the enum spellings in the blocks it decodes;
    // these are this app's own blocks, plus the /backprojection fields
    // again because the auxiliary formation decodes that block a second
    // time from a separate document.
    auto document = app::config::load_inherited_document(extras->config_path);
    document = app::config::normalize_enum_field<radar::BackprojectionBackend>(
        std::move(document), "/backprojection/backend");
    document = app::config::normalize_enum_field<radar::GeometryChange>(
        std::move(document), "/backprojection/geometry_change");
    document = app::config::normalize_enum_field<radar::RangeInterpolation>(
        std::move(document), "/backprojection/interpolation");
    document = app::config::normalize_enum_field<radar::AutofocusAlgorithm>(std::move(document),
                                                                            "/autofocus/algorithm");
    document = app::config::normalize_enum_field<radar::AutofocusAzimuthAxis>(
        std::move(document), "/autofocus/azimuth_axis");
    document = app::config::normalize_enum_field<radar::PhaseCorrectionSense>(
        std::move(document), "/phase_correction/sense");

    // PreparedParams are immutable, so everything app-derived goes into
    // the seed and the YAML block layers on top.
    radar::PhaseCorrectionParams pc_seed;
    pc_seed.sense = radar::PhaseCorrectionSense::kSubtract;
    auto pc_params = app::config::decode_params(document, "/phase_correction", pc_seed);
    auto pc = graph.template op<radar::PhaseCorrectionOp<T>>("phase_correction", extents);
    pc.set_params(std::move(pc_params));
    bind_outputs(options, "phase_correction", {"range_profiles_out"});

    // A reform replay emits the whole retained aperture as one tensor, so
    // every edge carrying it is 9.4 GiB per slot on Kourou. The shared
    // depth-4 `queue` reserves four of those plus the pool's own slack --
    // 34.5 GiB on one edge -- while exactly one replay is ever in flight.
    const holoscan::ConnectionOptions replay{.queue_depth = 1U};

    // --- ingest, optionally through the known lie
    graph.add_flow(reader->stream_boundary, manager->stream_boundary, queue);
    // Everything on the ingest edge carries one pulses_per_tick batch, not
    // the retained aperture, and must be sized accordingly: at Kourou's
    // 23k pulses the difference is 17.45 GiB per pool.
    const radar::MaxTensorExtents ingest_extents = make_ingest_extents(extents, cfg);
    const auto& pe = extras->position_error;
    hud->enable_autofocus(pe.oracle);
    HOLOSCAN_LOG_INFO(
        "autofocus: {} over {} pulses; correction={}",
        pe.enabled ? "injected-error validation" : "input-data run (no injected error)",
        extras->aperture_pulses, pe.oracle ? "oracle" : "PGA");

    // Positions reach the manager in the scene frame, exactly as in every
    // sibling SAR app. This app needs the transform more than they do, not
    // less: `config.hpp` computes the look axis and the auxiliary lattice in
    // scene ENU whenever `transform_to_scene_enu` is set, so leaving the
    // runtime positions in ECEF puts the host-side geometry and the data an
    // Earth radius apart. All four smoke fixtures set `enu_origin_ecef`.
    const holoscan::Output<holoscan::Tensor>* src_positions = &reader->positions;
    const holoscan::Output<holoscan::Tensor>* src_reference = &reader->reference_position;
    if (cfg.transform_to_scene_enu) {
      auto enu = graph.template op<app::sar::EcefToSceneEnuOp>("ecef_to_enu", cfg.enu_origin_ecef,
                                                               cfg.quantize_phase_center_f32,
                                                               cfg.reader_pulses_per_tick);
      source.add(enu);
      bind_outputs(options, "ecef_to_enu", {"positions_out", "reference_position_out"});
      graph.add_flow(reader->positions, enu->positions, queue);
      graph.add_flow(reader->reference_position, enu->reference_position, queue);
      src_positions = &enu->positions_out;
      src_reference = &enu->reference_position_out;
    }

    const holoscan::Output<holoscan::Tensor>* ingest_profiles = &range->range_profiles;
    std::shared_ptr<app::sar::LieRecord> lies;
    if (pe.enabled) {
      HOLOSCAN_LOG_INFO(
          "autofocus: look-axis position_error over {} pulses "
          "(profile phase; dR cancels LOS APC)",
          pe.num_pulses);
      lies = std::make_shared<app::sar::LieRecord>();
      auto lie = graph.template op<app::sar::PerturbPositionsOp<T, TGeom>>(
          "perturb_positions", lies, pe.look, pe.profile, pe.num_pulses, cfg.reference_frequency,
          ingest_extents);
      bind_outputs(options, "perturb_positions",
                   {"range_profiles_out", "positions_out", "reference_position_out"});
      graph.add_flow(range->range_profiles, lie->range_profiles, queue);
      graph.add_flow(*src_positions, lie->positions, queue);
      graph.add_flow(*src_reference, lie->reference_position, queue);
      graph.add_flow(lie->positions_out, manager->positions, queue);
      graph.add_flow(lie->reference_position_out, manager->reference_position, queue);
      // The liar sits between range compression and the manager, so it
      // belongs on the chain it was spliced into: leaving `application` as
      // two disconnected nodes fails the exact-cover check.
      application.add(lie);
      ingest_profiles = &lie->range_profiles_out;
    } else {
      graph.add_flow(*src_positions, manager->positions, queue);
      graph.add_flow(*src_reference, manager->reference_position, queue);
    }

    auto estimate = graph.partition("estimate");
    if (pe.oracle) {
      // Skip the estimator and feed PhaseCorrectionOp the exact phase the
      // lie implies. If the product comes back focused, the whole
      // correction path is sound and any shortfall in a full run belongs
      // to AutofocusOp rather than to the plumbing around it.
      HOLOSCAN_LOG_INFO(
          "autofocus: oracle leftover -> PhaseCorrectionOp "
          "(skipping AutofocusOp; positions stay lied)");
      auto oracle = graph.template op<app::sar::OraclePhaseOp<T>>("oracle_phase", lies, extents);
      bind_outputs(options, "oracle_phase", {"phase_correction"});
      graph.add_flow(manager->range_profiles_out, oracle->range_profiles, replay);
      graph.add_flow(oracle->phase_correction, pc->phase_correction, queue);
      estimate.add(oracle);
    } else {
      // bp is the product formation: the driver already seeded it from
      // /backprojection with the product lattice. The auxiliary formation
      // decodes the same block so the two agree on coordinate_frame and
      // backend, then replaces only the lattice.
      radar::BackprojectionParams aux_seed;
      aux_seed.reference_frequency = cfg.reference_frequency;
      aux_seed.range_bin_spacing = cfg.range_bin_spacing;
      // A polar aux carries no SampleGeometry: it arrives on sample_points.
      if (!extras->aux_polar) {
        aux_seed.sample_geometry = extras->aux_sample_geometry;
      }
      auto aux_params = app::config::decode_params(document, "/backprojection", aux_seed);

      // The aux image is the estimation grid, so the pools must be sized
      // for it as well as for the product.
      radar::MaxTensorExtents aux_extents = extents;
      aux_extents.max_image_v = std::max(*extents.max_image_v, extras->aux_num_v());
      aux_extents.max_image_u = std::max(*extents.max_image_u, extras->aux_num_u());

      auto aux_bp = graph.template op<radar::BackprojectionOp<T, TGeom>>("aux_bp", aux_extents);
      aux_bp.set_params(std::move(aux_params));
      bind_outputs(options, "aux_bp", {"image"});

      // A polar aux is not a SampleGeometry, so it rides sample_points.
      // The source sits in series on the replay edge so the lattice and the
      // profiles leave one producer on one tick; a trigger-only tap races
      // the formation and loses -- see aux_points_op.hpp.
      if (extras->aux_polar) {
        auto aux_points =
            graph.template op<app::sar::AuxPointsSourceOp>("aux_points", extras->aux_polar_lattice);
        bind_outputs(options, "aux_points", {"range_profiles_out", "sample_points"});
        graph.add_flow(manager->range_profiles_out, aux_points->range_profiles, replay);
        graph.add_flow(aux_points->range_profiles_out, aux_bp->range_profiles, replay);
        graph.add_flow(aux_points->sample_points, aux_bp->sample_points, replay);
        estimate.add(aux_points);
      } else {
        graph.add_flow(manager->range_profiles_out, aux_bp->range_profiles, replay);
      }

      // PGA's first window is a fraction of the azimuth extent. On a
      // Nyquist aux that extent is the oversampled column count, not the
      // pulse count, so a fraction tuned for one is wrong for the other;
      // target a fixed ~64 columns unless the YAML says otherwise.
      radar::AutofocusParams af_seed;
      af_seed.azimuth_axis = radar::AutofocusAzimuthAxis::kU;
      if (!extras->window_fraction_in_yaml) {
        const double n_u = static_cast<double>(std::max(extras->aux_num_u(), int64_t{2}));
        af_seed.window_fraction = std::min(0.5, 64.0 / n_u);
        HOLOSCAN_LOG_INFO(
            "autofocus: window_fraction {:.5f} (first window ~{} of {} u)", af_seed.window_fraction,
            static_cast<int64_t>(std::llround(af_seed.window_fraction * n_u)), extras->aux_num_u());
      }
      auto decoded_af = app::config::decode_params(document, "/autofocus", af_seed);
      auto af = graph.template op<radar::AutofocusOp<T>>("autofocus", aux_extents);
      af.set_params(std::move(decoded_af));
      // Every declared tensor output needs a device placement before the
      // graph will compile, wired or not.
      bind_outputs(options, "autofocus",
                   {"image_out", "phase_correction", "sharpness", "termination", "iterations"});
      // All three reports are host-side; each is one scalar or a tiny trace.

      auto resample =
          graph.template op<app::sar::PhaseResampleOp>("phase_resample", extras->aperture_pulses);
      bind_outputs(options, "phase_resample", {"phase_correction_out"});

      // Figure 2 wants the pulse-indexed estimate, not the in-image
      // correction, but image_out is emitted every tick regardless and so
      // needs a consumer.
      auto af_drain = graph.template op<app::NullSinkOp>("af_image_drain");

      // sharpness needs a real consumer, not a drain: an unwired optional
      // output makes the operator skip the work behind it for the whole
      // run. The same report takes termination and iterations, which say
      // whether a flat trace was PGA refusing or PGA having nothing to do --
      // the one thing sharpness cannot distinguish.
      auto sharp =
          graph.template op<app::sar::AutofocusSharpnessReportOp>("autofocus_sharpness", hud);

      graph.add_flow(manager->positions_out, aux_bp->positions, queue);
      graph.add_flow(manager->reference_position_out, aux_bp->reference_position, queue);
      graph.add_flow(manager->control_out, aux_bp->control, queue);
      if (cfg.wire_weight) {
        graph.add_flow(manager->weight, aux_bp->weight, queue);
      }
      graph.add_flow(aux_bp->image, af->image, queue);
      graph.add_flow(af->image_out, af_drain->input, queue);
      graph.add_flow(af->sharpness, sharp->sharpness, queue);
      graph.add_flow(af->termination, sharp->termination, queue);
      graph.add_flow(af->iterations, sharp->iterations, queue);
      graph.add_flow(af->phase_correction, resample->phase_correction, queue);
      graph.add_flow(resample->phase_correction_out, pc->phase_correction, queue);
      estimate.add(aux_bp).add(af).add(resample).add(af_drain);
      // Each authored partition is one directed chain, and `estimate`
      // already ends at the resampler, so the sharpness tap is its own.
      graph.partition("sharpness").add(sharp);

      if (lies) {
        // Score the resampled, pulse-indexed estimate rather than the raw
        // one, because that is the vector the correction actually applies.
        std::string dump_dir;
        if (cfg.dump && !cfg.dump_path.empty()) {
          dump_dir = std::filesystem::path(cfg.dump_path).parent_path().string();
          if (dump_dir.empty()) {
            dump_dir = ".";
          }
        }
        auto score = graph.template op<app::sar::AutofocusScoreOp>(
            "autofocus_score", lies, pe.warn_rms_rad, cfg.reference_frequency, dump_dir);
        graph.add_flow(resample->phase_correction_out, score->phase_correction, queue);
        graph.partition("score").add(score);
      }
    }

    // The ingest chain is complete only once the estimation branch has had
    // its chance to splice the lattice source onto it.
    // The shared depth-4 `queue`, not `replay`: this edge carries one
    // `pulses_per_tick` batch into the manager, not the retained aperture
    // that `replay` exists to keep to a single slot. The hop immediately
    // before it is already wired with `queue`, and every sibling app wires
    // this one the same way.
    graph.add_flow(*ingest_profiles, manager->range_profiles, queue);

    // --- correct the same replayed pulses and form the product
    graph.add_flow(manager->range_profiles_out, pc->range_profiles_in, replay);
    graph.add_flow(pc->range_profiles_out, bp->range_profiles, replay);
    graph.add_flow(manager->positions_out, bp->positions, queue);
    graph.add_flow(manager->reference_position_out, bp->reference_position, queue);
    graph.add_flow(manager->control_out, bp->control, queue);
    if (cfg.wire_weight) {
      graph.add_flow(manager->weight, bp->weight, queue);
    }

    // Each authored partition must be a single directed chain, so the
    // correction sits apart from the product formation it feeds.
    graph.partition("correction").add(pc);
    return &bp->image;
  };
  return run_sar_graph_app(argc, argv, "autofocus", load, build);
}
