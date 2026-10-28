// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned extras for the autofocus app. The product lattice, CLI, dump and
// Holoviz flags come from the shared load_sar_config. This file adds the
// auxiliary conjugate lattice and the position-error diagnostic, and refuses
// formation shapes that are not Figure 2 (tiles, scroll, multi-collection,
// pass-through manager). Operator blocks still flow through from_config().
//
// load_sar_config returns a plain AppConfig, which is what the shared graph
// driver expects, so these extras travel beside it in AutofocusExtras: main
// owns one and hands it to both the load and the graph-building lambda.

#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "radar_app/sar/config.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace radar = holoscan::radar;

namespace af_detail {

inline std::array<double, 3> vec_sub(const std::array<double, 3>& a,
                                     const std::array<double, 3>& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline std::array<double, 3> vec_add(const std::array<double, 3>& a,
                                     const std::array<double, 3>& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline std::array<double, 3> vec_scale(const std::array<double, 3>& a, double s) {
  return {a[0] * s, a[1] * s, a[2] * s};
}

inline double vec_norm(const std::array<double, 3>& a) {
  return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline double vec_dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline std::array<double, 3> vec_cross(const std::array<double, 3>& a,
                                       const std::array<double, 3>& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline std::array<double, 3> ecef_to_scene_enu(const std::array<double, 3>& ecef,
                                               const std::array<double, 3>& origin) {
  constexpr double kA = 6378137.0;
  constexpr double kF = 1.0 / 298.257223563;
  const double b = kA * (1.0 - kF);
  const double e2 = 1.0 - (b * b) / (kA * kA);
  const double lon = std::atan2(origin[1], origin[0]);
  const double p = std::hypot(origin[0], origin[1]);
  const double theta = std::atan2(origin[2] * kA, p * b);
  const double slt = std::sin(theta);
  const double clt = std::cos(theta);
  const double ep2 = (kA * kA - b * b) / (b * b);
  const double lat =
      std::atan2(origin[2] + ep2 * b * slt * slt * slt, p - e2 * kA * clt * clt * clt);
  const double sl = std::sin(lat);
  const double cl = std::cos(lat);
  const double so = std::sin(lon);
  const double co = std::cos(lon);
  const double dx = ecef[0] - origin[0];
  const double dy = ecef[1] - origin[1];
  const double dz = ecef[2] - origin[2];
  return {-so * dx + co * dy, -sl * co * dx - sl * so * dy + cl * dz,
          cl * co * dx + cl * so * dy + sl * dz};
}

inline std::array<double, 3> read_xyz(const radar::cphd::CphdFile& file, const char* field,
                                      int64_t pulse) {
  const auto v = file.read_pvp(0, field, pulse, 1, 3);
  if (v.size() < 3) {
    throw std::runtime_error(std::string("PVP ") + field + " is not 3-vector");
  }
  return {v[0], v[1], v[2]};
}

// The slant-plane frame both aux lattices are built in: mid-aperture phase
// centre, the look to the SRP, and the in-plane axis perp to it (track with
// look removed). rho_az is the cross-range resolution the aperture supports
// at the SRP range.
struct AuxFrame {
  std::array<double, 3> apc_mid{};
  std::array<double, 3> srp{};
  std::array<double, 3> look_hat{};
  std::array<double, 3> uhat{};
  double look_n{0.0};
  double l_perp{0.0};
  double rho_az{0.0};
};

inline AuxFrame read_aux_frame(const AppConfig& cfg, int64_t first_pulse, int64_t last_pulse) {
  if (last_pulse - first_pulse < 2) {
    throw std::runtime_error("aux lattice needs at least 2 pulses in the aperture");
  }
  if (!(cfg.reference_frequency > 0.0) || !std::isfinite(cfg.reference_frequency)) {
    throw std::runtime_error("aux lattice derivation needs reference_frequency > 0");
  }
  radar::cphd::CphdFile file(cfg.cphd_path);
  if (!file.has_pvp_field("TxPos") || !file.has_pvp_field("SRPPos")) {
    throw std::runtime_error("aux lattice derivation needs PVP TxPos and SRPPos");
  }
  const int64_t last = last_pulse - 1;
  const int64_t mid = first_pulse + (last_pulse - first_pulse) / 2;
  auto apc0 = read_xyz(file, "TxPos", first_pulse);
  auto apc1 = read_xyz(file, "TxPos", last);
  AuxFrame f;
  f.apc_mid = read_xyz(file, "TxPos", mid);
  f.srp = read_xyz(file, "SRPPos", mid);
  if (cfg.transform_to_scene_enu) {
    apc0 = ecef_to_scene_enu(apc0, cfg.enu_origin_ecef);
    apc1 = ecef_to_scene_enu(apc1, cfg.enu_origin_ecef);
    f.apc_mid = ecef_to_scene_enu(f.apc_mid, cfg.enu_origin_ecef);
    f.srp = ecef_to_scene_enu(f.srp, cfg.enu_origin_ecef);
  }
  const auto track = vec_sub(apc1, apc0);
  if (!(vec_norm(track) > 0.0)) {
    throw std::runtime_error("aux lattice: first and last TxPos are coincident");
  }
  const auto look = vec_sub(f.srp, f.apc_mid);
  f.look_n = vec_norm(look);
  if (!(f.look_n > 0.0)) {
    throw std::runtime_error("aux lattice: SRPPos coincides with mid-aperture TxPos");
  }
  f.look_hat = vec_scale(look, 1.0 / f.look_n);
  const double along_look = vec_dot(track, f.look_hat);
  auto uvec = vec_sub(track, vec_scale(f.look_hat, along_look));
  f.l_perp = vec_norm(uvec);
  if (!(f.l_perp > 0.0)) {
    throw std::runtime_error("aux lattice: track is parallel to look (no along-track baseline)");
  }
  f.uhat = vec_scale(uvec, 1.0 / f.l_perp);
  const double lambda = radar::kSpeedOfLight / cfg.reference_frequency;
  f.rho_az = lambda * f.look_n / (2.0 * f.l_perp);
  if (!(f.rho_az > 0.0) || !std::isfinite(f.rho_az)) {
    throw std::runtime_error("aux lattice: rho_az is not finite and > 0");
  }
  return f;
}

inline std::pair<int64_t, int64_t> pulse_range_from_yaml(const YAML::Node& root,
                                                         const std::string& cphd_path) {
  radar::cphd::CphdFile file(cphd_path);
  int64_t first = 0;
  int64_t last = file.channel(0).num_vectors;
  if (const YAML::Node rd = root["cphd_reader"]) {
    if (const YAML::Node pr = rd["pulse_range"]) {
      const auto range = pr.as<std::vector<int64_t>>();
      if (range.size() != 2) {
        throw std::runtime_error("cphd_reader.pulse_range needs 2 entries");
      }
      first = range[0];
      last = range[1];
    }
  }
  if (last <= first) {
    throw std::runtime_error("pulse_range must be a non-empty half-open interval");
  }
  return {first, last};
}

}  // namespace af_detail

// Range-bearing (polar) auxiliary lattice, the grid PGA's Fourier relationship
// actually holds on: rows of constant slant range from the mid-aperture phase
// centre, columns uniform in bearing within the slant plane. A scatterer's
// phase history over the aperture is -(4 pi / lambda) R cos(theta_j - theta),
// so for the small angles a spotlight dwell subtends the image bearing axis is
// the conjugate of the aperture angle -- of slow time, on a uniformly sampled
// pass.
//
// A Cartesian look x track strip is not that grid, and the difference is not
// academic at spaceborne range: a straight uhat chord of L_perp over this
// dwell walks ~60 compressed range bins, which is what a constant-range row
// exists not to do. Jakowatz & Wahl (SPIE 7337, 2009) make the same point for
// backprojection -- PGA needs a range-bearing image, not a ground-plane one,
// which is Figure 2's reason for a second formation.
//
// Not a SampleGeometry: that descriptor is two unit axes and a uniform metre
// spacing, which cannot express a polar grid. It rides
// BackprojectionOp.sample_points instead (points mode), so no contract change
// is involved.
struct AuxPolarLattice {
  std::array<double, 3> apc{};       // mid-aperture phase centre
  std::array<double, 3> look_hat{};  // apc -> product centre
  std::array<double, 3> uhat{};      // in-plane, perpendicular to look
  double range0{0.0};                // slant range to the anchor (m)
  double range_spacing{0.0};         // m per row
  double bearing_spacing{0.0};       // rad per column
  int64_t num_v{0};
  int64_t num_u{0};
  double rho_az{0.0};
  double l_perp{0.0};

  // Row / column of a point, as a real-valued index. Row num_v / 2 is the
  // anchor range and column num_u / 2 the anchor bearing, so the anchor lands
  // exactly on (num_v / 2, num_u / 2).
  struct Cell {
    double row{0.0};
    double col{0.0};
    double range{0.0};
    double out_of_plane{0.0};  // signed metres off the slant plane
  };

  Cell locate(const std::array<double, 3>& ecef) const {
    const auto d = af_detail::vec_sub(ecef, apc);
    const auto normal = af_detail::vec_cross(look_hat, uhat);
    Cell c;
    c.range = af_detail::vec_norm(d);
    c.row = (c.range - range0) / range_spacing + static_cast<double>(num_v / 2);
    c.col =
        std::atan2(af_detail::vec_dot(d, uhat), af_detail::vec_dot(d, look_hat)) / bearing_spacing +
        static_cast<double>(num_u / 2);
    c.out_of_plane = af_detail::vec_dot(d, normal);
    return c;
  }
};

// The known lie: a polynomial position error along the look axis, injected so
// autofocus has something to find. Lives here rather than in the shared
// AppConfig because it is a diagnostic of this app alone.
struct PositionErrorConfig {
  bool enabled{false};
  std::vector<double> polynomial;  // metres vs s in [-1, 1]
  std::array<double, 3> look{};    // unit look axis, mid-aperture
  int64_t num_pulses{0};
  double warn_rms_rad{1.0};
  // Skip AutofocusOp and correct with the exact phase the lie implies. Proves
  // the correction path end to end, independently of the estimator.
  bool oracle{false};
};

struct AutofocusExtras {
  // The graph-building lambda gets cfg but not the YAML document, and the
  // auxiliary BackprojectionOp needs the same /backprojection block as the
  // product one (coordinate_frame above all). Keep the path so it can decode
  // that block itself rather than having the two formations drift.
  std::filesystem::path config_path;
  // Exactly one of these describes the aux: aux_polar selects the
  // range-bearing lattice on BackprojectionOp.sample_points, and
  // aux_sample_geometry the Cartesian look x track grid. Supplying both to one
  // operator is a build-time error there, by contract.
  radar::SampleGeometry aux_sample_geometry;
  bool aux_polar{false};
  AuxPolarLattice aux_polar_lattice;
  int64_t aperture_pulses{0};
  double aux_rho_az{0.0};
  double aux_l_perp{0.0};
  double aux_samples_per_psf{2.0};
  bool window_fraction_in_yaml{false};
  PositionErrorConfig position_error;

  // Columns of the aux azimuth axis, whichever lattice is active.
  int64_t aux_num_u() const {
    return aux_polar ? aux_polar_lattice.num_u : aux_sample_geometry.num_u();
  }
  int64_t aux_num_v() const {
    return aux_polar ? aux_polar_lattice.num_v : aux_sample_geometry.num_v();
  }
};

// Look x slant-plane lattice: uhat is track with look removed (perp to look in
// the slant plane), sampled at least Nyquist along slow time (samples_per_psf
// per rho_az). n_u may exceed the pulse count; Figure 2 interpolates
// phase_correction from that slow-time axis onto pulses. Product geometry is
// cfg.sample_geometry and is never this value.
inline radar::SampleGeometry derive_aux_geometry(const AppConfig& cfg, int64_t first_pulse,
                                                 int64_t last_pulse, int64_t num_v,
                                                 double samples_per_psf, double* rho_az_out,
                                                 double* l_perp_out) {
  if (last_pulse - first_pulse < 2) {
    throw std::runtime_error("aux lattice needs at least 2 pulses in the aperture");
  }
  if (num_v < 1) {
    throw std::runtime_error("aux_lattice.num_v must be >= 1");
  }
  if (!(samples_per_psf >= 1.0) || !std::isfinite(samples_per_psf)) {
    throw std::runtime_error("aux_lattice.samples_per_psf must be finite and >= 1");
  }
  if (!(cfg.range_bin_spacing > 0.0) || !std::isfinite(cfg.range_bin_spacing)) {
    throw std::runtime_error("aux lattice derivation needs range_bin_spacing > 0");
  }
  const auto frame = af_detail::read_aux_frame(cfg, first_pulse, last_pulse);
  const auto& srp = frame.srp;
  const double du_nyq = frame.rho_az / samples_per_psf;
  int64_t n_u_nyq = static_cast<int64_t>(std::ceil(frame.l_perp / du_nyq)) + 1;
  if (n_u_nyq < 2) {
    n_u_nyq = 2;
  }
  const int64_t n_pulses = last_pulse - first_pulse;
  const int64_t num_u = std::max(n_pulses, n_u_nyq);
  const double du = frame.l_perp / static_cast<double>(num_u - 1);
  if (rho_az_out != nullptr) {
    *rho_az_out = frame.rho_az;
  }
  if (l_perp_out != nullptr) {
    *l_perp_out = frame.l_perp;
  }
  radar::SampleGeometrySpec spec;
  spec.axis_u = frame.uhat;
  spec.axis_v = frame.look_hat;
  spec.spacing = {cfg.range_bin_spacing, du};
  spec.counts = {num_v, num_u};
  const double nv2 = 0.5 * static_cast<double>(num_v);
  const double nu2 = 0.5 * static_cast<double>(num_u);
  spec.origin = af_detail::vec_sub(
      srp, af_detail::vec_add(af_detail::vec_scale(spec.axis_v, nv2 * spec.spacing[0]),
                              af_detail::vec_scale(spec.axis_u, nu2 * spec.spacing[1])));
  return radar::SampleGeometry(spec);
}

// Range-bearing aux lattice. Rows step the compressed range bin, so the grid's
// range axis is the profile's own axis; columns step rho_az / R, the bearing
// step whose N-column span is exactly the unaliased conjugate swath (N *
// rho_az on the ground, the Nyquist limit of the per-pulse phase slope). With
// one column per pulse the estimate is already pulse-indexed and
// PhaseResampleOp is an identity copy.
//
// Anchored on the **product lattice centre**, not the SRP. Any set of
// constant-range rows is a valid polar grid, so the anchor is free, and the
// SRP is the wrong choice: a product centred away from the IARP then needs
// num_v to span the whole IARP-to-scene distance before it covers the scene at
// all. Kourou made that concrete -- 638 m of slant range at 0.273 m bins is
// 2335 rows of pure overhead, so a 2048-row aux missed a target that a
// product-anchored 2048 rows covers with margin. Rows are auto-sized to the
// product's own range span when num_v is not given.
// Ceiling on auto-sized aux rows. Loose on purpose -- see the clamp below for
// why cutting it further costs estimate quality rather than only memory.
inline constexpr int64_t kAuxAutoRowCap = 4096;

inline AuxPolarLattice derive_aux_polar_lattice(const AppConfig& cfg, int64_t first_pulse,
                                                int64_t last_pulse, int64_t num_v,
                                                int64_t num_u_request) {
  if (!(cfg.range_bin_spacing > 0.0) || !std::isfinite(cfg.range_bin_spacing)) {
    throw std::runtime_error("aux lattice derivation needs range_bin_spacing > 0");
  }
  if (!cfg.sample_geometry.is_valid()) {
    throw std::runtime_error(
        "a polar aux lattice is anchored on the product lattice, so sample_geometry must be valid");
  }
  const auto frame = af_detail::read_aux_frame(cfg, first_pulse, last_pulse);
  const int64_t n_pulses = last_pulse - first_pulse;
  const auto& pg = cfg.sample_geometry;
  const auto centre = pg.sample_position(pg.num_v() / 2, pg.num_u() / 2);
  const auto to_centre = af_detail::vec_sub(centre, frame.apc_mid);
  const double range_centre = af_detail::vec_norm(to_centre);
  if (!(range_centre > 0.0)) {
    throw std::runtime_error(
        "aux lattice: product centre coincides with the aperture phase centre");
  }
  AuxPolarLattice out;
  out.apc = frame.apc_mid;
  // Re-point the in-plane frame at the product centre, keeping uhat in the
  // original slant plane so bearing still runs along track.
  out.look_hat = af_detail::vec_scale(to_centre, 1.0 / range_centre);
  const double along = af_detail::vec_dot(frame.uhat, out.look_hat);
  auto uvec = af_detail::vec_sub(frame.uhat, af_detail::vec_scale(out.look_hat, along));
  const double un = af_detail::vec_norm(uvec);
  if (!(un > 0.0)) {
    throw std::runtime_error("aux lattice: track is parallel to the product-centre look");
  }
  out.uhat = af_detail::vec_scale(uvec, 1.0 / un);
  out.range0 = range_centre;
  out.range_spacing = cfg.range_bin_spacing;
  out.bearing_spacing = frame.rho_az / range_centre;
  out.num_u = num_u_request > 0 ? num_u_request : n_pulses;
  out.rho_az = frame.rho_az;
  out.l_perp = frame.l_perp;
  if (out.num_u < 2) {
    throw std::runtime_error("aux_lattice.num_u must be >= 2");
  }
  if (!(out.bearing_spacing > 0.0) || !std::isfinite(out.bearing_spacing)) {
    throw std::runtime_error("aux lattice: bearing spacing is not finite and > 0");
  }
  // Rows to cover the product's four corners, since those bound its slant-range
  // span, with 25% headroom for terrain relief the flat lattice ignores --
  // clamped, because that span is the product's requirement and not the
  // estimator's. The estimator keeps `min_scatterers` range bins (16 by
  // default) out of however many pass the peak-to-mean gate, so rows past the
  // point where good candidates are plentiful buy nothing but memory, and the
  // aux is the second-largest allocation in this graph at roughly 1.09 MiB per
  // row. A native-sized Kourou product asks for 8658 rows and does not fit in
  // 96 GiB; it forms and autofocuses identically on 2478.
  //
  // The clamp is deliberately loose. Row count is the candidate pool the
  // scatterer selection draws from, so cutting it too far leaves a worse best
  // sixteen: measured on Kourou, 4096 and 2478 rows both recover the full
  // sharpness gain, while 512 and 1024 produce an estimate bad enough that
  // iteration 0 regresses and the loop aborts. That failure is silent -- the
  // app still emits an image and exits zero -- so the clamp warns, and an
  // explicit aux_lattice.num_v is honoured as given rather than clamped.
  out.num_v = num_v;
  if (num_v <= 0) {
    double reach = 0.0;
    for (const int64_t iv : {int64_t{0}, pg.num_v() - 1}) {
      for (const int64_t iu : {int64_t{0}, pg.num_u() - 1}) {
        const auto corner = pg.sample_position(iv, iu);
        const double r = af_detail::vec_norm(af_detail::vec_sub(corner, frame.apc_mid));
        reach = std::max(reach, std::abs(r - range_centre));
      }
    }
    const int64_t rows = static_cast<int64_t>(std::ceil(2.5 * reach / cfg.range_bin_spacing));
    const int64_t spanning = std::max<int64_t>(64, rows + (rows & 1));
    out.num_v = std::min<int64_t>(spanning, kAuxAutoRowCap);
    if (out.num_v < spanning) {
      HOLOSCAN_LOG_WARN(
          "aux lattice: auto-sized rows clamped {} -> {} (saves {} rows, and the graph reserves "
          "several copies of each). The aux no longer spans the product's slant-range swath, "
          "which the estimator does not need, but it does narrow the pool its scatterer "
          "selection draws from. If the sharpness report shows no gain, raise aux_lattice.num_v.",
          spanning, out.num_v, spanning - out.num_v);
    }
  }
  if (out.num_v < 2) {
    throw std::runtime_error("aux_lattice.num_v must be >= 2 for a polar lattice");
  }
  return out;
}

inline void load_position_error(const YAML::Node& root, const AppConfig& cfg,
                                PositionErrorConfig* out) {
  const YAML::Node pe = root["position_error"];
  if (!pe || pe.IsNull()) {
    return;
  }
  out->enabled = true;
  const std::string axis = pe["axis"] ? pe["axis"].as<std::string>() : std::string("look");
  if (axis != "look") {
    throw std::runtime_error("position_error.axis must be look (mid-aperture APC to SRP)");
  }
  if (!pe["polynomial"]) {
    throw std::runtime_error("position_error.polynomial is required (metres vs s in [-1, 1])");
  }
  out->polynomial = pe["polynomial"].as<std::vector<double>>();
  if (out->polynomial.empty()) {
    throw std::runtime_error("position_error.polynomial must have at least one coefficient");
  }
  for (const double c : out->polynomial) {
    if (!std::isfinite(c)) {
      throw std::runtime_error("position_error.polynomial must be finite");
    }
  }
  if (const YAML::Node w = pe["score_warn_rms_rad"]) {
    out->warn_rms_rad = w.as<double>();
    if (!(out->warn_rms_rad > 0.0) || !std::isfinite(out->warn_rms_rad)) {
      throw std::runtime_error("position_error.score_warn_rms_rad must be finite and > 0");
    }
  }
  if (const YAML::Node o = pe["oracle"]) {
    out->oracle = o.as<bool>();
  }
  const auto window = af_detail::pulse_range_from_yaml(root, cfg.cphd_path);
  out->num_pulses = window.second - window.first;

  radar::cphd::CphdFile file(cfg.cphd_path);
  if (!file.has_pvp_field("TxPos") || !file.has_pvp_field("SRPPos")) {
    throw std::runtime_error("position_error axis: look needs PVP TxPos and SRPPos");
  }
  const int64_t mid = window.first + (window.second - window.first) / 2;
  auto apc = af_detail::read_xyz(file, "TxPos", mid);
  auto srp = af_detail::read_xyz(file, "SRPPos", mid);
  if (cfg.transform_to_scene_enu) {
    apc = af_detail::ecef_to_scene_enu(apc, cfg.enu_origin_ecef);
    srp = af_detail::ecef_to_scene_enu(srp, cfg.enu_origin_ecef);
  }
  const auto look = af_detail::vec_sub(srp, apc);
  const double n = af_detail::vec_norm(look);
  if (!(n > 0.0)) {
    throw std::runtime_error("position_error: mid-aperture TxPos coincides with SRPPos");
  }
  out->look = af_detail::vec_scale(look, 1.0 / n);
}

inline AppConfig load_autofocus_config(const std::filesystem::path& path,
                                       const std::filesystem::path& cphd_override,
                                       AutofocusExtras* extras) {
  AppConfig cfg = load_sar_config(path, cphd_override);
  extras->config_path = path;
  if (cfg.scroll) {
    throw std::runtime_error("autofocus is Figure 2 (spotlight reform), not scroll");
  }
  if (cfg.tile_geometries.size() > 1) {
    throw std::runtime_error("autofocus is Figure 2 (spotlight reform), not tiles");
  }
  if (cfg.cphd_paths.size() > 1) {
    throw std::runtime_error("autofocus is one collection; use independent_images for mode 5");
  }
  if (!cfg.nvradar_vph_path.empty()) {
    throw std::runtime_error("autofocus does not take nvradar_vph_path");
  }

  // Read these as their canonical spellings rather than through a YAML
  // converter: the operator blocks reach their operators through
  // decode_params, and this app only needs to know which topology was asked
  // for before the graph exists.
  const YAML::Node root = YAML::LoadFile(path.string());
  int64_t max_ap = 0;
  bool has_window = false;
  std::string emit(radar::to_string(radar::ApertureEmitMode::kContinue));
  if (const YAML::Node am = root["aperture_manager"]) {
    if (const YAML::Node m = am["max_aperture_pulses"]) {
      max_ap = m.as<int64_t>();
    }
    if (const YAML::Node e = am["emit_mode"]) {
      emit = e.as<std::string>();
    }
    if (const YAML::Node w = am["aperture_window"]) {
      if (const YAML::Node t = w["type"]) {
        has_window = t.as<std::string>() != radar::to_string(radar::WindowType::kNone);
      }
    }
  }
  if (max_ap <= 0) {
    throw std::runtime_error(
        "Figure 2 needs aperture_manager.max_aperture_pulses > 0 "
        "(AF-off spotlight cannot replay after estimation)");
  }
  if (emit != radar::to_string(radar::ApertureEmitMode::kReform)) {
    throw std::runtime_error("Figure 2 needs aperture_manager.emit_mode: reform");
  }
  // Retention here is replay, not eviction. Wiring weight would D2H an
  // all-ones vector every tick.
  cfg.wire_weight = has_window;
  cfg.show_playback_lag = false;

  load_position_error(root, cfg, &extras->position_error);
  if (extras->position_error.oracle && !extras->position_error.enabled) {
    throw std::runtime_error("position_error.oracle needs a position_error block");
  }

  const auto window = af_detail::pulse_range_from_yaml(root, cfg.cphd_path);
  const int64_t first_pulse = window.first;
  const int64_t last_pulse = window.second;
  extras->aperture_pulses = last_pulse - first_pulse;
  if (max_ap < extras->aperture_pulses) {
    throw std::runtime_error("aperture_manager.max_aperture_pulses must be >= the aperture (" +
                             std::to_string(extras->aperture_pulses) + " pulses)");
  }

  if (extras->position_error.oracle) {
    // No AutofocusOp, so the aux lattice is unused.
    return cfg;
  }

  if (const YAML::Node af = root["autofocus"]) {
    extras->window_fraction_in_yaml = static_cast<bool>(af["window_fraction"]);
  }

  int64_t num_v = 32;
  int64_t num_u = 0;
  double samples_per_psf = 2.0;
  bool polar = false;
  if (const YAML::Node aux = root["aux_lattice"]) {
    if (const YAML::Node m = aux["mode"]) {
      const auto mode = m.as<std::string>();
      if (mode == "polar") {
        polar = true;
        num_v = 0;  // 0 = auto-size from the product's range span
      } else if (mode != "cartesian") {
        throw std::runtime_error("aux_lattice.mode must be 'polar' or 'cartesian'");
      }
    }
    if (const YAML::Node geom = aux["sample_geometry"]) {
      if (polar) {
        throw std::runtime_error(
            "aux_lattice.sample_geometry cannot describe a polar lattice; "
            "drop one of mode: polar / sample_geometry");
      }
      extras->aux_sample_geometry = geom.as<radar::SampleGeometry>();
      return cfg;
    }
    if (const YAML::Node nv = aux["num_v"]) {
      num_v = nv.as<int64_t>();
    }
    if (const YAML::Node nu = aux["num_u"]) {
      if (!polar) {
        throw std::runtime_error(
            "aux_lattice.num_u applies to mode: polar only "
            "(the Cartesian grid takes samples_per_psf)");
      }
      num_u = nu.as<int64_t>();
    }
    if (const YAML::Node spp = aux["samples_per_psf"]) {
      if (polar) {
        throw std::runtime_error(
            "aux_lattice.samples_per_psf applies to the Cartesian grid; "
            "a polar lattice takes num_u");
      }
      samples_per_psf = spp.as<double>();
    }
  }
  if (polar) {
    extras->aux_polar = true;
    extras->aux_polar_lattice =
        derive_aux_polar_lattice(cfg, first_pulse, last_pulse, num_v, num_u);
    extras->aux_rho_az = extras->aux_polar_lattice.rho_az;
    extras->aux_l_perp = extras->aux_polar_lattice.l_perp;
    return cfg;
  }
  extras->aux_samples_per_psf = samples_per_psf;
  extras->aux_sample_geometry =
      derive_aux_geometry(cfg, first_pulse, last_pulse, num_v, samples_per_psf, &extras->aux_rho_az,
                          &extras->aux_l_perp);
  return cfg;
}
