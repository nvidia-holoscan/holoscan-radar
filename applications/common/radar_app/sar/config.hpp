// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared SAR config type and YAML loader. Each standalone app validates
// its topology in a colocated config.hpp. Operator blocks are NOT parsed here — they flow
// through Holoscan's from_config(). Exception: nfft and
// propagation_speed under range_compression are a read-only mirror
// so range_bin_spacing matches the operator if those keys are set.
// App-owned `nfft_pad: next_fast` computes nfft as
// next_fast_fft_size(max CPHD num_samples) and the Graph driver passes it
// as Arg("nfft"); mutually exclusive with range_compression.nfft.
//
// Lattice: `sample_geometry` (Atlanta ENU), sidecar truth, or
// `lattice` — explicit counts/spacing (IARP ENU; counts may be N
// or [num_v, num_u]), collection physics (XY square at the
// coarser GSD), `lattice.tiles` > 1 (mode 3: parent strip +
// ROI tiles along a travelling SRP; one file; omit
// counts/spacing), or `lattice.scroll` (mode 4: same parent,
// one window `max_side` deep along v). Along-track follows the
// SRP chord, not the IARP tangent plane. `max_side` caps each
// tile's along-track axis in mode 3 (the parent is n_tiles of
// those along v; tile and parent u may exceed the cap so the
// IFFT swath stays native GSD). Mode 4 uses `max_side` as the
// buffer depth, not a coarsening cap.
// `pixel_scale` is a per-collect factor on N. A guessed sidecar
// must not override an explicit YAML lattice.

#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/fft_size.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/sar_yaml_converters.hpp"
#include "radar_app/app/run_config.hpp"

namespace radar = holoscan::radar;

struct AppConfig : radar::app::CommonRunConfig {
  std::string cphd_path;
  std::vector<std::string> cphd_paths;
  std::vector<std::string> scene_names;
  std::string sidecar_path;
  // dump: true wires TensorDumpOp. Off by default.
  bool dump{false};
  std::string dump_path;
  int64_t dump_max_files{-1};
  radar::SampleGeometry sample_geometry;
  double reference_frequency{0.0};
  // Metres per compressed range bin. Mode 1 leaves range_bins
  // unconnected; this is BackprojectionOp.range_bin_spacing, with
  // the centred origin default. Derived from the CPHD (SCSS, N)
  // so it matches RC's axis when bandwidth = (N-1)*SCSS.
  double range_bin_spacing{0.0};
  // 0 = operator default (num_samples). Mirrored from
  // range_compression.nfft when that YAML key is set.
  int64_t nfft{0};
  // Mode 3: parent strip plus one lattice per tile along a
  // travelling SRP, from lattice.tiles > 1. Size 0 when not
  // tiling. collection_geometries stays length 1 so the mode-5
  // collection_boundary stager stays off; tile 0 is also
  // sample_geometry (BP parameter). parent_geometry is the
  // mosaic canvas (LUT / Holoviz / optional dump).
  std::vector<radar::SampleGeometry> tile_geometries;
  radar::SampleGeometry parent_geometry;
  int64_t tile_pulses{0};
  // Mode 4: lattice.scroll. Parent is the SRP-chord strip;
  // sample_geometry is the scrolling window (max_side along v).
  bool scroll{false};
  radar::SampleGeometry scroll_parent;
  int64_t scroll_anchor{0};
  // Read-only mirror of cphd_reader.loop so the tile stager can
  // restage tile 0 on wrap. Not parsed into the operator here.
  bool reader_loop{false};
  // Read-only mirrors of the reader / manager cadences so
  // the continuous-strip stage can slice a reader batch when the emit interval is
  // smaller than the reader batch. from_config() still owns the
  // operator parameters.
  int64_t reader_pulses_per_tick{0};
  int64_t emit_interval_pulses{0};
  // Mode 5: one lattice / dr / f0 per collection, aligned with
  // cphd_paths. Modes 1 and 3 have a single entry.
  std::vector<radar::SampleGeometry> collection_geometries;
  std::vector<double> collection_spacings;
  std::vector<double> collection_frequencies;
  // Timing cap on each physics-derived (or explicit-count) lattice
  // axis. 0 = no cap (native collection-physics N, or the YAML
  // counts). A positive cap coarsens GSD and keeps the scene.
  // Mode 3: each tile's along-track axis (u may exceed so the
  // IFFT swath stays native). Mode 2 / 5: the formed image.
  // Lower it on a slower GPU.
  int64_t lattice_max_side{0};
  // Per-collection factor on native N (both axes). 1 = native.
  // Same square, coarser GSD. Size matches cphd_paths.
  std::vector<double> lattice_pixel_scale;
  // Filled when `lattice` omits counts/spacing so compose can log
  // heading / swath / GSD. Empty on the explicit-count path.
  std::vector<radar::cphd::CphdFile::CollectionPhysics> collection_physics;
  // When set, the app graph rotates reader ECEF positions / SRP into
  // scene ENU at this IARP so sample_geometry can be an ENU lattice
  // (NVRadar Atlanta grid diagnostic).
  bool transform_to_scene_enu{false};
  std::array<double, 3> enu_origin_ecef{};
  // Diagnostic: round ENU APC through float32 (NVRadar .dat xyz).
  bool quantize_phase_center_f32{false};
  // Diagnostic: replace contract RC with their conditioned vph.
  std::string nvradar_vph_path;
  // Wire manager.weight → BP.weight. True when the manager
  // retains (max_aperture_pulses > 0, eviction −1) or stamps an
  // aperture_window (mode 3 Taylor). Mode 1 / 5 stay unconnected
  // (all +1, no D2H of weights).
  bool wire_weight{false};
  // Collect-vs-wall lag timeline HUD. Retention (mode 2) and
  // paced strips (mode 3 mosaic / mode 4 scroll). Modes 1 and
  // 5 omit it. Display still gates on HUD playback.paced so
  // an unpaced YAML stays blank.
  bool show_playback_lag{false};
  // When true, app-owned scroll/mosaic required emits wait on
  // kBackpressured. Library operators take the same-named static Param
  // on their YAML blocks (default false).
  bool wait_for_downstream{false};
  // Holoviz window: longest side (px) for square collects, and
  // the no-DISPLAY fallback for a parent strip. A parent strip
  // (mode 3 mosaic / mode 4 scroll blit) fits the desktop work
  // area (full width on a typical 16:9). LUT box-averages toward
  // that side; a dump is the formed lattice (parent when tiling
  // or scrolling).
  int64_t visualize_size{1024};
  // Strip Holoviz crop of parent u (half-open). 0 nu = full
  // union. Set from the range-window intersection so the
  // walking far/near edge is off-screen. BP / dump unchanged.
  int64_t viz_crop_u0{0};
  int64_t viz_crop_nu{0};
  // Seconds the Holoviz window keeps the formed image after emit.
  // Also the max wait for the first frame (Atlanta ~9 s). Mode 1
  // emits once; 0 closes with the graph. Negative: hold until
  // SIGINT (mode 3 / 5 looping).
  double visualize_hold_s{30.0};
};

struct CliPaths {
  std::filesystem::path config;
  std::filesystem::path cphd_override;
};

// First CLI argument is the YAML, unless it is a .cphd — then the
// staged default YAML is used and that file is the collection. A
// second argument always overrides cphd.path.
inline CliPaths parse_cli(int argc, char** argv, const char* default_yaml) {
  const auto bin_dir = std::filesystem::path(argv[0]).parent_path();
  CliPaths out;
  if (argc <= 1) {
    out.config = bin_dir / default_yaml;
    return out;
  }
  const std::filesystem::path first(argv[1]);
  const std::string ext = first.extension().string();
  if (ext == ".cphd" || ext == ".CPHD") {
    out.config = bin_dir / default_yaml;
    out.cphd_override = first;
    return out;
  }
  if (std::filesystem::is_regular_file(first)) {
    out.config = first;
  } else {
    const auto beside = bin_dir / first.filename();
    out.config = std::filesystem::is_regular_file(beside) ? beside : first;
  }
  if (argc > 2) {
    out.cphd_override = argv[2];
  }
  return out;
}

// dr = c / (2 * N_bins * df), df = SCSS. Same formula RC uses when
// bandwidth is the CPHD span (N_samples - 1)*SCSS. nfft_or_0 == 0
// means N_bins = num_samples (the mode-1 default).
inline double range_bin_spacing_from_cphd(const std::string& path, int64_t nfft_or_0,
                                          double propagation_speed) {
  radar::cphd::CphdFile file(path);
  if (file.sample_domain() != radar::cphd::SampleDomain::kFX) {
    throw std::runtime_error(
        "range_bin_spacing is derived from SCSS as a frequency "
        "step; collection is not FX-domain: " +
        path);
  }
  const int64_t num_samples = file.channel(0).num_samples;
  if (num_samples < 2) {
    throw std::runtime_error("collection has fewer than 2 samples; no range spacing: " + path);
  }
  if (!file.has_pvp_field("SCSS")) {
    throw std::runtime_error("collection has no SCSS PVP: " + path);
  }
  const double df = file.read_pvp(0, "SCSS", 0, 1, 1).at(0);
  if (!(df > 0.0) || !std::isfinite(df)) {
    throw std::runtime_error("SCSS must be finite and > 0 Hz: " + path);
  }
  const int64_t num_bins = nfft_or_0 > 0 ? nfft_or_0 : num_samples;
  if (num_bins < num_samples) {
    throw std::runtime_error("nfft must be >= num_samples for range_bin_spacing");
  }
  return propagation_speed / (2.0 * static_cast<double>(num_bins) * df);
}

// Stored sample N/2, matching SARRangeCompressionOp's IFFT DC after
// ifftshift. Same formula spotlight.yaml hardcodes for Atlanta.
inline double reference_frequency_from_cphd(const std::string& path) {
  radar::cphd::CphdFile file(path);
  if (file.sample_domain() != radar::cphd::SampleDomain::kFX) {
    throw std::runtime_error(
        "reference_frequency is derived from SC0/SCSS as a "
        "frequency; collection is not FX-domain: " +
        path);
  }
  const int64_t num_samples = file.channel(0).num_samples;
  if (num_samples < 2) {
    throw std::runtime_error("collection has fewer than 2 samples; no f0: " + path);
  }
  if (!file.has_pvp_field("SC0") || !file.has_pvp_field("SCSS")) {
    throw std::runtime_error("collection has no SC0/SCSS PVP: " + path);
  }
  const double sc0 = file.read_pvp(0, "SC0", 0, 1, 1).at(0);
  const double df = file.read_pvp(0, "SCSS", 0, 1, 1).at(0);
  if (!std::isfinite(sc0) || !std::isfinite(df)) {
    throw std::runtime_error("SC0/SCSS must be finite: " + path);
  }
  return sc0 + static_cast<double>(num_samples / 2) * df;
}

// Bowring WGS-84 + ECEF→ENU, matching EcefToSceneEnuOp. Rows are
// east, north, up in ECEF.
inline std::array<double, 9> ecef_to_enu_rotation(const std::array<double, 3>& ecef) {
  constexpr double kWgs84A = 6378137.0;
  constexpr double kWgs84F = 1.0 / 298.257223563;
  const double b = kWgs84A * (1.0 - kWgs84F);
  const double e2 = 1.0 - (b * b) / (kWgs84A * kWgs84A);
  const double x = ecef[0];
  const double y = ecef[1];
  const double z = ecef[2];
  const double lon = std::atan2(y, x);
  const double p = std::hypot(x, y);
  const double theta = std::atan2(z * kWgs84A, p * b);
  const double slt = std::sin(theta);
  const double clt = std::cos(theta);
  const double lat = std::atan2(z + e2 * (kWgs84A * kWgs84A / b) * slt * slt * slt,
                                p - e2 * kWgs84A * clt * clt * clt);
  const double sl = std::sin(lat);
  const double cl = std::cos(lat);
  const double so = std::sin(lon);
  const double co = std::cos(lon);
  return {-so, co, 0.0, -sl * co, -sl * so, cl, cl * co, cl * so, sl};
}

// Lattice centre (half-index along each axis), same frame as
// origin / axes.
inline std::array<double, 3> lattice_centre(const radar::SampleGeometry& geom) {
  const double cv = 0.5 * static_cast<double>(geom.num_v() - 1);
  const double cu = 0.5 * static_cast<double>(geom.num_u() - 1);
  const auto& o = geom.origin();
  const auto& av = geom.axis_v();
  const auto& au = geom.axis_u();
  const double dv = cv * geom.spacing()[0];
  const double du = cu * geom.spacing()[1];
  return {o[0] + dv * av[0] + du * au[0], o[1] + dv * av[1] + du * au[1],
          o[2] + dv * av[2] + du * au[2]};
}

inline std::array<double, 3> rotate_apc_to_scene(std::array<double, 3> apc_xyz, bool to_scene_enu,
                                                 const std::array<double, 3>& enu_origin) {
  if (!to_scene_enu) {
    return apc_xyz;
  }
  const auto R = ecef_to_enu_rotation(enu_origin);
  const double dx = apc_xyz[0] - enu_origin[0];
  const double dy = apc_xyz[1] - enu_origin[1];
  const double dz = apc_xyz[2] - enu_origin[2];
  return {R[0] * dx + R[1] * dy + R[2] * dz, R[3] * dx + R[4] * dy + R[5] * dz,
          R[6] * dx + R[7] * dy + R[8] * dz};
}

// APC minus lattice centre, metres along axis_u / axis_v (raster,
// not geographic). Chevron uses this so IA-aligned strips are
// not treated as east/south.
inline std::optional<std::array<double, 2>> apc_raster_uv(std::array<double, 3> apc_xyz,
                                                          const radar::SampleGeometry& geom,
                                                          bool to_scene_enu,
                                                          const std::array<double, 3>& enu_origin) {
  if (!geom.is_valid()) {
    return std::nullopt;
  }
  apc_xyz = rotate_apc_to_scene(apc_xyz, to_scene_enu, enu_origin);
  const auto c = lattice_centre(geom);
  const double d0 = apc_xyz[0] - c[0];
  const double d1 = apc_xyz[1] - c[1];
  const double d2 = apc_xyz[2] - c[2];
  const auto& au = geom.axis_u();
  const auto& av = geom.axis_v();
  const double ru = d0 * au[0] + d1 * au[1] + d2 * au[2];
  const double rv = d0 * av[0] + d1 * av[1] + d2 * av[2];
  if (!std::isfinite(ru) || !std::isfinite(rv)) {
    return std::nullopt;
  }
  if (ru * ru + rv * rv < 1.0) {
    return std::nullopt;
  }
  return std::array<double, 2>{ru, rv};
}

// Pixel-centre lattice covering the IA rectangle [x1, x2] × [y1, y2]
// metres from IARP along unit uIAX / uIAY. counts are (num_v, num_u)
// along IAY / IAX. Sample (0, 0) is the min-X min-Y pixel centre.
inline radar::SampleGeometry lattice_from_image_area(const std::array<double, 3>& iarp,
                                                     const std::array<double, 3>& u_iax,
                                                     const std::array<double, 3>& u_iay, double x1,
                                                     double y1, double x2, double y2, int64_t num_v,
                                                     int64_t num_u) {
  if (!(x2 > x1) || !(y2 > y1)) {
    throw std::runtime_error("image area must have x2 > x1 and y2 > y1");
  }
  if (num_v <= 0 || num_u <= 0) {
    throw std::runtime_error("lattice counts must be [num_v, num_u], both > 0");
  }
  const double dx = (x2 - x1) / static_cast<double>(num_u);
  const double dy = (y2 - y1) / static_cast<double>(num_v);
  const double x0 = x1 + 0.5 * dx;
  const double y0 = y1 + 0.5 * dy;
  radar::SampleGeometrySpec spec;
  spec.origin = {iarp[0] + x0 * u_iax[0] + y0 * u_iay[0], iarp[1] + x0 * u_iax[1] + y0 * u_iay[1],
                 iarp[2] + x0 * u_iax[2] + y0 * u_iay[2]};
  spec.axis_u = u_iax;
  spec.axis_v = u_iay;
  spec.spacing = {dy, dx};
  spec.counts = {num_v, num_u};
  return radar::SampleGeometry(spec);
}

// Square N, or [num_v, num_u].
inline std::array<int64_t, 2> parse_lattice_counts(const YAML::Node& node) {
  if (node.IsScalar()) {
    const int64_t n = node.as<int64_t>();
    if (n <= 0) {
      throw std::runtime_error("lattice.counts must be > 0");
    }
    return {n, n};
  }
  if (!node.IsSequence()) {
    throw std::runtime_error("lattice.counts must be N or [num_v, num_u]");
  }
  const auto counts = node.as<std::vector<int64_t>>();
  if (counts.size() != 2 || counts[0] <= 0 || counts[1] <= 0) {
    throw std::runtime_error("lattice.counts must be N or [num_v, num_u], all > 0");
  }
  return {counts[0], counts[1]};
}

// Range-Nyquist counts that fill the extents, then cap the longer
// side at max_side (0 = no cap). Timing knob: lower max_side
// coarsens spacing and keeps the declared scene. Spotlight /
// video-SAR images (one lattice).
inline std::array<int64_t, 2> counts_for_extent(double extent_v, double extent_u,
                                                double native_spacing, int64_t max_side) {
  if (!(native_spacing > 0.0) || !std::isfinite(native_spacing)) {
    throw std::runtime_error("native lattice spacing must be finite and > 0");
  }
  if (!(extent_v > 0.0) || !(extent_u > 0.0)) {
    throw std::runtime_error("image area extents must be > 0");
  }
  auto num_v = std::max(int64_t{1}, static_cast<int64_t>(std::llround(extent_v / native_spacing)));
  auto num_u = std::max(int64_t{1}, static_cast<int64_t>(std::llround(extent_u / native_spacing)));
  const int64_t longest = std::max(num_v, num_u);
  if (max_side > 0 && longest > max_side) {
    const double scale = static_cast<double>(longest) / static_cast<double>(max_side);
    num_v = std::max(int64_t{1},
                     static_cast<int64_t>(std::llround(static_cast<double>(num_v) / scale)));
    num_u = std::max(int64_t{1},
                     static_cast<int64_t>(std::llround(static_cast<double>(num_u) / scale)));
  }
  return {num_v, num_u};
}

// Same as counts_for_extent, but max_side caps each tile's
// along-track axis. n_tiles slices extent_v. Parent/tile u is
// the range-window envelope union and is not the cap; a 10:1
// strip does not spend the whole budget on along-track, and an
// IFFT swath wider than max_side stays at native GSD.
inline std::array<int64_t, 2> counts_for_tiled_extent(double extent_v, double extent_u,
                                                      double native_spacing, int64_t n_tiles,
                                                      int64_t max_side) {
  if (n_tiles < 1) {
    throw std::runtime_error("lattice.tiles must be >= 1");
  }
  if (!(native_spacing > 0.0) || !std::isfinite(native_spacing)) {
    throw std::runtime_error("native lattice spacing must be finite and > 0");
  }
  if (!(extent_v > 0.0) || !(extent_u > 0.0)) {
    throw std::runtime_error("image area extents must be > 0");
  }
  auto num_v = std::max(int64_t{1}, static_cast<int64_t>(std::llround(extent_v / native_spacing)));
  auto num_u = std::max(int64_t{1}, static_cast<int64_t>(std::llround(extent_u / native_spacing)));
  const int64_t tile_v = std::max(int64_t{1}, num_v / n_tiles);
  if (max_side > 0 && tile_v > max_side) {
    const double scale = static_cast<double>(tile_v) / static_cast<double>(max_side);
    num_v = std::max(int64_t{1},
                     static_cast<int64_t>(std::llround(static_cast<double>(num_v) / scale)));
  }
  return {num_v, num_u};
}

inline int64_t display_downsample_factor(int64_t height, int64_t width, int64_t max_side) {
  if (max_side <= 0) {
    throw std::runtime_error("visualize_size must be > 0");
  }
  const int64_t longest = std::max(height, width);
  if (longest <= max_side) {
    return 1;
  }
  return (longest + max_side - 1) / max_side;
}

// Holoviz (width, height). Landscape when the lattice is taller
// than wide (along-track on v), matching SarImageVizOp.
inline std::array<uint32_t, 2> holoviz_window_wh(int64_t num_v, int64_t num_u, int64_t max_side) {
  const int64_t factor = display_downsample_factor(num_v, num_u, max_side);
  const int64_t ov = num_v / factor;
  const int64_t ou = num_u / factor;
  const auto w = static_cast<uint32_t>(std::max(ov, ou));
  const auto h = static_cast<uint32_t>(std::min(ov, ou));
  return {std::max(uint32_t{1}, w), std::max(uint32_t{1}, h)};
}

// Fit a landscape lattice into the work area so at least one
// desktop edge is used (full width for a wide/short strip on
// 16:9). Caller passes _NET_WORKAREA size; GLFW client size is
// the result (title bar does not eat width).
inline std::array<uint32_t, 2> holoviz_fit_wh(int64_t num_v, int64_t num_u, int work_w,
                                              int work_h) {
  const double long_n = static_cast<double>(std::max(num_v, num_u));
  const double short_n = static_cast<double>(std::min(num_v, num_u));
  if (!(long_n > 0.0) || !(short_n > 0.0) || work_w <= 0 || work_h <= 0) {
    return {1, 1};
  }
  const double s =
      std::min(static_cast<double>(work_w) / long_n, static_cast<double>(work_h) / short_n);
  const int64_t wi = std::max(int64_t{1}, static_cast<int64_t>(std::llround(long_n * s)));
  const int64_t hi = std::max(int64_t{1}, static_cast<int64_t>(std::llround(short_n * s)));
  const auto w = static_cast<uint32_t>(std::min(wi, static_cast<int64_t>(work_w)));
  const auto h = static_cast<uint32_t>(std::min(hi, static_cast<int64_t>(work_h)));
  return {w, h};
}

inline radar::SampleGeometry ecef_lattice_at_iarp(const std::array<double, 3>& iarp,
                                                  const std::array<int64_t, 2>& counts,
                                                  const std::array<double, 2>& spacing) {
  const std::array<double, 9> rotation = ecef_to_enu_rotation(iarp);
  const std::array<double, 3> east{rotation[0], rotation[1], rotation[2]};
  const std::array<double, 3> north{rotation[3], rotation[4], rotation[5]};
  const std::array<double, 3> south{-north[0], -north[1], -north[2]};
  const double ext_v = static_cast<double>(counts[0]) * spacing[0];
  const double ext_u = static_cast<double>(counts[1]) * spacing[1];
  return lattice_from_image_area(iarp, east, south, -ext_u / 2.0, -ext_v / 2.0, ext_u / 2.0,
                                 ext_v / 2.0, counts[0], counts[1]);
}

// IARP-centred east-north square from collection_physics().
// pixel_scale multiplies native N (both axes). max_side 0 then
// keeps that N at the coarser GSD; a positive cap (after scale)
// keeps the same square and coarsens spacing.
inline radar::SampleGeometry ecef_physics_lattice(
    const std::array<double, 3>& iarp, const radar::cphd::CphdFile::CollectionPhysics& phys,
    int64_t max_side, double pixel_scale) {
  if (!(pixel_scale > 0.0) || !std::isfinite(pixel_scale)) {
    throw std::runtime_error("lattice.pixel_scale must be finite and > 0");
  }
  int64_t n = std::max(
      int64_t{1},
      static_cast<int64_t>(std::llround(static_cast<double>(phys.native_count) * pixel_scale)));
  if (max_side > 0 && n > max_side) {
    n = max_side;
  }
  const double spacing =
      (n == phys.native_count) ? phys.gsd_m : phys.xy_square_m / static_cast<double>(n);
  return ecef_lattice_at_iarp(iarp, {n, n}, {spacing, spacing});
}

inline double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline std::array<double, 3> sub3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline std::array<double, 3> add3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline std::array<double, 3> scale3(const std::array<double, 3>& a, double s) {
  return {a[0] * s, a[1] * s, a[2] * s};
}

inline double dist3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return std::sqrt(dot3(sub3(a, b), sub3(a, b)));
}

inline std::array<double, 3> unit3(const std::array<double, 3>& a, const std::string& what) {
  const double n = std::sqrt(dot3(a, a));
  if (!(n > 0.0) || !std::isfinite(n)) {
    throw std::runtime_error(what + " must be a non-zero vector");
  }
  return {a[0] / n, a[1] / n, a[2] / n};
}

struct StripmapTiling {
  radar::SampleGeometry parent;
  std::vector<radar::SampleGeometry> tiles;
  int64_t tile_pulses{0};
  // Holoviz crop of parent u (half-open [display_u0, +display_nu)).
  // 0 nu means show the full union. Formation and dump stay parent.
  int64_t display_u0{0};
  int64_t display_nu{0};
};

// Mode 3: parent strip plus ROI tiles. Along-track (v) follows
// the SRP chord in ECEF. Cross-track is the IA across-axis,
// orthogonalized against the chord. Parent u is every ground
// point the pulses can light (IFFT window on the plane, first
// / mid / last pulse of each tile). Every tile forms that
// full u so a shared v-edge is the same height; cropping u
// per tile made the staircase at the joins. Holoviz crops
// parent u to the intersection of those envelopes so the
// walking range-window edge is off-screen; dump stays the
// union. Capella ImageArea extents are not the scene.
// first/last is the half-open pulse window. max_side caps
// each tile's along-track axis; the parent is n_tiles of
// those along v.
inline StripmapTiling stripmap_tile_lattices(const radar::cphd::CphdFile& file, int64_t n_tiles,
                                             int64_t max_side, double pixel_scale,
                                             const radar::cphd::CphdFile::CollectionPhysics& phys,
                                             int64_t first_pulse, int64_t last_pulse,
                                             double half_slant) {
  if (n_tiles < 1) {
    throw std::runtime_error("lattice.tiles must be >= 1");
  }
  if (!(pixel_scale > 0.0) || !std::isfinite(pixel_scale)) {
    throw std::runtime_error("lattice.pixel_scale must be finite and > 0");
  }
  if (file.num_channels() < 1) {
    throw std::runtime_error("tiling needs a CPHD with a channel");
  }
  if (!file.has_pvp_field("SRPPos")) {
    throw std::runtime_error("tiling needs PVP SRPPos (travelling scene reference)");
  }
  if (!file.has_pvp_field("TxPos")) {
    throw std::runtime_error("tiling needs PVP TxPos (range-window envelope)");
  }
  if (!file.has_image_area()) {
    throw std::runtime_error("tiling needs planar SceneCoordinates uIAX/uIAY");
  }
  if (!file.has_iarp()) {
    throw std::runtime_error("tiling needs SceneCoordinates IARP");
  }
  const int64_t nvec = file.channel(0).num_vectors;
  if (first_pulse < 0 || last_pulse > nvec || first_pulse >= last_pulse) {
    throw std::runtime_error("pulse_range [" + std::to_string(first_pulse) + ", " +
                             std::to_string(last_pulse) + ") is out of range for tiling (" +
                             std::to_string(nvec) + " vectors)");
  }
  const int64_t span = last_pulse - first_pulse;
  if (span < n_tiles) {
    throw std::runtime_error("tiling needs at least as many pulses as tiles");
  }
  if (span % n_tiles != 0) {
    throw std::runtime_error("lattice.tiles must divide the pulse window (" + std::to_string(span) +
                             " pulses, " + std::to_string(n_tiles) +
                             " tiles); drop leftover vectors with pulse_range");
  }
  if (!(phys.gsd_range_m > 0.0)) {
    throw std::runtime_error("tiling needs a positive range GSD");
  }
  if (!(half_slant > 0.0) || !std::isfinite(half_slant)) {
    throw std::runtime_error("tiling needs a positive compressed-range half-span");
  }
  const auto& ia = file.image_area();
  const auto iarp = file.iarp_ecef();
  const auto iax = unit3(ia.u_iax, "uIAX");
  const auto iay = unit3(ia.u_iay, "uIAY");
  if (std::abs(dot3(iax, iay)) > 0.99) {
    throw std::runtime_error("uIAX and uIAY must not be parallel");
  }
  const auto srp_at = [&](int64_t i) {
    const auto v = file.read_pvp(0, "SRPPos", i, 1, 3);
    if (v.size() != 3) {
      throw std::runtime_error("SRPPos must have 3 components");
    }
    return std::array<double, 3>{v[0], v[1], v[2]};
  };
  const auto tx_at = [&](int64_t i) {
    const auto v = file.read_pvp(0, "TxPos", i, 1, 3);
    if (v.size() != 3) {
      throw std::runtime_error("TxPos must have 3 components");
    }
    return std::array<double, 3>{v[0], v[1], v[2]};
  };
  const auto proj = [&](const std::array<double, 3>& srp, const std::array<double, 3>& axis) {
    return dot3(sub3(srp, iarp), axis);
  };
  const auto s0 = srp_at(first_pulse);
  const auto s1 = srp_at(last_pulse - 1);
  const double x0 = proj(s0, iax);
  const double x1 = proj(s1, iax);
  const double y0 = proj(s0, iay);
  const double y1 = proj(s1, iay);
  const bool along_x = std::abs(x1 - x0) >= std::abs(y1 - y0);
  const std::array<double, 3> ia_across = along_x ? iay : iax;
  auto axis_along = unit3(sub3(s1, s0), "SRP track");
  double a0 = proj(s0, axis_along);
  double a1 = proj(s1, axis_along);
  if (a1 < a0) {
    axis_along = scale3(axis_along, -1.0);
    a0 = -a0;
    a1 = -a1;
  }
  if (!(a1 > a0) || !std::isfinite(a0) || !std::isfinite(a1)) {
    throw std::runtime_error("tiling needs a travelling SRP");
  }
  const auto axis_across =
      unit3(sub3(ia_across, scale3(axis_along, dot3(ia_across, axis_along))), "cross-track");
  const int64_t tile_pulses = span / n_tiles;
  const auto formable_across = [&](const std::array<double, 3>& srp,
                                   const std::array<double, 3>& tx) {
    const double a = proj(srp, axis_along);
    const double r0 = dist3(tx, srp);
    const auto los = unit3(sub3(srp, tx), "APC-SRP");
    const double los_across = std::abs(dot3(los, axis_across));
    if (!(los_across > 0.0) || !std::isfinite(los_across)) {
      throw std::runtime_error(
          "tiling range window mapped to an empty "
          "cross-track interval");
    }
    const double half_ground = half_slant / los_across;
    const double c_ref = proj(srp, axis_across);
    const auto in_win = [&](double c) {
      const auto p = add3(iarp, add3(scale3(axis_along, a), scale3(axis_across, c)));
      return std::abs(dist3(p, tx) - r0) <= half_slant;
    };
    if (!in_win(c_ref)) {
      throw std::runtime_error(
          "tiling: SRP projection is outside the range "
          "window on the lattice plane");
    }
    const auto edge = [&](double sign) {
      double lo = 0.0;
      double hi = 2.0 * half_ground;
      for (int i = 0; i < 40; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (in_win(c_ref + sign * mid)) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      return c_ref + sign * lo;
    };
    return std::array<double, 2>{edge(-1.0), edge(1.0)};
  };
  double cross_lo = 0.0;
  double cross_hi = 0.0;
  double inter_lo = 0.0;
  double inter_hi = 0.0;
  bool have_inter = false;
  const int64_t n_env = (n_tiles == 1) ? int64_t{11} : n_tiles;
  for (int64_t k = 0; k < n_env; ++k) {
    int64_t i0 = 0;
    int64_t i1 = 0;
    int64_t mid = 0;
    if (n_tiles == 1) {
      const int64_t i = first_pulse + k * (span - 1) / (n_env - 1);
      i0 = i1 = mid = std::min(i, last_pulse - 1);
    } else {
      i0 = first_pulse + k * tile_pulses;
      i1 = i0 + tile_pulses - 1;
      if (i1 >= last_pulse) {
        i1 = last_pulse - 1;
      }
      mid = i0 + tile_pulses / 2;
      if (mid >= last_pulse) {
        mid = last_pulse - 1;
      }
    }
    double clo = 0.0;
    double chi = 0.0;
    bool first = true;
    const int64_t pulses[3] = {i0, mid, i1};
    for (int64_t p = 0; p < 3; ++p) {
      const auto env = formable_across(srp_at(pulses[p]), tx_at(pulses[p]));
      if (first) {
        clo = env[0];
        chi = env[1];
        first = false;
      } else {
        clo = std::min(clo, env[0]);
        chi = std::max(chi, env[1]);
      }
      if (!have_inter) {
        inter_lo = env[0];
        inter_hi = env[1];
        have_inter = true;
      } else {
        inter_lo = std::max(inter_lo, env[0]);
        inter_hi = std::min(inter_hi, env[1]);
      }
    }
    if (!(chi > clo)) {
      throw std::runtime_error("tiling tile " + std::to_string(k + 1) +
                               " has an empty range-window envelope");
    }
    if (k == 0) {
      cross_lo = clo;
      cross_hi = chi;
    } else {
      cross_lo = std::min(cross_lo, clo);
      cross_hi = std::max(cross_hi, chi);
    }
  }
  const double spacing0 = phys.gsd_range_m / pixel_scale;
  const auto counts =
      counts_for_tiled_extent(a1 - a0, cross_hi - cross_lo, spacing0, n_tiles, max_side);
  StripmapTiling out;
  out.tile_pulses = tile_pulses;
  out.parent = lattice_from_image_area(iarp, axis_across, axis_along, cross_lo, a0, cross_hi, a1,
                                       counts[0], counts[1]);
  const int64_t nv_p = out.parent.num_v();
  const int64_t nu_p = out.parent.num_u();
  out.tiles.reserve(static_cast<size_t>(n_tiles));
  for (int64_t k = 0; k < n_tiles; ++k) {
    const int64_t v0 = k * nv_p / n_tiles;
    const int64_t v1 = (k + 1) * nv_p / n_tiles;
    const int64_t nv = v1 - v0;
    if (nv < 1) {
      throw std::runtime_error("lattice.max_side is too small to slice tiles along v");
    }
    radar::SampleGeometrySpec spec;
    spec.origin = out.parent.sample_position(v0, 0);
    spec.axis_u = out.parent.axis_u();
    spec.axis_v = out.parent.axis_v();
    spec.spacing = out.parent.spacing();
    spec.counts = {nv, nu_p};
    out.tiles.emplace_back(spec);
  }
  // Holoviz only: crop parent u to the intersection of the
  // sampled envelopes (plus a small inset) so the walking
  // range-window edge stays off-screen. BP / dump stay the
  // union parent. Skip if the core is empty or would drop
  // more than half the swath (a collapsing TOA window).
  if (have_inter && inter_hi > inter_lo) {
    const double dx = out.parent.spacing()[1];
    const auto u_at = [cross_lo, dx](double c) {
      return (c - cross_lo) / dx - 0.5;
    };
    int64_t u0 = static_cast<int64_t>(std::ceil(u_at(inter_lo)));
    int64_t u1 = static_cast<int64_t>(std::floor(u_at(inter_hi)));
    const int64_t inset = std::max(int64_t{16}, nu_p / 200);
    u0 += inset;
    u1 -= inset;
    if (u0 < 0) {
      u0 = 0;
    }
    if (u1 >= nu_p) {
      u1 = nu_p - 1;
    }
    const int64_t nu = u1 - u0 + 1;
    if (nu > nu_p / 2 && nu < nu_p) {
      out.display_u0 = u0;
      out.display_nu = nu;
    }
  }
  return out;
}

inline std::filesystem::path resolve_existing(const std::filesystem::path& given,
                                              const std::filesystem::path& yaml_dir) {
  if (given.empty()) {
    return {};
  }
  if (std::filesystem::is_regular_file(given)) {
    return std::filesystem::absolute(given);
  }
  const auto beside_yaml = yaml_dir / given;
  if (std::filesystem::is_regular_file(beside_yaml)) {
    return std::filesystem::absolute(beside_yaml);
  }
  return given;
}

inline AppConfig load_sar_config(const std::filesystem::path& path,
                                 const std::filesystem::path& cphd_override) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  const auto yaml_dir = path.parent_path();

  AppConfig cfg;
  std::vector<std::filesystem::path> cphd_files;
  if (!cphd_override.empty()) {
    cphd_files.push_back(resolve_existing(cphd_override, yaml_dir));
  } else if (const YAML::Node node = root["cphd"]) {
    if (const YAML::Node p = node["path"]) {
      if (p.IsSequence()) {
        for (const YAML::Node& item : p) {
          cphd_files.push_back(resolve_existing(item.as<std::string>(), yaml_dir));
        }
      } else {
        cphd_files.push_back(resolve_existing(p.as<std::string>(), yaml_dir));
      }
    }
  }
  for (const auto& cphd : cphd_files) {
    if (cphd.empty() || !std::filesystem::is_regular_file(cphd)) {
      throw std::runtime_error("cphd.path is empty or not a file: " + cphd.string());
    }
    cfg.cphd_paths.push_back(std::filesystem::absolute(cphd).string());
  }
  if (cfg.cphd_paths.empty()) {
    throw std::runtime_error("cphd.path must name at least one local CPHD file");
  }
  cfg.cphd_path = cfg.cphd_paths.front();
  const std::filesystem::path cphd(cfg.cphd_path);
  if (const YAML::Node node = root["cphd"]) {
    if (const YAML::Node n = node["names"]) {
      if (!n.IsSequence()) {
        throw std::runtime_error("cphd.names must be a list");
      }
      for (const YAML::Node& item : n) {
        cfg.scene_names.push_back(item.as<std::string>());
      }
    }
  }

  // Read-only mirror of operator keys that change dr. The CPHD
  // supplies df (SCSS) and num_samples; nfft / c come from the
  // same YAML block from_config() feeds the operator, when set.
  // nfft_pad: next_fast is app-owned (not an operator parameter).
  int64_t nfft = 0;
  double speed = radar::kSpeedOfLight;
  if (const YAML::Node rc = root["range_compression"]) {
    if (const YAML::Node n = rc["nfft"]) {
      nfft = n.as<int64_t>();
    }
    if (const YAML::Node p = rc["propagation_speed"]) {
      speed = p.as<double>();
    }
  }
  if (const YAML::Node pad = root["nfft_pad"]) {
    const auto policy = pad.as<std::string>();
    if (policy != "next_fast") {
      throw std::runtime_error(
          "nfft_pad must be 'next_fast' (omit the key for native "
          "N, or set range_compression.nfft to pin a length)");
    }
    if (nfft > 0) {
      throw std::runtime_error(
          "nfft_pad and range_compression.nfft are mutually "
          "exclusive");
    }
    int64_t max_samples = 0;
    for (const auto& p : cfg.cphd_paths) {
      radar::cphd::CphdFile file(p);
      if (file.num_channels() < 1) {
        throw std::runtime_error("nfft_pad: no channels in " + p);
      }
      max_samples = std::max(max_samples, file.channel(0).num_samples);
    }
    if (max_samples < 1) {
      throw std::runtime_error("nfft_pad: no CPHD has num_samples >= 1");
    }
    nfft = radar::next_fast_fft_size(max_samples);
  }
  cfg.nfft = nfft;

  std::array<int64_t, 2> lattice_counts{};
  std::array<double, 2> lattice_spacing{};
  const bool has_lattice = static_cast<bool>(root["lattice"]);
  if (has_lattice) {
    if (root["sample_geometry"]) {
      throw std::runtime_error(
          "lattice and sample_geometry are mutually exclusive: "
          "lattice derives an ECEF grid at each file's IARP");
    }
    const YAML::Node lat = root["lattice"];
    const bool has_counts = static_cast<bool>(lat["counts"]);
    const bool has_spacing = static_cast<bool>(lat["spacing"]);
    if (has_counts != has_spacing) {
      throw std::runtime_error(
          "lattice.counts and lattice.spacing must be set "
          "together (or omit both and derive from collection "
          "physics)");
    }
    int64_t n_tiles = 1;
    if (const YAML::Node t = lat["tiles"]) {
      n_tiles = t.as<int64_t>();
      if (n_tiles < 1) {
        throw std::runtime_error("lattice.tiles must be >= 1");
      }
    }
    if (n_tiles > 1) {
      if (cfg.cphd_paths.size() != 1) {
        throw std::runtime_error(
            "lattice.tiles is one-file (mode 3); mode 5 uses "
            "one lattice per cphd.path instead");
      }
      if (has_counts) {
        throw std::runtime_error(
            "lattice.tiles builds an SRP-track parent strip; "
            "omit counts/spacing");
      }
    }
    if (const YAML::Node sc = lat["scroll"]) {
      cfg.scroll = sc.as<bool>();
    }
    if (cfg.scroll) {
      if (n_tiles > 1) {
        throw std::runtime_error(
            "lattice.scroll and lattice.tiles are mutually "
            "exclusive (mode 4 is one window, not tiles)");
      }
      if (cfg.cphd_paths.size() != 1) {
        throw std::runtime_error("lattice.scroll is one-file (mode 4)");
      }
      if (has_counts) {
        throw std::runtime_error(
            "lattice.scroll builds an SRP-track strip window; "
            "omit counts/spacing");
      }
    }
    if (const YAML::Node side = lat["max_side"]) {
      cfg.lattice_max_side = side.as<int64_t>();
      if (cfg.lattice_max_side < 0) {
        throw std::runtime_error("lattice.max_side must be >= 0 (0 = no cap)");
      }
    }
    std::vector<double> pixel_scale(cfg.cphd_paths.size(), 1.0);
    if (const YAML::Node sc = lat["pixel_scale"]) {
      if (has_counts) {
        throw std::runtime_error(
            "lattice.pixel_scale applies to collection-physics "
            "derivation; omit it when counts/spacing are set");
      }
      if (sc.IsScalar()) {
        const double s = sc.as<double>();
        if (!(s > 0.0) || !std::isfinite(s)) {
          throw std::runtime_error("lattice.pixel_scale must be finite and > 0");
        }
        std::fill(pixel_scale.begin(), pixel_scale.end(), s);
      } else if (sc.IsSequence()) {
        const auto values = sc.as<std::vector<double>>();
        if (values.size() != cfg.cphd_paths.size()) {
          throw std::runtime_error("lattice.pixel_scale length must match cphd.path (" +
                                   std::to_string(cfg.cphd_paths.size()) + " files, got " +
                                   std::to_string(values.size()) + ")");
        }
        for (double s : values) {
          if (!(s > 0.0) || !std::isfinite(s)) {
            throw std::runtime_error(
                "lattice.pixel_scale entries must be finite "
                "and > 0");
          }
        }
        pixel_scale = values;
      } else {
        throw std::runtime_error("lattice.pixel_scale must be a number or a list");
      }
    }
    cfg.lattice_pixel_scale = pixel_scale;
    if (has_counts) {
      lattice_counts = parse_lattice_counts(lat["counts"]);
      const YAML::Node sp = lat["spacing"];
      if (sp.IsScalar()) {
        const double metres = sp.as<double>();
        if (!(metres > 0.0) || !std::isfinite(metres)) {
          throw std::runtime_error("lattice.spacing must be finite and > 0");
        }
        lattice_spacing = {metres, metres};
      } else {
        const auto values = sp.as<std::vector<double>>();
        if (values.size() != 2 || !(values[0] > 0.0) || !(values[1] > 0.0) ||
            !std::isfinite(values[0]) || !std::isfinite(values[1])) {
          throw std::runtime_error("lattice.spacing must be metres or [v, u], all > 0");
        }
        lattice_spacing = {values[0], values[1]};
      }
    }
    if (cfg.cphd_paths.size() > 1 && root["enu_origin_ecef"]) {
      throw std::runtime_error(
          "enu_origin_ecef is a single-scene rotate; mode 5 "
          "forms in ECEF at each IARP instead");
    }
    for (size_t i = 0; i < cfg.cphd_paths.size(); ++i) {
      const std::string& path = cfg.cphd_paths[i];
      radar::cphd::CphdFile file(path);
      if (!file.has_iarp()) {
        throw std::runtime_error("lattice needs SceneCoordinates/IARP/ECF in " + path);
      }
      const double dr = range_bin_spacing_from_cphd(path, nfft, speed);
      radar::SampleGeometry geom;
      if (cfg.scroll) {
        if (!file.has_collection_physics()) {
          throw std::runtime_error(
              "lattice.scroll needs collection physics "
              "(GrazeAngle / SlantRange / FxBW / TOASaved) in " +
              path);
        }
        const auto phys = file.collection_physics();
        int64_t first_pulse = 0;
        int64_t last_pulse = file.channel(0).num_vectors;
        if (const YAML::Node rd = root["cphd_reader"]) {
          if (const YAML::Node pr = rd["pulse_range"]) {
            const auto range = pr.as<std::vector<int64_t>>();
            if (range.size() != 2) {
              throw std::runtime_error("cphd_reader.pulse_range needs 2 entries");
            }
            first_pulse = range[0];
            last_pulse = range[1];
          }
        }
        const int64_t num_bins = nfft > 0 ? nfft : file.channel(0).num_samples;
        if (num_bins < 1) {
          throw std::runtime_error("scroll needs num_samples >= 1 in " + path);
        }
        const double half_slant = 0.5 * static_cast<double>(num_bins) * dr;
        const auto tiling = stripmap_tile_lattices(file, 1, 0, pixel_scale[i], phys, first_pulse,
                                                   last_pulse, half_slant);
        cfg.scroll_parent = tiling.parent;
        cfg.viz_crop_u0 = tiling.display_u0;
        cfg.viz_crop_nu = tiling.display_nu;
        int64_t buf = cfg.lattice_max_side;
        if (buf < 1) {
          buf = tiling.parent.num_v();
        }
        if (buf > tiling.parent.num_v()) {
          buf = tiling.parent.num_v();
        }
        radar::SampleGeometrySpec spec;
        spec.origin = tiling.parent.origin();
        spec.axis_u = tiling.parent.axis_u();
        spec.axis_v = tiling.parent.axis_v();
        spec.spacing = tiling.parent.spacing();
        spec.counts = {buf, tiling.parent.num_u()};
        geom = radar::SampleGeometry(spec);
        cfg.scroll_anchor = buf / 2;
        cfg.collection_physics.push_back(phys);
      } else if (n_tiles > 1) {
        if (!file.has_collection_physics()) {
          throw std::runtime_error(
              "lattice.tiles needs collection physics "
              "(GrazeAngle / SlantRange / FxBW / TOASaved) in " +
              path);
        }
        const auto phys = file.collection_physics();
        int64_t first_pulse = 0;
        int64_t last_pulse = file.channel(0).num_vectors;
        if (const YAML::Node rd = root["cphd_reader"]) {
          if (const YAML::Node pr = rd["pulse_range"]) {
            const auto range = pr.as<std::vector<int64_t>>();
            if (range.size() != 2) {
              throw std::runtime_error("cphd_reader.pulse_range needs 2 entries");
            }
            first_pulse = range[0];
            last_pulse = range[1];
          }
        }
        const int64_t num_bins = nfft > 0 ? nfft : file.channel(0).num_samples;
        if (num_bins < 1) {
          throw std::runtime_error("tiling needs num_samples >= 1 in " + path);
        }
        const double half_slant = 0.5 * static_cast<double>(num_bins) * dr;
        const auto tiling =
            stripmap_tile_lattices(file, n_tiles, cfg.lattice_max_side, pixel_scale[i], phys,
                                   first_pulse, last_pulse, half_slant);
        cfg.tile_geometries = tiling.tiles;
        cfg.parent_geometry = tiling.parent;
        cfg.viz_crop_u0 = tiling.display_u0;
        cfg.viz_crop_nu = tiling.display_nu;
        cfg.tile_pulses = tiling.tile_pulses;
        geom = cfg.tile_geometries.front();
        cfg.collection_physics.push_back(phys);
      } else if (has_counts) {
        geom = ecef_lattice_at_iarp(file.iarp_ecef(), lattice_counts, lattice_spacing);
        if (cfg.lattice_max_side > 0) {
          const auto capped = counts_for_extent(
              static_cast<double>(lattice_counts[0]) * lattice_spacing[0],
              static_cast<double>(lattice_counts[1]) * lattice_spacing[1],
              std::min(lattice_spacing[0], lattice_spacing[1]), cfg.lattice_max_side);
          const double ext_v = static_cast<double>(lattice_counts[0]) * lattice_spacing[0];
          const double ext_u = static_cast<double>(lattice_counts[1]) * lattice_spacing[1];
          const std::array<double, 9> rotation = ecef_to_enu_rotation(file.iarp_ecef());
          const std::array<double, 3> east{rotation[0], rotation[1], rotation[2]};
          const std::array<double, 3> north{rotation[3], rotation[4], rotation[5]};
          geom = lattice_from_image_area(file.iarp_ecef(), east, {-north[0], -north[1], -north[2]},
                                         -ext_u / 2.0, -ext_v / 2.0, ext_u / 2.0, ext_v / 2.0,
                                         capped[0], capped[1]);
        }
      } else {
        if (!file.has_collection_physics()) {
          throw std::runtime_error(
              "lattice derivation needs collection physics "
              "(GrazeAngle / SlantRange / AzimuthAngle, FxC / "
              "FxBW / TOASaved, Antenna GainZero, PVP SIGNAL / "
              "TxPos / SRPPos) in " +
              path);
        }
        const auto phys = file.collection_physics();
        geom = ecef_physics_lattice(file.iarp_ecef(), phys, cfg.lattice_max_side, pixel_scale[i]);
        cfg.collection_physics.push_back(phys);
      }
      cfg.collection_geometries.push_back(std::move(geom));
      cfg.collection_spacings.push_back(dr);
      cfg.collection_frequencies.push_back(reference_frequency_from_cphd(path));
    }
    cfg.sample_geometry = cfg.collection_geometries.front();
    cfg.range_bin_spacing = cfg.collection_spacings.front();
    cfg.reference_frequency = cfg.collection_frequencies.front();
  }

  if (!has_lattice) {
    if (cfg.cphd_paths.size() > 1) {
      throw std::runtime_error(
          "a multi-file cphd.path needs a lattice block "
          "(collection physics, or counts/spacing)");
    }
    cfg.range_bin_spacing = range_bin_spacing_from_cphd(cfg.cphd_path, nfft, speed);
  }

  if (!has_lattice) {
    std::filesystem::path sidecar;
    if (const YAML::Node node = root["cphd"]) {
      if (const YAML::Node s = node["sidecar"]) {
        const std::string given = s.as<std::string>();
        if (!given.empty()) {
          sidecar = resolve_existing(given, yaml_dir);
        }
      }
    }
    if (sidecar.empty()) {
      const auto guessed = cphd.parent_path() / (cphd.stem().string() + "_truth.json");
      if (std::filesystem::is_regular_file(guessed)) {
        sidecar = guessed;
      }
    }

    // YAML sample_geometry wins when present. A guessed sidecar must
    // not silently replace the Atlanta ENU lattice with a leftover
    // fixture file. Sidecar is the source only when the YAML omits
    // the lattice (generated smoke / spotlight fixtures).
    const bool yaml_has_lattice = static_cast<bool>(root["sample_geometry"]);
    if (!yaml_has_lattice && !sidecar.empty() && std::filesystem::is_regular_file(sidecar)) {
      cfg.sidecar_path = sidecar.string();
      // JSON is valid YAML. The converter runs SampleGeometry's
      // constructor, so a broken lattice fails here rather than
      // inside the operator.
      const YAML::Node truth = YAML::LoadFile(sidecar.string());
      const YAML::Node schema = truth["schema"];
      if (!schema || schema.as<std::string>() != "holoscan-radar/sar-sim-truth/1") {
        throw std::runtime_error("sidecar schema is not holoscan-radar/sar-sim-truth/1: " +
                                 sidecar.string());
      }
      if (!truth["sample_geometry"]) {
        throw std::runtime_error("sidecar is missing sample_geometry: " + sidecar.string());
      }
      cfg.sample_geometry = truth["sample_geometry"].as<radar::SampleGeometry>();
      if (!truth["collection"] || !truth["collection"]["fx_min_hz"] ||
          !truth["collection"]["fx_spacing_hz"] || !truth["collection"]["num_samples"]) {
        throw std::runtime_error(
            "sidecar is missing collection.fx_min_hz / "
            "fx_spacing_hz / num_samples: " +
            sidecar.string());
      }
      // Stored sample N/2, matching SARRangeCompressionOp's IFFT DC.
      const double fx_min = truth["collection"]["fx_min_hz"].as<double>();
      const double fx_step = truth["collection"]["fx_spacing_hz"].as<double>();
      const int64_t n_samp = truth["collection"]["num_samples"].as<int64_t>();
      cfg.reference_frequency = fx_min + static_cast<double>(n_samp / 2) * fx_step;
    } else {
      if (!root["sample_geometry"]) {
        throw std::runtime_error("no sidecar at " +
                                 (sidecar.empty() ? std::string("<none>") : sidecar.string()) +
                                 " and YAML is missing sample_geometry / "
                                 "reference_frequency");
      }
      cfg.sample_geometry = root["sample_geometry"].as<radar::SampleGeometry>();
      cfg.reference_frequency = root["reference_frequency"]
                                    ? root["reference_frequency"].as<double>()
                                    : reference_frequency_from_cphd(cfg.cphd_path);
    }
    cfg.collection_geometries = {cfg.sample_geometry};
    cfg.collection_spacings = {cfg.range_bin_spacing};
    cfg.collection_frequencies = {cfg.reference_frequency};
  }

  if (const YAML::Node origin = root["enu_origin_ecef"]) {
    const auto values = origin.as<std::vector<double>>();
    if (values.size() != 3) {
      throw std::runtime_error("enu_origin_ecef must have 3 ECEF metres");
    }
    cfg.enu_origin_ecef = {values[0], values[1], values[2]};
    cfg.transform_to_scene_enu = true;
  }
  if (const YAML::Node flag = root["quantize_phase_center_f32"]) {
    cfg.quantize_phase_center_f32 = flag.as<bool>();
  }
  if (const YAML::Node vph = root["nvradar_vph_path"]) {
    cfg.nvradar_vph_path = vph.as<std::string>();
  }

  if (const YAML::Node v = root["dump"]; v && !v.IsNull()) {
    cfg.dump = v.as<bool>();
  }
  if (const YAML::Node output = root["output"]) {
    if (const YAML::Node p = output["path"]) {
      const std::string given = p.as<std::string>();
      if (!given.empty()) {
        cfg.dump_path = given;
      }
    }
    if (const YAML::Node m = output["max_files"]) {
      cfg.dump_max_files = m.as<int64_t>();
    }
  }
  if (cfg.dump && cfg.dump_path.empty()) {
    cfg.dump_path = (cphd.parent_path() / (cphd.stem().string() + "_image_{index}.npy")).string();
  }

  if (const YAML::Node am = root["aperture_manager"]) {
    if (const YAML::Node m = am["max_aperture_pulses"]) {
      const auto max_ap = m.as<int64_t>();
      cfg.show_playback_lag = max_ap > 0;
      if (max_ap > 0) {
        cfg.wire_weight = true;
      }
    }
    if (const YAML::Node w = am["aperture_window"]) {
      const auto spec = w.as<radar::WindowSpec>();
      if (spec.type != radar::WindowType::kNone) {
        cfg.wire_weight = true;
      }
    }
    if (const YAML::Node e = am["emit_interval_pulses"]) {
      cfg.emit_interval_pulses = e.as<int64_t>();
    }
  }
  if (cfg.scroll || cfg.parent_geometry.is_valid()) {
    cfg.show_playback_lag = true;
  }
  // Read-only mirrors of operator YAML so the tile stager can
  // wrap and so cadence tiles stay even (one tile = N reader
  // ticks). from_config() still owns the operator parameters.
  if (const YAML::Node rd = root["cphd_reader"]) {
    if (const YAML::Node lp = rd["loop"]; lp && !lp.IsNull()) {
      cfg.reader_loop = lp.as<bool>();
    }
    if (const YAML::Node p = rd["pulses_per_tick"]) {
      cfg.reader_pulses_per_tick = p.as<int64_t>();
    }
  }
  if (cfg.tile_pulses > 0) {
    if (cfg.emit_interval_pulses != 0 && cfg.emit_interval_pulses != cfg.tile_pulses) {
      throw std::runtime_error(
          "aperture_manager.emit_interval_pulses must match "
          "vector_count / lattice.tiles, after pulse_range (" +
          std::to_string(cfg.tile_pulses) + ")");
    }
    if (cfg.reader_pulses_per_tick > 0 && cfg.tile_pulses % cfg.reader_pulses_per_tick != 0) {
      throw std::runtime_error(
          "cphd_reader.pulses_per_tick must divide the tile "
          "length (" +
          std::to_string(cfg.tile_pulses) + " pulses) so cadence tiles stay even");
    }
  }
  {
    if (cfg.parent_geometry.is_valid() || cfg.scroll) {
      if (cfg.scene_names.size() > 1) {
        throw std::runtime_error(
            "cphd.names must be one name (or omitted) when "
            "tiling a mosaic");
      }
      if (cfg.scene_names.empty()) {
        cfg.scene_names.emplace_back("strip");
      }
    } else {
      const size_t name_n = cfg.cphd_paths.size();
      if (cfg.scene_names.size() == 1 && name_n > 1) {
        const std::string base = cfg.scene_names.front();
        cfg.scene_names.clear();
        for (size_t i = 0; i < name_n; ++i) {
          cfg.scene_names.push_back(base + " " + std::to_string(i + 1) + "/" +
                                    std::to_string(name_n));
        }
      } else if (!cfg.scene_names.empty() && cfg.scene_names.size() != name_n) {
        throw std::runtime_error("cphd.names length must match cphd.path");
      }
    }
  }
  if (const YAML::Node v = root["visualize"]; v && !v.IsNull()) {
    cfg.visualize = v.as<bool>();
  }
  if (const YAML::Node v = root["visualize_size"]; v && !v.IsNull()) {
    cfg.visualize_size = v.as<int64_t>();
  }
  if (cfg.visualize_size <= 0) {
    throw std::runtime_error("visualize_size must be > 0");
  }
  if (const YAML::Node v = root["skip_display_check"]; v && !v.IsNull()) {
    cfg.skip_display_check = v.as<bool>();
  }
  if (const YAML::Node v = root["headless"]; v && !v.IsNull()) {
    cfg.headless = v.as<bool>();
  }
  if (const YAML::Node v = root["visualize_hold_s"]; v && !v.IsNull()) {
    cfg.visualize_hold_s = v.as<double>();
  }
  if (!std::isfinite(cfg.visualize_hold_s)) {
    throw std::runtime_error("visualize_hold_s must be finite");
  }
  if (const YAML::Node v = root["wait_for_downstream"]; v && !v.IsNull()) {
    cfg.wait_for_downstream = v.as<bool>();
  }
  return cfg;
}
