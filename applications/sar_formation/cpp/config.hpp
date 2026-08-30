// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned config for sar_formation: the AppConfig type and its YAML
// loader. Colocated with the app; the graph wiring lives in
// sar_formation.cpp. Operator blocks are NOT parsed here — they flow
// through Holoscan's from_config(). Exception: nfft and
// propagation_speed under range_compression are a read-only mirror
// so range_bin_spacing matches the operator if those keys are set.
// App-owned `nfft_pad: next_fast` computes nfft as
// next_fast_fft_size(max CPHD num_samples) and compose() passes it
// as Arg("nfft"); mutually exclusive with range_compression.nfft.
//
// Lattice: `sample_geometry` (Atlanta ENU), sidecar truth, or
// `lattice` — explicit counts/spacing (IARP ENU; counts may be N
// or [num_v, num_u]) or collection physics (XY square at the
// coarser GSD) with `max_side` as a formed-pixel cap (the
// slower-GPU knob) and `pixel_scale` as a per-collect factor on N.
// A guessed sidecar must not override an explicit YAML lattice.

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
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/dsp/fft_size.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace radar = holoscan::radar;

struct AppConfig {
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
  // Mode 5: one lattice / dr / f0 per collection, aligned with
  // cphd_paths. Mode 1 has a single entry matching the YAML lattice.
  std::vector<radar::SampleGeometry> collection_geometries;
  std::vector<double> collection_spacings;
  std::vector<double> collection_frequencies;
  // Timing cap on each physics-derived (or explicit-count) lattice
  // axis. 0 = no cap (native collection-physics N, or the YAML
  // counts). A positive cap coarsens GSD and keeps the square.
  // This is the formation-cost knob: lower it on a slower GPU.
  int64_t lattice_max_side{0};
  // Per-collection factor on native N (both axes). 1 = native.
  // Same square, coarser GSD. Size matches cphd_paths.
  std::vector<double> lattice_pixel_scale;
  // Filled when `lattice` omits counts/spacing so compose can log
  // heading / swath / GSD. Empty on the explicit-count path.
  std::vector<radar::cphd::CphdFile::CollectionPhysics>
      collection_physics;
  // When set, compose() rotates reader ECEF positions / SRP into
  // scene ENU at this IARP so sample_geometry can be an ENU lattice
  // (NVRadar Atlanta grid diagnostic).
  bool transform_to_scene_enu{false};
  std::array<double, 3> enu_origin_ecef{};
  // Diagnostic: round ENU APC through float32 (NVRadar .dat xyz).
  bool quantize_phase_center_f32{false};
  // Diagnostic: replace contract RC with their conditioned vph.
  std::string nvradar_vph_path;
  bool visualize{false};
  // Wire manager.weight → BP.weight. True when the manager
  // retains (max_aperture_pulses > 0) so eviction can emit −1.
  // Modes 1 / 5 stay unconnected (all +1, no D2H of weights).
  bool wire_weight{false};
  bool skip_display_check{false};
  bool headless{false};
  // Holoviz window (px, square). LUT box-averages the formed
  // image to this so R8_UINT is 1:1; a dump (dump: true) is the
  // BP lattice. 1024 fits a 1080p screen.
  int64_t visualize_size{1024};
  // Seconds the Holoviz window keeps the formed image after emit.
  // Also the max wait for the first frame (Atlanta ~9 s). Mode 1
  // emits once; 0 closes with the graph. Negative: hold until
  // SIGINT (mode 5 looping rotate).
  double visualize_hold_s{30.0};
};

struct CliPaths {
  std::filesystem::path config;
  std::filesystem::path cphd_override;
};

// First CLI argument is the YAML, unless it is a .cphd — then the
// staged default YAML is used and that file is the collection. A
// second argument always overrides cphd.path.
inline CliPaths parse_cli(int argc, char** argv,
                          const char* default_yaml) {
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
    out.config = std::filesystem::is_regular_file(beside) ? beside
                                                          : first;
  }
  if (argc > 2) {
    out.cphd_override = argv[2];
  }
  return out;
}

// dr = c / (2 * N_bins * df), df = SCSS. Same formula RC uses when
// bandwidth is the CPHD span (N_samples - 1)*SCSS. nfft_or_0 == 0
// means N_bins = num_samples (the mode-1 default).
inline double range_bin_spacing_from_cphd(const std::string& path,
                                          int64_t nfft_or_0,
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
    throw std::runtime_error(
        "collection has fewer than 2 samples; no range spacing: " +
        path);
  }
  if (!file.has_pvp_field("SCSS")) {
    throw std::runtime_error("collection has no SCSS PVP: " + path);
  }
  const double df = file.read_pvp(0, "SCSS", 0, 1, 1).at(0);
  if (!(df > 0.0) || !std::isfinite(df)) {
    throw std::runtime_error(
        "SCSS must be finite and > 0 Hz: " + path);
  }
  const int64_t num_bins =
      nfft_or_0 > 0 ? nfft_or_0 : num_samples;
  if (num_bins < num_samples) {
    throw std::runtime_error(
        "nfft must be >= num_samples for range_bin_spacing");
  }
  return propagation_speed /
         (2.0 * static_cast<double>(num_bins) * df);
}

// Stored sample N/2, matching SARRangeCompressionOp's IFFT DC after
// ifftshift. Same formula mode01.yaml hardcodes for Atlanta.
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
    throw std::runtime_error(
        "collection has fewer than 2 samples; no f0: " + path);
  }
  if (!file.has_pvp_field("SC0") || !file.has_pvp_field("SCSS")) {
    throw std::runtime_error(
        "collection has no SC0/SCSS PVP: " + path);
  }
  const double sc0 = file.read_pvp(0, "SC0", 0, 1, 1).at(0);
  const double df = file.read_pvp(0, "SCSS", 0, 1, 1).at(0);
  if (!std::isfinite(sc0) || !std::isfinite(df)) {
    throw std::runtime_error(
        "SC0/SCSS must be finite: " + path);
  }
  return sc0 + static_cast<double>(num_samples / 2) * df;
}

// Bowring WGS-84 + ECEF→ENU, matching EcefToSceneEnuOp. Rows are
// east, north, up in ECEF.
inline std::array<double, 9> ecef_to_enu_rotation(
    const std::array<double, 3>& ecef) {
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
  const double lat = std::atan2(
      z + e2 * (kWgs84A * kWgs84A / b) * slt * slt * slt,
      p - e2 * kWgs84A * clt * clt * clt);
  const double sl = std::sin(lat);
  const double cl = std::cos(lat);
  const double so = std::sin(lon);
  const double co = std::cos(lon);
  return {-so,      co,     0.0,
          -sl * co, -sl * so, cl,
          cl * co,  cl * so,  sl};
}

// Lattice centre (half-index along each axis), same frame as
// origin / axes. Used to project an APC onto east/north of the
// displayed map (axis_u = east, axis_v = south).
inline std::array<double, 3> lattice_centre(
    const radar::SampleGeometry& geom) {
  const double cv =
      0.5 * static_cast<double>(geom.num_v() - 1);
  const double cu =
      0.5 * static_cast<double>(geom.num_u() - 1);
  const auto& o = geom.origin();
  const auto& av = geom.axis_v();
  const auto& au = geom.axis_u();
  const double dv = cv * geom.spacing()[0];
  const double du = cu * geom.spacing()[1];
  return {o[0] + dv * av[0] + du * au[0],
          o[1] + dv * av[1] + du * au[1],
          o[2] + dv * av[2] + du * au[2]};
}

// Horizontal look of an APC relative to the scene: metres east
// and north of the lattice centre, in the displayed map frame
// (row 0 = north, col 0 = west). `apc_xyz` is ECEF unless
// `to_scene_enu`, in which case it is rotated at `enu_origin`
// first (Atlanta NVRadar path).
inline std::optional<std::array<double, 2>> apc_east_north(
    std::array<double, 3> apc_xyz,
    const radar::SampleGeometry& geom, bool to_scene_enu,
    const std::array<double, 3>& enu_origin) {
  if (!geom.is_valid()) {
    return std::nullopt;
  }
  if (to_scene_enu) {
    const auto R = ecef_to_enu_rotation(enu_origin);
    const double dx = apc_xyz[0] - enu_origin[0];
    const double dy = apc_xyz[1] - enu_origin[1];
    const double dz = apc_xyz[2] - enu_origin[2];
    apc_xyz = {R[0] * dx + R[1] * dy + R[2] * dz,
               R[3] * dx + R[4] * dy + R[5] * dz,
               R[6] * dx + R[7] * dy + R[8] * dz};
  }
  const auto c = lattice_centre(geom);
  const double d0 = apc_xyz[0] - c[0];
  const double d1 = apc_xyz[1] - c[1];
  const double d2 = apc_xyz[2] - c[2];
  const auto& au = geom.axis_u();
  const auto& av = geom.axis_v();
  const double east = d0 * au[0] + d1 * au[1] + d2 * au[2];
  const double south = d0 * av[0] + d1 * av[1] + d2 * av[2];
  const double north = -south;
  if (!std::isfinite(east) || !std::isfinite(north)) {
    return std::nullopt;
  }
  if (east * east + north * north < 1.0) {
    return std::nullopt;
  }
  return std::array<double, 2>{east, north};
}

// Pixel-centre lattice covering the IA rectangle [x1, x2] × [y1, y2]
// metres from IARP along unit uIAX / uIAY. counts are (num_v, num_u)
// along IAY / IAX. Sample (0, 0) is the min-X min-Y pixel centre.
inline radar::SampleGeometry lattice_from_image_area(
    const std::array<double, 3>& iarp,
    const std::array<double, 3>& u_iax,
    const std::array<double, 3>& u_iay, double x1, double y1,
    double x2, double y2, int64_t num_v, int64_t num_u) {
  if (!(x2 > x1) || !(y2 > y1)) {
    throw std::runtime_error(
        "image area must have x2 > x1 and y2 > y1");
  }
  if (num_v <= 0 || num_u <= 0) {
    throw std::runtime_error(
        "lattice counts must be [num_v, num_u], both > 0");
  }
  const double dx =
      (x2 - x1) / static_cast<double>(num_u);
  const double dy =
      (y2 - y1) / static_cast<double>(num_v);
  const double x0 = x1 + 0.5 * dx;
  const double y0 = y1 + 0.5 * dy;
  radar::SampleGeometrySpec spec;
  spec.origin = {
      iarp[0] + x0 * u_iax[0] + y0 * u_iay[0],
      iarp[1] + x0 * u_iax[1] + y0 * u_iay[1],
      iarp[2] + x0 * u_iax[2] + y0 * u_iay[2]};
  spec.axis_u = u_iax;
  spec.axis_v = u_iay;
  spec.spacing = {dy, dx};
  spec.counts = {num_v, num_u};
  return radar::SampleGeometry(spec);
}

// Square N, or [num_v, num_u].
inline std::array<int64_t, 2> parse_lattice_counts(
    const YAML::Node& node) {
  if (node.IsScalar()) {
    const int64_t n = node.as<int64_t>();
    if (n <= 0) {
      throw std::runtime_error(
          "lattice.counts must be > 0");
    }
    return {n, n};
  }
  if (!node.IsSequence()) {
    throw std::runtime_error(
        "lattice.counts must be N or [num_v, num_u]");
  }
  const auto counts = node.as<std::vector<int64_t>>();
  if (counts.size() != 2 || counts[0] <= 0 || counts[1] <= 0) {
    throw std::runtime_error(
        "lattice.counts must be N or [num_v, num_u], all > 0");
  }
  return {counts[0], counts[1]};
}

// Range-Nyquist counts that fill the extents, then cap the longer
// side at max_side (0 = no cap). Timing knob: lower max_side
// coarsens spacing and keeps the declared scene.
inline std::array<int64_t, 2> counts_for_extent(
    double extent_v, double extent_u, double native_spacing,
    int64_t max_side) {
  if (!(native_spacing > 0.0) || !std::isfinite(native_spacing)) {
    throw std::runtime_error(
        "native lattice spacing must be finite and > 0");
  }
  if (!(extent_v > 0.0) || !(extent_u > 0.0)) {
    throw std::runtime_error("image area extents must be > 0");
  }
  auto num_v = std::max(
      int64_t{1},
      static_cast<int64_t>(std::llround(extent_v / native_spacing)));
  auto num_u = std::max(
      int64_t{1},
      static_cast<int64_t>(std::llround(extent_u / native_spacing)));
  const int64_t longest = std::max(num_v, num_u);
  if (max_side > 0 && longest > max_side) {
    const double scale = static_cast<double>(longest) /
                         static_cast<double>(max_side);
    num_v = std::max(
        int64_t{1},
        static_cast<int64_t>(std::llround(
            static_cast<double>(num_v) / scale)));
    num_u = std::max(
        int64_t{1},
        static_cast<int64_t>(std::llround(
            static_cast<double>(num_u) / scale)));
  }
  return {num_v, num_u};
}
inline radar::SampleGeometry ecef_lattice_at_iarp(
    const std::array<double, 3>& iarp,
    const std::array<int64_t, 2>& counts,
    const std::array<double, 2>& spacing) {
  const std::array<double, 9> rotation = ecef_to_enu_rotation(iarp);
  const std::array<double, 3> east{rotation[0], rotation[1],
                                   rotation[2]};
  const std::array<double, 3> north{rotation[3], rotation[4],
                                    rotation[5]};
  const std::array<double, 3> south{-north[0], -north[1],
                                    -north[2]};
  const double ext_v =
      static_cast<double>(counts[0]) * spacing[0];
  const double ext_u =
      static_cast<double>(counts[1]) * spacing[1];
  return lattice_from_image_area(iarp, east, south, -ext_u / 2.0,
                                 -ext_v / 2.0, ext_u / 2.0,
                                 ext_v / 2.0, counts[0],
                                 counts[1]);
}

// IARP-centred east-north square from collection_physics().
// pixel_scale multiplies native N (both axes). max_side 0 then
// keeps that N at the coarser GSD; a positive cap (after scale)
// keeps the same square and coarsens spacing.
inline radar::SampleGeometry ecef_physics_lattice(
    const std::array<double, 3>& iarp,
    const radar::cphd::CphdFile::CollectionPhysics& phys,
    int64_t max_side, double pixel_scale) {
  if (!(pixel_scale > 0.0) || !std::isfinite(pixel_scale)) {
    throw std::runtime_error(
        "lattice.pixel_scale must be finite and > 0");
  }
  int64_t n = std::max(
      int64_t{1},
      static_cast<int64_t>(std::llround(
          static_cast<double>(phys.native_count) * pixel_scale)));
  if (max_side > 0 && n > max_side) {
    n = max_side;
  }
  const double spacing =
      (n == phys.native_count)
          ? phys.gsd_m
          : phys.xy_square_m / static_cast<double>(n);
  return ecef_lattice_at_iarp(iarp, {n, n}, {spacing, spacing});
}

inline std::filesystem::path resolve_existing(
    const std::filesystem::path& given,
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

inline AppConfig load_config(const std::filesystem::path& path,
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
          cphd_files.push_back(
              resolve_existing(item.as<std::string>(), yaml_dir));
        }
      } else {
        cphd_files.push_back(
            resolve_existing(p.as<std::string>(), yaml_dir));
      }
    }
  }
  for (const auto& cphd : cphd_files) {
    if (cphd.empty() || !std::filesystem::is_regular_file(cphd)) {
      throw std::runtime_error(
          "cphd.path is empty or not a file (" + cphd.string() +
          "). Mode 1 wants the Atlanta Umbra collect "
          "(see mode01.yaml). Mode 2: subset Houston first "
          "(mode02.yaml). Mode 5: run pull_cphd.py first. "
          "For the generated smoke fixture:\n"
          "  python3 tools/sar_sim/generate_cphd.py "
          "tools/sar_sim/recipes/mode01_smoke.yaml -o <dir>\n"
          "  sar_formation mode01_smoke.yaml "
          "<dir>/mode01_smoke.cphd");
    }
    cfg.cphd_paths.push_back(
        std::filesystem::absolute(cphd).string());
  }
  if (cfg.cphd_paths.empty()) {
    throw std::runtime_error(
        "cphd.path is empty or not a file. Mode 1 wants the "
        "Atlanta Umbra collect (see mode01.yaml). Mode 2: "
        "subset Houston (mode02.yaml). For the generated "
        "smoke fixture:\n"
        "  python3 tools/sar_sim/generate_cphd.py "
        "tools/sar_sim/recipes/mode01_smoke.yaml -o <dir>\n"
        "  sar_formation mode01_smoke.yaml "
        "<dir>/mode01_smoke.cphd");
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
      if (cfg.scene_names.size() != cfg.cphd_paths.size()) {
        throw std::runtime_error(
            "cphd.names length must match cphd.path");
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
        throw std::runtime_error(
            "nfft_pad: no channels in " + p);
      }
      max_samples =
          std::max(max_samples, file.channel(0).num_samples);
    }
    if (max_samples < 1) {
      throw std::runtime_error(
          "nfft_pad: no CPHD has num_samples >= 1");
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
    if (const YAML::Node side = lat["max_side"]) {
      cfg.lattice_max_side = side.as<int64_t>();
      if (cfg.lattice_max_side < 0) {
        throw std::runtime_error(
            "lattice.max_side must be >= 0 (0 = no cap)");
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
          throw std::runtime_error(
              "lattice.pixel_scale must be finite and > 0");
        }
        std::fill(pixel_scale.begin(), pixel_scale.end(), s);
      } else if (sc.IsSequence()) {
        const auto values = sc.as<std::vector<double>>();
        if (values.size() != cfg.cphd_paths.size()) {
          throw std::runtime_error(
              "lattice.pixel_scale length must match cphd.path (" +
              std::to_string(cfg.cphd_paths.size()) +
              " files, got " + std::to_string(values.size()) +
              ")");
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
        throw std::runtime_error(
            "lattice.pixel_scale must be a number or a list");
      }
    }
    cfg.lattice_pixel_scale = pixel_scale;
    if (has_counts) {
      lattice_counts = parse_lattice_counts(lat["counts"]);
      const YAML::Node sp = lat["spacing"];
      if (sp.IsScalar()) {
        const double metres = sp.as<double>();
        if (!(metres > 0.0) || !std::isfinite(metres)) {
          throw std::runtime_error(
              "lattice.spacing must be finite and > 0");
        }
        lattice_spacing = {metres, metres};
      } else {
        const auto values = sp.as<std::vector<double>>();
        if (values.size() != 2 || !(values[0] > 0.0) ||
            !(values[1] > 0.0) || !std::isfinite(values[0]) ||
            !std::isfinite(values[1])) {
          throw std::runtime_error(
              "lattice.spacing must be metres or [v, u], all > 0");
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
        throw std::runtime_error(
            "lattice needs SceneCoordinates/IARP/ECF in " + path);
      }
      const double dr =
          range_bin_spacing_from_cphd(path, nfft, speed);
      radar::SampleGeometry geom;
      if (has_counts) {
        geom = ecef_lattice_at_iarp(
            file.iarp_ecef(), lattice_counts, lattice_spacing);
        if (cfg.lattice_max_side > 0) {
          const auto capped = counts_for_extent(
              static_cast<double>(lattice_counts[0]) *
                  lattice_spacing[0],
              static_cast<double>(lattice_counts[1]) *
                  lattice_spacing[1],
              std::min(lattice_spacing[0], lattice_spacing[1]),
              cfg.lattice_max_side);
          const double ext_v =
              static_cast<double>(lattice_counts[0]) *
              lattice_spacing[0];
          const double ext_u =
              static_cast<double>(lattice_counts[1]) *
              lattice_spacing[1];
          const std::array<double, 9> rotation =
              ecef_to_enu_rotation(file.iarp_ecef());
          const std::array<double, 3> east{
              rotation[0], rotation[1], rotation[2]};
          const std::array<double, 3> north{
              rotation[3], rotation[4], rotation[5]};
          geom = lattice_from_image_area(
              file.iarp_ecef(), east,
              {-north[0], -north[1], -north[2]}, -ext_u / 2.0,
              -ext_v / 2.0, ext_u / 2.0, ext_v / 2.0, capped[0],
              capped[1]);
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
        geom = ecef_physics_lattice(file.iarp_ecef(), phys,
                                    cfg.lattice_max_side,
                                    pixel_scale[i]);
        cfg.collection_physics.push_back(phys);
      }
      cfg.collection_geometries.push_back(std::move(geom));
      cfg.collection_spacings.push_back(dr);
      cfg.collection_frequencies.push_back(
          reference_frequency_from_cphd(path));
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
    cfg.range_bin_spacing =
        range_bin_spacing_from_cphd(cfg.cphd_path, nfft, speed);
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
    const auto guessed = cphd.parent_path() /
                         (cphd.stem().string() + "_truth.json");
    if (std::filesystem::is_regular_file(guessed)) {
      sidecar = guessed;
    }
  }

  // YAML sample_geometry wins when present. A guessed sidecar must
  // not silently replace the Atlanta ENU lattice with a leftover
  // fixture file. Sidecar is the source only when the YAML omits
  // the lattice (generated smoke / mode01 fixtures).
  const bool yaml_has_lattice = root["sample_geometry"] &&
                                root["reference_frequency"];
  if (!yaml_has_lattice && !sidecar.empty() &&
      std::filesystem::is_regular_file(sidecar)) {
    cfg.sidecar_path = sidecar.string();
    // JSON is valid YAML. The converter runs SampleGeometry's
    // constructor, so a broken lattice fails here rather than
    // inside the operator.
    const YAML::Node truth = YAML::LoadFile(sidecar.string());
    const YAML::Node schema = truth["schema"];
    if (!schema || schema.as<std::string>() !=
                       "holoscan-radar/sar-sim-truth/1") {
      throw std::runtime_error(
          "sidecar schema is not holoscan-radar/sar-sim-truth/1: " +
          sidecar.string());
    }
    if (!truth["sample_geometry"]) {
      throw std::runtime_error(
          "sidecar is missing sample_geometry: " + sidecar.string());
    }
    cfg.sample_geometry =
        truth["sample_geometry"].as<radar::SampleGeometry>();
    if (!truth["collection"] ||
        !truth["collection"]["fx_min_hz"] ||
        !truth["collection"]["fx_spacing_hz"] ||
        !truth["collection"]["num_samples"]) {
      throw std::runtime_error(
          "sidecar is missing collection.fx_min_hz / "
          "fx_spacing_hz / num_samples: " +
          sidecar.string());
    }
    // Stored sample N/2, matching SARRangeCompressionOp's IFFT DC.
    const double fx_min =
        truth["collection"]["fx_min_hz"].as<double>();
    const double fx_step =
        truth["collection"]["fx_spacing_hz"].as<double>();
    const int64_t n_samp =
        truth["collection"]["num_samples"].as<int64_t>();
    cfg.reference_frequency =
        fx_min + static_cast<double>(n_samp / 2) * fx_step;
  } else {
    if (!root["sample_geometry"] || !root["reference_frequency"]) {
      throw std::runtime_error(
          "no sidecar at " +
          (sidecar.empty() ? std::string("<none>")
                           : sidecar.string()) +
          " and YAML is missing sample_geometry / "
          "reference_frequency");
    }
    cfg.sample_geometry =
        root["sample_geometry"].as<radar::SampleGeometry>();
    cfg.reference_frequency = root["reference_frequency"].as<double>();
  }
  cfg.collection_geometries = {cfg.sample_geometry};
  cfg.collection_spacings = {cfg.range_bin_spacing};
  cfg.collection_frequencies = {cfg.reference_frequency};
  }

  if (const YAML::Node origin = root["enu_origin_ecef"]) {
    const auto values = origin.as<std::vector<double>>();
    if (values.size() != 3) {
      throw std::runtime_error(
          "enu_origin_ecef must have 3 ECEF metres");
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
    cfg.dump_path =
        (cphd.parent_path() /
         (cphd.stem().string() + "_image_{index}.npy"))
            .string();
  }

  if (const YAML::Node am = root["aperture_manager"]) {
    if (const YAML::Node m = am["max_aperture_pulses"]) {
      cfg.wire_weight = m.as<int64_t>() > 0;
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
  return cfg;
}
