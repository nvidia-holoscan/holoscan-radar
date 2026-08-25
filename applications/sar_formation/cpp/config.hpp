// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned config for sar_formation: the AppConfig type and its YAML
// loader. Colocated with the app; the graph wiring lives in
// sar_formation.cpp. Operator blocks are NOT parsed here — they flow
// through Holoscan's from_config(). Exception: nfft and
// propagation_speed under range_compression are a read-only mirror
// so range_bin_spacing matches the operator if those keys are set.
//
// Lattice and the frequency of stored sample N/2 (IFFT DC after
// SARRangeCompressionOp's ifftshift): the app-owned YAML keys when
// present (Atlanta ENU lattice). Otherwise a simulator sidecar
// (<stem>_truth.json). A guessed sidecar must not override an
// explicit YAML lattice.

#pragma once

#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <vector>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace radar = holoscan::radar;

struct AppConfig {
  std::string cphd_path;
  std::string sidecar_path;
  std::string dump_path;
  radar::SampleGeometry sample_geometry;
  double reference_frequency{0.0};
  // Metres per compressed range bin. Mode 1 leaves range_bins
  // unconnected; this is BackprojectionOp.range_bin_spacing, with
  // the centred origin default. Derived from the CPHD (SCSS, N)
  // so it matches RC's axis when bandwidth = (N-1)*SCSS.
  double range_bin_spacing{0.0};
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
  bool skip_display_check{false};
  bool headless{false};
  // Seconds the Holoviz window keeps the formed image after emit.
  // Also the max wait for the first frame (Atlanta ~9 s). Mode 1
  // emits once; 0 closes with the graph.
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
  std::filesystem::path cphd;
  if (!cphd_override.empty()) {
    cphd = resolve_existing(cphd_override, yaml_dir);
  } else if (const YAML::Node node = root["cphd"]) {
    if (const YAML::Node p = node["path"]) {
      cphd = resolve_existing(p.as<std::string>(), yaml_dir);
    }
  }
  if (cphd.empty() || !std::filesystem::is_regular_file(cphd)) {
    throw std::runtime_error(
        "cphd.path is empty or not a file. Mode 1 wants the Atlanta "
        "Umbra collect (see mode01.yaml). For the generated smoke "
        "fixture:\n"
        "  python3 tools/sar_sim/generate_cphd.py "
        "tools/sar_sim/recipes/mode01_smoke.yaml -o <dir>\n"
        "  sar_formation mode01_smoke.yaml <dir>/mode01_smoke.cphd");
  }
  cfg.cphd_path = cphd.string();

  // Read-only mirror of operator keys that change dr. The CPHD
  // supplies df (SCSS) and num_samples; nfft / c come from the
  // same YAML block from_config() feeds the operator, when set.
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
  cfg.range_bin_spacing =
      range_bin_spacing_from_cphd(cfg.cphd_path, nfft, speed);

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

  if (const YAML::Node output = root["output"]) {
    if (const YAML::Node p = output["path"]) {
      const std::string given = p.as<std::string>();
      if (!given.empty()) {
        cfg.dump_path = given;
      }
    }
  }
  if (cfg.dump_path.empty()) {
    cfg.dump_path =
        (cphd.parent_path() /
         (cphd.stem().string() + "_image_{index}.npy"))
            .string();
  }

  if (const YAML::Node v = root["visualize"]; v && !v.IsNull()) {
    cfg.visualize = v.as<bool>();
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
  if (!std::isfinite(cfg.visualize_hold_s) ||
      cfg.visualize_hold_s < 0.0) {
    throw std::runtime_error(
        "visualize_hold_s must be finite and >= 0");
  }
  return cfg;
}
