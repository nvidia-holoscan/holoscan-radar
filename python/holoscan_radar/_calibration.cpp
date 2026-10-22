// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>
#include <type_traits>
#include <string_view>
#include <utility>
#include <stdexcept>

#include <holoscan/config/io.hpp>
#include <holoscan/params/config.hpp>

#include "detection/cfar_detector_op/cfar_calibration.hpp"
#include "detection/cfar_detector_op/params_config.hpp"

namespace nb = nanobind;
namespace radar = holoscan::radar;
namespace {
template <typename Enum>
void normalize_enum(holoscan::config::Document& doc, std::string_view path) {
  auto found = doc.try_find(path);
  if (!found)
    throw std::invalid_argument(holoscan::config::format_error(found.error()));
  if (!found->has_value())
    return;
  auto value = doc.try_get<Enum>(path);
  if (!value)
    throw std::invalid_argument(holoscan::config::format_error(value.error()));
  auto replaced = doc.try_replace(path, static_cast<std::underlying_type_t<Enum>>(*value));
  if (!replaced)
    throw std::invalid_argument(holoscan::config::format_error(replaced.error()));
  doc = std::move(*replaced);
}
radar::CFARParams decode(const std::string& json) {
  auto doc = holoscan::config_io::parse(json, holoscan::config_io::Format::kJson);
  normalize_enum<radar::CfarVariant>(doc, "/params/cfar_variant");
  normalize_enum<radar::CfarPeakMode>(doc, "/params/peak_mode");
  auto p = holoscan::params::try_decode<radar::CFARParams>(doc, "/params");
  if (!p)
    throw std::invalid_argument(holoscan::params::format_error(p.error()));
  return **p;
}
nb::dict progress_dict(const radar::CfarCalibrationProgress& p) {
  nb::dict d;
  d["completed_trials"] = p.completed_trials;
  d["total_trials"] = p.total_trials;
  d["completed_geometries"] = p.completed_geometries;
  d["total_geometries"] = p.total_geometries;
  d["elapsed_seconds"] = p.elapsed_seconds;
  d["remaining_seconds"] = p.remaining_seconds;
  d["started_at"] = p.started_at;
  d["status"] = p.status;
  return d;
}
nb::dict plan_dict(const radar::CfarCalibrationPlan& p) {
  nb::dict d, config;
  config["cfar_variant"] = std::string(radar::to_string(p.params->cfar_variant));
  config["pfa"] = p.params->pfa;
  config["os_rank"] = p.params->os_rank;
  config["guard_cells"] = p.params->guard_cells;
  config["training_cells"] = p.params->training_cells;
  config["nci_axes"] = p.params->nci_axes;
  config["preintegrated_looks"] = p.params->preintegrated_looks;
  config["calibration_relative_error"] = p.params->calibration_relative_error;
  config["calibration_confidence"] = p.params->calibration_confidence;
  d["params"] = config;
  d["input_shape"] = p.input_shape;
  d["effective_looks"] = p.effective_looks;
  d["seed"] = p.seed;
  d["algorithm_version"] = p.algorithm_version;
  d["method"] = p.method;
  d["trials_per_geometry"] = p.budget.trials;
  d["tail_rank"] = p.budget.tail_rank;
  if (p.method == "direct_monte_carlo")
    d["coverage"] = p.budget.coverage;
  else
    d["coverage"] = nb::none();
  d["total_trials"] = p.total_trials;
  d["estimated_seconds"] = p.estimated_seconds;
  d["lut_size"] = p.lut_size;
  nb::list geometries;
  for (auto g : p.geometries) {
    nb::dict v;
    v["lut_index"] = g.lut_index;
    v["doppler"] = g.doppler;
    v["range"] = g.range;
    geometries.append(v);
  }
  d["geometries"] = geometries;
  return d;
}
}  // namespace
void register_calibration_bindings(nb::module_& module) {
  module.def("_plan_cfar_calibration",
             [](const std::string& json, const std::vector<std::int64_t>& shape) {
               auto params = decode(json);
               radar::CfarCalibrationPlan plan;
               {
                 nb::gil_scoped_release release;
                 plan = radar::plan_cfar_calibration(params, shape);
               }
               return plan_dict(plan);
             });
  module.def(
      "_calibrate_cfar",
      [](const std::string& json, const std::vector<std::int64_t>& shape, nb::object progress) {
        auto params = decode(json);
        std::exception_ptr python_error;
        radar::CfarCalibrationResult result;
        try {
          nb::gil_scoped_release release;
          result = radar::calibrate_cfar(params, shape, [&](const auto& p) {
            nb::gil_scoped_acquire acquire;
            try {
              if (PyErr_CheckSignals() < 0)
                throw nb::python_error();
              if (!progress.is_none())
                progress(progress_dict(p));
            } catch (...) {
              python_error = std::current_exception();
              throw radar::CfarCalibrationCancelled{};
            }
          });
        } catch (...) {
          if (python_error)
            std::rethrow_exception(python_error);
          throw;
        }
        auto d = plan_dict(result.plan);
        d["alpha_lut"] = result.alpha_lut;
        d["geometry_alphas"] = result.geometry_alphas;
        d["started_at"] = result.started_at;
        d["finished_at"] = result.finished_at;
        d["elapsed_seconds"] = result.elapsed_seconds;
        return d;
      },
      nb::arg("config_json"), nb::arg("input_shape"), nb::arg("progress") = nb::none());
}
