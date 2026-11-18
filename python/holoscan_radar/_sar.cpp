// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/python/constructor_handle_capsule.hpp>
#include <holoscan/core/graph.hpp>

#include "radar_app/sar/graph_driver.hpp"
#include "radar_app/sar/coordinate_ops.hpp"
#include "radar_app/sar/mosaic_op.hpp"
#include "radar_app/report/report_ops.hpp"
#include "applications/sar/spotlight/cpp/config.hpp"
#include "applications/sar/video_sar/cpp/config.hpp"
#include "applications/sar/tiled_strip/cpp/config.hpp"
#include "applications/sar/tiled_strip/cpp/tile_stager.hpp"
#include "applications/sar/continuous_strip/cpp/config.hpp"
#include "applications/sar/continuous_strip/cpp/scroll_op.hpp"
#include "applications/sar/independent_images/cpp/config.hpp"
#include "applications/sar/independent_images/cpp/collection_stager.hpp"
#include "applications/sar/autofocus/cpp/config.hpp"
#include "applications/sar/autofocus/cpp/aux_points_op.hpp"
#include "applications/sar/autofocus/cpp/position_error_op.hpp"
#include "applications/sar/autofocus/cpp/phase_resample_op.hpp"
#include "applications/sar/autofocus/cpp/sharpness_report_op.hpp"
#include "sar/autofocus_op/autofocus_op.hpp"
#include "sar/phase_correction_op/phase_correction_op.hpp"

namespace nb = nanobind;
namespace authoring = holoscan::authoring;

namespace {

// These private authoring views use the SDK's erased native handles. Python
// owns the processing connections while native helpers retain constructor-only
// state. They add no payload type and never enter the processing data plane.
struct SarPort {
  authoring::OpaquePortRef ref;
  holoscan::PayloadIdentityRecord identity;
};

struct SarNode {
  authoring::OpaqueOperatorRef ref;

  SarPort port(const std::string& name) const {
    for (const auto& record : ref.ports()) {
      if (record.name == name)
        return {ref.port(name), record.payload_identity};
    }
    throw nb::attribute_error(name.c_str());
  }
};

template <typename Op>
SarNode node(const holoscan::OperatorHandle<Op>& handle) {
  return {authoring::OpaqueOperatorRef::from_handle(handle)};
}

struct SarSetup {
  std::string name;
  CliPaths paths;
  AppConfig config;
  AutofocusExtras extras;
};

SarSetup prepare(const std::string& name, const std::string& path,
                 const std::string& cphd_override) {
  SarSetup setup;
  setup.name = name;
  setup.paths.config = path;
  setup.paths.cphd_override = cphd_override;
  const auto& paths = setup.paths;
  if (name == "spotlight")
    setup.config = load_spotlight_config(paths.config, paths.cphd_override);
  else if (name == "video_sar")
    setup.config = load_video_sar_config(paths.config, paths.cphd_override);
  else if (name == "tiled_strip")
    setup.config = load_tiled_strip_config(paths.config, paths.cphd_override);
  else if (name == "continuous_strip")
    setup.config = load_continuous_strip_config(paths.config, paths.cphd_override);
  else if (name == "independent_images")
    setup.config = load_independent_images_config(paths.config, paths.cphd_override);
  else if (name == "autofocus")
    setup.config = load_autofocus_config(paths.config, paths.cphd_override, &setup.extras);
  else
    throw std::invalid_argument("Unknown SAR application: " + name);
  return setup;
}

template <typename T>
struct PythonSarContext : std::enable_shared_from_this<PythonSarContext<T>> {
  using Base = SarGraphRuntime<T, double>;
  AutofocusExtras extras;
  std::shared_ptr<app::sar::LieRecord> lies;
  std::map<std::string, holoscan::PartitionHandle> partitions;

  SarSetup setup;
  SarRuntimeInputs prepared;
  std::unique_ptr<Base> runtime;

  explicit PythonSarContext(SarSetup config)
  : extras(config.extras),
    setup(std::move(config)),
    prepared(prepare_sar_runtime(setup.config, setup.paths.config)) {}

  SarNode attach(holoscan::Graph& target) {
    if (runtime)
      throw std::logic_error("SAR context is already attached to a graph");
    runtime = std::make_unique<Base>(setup.name, setup.paths, setup.config, &target, &prepared);
    partitions.emplace("source", runtime->source);
    partitions.emplace("application", runtime->application);
    partitions.emplace("formation", runtime->formation);
    return node(runtime->reader);
  }

  Base& state() const {
    if (!runtime)
      throw std::logic_error("SAR context must be attached to a graph first");
    return *runtime;
  }

  nb::dict deployment() const {
    (void)state();
    nb::dict result;
    nb::list tensors, threads;
    for (const auto& binding : runtime->options.deployment.tensor_output_devices()) {
      tensors.append(
          nb::make_tuple(binding.operator_path, binding.output_port, binding.device.value));
    }
    for (const auto& binding : runtime->options.deployment.thread_placements()) {
      threads.append(binding.partition_name);
    }
    result["tensors"] = tensors;
    result["threads"] = threads;
    return result;
  }

  nb::dict flags() const {
    (void)state();
    nb::dict result;
    result["transform_to_scene_enu"] = runtime->cfg.transform_to_scene_enu;
    result["wire_weight"] = runtime->cfg.wire_weight;
    result["position_error"] = extras.position_error.enabled;
    result["oracle"] = extras.position_error.oracle;
    result["aux_polar"] = extras.aux_polar;
    return result;
  }

  nb::dict pools() const {
    (void)state();
    nb::dict result;
    result["ingest_pulses"] = *runtime->ingest_extents.max_pulses;
    result["aperture_pulses"] = *runtime->extents.max_pulses;
    result["range_bins"] = *runtime->extents.max_range_bins;
    result["image_v"] = *runtime->extents.max_image_v;
    result["image_u"] = *runtime->extents.max_image_u;
    return result;
  }

  template <typename Op, typename... Args>
  SarNode add(const std::string& name, Args&&... args) {
    auto handle = runtime->graph.template op<Op>(name, std::forward<Args>(args)...);
    const auto result = node(handle);
    for (const auto& port : result.ref.ports()) {
      if (port.direction == holoscan::PortDirection::kOutput &&
          port.payload_identity.kind == holoscan::PayloadIdentityKind::kTensor) {
        runtime->options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = name, .output_port = port.name, .device = runtime->devices.compute});
      }
    }
    return result;
  }

  holoscan::config::Document autofocus_document() const {
    auto document = app::config::load_inherited_document(extras.config_path);
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
    return app::config::normalize_enum_field<radar::PhaseCorrectionSense>(
        std::move(document), "/phase_correction/sense");
  }

  radar::MaxTensorExtents aux_extents() const {
    auto result = runtime->extents;
    result.max_image_v = std::max(*runtime->extents.max_image_v, extras.aux_num_v());
    result.max_image_u = std::max(*runtime->extents.max_image_u, extras.aux_num_u());
    return result;
  }

  SarNode op(const std::string& role) {
    (void)state();
    if (runtime->finished)
      throw std::logic_error("SAR graph was already finalized");
    if (role == "cphd_reader")
      return node(runtime->reader);
    if (role == "range_compression")
      return node(runtime->range);
    if (role == "aperture_manager")
      return node(runtime->manager);
    if (role == "backprojection")
      return node(runtime->bp);
    if (role == "ecef_to_enu") {
      return add<app::sar::EcefToSceneEnuOp>(role, runtime->cfg.enu_origin_ecef,
                                             runtime->cfg.quantize_phase_center_f32,
                                             runtime->cfg.reader_pulses_per_tick);
    }
    if (role == "tile_stager") {
      return add<app::sar::TileGeometryStagerOp>(role, runtime->cfg.tile_geometries,
                                                 runtime->cfg.tile_pulses);
    }
    if (role == "collection_stager") {
      return add<app::sar::CollectionBoundaryStagerOp>(role, runtime->cfg.collection_geometries,
                                                       runtime->cfg.collection_spacings,
                                                       runtime->cfg.collection_frequencies);
    }
    if (role == "scroll_stager") {
      const int64_t slice =
          runtime->cfg.emit_interval_pulses > 0 &&
                  runtime->cfg.reader_pulses_per_tick > runtime->cfg.emit_interval_pulses
              ? runtime->cfg.emit_interval_pulses
              : runtime->cfg.reader_pulses_per_tick;
      return add<app::sar::ScrollGeometryStagerOp>(
          role, runtime->cfg.scroll_parent, runtime->cfg.sample_geometry,
          runtime->cfg.scroll_anchor, runtime->cfg.reader_loop, slice,
          runtime->cfg.reader_pulses_per_tick, *runtime->extents.max_range_bins);
    }
    if (role == "sar_mosaic") {
      if (runtime->cfg.scroll) {
        return add<app::sar::SarMosaicOp<T>>(
            role, runtime->cfg.scroll_parent, runtime->cfg.reader_loop, runtime->cfg.viz_crop_u0,
            runtime->cfg.viz_crop_nu, 2 * runtime->cfg.visualize_size);
      }
      return add<app::sar::SarMosaicOp<T>>(role, runtime->cfg.parent_geometry,
                                           runtime->cfg.reader_loop);
    }
    if (role == "phase_correction") {
      radar::PhaseCorrectionParams seed;
      seed.sense = radar::PhaseCorrectionSense::kSubtract;
      auto result = add<radar::PhaseCorrectionOp<T>>(role, runtime->extents);
      runtime->graph.template handle<radar::PhaseCorrectionOp<T>>(role).set_params(
          app::config::decode_params(autofocus_document(), "/phase_correction", seed));
      return result;
    }
    if (role == "perturb_positions") {
      lies = std::make_shared<app::sar::LieRecord>();
      const auto& pe = extras.position_error;
      return add<app::sar::PerturbPositionsOp<T, double>>(
          role, lies, pe.look, pe.profile, pe.num_pulses, runtime->cfg.reference_frequency,
          runtime->ingest_extents);
    }
    if (role == "oracle_phase") {
      return add<app::sar::OraclePhaseOp<T>>(role, lies, runtime->extents);
    }
    if (role == "aux_bp") {
      radar::BackprojectionParams seed;
      seed.reference_frequency = runtime->cfg.reference_frequency;
      seed.range_bin_spacing = runtime->cfg.range_bin_spacing;
      if (!extras.aux_polar)
        seed.sample_geometry = extras.aux_sample_geometry;
      auto result = add<radar::BackprojectionOp<T, double>>(role, aux_extents());
      runtime->graph.template handle<radar::BackprojectionOp<T, double>>(role).set_params(
          app::config::decode_params(autofocus_document(), "/backprojection", seed));
      return result;
    }
    if (role == "aux_points") {
      return add<app::sar::AuxPointsSourceOp>(role, extras.aux_polar_lattice);
    }
    if (role == "autofocus") {
      radar::AutofocusParams seed;
      seed.azimuth_axis = radar::AutofocusAzimuthAxis::kU;
      if (!extras.window_fraction_in_yaml) {
        seed.window_fraction =
            std::min(0.5, 64.0 / static_cast<double>(std::max(extras.aux_num_u(), int64_t{2})));
      }
      auto result = add<radar::AutofocusOp<T>>(role, aux_extents());
      runtime->graph.template handle<radar::AutofocusOp<T>>(role).set_params(
          app::config::decode_params(autofocus_document(), "/autofocus", seed));
      return result;
    }
    if (role == "phase_resample") {
      return add<app::sar::PhaseResampleOp>(role, extras.aperture_pulses);
    }
    if (role == "af_image_drain")
      return add<app::NullSinkOp>(role);
    if (role == "autofocus_sharpness")
      return add<app::sar::AutofocusSharpnessReportOp>(role);
    if (role == "autofocus_score") {
      std::string dump_dir;
      if (runtime->cfg.dump && !runtime->cfg.dump_path.empty()) {
        dump_dir = std::filesystem::path(runtime->cfg.dump_path).parent_path().string();
        if (dump_dir.empty())
          dump_dir = ".";
      }
      return add<app::sar::AutofocusScoreOp>(role, lies, extras.position_error.warn_rms_rad,
                                             runtime->cfg.reference_frequency, dump_dir);
    }
    throw std::invalid_argument("Unknown SAR application node: " + role);
  }

  void partition(const std::string& name, const SarNode& member) {
    (void)state();
    if (runtime->finished)
      throw std::logic_error("SAR graph was already finalized");
    auto found = partitions.find(name);
    if (found == partitions.end())
      found = partitions.emplace(name, runtime->graph.partition(name)).first;
    authoring::OpaqueOperatorRef::partition_add(
        found->second, std::span<const authoring::OpaqueOperatorRef>(&member.ref, 1U));
  }

  void flow(const SarPort& output, const SarPort& input, uint32_t depth) {
    (void)state();
    if (runtime->finished)
      throw std::logic_error("SAR graph was already finalized");
    auto payload = authoring::authoring_registry().payload_type(output.identity);
    if (!payload)
      throw holoscan::RuntimeError{payload.error()};
    payload->get().add_flow(runtime->graph, output.ref, input.ref,
                            holoscan::ConnectionOptions{.queue_depth = depth});
  }

  void finish() {
    (void)state();
    if (runtime->cfg.scroll || !runtime->cfg.tile_geometries.empty()) {
      auto mosaic = runtime->graph.template handle<app::sar::SarMosaicOp<T>>("sar_mosaic");
      runtime->finish(&mosaic->image_out);
    } else {
      runtime->finish(&runtime->bp->image);
    }
  }
};

template <typename T>
const authoring::ConstructorHandleType<PythonSarContext<T>>& context_type() {
  static const authoring::ConstructorHandleDescriptor descriptor{
      sizeof(T) == 4 ? "holoscan.radar._app.SarF32Context" : "holoscan.radar._app.SarF64Context",
      "Native support owner for one Python-authored SAR graph"};
  static const authoring::ConstructorHandleType<PythonSarContext<T>> type{descriptor};
  return type;
}

template <typename T>
struct SarAnchorThunks
: authoring::ManualOperatorRegistrationThunks<radar::CPHDReaderOp<T, double>> {
  using Parent = authoring::ManualOperatorRegistrationThunks<radar::CPHDReaderOp<T, double>>;
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> add(
      holoscan::Graph& graph, std::string_view name,
      const authoring::OperatorConstructionInputs& inputs) {
    return Parent::invoke_add([&] {
      if (name != "cphd_reader")
        throw std::invalid_argument("SAR context anchor must be named cphd_reader");
      auto owner = inputs.require_shared("context", context_type<T>());
      if (!owner)
        throw holoscan::RuntimeError{owner.error()};
      return (*owner)->attach(graph).ref;
    });
  }
  static void set_params(const authoring::OpaqueOperatorRef&,
                         const authoring::MaterializedParameters&) {
    throw std::invalid_argument("SAR application constructor inputs are immutable");
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> describe(
      holoscan::Graph& graph, std::string_view name) {
    return Parent::invoke_add([&] {
      radar::MaxTensorExtents extents;
      extents.max_pulses = 1;
      extents.max_samples_per_pulse = 8;
      return authoring::OpaqueOperatorRef::from_handle(
          graph.op<radar::CPHDReaderOp<T, double>>(name, extents));
    });
  }
};

template <typename T>
void register_anchor() {
  using Thunks = SarAnchorThunks<T>;
  static const authoring::ConstructorHandleInput handle[]{
      {.name = "context",
       .description = "Per-graph native SAR support",
       .type = &context_type<T>().descriptor()}};
  static const authoring::RegisteredConstructorInputs inputs{
      .handles = handle,
      .add_to_graph_from_inputs = &Thunks::add,
      .add_for_description = &Thunks::describe};
  auto result = authoring::authoring_registry().register_operator(
      {.name = context_type<T>().descriptor().name,
       .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
       .add_to_graph = &Thunks::add_without_constructor_handles,
       .add_to_graph_from_assignments = &Thunks::reject_without_constructor_handles,
       .set_params = &Thunks::set_params,
       .port = &Thunks::port,
       .ports = &Thunks::ports,
       .region_add = &Thunks::region_add,
       .partition_add = &Thunks::partition_add,
       .lifecycle_before = &Thunks::lifecycle_before,
       .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
       .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
       .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
       .constructor_inputs = &inputs});
  if (!result)
    throw holoscan::RuntimeError{result.error()};
}

template <typename T>
void bind_context(nb::module_& module, const char* name) {
  using Context = PythonSarContext<T>;
  nb::class_<Context>(module, name)
      .def_prop_ro("registry_name",
                   [](Context&) {
                     return std::string(context_type<T>().descriptor().name);
                   })
      .def("__holoscan_constructor_handle__",
           [](Context& self, const std::string& requested) {
             auto* capsule = holoscan::python::export_constructor_handle(
                 requested, context_type<T>(), self.shared_from_this());
             if (!capsule)
               throw nb::python_error();
             return nb::steal<nb::object>(capsule);
           })
      .def_prop_ro("deployment", &Context::deployment)
      .def_prop_ro("flags", &Context::flags)
      .def_prop_ro("pools", &Context::pools)
      .def("op", &Context::op)
      .def("partition", &Context::partition)
      .def("add_flow", &Context::flow, nb::arg("output"), nb::arg("input"),
           nb::arg("queue_depth") = 4U)
      .def("finish", &Context::finish)
      .def(
          "run",
          [](Context& self) {
            return self.state().run(nullptr, [] {
              nb::gil_scoped_acquire gil;
              if (PyErr_CheckSignals() != 0)
                throw nb::python_error();
            });
          },
          nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace

void register_sar_bindings(nb::module_& module) {
  nb::class_<SarPort>(module, "_SarPort");
  nb::class_<SarNode>(module, "_SarNode")
      .def_prop_ro("name",
                   [](const SarNode& self) {
                     return std::string(self.ref.name());
                   })
      .def("__getattr__", &SarNode::port);
  register_anchor<float>();
  register_anchor<double>();
  bind_context<float>(module, "_SarF32Context");
  bind_context<double>(module, "_SarF64Context");
  module.def(
      "_prepare_sar",
      [](const std::string& name, const std::string& path, const std::string& cphd_override,
         const std::string& presenter, bool choose_presenter) -> nb::object {
        auto setup = prepare(name, path, cphd_override);
        setup.config.presenter = presenter;
        setup.config.choose_presenter = choose_presenter;
        if (single_sample_precision(path)) {
          return nb::cast(std::make_shared<PythonSarContext<float>>(std::move(setup)));
        }
        return nb::cast(std::make_shared<PythonSarContext<double>>(std::move(setup)));
      },
      nb::arg("application"), nb::arg("path"), nb::arg("cphd_override"), nb::arg("presenter") = "",
      nb::arg("choose_presenter") = false);
}
