// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <span>
#include <stdexcept>
#include <utility>
#include <string>
#include <string_view>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"

namespace holoscan::radar::perf {

template <typename Op>
struct PerfOperatorFactory {
  static holoscan::OperatorHandle<Op> create(holoscan::Graph& graph, std::string_view name,
                                             const MaxTensorExtents& extents) {
    return graph.op<Op>(name, extents);
  }
};

// HSDK's automatic registration requires a default constructor. Radar operators
// capture pool bounds in their constructor, so the performance authoring scope
// provides those bounds for one synchronous Graph declaration.
template <typename Op>
struct PerfRegisteredOperator : holoscan::authoring::ManualOperatorRegistrationThunks<Op> {
  using Base = holoscan::authoring::ManualOperatorRegistrationThunks<Op>;
  using Params = typename Op::Params;
  using Registry = holoscan::authoring::RegisteredOperatorType;
  using Ref = holoscan::authoring::OpaqueOperatorRef;
  using Materialized = holoscan::authoring::MaterializedParameters;
  using Assignment = holoscan::authoring::ParameterAssignment;

  inline static thread_local const MaxTensorExtents* extents = nullptr;
  inline static thread_local const Params* parameters = nullptr;

  struct Scope {
    Scope(const MaxTensorExtents& bounds, const Params& params) {
      if (extents || parameters)
        throw std::logic_error("nested performance operator authoring scope");
      extents = &bounds;
      parameters = &params;
    }
    ~Scope() {
      extents = nullptr;
      parameters = nullptr;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
  };

  static Ref add_to_graph(holoscan::Graph& graph, std::string_view name,
                          const Materialized* materialized) {
    if (!extents || !parameters)
      throw std::logic_error("performance operator authoring scope is missing");
    auto handle = PerfOperatorFactory<Op>::create(graph, name, *extents);
    handle.set_params(materialized ? *static_cast<const Params*>(materialized->data())
                                   : *parameters);
    return Base::make_operator_ref(handle);
  }

  static holoscan::expected<Ref, holoscan::Error> add_to_graph_from_assignments(
      holoscan::Graph& graph, std::string_view name, std::span<const Assignment> assignments) {
    if (assignments.empty()) {
      return Base::invoke_add([&] {
        return add_to_graph(graph, name, nullptr);
      });
    }
    auto materialized = holoscan::authoring::materialize_parameters(
        holoscan::param_type_descriptor<Params>(), holoscan::param_schema_descriptor<Params>(),
        assignments);
    if (!materialized)
      return holoscan::make_unexpected(std::move(materialized).error());
    return Base::invoke_add([&] {
      return add_to_graph(graph, name, &*materialized);
    });
  }

  static void set_params(const Ref& op, const Materialized& materialized) {
    if (!Base::holds_type(op) || materialized.type() != &holoscan::param_type_descriptor<Params>())
      throw std::invalid_argument("registered performance Params type mismatch");
    Base::typed_handle(op).set_params(*static_cast<const Params*>(materialized.data()));
  }

  static Registry row(std::string_view name) {
    return Registry{
        .name = name,
        .param_type = &holoscan::param_type_descriptor<Params>(),
        .param_schema = holoscan::param_schema_descriptor<Params>(),
        .parameter_introspection =
            std::same_as<Params, holoscan::NoParams>
                ? holoscan::authoring::OperatorParameterIntrospection::kNone
                : holoscan::authoring::OperatorParameterIntrospection::kDeclared,
        .required_parameters = holoscan::authoring::detail::required_parameter_names<Params>(),
        .add_to_graph = &add_to_graph,
        .add_to_graph_from_assignments = &add_to_graph_from_assignments,
        .set_params = &set_params,
        .port = &Base::port,
        .ports = &Base::ports,
        .region_add = &Base::region_add,
        .partition_add = &Base::partition_add,
        .lifecycle_before = &Base::lifecycle_before,
        .lifecycle_allow_self_restart = &Base::lifecycle_allow_self_restart,
        .lifecycle_allow_external_restart = &Base::lifecycle_allow_external_restart,
        .lifecycle_allow_restart_request = &Base::lifecycle_allow_restart_request};
  }

  static void ensure_registered(std::string_view name) {
    auto& registry = holoscan::authoring::authoring_registry();
    if (registry.operator_type(name))
      return;
    auto result = registry.register_operator(row(name));
    if (!result)
      throw std::runtime_error("cannot register performance operator " + std::string(name) + ": " +
                               result.error().message);
  }

  static holoscan::OperatorHandle<Op> typed_handle_for_flow(const Ref& op) {
    if (!Base::holds_type(op))
      throw std::invalid_argument("registered performance Operator type mismatch");
    return Base::typed_handle(op);
  }

  static Ref construct(holoscan::Graph& graph, std::string_view registry_name,
                       std::string_view instance_name, const MaxTensorExtents& bounds,
                       const Params& params) {
    ensure_registered(registry_name);
    Scope scope(bounds, params);
    auto registered = holoscan::authoring::authoring_registry().operator_type(registry_name);
    if (!registered)
      throw std::runtime_error("registered performance operator disappeared");
    return registered->get().add_to_graph(graph, instance_name, nullptr);
  }
};

}  // namespace holoscan::radar::perf
