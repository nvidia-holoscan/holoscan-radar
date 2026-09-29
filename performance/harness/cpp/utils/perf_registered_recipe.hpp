// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "utils/perf_authoring_registry.hpp"
#include "utils/perf_recipe.hpp"

namespace holoscan::radar::perf {

using OpaqueOperator = holoscan::authoring::OpaqueOperatorRef;

inline const holoscan::PortRecord& recipe_port(const std::vector<holoscan::PortRecord>& ports,
                                               std::string_view name,
                                               holoscan::PortDirection direction) {
  for (const auto& port : ports) {
    if (port.name == name && port.direction == direction)
      return port;
  }
  throw std::invalid_argument("performance recipe has unknown port: " + std::string(name));
}

inline void add_registered_flow(holoscan::Graph& graph, const OpaqueOperator& source,
                                std::string_view source_port, const OpaqueOperator& target,
                                std::string_view target_port,
                                const holoscan::ConnectionOptions& options) {
  const auto source_ports = source.ports();
  const auto target_ports = target.ports();
  const auto& out = recipe_port(source_ports, source_port, holoscan::PortDirection::kOutput);
  const auto& in = recipe_port(target_ports, target_port, holoscan::PortDirection::kInput);
  auto payload = holoscan::authoring::authoring_registry().payload_type(out.payload_identity);
  if (!payload || !payload->get().add_flow)
    throw std::invalid_argument("performance recipe has unregistered payload at " +
                                std::string(source_port));
  (void)in;  // The typed HSDK flow thunk checks the destination payload identity.
  payload->get().add_flow(graph, source.port(source_port), target.port(target_port), options);
}

struct RegisteredInputContract {
  std::string port;
  std::string dtype;
  std::string memory;
  std::size_t rank{};
};

inline void validate_authored_input(const holoscan::PortRecord& port,
                                    const RegisteredInputContract& declared) {
  const auto* contract = port.setup.tensor_contract ? &*port.setup.tensor_contract : nullptr;
  if (!contract)
    throw std::invalid_argument("operator data port lacks an HSDK Tensor contract: " +
                                declared.port);
  const auto& repr = contract->representation;
  if (repr.memory_kind && declared.memory == "device" &&
      *repr.memory_kind != holoscan::MemoryKind::kCudaDevice)
    throw std::invalid_argument("recipe memory placement disagrees with HSDK port: " +
                                declared.port);
  if (repr.rank && *repr.rank != declared.rank)
    throw std::invalid_argument("recipe rank disagrees with HSDK port: " + declared.port);
  if (repr.dtype) {
    const auto expected = declared.dtype == "complex64" ? DLDataType{kDLComplex, 64, 1}
                          : declared.dtype == "float32" ? DLDataType{kDLFloat, 32, 1}
                          : declared.dtype == "int32"   ? DLDataType{kDLInt, 32, 1}
                                                        : DLDataType{kDLInt, 64, 1};
    if (repr.dtype->code != expected.code || repr.dtype->bits != expected.bits ||
        repr.dtype->lanes != expected.lanes)
      throw std::invalid_argument("recipe dtype disagrees with HSDK port: " + declared.port);
  }
}

struct RegisteredOutputRecipe {
  std::string timed;
  std::vector<std::string> drain;
  std::vector<std::string> unwired;
  unsigned int queue_depth{4U};
  std::string timed_pair;
  std::map<std::string, std::string> drain_codecs;
};

template <typename Value>
void ensure_recipe_live_payload();

inline void wire_registered_outputs(holoscan::Graph& graph, holoscan::CompileOptions& options,
                                    const OpaqueOperator& op, std::string_view op_name,
                                    const RegisteredOutputRecipe& recipe,
                                    std::vector<OpaqueOperator>* members) {
  const holoscan::ConnectionOptions queue{.queue_depth = recipe.queue_depth};
  const auto records = op.ports();
  (void)recipe_port(records, recipe.timed, holoscan::PortDirection::kOutput);
  if (recipe.timed_pair.empty()) {
    auto terminal = graph.op<PerfTerminalSinkOp>(recipe.timed + "_perf_sink");
    auto terminal_ref = OpaqueOperator::from_handle(terminal);
    add_registered_flow(graph, op, recipe.timed, terminal_ref, "in", queue);
    bind_tensor_out(options, std::string(op_name).c_str(), recipe.timed.c_str());
    members->push_back(terminal_ref);
  } else {
    (void)recipe_port(records, recipe.timed_pair, holoscan::PortDirection::kOutput);
    auto terminal = graph.op<PerfDualTerminalSinkOp>("paired_perf_sink");
    auto terminal_ref = OpaqueOperator::from_handle(terminal);
    add_registered_flow(graph, op, recipe.timed, terminal_ref, "in0", queue);
    add_registered_flow(graph, op, recipe.timed_pair, terminal_ref, "in1", queue);
    bind_tensor_out(options, std::string(op_name).c_str(), recipe.timed.c_str());
    bind_tensor_out(options, std::string(op_name).c_str(), recipe.timed_pair.c_str());
    members->push_back(terminal_ref);
  }
  for (const auto& name : recipe.drain) {
    (void)recipe_port(records, name, holoscan::PortDirection::kOutput);
    const auto codec = recipe.drain_codecs.find(name);
    if (codec == recipe.drain_codecs.end()) {
      auto sink = graph.op<PerfNullSinkOp>(name + "_perf_drain");
      auto sink_ref = OpaqueOperator::from_handle(sink);
      add_registered_flow(graph, op, name, sink_ref, "in", queue);
      bind_tensor_out(options, std::string(op_name).c_str(), name.c_str());
      members->push_back(sink_ref);
    } else if (codec->second == "array_geometry") {
      ensure_recipe_live_payload<ArrayGeometry>();
      auto sink = graph.op<PerfPayloadSinkOp<WireArrayGeometry>>(name + "_perf_drain");
      auto sink_ref = OpaqueOperator::from_handle(sink);
      add_registered_flow(graph, op, name, sink_ref, "in", queue);
      members->push_back(sink_ref);
    } else {
      throw std::invalid_argument("unknown recipe drain codec: " + codec->second);
    }
  }
  for (const auto& name : recipe.unwired) {
    (void)recipe_port(records, name, holoscan::PortDirection::kOutput);
    bind_tensor_out(options, std::string(op_name).c_str(), name.c_str());
  }
}

template <typename Value>
struct RegisteredLiveRecipe {
  std::string physical_port;
  Value initial;
  Value changed;
  bool prime_changed{false};
};

template <typename Op, std::size_t N, typename SourceFactory, typename LiveConnect>
TimingCollector run_registered_recipe_core(
    std::array<std::shared_ptr<holoscan::Tensor>, N> inputs, const RunPlan& plan,
    std::string_view registry_name, const MaxTensorExtents& extents,
    const typename Op::Params& params, const std::array<std::string, N>& data_ports,
    const RegisteredOutputRecipe& outputs, SourceFactory source_factory, std::string_view live_port,
    LiveConnect live_connect, std::span<const RegisteredInputContract> input_contracts,
    PerfPublishMode publication) {
  constexpr auto source_ports = PerfDefaultSource<N>::ports;
  std::string graph_name = std::string(registry_name) + "-perf";
  return run_typed_recipe(
      std::move(inputs), plan, graph_name.c_str(), source_ports, source_factory,
      [&](holoscan::Graph& graph, const auto& source, holoscan::CompileOptions& options) {
        auto op =
            PerfRegisteredOperator<Op>::construct(graph, registry_name, "op", extents, params);
        auto src = OpaqueOperator::from_handle(source);
        if (!input_contracts.empty()) {
          if (input_contracts.size() != N)
            throw std::invalid_argument("recipe input contract count mismatch");
          const auto records = op.ports();
          for (std::size_t i = 0; i < N; ++i)
            validate_authored_input(
                recipe_port(records, data_ports[i], holoscan::PortDirection::kInput),
                input_contracts[i]);
        }
        const holoscan::ConnectionOptions queue{.queue_depth = outputs.queue_depth};
        for (std::size_t i = 0; i < N; ++i)
          add_registered_flow(graph, src, source_ports[i], op, data_ports[i], queue);
        if (!live_port.empty())
          live_connect(graph, source, src, op, live_port, queue);
        std::vector<OpaqueOperator> members{src, op};
        wire_registered_outputs(graph, options, op, "op", outputs, &members);
        auto partition = graph.partition("perf");
        OpaqueOperator::partition_add(partition, std::span<const OpaqueOperator>(members));
      },
      publication);
}

template <typename Op, std::size_t N>
TimingCollector run_registered_recipe(std::array<std::shared_ptr<holoscan::Tensor>, N> inputs,
                                      const RunPlan& plan, std::string_view registry_name,
                                      const MaxTensorExtents& extents,
                                      const typename Op::Params& params,
                                      const std::array<std::string, N>& data_ports,
                                      const RegisteredOutputRecipe& outputs,
                                      std::span<const RegisteredInputContract> contracts = {},
                                      PerfPublishMode publication = PerfPublishMode::kCopy) {
  return run_registered_recipe_core<Op>(
      std::move(inputs), plan, registry_name, extents, params, data_ports, outputs,
      [](holoscan::Graph& graph) {
        return graph.op<typename PerfDefaultSource<N>::Op>("src");
      },
      {}, [](auto&, const auto&, const auto&, const auto&, auto, const auto&) {}, contracts,
      publication);
}

template <typename Value>
void ensure_recipe_live_payload() {
  using Payload = typename PerfLiveCodec<Value>::PortValue;
  auto& registry = holoscan::authoring::authoring_registry();
  if (registry.payload_type(holoscan::detail::operator_spec_payload_identity<Payload>()))
    return;
  if constexpr (holoscan::detail::ExternalTopicPayload<Payload>) {
    auto registered = registry.register_payload(
        holoscan::authoring::detail::make_registered_payload<Payload>(typeid(Payload).name()));
    if (!registered)
      throw std::runtime_error("cannot register live recipe payload: " +
                               registered.error().message);
  } else {
    throw std::invalid_argument("live recipe payload requires a typed flow extension");
  }
}

template <typename Op, std::size_t N, typename Value>
TimingCollector run_registered_live_recipe(std::array<std::shared_ptr<holoscan::Tensor>, N> inputs,
                                           const RunPlan& plan, std::string_view registry_name,
                                           const MaxTensorExtents& extents,
                                           const typename Op::Params& params,
                                           const std::array<std::string, N>& data_ports,
                                           const RegisteredOutputRecipe& outputs,
                                           const RegisteredLiveRecipe<Value>& live,
                                           std::span<const RegisteredInputContract> contracts = {},
                                           PerfPublishMode publication = PerfPublishMode::kCopy) {
  ensure_recipe_live_payload<Value>();
  if (!plan.has_override() || plan.override_at >= plan.total_frames)
    throw std::invalid_argument("performance recipe requires one changed frame");
  return run_registered_recipe_core<Op>(
      std::move(inputs), plan, registry_name, extents, params, data_ports, outputs,
      [&](holoscan::Graph& graph) {
        return graph.op<PerfSelectedLiveSourceOp<N, Value>>("src", plan.override_at, live.initial,
                                                            live.changed, live.prime_changed);
      },
      live.physical_port,
      [](auto& graph, const auto&, const auto& src, const auto& op, auto port, const auto& q) {
        add_registered_flow(graph, src, "override_out", op, port, q);
      },
      contracts, publication);
}

template <typename Op, std::size_t N, typename Value, typename Connect>
TimingCollector run_registered_live_recipe_with_connector(
    std::array<std::shared_ptr<holoscan::Tensor>, N> inputs, const RunPlan& plan,
    std::string_view registry_name, const MaxTensorExtents& extents,
    const typename Op::Params& params, const std::array<std::string, N>& data_ports,
    const RegisteredOutputRecipe& outputs, const RegisteredLiveRecipe<Value>& live, Connect connect,
    std::span<const RegisteredInputContract> contracts = {},
    PerfPublishMode publication = PerfPublishMode::kCopy) {
  if (!plan.has_override() || plan.override_at >= plan.total_frames)
    throw std::invalid_argument("performance recipe requires one changed frame");
  return run_registered_recipe_core<Op>(
      std::move(inputs), plan, registry_name, extents, params, data_ports, outputs,
      [&](holoscan::Graph& graph) {
        return graph.op<PerfSelectedLiveSourceOp<N, Value>>("src", plan.override_at, live.initial,
                                                            live.changed, live.prime_changed);
      },
      live.physical_port,
      [&](auto& graph, const auto& source, const auto&, const auto& op, auto, const auto& q) {
        connect(graph, source, PerfRegisteredOperator<Op>::typed_handle_for_flow(op), q);
      },
      contracts, publication);
}

}  // namespace holoscan::radar::perf
