// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Typed Graph assembly for individual-operator performance recipes. Adapters
// supply valid tensors, an operator handle, and named port connections; this
// file owns staged-source setup, output sinks, partitioning, and execution.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>

#include "utils/perf_graph.hpp"
#include "utils/perf_payloads.hpp"
#include "utils/perf_runner.hpp"

namespace holoscan::radar::perf {

template <std::size_t N>
void stage_recipe_inputs(std::array<std::shared_ptr<holoscan::Tensor>, N> inputs, int total_frames,
                         TimingCollector* collector, PerfPublishMode publish) {
  PerfFrameBank::reset(std::move(inputs), total_frames, collector, publish);
}

// SourceFactory creates a typed source handle after frames are staged.
// BuildFn creates the operator and declares its connections and outputs.
template <std::size_t N, typename SourceFactory, typename BuildFn>
TimingCollector run_typed_recipe(std::array<std::shared_ptr<holoscan::Tensor>, N> inputs,
                                 const RunPlan& plan, const char* graph_name,
                                 const std::array<const char*, N>& source_ports,
                                 SourceFactory source_factory, BuildFn build,
                                 PerfPublishMode publish = PerfPublishMode::kCopy) {
  TimingCollector collector;
  collector.t_construct = Clock::now();
  stage_recipe_inputs(std::move(inputs), plan.total_frames, &collector, publish);
  holoscan::Graph graph{graph_name};
  auto source = source_factory(graph);
  holoscan::CompileOptions options;
  for (const char* port : source_ports)
    bind_tensor_out(options, "src", port);
  build(graph, source, options);
  execute_perf_graph(graph, std::move(options), plan.total_frames, graph_name, &collector);
  return collector;
}

template <std::size_t N>
class PerfDataSourceOp;

template <std::size_t N>
struct PerfDefaultSource;

template <>
struct PerfDefaultSource<1> {
  using Op = PerfDataSourceOp<1>;
  static constexpr std::array<const char*, 1> ports{"out"};
};

template <>
struct PerfDefaultSource<2> {
  using Op = PerfDataSourceOp<2>;
  static constexpr std::array<const char*, 2> ports{"out_a", "out_b"};
};

template <>
struct PerfDefaultSource<3> {
  using Op = PerfDataSourceOp<3>;
  static constexpr std::array<const char*, 3> ports{"out_a", "out_b", "out_c"};
};

template <>
struct PerfDefaultSource<6> {
  using Op = PerfDataSourceOp<6>;
  static constexpr std::array<const char*, 6> ports{"a", "b", "c", "d", "e", "f"};
};

template <std::size_t N, typename BuildFn>
TimingCollector run_data_recipe(std::array<std::shared_ptr<holoscan::Tensor>, N> inputs,
                                const RunPlan& plan, const char* graph_name, BuildFn build,
                                PerfPublishMode publish = PerfPublishMode::kCopy) {
  return run_typed_recipe(
      std::move(inputs), plan, graph_name, PerfDefaultSource<N>::ports,
      [](holoscan::Graph& graph) {
        return graph.op<typename PerfDefaultSource<N>::Op>("src");
      },
      build, publish);
}

template <typename Value, typename Connect>
struct PerfLiveCase {
  std::string_view port;
  Value initial;
  Value changed;
  Connect connect;
  bool prime_changed{false};
};

template <typename Value, typename Connect>
auto live_case(std::string_view port, Value initial, Value changed, Connect connect,
               bool prime_changed = false) {
  return PerfLiveCase<Value, Connect>{port, std::move(initial), std::move(changed),
                                      std::move(connect), prime_changed};
}

template <typename Value, typename Connect>
auto selected_live_case(std::string_view port, Value changed, Connect connect,
                        bool prime_changed = false) {
  return PerfLiveCase<Value, Connect>{port, changed, std::move(changed), std::move(connect),
                                      prime_changed};
}

template <typename Factory>
struct PerfLiveOption {
  std::string_view port;
  Factory make;
};

template <typename Factory>
auto live_option(std::string_view port, Factory make) {
  return PerfLiveOption<Factory>{port, std::move(make)};
}

template <std::size_t Index = 0, typename Cases, typename Visit>
TimingCollector dispatch_live_case(std::string_view port, const Cases& cases, Visit visit) {
  if constexpr (Index == std::tuple_size_v<Cases>) {
    throw std::invalid_argument("unknown live port in performance recipe: " + std::string(port));
  } else {
    const auto& candidate = std::get<Index>(cases);
    if (candidate.port == port) {
      auto selected = candidate.make();
      if (selected.port != port)
        throw std::invalid_argument("mismatched live port in performance recipe: " +
                                    std::string(port));
      return visit(std::move(selected));
    }
    return dispatch_live_case<Index + 1>(port, cases, visit);
  }
}

// Logical live values map to exactly one physical HSDK output. The source
// declares only the selected port, so an adapter never needs a multi-port
// source class just to emit one change.
template <typename Value>
struct PerfLiveCodec {
  using PortValue = Value;
  static void setup(holoscan::OperatorSpec& spec, holoscan::Output<Value>& output,
                    const Value& initial, const Value& changed) {
    if constexpr (std::is_same_v<Value, WireFMCWWaveform>)
      detail::setup_live_output(spec, output, initial, changed);
    else
      setup_scalar_payload(spec, output);
  }
  static holoscan::expected<void, holoscan::Error> emit(holoscan::Output<Value>& output,
                                                        holoscan::ExecutionContext& context,
                                                        const Value& value) {
    if constexpr (std::is_same_v<Value, WireFMCWWaveform>)
      return detail::emit_live_value(output, context, value);
    else
      return emit_scalar_payload(output, value, context.activation_time());
  }
};

template <typename Element>
struct PerfLiveCodec<std::vector<Element>> {
  static_assert(std::is_same_v<Element, double> || std::is_same_v<Element, int64_t>);
  using PortValue = holoscan::Tensor;
  static void setup(holoscan::OperatorSpec& spec, holoscan::Output<PortValue>& output,
                    const std::vector<Element>& initial, const std::vector<Element>& changed) {
    constexpr DLDataType dtype = std::is_same_v<Element, double> ? DLDataType{kDLFloat, 64U, 1U}
                                                                 : DLDataType{kDLInt, 64U, 1U};
    const std::size_t span =
        sizeof(Element) * std::max<std::size_t>(1U, std::max(initial.size(), changed.size()));
    setup_host_payload(spec, output, 1U, 1U, dtype, span);
  }
  static holoscan::expected<void, holoscan::Error> emit(holoscan::Output<PortValue>& output,
                                                        holoscan::ExecutionContext& context,
                                                        const std::vector<Element>& value) {
    return emit_host_vector(output, value, context.activation_time());
  }
};

template <>
struct PerfLiveCodec<ArrayGeometry> {
  using PortValue = WireArrayGeometry;
  static void setup(holoscan::OperatorSpec& spec, holoscan::Output<PortValue>& output,
                    const ArrayGeometry& initial, const ArrayGeometry& changed) {
    const std::size_t span =
        sizeof(double) * 3U *
        static_cast<std::size_t>(std::max(initial.num_elements(), changed.num_elements()) + 1);
    setup_host_payload(spec, output, 2U, 2U, DLDataType{kDLFloat, 64U, 1U}, span);
  }
  static holoscan::expected<void, holoscan::Error> emit(holoscan::Output<PortValue>& output,
                                                        holoscan::ExecutionContext& context,
                                                        const ArrayGeometry& value) {
    return emit_geometry_payload(output, value, context.activation_time());
  }
};

template <>
struct PerfLiveCodec<WindowSpec> {
  using PortValue = WireWindowSpec;
  static void setup(holoscan::OperatorSpec& spec, holoscan::Output<PortValue>& output,
                    const WindowSpec& initial, const WindowSpec& changed) {
    const std::size_t span =
        sizeof(double) * std::max<std::size_t>(1U, std::max(initial.coefficients.size(),
                                                            changed.coefficients.size()));
    setup_host_payload(spec, output, 1U, 1U, DLDataType{kDLFloat, 64U, 1U}, span);
  }
  static holoscan::expected<void, holoscan::Error> emit(holoscan::Output<PortValue>& output,
                                                        holoscan::ExecutionContext& context,
                                                        const WindowSpec& value) {
    return emit_window_payload(output, value, context.activation_time());
  }
};

template <std::size_t N>
struct PerfDataPorts;

template <>
struct PerfDataPorts<1> {
  holoscan::Output<holoscan::Tensor> out{"out"};
  void setup(holoscan::OperatorSpec& spec) {
    setup_device_input(spec, out, PerfFrameBank::span(0));
  }
  holoscan::expected<void, holoscan::Error> emit(
      holoscan::ExecutionContext& context,
      const std::array<std::shared_ptr<holoscan::Tensor>, 1>& frames, PerfPublishMode mode) {
    return detail::emit_device_frame(out, context, frames[0], mode);
  }
};

template <>
struct PerfDataPorts<2> {
  holoscan::Output<holoscan::Tensor> out_a{"out_a"};
  holoscan::Output<holoscan::Tensor> out_b{"out_b"};
  void setup(holoscan::OperatorSpec& spec) {
    setup_device_input(spec, out_a, PerfFrameBank::span(0));
    setup_device_input(spec, out_b, PerfFrameBank::span(1));
  }
  holoscan::expected<void, holoscan::Error> emit(
      holoscan::ExecutionContext& context,
      const std::array<std::shared_ptr<holoscan::Tensor>, 2>& frames, PerfPublishMode mode) {
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(out_a, context, frames[0], mode));
    return detail::emit_device_frame(out_b, context, frames[1], mode);
  }
};

template <>
struct PerfDataPorts<3> {
  holoscan::Output<holoscan::Tensor> out_a{"out_a"};
  holoscan::Output<holoscan::Tensor> out_b{"out_b"};
  holoscan::Output<holoscan::Tensor> out_c{"out_c"};
  void setup(holoscan::OperatorSpec& spec) {
    setup_device_input(spec, out_a, PerfFrameBank::span(0));
    setup_device_input(spec, out_b, PerfFrameBank::span(1));
    setup_device_input(spec, out_c, PerfFrameBank::span(2));
  }
  holoscan::expected<void, holoscan::Error> emit(
      holoscan::ExecutionContext& context,
      const std::array<std::shared_ptr<holoscan::Tensor>, 3>& frames, PerfPublishMode mode) {
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(out_a, context, frames[0], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(out_b, context, frames[1], mode));
    return detail::emit_device_frame(out_c, context, frames[2], mode);
  }
};

template <>
struct PerfDataPorts<6> {
  holoscan::Output<holoscan::Tensor> a{"a"};
  holoscan::Output<holoscan::Tensor> b{"b"};
  holoscan::Output<holoscan::Tensor> c{"c"};
  holoscan::Output<holoscan::Tensor> d{"d"};
  holoscan::Output<holoscan::Tensor> e{"e"};
  holoscan::Output<holoscan::Tensor> f{"f"};
  void setup(holoscan::OperatorSpec& spec) {
    setup_device_input(spec, a, PerfFrameBank::span(0));
    setup_device_input(spec, b, PerfFrameBank::span(1));
    setup_device_input(spec, c, PerfFrameBank::span(2));
    setup_device_input(spec, d, PerfFrameBank::span(3));
    setup_device_input(spec, e, PerfFrameBank::span(4));
    setup_device_input(spec, f, PerfFrameBank::span(5));
  }
  holoscan::expected<void, holoscan::Error> emit(
      holoscan::ExecutionContext& context,
      const std::array<std::shared_ptr<holoscan::Tensor>, 6>& frames, PerfPublishMode mode) {
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(a, context, frames[0], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(b, context, frames[1], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(c, context, frames[2], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(d, context, frames[3], mode));
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_device_frame(e, context, frames[4], mode));
    return detail::emit_device_frame(f, context, frames[5], mode);
  }
};

template <std::size_t N>
class PerfDataSourceOp final : public holoscan::Operator<>, public PerfDataPorts<N> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { PerfDataPorts<N>::setup(spec); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto staged = detail::take_staged_frames<N>();
    if (!staged)
      return {};
    return PerfDataPorts<N>::emit(context, staged->frames, staged->mode);
  }
};

template <std::size_t N, typename Value>
class PerfSelectedLiveSourceOp final : public holoscan::Operator<>, public PerfDataPorts<N> {
 public:
  PerfSelectedLiveSourceOp(int64_t override_at, Value initial, Value changed,
                           bool prime_changed = false)
  : override_at_(override_at),
    initial_(std::move(initial)),
    changed_(std::move(changed)),
    prime_changed_(prime_changed) {}

  void setup(holoscan::OperatorSpec& spec) override {
    PerfDataPorts<N>::setup(spec);
    PerfLiveCodec<Value>::setup(spec, override_out, initial_, changed_);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto staged = detail::take_staged_frames<N>();
    if (!staged)
      return {};
    if (frame_index_ == override_at_ || (prime_changed_ && frame_index_ == 0)) {
      HOLOSCAN_RETURN_IF_ERROR(PerfLiveCodec<Value>::emit(override_out, context, changed_));
      detail::record_live_emission(frame_index_);
    }
    ++frame_index_;
    return PerfDataPorts<N>::emit(context, staged->frames, staged->mode);
  }

  holoscan::Output<typename PerfLiveCodec<Value>::PortValue> override_out{"override_out"};

 private:
  int64_t override_at_{-1};
  int64_t frame_index_{0};
  Value initial_;
  Value changed_;
  bool prime_changed_{false};
};

template <std::size_t N, typename Value, typename Connect, typename Configure>
TimingCollector run_selected_live_recipe(std::array<std::shared_ptr<holoscan::Tensor>, N> inputs,
                                         const RunPlan& plan, const char* graph_name,
                                         const PerfLiveCase<Value, Connect>& live,
                                         Configure configure,
                                         PerfPublishMode publish = PerfPublishMode::kCopy) {
  static_assert(N == 1 || N == 2 || N == 3);
  if (!plan.has_override() || plan.override_at >= plan.total_frames)
    throw std::invalid_argument("performance selected live recipe requires a changed frame");
  return run_typed_recipe(
      std::move(inputs), plan, graph_name, PerfDefaultSource<N>::ports,
      [&](holoscan::Graph& graph) {
        return graph.op<PerfSelectedLiveSourceOp<N, Value>>("src", plan.override_at, live.initial,
                                                            live.changed, live.prime_changed);
      },
      [&](holoscan::Graph& graph, const auto& source, holoscan::CompileOptions& options) {
        configure(graph, source, options, live.connect);
      },
      publish);
}

class PerfRecipeOutputs {
 public:
  PerfRecipeOutputs(holoscan::Graph& graph, holoscan::CompileOptions& options,
                    const char* operator_name, unsigned int queue_depth = 4U)
  : graph_(graph),
    options_(options),
    operator_name_(operator_name),
    queue_{.queue_depth = queue_depth} {}

  void timed(const holoscan::Output<holoscan::Tensor>& output, const char* name) {
    require_single_terminal();
    auto sink = graph_.op<PerfTerminalSinkOp>(std::string(name) + "_perf_sink");
    graph_.add_flow(output, sink->in, queue_);
    sinks_.emplace_back([sink](holoscan::PartitionHandle& partition) {
      partition.add(sink);
    });
    bind_tensor_out(options_, operator_name_, name);
  }

  void timed_pair(const holoscan::Output<holoscan::Tensor>& first, const char* first_name,
                  const holoscan::Output<holoscan::Tensor>& second, const char* second_name) {
    require_single_terminal();
    auto sink = graph_.op<PerfDualTerminalSinkOp>("paired_perf_sink");
    graph_.add_flow(first, sink->in0, queue_);
    graph_.add_flow(second, sink->in1, queue_);
    sinks_.emplace_back([sink](holoscan::PartitionHandle& partition) {
      partition.add(sink);
    });
    bind_tensor_out(options_, operator_name_, first_name);
    bind_tensor_out(options_, operator_name_, second_name);
  }

  template <typename Payload>
  void drain(const holoscan::Output<Payload>& output, const char* name) {
    if constexpr (std::is_same_v<Payload, holoscan::Tensor>) {
      auto sink = graph_.op<PerfNullSinkOp>(std::string(name) + "_perf_drain");
      graph_.add_flow(output, sink->in, queue_);
      sinks_.emplace_back([sink](holoscan::PartitionHandle& partition) {
        partition.add(sink);
      });
      bind_tensor_out(options_, operator_name_, name);
    } else {
      auto sink = graph_.op<PerfPayloadSinkOp<Payload>>(std::string(name) + "_perf_drain");
      graph_.add_flow(output, sink->in, queue_);
      sinks_.emplace_back([sink](holoscan::PartitionHandle& partition) {
        partition.add(sink);
      });
    }
  }

  // Some optional Tensor outputs are deliberately unwired but still need
  // device placement in HSDK 5. Keep that declaration with the other outputs.
  void unwired(const holoscan::Output<holoscan::Tensor>&, const char* name) {
    bind_tensor_out(options_, operator_name_, name);
  }

  template <typename Source, typename Operator>
  void finish(const Source& source, const Operator& op) {
    if (!has_terminal_)
      throw std::invalid_argument("performance recipe requires a timed output");
    auto partition = graph_.partition("perf");
    partition.add(source).add(op);
    for (auto& add_sink : sinks_)
      add_sink(partition);
  }

 private:
  void require_single_terminal() {
    if (has_terminal_)
      throw std::invalid_argument("performance recipe has multiple timed sinks");
    has_terminal_ = true;
  }

  holoscan::Graph& graph_;
  holoscan::CompileOptions& options_;
  const char* operator_name_;
  holoscan::ConnectionOptions queue_;
  bool has_terminal_{false};
  std::vector<std::function<void(holoscan::PartitionHandle&)>> sinks_;
};

}  // namespace holoscan::radar::perf
