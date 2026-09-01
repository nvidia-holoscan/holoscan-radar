// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Focused HSDK 5 Graph probe for a plain, pageable-host Tensor used as a
// rank-1 configuration vector under OnEach(data) + Exact + SafeDefault.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "radar_common/sdk/config_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::add_live_override_to_contract;
using holoscan::radar::dldatatype_equals;
using holoscan::radar::read_host_config_vector;
using holoscan::radar::receive_tensor;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;

constexpr DLDataType kConfigDtype{kDLInt, 64U, 1U};
constexpr auto kTick = std::chrono::milliseconds{5};
constexpr auto kDeadline = std::chrono::seconds{15};
constexpr auto kPoll = std::chrono::milliseconds{25};
constexpr auto kSkew = std::chrono::nanoseconds{1};

struct ConfigTick {
  int64_t data;
  std::optional<std::vector<int64_t>> config;
  std::chrono::nanoseconds config_skew{0};
};

struct Observation {
  int64_t data;
  std::vector<int64_t> config;
  bool config_missing;
};

struct ProbeBank {
  static void reset(std::vector<ConfigTick> new_ticks) {
    const std::lock_guard lock(mutex);
    ticks = std::move(new_ticks);
    observations.clear();
    index = 0;
  }

  static inline std::mutex mutex;
  static inline std::vector<ConfigTick> ticks;
  static inline std::vector<Observation> observations;
  static inline std::size_t index{0};
};

class HostConfigSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kHostConfig{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = kConfigDtype,
        .rank = 1U,
    };
    spec.output(data, "data");
    spec.output(config, "config")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kHostConfig, 1U, 64U * sizeof(int64_t)));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    ConfigTick tick;
    {
      const std::lock_guard lock(ProbeBank::mutex);
      if (ProbeBank::index >= ProbeBank::ticks.size()) {
        return {};
      }
      tick = ProbeBank::ticks[ProbeBank::index++];
    }

    const holoscan::TimePoint data_time = context.activation_time();
    if (tick.config.has_value()) {
      const std::vector<int64_t>& values = *tick.config;
      const std::vector<int64_t> shape{
          static_cast<int64_t>(values.size())};
      auto loan = config.allocate_tensor(
          holoscan::TensorLoanRequest{.shape = shape, .dtype = kConfigDtype});
      if (!loan) {
        return holoscan::make_unexpected(std::move(loan).error());
      }
      auto written = loan->write_host_and_commit(
          [&values](const holoscan::TensorOutputWriteGuard& writer) {
            std::copy(values.begin(), values.end(),
                      writer.data_as<int64_t>());
          });
      if (!written) {
        return holoscan::make_unexpected(std::move(written).error());
      }
      holoscan::TimePoint config_time = data_time;
      config_time.timestamp_ns += tick.config_skew.count();
      HOLOSCAN_RETURN_IF_ERROR(config.emit(
          std::move(*loan),
          holoscan::EmitOptions{.capture_time = config_time}));
    }
    return data.emit(tick.data,
                     holoscan::EmitOptions{.capture_time = data_time});
  }

  holoscan::Output<int64_t> data{"data"};
  holoscan::Output<holoscan::Tensor> config{"config"};
};

class HostTensorOverrideProbeOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kHostConfig{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = kConfigDtype,
        .rank = 1U,
    };
    spec.input(data, "data").queue_depth(4U);
    spec.input(config, "config")
        .queue_depth(4U)
        .expects_tensor(holoscan::radar::tensor_input(kHostConfig));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{data});
    add_live_override_to_contract(result, config);
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto data_sample = data.receive_data();
    if (!data_sample) {
      return holoscan::make_unexpected(std::move(data_sample).error());
    }

    const bool missing = config.is_missing();
    std::vector<int64_t> effective{1};
    if (!missing) {
      HOLOSCAN_TRY(auto tensor,
                   receive_tensor(config, "HostTensorOverrideProbeOp"));
      HOLOSCAN_TRY(effective, read_host_config_vector<int64_t>(
                                  tensor, kConfigDtype, 64, "config"));
    }

    {
      const std::lock_guard lock(ProbeBank::mutex);
      ProbeBank::observations.push_back(
          Observation{*data_sample, std::move(effective), missing});
    }
    return {};
  }

  holoscan::Input<int64_t> data{"data"};
  holoscan::Input<holoscan::Tensor> config{"config"};
};

struct GraphRunResult {
  bool plan_ok{true};
  std::string plan_json;
  holoscan::RunTermination termination{holoscan::RunTermination::kStopped};
  std::vector<std::string> diagnostics;
};

GraphRunResult run_probe(std::vector<ConfigTick> ticks) {
  GraphRunResult result;
  const std::size_t expected = ticks.size();
  ProbeBank::reset(std::move(ticks));

  holoscan::Graph graph{"host-tensor-exact-override"};
  const auto source = graph.op<HostConfigSourceOp>("source");
  const auto probe = graph.op<HostTensorOverrideProbeOp>("probe");
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->data, probe->data, q);
  graph.add_flow(source->config, probe->config, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  try {
    // Deliberately no device placement or CUDA-copy fallback: the config
    // Output allocates and publishes a host Tensor loan directly.
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, holoscan::CompileOptions{});
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() + kDeadline;
    while (session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      {
        const std::lock_guard lock(ProbeBank::mutex);
        if (ProbeBank::observations.size() >= expected) {
          break;
        }
      }
      std::this_thread::sleep_for(kPoll);
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& diagnostic : session.diagnostics()) {
      result.diagnostics.emplace_back(diagnostic.token);
    }
  } catch (const holoscan::RuntimeError& error) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + error.what();
  }
  return result;
}

}  // namespace

TEST(HostTensorExactOverrideProbe, ExactMatchThenRevertToDefaultAfterMiss) {
  const std::vector<int64_t> exact{3, 5, 8};
  const GraphRunResult result = run_probe({
      ConfigTick{.data = 10, .config = exact},
      ConfigTick{
          .data = 20, .config = std::vector<int64_t>{13, 21},
          .config_skew = kSkew},
  });

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  const std::lock_guard lock(ProbeBank::mutex);
  ASSERT_EQ(ProbeBank::observations.size(), 2U);
  EXPECT_EQ(ProbeBank::observations[0].data, 10);
  EXPECT_EQ(ProbeBank::observations[0].config, exact);
  EXPECT_FALSE(ProbeBank::observations[0].config_missing);
  EXPECT_EQ(ProbeBank::observations[1].data, 20);
  EXPECT_EQ(ProbeBank::observations[1].config, (std::vector<int64_t>{1}));
  EXPECT_TRUE(ProbeBank::observations[1].config_missing);
}

TEST(HostTensorExactOverrideProbe, ValidatesDtypeRankAndPageableHostMemory) {
  const auto valid = to_pageable_host_tensor<int64_t>(
      {3, 5, 8}, {3}, kConfigDtype);
  EXPECT_TRUE(read_host_config_vector<int64_t>(
      *valid, kConfigDtype, 64, "config"));

  const auto wrong_dtype = to_pageable_host_tensor<float>(
      {3.0F, 5.0F}, {2}, DLDataType{kDLFloat, 32U, 1U});
  EXPECT_FALSE(read_host_config_vector<int64_t>(
      *wrong_dtype, kConfigDtype, 64, "config"));

  const auto wrong_rank = to_pageable_host_tensor<int64_t>(
      {3, 5}, {1, 2}, kConfigDtype);
  EXPECT_FALSE(read_host_config_vector<int64_t>(
      *wrong_rank, kConfigDtype, 64, "config"));

  const auto cuda_memory = to_device_tensor<int64_t>(
      {3, 5, 8}, {3}, kConfigDtype);
  EXPECT_FALSE(read_host_config_vector<int64_t>(
      *cuda_memory, kConfigDtype, 64, "config"));
  EXPECT_TRUE(dldatatype_equals(cuda_memory->dtype(), kConfigDtype));
}
