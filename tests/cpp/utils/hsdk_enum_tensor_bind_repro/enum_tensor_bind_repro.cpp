// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone Holoscan Graph reproducer (no radar / GTest).
// EA v5.0.0.1: TensorRepresentation + TensorBounds.
//
// Question for Holoscan: kLocalEnum is "admitted only on local typed Graph
// edges". Does a local edge that also carries a CUDA Tensor bind, or is
// Output<enum class> unsupported next to bind_tensor_output_device?
// cuda_tensor_scalar_override uses uint32_t, which is the control below.
//
// Build (copy this directory only):
//   cmake -S . -B build
//   cmake --build build
//   ./build/enum_tensor_bind_repro
// If Holoscan is not on the default prefix:
//   cmake -S . -B build -Dholoscan_DIR=/path/to/holoscan/lib/cmake/holoscan

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
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

namespace {

inline constexpr auto kTick = std::chrono::milliseconds{5};
inline constexpr auto kDeadline = std::chrono::seconds{5};
inline constexpr auto kPoll = std::chrono::milliseconds{25};
inline constexpr int kFrames = 3;
inline constexpr std::uint32_t kSidebandValue = 7U;

// Same underlying width as uint32_t; Graph labels this kLocalEnum, not
// kUInt32 (holoscan/core/payload_identity.hpp).
enum class LocalMode : std::uint32_t {
  kValue = kSidebandValue,
};

struct Bank {
  static void reset(int n) {
    const std::lock_guard lock(mutex);
    remaining = n;
    captured = 0;
  }
  static inline std::mutex mutex;
  static inline int remaining{0};
  static inline int captured{0};
};

void bind_device(holoscan::CompileOptions& options, const char* op,
                 const char* port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = op,
          .output_port = port,
          .device = holoscan::DeviceId{0},
      });
}

holoscan::expected<void, holoscan::Error> emit_one_f32(
    holoscan::Output<holoscan::Tensor>& out,
    holoscan::ExecutionContext& context) {
  const std::vector<int64_t> dims{1};
  const DLDataType dtype{kDLFloat, 32, 1};
  auto loan = out.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = holoscan::TensorShapeView{dims.data(), dims.size()},
      .dtype = dtype,
  });
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const float one = 1.0F;
  if (cudaMemcpyAsync(writer->data(), &one, sizeof(one),
                      cudaMemcpyHostToDevice,
                      context.cuda_stream().get()) != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        "emit_one_f32: cudaMemcpyAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(
      std::move(*loan),
      holoscan::EmitOptions{.capture_time = context.activation_time()});
}

template <typename Sideband>
class CoEmitSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(tensor_out, "tensor_out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kDevice,
            .bounds = holoscan::TensorBounds{.max_rank = 1U,
                                             .max_byte_span = sizeof(float)},
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
    spec.output(sideband_out, "sideband_out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    {
      const std::lock_guard lock(Bank::mutex);
      if (Bank::remaining <= 0) {
        return {};
      }
      --Bank::remaining;
    }
    Sideband value{};
    if constexpr (std::is_enum_v<Sideband>) {
      value = Sideband::kValue;
    } else {
      value = static_cast<Sideband>(kSidebandValue);
    }
    HOLOSCAN_RETURN_IF_ERROR(sideband_out.emit(
        value,
        holoscan::EmitOptions{.capture_time = context.activation_time()}));
    return emit_one_f32(tensor_out, context);
  }

  holoscan::Output<holoscan::Tensor> tensor_out{"tensor_out"};
  holoscan::Output<Sideband> sideband_out{"sideband_out"};
};

template <typename Sideband>
class PassThroughOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(in, "in")
        .queue_depth(4U)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kDevice});
    spec.input(sideband, "sideband").queue_depth(4U);
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kDevice,
            .bounds = holoscan::TensorBounds{.max_rank = 1U,
                                             .max_byte_span = sizeof(float)},
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    result
        .input(sideband,
               holoscan::LatestWithin{.max_age = std::chrono::years{100}})
        .on_missing(sideband, holoscan::SafeDefault{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = in.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    (void)sideband.receive_data();
    const holoscan::Tensor& input = *sample;
    auto loan = out.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = input.shape(),
        .dtype = input.dtype(),
    });
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (cudaMemcpyAsync(writer->data(), input.data(), writer->byte_size(),
                        cudaMemcpyDeviceToDevice,
                        context.cuda_stream().get()) != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure,
                          "PassThroughOp: D2D copy failed"});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return out.emit(
        std::move(*loan),
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
  holoscan::Input<Sideband> sideband{"sideband"};
  holoscan::Output<holoscan::Tensor> out{"out"};
};

class CaptureSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = in.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    cudaDeviceSynchronize();
    {
      const std::lock_guard lock(Bank::mutex);
      ++Bank::captured;
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

struct ReproResult {
  bool plan_ok{false};
  int captured{0};
  std::vector<std::string> diagnostics;
  std::string plan_json;
};

template <typename Sideband>
ReproResult run_graph(const char* graph_name) {
  Bank::reset(kFrames);
  holoscan::Graph graph{graph_name};
  const auto src = graph.op<CoEmitSourceOp<Sideband>>("src");
  const auto op = graph.op<PassThroughOp<Sideband>>("op");
  const auto sink = graph.op<CaptureSinkOp>("sink");
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(src->tensor_out, op->in, q);
  graph.add_flow(src->sideband_out, op->sideband, q);
  graph.add_flow(op->out, sink->in, q);
  graph.partition("perf").add(src).add(op).add(sink);

  holoscan::CompileOptions options;
  bind_device(options, "src", "tensor_out");
  bind_device(options, "op", "out");
  graph.set_default_clock(
      graph.add_clock<holoscan::RealtimeClock>("clock"));

  ReproResult result;
  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  result.plan_ok = plan.ok();
  result.plan_json = std::string(plan.json());
  if (!plan.ok()) {
    return result;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + kDeadline;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      const std::lock_guard lock(Bank::mutex);
      if (Bank::captured >= kFrames) {
        break;
      }
    }
    if (!session.diagnostics().empty()) {
      break;
    }
    std::this_thread::sleep_for(kPoll);
  }
  session.request_stop();
  session.wait();
  for (const auto& d : session.diagnostics()) {
    result.diagnostics.emplace_back(d.token);
  }
  {
    const std::lock_guard lock(Bank::mutex);
    result.captured = Bank::captured;
  }
  return result;
}

std::string summarize(const ReproResult& r) {
  std::string msg = "plan_ok=";
  msg += r.plan_ok ? "true" : "false";
  msg += " captured=";
  msg += std::to_string(r.captured);
  msg += " diagnostics=";
  for (const auto& t : r.diagnostics) {
    msg += " ";
    msg += t;
  }
  return msg;
}

bool ran_clean(const ReproResult& r) {
  return r.plan_ok && r.diagnostics.empty() && r.captured == kFrames;
}

bool has_bind_token(const ReproResult& r) {
  const char* known[] = {
      "RUNTIME_PORT_BINDING_UNAVAILABLE",
      "RUNTIME_EXECUTION_LANE_PLAN_INVALID",
      "RUNTIME_OPERATOR_CALLBACK_FAILED",
      "RUNTIME_OPERATOR_REPLAY_MISMATCH",
      "PARTITION_MANUAL_AUTHORING_REQUIRED",
  };
  for (const auto& t : r.diagnostics) {
    for (const char* k : known) {
      if (t == k) {
        return true;
      }
    }
  }
  for (const char* k : known) {
    if (r.plan_json.find(k) != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main() {
  std::cout
      << "Graph: co-emit CUDA Tensor + sideband, one partition,\n"
      << "       sideband in contract (LatestWithin + SafeDefault).\n"
      << "Control sideband: uint32_t (kUInt32).\n"
      << "Probe sideband:   enum class LocalMode : uint32_t (kLocalEnum).\n\n";

  const ReproResult uint32_result =
      run_graph<std::uint32_t>("enum-bind-uint32");
  std::cout << "uint32_t:    " << summarize(uint32_result) << "\n";

  const ReproResult enum_result =
      run_graph<LocalMode>("enum-bind-local-enum");
  std::cout << "enum class:  " << summarize(enum_result) << "\n\n";

  if (!ran_clean(uint32_result)) {
    std::cerr
        << "CONTROL FAILED: uint32_t sideband did not run. The graph shape "
           "is wrong for this SDK; this is not an enum-only result.\n";
    return 2;
  }
  if (ran_clean(enum_result)) {
    std::cout
        << "BOTH PASSED: kLocalEnum + CUDA Tensor co-emit works on this "
           "SDK (EA gap closed).\n";
    return 0;
  }
  if (has_bind_token(enum_result)) {
    std::cout
        << "REPRODUCED: uint32_t works, enum class fails to bind/run.\n";
    return 0;
  }
  std::cerr << "UNEXPECTED enum failure (no known diagnostic token).\n";
  return 3;
}
