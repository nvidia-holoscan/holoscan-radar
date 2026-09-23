// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MultiSourceExpansionOp performance characterization. Sweeps original rows,
// selected fraction, confirmed source count, and detection-index width.
// Measurement only: no timing thresholds.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "detection/multi_source_expansion_op/multi_source_expansion_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;
constexpr DLDataType kInt64Dtype{kDLInt, 64U, 1U};
constexpr DLDataType kInt32Dtype{kDLInt, 32U, 1U};

struct SizeSpec {
  std::string id;
  int64_t rows;
  std::string role;
};

struct ExpansionScenario {
  std::string id;
  std::string size_id;
  int64_t rows{0};
  double selected_fraction{0.25};
  int64_t sources{2};
  int64_t index_columns{2};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {{"xs", 32, "small; performance smoke"},
                                               {"sm", 256, "compact detection set"},
                                               {"md", 2048, "common sparse-angle workload"},
                                               {"lg", 16384, "dense detection workload"},
                                               {"xl", 65536, "upper characterization bound"}};
  return ladder;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& size : size_ladder()) {
    if (size.id == id) {
      return &size;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* size = find_size(id);
  if (size == nullptr) {
    std::cerr << "unknown size id '" << id << "'\n";
    std::abort();
  }
  return *size;
}

template <typename Element>
std::shared_ptr<holoscan::Tensor> upload(const std::vector<Element>& host,
                                         const std::vector<int64_t>& shape, DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto writable = tensor->mutable_data();
    if (!writable || cudaMemcpy(writable.value(), host.data(), host.size() * sizeof(Element),
                                cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("perf_multi_source_expansion: input upload failed");
    }
  }
  return tensor;
}

int64_t selected_rows(const ExpansionScenario& scenario) {
  return std::max<int64_t>(
      1, static_cast<int64_t>(
             std::llround(scenario.selected_fraction * static_cast<double>(scenario.rows))));
}

struct ExpansionInputs {
  std::shared_ptr<holoscan::Tensor> detection_idx;
  std::shared_ptr<holoscan::Tensor> baseline;
  std::shared_ptr<holoscan::Tensor> selected;
  std::shared_ptr<holoscan::Tensor> resolved;
  std::shared_ptr<holoscan::Tensor> multi;
  std::shared_ptr<holoscan::Tensor> count;
};

ExpansionInputs make_inputs(const ExpansionScenario& scenario) {
  const int64_t selected_count = selected_rows(scenario);
  std::vector<int64_t> idx(static_cast<size_t>(scenario.rows * scenario.index_columns));
  std::vector<PerfPrecision> baseline(static_cast<size_t>(scenario.rows * 2));
  for (int64_t row = 0; row < scenario.rows; ++row) {
    for (int64_t column = 0; column < scenario.index_columns; ++column) {
      idx[static_cast<size_t>(row * scenario.index_columns + column)] = row * 11 + column;
    }
    baseline[static_cast<size_t>(row * 2)] = static_cast<PerfPrecision>((row % 101) * 0.001);
  }

  std::vector<int64_t> selected(static_cast<size_t>(selected_count));
  std::vector<PerfPrecision> resolved(static_cast<size_t>(selected_count * 2));
  std::vector<PerfPrecision> multi(static_cast<size_t>(selected_count * scenario.sources * 2));
  std::vector<int32_t> count(static_cast<size_t>(selected_count),
                             static_cast<int32_t>(scenario.sources));
  for (int64_t row = 0; row < selected_count; ++row) {
    selected[static_cast<size_t>(row)] = row;
    resolved[static_cast<size_t>(row * 2)] = -0.1F;
    for (int64_t source = 0; source < scenario.sources; ++source) {
      multi[static_cast<size_t>((row * scenario.sources + source) * 2)] =
          -0.1F + 0.05F * static_cast<PerfPrecision>(source);
    }
  }

  return ExpansionInputs{
      .detection_idx = upload(idx, {scenario.rows, scenario.index_columns}, kInt64Dtype),
      .baseline = upload(baseline, {scenario.rows, 2}, hr::real_dtype<PerfPrecision>()),
      .selected = upload(selected, {selected_count}, kInt64Dtype),
      .resolved = upload(resolved, {selected_count, 2}, hr::real_dtype<PerfPrecision>()),
      .multi =
          upload(multi, {selected_count, scenario.sources, 2}, hr::real_dtype<PerfPrecision>()),
      .count = upload(count, {selected_count}, kInt32Dtype)};
}

struct ExtraInputBank {
  static inline std::shared_ptr<holoscan::Tensor> d;
  static inline std::shared_ptr<holoscan::Tensor> e;
  static inline std::shared_ptr<holoscan::Tensor> f;

  static std::size_t span(const std::shared_ptr<holoscan::Tensor>& tensor) {
    if (!tensor || tensor->nbytes() <= 0) {
      throw std::runtime_error("six-input perf source has empty frame");
    }
    return static_cast<std::size_t>(tensor->nbytes());
  }
};

class SixTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation device{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt};
    spec.output(a, "a").produces_tensor(
        hr::pooled_tensor_output(device, 8U, perf::PerfFrameBank::span_a()));
    spec.output(b, "b").produces_tensor(
        hr::pooled_tensor_output(device, 8U, perf::PerfFrameBank::span_b()));
    spec.output(c, "c").produces_tensor(
        hr::pooled_tensor_output(device, 8U, perf::PerfFrameBank::span_c()));
    spec.output(d, "d").produces_tensor(
        hr::pooled_tensor_output(device, 8U, ExtraInputBank::span(ExtraInputBank::d)));
    spec.output(e, "e").produces_tensor(
        hr::pooled_tensor_output(device, 8U, ExtraInputBank::span(ExtraInputBank::e)));
    spec.output(f, "f").produces_tensor(
        hr::pooled_tensor_output(device, 8U, ExtraInputBank::span(ExtraInputBank::f)));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = perf::kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frames[6];
    perf::PerfPublishMode mode = perf::PerfPublishMode::kCopy;
    {
      const std::lock_guard lock(perf::PerfFrameBank::mutex);
      if (perf::PerfFrameBank::frames_remaining <= 0) {
        return {};
      }
      --perf::PerfFrameBank::frames_remaining;
      frames[0] = perf::PerfFrameBank::input_frame;
      frames[1] = perf::PerfFrameBank::input_frame_b;
      frames[2] = perf::PerfFrameBank::input_frame_c;
      frames[3] = ExtraInputBank::d;
      frames[4] = ExtraInputBank::e;
      frames[5] = ExtraInputBank::f;
      mode = perf::PerfFrameBank::publish_mode;
      perf::detail::record_start_locked();
    }
    HOLOSCAN_RETURN_IF_ERROR(perf::detail::emit_device_frame(a, context, frames[0], mode));
    HOLOSCAN_RETURN_IF_ERROR(perf::detail::emit_device_frame(b, context, frames[1], mode));
    HOLOSCAN_RETURN_IF_ERROR(perf::detail::emit_device_frame(c, context, frames[2], mode));
    HOLOSCAN_RETURN_IF_ERROR(perf::detail::emit_device_frame(d, context, frames[3], mode));
    HOLOSCAN_RETURN_IF_ERROR(perf::detail::emit_device_frame(e, context, frames[4], mode));
    return perf::detail::emit_device_frame(f, context, frames[5], mode);
  }

  holoscan::Output<holoscan::Tensor> a{"a"};
  holoscan::Output<holoscan::Tensor> b{"b"};
  holoscan::Output<holoscan::Tensor> c{"c"};
  holoscan::Output<holoscan::Tensor> d{"d"};
  holoscan::Output<holoscan::Tensor> e{"e"};
  holoscan::Output<holoscan::Tensor> f{"f"};
};

perf::TimingCollector run_expansion_graph(const ExpansionScenario& scenario, int total_frames) {
  auto inputs = make_inputs(scenario);
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::now();
  perf::PerfFrameBank::reset_triple(std::move(inputs.detection_idx), std::move(inputs.baseline),
                                    std::move(inputs.selected), total_frames, &collector);
  ExtraInputBank::d = std::move(inputs.resolved);
  ExtraInputBank::e = std::move(inputs.multi);
  ExtraInputBank::f = std::move(inputs.count);

  holoscan::Graph graph{"multi-source-expansion-perf"};
  const auto source = graph.op<SixTensorSourceOp>("source");

  hr::MaxTensorExtents extents;
  extents.max_items = scenario.rows;
  extents.max_sources = scenario.sources;
  extents.max_points = scenario.rows + selected_rows(scenario) * (scenario.sources - 1);
  const auto expansion = graph.op<hr::MultiSourceExpansionOp<PerfPrecision>>("expansion", extents);
  const auto timed = graph.op<perf::PerfTerminalSinkOp>("idx_sink");
  const auto aoa_drain = graph.op<perf::PerfNullSinkOp>("aoa_drain");
  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->a, expansion->detection_idx, queue);
  graph.add_flow(source->b, expansion->baseline_aoa, queue);
  graph.add_flow(source->c, expansion->selected_idx, queue);
  graph.add_flow(source->d, expansion->resolved_aoa, queue);
  graph.add_flow(source->e, expansion->doa_multi, queue);
  graph.add_flow(source->f, expansion->source_count, queue);
  graph.add_flow(expansion->detection_idx_out, timed->in, queue);
  graph.add_flow(expansion->aoa_estimates_out, aoa_drain->in, queue);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const char* port : {"a", "b", "c", "d", "e", "f"}) {
    perf::bind_tensor_out(options, "source", port);
  }
  perf::bind_tensor_out(options, "expansion", "detection_idx_out");
  perf::bind_tensor_out(options, "expansion", "aoa_estimates_out");
  const auto plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    throw std::runtime_error("multi-source-expansion perf compile failed: " +
                             std::string(plan.json()));
  }

  auto session = holoscan::run_async(plan);
  perf::detail::wait_for_frames(session, total_frames);
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    throw std::runtime_error("multi-source-expansion perf run produced diagnostics");
  }
  if (collector.ends.size() < static_cast<size_t>(total_frames)) {
    throw std::runtime_error(
        perf::detail::stall_message("multi-source-expansion", collector.ends.size(), total_frames));
  }
  return collector;
}

ExpansionScenario make_scenario(const SizeSpec& size, double fraction, int64_t sources,
                                int64_t columns) {
  ExpansionScenario scenario;
  scenario.size_id = size.id;
  scenario.rows = size.rows;
  scenario.selected_fraction = fraction;
  scenario.sources = sources;
  scenario.index_columns = columns;
  std::ostringstream id;
  id << "expand_" << size.id << "_n" << size.rows << "_sel" << static_cast<int>(fraction * 100.0)
     << "_s" << sources << "_w" << columns;
  scenario.id = id.str();
  return scenario;
}

perf::ResultRow base_row(const ExpansionScenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "MultiSourceExpansionOp";
  row.variant = "confirmed_sources";
  row.size_id = scenario.size_id;
  row.input_shape = {scenario.rows, scenario.index_columns};
  row.out_batch = scenario.rows + selected_rows(scenario) * (scenario.sources - 1);
  row.k = scenario.sources;
  row.notes = "both outputs drained";
  return row;
}

void fill_throughput(perf::ResultRow* row, const ExpansionScenario& scenario, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double seconds = median_ms / 1000.0;
  row->cells_per_second = static_cast<double>(scenario.rows) / seconds;
  row->slices_per_second = static_cast<double>(row->out_batch) / seconds;
}

struct ExpansionTraits : perf::PerfDriverDefaults {
  using Scenario = ExpansionScenario;

  static const char* tool_name() { return "perf_multi_source_expansion"; }
  static const char* tool_description() {
    return "MultiSourceExpansionOp HSDK 5 Graph characterization";
  }
  static const char* sizes_all_help() {
    return "Original-row sweep at 25 percent selected and two sources";
  }
  static const char* sizes_subset_help() { return "Selected row-count subset"; }
  static const char* full_help() {
    return "Rows x selected fraction x sources x index-column width";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    for (const auto& size : size_ladder()) {
      result.push_back(perf::SizeDef{
          .id = size.id, .shape = {size.rows, 2}, .cells = size.rows, .role = size.role});
    }
    return result;
  }

  static std::vector<Scenario> build_default_matrix() {
    const auto& medium = must_find_size("md");
    return {make_scenario(must_find_size("xs"), 0.25, 2, 2),
            make_scenario(medium, 0.10, 2, 2),
            make_scenario(medium, 0.50, 2, 2),
            make_scenario(medium, 0.25, 4, 2),
            make_scenario(medium, 0.25, 2, 10),
            make_scenario(must_find_size("lg"), 0.25, 2, 2)};
  }

  static std::vector<Scenario> build_size_sweep(const std::vector<std::string>& sizes) {
    std::vector<Scenario> result;
    for (const auto& id : sizes) {
      if (const auto* size = find_size(id)) {
        result.push_back(make_scenario(*size, 0.25, 2, 2));
      }
    }
    return result;
  }

  static std::vector<Scenario> build_full_matrix() {
    std::vector<Scenario> result;
    for (const auto& size : size_ladder()) {
      for (double fraction : {0.10, 0.50, 1.0}) {
        for (int64_t sources : {2, 3, 4}) {
          for (int64_t columns : {2, 4, 10}) {
            result.push_back(make_scenario(size, fraction, sources, columns));
          }
        }
      }
    }
    return result;
  }

  static void run_scenario(const Scenario& scenario, const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const auto plan = perf::make_run_plan(settings, /*has_override=*/false);
    const auto samples =
        perf::collect_scenario_samples(settings, plan, [&](int, const perf::RunPlan& run_plan) {
          return run_expansion_graph(scenario, run_plan.total_frames);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    auto row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scenario, row.latency.median_ms);
    row.notes =
        "cold_ms is first frame; median/p90 are steady state; both outputs "
        "are drained; cells/s is original rows/s; slices/s is emitted rows/s";
    report->results.push_back(std::move(row));
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs"), 0.25, 2, 2), settings, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<ExpansionTraits>(argc, argv);
}
