// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modality-independent integration test for the reusable selective-angle
// cascade. The fixture uses synthetic array snapshots and a deterministic
// estimator stub; it deliberately has no simulator, TDM, or application code.

#include <gtest/gtest.h>

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
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

#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "detection/multi_source_expansion_op/multi_source_expansion_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/integration_graph_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace hr = holoscan::radar;
namespace test = holoscan::radar::testing;

constexpr DLDataType kInt64Dtype{kDLInt, 64U, 1U};
constexpr DLDataType kInt32Dtype{kDLInt, 32U, 1U};

struct FixtureBank {
  static void reset() {
    const std::lock_guard lock(mutex);
    snapshot.reset();
    baseline.reset();
    detection_idx.reset();
    source_time.reset();
    result_idx = {};
    result_aoa = {};
    valid = {};
    score = {};
    mask = {};
    result_time.reset();
    diagnostics_time.reset();
    result_ready = false;
    diagnostics_ready = false;
    frames.clear();
    source_times.clear();
    result_times.clear();
    emitted = 0;
    results = 0;
    diagnostics = 0;
    retain = false;
    retained_idx.reset();
    retained_aoa.reset();
    retained_points.reset();
    result_points = {};
    later_idx = {};
    later_aoa = {};
    later_points = {};
    idx_addresses.clear();
    aoa_addresses.clear();
    point_addresses.clear();
    isolated_backing = true;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::shared_ptr<holoscan::Tensor> snapshot;
  static inline std::shared_ptr<holoscan::Tensor> baseline;
  static inline std::shared_ptr<holoscan::Tensor> detection_idx;
  static inline std::optional<holoscan::TimePoint> source_time;
  static inline test::HostTensor result_idx;
  static inline test::HostTensor result_aoa;
  static inline test::HostTensor valid;
  static inline test::HostTensor score;
  static inline test::HostTensor mask;
  static inline std::optional<holoscan::TimePoint> result_time;
  static inline std::optional<holoscan::TimePoint> diagnostics_time;
  static inline bool result_ready{false};
  static inline bool diagnostics_ready{false};
  struct Frame {
    std::shared_ptr<holoscan::Tensor> snapshot;
    std::shared_ptr<holoscan::Tensor> baseline;
    std::shared_ptr<holoscan::Tensor> detection_idx;
  };
  static inline std::vector<Frame> frames;
  static inline std::vector<holoscan::TimePoint> source_times;
  static inline std::vector<holoscan::TimePoint> result_times;
  static inline size_t emitted{0};
  static inline size_t results{0};
  static inline size_t diagnostics{0};
  static inline bool retain{false};
  // Tensor values are SDK native read handles, each retaining a pool backing claim.
  // These are consumer-received values, never exported snapshots or copied payloads.
  static inline std::optional<holoscan::Tensor> retained_idx;
  static inline std::optional<holoscan::Tensor> retained_aoa;
  static inline std::optional<holoscan::Tensor> retained_points;
  static inline test::HostTensor result_points;
  static inline test::HostTensor later_idx;
  static inline test::HostTensor later_aoa;
  static inline test::HostTensor later_points;
  static inline std::set<const void*> idx_addresses;
  static inline std::set<const void*> aoa_addresses;
  static inline std::set<const void*> point_addresses;
  static inline bool isolated_backing{true};
};

class FixtureSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation device{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt};
    constexpr std::size_t kMaxBytes = 1024U * 1024U;
    spec.output(snapshot, "snapshot")
        .produces_tensor(hr::pooled_tensor_output(device, 3U, kMaxBytes));
    spec.output(baseline, "baseline")
        .produces_tensor(hr::pooled_tensor_output(device, 2U, kMaxBytes));
    spec.output(detection_idx, "detection_idx")
        .produces_tensor(hr::pooled_tensor_output(device, 2U, kMaxBytes));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{2}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> snap;
    std::shared_ptr<holoscan::Tensor> base;
    std::shared_ptr<holoscan::Tensor> idx;
    {
      const std::lock_guard lock(FixtureBank::mutex);
      // Admit one whole-chain frame only after both sinks release transient handles.
      // Keeping the first frame consumes one extra slot, not unbounded history.
      if (FixtureBank::emitted >= FixtureBank::frames.size() ||
          FixtureBank::results != FixtureBank::emitted ||
          FixtureBank::diagnostics != FixtureBank::emitted) {
        return {};
      }
      auto& frame = FixtureBank::frames[FixtureBank::emitted++];
      FixtureBank::source_times.push_back(context.activation_time());
      if (!FixtureBank::source_time) {
        FixtureBank::source_time = context.activation_time();
      }
      snap = frame.snapshot;
      base = frame.baseline;
      idx = frame.detection_idx;
    }
    HOLOSCAN_RETURN_IF_ERROR(
        test::integration::emit_bank_frame(snapshot, context, snap, "FixtureSourceOp.snapshot"));
    HOLOSCAN_RETURN_IF_ERROR(
        test::integration::emit_bank_frame(baseline, context, base, "FixtureSourceOp.baseline"));
    return test::integration::emit_bank_frame(detection_idx, context, idx,
                                              "FixtureSourceOp.detection_idx");
  }

  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Output<holoscan::Tensor> baseline{"baseline"};
  holoscan::Output<holoscan::Tensor> detection_idx{"detection_idx"};
};

template <typename Element>
holoscan::expected<void, holoscan::Error> emit_values(holoscan::Output<holoscan::Tensor>& output,
                                                      holoscan::ExecutionContext& context,
                                                      const std::vector<Element>& values,
                                                      const std::vector<int64_t>& shape,
                                                      DLDataType dtype) {
  auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  HOLOSCAN_TRY(auto writer, loan->write(context.cuda_stream()));
  if (!values.empty() &&
      cudaMemcpyAsync(writer.data(), values.data(), values.size() * sizeof(Element),
                      cudaMemcpyHostToDevice, context.cuda_stream().get()) != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "SyntheticEstimatorOp: H2D failed"});
  }
  HOLOSCAN_TRY(std::move(writer).commit());
  return output.emit(std::move(*loan),
                     holoscan::EmitOptions{.capture_time = context.activation_time()});
}

template <typename T>
class SyntheticEstimatorOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation device{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt};
    spec.input(snapshot, "snapshot").expects_tensor(hr::tensor_input(device));
    spec.input(selected_idx, "selected_idx").expects_tensor(hr::tensor_input(device));
    spec.output(aoa_estimates, "aoa_estimates")
        .produces_tensor(hr::pooled_tensor_output(device, 2U, 4096U));
    spec.output(doa_multi, "doa_multi")
        .produces_tensor(hr::pooled_tensor_output(device, 3U, 8192U));
    spec.output(source_count, "source_count")
        .produces_tensor(hr::pooled_tensor_output(device, 1U, 4096U));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned{.ports = holoscan::ports(snapshot, selected_idx),
                                     .window = std::chrono::nanoseconds{0}})
        .input(snapshot, holoscan::Exact{})
        .input(selected_idx, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    HOLOSCAN_TRY(auto snapshots, hr::receive_tensor(snapshot, "SyntheticEstimatorOp"));
    HOLOSCAN_TRY(auto selected, hr::receive_tensor(selected_idx, "SyntheticEstimatorOp"));
    const int64_t rows = selected.shape().empty() ? 0 : selected.shape()[0];
    if (snapshots.shape().size() != 3U || snapshots.shape()[0] != rows) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                       "SyntheticEstimatorOp: row mismatch"});
    }

    std::vector<T> dominant(static_cast<size_t>(rows * 2), 0.0F);
    std::vector<T> multi(static_cast<size_t>(rows * 4), 0.0F);
    std::vector<int32_t> count(static_cast<size_t>(rows), 2);
    if (rows > 1) {
      count[1] = 1;
    }
    const T delta = static_cast<T>(frame_++) * static_cast<T>(0.01F);
    for (int64_t row = 0; row < rows; ++row) {
      dominant[static_cast<size_t>(row * 2)] = static_cast<T>(-0.10F) + delta;
      multi[static_cast<size_t>(row * 4)] = static_cast<T>(-0.10F) + delta;
      multi[static_cast<size_t>(row * 4 + 2)] = static_cast<T>(0.10F) + delta;
    }
    HOLOSCAN_RETURN_IF_ERROR(
        emit_values(aoa_estimates, context, dominant, {rows, 2}, hr::real_dtype<T>()));
    HOLOSCAN_RETURN_IF_ERROR(
        emit_values(doa_multi, context, multi, {rows, 2, 2}, hr::real_dtype<T>()));
    return emit_values(source_count, context, count, {rows}, kInt32Dtype);
  }

  size_t frame_{0};
  holoscan::Input<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Input<holoscan::Tensor> selected_idx{"selected_idx"};
  holoscan::Output<holoscan::Tensor> aoa_estimates{"aoa_estimates"};
  holoscan::Output<holoscan::Tensor> doa_multi{"doa_multi"};
  holoscan::Output<holoscan::Tensor> source_count{"source_count"};
};

class ResultSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(idx, "idx");
    spec.input(aoa, "aoa");
    spec.input(points, "points");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned{.ports = holoscan::ports(idx, aoa, points),
                                     .window = std::chrono::nanoseconds{0}})
        .input(idx, holoscan::Exact{})
        .input(aoa, holoscan::Exact{})
        .input(points, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    {
      HOLOSCAN_TRY(auto idx_sample, idx.receive());
      HOLOSCAN_TRY(auto aoa_sample, aoa.receive());
      HOLOSCAN_TRY(auto points_sample, points.receive());
      auto idx_value = std::move(idx_sample.data);
      auto aoa_value = std::move(aoa_sample.data);
      auto points_value = std::move(points_sample.data);
      {
        const std::lock_guard lock(FixtureBank::mutex);
        if (idx_sample.metadata.capture_time() != aoa_sample.metadata.capture_time() ||
            idx_sample.metadata.capture_time() != points_sample.metadata.capture_time()) {
          return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure,
                                                           "ResultSinkOp: capture-time mismatch"});
        }
        FixtureBank::result_times.push_back(context.activation_time());
        if (FixtureBank::retain && FixtureBank::results == 0) {
          // Move the actual native read handles. Do not inspect/copy their data yet.
          FixtureBank::retained_idx.emplace(std::move(idx_value));
          FixtureBank::retained_aoa.emplace(std::move(aoa_value));
          FixtureBank::retained_points.emplace(std::move(points_value));
        } else {
          if (FixtureBank::retain) {
            FixtureBank::isolated_backing &=
                idx_value.data() != FixtureBank::retained_idx->data() &&
                aoa_value.data() != FixtureBank::retained_aoa->data() &&
                points_value.data() != FixtureBank::retained_points->data();
            FixtureBank::idx_addresses.insert(idx_value.data());
            FixtureBank::aoa_addresses.insert(aoa_value.data());
            FixtureBank::point_addresses.insert(points_value.data());
          }
          FixtureBank::later_idx = test::to_host(idx_value);
          FixtureBank::later_aoa = test::to_host(aoa_value);
          FixtureBank::later_points = test::to_host(points_value);
          if (!FixtureBank::retain) {
            FixtureBank::result_idx = FixtureBank::later_idx;
            FixtureBank::result_aoa = FixtureBank::later_aoa;
            FixtureBank::result_points = FixtureBank::later_points;
          }
        }
        if (FixtureBank::results + 1U == FixtureBank::frames.size()) {
          if (FixtureBank::retain) {
            // Resolve producer readiness before runtime stream teardown, as SDK Retain does.
            HOLOSCAN_TRY(FixtureBank::retained_idx->synchronize_for_host());
            HOLOSCAN_TRY(FixtureBank::retained_aoa->synchronize_for_host());
            HOLOSCAN_TRY(FixtureBank::retained_points->synchronize_for_host());
            FixtureBank::result_idx = test::to_host(*FixtureBank::retained_idx);
            FixtureBank::result_aoa = test::to_host(*FixtureBank::retained_aoa);
            FixtureBank::result_points = test::to_host(*FixtureBank::retained_points);
          }
          FixtureBank::result_time = FixtureBank::source_times.front();
          FixtureBank::result_ready = true;
        }
      }
    }  // Release received Samples and transient Tensor handles before admitting another frame.
    {
      const std::lock_guard lock(FixtureBank::mutex);
      ++FixtureBank::results;
    }
    FixtureBank::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> idx{"idx"};
  holoscan::Input<holoscan::Tensor> aoa{"aoa"};
  holoscan::Input<holoscan::Tensor> points{"points"};
};

class DiagnosticsSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(valid, "valid");
    spec.input(score, "score");
    spec.input(mask, "mask");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned{.ports = holoscan::ports(valid, score, mask),
                                     .window = std::chrono::nanoseconds{0}})
        .input(valid, holoscan::Exact{})
        .input(score, holoscan::Exact{})
        .input(mask, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    {
      HOLOSCAN_TRY(auto valid_value, valid.receive_data());
      HOLOSCAN_TRY(auto score_value, score.receive_data());
      HOLOSCAN_TRY(auto mask_value, mask.receive_data());
      {
        const std::lock_guard lock(FixtureBank::mutex);
        FixtureBank::valid = test::to_host(valid_value);
        FixtureBank::score = test::to_host(score_value);
        FixtureBank::mask = test::to_host(mask_value);
        if (!FixtureBank::diagnostics_time) {
          FixtureBank::diagnostics_time = context.activation_time();
        }
      }
    }
    {
      const std::lock_guard lock(FixtureBank::mutex);
      ++FixtureBank::diagnostics;
      FixtureBank::diagnostics_ready = FixtureBank::diagnostics == FixtureBank::frames.size();
    }
    FixtureBank::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> valid{"valid"};
  holoscan::Input<holoscan::Tensor> score{"score"};
  holoscan::Input<holoscan::Tensor> mask{"mask"};
};

hr::ArrayGeometry make_geometry(int64_t channels) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = hr::ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < channels; ++channel) {
    geometry.element_positions.insert(geometry.element_positions.end(),
                                      {0.0, 0.5 * static_cast<double>(channel), 0.0});
  }
  return geometry;
}

template <typename T>
std::vector<cuda::std::complex<T>> steering(const hr::ArrayGeometry& geometry, T azimuth) {
  std::vector<cuda::std::complex<T>> result(static_cast<size_t>(geometry.num_elements()));
  for (int64_t channel = 0; channel < geometry.num_elements(); ++channel) {
    const double y = geometry.element_positions[static_cast<size_t>(channel * 3 + 1)];
    const double phase = 2.0 * M_PI * y * std::sin(azimuth);
    result[static_cast<size_t>(channel)] = {static_cast<T>(std::cos(phase)),
                                            static_cast<T>(std::sin(phase))};
  }
  return result;
}

void bind_output(holoscan::CompileOptions* options, const char* op, const char* port) {
  options->deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
}

// Release device inputs and native output claims while SDK/CUDA process state is
// alive, including when a fatal assertion exits the test helper early. The
// RunSession is declared later and therefore tears down before this guard runs.
struct FixtureBankCleanup {
  ~FixtureBankCleanup() { FixtureBank::reset(); }
};

template <typename T>
void run_selective_chain(bool retain) {
  constexpr int64_t kItems = 4;
  constexpr int64_t kChannels = 8;
  constexpr int64_t kLooks = 2;
  FixtureBank::reset();
  const FixtureBankCleanup cleanup;

  const auto geometry = make_geometry(kChannels);
  const auto boresight = steering<T>(geometry, 0.0F);
  const auto secondary = steering<T>(geometry, 0.25F);
  const auto isolated = steering<T>(geometry, -0.15F);
  std::vector<cuda::std::complex<T>> snapshots(static_cast<size_t>(kItems * kChannels * kLooks));
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    for (int64_t look = 0; look < kLooks; ++look) {
      snapshots[static_cast<size_t>(channel * kLooks + look)] =
          boresight[static_cast<size_t>(channel)];
      snapshots[static_cast<size_t>((kChannels + channel) * kLooks + look)] =
          boresight[static_cast<size_t>(channel)] + secondary[static_cast<size_t>(channel)];
      snapshots[static_cast<size_t>((2 * kChannels + channel) * kLooks + look)] =
          boresight[static_cast<size_t>(channel)] + secondary[static_cast<size_t>(channel)];
      snapshots[static_cast<size_t>((3 * kChannels + channel) * kLooks + look)] =
          isolated[static_cast<size_t>(channel)];
    }
  }
  FixtureBank::snapshot =
      test::to_device_tensor(snapshots, {kItems, kChannels, kLooks}, hr::complex_dtype<T>());
  FixtureBank::baseline =
      test::to_device_tensor(std::vector<T>{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -0.15F, 0.0F},
                             {kItems, 2}, hr::real_dtype<T>());
  FixtureBank::detection_idx = test::to_device_tensor(
      std::vector<int64_t>{1, 10, 100, 1000, 2, 20, 200, 2000, 3, 30, 300, 3000, 4, 40, 400, 4000},
      {kItems, 4}, kInt64Dtype);

  holoscan::Graph graph{"selective-aoa-integration"};
  const auto source = graph.op<FixtureSourceOp>("source");

  hr::MaxTensorExtents screen_extents;
  screen_extents.max_items = kItems;
  screen_extents.max_channels = kChannels;
  screen_extents.max_snapshots = kLooks;
  auto screen = graph.op<hr::RankOneResidualScreenOp<T>>("screen", screen_extents);
  hr::RankOneResidualScreenParams screen_params;
  screen_params.array_geometry = geometry;
  screen_params.residual_threshold = 0.03;
  screen.set_params(screen_params);

  const auto estimator = graph.op<SyntheticEstimatorOp<T>>("synthetic_estimator");

  hr::MaxTensorExtents expansion_extents;
  expansion_extents.max_items = kItems;
  expansion_extents.max_points = 8;
  expansion_extents.max_sources = 2;
  const auto expansion = graph.op<hr::MultiSourceExpansionOp<T>>("expansion", expansion_extents);
  hr::MaxTensorExtents feature_extents;
  feature_extents.max_points = 8;
  feature_extents.max_features = 4;
  const auto features = graph.op<hr::FeatureAssemblyOp<T>>(
      "features", feature_extents,
      std::vector<hr::AssemblyFeature>{
          hr::AssemblyFeature::kDopplerBin, hr::AssemblyFeature::kRangeBin,
          hr::AssemblyFeature::kAzEstimate, hr::AssemblyFeature::kElEstimate});
  const auto result_sink = graph.op<ResultSinkOp>("result_sink");
  const auto diagnostic_sink = graph.op<DiagnosticsSinkOp>("diagnostic_sink");

  const holoscan::ConnectionOptions queue{.queue_depth = 1U};
  graph.add_flow(source->snapshot, screen->snapshot, queue);
  graph.add_flow(source->baseline, screen->baseline_aoa, queue);
  graph.add_flow(source->detection_idx, expansion->detection_idx, queue);
  graph.add_flow(source->baseline, expansion->baseline_aoa, queue);
  graph.add_flow(screen->snapshot_out, estimator->snapshot, queue);
  graph.add_flow(screen->selected_idx, estimator->selected_idx, queue);
  graph.add_flow(screen->selected_idx, expansion->selected_idx, queue);
  graph.add_flow(estimator->aoa_estimates, expansion->resolved_aoa, queue);
  graph.add_flow(estimator->doa_multi, expansion->doa_multi, queue);
  graph.add_flow(estimator->source_count, expansion->source_count, queue);
  graph.add_flow(expansion->detection_idx_out, features->detection_idx, queue);
  graph.add_flow(expansion->aoa_estimates_out, features->aoa_estimates, queue);
  graph.add_flow(features->points, result_sink->points, queue);
  graph.add_flow(expansion->detection_idx_out, result_sink->idx, queue);
  graph.add_flow(expansion->aoa_estimates_out, result_sink->aoa, queue);
  graph.add_flow(screen->is_valid_out, diagnostic_sink->valid, queue);
  graph.add_flow(screen->residual_score, diagnostic_sink->score, queue);
  graph.add_flow(screen->selection_mask, diagnostic_sink->mask, queue);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const char* port : {"snapshot", "baseline", "detection_idx"}) {
    bind_output(&options, "source", port);
  }
  for (const char* port :
       {"snapshot_out", "selected_idx", "is_valid_out", "residual_score", "selection_mask"}) {
    bind_output(&options, "screen", port);
  }
  for (const char* port : {"aoa_estimates", "doa_multi", "source_count"}) {
    bind_output(&options, "synthetic_estimator", port);
  }
  bind_output(&options, "expansion", "detection_idx_out");
  bind_output(&options, "expansion", "aoa_estimates_out");
  bind_output(&options, "features", "points");

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  size_t max_slots = 0;
  for (const auto& allocation : plan.tensor_allocations()) {
    max_slots = std::max(max_slots, static_cast<size_t>(allocation.slot_count));
    // One retained history slot and one current loan suffice: source pacing and
    // synchronous sink reads complete transient retirement before the next tick.
    ASSERT_GE(allocation.slot_count, 2U) << allocation.pool_id;
    ::testing::Test::RecordProperty("pool_" + std::to_string(allocation.output.value()),
                                    std::to_string(allocation.slot_count));
  }
  ASSERT_GT(max_slots, 0U);
  ASSERT_LE(max_slots, 125U);
  const size_t frame_count = retain ? max_slots + 3U : 1U;
  FixtureBank::retain = retain;
  for (size_t frame = 0; frame < frame_count; ++frame) {
    auto indices = test::to_host(*FixtureBank::detection_idx);
    std::vector<int64_t> changed(static_cast<size_t>(indices.size()));
    for (size_t i = 0; i < changed.size(); ++i) {
      changed[i] = indices.as<int64_t>()[i] + static_cast<int64_t>(100U * frame);
    }
    FixtureBank::frames.push_back(
        FixtureBank::Frame{FixtureBank::snapshot, FixtureBank::baseline,
                           test::to_device_tensor(changed, {kItems, 4}, kInt64Dtype)});
  }
  // The horizon exceeds every compiled pool's slot bound while retaining only
  // the first frame. No output pool capacity or production extents are enlarged.
  {
    holoscan::RunSession session = holoscan::run_async(plan);
    {
      std::unique_lock lock(FixtureBank::mutex);
      ASSERT_TRUE(FixtureBank::ready.wait_for(lock, std::chrono::seconds{15}, [] {
        return FixtureBank::result_ready && FixtureBank::diagnostics_ready;
      }));
    }
    session.request_stop();
    session.wait();
    ASSERT_TRUE(session.diagnostics().empty());
  }  // Runtime-owned streams are gone; native Tensor read handles remain alive.
  if (retain) {
    ASSERT_TRUE(FixtureBank::retained_idx.has_value());
    ASSERT_TRUE(FixtureBank::retained_aoa.has_value());
    ASSERT_TRUE(FixtureBank::retained_points.has_value());
    for (const auto* tensor : {&*FixtureBank::retained_idx, &*FixtureBank::retained_aoa,
                               &*FixtureBank::retained_points}) {
      EXPECT_EQ(tensor->device().device_type, kDLCUDA);
      EXPECT_NE(tensor->data(), nullptr);
    }
    EXPECT_EQ(test::to_host(*FixtureBank::retained_idx).data, FixtureBank::result_idx.data);
    EXPECT_EQ(test::to_host(*FixtureBank::retained_aoa).data, FixtureBank::result_aoa.data);
    EXPECT_EQ(test::to_host(*FixtureBank::retained_points).data, FixtureBank::result_points.data);
    EXPECT_TRUE(FixtureBank::isolated_backing);
    EXPECT_LT(FixtureBank::idx_addresses.size(), frame_count - 1U);
    EXPECT_LT(FixtureBank::aoa_addresses.size(), frame_count - 1U);
    EXPECT_LT(FixtureBank::point_addresses.size(), frame_count - 1U);
    EXPECT_EQ(FixtureBank::later_idx.as<int64_t>()[0],
              1 + static_cast<int64_t>(100U * (frame_count - 1U)));
    EXPECT_NE(FixtureBank::later_aoa.data, FixtureBank::result_aoa.data);
    EXPECT_NE(FixtureBank::later_points.data, FixtureBank::result_points.data);
  }
  ASSERT_EQ(FixtureBank::results, frame_count);
  ASSERT_EQ(FixtureBank::diagnostics, frame_count);
  EXPECT_EQ(FixtureBank::result_times.size(), FixtureBank::source_times.size());
  for (size_t frame = 0; frame < frame_count; ++frame) {
    EXPECT_EQ(FixtureBank::result_times[frame].timestamp_ns,
              FixtureBank::source_times[frame].timestamp_ns);
  }

  EXPECT_EQ(FixtureBank::mask.shape, (std::vector<int64_t>{kItems}));
  EXPECT_EQ(FixtureBank::valid.shape, (std::vector<int64_t>{2, kLooks}));
  EXPECT_EQ(FixtureBank::mask.as<uint8_t>()[0], 0);
  EXPECT_EQ(FixtureBank::mask.as<uint8_t>()[1], 1);
  EXPECT_EQ(FixtureBank::mask.as<uint8_t>()[2], 1);
  EXPECT_EQ(FixtureBank::mask.as<uint8_t>()[3], 0);
  EXPECT_NEAR(FixtureBank::score.as<T>()[0], 0.0F, 1e-5F);
  EXPECT_GT(FixtureBank::score.as<T>()[1], 0.03F);
  EXPECT_GT(FixtureBank::score.as<T>()[2], 0.03F);
  EXPECT_NEAR(FixtureBank::score.as<T>()[3], 0.0F, 1e-5F);

  ASSERT_EQ(FixtureBank::result_idx.shape, (std::vector<int64_t>{5, 4}));
  const std::vector<int64_t> expected_idx{1,   10,   100, 1000, 2,   20,   200, 2000, 2,   20,
                                          200, 2000, 3,   30,   300, 3000, 4,   40,   400, 4000};
  for (size_t i = 0; i < expected_idx.size(); ++i) {
    EXPECT_EQ(FixtureBank::result_idx.as<int64_t>()[i], expected_idx[i]);
  }
  ASSERT_EQ(FixtureBank::result_aoa.shape, (std::vector<int64_t>{5, 2}));
  const std::vector<T> expected_aoa{0.0F, 0.0F, -0.10F, 0.0F,   0.10F,
                                    0.0F, 0.0F, 0.0F,   -0.15F, 0.0F};
  for (size_t i = 0; i < expected_aoa.size(); ++i) {
    EXPECT_NEAR(FixtureBank::result_aoa.as<T>()[i], expected_aoa[i], 1e-6F);
  }

  ASSERT_EQ(FixtureBank::result_points.shape, (std::vector<int64_t>{5, 4}));
  for (size_t row = 0; row < 5U; ++row) {
    EXPECT_EQ(FixtureBank::result_points.as<T>()[4U * row],
              static_cast<T>(expected_idx[4U * row + 2U]));
    EXPECT_EQ(FixtureBank::result_points.as<T>()[4U * row + 1U],
              static_cast<T>(expected_idx[4U * row + 3U]));
    EXPECT_EQ(FixtureBank::result_points.as<T>()[4U * row + 2U], expected_aoa[2U * row]);
    EXPECT_EQ(FixtureBank::result_points.as<T>()[4U * row + 3U], expected_aoa[2U * row + 1U]);
  }

  if (retain) {
    const T delta = static_cast<T>(frame_count - 1U) * static_cast<T>(0.01F);
    for (size_t row = 0; row < 5U; ++row) {
      for (size_t column = 0; column < 4U; ++column) {
        EXPECT_EQ(
            FixtureBank::later_idx.as<int64_t>()[4U * row + column],
            expected_idx[4U * row + column] + static_cast<int64_t>(100U * (frame_count - 1U)));
      }
      const T azimuth = expected_aoa[2U * row] + ((row == 1U || row == 2U) ? delta : T{0});
      EXPECT_EQ(FixtureBank::later_aoa.as<T>()[2U * row], azimuth);
      EXPECT_EQ(FixtureBank::later_aoa.as<T>()[2U * row + 1U], T{0});
      EXPECT_EQ(FixtureBank::later_points.as<T>()[4U * row],
                static_cast<T>(expected_idx[4U * row + 2U] +
                               static_cast<int64_t>(100U * (frame_count - 1U))));
      EXPECT_EQ(FixtureBank::later_points.as<T>()[4U * row + 1U],
                static_cast<T>(expected_idx[4U * row + 3U] +
                               static_cast<int64_t>(100U * (frame_count - 1U))));
      EXPECT_EQ(FixtureBank::later_points.as<T>()[4U * row + 2U], azimuth);
      EXPECT_EQ(FixtureBank::later_points.as<T>()[4U * row + 3U], T{0});
    }
  }

  ASSERT_TRUE(FixtureBank::source_time.has_value());
  ASSERT_TRUE(FixtureBank::result_time.has_value());
  ASSERT_TRUE(FixtureBank::diagnostics_time.has_value());
  EXPECT_EQ(FixtureBank::result_time->timestamp_ns, FixtureBank::source_time->timestamp_ns);
  EXPECT_EQ(FixtureBank::diagnostics_time->timestamp_ns, FixtureBank::source_time->timestamp_ns);
}

TEST(SelectiveAoaOperatorsIntegration, ScreensAndExpandsSyntheticModalityIndependentSnapshots) {
  run_selective_chain<float>(false);
}

TEST(SelectiveAoaOperatorsIntegration, ScreensExpandsAndAssemblesDouble) {
  run_selective_chain<double>(false);
}

TEST(SelectiveAoaOperatorsIntegration, RetainsNativeExpansionAndFeatureLoansFloat) {
  run_selective_chain<float>(true);
}

TEST(SelectiveAoaOperatorsIntegration, RetainsNativeExpansionAndFeatureLoansDouble) {
  run_selective_chain<double>(true);
}

}  // namespace
