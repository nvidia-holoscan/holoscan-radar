// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_app/report/cadence_meter.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sim/scene.hpp"
#include "radar_common/sim/simulator.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"

namespace holoscan::radar::app {

using FrameToken = uint64_t;

struct TdmTransmitProfile {
  std::string name;
  FMCWWaveformDescriptor waveform;
  sim::SimWaveform physical_waveform;
  int64_t chirps_per_tx{0};
  int64_t samples_per_chirp{0};
};

struct TransmitReceipt {
  FrameToken token{0};
  holoscan::TimePoint transmit_time{};
  std::size_t profile_index{0U};
};

struct ReceivedAdcFrame {
  FrameToken token{0};
  std::size_t profile_index{0U};
  std::shared_ptr<holoscan::Tensor> adc_cube;
};

class TransmitFrameCoordinator {
 public:
  enum class ClaimStatus { kMissing, kNotReady, kReady };

  struct ClaimResult {
    ClaimStatus status{ClaimStatus::kMissing};
    holoscan::TimePoint transmit_time{};
    std::size_t profile_index{0U};
  };

  explicit TransmitFrameCoordinator(std::size_t capacity)
      : capacity_(capacity), tombstone_capacity_(capacity * 2U) {
    if (capacity_ == 0U) {
      throw std::invalid_argument("TransmitFrameCoordinator capacity must be > 0");
    }
  }

  void register_transmit(const TransmitReceipt& receipt) {
    std::lock_guard lock(mutex_);
    if (records_.count(receipt.token) != 0U ||
        tombstones_.count(receipt.token) != 0U) {
      throw std::runtime_error("duplicate transmit frame token");
    }
    if (records_.size() >= capacity_) {
      throw std::runtime_error("transmit frame coordinator capacity exceeded");
    }
    records_.emplace(receipt.token, Record{receipt.transmit_time, receipt.profile_index, false});
  }

  void mark_ready(FrameToken token) {
    std::lock_guard lock(mutex_);
    auto found = records_.find(token);
    if (found == records_.end()) {
      throw std::runtime_error("cannot ready unknown transmit frame token");
    }
    if (found->second.ready) {
      throw std::runtime_error("transmit frame token already ready");
    }
    found->second.ready = true;
  }

  void cancel(FrameToken token) {
    std::lock_guard lock(mutex_);
    if (records_.erase(token) != 0U) {
      add_tombstone(token);
    }
  }

  ClaimResult claim(FrameToken token) {
    std::lock_guard lock(mutex_);
    auto found = records_.find(token);
    if (found == records_.end()) {
      if (tombstones_.count(token) != 0U) {
        throw std::runtime_error("duplicate claim for retired frame token");
      }
      return {};
    }
    if (!found->second.ready) {
      return ClaimResult{ClaimStatus::kNotReady, found->second.transmit_time,
                         found->second.profile_index};
    }
    ClaimResult result{ClaimStatus::kReady, found->second.transmit_time,
                       found->second.profile_index};
    records_.erase(found);
    add_tombstone(token);
    return result;
  }

  void register_receive(FrameToken token) {
    std::lock_guard lock(mutex_);
    if (pending_receives_.count(token) != 0U ||
        tombstones_.count(token) != 0U) {
      throw std::runtime_error("duplicate receive frame token");
    }
    if (pending_receives_.size() >= capacity_) {
      throw std::runtime_error("receive frame coordinator capacity exceeded");
    }
    pending_receives_.insert(token);
  }

  void complete_receive(FrameToken token) {
    std::lock_guard lock(mutex_);
    if (pending_receives_.erase(token) == 0U) {
      throw std::runtime_error("cannot complete unknown receive frame token");
    }
  }

  void advance_poll(std::size_t max_polls) {
    std::lock_guard lock(mutex_);
    for (auto& [token, record] : records_) {
      (void)token;
      if (++record.poll_age > max_polls) {
        throw std::runtime_error("unmatched TX frame expired");
      }
    }
  }

  void finalize() const {
    std::lock_guard lock(mutex_);
    if (!records_.empty()) {
      throw std::runtime_error("frame coordinator finalized with unmatched TX frames");
    }
    if (!pending_receives_.empty()) {
      throw std::runtime_error("frame coordinator finalized with pending RX frames");
    }
  }

  [[nodiscard]] std::size_t outstanding() const {
    std::lock_guard lock(mutex_);
    return records_.size();
  }

 private:
  struct Record {
    holoscan::TimePoint transmit_time;
    std::size_t profile_index;
    bool ready;
    std::size_t poll_age{0U};
  };

  void add_tombstone(FrameToken token) {
    tombstones_.insert(token);
    tombstone_order_.push_back(token);
    if (tombstone_order_.size() > tombstone_capacity_) {
      tombstones_.erase(tombstone_order_.front());
      tombstone_order_.pop_front();
    }
  }

  std::size_t capacity_;
  std::size_t tombstone_capacity_;
  mutable std::mutex mutex_;
  std::unordered_map<FrameToken, Record> records_;
  std::unordered_set<FrameToken> pending_receives_;
  std::unordered_set<FrameToken> tombstones_;
  std::deque<FrameToken> tombstone_order_;
};

class FMCWTransmitDriver {
 public:
  virtual ~FMCWTransmitDriver() = default;
  virtual TransmitReceipt transmit(std::size_t profile_index,
                                   holoscan::TimePoint requested_time) = 0;
};

class MimoTdmReceiveDriver {
 public:
  virtual ~MimoTdmReceiveDriver() = default;
  virtual std::optional<ReceivedAdcFrame> try_receive(
      holoscan::ExecutionContext& context) = 0;
  virtual void finalize() = 0;
};

struct SimulatedTdmLinkConfig {
  std::vector<TdmTransmitProfile> profiles;
  sim::SimScene scene;
  TXSchedule tx_schedule;
  double frame_period{0.1};
  int polls_before_ready{2};
};

class SimulatedTdmLink final : public FMCWTransmitDriver,
                               public MimoTdmReceiveDriver {
 public:
  explicit SimulatedTdmLink(SimulatedTdmLinkConfig config)
      : config_(std::move(config)) {
    const int polls_before_ready = config_.polls_before_ready;
    if (polls_before_ready < 0) {
      throw std::invalid_argument("polls_before_ready must be >= 0");
    }
    polls_before_ready_ = polls_before_ready;
    if (config_.profiles.empty()) {
      throw std::invalid_argument("simulated link requires profiles");
    }
  }

  TransmitReceipt transmit(std::size_t profile_index,
                           holoscan::TimePoint requested_time) override {
    std::lock_guard lock(mutex_);
    const FrameToken token = next_token_++;
    pending_.push_back(DeviceFrame{token, profile_index, polls_before_ready_});
    return TransmitReceipt{token, requested_time, profile_index};
  }

  std::optional<ReceivedAdcFrame> try_receive(
      holoscan::ExecutionContext& context) override {
    DeviceFrame frame;
    {
      std::lock_guard lock(mutex_);
      if (pending_.empty()) {
        return std::nullopt;
      }
      if (pending_.front().polls_remaining-- > 0) {
        return std::nullopt;
      }
      frame = pending_.front();
      pending_.pop_front();
    }
    if (frame.profile_index >= config_.profiles.size()) {
      throw std::runtime_error("simulated RX received invalid profile index");
    }
    const auto& profile = config_.profiles[frame.profile_index];
    config_.scene.waveform = profile.physical_waveform;
    config_.scene.num_chirps = profile.chirps_per_tx;
    config_.scene.samples_per_chirp = profile.samples_per_chirp;
    const int64_t num_tx = static_cast<int64_t>(config_.scene.tx_positions.size());
    const int64_t slow_time = num_tx * profile.chirps_per_tx;
    const std::vector<int64_t> pattern =
        config_.tx_schedule.expand_chirp_tx_index(slow_time);
    std::vector<std::complex<double>> weights(
        static_cast<std::size_t>(num_tx * slow_time), {0.0, 0.0});
    for (int64_t chirp = 0; chirp < slow_time; ++chirp) {
      weights[static_cast<std::size_t>(
          pattern[static_cast<std::size_t>(chirp)] * slow_time + chirp)] =
          {1.0, 0.0};
    }
    const cudaStream_t stream = context.cuda_stream().get();
    auto adc = sim::simulate_mimo_gpu(config_.scene, weights, slow_time, stream);
    sim::add_thermal_noise_gpu(*adc, config_.scene.snr_db,
                               config_.scene.seed + frames_received_, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("simulated RX stream synchronization failed");
    }
    sim::advance_scene(&config_.scene, config_.frame_period);
    ++frames_received_;
    return ReceivedAdcFrame{frame.token, frame.profile_index, std::move(adc)};
  }

  void finalize() override {
    std::lock_guard lock(mutex_);
    if (!pending_.empty()) {
      throw std::runtime_error("simulated link finalized with pending frames");
    }
  }

 private:
  struct DeviceFrame {
    FrameToken token;
    std::size_t profile_index;
    int polls_remaining;
  };

  int polls_before_ready_;
  SimulatedTdmLinkConfig config_;
  FrameToken next_token_{1};
  uint64_t frames_received_{0U};
  std::mutex mutex_;
  std::deque<DeviceFrame> pending_;
};

struct FMCWTransmitSourceConfig {
  std::vector<TdmTransmitProfile> profiles;
  std::vector<std::size_t> profile_sequence;
  double frame_period{0.1};
  int64_t num_frames{1};
  std::shared_ptr<FMCWTransmitDriver> driver;
  std::shared_ptr<TransmitFrameCoordinator> coordinator;
};

class FMCWTransmitSourceOp final : public holoscan::Operator<> {
 public:
  explicit FMCWTransmitSourceOp(FMCWTransmitSourceConfig config)
      : config_(std::move(config)) {
    if (config_.profiles.empty() || config_.profile_sequence.empty() ||
        !config_.driver || !config_.coordinator) {
      throw std::invalid_argument("FMCWTransmitSourceOp has incomplete configuration");
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kWaveform{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLComplex, 128U, 1U},
        .rank = 1U,
    };
    std::size_t max_reference_bytes = sizeof(std::complex<double>);
    for (const auto& profile : config_.profiles) {
      max_reference_bytes = std::max(
          max_reference_bytes,
          profile.waveform.reference_samples().size() * sizeof(std::complex<double>));
    }
    spec.output(waveform)
        .max_emits_per_compute(1U)
        .produces_tensor(pooled_tensor_output(kWaveform, 1U, max_reference_bytes));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(config_.frame_period))});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (frames_emitted_ >= config_.num_frames) {
      return {};
    }
    const std::size_t profile_index = config_.profile_sequence[
        static_cast<std::size_t>(frames_emitted_) % config_.profile_sequence.size()];
    TransmitReceipt receipt;
    try {
      receipt = config_.driver->transmit(profile_index, context.activation_time());
      if (receipt.profile_index >= config_.profiles.size()) {
        throw std::runtime_error("TX driver returned an invalid applied profile");
      }
      config_.coordinator->register_transmit(receipt);
    } catch (const std::exception& error) {
      return failure("transmit", error.what());
    }

    const auto& descriptor = config_.profiles[receipt.profile_index].waveform;
    auto wire = to_wire(descriptor);
    const auto& reference = descriptor.reference_samples();
    const std::array<int64_t, 1> shape{static_cast<int64_t>(reference.size())};
    auto loan = waveform.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = shape, .dtype = DLDataType{kDLComplex, 128U, 1U}});
    if (!loan) {
      config_.coordinator->cancel(receipt.token);
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto written = loan->write_host_and_commit(
        [&reference](const holoscan::TensorOutputWriteGuard& writer) {
          std::copy(reference.begin(), reference.end(),
                    writer.data_as<std::complex<double>>());
        });
    if (!written) {
      config_.coordinator->cancel(receipt.token);
      return holoscan::make_unexpected(std::move(written).error());
    }
    auto emitted = waveform.emit_tensor(
        std::move(*loan), std::move(wire),
        holoscan::EmitOptions{.capture_time = receipt.transmit_time});
    if (!emitted) {
      config_.coordinator->cancel(receipt.token);
      return emitted;
    }
    try {
      config_.coordinator->mark_ready(receipt.token);
    } catch (const std::exception& error) {
      return failure("publish", error.what());
    }
    const auto& applied = config_.profiles[receipt.profile_index];
    const auto effective_pri_ns = static_cast<int64_t>(std::llround(
        applied.waveform.chirp_repetition_interval() * 1.0e9));
    HOLOSCAN_LOG_INFO(
        "multi-profile axes: frame={} profile={} range_bins={} "
        "velocity_bins={} effective_pri_ns={}",
        frames_emitted_, applied.name, applied.samples_per_chirp,
        applied.chirps_per_tx, effective_pri_ns);
    ++frames_emitted_;
    return {};
  }

  holoscan::Output<wire::FMCWWaveformT> waveform{"waveform"};

 private:
  static holoscan::expected<void, holoscan::Error> failure(
      const char* action, const char* message) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string("FMCWTransmitSourceOp ") + action + " failed: " + message});
  }

  FMCWTransmitSourceConfig config_;
  int64_t frames_emitted_{0};
};

struct CorrelatedMimoTdmAdcSourceConfig {
  double poll_period{0.001};
  double frame_period{0.1};
  int64_t num_frames{1};
  std::size_t pending_capacity{4U};
  std::shared_ptr<MimoTdmReceiveDriver> driver;
  std::shared_ptr<TransmitFrameCoordinator> coordinator;
  std::shared_ptr<CadenceMeter> cadence;
};

class CorrelatedMimoTdmAdcSourceOp final : public holoscan::Operator<> {
 public:
  CorrelatedMimoTdmAdcSourceOp(const MaxTensorExtents& extents,
                               CorrelatedMimoTdmAdcSourceConfig config)
      : config_(std::move(config)) {
    const int64_t max_rx = require_extent(extents.max_rx, "max_rx", name());
    const int64_t max_samples = require_extent(
        extents.max_samples_per_chirp, "max_samples_per_chirp", name());
    const int64_t max_interleaved = require_extent(
        extents.max_interleaved_chirps, "max_interleaved_chirps", name());
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_rx, max_interleaved, max_samples},
        complex_dtype<float>());
    if (!config_.driver || !config_.coordinator ||
        config_.pending_capacity == 0U || config_.poll_period <= 0.0 ||
        config_.frame_period <= 0.0) {
      throw std::invalid_argument(std::string(name()) + " has incomplete configuration");
    }
    max_pending_polls_ = std::max<std::size_t>(
        1U, static_cast<std::size_t>(
                std::ceil(2.0 * config_.frame_period / config_.poll_period)));
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(adc_cube).produces_tensor(
        pooled_tensor_output(kDevice, 3U, max_byte_span_));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(config_.poll_period))});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (frames_emitted_ >= config_.num_frames) {
      return {};
    }
    try {
      config_.coordinator->advance_poll(max_pending_polls_);
    } catch (const std::exception& error) {
      return failure(error.what());
    }
    std::optional<ReceivedAdcFrame> received;
    try {
      received = config_.driver->try_receive(context);
    } catch (const std::exception& error) {
      return failure(error.what());
    }
    if (received.has_value()) {
      auto& frame = *received;
      if (!frame.adc_cube) {
        return failure("RX driver returned no ADC tensor");
      }
      if (pending_.size() >= config_.pending_capacity) {
        return failure("pending RX frame capacity exceeded");
      }
      try {
        config_.coordinator->register_receive(frame.token);
      } catch (const std::exception& error) {
        return failure(error.what());
      }
      pending_.push_back(PendingFrame{std::move(frame), 0U});
    }
    for (auto it = pending_.begin(); it != pending_.end(); ++it) {
      const auto claim = config_.coordinator->claim(it->frame.token);
      if (claim.status == TransmitFrameCoordinator::ClaimStatus::kMissing ||
          claim.status == TransmitFrameCoordinator::ClaimStatus::kNotReady) {
        if (++it->polls_waited > max_pending_polls_) {
          return failure("RX frame token did not acquire a published waveform");
        }
        continue;
      }
      if (claim.profile_index != it->frame.profile_index) {
        return failure("RX/TX applied-profile correlation mismatch");
      }
      const ReceivedAdcFrame frame = it->frame;
      pending_.erase(it);
      auto emitted = copy_and_emit(context, frame, claim);
      if (!emitted) {
        return emitted;
      }
      try {
        config_.coordinator->complete_receive(frame.token);
      } catch (const std::exception& error) {
        return failure(error.what());
      }
      return {};
    }
    return {};
  }

  holoscan::Output<holoscan::Tensor> adc_cube{"adc_cube"};

  void stop() override {
    if (!pending_.empty()) {
      throw std::runtime_error(
          "CorrelatedMimoTdmAdcSourceOp finalized with pending RX frames");
    }
    config_.driver->finalize();
    config_.coordinator->finalize();
  }

 private:
  static constexpr const char* name() { return "CorrelatedMimoTdmAdcSourceOp"; }

  static holoscan::expected<void, holoscan::Error> failure(const char* message) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, std::string(name()) + ": " + message});
  }

  holoscan::expected<void, holoscan::Error> copy_and_emit(
      holoscan::ExecutionContext& context, const ReceivedAdcFrame& frame,
      const TransmitFrameCoordinator::ClaimResult& claim) {
    auto loan = adc_cube.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = frame.adc_cube->shape(), .dtype = frame.adc_cube->dtype()});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (cudaMemcpyAsync(writer->data(), frame.adc_cube->data(), writer->byte_size(),
                        cudaMemcpyDeviceToDevice,
                        context.cuda_stream().get()) != cudaSuccess) {
      return failure("output D2D failed");
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    HOLOSCAN_RETURN_IF_ERROR(adc_cube.emit(
        std::move(*loan),
        holoscan::EmitOptions{.capture_time = claim.transmit_time}));
    if (config_.cadence) {
      config_.cadence->on_source_emit();
    }
    ++frames_emitted_;
    return {};
  }

  CorrelatedMimoTdmAdcSourceConfig config_;
  struct PendingFrame {
    ReceivedAdcFrame frame;
    std::size_t polls_waited;
  };
  std::size_t max_byte_span_{0U};
  std::size_t max_pending_polls_{1U};
  int64_t frames_emitted_{0};
  std::deque<PendingFrame> pending_;
};

}  // namespace holoscan::radar::app
