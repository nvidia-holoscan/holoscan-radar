// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned operators for mode 1: the control operator (a spotlight
// retains nothing, so ApertureManagerOp is off this path —
// OPERATOR_API.md → Running without a manager), the Holoviz
// rasterizer / hold pair, and the NVRadar-parity diagnostics.
//
// StreamBoundaryControlOp watches the reader's D22 announcement and
// issues one emit-and-reset on that same tick so the closing batch
// lands in the image. It is not a source: metadata arrives only
// through receive(), so it must be fed from the reader (fan-out of
// phase_history). Absent control → BackprojectionOp accumulates and
// does not emit.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/sdk/stream_boundary.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace radar = holoscan::radar;

class StreamBoundaryControlOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StreamBoundaryControlOp)
  StreamBoundaryControlOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("announce");
    // kNone: silent on non-boundary ticks (absent control = accumulate).
    spec.output<holoscan::TensorMap>("control")
        .condition(holoscan::ConditionType::kNone);
  }

  void start() override {
    emit_and_reset_ =
        radar::make_device_tensor({1}, DLDataType{kDLUInt, 8, 1});
    const uint8_t value = static_cast<uint8_t>(
        radar::FormationControl::kEmitAndReset);
    if (cudaMemcpy(emit_and_reset_->data(), &value, sizeof(value),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "StreamBoundaryControlOp: control H2D copy failed");
    }
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    // Pull the message so the reader's metadata lands on this operator.
    radar::receive_single(op_input, "announce",
                          "StreamBoundaryControlOp");
    // Interim string encoding (OPEN_ITEMS item 11), not the contract's
    // uint8 StreamBoundary enum. Both collection seams and the final
    // tick close an aperture; mode 1 is one file so only
    // end_of_stream appears.
    const std::string boundary = metadata()->get<std::string>(
        radar::kStreamBoundaryKey, std::string());
    if (boundary == radar::kStreamBoundaryEndOfStream ||
        boundary == radar::kStreamBoundaryCollection) {
      radar::emit_single(op_output, "control", emit_and_reset_);
    }
  }

 private:
  std::shared_ptr<holoscan::Tensor> emit_and_reset_;
};

// Diagnostic: rotate reader ECEF positions / SRP into scene ENU at a
// fixed IARP, matching NVRadar's spaceborne Atlanta converter
// (Bowring WGS-84 + ECEF→ENU). The kernel is convention-free; this
// exists so a SampleGeometry written in that ENU can be paired with
// APC / MCP in the same numbers. Not a contract operator.
class EcefToSceneEnuOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(EcefToSceneEnuOp)
  EcefToSceneEnuOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("positions");
    spec.input<holoscan::TensorMap>("reference_position");
    spec.output<holoscan::TensorMap>("positions");
    spec.output<holoscan::TensorMap>("reference_position");
    spec.param(origin_ecef_, "origin_ecef", "ENU origin (ECEF m)",
               "IARP in ECEF metres. Subtracted, then rotated.");
    spec.param(quantize_f32_, "quantize_phase_center_f32",
               "Quantize APC to float32",
               "Round positions through float32 after the rotate, "
               "matching NVRadar's .dat xyz. Reference is not "
               "quantized. Tensor dtype stays float64.",
               false);
  }

  void start() override {
    // Both inputs are a fan-out of the reader, so they carry the
    // same D22 key. Raise would abort on the second receive;
    // BackprojectionOp uses kUpdate for the same reason.
    metadata_policy(holoscan::MetadataPolicy::kUpdate);
    const std::vector<double>& origin = origin_ecef_.get();
    if (origin.size() != 3) {
      throw std::invalid_argument(
          "EcefToSceneEnuOp: origin_ecef must have 3 elements");
    }
    for (double component : origin) {
      if (!std::isfinite(component)) {
        throw std::invalid_argument(
            "EcefToSceneEnuOp: origin_ecef must be finite");
      }
    }
    origin_ = {origin[0], origin[1], origin[2]};
    rotation_ = ecef_to_enu_rotation(origin_);
    do_quantize_ = quantize_f32_.get();
    HOLOSCAN_LOG_INFO(
        "EcefToSceneEnuOp: IARP ECEF [{:.6f}, {:.6f}, {:.6f}]{}",
        origin_[0], origin_[1], origin_[2],
        do_quantize_ ? "  APC through float32" : "");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    auto positions = radar::receive_single(op_input, "positions",
                                           "EcefToSceneEnuOp");
    auto reference = radar::receive_single(
        op_input, "reference_position", "EcefToSceneEnuOp");
    radar::require_contiguous(*positions, "positions",
                              "EcefToSceneEnuOp");
    radar::require_contiguous(*reference, "reference_position",
                              "EcefToSceneEnuOp");
    radar::require_cuda_accessible(*positions, "positions",
                                   "EcefToSceneEnuOp");
    radar::require_cuda_accessible(*reference, "reference_position",
                                   "EcefToSceneEnuOp");
    cudaStream_t stream = op_input.receive_cuda_stream("positions");
    auto pos_out = transform(*positions, do_quantize_, stream);
    auto ref_out = transform(*reference, false, stream);
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "positions");
      op_output.set_cuda_stream(stream, "reference_position");
    }
    radar::emit_single(op_output, "positions", pos_out);
    radar::emit_single(op_output, "reference_position", ref_out);
  }

 private:
  // NVRadar cphd_data_convert.py — Bowring, then the standard
  // ECEF→ENU matrix at that geodetic point.
  static constexpr double kWgs84A = 6378137.0;
  static constexpr double kWgs84F = 1.0 / 298.257223563;

  static std::array<double, 9> ecef_to_enu_rotation(
      const std::array<double, 3>& ecef) {
    const double b = kWgs84A * (1.0 - kWgs84F);
    const double e2 = 1.0 - (b * b) / (kWgs84A * kWgs84A);
    const double x = ecef[0];
    const double y = ecef[1];
    const double z = ecef[2];
    const double lon = std::atan2(y, x);
    const double p = std::hypot(x, y);
    const double theta = std::atan2(z * kWgs84A, p * b);
    const double slt = std::sin(theta);
    const double clt = std::cos(theta);
    const double lat = std::atan2(
        z + e2 * (kWgs84A * kWgs84A / b) * slt * slt * slt,
        p - e2 * kWgs84A * clt * clt * clt);
    const double sl = std::sin(lat);
    const double cl = std::cos(lat);
    const double so = std::sin(lon);
    const double co = std::cos(lon);
    return {-so,      co,     0.0,
            -sl * co, -sl * so, cl,
            cl * co,  cl * so,  sl};
  }

  static void apply(const double* in, double* out, int64_t count,
                    const std::array<double, 3>& origin,
                    const std::array<double, 9>& rotation) {
    for (int64_t i = 0; i < count; ++i) {
      const double dx = in[i * 3 + 0] - origin[0];
      const double dy = in[i * 3 + 1] - origin[1];
      const double dz = in[i * 3 + 2] - origin[2];
      out[i * 3 + 0] = rotation[0] * dx + rotation[1] * dy +
                       rotation[2] * dz;
      out[i * 3 + 1] = rotation[3] * dx + rotation[4] * dy +
                       rotation[5] * dz;
      out[i * 3 + 2] = rotation[6] * dx + rotation[7] * dy +
                       rotation[8] * dz;
    }
  }

  std::shared_ptr<holoscan::Tensor> transform(
      const holoscan::Tensor& in, bool quantize_f32,
      cudaStream_t stream) {
    const std::vector<int64_t> in_shape = in.shape();
    if (in_shape.size() < 1 ||
        in_shape[in_shape.size() - 1] != 3) {
      throw std::invalid_argument(
          "EcefToSceneEnuOp: last axis must be 3");
    }
    if (in.dtype().code != kDLFloat || in.dtype().bits != 64) {
      throw std::invalid_argument(
          "EcefToSceneEnuOp: expected float64 positions");
    }
    int64_t count = 1;
    for (size_t d = 0; d + 1 < in_shape.size(); ++d) {
      count *= in_shape[d];
    }
    std::vector<int64_t> shape(in_shape.begin(), in_shape.end());
    std::vector<double> host(static_cast<size_t>(count) * 3);
    const size_t bytes = host.size() * sizeof(double);
    if (cudaMemcpyAsync(host.data(), in.data(), bytes,
                        cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
      throw std::runtime_error(
          "EcefToSceneEnuOp: D2H copy failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error(
          "EcefToSceneEnuOp: D2H sync failed");
    }
    apply(host.data(), host.data(), count, origin_, rotation_);
    if (quantize_f32) {
      for (double& component : host) {
        component = static_cast<double>(static_cast<float>(component));
      }
    }
    if (!logged_) {
      HOLOSCAN_LOG_INFO(
          "EcefToSceneEnuOp: pulse0 ENU [{:.6f}, {:.6f}, {:.6f}]",
          host[0], host[1], host[2]);
      logged_ = true;
    }
    auto out = radar::make_device_tensor(
        shape, DLDataType{kDLFloat, 64, 1});
    if (cudaMemcpyAsync(out->data(), host.data(), bytes,
                        cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
      throw std::runtime_error(
          "EcefToSceneEnuOp: H2D copy failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error(
          "EcefToSceneEnuOp: H2D sync failed");
    }
    return out;
  }

  holoscan::Parameter<std::vector<double>> origin_ecef_;
  holoscan::Parameter<bool> quantize_f32_;
  std::array<double, 3> origin_{};
  std::array<double, 9> rotation_{};
  bool do_quantize_{false};
  bool logged_{false};
};

// Diagnostic: replace contract RC with NVRadar's kernel vph. The
// file is their .dat samples after DataConditionerOp
// (fftshift(ifft(·, ortho))), stored as complex64, pulse-major.
// Promoted to complex128 for BackprojectionOp<double>; the kernel
// downcasts back to float32 so those bits match. Not a contract op.
class NvradarVphOverrideOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NvradarVphOverrideOp)
  NvradarVphOverrideOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("range_profiles");
    spec.output<holoscan::TensorMap>("range_profiles");
    spec.param(path_, "path", "NVRadar vph file",
               "Headered complex64 dump: uint64 Np, uint64 N, then "
               "Np*N interleaved complex64 samples.",
               std::string());
  }

  void start() override {
    const std::string path = path_.get();
    if (path.empty()) {
      throw std::invalid_argument(
          "NvradarVphOverrideOp: path is empty");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error(
          "NvradarVphOverrideOp: cannot open " + path);
    }
    uint64_t np = 0;
    uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&np), sizeof(np));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in || np == 0 || n == 0) {
      throw std::runtime_error(
          "NvradarVphOverrideOp: bad header in " + path);
    }
    host_.resize(static_cast<size_t>(np * n));
    const auto bytes = static_cast<std::streamsize>(
        host_.size() * sizeof(std::complex<float>));
    in.read(reinterpret_cast<char*>(host_.data()), bytes);
    if (!in || in.gcount() != bytes) {
      throw std::runtime_error(
          "NvradarVphOverrideOp: short read from " + path);
    }
    num_pulses_ = static_cast<int64_t>(np);
    num_bins_ = static_cast<int64_t>(n);
    cursor_ = 0;
    HOLOSCAN_LOG_INFO(
        "NvradarVphOverrideOp: {}  {} pulses x {} bins  "
        "pulse0[0]=({:.6g}, {:.6g})",
        path, num_pulses_, num_bins_, host_[0].real(),
        host_[0].imag());
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    auto incoming = radar::receive_single(
        op_input, "range_profiles", "NvradarVphOverrideOp");
    const std::vector<int64_t> shape = incoming->shape();
    if (shape.size() < 2 || shape.back() != num_bins_) {
      throw std::invalid_argument(
          "NvradarVphOverrideOp: last axis must be " +
          std::to_string(num_bins_));
    }
    int64_t count = 1;
    for (size_t d = 0; d + 1 < shape.size(); ++d) {
      count *= shape[d];
    }
    if (cursor_ + count > num_pulses_) {
      throw std::runtime_error(
          "NvradarVphOverrideOp: past end of file (" +
          std::to_string(cursor_) + " + " + std::to_string(count) +
          " > " + std::to_string(num_pulses_) + ")");
    }
    const size_t samples = static_cast<size_t>(count * num_bins_);
    std::vector<std::complex<double>> wide(samples);
    const std::complex<float>* src =
        host_.data() + static_cast<size_t>(cursor_ * num_bins_);
    for (size_t i = 0; i < samples; ++i) {
      wide[i] = std::complex<double>(
          static_cast<double>(src[i].real()),
          static_cast<double>(src[i].imag()));
    }
    auto out = radar::make_device_tensor(
        shape, radar::complex_dtype<double>());
    const size_t bytes = samples * sizeof(std::complex<double>);
    if (cudaMemcpy(out->data(), wide.data(), bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "NvradarVphOverrideOp: H2D copy failed");
    }
    cursor_ += count;
    radar::emit_single(op_output, "range_profiles", out);
  }

 private:
  holoscan::Parameter<std::string> path_;
  std::vector<std::complex<float>> host_;
  int64_t num_pulses_{0};
  int64_t num_bins_{0};
  int64_t cursor_{0};
};

// Magnitude-and-log rasterizer for Holoviz. Matches
// tools/sar_image/sar_image/render.py: dB relative to the 99.8th
// percentile of full-resolution |z| (valid samples only), clipped
// 30 dB down, mapped onto the greyscale LUT. Display-only; the
// TensorDump .npy stays full-resolution complex. GPU |z|, then a
// host magnitude copy (256 MiB at Atlanta 8192² float) so the
// percentile matches the PNG; box-average is after that, for the
// displayed grid only. compute() is in sar_image_lut.cu.
class SarImageToLutOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SarImageToLutOp)
  SarImageToLutOp() = default;

  // Same window as the offline PNG (render.py floor_db / percentile).
  double floor_db{-30.0};
  double high_percentile{99.8};
  // Longest displayed side; the other axis scales. 2048 keeps the
  // Atlanta lattice readable in a 1024² window.
  int64_t max_side{2048};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("image");
    spec.output<holoscan::TensorMap>("sar_image");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
};

// Mode 1 emits one image, then the reader stops the graph. Re-emit
// the raster for `hold_s` so the Holoviz window stays up. Periodic
// condition is attached in compose(); this operator disables itself
// when the hold expires. `hold_s` is also the max wait for the
// first frame (Atlanta forms in ~9 s; default 30 s still covers
// that). The deadline arms in start() so a silent formation path
// cannot hang the window; the first image re-arms so the hold is
// still counted from emit.
class HoldLastImageOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HoldLastImageOp)
  HoldLastImageOp() = default;

  double hold_s{30.0};

  void initialize() override {
    stop_condition_ =
        fragment()->make_condition<holoscan::BooleanCondition>(
            "hold_stop");
    add_arg(stop_condition_);
    holoscan::Operator::initialize();
  }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("image")
        .condition(holoscan::ConditionType::kNone);
    spec.output<holoscan::TensorMap>("image");
  }

  void start() override { arm_deadline(); }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    auto maybe = op_input.receive<holoscan::TensorMap>("image");
    if (maybe && !maybe->empty()) {
      cached_ = maybe.value();
      if (waiting_for_first_) {
        arm_deadline();
        waiting_for_first_ = false;
      }
    }
    if (have_deadline_ &&
        std::chrono::steady_clock::now() >= deadline_) {
      stop_condition_->disable_tick();
    }
    if (cached_.empty()) {
      return;
    }
    op_output.emit(cached_, "image");
  }

 private:
  void arm_deadline() {
    const auto ms = static_cast<int64_t>(
        std::max(0.0, hold_s) * 1000.0);
    deadline_ = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(ms);
    have_deadline_ = true;
  }

  holoscan::TensorMap cached_;
  std::chrono::steady_clock::time_point deadline_{};
  bool have_deadline_{false};
  bool waiting_for_first_{true};
  std::shared_ptr<holoscan::BooleanCondition> stop_condition_;
};
