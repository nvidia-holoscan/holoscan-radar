// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"

namespace holoscan::radar {

namespace detail {
template <typename T>
struct BackprojectionState;
}  // namespace detail

// Time-domain SAR image formation. Accumulating across ticks — emits only
// when a control message says so. This header is the contract reference
// for ports, shapes, dtypes and parameters; the locked contract is
// docs/planning/SAR/OPERATOR_API.md.
//
// In:  range_profiles      complex, (..., num_pulses, num_range_bins)
// In:  positions           real,    (..., num_pulses, 3)
// In:  reference_position  real,    (..., num_pulses, 3)
// In:  control             uint8,   (1,)  required connection, optional
//                                      delivery (non-blocking)
// In:  range_bins          optional real (num_range_bins,)
// In:  sample_points       optional real (num_v, num_u, 3)
// In:  weight              optional real (..., num_pulses)
// In:  initial_image       optional; rejected this slice
// Out: image               complex, (..., num_v, num_u) on emit only
// Out: sample_geometry     optional SampleGeometry, grid mode on emit
//
// Precision is fixed at compile time (T = float | double).
template <typename T>
class BackprojectionOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "BackprojectionOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BackprojectionOp)

  BackprojectionOp() = default;
  ~BackprojectionOp() override;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

  // Host wall time of compute() summed over the aperture that
  // produced the last emit, after a stream sync on that emit.
  // Not a port. The formation app reads this for the overlay.
  struct LastImageTiming {
    double compute_s{0.0};
    int64_t pulses{0};
    int64_t pixels{0};
    int64_t num_v{0};
    int64_t num_u{0};
    // Wall time when the formed image was ready (after the
    // emit-path stream sync, before the image port is emitted).
    std::optional<std::chrono::steady_clock::time_point> emit_at;
  };
  LastImageTiming last_image_timing() const {
    LastImageTiming t;
    t.compute_s =
        last_image_compute_s_.load(std::memory_order_acquire);
    t.pulses = last_image_pulses_.load(std::memory_order_relaxed);
    t.pixels = last_image_pixels_.load(std::memory_order_relaxed);
    t.num_v = last_image_num_v_.load(std::memory_order_relaxed);
    t.num_u = last_image_num_u_.load(std::memory_order_relaxed);
    if (last_image_emit_valid_.load(std::memory_order_acquire)) {
      const auto ns =
          last_image_emit_ns_.load(std::memory_order_relaxed);
      t.emit_at = std::chrono::steady_clock::time_point{
          std::chrono::nanoseconds{ns}};
    }
    return t;
  }

 private:
  Parameter<CoordinateFrame> coordinate_frame_;
  Parameter<SampleGeometry> sample_geometry_;
  Parameter<double> range_bin_spacing_;
  Parameter<double> range_bin_origin_;
  Parameter<double> reference_frequency_;
  Parameter<RangeInterpolation> interpolation_;
  Parameter<BackprojectionBackend> backend_;
  Parameter<GeometryChange> geometry_change_;

  CoordinateFrame effective_frame_{};
  SampleGeometry effective_geometry_{};
  std::optional<double> effective_spacing_{};
  std::optional<double> effective_origin_{};
  double effective_frequency_{0.0};
  RangeInterpolation effective_interpolation_{RangeInterpolation::kLinear};
  BackprojectionBackend backend_value_{kDefaultBackprojectionBackend};
  GeometryChange geometry_change_value_{GeometryChange::kReset};

  bool sample_points_connected_{false};
  bool range_bins_connected_{false};
  bool control_connected_{false};

  int64_t unemitted_pulses_{0};
  bool locality_warned_{false};
  double aperture_compute_s_{0.0};
  std::atomic<double> last_image_compute_s_{0.0};
  std::atomic<int64_t> last_image_pulses_{0};
  std::atomic<int64_t> last_image_pixels_{0};
  std::atomic<int64_t> last_image_num_v_{0};
  std::atomic<int64_t> last_image_num_u_{0};
  std::atomic<int64_t> last_image_emit_ns_{0};
  std::atomic<bool> last_image_emit_valid_{false};

  std::shared_ptr<detail::BackprojectionState<T>> state_;
  std::shared_ptr<holoscan::Tensor> retained_profiles_;
  std::shared_ptr<holoscan::Tensor> retained_positions_;
  std::shared_ptr<holoscan::Tensor> retained_reference_;
  std::shared_ptr<holoscan::Tensor> retained_sample_points_;
  std::shared_ptr<holoscan::Tensor> retained_weight_;
};

extern template class BackprojectionOp<float>;
extern template class BackprojectionOp<double>;

}  // namespace holoscan::radar
