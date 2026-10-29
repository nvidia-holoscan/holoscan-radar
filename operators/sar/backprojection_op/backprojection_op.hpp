// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sar/formation_hud.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"

namespace holoscan::radar {
namespace detail {
template <typename T, typename TGeom>
struct BackprojectionState;
}

struct BackprojectionParams {
  CoordinateFrame coordinate_frame{};
  SampleGeometry sample_geometry{};
  std::optional<double> range_bin_spacing{};
  std::optional<double> range_bin_origin{};
  double reference_frequency{0.0};
  RangeInterpolation interpolation{RangeInterpolation::kLinear};
  BackprojectionBackend backend{kDefaultBackprojectionBackend};
  GeometryChange geometry_change{GeometryChange::kReset};
};
HOLOSCAN_PARAM_SCHEMA(
    BackprojectionParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.BackprojectionParams", 3),
    holoscan::field<&BackprojectionParams::coordinate_frame>("coordinate_frame"),
    holoscan::field<&BackprojectionParams::sample_geometry>("sample_geometry"),
    holoscan::field<&BackprojectionParams::range_bin_spacing>("range_bin_spacing"),
    holoscan::field<&BackprojectionParams::range_bin_origin>("range_bin_origin"),
    holoscan::field<&BackprojectionParams::reference_frequency>("reference_frequency"),
    holoscan::field<&BackprojectionParams::interpolation>("interpolation"),
    holoscan::field<&BackprojectionParams::backend>("backend"),
    holoscan::field<&BackprojectionParams::geometry_change>("geometry_change"));

// `T` is the sample and image precision; `TGeom` is the precision of the
// geometry ports (`positions`, `reference_position`, `sample_points`). They
// default to the same type, which is the only combination the contract
// describes. `<float, double>` exists because ECEF positions do not survive
// float32: one ulp at a 6.4e6 m radius is 0.76 m, some 25 wavelengths at X
// band, which destroys the phase history well before the samples' own
// precision matters. See the operator README.
template <typename T, typename TGeom = T>
class BackprojectionOp : public holoscan::Operator<BackprojectionParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  static_assert(std::is_same_v<TGeom, float> || std::is_same_v<TGeom, double>);
  static_assert(sizeof(TGeom) >= sizeof(T), "geometry may not be coarser than samples");

 public:
  explicit BackprojectionOp(const MaxTensorExtents& extents,
                            std::shared_ptr<FormationHud> hud = {});
  ~BackprojectionOp() override;
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  void stop() override;

  struct LastImageTiming {
    double compute_s{0.0};
    int64_t pulses{0}, pixels{0}, num_v{0}, num_u{0};
    std::optional<std::chrono::steady_clock::time_point> emit_at;
  };
  LastImageTiming last_image_timing() const {
    LastImageTiming t;
    t.compute_s = last_image_compute_s_.load(std::memory_order_acquire);
    t.pulses = last_image_pulses_.load(std::memory_order_relaxed);
    t.pixels = last_image_pixels_.load(std::memory_order_relaxed);
    t.num_v = last_image_num_v_.load(std::memory_order_relaxed);
    t.num_u = last_image_num_u_.load(std::memory_order_relaxed);
    if (last_image_emit_valid_.load(std::memory_order_acquire)) {
      t.emit_at = std::chrono::steady_clock::time_point{
          std::chrono::nanoseconds{last_image_emit_ns_.load(std::memory_order_relaxed)}};
    }
    return t;
  }

  Input<Tensor> range_profiles{"range_profiles"};
  Input<Tensor> positions{"positions"};
  Input<Tensor> reference_position{"reference_position"};
  Input<Tensor> control{"control"};
  Input<Tensor> range_bins{"range_bins"};
  Input<Tensor> sample_points{"sample_points"};
  Input<Tensor> weight{"weight"};
  Input<Tensor> initial_image{"initial_image"};
  Input<WireSampleGeometry> sample_geometry{"sample_geometry"};
  Input<double> range_bin_spacing{"range_bin_spacing"};
  Input<double> range_bin_origin{"range_bin_origin"};
  Input<double> reference_frequency{"reference_frequency"};
  Input<RangeInterpolation> interpolation{"interpolation"};
  Output<Tensor> image{"image"};
  Output<WireSampleGeometry> sample_geometry_out{"sample_geometry_out"};

 private:
  MaxTensorExtents extents_;
  CoordinateFrame effective_frame_{};
  SampleGeometry effective_geometry_{};
  std::optional<double> effective_spacing_{}, effective_origin_{};
  double effective_frequency_{0.0};
  RangeInterpolation effective_interpolation_{RangeInterpolation::kLinear};
  BackprojectionBackend backend_value_{kDefaultBackprojectionBackend};
  GeometryChange geometry_change_value_{GeometryChange::kReset};
  bool sample_points_connected_{false}, range_bins_connected_{false}, weight_connected_{false};
  int64_t unemitted_pulses_{0};
  bool locality_warned_{false};
  double aperture_compute_s_{0.0};
  std::atomic<double> last_image_compute_s_{0.0};
  std::atomic<int64_t> last_image_pulses_{0}, last_image_pixels_{0};
  std::atomic<int64_t> last_image_num_v_{0}, last_image_num_u_{0};
  std::atomic<int64_t> last_image_emit_ns_{0};
  std::atomic<bool> last_image_emit_valid_{false};
  std::shared_ptr<FormationHud> hud_{};
  std::shared_ptr<detail::BackprojectionState<T, TGeom>> state_;
  std::shared_ptr<Tensor> retained_profiles_, retained_positions_, retained_reference_;
  std::shared_ptr<Tensor> retained_sample_points_, retained_weight_;
  LatchedOutput geometry_output_;
};

extern template class BackprojectionOp<float, float>;
extern template class BackprojectionOp<double, double>;
extern template class BackprojectionOp<float, double>;
}  // namespace holoscan::radar
