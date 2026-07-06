// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SISO FMCW pipeline example: a synthetic ADC source feeding the split
// signal-processing chain
//
//   AdcSourceOp -> FMCWRangeProcessorOp -> ClutterFilterOp
//               -> FMCWDopplerProcessorOp -> PeakReportOp
//
// The source synthesizes one static clutter return and one moving target
// with the analytic dechirped point-target model; the sink locates the
// range-Doppler peak and reports it in physical units via the operators'
// range_bins / velocity_bins axis outputs.
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/radar_pipeline/cpp/radar_pipeline

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace radar = holoscan::radar;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int64_t kNumChirps = 128;
constexpr int64_t kSamplesPerChirp = 256;

radar::FMCWWaveformDescriptor demo_waveform() {
  radar::FMCWWaveformDescriptor waveform;
  waveform.carrier_frequency = 77e9;
  waveform.bandwidth = 150e6;
  waveform.chirp_duration = 20e-6;
  waveform.sample_rate = 12.8e6;
  waveform.chirp_repetition_interval = 25e-6;
  return waveform;
}

// Analytic dechirped point-target contribution (stop-and-hop), matching
// the operators' sign conventions: beat = slope * tau (positive), carrier
// phase -2*pi*f_c*tau so radial velocity v_r > 0 (receding) satisfies
// v = -f_d * lambda / 2.
void add_target(std::vector<std::complex<float>>* cube, double range_m,
                double radial_velocity, double amplitude) {
  const radar::FMCWWaveformDescriptor waveform = demo_waveform();
  const double c = radar::kDefaultPropagationSpeed;
  const double slope = waveform.slope();
  const double pri = waveform.chirp_repetition_interval;
  for (int64_t k = 0; k < kNumChirps; ++k) {
    const double range_k =
        range_m + radial_velocity * static_cast<double>(k) * pri;
    const double tau = 2.0 * range_k / c;
    const double beat = slope * tau;
    for (int64_t s = 0; s < kSamplesPerChirp; ++s) {
      const double t = static_cast<double>(s) / waveform.sample_rate;
      const double phase =
          2.0 * kPi * beat * t - 2.0 * kPi * waveform.carrier_frequency * tau;
      (*cube)[static_cast<size_t>(k * kSamplesPerChirp + s)] +=
          std::complex<float>(
              static_cast<float>(amplitude * std::cos(phase)),
              static_cast<float>(amplitude * std::sin(phase)));
    }
  }
}

}  // namespace

// Emits one synthetic complex-IF ADC cube per tick.
class AdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdcSourceOp)
  AdcSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    std::vector<std::complex<float>> cube(
        static_cast<size_t>(kNumChirps * kSamplesPerChirp), {0.0F, 0.0F});
    add_target(&cube, /*range_m=*/60.0, /*radial_velocity=*/0.0,
               /*amplitude=*/1.0);  // static clutter
    add_target(&cube, /*range_m=*/42.0, /*radial_velocity=*/5.5,
               /*amplitude=*/0.5);  // moving target

    auto tensor = radar::make_device_tensor(
        {kNumChirps, kSamplesPerChirp}, radar::complex_dtype<float>());
    if (cudaMemcpy(tensor->data(), cube.data(),
                   sizeof(std::complex<float>) * cube.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("AdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
  }
};

// Locates the range-Doppler peak and reports it in physical units via the
// upstream operators' axis outputs.
class PeakReportOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PeakReportOp)
  PeakReportOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("rd_maps");
    spec.input<holoscan::TensorMap>("range_bins");
    spec.input<holoscan::TensorMap>("velocity_bins");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    auto rd_map = single_tensor(op_input, "rd_maps");
    auto range_bins = single_tensor(op_input, "range_bins");
    auto velocity_bins = single_tensor(op_input, "velocity_bins");

    const auto shape = rd_map->shape();
    const int64_t num_doppler = shape[shape.size() - 2];
    const int64_t num_range = shape[shape.size() - 1];

    std::vector<std::complex<float>> rd(
        static_cast<size_t>(num_doppler * num_range));
    std::vector<float> ranges(static_cast<size_t>(num_range));
    std::vector<float> velocities(static_cast<size_t>(num_doppler));
    cudaDeviceSynchronize();
    cudaMemcpy(rd.data(), rd_map->data(),
               static_cast<size_t>(rd_map->nbytes()),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ranges.data(), range_bins->data(),
               static_cast<size_t>(range_bins->nbytes()),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities.data(), velocity_bins->data(),
               static_cast<size_t>(velocity_bins->nbytes()),
               cudaMemcpyDeviceToHost);

    // The zero-Doppler column is suppressed by the clutter filter; report
    // the strongest remaining (moving-target) cell.
    size_t peak = 0;
    float peak_mag = 0.0F;
    for (size_t i = 0; i < rd.size(); ++i) {
      const float mag = std::abs(rd[i]);
      if (mag > peak_mag) {
        peak_mag = mag;
        peak = i;
      }
    }
    const size_t doppler_bin = peak / static_cast<size_t>(num_range);
    const size_t range_bin = peak % static_cast<size_t>(num_range);
    HOLOSCAN_LOG_INFO(
        "Peak |RD| = {:.1f} at range bin {} ({:.2f} m), Doppler bin {} "
        "({:+.2f} m/s)",
        peak_mag, range_bin, ranges[range_bin], doppler_bin,
        velocities[doppler_bin]);
  }

 private:
  static std::shared_ptr<holoscan::Tensor> single_tensor(
      holoscan::InputContext& op_input, const char* port) {
    auto maybe = op_input.receive<holoscan::TensorMap>(port);
    if (!maybe || maybe->size() != 1) {
      throw std::runtime_error(std::string("PeakReportOp: expected one "
                                           "tensor on port ") + port);
    }
    return maybe->begin()->second;
  }
};

class RadarPipelineApp : public holoscan::Application {
 public:
  void compose() override {
    using holoscan::Arg;
    const radar::FMCWWaveformDescriptor waveform = demo_waveform();

    auto source = make_operator<AdcSourceOp>(
        "adc_source", make_condition<holoscan::CountCondition>(1));
    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", Arg("waveform") = waveform);
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter");
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", Arg("waveform") = waveform);
    auto sink = make_operator<PeakReportOp>("peak_report");

    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
    add_flow(doppler, sink, {{"rd_maps", "rd_maps"}});
    add_flow(range, sink, {{"range_bins", "range_bins"}});
    add_flow(doppler, sink, {{"velocity_bins", "velocity_bins"}});
  }
};

int main(int, char**) {
  auto app = holoscan::make_application<RadarPipelineApp>();
  app->run();
  return 0;
}
