// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyFMCWDopplerProcessorOp : public FMCWDopplerProcessorOp<T> {
 public:
  using FMCWDopplerProcessorOp<T>::FMCWDopplerProcessorOp;

  PyFMCWDopplerProcessorOp(
      Fragment* fragment, const py::args& args,
      const WindowSpec& doppler_window = WindowSpec{},
      bool center_zero_doppler = true,
      std::optional<FMCWWaveformDescriptor> waveform = std::nullopt,
      FftNorm fft_norm = FftNorm::kBackward,
      std::optional<int64_t> nfft = std::nullopt,
      const std::string& name = "fmcw_doppler_processor")
      : FMCWDopplerProcessorOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("doppler_window") = doppler_window);
    this->add_arg(Arg("center_zero_doppler") = center_zero_doppler);
    if (waveform.has_value()) {
      this->add_arg(Arg("waveform") = waveform.value());
    }
    this->add_arg(Arg("fft_norm") = fft_norm);
    if (nfft.has_value()) {
      this->add_arg(Arg("nfft") = nfft.value());
    }
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_fmcw_doppler_processor(py::module_& m, const char* class_name) {
  using Op = FMCWDopplerProcessorOp<T>;
  using PyOp = PyFMCWDopplerProcessorOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "FMCW Doppler compression: windowed FFT along slow-time.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "doppler_window : WindowSpec, optional\n"
      "center_zero_doppler : bool, optional\n"
      "waveform : FMCWWaveformDescriptor, optional\n"
      "fft_norm : FftNorm, optional\n"
      "nfft : int, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, const WindowSpec&, bool,
                    std::optional<FMCWWaveformDescriptor>, FftNorm,
                    std::optional<int64_t>, const std::string&>(),
           py::arg("fragment"),
           py::arg("doppler_window") = WindowSpec{},
           py::arg("center_zero_doppler") = true,
           py::arg("waveform") = py::none(),
           py::arg("fft_norm") = FftNorm::kBackward,
           py::arg("nfft") = py::none(),
           py::arg("name") = "fmcw_doppler_processor")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_fmcw_doppler_processor_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "FMCWDopplerProcessorOp Python bindings";
  holoscan::radar::bind_fmcw_doppler_processor<float>(
      m, "FMCWDopplerProcessorOpF32");
  holoscan::radar::bind_fmcw_doppler_processor<double>(
      m, "FMCWDopplerProcessorOpF64");
}
