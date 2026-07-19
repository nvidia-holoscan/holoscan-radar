// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyFMCWRangeProcessorOp : public FMCWRangeProcessorOp<T> {
 public:
  using FMCWRangeProcessorOp<T>::FMCWRangeProcessorOp;

  PyFMCWRangeProcessorOp(
      Fragment* fragment, const py::args& args,
      const WindowSpec& range_window = WindowSpec{},
      bool apply_dechirp = false,
      std::optional<FMCWWaveformDescriptor> waveform = std::nullopt,
      FftNorm fft_norm = FftNorm::kBackward,
      double propagation_speed = kDefaultPropagationSpeed,
      std::optional<int64_t> nfft = std::nullopt,
      const std::string& name = "fmcw_range_processor")
      : FMCWRangeProcessorOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("range_window") = range_window);
    this->add_arg(Arg("apply_dechirp") = apply_dechirp);
    if (waveform.has_value()) {
      this->add_arg(Arg("waveform") = waveform.value());
    }
    this->add_arg(Arg("fft_norm") = fft_norm);
    this->add_arg(Arg("propagation_speed") = propagation_speed);
    if (nfft.has_value()) {
      this->add_arg(Arg("nfft") = nfft.value());
    }
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_fmcw_range_processor(py::module_& m, const char* class_name) {
  using Op = FMCWRangeProcessorOp<T>;
  using PyOp = PyFMCWRangeProcessorOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "FMCW range compression: windowed FFT along fast-time.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "range_window : WindowSpec, optional\n"
      "apply_dechirp : bool, optional\n"
      "waveform : FMCWWaveformDescriptor, optional\n"
      "fft_norm : FftNorm, optional\n"
      "propagation_speed : float, optional\n"
      "nfft : int, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, const WindowSpec&, bool,
                    std::optional<FMCWWaveformDescriptor>, FftNorm, double,
                    std::optional<int64_t>, const std::string&>(),
           py::arg("fragment"),
           py::arg("range_window") = WindowSpec{},
           py::arg("apply_dechirp") = false,
           py::arg("waveform") = py::none(),
           py::arg("fft_norm") = FftNorm::kBackward,
           py::arg("propagation_speed") = kDefaultPropagationSpeed,
           py::arg("nfft") = py::none(),
           py::arg("name") = "fmcw_range_processor")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_fmcw_range_processor_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "FMCWRangeProcessorOp Python bindings";
  holoscan::radar::bind_fmcw_range_processor<float>(
      m, "FMCWRangeProcessorOpF32");
  holoscan::radar::bind_fmcw_range_processor<double>(
      m, "FMCWRangeProcessorOpF64");
}
