// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyMIMOCoherentCompensationOp : public MIMOCoherentCompensationOp<T> {
 public:
  using MIMOCoherentCompensationOp<T>::MIMOCoherentCompensationOp;

  PyMIMOCoherentCompensationOp(
      Fragment* fragment, const py::args& args,
      std::int64_t max_tx = 8,
      std::int64_t max_rx = 16,
      std::int64_t max_doppler_bins = 512,
      std::int64_t max_range_bins = 4096,
      MultiplexingScheme multiplexing_scheme =
          MultiplexingScheme::kUnspecified,
      const FMCWWaveformDescriptor& waveform =
          FMCWWaveformDescriptor{},
      const DDMCodeDescriptor& ddm_code =
          DDMCodeDescriptor{},
      const std::string& name =
          "mimo_coherent_compensation")
      : MIMOCoherentCompensationOp<T>([&] {
          MaxTensorExtents extents;
          extents.max_tx = max_tx;
          extents.max_rx = max_rx;
          extents.max_doppler_bins = max_doppler_bins;
          extents.max_range_bins = max_range_bins;
          return extents;
        }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("multiplexing_scheme") = multiplexing_scheme);
    this->add_arg(Arg("waveform") = waveform);
    this->add_arg(Arg("ddm_code") = ddm_code);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_mimo_coherent_compensation(py::module_& m, const char* class_name) {
  using Op = MIMOCoherentCompensationOp<T>;
  using PyOp = PyMIMOCoherentCompensationOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "MIMO coherent compensation (TDM phase / DDM demux).\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "max_tx : int, optional\n"
      "max_rx : int, optional\n"
      "max_doppler_bins : int, optional\n"
      "max_range_bins : int, optional\n"
      "multiplexing_scheme : MultiplexingScheme, optional\n"
      "waveform : FMCWWaveformDescriptor, optional\n"
      "ddm_code : DDMCodeDescriptor, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&,
                    std::int64_t, std::int64_t,
                    std::int64_t, std::int64_t,
                    MultiplexingScheme,
                    const FMCWWaveformDescriptor&,
                    const DDMCodeDescriptor&,
                    const std::string&>(),
           py::arg("fragment"),
           py::arg("max_tx") = 8,
           py::arg("max_rx") = 16,
           py::arg("max_doppler_bins") = 512,
           py::arg("max_range_bins") = 4096,
           py::arg("multiplexing_scheme") =
               MultiplexingScheme::kUnspecified,
           py::arg("waveform") =
               FMCWWaveformDescriptor{},
           py::arg("ddm_code") = DDMCodeDescriptor{},
           py::arg("name") =
               "mimo_coherent_compensation")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_mimo_coherent_compensation_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "MIMOCoherentCompensationOp Python bindings";
  holoscan::radar::bind_mimo_coherent_compensation<float>(
      m, "MIMOCoherentCompensationOpF32");
  holoscan::radar::bind_mimo_coherent_compensation<double>(
      m, "MIMOCoherentCompensationOpF64");
}
