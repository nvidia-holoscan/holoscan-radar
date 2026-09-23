// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {
namespace {

template <typename T>
class PyDopplerNotchOp : public DopplerNotchOp<T> {
 public:
  using DopplerNotchOp<T>::DopplerNotchOp;

  PyDopplerNotchOp(
      Fragment* fragment, const py::args& args,
      double center_velocity = 0.0, double stop_half_width = 0.0,
      double transition_width = 0.0, bool cyclic_velocity_axis = true,
      int64_t max_doppler_bins = 512, int64_t max_range_bins = 4096,
      int64_t max_rx = 1, int64_t max_tx = 0,
      const std::string& name = "doppler_notch")
      : DopplerNotchOp<T>([&] {
          MaxTensorExtents extents;
          extents.max_doppler_bins = max_doppler_bins;
          extents.max_range_bins = max_range_bins;
          if (max_rx > 1) extents.max_rx = max_rx;
          if (max_tx > 0) {
            extents.max_tx = max_tx;
            extents.max_rx = max_rx;
          }
          return extents;
        }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("center_velocity") = center_velocity);
    this->add_arg(Arg("stop_half_width") = stop_half_width);
    this->add_arg(Arg("transition_width") = transition_width);
    this->add_arg(Arg("cyclic_velocity_axis") = cyclic_velocity_axis);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_doppler_notch(py::module_& module, const char* class_name) {
  using Op = DopplerNotchOp<T>;
  using PyOp = PyDopplerNotchOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(module, class_name)
      .def(py::init<Fragment*, const py::args&, double, double, double, bool,
                    int64_t, int64_t, int64_t, int64_t,
                    const std::string&>(),
           py::arg("fragment"), py::arg("center_velocity") = 0.0,
           py::arg("stop_half_width") = 0.0,
           py::arg("transition_width") = 0.0,
           py::arg("cyclic_velocity_axis") = true,
           py::arg("max_doppler_bins") = 512,
           py::arg("max_range_bins") = 4096, py::arg("max_rx") = 1,
           py::arg("max_tx") = 0, py::arg("name") = "doppler_notch")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace
}  // namespace holoscan::radar

PYBIND11_MODULE(_doppler_notch_op, module) {
  holoscan::radar::ensure_radar_arg_setters();
  holoscan::radar::bind_doppler_notch<float>(
      module, "DopplerNotchOpF32");
  holoscan::radar::bind_doppler_notch<double>(
      module, "DopplerNotchOpF64");
}
