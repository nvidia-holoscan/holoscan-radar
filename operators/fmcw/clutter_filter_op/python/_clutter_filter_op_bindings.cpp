// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyClutterFilterOp : public ClutterFilterOp<T> {
 public:
  using ClutterFilterOp<T>::ClutterFilterOp;

  PyClutterFilterOp(
      Fragment* fragment, const py::args& args,
      ClutterFilterMethod method = ClutterFilterMethod::kMeanSubtraction,
      const std::string& name = "clutter_filter")
      : ClutterFilterOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("method") = method);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_clutter_filter(py::module_& m, const char* class_name) {
  using Op = ClutterFilterOp<T>;
  using PyOp = PyClutterFilterOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Slow-time clutter / zero-Doppler suppression.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "method : ClutterFilterMethod, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, ClutterFilterMethod,
                    const std::string&>(),
           py::arg("fragment"),
           py::arg("method") = ClutterFilterMethod::kMeanSubtraction,
           py::arg("name") = "clutter_filter")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_clutter_filter_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "ClutterFilterOp Python bindings";
  holoscan::radar::bind_clutter_filter<float>(m, "ClutterFilterOpF32");
  holoscan::radar::bind_clutter_filter<double>(m, "ClutterFilterOpF64");
}
