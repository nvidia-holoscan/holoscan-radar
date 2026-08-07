// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyMIMODeinterleaveOp : public MIMODeinterleaveOp<T> {
 public:
  using MIMODeinterleaveOp<T>::MIMODeinterleaveOp;

  PyMIMODeinterleaveOp(
      Fragment* fragment, const py::args& args,
      const TXSchedule& tx_schedule = TXSchedule{},
      const std::string& name = "mimo_deinterleave")
      : MIMODeinterleaveOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("tx_schedule") = tx_schedule);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_mimo_deinterleave(py::module_& m, const char* class_name) {
  using Op = MIMODeinterleaveOp<T>;
  using PyOp = PyMIMODeinterleaveOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "TDM front-end marshaling: split interleaved slow-time into TX axis.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "tx_schedule : TXSchedule, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, const TXSchedule&,
                    const std::string&>(),
           py::arg("fragment"),
           py::arg("tx_schedule") = TXSchedule{},
           py::arg("name") = "mimo_deinterleave")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_mimo_deinterleave_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "MIMODeinterleaveOp Python bindings";
  holoscan::radar::bind_mimo_deinterleave<float>(m, "MIMODeinterleaveOpF32");
  holoscan::radar::bind_mimo_deinterleave<double>(m, "MIMODeinterleaveOpF64");
}
