// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/condition.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resource.hpp>

#include <memory>
#include <string>

#include <radar_op/radar_op.hpp>

namespace py = pybind11;

namespace holoscan::radar {

// Forward optional Condition / Resource positional args from py::args onto the
// operator. Takes a raw Operator* — calling shared_from_this() inside the
// constructor doesn't work because the object isn't yet owned by a shared_ptr.
static void add_conditions_and_resources(holoscan::Operator* op, const py::args& args) {
  for (const auto& item : args) {
    auto obj = py::cast<py::object>(item);
    try {
      op->add_arg(obj.cast<std::shared_ptr<holoscan::Condition>>());
      continue;
    } catch (const py::cast_error&) {}
    try {
      op->add_arg(obj.cast<std::shared_ptr<holoscan::Resource>>());
    } catch (const py::cast_error&) {
      HOLOSCAN_LOG_WARN(
          "Operator '{}': positional arg of type '{}' is neither a Condition nor a Resource — "
          "add_arg skipped.",
          op->name(),
          py::str(obj.get_type()).cast<std::string>());
    }
  }
}

class PyRadarOp : public RadarOp {
 public:
  using RadarOp::RadarOp;

  PyRadarOp(holoscan::Fragment* fragment, const py::args& args,
                  // TODO: add keyword parameters here, e.g.:
                  // int my_param = 1,
                  const std::string& name = "radar_op")
      : RadarOp() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
    // TODO: forward keyword parameters to args_, e.g.:
    // this->add_arg(holoscan::Arg("my_param") = my_param);
    setup(*spec_);
    add_conditions_and_resources(this, args);
  }
};

}  // namespace holoscan::radar

PYBIND11_MODULE(_radar_op, m) {
  using namespace holoscan::radar;

  m.doc() = "RadarOp Python bindings";

  py::class_<RadarOp, PyRadarOp, holoscan::Operator, std::shared_ptr<RadarOp>>(
      m, "RadarOp",
      "Construct RadarOp.\n\n"
      // TODO: expand the docstring to document ports and parameters.
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "    The fragment this operator belongs to.\n"
      "*args\n    Optional Condition or Resource objects.\n"
      "name : str, optional\n"
      "    Operator name (default 'radar_op').\n")
      .def(py::init<holoscan::Fragment*, const py::args&, const std::string&>(),
           py::arg("fragment"),
           py::arg("name") = "radar_op")
      .def("setup", &RadarOp::setup, py::arg("spec"));
}
