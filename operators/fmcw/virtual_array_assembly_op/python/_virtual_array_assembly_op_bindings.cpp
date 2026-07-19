// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyVirtualArrayAssemblyOp : public VirtualArrayAssemblyOp<T> {
 public:
  using VirtualArrayAssemblyOp<T>::VirtualArrayAssemblyOp;

  PyVirtualArrayAssemblyOp(
      Fragment* fragment, const py::args& args,
      const std::vector<double>& tx_positions = {},
      const std::vector<double>& rx_positions = {},
      double wavelength = 0.0,
      OverlapPolicy overlap_policy = OverlapPolicy::kDedup,
      const std::string& name = "virtual_array_assembly")
      : VirtualArrayAssemblyOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("tx_positions") = tx_positions);
    this->add_arg(Arg("rx_positions") = rx_positions);
    this->add_arg(Arg("wavelength") = wavelength);
    this->add_arg(Arg("overlap_policy") = overlap_policy);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_virtual_array_assembly(py::module_& m, const char* class_name) {
  using Op = VirtualArrayAssemblyOp<T>;
  using PyOp = PyVirtualArrayAssemblyOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Flatten (tx, rx) RD maps into a virtual channel axis.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "tx_positions : list[float], optional\n"
      "rx_positions : list[float], optional\n"
      "wavelength : float, optional\n"
      "overlap_policy : OverlapPolicy, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, const std::vector<double>&,
                    const std::vector<double>&, double, OverlapPolicy,
                    const std::string&>(),
           py::arg("fragment"),
           py::arg("tx_positions") = std::vector<double>{},
           py::arg("rx_positions") = std::vector<double>{},
           py::arg("wavelength") = 0.0,
           py::arg("overlap_policy") = OverlapPolicy::kDedup,
           py::arg("name") = "virtual_array_assembly")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_virtual_array_assembly_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "VirtualArrayAssemblyOp Python bindings";
  holoscan::radar::bind_virtual_array_assembly<float>(
      m, "VirtualArrayAssemblyOpF32");
  holoscan::radar::bind_virtual_array_assembly<double>(
      m, "VirtualArrayAssemblyOpF64");
}
