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

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyPhaseShiftBeamformerOp : public PhaseShiftBeamformerOp<T> {
 public:
  using PhaseShiftBeamformerOp<T>::PhaseShiftBeamformerOp;

  PyPhaseShiftBeamformerOp(
      Fragment* fragment, const py::args& args,
      const ArrayGeometry& array_geometry = ArrayGeometry{},
      const AngleConvention& angle_convention = AngleConvention{},
      const std::vector<double>& az_angles = {},
      const std::vector<double>& el_angles = {},
      const WindowSpec& array_taper = WindowSpec{},
      const std::string& name = "phase_shift_beamformer")
      : PhaseShiftBeamformerOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("array_geometry") = array_geometry);
    this->add_arg(Arg("angle_convention") = angle_convention);
    this->add_arg(Arg("az_angles") = az_angles);
    this->add_arg(Arg("el_angles") = el_angles);
    this->add_arg(Arg("array_taper") = array_taper);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_phase_shift_beamformer(py::module_& m, const char* class_name) {
  using Op = PhaseShiftBeamformerOp<T>;
  using PyOp = PyPhaseShiftBeamformerOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Conventional (Bartlett) dense-path beamformer.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "array_geometry : ArrayGeometry, optional\n"
      "angle_convention : AngleConvention, optional\n"
      "az_angles : list[float], optional\n"
      "el_angles : list[float], optional\n"
      "array_taper : WindowSpec, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, const ArrayGeometry&,
                    const AngleConvention&, const std::vector<double>&,
                    const std::vector<double>&, const WindowSpec&,
                    const std::string&>(),
           py::arg("fragment"),
           py::arg("array_geometry") = ArrayGeometry{},
           py::arg("angle_convention") = AngleConvention{},
           py::arg("az_angles") = std::vector<double>{},
           py::arg("el_angles") = std::vector<double>{},
           py::arg("array_taper") = WindowSpec{},
           py::arg("name") = "phase_shift_beamformer")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_phase_shift_beamformer_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "PhaseShiftBeamformerOp Python bindings";
  holoscan::radar::bind_phase_shift_beamformer<float>(
      m, "PhaseShiftBeamformerOpF32");
  holoscan::radar::bind_phase_shift_beamformer<double>(
      m, "PhaseShiftBeamformerOpF64");
}
