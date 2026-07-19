// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyMVDREstimatorOp : public MVDREstimatorOp<T> {
 public:
  using MVDREstimatorOp<T>::MVDREstimatorOp;

  PyMVDREstimatorOp(
      Fragment* fragment, const py::args& args,
      const ArrayGeometry& array_geometry = ArrayGeometry{},
      const AngleConvention& angle_convention = AngleConvention{},
      const std::vector<double>& az_angles = {},
      const std::vector<double>& el_angles = {},
      double diagonal_loading = -1.0,
      int64_t max_sources = 1,
      const std::string& name = "mvdr_estimator")
      : MVDREstimatorOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("array_geometry") = array_geometry);
    this->add_arg(Arg("angle_convention") = angle_convention);
    this->add_arg(Arg("az_angles") = az_angles);
    this->add_arg(Arg("el_angles") = el_angles);
    this->add_arg(Arg("diagonal_loading") = diagonal_loading);
    this->add_arg(Arg("max_sources") = max_sources);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_mvdr_estimator(py::module_& m, const char* class_name) {
  using Op = MVDREstimatorOp<T>;
  using PyOp = PyMVDREstimatorOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Adaptive (MVDR / Capon) sparse-path DOA estimator.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "array_geometry : ArrayGeometry, optional\n"
      "angle_convention : AngleConvention, optional\n"
      "az_angles : list[float], optional\n"
      "el_angles : list[float], optional\n"
      "diagonal_loading : float, optional\n"
      "max_sources : int, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, const ArrayGeometry&,
                    const AngleConvention&, const std::vector<double>&,
                    const std::vector<double>&, double, int64_t,
                    const std::string&>(),
           py::arg("fragment"),
           py::arg("array_geometry") = ArrayGeometry{},
           py::arg("angle_convention") = AngleConvention{},
           py::arg("az_angles") = std::vector<double>{},
           py::arg("el_angles") = std::vector<double>{},
           py::arg("diagonal_loading") = -1.0,
           py::arg("max_sources") = int64_t{1},
           py::arg("name") = "mvdr_estimator")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_mvdr_estimator_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "MVDREstimatorOp Python bindings";
  holoscan::radar::bind_mvdr_estimator<float>(m, "MVDREstimatorOpF32");
  holoscan::radar::bind_mvdr_estimator<double>(m, "MVDREstimatorOpF64");

  m.def("retire_matx_solver_caches",
        &holoscan::radar::retire_matx_solver_caches,
        "Destroy MatX eig/inv plan caches while CUDA is still alive. "
        "Register via install_matx_cache_atexit() after importing cupy.");
}
