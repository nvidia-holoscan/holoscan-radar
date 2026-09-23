// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyBeamscanEstimatorOp : public BeamscanEstimatorOp<T> {
 public:
  using BeamscanEstimatorOp<T>::BeamscanEstimatorOp;

  PyBeamscanEstimatorOp(Fragment* fragment, const py::args& args,
                        const ArrayGeometry& array_geometry = ArrayGeometry{},
                        const AngleConvention& angle_convention = AngleConvention{},
                        const std::vector<double>& az_coordinates = {},
                        const std::vector<double>& el_coordinates = {},
                        const WindowSpec& array_taper = WindowSpec{},
                        PeakInterpolation peak_interpolation = PeakInterpolation::kNone,
                        std::int64_t max_items = 16384, std::int64_t max_az = 101,
                        std::int64_t max_el = 51, const std::string& name = "beamscan_estimator")
  : BeamscanEstimatorOp<T>([&] {
      MaxTensorExtents extents;
      extents.max_items = max_items;
      extents.max_az = max_az;
      extents.max_el = max_el;
      return extents;
    }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("array_geometry") = array_geometry);
    this->add_arg(Arg("angle_convention") = angle_convention);
    this->add_arg(Arg("az_coordinates") = az_coordinates);
    this->add_arg(Arg("el_coordinates") = el_coordinates);
    this->add_arg(Arg("array_taper") = array_taper);
    this->add_arg(Arg("peak_interpolation") = peak_interpolation);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_beamscan_estimator(py::module_& m, const char* class_name) {
  using Op = BeamscanEstimatorOp<T>;
  using PyOp = PyBeamscanEstimatorOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Conventional (Bartlett) sparse-path DOA estimator.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "array_geometry : ArrayGeometry, optional\n"
      "angle_convention : AngleConvention, optional\n"
      "az_coordinates : list[float], optional\n"
      "el_coordinates : list[float], optional\n"
      "array_taper : WindowSpec, optional\n"
      "peak_interpolation : PeakInterpolation, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, const ArrayGeometry&, const AngleConvention&,
                    const std::vector<double>&, const std::vector<double>&, const WindowSpec&,
                    PeakInterpolation, std::int64_t, std::int64_t, std::int64_t,
                    const std::string&>(),
           py::arg("fragment"), py::arg("array_geometry") = ArrayGeometry{},
           py::arg("angle_convention") = AngleConvention{},
           py::arg("az_coordinates") = std::vector<double>{},
           py::arg("el_coordinates") = std::vector<double>{}, py::arg("array_taper") = WindowSpec{},
           py::arg("peak_interpolation") = PeakInterpolation::kNone, py::arg("max_items") = 16384,
           py::arg("max_az") = 101, py::arg("max_el") = 51, py::arg("name") = "beamscan_estimator")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_beamscan_estimator_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "BeamscanEstimatorOp Python bindings";
  holoscan::radar::bind_beamscan_estimator<float>(m, "BeamscanEstimatorOpF32");
  holoscan::radar::bind_beamscan_estimator<double>(m, "BeamscanEstimatorOpF64");
}
