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

#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyRootMUSICEstimatorOp : public RootMUSICEstimatorOp<T> {
 public:
  using RootMUSICEstimatorOp<T>::RootMUSICEstimatorOp;

  PyRootMUSICEstimatorOp(
      Fragment* fragment, const py::args& args,
      const ArrayGeometry& array_geometry = ArrayGeometry{},
      const AngleConvention& angle_convention = AngleConvention{},
      int64_t num_sources = -1, double source_count_pfa = 1e-2,
      int64_t max_sources = 4,
      const std::vector<int64_t>& spatial_smoothing = {},
      bool forward_backward = true,
      const std::string& name = "root_music_estimator")
      : RootMUSICEstimatorOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("array_geometry") = array_geometry);
    this->add_arg(Arg("angle_convention") = angle_convention);
    this->add_arg(Arg("num_sources") = num_sources);
    this->add_arg(Arg("source_count_pfa") = source_count_pfa);
    this->add_arg(Arg("max_sources") = max_sources);
    this->add_arg(Arg("spatial_smoothing") = spatial_smoothing);
    this->add_arg(Arg("forward_backward") = forward_backward);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_root_music_estimator(py::module_& module,
                               const char* class_name) {
  using Op = RootMUSICEstimatorOp<T>;
  using PyOp = PyRootMUSICEstimatorOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      module, class_name,
      "Gridless Root-MUSIC sparse-path ULA DOA estimator.")
      .def(py::init<Fragment*, const py::args&, const ArrayGeometry&,
                    const AngleConvention&, int64_t, double, int64_t,
                    const std::vector<int64_t>&, bool, const std::string&>(),
           py::arg("fragment"),
           py::arg("array_geometry") = ArrayGeometry{},
           py::arg("angle_convention") = AngleConvention{},
           py::arg("num_sources") = int64_t{-1},
           py::arg("source_count_pfa") = 1e-2,
           py::arg("max_sources") = int64_t{4},
           py::arg("spatial_smoothing") = std::vector<int64_t>{},
           py::arg("forward_backward") = true,
           py::arg("name") = "root_music_estimator")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_root_music_estimator_op, module) {
  holoscan::radar::ensure_radar_arg_setters();
  module.doc() = "RootMUSICEstimatorOp Python bindings";
  holoscan::radar::bind_root_music_estimator<float>(
      module, "RootMUSICEstimatorOpF32");
  holoscan::radar::bind_root_music_estimator<double>(
      module, "RootMUSICEstimatorOpF64");
}
