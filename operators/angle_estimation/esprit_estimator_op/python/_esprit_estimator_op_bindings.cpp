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

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyESPRITEstimatorOp : public ESPRITEstimatorOp<T> {
 public:
  using ESPRITEstimatorOp<T>::ESPRITEstimatorOp;

  PyESPRITEstimatorOp(
      Fragment* fragment, const py::args& args,
      const ArrayGeometry& array_geometry = ArrayGeometry{},
      const AngleConvention& angle_convention = AngleConvention{},
      int64_t num_sources = -1, double source_count_pfa = 1e-2,
      int64_t max_sources = 4,
      const std::vector<int64_t>& spatial_smoothing = {},
      bool forward_backward = true,
      const ShiftInvariantSubarrayPair& subarray_spec =
          ShiftInvariantSubarrayPair{},
      const std::string& name = "esprit_estimator")
      : ESPRITEstimatorOp<T>() {
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
    this->add_arg(Arg("subarray_spec") = subarray_spec);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_esprit_estimator(py::module_& module, const char* class_name) {
  using Op = ESPRITEstimatorOp<T>;
  using PyOp = PyESPRITEstimatorOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      module, class_name,
      "Gridless ESPRIT sparse-path shift-invariant DOA estimator.")
      .def(py::init<Fragment*, const py::args&, const ArrayGeometry&,
                    const AngleConvention&, int64_t, double, int64_t,
                    const std::vector<int64_t>&, bool,
                    const ShiftInvariantSubarrayPair&, const std::string&>(),
           py::arg("fragment"),
           py::arg("array_geometry") = ArrayGeometry{},
           py::arg("angle_convention") = AngleConvention{},
           py::arg("num_sources") = int64_t{-1},
           py::arg("source_count_pfa") = 1e-2,
           py::arg("max_sources") = int64_t{4},
           py::arg("spatial_smoothing") = std::vector<int64_t>{},
           py::arg("forward_backward") = true,
           py::arg("subarray_spec") = ShiftInvariantSubarrayPair{},
           py::arg("name") = "esprit_estimator")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_esprit_estimator_op, module) {
  holoscan::radar::ensure_radar_arg_setters();
  module.doc() = "ESPRITEstimatorOp Python bindings";
  holoscan::radar::bind_esprit_estimator<float>(module, "ESPRITEstimatorOpF32");
  holoscan::radar::bind_esprit_estimator<double>(module, "ESPRITEstimatorOpF64");
}
