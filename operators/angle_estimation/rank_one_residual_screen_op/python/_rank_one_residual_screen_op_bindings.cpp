// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {
namespace {

template <typename T>
class PyRankOneResidualScreenOp : public RankOneResidualScreenOp<T> {
 public:
  PyRankOneResidualScreenOp(
      Fragment* fragment, const py::args& args,
      const ArrayGeometry& array_geometry = ArrayGeometry{},
      const AngleConvention& angle_convention = AngleConvention{},
      double residual_threshold = 0.10, int64_t max_items = 16384,
      int64_t max_channels = 64, int64_t max_snapshots = 9,
      const std::string& name = "rank_one_residual_screen")
      : RankOneResidualScreenOp<T>([&] {
          MaxTensorExtents extents;
          extents.max_items = max_items;
          extents.max_channels = max_channels;
          extents.max_snapshots = max_snapshots;
          return extents;
        }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("array_geometry") = array_geometry);
    this->add_arg(Arg("angle_convention") = angle_convention);
    this->add_arg(Arg("residual_threshold") = residual_threshold);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_screen(py::module_& module, const char* class_name) {
  using Op = RankOneResidualScreenOp<T>;
  using PyOp = PyRankOneResidualScreenOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      module, class_name,
      "Routes snapshots with a large normalized rank-one residual.")
      .def(py::init<Fragment*, const py::args&, const ArrayGeometry&,
                    const AngleConvention&, double, int64_t, int64_t,
                    int64_t, const std::string&>(),
           py::arg("fragment"),
           py::arg("array_geometry") = ArrayGeometry{},
           py::arg("angle_convention") = AngleConvention{},
           py::arg("residual_threshold") = 0.10,
           py::arg("max_items") = int64_t{16384},
           py::arg("max_channels") = int64_t{64},
           py::arg("max_snapshots") = int64_t{9},
           py::arg("name") = "rank_one_residual_screen")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace
}  // namespace holoscan::radar

PYBIND11_MODULE(_rank_one_residual_screen_op, module) {
  holoscan::radar::ensure_radar_arg_setters();
  module.doc() = "RankOneResidualScreenOp Python bindings";
  holoscan::radar::bind_screen<float>(module, "RankOneResidualScreenOpF32");
  holoscan::radar::bind_screen<double>(module, "RankOneResidualScreenOpF64");
}
