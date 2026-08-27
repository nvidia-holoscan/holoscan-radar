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

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyClusterDBSCANOp : public ClusterDBSCANOp<T> {
 public:
  using ClusterDBSCANOp<T>::ClusterDBSCANOp;

  PyClusterDBSCANOp(
      Fragment* fragment, const py::args& args,
      const std::vector<double>& epsilon = {},
      int64_t min_points = 0,
      int64_t max_clusters = 0,
      int64_t max_points_ext = 8192,
      int64_t max_clusters_ext = 256,
      const std::string& name = "cluster_dbscan")
      : ClusterDBSCANOp<T>([&] {
          MaxTensorExtents extents;
          extents.max_points = max_points_ext;
          extents.max_clusters = max_clusters_ext;
          return extents;
        }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("epsilon") = epsilon);
    this->add_arg(Arg("min_points") = min_points);
    this->add_arg(Arg("max_clusters") = max_clusters);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_cluster_dbscan(py::module_& m, const char* class_name) {
  using Op = ClusterDBSCANOp<T>;
  using PyOp = PyClusterDBSCANOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Density clustering (DBSCAN) over a feature matrix.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "epsilon : list[float], optional\n"
      "min_points : int, optional\n"
      "max_clusters : int, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&,
                    const std::vector<double>&, int64_t, int64_t,
                    int64_t, int64_t, const std::string&>(),
           py::arg("fragment"),
           py::arg("epsilon") = std::vector<double>{},
           py::arg("min_points") = int64_t{0},
           py::arg("max_clusters") = int64_t{0},
           py::arg("max_points_ext") = int64_t{8192},
           py::arg("max_clusters_ext") = int64_t{256},
           py::arg("name") = "cluster_dbscan")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_cluster_dbscan_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "ClusterDBSCANOp Python bindings";
  holoscan::radar::bind_cluster_dbscan<float>(m, "ClusterDBSCANOpF32");
  holoscan::radar::bind_cluster_dbscan<double>(m, "ClusterDBSCANOpF64");
}
