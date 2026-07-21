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

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PySnapshotGatherOp : public SnapshotGatherOp<T> {
 public:
  using SnapshotGatherOp<T>::SnapshotGatherOp;

  PySnapshotGatherOp(
      Fragment* fragment, const py::args& args,
      const std::vector<int64_t>& snapshot_window = {0, 0},
      SnapshotEdgeMode doppler_edge = SnapshotEdgeMode::kWrap,
      SnapshotEdgeMode range_edge = SnapshotEdgeMode::kShift,
      const std::string& name = "snapshot_gather")
      : SnapshotGatherOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("snapshot_window") = snapshot_window);
    this->add_arg(Arg("doppler_edge") = doppler_edge);
    this->add_arg(Arg("range_edge") = range_edge);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_snapshot_gather(py::module_& m, const char* class_name) {
  using Op = SnapshotGatherOp<T>;
  using PyOp = PySnapshotGatherOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Gathers per-detection channel snapshots from RD maps.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "snapshot_window : list[int], optional\n"
      "doppler_edge : SnapshotEdgeMode, optional\n"
      "range_edge : SnapshotEdgeMode, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, const std::vector<int64_t>&,
                    SnapshotEdgeMode, SnapshotEdgeMode, const std::string&>(),
           py::arg("fragment"),
           py::arg("snapshot_window") = std::vector<int64_t>{0, 0},
           py::arg("doppler_edge") = SnapshotEdgeMode::kWrap,
           py::arg("range_edge") = SnapshotEdgeMode::kShift,
           py::arg("name") = "snapshot_gather")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_snapshot_gather_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "SnapshotGatherOp Python bindings";
  holoscan::radar::bind_snapshot_gather<float>(m, "SnapshotGatherOpF32");
  holoscan::radar::bind_snapshot_gather<double>(m, "SnapshotGatherOpF64");
}
