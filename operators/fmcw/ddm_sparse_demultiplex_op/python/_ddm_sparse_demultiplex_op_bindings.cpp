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

#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_op.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;
namespace holoscan::radar {
namespace {

template <typename T>
class PyDDMSparseDemultiplexOp : public DDMSparseDemultiplexOp<T> {
 public:
  PyDDMSparseDemultiplexOp(
      Fragment* fragment, const py::args& args, int64_t max_tx = 8,
      int64_t max_rx = 16, int64_t max_doppler_bins = 512,
      int64_t max_range_bins = 4096, int64_t max_detections = 4096,
      int64_t max_virtual_channels = 128, int64_t max_snapshots = 81,
      const DDMCodeDescriptor& ddm_code = {},
      const std::vector<double>& tx_positions = {},
      const std::vector<double>& rx_positions = {},
      OverlapPolicy overlap_policy = OverlapPolicy::kDedup,
      double wavelength = 0.0,
      const std::vector<int64_t>& snapshot_window = {0, 0},
      SnapshotEdgeMode doppler_edge = SnapshotEdgeMode::kWrap,
      SnapshotEdgeMode range_edge = SnapshotEdgeMode::kShift,
      const std::string& name = "ddm_sparse_demultiplex")
      : DDMSparseDemultiplexOp<T>([&] {
          MaxTensorExtents extents;
          extents.max_tx = max_tx;
          extents.max_rx = max_rx;
          extents.max_doppler_bins = max_doppler_bins;
          extents.max_range_bins = max_range_bins;
          extents.max_items = max_detections;
          extents.max_detections = max_detections;
          extents.max_virtual_channels = max_virtual_channels;
          extents.max_snapshots = max_snapshots;
          return extents;
        }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("ddm_code") = ddm_code);
    this->add_arg(Arg("tx_positions") = tx_positions);
    this->add_arg(Arg("rx_positions") = rx_positions);
    this->add_arg(Arg("overlap_policy") = overlap_policy);
    this->add_arg(Arg("wavelength") = wavelength);
    this->add_arg(Arg("snapshot_window") = snapshot_window);
    this->add_arg(Arg("doppler_edge") = doppler_edge);
    this->add_arg(Arg("range_edge") = range_edge);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind(py::module_& module, const char* name) {
  using Op = DDMSparseDemultiplexOp<T>;
  using PyOp = PyDDMSparseDemultiplexOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(module, name)
      .def(py::init<
               Fragment*, const py::args&, int64_t, int64_t, int64_t, int64_t,
               int64_t, int64_t, int64_t, const DDMCodeDescriptor&,
               const std::vector<double>&, const std::vector<double>&,
               OverlapPolicy, double, const std::vector<int64_t>&,
               SnapshotEdgeMode, SnapshotEdgeMode, const std::string&>(),
           py::arg("fragment"), py::arg("max_tx") = 8, py::arg("max_rx") = 16,
           py::arg("max_doppler_bins") = 512,
           py::arg("max_range_bins") = 4096,
           py::arg("max_detections") = 4096,
           py::arg("max_virtual_channels") = 128,
           py::arg("max_snapshots") = 81,
           py::arg("ddm_code") = DDMCodeDescriptor{},
           py::arg("tx_positions") = std::vector<double>{},
           py::arg("rx_positions") = std::vector<double>{},
           py::arg("overlap_policy") = OverlapPolicy::kDedup,
           py::arg("wavelength") = 0.0,
           py::arg("snapshot_window") = std::vector<int64_t>{0, 0},
           py::arg("doppler_edge") = SnapshotEdgeMode::kWrap,
           py::arg("range_edge") = SnapshotEdgeMode::kShift,
           py::arg("name") = "ddm_sparse_demultiplex")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace
}  // namespace holoscan::radar

PYBIND11_MODULE(_ddm_sparse_demultiplex_op, module) {
  holoscan::radar::ensure_radar_arg_setters();
  holoscan::radar::bind<float>(module, "DDMSparseDemultiplexOpF32");
  holoscan::radar::bind<double>(module, "DDMSparseDemultiplexOpF64");
}
