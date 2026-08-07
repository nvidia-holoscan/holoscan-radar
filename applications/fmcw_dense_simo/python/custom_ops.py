# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# App-specific operators for fmcw_dense_simo: the dense-path bridges
# (AnglePeakPickOp, CubeAoaLookupOp, TrailingIdxColumnsOp) plus
# DenseClusterReportOp. Colocated with the app (only fmcw_dense_simo needs
# these); the shared ADC source lives in applications/common/python/ops.py.
#
# DenseClusterReportOp duplicates the shared (C++-bound) ClusterReportOp's
# 4-column logic in pure Python rather than reusing it: the Python
# ClusterReportOp binding does not expose az_grid / el_grid / cadence as
# read-write (unlike the C++ struct fields the app sets directly - see
# fmcw_dense_simo.cpp), so there is no way to configure the dense path from
# Python today. That is a binding gap, not fixed here.
#
# Imported bare (``from custom_ops import ...``) from fmcw_dense_simo.py: see
# config.py's docstring for why this is script-local, not a shared package.

from __future__ import annotations

import logging

import cupy as cp
import numpy as np
from common.python.demo_config import port_tensor, tensor_to_numpy

from holoscan.core import Operator, OperatorSpec
from holoscan.radar import CadenceMeter

logger = logging.getLogger("fmcw_dense_simo")


class AnglePeakPickOp(Operator):
    """One detection per RD cell: argmax (az, el) beam by CFAR score."""

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("detection_idx")
        spec.input("detection_score")
        spec.output("detection_idx")

    def compute(self, op_input, op_output, context) -> None:
        idx = port_tensor(op_input.receive("detection_idx"), "detection_idx")
        score = port_tensor(op_input.receive("detection_score"), "detection_score")
        host_idx = tensor_to_numpy(idx).astype(np.int64, copy=False)
        host_score = tensor_to_numpy(score).astype(np.float32, copy=False)

        if host_idx.ndim != 2 or host_idx.shape[1] != 4:
            raise RuntimeError("AnglePeakPickOp: detection_idx must be (N, 4)")
        if host_score.ndim != 4:
            raise RuntimeError(
                "AnglePeakPickOp: detection_score must be rank-4 "
                "(num_az, num_el, num_doppler, num_range)"
            )

        n = host_idx.shape[0]
        na, ne, nd, nr = host_score.shape
        seen: set[tuple[int, int]] = set()
        cells: list[tuple[int, int]] = []
        for i in range(n):
            d = int(host_idx[i, 2])
            r = int(host_idx[i, 3])
            if d < 0 or d >= nd or r < 0 or r >= nr:
                raise RuntimeError(
                    "AnglePeakPickOp: detection (doppler, range) out of score bounds"
                )
            key = (d, r)
            if key not in seen:
                seen.add(key)
                cells.append(key)

        out = np.zeros((len(cells), 4), dtype=np.int64)
        for c, (d, r) in enumerate(cells):
            slab = host_score[:, :, d, r]
            flat = int(np.argmax(slab))
            best_az = flat // ne
            best_el = flat % ne
            out[c, 0] = best_az
            out[c, 1] = best_el
            out[c, 2] = d
            out[c, 3] = r

        op_output.emit(
            {"detection_idx": cp.asarray(out)},
            "detection_idx",
        )


class CubeAoaLookupOp(Operator):
    """Map (az_bin, el_bin) columns to physical angles; emit row_index."""

    def __init__(
        self,
        fragment,
        *args,
        az_grid: list[float],
        el_grid: list[float],
        **kwargs,
    ):
        self.az_grid = az_grid
        self.el_grid = el_grid
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("detection_idx")
        spec.output("aoa_estimates")
        spec.output("row_index")

    def compute(self, op_input, op_output, context) -> None:
        idx = port_tensor(op_input.receive("detection_idx"), "detection_idx")
        host_idx = tensor_to_numpy(idx).astype(np.int64, copy=False)
        if host_idx.ndim != 2 or host_idx.shape[1] < 4:
            raise RuntimeError("CubeAoaLookupOp: detection_idx must be (num_points, >= 4)")
        n = host_idx.shape[0]
        num_az = len(self.az_grid)
        num_el = len(self.el_grid)
        aoa = np.zeros((n, 2), dtype=np.float32)
        rows = np.arange(n, dtype=np.int64)
        for i in range(n):
            az_bin = int(host_idx[i, 0])
            el_bin = int(host_idx[i, 1])
            if az_bin < 0 or az_bin >= num_az or el_bin < 0 or el_bin >= num_el:
                raise RuntimeError("CubeAoaLookupOp: detection az/el bin out of grid range")
            aoa[i, 0] = self.az_grid[az_bin]
            aoa[i, 1] = self.el_grid[el_bin]
        op_output.emit({"aoa_estimates": cp.asarray(aoa)}, "aoa_estimates")
        op_output.emit({"row_index": cp.asarray(rows)}, "row_index")


class DenseClusterReportOp(Operator):
    """Dense-aware cluster log (4-column points / rep_idx)."""

    def __init__(
        self,
        fragment,
        *args,
        az_grid: list[float],
        el_grid: list[float],
        cadence: CadenceMeter | None = None,
        **kwargs,
    ):
        self.az_grid = az_grid
        self.el_grid = el_grid
        self.cadence = cadence
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("points")
        spec.input("labels")
        spec.input("sizes")
        spec.input("rep_idx")
        spec.input("range_bins")
        spec.input("velocity_bins")

    def compute(self, op_input, op_output, context) -> None:
        points = tensor_to_numpy(port_tensor(op_input.receive("points"), "points")).astype(
            np.float32, copy=False
        )
        labels = tensor_to_numpy(port_tensor(op_input.receive("labels"), "labels")).astype(
            np.int64, copy=False
        )
        sizes = tensor_to_numpy(port_tensor(op_input.receive("sizes"), "sizes")).astype(
            np.int64, copy=False
        )
        rep_idx = tensor_to_numpy(port_tensor(op_input.receive("rep_idx"), "rep_idx")).astype(
            np.int64, copy=False
        )
        ranges = tensor_to_numpy(port_tensor(op_input.receive("range_bins"), "range_bins")).astype(
            np.float32, copy=False
        )
        velocities = tensor_to_numpy(
            port_tensor(op_input.receive("velocity_bins"), "velocity_bins")
        ).astype(np.float32, copy=False)

        num_points = points.shape[0]
        num_clusters = sizes.shape[0]

        noise = int(np.sum(labels < 0))
        logger.info(
            "dense: %d detections -> %d clusters (%d noise)",
            num_points,
            num_clusters,
            noise,
        )

        for c in range(num_clusters):
            n = int(sizes[c])
            mask = labels == c
            inv = 1.0 / n if n > 0 else 0.0
            centroid_r = float(np.sum(points[mask, 0]) * inv)
            centroid_v = float(np.sum(points[mask, 1]) * inv)
            centroid_az = float(np.sum(points[mask, 2]) * inv)
            centroid_el = float(np.sum(points[mask, 3]) * inv)
            az_bin = int(rep_idx[c, 0])
            el_bin = int(rep_idx[c, 1])
            dd = int(rep_idx[c, 2])
            rr = int(rep_idx[c, 3])
            logger.info(
                "  cluster %d: %d pts, centroid range %.2f m, vel %+.2f m/s, "
                "az %.3f rad, el %.3f rad; rep range %.2f m, vel %+.2f m/s, "
                "az %.3f rad, el %.3f rad",
                c,
                n,
                centroid_r,
                centroid_v,
                centroid_az,
                centroid_el,
                float(ranges[rr]),
                float(velocities[dd]),
                self.az_grid[az_bin],
                self.el_grid[el_bin],
            )

        if self.cadence is not None:
            self.cadence.on_sink_done()


class TrailingIdxColumnsOp(Operator):
    """Slice trailing (doppler_bin, range_bin) from dense 4-column idx."""

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("idx")
        spec.output("idx")

    def compute(self, op_input, op_output, context) -> None:
        idx = port_tensor(op_input.receive("idx"), "idx")
        host = tensor_to_numpy(idx).astype(np.int64, copy=False)
        if host.ndim != 2 or host.shape[1] < 2:
            raise RuntimeError("TrailingIdxColumnsOp: expected (num_points, >= 2) int64 idx")
        trimmed = host[:, -2:].copy()
        op_output.emit({"idx": cp.asarray(trimmed)}, "idx")
