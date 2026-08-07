# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import / construct smoke for all landed operator bindings."""

import pytest

holoscan = pytest.importorskip("holoscan")


def test_import_all_operators_and_descriptors():
    import holoscan.radar as r

    for name in (
        "ClutterFilterOp",
        "FMCWRangeProcessorOp",
        "FMCWDopplerProcessorOp",
        "FMCWRangeDopplerProcessorOp",
        "CFARDetectorOp",
        "FeatureAssemblyOp",
        "DetectionReductionOp",
        "ClusterDBSCANOp",
        "SnapshotGatherOp",
        "BeamscanEstimatorOp",
        "PhaseShiftBeamformerOp",
        "MVDREstimatorOp",
        "MIMODeinterleaveOp",
        "MIMOCoherentCompensationOp",
        "VirtualArrayAssemblyOp",
        "CfarVariant",
        "AssemblyFeature",
        "WindowSpec",
        "ArrayGeometry",
        "AngleConvention",
        "FMCWWaveformDescriptor",
        "TXSchedule",
        "DDMCodeDescriptor",
        "NullSinkOp",
        "DetectionReportOp",
        "ClusterReportOp",
        "RdMapToImageOp",
        "ClusterOverlayOp",
        "RollingPlotOp",
        "BoresightViewOp",
        "RdChannelSliceOp",
        "gui_available",
        "make_turbo_lut",
        "cluster_palette",
    ):
        assert hasattr(r, name), name
        if name.endswith("Op") and name not in (
            "NullSinkOp",
            "DetectionReportOp",
            "ClusterReportOp",
            "RdMapToImageOp",
            "ClusterOverlayOp",
            "RollingPlotOp",
            "BoresightViewOp",
            "RdChannelSliceOp",
        ):
            assert getattr(r, name) is getattr(r, f"{name}F32")
            assert getattr(r, f"{name}F64") is not getattr(r, f"{name}F32")


def test_construct_clutter_filter_op():
    from holoscan.core import Application
    from holoscan.radar import ClutterFilterMethod, ClutterFilterOp

    class App(Application):
        def compose(self):
            self.cf = ClutterFilterOp(
                self,
                method=ClutterFilterMethod.SINGLE_DELAY,
                name="cf",
            )

    app = App()
    app.compose()
    assert app.cf.name == "cf"


def test_construct_cfar_and_feature_assembly():
    from holoscan.core import Application
    from holoscan.radar import (
        AssemblyFeature,
        CFARDetectorOp,
        CfarVariant,
        FeatureAssemblyOp,
    )

    class App(Application):
        def compose(self):
            self.cfar = CFARDetectorOp(self, cfar_variant=CfarVariant.CA, name="cfar")
            self.fa = FeatureAssemblyOp(
                self,
                features=[AssemblyFeature.RANGE, AssemblyFeature.VELOCITY],
                name="fa",
            )

    app = App()
    app.compose()
    assert app.cfar.name == "cfar"
    assert app.fa.name == "fa"
