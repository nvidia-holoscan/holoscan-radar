# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HSDK 5 FMCW Graph operator facade."""

from collections.abc import Mapping
from typing import Any, Protocol

from holoscan import Graph
from holoscan.operators import OperatorHandle

class RadarOperator(Protocol):
    registry_name: str
    def __call__(
        self, graph: Graph, name: str, *, extents: Mapping[str, int], **params: Any
    ) -> OperatorHandle: ...

BeamscanEstimatorOpF32: RadarOperator
BeamscanEstimatorOpF64: RadarOperator
CFARDetectorOpF32: RadarOperator
CFARDetectorOpF64: RadarOperator
ClusterDBSCANOpF32: RadarOperator
ClusterDBSCANOpF64: RadarOperator
ClutterFilterOpF32: RadarOperator
ClutterFilterOpF64: RadarOperator
DDMDenseDemultiplexOpF32: RadarOperator
DDMDenseDemultiplexOpF64: RadarOperator
DDMPowerFoldOpF32: RadarOperator
DDMPowerFoldOpF64: RadarOperator
DDMSparseDemultiplexOpF32: RadarOperator
DDMSparseDemultiplexOpF64: RadarOperator
DetectionReductionOpF32: RadarOperator
DetectionReductionOpF64: RadarOperator
DopplerNotchOpF32: RadarOperator
DopplerNotchOpF64: RadarOperator
ESPRITEstimatorOpF32: RadarOperator
ESPRITEstimatorOpF64: RadarOperator
FMCWDopplerProcessorOpF32: RadarOperator
FMCWDopplerProcessorOpF64: RadarOperator
FMCWRangeDopplerProcessorOpF32: RadarOperator
FMCWRangeDopplerProcessorOpF64: RadarOperator
FMCWRangeProcessorOpF32: RadarOperator
FMCWRangeProcessorOpF64: RadarOperator
FeatureAssemblyOpF32: RadarOperator
FeatureAssemblyOpF64: RadarOperator
MIMODeinterleaveOpF32: RadarOperator
MIMODeinterleaveOpF64: RadarOperator
MUSICEstimatorOpF32: RadarOperator
MUSICEstimatorOpF64: RadarOperator
MVDREstimatorOpF32: RadarOperator
MVDREstimatorOpF64: RadarOperator
MultiSourceExpansionOpF32: RadarOperator
MultiSourceExpansionOpF64: RadarOperator
PhaseShiftBeamformerOpF32: RadarOperator
PhaseShiftBeamformerOpF64: RadarOperator
RankOneResidualScreenOpF32: RadarOperator
RankOneResidualScreenOpF64: RadarOperator
RootMUSICEstimatorOpF32: RadarOperator
RootMUSICEstimatorOpF64: RadarOperator
SnapshotGatherOpF32: RadarOperator
SnapshotGatherOpF64: RadarOperator
SubarraySelectionOpF32: RadarOperator
SubarraySelectionOpF64: RadarOperator
TDMDenseCompensationOpF32: RadarOperator
TDMDenseCompensationOpF64: RadarOperator
TDMSparseCompensationOpF32: RadarOperator
TDMSparseCompensationOpF64: RadarOperator
VirtualArrayAssemblyOpF32: RadarOperator
VirtualArrayAssemblyOpF64: RadarOperator
