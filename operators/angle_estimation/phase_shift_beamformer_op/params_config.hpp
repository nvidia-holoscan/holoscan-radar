// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include <holoscan/radar/angle_estimation/phase_shift_beamformer_op.hpp>
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using BeamformerParamsCodec =
    params::DefaultedParamSchemaCodec<radar::PhaseShiftBeamformerParams>;  // NOLINT

template <>
struct ConfigCodec<radar::PhaseShiftBeamformerParams> : BeamformerParamsCodec {};  // NOLINT

}  // namespace holoscan::config
