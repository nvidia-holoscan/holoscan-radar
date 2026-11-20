// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "radar_common/descriptors/ddm_code_descriptor.hpp"

namespace holoscan::radar {

// Build deterministic nonlinear DDM descriptors on the host. These helpers
// define code math only; applications retain ownership of preset naming and
// must provide the same descriptor to the transmitter/source and demultiplexer.
// Reject non-positive dimensions and unrepresentable element/byte counts
// before allocation or root search. No resource budget is imposed.
DDMCodeDescriptor make_zadoff_chu_ddm_code(int64_t num_tx, int64_t num_chirps);

DDMCodeDescriptor make_sinusoidal_ddm_stress_code(int64_t num_tx, int64_t num_chirps);

}  // namespace holoscan::radar
