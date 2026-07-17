// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device-side conventional (Bartlett) steering-weight builder.
//
// The per-element phase term taper[m]·exp(±j·k·(pₘ·dir_b)) — the
// O(num_beams·num_channels) work that dominates a large steering-table build —
// runs in a kernel here instead of a serial host loop. Only the per-beam
// direction vectors + validity (O(num_beams), the convention-dependent part)
// and the amplitude taper (O(num_channels)) are computed host-side and
// uploaded; that upload is *smaller* than the full complex table it replaces.
//
// The host launch wrapper is declared here and defined (with float/double
// explicit instantiations) in steering_weights_device.cu, compiled into
// holoscan::radar_common_steering. Link that library; do not recompile the .cu
// into another target (ODR). radar_common itself stays CUDA-free.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"

namespace holoscan::radar {

// How the 2-D (az × el) scan grid flattens onto the weight table's beam axis.
enum class BeamOrder {
  kElMajor,  // beam = ie * num_az + ia (az fastest) — Beamscan reduce order
  kAzMajor,  // beam = ia * num_el + ie (el fastest) — dense RDA-cube order
};

// Memory layout of the (num_beams × num_channels) weight table.
enum class SteeringLayout {
  kBeamMajor,     // shape (num_beams, num_channels) — dense beamformer GEMM A
  kChannelMajor,  // shape (num_channels, num_beams) — Beamscan GEMM Wᶜ
};

// Build the steering weights directly on the device, in the requested beam
// order / layout, optionally conjugated (both angle operators store conj(w) so
// the hot path is a plain matmul). Invalid beams (direction-cosine invisible
// region) get zero weights. Returns a device holoscan::Tensor of complex<T>.
// All work is enqueued on `stream`; scratch is stream-ordered.
template <typename T>
std::shared_ptr<holoscan::Tensor> build_steering_weights_device(
    const ArrayGeometry& geometry, const AngleConvention& convention,
    const std::vector<double>& az_angles, const std::vector<double>& el_angles,
    const WindowSpec& array_taper, BeamOrder beam_order, SteeringLayout layout,
    bool conjugate, cudaStream_t stream);

}  // namespace holoscan::radar
