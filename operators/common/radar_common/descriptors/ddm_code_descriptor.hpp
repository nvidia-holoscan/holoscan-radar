// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace holoscan::radar {

// Doppler-division multiplexing layout for TDMDenseCompensationOp / DDMDenseDemultiplexOp (DDM
// mode). Lands in radar_common as one of the seven shared descriptor types.
struct DDMCodeDescriptor {
  int64_t num_tx{0};
  std::vector<std::complex<double>> phase_codes{};
  std::vector<double> doppler_offsets{};
  int64_t guard_bands{0};

  void validate() const;
};

}  // namespace holoscan::radar
