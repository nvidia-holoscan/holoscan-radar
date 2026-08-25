// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The D22 boundary announcement: the key the reader stamps and the
// interim string values it writes. One named constant for the key is
// the standing R10 instruction — promotion to the module level is then
// a change to this file, not a hunt. See docs/planning/SAR/TEAM_BLOCKED.md
// (R10) and operators/io/cphd_reader_op/OPEN_ITEMS.md item 11.
//
// This is not a new shared descriptor (invariant 2) and it does not
// close R10 or item 11. The contract's uint8 StreamBoundary enum waits
// on ApertureManagerOp; these strings are what CPHDReaderOp writes
// today and what the mode-1 control operator reads. Changing a value
// is a breaking change between those two sites.

namespace holoscan::radar {

// Interim SAR-tier key. R10 has not ruled on a module-wide prefix.
inline constexpr const char* kStreamBoundaryKey =
    "holoscan.radar.sar.stream_boundary";

// Interim string encoding. Not the contract's uint8 enum (item 11).
inline constexpr const char* kStreamBoundaryEndOfStream = "end_of_stream";
inline constexpr const char* kStreamBoundaryCollection =
    "collection_boundary";

}  // namespace holoscan::radar
