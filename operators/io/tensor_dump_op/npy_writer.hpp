// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace holoscan::radar::npy {

// Minimal NumPy .npy v1.0 writer, host-side and free of Holoscan and CUDA so
// the format can be tested without a graph. Only what TensorDumpOp needs: one
// C-contiguous array per file, no pickled objects, no v2/v3 headers (which
// exist for headers above 64 KiB; header() enforces that bound rather than
// assuming a shape tuple stays under it).

// The NumPy type string for a DLPack dtype, e.g. "<f4" or "|u1". Throws
// std::invalid_argument for a dtype .npy cannot express, naming it: bfloat16
// has no NumPy equivalent, and a multi-lane (vector) dtype is not an array
// element type.
std::string dtype_descr(DLDataType dtype);

// Serialize the .npy header for `dtype` and `shape`, padded so the data that
// follows begins on a 64-byte boundary as the format requires. Throws
// std::invalid_argument if the shape has enough dimensions that the header
// outgrows the 65535 bytes v1.0's two-byte length field can express, rather
// than writing a file whose length field has silently wrapped.
std::string header(DLDataType dtype, const std::vector<int64_t>& shape);

// Write `nbytes` of C-contiguous element data as a .npy file. The caller has
// already brought the bytes to host memory; this function does no device work.
// Throws std::runtime_error if the file cannot be opened or written.
void write(const std::string& path, DLDataType dtype,
           const std::vector<int64_t>& shape, const void* data, size_t nbytes);

// Write the element bytes with no header at all. Shape and dtype are lost,
// which is the format's whole trade -- the caller must know them already.
void write_raw(const std::string& path, const void* data, size_t nbytes);

}  // namespace holoscan::radar::npy
