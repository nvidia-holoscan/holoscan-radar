// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/tensor_dump_op/npy_writer.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

namespace holoscan::radar::npy {

namespace {

// The byte-order character. Every machine this module runs on is
// little-endian -- invariant 8 makes a CUDA-capable NVIDIA GPU a hard
// dependency, and CUDA ships for x86_64, aarch64 and ppc64le only -- so the
// descriptor is written little-endian unconditionally rather than detected.
// The same precondition the CPHD parser asserts, for the same reason.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "npy_writer writes little-endian type descriptors; this module "
              "requires a CUDA-capable host, all of which are little-endian");

char order_char(uint8_t bits) {
  // NumPy spells single-byte types '|' (not applicable) rather than '<',
  // and numpy.lib.format rejects '<u1'.
  return bits <= 8 ? '|' : '<';
}

}  // namespace

std::string dtype_descr(DLDataType dtype) {
  std::ostringstream unsupported;
  if (dtype.lanes != 1) {
    unsupported << "npy: dtype has " << static_cast<int>(dtype.lanes)
                << " lanes; .npy element types are scalar";
    throw std::invalid_argument(unsupported.str());
  }
  const char order = order_char(dtype.bits);
  const int bytes = static_cast<int>(dtype.bits) / 8;
  std::ostringstream descr;
  switch (dtype.code) {
    case kDLFloat:
      if (dtype.bits != 16 && dtype.bits != 32 && dtype.bits != 64) {
        break;
      }
      descr << order << 'f' << bytes;
      return descr.str();
    case kDLComplex:
      if (dtype.bits != 64 && dtype.bits != 128) {
        break;
      }
      descr << order << 'c' << bytes;
      return descr.str();
    case kDLInt:
      if (dtype.bits != 8 && dtype.bits != 16 && dtype.bits != 32 &&
          dtype.bits != 64) {
        break;
      }
      descr << order << 'i' << bytes;
      return descr.str();
    case kDLUInt:
      if (dtype.bits != 8 && dtype.bits != 16 && dtype.bits != 32 &&
          dtype.bits != 64) {
        break;
      }
      descr << order << 'u' << bytes;
      return descr.str();
    case kDLBool:
      if (dtype.bits != 8) {
        break;
      }
      return "|b1";
    default:
      break;
  }
  unsupported << "npy: dtype code " << static_cast<int>(dtype.code) << " with "
              << static_cast<int>(dtype.bits)
              << " bits has no .npy type string. Dump it with format: raw, "
                 "which writes the element bytes without a header.";
  throw std::invalid_argument(unsupported.str());
}

std::string header(DLDataType dtype, const std::vector<int64_t>& shape) {
  std::ostringstream dict;
  dict << "{'descr': '" << dtype_descr(dtype)
       << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    dict << shape[i];
    // A one-element tuple needs its trailing comma or Python parses it as a
    // parenthesized integer, and numpy.lib.format evaluates this literally.
    if (i + 1 < shape.size() || shape.size() == 1) {
      dict << ", ";
    }
  }
  dict << "), }";

  std::string text = dict.str();
  // Magic (6) + version (2) + header length (2) = 10 bytes precede the text,
  // and the format requires the data that follows to start 64-byte aligned.
  constexpr size_t kPreamble = 10;
  constexpr size_t kAlign = 64;
  const size_t unpadded = kPreamble + text.size() + 1;  // +1 for the newline
  const size_t padded = ((unpadded + kAlign - 1) / kAlign) * kAlign;
  text.append(padded - unpadded, ' ');
  text.push_back('\n');

  const size_t length = text.size();
  if (length > 0xFFFF) {
    throw std::invalid_argument(
        "npy: header exceeds the 65535 bytes .npy v1.0 allows; the shape has "
        "too many dimensions to record");
  }

  std::string out;
  out.append("\x93NUMPY", 6);
  out.push_back('\x01');  // version major
  out.push_back('\x00');  // version minor
  out.push_back(static_cast<char>(length & 0xFF));
  out.push_back(static_cast<char>((length >> 8) & 0xFF));
  out.append(text);
  return out;
}

void write(const std::string& path, DLDataType dtype,
           const std::vector<int64_t>& shape, const void* data,
           size_t nbytes) {
  const std::string head = header(dtype, shape);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("npy: cannot open '" + path + "' for writing");
  }
  out.write(head.data(), static_cast<std::streamsize>(head.size()));
  if (nbytes > 0) {
    out.write(static_cast<const char*>(data),
              static_cast<std::streamsize>(nbytes));
  }
  out.close();
  if (!out) {
    throw std::runtime_error("npy: failed writing '" + path +
                             "' (disk full or permission denied?)");
  }
}

void write_raw(const std::string& path, const void* data, size_t nbytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("npy: cannot open '" + path + "' for writing");
  }
  if (nbytes > 0) {
    out.write(static_cast<const char*>(data),
              static_cast<std::streamsize>(nbytes));
  }
  out.close();
  if (!out) {
    throw std::runtime_error("npy: failed writing '" + path +
                             "' (disk full or permission denied?)");
  }
}

}  // namespace holoscan::radar::npy
