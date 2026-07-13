// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/tensor/tensor_utils.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::radar {

std::string dldatatype_name(DLDataType dtype) {
  std::ostringstream name;
  switch (dtype.code) {
    case kDLFloat: name << "float"; break;
    case kDLComplex: name << "complex"; break;
    case kDLInt: name << "int"; break;
    case kDLUInt: name << "uint"; break;
    case kDLBfloat: name << "bfloat"; break;
    case kDLBool: name << "bool"; break;
    default: name << "code" << static_cast<int>(dtype.code); break;
  }
  name << static_cast<int>(dtype.bits);
  if (dtype.lanes != 1) {
    name << "x" << static_cast<int>(dtype.lanes);
  }
  return name.str();
}

BatchedShape collapse_batch(const holoscan::Tensor& tensor,
                            const char* port_name) {
  const auto shape = tensor.shape();
  if (shape.size() < 2) {
    std::ostringstream msg;
    msg << port_name << ": expected a tensor of rank >= 2 (..., rows, cols)"
        << ", received rank " << shape.size();
    throw std::invalid_argument(msg.str());
  }
  if (!tensor.is_contiguous()) {
    std::ostringstream msg;
    msg << port_name
        << ": strided (non-contiguous) inputs are not supported; provide a "
           "compact row-major tensor";
    throw std::invalid_argument(msg.str());
  }

  BatchedShape out;
  out.leading.assign(shape.begin(), shape.end() - 2);
  out.batch = 1;
  for (int64_t dim : out.leading) {
    if (dim <= 0) {
      std::ostringstream msg;
      msg << port_name << ": zero/negative-sized batch axis in input shape";
      throw std::invalid_argument(msg.str());
    }
    out.batch *= dim;
  }
  out.rows = shape[shape.size() - 2];
  out.cols = shape[shape.size() - 1];
  if (out.rows <= 0 || out.cols <= 0) {
    std::ostringstream msg;
    msg << port_name << ": trailing (rows, cols) axes must be non-empty";
    throw std::invalid_argument(msg.str());
  }
  return out;
}

std::vector<int64_t> full_rank_shape(const BatchedShape& shape, int64_t rows,
                                     int64_t cols) {
  std::vector<int64_t> out = shape.leading;
  out.push_back(rows);
  out.push_back(cols);
  return out;
}

bool is_cuda_accessible(const holoscan::Tensor& tensor) {
  switch (tensor.device().device_type) {
    case kDLCUDA:
    case kDLCUDAManaged:
    case kDLCUDAHost:
      return true;
    default:
      return false;
  }
}

template <typename T>
InputPrecision classify_input_dtype(const holoscan::Tensor& tensor,
                                    const char* port_name, bool allow_real) {
  const DLDataType dtype = tensor.dtype();
  const char* precision_name = sizeof(T) == 4 ? "float" : "double";
  if (dldatatype_equals(dtype, complex_dtype<T>())) {
    return InputPrecision::kComplex;
  }
  if (dldatatype_equals(dtype, real_dtype<T>())) {
    if (allow_real) {
      return InputPrecision::kReal;
    }
    std::ostringstream msg;
    msg << port_name << ": real-valued input " << dldatatype_name(dtype)
        << " is not supported here; provide " << dldatatype_name(
               complex_dtype<T>());
    throw std::invalid_argument(msg.str());
  }
  std::ostringstream msg;
  msg << port_name << ": received dtype " << dldatatype_name(dtype)
      << " does not match this operator's " << precision_name
      << "-precision instantiation (expected "
      << dldatatype_name(complex_dtype<T>());
  if (allow_real) {
    msg << " or " << dldatatype_name(real_dtype<T>());
  }
  msg << ")";
  throw std::invalid_argument(msg.str());
}

template InputPrecision classify_input_dtype<float>(const holoscan::Tensor&,
                                                    const char*, bool);
template InputPrecision classify_input_dtype<double>(const holoscan::Tensor&,
                                                     const char*, bool);

}  // namespace holoscan::radar
