# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED TRAITS_FILE)
    message(FATAL_ERROR "TRAITS_FILE is required")
endif()

file(READ "${TRAITS_FILE}" _traits)
if(_traits MATCHES "HOLOSCAN_RADAR_WIRE_CREATE_TENSOR")
    return()
endif()

set(_original
    [=[inline flatbuffers::Offset<holoscan::schema::Tensor> CreateTensor(
    flatbuffers::FlatBufferBuilder& builder, const holoscan::Tensor* tensor,
    const flatbuffers::rehasher_function_t* rehasher = nullptr) {
  return holoscan::schema::CreateTensor(builder, tensor, rehasher);
}]=])
set(_guarded
    [=[#ifndef HOLOSCAN_RADAR_WIRE_CREATE_TENSOR
#define HOLOSCAN_RADAR_WIRE_CREATE_TENSOR
inline flatbuffers::Offset<holoscan::schema::Tensor> CreateTensor(
    flatbuffers::FlatBufferBuilder& builder, const holoscan::Tensor* tensor,
    const flatbuffers::rehasher_function_t* rehasher = nullptr) {
  return holoscan::schema::CreateTensor(builder, tensor, rehasher);
}
#endif]=])

string(FIND "${_traits}" "${_original}" _found)
if(_found EQUAL -1)
    message(FATAL_ERROR
        "CreateTensor forwarding block not found in ${TRAITS_FILE}")
endif()
string(REPLACE "${_original}" "${_guarded}" _traits "${_traits}")
file(WRITE "${TRAITS_FILE}" "${_traits}")
