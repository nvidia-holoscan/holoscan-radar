# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Add the locally built SDK install tree to its build image.
# Delete this entire file before the HSDK 5 public release.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HSDK_INSTALL_DIR

COPY ${HSDK_INSTALL_DIR} /opt/nvidia/holoscan

ENV CMAKE_PREFIX_PATH=/opt/nvidia/holoscan
ENV HOLOSCAN_LIB_PATH=/opt/nvidia/holoscan/lib
ENV PYTHONPATH=/opt/nvidia/holoscan/python/lib
ENV PATH=${PATH}:/opt/nvidia/holoscan/bin

RUN echo "/opt/nvidia/holoscan/lib" > /etc/ld.so.conf.d/holoscan.conf \
    && ldconfig
