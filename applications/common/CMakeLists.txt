# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared application-support library, pulled in by the FMCW example apps. Not
# gated behind an APP_/OP_ option — it is header-only (INTERFACE) with no build
# cost, and every application here links it.
add_subdirectory(radar_app)
