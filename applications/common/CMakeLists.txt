# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared application-support library, pulled in by the FMCW example apps. Not
# gated behind an APP_/OP_ option. Its base target is header-only; optional
# visualization components may add compiled CUDA targets.
add_subdirectory(radar_app)
