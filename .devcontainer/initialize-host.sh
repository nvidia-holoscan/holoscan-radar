#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

workspace="${1:?workspace folder required}"
data_dir="${HOLOSCAN_RADAR_DATA_DIR:-${workspace}/data}"
mkdir -p "${data_dir}"
