#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

workspace="${1:?workspace folder required}"
data_dir="${HOLOSCAN_RADAR_DATA_DIR:-${workspace}/data}"
mkdir -p "${data_dir}"

# Record the host user's UID/GID so Compose builds the in-container `dev` user
# with matching ids, keeping the bind-mounted source writable. Compose auto-loads
# .env from its project directory (this .devcontainer/ folder).
env_file="${workspace}/.devcontainer/.env"
{
  echo "DEV_UID=$(id -u)"
  echo "DEV_GID=$(id -g)"
} >"${env_file}"
