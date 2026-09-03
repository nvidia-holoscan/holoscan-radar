#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <holohub-wrapper>" >&2
    exit 2
fi

wrapper=$1
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT

fake_python="$test_dir/python"
cat > "$fake_python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ $1 == "-m" && $2 == "holoscan_cli" && $3 == "--help" ]]; then
    if [[ -f $TEST_STATE_FILE ]]; then
        echo build
    fi
    exit 0
fi

if [[ $1 == "-m" && $2 == "pip" && $3 == "--version" ]]; then
    exit 0
fi

if [[ $1 == "-m" && $2 == "pip" && $3 == "install" ]]; then
    printf '%s\n' "$@" > "$TEST_ARGS_FILE"
    touch "$TEST_STATE_FILE"
    exit 0
fi

echo "Unexpected fake Python invocation: $*" >&2
exit 1
EOF
chmod +x "$fake_python"

export HOLOSCAN_CLI_PYTHON_BIN="$fake_python"
export HOLOSCAN_CLI_INSTALL_ARGS='--pre --extra-index-url https://pypi.nvidia.com holoscan-cli>4.2.0 *'
export TEST_ARGS_FILE="$test_dir/pip-args"
export TEST_STATE_FILE="$test_dir/cli-installed"

"$wrapper" --help > /dev/null

diff -u <(printf '%s\n' \
    -m pip install --pre --extra-index-url https://pypi.nvidia.com \
    'holoscan-cli>4.2.0' '*') "$TEST_ARGS_FILE"
