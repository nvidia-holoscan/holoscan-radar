#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <holohub-wrapper>" >&2
    exit 2
fi

wrapper="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT
unset HOLOSCAN_CLI_SOURCE

fake_python="$test_dir/python"
cat > "$fake_python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Interpreter version probe.
if [[ $1 == "-c" ]]; then
    if [[ -n ${TEST_PYTHON_TOO_OLD:-} ]]; then
        exit 1
    fi
    exit 0
fi

if [[ $1 == "-m" && $2 == "holoscan_cli" ]]; then
    printf '%s\n' "$HOLOSCAN_CLI_PYTHON_BIN" > "$TEST_CLI_PYTHON_FILE"
    exit 0
fi

if [[ $1 == "-m" && $2 == "pip" && $3 == "--version" ]]; then
    exit 0
fi

if [[ $1 == "-m" && $2 == "pip" && $3 == "install" ]]; then
    printf '%s\n' "$@" > "$TEST_ARGS_FILE"
    exit 0
fi

echo "Unexpected fake Python invocation: $*" >&2
exit 1
EOF
chmod +x "$fake_python"

export HOLOSCAN_CLI_INSTALL_ARGS='--no-index --find-links *'
export TEST_ARGS_FILE="$test_dir/pip-args"
export TEST_CLI_PYTHON_FILE="$test_dir/cli-python"

# Pip options pass through without pathname expansion, ahead of the shared pin.
HOLOSCAN_CLI_PYTHON_BIN="$fake_python" "$wrapper" --help > /dev/null
diff -u <(printf '%s\n' \
    -m pip install --quiet --disable-pip-version-check --no-index --find-links '*' \
    -r "$(dirname "$wrapper")/requirements-cli.txt") "$TEST_ARGS_FILE"

# A relative interpreter reaches the CLI as an absolute path.
(cd "$test_dir" && HOLOSCAN_CLI_PYTHON_BIN=./python "$wrapper" --help > /dev/null)
diff -u <(printf '%s\n' "$fake_python") "$TEST_CLI_PYTHON_FILE"

# An interpreter older than Python 3.11 is rejected before pip runs.
rm -f "$TEST_ARGS_FILE"
if HOLOSCAN_CLI_PYTHON_BIN="$fake_python" TEST_PYTHON_TOO_OLD=1 \
    "$wrapper" --help > /dev/null 2> "$test_dir/stderr"; then
    echo "Expected the wrapper to reject Python older than 3.11" >&2
    exit 1
fi
grep -q 'requires Python 3.11' "$test_dir/stderr"
[[ ! -e $TEST_ARGS_FILE ]]
