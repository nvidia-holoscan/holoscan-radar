#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Checks how ./holohub selects the CLI interpreter, using a fake Python.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <holohub-wrapper>" >&2
    exit 2
fi

wrapper="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
requirements="$(dirname "$wrapper")/requirements-cli.txt"
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT
unset VIRTUAL_ENV HOLOSCAN_CLI_PYTHON_BIN HOLOSCAN_CLI_VENV
export TEST_LOG="$test_dir/log"

# FAKE_CLI=1: the CLI imports. FAKE_OLD=1: Python older than 3.11.
fake_python="$test_dir/python"
cat > "$fake_python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$1 ${2:-}" in
    "-c import holoscan_cli.project_context") [[ -n ${FAKE_CLI:-} ]] ;;
    "-c "*) [[ -z ${FAKE_OLD:-} ]] ;;
    "-m venv") mkdir -p "$3/bin" && cp "$0" "$3/bin/python" ;;
    "-m pip" | "-m holoscan_cli") echo "$0 $* ${HOLOSCAN_CLI_PYTHON_BIN:-}" >> "$TEST_LOG" ;;
    *) echo "Unexpected fake Python invocation: $*" >&2; exit 1 ;;
esac
EOF
chmod +x "$fake_python"

# An interpreter with the CLI is used as-is and exported as an absolute path.
(cd "$test_dir" && FAKE_CLI=1 HOLOSCAN_CLI_PYTHON_BIN=./python "$wrapper" --help)
diff -u <(echo "$fake_python -m holoscan_cli --help $fake_python") "$TEST_LOG"

# An explicit interpreter without the CLI is reported, not modified.
rm -f "$TEST_LOG"
if HOLOSCAN_CLI_PYTHON_BIN="$fake_python" "$wrapper" --help 2> "$test_dir/err"; then
    echo "Expected a missing CLI to fail" >&2
    exit 1
fi
grep -q 'pip install -r' "$test_dir/err"
[[ ! -e $TEST_LOG ]]

# Otherwise a managed venv is created and the pin installed only once.
bin="$test_dir/bin"
mkdir "$bin"
ln -s "$fake_python" "$bin/python3"
venv="$test_dir/venv"
for _ in 1 2; do
    PATH="$bin:$PATH" HOLOSCAN_CLI_VENV="$venv" "$wrapper" --help
done
diff -u <(printf '%s\n' \
    "$venv/bin/python -m pip install --quiet --disable-pip-version-check -r $requirements " \
    "$venv/bin/python -m holoscan_cli --help $venv/bin/python" \
    "$venv/bin/python -m pip --version " \
    "$venv/bin/python -m holoscan_cli --help $venv/bin/python") "$TEST_LOG"

# Without Python 3.11+, the wrapper fails before creating a venv.
old_bin="$test_dir/old-bin"
mkdir "$old_bin"
ln -s "$fake_python" "$old_bin/python3"
for tool in bash basename cmp cp dirname; do
    ln -s "$(type -P "$tool")" "$old_bin/$tool"
done
if FAKE_OLD=1 PATH="$old_bin" HOLOSCAN_CLI_VENV="$test_dir/old-venv" \
    "$wrapper" --help 2> "$test_dir/err"; then
    echo "Expected Python older than 3.11 to fail" >&2
    exit 1
fi
grep -q 'needs Python 3.11' "$test_dir/err"
[[ ! -e $test_dir/old-venv ]]
