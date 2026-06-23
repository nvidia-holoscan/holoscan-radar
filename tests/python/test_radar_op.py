# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Functional tests for RadarOp.
# TODO: extend test_placeholder() with real pipeline coverage.
# See holoscan-example-module/tests/python/ for worked examples.

import importlib
import inspect

import pytest

# Skip the whole module only when the Holoscan SDK itself is unavailable
# (e.g. CUDA missing outside the dev container). We deliberately do NOT
# importorskip("holoscan.radar") at module level — that
# would mask an actual build/import failure of *our* module as a "Skipped"
# result, and CTest would silently report success-with-zero-tests (exit code 5).
pytest.importorskip("holoscan", reason="holoscan SDK not installed", exc_type=ImportError)


EXPECTED_OPERATORS = (
    "RadarOp",
)


@pytest.mark.parametrize("operator_name", EXPECTED_OPERATORS)
def test_operator_is_importable(operator_name):
    module = importlib.import_module("holoscan.radar")

    assert hasattr(module, operator_name), (
        f"holoscan.radar does not expose {operator_name}; "
        f"available names: {sorted(n for n in dir(module) if not n.startswith('_'))}"
    )
    operator_cls = getattr(module, operator_name)
    assert inspect.isclass(operator_cls), (
        f"holoscan.radar.{operator_name} is not a class "
        f"(got {type(operator_cls)!r})"
    )


def test_placeholder():
    # TODO: build a minimal Application, call app.run(), assert on results.
    pass
