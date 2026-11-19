# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for CLI error presentation, independent of Graph execution."""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Load only the CLI: these checks exercise formatting and dispatch, not native operators.
SOURCE = Path(__file__).resolve().parents[2] / "python/holoscan_radar/applications/cli.py"
SPEC = importlib.util.spec_from_file_location("radar_application_cli", SOURCE)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)

DIAGNOSTIC = (
    "CPHD input file not found:\n  sar/missing.cphd\n\n"
    "Next step: prepare the data\nRun on the host, from the repository root:\n\n"
    "  uv run tools/sar_data/pull_cphd.py --config 'custom.yaml'\n\n"
    "Add --list to preview without downloading.\n\n"
    "More information\n  tools/sar_data/README.md\n  applications/sar/README.md"
)


class ApplicationCliErrorTest(unittest.TestCase):
    def render(self, *, tty, environment):
        output = io.StringIO()
        with (
            mock.patch.object(output, "isatty", return_value=tty),
            mock.patch.dict(os.environ, environment, clear=True),
            contextlib.redirect_stderr(output),
        ):
            cli._print_sar_input_error(RuntimeError(DIAGNOSTIC))
        return output.getvalue()

    def test_redirected_output_keeps_instructions_without_ansi(self):
        self.assertEqual(self.render(tty=False, environment={}), f"Error: {DIAGNOSTIC}\n")

    def test_terminal_styles_error_and_distinct_instruction_headings(self):
        output = self.render(tty=True, environment={"TERM": "xterm-256color"})
        self.assertIn("\033[1;31mError: CPHD input file not found:\033[0m", output)
        self.assertIn("\033[1;36mNext step: prepare the data\033[0m", output)
        self.assertIn("\033[1mMore information\033[0m", output)
        # Keep the actual command unstyled and directly copyable.
        self.assertIn("\n  uv run tools/sar_data/pull_cphd.py --config 'custom.yaml'\n", output)

    def test_no_color_and_dumb_terminal_disable_styling(self):
        for environment in ({"NO_COLOR": ""}, {"NO_COLOR": "1"}, {"TERM": "dumb"}):
            with self.subTest(environment=environment):
                self.assertNotIn("\033[", self.render(tty=True, environment=environment))

    def test_missing_input_returns_failure_with_no_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "custom.yaml"
            config.write_text("{}")
            build = mock.Mock(side_effect=RuntimeError(DIAGNOSTIC))
            output = io.StringIO()
            with (
                mock.patch.object(
                    cli, "import_module", return_value=SimpleNamespace(build_graph=build)
                ),
                contextlib.redirect_stderr(output),
            ):
                self.assertEqual(cli.main(["autofocus", str(config)]), 1)
            self.assertIn("Error: CPHD input file not found:", output.getvalue())
            self.assertIn("Next step: prepare the data", output.getvalue())
            self.assertNotIn("Traceback", output.getvalue())
            build.assert_called_once_with(
                config, cphd_override=None, presenter=None, choose_presenter=False
            )

    def test_unexpected_builder_failure_propagates(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "custom.yaml"
            config.write_text("{}")
            build = mock.Mock(side_effect=RuntimeError("unexpected CUDA failure"))
            with (
                mock.patch.object(
                    cli, "import_module", return_value=SimpleNamespace(build_graph=build)
                ),
                self.assertRaisesRegex(RuntimeError, "unexpected CUDA failure"),
            ):
                cli.main(["autofocus", str(config)])

    def test_execution_errors_are_not_treated_as_configuration_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "custom.yaml"
            config.write_text("{}")
            workflow = SimpleNamespace(run=mock.Mock(side_effect=RuntimeError(DIAGNOSTIC)))
            build = mock.Mock(return_value=workflow)
            with (
                mock.patch.object(
                    cli, "import_module", return_value=SimpleNamespace(build_graph=build)
                ),
                self.assertRaisesRegex(RuntimeError, "CPHD input file not found"),
            ):
                cli.main(["autofocus", str(config)])


if __name__ == "__main__":
    unittest.main()
