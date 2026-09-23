# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the shared opening structure of public operator READMEs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OPERATORS_ROOT = REPO_ROOT / "operators"
REQUIRED_HEADINGS = (
    "Purpose",
    "Responsibilities",
    "When to use",
    "Ports",
    "Construction",
    "Parameters",
)
REQUIRED_TABLE_HEADERS = {
    "Ports": (
        "| Kind | Port | Payload type / dtype | Shape | Timing / optionality | " "Responsibility |"
    ),
    "Construction": "| Item | Type | Requirement | Responsibility |",
    "Parameters": "| Parameter | Type | Default | Mutability | Responsibility |",
}
HEADING_PATTERN = re.compile(r"^## (.+)$", re.MULTILINE)


def public_operator_directories() -> list[Path]:
    """Return metadata-backed public operator directories."""
    return sorted(
        directory
        for directory in OPERATORS_ROOT.glob("*/*_op")
        if (directory / "metadata.json").is_file()
    )


def check_readme(directory: Path) -> list[str]:
    """Return structural errors for one operator README."""
    readme = directory / "README.md"
    relative = readme.relative_to(REPO_ROOT)
    if not readme.is_file():
        return [f"{relative}: missing README.md"]

    text = readme.read_text(encoding="utf-8")
    errors: list[str] = []
    if not text.startswith("# "):
        errors.append(f"{relative}: first line must be an H1 operator title")

    matches = list(HEADING_PATTERN.finditer(text))
    headings = [match.group(1) for match in matches]
    if tuple(headings[: len(REQUIRED_HEADINGS)]) != REQUIRED_HEADINGS:
        errors.append(
            f"{relative}: first six H2 headings must be " + " -> ".join(REQUIRED_HEADINGS)
        )

    for heading in REQUIRED_HEADINGS:
        count = headings.count(heading)
        if count != 1:
            errors.append(f"{relative}: expected exactly one ## {heading}, found {count}")

    for heading, table_header in REQUIRED_TABLE_HEADERS.items():
        if heading not in headings:
            continue
        index = headings.index(heading)
        start = matches[index].end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if table_header not in text[start:end]:
            errors.append(f"{relative}: ## {heading} must contain table header {table_header}")

    if "When to use" in headings:
        index = headings.index("When to use")
        start = matches[index].end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end]
        prose = re.sub(r"```.*?```", "", section, flags=re.DOTALL)
        prose = " ".join(line.strip() for line in prose.splitlines() if line.strip())
        if len(prose) < 100:
            errors.append(
                f"{relative}: ## When to use needs substantive prose in addition "
                "to any placement diagram"
            )
    return errors


def main() -> int:
    """Check every public operator and print actionable failures."""
    directories = public_operator_directories()
    errors = [error for directory in directories for error in check_readme(directory)]
    if errors:
        print("Operator README structure check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Operator README structure check passed ({len(directories)} operators).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
