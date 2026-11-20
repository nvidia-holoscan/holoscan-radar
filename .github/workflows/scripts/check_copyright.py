"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# This file is modified from the RAPIDS RAFT project which is under the
# Apache 2.0 license.
# (https://github.com/rapidsai/raft/blob/branch-22.08/ci/checks/copyright.py)

import argparse
import datetime
import fnmatch
import itertools
import os
import re
import sys

import gitutils

FilesToCheck = [
    re.compile(r"[.](cmake|cpp|cu|cuh|h|hpp|sh|pxd|py|pyx|yaml)$"),
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"Dockerfile$"),
    re.compile(r"[.]dockerfile$"),
    re.compile(r"CMakeLists_standalone[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"[.]flake8[.]cython$"),
    re.compile(r"meta[.]yaml$"),
]
ExemptFiles = []

MAX_EXCLUDE_PATTERN_LENGTH = 256
MAX_HEADER_BYTES = 64 * 1024
MAX_HEADER_LINES = 100

# Require the full NVIDIA entity; bare "NVIDIA" must not pass.
_NVIDIA_ENTITY = r"NVIDIA CORPORATION & AFFILIATES(?:\. All rights reserved)?"

CheckSPDXWithCopyright = re.compile(
    rf"^(\s*(?:(?:#|//|/\*|\*|<!--)\s*)?)SPDX-FileCopyrightText: Copyright(?: \(c\))? *(\d{{4}})(?:-(\d{{4}}))?,? ({_NVIDIA_ENTITY})\.?",
    re.IGNORECASE,
)
CheckSPDXREUSE = re.compile(
    rf"^(\s*(?:(?:#|//|/\*|\*|<!--)\s*)?)SPDX-FileCopyrightText: *(\d{{4}})(?:-(\d{{4}}))? ({_NVIDIA_ENTITY})",
    re.IGNORECASE,
)
CheckNonSPDX = re.compile(
    rf"^(\s*(?:(?:#|//|/\*|\*|<!--)\s*)?)Copyright(?: \(c\))? *(\d{{4}})(?:-(\d{{4}}))?,? ({_NVIDIA_ENTITY})\.?",
    re.IGNORECASE,
)


def matches_exclude(path, pattern):
    """Match a short shell glob as a path substring."""
    normalized = path.replace(os.sep, "/")
    return fnmatch.fnmatchcase(normalized, f"*{pattern}*")


def check_this_file(f):
    # This check covers things like symlinks which point to files that DONE
    if not (os.path.exists(f)):
        return False
    if gitutils and gitutils.is_file_empty(f):
        return False
    for exempt in ExemptFiles:
        if matches_exclude(f, exempt):
            return False
    return any(checker.search(f) for checker in FilesToCheck)


def get_copyright_years(lines):
    """Return the first copyright year range in bounded header lines."""
    for line_num, line in enumerate(lines, start=1):
        for pattern in [CheckSPDXWithCopyright, CheckSPDXREUSE, CheckNonSPDX]:
            res = pattern.search(line)
            if res:
                start_year = int(res.group(2))
                end_year = int(res.group(3)) if res.group(3) else start_year
                return (start_year, end_year, line_num, line)

    return (None, None, None, None)


def replace_current_year(line, start, end):
    def replace_spdx_with_copyright(match):
        comment_prefix = match.group(1) or ""
        affiliation = match.group(4)
        return f"{comment_prefix}SPDX-FileCopyrightText: Copyright (c) {start}-{end} {affiliation}"

    def replace_spdx_reuse(match):
        comment_prefix = match.group(1) or ""
        affiliation = match.group(4)
        return f"{comment_prefix}SPDX-FileCopyrightText: {start}-{end} {affiliation}"

    def replace_non_spdx(match):
        comment_prefix = match.group(1) or ""
        affiliation = match.group(4)
        return f"{comment_prefix}Copyright (c) {start}-{end}, {affiliation}"

    # Apply each pattern's replacement
    res = CheckSPDXWithCopyright.sub(replace_spdx_with_copyright, line)
    res = CheckSPDXREUSE.sub(replace_spdx_reuse, res)
    res = CheckNonSPDX.sub(replace_non_spdx, res)

    return res


def read_header_lines(path):
    """Read only the bounded region in which a file header may appear."""
    with open(path, "rb") as fp:
        prefix = fp.read(MAX_HEADER_BYTES)
    return prefix.decode("utf-8", errors="replace").splitlines(keepends=True)[:MAX_HEADER_LINES]


def check_copyright(f, update_current_year, ignore_year_mismatch=False):
    """
    Checks for copyright headers and their years
    """
    errs = []
    this_year = datetime.datetime.now().year
    cr_found = False
    year_matched = False
    lines = read_header_lines(f)

    start, end, line_num, header_line = get_copyright_years(lines)
    if start is not None:
        cr_found = True
        if start > end:
            e = [
                f,
                line_num,
                "First year after second year in the copyright header (manual fix required)",
                None,
            ]
            errs.append(e)
        if not ignore_year_mismatch and (this_year < start or this_year > end):
            e = [f, line_num, "Current year not included in the copyright header", None]
            if this_year < start:
                e[-1] = replace_current_year(header_line, this_year, end)
            if this_year > end:
                e[-1] = replace_current_year(header_line, start, this_year)
            errs.append(e)
        else:
            year_matched = True
    # copyright header itself not found
    if not cr_found:
        e = [
            f,
            0,
            "Copyright header missing or formatted incorrectly (manual fix required)",
            None,
        ]
        errs.append(e)
    # even if the year matches a copyright header, make the check pass
    if year_matched:
        errs = []

    if update_current_year:
        errs_update = [x for x in errs if x[-1] is not None]
        if len(errs_update) > 0:
            print(
                "File: {}. Changing line(s) {}".format(
                    f, ", ".join(str(x[1]) for x in errs if x[-1] is not None)
                )
            )
            with open(f, encoding="utf-8", newline="") as fp:
                file_lines = fp.readlines()
            for _, line_num, __, replacement in errs_update:
                file_lines[line_num - 1] = replacement
            with open(f, "w", encoding="utf-8", newline="") as out_file:
                out_file.writelines(file_lines)
        errs = [x for x in errs if x[-1] is None]

    return errs


def get_all_files_under_dir(root):
    ret_list = []
    for dirpath, _, filenames in os.walk(root):
        ret_list.extend([os.path.join(dirpath, fn) for fn in filenames])
    return ret_list


def check_copyright_main():
    """
    Checks for copyright headers in all the modified files. In case of local
    repo, this script will just look for uncommitted files and in case of CI
    it compares between branches "$PR_TARGET_BRANCH" and "current-pr-branch"
    """
    ret_val = 0
    global ExemptFiles

    argparser = argparse.ArgumentParser(
        "Checks for a consistent copyright header in git's modified files"
    )
    argparser.add_argument(
        "--update-current-year",
        dest="update_current_year",
        action="store_true",
        required=False,
        help="If set, update the current year if a header is already present and well formatted.",
    )
    argparser.add_argument(
        "--git-modified-only",
        dest="git_modified_only",
        action="store",
        type=str,
        nargs="?",
        default=None,
        const="no-target",
        required=False,
        help="If set, "
        "only files seen as modified by git will be "
        "processed. It will look for local modifications"
        "(unstaged, untracked) if no git reference is provided.",
    )
    argparser.add_argument(
        "--exclude",
        dest="exclude",
        action="append",
        required=False,
        default=[],
        help=("Exclude paths matching a glob. Can be specified multiple times."),
    )
    argparser.add_argument(
        "--exclude-config",
        dest="exclude_config",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to a file containing exclude globs (one per line). "
            "Lines starting with # are treated as comments."
        ),
    )
    argparser.add_argument(
        "--ignore-year-mismatch",
        dest="ignore_year_mismatch",
        action="store_true",
        required=False,
        help="If set, ignore year mismatches in copyright headers (when current year is not within the copyright year range).",
    )

    args, dirs = argparser.parse_known_args()

    # Read excludes from config file if specified
    config_excludes = []
    if args.exclude_config:
        config_path = args.exclude_config
        if not os.path.isabs(config_path):
            # If relative path, try current working directory first
            if os.path.exists(config_path):
                config_path = os.path.abspath(config_path)
            else:
                # If not found in current directory, try relative to script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(script_dir, os.path.basename(config_path))

        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        config_excludes.append(line)
        else:
            print(f"Warning: Config file not found at {config_path}")

    # Combine config file excludes with command line excludes. These are
    # bounded globs, never caller-controlled regular expressions.
    all_excludes = config_excludes + args.exclude
    if any(len(pattern) > MAX_EXCLUDE_PATTERN_LENGTH for pattern in all_excludes):
        print(f"Exclude globs may not exceed {MAX_EXCLUDE_PATTERN_LENGTH} characters")
        return 1
    ExemptFiles = ExemptFiles + all_excludes

    all_files = []
    if dirs:
        for d in [os.path.abspath(d) for d in dirs]:
            if not (os.path.isdir(d)):
                raise ValueError(f"{d} is not a directory.")
            all_files += get_all_files_under_dir(d)

    if args.git_modified_only:
        target_branch = None
        if args.git_modified_only != "no-target":
            target_branch = args.git_modified_only
            print(f"Checking copyright headers in modified files between {target_branch} and HEAD")
        modified_files = gitutils.modified_files(target_branch, True)
        all_files = list(set(all_files).intersection(modified_files)) if dirs else modified_files

    files = [f for f in all_files if check_this_file(f)]

    # Print progress information
    print(f"Checking copyright headers in {len(files)} files out of {len(all_files)} total files")
    if len(files) > 0:
        print(f"Example files being checked: {', '.join(files[:3])}")
        if len(files) > 3:
            print(f"... and {len(files) - 3} more files")

    errors = tuple(
        itertools.chain(
            *[
                check_copyright(f, args.update_current_year, args.ignore_year_mismatch)
                for f in files
            ]
        )
    )
    if errors:
        print("Copyright headers incomplete in some of the files!")
        for file_name, line_no, err_msg, _ in errors:
            print(f"  {file_name}:{line_no} Issue: {err_msg}")
        print("")
        n_fixable = sum(1 for e in errors if e[-1] is not None)
        if n_fixable > 0:
            print(
                f"You can run `python3 {' '.join(sys.argv)} --update-current-year` to fix "
                f"{n_fixable} of these errors."
            )
        ret_val = 1
    else:
        print("Copyright check passed")

    return ret_val


if __name__ == "__main__":
    import sys

    sys.exit(check_copyright_main())
