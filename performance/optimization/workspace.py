# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only archive relocation audit and current-best scratch restore receipts."""

import argparse
import time
from pathlib import Path

from artifacts import checkpoint, digest, fingerprint, read_json, restore, write_json
from timing import verify_build


def audit_archive(session, mappings):
    roots = []
    for old, new in mappings.items():
        if not Path(old).is_absolute() or not Path(new).is_absolute():
            raise ValueError("relocation roots must be absolute")
        roots.append((Path(old), Path(new).resolve()))
    roots.sort(key=lambda pair: len(pair[0].parts), reverse=True)

    def resolve(path):
        original = Path(path)
        for old, new in roots:
            if original.is_relative_to(old):
                mapped = new / original.relative_to(old)
                if not mapped.resolve().is_relative_to(new):
                    raise ValueError("relocated artifact escapes its archive root")
                return mapped
        return original

    checked, failures, visited = [], [], set()

    def walk(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                identity = (value["path"], value["sha256"])
                if identity in visited:
                    return
                visited.add(identity)
                target = resolve(value["path"])
                if not target.is_file() or digest(target) != value["sha256"]:
                    failures.append({"original": value["path"], "resolved": str(target)})
                else:
                    checked.append(
                        {
                            "original": value["path"],
                            "resolved": str(target),
                            "sha256": value["sha256"],
                        }
                    )
                    if target.suffix == ".json":
                        walk(read_json(target))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(read_json(session))
    return {
        "passed": not failures,
        "checked": checked,
        "failures": failures,
        "session_sha256": digest(session),
        "relocations": mappings,
        "scope": "Read-only path/hash audit, not new-policy candidate acceptance. Original JSON is unchanged.",
    }


def scratch_begin(root, files, targets, expected_source_id):
    if not targets or any(not Path(target).is_absolute() for target in targets):
        raise ValueError("declare absolute paths for every target touched by scratch builds")
    saved = checkpoint(root, files)
    source = fingerprint(root)
    if source != expected_source_id:
        raise ValueError("scratch baseline is not the expected current best")
    saved.update(source_id=source, rebuild_targets=list(targets))
    return saved


def scratch_restore(saved, sealed):
    restore(saved, sealed)
    if fingerprint(saved["root"]) != saved["source_id"]:
        raise ValueError("scratch files restored, but full current-best fingerprint differs")
    return {
        "root": saved["root"],
        "source_id": saved["source_id"],
        "restored_at": time.time(),
        "rebuild_targets": saved["rebuild_targets"],
        "status": "rebuild_required",
    }


def verify_scratch_rebuild(receipt, manifests):
    if fingerprint(receipt["root"]) != receipt["source_id"]:
        raise ValueError("current best changed after scratch restore")
    covered = set()
    for manifest in manifests:
        verify_build(manifest)
        if manifest["source_id"] != receipt["source_id"]:
            raise ValueError("scratch rebuild manifest source mismatch")
        covered.update(manifest["artifacts"])
    for target in receipt["rebuild_targets"]:
        if target not in covered or Path(target).stat().st_mtime < receipt["restored_at"]:
            raise ValueError("scratch target not rebuilt after restore: " + target)
    return receipt | {"status": "validated_rebuild", "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--session", required=True)
    audit.add_argument("--relocations", required=True)
    audit.add_argument("--out", required=True)
    begin = commands.add_parser("scratch-begin")
    begin.add_argument("--root", required=True)
    begin.add_argument("--files", nargs="+", required=True)
    begin.add_argument("--targets", nargs="+", required=True)
    begin.add_argument("--out", required=True)
    begin.add_argument("--source-id", required=True)
    undo = commands.add_parser("scratch-restore")
    undo.add_argument("--checkpoint", required=True)
    undo.add_argument("--state", required=True)
    undo.add_argument("--out", required=True)
    check = commands.add_parser("scratch-check")
    check.add_argument("--receipt", required=True)
    check.add_argument("--manifests", nargs="+", required=True)
    check.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "audit":
        value = audit_archive(args.session, read_json(args.relocations))
    elif args.command == "scratch-begin":
        value = scratch_begin(args.root, args.files, args.targets, args.source_id)
    elif args.command == "scratch-restore":
        value = scratch_restore(read_json(args.checkpoint), read_json(args.state))
    else:
        value = verify_scratch_rebuild(
            read_json(args.receipt), [read_json(path) for path in args.manifests]
        )
    write_json(args.out, value)
    return 0 if value.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
