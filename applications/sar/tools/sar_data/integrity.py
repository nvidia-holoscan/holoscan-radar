# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Header/XML baselines for bundled examples, never a general CPHD format restriction."""

import hashlib
import os
import urllib.parse
from functools import lru_cache
from pathlib import Path

import yaml

_BUCKET_HOSTS = {
    "umbra-open-data-catalog.s3.us-west-2.amazonaws.com": "umbra-open-data-catalog",
    "capella-open-data.s3.amazonaws.com": "capella-open-data",
    "capella-open-data.s3.us-west-2.amazonaws.com": "capella-open-data",
}
_HASH_CHUNK_BYTES = 1024 * 1024


@lru_cache(maxsize=1)
def _manifest() -> dict:
    with Path(__file__).with_name("metadata_pins.yaml").open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if manifest["version"] != 1 or manifest["scope"] != "header_xml_only":
        raise RuntimeError("unsupported bundled dataset metadata manifest")
    return manifest["sources"]


def metadata_pin(source: str | Path) -> dict | None:
    """Look up only known remote objects; local and custom CPHDs have no pin."""
    parsed = urllib.parse.urlsplit(str(source))
    if parsed.scheme == "s3":
        bucket = parsed.netloc
    elif parsed.scheme in ("http", "https"):
        bucket = _BUCKET_HOSTS.get(parsed.hostname)
    else:
        return None
    if not bucket:
        return None
    key = urllib.parse.unquote(parsed.path.removeprefix("/"))
    return _manifest().get(f"s3://{bucket}/{key}")


def verify_metadata(pin: dict, source_size: int, read) -> None:
    """Check trusted ranges before the caller parses any downloaded metadata.

    These public hashes detect changes from the repository baseline. They are
    not publisher signatures and do not authenticate PVP/support/sample bytes.
    """
    if source_size != pin["size_bytes"]:
        raise RuntimeError("bundled CPHD source size does not match its pinned metadata baseline")
    for name in ("header", "xml"):
        region = pin[name]
        digest = hashlib.sha256()
        offset, remaining = region["offset"], region["size"]
        while remaining:
            size = min(remaining, _HASH_CHUNK_BYTES)
            block = read(offset, size)
            if len(block) != size:
                raise RuntimeError(f"short bundled CPHD {name} read during metadata verification")
            digest.update(block)
            offset += size
            remaining -= size
        if digest.hexdigest() != region["sha256"]:
            raise RuntimeError(
                f"bundled CPHD {name} does not match its pinned metadata baseline; "
                "refusing to accept a changed example dataset"
            )


def verify_metadata_file(path: Path, source: str | Path) -> None:
    pin = metadata_pin(source)
    if pin is None:
        return
    with path.open("rb") as handle:

        def read(offset, size):
            handle.seek(offset)
            return handle.read(size)

        verify_metadata(pin, os.fstat(handle.fileno()).st_size, read)
