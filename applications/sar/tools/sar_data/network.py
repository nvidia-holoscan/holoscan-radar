# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit HTTP source origins; recipes cannot extend the caller's trust policy."""

import urllib.parse
import urllib.request

DEFAULT_SOURCE_ORIGINS = frozenset(
    {
        "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com",
        "https://capella-open-data.s3.amazonaws.com",
        "https://capella-open-data.s3.us-west-2.amazonaws.com",
    }
)


def _parse_url(url: str) -> tuple[urllib.parse.SplitResult, str]:
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url
    ):
        raise ValueError("HTTP source URL must not contain whitespace or control characters")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("HTTP source needs an http:// or https:// host")
    if parsed.username is not None or parsed.password is not None or "\\" in parsed.netloc:
        raise ValueError("HTTP source URL must not contain credentials or a backslash host")
    if parsed.fragment:
        raise ValueError("HTTP source URL must not contain a fragment")
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    suffix = "" if port is None or port == {"http": 80, "https": 443}[parsed.scheme] else f":{port}"
    return parsed, f"{parsed.scheme}://{host}{suffix}"


def source_origins(extra_origins: tuple[str, ...] = ()) -> frozenset[str]:
    origins = set(DEFAULT_SOURCE_ORIGINS)
    for value in extra_origins:
        parsed, origin = _parse_url(value)
        if parsed.path not in ("", "/") or parsed.query:
            raise ValueError("--allow-source-origin needs only scheme, host, and optional port")
        origins.add(origin)
    return frozenset(origins)


def validate_source_url(url: str, extra_origins: tuple[str, ...] = ()) -> urllib.parse.SplitResult:
    parsed, origin = _parse_url(url)
    if origin not in source_origins(extra_origins):
        raise ValueError(
            f"HTTP source origin {origin} is not approved; "
            "use --allow-source-origin to explicitly trust this server"
        )
    return parsed


class SourceRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Check the new origin before urllib can follow a redirect."""

    def __init__(self, extra_origins: tuple[str, ...] = ()):
        self.extra_origins = tuple(extra_origins)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_source_url(newurl, self.extra_origins)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_source_url(request, *, timeout: float, extra_origins: tuple[str, ...] = ()):
    """Open a validated source, rechecking every redirect before connecting."""
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    validate_source_url(url, extra_origins)
    opener = urllib.request.build_opener(SourceRedirectHandler(extra_origins))
    return opener.open(request, timeout=timeout)
