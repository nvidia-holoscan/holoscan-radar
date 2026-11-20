# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP source authorization, including redirects before any target connection."""

import contextlib
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from sar_data.network import (
    DEFAULT_SOURCE_ORIGINS,
    SourceRedirectHandler,
    open_source_url,
    source_origins,
    validate_source_url,
)


@contextlib.contextmanager
def server(redirect=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(self.path)
            if redirect:
                self.send_response(302)
                self.send_header("Location", redirect)
            else:
                self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


class SourceOriginTest(unittest.TestCase):
    def test_all_builtin_endpoints_allow_objects_and_queries(self):
        for origin in DEFAULT_SOURCE_ORIGINS:
            with self.subTest(origin=origin):
                validate_source_url(origin + "/path/to/object?versionId=fixed")
                validate_source_url(origin.upper() + ":443/object")

    def test_unapproved_origins_fail_without_resolving_or_connecting(self):
        host = "umbra-open-data-catalog.s3.us-west-2.amazonaws.com"
        for url in (
            "http://127.0.0.1/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/admin",
            "https://10.0.0.1/",
            "https://internal.example/",
            "https://arbitrary-public.example/",
            f"https://{host}.evil.example/",
            f"https://{host}:8443/",
            f"http://{host}/",
            f"https://{host}./",
            "http://2130706433/",
        ):
            with (
                self.subTest(url=url),
                mock.patch("urllib.request.build_opener") as opener,
                self.assertRaisesRegex(ValueError, "not approved"),
            ):
                try:
                    open_source_url(url, timeout=1)
                finally:
                    opener.assert_not_called()

    def test_rejects_ambiguous_or_credential_bearing_urls(self):
        for url in (
            "https://user:secret@umbra-open-data-catalog.s3.us-west-2.amazonaws.com/",
            "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/#fragment",
            "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/\nadmin",
            "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com\\@example.test/",
            "https://example.test:bad/",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_source_url(url)

    def test_approval_is_exact_scheme_host_port_and_not_a_path(self):
        validate_source_url("http://127.0.0.1:1234/object", ("http://127.0.0.1:1234",))
        validate_source_url("https://custom.example/object", ("https://CUSTOM.example:443/",))
        with self.assertRaisesRegex(ValueError, "not approved"):
            validate_source_url("http://127.0.0.1:1235/object", ("http://127.0.0.1:1234",))
        for bad in ("https://custom.example/path", "https://custom.example/?q=1"):
            with self.subTest(origin=bad), self.assertRaisesRegex(ValueError, "only scheme"):
                source_origins((bad,))

    def test_redirect_to_an_unapproved_origin_never_reaches_the_target(self):
        with server() as (target, target_requests):
            with server(target + "/admin") as (origin, initial_requests):
                with self.assertRaisesRegex(ValueError, "not approved"):
                    open_source_url(origin + "/object", timeout=2, extra_origins=(origin,))
                self.assertEqual(initial_requests, ["/object"])
                self.assertEqual(target_requests, [])

    def test_explicit_approval_of_both_servers_allows_redirect(self):
        with server() as (target, target_requests):
            with server(target + "/object") as (origin, _requests):
                with open_source_url(
                    origin + "/object", timeout=2, extra_origins=(origin, target)
                ) as response:
                    self.assertEqual(response.status, 200)
                self.assertEqual(target_requests, ["/object"])

    def test_redirect_cannot_downgrade_a_builtin_https_endpoint(self):
        host = "umbra-open-data-catalog.s3.us-west-2.amazonaws.com"
        request = urllib.request.Request(f"https://{host}/object")
        with self.assertRaisesRegex(ValueError, "not approved"):
            SourceRedirectHandler().redirect_request(
                request, None, 302, "Found", {}, f"http://{host}/object"
            )


if __name__ == "__main__":
    unittest.main()
