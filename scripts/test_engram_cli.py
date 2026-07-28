#!/usr/bin/env python3
"""Unit tests for engram_cli.py.

No network, no live engram/qdrant: a stdlib http.server stub records the last
request (method, path, headers, body) and returns canned JSON. Run with:

    python3 scripts/test_engram_cli.py
"""

import io
import json
import os
import sys
import threading
import unittest
from contextlib import redirect_stdout, redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engram_cli  # noqa: E402


# Shared record of the last request seen by the stub server.
LAST = {}


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _handle(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        LAST.clear()
        LAST.update({
            "method": self.command,
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": json.loads(raw) if raw else None,
        })
        # Emulate the 404 path for a magic id so we can test non-2xx exit.
        if self.path.endswith("/notfound"):
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"not found"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "echo_path": self.path}).encode())

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle


class CLITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), StubHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:%d" % cls.port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def run_cli(self, extra_args):
        """Run the CLI with base --url injected. Returns (exit_code, stdout, stderr)."""
        argv = ["--url", self.url] + extra_args
        out, err = io.StringIO(), io.StringIO()
        code = 0
        try:
            with redirect_stdout(out), redirect_stderr(err):
                engram_cli.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        return code, out.getvalue(), err.getvalue()

    # ── add ──
    def test_add(self):
        code, out, err = self.run_cli([
            "add", "hello world", "--type", "insight",
            "--importance", "7", "--tags", "a", "b",
        ])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["method"], "POST")
        self.assertEqual(LAST["path"], "/memories")
        self.assertEqual(LAST["body"]["content"], "hello world")
        self.assertEqual(LAST["body"]["type"], "insight")
        self.assertEqual(LAST["body"]["importance"], 7)
        self.assertEqual(LAST["body"]["tags"], ["a", "b"])
        self.assertIn('"ok": true', out.lower().replace(" ", " ") or out)

    def test_add_collection_route(self):
        code, out, err = self.run_cli([
            "--caller-type", "reflection",
            "add", "x", "--collection", "engram_reflection",
        ])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["path"], "/collections/engram_reflection/memories")
        self.assertEqual(LAST["headers"].get("X-Caller-Type"), "reflection")

    # ── search ──
    def test_search(self):
        code, out, err = self.run_cli([
            "search", "find me", "--limit", "3", "--collection", "engram_user",
        ])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["method"], "POST")
        self.assertEqual(LAST["path"], "/memories/search")
        self.assertEqual(LAST["body"]["query"], "find me")
        self.assertEqual(LAST["body"]["limit"], 3)
        self.assertEqual(LAST["body"]["collection"], "engram_user")

    # ── get ──
    def test_get(self):
        code, out, err = self.run_cli(["get", "abc123"])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["method"], "GET")
        self.assertEqual(LAST["path"], "/memories/abc123")

    # ── delete ──
    def test_delete(self):
        code, out, err = self.run_cli(["delete", "abc123"])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["method"], "DELETE")
        self.assertEqual(LAST["path"], "/memories/abc123")

    # ── update PATCH vs PUT ──
    def test_update_patch(self):
        code, out, err = self.run_cli(["update", "id1", "--importance", "9"])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["method"], "PATCH")
        self.assertEqual(LAST["body"], {"importance": 9})

    def test_update_put_with_content(self):
        code, out, err = self.run_cli(["update", "id1", "--content", "new text"])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["method"], "PUT")
        self.assertEqual(LAST["body"]["content"], "new text")

    def test_update_empty_patch_errors(self):
        code, out, err = self.run_cli(["update", "id1"])
        self.assertNotEqual(code, 0)
        self.assertIn("nothing to change", err)

    # ── auth header injection ──
    def test_auth_header(self):
        code, out, err = self.run_cli(["--api-key", "sekret", "health"])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["headers"].get("Authorization"), "Bearer sekret")

    def test_auth_header_from_env(self):
        os.environ["ENGRAM_API_KEY"] = "envkey"
        try:
            code, out, err = self.run_cli(["health"])
        finally:
            del os.environ["ENGRAM_API_KEY"]
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["headers"].get("Authorization"), "Bearer envkey")

    # ── non-2xx exit code ──
    def test_non_2xx_exit(self):
        code, out, err = self.run_cli(["get", "notfound"])
        self.assertNotEqual(code, 0)
        self.assertIn("HTTP 404", err)
        self.assertIn("not found", err)
        self.assertEqual(out, "")

    # ── pretty output ──
    def test_pretty(self):
        code, out, err = self.run_cli(["--pretty", "health"])
        self.assertEqual(code, 0, err)
        self.assertIn("\n  ", out)  # indentation present

    # ── cross-search ──
    def test_cross_search(self):
        code, out, err = self.run_cli([
            "cross-search", "q", "--collections", "engram_user", "engram_reflection",
        ])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["path"], "/memories/cross-search")
        self.assertEqual(LAST["body"]["collections"], ["engram_user", "engram_reflection"])

    # ── reflect ──
    def test_reflect_dry_run(self):
        code, out, err = self.run_cli(["reflect", "--dry-run"])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["method"], "POST")
        self.assertEqual(LAST["path"], "/reflect")
        self.assertEqual(LAST["body"], {"dry_run": True})

    def test_reflect_check(self):
        code, out, err = self.run_cli(["reflect-check"])
        self.assertEqual(code, 0, err)
        self.assertEqual(LAST["method"], "GET")
        self.assertEqual(LAST["path"], "/reflect/check")


if __name__ == "__main__":
    unittest.main(verbosity=2)
