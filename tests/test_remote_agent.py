from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from remote_agent import (
    _post_json,
    build_child_environment,
    select_clock_observation,
    validate_agent_base_url,
)


class RemoteAgentProtocolTest(unittest.TestCase):
    def test_authenticated_post_does_not_follow_redirects(self) -> None:
        class SinkHandler(BaseHTTPRequestHandler):
            request_count = 0

            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                SinkHandler.request_count += 1
                self.send_response(200)
                self.end_headers()

        sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)

        class RedirectHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(307)
                self.send_header(
                    "Location", f"http://127.0.0.1:{sink.server_port}/capture"
                )
                self.send_header("Content-Length", "0")
                self.end_headers()

        redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (sink, redirect)
        ]
        for thread in threads:
            thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "HTTP 307"):
                _post_json(
                    f"http://127.0.0.1:{redirect.server_port}",
                    "/v1/clock",
                    {"schema_version": 1, "challenge": "a" * 64},
                    api_key="redirect-test-agent-key",
                    timeout_seconds=2,
                )
            self.assertEqual(SinkHandler.request_count, 0)
        finally:
            redirect.shutdown()
            sink.shutdown()
            redirect.server_close()
            sink.server_close()
            for thread in threads:
                thread.join(timeout=2)

    def test_agent_url_requires_https_except_for_loopback(self) -> None:
        self.assertEqual(
            validate_agent_base_url("https://agent.example.test:8443/"),
            "https://agent.example.test:8443",
        )
        self.assertEqual(
            validate_agent_base_url("http://127.0.0.1:8123"),
            "http://127.0.0.1:8123",
        )
        self.assertEqual(
            validate_agent_base_url("http://[::1]:8123/"),
            "http://[::1]:8123",
        )

        rejected = (
            "http://agent.example.test:8123",
            "https://user:secret@agent.example.test",
            "https://agent.example.test/api",
            "https://agent.example.test?token=secret",
            "file:///tmp/agent.sock",
        )
        for value in rejected:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_agent_base_url(value)

    def test_clock_selection_uses_minimum_network_delay_sample(self) -> None:
        selected = select_clock_observation(
            [
                {
                    "coordinator_send_unix_ns": 1_000,
                    "agent_receive_unix_ns": 1_120,
                    "agent_send_unix_ns": 1_130,
                    "coordinator_receive_unix_ns": 1_050,
                },
                {
                    "coordinator_send_unix_ns": 2_000,
                    "agent_receive_unix_ns": 2_130,
                    "agent_send_unix_ns": 2_140,
                    "coordinator_receive_unix_ns": 2_080,
                },
            ]
        )

        self.assertEqual(selected["clock_offset_agent_minus_coordinator_ns"], 100)
        self.assertEqual(selected["clock_network_delay_ns"], 40)
        self.assertEqual(selected["clock_uncertainty_ns"], 20)
        self.assertEqual(selected["clock_sample_count"], 2)
        self.assertEqual(selected["clock_selection"], "minimum_network_delay")

    def test_clock_selection_rejects_impossible_exchange(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative network delay"):
            select_clock_observation(
                [
                    {
                        "coordinator_send_unix_ns": 1_000,
                        "agent_receive_unix_ns": 1_010,
                        "agent_send_unix_ns": 1_100,
                        "coordinator_receive_unix_ns": 1_050,
                    }
                ]
            )

    def test_child_environment_requires_agent_side_opt_in_and_excludes_agent_key(self) -> None:
        source = {
            "PATH": "/usr/bin",
            "AGENT_KEY": "agent-secret",
            "TARGET_KEY": "target-secret",
            "AMBIENT_KEY": "ambient-secret",
        }
        child = build_child_environment(
            "AGENT_KEY", ["TARGET_KEY"], source=source
        )

        self.assertEqual(
            child,
            {
                "PATH": "/usr/bin",
                "PYTHONIOENCODING": "utf-8",
                "TARGET_KEY": "target-secret",
            },
        )
        self.assertNotIn("AGENT_KEY", child)
        self.assertNotIn("AMBIENT_KEY", child)

        with self.assertRaisesRegex(ValueError, "agent API key"):
            build_child_environment("AGENT_KEY", ["AGENT_KEY"], source=source)
        with self.assertRaisesRegex(ValueError, "is not set"):
            build_child_environment("AGENT_KEY", ["MISSING_KEY"], source=source)


if __name__ == "__main__":
    unittest.main()
