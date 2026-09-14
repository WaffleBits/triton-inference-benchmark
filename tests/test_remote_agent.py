from __future__ import annotations

import json
import sqlite3
import stat
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from remote_agent import (
    AgentProtocolError,
    AgentTransportError,
    BenchmarkAgentServer,
    _post_json,
    build_child_environment,
    build_coordinated_agent_artifact,
    run_agent_benchmark,
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

    def test_agent_artifact_projection_omits_endpoint_prompt_and_output_data(self) -> None:
        projected = build_coordinated_agent_artifact(
            {
                "mode": "openai",
                "num_requests": 4,
                "successful_requests": 4,
                "failed_requests": 0,
                "duration_seconds": 1.0,
                "retry": {"logical_requests": 4},
                "load_schedule": None,
                "coordination": {"client_index": 0},
                "server_url": "https://private.example.test/v1",
                "config": {
                    "server_url": "https://private.example.test/v1",
                    "openai_prompt_sha256": "a" * 64,
                },
                "private_output": "sensitive response",
            }
        )

        serialized = str(projected)
        self.assertEqual(
            set(projected),
            {
                "mode",
                "num_requests",
                "successful_requests",
                "failed_requests",
                "duration_seconds",
                "retry",
                "load_schedule",
                "coordination",
            },
        )
        self.assertNotIn("private.example.test", serialized)
        self.assertNotIn("sensitive response", serialized)

    def test_completed_run_is_cached_but_conflicts_and_expired_results_fail_closed(self) -> None:
        server = BenchmarkAgentServer(
            ("127.0.0.1", 0),
            api_key="unit-test-agent-key",
            agent_id="unit-test-agent",
            child_timeout_seconds=5,
            child_environment={"PATH": "", "PYTHONIOENCODING": "utf-8"},
            max_cached_results=1,
        )
        try:
            first_artifact = {"successful_requests": 4}
            self.assertIsNone(server.begin_run("run-a", "a" * 64))
            server.complete_run("run-a", first_artifact)
            self.assertEqual(
                server.begin_run("run-a", "a" * 64), first_artifact
            )

            with self.assertRaisesRegex(AgentProtocolError, "different request"):
                server.begin_run("run-a", "b" * 64)

            self.assertIsNone(server.begin_run("unfinished", "d" * 64))
            with self.assertRaisesRegex(AgentProtocolError, "did not complete"):
                server.begin_run("unfinished", "d" * 64)

            self.assertIsNone(server.begin_run("run-b", "c" * 64))
            server.complete_run("run-b", {"successful_requests": 2})
            with self.assertRaisesRegex(AgentProtocolError, "expired"):
                server.begin_run("run-a", "a" * 64)
        finally:
            server.server_close()

    def test_durable_state_recovers_completed_result_after_server_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "agent-state.sqlite3"
            identity_a = "a" * 64
            identity_b = "b" * 64
            fingerprint_a = "c" * 64
            fingerprint_b = "d" * 64
            artifact_a = {"successful_requests": 4}
            artifact_b = {"successful_requests": 2}

            first = BenchmarkAgentServer(
                ("127.0.0.1", 0),
                api_key="unit-test-agent-key",
                agent_id="unit-test-agent",
                child_timeout_seconds=5,
                child_environment={"PATH": "", "PYTHONIOENCODING": "utf-8"},
                max_cached_results=1,
                state_db=state_path,
            )
            try:
                self.assertIsNone(first.begin_run(identity_a, fingerprint_a))
                first.complete_run(identity_a, artifact_a)
                self.assertIsNone(first.begin_run(identity_b, fingerprint_b))
                first.complete_run(identity_b, artifact_b)
            finally:
                first.server_close()

            self.assertEqual(stat.S_IMODE(state_path.stat().st_mode), 0o600)
            with sqlite3.connect(state_path) as connection:
                rows = connection.execute(
                    "SELECT identity, request_fingerprint, state, artifact_json "
                    "FROM agent_run_records ORDER BY accepted_order"
                ).fetchall()
                metadata = connection.execute(
                    "SELECT key, value FROM agent_state_metadata ORDER BY key"
                ).fetchall()
            serialized_rows = json.dumps([rows, metadata])
            self.assertNotIn("unit-test-agent-key", serialized_rows)
            self.assertNotIn("unit-test-agent", serialized_rows)
            self.assertEqual(rows[0][0], identity_a)
            self.assertEqual(rows[0][2], "completed_result_expired")
            self.assertIsNone(rows[0][3])
            self.assertEqual(json.loads(rows[1][3]), artifact_b)

            second = BenchmarkAgentServer(
                ("127.0.0.1", 0),
                api_key="unit-test-agent-key",
                agent_id="unit-test-agent",
                child_timeout_seconds=5,
                child_environment={"PATH": "", "PYTHONIOENCODING": "utf-8"},
                max_cached_results=1,
                state_db=state_path,
            )
            try:
                self.assertEqual(
                    second.begin_run(identity_b, fingerprint_b), artifact_b
                )
                with self.assertRaisesRegex(AgentProtocolError, "different request"):
                    second.begin_run(identity_b, "e" * 64)
                with self.assertRaisesRegex(AgentProtocolError, "expired"):
                    second.begin_run(identity_a, fingerprint_a)

                unfinished_identity = "f" * 64
                unfinished_fingerprint = "1" * 64
                self.assertIsNone(
                    second.begin_run(unfinished_identity, unfinished_fingerprint)
                )
                with self.assertRaisesRegex(ValueError, "unsupported field"):
                    second.complete_run(
                        unfinished_identity,
                        {"private_output": "must never be stored"},
                    )
            finally:
                second.server_close()

            third = BenchmarkAgentServer(
                ("127.0.0.1", 0),
                api_key="unit-test-agent-key",
                agent_id="unit-test-agent",
                child_timeout_seconds=5,
                child_environment={"PATH": "", "PYTHONIOENCODING": "utf-8"},
                max_cached_results=1,
                state_db=state_path,
            )
            try:
                with self.assertRaisesRegex(AgentProtocolError, "did not complete"):
                    third.begin_run(unfinished_identity, unfinished_fingerprint)
            finally:
                third.server_close()

            with self.assertRaisesRegex(ValueError, "different agent identity"):
                BenchmarkAgentServer(
                    ("127.0.0.1", 0),
                    api_key="unit-test-agent-key",
                    agent_id="different-unit-test-agent",
                    child_timeout_seconds=5,
                    child_environment={"PATH": "", "PYTHONIOENCODING": "utf-8"},
                    max_cached_results=1,
                    state_db=state_path,
                )

    def test_transport_failure_retries_same_run_and_records_cached_recovery(self) -> None:
        response = {
            "schema_version": 1,
            "agent_id_sha256": "d" * 64,
            "result_source": "cached",
            "artifact": {"successful_requests": 4},
        }
        with patch(
            "remote_agent._post_json",
            side_effect=[AgentTransportError("response lost"), response],
        ) as post:
            artifact = run_agent_benchmark(
                "https://agent.example.test",
                api_key="unit-test-agent-key",
                expected_agent_id_sha256="d" * 64,
                run_id="private-run-id",
                client_index=0,
                client_count=2,
                planned_start_agent_unix_ns=1_000_000_000,
                benchmark_args=["--mode", "mock", "--num-requests", "4"],
                timeout_seconds=5,
                recovery_attempts=1,
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(artifact["successful_requests"], 4)
        self.assertEqual(
            artifact["_remote_result_delivery"],
            {
                "result_source": "cached",
                "transport_retries": 1,
                "max_transport_recovery_attempts": 1,
            },
        )

    def test_explicit_http_rejection_is_not_retried(self) -> None:
        with patch(
            "remote_agent._post_json",
            side_effect=RuntimeError("agent returned HTTP 409: conflict"),
        ) as post:
            with self.assertRaisesRegex(RuntimeError, "HTTP 409"):
                run_agent_benchmark(
                    "https://agent.example.test",
                    api_key="unit-test-agent-key",
                    expected_agent_id_sha256="d" * 64,
                    run_id="private-run-id",
                    client_index=0,
                    client_count=2,
                    planned_start_agent_unix_ns=1_000_000_000,
                    benchmark_args=["--mode", "mock", "--num-requests", "4"],
                    timeout_seconds=5,
                    recovery_attempts=3,
                )
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
