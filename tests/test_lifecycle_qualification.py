from __future__ import annotations

import json
import unittest

import lifecycle_qualification


BENCHMARK = {
    "mode": "openai",
    "num_requests": 4,
    "successful_requests": 4,
    "failed_requests": 0,
    "success_rate": 1.0,
    "throughput_rps": 12.5,
    "duration_seconds": 0.32,
    "latency_ms": {"p50": 12.0, "p95": 18.0, "p99": 20.0},
    "server_url": "http://127.0.0.1:1234/v1",
    "config": {"openai_prompt": "private fixture prompt"},
}


class LifecycleQualificationTests(unittest.TestCase):
    def test_health_url_is_loopback_only_and_has_no_credentials(self) -> None:
        self.assertEqual(
            lifecycle_qualification.validate_health_url(
                "http://127.0.0.1:8080/healthz"
            ),
            "http://127.0.0.1:8080/healthz",
        )
        with self.assertRaises(ValueError):
            lifecycle_qualification.validate_health_url("http://example.com/healthz")
        with self.assertRaises(ValueError):
            lifecycle_qualification.validate_health_url(
                "http://user:secret@127.0.0.1:8080/healthz"
            )
        with self.assertRaises(ValueError):
            lifecycle_qualification.validate_health_url(
                "http://127.0.0.1:8080/healthz?token=secret"
            )

    def test_port_placeholder_is_substituted_without_shell_parsing(self) -> None:
        command = ["python", "fixture.py", "--port", "{port}"]
        self.assertEqual(
            lifecycle_qualification.substitute_port(command, 4321),
            ["python", "fixture.py", "--port", "4321"],
        )
        with self.assertRaises(ValueError):
            lifecycle_qualification.substitute_port(["python", "fixture.py"], 4321)
        with self.assertRaises(ValueError):
            lifecycle_qualification.substitute_port(["python", "{other}"], 4321)

    def test_benchmark_arguments_cannot_override_wrapper_output(self) -> None:
        with self.assertRaises(ValueError):
            lifecycle_qualification.validate_benchmark_arguments(
                ["--mode", "mock", "--output-dir", "private-results"]
            )
        with self.assertRaises(ValueError):
            lifecycle_qualification.validate_benchmark_arguments(
                ["--mode=mock", "--output-dir=private-results"]
            )

    def test_projection_keeps_measured_values_and_hashes_private_inputs(self) -> None:
        report = lifecycle_qualification.build_lifecycle_report(
            startup_latency_ms=123.45678,
            readiness_probe_count=4,
            service_exit_code=-15,
            termination="SIGTERM",
            command=["python", "fixture.py", "--secret", "not-for-artifact"],
            health_url="http://127.0.0.1:4321/healthz",
            benchmark_artifact=BENCHMARK,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["lifecycle"]["startup_latency_ms"], 123.4568)
        self.assertEqual(report["lifecycle"]["readiness_probe_count"], 4)
        self.assertEqual(report["lifecycle"]["termination"], "SIGTERM")
        self.assertTrue(report["lifecycle"]["command_sha256"])
        self.assertTrue(report["lifecycle"]["health_url_sha256"])
        serialized = json.dumps(report)
        self.assertNotIn("not-for-artifact", serialized)
        self.assertNotIn("127.0.0.1:4321", serialized)
        self.assertNotIn("private fixture prompt", serialized)
        self.assertNotIn("server_url", serialized)
        self.assertNotIn("config", serialized)
        self.assertEqual(report["benchmark"]["successful_requests"], 4)

    def test_markdown_states_scope_without_private_values(self) -> None:
        report = lifecycle_qualification.build_lifecycle_report(
            startup_latency_ms=10.0,
            readiness_probe_count=1,
            service_exit_code=-15,
            termination="SIGTERM",
            command=["python", "fixture.py"],
            health_url="http://127.0.0.1:4321/healthz",
            benchmark_artifact=BENCHMARK,
        )
        markdown = lifecycle_qualification.render_markdown(report)
        self.assertIn("Controlled lifecycle qualification", markdown)
        self.assertIn("process launch to the selected HTTP-200 readiness", markdown)
        self.assertNotIn("fixture.py", markdown)
        self.assertNotIn("127.0.0.1", markdown)


if __name__ == "__main__":
    unittest.main()
