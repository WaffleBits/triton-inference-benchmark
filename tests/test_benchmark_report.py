from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import benchmark_report


BASELINE = {
    "mode": "mock",
    "num_requests": 10,
    "successful_requests": 10,
    "failed_requests": 0,
    "success_rate": 1.0,
    "throughput_rps": 100.0,
    "duration_seconds": 0.1,
    "latency_ms": {"p50": 4.0, "p95": 8.0, "p99": 10.0},
    "retry": {
        "client_attempt_amplification": 1.0,
        "retry_attempts": 0,
        "retried_requests": 0,
        "recovered_requests": 0,
        "exhausted_requests": 0,
    },
    "server_url": "https://private.example/v1",
    "config": {"openai_prompt_sha256": "private-prompt-digest"},
}


class BenchmarkReportTests(unittest.TestCase):
    def test_projection_excludes_private_fields_and_keeps_measured_values(self) -> None:
        projected = benchmark_report.project_run(BASELINE, 1)

        self.assertEqual(projected["run_index"], 1)
        self.assertEqual(projected["mode"], "mock")
        self.assertEqual(projected["throughput_rps"], 100.0)
        self.assertEqual(projected["latency_ms"]["p95"], 8.0)
        self.assertEqual(projected["retry"]["client_attempt_amplification"], 1.0)
        serialized = json.dumps(projected)
        self.assertNotIn("private.example", serialized)
        self.assertNotIn("private-prompt-digest", serialized)
        self.assertNotIn("server_url", serialized)
        self.assertNotIn("config", serialized)

    def test_adjacent_comparison_gates_latency_success_throughput_and_retries(self) -> None:
        candidate = json.loads(json.dumps(BASELINE))
        candidate["throughput_rps"] = 80.0
        candidate["success_rate"] = 0.9
        candidate["successful_requests"] = 9
        candidate["failed_requests"] = 1
        candidate["latency_ms"]["p95"] = 9.6
        candidate["retry"]["client_attempt_amplification"] = 1.1

        report = benchmark_report.build_trend_report(
            [
                benchmark_report.project_run(BASELINE, 1),
                benchmark_report.project_run(candidate, 2),
            ],
            max_p95_regression_pct=10.0,
            max_success_rate_drop=0.01,
            max_throughput_drop_pct=10.0,
            max_attempt_amplification_increase=0.05,
        )

        comparison = report["comparisons"][0]
        self.assertTrue(comparison["regression"])
        reasons = " ".join(comparison["regression_reasons"])
        self.assertIn("p95 latency", reasons)
        self.assertIn("success rate", reasons)
        self.assertIn("throughput", reasons)
        self.assertIn("attempt amplification", reasons)
        self.assertEqual(comparison["changes"]["success_rate_delta"], -0.1)
        self.assertEqual(comparison["changes"]["throughput_delta_pct"], -20.0)

    def test_zero_baseline_fails_closed_for_percentage_checks(self) -> None:
        baseline = json.loads(json.dumps(BASELINE))
        baseline["throughput_rps"] = 0.0
        candidate = json.loads(json.dumps(BASELINE))
        candidate["throughput_rps"] = 1.0

        report = benchmark_report.build_trend_report(
            [
                benchmark_report.project_run(baseline, 1),
                benchmark_report.project_run(candidate, 2),
            ]
        )

        comparison = report["comparisons"][0]
        self.assertTrue(comparison["regression"])
        self.assertIsNone(comparison["changes"]["throughput_delta_pct"])
        self.assertFalse(comparison["checks"]["throughput"]["evaluable"])
        self.assertIn("throughput comparison unavailable", comparison["regression_reasons"][0])

    def test_missing_retry_data_is_reported_without_inventing_a_value(self) -> None:
        first = json.loads(json.dumps(BASELINE))
        second = json.loads(json.dumps(BASELINE))
        first.pop("retry")
        second.pop("retry")

        report = benchmark_report.build_trend_report(
            [
                benchmark_report.project_run(first, 1),
                benchmark_report.project_run(second, 2),
            ]
        )

        comparison = report["comparisons"][0]
        self.assertIsNone(comparison["changes"]["client_attempt_amplification_delta"])
        self.assertFalse(comparison["checks"]["attempt_amplification"]["evaluable"])
        self.assertFalse(comparison["regression"])

    def test_invalid_run_is_rejected(self) -> None:
        invalid = json.loads(json.dumps(BASELINE))
        invalid["success_rate"] = 2.0
        with self.assertRaises(ValueError):
            benchmark_report.project_run(invalid, 1)

    def test_markdown_does_not_include_input_paths_or_private_fields(self) -> None:
        report = benchmark_report.build_trend_report(
            [
                benchmark_report.project_run(BASELINE, 1),
                benchmark_report.project_run(BASELINE, 2),
            ]
        )
        markdown = benchmark_report.render_markdown(report)
        self.assertIn("Benchmark trend report", markdown)
        self.assertIn("measured benchmark phase", markdown)
        self.assertNotIn("private.example", markdown)
        self.assertNotIn("server_url", markdown)
        self.assertNotIn("/tmp/", markdown)

    def test_real_json_loader_rejects_oversized_or_non_object_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "run.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark_report.load_run(path, 1)

    def test_legacy_committed_example_keeps_missing_fields_explicit(self) -> None:
        path = Path(__file__).resolve().parents[1] / "sample_results" / "mock_run.json"
        projected = benchmark_report.load_run(path, 1)
        self.assertEqual(projected["failed_requests"], 0)
        self.assertEqual(projected["duration_seconds"], 0.3274)
        self.assertNotIn("retry", projected)


if __name__ == "__main__":
    unittest.main()
