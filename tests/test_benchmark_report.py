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

    def _paired_runs(self, *, candidate: dict[str, object] | None = None) -> list[dict[str, object]]:
        baseline = json.loads(json.dumps(BASELINE))
        candidate_artifact = json.loads(json.dumps(candidate or BASELINE))
        runs = [
            benchmark_report.project_run(baseline, index)
            for index in range(1, 8)
        ]
        runs.extend(
            benchmark_report.project_run(candidate_artifact, index)
            for index in range(8, 15)
        )
        return runs

    def test_exact_one_sided_binomial_tail_and_holm_adjustment(self) -> None:
        self.assertEqual(benchmark_report._binomial_upper_tail(7, 7), 1 / 128)
        self.assertEqual(benchmark_report._binomial_upper_tail(6, 7), 8 / 128)
        self.assertEqual(benchmark_report._binomial_upper_tail(0, 7), 1.0)
        self.assertEqual(
            benchmark_report._holm_adjusted_p_values([0.01, 0.04, 1.0, 0.03]),
            [0.04, 0.09, 1.0, 0.09],
        )

    def test_paired_gate_detects_repeated_practical_regressions_after_holm(self) -> None:
        candidate = json.loads(json.dumps(BASELINE))
        candidate["throughput_rps"] = 80.0
        candidate["success_rate"] = 0.9
        candidate["successful_requests"] = 9
        candidate["failed_requests"] = 1
        candidate["latency_ms"]["p95"] = 9.6
        candidate["retry"]["client_attempt_amplification"] = 1.1

        report = benchmark_report.build_trend_report(
            self._paired_runs(candidate=candidate),
            paired_baseline_runs=7,
            paired_significance_level=0.05,
        )

        self.assertEqual(report["schema_version"], 2)
        self.assertTrue(report["statistical_regression"])
        analysis = report["paired_statistical_analysis"]
        self.assertEqual(
            analysis["method"],
            "one_sided_exact_binomial_test_of_paired_threshold_exceedances",
        )
        self.assertEqual(analysis["run_count_per_group"], 7)
        self.assertEqual(analysis["multiple_comparison_correction"], "Holm step-down")
        self.assertIn("first N inputs are baseline", analysis["pairing"])
        self.assertIn("independent", analysis["operator_assumptions"])
        tests = {item["metric"]: item for item in analysis["tests"]}
        self.assertEqual(set(tests), {
            "p95_latency", "success_rate", "throughput", "attempt_amplification"
        })
        for result in tests.values():
            self.assertEqual(result["pairs"], 7)
            self.assertEqual(result["exceeding_pairs"], 7)
            self.assertEqual(result["p_value"], 1 / 128)
            self.assertEqual(result["adjusted_p_value"], 0.03125)
            self.assertTrue(result["regression"])

    def test_paired_gate_treats_budget_boundary_as_non_exceedance(self) -> None:
        baseline = json.loads(json.dumps(BASELINE))
        baseline["num_requests"] = 100
        baseline["successful_requests"] = 100
        candidate = json.loads(json.dumps(baseline))
        candidate["successful_requests"] = 99
        candidate["failed_requests"] = 1
        candidate["success_rate"] = 0.99
        candidate["throughput_rps"] = 90.0
        candidate["latency_ms"]["p95"] = 8.8
        candidate["retry"]["client_attempt_amplification"] = 1.05
        runs = [
            benchmark_report.project_run(baseline, index)
            for index in range(1, 8)
        ] + [
            benchmark_report.project_run(candidate, index)
            for index in range(8, 15)
        ]

        report = benchmark_report.build_trend_report(
            runs,
            paired_baseline_runs=7,
            paired_significance_level=0.05,
        )

        self.assertFalse(report["statistical_regression"])
        for result in report["paired_statistical_analysis"]["tests"]:
            self.assertEqual(result["exceeding_pairs"], 0)
            self.assertEqual(result["p_value"], 1.0)
            self.assertFalse(result["regression"])

    def test_paired_gate_requires_seven_pairs_and_exact_input_cardinality(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 7"):
            benchmark_report.build_trend_report(
                self._paired_runs(), paired_baseline_runs=6
            )
        with self.assertRaisesRegex(ValueError, "exactly 2 times"):
            benchmark_report.build_trend_report(
                self._paired_runs()[:-1], paired_baseline_runs=7
            )
        with self.assertRaisesRegex(ValueError, "significance level"):
            benchmark_report.build_trend_report(
                self._paired_runs(),
                paired_baseline_runs=7,
                paired_significance_level=1.0,
            )

    def test_paired_gate_keeps_unavailable_metrics_in_fixed_holm_family(self) -> None:
        candidate = json.loads(json.dumps(BASELINE))
        candidate.pop("retry")
        report = benchmark_report.build_trend_report(
            self._paired_runs(candidate=candidate), paired_baseline_runs=7
        )
        tests = {
            item["metric"]: item
            for item in report["paired_statistical_analysis"]["tests"]
        }
        retry = tests["attempt_amplification"]
        self.assertFalse(retry["evaluable"])
        self.assertEqual(retry["p_value"], 1.0)
        self.assertEqual(retry["adjusted_p_value"], 1.0)
        self.assertIn("not present", retry["note"])

        zero_baseline = json.loads(json.dumps(BASELINE))
        zero_baseline["throughput_rps"] = 0.0
        nonzero_candidate = json.loads(json.dumps(zero_baseline))
        nonzero_candidate["throughput_rps"] = 1.0
        zero_runs = [
            benchmark_report.project_run(zero_baseline, index)
            for index in range(1, 8)
        ] + [
            benchmark_report.project_run(nonzero_candidate, index)
            for index in range(8, 15)
        ]
        zero_report = benchmark_report.build_trend_report(
            zero_runs, paired_baseline_runs=7
        )
        throughput = {
            item["metric"]: item
            for item in zero_report["paired_statistical_analysis"]["tests"]
        }["throughput"]
        self.assertFalse(throughput["evaluable"])
        self.assertEqual(throughput["p_value"], 1.0)
        self.assertEqual(throughput["adjusted_p_value"], 1.0)

    def test_paired_report_markdown_omits_private_artifact_fields(self) -> None:
        report = benchmark_report.build_trend_report(
            self._paired_runs(), paired_baseline_runs=7
        )
        markdown = benchmark_report.render_markdown(report)
        self.assertIn("Paired statistical analysis", markdown)
        self.assertIn("Holm step-down", markdown)
        self.assertIn("run-level", markdown)
        self.assertIn("first N inputs are baseline", markdown)
        self.assertIn("Operator assumptions:", markdown)
        self.assertIn("Threshold", markdown)
        self.assertIn("Raw p-value", markdown)
        self.assertIn("Holm-adjusted p-value", markdown)
        self.assertNotIn("private.example", markdown)
        self.assertNotIn("private prompt", markdown)
        self.assertNotIn("server_url", markdown)
        self.assertNotIn("/tmp/", markdown)

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
