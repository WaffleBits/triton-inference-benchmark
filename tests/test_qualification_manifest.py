from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import benchmark_report
import qualification_manifest


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
    "config": {
        "openai_prompt": "private prompt must not be copied",
        "authorization": "private credential must not be copied",
    },
}


class QualificationManifestTests(unittest.TestCase):
    def _write_case(
        self,
        root: Path,
        *,
        candidate: dict[str, object] | None = None,
    ) -> tuple[Path, list[Path], dict[str, object]]:
        first = root / "first.json"
        second = root / "second.json"
        report_path = root / "trend.json"
        first.write_text(json.dumps(BASELINE, indent=2) + "\n", encoding="utf-8")
        candidate_artifact = copy.deepcopy(candidate or BASELINE)
        second.write_text(
            json.dumps(candidate_artifact, indent=2) + "\n", encoding="utf-8"
        )
        runs = [
            benchmark_report.load_run(first, 1),
            benchmark_report.load_run(second, 2),
        ]
        report = benchmark_report.build_trend_report(runs)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report_path, [first, second], report

    def test_manifest_rederives_report_and_omits_private_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path, inputs, report = self._write_case(Path(temp_dir))
            manifest = qualification_manifest.create_manifest(report_path, inputs)
            serialized = json.dumps(manifest)

            self.assertEqual(manifest["kind"], "benchmark_qualification_manifest")
            self.assertEqual(manifest["trend_report"]["run_count"], 2)
            self.assertEqual(len(manifest["input_artifacts"]), 2)
            self.assertFalse(manifest["scope"]["paths_persisted"])
            self.assertFalse(manifest["scope"]["content_persisted"])
            self.assertNotIn("private.example", serialized)
            self.assertNotIn("private prompt", serialized)
            self.assertNotIn("private credential", serialized)
            self.assertNotIn(str(Path(temp_dir)), serialized)
            self.assertEqual(manifest["trend_report"]["regression"], report["regression"])

    def test_verify_accepts_exact_artifacts_and_rejects_whitespace_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path, inputs, _ = self._write_case(root)
            manifest_path = root / "manifest.json"
            manifest = qualification_manifest.create_manifest(report_path, inputs)
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            result = qualification_manifest.verify_manifest(
                manifest_path, report_path, inputs
            )
            self.assertEqual(result["verified"], True)
            self.assertEqual(result["run_count"], 2)

            inputs[0].write_text(inputs[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                qualification_manifest.verify_manifest(manifest_path, report_path, inputs)

    def test_require_pass_rejects_regressed_report(self) -> None:
        candidate = copy.deepcopy(BASELINE)
        candidate["throughput_rps"] = 50.0
        candidate["latency_ms"] = {"p50": 4.0, "p95": 20.0, "p99": 25.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path, inputs, report = self._write_case(
                Path(temp_dir), candidate=candidate
            )
            self.assertTrue(report["regression"])
            with self.assertRaisesRegex(ValueError, "regression gate"):
                qualification_manifest.create_manifest(
                    report_path, inputs, require_pass=True
                )


if __name__ == "__main__":
    unittest.main()
