"""Exercise benchmark, trend-report, and qualification-manifest CLIs together."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_benchmark(output_dir: Path, seed: int) -> Path:
    subprocess.run(
        [
            sys.executable,
            "benchmark.py",
            "--mode",
            "mock",
            "--num-requests",
            "8",
            "--concurrency",
            "2",
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    results = sorted(output_dir.glob("benchmark_*.json"))
    assert len(results) == 1, results
    return results[0]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="benchmark-qualification-") as temp_dir:
        root = Path(temp_dir)
        first = run_benchmark(root / "first", seed=7)
        second = run_benchmark(root / "second", seed=8)
        report_dir = root / "report"
        subprocess.run(
            [
                sys.executable,
                "benchmark_report.py",
                "--input",
                str(first),
                "--input",
                str(second),
                "--output-dir",
                str(report_dir),
                "--max-p95-regression-pct",
                "1000",
                "--max-success-rate-drop",
                "1",
                "--max-throughput-drop-pct",
                "1000",
                "--max-attempt-amplification-increase",
                "1000",
                "--fail-on-regression",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        report = report_dir / "benchmark_trend.json"
        manifest = root / "qualification_manifest.json"

        create = subprocess.run(
            [
                sys.executable,
                "qualification_manifest.py",
                "create",
                "--report",
                str(report),
                "--input",
                str(first),
                "--input",
                str(second),
                "--output",
                str(manifest),
                "--require-pass",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        verify = subprocess.run(
            [
                sys.executable,
                "qualification_manifest.py",
                "verify",
                "--manifest",
                str(manifest),
                "--report",
                str(report),
                "--input",
                str(first),
                "--input",
                str(second),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        serialized = manifest.read_text(encoding="utf-8")
        assert manifest_data["kind"] == "benchmark_qualification_manifest"
        assert manifest_data["gate_policy"]["require_trend_pass"] is True
        assert manifest_data["trend_report"]["regression"] is False
        assert len(manifest_data["input_artifacts"]) == 2
        assert all(len(entry["sha256"]) == 64 for entry in manifest_data["input_artifacts"])
        assert "server_url" not in serialized
        assert "openai_prompt" not in serialized
        assert "Authorization" not in serialized
        assert str(root) not in serialized
        assert json.loads(verify.stdout)["verified"] is True
        assert "Saved qualification manifest" in create.stdout

        source_metrics = json.loads(first.read_text(encoding="utf-8"))
        paired_inputs: list[Path] = []
        for group in ("baseline", "candidate"):
            for index in range(7):
                metrics = json.loads(json.dumps(source_metrics))
                baseline_p95 = source_metrics["latency_ms"]["p95"] * 1.09**index
                metrics["latency_ms"]["p95"] = (
                    baseline_p95 if group == "baseline" else baseline_p95 * 1.11
                )
                metrics["latency_ms"]["p99"] = max(
                    source_metrics["latency_ms"]["p99"],
                    metrics["latency_ms"]["p95"],
                )
                artifact = root / f"{group}-{index}.json"
                artifact.write_text(json.dumps(metrics), encoding="utf-8")
                paired_inputs.append(artifact)

        paired_report_dir = root / "paired-report"
        paired_command = [sys.executable, "benchmark_report.py"]
        for artifact in paired_inputs:
            paired_command.extend(["--input", str(artifact)])
        paired_command.extend(
            ["--paired-baseline-runs", "7", "--output-dir", str(paired_report_dir)]
        )
        subprocess.run(
            paired_command,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        paired_report = paired_report_dir / "benchmark_trend.json"
        paired_data = json.loads(paired_report.read_text(encoding="utf-8"))
        assert paired_data["regression"] is False
        assert paired_data["statistical_regression"] is True

        paired_manifest = root / "paired-qualification-manifest.json"
        paired_create = [
            sys.executable,
            "qualification_manifest.py",
            "create",
            "--report",
            str(paired_report),
        ]
        for artifact in paired_inputs:
            paired_create.extend(["--input", str(artifact)])
        paired_create.extend(["--output", str(paired_manifest)])
        subprocess.run(
            paired_create,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        paired_verify = [
            sys.executable,
            "qualification_manifest.py",
            "verify",
            "--manifest",
            str(paired_manifest),
            "--report",
            str(paired_report),
        ]
        for artifact in paired_inputs:
            paired_verify.extend(["--input", str(artifact)])
        paired_verified = subprocess.run(
            paired_verify,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        paired_manifest_data = json.loads(paired_manifest.read_text(encoding="utf-8"))
        assert paired_manifest_data["schema_version"] == 2
        assert paired_manifest_data["trend_report"]["statistical_regression"] is True
        assert json.loads(paired_verified.stdout)["statistical_regression"] is True

        rejected_manifest = root / "rejected-manifest.json"
        require_statistical_pass = paired_create[:-2] + [
            "--output",
            str(rejected_manifest),
            "--require-statistical-pass",
        ]
        rejected = subprocess.run(
            require_statistical_pass,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert rejected.returncode == 2, rejected
        assert not rejected_manifest.exists()

        first.write_text(first.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        tampered = subprocess.run(
            [
                sys.executable,
                "qualification_manifest.py",
                "verify",
                "--manifest",
                str(manifest),
                "--report",
                str(report),
                "--input",
                str(first),
                "--input",
                str(second),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert tampered.returncode == 2, tampered
        assert "qualification manifest error" in tampered.stderr

    print("qualification manifest CLI fixture passed")


if __name__ == "__main__":
    main()
