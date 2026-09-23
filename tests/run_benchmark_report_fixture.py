"""Exercise the saved trend-report CLI against real benchmark output."""

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
    with tempfile.TemporaryDirectory(prefix="benchmark-trend-fixture-") as temp_dir:
        root = Path(temp_dir)
        first_path = run_benchmark(root / "first", seed=7)
        second_path = run_benchmark(root / "second", seed=8)
        report_dir = root / "report"
        subprocess.run(
            [
                sys.executable,
                "benchmark_report.py",
                "--input",
                str(first_path),
                "--input",
                str(second_path),
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

        json_path = report_dir / "benchmark_trend.json"
        markdown_path = report_dir / "benchmark_trend.md"
        report = json.loads(json_path.read_text(encoding="utf-8"))
        serialized = json_path.read_text(encoding="utf-8") + markdown_path.read_text(
            encoding="utf-8"
        )

        assert report["schema_version"] == 1
        assert report["run_count"] == 2
        assert len(report["runs"]) == 2
        assert len(report["comparisons"]) == 1
        assert report["regression"] is False
        assert report["runs"][0]["num_requests"] == 8
        assert report["runs"][1]["num_requests"] == 8
        assert "server_url" not in serialized
        assert '"config":' not in serialized
        assert "openai_prompt" not in serialized
        assert "Return a short deterministic benchmark response." not in serialized
        assert str(root) not in serialized
        assert "source paths" in serialized
        assert "client-observed" in serialized

    print("saved benchmark trend report fixture passed")


if __name__ == "__main__":
    main()
