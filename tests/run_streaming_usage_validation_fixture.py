"""Exercise fail-closed streaming token-usage validation through the real CLI."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="streaming-usage-validation-") as temp_dir:
        root = Path(temp_dir)
        port_file = root / "port"
        trace_file = root / "traces"
        output_dir = root / "results"
        server = subprocess.Popen(
            [
                sys.executable,
                "tests/openai_fixture_server.py",
                "--port-file",
                str(port_file),
                "--trace-file",
                str(trace_file),
                "--invalid-token-usage",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while not port_file.exists() and time.monotonic() < deadline:
                if server.poll() is not None:
                    stderr = server.stderr.read() if server.stderr else ""
                    raise AssertionError(f"SSE fixture exited early: {stderr}")
                time.sleep(0.05)
            assert port_file.exists(), "SSE fixture did not become ready"
            port = port_file.read_text(encoding="utf-8").strip()

            subprocess.run(
                [
                    sys.executable,
                    "benchmark.py",
                    "--mode",
                    "openai",
                    "--server-url",
                    f"http://127.0.0.1:{port}/v1",
                    "--model-name",
                    "fixture-model",
                    "--num-requests",
                    "2",
                    "--concurrency",
                    "1",
                    "--retries",
                    "0",
                    "--propagate-trace-context",
                    "--prometheus",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )

            artifacts = list(output_dir.glob("benchmark_*.json"))
            assert len(artifacts) == 1, artifacts
            serialized = artifacts[0].read_text(encoding="utf-8")
            metrics = json.loads(serialized)
            streaming = metrics["streaming"]
            assert streaming["request_count"] == 2, streaming
            assert streaming["reported_token_request_count"] == 0, streaming
            assert streaming["missing_token_usage_requests"] == 0, streaming
            assert streaming["invalid_token_usage_requests"] == 2, streaming
            assert streaming["reported_output_tokens"] == 0, streaming
            assert streaming["output_tokens_per_second"] is None, streaming
            prometheus_path = artifacts[0].with_suffix(".prom")
            assert prometheus_path.is_file()
            prometheus = prometheus_path.read_text(encoding="utf-8")
            assert "triton_benchmark_streaming_invalid_token_usage_requests" in prometheus
            assert "triton_benchmark_streaming_missing_token_usage_requests" in prometheus
            assert "triton_benchmark_streaming_reported_token_requests" in prometheus
            assert "triton_benchmark_streaming_output_tokens_per_second{" not in prometheus
            serialized += prometheus
            assert "Authorization" not in serialized
            assert "fixture benchmark prompt" not in serialized
            assert "fixture response" not in serialized
            assert not re.search(
                r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", serialized
            )
            print("streaming usage validation CLI fixture passed")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    main()
