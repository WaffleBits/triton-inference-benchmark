"""Exercise two coordinated benchmark CLIs against one deterministic SSE fixture."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$"
)
SERIALIZED_TRACEPARENT_PATTERN = re.compile(
    r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}"
)


def wait_for_port_file(path: Path, process: subprocess.Popen[str]) -> int:
    for _ in range(100):
        if path.is_file() and path.stat().st_size:
            return int(path.read_text(encoding="utf-8"))
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"fixture process exited before readiness with status {return_code}"
            )
        time.sleep(0.05)
    raise RuntimeError("fixture process did not become ready")


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def main() -> None:
    fixture_process: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory(prefix="coordinated-client-fixture-") as temp_dir:
        temp_path = Path(temp_dir)
        port_path = temp_path / "fixture.port"
        trace_path = temp_path / "fixture.trace"
        result_dir = temp_path / "results"
        prompt = "coordinated private fixture prompt"
        try:
            fixture_process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests" / "openai_fixture_server.py"),
                    "--port-file",
                    str(port_path),
                    "--trace-file",
                    str(trace_path),
                ],
                cwd=ROOT,
                text=True,
            )
            port = wait_for_port_file(port_path, fixture_process)
            endpoint = f"http://127.0.0.1:{port}"

            command = [
                sys.executable,
                str(ROOT / "coordinated_benchmark.py"),
                "--clients",
                "2",
                "--output-dir",
                str(result_dir),
                "--lead-time-ms",
                "500",
                # Keep fixture acceptance bounded but tolerant of constrained CI hosts;
                # production thresholds remain operator-selected and fail closed.
                "--max-start-skew-ms",
                "300",
                "--timeout-seconds",
                "30",
                "--",
                "--mode",
                "openai",
                "--server-url",
                endpoint,
                "--model-name",
                "fixture-model",
                "--openai-prompt",
                prompt,
                "--num-requests",
                "4",
                "--concurrency",
                "1",
                "--retries",
                "0",
                "--request-rate-rps",
                "10",
                "--propagate-trace-context",
                "--fail-on-trace-context-gap",
                "--prometheus",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=35,
            )
            if completed.returncode:
                raise RuntimeError(
                    "coordinated benchmark CLI failed\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            aggregate_path = result_dir / "coordinated_benchmark.json"
            prometheus_path = result_dir / "coordinated_benchmark.prom"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            prometheus = prometheus_path.read_text(encoding="utf-8")
            serialized_aggregate = aggregate_path.read_text(encoding="utf-8") + prometheus

            assert aggregate["scope"] == "single_host_multi_process"
            assert aggregate["client_count"] == 2
            assert aggregate["client_indexes"] == [0, 1]
            assert aggregate["logical_requests"] == 8
            assert aggregate["successful_requests"] == 8
            assert aggregate["failed_requests"] == 0
            assert aggregate["client_attempts"] == 8
            assert aggregate["retry_attempts"] == 0
            assert aggregate["configured_aggregate_request_rate_rps"] == 20
            assert aggregate["throughput_rps"] > 0
            assert aggregate["window"]["overlap_duration_seconds"] > 0
            assert aggregate["window"]["start_skew_ms"] <= 300
            assert aggregate["coordination_gate"]["passed"] is True
            assert aggregate["latency"]["global_percentiles_available"] is False
            assert aggregate["privacy"] == {
                "child_artifact_paths_persisted": False,
                "server_urls_persisted": False,
                "prompts_or_outputs_persisted": False,
                "trace_identifiers_persisted": False,
            }
            assert endpoint not in serialized_aggregate
            assert prompt not in serialized_aggregate
            assert str(result_dir) not in serialized_aggregate
            assert SERIALIZED_TRACEPARENT_PATTERN.search(serialized_aggregate) is None
            assert "triton_coordinated_clients 2" in prometheus
            assert 'triton_coordinated_requests_total{outcome="success"} 8' in prometheus
            assert "triton_coordinated_gate_passed 1" in prometheus

            child_artifacts = sorted(result_dir.glob("client-*/benchmark_*.json"))
            assert len(child_artifacts) == 2, child_artifacts
            child_metrics = [
                json.loads(path.read_text(encoding="utf-8")) for path in child_artifacts
            ]
            child_coordination = [metrics["coordination"] for metrics in child_metrics]
            assert {record["client_index"] for record in child_coordination} == {0, 1}
            assert len({record["run_id_sha256"] for record in child_coordination}) == 1
            assert len(
                {record["config_fingerprint_sha256"] for record in child_coordination}
            ) == 1
            assert all(record["run_id_persisted"] is False for record in child_coordination)
            assert all(prompt not in json.dumps(metrics) for metrics in child_metrics)

            traceparents = trace_path.read_text(encoding="utf-8").splitlines()
            assert len(traceparents) == 8, traceparents
            assert len(set(traceparents)) == 8, traceparents
            assert all(TRACEPARENT_PATTERN.fullmatch(value) for value in traceparents)
            assert all(value not in serialized_aggregate for value in traceparents)

            print(
                json.dumps(
                    {
                        "client_processes": aggregate["client_count"],
                        "logical_requests": aggregate["logical_requests"],
                        "client_attempts": aggregate["client_attempts"],
                        "successful_requests": aggregate["successful_requests"],
                        "configured_aggregate_request_rate_rps": aggregate[
                            "configured_aggregate_request_rate_rps"
                        ],
                        "start_skew_ms": aggregate["window"]["start_skew_ms"],
                        "overlap_duration_seconds": aggregate["window"][
                            "overlap_duration_seconds"
                        ],
                        "coordination_gate": aggregate["coordination_gate"]["passed"],
                        "unique_traceparents": len(set(traceparents)),
                        "global_latency_percentiles_available": aggregate["latency"][
                            "global_percentiles_available"
                        ],
                    },
                    indent=2,
                )
            )
        finally:
            stop_process(fixture_process)


if __name__ == "__main__":
    main()
