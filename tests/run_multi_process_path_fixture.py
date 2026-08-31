"""Exercise the real CLI against separate router and backend fixture processes."""

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
    raise RuntimeError(f"fixture process did not write a port file: {path.name}")


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def validate_artifacts(
    result_dir: Path,
    router_trace_path: Path,
    backend_trace_path: Path,
    backend_state_path: Path,
    backend_port: int,
    supervisor_port: int,
) -> dict[str, object]:
    results = list(result_dir.glob("benchmark_*.json"))
    if len(results) != 1:
        raise AssertionError(f"expected one JSON artifact, found: {results}")
    result_path = results[0]
    prometheus_path = result_path.with_suffix(".prom")
    metrics = json.loads(result_path.read_text(encoding="utf-8"))
    prometheus = prometheus_path.read_text(encoding="utf-8")
    router_traceparents = router_trace_path.read_text(encoding="utf-8").splitlines()
    backend_traceparents = backend_trace_path.read_text(encoding="utf-8").splitlines()

    assert len(router_traceparents) == 6, router_traceparents
    assert len(set(router_traceparents)) == 6, router_traceparents
    assert all(TRACEPARENT_PATTERN.fullmatch(value) for value in router_traceparents)
    assert len(backend_traceparents) == 5, backend_traceparents
    assert backend_traceparents == router_traceparents[1:]

    retry = metrics["retry"]
    assert retry["logical_requests"] == 4
    assert retry["client_attempts"] == 6
    assert retry["retry_attempts"] == 2
    assert retry["retried_requests"] == 2
    assert retry["recovered_requests"] == 2
    assert retry["exhausted_requests"] == 0
    assert retry["client_attempt_amplification"] == 1.5
    recovery_latency = retry["recovered_request_latency_ms"]
    assert recovery_latency["request_count"] == 2
    assert recovery_latency["p95"] >= 0
    assert "not service MTTR" in recovery_latency["note"]

    request_path = metrics["request_path"]
    path_deltas = {
        stage: summary["delta"]
        for stage, summary in request_path["stages"].items()
    }
    assert path_deltas == {"ingress": 6, "backend": 5, "success": 4}
    assert request_path["endpoint_count"] == 3
    assert request_path["endpoint_urls_persisted"] is False
    assert metrics["telemetry"]["endpoint_count"] == 3
    assert metrics["telemetry_window"]["endpoint_count"] == 3
    assert metrics["request_path_gate"]["passed"] is True
    assert metrics["retry_gate"]["passed"] is True
    assert metrics["trace_context_gate"]["passed"] is True
    assert metrics["config"]["retry_backoff_seconds"] == 0.25

    backend_state = json.loads(backend_state_path.read_text(encoding="utf-8"))
    assert backend_state == {
        "backend": 5,
        "crash_triggered": True,
        "success": 4,
    }

    service_lifecycle = metrics["service_lifecycle"]
    assert service_lifecycle["valid"] is True
    assert service_lifecycle["restart_count"] == 1
    assert service_lifecycle["endpoint_count"] == 3
    assert service_lifecycle["endpoint_urls_persisted"] is False
    assert metrics["service_lifecycle_gate"]["passed"] is True

    assert "triton_benchmark_recovered_request_latency_ms" in prometheus
    assert "triton_benchmark_service_restart_delta" in prometheus
    assert "triton_benchmark_service_lifecycle_gate_passed" in prometheus
    assert 'stage="ingress"} 6' in prometheus
    assert 'stage="backend"} 5' in prometheus
    assert 'stage="success"} 4' in prometheus
    serialized = result_path.read_text(encoding="utf-8") + prometheus
    assert "fixture_ingress_requests_total" not in serialized
    assert "fixture_backend_requests_total" not in serialized
    assert "fixture_backend_success_total" not in serialized
    assert "fixture_backend_restarts_total" not in serialized
    assert 'service="router"' not in serialized
    assert 'service="backend"' not in serialized
    assert 'service="supervisor"' not in serialized
    assert f"127.0.0.1:{backend_port}" not in serialized
    assert f"127.0.0.1:{supervisor_port}" not in serialized
    assert "/metrics" not in serialized
    assert SERIALIZED_TRACEPARENT_PATTERN.search(serialized) is None
    assert all(value not in serialized for value in router_traceparents)
    assert "Return a short deterministic benchmark response." not in serialized
    assert "Authorization" not in serialized

    return {
        "client_attempts": retry["client_attempts"],
        "recovered_requests": retry["recovered_requests"],
        "recovered_request_latency_ms": recovery_latency,
        "path_deltas": path_deltas,
        "telemetry_endpoint_count": request_path["endpoint_count"],
        "completed_service_restarts": service_lifecycle["restart_count"],
        "controlled_backend_crash_triggered": backend_state["crash_triggered"],
        "gates": {
            "request_path": metrics["request_path_gate"]["passed"],
            "retry": metrics["retry_gate"]["passed"],
            "trace_context": metrics["trace_context_gate"]["passed"],
            "service_lifecycle": metrics["service_lifecycle_gate"]["passed"],
        },
    }


def main() -> None:
    supervisor_process: subprocess.Popen[str] | None = None
    router_process: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory(prefix="triton-path-fixture-") as temp_dir:
        temp_path = Path(temp_dir)
        backend_port_path = temp_path / "backend.port"
        supervisor_port_path = temp_path / "supervisor.port"
        router_port_path = temp_path / "router.port"
        backend_trace_path = temp_path / "backend.trace"
        backend_state_path = temp_path / "backend.state.json"
        router_trace_path = temp_path / "router.trace"
        result_dir = temp_path / "results"
        try:
            supervisor_process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests" / "openai_backend_supervisor.py"),
                    "--port-file",
                    str(supervisor_port_path),
                    "--backend-port-file",
                    str(backend_port_path),
                    "--backend-trace-file",
                    str(backend_trace_path),
                    "--backend-state-file",
                    str(backend_state_path),
                    "--crash-request-number",
                    "2",
                ],
                cwd=ROOT,
                text=True,
            )
            supervisor_port = wait_for_port_file(
                supervisor_port_path,
                supervisor_process,
            )
            backend_port = wait_for_port_file(backend_port_path, supervisor_process)

            router_process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests" / "openai_path_router_server.py"),
                    "--port-file",
                    str(router_port_path),
                    "--trace-file",
                    str(router_trace_path),
                    "--backend-url",
                    f"http://127.0.0.1:{backend_port}/v1/completions",
                    "--fail-request-number",
                    "1",
                ],
                cwd=ROOT,
                text=True,
            )
            router_port = wait_for_port_file(router_port_path, router_process)

            command = [
                sys.executable,
                str(ROOT / "benchmark.py"),
                "--mode",
                "openai",
                "--server-url",
                f"http://127.0.0.1:{router_port}",
                "--model-name",
                "fixture-model",
                "--num-requests",
                "4",
                "--concurrency",
                "1",
                "--retries",
                "2",
                "--retry-backoff-seconds",
                "0.25",
                "--propagate-trace-context",
                "--fail-on-trace-context-gap",
                "--telemetry-url",
                f"http://127.0.0.1:{router_port}/metrics",
                "--telemetry-url",
                f"http://127.0.0.1:{backend_port}/metrics",
                "--telemetry-url",
                f"http://127.0.0.1:{supervisor_port}/metrics",
                "--request-path-ingress-metric",
                "fixture_ingress_requests_total",
                "--request-path-backend-metric",
                "fixture_backend_requests_total",
                "--request-path-success-metric",
                "fixture_backend_success_total",
                "--fail-on-request-path-gap",
                "--service-restart-metric",
                "fixture_backend_restarts_total",
                "--min-service-restarts",
                "1",
                "--max-service-restarts",
                "1",
                "--fail-on-service-lifecycle-gap",
                "--max-client-attempt-amplification",
                "1.5",
                "--fail-on-retry-gate",
                "--prometheus",
                "--output-dir",
                str(result_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if completed.returncode:
                raise RuntimeError(
                    "benchmark CLI failed\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            summary = validate_artifacts(
                result_dir,
                router_trace_path,
                backend_trace_path,
                backend_state_path,
                backend_port,
                supervisor_port,
            )
            print(json.dumps(summary, indent=2))
        finally:
            stop_process(router_process)
            stop_process(supervisor_process)


if __name__ == "__main__":
    main()
