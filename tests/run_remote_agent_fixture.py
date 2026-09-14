"""Exercise authenticated remote-agent coordination against a local SSE fixture."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remote_agent import probe_agent_clock, run_agent_benchmark

TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$"
)
SERIALIZED_TRACEPARENT_PATTERN = re.compile(
    r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}"
)
API_KEY_ENV = "BENCHMARK_AGENT_FIXTURE_KEY"
API_KEY = "fixture-agent-key-32-characters-long"


class OneShotRunResponseDropProxy(ThreadingHTTPServer):
    """Forward agent calls but discard the first complete run response."""

    def __init__(self, target_url: str) -> None:
        super().__init__(("127.0.0.1", 0), OneShotRunResponseDropHandler)
        self.target_url = target_url
        self.dropped_run_responses = 0


class OneShotRunResponseDropHandler(BaseHTTPRequestHandler):
    server: OneShotRunResponseDropProxy

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        request = urllib.request.Request(
            self.server.target_url + self.path,
            data=body,
            headers={
                "Authorization": self.headers.get("Authorization", ""),
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                status = response.status
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            response_body = exc.read()

        if self.path == "/v1/run" and self.server.dropped_run_responses == 0:
            self.server.dropped_run_responses += 1
            self.close_connection = True
            return

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response_body)


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


def assert_wrong_key_is_rejected(agent_url: str) -> None:
    body = json.dumps(
        {"schema_version": 1, "challenge": "a" * 64}, separators=(",", ":")
    ).encode("utf-8")
    request = urllib.request.Request(
        agent_url + "/v1/clock",
        data=body,
        headers={
            "Authorization": "Bearer definitely-the-wrong-agent-key",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=3)
    except urllib.error.HTTPError as exc:
        assert exc.code == 401, exc.code
    else:
        raise AssertionError("agent accepted an invalid bearer key")


def assert_idempotent_repeat_and_conflict(agent_url: str) -> None:
    profile = probe_agent_clock(
        agent_url,
        api_key=API_KEY,
        sample_count=3,
        timeout_seconds=3,
    )
    agent_hash = profile["agent_id_sha256"]
    offset_ns = profile["clock_offset_agent_minus_coordinator_ns"]
    assert isinstance(agent_hash, str)
    assert isinstance(offset_ns, int)
    arguments = ["--mode", "mock", "--num-requests", "2", "--concurrency", "1"]
    planned_agent_ns = time.time_ns() + 200_000_000 + offset_ns
    first = run_agent_benchmark(
        agent_url,
        api_key=API_KEY,
        expected_agent_id_sha256=agent_hash,
        run_id="fixture-replay-identity",
        client_index=0,
        client_count=2,
        planned_start_agent_unix_ns=planned_agent_ns,
        benchmark_args=arguments,
        timeout_seconds=10,
    )
    assert first["successful_requests"] == 2
    repeated = run_agent_benchmark(
        agent_url,
        api_key=API_KEY,
        expected_agent_id_sha256=agent_hash,
        run_id="fixture-replay-identity",
        client_index=0,
        client_count=2,
        planned_start_agent_unix_ns=planned_agent_ns,
        benchmark_args=arguments,
        timeout_seconds=10,
    )
    assert repeated["successful_requests"] == 2
    assert repeated["_remote_result_delivery"]["result_source"] == "cached"

    try:
        run_agent_benchmark(
            agent_url,
            api_key=API_KEY,
            expected_agent_id_sha256=agent_hash,
            run_id="fixture-replay-identity",
            client_index=0,
            client_count=2,
            planned_start_agent_unix_ns=planned_agent_ns,
            benchmark_args=[*arguments, "--seed", "99"],
            timeout_seconds=10,
        )
    except RuntimeError as exc:
        assert "HTTP 409" in str(exc), str(exc)
    else:
        raise AssertionError("agent accepted one identity for a different request")


def main() -> None:
    fixture_process: subprocess.Popen[str] | None = None
    agent_processes: list[subprocess.Popen[str]] = []
    drop_proxy: OneShotRunResponseDropProxy | None = None
    drop_proxy_thread: threading.Thread | None = None
    with tempfile.TemporaryDirectory(prefix="remote-agent-fixture-") as temp_dir:
        temp_path = Path(temp_dir)
        fixture_port_path = temp_path / "fixture.port"
        trace_path = temp_path / "fixture.trace"
        result_dir = temp_path / "results"
        prompt = "remote agent private fixture prompt"
        child_env = dict(os.environ)
        child_env[API_KEY_ENV] = API_KEY
        try:
            fixture_process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests" / "openai_fixture_server.py"),
                    "--port-file",
                    str(fixture_port_path),
                    "--trace-file",
                    str(trace_path),
                ],
                cwd=ROOT,
                text=True,
            )
            target_port = wait_for_port_file(fixture_port_path, fixture_process)
            endpoint = f"http://127.0.0.1:{target_port}"

            agent_urls: list[str] = []
            for index in range(2):
                port_path = temp_path / f"agent-{index}.port"
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "remote_agent.py"),
                        "--listen-host",
                        "127.0.0.1",
                        "--port",
                        "0",
                        "--port-file",
                        str(port_path),
                        "--agent-id",
                        f"loopback-agent-{index}",
                        "--api-key-env",
                        API_KEY_ENV,
                        "--child-timeout-seconds",
                        "30",
                    ],
                    cwd=ROOT,
                    env=child_env,
                    text=True,
                )
                agent_processes.append(process)
                port = wait_for_port_file(port_path, process)
                agent_urls.append(f"http://127.0.0.1:{port}")

            assert_wrong_key_is_rejected(agent_urls[0])
            assert_idempotent_repeat_and_conflict(agent_urls[0])

            drop_proxy = OneShotRunResponseDropProxy(agent_urls[0])
            drop_proxy_thread = threading.Thread(
                target=drop_proxy.serve_forever, daemon=True
            )
            drop_proxy_thread.start()
            coordinator_agent_urls = [
                f"http://127.0.0.1:{drop_proxy.server_port}",
                agent_urls[1],
            ]

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
                "--max-clock-uncertainty-ms",
                "100",
                "--clock-samples",
                "5",
                "--timeout-seconds",
                "30",
                "--agent-run-recovery-attempts",
                "1",
                "--agent-url",
                coordinator_agent_urls[0],
                "--agent-url",
                coordinator_agent_urls[1],
                "--agent-api-key-env",
                API_KEY_ENV,
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
                env=child_env,
                text=True,
                capture_output=True,
                timeout=40,
            )
            if completed.returncode:
                raise RuntimeError(
                    "remote coordinator CLI failed\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            aggregate_path = result_dir / "coordinated_benchmark.json"
            prometheus_path = result_dir / "coordinated_benchmark.prom"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            prometheus = prometheus_path.read_text(encoding="utf-8")
            serialized = aggregate_path.read_text(encoding="utf-8") + prometheus

            assert aggregate["scope"] == "authenticated_remote_agents"
            assert aggregate["schema_version"] == 2
            assert aggregate["client_count"] == 2
            assert aggregate["client_indexes"] == [0, 1]
            assert len(aggregate["agent_identity_fingerprints_sha256"]) == 2
            assert len(set(aggregate["agent_identity_fingerprints_sha256"])) == 2
            assert aggregate["logical_requests"] == 8
            assert aggregate["successful_requests"] == 8
            assert aggregate["failed_requests"] == 0
            assert aggregate["client_attempts"] == 8
            assert aggregate["configured_aggregate_request_rate_rps"] == 20
            assert aggregate["result_delivery"] == {
                "max_transport_recovery_attempts_per_agent": 1,
                "transport_retries": 1,
                "executed_results": 1,
                "cached_results": 1,
                "durable_results": 0,
                "recovered_after_transport_failure": 1,
            }
            assert aggregate["clock_quality"]["passed"] is True
            assert aggregate["clock_quality"]["max_uncertainty_ms"] <= 100
            assert all(
                agent["clock_sample_count"] == 5
                for agent in aggregate["clock_quality"]["agents"]
            )
            assert aggregate["window"]["overlap_duration_lower_bound_seconds"] > 0
            assert aggregate["window"]["start_skew_upper_bound_ms"] <= 300
            assert aggregate["throughput"]["observed_normalized_rps"] > 0
            assert aggregate["throughput"]["conservative_lower_bound_rps"] > 0
            assert aggregate["coordination_gate"]["passed"] is True
            assert aggregate["latency"]["global_percentiles_available"] is False
            assert aggregate["privacy"]["agent_urls_persisted"] is False
            assert aggregate["privacy"]["authorization_persisted"] is False
            assert aggregate["privacy"]["clock_challenges_persisted"] is False
            assert aggregate["privacy"]["raw_agent_ids_persisted"] is False

            for private_value in (
                endpoint,
                prompt,
                API_KEY,
                API_KEY_ENV,
                *agent_urls,
                *coordinator_agent_urls,
                "loopback-agent-0",
                "loopback-agent-1",
                str(result_dir),
            ):
                assert private_value not in serialized, private_value
            assert SERIALIZED_TRACEPARENT_PATTERN.search(serialized) is None
            assert "triton_coordinated_clock_uncertainty_max_ms" in prometheus
            assert "triton_coordinated_start_skew_upper_bound_ms" in prometheus
            assert "triton_coordinated_throughput_lower_bound_rps" in prometheus
            assert "triton_coordinated_agent_transport_retries_total 1" in prometheus
            assert "triton_coordinated_agent_cached_results_total 1" in prometheus
            assert "triton_coordinated_agent_durable_results_total 0" in prometheus
            assert "triton_coordinated_agent_recovered_results_total 1" in prometheus
            assert "triton_coordinated_gate_passed 1" in prometheus
            assert sorted(path.name for path in result_dir.iterdir()) == [
                "coordinated_benchmark.json",
                "coordinated_benchmark.prom",
            ]

            traceparents = trace_path.read_text(encoding="utf-8").splitlines()
            assert len(traceparents) == 8, traceparents
            assert len(set(traceparents)) == 8, traceparents
            assert all(TRACEPARENT_PATTERN.fullmatch(value) for value in traceparents)
            assert all(value not in serialized for value in traceparents)
            assert drop_proxy.dropped_run_responses == 1

            print(
                json.dumps(
                    {
                        "agent_services": aggregate["client_count"],
                        "logical_requests": aggregate["logical_requests"],
                        "successful_requests": aggregate["successful_requests"],
                        "unique_traceparents": len(set(traceparents)),
                        "max_clock_uncertainty_ms": aggregate["clock_quality"][
                            "max_uncertainty_ms"
                        ],
                        "observed_start_skew_ms": aggregate["window"][
                            "observed_start_skew_ms"
                        ],
                        "start_skew_upper_bound_ms": aggregate["window"][
                            "start_skew_upper_bound_ms"
                        ],
                        "overlap_lower_bound_seconds": aggregate["window"][
                            "overlap_duration_lower_bound_seconds"
                        ],
                        "coordination_gate": aggregate["coordination_gate"]["passed"],
                        "wrong_key_rejected": True,
                        "conflicting_replay_rejected": True,
                        "ambiguous_response_dropped": True,
                        "transport_retries": aggregate["result_delivery"][
                            "transport_retries"
                        ],
                        "cached_results": aggregate["result_delivery"][
                            "cached_results"
                        ],
                    },
                    indent=2,
                )
            )
        finally:
            if drop_proxy is not None:
                drop_proxy.shutdown()
                drop_proxy.server_close()
            if drop_proxy_thread is not None:
                drop_proxy_thread.join(timeout=2)
            for process in agent_processes:
                stop_process(process)
            stop_process(fixture_process)


if __name__ == "__main__":
    main()
