"""Prove completed-shard reconciliation after a coordinator process restart."""

from __future__ import annotations

import json
import os
import re
import stat
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

from coordinated_benchmark import derive_coordinator_run_id

TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$"
)
SERIALIZED_TRACEPARENT_PATTERN = re.compile(
    r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}"
)
API_KEY_ENV = "BENCHMARK_COORDINATOR_RESTART_AGENT_KEY"
API_KEY = "coordinator-restart-agent-key-32-chars"
RESUME_TOKEN_ENV = "BENCHMARK_COORDINATOR_RESUME_TOKEN"
RESUME_TOKEN = "coordinator-restart-resume-token-32-characters"
PROMPT = "private coordinator restart fixture prompt"


class HoldCompletedRunResponseProxy(ThreadingHTTPServer):
    """Forward agent calls but withhold completed run responses until released."""

    def __init__(self, target_url: str) -> None:
        super().__init__(("127.0.0.1", 0), HoldCompletedRunResponseHandler)
        self.target_url = target_url
        self.completed_run_responses = 0
        self.completed = threading.Event()
        self.release = threading.Event()


class HoldCompletedRunResponseHandler(BaseHTTPRequestHandler):
    server: HoldCompletedRunResponseProxy

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

        if self.path == "/v1/run" and status == 200:
            self.server.completed_run_responses += 1
            self.server.completed.set()
            self.server.release.wait(timeout=35)
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


def coordinator_command(
    output_dir: Path,
    state_path: Path,
    agent_urls: list[str],
    endpoint: str,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "coordinated_benchmark.py"),
        "--clients",
        "2",
        "--output-dir",
        str(output_dir),
        "--lead-time-ms",
        "500",
        "--max-start-skew-ms",
        "300",
        "--max-clock-uncertainty-ms",
        "100",
        "--clock-samples",
        "5",
        "--timeout-seconds",
        "30",
        "--agent-run-recovery-attempts",
        "0",
        "--coordinator-state-file",
        str(state_path),
        "--coordinator-resume-token-env",
        RESUME_TOKEN_ENV,
        "--agent-url",
        agent_urls[0],
        "--agent-url",
        agent_urls[1],
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
        PROMPT,
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


def main() -> None:
    fixture_process: subprocess.Popen[str] | None = None
    first_coordinator: subprocess.Popen[str] | None = None
    agent_processes: list[subprocess.Popen[str]] = []
    proxies: list[HoldCompletedRunResponseProxy] = []
    proxy_threads: list[threading.Thread] = []
    with tempfile.TemporaryDirectory(prefix="coordinator-restart-fixture-") as temp_dir:
        temp_path = Path(temp_dir)
        fixture_port_path = temp_path / "fixture.port"
        trace_path = temp_path / "fixture.trace"
        output_dir = temp_path / "results"
        state_path = temp_path / "coordinator-state.json"
        child_env = dict(os.environ)
        child_env[API_KEY_ENV] = API_KEY
        child_env[RESUME_TOKEN_ENV] = RESUME_TOKEN
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
                        f"coordinator-restart-agent-{index}",
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

            proxy_urls: list[str] = []
            for agent_url in agent_urls:
                proxy = HoldCompletedRunResponseProxy(agent_url)
                thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                thread.start()
                proxies.append(proxy)
                proxy_threads.append(thread)
                proxy_urls.append(f"http://127.0.0.1:{proxy.server_port}")

            first_coordinator = subprocess.Popen(
                coordinator_command(output_dir, state_path, proxy_urls, endpoint),
                cwd=ROOT,
                env=child_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                if first_coordinator.poll() is not None:
                    stdout, stderr = first_coordinator.communicate()
                    raise RuntimeError(
                        "first coordinator exited before both responses were held\n"
                        f"stdout:\n{stdout}\nstderr:\n{stderr}"
                    )
                if state_path.is_file() and all(proxy.completed.is_set() for proxy in proxies):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("agents did not complete before fixture deadline")

            traceparents_before_restart = trace_path.read_text(
                encoding="utf-8"
            ).splitlines()
            assert len(traceparents_before_restart) == 8, traceparents_before_restart
            assert len(set(traceparents_before_restart)) == 8
            assert all(
                TRACEPARENT_PATTERN.fullmatch(value)
                for value in traceparents_before_restart
            )
            assert all(proxy.completed_run_responses == 1 for proxy in proxies)
            assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
            assert output_dir.is_dir() and not any(output_dir.iterdir())

            first_coordinator.terminate()
            first_coordinator.wait(timeout=3)
            first_return_code = first_coordinator.returncode
            first_coordinator = None
            for proxy in proxies:
                proxy.release.set()
                proxy.shutdown()
                proxy.server_close()
            for thread in proxy_threads:
                thread.join(timeout=2)
            proxies.clear()
            proxy_threads.clear()

            completed = subprocess.run(
                coordinator_command(output_dir, state_path, agent_urls, endpoint),
                cwd=ROOT,
                env=child_env,
                text=True,
                capture_output=True,
                timeout=40,
            )
            if completed.returncode:
                raise RuntimeError(
                    "resumed coordinator CLI failed\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            aggregate_path = output_dir / "coordinated_benchmark.json"
            prometheus_path = output_dir / "coordinated_benchmark.prom"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            prometheus = prometheus_path.read_text(encoding="utf-8")
            state_text = state_path.read_text(encoding="utf-8")
            state = json.loads(state_text)
            serialized = (
                aggregate_path.read_text(encoding="utf-8")
                + prometheus
                + state_text
            )

            assert first_return_code is not None and first_return_code != 0
            assert aggregate["scope"] == "authenticated_remote_agents"
            assert aggregate["client_count"] == 2
            assert aggregate["logical_requests"] == 8
            assert aggregate["successful_requests"] == 8
            assert aggregate["failed_requests"] == 0
            assert aggregate["result_delivery"] == {
                "max_transport_recovery_attempts_per_agent": 0,
                "transport_retries": 0,
                "executed_results": 0,
                "cached_results": 2,
                "durable_results": 0,
                "recovered_after_transport_failure": 0,
            }
            assert aggregate["coordinator_recovery"] == {
                "state_enabled": True,
                "resumed_after_process_restart": True,
                "completed_statuses_verified_before_retrieval": 2,
                "missing_or_incomplete_shards_launched": False,
                "state_authentication": "hmac_sha256_explicit_resume_token",
                "resume_scope": "completed_shards_only",
            }
            assert aggregate["coordination_gate"]["passed"] is True
            assert aggregate["privacy"]["coordinator_state_path_persisted"] is False
            assert aggregate["privacy"]["coordinator_resume_token_persisted"] is False
            assert aggregate["privacy"]["coordinator_benchmark_arguments_persisted"] is False
            assert aggregate["privacy"]["coordinator_result_bodies_persisted"] is False
            assert state["privacy"]["result_bodies_persisted"] is False
            assert "completed-result status verification" in aggregate["claim_boundary"]

            run_id = derive_coordinator_run_id(RESUME_TOKEN, state["workflow_nonce"])
            for private_value in (
                API_KEY,
                API_KEY_ENV,
                RESUME_TOKEN,
                RESUME_TOKEN_ENV,
                PROMPT,
                endpoint,
                *agent_urls,
                *proxy_urls,
                "coordinator-restart-agent-0",
                "coordinator-restart-agent-1",
                str(output_dir),
                str(state_path),
                run_id,
            ):
                assert private_value not in serialized, private_value
            assert '"benchmark_args"' not in state_text
            assert SERIALIZED_TRACEPARENT_PATTERN.search(serialized) is None
            assert "triton_coordinated_coordinator_state_enabled 1" in prometheus
            assert "triton_coordinated_coordinator_resumed 1" in prometheus
            assert "triton_coordinated_resume_statuses_verified 2" in prometheus
            assert "triton_coordinated_agent_cached_results_total 2" in prometheus
            assert sorted(path.name for path in output_dir.iterdir()) == [
                "coordinated_benchmark.json",
                "coordinated_benchmark.prom",
            ]

            traceparents_after_resume = trace_path.read_text(
                encoding="utf-8"
            ).splitlines()
            assert traceparents_after_resume == traceparents_before_restart

            wrong_token_env = dict(child_env)
            wrong_token_env[RESUME_TOKEN_ENV] = (
                "different-coordinator-resume-token-32-characters"
            )
            rejected = subprocess.run(
                coordinator_command(
                    temp_path / "wrong-token-results",
                    state_path,
                    agent_urls,
                    endpoint,
                ),
                cwd=ROOT,
                env=wrong_token_env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            assert rejected.returncode != 0
            assert "authentication failed" in rejected.stderr
            assert trace_path.read_text(encoding="utf-8").splitlines() == (
                traceparents_before_restart
            )

            print(
                json.dumps(
                    {
                        "coordinator_processes": 2,
                        "coordinator_restarts": 1,
                        "agent_services": aggregate["client_count"],
                        "completed_statuses_verified": aggregate[
                            "coordinator_recovery"
                        ]["completed_statuses_verified_before_retrieval"],
                        "logical_requests": aggregate["logical_requests"],
                        "successful_requests": aggregate["successful_requests"],
                        "unique_traceparents": len(set(traceparents_after_resume)),
                        "cached_results_retrieved": aggregate["result_delivery"][
                            "cached_results"
                        ],
                        "duplicate_target_requests": 0,
                        "wrong_resume_token_rejected": True,
                        "state_mode": oct(stat.S_IMODE(state_path.stat().st_mode)),
                        "private_values_persisted": False,
                        "coordination_gate": aggregate["coordination_gate"]["passed"],
                    },
                    indent=2,
                )
            )
        finally:
            stop_process(first_coordinator)
            for proxy in proxies:
                proxy.release.set()
                proxy.shutdown()
                proxy.server_close()
            for thread in proxy_threads:
                thread.join(timeout=2)
            for process in agent_processes:
                stop_process(process)
            stop_process(fixture_process)


if __name__ == "__main__":
    main()
