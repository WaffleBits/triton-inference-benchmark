"""Prove durable agent-result recovery across a real process restart."""

from __future__ import annotations

import json
import os
import re
import sqlite3
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

from remote_agent import AgentTransportError, probe_agent_clock, run_agent_benchmark

TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$"
)
API_KEY_ENV = "BENCHMARK_AGENT_RESTART_FIXTURE_KEY"
API_KEY = "restart-fixture-agent-key-32-chars"
AGENT_ID = "restart-fixture-agent"
RUN_ID = "private-restart-fixture-run"
PROMPT = "private durable restart fixture prompt"


class DropCompletedRunResponseProxy(ThreadingHTTPServer):
    """Forward one request through completion, then discard its response."""

    def __init__(self, target_url: str) -> None:
        super().__init__(("127.0.0.1", 0), DropCompletedRunResponseHandler)
        self.target_url = target_url
        self.dropped_run_responses = 0


class DropCompletedRunResponseHandler(BaseHTTPRequestHandler):
    server: DropCompletedRunResponseProxy

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


def start_agent(
    temp_path: Path,
    state_path: Path,
    child_env: dict[str, str],
    generation: int,
) -> tuple[subprocess.Popen[str], str]:
    port_path = temp_path / f"agent-{generation}.port"
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
            AGENT_ID,
            "--api-key-env",
            API_KEY_ENV,
            "--child-timeout-seconds",
            "30",
            "--state-db",
            str(state_path),
        ],
        cwd=ROOT,
        env=child_env,
        text=True,
    )
    port = wait_for_port_file(port_path, process)
    return process, f"http://127.0.0.1:{port}"


def read_traceparents(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    fixture_process: subprocess.Popen[str] | None = None
    agent_process: subprocess.Popen[str] | None = None
    proxy: DropCompletedRunResponseProxy | None = None
    proxy_thread: threading.Thread | None = None
    with tempfile.TemporaryDirectory(prefix="agent-restart-fixture-") as temp_dir:
        temp_path = Path(temp_dir)
        fixture_port_path = temp_path / "fixture.port"
        trace_path = temp_path / "fixture.trace"
        state_path = temp_path / "agent-state.sqlite3"
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

            agent_process, first_agent_url = start_agent(
                temp_path, state_path, child_env, generation=1
            )
            first_profile = probe_agent_clock(
                first_agent_url,
                api_key=API_KEY,
                sample_count=3,
                timeout_seconds=3,
            )
            agent_hash = first_profile["agent_id_sha256"]
            offset_ns = first_profile["clock_offset_agent_minus_coordinator_ns"]
            assert isinstance(agent_hash, str)
            assert isinstance(offset_ns, int)

            arguments = [
                "--mode",
                "openai",
                "--server-url",
                endpoint,
                "--model-name",
                "fixture-model",
                "--openai-prompt",
                PROMPT,
                "--num-requests",
                "2",
                "--concurrency",
                "1",
                "--retries",
                "0",
                "--propagate-trace-context",
                "--fail-on-trace-context-gap",
            ]
            planned_agent_ns = time.time_ns() + 500_000_000 + offset_ns
            proxy = DropCompletedRunResponseProxy(first_agent_url)
            proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            proxy_thread.start()

            try:
                run_agent_benchmark(
                    f"http://127.0.0.1:{proxy.server_port}",
                    api_key=API_KEY,
                    expected_agent_id_sha256=agent_hash,
                    run_id=RUN_ID,
                    client_index=0,
                    client_count=2,
                    planned_start_agent_unix_ns=planned_agent_ns,
                    benchmark_args=arguments,
                    timeout_seconds=30,
                    recovery_attempts=0,
                )
            except AgentTransportError:
                pass
            else:
                raise AssertionError("proxy did not create an ambiguous response loss")

            assert proxy.dropped_run_responses == 1
            before_restart = read_traceparents(trace_path)
            assert len(before_restart) == 2, before_restart
            assert len(set(before_restart)) == 2, before_restart
            assert all(TRACEPARENT_PATTERN.fullmatch(value) for value in before_restart)

            stop_process(agent_process)
            agent_process = None
            proxy.shutdown()
            proxy.server_close()
            proxy = None
            proxy_thread.join(timeout=2)
            proxy_thread = None

            agent_process, second_agent_url = start_agent(
                temp_path, state_path, child_env, generation=2
            )
            second_profile = probe_agent_clock(
                second_agent_url,
                api_key=API_KEY,
                sample_count=3,
                timeout_seconds=3,
            )
            assert second_profile["agent_id_sha256"] == agent_hash

            recovered = run_agent_benchmark(
                second_agent_url,
                api_key=API_KEY,
                expected_agent_id_sha256=agent_hash,
                run_id=RUN_ID,
                client_index=0,
                client_count=2,
                planned_start_agent_unix_ns=planned_agent_ns,
                benchmark_args=arguments,
                timeout_seconds=30,
                recovery_attempts=0,
            )
            assert recovered["successful_requests"] == 2
            assert recovered["failed_requests"] == 0
            assert recovered["_remote_result_delivery"] == {
                "result_source": "durable",
                "transport_retries": 0,
                "max_transport_recovery_attempts": 0,
            }
            after_recovery = read_traceparents(trace_path)
            assert after_recovery == before_restart

            try:
                run_agent_benchmark(
                    second_agent_url,
                    api_key=API_KEY,
                    expected_agent_id_sha256=agent_hash,
                    run_id=RUN_ID,
                    client_index=0,
                    client_count=2,
                    planned_start_agent_unix_ns=planned_agent_ns,
                    benchmark_args=[*arguments, "--seed", "99"],
                    timeout_seconds=30,
                    recovery_attempts=0,
                )
            except RuntimeError as exc:
                assert "HTTP 409" in str(exc), str(exc)
            else:
                raise AssertionError("durable identity accepted a conflicting request")
            assert read_traceparents(trace_path) == before_restart

            assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
            with sqlite3.connect(state_path) as connection:
                rows = connection.execute(
                    "SELECT identity, request_fingerprint, state, artifact_json "
                    "FROM agent_run_records ORDER BY accepted_order"
                ).fetchall()
                metadata = connection.execute(
                    "SELECT key, value FROM agent_state_metadata ORDER BY key"
                ).fetchall()
                integrity = connection.execute("PRAGMA integrity_check").fetchone()
            assert integrity == ("ok",), integrity
            assert len(rows) == 1, rows
            assert rows[0][2] == "completed"
            persisted_artifact = json.loads(rows[0][3])
            assert persisted_artifact["successful_requests"] == 2
            persisted = json.dumps([rows, metadata])
            for private_value in (
                API_KEY,
                API_KEY_ENV,
                AGENT_ID,
                RUN_ID,
                PROMPT,
                endpoint,
                first_agent_url,
                second_agent_url,
                str(state_path),
                *before_restart,
            ):
                assert private_value not in persisted, private_value

            print(
                json.dumps(
                    {
                        "agent_processes": 2,
                        "agent_restarts": 1,
                        "ambiguous_response_dropped": True,
                        "logical_target_requests": len(after_recovery),
                        "successful_target_requests": len(after_recovery),
                        "durable_results": 1,
                        "duplicate_target_requests": 0,
                        "conflicting_replay_rejected": True,
                        "database_integrity": integrity[0],
                        "database_mode": oct(stat.S_IMODE(state_path.stat().st_mode)),
                        "private_values_persisted": False,
                    },
                    indent=2,
                )
            )
        finally:
            if proxy is not None:
                proxy.shutdown()
                proxy.server_close()
            if proxy_thread is not None:
                proxy_thread.join(timeout=2)
            stop_process(agent_process)
            stop_process(fixture_process)


if __name__ == "__main__":
    main()
