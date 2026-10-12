"""Exercise remote-agent cancellation through the real coordinator CLI."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_KEY_ENV = "BENCHMARK_CANCELLATION_FIXTURE_KEY"
API_KEY = "remote-cancellation-fixture-key-32-characters"


def children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text(encoding="utf-8").split()]
    except FileNotFoundError:
        return []


def process_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
    except FileNotFoundError:
        return False
    return fields.lstrip().split()[0] != "Z"


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
    signal_name = sys.argv[1] if len(sys.argv) == 2 else "SIGTERM"
    if signal_name not in {"SIGINT", "SIGTERM"}:
        raise ValueError("fixture accepts SIGINT or SIGTERM")
    selected_signal = getattr(signal, signal_name)

    sys.path.insert(0, str(ROOT))
    from remote_agent import BenchmarkAgentServer

    target: subprocess.Popen[str] | None = None
    coordinator: subprocess.Popen[str] | None = None
    agents: list[BenchmarkAgentServer] = []
    agent_threads: list[threading.Thread] = []
    with tempfile.TemporaryDirectory(prefix="remote-agent-cancel-") as temp_dir:
        temp = Path(temp_dir)
        target_port_file = temp / "target.port"
        trace_file = temp / "target.trace"
        result_dir = temp / "results"
        result_dir.mkdir()
        try:
            target = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests" / "openai_fixture_server.py"),
                    "--port-file",
                    str(target_port_file),
                    "--trace-file",
                    str(trace_file),
                    "--response-delay-ms",
                    "30000",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not target_port_file.exists():
                if target.poll() is not None:
                    raise RuntimeError("target fixture exited before readiness")
                time.sleep(0.02)
            assert target_port_file.is_file(), "target fixture did not become ready"
            target_port = int(target_port_file.read_text(encoding="utf-8"))

            for index in range(2):
                server = BenchmarkAgentServer(
                    ("127.0.0.1", 0),
                    api_key=API_KEY,
                    agent_id=f"cancellation-fixture-agent-{index}",
                    child_timeout_seconds=60,
                    child_environment={
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONIOENCODING": "utf-8",
                    },
                )
                agents.append(server)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                agent_threads.append(thread)
                thread.start()

            env = dict(os.environ)
            env[API_KEY_ENV] = API_KEY
            command = [
                sys.executable,
                str(ROOT / "coordinated_benchmark.py"),
                "--clients",
                "2",
                "--output-dir",
                str(result_dir),
                "--lead-time-ms",
                "500",
                "--max-start-skew-ms",
                "500",
                "--max-clock-uncertainty-ms",
                "1000",
                "--clock-samples",
                "2",
                "--timeout-seconds",
                "60",
            ]
            for server in agents:
                command.extend(
                    ["--agent-url", f"http://127.0.0.1:{server.server_port}"]
                )
            command.extend(
                [
                    "--agent-api-key-env",
                    API_KEY_ENV,
                    "--",
                    "--mode",
                    "openai",
                    "--server-url",
                    f"http://127.0.0.1:{target_port}/v1",
                    "--model-name",
                    "fixture-model",
                    "--num-requests",
                    "1",
                    "--concurrency",
                    "1",
                    "--retries",
                    "0",
                    "--propagate-trace-context",
                ]
            )
            coordinator = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if coordinator.poll() is not None:
                    stdout, stderr = coordinator.communicate()
                    raise RuntimeError(
                        "coordinator exited before remote requests were held open\n"
                        f"stdout:\n{stdout}\nstderr:\n{stderr}"
                    )
                trace_count = (
                    len(trace_file.read_text(encoding="utf-8").splitlines())
                    if trace_file.exists()
                    else 0
                )
                if trace_count >= 2:
                    break
                time.sleep(0.02)
            else:
                raise RuntimeError("did not observe both remote benchmark requests")

            coordinator.send_signal(selected_signal)
            try:
                stdout, stderr = coordinator.communicate(timeout=8)
            except subprocess.TimeoutExpired as exc:
                raise AssertionError(
                    "coordinator did not finish remote cancellation promptly"
                ) from exc
            expected_status = 128 + selected_signal
            assert coordinator.returncode == expected_status, (
                coordinator.returncode,
                stdout,
                stderr,
            )
            assert "remote agent cancellation confirmed: 2/2" in stderr.lower(), stderr
            assert not list(result_dir.iterdir()), "cancelled run wrote a success artifact"
            for index, server in enumerate(agents):
                assert len(server.run_records) == 1, (index, server.run_records)
                record = next(iter(server.run_records.values()))
                assert record["state"] == "cancelled", (index, record)
                assert not server.active_processes, (index, server.active_processes)

            # The agents stay available after cancellation, but their benchmark
            # subprocesses must be gone before the cancellation response is accepted.
            children_left = [pid for pid in children(os.getpid()) if process_running(pid)]
            expected = {target.pid}
            assert set(children_left) <= expected, children_left
        finally:
            if coordinator is not None and coordinator.poll() is None:
                try:
                    os.killpg(coordinator.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                coordinator.wait(timeout=3)
            for server in agents:
                for process in list(getattr(server, "active_processes", {}).values()):
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)
            for pid in children(os.getpid()):
                if target is not None and pid == target.pid:
                    continue
                if process_running(pid):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            for server in agents:
                try:
                    server.shutdown()
                except Exception:
                    pass
                server.server_close()
            for thread in agent_threads:
                thread.join(timeout=3)
            stop_process(target)

    print(f"remote agent {signal_name} cancellation fixture passed")


if __name__ == "__main__":
    main()
