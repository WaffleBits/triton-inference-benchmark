"""Assert SIGTERM cancels a live lifecycle run without orphaning its children."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _process_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
    except FileNotFoundError:
        return False
    return fields.lstrip().split()[0] != "Z"


def _children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text(encoding="utf-8").split()]
    except FileNotFoundError:
        return []


def main() -> None:
    signal_name = sys.argv[1] if len(sys.argv) == 2 else "SIGTERM"
    if signal_name not in {"SIGINT", "SIGTERM"}:
        raise ValueError("signal fixture accepts SIGINT or SIGTERM")
    selected_signal = getattr(signal, signal_name)
    with tempfile.TemporaryDirectory(prefix="lifecycle-signal-fixture-") as temp_dir:
        temp = Path(temp_dir)
        output_dir = temp / "report"
        state_file = temp / "service-state.json"
        pid_file = temp / "service.pid"
        termination_file = temp / "service-terminated"
        service_command = shlex.join(
            [
                sys.executable,
                str(ROOT / "tests" / "lifecycle_fixture_server.py"),
                "--port",
                "{port}",
                "--ready-delay-ms",
                "0",
                "--response-delay-ms",
                "10000",
                "--state-file",
                str(state_file),
                "--pid-file",
                str(pid_file),
                "--termination-file",
                str(termination_file),
            ]
        )
        command = [
            sys.executable,
            str(ROOT / "lifecycle_qualification.py"),
            "--service-command",
            service_command,
            "--health-url",
            "http://127.0.0.1:{port}/healthz",
            "--startup-timeout-seconds",
            "10",
            "--probe-interval-seconds",
            "0.02",
            "--benchmark-timeout-seconds",
            "30",
            "--shutdown-timeout-seconds",
            "1",
            "--output-dir",
            str(output_dir),
            "--benchmark",
            "--mode",
            "openai",
            "--server-url",
            "http://127.0.0.1:{port}/v1",
            "--model-name",
            "fixture-model",
            "--num-requests",
            "1",
            "--concurrency",
            "1",
            "--retries",
            "0",
        ]
        wrapper = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        service_pid: int | None = None
        benchmark_pids: list[int] = []
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if wrapper.poll() is not None:
                    stdout, stderr = wrapper.communicate()
                    raise RuntimeError(
                        "lifecycle wrapper exited before the held request was observed\n"
                        f"stdout:\n{stdout}\nstderr:\n{stderr}"
                    )
                if pid_file.exists() and state_file.exists():
                    try:
                        service_pid = int(pid_file.read_text(encoding="utf-8").strip())
                        state = json.loads(state_file.read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        # The fixture publishes these small files with write_text;
                        # retry while an initial write is still being completed.
                        time.sleep(0.02)
                        continue
                    if state["requests"] == 1:
                        benchmark_pids = [pid for pid in _children(wrapper.pid) if pid != service_pid]
                        if benchmark_pids:
                            break
                time.sleep(0.02)
            else:
                raise RuntimeError("did not observe a live benchmark request")

            wrapper.send_signal(selected_signal)
            stdout, stderr = wrapper.communicate(timeout=15)
            assert wrapper.returncode == 128 + selected_signal, (
                wrapper.returncode,
                stdout,
                stderr,
            )
            assert f"interrupted by {signal_name}" in stderr, stderr
            assert termination_file.read_text(encoding="utf-8").strip() == "terminated"
            assert json.loads(state_file.read_text(encoding="utf-8"))["requests"] == 1
            assert not output_dir.exists(), "cancelled runs must not publish a success report"

            for pid in [service_pid, *benchmark_pids]:
                child_deadline = time.monotonic() + 5
                while _process_running(pid) and time.monotonic() < child_deadline:
                    time.sleep(0.02)
                assert not _process_running(pid), f"child process {pid} remained active"
        finally:
            if wrapper.poll() is None:
                wrapper.kill()
                wrapper.wait(timeout=5)
            if service_pid is not None:
                try:
                    os.killpg(service_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for pid in benchmark_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    print(f"lifecycle {signal_name} cleanup fixture passed")


if __name__ == "__main__":
    main()
