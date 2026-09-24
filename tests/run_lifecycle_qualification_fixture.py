"""Exercise lifecycle_qualification.py against a real delayed local service."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lifecycle-cli-fixture-") as temp_dir:
        temp = Path(temp_dir)
        output_dir = temp / "report"
        state_file = temp / "service-state.json"
        service_command = " ".join(
            [
                shlex.quote(sys.executable),
                shlex.quote(str(ROOT / "tests" / "lifecycle_fixture_server.py")),
                "--port",
                "{port}",
                "--ready-delay-ms",
                "120",
                "--state-file",
                shlex.quote(str(state_file)),
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
            "4",
            "--concurrency",
            "1",
            "--retries",
            "0",
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
                "lifecycle qualification CLI failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        json_path = output_dir / "lifecycle_qualification.json"
        markdown_path = output_dir / "lifecycle_qualification.md"
        report = json.loads(json_path.read_text(encoding="utf-8"))
        serialized = json_path.read_text(encoding="utf-8") + markdown_path.read_text(
            encoding="utf-8"
        )
        lifecycle = report["lifecycle"]
        benchmark = report["benchmark"]
        assert lifecycle["kind"] == "local_subprocess"
        assert lifecycle["readiness"] == "http_200"
        assert lifecycle["startup_latency_ms"] >= 80
        assert lifecycle["readiness_probe_count"] >= 2
        assert lifecycle["termination"] == "SIGTERM"
        assert benchmark["mode"] == "openai"
        assert benchmark["successful_requests"] == 4
        assert benchmark["failed_requests"] == 0
        assert json.loads(state_file.read_text(encoding="utf-8")) == {"requests": 4}

        for private_value in (
            "lifecycle_fixture_server.py",
            str(state_file),
            "127.0.0.1",
            str(temp),
            "Return a short deterministic benchmark response.",
            "Authorization",
        ):
            assert private_value not in serialized, private_value
        assert "process launch to the selected HTTP-200 readiness" in serialized

    print("controlled lifecycle qualification fixture passed")


if __name__ == "__main__":
    main()
