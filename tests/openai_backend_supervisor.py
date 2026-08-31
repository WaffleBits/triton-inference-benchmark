"""Supervise a crash-once backend and expose a cumulative restart counter."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_CRASH_EXIT_CODE = 86


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--backend-port-file", required=True)
    parser.add_argument("--backend-trace-file", required=True)
    parser.add_argument("--backend-state-file", required=True)
    parser.add_argument("--crash-request-number", type=int, default=2)
    args = parser.parse_args()
    if args.crash_request_number <= 0:
        parser.error("--crash-request-number must be greater than zero")
    return args


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def wait_for_backend(
    port_file: Path,
    process: subprocess.Popen[str],
    expected_port: int,
) -> None:
    for _ in range(200):
        if port_file.is_file() and port_file.stat().st_size:
            observed = int(port_file.read_text(encoding="utf-8"))
            if observed != expected_port:
                raise RuntimeError(
                    f"backend reported port {observed}, expected {expected_port}"
                )
            return
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"backend exited before readiness with status {return_code}"
            )
        time.sleep(0.01)
    raise RuntimeError("backend did not become ready")


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
    args = parse_args()
    backend_port_file = Path(args.backend_port_file)
    backend_trace_file = Path(args.backend_trace_file)
    backend_state_file = Path(args.backend_state_file)
    backend_port = reserve_local_port()
    state_lock = threading.Lock()
    stop_requested = threading.Event()
    state: dict[str, object] = {"process": None, "restarts": 0, "error": None}

    def start_backend() -> subprocess.Popen[str]:
        backend_port_file.unlink(missing_ok=True)
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "tests" / "openai_path_backend_server.py"),
                "--port-file",
                str(backend_port_file),
                "--trace-file",
                str(backend_trace_file),
                "--state-file",
                str(backend_state_file),
                "--listen-port",
                str(backend_port),
                "--crash-request-number",
                str(args.crash_request_number),
            ],
            cwd=ROOT,
            text=True,
        )
        wait_for_backend(backend_port_file, process, backend_port)
        return process

    initial_process = start_backend()
    state["process"] = initial_process

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            with state_lock:
                restarts = int(state["restarts"])
                error = state["error"]
            if error is not None:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = (
                f'fixture_backend_restarts_total{{service="supervisor"}} {restarts}\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    def monitor_backend() -> None:
        process = initial_process
        while not stop_requested.is_set():
            return_code = process.wait()
            if stop_requested.is_set():
                return
            if return_code != CONTROLLED_CRASH_EXIT_CODE:
                with state_lock:
                    state["error"] = (
                        f"backend exited unexpectedly with status {return_code}"
                    )
                server.shutdown()
                return
            try:
                process = start_backend()
            except Exception as exc:  # noqa: BLE001 - surfaced through supervisor health.
                with state_lock:
                    state["error"] = str(exc)
                server.shutdown()
                return
            with state_lock:
                state["process"] = process
                state["restarts"] = int(state["restarts"]) + 1

    monitor = threading.Thread(
        target=monitor_backend,
        name="fixture-backend-supervisor",
        daemon=True,
    )
    monitor.start()
    Path(args.port_file).write_text(str(server.server_port), encoding="utf-8")
    try:
        server.serve_forever()
    finally:
        stop_requested.set()
        server.server_close()
        with state_lock:
            process = state.get("process")
        stop_process(process if isinstance(process, subprocess.Popen) else None)
        monitor.join(timeout=3)
    with state_lock:
        error = state["error"]
    if error is not None:
        raise RuntimeError(str(error))


if __name__ == "__main__":
    main()
