"""Deterministic OpenAI-compatible backend for multi-process path tests."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$"
)
CONTROLLED_CRASH_EXIT_CODE = 86


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--state-file")
    parser.add_argument("--fail-request-number", type=int, default=0)
    parser.add_argument("--crash-request-number", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.listen_port <= 65535:
        parser.error("--listen-port must be between zero and 65535")
    if args.fail_request_number < 0:
        parser.error("--fail-request-number must be zero or greater")
    if args.crash_request_number < 0:
        parser.error("--crash-request-number must be zero or greater")
    if args.fail_request_number and args.fail_request_number == args.crash_request_number:
        parser.error("failure and crash request numbers must be distinct")
    return args


def load_state(state_path: Path | None) -> dict[str, int | bool]:
    if state_path is None or not state_path.exists():
        return {"backend": 0, "success": 0, "crash_triggered": False}
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("fixture state must be a JSON object")
    backend = raw.get("backend")
    success = raw.get("success")
    crash_triggered = raw.get("crash_triggered")
    if (
        not isinstance(backend, int)
        or isinstance(backend, bool)
        or backend < 0
        or not isinstance(success, int)
        or isinstance(success, bool)
        or not 0 <= success <= backend
        or not isinstance(crash_triggered, bool)
    ):
        raise ValueError("fixture state contains invalid counters")
    return {
        "backend": backend,
        "success": success,
        "crash_triggered": crash_triggered,
    }


def persist_state(state_path: Path | None, state: dict[str, int | bool]) -> None:
    if state_path is None:
        return
    temporary_path = state_path.with_name(f".{state_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(state_path)


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace_file)
    state_path = Path(args.state_file) if args.state_file else None
    state_lock = threading.Lock()
    request_state = load_state(state_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            with state_lock:
                payload = (
                    f'fixture_backend_requests_total{{service="backend"}} '
                    f'{request_state["backend"]}\n'
                    f'fixture_backend_success_total{{service="backend"}} '
                    f'{request_state["success"]}\n'
                ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_response(400)
                self.end_headers()
                return
            traceparent = self.headers.get("traceparent", "")
            if (
                self.path != "/v1/completions"
                or payload.get("stream") is not True
                or TRACEPARENT_PATTERN.fullmatch(traceparent) is None
            ):
                self.send_response(400)
                self.end_headers()
                return

            with state_lock:
                backend_count = int(request_state["backend"]) + 1
                request_state["backend"] = backend_count
                with trace_path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(f"{traceparent}\n")
                should_crash = (
                    backend_count == args.crash_request_number
                    and request_state["crash_triggered"] is False
                )
                if should_crash:
                    request_state["crash_triggered"] = True
                persist_state(state_path, request_state)

            if should_crash:
                os._exit(CONTROLLED_CRASH_EXIT_CODE)

            if backend_count == args.fail_request_number:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            request_parts = traceparent.split("-")
            response_span_id = (
                "2222222222222222"
                if request_parts[2] == "1111111111111111"
                else "1111111111111111"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header(
                "traceparent",
                f"00-{request_parts[1]}-{response_span_id}-01",
            )
            self.end_headers()
            events = (
                {"choices": [{"text": "fixture"}]},
                {"choices": [{"text": " response"}]},
                {"choices": [], "usage": {"completion_tokens": 2}},
            )
            for event in events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            with state_lock:
                request_state["success"] = int(request_state["success"]) + 1
                persist_state(state_path, request_state)

    server = ThreadingHTTPServer(("127.0.0.1", args.listen_port), Handler)
    Path(args.port_file).write_text(str(server.server_port), encoding="utf-8")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
