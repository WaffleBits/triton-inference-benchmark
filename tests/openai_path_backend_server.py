"""Deterministic OpenAI-compatible backend for multi-process path tests."""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--fail-request-number", type=int, default=0)
    args = parser.parse_args()
    if args.fail_request_number < 0:
        parser.error("--fail-request-number must be zero or greater")
    return args


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace_file)
    state_lock = threading.Lock()
    request_state = {"backend": 0, "success": 0}

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
                request_state["backend"] += 1
                request_number = request_state["backend"]
                with trace_path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(f"{traceparent}\n")

            if request_number == args.fail_request_number:
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
                request_state["success"] += 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Path(args.port_file).write_text(str(server.server_port), encoding="utf-8")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
