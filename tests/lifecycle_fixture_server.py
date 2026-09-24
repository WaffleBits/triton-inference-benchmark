"""Deterministic local service for the lifecycle qualification CLI fixture."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-delay-ms", type=float, default=0.0)
    parser.add_argument("--state-file", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.ready_delay_ms < 0:
        parser.error("--ready-delay-ms must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    state_lock = threading.Lock()
    state = {"requests": 0}
    ready_at = time.monotonic() + (args.ready_delay_ms / 1000.0)

    def persist_state() -> None:
        with state_lock:
            payload = json.dumps(state, sort_keys=True) + "\n"
        args.state_file.write_text(payload, encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/healthz":
                self.send_response(404)
                self.end_headers()
                return
            ready = time.monotonic() >= ready_at
            payload = (b"ready\n" if ready else b"starting\n")
            self.send_response(200 if ready else 503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/v1/completions" or time.monotonic() < ready_at:
                self.send_response(503)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if payload.get("stream") is not True:
                self.send_response(400)
                self.end_headers()
                return
            with state_lock:
                state["requests"] += 1
            persist_state()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
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

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    persist_state()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        persist_state()


if __name__ == "__main__":
    main()
