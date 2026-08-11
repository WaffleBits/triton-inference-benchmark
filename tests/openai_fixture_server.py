"""Deterministic OpenAI-compatible SSE fixture for trace-context CLI tests."""

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace_file)
    trace_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            traceparent = self.headers.get("traceparent", "")
            if (
                self.path != "/v1/completions"
                or payload.get("stream") is not True
                or TRACEPARENT_PATTERN.fullmatch(traceparent) is None
            ):
                self.send_response(400)
                self.end_headers()
                return

            with trace_lock:
                with trace_path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(f"{traceparent}\n")

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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Path(args.port_file).write_text(str(server.server_port), encoding="utf-8")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
