"""Deterministic forwarding router for multi-process request-path tests."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--fail-request-number", type=int, default=0)
    args = parser.parse_args()
    if args.fail_request_number < 0:
        parser.error("--fail-request-number must be zero or greater")
    parsed = urllib.parse.urlsplit(args.backend_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        parser.error("--backend-url must be an unauthenticated local HTTP URL")
    return args


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace_file)
    state_lock = threading.Lock()
    request_state = {"ingress": 0}

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
                    f'fixture_ingress_requests_total{{service="router"}} '
                    f'{request_state["ingress"]}\n'
                ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(length)
            try:
                payload = json.loads(request_body)
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
                request_state["ingress"] += 1
                request_number = request_state["ingress"]
                with trace_path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(f"{traceparent}\n")

            if request_number == args.fail_request_number:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            backend_request = urllib.request.Request(
                args.backend_url,
                data=request_body,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    "traceparent": traceparent,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(backend_request, timeout=5) as response:
                    status = response.status
                    response_body = response.read()
                    content_type = response.headers.get("Content-Type")
                    response_traceparent = response.headers.get("traceparent")
            except urllib.error.HTTPError as exc:
                status = exc.code
                response_body = exc.read()
                content_type = exc.headers.get("Content-Type")
                response_traceparent = exc.headers.get("traceparent")
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                status = 502
                response_body = b""
                content_type = None
                response_traceparent = None

            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            if response_traceparent:
                self.send_header("traceparent", response_traceparent)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Path(args.port_file).write_text(str(server.server_port), encoding="utf-8")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
