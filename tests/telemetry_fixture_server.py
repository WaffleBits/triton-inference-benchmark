from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SNAPSHOTS = (
    {
        "success": 400,
        "failure": 0,
        "request_duration_us": 9_400_000,
        "queue_duration_us": 210_000,
        "compute_infer_duration_us": 7_600_000,
        "gpu_utilization": 40,
    },
    {
        "success": 500,
        "failure": 1,
        "request_duration_us": 18_400_000,
        "queue_duration_us": 710_000,
        "compute_infer_duration_us": 15_100_000,
        "gpu_utilization": 72,
    },
)


class FixtureServer(ThreadingHTTPServer):
    scrape_count = 0


class MetricsHandler(BaseHTTPRequestHandler):
    server: FixtureServer

    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_error(404)
            return

        snapshot = SNAPSHOTS[min(self.server.scrape_count, len(SNAPSHOTS) - 1)]
        self.server.scrape_count += 1
        model = "resnet50_trt_fp16"
        payload = (
            f'DCGM_FI_DEV_GPU_UTIL{{gpu="0"}} {snapshot["gpu_utilization"]}\n'
            f'nv_inference_request_success{{model="{model}",version="1"}} '
            f'{snapshot["success"]}\n'
            f'nv_inference_request_failure{{model="{model}",version="1"}} '
            f'{snapshot["failure"]}\n'
            f'nv_inference_request_duration_us{{model="{model}",version="1"}} '
            f'{snapshot["request_duration_us"]}\n'
            f'nv_inference_queue_duration_us{{model="{model}",version="1"}} '
            f'{snapshot["queue_duration_us"]}\n'
            f'nv_inference_compute_infer_duration_us{{model="{model}",version="1"}} '
            f'{snapshot["compute_infer_duration_us"]}\n'
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve two deterministic Prometheus snapshots for CLI tests."
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = FixtureServer(("127.0.0.1", args.port), MetricsHandler)
    port_file = Path(args.port_file)
    port_file.write_text(str(server.server_port), encoding="utf-8")
    try:
        server.serve_forever()
    finally:
        port_file.unlink(missing_ok=True)
        server.server_close()


if __name__ == "__main__":
    main()
