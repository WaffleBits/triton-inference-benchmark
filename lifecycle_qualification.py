"""Run a benchmark behind an explicitly controlled local service lifecycle.

The wrapper measures only process launch to an operator-selected HTTP-200 health
response. It does not infer model cold-start time or serialize operator inputs.
Service and benchmark commands are executed without a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

import benchmark_report

ROOT = Path(__file__).resolve().parent
MAX_COMMAND_LENGTH = 8 * 1024
MAX_BENCHMARK_ARGUMENTS = 128
MAX_BENCHMARK_TIMEOUT_SECONDS = 15 * 60
MAX_STARTUP_TIMEOUT_SECONDS = 10 * 60
MAX_PROBE_INTERVAL_SECONDS = 10.0
MAX_SHUTDOWN_TIMEOUT_SECONDS = 60.0
MAX_HEALTH_BODY_BYTES = 4096


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not let a health probe leave the explicitly selected origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        return None


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_health_url(value: str) -> str:
    """Accept only a credential-free loopback HTTP(S) health URL."""
    if not isinstance(value, str) or not value:
        raise ValueError("health URL must be a non-empty string")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("health URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("health URL must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("health URL must not contain a query or fragment")
    try:
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("health URL contains an invalid port") from exc
    if hostname is None or not _loopback(hostname):
        raise ValueError("health URL must target loopback")
    if not parsed.path or parsed.path == "/":
        raise ValueError("health URL must name an explicit endpoint")
    return value


def _substitute_tokens(
    tokens: Sequence[str],
    port: int,
    *,
    require_placeholder: bool,
) -> list[str]:
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not tokens:
        raise ValueError("command must not be empty")
    substituted: list[str] = []
    placeholder_count = 0
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("command arguments must be non-empty strings")
        placeholder_count += token.count("{port}")
        replaced = token.replace("{port}", str(port))
        if "{" in replaced or "}" in replaced:
            raise ValueError("unsupported command placeholder")
        substituted.append(replaced)
    if require_placeholder and placeholder_count == 0:
        raise ValueError("command must contain the {port} placeholder")
    return substituted


def substitute_port(tokens: Sequence[str], port: int) -> list[str]:
    """Substitute the required literal ``{port}`` without shell evaluation."""
    return _substitute_tokens(tokens, port, require_placeholder=True)


def substitute_optional_port(tokens: Sequence[str], port: int) -> list[str]:
    """Substitute ``{port}`` when present, while permitting mock arguments."""
    return _substitute_tokens(tokens, port, require_placeholder=False)


def validate_benchmark_arguments(arguments: Sequence[str]) -> None:
    """Reject options that could replace the wrapper's private output directory."""
    if not arguments:
        raise ValueError("at least one benchmark argument is required")
    if len(arguments) > MAX_BENCHMARK_ARGUMENTS:
        raise ValueError("too many benchmark arguments")
    for token in arguments:
        option = token.split("=", 1)[0]
        if option == "--output-dir":
            raise ValueError("benchmark arguments must not set --output-dir")
        if token == "--":
            raise ValueError("benchmark arguments must not contain a command separator")


def _sha256_json(value: Sequence[str]) -> str:
    raw = json.dumps(list(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_lifecycle_report(
    *,
    startup_latency_ms: float,
    readiness_probe_count: int,
    service_exit_code: int | None,
    termination: str,
    command: Sequence[str],
    health_url: str,
    benchmark_artifact: dict[str, object],
) -> dict[str, object]:
    """Project measured lifecycle and benchmark data into a public-safe report."""
    latency = _finite_nonnegative(startup_latency_ms, "startup latency")
    if (
        not isinstance(readiness_probe_count, int)
        or isinstance(readiness_probe_count, bool)
        or readiness_probe_count < 1
    ):
        raise ValueError("readiness probe count must be a positive integer")
    if service_exit_code is not None and (
        not isinstance(service_exit_code, int) or isinstance(service_exit_code, bool)
    ):
        raise ValueError("service exit code must be an integer or null")
    if termination not in {"SIGTERM", "SIGKILL", "already-exited"}:
        raise ValueError("unsupported service termination status")
    if not command or any(not isinstance(item, str) for item in command):
        raise ValueError("service command must be a non-empty string sequence")
    validate_health_url(health_url)

    return {
        "schema_version": 1,
        "lifecycle": {
            "kind": "local_subprocess",
            "readiness": "http_200",
            "startup_scope": "process_launch_to_selected_health_response",
            "startup_latency_ms": round(latency, 4),
            "readiness_probe_count": readiness_probe_count,
            "service_exit_code": service_exit_code,
            "termination": termination,
            "command_sha256": _sha256_json(command),
            "health_url_sha256": hashlib.sha256(health_url.encode("utf-8")).hexdigest(),
            "command_persisted": False,
            "health_url_persisted": False,
        },
        "benchmark": benchmark_report.project_run(benchmark_artifact, 1),
        "privacy": {
            "command_persisted": False,
            "health_url_persisted": False,
            "benchmark_source_path_persisted": False,
            "benchmark_private_fields_persisted": False,
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    """Render a deterministic, privacy-safe human-readable report."""
    lifecycle = report["lifecycle"]
    benchmark = report["benchmark"]
    assert isinstance(lifecycle, dict)
    assert isinstance(benchmark, dict)
    latency = lifecycle["startup_latency_ms"]
    probes = lifecycle["readiness_probe_count"]
    termination = lifecycle["termination"]
    return "\n".join(
        [
            "# Controlled lifecycle qualification",
            "",
            "- Readiness: HTTP 200",
            f"- process launch to the selected HTTP-200 readiness: {latency} ms",
            f"- Readiness probes: {probes}",
            f"- Service termination: {termination}",
            f"- Benchmark mode: {benchmark['mode']}",
            f"- Successful requests: {benchmark['successful_requests']}/{benchmark['num_requests']}",
            f"- Completion throughput: {benchmark['throughput_rps']} requests/s",
            "",
            "Startup latency is local process-launch to the selected health response;",
            "it is not model cold-start time, accelerator initialization time, service MTTR,",
            "a remote-host timing, or a production SLO.",
            "",
        ]
    )


def _atomic_write(path: Path, content: bytes) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError(f"output path must not be a symbolic link: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_report(output_dir: Path, report: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("output directory must be a real directory")
    json_path = output_dir / "lifecycle_qualification.json"
    markdown_path = output_dir / "lifecycle_qualification.md"
    _atomic_write(
        json_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(markdown_path, render_markdown(report).encode("utf-8"))


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _health_ready(url: str, timeout_seconds: float) -> bool:
    request = urllib.request.Request(url, headers={"Accept": "*/*"}, method="GET")
    opener = urllib.request.build_opener(_NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            response.read(MAX_HEALTH_BODY_BYTES + 1)
            return response.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return False


def _wait_for_readiness(
    process: subprocess.Popen[bytes],
    health_url: str,
    *,
    timeout_seconds: float,
    probe_interval_seconds: float,
) -> tuple[float, int]:
    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    probes = 0
    while True:
        probes += 1
        if _health_ready(health_url, min(probe_interval_seconds, 1.0)):
            elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000
            return elapsed_ms, probes
        if process.poll() is not None:
            raise RuntimeError("service exited before health readiness")
        if time.monotonic() >= deadline:
            raise TimeoutError("service did not reach health readiness before the timeout")
        time.sleep(probe_interval_seconds)


def terminate_process(
    process: subprocess.Popen[bytes], timeout_seconds: float
) -> tuple[int | None, str]:
    """Terminate the service process/group and report the actual outcome."""
    timeout = _finite_nonnegative(timeout_seconds, "shutdown timeout")
    if timeout > MAX_SHUTDOWN_TIMEOUT_SECONDS:
        raise ValueError("shutdown timeout is too large")
    if process.poll() is not None:
        return process.returncode, "already-exited"

    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        returncode = process.wait(timeout=timeout)
        return returncode, "SIGTERM"
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        return process.wait(timeout=timeout), "SIGKILL"


def _parse_service_command(value: str) -> list[str]:
    if not isinstance(value, str) or not value or len(value) > MAX_COMMAND_LENGTH:
        raise ValueError("service command must be non-empty and bounded")
    try:
        command = shlex.split(value, posix=True)
    except ValueError as exc:
        raise ValueError("service command is not valid shell-style argument syntax") from exc
    if not command:
        raise ValueError("service command must not be empty")
    return command


def _validate_timeout(value: float, field: str, maximum: float) -> float:
    result = _finite_nonnegative(value, field)
    if result == 0 or result > maximum:
        raise ValueError(f"{field} must be greater than zero and at most {maximum:g}")
    return result


def run_qualification(args: argparse.Namespace) -> dict[str, object]:
    service_command = _parse_service_command(args.service_command)
    validate_benchmark_arguments(args.benchmark)
    startup_timeout = _validate_timeout(
        args.startup_timeout_seconds, "startup timeout", MAX_STARTUP_TIMEOUT_SECONDS
    )
    probe_interval = _validate_timeout(
        args.probe_interval_seconds, "probe interval", MAX_PROBE_INTERVAL_SECONDS
    )
    benchmark_timeout = _validate_timeout(
        args.benchmark_timeout_seconds,
        "benchmark timeout",
        MAX_BENCHMARK_TIMEOUT_SECONDS,
    )
    shutdown_timeout = _validate_timeout(
        args.shutdown_timeout_seconds,
        "shutdown timeout",
        MAX_SHUTDOWN_TIMEOUT_SECONDS,
    )

    port = _reserve_loopback_port()
    service_command = substitute_port(service_command, port)
    health_url = substitute_optional_port([args.health_url], port)[0]
    health_url = validate_health_url(health_url)
    benchmark_arguments = substitute_optional_port(args.benchmark, port)

    process: subprocess.Popen[bytes] | None = None
    startup_latency_ms: float | None = None
    readiness_probe_count = 0
    termination = "already-exited"
    service_exit_code: int | None = None
    try:
        process = subprocess.Popen(
            service_command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
        startup_latency_ms, readiness_probe_count = _wait_for_readiness(
            process,
            health_url,
            timeout_seconds=startup_timeout,
            probe_interval_seconds=probe_interval,
        )
        with tempfile.TemporaryDirectory(prefix="lifecycle-benchmark-") as temp_dir:
            result_dir = Path(temp_dir)
            benchmark_command = [
                sys.executable,
                str(ROOT / "benchmark.py"),
                *benchmark_arguments,
                "--output-dir",
                str(result_dir),
            ]
            completed = subprocess.run(
                benchmark_command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=benchmark_timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark child exited with status {completed.returncode}"
                )
            artifacts = sorted(result_dir.glob("benchmark_*.json"))
            if len(artifacts) != 1:
                raise RuntimeError("benchmark child did not produce exactly one JSON artifact")
            benchmark_artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
            if not isinstance(benchmark_artifact, dict):
                raise RuntimeError("benchmark artifact was not a JSON object")
    finally:
        if process is not None:
            service_exit_code, termination = terminate_process(process, shutdown_timeout)

    assert startup_latency_ms is not None
    report = build_lifecycle_report(
        startup_latency_ms=startup_latency_ms,
        readiness_probe_count=readiness_probe_count,
        service_exit_code=service_exit_code,
        termination=termination,
        command=service_command,
        health_url=health_url,
        benchmark_artifact=benchmark_artifact,
    )
    write_report(Path(args.output_dir), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing benchmark behind a controlled local service lifecycle."
    )
    parser.add_argument(
        "--service-command",
        required=True,
        help="quoted executable and arguments; include one literal {port} placeholder",
    )
    parser.add_argument(
        "--health-url",
        required=True,
        help="loopback HTTP(S) health URL; include {port} for the selected service port",
    )
    parser.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--probe-interval-seconds", type=float, default=0.05)
    parser.add_argument("--benchmark-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--benchmark",
        dest="benchmark",
        nargs=argparse.REMAINDER,
        required=True,
        help="remaining arguments passed to benchmark.py; --output-dir is reserved",
    )
    args = parser.parse_args(argv)
    if not args.benchmark:
        parser.error("--benchmark requires at least one benchmark.py argument")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = run_qualification(args)
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"lifecycle qualification failed: {exc}", file=sys.stderr)
        return 1
    lifecycle = report["lifecycle"]
    assert isinstance(lifecycle, dict)
    print(
        "controlled lifecycle qualification passed: "
        f"{lifecycle['startup_latency_ms']} ms to HTTP-200 readiness"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
