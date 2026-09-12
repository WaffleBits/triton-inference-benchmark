"""Authenticated remote benchmark agent and coordinator-side protocol helpers."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CHALLENGE_PATTERN = HASH_PATTERN
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_BENCHMARK_ARGUMENT_BYTES = 32 * 1024
MAX_REPLAY_IDENTITIES = 1024


class AgentProtocolError(ValueError):
    """A bounded protocol validation error with an HTTP status."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep bearer credentials pinned to the operator-selected agent origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


def validate_agent_base_url(value: str) -> str:
    """Validate a remote agent URL, allowing cleartext only on loopback."""
    if not isinstance(value, str) or not value:
        raise ValueError("agent URL must be a non-empty string")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("agent URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("agent URL must not contain embedded credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("agent URL must not contain a path, query, or fragment")
    try:
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("agent URL contains an invalid port") from exc
    if hostname is None:
        raise ValueError("agent URL must contain a hostname")

    loopback = hostname.lower() == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
    if parsed.scheme == "http" and not loopback:
        raise ValueError("non-loopback agent URLs must use HTTPS")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def build_child_environment(
    agent_api_key_env: str,
    allowed_names: list[str],
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal child environment without the agent authentication key."""
    environment = os.environ if source is None else source
    child_environment = {
        "PATH": environment.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
    }
    env_name_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for name in allowed_names:
        if env_name_pattern.fullmatch(name) is None:
            raise ValueError(
                "--allow-child-env names must be valid environment variables"
            )
        if name == agent_api_key_env:
            raise ValueError(
                "the agent API key cannot be exposed to benchmark children"
            )
        value = environment.get(name)
        if value is None:
            raise ValueError(f"allowed child environment variable {name} is not set")
        child_environment[name] = value
    return child_environment


def _integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"clock sample {key} must be an integer")
    return value


def select_clock_observation(
    observations: list[dict[str, int]],
) -> dict[str, int | str]:
    """Select the NTP-style sample with minimum network delay."""
    if not observations:
        raise ValueError("at least one clock observation is required")

    candidates: list[dict[str, int]] = []
    for raw in observations:
        sample = dict(raw)
        t0 = _integer(sample, "coordinator_send_unix_ns")
        t1 = _integer(sample, "agent_receive_unix_ns")
        t2 = _integer(sample, "agent_send_unix_ns")
        t3 = _integer(sample, "coordinator_receive_unix_ns")
        if t3 < t0 or t2 < t1:
            raise ValueError("clock sample timestamps are not monotonic per host")
        network_delay_ns = (t3 - t0) - (t2 - t1)
        if network_delay_ns < 0:
            raise ValueError("clock sample has negative network delay")
        offset_ns = ((t1 - t0) + (t2 - t3)) // 2
        candidates.append(
            {
                "clock_offset_agent_minus_coordinator_ns": offset_ns,
                "clock_network_delay_ns": network_delay_ns,
                "clock_uncertainty_ns": (network_delay_ns + 1) // 2,
            }
        )

    selected = min(candidates, key=lambda item: item["clock_network_delay_ns"])
    return {
        **selected,
        "clock_sample_count": len(candidates),
        "clock_selection": "minimum_network_delay",
    }


def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, object],
    *,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    normalized = validate_agent_base_url(base_url)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        normalized + path,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(_NoRedirectHandler())
        with opener.open(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            detail_raw = exc.read(4097)
            detail_data = json.loads(detail_raw[:4096].decode("utf-8"))
            detail = detail_data.get("error", "request rejected")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            detail = "request rejected"
        raise RuntimeError(f"agent returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("agent request failed") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("agent response exceeded the size limit")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("agent response was not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("agent response must be a JSON object")
    return decoded


def probe_agent_clock(
    base_url: str,
    *,
    api_key: str,
    sample_count: int,
    timeout_seconds: float,
) -> dict[str, object]:
    """Measure one agent clock without retaining URLs or raw challenges."""
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise ValueError("clock sample count must be positive")
    observations: list[dict[str, int]] = []
    agent_hashes: set[str] = set()
    for _ in range(sample_count):
        challenge = os.urandom(32).hex()
        sent_ns = time.time_ns()
        response = _post_json(
            base_url,
            "/v1/clock",
            {"schema_version": 1, "challenge": challenge},
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        received_ns = time.time_ns()
        if response.get("schema_version") != 1:
            raise RuntimeError("agent clock response has an unsupported schema")
        agent_hash = response.get("agent_id_sha256")
        if not isinstance(agent_hash, str) or HASH_PATTERN.fullmatch(agent_hash) is None:
            raise RuntimeError("agent identity fingerprint is invalid")
        expected_challenge_hash = hashlib.sha256(challenge.encode("ascii")).hexdigest()
        if not hmac.compare_digest(
            str(response.get("challenge_sha256", "")), expected_challenge_hash
        ):
            raise RuntimeError("agent clock challenge did not match")
        agent_receive_ns = response.get("agent_receive_unix_ns")
        agent_send_ns = response.get("agent_send_unix_ns")
        if (
            not isinstance(agent_receive_ns, int)
            or isinstance(agent_receive_ns, bool)
            or not isinstance(agent_send_ns, int)
            or isinstance(agent_send_ns, bool)
        ):
            raise RuntimeError("agent clock response timestamps are invalid")
        observations.append(
            {
                "coordinator_send_unix_ns": sent_ns,
                "agent_receive_unix_ns": agent_receive_ns,
                "agent_send_unix_ns": agent_send_ns,
                "coordinator_receive_unix_ns": received_ns,
            }
        )
        agent_hashes.add(agent_hash)

    if len(agent_hashes) != 1:
        raise RuntimeError("agent identity changed during clock sampling")
    return {
        "agent_id_sha256": next(iter(agent_hashes)),
        "agent_id_persisted": False,
        **select_clock_observation(observations),
    }


def run_agent_benchmark(
    base_url: str,
    *,
    api_key: str,
    expected_agent_id_sha256: str,
    run_id: str,
    client_index: int,
    client_count: int,
    planned_start_agent_unix_ns: int,
    benchmark_args: list[str],
    timeout_seconds: float,
) -> dict[str, object]:
    """Run one benchmark child through an authenticated agent."""
    response = _post_json(
        base_url,
        "/v1/run",
        {
            "schema_version": 1,
            "run_id": run_id,
            "client_index": client_index,
            "client_count": client_count,
            "planned_start_unix_ns": planned_start_agent_unix_ns,
            "benchmark_args": benchmark_args,
            "timeout_seconds": timeout_seconds,
        },
        api_key=api_key,
        timeout_seconds=timeout_seconds + 5,
    )
    if response.get("schema_version") != 1:
        raise RuntimeError("agent run response has an unsupported schema")
    response_agent_hash = response.get("agent_id_sha256")
    if not isinstance(response_agent_hash, str) or not hmac.compare_digest(
        response_agent_hash, expected_agent_id_sha256
    ):
        raise RuntimeError("agent identity changed between clock probe and run")
    artifact = response.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError("agent run response did not contain an artifact object")
    return dict(artifact)


def _require_int(
    payload: dict[str, Any], key: str, *, minimum: int, maximum: int | None = None
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AgentProtocolError(f"{key} is invalid")
    if maximum is not None and value > maximum:
        raise AgentProtocolError(f"{key} is invalid")
    return value


class BenchmarkAgentServer(HTTPServer):
    """Single-job-at-a-time HTTP server with bounded replay memory."""

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        api_key: str,
        agent_id: str,
        child_timeout_seconds: float,
        child_environment: dict[str, str],
    ) -> None:
        super().__init__(server_address, BenchmarkAgentHandler)
        self.api_key = api_key
        self.agent_id_sha256 = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
        self.child_timeout_seconds = child_timeout_seconds
        self.child_environment = dict(child_environment)
        self.replay_identities: set[str] = set()
        self.replay_order: deque[str] = deque()

    def remember_run(self, identity: str) -> None:
        if identity in self.replay_identities:
            raise AgentProtocolError("run/client identity was already accepted", status=409)
        self.replay_identities.add(identity)
        self.replay_order.append(identity)
        if len(self.replay_order) > MAX_REPLAY_IDENTITIES:
            expired = self.replay_order.popleft()
            self.replay_identities.discard(expired)


class BenchmarkAgentHandler(BaseHTTPRequestHandler):
    server: BenchmarkAgentServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorize(self) -> None:
        expected = f"Bearer {self.server.api_key}"
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, expected):
            raise AgentProtocolError("unauthorized", status=401)

    def _read_payload(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise AgentProtocolError("content type must be application/json", status=415)
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError as exc:
            raise AgentProtocolError("content length is invalid", status=411) from exc
        if length < 2 or length > MAX_REQUEST_BYTES:
            raise AgentProtocolError("request body size is invalid", status=413)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AgentProtocolError("request body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise AgentProtocolError("request body must be a JSON object")
        if payload.get("schema_version") != 1:
            raise AgentProtocolError("unsupported protocol schema")
        return payload

    def do_POST(self) -> None:
        received_ns = time.time_ns()
        try:
            self._authorize()
            payload = self._read_payload()
            if self.path == "/v1/clock":
                self._handle_clock(payload, received_ns)
            elif self.path == "/v1/run":
                self._handle_run(payload)
            else:
                raise AgentProtocolError("not found", status=404)
        except AgentProtocolError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except Exception:
            self._send_json(500, {"error": "agent request failed"})

    def _handle_clock(self, payload: dict[str, Any], received_ns: int) -> None:
        if set(payload) != {"schema_version", "challenge"}:
            raise AgentProtocolError("clock request contains unsupported fields")
        challenge = payload.get("challenge")
        if not isinstance(challenge, str) or CHALLENGE_PATTERN.fullmatch(challenge) is None:
            raise AgentProtocolError("clock challenge must be 32 random bytes in hex")
        sent_ns = time.time_ns()
        self._send_json(
            200,
            {
                "schema_version": 1,
                "agent_id_sha256": self.server.agent_id_sha256,
                "challenge_sha256": hashlib.sha256(challenge.encode("ascii")).hexdigest(),
                "agent_receive_unix_ns": received_ns,
                "agent_send_unix_ns": sent_ns,
            },
        )

    def _handle_run(self, payload: dict[str, Any]) -> None:
        allowed_fields = {
            "schema_version",
            "run_id",
            "client_index",
            "client_count",
            "planned_start_unix_ns",
            "benchmark_args",
            "timeout_seconds",
        }
        if set(payload) != allowed_fields:
            raise AgentProtocolError("run request contains unsupported fields")
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not 1 <= len(run_id) <= 256:
            raise AgentProtocolError("run_id is invalid")
        client_count = _require_int(payload, "client_count", minimum=2, maximum=64)
        client_index = _require_int(
            payload, "client_index", minimum=0, maximum=client_count - 1
        )
        planned_start_ns = _require_int(payload, "planned_start_unix_ns", minimum=1)
        timeout_value = payload.get("timeout_seconds")
        if (
            not isinstance(timeout_value, (int, float))
            or isinstance(timeout_value, bool)
            or not math.isfinite(float(timeout_value))
            or float(timeout_value) <= 0
        ):
            raise AgentProtocolError("timeout_seconds is invalid")
        timeout_seconds = min(float(timeout_value), self.server.child_timeout_seconds)
        benchmark_args = payload.get("benchmark_args")
        if not isinstance(benchmark_args, list) or not all(
            isinstance(value, str) for value in benchmark_args
        ):
            raise AgentProtocolError("benchmark_args must be a list of strings")
        if sum(len(value.encode("utf-8")) for value in benchmark_args) > MAX_BENCHMARK_ARGUMENT_BYTES:
            raise AgentProtocolError("benchmark arguments exceeded the size limit", status=413)

        from coordinated_benchmark import validate_benchmark_args

        try:
            validate_benchmark_args(list(benchmark_args))
        except ValueError as exc:
            raise AgentProtocolError(str(exc)) from exc
        replay_identity = hashlib.sha256(
            f"{run_id}\0{client_index}".encode("utf-8")
        ).hexdigest()
        self.server.remember_run(replay_identity)

        with tempfile.TemporaryDirectory(prefix="benchmark-agent-") as temp_dir:
            command = [
                sys.executable,
                str(ROOT / "benchmark.py"),
                *benchmark_args,
                f"--coordinated-run-id={run_id}",
                "--coordinated-client-index",
                str(client_index),
                "--coordinated-client-count",
                str(client_count),
                "--coordinated-start-unix-ns",
                str(planned_start_ns),
                "--output-dir",
                temp_dir,
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=self.server.child_environment,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise AgentProtocolError("benchmark child timed out", status=504) from exc
            if completed.returncode:
                detail = completed.stderr.strip()[:2000] or "no diagnostic"
                raise AgentProtocolError(
                    f"benchmark child failed with status {completed.returncode}: {detail}",
                    status=422,
                )
            artifacts = list(Path(temp_dir).glob("benchmark_*.json"))
            if len(artifacts) != 1:
                raise AgentProtocolError(
                    "benchmark child did not write exactly one JSON artifact", status=500
                )
            try:
                artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AgentProtocolError("benchmark child artifact is invalid", status=500) from exc
            if not isinstance(artifact, dict):
                raise AgentProtocolError("benchmark child artifact is invalid", status=500)

        self._send_json(
            200,
            {
                "schema_version": 1,
                "agent_id_sha256": self.server.agent_id_sha256,
                "artifact": artifact,
            },
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an authenticated benchmark agent.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--port-file")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument(
        "--allow-child-env",
        action="append",
        default=[],
        help="Environment variable explicitly allowed into benchmark children.",
    )
    parser.add_argument("--child-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if not args.agent_id or len(args.agent_id) > 256:
        parser.error("--agent-id must contain 1 to 256 characters")
    if not math.isfinite(args.child_timeout_seconds) or args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be finite and positive")
    api_key = os.environ.get(args.api_key_env)
    if api_key is None or not 16 <= len(api_key) <= 4096 or "\r" in api_key or "\n" in api_key:
        parser.error("the selected API-key environment variable must contain 16 to 4096 characters")
    try:
        child_environment = build_child_environment(
            args.api_key_env, args.allow_child_env
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.api_key = api_key
    args.child_environment = child_environment
    return args


def main() -> None:
    args = _parse_args()
    server = BenchmarkAgentServer(
        (args.listen_host, args.port),
        api_key=args.api_key,
        agent_id=args.agent_id,
        child_timeout_seconds=args.child_timeout_seconds,
        child_environment=args.child_environment,
    )
    if args.port_file:
        Path(args.port_file).write_text(str(server.server_address[1]), encoding="utf-8")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
