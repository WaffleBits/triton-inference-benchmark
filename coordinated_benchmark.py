"""Coordinate independent benchmark CLI clients on one host and reconcile results."""

from __future__ import annotations

import argparse
import json
import math
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_CLIENTS = 64
UNSUPPORTED_BENCHMARK_OPTIONS = {
    "-h",
    "--help",
    "--output-dir",
    "--coordinated-run-id",
    "--coordinated-client-index",
    "--coordinated-client-count",
    "--coordinated-start-unix-ns",
    "--baseline",
    "--fail-on-regression",
    "--telemetry-prometheus",
    "--telemetry-baseline-prometheus",
    "--telemetry-url",
    "--telemetry-timeout-seconds",
    "--telemetry-sample-interval-seconds",
    "--telemetry-api-key-env",
    "--request-path-ingress-metric",
    "--request-path-backend-metric",
    "--request-path-success-metric",
    "--fail-on-request-path-gap",
    "--service-restart-metric",
    "--min-service-restarts",
    "--max-service-restarts",
    "--fail-on-service-lifecycle-gap",
    "--max-server-failure-rate",
    "--max-server-queue-fraction",
    "--fail-on-telemetry-gate",
}


def validate_benchmark_args(arguments: list[str]) -> None:
    """Reject child options that conflict with coordination or share counter windows."""
    for token in arguments:
        option = token.split("=", 1)[0]
        if option in UNSUPPORTED_BENCHMARK_OPTIONS:
            raise ValueError(
                f"benchmark option {option} is not supported by the coordinator"
            )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _nonnegative_int(mapping: dict[str, Any], key: str, scope: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{scope}.{key} must be a non-negative integer")
    return value


def _positive_int(mapping: dict[str, Any], key: str, scope: str) -> int:
    value = _nonnegative_int(mapping, key, scope)
    if value <= 0:
        raise ValueError(f"{scope}.{key} must be positive")
    return value


def _finite_number(mapping: dict[str, Any], key: str, scope: str) -> float:
    value = mapping.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{scope}.{key} must be finite")
    return float(value)


def build_coordinated_summary(
    shards: list[dict[str, object]],
    max_start_skew_ms: float,
) -> dict[str, object]:
    """Validate and aggregate same-host client artifacts without merging quantiles."""
    if not shards:
        raise ValueError("at least one coordinated client artifact is required")
    if not math.isfinite(max_start_skew_ms) or max_start_skew_ms < 0:
        raise ValueError("maximum start skew must be finite and non-negative")

    clients: list[dict[str, object]] = []
    run_hashes: set[str] = set()
    config_hashes: set[str] = set()
    expected_counts: set[int] = set()
    planned_starts: set[int] = set()
    modes: set[str] = set()
    indexes: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    configured_rates: list[float | None] = []

    total_logical = 0
    total_successful = 0
    total_failed = 0
    total_attempts = 0
    total_retry_attempts = 0
    total_retried = 0
    total_recovered = 0
    total_exhausted = 0

    for position, raw_shard in enumerate(shards):
        shard_data = _mapping(raw_shard, f"shard[{position}]")
        coordination = _mapping(
            shard_data.get("coordination"), f"shard[{position}].coordination"
        )
        retry = _mapping(shard_data.get("retry"), f"shard[{position}].retry")

        run_hash = coordination.get("run_id_sha256")
        config_hash = coordination.get("config_fingerprint_sha256")
        if not isinstance(run_hash, str) or HASH_PATTERN.fullmatch(run_hash) is None:
            raise ValueError("coordinated run fingerprint must be lowercase SHA-256")
        if not isinstance(config_hash, str) or HASH_PATTERN.fullmatch(config_hash) is None:
            raise ValueError("configuration fingerprint must be lowercase SHA-256")
        if coordination.get("run_id_persisted") is not False:
            raise ValueError("child artifact did not confirm run-ID redaction")
        if coordination.get("clock") != "host_wall_clock":
            raise ValueError("coordinated child must use the host wall clock")

        client_index = _nonnegative_int(
            coordination, "client_index", f"shard[{position}].coordination"
        )
        client_count = _positive_int(
            coordination, "client_count", f"shard[{position}].coordination"
        )
        planned_start = _positive_int(
            coordination,
            "planned_start_unix_ns",
            f"shard[{position}].coordination",
        )
        measured_start = _positive_int(
            coordination,
            "measured_start_unix_ns",
            f"shard[{position}].coordination",
        )
        measured_end = _positive_int(
            coordination,
            "measured_end_unix_ns",
            f"shard[{position}].coordination",
        )
        if measured_end <= measured_start:
            raise ValueError("coordinated child measured window must have positive duration")

        logical = _nonnegative_int(retry, "logical_requests", f"shard[{position}].retry")
        attempts = _nonnegative_int(retry, "client_attempts", f"shard[{position}].retry")
        retry_attempts = _nonnegative_int(
            retry, "retry_attempts", f"shard[{position}].retry"
        )
        retried = _nonnegative_int(
            retry, "retried_requests", f"shard[{position}].retry"
        )
        recovered = _nonnegative_int(
            retry, "recovered_requests", f"shard[{position}].retry"
        )
        exhausted = _nonnegative_int(
            retry, "exhausted_requests", f"shard[{position}].retry"
        )
        successful = _nonnegative_int(
            shard_data, "successful_requests", f"shard[{position}]"
        )
        failed = _nonnegative_int(
            shard_data, "failed_requests", f"shard[{position}]"
        )
        num_requests = _nonnegative_int(
            shard_data, "num_requests", f"shard[{position}]"
        )
        if logical <= 0:
            raise ValueError("coordinated child must contain measured logical requests")
        if num_requests != logical or successful + failed != logical:
            raise ValueError("coordinated child request counts did not reconcile")
        if attempts < logical or retry_attempts != attempts - logical:
            raise ValueError("coordinated child attempt counts did not reconcile")
        if exhausted != failed or recovered > retried:
            raise ValueError("coordinated child retry outcomes did not reconcile")

        mode = shard_data.get("mode")
        if not isinstance(mode, str) or not mode:
            raise ValueError("coordinated child mode must be a non-empty string")
        duration = _finite_number(shard_data, "duration_seconds", f"shard[{position}]")
        if duration <= 0:
            raise ValueError("coordinated child duration must be positive")

        schedule = shard_data.get("load_schedule")
        if schedule is None:
            configured_rate = None
        else:
            schedule_data = _mapping(schedule, f"shard[{position}].load_schedule")
            configured_rate = _finite_number(
                schedule_data,
                "configured_request_rate_rps",
                f"shard[{position}].load_schedule",
            )
            if configured_rate <= 0:
                raise ValueError("coordinated child request rate must be positive")
        configured_rates.append(configured_rate)

        run_hashes.add(run_hash)
        config_hashes.add(config_hash)
        expected_counts.add(client_count)
        planned_starts.add(planned_start)
        modes.add(mode)
        indexes.append(client_index)
        starts.append(measured_start)
        ends.append(measured_end)

        total_logical += logical
        total_successful += successful
        total_failed += failed
        total_attempts += attempts
        total_retry_attempts += retry_attempts
        total_retried += retried
        total_recovered += recovered
        total_exhausted += exhausted
        clients.append(
            {
                "client_index": client_index,
                "logical_requests": logical,
                "successful_requests": successful,
                "failed_requests": failed,
                "client_attempts": attempts,
                "duration_seconds": duration,
            }
        )

    if len(run_hashes) != 1:
        raise ValueError("coordinated run fingerprint mismatch")
    if len(config_hashes) != 1:
        raise ValueError("configuration fingerprint mismatch")
    if len(expected_counts) != 1:
        raise ValueError("coordinated client-count mismatch")
    if len(planned_starts) != 1:
        raise ValueError("coordinated planned-start mismatch")
    if len(modes) != 1:
        raise ValueError("coordinated benchmark mode mismatch")
    if len(set(indexes)) != len(indexes):
        raise ValueError("coordinated client indexes must be unique")

    expected_count = next(iter(expected_counts))
    if expected_count < 2 or expected_count > MAX_CLIENTS:
        raise ValueError(f"coordinated client count must be between 2 and {MAX_CLIENTS}")
    if len(shards) != expected_count or set(indexes) != set(range(expected_count)):
        raise ValueError("coordinated artifacts must contain the complete client index set")

    if all(rate is None for rate in configured_rates):
        aggregate_configured_rate = None
    elif all(rate is not None for rate in configured_rates):
        aggregate_configured_rate = round(
            sum(float(rate) for rate in configured_rates if rate is not None), 4
        )
    else:
        raise ValueError("coordinated child pacing metadata was inconsistent")

    earliest_start = min(starts)
    latest_start = max(starts)
    earliest_end = min(ends)
    latest_end = max(ends)
    start_skew_ms = (latest_start - earliest_start) / 1_000_000
    union_duration_seconds = (latest_end - earliest_start) / 1_000_000_000
    overlap_duration_seconds = max(0, earliest_end - latest_start) / 1_000_000_000
    throughput_rps = total_successful / union_duration_seconds

    failure_reasons: list[str] = []
    if start_skew_ms > max_start_skew_ms:
        failure_reasons.append(
            f"measured client start skew {start_skew_ms:g} ms exceeded "
            f"{max_start_skew_ms:g} ms"
        )
    if overlap_duration_seconds <= 0:
        failure_reasons.append("measured client windows did not overlap")

    return {
        "schema_version": 1,
        "scope": "single_host_multi_process",
        "mode": next(iter(modes)),
        "run_id_sha256": next(iter(run_hashes)),
        "run_id_persisted": False,
        "config_fingerprint_sha256": next(iter(config_hashes)),
        "client_count": expected_count,
        "client_indexes": sorted(indexes),
        "logical_requests": total_logical,
        "successful_requests": total_successful,
        "failed_requests": total_failed,
        "success_rate": round(total_successful / total_logical, 4),
        "client_attempts": total_attempts,
        "retry_attempts": total_retry_attempts,
        "retried_requests": total_retried,
        "recovered_requests": total_recovered,
        "exhausted_requests": total_exhausted,
        "throughput_rps": round(throughput_rps, 4),
        "configured_aggregate_request_rate_rps": aggregate_configured_rate,
        "window": {
            "clock": "single_host_wall_clock",
            "planned_start_unix_ns": next(iter(planned_starts)),
            "earliest_measured_start_unix_ns": earliest_start,
            "latest_measured_end_unix_ns": latest_end,
            "union_duration_seconds": round(union_duration_seconds, 6),
            "overlap_duration_seconds": round(overlap_duration_seconds, 6),
            "start_skew_ms": round(start_skew_ms, 6),
        },
        "coordination_gate": {
            "passed": not failure_reasons,
            "max_start_skew_ms": max_start_skew_ms,
            "requires_overlapping_windows": True,
            "failure_reasons": failure_reasons,
        },
        "latency": {
            "global_percentiles_available": False,
            "note": (
                "Child percentile summaries are not mergeable; this aggregate does not "
                "average or relabel them as a global latency percentile."
            ),
        },
        "clients": sorted(clients, key=lambda item: int(item["client_index"])),
        "privacy": {
            "child_artifact_paths_persisted": False,
            "server_urls_persisted": False,
            "prompts_or_outputs_persisted": False,
            "trace_identifiers_persisted": False,
        },
        "claim_boundary": (
            "This artifact proves same-host process coordination and count/window "
            "reconciliation. It does not prove multi-node load, cross-host clock "
            "synchronization, a model or GPU, production isolation, or global latency "
            "percentiles."
        ),
    }


def format_coordinated_prometheus(summary: dict[str, object]) -> str:
    """Export aggregate scalar evidence without run IDs, paths, or endpoints."""
    window = _mapping(summary.get("window"), "window")
    gate = _mapping(summary.get("coordination_gate"), "coordination_gate")
    lines = [
        "# HELP triton_coordinated_clients Independent benchmark client processes.",
        "# TYPE triton_coordinated_clients gauge",
        f"triton_coordinated_clients {_nonnegative_int(summary, 'client_count', 'summary')}",
        "# HELP triton_coordinated_requests_total Measured logical requests by outcome.",
        "# TYPE triton_coordinated_requests_total counter",
        (
            'triton_coordinated_requests_total{outcome="success"} '
            f"{_nonnegative_int(summary, 'successful_requests', 'summary')}"
        ),
        (
            'triton_coordinated_requests_total{outcome="failure"} '
            f"{_nonnegative_int(summary, 'failed_requests', 'summary')}"
        ),
        "# HELP triton_coordinated_client_attempts_total Physical client attempts across all processes.",
        "# TYPE triton_coordinated_client_attempts_total counter",
        f"triton_coordinated_client_attempts_total {_nonnegative_int(summary, 'client_attempts', 'summary')}",
        "# HELP triton_coordinated_window_duration_seconds Union of same-host measured client windows.",
        "# TYPE triton_coordinated_window_duration_seconds gauge",
        (
            "triton_coordinated_window_duration_seconds "
            f"{_finite_number(window, 'union_duration_seconds', 'window'):g}"
        ),
        "# HELP triton_coordinated_throughput_rps Successful requests divided by the same-host union window.",
        "# TYPE triton_coordinated_throughput_rps gauge",
        f"triton_coordinated_throughput_rps {_finite_number(summary, 'throughput_rps', 'summary'):g}",
        "# HELP triton_coordinated_start_skew_ms Difference between latest and earliest measured client starts.",
        "# TYPE triton_coordinated_start_skew_ms gauge",
        f"triton_coordinated_start_skew_ms {_finite_number(window, 'start_skew_ms', 'window'):g}",
        "# HELP triton_coordinated_gate_passed Whether client completeness, overlap, and start-skew checks passed.",
        "# TYPE triton_coordinated_gate_passed gauge",
        f"triton_coordinated_gate_passed {1 if gate.get('passed') is True else 0}",
    ]
    configured_rate = summary.get("configured_aggregate_request_rate_rps")
    if isinstance(configured_rate, (int, float)) and not isinstance(configured_rate, bool):
        lines.extend(
            [
                "# HELP triton_coordinated_configured_request_rate_rps Sum of configured per-client open-loop request rates.",
                "# TYPE triton_coordinated_configured_request_rate_rps gauge",
                f"triton_coordinated_configured_request_rate_rps {float(configured_rate):g}",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coordinate independent same-host benchmark client processes."
    )
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lead-time-ms", type=float, default=750.0)
    parser.add_argument("--max-start-skew-ms", type=float, default=100.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 2 <= args.clients <= MAX_CLIENTS:
        parser.error(f"--clients must be between 2 and {MAX_CLIENTS}")
    if not math.isfinite(args.lead_time_ms) or args.lead_time_ms <= 0:
        parser.error("--lead-time-ms must be finite and greater than zero")
    if not math.isfinite(args.max_start_skew_ms) or args.max_start_skew_ms < 0:
        parser.error("--max-start-skew-ms must be finite and non-negative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and greater than zero")
    benchmark_args = list(args.benchmark_args)
    if benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]
    try:
        validate_benchmark_args(benchmark_args)
    except ValueError as exc:
        parser.error(str(exc))
    args.benchmark_args = benchmark_args
    return args


def _stop_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("coordinated output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = secrets.token_hex(32)
    planned_start_ns = time.time_ns() + int(args.lead_time_ms * 1_000_000)
    processes: list[subprocess.Popen[str]] = []
    client_dirs: list[Path] = []
    try:
        for client_index in range(args.clients):
            client_dir = output_dir / f"client-{client_index}"
            client_dir.mkdir()
            client_dirs.append(client_dir)
            command = [
                sys.executable,
                str(ROOT / "benchmark.py"),
                *args.benchmark_args,
                f"--coordinated-run-id={run_id}",
                "--coordinated-client-index",
                str(client_index),
                "--coordinated-client-count",
                str(args.clients),
                "--coordinated-start-unix-ns",
                str(planned_start_ns),
                "--output-dir",
                str(client_dir),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            )

        deadline = time.monotonic() + args.timeout_seconds
        for client_index, process in enumerate(processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("coordinated benchmark timed out")
            try:
                _, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("coordinated benchmark timed out") from exc
            if process.returncode:
                detail = stderr.strip() if stderr else "no stderr"
                raise RuntimeError(
                    f"coordinated client {client_index} failed with status "
                    f"{process.returncode}: {detail}"
                )
    except Exception:
        _stop_processes(processes)
        raise

    shards: list[dict[str, object]] = []
    for client_index, client_dir in enumerate(client_dirs):
        artifacts = list(client_dir.glob("benchmark_*.json"))
        if len(artifacts) != 1:
            raise RuntimeError(
                f"coordinated client {client_index} wrote {len(artifacts)} JSON artifacts"
            )
        raw = json.loads(artifacts[0].read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError("coordinated client artifact must contain a JSON object")
        shards.append(raw)

    summary = build_coordinated_summary(shards, args.max_start_skew_ms)
    json_path = output_dir / "coordinated_benchmark.json"
    prometheus_path = output_dir / "coordinated_benchmark.prom"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    prometheus_path.write_text(
        format_coordinated_prometheus(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved coordinated metrics to {json_path}")
    print(f"Saved coordinated Prometheus metrics to {prometheus_path}")

    gate = summary.get("coordination_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise SystemExit(9)


if __name__ == "__main__":
    main()
