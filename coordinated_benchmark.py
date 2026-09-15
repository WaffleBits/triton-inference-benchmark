"""Coordinate independent benchmark CLI clients on one host and reconcile results."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from remote_agent import (
    probe_agent_clock,
    query_agent_run_status,
    run_agent_benchmark,
    validate_agent_base_url,
)

ROOT = Path(__file__).resolve().parent
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_CLIENTS = 64
MAX_COORDINATOR_STATE_BYTES = 64 * 1024
COORDINATOR_STATE_SCHEMA_VERSION = 1
COORDINATOR_STATE_HMAC_CONTEXT = b"triton-coordinator-state-v1\0"
COORDINATOR_RUN_ID_CONTEXT = b"triton-coordinator-run-v1\0"
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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _validate_resume_token(value: str) -> None:
    if (
        not isinstance(value, str)
        or not 32 <= len(value) <= 4096
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError("coordinator resume token must contain 32 to 4096 characters")


def derive_coordinator_run_id(resume_token: str, workflow_nonce: object) -> str:
    """Derive a stable private run ID without storing it in coordinator state."""
    _validate_resume_token(resume_token)
    if not isinstance(workflow_nonce, str) or HASH_PATTERN.fullmatch(workflow_nonce) is None:
        raise ValueError("coordinator workflow nonce must be 32 random bytes in hex")
    return hmac.new(
        resume_token.encode("utf-8"),
        COORDINATOR_RUN_ID_CONTEXT + workflow_nonce.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def build_coordinator_configuration_fingerprint(
    *,
    client_count: int,
    benchmark_args: list[str],
    timeout_seconds: float,
    max_start_skew_ms: float,
    max_clock_uncertainty_ms: float,
    agent_run_recovery_attempts: int,
    agent_hashes: list[str],
) -> str:
    """Bind a resume manifest to all request and reconciliation semantics."""
    if not 2 <= client_count <= MAX_CLIENTS or len(agent_hashes) != client_count:
        raise ValueError("coordinator configuration client count is invalid")
    if any(
        not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None
        for value in agent_hashes
    ):
        raise ValueError("coordinator configuration agent identity is invalid")
    if len(set(agent_hashes)) != len(agent_hashes):
        raise ValueError("coordinator configuration requires unique agent identities")
    validate_benchmark_args(benchmark_args)
    for name, value, minimum in (
        ("timeout", timeout_seconds, 0.0),
        ("maximum start skew", max_start_skew_ms, 0.0),
        ("maximum clock uncertainty", max_clock_uncertainty_ms, 0.0),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or (float(value) <= minimum if name == "timeout" else float(value) < minimum)
        ):
            raise ValueError(f"coordinator configuration {name} is invalid")
    if (
        not isinstance(agent_run_recovery_attempts, int)
        or isinstance(agent_run_recovery_attempts, bool)
        or not 0 <= agent_run_recovery_attempts <= 3
    ):
        raise ValueError("coordinator configuration recovery attempts are invalid")
    payload = {
        "schema_version": 1,
        "client_count": client_count,
        "benchmark_args": list(benchmark_args),
        "timeout_seconds": float(timeout_seconds),
        "max_start_skew_ms": float(max_start_skew_ms),
        "max_clock_uncertainty_ms": float(max_clock_uncertainty_ms),
        "agent_run_recovery_attempts": agent_run_recovery_attempts,
        "agent_identity_fingerprints_sha256": list(agent_hashes),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _manifest_without_authentication(manifest: dict[str, object]) -> dict[str, object]:
    unsigned = dict(manifest)
    unsigned.pop("integrity_hmac_sha256", None)
    return unsigned


def _validate_coordinator_resume_manifest_fields(
    manifest: dict[str, object],
) -> None:
    expected_fields = {
        "schema_version",
        "workflow_nonce",
        "configuration_sha256",
        "planned_start_coordinator_unix_ns",
        "agents",
        "privacy",
        "integrity_hmac_sha256",
    }
    if set(manifest) != expected_fields:
        raise ValueError("coordinator resume manifest contains unsupported fields")
    if manifest.get("schema_version") != COORDINATOR_STATE_SCHEMA_VERSION:
        raise ValueError("coordinator resume manifest schema is unsupported")
    nonce = manifest.get("workflow_nonce")
    configuration_hash = manifest.get("configuration_sha256")
    integrity = manifest.get("integrity_hmac_sha256")
    if not isinstance(nonce, str) or HASH_PATTERN.fullmatch(nonce) is None:
        raise ValueError("coordinator resume manifest nonce is invalid")
    if (
        not isinstance(configuration_hash, str)
        or HASH_PATTERN.fullmatch(configuration_hash) is None
    ):
        raise ValueError("coordinator resume manifest configuration is invalid")
    if not isinstance(integrity, str) or HASH_PATTERN.fullmatch(integrity) is None:
        raise ValueError("coordinator resume manifest authentication is invalid")
    planned_start = manifest.get("planned_start_coordinator_unix_ns")
    if (
        not isinstance(planned_start, int)
        or isinstance(planned_start, bool)
        or planned_start <= 0
    ):
        raise ValueError("coordinator resume manifest planned start is invalid")

    agents = manifest.get("agents")
    if not isinstance(agents, list) or not 2 <= len(agents) <= MAX_CLIENTS:
        raise ValueError("coordinator resume manifest agents are invalid")
    seen_indexes: set[int] = set()
    seen_hashes: set[str] = set()
    for raw_agent in agents:
        if not isinstance(raw_agent, dict) or set(raw_agent) != {
            "client_index",
            "agent_id_sha256",
            "agent_id_persisted",
            "clock_offset_agent_minus_coordinator_ns",
            "clock_network_delay_ns",
            "clock_uncertainty_ns",
            "clock_sample_count",
            "clock_selection",
            "planned_start_agent_unix_ns",
        }:
            raise ValueError("coordinator resume manifest agent record is invalid")
        client_index = raw_agent.get("client_index")
        agent_hash = raw_agent.get("agent_id_sha256")
        offset_ns = raw_agent.get("clock_offset_agent_minus_coordinator_ns")
        network_delay_ns = raw_agent.get("clock_network_delay_ns")
        uncertainty_ns = raw_agent.get("clock_uncertainty_ns")
        sample_count = raw_agent.get("clock_sample_count")
        planned_agent_ns = raw_agent.get("planned_start_agent_unix_ns")
        if (
            not isinstance(client_index, int)
            or isinstance(client_index, bool)
            or not 0 <= client_index < len(agents)
        ):
            raise ValueError("coordinator resume manifest client index is invalid")
        if not isinstance(agent_hash, str) or HASH_PATTERN.fullmatch(agent_hash) is None:
            raise ValueError("coordinator resume manifest agent identity is invalid")
        for field_name, value in (
            ("clock offset", offset_ns),
            ("clock network delay", network_delay_ns),
            ("clock uncertainty", uncertainty_ns),
            ("clock sample count", sample_count),
            ("agent planned start", planned_agent_ns),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"coordinator resume manifest {field_name} is invalid")
        assert isinstance(network_delay_ns, int)
        assert isinstance(uncertainty_ns, int)
        assert isinstance(sample_count, int)
        assert isinstance(offset_ns, int)
        assert isinstance(planned_agent_ns, int)
        if (
            network_delay_ns < 0
            or uncertainty_ns != (network_delay_ns + 1) // 2
            or sample_count <= 0
            or planned_agent_ns <= 0
            or planned_agent_ns - offset_ns != planned_start
            or raw_agent.get("agent_id_persisted") is not False
            or raw_agent.get("clock_selection") != "minimum_network_delay"
        ):
            raise ValueError("coordinator resume manifest clock record is invalid")
        seen_indexes.add(client_index)
        seen_hashes.add(agent_hash)
    if seen_indexes != set(range(len(agents))) or len(seen_hashes) != len(agents):
        raise ValueError("coordinator resume manifest agent set is invalid")

    privacy = manifest.get("privacy")
    if not isinstance(privacy, dict) or privacy != {
        "resume_token_persisted": False,
        "run_id_persisted": False,
        "benchmark_arguments_persisted": False,
        "agent_urls_persisted": False,
        "result_bodies_persisted": False,
        "clock_challenges_persisted": False,
    }:
        raise ValueError("coordinator resume manifest privacy record is invalid")


def create_coordinator_resume_manifest(
    *,
    resume_token: str,
    configuration_sha256: str,
    planned_start_coordinator_unix_ns: int,
    clock_profiles: list[dict[str, object]],
    workflow_nonce: str | None = None,
) -> dict[str, object]:
    """Create authenticated, privacy-safe state that can reconstruct exact requests."""
    _validate_resume_token(resume_token)
    nonce = workflow_nonce if workflow_nonce is not None else secrets.token_hex(32)
    agents: list[dict[str, object]] = []
    for client_index, profile in enumerate(clock_profiles):
        offset_ns = profile.get("clock_offset_agent_minus_coordinator_ns")
        if not isinstance(offset_ns, int) or isinstance(offset_ns, bool):
            raise ValueError("coordinator clock profile offset is invalid")
        agents.append(
            {
                "client_index": client_index,
                "agent_id_sha256": profile.get("agent_id_sha256"),
                "agent_id_persisted": profile.get("agent_id_persisted"),
                "clock_offset_agent_minus_coordinator_ns": offset_ns,
                "clock_network_delay_ns": profile.get("clock_network_delay_ns"),
                "clock_uncertainty_ns": profile.get("clock_uncertainty_ns"),
                "clock_sample_count": profile.get("clock_sample_count"),
                "clock_selection": profile.get("clock_selection"),
                "planned_start_agent_unix_ns": (
                    planned_start_coordinator_unix_ns + offset_ns
                ),
            }
        )
    unsigned: dict[str, object] = {
        "schema_version": COORDINATOR_STATE_SCHEMA_VERSION,
        "workflow_nonce": nonce,
        "configuration_sha256": configuration_sha256,
        "planned_start_coordinator_unix_ns": planned_start_coordinator_unix_ns,
        "agents": agents,
        "privacy": {
            "resume_token_persisted": False,
            "run_id_persisted": False,
            "benchmark_arguments_persisted": False,
            "agent_urls_persisted": False,
            "result_bodies_persisted": False,
            "clock_challenges_persisted": False,
        },
    }
    manifest = {
        **unsigned,
        "integrity_hmac_sha256": hmac.new(
            resume_token.encode("utf-8"),
            COORDINATOR_STATE_HMAC_CONTEXT + _canonical_json_bytes(unsigned),
            hashlib.sha256,
        ).hexdigest(),
    }
    _validate_coordinator_resume_manifest_fields(manifest)
    return manifest


def write_coordinator_resume_manifest(
    path: Path, manifest: dict[str, object]
) -> None:
    """Durably publish a new owner-only manifest without exposing partial JSON."""
    _validate_coordinator_resume_manifest_fields(manifest)
    if path.exists() or path.is_symlink():
        raise ValueError("coordinator resume manifest already exists")
    if not path.parent.is_dir():
        raise ValueError("coordinator resume manifest parent directory does not exist")
    raw = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if len(raw) > MAX_COORDINATOR_STATE_BYTES:
        raise ValueError("coordinator resume manifest exceeded the size limit")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(temporary_path, 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ValueError("coordinator resume manifest already exists") from exc
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_coordinator_resume_manifest(
    path: Path, resume_token: str
) -> dict[str, object]:
    """Read and authenticate a bounded owner-only manifest without following links."""
    _validate_resume_token(resume_token)
    if path.is_symlink():
        raise ValueError("coordinator resume manifest must not be a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ValueError("coordinator resume manifest could not be opened") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("coordinator resume manifest must be a regular file")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError("coordinator resume manifest must be owner-only")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(MAX_COORDINATOR_STATE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_COORDINATOR_STATE_BYTES:
        raise ValueError("coordinator resume manifest exceeded the size limit")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("coordinator resume manifest is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("coordinator resume manifest must be a JSON object")
    manifest = dict(decoded)
    supplied_integrity = manifest.get("integrity_hmac_sha256")
    if (
        not isinstance(supplied_integrity, str)
        or HASH_PATTERN.fullmatch(supplied_integrity) is None
    ):
        raise ValueError("coordinator resume manifest authentication is invalid")
    expected = hmac.new(
        resume_token.encode("utf-8"),
        COORDINATOR_STATE_HMAC_CONTEXT
        + _canonical_json_bytes(_manifest_without_authentication(manifest)),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(supplied_integrity, expected):
        raise ValueError("coordinator resume manifest authentication failed")
    _validate_coordinator_resume_manifest_fields(manifest)
    return manifest


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


def _integer_value(
    mapping: dict[str, Any], key: str, scope: str, *, nonnegative: bool = False
) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{scope}.{key} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{scope}.{key} must be non-negative")
    return value


def build_remote_coordinated_summary(
    shards: list[dict[str, object]],
    max_start_skew_ms: float,
    max_clock_uncertainty_ms: float,
    coordinator_recovery: dict[str, object] | None = None,
) -> dict[str, object]:
    """Normalize authenticated agent windows and report conservative time bounds."""
    if (
        not math.isfinite(max_clock_uncertainty_ms)
        or max_clock_uncertainty_ms < 0
    ):
        raise ValueError("maximum clock uncertainty must be finite and non-negative")

    normalized_shards: list[dict[str, object]] = []
    agent_records: list[dict[str, object]] = []
    agent_hashes: set[str] = set()
    planned_starts: set[int] = set()
    uncertainties_ns: list[int] = []
    total_transport_retries = 0
    executed_results = 0
    cached_results = 0
    durable_results = 0
    recovered_results = 0
    configured_recovery_attempts: set[int] = set()

    for position, raw_shard in enumerate(shards):
        shard_data = dict(raw_shard)
        remote = _mapping(
            shard_data.pop("_remote_agent", None), f"shard[{position}]._remote_agent"
        )
        delivery = _mapping(
            shard_data.pop("_remote_result_delivery", None),
            f"shard[{position}]._remote_result_delivery",
        )
        coordination = _mapping(
            shard_data.get("coordination"), f"shard[{position}].coordination"
        )
        if coordination.get("clock") != "host_wall_clock":
            raise ValueError("remote child artifact must use its agent wall clock")

        agent_hash = remote.get("agent_id_sha256")
        if not isinstance(agent_hash, str) or HASH_PATTERN.fullmatch(agent_hash) is None:
            raise ValueError("remote agent identity must be lowercase SHA-256")
        if remote.get("agent_id_persisted") is not False:
            raise ValueError("remote agent did not confirm raw identity redaction")
        if remote.get("clock_selection") != "minimum_network_delay":
            raise ValueError("remote clock selection method is unsupported")
        result_source = delivery.get("result_source")
        if result_source not in {"executed", "cached", "durable"}:
            raise ValueError("remote result source is invalid")
        transport_retries = _integer_value(
            delivery,
            "transport_retries",
            f"shard[{position}]._remote_result_delivery",
            nonnegative=True,
        )
        if transport_retries > 3:
            raise ValueError("remote transport retry count exceeded the protocol limit")
        max_recovery_attempts = _integer_value(
            delivery,
            "max_transport_recovery_attempts",
            f"shard[{position}]._remote_result_delivery",
            nonnegative=True,
        )
        if max_recovery_attempts > 3 or transport_retries > max_recovery_attempts:
            raise ValueError("remote transport retries exceeded the configured limit")
        configured_recovery_attempts.add(max_recovery_attempts)
        total_transport_retries += transport_retries
        if result_source == "executed":
            executed_results += 1
        elif result_source == "cached":
            cached_results += 1
            if transport_retries > 0:
                recovered_results += 1
        else:
            durable_results += 1
            if transport_retries > 0:
                recovered_results += 1
        offset_ns = _integer_value(
            remote,
            "clock_offset_agent_minus_coordinator_ns",
            f"shard[{position}]._remote_agent",
        )
        network_delay_ns = _integer_value(
            remote,
            "clock_network_delay_ns",
            f"shard[{position}]._remote_agent",
            nonnegative=True,
        )
        uncertainty_ns = _integer_value(
            remote,
            "clock_uncertainty_ns",
            f"shard[{position}]._remote_agent",
            nonnegative=True,
        )
        sample_count = _integer_value(
            remote,
            "clock_sample_count",
            f"shard[{position}]._remote_agent",
            nonnegative=True,
        )
        if sample_count <= 0:
            raise ValueError("remote clock sample count must be positive")
        if uncertainty_ns != (network_delay_ns + 1) // 2:
            raise ValueError("remote clock uncertainty did not match network delay")
        planned_coordinator_ns = _integer_value(
            remote,
            "planned_start_coordinator_unix_ns",
            f"shard[{position}]._remote_agent",
            nonnegative=True,
        )
        if planned_coordinator_ns <= 0:
            raise ValueError("remote coordinator planned start must be positive")

        planned_agent_ns = _positive_int(
            coordination,
            "planned_start_unix_ns",
            f"shard[{position}].coordination",
        )
        if planned_agent_ns - offset_ns != planned_coordinator_ns:
            raise ValueError("remote planned start did not reconcile across clock domains")
        measured_start_agent_ns = _positive_int(
            coordination,
            "measured_start_unix_ns",
            f"shard[{position}].coordination",
        )
        measured_end_agent_ns = _positive_int(
            coordination,
            "measured_end_unix_ns",
            f"shard[{position}].coordination",
        )
        normalized_start_ns = measured_start_agent_ns - offset_ns
        normalized_end_ns = measured_end_agent_ns - offset_ns
        if normalized_start_ns <= 0 or normalized_end_ns <= normalized_start_ns:
            raise ValueError("remote normalized measured window is invalid")

        normalized_coordination = dict(coordination)
        normalized_coordination.update(
            {
                "planned_start_unix_ns": planned_coordinator_ns,
                "measured_start_unix_ns": normalized_start_ns,
                "measured_end_unix_ns": normalized_end_ns,
                "clock": "host_wall_clock",
            }
        )
        shard_data["coordination"] = normalized_coordination
        normalized_shards.append(shard_data)
        agent_hashes.add(agent_hash)
        planned_starts.add(planned_coordinator_ns)
        uncertainties_ns.append(uncertainty_ns)
        agent_records.append(
            {
                "agent_id_sha256": agent_hash,
                "clock_offset_agent_minus_coordinator_ns": offset_ns,
                "clock_network_delay_ms": round(network_delay_ns / 1_000_000, 6),
                "clock_uncertainty_ms": round(uncertainty_ns / 1_000_000, 6),
                "clock_sample_count": sample_count,
                "result_source": result_source,
                "transport_retries": transport_retries,
            }
        )

    if len(agent_hashes) != len(shards):
        raise ValueError("remote coordination requires unique agent identities")
    if len(planned_starts) != 1:
        raise ValueError("remote coordinator planned-start mismatch")
    if len(configured_recovery_attempts) != 1:
        raise ValueError("remote result recovery configuration was inconsistent")

    summary = build_coordinated_summary(normalized_shards, max_start_skew_ms)
    window = _mapping(summary.get("window"), "window")
    observed_union_seconds = _finite_number(
        window, "union_duration_seconds", "window"
    )
    observed_overlap_seconds = _finite_number(
        window, "overlap_duration_seconds", "window"
    )
    observed_start_skew_ms = _finite_number(window, "start_skew_ms", "window")
    max_uncertainty_ns = max(uncertainties_ns)
    two_clock_margin_ns = 2 * max_uncertainty_ns
    start_skew_upper_bound_ms = observed_start_skew_ms + (
        two_clock_margin_ns / 1_000_000
    )
    union_upper_bound_seconds = observed_union_seconds + (
        two_clock_margin_ns / 1_000_000_000
    )
    overlap_lower_bound_seconds = max(
        0.0,
        observed_overlap_seconds - (two_clock_margin_ns / 1_000_000_000),
    )
    max_uncertainty_ms = max_uncertainty_ns / 1_000_000

    failure_reasons: list[str] = []
    if max_uncertainty_ms > max_clock_uncertainty_ms:
        failure_reasons.append(
            f"clock uncertainty {max_uncertainty_ms:g} ms exceeded "
            f"{max_clock_uncertainty_ms:g} ms"
        )
    if start_skew_upper_bound_ms > max_start_skew_ms:
        failure_reasons.append(
            f"conservative client start skew {start_skew_upper_bound_ms:g} ms "
            f"exceeded {max_start_skew_ms:g} ms"
        )
    if overlap_lower_bound_seconds <= 0:
        failure_reasons.append(
            "client windows did not conservatively overlap after clock uncertainty"
        )

    observed_throughput = _finite_number(summary, "throughput_rps", "summary")
    successful_requests = _nonnegative_int(
        summary, "successful_requests", "summary"
    )
    summary.pop("throughput_rps")
    recovery = coordinator_recovery or {
        "state_enabled": False,
        "resumed_after_process_restart": False,
        "completed_statuses_verified_before_retrieval": 0,
        "missing_or_incomplete_shards_launched": False,
        "state_authentication": None,
        "resume_scope": "disabled",
    }
    summary.update(
        {
            "schema_version": 2,
            "scope": "authenticated_remote_agents",
            "agent_protocol": {
                "schema_version": 1,
                "authentication": "explicit_bearer_key_environment_variable",
                "transport_policy": "https_or_loopback_http",
                "replay_identity": "sha256_run_id_and_client_index",
                "result_recovery": "bounded_agent_selected_identical_request_store",
            },
            "agent_identity_fingerprints_sha256": sorted(agent_hashes),
            "result_delivery": {
                "max_transport_recovery_attempts_per_agent": next(
                    iter(configured_recovery_attempts)
                ),
                "transport_retries": total_transport_retries,
                "executed_results": executed_results,
                "cached_results": cached_results,
                "durable_results": durable_results,
                "recovered_after_transport_failure": recovered_results,
            },
            "coordinator_recovery": recovery,
            "clock_quality": {
                "method": "ntp_style_minimum_network_delay_sample",
                "max_uncertainty_ms": round(max_uncertainty_ms, 6),
                "max_allowed_uncertainty_ms": max_clock_uncertainty_ms,
                "passed": max_uncertainty_ms <= max_clock_uncertainty_ms,
                "agents": sorted(
                    agent_records, key=lambda item: str(item["agent_id_sha256"])
                ),
            },
            "window": {
                "clock": "coordinator_wall_clock_estimated_from_agent_samples",
                "planned_start_unix_ns": next(iter(planned_starts)),
                "observed_earliest_normalized_start_unix_ns": window[
                    "earliest_measured_start_unix_ns"
                ],
                "observed_latest_normalized_end_unix_ns": window[
                    "latest_measured_end_unix_ns"
                ],
                "observed_union_duration_seconds": round(
                    observed_union_seconds, 6
                ),
                "union_duration_upper_bound_seconds": round(
                    union_upper_bound_seconds, 6
                ),
                "observed_overlap_duration_seconds": round(
                    observed_overlap_seconds, 6
                ),
                "overlap_duration_lower_bound_seconds": round(
                    overlap_lower_bound_seconds, 6
                ),
                "observed_start_skew_ms": round(observed_start_skew_ms, 6),
                "start_skew_upper_bound_ms": round(
                    start_skew_upper_bound_ms, 6
                ),
            },
            "throughput": {
                "observed_normalized_rps": round(observed_throughput, 4),
                "conservative_lower_bound_rps": round(
                    successful_requests / union_upper_bound_seconds, 4
                ),
                "note": (
                    "Both values use clock-normalized agent windows. The lower bound "
                    "uses the union-duration upper bound from sampled clock uncertainty."
                ),
            },
            "coordination_gate": {
                "passed": not failure_reasons,
                "max_start_skew_ms": max_start_skew_ms,
                "max_clock_uncertainty_ms": max_clock_uncertainty_ms,
                "requires_conservative_overlapping_windows": True,
                "failure_reasons": failure_reasons,
            },
            "claim_boundary": (
                "This artifact proves authenticated agent protocol execution, sampled "
                "clock normalization, and conservative shard reconciliation. Agent "
                "identity fingerprints do not prove separate physical hosts. It does "
                "not prove partial-workflow continuation, production-network behavior, "
                "synchronized hardware clocks, a model or GPU, traffic isolation, or "
                "fleet scale."
            ),
        }
    )
    privacy = _mapping(summary.get("privacy"), "privacy")
    privacy.update(
        {
            "agent_urls_persisted": False,
            "authorization_persisted": False,
            "clock_challenges_persisted": False,
            "raw_agent_ids_persisted": False,
            "coordinator_state_path_persisted": False,
            "coordinator_resume_token_persisted": False,
            "coordinator_benchmark_arguments_persisted": False,
            "coordinator_result_bodies_persisted": False,
        }
    )
    summary["privacy"] = privacy
    return summary


def _format_remote_prometheus(summary: dict[str, object]) -> str:
    window = _mapping(summary.get("window"), "window")
    throughput = _mapping(summary.get("throughput"), "throughput")
    clock = _mapping(summary.get("clock_quality"), "clock_quality")
    gate = _mapping(summary.get("coordination_gate"), "coordination_gate")
    delivery = _mapping(summary.get("result_delivery"), "result_delivery")
    recovery = _mapping(summary.get("coordinator_recovery"), "coordinator_recovery")
    lines = [
        "# HELP triton_coordinated_clients Authenticated benchmark agent clients.",
        "# TYPE triton_coordinated_clients gauge",
        f"triton_coordinated_clients {_nonnegative_int(summary, 'client_count', 'summary')}",
        "# HELP triton_coordinated_requests_total Measured logical requests by outcome.",
        "# TYPE triton_coordinated_requests_total counter",
        'triton_coordinated_requests_total{outcome="success"} '
        f"{_nonnegative_int(summary, 'successful_requests', 'summary')}",
        'triton_coordinated_requests_total{outcome="failure"} '
        f"{_nonnegative_int(summary, 'failed_requests', 'summary')}",
        "# HELP triton_coordinated_client_attempts_total Physical client attempts across all agents.",
        "# TYPE triton_coordinated_client_attempts_total counter",
        f"triton_coordinated_client_attempts_total {_nonnegative_int(summary, 'client_attempts', 'summary')}",
        "# HELP triton_coordinated_window_observed_duration_seconds Clock-normalized observed union window.",
        "# TYPE triton_coordinated_window_observed_duration_seconds gauge",
        "triton_coordinated_window_observed_duration_seconds "
        f"{_finite_number(window, 'observed_union_duration_seconds', 'window'):g}",
        "# HELP triton_coordinated_window_duration_upper_bound_seconds Conservative union window after clock uncertainty.",
        "# TYPE triton_coordinated_window_duration_upper_bound_seconds gauge",
        "triton_coordinated_window_duration_upper_bound_seconds "
        f"{_finite_number(window, 'union_duration_upper_bound_seconds', 'window'):g}",
        "# HELP triton_coordinated_throughput_observed_rps Successful requests over the normalized observed union window.",
        "# TYPE triton_coordinated_throughput_observed_rps gauge",
        "triton_coordinated_throughput_observed_rps "
        f"{_finite_number(throughput, 'observed_normalized_rps', 'throughput'):g}",
        "# HELP triton_coordinated_throughput_lower_bound_rps Successful requests over the conservative union-window upper bound.",
        "# TYPE triton_coordinated_throughput_lower_bound_rps gauge",
        "triton_coordinated_throughput_lower_bound_rps "
        f"{_finite_number(throughput, 'conservative_lower_bound_rps', 'throughput'):g}",
        "# HELP triton_coordinated_start_skew_observed_ms Clock-normalized observed client start skew.",
        "# TYPE triton_coordinated_start_skew_observed_ms gauge",
        "triton_coordinated_start_skew_observed_ms "
        f"{_finite_number(window, 'observed_start_skew_ms', 'window'):g}",
        "# HELP triton_coordinated_start_skew_upper_bound_ms Conservative client start skew after clock uncertainty.",
        "# TYPE triton_coordinated_start_skew_upper_bound_ms gauge",
        "triton_coordinated_start_skew_upper_bound_ms "
        f"{_finite_number(window, 'start_skew_upper_bound_ms', 'window'):g}",
        "# HELP triton_coordinated_clock_uncertainty_max_ms Maximum selected agent clock uncertainty.",
        "# TYPE triton_coordinated_clock_uncertainty_max_ms gauge",
        "triton_coordinated_clock_uncertainty_max_ms "
        f"{_finite_number(clock, 'max_uncertainty_ms', 'clock_quality'):g}",
        "# HELP triton_coordinated_gate_passed Whether authentication, clock, overlap, and start-skew checks passed.",
        "# TYPE triton_coordinated_gate_passed gauge",
        f"triton_coordinated_gate_passed {1 if gate.get('passed') is True else 0}",
        "# HELP triton_coordinated_agent_transport_retries_total Ambiguous agent response failures retried by the coordinator.",
        "# TYPE triton_coordinated_agent_transport_retries_total counter",
        "triton_coordinated_agent_transport_retries_total "
        f"{_nonnegative_int(delivery, 'transport_retries', 'result_delivery')}",
        "# HELP triton_coordinated_agent_cached_results_total Completed results served from bounded agent memory.",
        "# TYPE triton_coordinated_agent_cached_results_total counter",
        "triton_coordinated_agent_cached_results_total "
        f"{_nonnegative_int(delivery, 'cached_results', 'result_delivery')}",
        "# HELP triton_coordinated_agent_durable_results_total Completed results served from opt-in durable agent state.",
        "# TYPE triton_coordinated_agent_durable_results_total counter",
        "triton_coordinated_agent_durable_results_total "
        f"{_nonnegative_int(delivery, 'durable_results', 'result_delivery')}",
        "# HELP triton_coordinated_agent_recovered_results_total Stored results recovered after an ambiguous transport failure.",
        "# TYPE triton_coordinated_agent_recovered_results_total counter",
        "triton_coordinated_agent_recovered_results_total "
        f"{_nonnegative_int(delivery, 'recovered_after_transport_failure', 'result_delivery')}",
        "# HELP triton_coordinated_coordinator_state_enabled Whether opt-in restart state was selected.",
        "# TYPE triton_coordinated_coordinator_state_enabled gauge",
        "triton_coordinated_coordinator_state_enabled "
        f"{1 if recovery.get('state_enabled') is True else 0}",
        "# HELP triton_coordinated_coordinator_resumed Whether this aggregate followed a coordinator process restart.",
        "# TYPE triton_coordinated_coordinator_resumed gauge",
        "triton_coordinated_coordinator_resumed "
        f"{1 if recovery.get('resumed_after_process_restart') is True else 0}",
        "# HELP triton_coordinated_resume_statuses_verified Completed agent states verified before restart retrieval.",
        "# TYPE triton_coordinated_resume_statuses_verified gauge",
        "triton_coordinated_resume_statuses_verified "
        f"{_nonnegative_int(recovery, 'completed_statuses_verified_before_retrieval', 'coordinator_recovery')}",
    ]
    configured_rate = summary.get("configured_aggregate_request_rate_rps")
    if isinstance(configured_rate, (int, float)) and not isinstance(
        configured_rate, bool
    ):
        lines.extend(
            [
                "# HELP triton_coordinated_configured_request_rate_rps Sum of configured per-agent open-loop request rates.",
                "# TYPE triton_coordinated_configured_request_rate_rps gauge",
                f"triton_coordinated_configured_request_rate_rps {float(configured_rate):g}",
            ]
        )
    return "\n".join(lines) + "\n"


def format_coordinated_prometheus(summary: dict[str, object]) -> str:
    """Export aggregate scalar evidence without run IDs, paths, or endpoints."""
    if summary.get("scope") == "authenticated_remote_agents":
        return _format_remote_prometheus(summary)

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
        description=(
            "Coordinate independent benchmark client processes locally or through "
            "authenticated remote agents."
        )
    )
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lead-time-ms", type=float, default=750.0)
    parser.add_argument("--max-start-skew-ms", type=float, default=100.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--agent-url",
        action="append",
        default=[],
        help="Authenticated agent base URL; repeat once per client.",
    )
    parser.add_argument(
        "--agent-api-key-env",
        help="Environment variable containing the bearer key for all selected agents.",
    )
    parser.add_argument("--clock-samples", type=int, default=5)
    parser.add_argument("--max-clock-uncertainty-ms", type=float, default=25.0)
    parser.add_argument(
        "--agent-run-recovery-attempts",
        type=int,
        default=1,
        help="Retries for ambiguous agent run-response failures (0 to 3).",
    )
    parser.add_argument(
        "--coordinator-state-file",
        type=Path,
        help=(
            "Optional owner-only authenticated manifest used to reconcile an "
            "already-completed agent run after coordinator restart."
        ),
    )
    parser.add_argument(
        "--coordinator-resume-token-env",
        help=(
            "Explicit environment variable containing the coordinator state "
            "authentication and run-derivation token."
        ),
    )
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
    if not 1 <= args.clock_samples <= 20:
        parser.error("--clock-samples must be between 1 and 20")
    if not 0 <= args.agent_run_recovery_attempts <= 3:
        parser.error("--agent-run-recovery-attempts must be between 0 and 3")
    if (
        not math.isfinite(args.max_clock_uncertainty_ms)
        or args.max_clock_uncertainty_ms < 0
    ):
        parser.error("--max-clock-uncertainty-ms must be finite and non-negative")

    try:
        args.agent_url = [validate_agent_base_url(url) for url in args.agent_url]
    except ValueError as exc:
        parser.error(str(exc))
    if args.agent_url:
        if len(args.agent_url) != args.clients:
            parser.error("--agent-url must be repeated exactly once per client")
        if len(set(args.agent_url)) != len(args.agent_url):
            parser.error("agent URLs must be unique")
        if not args.agent_api_key_env:
            parser.error("--agent-api-key-env is required with --agent-url")
        api_key = os.environ.get(args.agent_api_key_env)
        if (
            api_key is None
            or not 16 <= len(api_key) <= 4096
            or "\r" in api_key
            or "\n" in api_key
        ):
            parser.error(
                "the selected agent API-key environment variable must contain "
                "16 to 4096 characters"
            )
        args.agent_api_key = api_key
        state_selected = args.coordinator_state_file is not None
        token_selected = args.coordinator_resume_token_env is not None
        if state_selected != token_selected:
            parser.error(
                "--coordinator-state-file and --coordinator-resume-token-env "
                "must be supplied together"
            )
        if state_selected:
            if args.coordinator_resume_token_env == args.agent_api_key_env:
                parser.error(
                    "coordinator resume and agent API keys must use different "
                    "environment variables"
                )
            resume_token = os.environ.get(args.coordinator_resume_token_env)
            try:
                _validate_resume_token(resume_token)  # type: ignore[arg-type]
            except ValueError as exc:
                parser.error(str(exc))
            assert isinstance(resume_token, str)
            if hmac.compare_digest(resume_token, api_key):
                parser.error("coordinator resume and agent API keys must be distinct")
            output_path = Path(args.output_dir).absolute()
            state_path = args.coordinator_state_file.absolute()
            if state_path == output_path or output_path in state_path.parents:
                parser.error("coordinator state file must be outside the output directory")
            args.coordinator_state_file = state_path
            args.coordinator_resume_token = resume_token
        else:
            args.coordinator_resume_token = None
    else:
        if args.agent_api_key_env:
            parser.error("--agent-api-key-env requires at least one --agent-url")
        if args.coordinator_state_file or args.coordinator_resume_token_env:
            parser.error("coordinator restart state requires authenticated agents")
        args.agent_api_key = None
        args.coordinator_resume_token = None

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


def _run_local_clients(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
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
    return build_coordinated_summary(shards, args.max_start_skew_ms)


def _run_remote_clients(args: argparse.Namespace) -> dict[str, object]:
    api_key = args.agent_api_key
    if not isinstance(api_key, str):
        raise RuntimeError("remote agent authentication was not configured")
    state_path = args.coordinator_state_file
    resume_token = args.coordinator_resume_token
    state_enabled = isinstance(state_path, Path) and isinstance(resume_token, str)
    resumed = bool(state_enabled and (state_path.exists() or state_path.is_symlink()))
    manifest = (
        load_coordinator_resume_manifest(state_path, resume_token)
        if resumed and isinstance(state_path, Path) and isinstance(resume_token, str)
        else None
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.clients) as executor:
        clock_futures = [
            executor.submit(
                probe_agent_clock,
                agent_url,
                api_key=api_key,
                sample_count=args.clock_samples,
                timeout_seconds=min(args.timeout_seconds, 10.0),
            )
            for agent_url in args.agent_url
        ]
        current_clock_profiles = [future.result() for future in clock_futures]

    current_agent_hashes = [
        profile.get("agent_id_sha256") for profile in current_clock_profiles
    ]
    if (
        any(
            not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None
            for value in current_agent_hashes
        )
        or len(set(current_agent_hashes)) != args.clients
    ):
        raise RuntimeError("remote coordination requires unique stable agent identities")
    agent_hashes = [str(value) for value in current_agent_hashes]
    configuration_hash = build_coordinator_configuration_fingerprint(
        client_count=args.clients,
        benchmark_args=args.benchmark_args,
        timeout_seconds=args.timeout_seconds,
        max_start_skew_ms=args.max_start_skew_ms,
        max_clock_uncertainty_ms=args.max_clock_uncertainty_ms,
        agent_run_recovery_attempts=args.agent_run_recovery_attempts,
        agent_hashes=agent_hashes,
    )

    completed_statuses_verified = 0
    if manifest is not None:
        if not hmac.compare_digest(
            str(manifest["configuration_sha256"]), configuration_hash
        ):
            raise RuntimeError(
                "coordinator resume manifest does not match this configuration"
            )
        raw_agents = manifest.get("agents")
        assert isinstance(raw_agents, list)
        manifest_agents = sorted(
            [dict(value) for value in raw_agents if isinstance(value, dict)],
            key=lambda value: int(value["client_index"]),
        )
        if len(manifest_agents) != args.clients or [
            value.get("agent_id_sha256") for value in manifest_agents
        ] != agent_hashes:
            raise RuntimeError(
                "coordinator resume manifest agent identities do not match current agents"
            )
        planned_coordinator_ns = int(
            manifest["planned_start_coordinator_unix_ns"]
        )
        clock_profiles = [
            {
                key: value[key]
                for key in (
                    "agent_id_sha256",
                    "agent_id_persisted",
                    "clock_offset_agent_minus_coordinator_ns",
                    "clock_network_delay_ns",
                    "clock_uncertainty_ns",
                    "clock_sample_count",
                    "clock_selection",
                )
            }
            for value in manifest_agents
        ]
        planned_agent_starts = [
            int(value["planned_start_agent_unix_ns"]) for value in manifest_agents
        ]
        assert isinstance(resume_token, str)
        run_id = derive_coordinator_run_id(resume_token, manifest["workflow_nonce"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.clients) as executor:
            status_futures = [
                executor.submit(
                    query_agent_run_status,
                    agent_url,
                    api_key=api_key,
                    expected_agent_id_sha256=agent_hash,
                    run_id=run_id,
                    client_index=client_index,
                    client_count=args.clients,
                    planned_start_agent_unix_ns=planned_agent_starts[client_index],
                    benchmark_args=args.benchmark_args,
                    timeout_seconds=args.timeout_seconds,
                )
                for client_index, (agent_url, agent_hash) in enumerate(
                    zip(args.agent_url, agent_hashes)
                )
            ]
            statuses = [future.result() for future in status_futures]
        unavailable = [
            (index, status.get("state"))
            for index, status in enumerate(statuses)
            if status.get("state") != "completed"
            or status.get("result_available") is not True
        ]
        if unavailable:
            detail = ", ".join(
                f"client {index}: {state}" for index, state in unavailable
            )
            raise RuntimeError(
                "coordinator resume refused missing, incomplete, or expired shards; "
                f"no workload was launched ({detail})"
            )
        completed_statuses_verified = len(statuses)
    else:
        uncertainties_ns = [
            profile.get("clock_uncertainty_ns") for profile in current_clock_profiles
        ]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in uncertainties_ns
        ):
            raise RuntimeError("remote agent returned invalid clock uncertainty")
        max_uncertainty_ms = (
            max(int(value) for value in uncertainties_ns) / 1_000_000
        )
        if max_uncertainty_ms > args.max_clock_uncertainty_ms:
            raise RuntimeError(
                f"remote clock uncertainty {max_uncertainty_ms:g} ms exceeded "
                f"{args.max_clock_uncertainty_ms:g} ms before workload launch"
            )

        clock_profiles = current_clock_profiles
        planned_coordinator_ns = time.time_ns() + int(
            args.lead_time_ms * 1_000_000
        )
        planned_agent_starts = []
        for profile in clock_profiles:
            offset_ns = profile.get("clock_offset_agent_minus_coordinator_ns")
            if not isinstance(offset_ns, int) or isinstance(offset_ns, bool):
                raise RuntimeError("remote agent returned invalid clock offset")
            planned_agent_starts.append(planned_coordinator_ns + offset_ns)

        if state_enabled:
            assert isinstance(state_path, Path)
            assert isinstance(resume_token, str)
            manifest = create_coordinator_resume_manifest(
                resume_token=resume_token,
                configuration_sha256=configuration_hash,
                planned_start_coordinator_unix_ns=planned_coordinator_ns,
                clock_profiles=clock_profiles,
            )
            write_coordinator_resume_manifest(state_path, manifest)
            run_id = derive_coordinator_run_id(
                resume_token, manifest["workflow_nonce"]
            )
        else:
            run_id = secrets.token_hex(32)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.clients) as executor:
        run_futures = [
            executor.submit(
                run_agent_benchmark,
                agent_url,
                api_key=api_key,
                expected_agent_id_sha256=agent_hash,
                run_id=run_id,
                client_index=client_index,
                client_count=args.clients,
                planned_start_agent_unix_ns=planned_agent_starts[client_index],
                benchmark_args=args.benchmark_args,
                timeout_seconds=args.timeout_seconds,
                recovery_attempts=args.agent_run_recovery_attempts,
            )
            for client_index, (agent_url, agent_hash) in enumerate(
                zip(args.agent_url, agent_hashes)
            )
        ]
        shards = [future.result() for future in run_futures]

    for shard, profile in zip(shards, clock_profiles):
        shard["_remote_agent"] = {
            **profile,
            "planned_start_coordinator_unix_ns": planned_coordinator_ns,
        }
    recovery = {
        "state_enabled": state_enabled,
        "resumed_after_process_restart": resumed,
        "completed_statuses_verified_before_retrieval": (
            completed_statuses_verified
        ),
        "missing_or_incomplete_shards_launched": False,
        "state_authentication": (
            "hmac_sha256_explicit_resume_token" if state_enabled else None
        ),
        "resume_scope": (
            "completed_shards_only" if state_enabled else "disabled"
        ),
    }
    summary = build_remote_coordinated_summary(
        shards,
        max_start_skew_ms=args.max_start_skew_ms,
        max_clock_uncertainty_ms=args.max_clock_uncertainty_ms,
        coordinator_recovery=recovery,
    )
    if resumed:
        summary["claim_boundary"] = (
            "This artifact proves completed-result status verification, retrieval, "
            "and reconciliation after one coordinator process restart. Resume refuses "
            "missing, incomplete, expired, or conflicting shards rather than launching "
            "work against the stale clock plan. It does not prove partial-workflow or "
            "interrupted-child continuation, agent recovery without retained results, "
            "production-network behavior, synchronized hardware clocks, a model or "
            "GPU, traffic isolation, or fleet scale."
        )
    return summary


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("coordinated output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.agent_url:
        summary = _run_remote_clients(args)
    else:
        summary = _run_local_clients(args, output_dir)

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
