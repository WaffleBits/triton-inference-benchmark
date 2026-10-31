"""Prometheus parsing, request-path accounting and telemetry gates."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from bench_core.config import (
    GPU_GAUGE_METRIC_ALIASES,
    GPU_MEMORY_COPY_METRICS,
    GPU_MEMORY_USED_METRICS,
    GPU_UTILIZATION_METRICS,
    GpuGaugeCapture,
    PROMETHEUS_METRIC_NAME_PATTERN,
    PrometheusSample,
    RequestPathMetricNames,
    TELEMETRY_COUNTER_KEYS,
    TRITON_COMPUTE_INFER_DURATION_METRICS,
    TRITON_COUNTER_METRIC_ALIASES,
    TRITON_QUEUE_DURATION_METRICS,
    TRITON_REQUEST_DURATION_METRICS,
    TRITON_REQUEST_FAILURE_METRICS,
    TRITON_REQUEST_SUCCESS_METRICS,
    percentile,
)


def _split_prometheus_labels(raw_labels: str) -> list[str]:
    labels: list[str] = []
    current: list[str] = []
    in_quotes = False
    escaped = False

    for char in raw_labels:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
            continue
        if char == "," and not in_quotes:
            labels.append("".join(current))
            current = []
            continue
        current.append(char)

    if current:
        labels.append("".join(current))
    return labels


def _unescape_prometheus_label(value: str) -> str:
    return value.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


def _parse_metric_and_labels(metric_with_labels: str) -> tuple[str, dict[str, str]]:
    if "{" not in metric_with_labels:
        return metric_with_labels, {}

    metric_name, raw_labels = metric_with_labels.split("{", 1)
    raw_labels = raw_labels.rstrip("}")
    labels: dict[str, str] = {}

    for raw_label in _split_prometheus_labels(raw_labels):
        if "=" not in raw_label:
            continue
        key, raw_value = raw_label.split("=", 1)
        raw_value = raw_value.strip()
        if len(raw_value) >= 2 and raw_value[0] == '"' and raw_value[-1] == '"':
            labels[key.strip()] = _unescape_prometheus_label(raw_value[1:-1])

    return metric_name, labels


def _split_prometheus_sample_line(line: str, line_number: int) -> tuple[str, str]:
    in_braces = False
    in_quotes = False
    escaped = False

    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            continue
        if char == "{" and not in_quotes:
            in_braces = True
            continue
        if char == "}" and not in_quotes:
            in_braces = False
            continue
        if char.isspace() and not in_braces and not in_quotes:
            metric_with_labels = line[:index]
            remainder = line[index:].strip()
            if metric_with_labels and remainder:
                return metric_with_labels, remainder
            break

    raise ValueError(f"Invalid Prometheus sample on line {line_number}: {line}")


def parse_prometheus_samples(prometheus_text: str) -> list[PrometheusSample]:
    samples: list[PrometheusSample] = []

    for line_number, raw_line in enumerate(prometheus_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        metric_with_labels, remainder = _split_prometheus_sample_line(line, line_number)
        metric_name, labels = _parse_metric_and_labels(metric_with_labels)
        raw_value = remainder.split(maxsplit=1)[0]
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid Prometheus sample value on line {line_number}: {raw_value}"
            ) from exc
        samples.append(PrometheusSample(metric_name, labels, value))

    return samples


def _sample_matches_model(sample: PrometheusSample, model_name: str) -> bool:
    if "model" in sample.labels:
        return sample.labels["model"] == model_name
    if "model_name" in sample.labels:
        return sample.labels["model_name"] == model_name
    return True


def _values_for_metrics(
    samples: list[PrometheusSample],
    metric_names: set[str],
    model_name: str | None = None,
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        if sample.metric not in metric_names:
            continue
        if model_name and not _sample_matches_model(sample, model_name):
            continue
        value = sample.value
        if sample.metric.endswith("_bytes"):
            value = value / (1024 * 1024)
        values.append(value)
    return values


def _build_series_membership(
    samples: list[PrometheusSample],
    metric_aliases: dict[str, str],
    model_name: str | None = None,
) -> dict[str, object]:
    """Hash selected Prometheus series identities without retaining raw labels."""
    identities: set[str] = set()
    for sample in samples:
        logical_metric = metric_aliases.get(sample.metric)
        if logical_metric is None:
            continue
        if model_name and not _sample_matches_model(sample, model_name):
            continue
        identities.add(
            json.dumps(
                [logical_metric, sorted(sample.labels.items())],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    ordered_identities = sorted(identities)
    fingerprint = (
        hashlib.sha256(
            json.dumps(
                ordered_identities,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if ordered_identities
        else None
    )
    return {
        "series_count": len(ordered_identities),
        "fingerprint_sha256": fingerprint,
        "fingerprint_scope": (
            "logical metric names and sorted labels; sample values excluded; "
            "raw series identities not persisted"
        ),
    }


def _request_path_counter_series(
    samples: list[PrometheusSample],
    metric_name: str,
) -> tuple[dict[str, float], set[str]]:
    """Select one counter family while retaining labels only in ephemeral keys."""
    values: dict[str, float] = {}
    issues: set[str] = set()
    for sample in samples:
        if sample.metric != metric_name:
            continue
        identity = json.dumps(
            sorted(sample.labels.items()),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if identity in values:
            issues.add("duplicate_series")
        values[identity] = sample.value
        if not math.isfinite(sample.value):
            issues.add("non_finite_counter")
        elif sample.value < 0:
            issues.add("negative_counter")
        elif not sample.value.is_integer():
            issues.add("non_integral_counter")
    return values, issues


def _request_path_membership_fingerprint(stage: str, identities: set[str]) -> str | None:
    if not identities:
        return None
    payload = json.dumps(
        [[stage, identity] for identity in sorted(identities)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_request_path_stage_window(
    stage: str,
    metric_name: str,
    before_samples: list[PrometheusSample],
    after_samples: list[PrometheusSample],
) -> dict[str, object]:
    before, before_issues = _request_path_counter_series(before_samples, metric_name)
    after, after_issues = _request_path_counter_series(after_samples, metric_name)
    issues = set(before_issues) | set(after_issues)
    if not before or not after:
        issues.add("counter_missing")

    before_identities = set(before)
    after_identities = set(after)
    membership_stable = bool(before_identities) and before_identities == after_identities
    if before_identities and after_identities and not membership_stable:
        issues.add("series_membership_changed")

    delta: int | None = None
    if membership_stable and not issues:
        deltas = [after[key] - before[key] for key in sorted(before)]
        if any(value < 0 for value in deltas):
            issues.add("counter_reset")
        else:
            delta = int(sum(deltas))

    before_fingerprint = _request_path_membership_fingerprint(stage, before_identities)
    after_fingerprint = _request_path_membership_fingerprint(stage, after_identities)
    membership: dict[str, object] = {
        "stable": membership_stable,
        "before_series_count": len(before_identities),
        "after_series_count": len(after_identities),
        "series_count": len(before_identities) if membership_stable else None,
        "fingerprint_sha256": before_fingerprint if membership_stable else None,
        "fingerprint_scope": (
            "fixed request-path stage and sorted labels; sample values and raw "
            "series identities excluded"
        ),
    }
    if before_fingerprint != after_fingerprint:
        membership["before_fingerprint_sha256"] = before_fingerprint
        membership["after_fingerprint_sha256"] = after_fingerprint

    return {
        "valid": not issues,
        "delta": delta,
        "metric_name_sha256": hashlib.sha256(metric_name.encode("utf-8")).hexdigest(),
        "series_membership": membership,
        "issues": sorted(issues),
    }


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def build_request_path_accounting(
    metrics: dict[str, object],
    before_prometheus_text: str,
    after_prometheus_text: str,
    metric_names: RequestPathMetricNames,
    *,
    source: str = "paired_prometheus_counter_snapshots",
    alignment: str = "operator_supplied_unverified",
) -> dict[str, object]:
    """Correlate privacy-safe aggregate ingress/backend/success counter deltas."""
    names = metric_names.by_stage()
    if len(set(names.values())) != len(names):
        raise ValueError("request-path metric names must be distinct")
    for metric_name in names.values():
        if PROMETHEUS_METRIC_NAME_PATTERN.fullmatch(metric_name) is None:
            raise ValueError(f"invalid Prometheus metric name: {metric_name!r}")

    before_samples = parse_prometheus_samples(before_prometheus_text)
    after_samples = parse_prometheus_samples(after_prometheus_text)
    stages = {
        stage: _build_request_path_stage_window(
            stage,
            metric_name,
            before_samples,
            after_samples,
        )
        for stage, metric_name in names.items()
    }

    retry = metrics.get("retry")
    retry = retry if isinstance(retry, dict) else {}
    logical_requests = _nonnegative_int(retry.get("logical_requests"))
    client_attempts = _nonnegative_int(retry.get("client_attempts"))
    successful_requests = _nonnegative_int(metrics.get("successful_requests"))

    issues = {
        issue
        for stage_summary in stages.values()
        for issue in stage_summary.get("issues", [])
        if isinstance(issue, str)
    }
    if (
        logical_requests is None
        or client_attempts is None
        or successful_requests is None
    ):
        issues.add("client_summary_unavailable")

    ingress = stages["ingress"].get("delta")
    backend = stages["backend"].get("delta")
    success = stages["success"].get("delta")

    def ratio(numerator: object, denominator: object) -> float | None:
        if (
            isinstance(numerator, int)
            and not isinstance(numerator, bool)
            and isinstance(denominator, int)
            and not isinstance(denominator, bool)
            and denominator > 0
        ):
            return round(numerator / denominator, 6)
        return None

    notes = [
        "counter deltas are aggregate observations and do not establish per-request causality",
        "the selected counter series must be isolated to this run for exact accounting",
        "metric names and labels are represented only by SHA-256 fingerprints",
    ]
    if alignment == "harness_bracketed_measured_phase":
        notes.insert(
            0,
            "the harness fetched counters immediately before and after measured requests",
        )
    else:
        notes[:0] = [
            "snapshot timing is operator supplied and not verified by the harness",
        ]

    return {
        "source": source,
        "alignment": alignment,
        "valid": not issues,
        "client": {
            "logical_requests": logical_requests,
            "client_attempts": client_attempts,
            "successful_requests": successful_requests,
        },
        "stages": stages,
        "ratios": {
            "ingress_per_client_attempt": ratio(ingress, client_attempts),
            "backend_per_ingress": ratio(backend, ingress),
            "success_per_backend": ratio(success, backend),
        },
        "issues": sorted(issues),
        "notes": notes,
    }


def build_request_path_gate(accounting: dict[str, object]) -> dict[str, object]:
    """Fail closed on exact request-path accounting for an isolated counter scope."""
    client = accounting.get("client")
    stages = accounting.get("stages")
    client = client if isinstance(client, dict) else {}
    stages = stages if isinstance(stages, dict) else {}

    def stage_delta(stage: str) -> int | None:
        summary = stages.get(stage)
        return (
            _nonnegative_int(summary.get("delta"))
            if isinstance(summary, dict)
            else None
        )

    client_attempts = _nonnegative_int(client.get("client_attempts"))
    successful_requests = _nonnegative_int(client.get("successful_requests"))
    ingress = stage_delta("ingress")
    backend = stage_delta("backend")
    success = stage_delta("success")

    ingress_evaluable = ingress is not None and client_attempts is not None
    ingress_passed = ingress_evaluable and ingress == client_attempts
    order_evaluable = ingress is not None and backend is not None and success is not None
    order_passed = order_evaluable and ingress >= backend >= success
    success_evaluable = success is not None and successful_requests is not None
    success_passed = success_evaluable and success == successful_requests

    checks = {
        "ingress_matches_client_attempts": {
            "observed": ingress,
            "expected": client_attempts,
            "evaluable": ingress_evaluable,
            "passed": ingress_passed,
        },
        "stage_counts_non_increasing": {
            "ingress": ingress,
            "backend": backend,
            "success": success,
            "evaluable": order_evaluable,
            "passed": order_passed,
        },
        "success_matches_successful_requests": {
            "observed": success,
            "expected": successful_requests,
            "evaluable": success_evaluable,
            "passed": success_passed,
        },
    }
    failure_reasons: list[str] = []
    if accounting.get("valid") is not True:
        failure_reasons.append("request-path counter evidence is invalid")
    if not ingress_passed:
        failure_reasons.append("ingress receipts did not equal measured client attempts")
    if not order_passed:
        failure_reasons.append("request-path stage counts were unavailable or out of order")
    if not success_passed:
        failure_reasons.append(
            "success receipts did not equal successful measured logical requests"
        )

    return {
        "passed": not failure_reasons,
        "checks": checks,
        "failure_reasons": failure_reasons,
        "scope": (
            "exact aggregate accounting for counter series isolated to the measured run; "
            "no per-request causality is claimed"
        ),
    }


def build_service_lifecycle_accounting(
    before_prometheus_text: str,
    after_prometheus_text: str,
    metric_name: str,
    *,
    source: str = "paired_prometheus_counter_snapshots",
    alignment: str = "operator_supplied_unverified",
) -> dict[str, object]:
    """Derive a privacy-safe completed service-restart counter delta."""
    if PROMETHEUS_METRIC_NAME_PATTERN.fullmatch(metric_name) is None:
        raise ValueError(f"invalid Prometheus metric name: {metric_name!r}")

    before_samples = parse_prometheus_samples(before_prometheus_text)
    after_samples = parse_prometheus_samples(after_prometheus_text)
    window = _build_request_path_stage_window(
        "service_restart",
        metric_name,
        before_samples,
        after_samples,
    )
    issues = window.get("issues", [])
    notes = [
        "the selected cumulative counter is an aggregate observation",
        "a restart delta does not prove which request, retry, or outcome it affected",
        "restart count is not process downtime, readiness duration, or service MTTR",
        "the metric name and labels are represented only by SHA-256 fingerprints",
    ]
    if alignment == "harness_bracketed_measured_phase":
        notes.insert(
            0,
            "the harness fetched the counter immediately before and after measured requests",
        )
    else:
        notes.insert(0, "snapshot timing is operator supplied and not verified by the harness")

    return {
        "source": source,
        "alignment": alignment,
        "valid": window.get("valid") is True,
        "restart_count": window.get("delta"),
        "metric_name_sha256": window.get("metric_name_sha256"),
        "series_membership": window.get("series_membership"),
        "issues": list(issues) if isinstance(issues, list) else [],
        "notes": notes,
    }


def build_service_lifecycle_gate(
    accounting: dict[str, object],
    *,
    min_service_restarts: int | None = None,
    max_service_restarts: int | None = None,
) -> dict[str, object]:
    """Gate a bracketed service-restart count against explicit run bounds."""
    if min_service_restarts is None and max_service_restarts is None:
        raise ValueError("service lifecycle gate requires a restart bound")
    for label, value in (
        ("minimum", min_service_restarts),
        ("maximum", max_service_restarts),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{label} service restarts must be a non-negative integer")
    if (
        min_service_restarts is not None
        and max_service_restarts is not None
        and min_service_restarts > max_service_restarts
    ):
        raise ValueError("minimum service restarts must not exceed maximum")

    observed = _nonnegative_int(accounting.get("restart_count"))
    checks: dict[str, dict[str, object]] = {}
    if min_service_restarts is not None:
        checks["minimum_restarts"] = {
            "observed": observed,
            "minimum": min_service_restarts,
            "evaluable": observed is not None,
            "passed": observed is not None and observed >= min_service_restarts,
        }
    if max_service_restarts is not None:
        checks["maximum_restarts"] = {
            "observed": observed,
            "maximum": max_service_restarts,
            "evaluable": observed is not None,
            "passed": observed is not None and observed <= max_service_restarts,
        }

    failure_reasons: list[str] = []
    if accounting.get("valid") is not True:
        failure_reasons.append("service lifecycle counter evidence is invalid")
    if observed is None:
        failure_reasons.append("service restart count is unavailable")
    else:
        if min_service_restarts is not None and observed < min_service_restarts:
            failure_reasons.append(
                f"service restart count {observed} was below {min_service_restarts} minimum"
            )
        if max_service_restarts is not None and observed > max_service_restarts:
            failure_reasons.append(
                f"service restart count {observed} exceeded {max_service_restarts} maximum"
            )

    return {
        "passed": not failure_reasons,
        "checks": checks,
        "failure_reasons": failure_reasons,
        "scope": (
            "aggregate completed-restart counter delta inside the selected telemetry "
            "window; no request-level causality or MTTR is claimed"
        ),
    }


def _stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "avg": round(statistics.fmean(values), 4),
        "max": round(max(values), 4),
    }


def _sum_values(values: list[float]) -> float:
    return round(sum(values), 4)


def _gauge_window_stat_summary(
    values: list[float],
    matched_scrape_count: int,
) -> dict[str, float | int]:
    if not values:
        return {}
    return {
        "sample_count": len(values),
        "matched_scrape_count": matched_scrape_count,
        "avg": round(statistics.fmean(values), 4),
        "min": round(min(values), 4),
        "p50": round(percentile(values, 50), 4),
        "p95": round(percentile(values, 95), 4),
        "max": round(max(values), 4),
    }


def _extract_gpu_gauge_values(prometheus_text: str) -> dict[str, list[float]]:
    samples = parse_prometheus_samples(prometheus_text)
    return _extract_gpu_gauge_values_from_samples(samples)


def _extract_gpu_gauge_values_from_samples(
    samples: list[PrometheusSample],
) -> dict[str, list[float]]:
    return {
        "utilization_pct": _values_for_metrics(samples, GPU_UTILIZATION_METRICS),
        "memory_copy_utilization_pct": _values_for_metrics(
            samples,
            GPU_MEMORY_COPY_METRICS,
        ),
        "memory_used_mib": _values_for_metrics(samples, GPU_MEMORY_USED_METRICS),
    }


def _build_gpu_gauge_capture(prometheus_text: str) -> GpuGaugeCapture:
    samples = parse_prometheus_samples(prometheus_text)
    return GpuGaugeCapture(
        values=_extract_gpu_gauge_values_from_samples(samples),
        series_membership=_build_series_membership(
            samples,
            GPU_GAUGE_METRIC_ALIASES,
        ),
    )


def _build_gpu_gauge_window(
    gauge_captures: list[dict[str, list[float]]],
    series_memberships: list[dict[str, object]],
    *,
    in_window_scrape_count: int,
    configured_interval_seconds: float,
) -> dict[str, object]:
    if len(gauge_captures) < 3:
        raise ValueError("sampled GPU gauge windows require at least three scrapes")
    if not 0 < in_window_scrape_count <= len(gauge_captures) - 2:
        raise ValueError("in-window scrape count must match the sampled gauge window")
    if configured_interval_seconds <= 0:
        raise ValueError("configured telemetry sample interval must be greater than zero")
    if len(series_memberships) != len(gauge_captures):
        raise ValueError("every GPU gauge scrape requires a series-membership summary")

    fingerprints = [
        membership.get("fingerprint_sha256") for membership in series_memberships
    ]
    present_fingerprints = [
        fingerprint for fingerprint in fingerprints if isinstance(fingerprint, str)
    ]
    if present_fingerprints and (
        len(present_fingerprints) != len(fingerprints)
        or len(set(present_fingerprints)) != 1
    ):
        raise ValueError("GPU telemetry series membership changed across sampled window")

    membership_evaluable = len(present_fingerprints) == len(fingerprints)
    membership_fingerprint = (
        present_fingerprints[0] if membership_evaluable else None
    )
    membership_series_count = (
        int(series_memberships[0].get("series_count", 0))
        if membership_evaluable
        else 0
    )

    gauge_names = (
        "utilization_pct",
        "memory_copy_utilization_pct",
        "memory_used_mib",
    )
    values_by_metric: dict[str, list[float]] = {name: [] for name in gauge_names}
    matched_scrapes = {name: 0 for name in gauge_names}
    for capture in gauge_captures:
        for logical_name in gauge_names:
            values = capture.get(logical_name, [])
            if values:
                matched_scrapes[logical_name] += 1
                values_by_metric[logical_name].extend(values)

    notes = [
        "GPU values are sample statistics across bounded scrapes and are not time-weighted",
        (
            "the pre-boundary and post-boundary scrapes are included with samples "
            "initiated during measured work"
        ),
        "known GPU series membership is fingerprinted without persisting raw labels",
        "a shared telemetry endpoint can include activity unrelated to this benchmark",
    ]
    if not any(values_by_metric.values()):
        notes.append("no GPU telemetry samples matched known DCGM metric names")

    return {
        "source": "sampled_http_prometheus_scrapes",
        "alignment": "harness_bracketed_measured_phase",
        "scrape_count": len(gauge_captures),
        "boundary_scrape_count": 2,
        "in_window_scrape_count": in_window_scrape_count,
        "configured_interval_seconds": configured_interval_seconds,
        "series_membership": {
            "evaluable": membership_evaluable,
            "stable": True if membership_evaluable else None,
            "series_count": membership_series_count,
            "fingerprint_sha256": membership_fingerprint,
            "fingerprint_scope": (
                "logical metric names and sorted labels; sample values excluded; "
                "raw series identities not persisted"
            ),
        },
        "gpu": {
            name: _gauge_window_stat_summary(
                values_by_metric[name],
                matched_scrapes[name],
            )
            for name in gauge_names
        },
        "notes": notes,
    }


def build_gpu_gauge_window(
    prometheus_snapshots: list[str],
    *,
    in_window_scrape_count: int,
    configured_interval_seconds: float,
) -> dict[str, object]:
    """Aggregate DCGM gauges across bounded pre, in-window, and post scrapes."""
    captures = [_build_gpu_gauge_capture(text) for text in prometheus_snapshots]
    return _build_gpu_gauge_window(
        [capture.values for capture in captures],
        [capture.series_membership for capture in captures],
        in_window_scrape_count=in_window_scrape_count,
        configured_interval_seconds=configured_interval_seconds,
    )


def build_telemetry_summary(
    prometheus_text: str,
    model_name: str,
    source: str = "prometheus_snapshot",
) -> dict[str, object]:
    samples = parse_prometheus_samples(prometheus_text)
    gpu_utilization = _values_for_metrics(samples, GPU_UTILIZATION_METRICS)
    gpu_memory_copy = _values_for_metrics(samples, GPU_MEMORY_COPY_METRICS)
    gpu_memory_used = _values_for_metrics(samples, GPU_MEMORY_USED_METRICS)

    triton_success = _values_for_metrics(
        samples,
        TRITON_REQUEST_SUCCESS_METRICS,
        model_name=model_name,
    )
    triton_failure = _values_for_metrics(
        samples,
        TRITON_REQUEST_FAILURE_METRICS,
        model_name=model_name,
    )
    triton_request_duration = _values_for_metrics(
        samples,
        TRITON_REQUEST_DURATION_METRICS,
        model_name=model_name,
    )
    triton_queue_duration = _values_for_metrics(
        samples,
        TRITON_QUEUE_DURATION_METRICS,
        model_name=model_name,
    )
    triton_compute_infer_duration = _values_for_metrics(
        samples,
        TRITON_COMPUTE_INFER_DURATION_METRICS,
        model_name=model_name,
    )
    series_membership = {
        "gpu_gauges": _build_series_membership(
            samples,
            GPU_GAUGE_METRIC_ALIASES,
        ),
        "triton_counters": _build_series_membership(
            samples,
            TRITON_COUNTER_METRIC_ALIASES,
            model_name=model_name,
        ),
    }

    notes: list[str] = []
    if not any((gpu_utilization, gpu_memory_copy, gpu_memory_used)):
        notes.append("no GPU telemetry samples matched known DCGM metric names")
    if not any(
        (
            triton_success,
            triton_failure,
            triton_request_duration,
            triton_queue_duration,
            triton_compute_infer_duration,
        )
    ):
        notes.append("no Triton server samples matched the configured model name")

    return {
        "source": source,
        "sample_count": len(samples),
        "series_membership": series_membership,
        "gpu": {
            "utilization_pct": _stat_summary(gpu_utilization),
            "memory_copy_utilization_pct": _stat_summary(gpu_memory_copy),
            "memory_used_mib": _stat_summary(gpu_memory_used),
        },
        "triton": {
            "model_name": model_name,
            "request_success_total": _sum_values(triton_success),
            "request_failure_total": _sum_values(triton_failure),
            "request_duration_us_total": _sum_values(triton_request_duration),
            "queue_duration_us_total": _sum_values(triton_queue_duration),
            "compute_infer_duration_us_total": _sum_values(triton_compute_infer_duration),
            "counter_sample_counts": {
                "request_success": len(triton_success),
                "request_failure": len(triton_failure),
                "request_duration_us": len(triton_request_duration),
                "queue_duration_us": len(triton_queue_duration),
                "compute_infer_duration_us": len(triton_compute_infer_duration),
            },
        },
        "notes": notes,
    }


def build_telemetry_counter_window(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    source: str = "paired_prometheus_counter_snapshots",
    alignment: str = "operator_supplied_unverified",
) -> dict[str, object]:
    """Derive observed-window values from paired cumulative counter snapshots."""
    baseline_triton = baseline.get("triton", {})
    candidate_triton = candidate.get("triton", {})
    if not isinstance(baseline_triton, dict) or not isinstance(candidate_triton, dict):
        raise ValueError("telemetry summaries must include Triton counter records")

    baseline_counts = baseline_triton.get("counter_sample_counts", {})
    candidate_counts = candidate_triton.get("counter_sample_counts", {})
    if not isinstance(baseline_counts, dict) or not isinstance(candidate_counts, dict):
        raise ValueError("telemetry summaries must include counter sample counts")

    baseline_memberships = baseline.get("series_membership", {})
    candidate_memberships = candidate.get("series_membership", {})
    if not isinstance(baseline_memberships, dict):
        baseline_memberships = {}
    if not isinstance(candidate_memberships, dict):
        candidate_memberships = {}
    baseline_membership = baseline_memberships.get("triton_counters", {})
    candidate_membership = candidate_memberships.get("triton_counters", {})
    if not isinstance(baseline_membership, dict):
        baseline_membership = {}
    if not isinstance(candidate_membership, dict):
        candidate_membership = {}
    before_fingerprint = baseline_membership.get("fingerprint_sha256")
    after_fingerprint = candidate_membership.get("fingerprint_sha256")
    membership_evaluable = isinstance(before_fingerprint, str) and isinstance(
        after_fingerprint,
        str,
    )
    membership_stable = (
        membership_evaluable and before_fingerprint == after_fingerprint
    )
    before_series_count = int(baseline_membership.get("series_count", 0))
    after_series_count = int(candidate_membership.get("series_count", 0))
    series_membership: dict[str, object] = {
        "evaluable": membership_evaluable,
        "stable": membership_stable,
        "before_series_count": before_series_count,
        "after_series_count": after_series_count,
        "series_count": before_series_count if membership_stable else None,
        "fingerprint_sha256": before_fingerprint if membership_stable else None,
        "fingerprint_scope": (
            "logical metric names and sorted labels; sample values excluded; "
            "raw series identities not persisted"
        ),
    }
    if membership_evaluable and not membership_stable:
        series_membership["before_fingerprint_sha256"] = before_fingerprint
        series_membership["after_fingerprint_sha256"] = after_fingerprint

    deltas: dict[str, float | None] = {}
    unavailable_counters: list[str] = []
    counter_resets: list[str] = []
    for logical_name, summary_key in TELEMETRY_COUNTER_KEYS.items():
        if not baseline_counts.get(logical_name) or not candidate_counts.get(logical_name):
            deltas[logical_name] = None
            unavailable_counters.append(logical_name)
            continue

        before = baseline_triton.get(summary_key)
        after = candidate_triton.get(summary_key)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            deltas[logical_name] = None
            unavailable_counters.append(logical_name)
            continue
        if after < before:
            deltas[logical_name] = None
            counter_resets.append(logical_name)
            continue
        deltas[logical_name] = round(float(after) - float(before), 4)

    success_delta = deltas["request_success"]
    failure_delta = deltas["request_failure"]
    request_total = (
        round(success_delta + failure_delta, 4)
        if success_delta is not None and failure_delta is not None
        else None
    )
    server_failure_rate = (
        round(failure_delta / request_total, 6)
        if failure_delta is not None and request_total is not None and request_total > 0
        else None
    )

    request_duration_delta = deltas["request_duration_us"]
    queue_duration_delta = deltas["queue_duration_us"]
    server_queue_fraction = (
        round(queue_duration_delta / request_duration_delta, 6)
        if queue_duration_delta is not None
        and request_duration_delta is not None
        and request_duration_delta > 0
        else None
    )

    notes = [
        "DCGM gauges remain point-in-time values in the post-run telemetry summary",
        "counter-series membership is compared by a privacy-preserving fingerprint",
        "counter deltas may include unrelated server traffic even when membership is stable",
    ]
    if alignment == "harness_bracketed_measured_phase":
        notes.insert(
            0,
            "the harness fetched counters immediately before and after its measured request phase",
        )
    else:
        notes[:0] = [
            "values are deltas between operator-supplied before/after Triton counters",
            "the harness cannot prove that supplied snapshots bracket this invocation",
        ]

    return {
        "source": source,
        "alignment": alignment,
        "valid": (
            not unavailable_counters
            and not counter_resets
            and membership_stable
        ),
        "series_membership": series_membership,
        "counter_resets": sorted(counter_resets),
        "unavailable_counters": sorted(unavailable_counters),
        "deltas": deltas,
        "derived": {
            "request_total": request_total,
            "server_failure_rate": server_failure_rate,
            "server_queue_fraction": server_queue_fraction,
        },
        "notes": notes,
    }


def build_telemetry_gate(
    counter_window: dict[str, object],
    max_server_failure_rate: float | None = None,
    max_server_queue_fraction: float | None = None,
) -> dict[str, object]:
    """Evaluate configured server-side thresholds and fail closed when unevaluable."""
    derived = counter_window.get("derived", {})
    if not isinstance(derived, dict):
        derived = {}

    checks: dict[str, dict[str, object]] = {}
    failure_reasons: list[str] = []
    series_membership = counter_window.get("series_membership")
    if isinstance(series_membership, dict):
        if not series_membership.get("evaluable"):
            failure_reasons.append(
                "telemetry series membership unavailable from paired counter snapshots"
            )
        elif series_membership.get("stable") is not True:
            failure_reasons.append(
                "telemetry series membership changed between paired counter snapshots"
            )
    specifications = (
        (
            "server_failure_rate",
            max_server_failure_rate,
            "server failure rate",
        ),
        (
            "server_queue_fraction",
            max_server_queue_fraction,
            "server queue fraction",
        ),
    )
    for key, maximum, label in specifications:
        if maximum is None:
            continue
        observed = derived.get(key)
        evaluable = isinstance(observed, (int, float))
        passed = evaluable and float(observed) <= maximum
        checks[key] = {
            "observed": observed if evaluable else None,
            "maximum": maximum,
            "evaluable": evaluable,
            "passed": passed,
        }
        if not evaluable:
            failure_reasons.append(
                f"{label} unavailable from paired Prometheus counter snapshots"
            )
        elif not passed:
            failure_reasons.append(
                f"{label} {float(observed):g} exceeded {maximum:g} threshold"
            )

    return {
        "passed": bool(checks) and not failure_reasons,
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def attach_telemetry_summary(
    metrics: dict[str, object],
    telemetry_prometheus_path: str | Path,
    telemetry_baseline_prometheus_path: str | Path | None = None,
    max_server_failure_rate: float | None = None,
    max_server_queue_fraction: float | None = None,
    request_path_metrics: RequestPathMetricNames | None = None,
    service_restart_metric: str | None = None,
    min_service_restarts: int | None = None,
    max_service_restarts: int | None = None,
) -> dict[str, object]:
    path = Path(telemetry_prometheus_path)
    baseline_text = None
    if telemetry_baseline_prometheus_path is not None:
        baseline_path = Path(telemetry_baseline_prometheus_path)
        baseline_text = baseline_path.read_text(encoding="utf-8")
    return attach_telemetry_snapshots(
        metrics,
        path.read_text(encoding="utf-8"),
        baseline_prometheus_text=baseline_text,
        max_server_failure_rate=max_server_failure_rate,
        max_server_queue_fraction=max_server_queue_fraction,
        request_path_metrics=request_path_metrics,
        service_restart_metric=service_restart_metric,
        min_service_restarts=min_service_restarts,
        max_service_restarts=max_service_restarts,
    )


def attach_telemetry_snapshots(
    metrics: dict[str, object],
    prometheus_text: str,
    baseline_prometheus_text: str | None = None,
    *,
    source: str = "prometheus_snapshot",
    alignment: str = "operator_supplied_unverified",
    counter_window_source: str = "paired_prometheus_counter_snapshots",
    max_server_failure_rate: float | None = None,
    max_server_queue_fraction: float | None = None,
    request_path_metrics: RequestPathMetricNames | None = None,
    service_restart_metric: str | None = None,
    min_service_restarts: int | None = None,
    max_service_restarts: int | None = None,
) -> dict[str, object]:
    if (
        min_service_restarts is not None or max_service_restarts is not None
    ) and service_restart_metric is None:
        raise ValueError("service restart bounds require a restart metric")
    enriched_metrics = dict(metrics)
    model_name = str(metrics.get("model_name", "unknown"))
    enriched_metrics["telemetry"] = build_telemetry_summary(
        prometheus_text,
        model_name=model_name,
        source=source,
    )
    if baseline_prometheus_text is not None:
        baseline = build_telemetry_summary(
            baseline_prometheus_text,
            model_name=model_name,
            source=source,
        )
        counter_window = build_telemetry_counter_window(
            baseline,
            enriched_metrics["telemetry"],
            source=counter_window_source,
            alignment=alignment,
        )
        enriched_metrics["telemetry_window"] = counter_window
        if (
            max_server_failure_rate is not None
            or max_server_queue_fraction is not None
        ):
            enriched_metrics["telemetry_gate"] = build_telemetry_gate(
                counter_window,
                max_server_failure_rate=max_server_failure_rate,
                max_server_queue_fraction=max_server_queue_fraction,
            )
        if request_path_metrics is not None:
            request_path = build_request_path_accounting(
                metrics,
                baseline_prometheus_text,
                prometheus_text,
                request_path_metrics,
                source=counter_window_source,
                alignment=alignment,
            )
            enriched_metrics["request_path"] = request_path
            enriched_metrics["request_path_gate"] = build_request_path_gate(request_path)
        if service_restart_metric is not None:
            service_lifecycle = build_service_lifecycle_accounting(
                baseline_prometheus_text,
                prometheus_text,
                service_restart_metric,
                source=counter_window_source,
                alignment=alignment,
            )
            enriched_metrics["service_lifecycle"] = service_lifecycle
            if min_service_restarts is not None or max_service_restarts is not None:
                enriched_metrics["service_lifecycle_gate"] = (
                    build_service_lifecycle_gate(
                        service_lifecycle,
                        min_service_restarts=min_service_restarts,
                        max_service_restarts=max_service_restarts,
                    )
                )
    else:
        if max_server_failure_rate is not None or max_server_queue_fraction is not None:
            raise ValueError(
                "telemetry thresholds require paired pre-run and post-run snapshots"
            )
        if request_path_metrics is not None:
            raise ValueError("request-path accounting requires paired counter snapshots")
        if service_restart_metric is not None:
            raise ValueError("service lifecycle accounting requires paired counter snapshots")
    return enriched_metrics
