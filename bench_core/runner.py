"""Request execution, retries, load schedules and the benchmark phases."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from bench_core.config import (
    BenchmarkConfig,
    CoordinatedClientConfig,
    GpuGaugeCapture,
    InferenceResult,
    OutputInferenceResult,
    OutputObservation,
    RequestPathMetricNames,
    StreamingInferenceObservation,
    percentile,
)
from bench_core.telemetry_clients import (
    TelemetrySnapshotClient,
)
from bench_core.clients import (
    InferenceClient,
    OutputInferenceClient,
)
from bench_core.summary import (
    _summarize_retry_attempts,
    summarize_results,
)
from bench_core.telemetry import (
    _build_gpu_gauge_capture,
    _build_gpu_gauge_window,
    attach_telemetry_snapshots,
)


def execute_with_retries(
    client: InferenceClient,
    retries: int,
    retry_backoff_seconds: float = 0.0,
) -> InferenceResult:
    if not math.isfinite(retry_backoff_seconds) or retry_backoff_seconds < 0:
        raise ValueError("retry backoff must be finite and non-negative")
    start = time.perf_counter()
    last_error: str | None = None

    for attempt_count in range(1, retries + 2):
        try:
            observation = client.infer()
            latency_ms = (time.perf_counter() - start) * 1000
            return InferenceResult(
                ok=True,
                latency_ms=latency_ms,
                streaming=(
                    observation
                    if isinstance(observation, StreamingInferenceObservation)
                    else None
                ),
                request_started_at=start,
                attempt_count=attempt_count,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark harness records client failures.
            last_error = str(exc)
            if attempt_count <= retries and retry_backoff_seconds:
                time.sleep(retry_backoff_seconds)

    latency_ms = (time.perf_counter() - start) * 1000
    return InferenceResult(
        ok=False,
        latency_ms=latency_ms,
        error=last_error,
        request_started_at=start,
        attempt_count=retries + 1,
    )


def execute_output_with_retries(
    client: OutputInferenceClient,
    sample_id: int,
    retries: int,
) -> OutputInferenceResult:
    start = time.perf_counter()
    last_error: str | None = None

    for _ in range(retries + 1):
        try:
            captured_output = client.infer_output(sample_id)
            observation = (
                captured_output
                if isinstance(captured_output, OutputObservation)
                else OutputObservation(fingerprint=captured_output)
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return OutputInferenceResult(
                sample_id=sample_id,
                ok=True,
                latency_ms=latency_ms,
                output_observation=observation,
            )
        except Exception as exc:  # noqa: BLE001 - probe records client failures.
            last_error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    return OutputInferenceResult(
        sample_id=sample_id,
        ok=False,
        latency_ms=latency_ms,
        error=last_error,
    )


def _compare_output_observations(
    baseline: OutputObservation,
    candidate: OutputObservation,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    if baseline.fingerprint == candidate.fingerprint:
        has_numeric_evidence = (
            baseline.numeric_outputs is not None
            and candidate.numeric_outputs is not None
        )
        return {
            "matched": True,
            "exact": True,
            "tolerance_match": False,
            "numeric_comparison": has_numeric_evidence,
            "incompatible": False,
            "reason": None,
            "max_absolute_error": 0.0 if has_numeric_evidence else None,
            "max_relative_error": 0.0 if has_numeric_evidence else None,
        }

    if absolute_tolerance == 0 and relative_tolerance == 0:
        return {
            "matched": False,
            "exact": False,
            "tolerance_match": False,
            "numeric_comparison": False,
            "incompatible": False,
            "reason": "exact_fingerprint_mismatch",
            "max_absolute_error": None,
            "max_relative_error": None,
        }

    baseline_outputs = baseline.numeric_outputs
    candidate_outputs = candidate.numeric_outputs
    if baseline_outputs is None or candidate_outputs is None:
        return {
            "matched": False,
            "exact": False,
            "tolerance_match": False,
            "numeric_comparison": False,
            "incompatible": True,
            "reason": "non_numeric_output",
            "max_absolute_error": None,
            "max_relative_error": None,
        }

    baseline_structure = [
        (output.name, output.dtype, output.shape, len(output.values))
        for output in baseline_outputs
    ]
    candidate_structure = [
        (output.name, output.dtype, output.shape, len(output.values))
        for output in candidate_outputs
    ]
    if baseline_structure != candidate_structure:
        return {
            "matched": False,
            "exact": False,
            "tolerance_match": False,
            "numeric_comparison": False,
            "incompatible": True,
            "reason": "structural_incompatibility",
            "max_absolute_error": None,
            "max_relative_error": None,
        }

    max_absolute_error = 0.0
    relative_errors: list[float] = []
    within_tolerance = True
    for baseline_output, candidate_output in zip(
        baseline_outputs,
        candidate_outputs,
    ):
        for baseline_value, candidate_value in zip(
            baseline_output.values,
            candidate_output.values,
        ):
            baseline_magnitude = float(abs(baseline_value))
            candidate_magnitude = float(abs(candidate_value))
            if not (
                math.isfinite(baseline_magnitude)
                and math.isfinite(candidate_magnitude)
            ):
                return {
                    "matched": False,
                    "exact": False,
                    "tolerance_match": False,
                    "numeric_comparison": False,
                    "incompatible": True,
                    "reason": "non_finite_values",
                    "max_absolute_error": None,
                    "max_relative_error": None,
                }
            absolute_error = float(abs(candidate_value - baseline_value))
            allowed_error = absolute_tolerance + relative_tolerance * baseline_magnitude
            if not (math.isfinite(absolute_error) and math.isfinite(allowed_error)):
                return {
                    "matched": False,
                    "exact": False,
                    "tolerance_match": False,
                    "numeric_comparison": False,
                    "incompatible": True,
                    "reason": "non_finite_values",
                    "max_absolute_error": None,
                    "max_relative_error": None,
                }
            max_absolute_error = max(max_absolute_error, absolute_error)
            if baseline_magnitude > 0:
                relative_errors.append(absolute_error / baseline_magnitude)
            if absolute_error > allowed_error:
                within_tolerance = False

    return {
        "matched": within_tolerance,
        "exact": False,
        "tolerance_match": within_tolerance,
        "numeric_comparison": True,
        "incompatible": False,
        "reason": None if within_tolerance else "outside_tolerance",
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max(relative_errors, default=0.0),
    }


def run_batch_invariance_probe(
    client: OutputInferenceClient,
    probe_count: int,
    concurrency: int,
    retries: int = 0,
    seed: int = 7,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> dict[str, object]:
    if probe_count <= 0:
        raise ValueError("probe_count must be greater than zero")
    if concurrency <= 1:
        raise ValueError("concurrency must be greater than one")
    if any(
        not math.isfinite(value) or value < 0
        for value in (absolute_tolerance, relative_tolerance)
    ):
        raise ValueError("batch output tolerances must be finite and non-negative")

    probe_ids = list(range(probe_count))
    baseline_results = {
        sample_id: execute_output_with_retries(client, sample_id, retries)
        for sample_id in probe_ids
    }

    noise_ids = [1_000_000 + index for index in range(probe_count)]
    candidate_work = [("probe", sample_id) for sample_id in probe_ids]
    candidate_work.extend(("noise", sample_id) for sample_id in noise_ids)
    random.Random(seed).shuffle(candidate_work)

    candidate_results: dict[int, OutputInferenceResult] = {}
    noise_failures = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(execute_output_with_retries, client, sample_id, retries): (
                workload_type,
                sample_id,
            )
            for workload_type, sample_id in candidate_work
        }
        for future in as_completed(futures):
            workload_type, sample_id = futures[future]
            result = future.result()
            if workload_type == "probe":
                candidate_results[sample_id] = result
            elif not result.ok:
                noise_failures += 1

    mismatched_sample_ids: list[int] = []
    matched_outputs = 0
    exact_matches = 0
    tolerance_matches = 0
    numeric_comparisons = 0
    incompatible_outputs = 0
    compared_outputs = 0
    errors: list[dict[str, object]] = []
    mismatch_reasons: dict[str, int] = {}
    observed_absolute_errors: list[float] = []
    observed_relative_errors: list[float] = []

    for sample_id in probe_ids:
        baseline = baseline_results[sample_id]
        candidate = candidate_results[sample_id]
        if not baseline.ok:
            errors.append(
                {
                    "phase": "isolated",
                    "sample_id": sample_id,
                    "error": baseline.error,
                }
            )
            continue
        if not candidate.ok:
            errors.append(
                {
                    "phase": "concurrent",
                    "sample_id": sample_id,
                    "error": candidate.error,
                }
            )
            continue

        compared_outputs += 1
        baseline_observation = baseline.output_observation
        candidate_observation = candidate.output_observation
        if baseline_observation is None or candidate_observation is None:
            comparison = {
                "matched": False,
                "exact": False,
                "tolerance_match": False,
                "numeric_comparison": False,
                "incompatible": True,
                "reason": "missing_output_evidence",
                "max_absolute_error": None,
                "max_relative_error": None,
            }
        else:
            comparison = _compare_output_observations(
                baseline_observation,
                candidate_observation,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )

        if comparison["numeric_comparison"]:
            numeric_comparisons += 1
        if comparison["incompatible"]:
            incompatible_outputs += 1
        absolute_error = comparison["max_absolute_error"]
        relative_error = comparison["max_relative_error"]
        if isinstance(absolute_error, (int, float)):
            observed_absolute_errors.append(float(absolute_error))
        if isinstance(relative_error, (int, float)):
            observed_relative_errors.append(float(relative_error))

        if comparison["matched"]:
            matched_outputs += 1
            if comparison["exact"]:
                exact_matches += 1
            elif comparison["tolerance_match"]:
                tolerance_matches += 1
        else:
            mismatched_sample_ids.append(sample_id)
            reason = str(comparison["reason"])
            mismatch_reasons[reason] = mismatch_reasons.get(reason, 0) + 1

    failed_probes = len(errors)
    exact_match = (
        failed_probes == 0
        and noise_failures == 0
        and compared_outputs == probe_count
        and exact_matches == probe_count
    )
    passed = (
        failed_probes == 0
        and noise_failures == 0
        and compared_outputs == probe_count
        and matched_outputs == probe_count
    )

    return {
        "probe_count": probe_count,
        "concurrency": concurrency,
        "noise_request_count": len(noise_ids),
        "compared_outputs": compared_outputs,
        "matched_outputs": matched_outputs,
        "exact_matches": exact_matches,
        "tolerance_matches": tolerance_matches,
        "numeric_comparisons": numeric_comparisons,
        "incompatible_outputs": incompatible_outputs,
        "mismatched_outputs": len(mismatched_sample_ids),
        "failed_probes": failed_probes,
        "noise_failures": noise_failures,
        "match_rate": round(matched_outputs / probe_count, 4),
        "exact_match": exact_match,
        "passed": passed,
        "comparison_policy": {
            "mode": (
                "numeric_tolerance"
                if absolute_tolerance > 0 or relative_tolerance > 0
                else "exact_fingerprint"
            ),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "criterion": (
                "absolute_error <= absolute_tolerance + "
                "relative_tolerance * abs(isolated_value)"
            ),
            "scope": "this benchmark run and model only",
            "output_values_persisted": False,
            "output_fingerprints_persisted": False,
        },
        "max_observed_absolute_error": max(observed_absolute_errors, default=None),
        "max_observed_relative_error": max(observed_relative_errors, default=None),
        "mismatched_sample_ids": mismatched_sample_ids,
        "mismatch_reasons": mismatch_reasons,
        "errors": errors,
    }


def build_constant_rate_submission_offsets(
    request_count: int,
    request_rate_rps: float,
) -> list[float]:
    """Return monotonic deadline offsets for an open-loop constant-rate schedule."""
    if request_rate_rps <= 0:
        raise ValueError("request rate must be greater than zero")
    return [index / request_rate_rps for index in range(request_count)]


def _build_load_schedule_summary(
    scheduled_offsets: list[float],
    submission_offsets: list[float],
    request_start_offsets: list[float],
    request_rate_rps: float,
) -> dict[str, object]:
    if not (
        len(scheduled_offsets)
        == len(submission_offsets)
        == len(request_start_offsets)
    ):
        raise ValueError("scheduled, submitted, and started requests must have equal counts")
    submission_lags_ms = [
        max(0.0, submitted - scheduled) * 1000
        for scheduled, submitted in zip(scheduled_offsets, submission_offsets)
    ]
    executor_queue_ms = [
        max(0.0, started - submitted) * 1000
        for submitted, started in zip(submission_offsets, request_start_offsets)
    ]
    dispatch_lags_ms = [
        max(0.0, started - scheduled) * 1000
        for scheduled, started in zip(scheduled_offsets, request_start_offsets)
    ]
    scheduled_span = scheduled_offsets[-1] if scheduled_offsets else 0.0
    submission_span = (
        submission_offsets[-1] - submission_offsets[0]
        if len(submission_offsets) > 1
        else 0.0
    )
    request_start_span = (
        max(request_start_offsets) - min(request_start_offsets)
        if len(request_start_offsets) > 1
        else 0.0
    )
    achieved_submission_rate = (
        (len(submission_offsets) - 1) / submission_span
        if len(submission_offsets) > 1 and submission_span > 0
        else None
    )
    achieved_request_start_rate = (
        (len(request_start_offsets) - 1) / request_start_span
        if len(request_start_offsets) > 1 and request_start_span > 0
        else None
    )

    def distribution(values: list[float]) -> dict[str, float]:
        return {
            "avg": round(statistics.fmean(values), 4) if values else 0.0,
            "p50": round(percentile(values, 50), 4),
            "p95": round(percentile(values, 95), 4),
            "p99": round(percentile(values, 99), 4),
            "max": round(max(values), 4) if values else 0.0,
        }

    return {
        "mode": "open_loop_constant_rate",
        "configured_request_rate_rps": request_rate_rps,
        "request_count": len(submission_offsets),
        "scheduled_submission_span_seconds": round(scheduled_span, 6),
        "observed_submission_span_seconds": round(submission_span, 6),
        "observed_request_start_span_seconds": round(request_start_span, 6),
        "achieved_submission_rate_rps": round(achieved_submission_rate, 4)
        if achieved_submission_rate is not None
        else None,
        "achieved_request_start_rate_rps": round(achieved_request_start_rate, 4)
        if achieved_request_start_rate is not None
        else None,
        "submission_lag_ms": distribution(submission_lags_ms),
        "executor_queue_ms": distribution(executor_queue_ms),
        "request_start_lag_ms": distribution(dispatch_lags_ms),
        "clock": "client_monotonic",
        "scope": (
            "client executor submission and worker-start timing; successful completion "
            "throughput is reported separately"
        ),
        "note": (
            "This schedule does not prove server arrival timing, queue isolation, or "
            "distributed clock alignment."
        ),
    }


def _run_request_phase(
    client: InferenceClient,
    request_count: int,
    concurrency: int,
    retries: int,
    retry_backoff_seconds: float = 0.0,
    phase_started: threading.Event | None = None,
    request_rate_rps: float = 0.0,
) -> tuple[list[InferenceResult], float, dict[str, object] | None]:
    start = time.perf_counter()
    results: list[InferenceResult] = []
    scheduled_offsets = (
        build_constant_rate_submission_offsets(request_count, request_rate_rps)
        if request_rate_rps
        else []
    )
    submission_offsets: list[float] = []
    request_start_offsets = [0.0] * request_count

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for request_index in range(request_count):
            if scheduled_offsets:
                deadline = start + scheduled_offsets[request_index]
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                submission_offsets.append(time.perf_counter() - start)
            future = executor.submit(
                execute_with_retries,
                client,
                retries,
                retry_backoff_seconds,
            )
            futures[future] = request_index
            if request_index == 0 and phase_started is not None:
                phase_started.set()
        if not futures and phase_started is not None:
            phase_started.set()
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            request_index = futures[future]
            if result.request_started_at is None:
                raise RuntimeError("paced request did not record a worker start time")
            request_start_offsets[request_index] = result.request_started_at - start

    duration_seconds = time.perf_counter() - start
    load_schedule = (
        _build_load_schedule_summary(
            scheduled_offsets,
            submission_offsets,
            request_start_offsets,
            request_rate_rps,
        )
        if scheduled_offsets
        else None
    )
    return results, duration_seconds, load_schedule


def _sample_telemetry_during_phase(
    telemetry_client: TelemetrySnapshotClient,
    interval_seconds: float,
    phase_started: threading.Event,
    stop_requested: threading.Event,
    snapshots: list[GpuGaugeCapture],
    errors: list[Exception],
) -> None:
    phase_started.wait()
    while not stop_requested.is_set():
        try:
            snapshots.append(_build_gpu_gauge_capture(telemetry_client.scrape()))
        except Exception as exc:  # noqa: BLE001 - propagated on the benchmark thread.
            errors.append(exc)
            stop_requested.set()
            return
        if stop_requested.wait(interval_seconds):
            return


def build_retry_gate(
    retry_summary: dict[str, object],
    max_client_attempt_amplification: float,
) -> dict[str, object]:
    """Gate measured client-attempt amplification against an explicit budget."""
    if (
        not math.isfinite(max_client_attempt_amplification)
        or max_client_attempt_amplification < 1
    ):
        raise ValueError("maximum client-attempt amplification must be finite and at least 1")
    client_attempts = retry_summary.get("client_attempts")
    logical_requests = retry_summary.get("logical_requests")
    if isinstance(client_attempts, int) and isinstance(logical_requests, int):
        evaluable = logical_requests > 0 and client_attempts >= logical_requests
        observed = client_attempts / logical_requests if evaluable else None
    else:
        summarized = retry_summary.get("client_attempt_amplification")
        evaluable = isinstance(summarized, (int, float)) and math.isfinite(
            float(summarized)
        )
        observed = float(summarized) if evaluable else None
    passed = (
        evaluable
        and observed is not None
        and observed <= max_client_attempt_amplification
    )
    failure_reasons: list[str] = []
    if not evaluable or observed is None:
        failure_reasons.append("client-attempt amplification is unavailable")
    elif not passed:
        failure_reasons.append(
            "client-attempt amplification "
            f"{observed:g} exceeded {max_client_attempt_amplification:g} threshold"
        )
    return {
        "passed": passed,
        "checks": {
            "client_attempt_amplification": {
                "observed": round(observed, 6) if observed is not None else None,
                "maximum": max_client_attempt_amplification,
                "evaluable": evaluable,
                "passed": passed,
            }
        },
        "failure_reasons": failure_reasons,
        "scope": "measured logical requests only; warmup is excluded",
    }


def _summarize_request_phase(
    results: list[InferenceResult],
    duration_seconds: float,
    configured_retries: int,
) -> dict[str, object]:
    latencies = [result.latency_ms for result in results if result.ok]
    successes = len(latencies)
    total = len(results)
    return {
        "request_count": total,
        "duration_seconds": round(duration_seconds, 4),
        "successful_requests": successes,
        "failed_requests": total - successes,
        "success_rate": round(successes / total, 4) if total else 0,
        "throughput_rps": round(successes / duration_seconds, 4)
        if duration_seconds > 0
        else 0,
        "latency_ms": {
            "avg": round(statistics.fmean(latencies), 4) if latencies else 0,
            "p50": round(percentile(latencies, 50), 4),
            "p95": round(percentile(latencies, 95), 4),
            "p99": round(percentile(latencies, 99), 4),
            "min": round(min(latencies), 4) if latencies else 0,
            "max": round(max(latencies), 4) if latencies else 0,
        },
        "retry": _summarize_retry_attempts(results, configured_retries),
    }


def run_benchmark(
    client: InferenceClient,
    config: BenchmarkConfig,
    telemetry_client: TelemetrySnapshotClient | None = None,
    max_server_failure_rate: float | None = None,
    max_server_queue_fraction: float | None = None,
    telemetry_sample_interval_seconds: float = 0.0,
    max_client_attempt_amplification: float | None = None,
    request_path_metrics: RequestPathMetricNames | None = None,
    service_restart_metric: str | None = None,
    min_service_restarts: int | None = None,
    max_service_restarts: int | None = None,
    coordination: CoordinatedClientConfig | None = None,
) -> dict[str, object]:
    if telemetry_sample_interval_seconds < 0:
        raise ValueError("telemetry sample interval must be zero or greater")
    if telemetry_sample_interval_seconds and telemetry_client is None:
        raise ValueError("telemetry sampling requires a telemetry client")
    if request_path_metrics is not None and telemetry_client is None:
        raise ValueError("request-path accounting requires a telemetry client")
    if service_restart_metric is not None and telemetry_client is None:
        raise ValueError("service lifecycle accounting requires a telemetry client")
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
    if (
        min_service_restarts is not None or max_service_restarts is not None
    ) and service_restart_metric is None:
        raise ValueError("service restart bounds require a restart metric")
    if coordination is not None:
        if not coordination.run_id or len(coordination.run_id) > 256:
            raise ValueError("coordinated run ID must contain 1 to 256 characters")
        if (
            not isinstance(coordination.client_count, int)
            or isinstance(coordination.client_count, bool)
            or not 2 <= coordination.client_count <= 64
        ):
            raise ValueError("coordinated client count must be between 2 and 64")
        if (
            not isinstance(coordination.client_index, int)
            or isinstance(coordination.client_index, bool)
            or not 0 <= coordination.client_index < coordination.client_count
        ):
            raise ValueError("coordinated client index is outside the client set")
        if (
            not isinstance(coordination.planned_start_unix_ns, int)
            or isinstance(coordination.planned_start_unix_ns, bool)
            or coordination.planned_start_unix_ns <= 0
        ):
            raise ValueError("coordinated start must be a positive Unix nanosecond value")

    warmup_results: list[InferenceResult] = []
    warmup_duration_seconds = 0.0
    if config.warmup_requests:
        warmup_results, warmup_duration_seconds, _ = _run_request_phase(
            client,
            request_count=config.warmup_requests,
            concurrency=config.concurrency,
            retries=config.retries,
            retry_backoff_seconds=config.retry_backoff_seconds,
        )

    telemetry_before = telemetry_client.scrape() if telemetry_client else None
    in_window_telemetry: list[GpuGaugeCapture] = []
    telemetry_errors: list[Exception] = []
    phase_started = threading.Event()
    stop_sampling = threading.Event()
    sampler: threading.Thread | None = None
    if telemetry_client is not None and telemetry_sample_interval_seconds:
        sampler = threading.Thread(
            target=_sample_telemetry_during_phase,
            args=(
                telemetry_client,
                telemetry_sample_interval_seconds,
                phase_started,
                stop_sampling,
                in_window_telemetry,
                telemetry_errors,
            ),
            name="benchmark-telemetry-sampler",
            daemon=True,
        )
        sampler.start()

    if coordination is not None:
        remaining_ns = coordination.planned_start_unix_ns - time.time_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1_000_000_000)
        measured_start_unix_ns = time.time_ns()
    else:
        measured_start_unix_ns = None

    try:
        results, duration_seconds, load_schedule = _run_request_phase(
            client,
            request_count=config.num_requests,
            concurrency=config.concurrency,
            retries=config.retries,
            retry_backoff_seconds=config.retry_backoff_seconds,
            phase_started=phase_started if sampler is not None else None,
            request_rate_rps=config.request_rate_rps,
        )
        measured_end_unix_ns = (
            time.time_ns() if coordination is not None else None
        )
    finally:
        stop_sampling.set()
        if sampler is not None:
            sampler.join()

    if telemetry_errors:
        raise RuntimeError("in-window telemetry scrape failed") from telemetry_errors[0]
    if sampler is not None and not in_window_telemetry:
        raise RuntimeError("telemetry sampling produced no in-window scrapes")
    telemetry_after = telemetry_client.scrape() if telemetry_client else None
    summary = summarize_results(results, duration_seconds, config)
    if max_client_attempt_amplification is not None:
        retry_summary = summary.get("retry")
        assert isinstance(retry_summary, dict)
        summary["retry_gate"] = build_retry_gate(
            retry_summary,
            max_client_attempt_amplification,
        )
    summary["measurement_scope"] = {
        "headline_phase": "measured requests",
        "warmup_excluded": bool(config.warmup_requests),
        "note": (
            "Warmup requests precondition the client and server path; they do not "
            "establish a process, model, or accelerator cold-start measurement."
        ),
    }
    if load_schedule is not None:
        summary["load_schedule"] = load_schedule
    if coordination is not None:
        assert measured_start_unix_ns is not None
        assert measured_end_unix_ns is not None
        persisted_config = summary.get("config")
        assert isinstance(persisted_config, dict)
        fingerprint_config = dict(persisted_config)
        fingerprint_config.pop("output_dir", None)
        config_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        summary["coordination"] = {
            "run_id_sha256": hashlib.sha256(
                coordination.run_id.encode("utf-8")
            ).hexdigest(),
            "run_id_persisted": False,
            "config_fingerprint_sha256": config_fingerprint,
            "client_index": coordination.client_index,
            "client_count": coordination.client_count,
            "planned_start_unix_ns": coordination.planned_start_unix_ns,
            "measured_start_unix_ns": measured_start_unix_ns,
            "measured_end_unix_ns": measured_end_unix_ns,
            "clock": "host_wall_clock",
            "scope": "single-host child process measured-request window",
            "note": (
                "Wall-clock timestamps support same-host coordination only; they do not "
                "establish synchronized clocks across hosts."
            ),
        }
    if warmup_results:
        summary["warmup"] = _summarize_request_phase(
            warmup_results,
            warmup_duration_seconds,
            config.retries,
        )
    if telemetry_before is not None and telemetry_after is not None:
        endpoint_count = int(getattr(telemetry_client, "source_count", 1))
        multiple_endpoints = endpoint_count > 1
        summary = attach_telemetry_snapshots(
            summary,
            telemetry_after,
            baseline_prometheus_text=telemetry_before,
            source=(
                "combined_http_prometheus_snapshot"
                if multiple_endpoints
                else "http_prometheus_snapshot"
            ),
            alignment="harness_bracketed_measured_phase",
            counter_window_source=(
                "paired_multi_endpoint_http_prometheus_scrapes"
                if multiple_endpoints
                else "paired_http_prometheus_scrapes"
            ),
            max_server_failure_rate=max_server_failure_rate,
            max_server_queue_fraction=max_server_queue_fraction,
            request_path_metrics=request_path_metrics,
            service_restart_metric=service_restart_metric,
            min_service_restarts=min_service_restarts,
            max_service_restarts=max_service_restarts,
        )
        for artifact_key in (
            "telemetry",
            "telemetry_window",
            "request_path",
            "service_lifecycle",
        ):
            artifact = summary.get(artifact_key)
            if isinstance(artifact, dict):
                artifact["endpoint_count"] = endpoint_count
                artifact["endpoint_urls_persisted"] = False
        if telemetry_sample_interval_seconds:
            gauge_captures = [
                _build_gpu_gauge_capture(telemetry_before),
                *in_window_telemetry,
                _build_gpu_gauge_capture(telemetry_after),
            ]
            summary["telemetry_gauge_window"] = _build_gpu_gauge_window(
                [capture.values for capture in gauge_captures],
                [capture.series_membership for capture in gauge_captures],
                in_window_scrape_count=len(in_window_telemetry),
                configured_interval_seconds=telemetry_sample_interval_seconds,
            )
    return summary
