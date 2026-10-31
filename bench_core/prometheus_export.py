"""Prometheus text export of a benchmark result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench_core.config import (
    TELEMETRY_COUNTER_KEYS,
    _config_number,
    _latency,
    _nested_number,
    _number,
)


def _escape_label(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _base_labels(metrics: dict[str, Any]) -> str:
    mode = _escape_label(metrics.get("mode", "unknown"))
    model_name = _escape_label(metrics.get("model_name", "unknown"))
    return f'mode="{mode}",model="{model_name}"'


def format_prometheus_metrics(metrics: dict[str, object]) -> str:
    typed_metrics: dict[str, Any] = dict(metrics)
    labels = _base_labels(typed_metrics)
    lines = [
        "# HELP triton_benchmark_requests_total Total benchmark requests by outcome.",
        "# TYPE triton_benchmark_requests_total counter",
        (
            f'triton_benchmark_requests_total{{{labels},outcome="success"}} '
            f'{_number(typed_metrics, "successful_requests"):g}'
        ),
        (
            f'triton_benchmark_requests_total{{{labels},outcome="failure"}} '
            f'{_number(typed_metrics, "failed_requests"):g}'
        ),
        "# HELP triton_benchmark_success_rate Successful request ratio for the benchmark run.",
        "# TYPE triton_benchmark_success_rate gauge",
        f"triton_benchmark_success_rate{{{labels}}} {_number(typed_metrics, 'success_rate'):g}",
        "# HELP triton_benchmark_duration_seconds Wall-clock benchmark duration.",
        "# TYPE triton_benchmark_duration_seconds gauge",
        f"triton_benchmark_duration_seconds{{{labels}}} {_number(typed_metrics, 'duration_seconds'):g}",
        "# HELP triton_benchmark_throughput_rps Successful requests per second.",
        "# TYPE triton_benchmark_throughput_rps gauge",
        f"triton_benchmark_throughput_rps{{{labels}}} {_number(typed_metrics, 'throughput_rps'):g}",
        "# HELP triton_benchmark_latency_ms End-to-end successful request latency.",
        "# TYPE triton_benchmark_latency_ms gauge",
        f'triton_benchmark_latency_ms{{{labels},stat="avg"}} {_latency(typed_metrics, "avg"):g}',
        f'triton_benchmark_latency_ms{{{labels},stat="min"}} {_latency(typed_metrics, "min"):g}',
        f'triton_benchmark_latency_ms{{{labels},stat="max"}} {_latency(typed_metrics, "max"):g}',
        f'triton_benchmark_latency_ms{{{labels},quantile="0.50"}} {_latency(typed_metrics, "p50"):g}',
        f'triton_benchmark_latency_ms{{{labels},quantile="0.95"}} {_latency(typed_metrics, "p95"):g}',
        f'triton_benchmark_latency_ms{{{labels},quantile="0.99"}} {_latency(typed_metrics, "p99"):g}',
        "# HELP triton_benchmark_concurrency Configured concurrent workers.",
        "# TYPE triton_benchmark_concurrency gauge",
        f"triton_benchmark_concurrency{{{labels}}} {_config_number(typed_metrics, 'concurrency'):g}",
        "# HELP triton_benchmark_retries Configured retry attempts per request.",
        "# TYPE triton_benchmark_retries gauge",
        f"triton_benchmark_retries{{{labels}}} {_config_number(typed_metrics, 'retries'):g}",
    ]

    retry = typed_metrics.get("retry")
    if isinstance(retry, dict):
        retry_specs = (
            (
                "client_attempts",
                "triton_benchmark_client_attempts_total",
                "Client calls made for measured logical requests.",
            ),
            (
                "retry_attempts",
                "triton_benchmark_retry_attempts_total",
                "Client calls after the initial measured-request attempt.",
            ),
            (
                "retried_requests",
                "triton_benchmark_retried_requests_total",
                "Measured logical requests that needed more than one client attempt.",
            ),
            (
                "recovered_requests",
                "triton_benchmark_recovered_requests_total",
                "Measured logical requests that succeeded after a failed client attempt.",
            ),
            (
                "exhausted_requests",
                "triton_benchmark_exhausted_requests_total",
                "Measured logical requests that remained failed after all configured attempts.",
            ),
        )
        for source_key, metric_name, help_text in retry_specs:
            value = retry.get(source_key)
            if isinstance(value, (int, float)):
                lines.extend(
                    [
                        f"# HELP {metric_name} {help_text}",
                        f"# TYPE {metric_name} counter",
                        f"{metric_name}{{{labels}}} {value:g}",
                    ]
                )
        amplification = retry.get("client_attempt_amplification")
        if isinstance(amplification, (int, float)):
            lines.extend(
                [
                    "# HELP triton_benchmark_client_attempt_amplification Measured client attempts divided by measured logical requests.",
                    "# TYPE triton_benchmark_client_attempt_amplification gauge",
                    (
                        "triton_benchmark_client_attempt_amplification"
                        f"{{{labels}}} {amplification:g}"
                    ),
                ]
            )
        recovered_latency = retry.get("recovered_request_latency_ms")
        if isinstance(recovered_latency, dict):
            latency_samples: list[str] = []
            for statistic in ("avg", "p50", "p95", "p99", "min", "max"):
                value = recovered_latency.get(statistic)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    latency_samples.append(
                        "triton_benchmark_recovered_request_latency_ms"
                        f'{{{labels},statistic="{statistic}"}} {value:g}'
                    )
            if latency_samples:
                lines.extend(
                    [
                        "# HELP triton_benchmark_recovered_request_latency_ms Client-observed end-to-end latency for measured logical requests that succeeded after retry.",
                        "# TYPE triton_benchmark_recovered_request_latency_ms gauge",
                        *latency_samples,
                    ]
                )

    retry_gate = typed_metrics.get("retry_gate")
    if isinstance(retry_gate, dict):
        checks = retry_gate.get("checks")
        check = (
            checks.get("client_attempt_amplification")
            if isinstance(checks, dict)
            else None
        )
        if isinstance(check, dict):
            maximum = check.get("maximum")
            if isinstance(maximum, (int, float)):
                lines.extend(
                    [
                        "# HELP triton_benchmark_retry_gate_max_amplification Configured maximum measured client-attempt amplification.",
                        "# TYPE triton_benchmark_retry_gate_max_amplification gauge",
                        (
                            "triton_benchmark_retry_gate_max_amplification"
                            f"{{{labels}}} {maximum:g}"
                        ),
                        "# HELP triton_benchmark_retry_gate_passed Whether measured client-attempt amplification stayed within budget.",
                        "# TYPE triton_benchmark_retry_gate_passed gauge",
                        (
                            "triton_benchmark_retry_gate_passed"
                            f"{{{labels}}} {1 if retry_gate.get('passed') else 0}"
                        ),
                    ]
                )

    request_path = typed_metrics.get("request_path")
    if isinstance(request_path, dict):
        valid = request_path.get("valid")
        if isinstance(valid, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_request_path_valid Whether every selected request-path counter window was valid.",
                    "# TYPE triton_benchmark_request_path_valid gauge",
                    f"triton_benchmark_request_path_valid{{{labels}}} {int(valid)}",
                ]
            )
        stages = request_path.get("stages")
        if isinstance(stages, dict):
            lines.extend(
                [
                    "# HELP triton_benchmark_request_path_events_total Aggregate counter delta by fixed serving-path stage.",
                    "# TYPE triton_benchmark_request_path_events_total counter",
                ]
            )
            for stage in ("ingress", "backend", "success"):
                stage_summary = stages.get(stage)
                delta = (
                    stage_summary.get("delta")
                    if isinstance(stage_summary, dict)
                    else None
                )
                if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                    lines.append(
                        "triton_benchmark_request_path_events_total"
                        f'{{{labels},stage="{stage}"}} {delta:g}'
                    )
        ratios = request_path.get("ratios")
        if isinstance(ratios, dict):
            ratio_specs = (
                ("ingress_per_client_attempt", "ingress_per_client_attempt"),
                ("backend_per_ingress", "backend_per_ingress"),
                ("success_per_backend", "success_per_backend"),
            )
            ratio_samples: list[str] = []
            for source_key, relation in ratio_specs:
                value = ratios.get(source_key)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                ratio_samples.append(
                    "triton_benchmark_request_path_ratio"
                    f'{{{labels},relation="{relation}"}} {value:g}'
                )
            if ratio_samples:
                lines.extend(
                    [
                        "# HELP triton_benchmark_request_path_ratio Aggregate ratio between adjacent request-path observations.",
                        "# TYPE triton_benchmark_request_path_ratio gauge",
                        *ratio_samples,
                    ]
                )

    request_path_gate = typed_metrics.get("request_path_gate")
    if isinstance(request_path_gate, dict):
        passed = request_path_gate.get("passed")
        if isinstance(passed, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_request_path_gate_passed Whether isolated request-path counters exactly reconciled with harness outcomes.",
                    "# TYPE triton_benchmark_request_path_gate_passed gauge",
                    f"triton_benchmark_request_path_gate_passed{{{labels}}} {int(passed)}",
                ]
            )

    service_lifecycle = typed_metrics.get("service_lifecycle")
    if isinstance(service_lifecycle, dict):
        restart_count = service_lifecycle.get("restart_count")
        if isinstance(restart_count, int) and not isinstance(restart_count, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_service_restart_delta Aggregate completed service restarts in the paired telemetry window.",
                    "# TYPE triton_benchmark_service_restart_delta gauge",
                    f"triton_benchmark_service_restart_delta{{{labels}}} {restart_count}",
                ]
            )
        valid = service_lifecycle.get("valid")
        if isinstance(valid, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_service_lifecycle_valid Whether the selected restart counter window was valid.",
                    "# TYPE triton_benchmark_service_lifecycle_valid gauge",
                    f"triton_benchmark_service_lifecycle_valid{{{labels}}} {int(valid)}",
                ]
            )

    service_lifecycle_gate = typed_metrics.get("service_lifecycle_gate")
    if isinstance(service_lifecycle_gate, dict):
        checks = service_lifecycle_gate.get("checks")
        checks = checks if isinstance(checks, dict) else {}
        for check_name, bound_key, metric_name, help_text in (
            (
                "minimum_restarts",
                "minimum",
                "triton_benchmark_service_lifecycle_min_restarts",
                "Configured minimum completed service restarts in the telemetry window.",
            ),
            (
                "maximum_restarts",
                "maximum",
                "triton_benchmark_service_lifecycle_max_restarts",
                "Configured maximum completed service restarts in the telemetry window.",
            ),
        ):
            check = checks.get(check_name)
            bound = check.get(bound_key) if isinstance(check, dict) else None
            if isinstance(bound, int) and not isinstance(bound, bool):
                lines.extend(
                    [
                        f"# HELP {metric_name} {help_text}",
                        f"# TYPE {metric_name} gauge",
                        f"{metric_name}{{{labels}}} {bound}",
                    ]
                )
        passed = service_lifecycle_gate.get("passed")
        if isinstance(passed, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_service_lifecycle_gate_passed Whether the aggregate restart count stayed within configured bounds.",
                    "# TYPE triton_benchmark_service_lifecycle_gate_passed gauge",
                    (
                        "triton_benchmark_service_lifecycle_gate_passed"
                        f"{{{labels}}} {int(passed)}"
                    ),
                ]
            )

    load_schedule = typed_metrics.get("load_schedule")
    if isinstance(load_schedule, dict):
        configured_rate = load_schedule.get("configured_request_rate_rps")
        achieved_rate = load_schedule.get("achieved_submission_rate_rps")
        lines.extend(
            [
                "# HELP triton_benchmark_configured_request_rate_rps Configured open-loop measured-request submission rate.",
                "# TYPE triton_benchmark_configured_request_rate_rps gauge",
                (
                    "triton_benchmark_configured_request_rate_rps"
                    f"{{{labels}}} {float(configured_rate):g}"
                ),
            ]
        )
        if isinstance(achieved_rate, (int, float)):
            lines.extend(
                [
                    "# HELP triton_benchmark_achieved_submission_rate_rps Observed client submission rate over the measured submission span.",
                    "# TYPE triton_benchmark_achieved_submission_rate_rps gauge",
                    (
                        "triton_benchmark_achieved_submission_rate_rps"
                        f"{{{labels}}} {float(achieved_rate):g}"
                    ),
                ]
            )
        achieved_start_rate = load_schedule.get("achieved_request_start_rate_rps")
        if isinstance(achieved_start_rate, (int, float)):
            lines.extend(
                [
                    "# HELP triton_benchmark_achieved_request_start_rate_rps Observed client worker-start rate over the measured request-start span.",
                    "# TYPE triton_benchmark_achieved_request_start_rate_rps gauge",
                    (
                        "triton_benchmark_achieved_request_start_rate_rps"
                        f"{{{labels}}} {float(achieved_start_rate):g}"
                    ),
                ]
            )
        lag = load_schedule.get("submission_lag_ms")
        if isinstance(lag, dict):
            lines.extend(
                [
                    "# HELP triton_benchmark_submission_lag_ms Client submission delay after the configured monotonic deadline.",
                    "# TYPE triton_benchmark_submission_lag_ms gauge",
                ]
            )
            for stat in ("avg", "max"):
                value = lag.get(stat)
                if isinstance(value, (int, float)):
                    lines.append(
                        f'triton_benchmark_submission_lag_ms{{{labels},stat="{stat}"}} {value:g}'
                    )
            for stat, quantile in (
                ("p50", "0.50"),
                ("p95", "0.95"),
                ("p99", "0.99"),
            ):
                value = lag.get(stat)
                if isinstance(value, (int, float)):
                    lines.append(
                        "triton_benchmark_submission_lag_ms"
                        f'{{{labels},quantile="{quantile}"}} {value:g}'
                    )
        for source_key, metric_name, help_text in (
            (
                "executor_queue_ms",
                "triton_benchmark_executor_queue_ms",
                "Client delay between executor submission and worker start.",
            ),
            (
                "request_start_lag_ms",
                "triton_benchmark_request_start_lag_ms",
                "Client worker-start delay after the configured monotonic deadline.",
            ),
        ):
            distribution = load_schedule.get(source_key)
            if not isinstance(distribution, dict):
                continue
            lines.extend(
                [
                    f"# HELP {metric_name} {help_text}",
                    f"# TYPE {metric_name} gauge",
                ]
            )
            for stat in ("avg", "max"):
                value = distribution.get(stat)
                if isinstance(value, (int, float)):
                    lines.append(f'{metric_name}{{{labels},stat="{stat}"}} {value:g}')
            for stat, quantile in (
                ("p50", "0.50"),
                ("p95", "0.95"),
                ("p99", "0.99"),
            ):
                value = distribution.get(stat)
                if isinstance(value, (int, float)):
                    lines.append(
                        f'{metric_name}{{{labels},quantile="{quantile}"}} {value:g}'
                    )

    trace_context = typed_metrics.get("trace_context")
    if isinstance(trace_context, dict):
        lines.extend(
            [
                "# HELP triton_benchmark_trace_context_enabled Whether outbound live requests were configured to carry fresh W3C traceparent headers.",
                "# TYPE triton_benchmark_trace_context_enabled gauge",
                f"triton_benchmark_trace_context_enabled{{{labels}}} 1",
            ]
        )
        continuation = trace_context.get("response_continuation")
        if isinstance(continuation, dict):
            lines.extend(
                [
                    "# HELP triton_benchmark_trace_response_total Successful measured HTTP responses by privacy-safe trace-continuation classification.",
                    "# TYPE triton_benchmark_trace_response_total gauge",
                ]
            )
            for source_key, status in (
                ("matched_responses", "matched"),
                ("missing_responses", "missing"),
                ("invalid_responses", "invalid"),
                ("mismatched_responses", "mismatched"),
            ):
                value = continuation.get(source_key)
                if isinstance(value, int) and not isinstance(value, bool):
                    lines.append(
                        "triton_benchmark_trace_response_total"
                        f'{{{labels},status="{status}"}} {value}'
                    )
            match_coverage = continuation.get("match_coverage")
            if isinstance(match_coverage, (int, float)):
                lines.extend(
                    [
                        "# HELP triton_benchmark_trace_response_match_coverage Ratio of successful measured responses that continued the outbound trace ID.",
                        "# TYPE triton_benchmark_trace_response_match_coverage gauge",
                        (
                            "triton_benchmark_trace_response_match_coverage"
                            f"{{{labels}}} {match_coverage:g}"
                        ),
                    ]
                )

    trace_context_gate = typed_metrics.get("trace_context_gate")
    if isinstance(trace_context_gate, dict):
        passed = trace_context_gate.get("passed")
        if isinstance(passed, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_trace_context_gate_passed Whether every measured OpenAI response continued its outbound trace.",
                    "# TYPE triton_benchmark_trace_context_gate_passed gauge",
                    f"triton_benchmark_trace_context_gate_passed{{{labels}}} {int(passed)}",
                ]
            )

    warmup = typed_metrics.get("warmup")
    if isinstance(warmup, dict):
        warmup_labels = f'{labels},phase="warmup"'
        lines.extend(
            [
                "# HELP triton_benchmark_warmup_requests_total Warmup requests by outcome; excluded from headline benchmark results.",
                "# TYPE triton_benchmark_warmup_requests_total counter",
                (
                    f'triton_benchmark_warmup_requests_total{{{warmup_labels},outcome="success"}} '
                    f'{_number(warmup, "successful_requests"):g}'
                ),
                (
                    f'triton_benchmark_warmup_requests_total{{{warmup_labels},outcome="failure"}} '
                    f'{_number(warmup, "failed_requests"):g}'
                ),
                "# HELP triton_benchmark_warmup_duration_seconds Warmup phase wall-clock duration.",
                "# TYPE triton_benchmark_warmup_duration_seconds gauge",
                (
                    f"triton_benchmark_warmup_duration_seconds{{{warmup_labels}}} "
                    f'{_number(warmup, "duration_seconds"):g}'
                ),
                "# HELP triton_benchmark_warmup_throughput_rps Successful warmup requests per second.",
                "# TYPE triton_benchmark_warmup_throughput_rps gauge",
                (
                    f"triton_benchmark_warmup_throughput_rps{{{warmup_labels}}} "
                    f'{_number(warmup, "throughput_rps"):g}'
                ),
                "# HELP triton_benchmark_warmup_latency_ms End-to-end successful warmup request latency.",
                "# TYPE triton_benchmark_warmup_latency_ms gauge",
            ]
        )
        warmup_latency = warmup.get("latency_ms")
        if isinstance(warmup_latency, dict):
            for stat in ("avg", "min", "max"):
                value = warmup_latency.get(stat)
                if isinstance(value, (int, float)):
                    lines.append(
                        f'triton_benchmark_warmup_latency_ms{{{warmup_labels},stat="{stat}"}} {value:g}'
                    )
            for stat, quantile in (
                ("p50", "0.50"),
                ("p95", "0.95"),
                ("p99", "0.99"),
            ):
                value = warmup_latency.get(stat)
                if isinstance(value, (int, float)):
                    lines.append(
                        f'triton_benchmark_warmup_latency_ms{{{warmup_labels},quantile="{quantile}"}} {value:g}'
                    )

    telemetry = typed_metrics.get("telemetry")
    if isinstance(telemetry, dict):
        telemetry_specs = [
            (
                "# HELP triton_benchmark_gpu_utilization_percent Correlated GPU utilization from telemetry snapshot.",
                "# TYPE triton_benchmark_gpu_utilization_percent gauge",
                "triton_benchmark_gpu_utilization_percent",
                ("gpu", "utilization_pct"),
            ),
            (
                "# HELP triton_benchmark_gpu_memory_copy_utilization_percent Correlated GPU memory-copy utilization from telemetry snapshot.",
                "# TYPE triton_benchmark_gpu_memory_copy_utilization_percent gauge",
                "triton_benchmark_gpu_memory_copy_utilization_percent",
                ("gpu", "memory_copy_utilization_pct"),
            ),
            (
                "# HELP triton_benchmark_gpu_memory_used_mib Correlated GPU memory usage from telemetry snapshot.",
                "# TYPE triton_benchmark_gpu_memory_used_mib gauge",
                "triton_benchmark_gpu_memory_used_mib",
                ("gpu", "memory_used_mib"),
            ),
        ]
        for help_line, type_line, metric_name, prefix in telemetry_specs:
            emitted_header = False
            for stat in ("avg", "max"):
                value = _nested_number(telemetry, (*prefix, stat))
                if value is None:
                    continue
                if not emitted_header:
                    lines.extend([help_line, type_line])
                    emitted_header = True
                lines.append(f'{metric_name}{{{labels},stat="{stat}"}} {value:g}')

        triton_specs = [
            (
                "request_success_total",
                "triton_benchmark_server_request_success_total",
                "Correlated Triton successful request counter from telemetry snapshot.",
            ),
            (
                "request_failure_total",
                "triton_benchmark_server_request_failure_total",
                "Correlated Triton failed request counter from telemetry snapshot.",
            ),
            (
                "request_duration_us_total",
                "triton_benchmark_server_request_duration_us_total",
                "Correlated Triton request-duration counter from telemetry snapshot.",
            ),
            (
                "queue_duration_us_total",
                "triton_benchmark_server_queue_duration_us_total",
                "Correlated Triton queue-duration counter from telemetry snapshot.",
            ),
            (
                "compute_infer_duration_us_total",
                "triton_benchmark_server_compute_infer_duration_us_total",
                "Correlated Triton compute-infer-duration counter from telemetry snapshot.",
            ),
        ]
        for source_key, metric_name, help_text in triton_specs:
            value = _nested_number(telemetry, ("triton", source_key))
            if value is None:
                continue
            lines.extend([f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"])
            lines.append(f"{metric_name}{{{labels}}} {value:g}")

    telemetry_gauge_window = typed_metrics.get("telemetry_gauge_window")
    if isinstance(telemetry_gauge_window, dict):
        scrape_specs = (
            ("scrape_count", "window"),
            ("in_window_scrape_count", "in_window"),
        )
        emitted_scrape_header = False
        for source_key, scope in scrape_specs:
            value = telemetry_gauge_window.get(source_key)
            if not isinstance(value, (int, float)):
                continue
            if not emitted_scrape_header:
                lines.extend(
                    [
                        (
                            "# HELP triton_benchmark_gpu_window_scrapes Prometheus "
                            "scrapes used for the sampled GPU gauge window."
                        ),
                        "# TYPE triton_benchmark_gpu_window_scrapes gauge",
                    ]
                )
                emitted_scrape_header = True
            lines.append(
                "triton_benchmark_gpu_window_scrapes"
                f'{{{labels},phase="measured",scope="{scope}"}} {value:g}'
            )

        gauge_membership = telemetry_gauge_window.get("series_membership")
        if isinstance(gauge_membership, dict):
            stable = gauge_membership.get("stable")
            if isinstance(stable, bool):
                lines.extend(
                    [
                        "# HELP triton_benchmark_gpu_window_series_membership_stable Whether known GPU series membership stayed stable across every sampled scrape.",
                        "# TYPE triton_benchmark_gpu_window_series_membership_stable gauge",
                        (
                            "triton_benchmark_gpu_window_series_membership_stable"
                            f"{{{labels}}} {int(stable)}"
                        ),
                    ]
                )
            series_count = gauge_membership.get("series_count")
            if isinstance(series_count, int) and not isinstance(series_count, bool):
                lines.extend(
                    [
                        "# HELP triton_benchmark_gpu_window_series_count Known GPU series represented by each stable scrape.",
                        "# TYPE triton_benchmark_gpu_window_series_count gauge",
                        f"triton_benchmark_gpu_window_series_count{{{labels}}} {series_count}",
                    ]
                )

        gauge_specs = (
            (
                "utilization_pct",
                "triton_benchmark_gpu_window_utilization_percent",
                "Sampled GPU utilization across the bracketed request window.",
            ),
            (
                "memory_copy_utilization_pct",
                "triton_benchmark_gpu_window_memory_copy_utilization_percent",
                "Sampled GPU memory-copy utilization across the bracketed request window.",
            ),
            (
                "memory_used_mib",
                "triton_benchmark_gpu_window_memory_used_mib",
                "Sampled GPU memory use across the bracketed request window.",
            ),
        )
        gpu_window = telemetry_gauge_window.get("gpu")
        if isinstance(gpu_window, dict):
            for source_key, metric_name, help_text in gauge_specs:
                distribution = gpu_window.get(source_key)
                if not isinstance(distribution, dict) or not distribution:
                    continue
                lines.extend(
                    [f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"]
                )
                for stat in ("avg", "min", "max"):
                    value = distribution.get(stat)
                    if isinstance(value, (int, float)):
                        lines.append(
                            f'{metric_name}{{{labels},stat="{stat}"}} {value:g}'
                        )
                for stat, quantile in (("p50", "0.50"), ("p95", "0.95")):
                    value = distribution.get(stat)
                    if isinstance(value, (int, float)):
                        lines.append(
                            f'{metric_name}{{{labels},quantile="{quantile}"}} {value:g}'
                        )

    telemetry_window = typed_metrics.get("telemetry_window")
    if isinstance(telemetry_window, dict):
        counter_membership = telemetry_window.get("series_membership")
        if isinstance(counter_membership, dict):
            stable = counter_membership.get("stable")
            if isinstance(stable, bool):
                lines.extend(
                    [
                        "# HELP triton_benchmark_server_series_membership_stable Whether Triton counter series membership matched across paired snapshots.",
                        "# TYPE triton_benchmark_server_series_membership_stable gauge",
                        (
                            "triton_benchmark_server_series_membership_stable"
                            f"{{{labels}}} {int(stable)}"
                        ),
                    ]
                )
            membership_counts = (
                ("before_series_count", "before"),
                ("after_series_count", "after"),
            )
            emitted_count_header = False
            for source_key, snapshot in membership_counts:
                series_count = counter_membership.get(source_key)
                if not isinstance(series_count, int) or isinstance(series_count, bool):
                    continue
                if not emitted_count_header:
                    lines.extend(
                        [
                            "# HELP triton_benchmark_server_series_count Triton counter series represented by each paired snapshot.",
                            "# TYPE triton_benchmark_server_series_count gauge",
                        ]
                    )
                    emitted_count_header = True
                lines.append(
                    "triton_benchmark_server_series_count"
                    f'{{{labels},snapshot="{snapshot}"}} {series_count}'
                )

        counter_deltas = telemetry_window.get("deltas")
        if isinstance(counter_deltas, dict):
            emitted_header = False
            for counter_name in TELEMETRY_COUNTER_KEYS:
                value = counter_deltas.get(counter_name)
                if not isinstance(value, (int, float)):
                    continue
                if not emitted_header:
                    lines.extend(
                        [
                            "# HELP triton_benchmark_server_counter_delta Triton counter increase between paired pre-run and post-run snapshots.",
                            "# TYPE triton_benchmark_server_counter_delta gauge",
                        ]
                    )
                    emitted_header = True
                lines.append(
                    f'triton_benchmark_server_counter_delta{{{labels},counter="{counter_name}"}} {value:g}'
                )

        derived_specs = (
            (
                "server_failure_rate",
                "triton_benchmark_server_failure_rate",
                "Failed Triton requests divided by all Triton requests in the paired counter window.",
            ),
            (
                "server_queue_fraction",
                "triton_benchmark_server_queue_duration_fraction",
                "Triton queue-duration delta divided by request-duration delta in the paired counter window.",
            ),
        )
        for source_key, metric_name, help_text in derived_specs:
            value = _nested_number(telemetry_window, ("derived", source_key))
            if value is None:
                continue
            lines.extend(
                [f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"]
            )
            lines.append(f"{metric_name}{{{labels}}} {value:g}")

    telemetry_gate = typed_metrics.get("telemetry_gate")
    if isinstance(telemetry_gate, dict):
        passed = telemetry_gate.get("passed")
        if isinstance(passed, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_telemetry_gate_passed Whether every configured paired-counter telemetry check passed.",
                    "# TYPE triton_benchmark_telemetry_gate_passed gauge",
                    f"triton_benchmark_telemetry_gate_passed{{{labels}}} {int(passed)}",
                ]
            )

    batch_invariance = typed_metrics.get("batch_invariance")
    if isinstance(batch_invariance, dict):
        batch_specs = [
            (
                "probe_count",
                "triton_benchmark_batch_invariance_probes_total",
                "Total fixed inputs checked for batch-invariant outputs.",
            ),
            (
                "mismatched_outputs",
                "triton_benchmark_batch_invariance_mismatches_total",
                "Outputs outside the configured comparison policy.",
            ),
            (
                "exact_matches",
                "triton_benchmark_batch_invariance_exact_matches_total",
                "Outputs with identical isolated and concurrent fingerprints.",
            ),
            (
                "tolerance_matches",
                "triton_benchmark_batch_invariance_tolerance_matches_total",
                "Numerically different outputs accepted by the run-scoped policy.",
            ),
            (
                "failed_probes",
                "triton_benchmark_batch_invariance_failures_total",
                "Probe inputs that failed in isolated or concurrent execution.",
            ),
            (
                "noise_failures",
                "triton_benchmark_batch_invariance_noise_failures_total",
                "Concurrent noise requests that failed during the probe.",
            ),
            (
                "match_rate",
                "triton_benchmark_batch_invariance_match_rate",
                "Ratio of fixed inputs within the configured output policy.",
            ),
            (
                "max_observed_absolute_error",
                "triton_benchmark_batch_invariance_max_absolute_error",
                "Maximum finite absolute numeric error across compared outputs.",
            ),
            (
                "max_observed_relative_error",
                "triton_benchmark_batch_invariance_max_relative_error",
                "Maximum finite relative error against isolated outputs.",
            ),
        ]
        for source_key, metric_name, help_text in batch_specs:
            value = _nested_number(batch_invariance, (source_key,))
            if value is None:
                continue
            lines.extend([f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"])
            lines.append(f"{metric_name}{{{labels}}} {value:g}")

        exact_match = batch_invariance.get("exact_match")
        if isinstance(exact_match, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_batch_invariance_exact_match Whether all probe outputs matched exactly.",
                    "# TYPE triton_benchmark_batch_invariance_exact_match gauge",
                    (
                        f"triton_benchmark_batch_invariance_exact_match{{{labels}}} "
                        f"{int(exact_match)}"
                    ),
                ]
            )

        passed = batch_invariance.get("passed")
        if isinstance(passed, bool):
            lines.extend(
                [
                    "# HELP triton_benchmark_batch_invariance_passed Whether all probes passed the configured output policy without request or noise failures.",
                    "# TYPE triton_benchmark_batch_invariance_passed gauge",
                    (
                        f"triton_benchmark_batch_invariance_passed{{{labels}}} "
                        f"{int(passed)}"
                    ),
                ]
            )

        comparison_policy = batch_invariance.get("comparison_policy")
        if isinstance(comparison_policy, dict):
            tolerance_specs = [
                (
                    "absolute_tolerance",
                    "triton_benchmark_batch_invariance_absolute_tolerance",
                    "Configured run-scoped absolute output tolerance.",
                ),
                (
                    "relative_tolerance",
                    "triton_benchmark_batch_invariance_relative_tolerance",
                    "Configured run-scoped relative output tolerance.",
                ),
            ]
            for source_key, metric_name, help_text in tolerance_specs:
                value = _nested_number(comparison_policy, (source_key,))
                if value is None:
                    continue
                lines.extend(
                    [f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"]
                )
                lines.append(f"{metric_name}{{{labels}}} {value:g}")

    streaming = typed_metrics.get("streaming")
    if isinstance(streaming, dict):
        distribution_specs = [
            (
                "time_to_first_token_ms",
                "triton_benchmark_streaming_ttft_ms",
                "Measured time to first non-empty streamed text event.",
            ),
            (
                "inter_chunk_latency_ms",
                "triton_benchmark_streaming_inter_chunk_latency_ms",
                "Measured gap between non-empty streamed text events.",
            ),
        ]
        for source_key, metric_name, help_text in distribution_specs:
            values = streaming.get(source_key)
            if not isinstance(values, dict):
                continue
            lines.extend([f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"])
            for stat in ("avg", "min", "max"):
                value = values.get(stat)
                if isinstance(value, (int, float)):
                    lines.append(f'{metric_name}{{{labels},stat="{stat}"}} {value:g}')
            for stat, quantile in (("p50", "0.50"), ("p95", "0.95"), ("p99", "0.99")):
                value = values.get(stat)
                if isinstance(value, (int, float)):
                    lines.append(
                        f'{metric_name}{{{labels},quantile="{quantile}"}} {value:g}'
                    )

        scalar_specs = [
            (
                "reported_token_request_count",
                "triton_benchmark_streaming_reported_token_requests",
                "Successful streams with valid server-reported completion-token usage.",
            ),
            (
                "missing_token_usage_requests",
                "triton_benchmark_streaming_missing_token_usage_requests",
                "Successful streams without a server-reported completion-token count.",
            ),
            (
                "invalid_token_usage_requests",
                "triton_benchmark_streaming_invalid_token_usage_requests",
                "Successful streams with invalid server-reported completion-token usage.",
            ),
            (
                "reported_output_tokens",
                "triton_benchmark_streaming_reported_output_tokens_total",
                "Server-reported output tokens across successful streaming requests.",
            ),
            (
                "observed_output_chunks",
                "triton_benchmark_streaming_observed_output_chunks_total",
                "Observed non-empty streamed text events; chunks are not tokens.",
            ),
            (
                "output_bytes",
                "triton_benchmark_streaming_output_bytes_total",
                "UTF-8 output bytes observed across successful streaming requests.",
            ),
            (
                "output_tokens_per_second",
                "triton_benchmark_streaming_output_tokens_per_second",
                "Server-reported output-token throughput when usage coverage is complete.",
            ),
        ]
        for source_key, metric_name, help_text in scalar_specs:
            value = streaming.get(source_key)
            if not isinstance(value, (int, float)):
                continue
            lines.extend([f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"])
            lines.append(f"{metric_name}{{{labels}}} {value:g}")

    cost_model = typed_metrics.get("cost_model")
    if isinstance(cost_model, dict):
        workload_specs = [
            (("workload", "successful_input_tokens"), "input"),
            (("workload", "successful_output_tokens"), "output"),
            (("workload", "successful_total_tokens"), "total"),
        ]
        lines.extend(
            [
                "# HELP triton_benchmark_workload_tokens_total Successful tokens represented by the benchmark workload.",
                "# TYPE triton_benchmark_workload_tokens_total gauge",
            ]
        )
        for keys, direction in workload_specs:
            value = _nested_number(cost_model, keys)
            if value is not None:
                lines.append(
                    f'triton_benchmark_workload_tokens_total{{{labels},'
                    f'direction="{direction}"}} {value:g}'
                )

        throughput_specs = [
            (("workload", "output_tokens_per_second"), "output"),
            (("workload", "total_tokens_per_second"), "total"),
        ]
        lines.extend(
            [
                "# HELP triton_benchmark_token_throughput_per_second Successful token throughput.",
                "# TYPE triton_benchmark_token_throughput_per_second gauge",
            ]
        )
        for keys, direction in throughput_specs:
            value = _nested_number(cost_model, keys)
            if value is not None:
                lines.append(
                    f'triton_benchmark_token_throughput_per_second{{{labels},'
                    f'direction="{direction}"}} {value:g}'
                )

        cost_specs = [
            (("cost", "accelerator_cost_usd"), "accelerator"),
            (("cost", "electricity_cost_usd"), "electricity"),
            (("cost", "total_estimated_cost_usd"), "total"),
        ]
        lines.extend(
            [
                "# HELP triton_benchmark_estimated_cost_usd Estimated benchmark-run cost by component.",
                "# TYPE triton_benchmark_estimated_cost_usd gauge",
            ]
        )
        for keys, component in cost_specs:
            value = _nested_number(cost_model, keys)
            if value is not None:
                lines.append(
                    f'triton_benchmark_estimated_cost_usd{{{labels},'
                    f'component="{component}"}} {value:g}'
                )

        normalized_cost_specs = [
            ("cost_per_million_requests_usd", "request"),
            ("cost_per_million_input_tokens_usd", "input_token"),
            ("cost_per_million_output_tokens_usd", "output_token"),
            ("cost_per_million_total_tokens_usd", "total_token"),
        ]
        emitted_normalized_header = False
        for source_key, unit in normalized_cost_specs:
            value = _nested_number(cost_model, ("cost", source_key))
            if value is None:
                continue
            if not emitted_normalized_header:
                lines.extend(
                    [
                        "# HELP triton_benchmark_estimated_cost_per_million_usd Estimated cost per million successful units.",
                        "# TYPE triton_benchmark_estimated_cost_per_million_usd gauge",
                    ]
                )
                emitted_normalized_header = True
            lines.append(
                f'triton_benchmark_estimated_cost_per_million_usd{{{labels},'
                f'unit="{unit}"}} {value:g}'
            )

    llm_metrics = typed_metrics.get("llm_metrics")
    if isinstance(llm_metrics, dict):
        latency_specs = [
            (("latency_ms", "time_to_first_token"), "ttft"),
            (("latency_ms", "inter_token"), "itl"),
        ]
        emitted_latency_header = False
        for keys, phase in latency_specs:
            value = _nested_number(llm_metrics, keys)
            if value is None:
                continue
            if not emitted_latency_header:
                lines.extend(
                    [
                        "# HELP triton_benchmark_llm_latency_ms LLM decode latency by phase.",
                        "# TYPE triton_benchmark_llm_latency_ms gauge",
                    ]
                )
                emitted_latency_header = True
            lines.append(
                f'triton_benchmark_llm_latency_ms{{{labels},phase="{phase}"}} {value:g}'
            )

        llm_specs = [
            (
                ("context_tokens_per_request",),
                "triton_benchmark_llm_context_tokens",
                "Prompt context tokens represented by each request.",
            ),
            (
                ("batch_size",),
                "triton_benchmark_llm_batch_size",
                "Logical decode batch size.",
            ),
            (
                ("memory", "kv_cache_bytes_per_request"),
                "triton_benchmark_llm_kv_cache_bytes",
                "Logical KV-cache bytes per request.",
            ),
            (
                ("memory", "bytes_read_per_output_token"),
                "triton_benchmark_llm_bytes_read_per_output_token",
                "Logical bytes read for each generated output token.",
            ),
            (
                ("energy", "estimated_joules_per_output_token"),
                "triton_benchmark_llm_joules_per_output_token",
                "Estimated board joules per generated output token.",
            ),
            (
                ("quality", "relative_degradation_percent"),
                "triton_benchmark_llm_quality_degradation_percent",
                "Relative quality-score degradation from baseline.",
            ),
        ]
        for keys, metric_name, help_text in llm_specs:
            value = _nested_number(llm_metrics, keys)
            if value is None:
                continue
            lines.extend(
                [f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} gauge"]
            )
            lines.append(f"{metric_name}{{{labels}}} {value:g}")
    return "\n".join(lines) + "\n"


def save_prometheus_metrics(metrics: dict[str, object], metrics_path: Path) -> Path:
    prometheus_path = metrics_path.with_suffix(".prom")
    prometheus_path.write_text(format_prometheus_metrics(metrics), encoding="utf-8")
    return prometheus_path
