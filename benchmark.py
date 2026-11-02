from __future__ import annotations

import argparse
import json
import math
import os
import time  # noqa: F401
import urllib.request  # noqa: F401
from dataclasses import asdict

from bench_core.config import (  # noqa: F401
    BenchmarkConfig,
    CliOptions,
    CoordinatedClientConfig,
    CostModelConfig,
    GPU_GAUGE_METRIC_ALIASES,
    GPU_MEMORY_COPY_METRICS,
    GPU_MEMORY_USED_METRICS,
    GPU_UTILIZATION_METRICS,
    GpuGaugeCapture,
    InferenceResult,
    LlmMetricsConfig,
    MAX_TELEMETRY_ENDPOINTS,
    MAX_TELEMETRY_RESPONSE_BYTES,
    NumericOutput,
    OutputInferenceResult,
    OutputObservation,
    PROMETHEUS_METRIC_NAME_PATTERN,
    PrometheusSample,
    RequestPathMetricNames,
    StreamingInferenceObservation,
    TELEMETRY_COUNTER_KEYS,
    TRACEPARENT_VERSION_00_PATTERN,
    TRITON_COMPUTE_INFER_DURATION_METRICS,
    TRITON_COUNTER_METRIC_ALIASES,
    TRITON_QUEUE_DURATION_METRICS,
    TRITON_REQUEST_DURATION_METRICS,
    TRITON_REQUEST_FAILURE_METRICS,
    TRITON_REQUEST_SUCCESS_METRICS,
    WORKLOAD_PROFILES,
    WorkloadProfile,
    _config_number,
    _latency,
    _nested_number,
    _nonzero_random_hex,
    _number,
    build_traceparent,
    classify_response_traceparent,
    normalize_openai_completions_url,
    percentile,
    sanitize_server_url,
)
from bench_core.telemetry_clients import (  # noqa: F401
    HttpPrometheusTelemetryClient,
    MultiEndpointPrometheusTelemetryClient,
    TelemetrySnapshotClient,
    _normalize_telemetry_endpoint_urls,
    build_http_telemetry_client,
)
from bench_core.clients import (  # noqa: F401
    InferenceClient,
    MockInferenceClient,
    OpenAICompatibleStreamingClient,
    OutputInferenceClient,
    TritonHttpInferenceClient,
    capture_triton_outputs,
    fingerprint_triton_outputs,
)
from bench_core.summary import (  # noqa: F401
    _complete_streaming_output_tokens,
    _summarize_retry_attempts,
    build_cost_model,
    build_llm_metrics,
    build_trace_context_gate,
    save_metrics,
    summarize_results,
)
from bench_core.telemetry import (  # noqa: F401
    _build_gpu_gauge_capture,
    _build_gpu_gauge_window,
    _build_request_path_stage_window,
    _build_series_membership,
    _extract_gpu_gauge_values,
    _extract_gpu_gauge_values_from_samples,
    _gauge_window_stat_summary,
    _nonnegative_int,
    _parse_metric_and_labels,
    _request_path_counter_series,
    _request_path_membership_fingerprint,
    _sample_matches_model,
    _split_prometheus_labels,
    _split_prometheus_sample_line,
    _stat_summary,
    _sum_values,
    _unescape_prometheus_label,
    _values_for_metrics,
    attach_telemetry_snapshots,
    attach_telemetry_summary,
    build_gpu_gauge_window,
    build_request_path_accounting,
    build_request_path_gate,
    build_service_lifecycle_accounting,
    build_service_lifecycle_gate,
    build_telemetry_counter_window,
    build_telemetry_gate,
    build_telemetry_summary,
    parse_prometheus_samples,
)
from bench_core.prometheus_export import (  # noqa: F401
    _base_labels,
    _escape_label,
    format_prometheus_metrics,
    save_prometheus_metrics,
)
from bench_core.regression import (  # noqa: F401
    _percent_change,
    build_regression_report,
    load_metrics,
    save_regression_report,
)
from bench_core.runner import (  # noqa: F401
    _build_load_schedule_summary,
    _compare_output_observations,
    _run_request_phase,
    _sample_telemetry_during_phase,
    _summarize_request_phase,
    build_constant_rate_submission_offsets,
    build_retry_gate,
    execute_output_with_retries,
    execute_with_retries,
    run_batch_invariance_probe,
    run_benchmark,
)


def build_client(config: BenchmarkConfig) -> InferenceClient:
    if config.mode == "mock":
        return MockInferenceClient(seed=config.seed)
    if config.mode == "triton":
        return TritonHttpInferenceClient(
            server_url=config.server_url,
            model_name=config.model_name,
            input_name=config.input_name,
            input_shape=config.input_shape,
            propagate_trace_context=config.propagate_trace_context,
        )
    if config.mode == "openai":
        return OpenAICompatibleStreamingClient(
            server_url=config.server_url,
            model_name=config.model_name,
            prompt=config.openai_prompt,
            max_tokens=config.openai_max_tokens,
            timeout_seconds=config.openai_timeout_seconds,
            api_key=(
                os.environ.get(config.openai_api_key_env)
                if config.openai_api_key_env
                else None
            ),
            propagate_trace_context=config.propagate_trace_context,
        )
    raise ValueError(f"Unsupported mode: {config.mode}")


def resolve_workload_profile(name: str | None) -> WorkloadProfile | None:
    if name is None:
        return None
    try:
        return WORKLOAD_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(WORKLOAD_PROFILES))
        raise ValueError(f"Unknown workload profile {name!r}; choose from {choices}") from exc


def parse_shape(raw_shape: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw_shape.split(",") if part.strip())


def parse_args() -> CliOptions:
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Benchmark Triton-style inference workloads.")
    parser.add_argument("--mode", choices=["mock", "triton", "openai"], default="mock")
    parser.add_argument("--server-url", default="localhost:8000")
    parser.add_argument("--model-name", default="resnet50_trt_fp16")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help=(
            "Requests run before the measured phase and reported separately; "
            "they are excluded from headline metrics and cost calculations."
        ),
    )
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.0,
        help=(
            "Fixed client delay between failed attempts; a zero value retries immediately. "
            "The configured delay is not measured service recovery time."
        ),
    )
    parser.add_argument(
        "--max-client-attempt-amplification",
        type=float,
        help=(
            "Maximum measured client calls per measured logical request; values "
            "must be at least 1."
        ),
    )
    parser.add_argument(
        "--fail-on-retry-gate",
        action="store_true",
        help=(
            "Exit with status 6 when measured client-attempt amplification exceeds "
            "--max-client-attempt-amplification."
        ),
    )
    parser.add_argument(
        "--request-rate-rps",
        type=float,
        default=0.0,
        help=(
            "Open-loop constant measured-request submission rate; zero submits "
            "immediately as before. Warmup is not paced."
        ),
    )
    parser.add_argument(
        "--propagate-trace-context",
        action="store_true",
        help=(
            "Add a fresh sampled W3C traceparent to each live HTTP request attempt; "
            "identifiers are not written to artifacts."
        ),
    )
    parser.add_argument(
        "--fail-on-trace-context-gap",
        action="store_true",
        help=(
            "Exit with status 5 unless every measured successful OpenAI response "
            "returns a valid traceparent on the outbound trace."
        ),
    )
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--coordinated-run-id",
        help=(
            "Coordinator-issued run identifier; only its SHA-256 fingerprint is "
            "written to the child artifact."
        ),
    )
    parser.add_argument("--coordinated-client-index", type=int)
    parser.add_argument("--coordinated-client-count", type=int)
    parser.add_argument("--coordinated-start-unix-ns", type=int)
    parser.add_argument(
        "--openai-prompt",
        default="Return a short deterministic benchmark response.",
        help="Synthetic prompt used only by OpenAI-compatible streaming mode.",
    )
    parser.add_argument(
        "--openai-max-tokens",
        type=int,
        default=0,
        help="Maximum streamed output tokens; defaults to the workload profile or 128.",
    )
    parser.add_argument(
        "--openai-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request timeout for OpenAI-compatible streaming mode.",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="",
        help="Optional environment variable containing a bearer token.",
    )
    parser.add_argument(
        "--workload-profile",
        choices=sorted(WORKLOAD_PROFILES),
        help=(
            "Apply a named, workload-shaped LLM profile. Explicit token and "
            "latency flags override the profile values."
        ),
    )
    parser.add_argument(
        "--input-tokens-per-request",
        type=int,
        default=0,
        help="Estimated input tokens represented by each successful request.",
    )
    parser.add_argument(
        "--output-tokens-per-request",
        type=int,
        default=0,
        help="Estimated output tokens represented by each successful request.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Accelerators reserved for the run when estimating capacity and cost.",
    )
    parser.add_argument(
        "--gpu-hourly-cost-usd",
        type=float,
        default=0.0,
        help="Hourly price per reserved accelerator.",
    )
    parser.add_argument(
        "--power-watts-per-gpu",
        type=float,
        default=0.0,
        help="Average accelerator power draw used for the energy estimate.",
    )
    parser.add_argument(
        "--electricity-cost-usd-per-kwh",
        type=float,
        default=0.0,
        help="Electricity price added when modeling owned accelerator capacity.",
    )
    parser.add_argument(
        "--context-tokens-per-request",
        type=int,
        default=0,
        help="Prompt context represented by each LLM request.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=1,
        help="Logical decode batch size represented by the LLM measurement.",
    )
    parser.add_argument(
        "--time-to-first-token-ms",
        type=float,
        help="Measured model-forward time to first generated token.",
    )
    parser.add_argument(
        "--inter-token-latency-ms",
        type=float,
        help="Measured steady-state decode latency per generated token.",
    )
    parser.add_argument(
        "--kv-cache-bytes-per-request",
        type=int,
        default=0,
        help="Logical KV-cache capacity occupied by one request.",
    )
    parser.add_argument(
        "--bytes-read-per-output-token",
        type=int,
        default=0,
        help="Logical memory bytes read for each generated token.",
    )
    parser.add_argument(
        "--baseline-quality-score",
        type=float,
        help="Reference score from an identified model-quality evaluation.",
    )
    parser.add_argument(
        "--candidate-quality-score",
        type=float,
        help="Candidate score from the same model-quality evaluation.",
    )
    parser.add_argument(
        "--prometheus",
        action="store_true",
        help="Write a Prometheus text-format .prom file beside the JSON result.",
    )
    parser.add_argument(
        "--telemetry-prometheus",
        help=(
            "Optional post-run Prometheus text snapshot from Triton/DCGM; a correlated "
            "telemetry summary is attached to the JSON and .prom outputs."
        ),
    )
    parser.add_argument(
        "--telemetry-baseline-prometheus",
        help=(
            "Optional pre-run Prometheus snapshot paired with --telemetry-prometheus "
            "to derive operator-supplied observed-window Triton counter deltas."
        ),
    )
    parser.add_argument(
        "--telemetry-url",
        action="append",
        default=[],
        help=(
            "Optional HTTP(S) Prometheus endpoint scraped after warmup, immediately "
            "before measured requests, and immediately after they complete. Repeat "
            "the option to capture independent sources concurrently."
        ),
    )
    parser.add_argument(
        "--telemetry-timeout-seconds",
        type=float,
        default=10.0,
        help="Timeout for each opt-in --telemetry-url scrape.",
    )
    parser.add_argument(
        "--telemetry-sample-interval-seconds",
        type=float,
        default=0.0,
        help=(
            "Sample GPU gauges at this interval while measured requests are in flight; "
            "zero disables in-window sampling."
        ),
    )
    parser.add_argument(
        "--telemetry-api-key-env",
        default="",
        help=(
            "Optional environment variable containing the telemetry endpoint bearer "
            "token; no ambient API key is sent."
        ),
    )
    parser.add_argument(
        "--request-path-ingress-metric",
        help="Cumulative Prometheus counter for requests accepted by the ingress layer.",
    )
    parser.add_argument(
        "--request-path-backend-metric",
        help="Cumulative Prometheus counter for requests accepted by the backend layer.",
    )
    parser.add_argument(
        "--request-path-success-metric",
        help="Cumulative Prometheus counter for successful backend completions.",
    )
    parser.add_argument(
        "--fail-on-request-path-gap",
        action="store_true",
        help=(
            "Exit with status 7 unless isolated ingress/backend/success counters "
            "exactly reconcile with measured client attempts and outcomes."
        ),
    )
    parser.add_argument(
        "--service-restart-metric",
        help=(
            "Cumulative Prometheus counter for completed service restarts; the metric "
            "name and labels are not persisted."
        ),
    )
    parser.add_argument(
        "--min-service-restarts",
        type=int,
        help="Minimum completed service restarts required in the paired counter window.",
    )
    parser.add_argument(
        "--max-service-restarts",
        type=int,
        help="Maximum completed service restarts allowed in the paired counter window.",
    )
    parser.add_argument(
        "--fail-on-service-lifecycle-gap",
        action="store_true",
        help=(
            "Exit with status 8 unless the selected restart counter is valid and "
            "within the configured bounds."
        ),
    )
    parser.add_argument(
        "--max-server-failure-rate",
        type=float,
        help="Maximum failed-request fraction in the paired Triton counter window.",
    )
    parser.add_argument(
        "--max-server-queue-fraction",
        type=float,
        help=(
            "Maximum queue-duration delta divided by request-duration delta in the "
            "paired Triton counter window."
        ),
    )
    parser.add_argument(
        "--fail-on-telemetry-gate",
        action="store_true",
        help="Exit with status 4 when a configured telemetry check fails or is unavailable.",
    )
    parser.add_argument(
        "--batch-invariance-probes",
        type=int,
        default=0,
        help=(
            "Run this many fixed inputs in isolation and under concurrent noise traffic, "
            "then compare output correctness under the configured policy."
        ),
    )
    parser.add_argument(
        "--batch-output-atol",
        type=float,
        default=0.0,
        help=(
            "Run-scoped absolute numeric tolerance for batch-invariance outputs; "
            "zero preserves exact comparison unless a relative tolerance is set."
        ),
    )
    parser.add_argument(
        "--batch-output-rtol",
        type=float,
        default=0.0,
        help=(
            "Run-scoped relative numeric tolerance for batch-invariance outputs; "
            "evaluated against the isolated value."
        ),
    )
    parser.add_argument(
        "--baseline",
        help="Optional prior benchmark JSON file used for baseline-versus-candidate comparison.",
    )
    parser.add_argument(
        "--max-p95-regression-pct",
        type=float,
        default=10.0,
        help="Allowed p95 latency increase before the comparison is marked as a regression.",
    )
    parser.add_argument(
        "--max-success-rate-drop",
        type=float,
        default=0.01,
        help="Allowed success-rate drop before the comparison is marked as a regression.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 2 when the baseline comparison is marked as a regression.",
    )
    parser.add_argument(
        "--fail-on-batch-variance",
        action="store_true",
        help="Exit with status 3 when any batch-invariance probe fails or changes output.",
    )
    args = parser.parse_args()
    telemetry_urls = tuple(args.telemetry_url)
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be zero or greater")
    if args.request_rate_rps < 0:
        parser.error("--request-rate-rps must be zero or greater")
    if args.retries < 0:
        parser.error("--retries must be zero or greater")
    coordination_values = (
        args.coordinated_run_id,
        args.coordinated_client_index,
        args.coordinated_client_count,
        args.coordinated_start_unix_ns,
    )
    coordination: CoordinatedClientConfig | None = None
    if any(value is not None for value in coordination_values) and not all(
        value is not None for value in coordination_values
    ):
        parser.error("coordinated client options must be supplied together")
    if all(value is not None for value in coordination_values):
        assert args.coordinated_run_id is not None
        assert args.coordinated_client_index is not None
        assert args.coordinated_client_count is not None
        assert args.coordinated_start_unix_ns is not None
        if not args.coordinated_run_id or len(args.coordinated_run_id) > 256:
            parser.error("--coordinated-run-id must contain 1 to 256 characters")
        if not 2 <= args.coordinated_client_count <= 64:
            parser.error("--coordinated-client-count must be between 2 and 64")
        if not 0 <= args.coordinated_client_index < args.coordinated_client_count:
            parser.error("--coordinated-client-index is outside the client set")
        if args.coordinated_start_unix_ns <= 0:
            parser.error("--coordinated-start-unix-ns must be positive")
        coordination = CoordinatedClientConfig(
            run_id=args.coordinated_run_id,
            client_index=args.coordinated_client_index,
            client_count=args.coordinated_client_count,
            planned_start_unix_ns=args.coordinated_start_unix_ns,
        )
    if not math.isfinite(args.retry_backoff_seconds) or args.retry_backoff_seconds < 0:
        parser.error("--retry-backoff-seconds must be finite and non-negative")
    if args.max_client_attempt_amplification is not None and (
        not math.isfinite(args.max_client_attempt_amplification)
        or args.max_client_attempt_amplification < 1
    ):
        parser.error("--max-client-attempt-amplification must be finite and at least 1")
    if args.fail_on_retry_gate and args.max_client_attempt_amplification is None:
        parser.error(
            "--fail-on-retry-gate requires --max-client-attempt-amplification"
        )
    if args.batch_invariance_probes < 0:
        parser.error("--batch-invariance-probes must be zero or greater")
    if args.batch_invariance_probes and args.concurrency <= 1:
        parser.error("--batch-invariance-probes requires --concurrency greater than one")
    batch_output_tolerances = (args.batch_output_atol, args.batch_output_rtol)
    if any(
        not math.isfinite(value) or value < 0
        for value in batch_output_tolerances
    ):
        parser.error("batch output tolerances must be finite and non-negative")
    if any(value > 0 for value in batch_output_tolerances) and not (
        args.batch_invariance_probes
    ):
        parser.error("batch output tolerances require --batch-invariance-probes")
    if args.fail_on_batch_variance and not args.batch_invariance_probes:
        parser.error("--fail-on-batch-variance requires --batch-invariance-probes")
    if args.propagate_trace_context and args.mode == "mock":
        parser.error("--propagate-trace-context requires triton or openai mode")
    if args.fail_on_trace_context_gap and not (
        args.mode == "openai" and args.propagate_trace_context
    ):
        parser.error(
            "--fail-on-trace-context-gap requires openai mode and "
            "--propagate-trace-context"
        )
    telemetry_thresholds = (
        args.max_server_failure_rate,
        args.max_server_queue_fraction,
    )
    paired_snapshot_files = bool(
        args.telemetry_baseline_prometheus and args.telemetry_prometheus
    )
    request_path_values = (
        args.request_path_ingress_metric,
        args.request_path_backend_metric,
        args.request_path_success_metric,
    )
    request_path_metrics: RequestPathMetricNames | None = None
    if any(request_path_values) and not all(request_path_values):
        parser.error(
            "request-path accounting requires ingress, backend, and success metric names"
        )
    if all(request_path_values):
        request_path_metrics = RequestPathMetricNames(
            ingress=args.request_path_ingress_metric,
            backend=args.request_path_backend_metric,
            success=args.request_path_success_metric,
        )
        metric_names = tuple(request_path_metrics.by_stage().values())
        if len(set(metric_names)) != len(metric_names):
            parser.error("request-path metric names must be distinct")
        if any(
            PROMETHEUS_METRIC_NAME_PATTERN.fullmatch(name) is None
            for name in metric_names
        ):
            parser.error("request-path metric names must be valid Prometheus names")
        if not (paired_snapshot_files or telemetry_urls):
            parser.error(
                "request-path accounting requires --telemetry-url or paired telemetry snapshots"
            )
    if args.fail_on_request_path_gap and request_path_metrics is None:
        parser.error("--fail-on-request-path-gap requires request-path metric names")
    service_restart_bounds = (
        args.min_service_restarts,
        args.max_service_restarts,
    )
    if any(value is not None and value < 0 for value in service_restart_bounds):
        parser.error("service restart bounds must be zero or greater")
    if (
        args.min_service_restarts is not None
        and args.max_service_restarts is not None
        and args.min_service_restarts > args.max_service_restarts
    ):
        parser.error("--min-service-restarts must not exceed --max-service-restarts")
    if args.service_restart_metric is not None and (
        PROMETHEUS_METRIC_NAME_PATTERN.fullmatch(args.service_restart_metric) is None
    ):
        parser.error("--service-restart-metric must be a valid Prometheus name")
    if any(value is not None for value in service_restart_bounds) and (
        args.service_restart_metric is None
    ):
        parser.error("service restart bounds require --service-restart-metric")
    if args.service_restart_metric is not None and not (
        paired_snapshot_files or telemetry_urls
    ):
        parser.error(
            "service lifecycle accounting requires --telemetry-url or paired telemetry snapshots"
        )
    if args.fail_on_service_lifecycle_gap and (
        args.service_restart_metric is None
        or not any(value is not None for value in service_restart_bounds)
    ):
        parser.error(
            "--fail-on-service-lifecycle-gap requires a service restart metric and bound"
        )
    if (
        request_path_metrics is not None
        and args.service_restart_metric in request_path_metrics.by_stage().values()
    ):
        parser.error("service restart and request-path metric names must be distinct")
    if telemetry_urls and (
        args.telemetry_baseline_prometheus or args.telemetry_prometheus
    ):
        parser.error(
            "--telemetry-url is mutually exclusive with telemetry snapshot files"
        )
    if args.telemetry_api_key_env and not telemetry_urls:
        parser.error("--telemetry-api-key-env requires --telemetry-url")
    if args.telemetry_sample_interval_seconds < 0:
        parser.error("--telemetry-sample-interval-seconds must be zero or greater")
    if args.telemetry_sample_interval_seconds and not telemetry_urls:
        parser.error("--telemetry-sample-interval-seconds requires --telemetry-url")
    if args.telemetry_timeout_seconds <= 0:
        parser.error("--telemetry-timeout-seconds must be greater than zero")
    if telemetry_urls:
        try:
            _normalize_telemetry_endpoint_urls(telemetry_urls)
            for telemetry_url in telemetry_urls:
                HttpPrometheusTelemetryClient(
                    telemetry_url,
                    timeout_seconds=args.telemetry_timeout_seconds,
                )
        except ValueError as exc:
            parser.error(str(exc))
    if any(value is not None for value in telemetry_thresholds) and not (
        paired_snapshot_files or telemetry_urls
    ):
        parser.error(
            "telemetry thresholds require --telemetry-url or paired telemetry snapshots"
        )
    if args.telemetry_baseline_prometheus and not args.telemetry_prometheus:
        parser.error(
            "--telemetry-baseline-prometheus requires --telemetry-prometheus"
        )
    if args.fail_on_telemetry_gate and not any(
        value is not None for value in telemetry_thresholds
    ):
        parser.error("--fail-on-telemetry-gate requires a telemetry threshold")
    if any(
        value is not None and not 0 <= value <= 1
        for value in telemetry_thresholds
    ):
        parser.error("telemetry rate and fraction thresholds must be between zero and one")
    if args.mode == "openai" and args.batch_invariance_probes:
        parser.error("batch-invariance probes are not supported in openai mode")
    if not args.openai_prompt.strip():
        parser.error("--openai-prompt must not be empty")
    if args.openai_max_tokens < 0:
        parser.error("--openai-max-tokens must be zero or greater")
    if args.openai_timeout_seconds <= 0:
        parser.error("--openai-timeout-seconds must be greater than zero")
    if args.input_tokens_per_request < 0 or args.output_tokens_per_request < 0:
        parser.error("token counts must be zero or greater")
    if args.gpu_count <= 0:
        parser.error("--gpu-count must be greater than zero")
    if (
        args.gpu_hourly_cost_usd < 0
        or args.power_watts_per_gpu < 0
        or args.electricity_cost_usd_per_kwh < 0
    ):
        parser.error("cost-model inputs must be zero or greater")
    if (
        args.context_tokens_per_request < 0
        or args.llm_batch_size <= 0
        or args.kv_cache_bytes_per_request < 0
        or args.bytes_read_per_output_token < 0
    ):
        parser.error("LLM context, batch, and byte inputs must be non-negative")
    if (
        args.time_to_first_token_ms is not None
        and args.time_to_first_token_ms < 0
    ) or (
        args.inter_token_latency_ms is not None
        and args.inter_token_latency_ms < 0
    ):
        parser.error("LLM latency inputs must be zero or greater")
    if (args.baseline_quality_score is None) != (
        args.candidate_quality_score is None
    ):
        parser.error("baseline and candidate quality scores must be supplied together")

    workload_profile = resolve_workload_profile(args.workload_profile)
    profile_defaults = workload_profile or WorkloadProfile(
        name="",
        description="",
        context_tokens_per_request=0,
        output_tokens_per_request=0,
        batch_size=1,
        time_to_first_token_ms=0.0,
        inter_token_latency_ms=0.0,
        kv_cache_bytes_per_request=0,
        bytes_read_per_output_token=0,
    )

    def profile_value(explicit: object, default_value: object) -> object:
        if workload_profile is None:
            return explicit
        if explicit is None or explicit == 0:
            return default_value
        return explicit

    input_tokens_per_request = int(
        profile_value(
            args.input_tokens_per_request,
            profile_defaults.context_tokens_per_request,
        )
    )
    output_tokens_per_request = int(
        profile_value(
            args.output_tokens_per_request,
            profile_defaults.output_tokens_per_request,
        )
    )
    context_tokens_per_request = int(
        profile_value(
            args.context_tokens_per_request,
            profile_defaults.context_tokens_per_request,
        )
    )
    llm_batch_size = int(
        workload_profile.batch_size
        if workload_profile is not None and args.llm_batch_size == 1
        else args.llm_batch_size
    )
    time_to_first_token_ms = profile_value(
        args.time_to_first_token_ms,
        profile_defaults.time_to_first_token_ms,
    )
    inter_token_latency_ms = profile_value(
        args.inter_token_latency_ms,
        profile_defaults.inter_token_latency_ms,
    )
    kv_cache_bytes_per_request = int(
        profile_value(
            args.kv_cache_bytes_per_request,
            profile_defaults.kv_cache_bytes_per_request,
        )
    )
    bytes_read_per_output_token = int(
        profile_value(
            args.bytes_read_per_output_token,
            profile_defaults.bytes_read_per_output_token,
        )
    )
    openai_max_tokens = (
        args.openai_max_tokens or output_tokens_per_request or 128
    )

    cost_model_enabled = workload_profile is not None or any(
        (
            args.input_tokens_per_request,
            args.output_tokens_per_request,
            args.gpu_hourly_cost_usd,
            args.power_watts_per_gpu,
            args.electricity_cost_usd_per_kwh,
        )
    )
    llm_metrics_enabled = workload_profile is not None or any(
        (
            args.context_tokens_per_request,
            args.time_to_first_token_ms is not None,
            args.inter_token_latency_ms is not None,
            args.kv_cache_bytes_per_request,
            args.bytes_read_per_output_token,
            args.baseline_quality_score is not None,
        )
    )

    return CliOptions(
        config=BenchmarkConfig(
            mode=args.mode,
            server_url=args.server_url,
            model_name=args.model_name,
            input_name=args.input_name,
            input_shape=parse_shape(args.input_shape),
            warmup_requests=args.warmup_requests,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            request_rate_rps=args.request_rate_rps,
            propagate_trace_context=args.propagate_trace_context,
            output_dir=args.output_dir,
            seed=args.seed,
            openai_prompt=args.openai_prompt,
            openai_max_tokens=openai_max_tokens,
            openai_timeout_seconds=args.openai_timeout_seconds,
            openai_api_key_env=args.openai_api_key_env,
        ),
        coordination=coordination,
        cost_model_config=CostModelConfig(
            input_tokens_per_request=input_tokens_per_request,
            output_tokens_per_request=output_tokens_per_request,
            gpu_count=args.gpu_count,
            gpu_hourly_cost_usd=args.gpu_hourly_cost_usd,
            power_watts_per_gpu=args.power_watts_per_gpu,
            electricity_cost_usd_per_kwh=args.electricity_cost_usd_per_kwh,
        )
        if cost_model_enabled
        else None,
        llm_metrics_config=LlmMetricsConfig(
            context_tokens_per_request=context_tokens_per_request,
            batch_size=llm_batch_size,
            time_to_first_token_ms=time_to_first_token_ms,
            inter_token_latency_ms=inter_token_latency_ms,
            kv_cache_bytes_per_request=kv_cache_bytes_per_request,
            bytes_read_per_output_token=bytes_read_per_output_token,
            baseline_quality_score=args.baseline_quality_score,
            candidate_quality_score=args.candidate_quality_score,
        )
        if llm_metrics_enabled
        else None,
        workload_profile=workload_profile,
        export_prometheus=args.prometheus,
        telemetry_prometheus_path=args.telemetry_prometheus,
        telemetry_baseline_prometheus_path=args.telemetry_baseline_prometheus,
        telemetry_urls=telemetry_urls,
        telemetry_timeout_seconds=args.telemetry_timeout_seconds,
        telemetry_sample_interval_seconds=args.telemetry_sample_interval_seconds,
        telemetry_api_key_env=args.telemetry_api_key_env,
        max_server_failure_rate=args.max_server_failure_rate,
        max_server_queue_fraction=args.max_server_queue_fraction,
        request_path_metrics=request_path_metrics,
        service_restart_metric=args.service_restart_metric,
        min_service_restarts=args.min_service_restarts,
        max_service_restarts=args.max_service_restarts,
        batch_invariance_probes=args.batch_invariance_probes,
        batch_output_atol=args.batch_output_atol,
        batch_output_rtol=args.batch_output_rtol,
        baseline_path=args.baseline,
        max_p95_regression_pct=args.max_p95_regression_pct,
        max_success_rate_drop=args.max_success_rate_drop,
        max_client_attempt_amplification=args.max_client_attempt_amplification,
        fail_on_regression=args.fail_on_regression,
        fail_on_batch_variance=args.fail_on_batch_variance,
        fail_on_telemetry_gate=args.fail_on_telemetry_gate,
        fail_on_trace_context_gap=args.fail_on_trace_context_gap,
        fail_on_retry_gate=args.fail_on_retry_gate,
        fail_on_request_path_gap=args.fail_on_request_path_gap,
        fail_on_service_lifecycle_gap=args.fail_on_service_lifecycle_gap,
    )


def main() -> None:
    options = parse_args()
    config = options.config
    telemetry_client = (
        build_http_telemetry_client(
            options.telemetry_urls,
            timeout_seconds=options.telemetry_timeout_seconds,
            api_key_env=options.telemetry_api_key_env,
        )
        if options.telemetry_urls
        else None
    )
    client = build_client(config)
    metrics = run_benchmark(
        client,
        config,
        telemetry_client=telemetry_client,
        telemetry_sample_interval_seconds=options.telemetry_sample_interval_seconds,
        max_server_failure_rate=options.max_server_failure_rate,
        max_server_queue_fraction=options.max_server_queue_fraction,
        max_client_attempt_amplification=(
            options.max_client_attempt_amplification
        ),
        request_path_metrics=(
            options.request_path_metrics if options.telemetry_urls else None
        ),
        service_restart_metric=(
            options.service_restart_metric if options.telemetry_urls else None
        ),
        min_service_restarts=(
            options.min_service_restarts if options.telemetry_urls else None
        ),
        max_service_restarts=(
            options.max_service_restarts if options.telemetry_urls else None
        ),
        coordination=options.coordination,
    )
    if options.workload_profile:
        metrics["workload_profile"] = asdict(options.workload_profile)
    if options.cost_model_config:
        metrics["cost_model"] = build_cost_model(metrics, options.cost_model_config)
    if options.llm_metrics_config:
        metrics["llm_metrics"] = build_llm_metrics(
            metrics,
            options.llm_metrics_config,
            options.cost_model_config,
        )
    if options.telemetry_prometheus_path:
        metrics = attach_telemetry_summary(
            metrics,
            options.telemetry_prometheus_path,
            telemetry_baseline_prometheus_path=(
                options.telemetry_baseline_prometheus_path
            ),
            max_server_failure_rate=options.max_server_failure_rate,
            max_server_queue_fraction=options.max_server_queue_fraction,
            request_path_metrics=options.request_path_metrics,
            service_restart_metric=options.service_restart_metric,
            min_service_restarts=options.min_service_restarts,
            max_service_restarts=options.max_service_restarts,
        )
    if options.batch_invariance_probes:
        metrics["batch_invariance"] = run_batch_invariance_probe(
            client,
            probe_count=options.batch_invariance_probes,
            concurrency=config.concurrency,
            retries=config.retries,
            seed=config.seed,
            absolute_tolerance=options.batch_output_atol,
            relative_tolerance=options.batch_output_rtol,
        )
    if options.fail_on_trace_context_gap:
        metrics["trace_context_gate"] = build_trace_context_gate(metrics)
    metrics_path = save_metrics(metrics, config.output_dir)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    if options.export_prometheus:
        prometheus_path = save_prometheus_metrics(metrics, metrics_path)
        print(f"Saved Prometheus metrics to {prometheus_path}")

    exit_code = 0
    if options.baseline_path:
        baseline = load_metrics(options.baseline_path)
        regression_report = build_regression_report(
            baseline,
            metrics,
            max_p95_regression_pct=options.max_p95_regression_pct,
            max_success_rate_drop=options.max_success_rate_drop,
        )
        report_path = save_regression_report(regression_report, metrics_path)
        print(json.dumps(regression_report, indent=2))
        print(f"Saved comparison report to {report_path}")
        if options.fail_on_regression and regression_report["regression"]:
            exit_code = 2

    batch_invariance = metrics.get("batch_invariance")
    if (
        options.fail_on_batch_variance
        and isinstance(batch_invariance, dict)
        and not batch_invariance.get("passed", False)
        and exit_code == 0
    ):
        exit_code = 3

    telemetry_gate = metrics.get("telemetry_gate")
    if (
        options.fail_on_telemetry_gate
        and isinstance(telemetry_gate, dict)
        and not telemetry_gate.get("passed", False)
        and exit_code == 0
    ):
        exit_code = 4

    trace_context_gate = metrics.get("trace_context_gate")
    if (
        options.fail_on_trace_context_gap
        and isinstance(trace_context_gate, dict)
        and not trace_context_gate.get("passed", False)
        and exit_code == 0
    ):
        exit_code = 5

    retry_gate = metrics.get("retry_gate")
    if (
        options.fail_on_retry_gate
        and isinstance(retry_gate, dict)
        and not retry_gate.get("passed", False)
        and exit_code == 0
    ):
        exit_code = 6

    request_path_gate = metrics.get("request_path_gate")
    if (
        options.fail_on_request_path_gap
        and (
            not isinstance(request_path_gate, dict)
            or not request_path_gate.get("passed", False)
        )
        and exit_code == 0
    ):
        exit_code = 7

    service_lifecycle_gate = metrics.get("service_lifecycle_gate")
    if (
        options.fail_on_service_lifecycle_gap
        and (
            not isinstance(service_lifecycle_gate, dict)
            or not service_lifecycle_gate.get("passed", False)
        )
        and exit_code == 0
    ):
        exit_code = 8

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
