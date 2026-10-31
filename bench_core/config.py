"""Configuration, workload profiles and result types."""

from __future__ import annotations

import re
import secrets
import urllib.request
from dataclasses import dataclass
from typing import Any


GPU_UTILIZATION_METRICS = {"DCGM_FI_DEV_GPU_UTIL", "dcgm_gpu_utilization"}


GPU_MEMORY_COPY_METRICS = {"DCGM_FI_DEV_MEM_COPY_UTIL", "dcgm_gpu_memory_copy_utilization"}


GPU_MEMORY_USED_METRICS = {
    "DCGM_FI_DEV_FB_USED",
    "dcgm_gpu_memory_used_mib",
    "dcgm_gpu_memory_used_bytes",
}


TRITON_REQUEST_SUCCESS_METRICS = {"nv_inference_request_success", "nv_inference_count"}


TRITON_REQUEST_FAILURE_METRICS = {"nv_inference_request_failure"}


TRITON_REQUEST_DURATION_METRICS = {"nv_inference_request_duration_us"}


TRITON_QUEUE_DURATION_METRICS = {"nv_inference_queue_duration_us"}


TRITON_COMPUTE_INFER_DURATION_METRICS = {"nv_inference_compute_infer_duration_us"}


MAX_TELEMETRY_RESPONSE_BYTES = 10 * 1024 * 1024


MAX_TELEMETRY_ENDPOINTS = 16


PROMETHEUS_METRIC_NAME_PATTERN = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


GPU_GAUGE_METRIC_ALIASES = {
    **{name: "utilization_pct" for name in GPU_UTILIZATION_METRICS},
    **{name: "memory_copy_utilization_pct" for name in GPU_MEMORY_COPY_METRICS},
    **{name: "memory_used_mib" for name in GPU_MEMORY_USED_METRICS},
}


TRITON_COUNTER_METRIC_ALIASES = {
    **{name: "request_success" for name in TRITON_REQUEST_SUCCESS_METRICS},
    **{name: "request_failure" for name in TRITON_REQUEST_FAILURE_METRICS},
    **{name: "request_duration_us" for name in TRITON_REQUEST_DURATION_METRICS},
    **{name: "queue_duration_us" for name in TRITON_QUEUE_DURATION_METRICS},
    **{
        name: "compute_infer_duration_us"
        for name in TRITON_COMPUTE_INFER_DURATION_METRICS
    },
}


def normalize_openai_completions_url(server_url: str) -> str:
    """Return the OpenAI-compatible streaming completions endpoint."""
    normalized = server_url.rstrip("/")
    if normalized.endswith("/v1/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/completions"
    return f"{normalized}/v1/completions"


def sanitize_server_url(server_url: str) -> str:
    """Remove credentials, query parameters, and fragments from persisted URLs."""
    has_scheme = "://" in server_url
    parsed = urllib.parse.urlsplit(server_url if has_scheme else f"//{server_url}")
    if parsed.hostname is None:
        return server_url.split("?", 1)[0].split("#", 1)[0].rsplit("@", 1)[-1]
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    sanitized = urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, "", "")
    )
    return sanitized if has_scheme else sanitized.removeprefix("//")


def _nonzero_random_hex(byte_count: int) -> str:
    while True:
        value = secrets.token_hex(byte_count)
        if int(value, 16) != 0:
            return value


def build_traceparent() -> str:
    """Create a sampled W3C Trace Context header without retaining identifiers."""
    trace_id = _nonzero_random_hex(16)
    parent_id = _nonzero_random_hex(8)
    return f"00-{trace_id}-{parent_id}-01"


TRACEPARENT_VERSION_00_PATTERN = re.compile(
    r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)


def classify_response_traceparent(
    request_traceparent: str,
    response_traceparent: str | None,
) -> str:
    """Classify response trace continuation without retaining either identifier."""
    if response_traceparent is None:
        return "missing"
    request_match = TRACEPARENT_VERSION_00_PATTERN.fullmatch(request_traceparent)
    response_match = TRACEPARENT_VERSION_00_PATTERN.fullmatch(
        response_traceparent.strip()
    )
    if request_match is None or response_match is None:
        return "invalid"
    if response_match.group(1) == "0" * 32 or response_match.group(2) == "0" * 16:
        return "invalid"
    if response_match.group(2) == request_match.group(2):
        return "invalid"
    if response_match.group(1) != request_match.group(1):
        return "mismatched"
    return "matched"


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str = "mock"
    server_url: str = "localhost:8000"
    model_name: str = "resnet50_trt_fp16"
    input_name: str = "input"
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    warmup_requests: int = 0
    num_requests: int = 200
    concurrency: int = 10
    retries: int = 2
    retry_backoff_seconds: float = 0.0
    request_rate_rps: float = 0.0
    propagate_trace_context: bool = False
    output_dir: str = "benchmark_results"
    seed: int = 7
    openai_prompt: str = "Return a short deterministic benchmark response."
    openai_max_tokens: int = 128
    openai_timeout_seconds: float = 60.0
    openai_api_key_env: str = ""


@dataclass(frozen=True)
class CoordinatedClientConfig:
    """Process-local inputs supplied by the single-host coordinator."""

    run_id: str
    client_index: int
    client_count: int
    planned_start_unix_ns: int


@dataclass(frozen=True)
class CostModelConfig:
    input_tokens_per_request: int = 0
    output_tokens_per_request: int = 0
    gpu_count: int = 1
    gpu_hourly_cost_usd: float = 0.0
    power_watts_per_gpu: float = 0.0
    electricity_cost_usd_per_kwh: float = 0.0


@dataclass(frozen=True)
class LlmMetricsConfig:
    context_tokens_per_request: int = 0
    batch_size: int = 1
    time_to_first_token_ms: float | None = None
    inter_token_latency_ms: float | None = None
    kv_cache_bytes_per_request: int = 0
    bytes_read_per_output_token: int = 0
    baseline_quality_score: float | None = None
    candidate_quality_score: float | None = None


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    description: str
    context_tokens_per_request: int
    output_tokens_per_request: int
    batch_size: int
    time_to_first_token_ms: float
    inter_token_latency_ms: float
    kv_cache_bytes_per_request: int
    bytes_read_per_output_token: int


WORKLOAD_PROFILES = {
    "interactive": WorkloadProfile(
        name="interactive",
        description="Short-context, latency-sensitive assistant traffic.",
        context_tokens_per_request=512,
        output_tokens_per_request=128,
        batch_size=1,
        time_to_first_token_ms=80.0,
        inter_token_latency_ms=25.0,
        kv_cache_bytes_per_request=4 * 1024 * 1024,
        bytes_read_per_output_token=4096,
    ),
    "long-context": WorkloadProfile(
        name="long-context",
        description="Long-context requests that stress prefill and KV capacity.",
        context_tokens_per_request=32768,
        output_tokens_per_request=256,
        batch_size=4,
        time_to_first_token_ms=450.0,
        inter_token_latency_ms=35.0,
        kv_cache_bytes_per_request=256 * 1024 * 1024,
        bytes_read_per_output_token=16384,
    ),
    "throughput": WorkloadProfile(
        name="throughput",
        description="Higher-batch decode traffic for capacity qualification.",
        context_tokens_per_request=2048,
        output_tokens_per_request=512,
        batch_size=16,
        time_to_first_token_ms=180.0,
        inter_token_latency_ms=18.0,
        kv_cache_bytes_per_request=32 * 1024 * 1024,
        bytes_read_per_output_token=8192,
    ),
}


@dataclass(frozen=True)
class RequestPathMetricNames:
    """Operator-selected cumulative counters for three serving-path stages."""

    ingress: str
    backend: str
    success: str

    def by_stage(self) -> dict[str, str]:
        return {
            "ingress": self.ingress,
            "backend": self.backend,
            "success": self.success,
        }


@dataclass(frozen=True)
class CliOptions:
    config: BenchmarkConfig
    coordination: CoordinatedClientConfig | None = None
    workload_profile: WorkloadProfile | None = None
    cost_model_config: CostModelConfig | None = None
    llm_metrics_config: LlmMetricsConfig | None = None
    export_prometheus: bool = False
    telemetry_prometheus_path: str | None = None
    telemetry_baseline_prometheus_path: str | None = None
    telemetry_urls: tuple[str, ...] = ()
    telemetry_timeout_seconds: float = 10.0
    telemetry_sample_interval_seconds: float = 0.0
    telemetry_api_key_env: str = ""
    max_server_failure_rate: float | None = None
    max_server_queue_fraction: float | None = None
    request_path_metrics: RequestPathMetricNames | None = None
    service_restart_metric: str | None = None
    min_service_restarts: int | None = None
    max_service_restarts: int | None = None
    batch_invariance_probes: int = 0
    batch_output_atol: float = 0.0
    batch_output_rtol: float = 0.0
    baseline_path: str | None = None
    max_p95_regression_pct: float = 10.0
    max_success_rate_drop: float = 0.01
    max_client_attempt_amplification: float | None = None
    fail_on_regression: bool = False
    fail_on_batch_variance: bool = False
    fail_on_telemetry_gate: bool = False
    fail_on_trace_context_gap: bool = False
    fail_on_retry_gate: bool = False
    fail_on_request_path_gap: bool = False
    fail_on_service_lifecycle_gap: bool = False


@dataclass(frozen=True)
class StreamingInferenceObservation:
    time_to_first_token_ms: float
    inter_chunk_latency_ms: float
    observed_output_chunks: int
    reported_output_tokens: int | None
    output_bytes: int
    response_trace_context: str | None = None
    invalid_reported_output_tokens: bool = False


@dataclass(frozen=True)
class InferenceResult:
    ok: bool
    latency_ms: float
    error: str | None = None
    streaming: StreamingInferenceObservation | None = None
    request_started_at: float | None = None
    attempt_count: int = 1


@dataclass(frozen=True)
class OutputInferenceResult:
    sample_id: int
    ok: bool
    latency_ms: float
    output_observation: OutputObservation | None = None
    error: str | None = None


@dataclass(frozen=True)
class NumericOutput:
    """One numeric Triton output retained only for in-process comparison."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    values: tuple[bool | int | float | complex, ...]


@dataclass(frozen=True)
class OutputObservation:
    """Private output evidence; fingerprints and values are never serialized."""

    fingerprint: str
    numeric_outputs: tuple[NumericOutput, ...] | None = None


@dataclass(frozen=True)
class PrometheusSample:
    metric: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class GpuGaugeCapture:
    values: dict[str, list[float]]
    series_membership: dict[str, object]


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    if percentile_rank <= 0:
        return min(values)
    if percentile_rank >= 100:
        return max(values)

    sorted_values = sorted(values)
    index = round((percentile_rank / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


TELEMETRY_COUNTER_KEYS = {
    "request_success": "request_success_total",
    "request_failure": "request_failure_total",
    "request_duration_us": "request_duration_us_total",
    "queue_duration_us": "queue_duration_us_total",
    "compute_infer_duration_us": "compute_infer_duration_us_total",
}


def _number(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _latency(metrics: dict[str, Any], key: str) -> float:
    latency = metrics.get("latency_ms", {})
    if not isinstance(latency, dict):
        return 0.0
    value = latency.get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _nested_number(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _config_number(metrics: dict[str, Any], key: str) -> float:
    config = metrics.get("config", {})
    if isinstance(config, dict):
        value = config.get(key, metrics.get(key, 0))
    else:
        value = metrics.get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0
