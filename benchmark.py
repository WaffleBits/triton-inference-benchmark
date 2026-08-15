from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import secrets
import statistics
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


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
    request_rate_rps: float = 0.0
    propagate_trace_context: bool = False
    output_dir: str = "benchmark_results"
    seed: int = 7
    openai_prompt: str = "Return a short deterministic benchmark response."
    openai_max_tokens: int = 128
    openai_timeout_seconds: float = 60.0
    openai_api_key_env: str = ""


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
class CliOptions:
    config: BenchmarkConfig
    workload_profile: WorkloadProfile | None = None
    cost_model_config: CostModelConfig | None = None
    llm_metrics_config: LlmMetricsConfig | None = None
    export_prometheus: bool = False
    telemetry_prometheus_path: str | None = None
    telemetry_baseline_prometheus_path: str | None = None
    telemetry_url: str | None = None
    telemetry_timeout_seconds: float = 10.0
    telemetry_sample_interval_seconds: float = 0.0
    telemetry_api_key_env: str = ""
    max_server_failure_rate: float | None = None
    max_server_queue_fraction: float | None = None
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


@dataclass(frozen=True)
class StreamingInferenceObservation:
    time_to_first_token_ms: float
    inter_chunk_latency_ms: float
    observed_output_chunks: int
    reported_output_tokens: int | None
    output_bytes: int
    response_trace_context: str | None = None


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


class TelemetrySnapshotClient(Protocol):
    def scrape(self) -> str:
        """Fetch one Prometheus text snapshot or raise an exception."""


class HttpPrometheusTelemetryClient:
    """Opt-in HTTP client for bounded Prometheus text snapshots."""

    def __init__(
        self,
        endpoint_url: str,
        timeout_seconds: float = 10.0,
        bearer_token: str | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(endpoint_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("telemetry endpoint must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("telemetry endpoint URL must not contain credentials")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("telemetry endpoint URL contains an invalid port") from exc
        if timeout_seconds <= 0:
            raise ValueError("telemetry timeout must be greater than zero")

        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds
        self.bearer_token = bearer_token

    def scrape(self) -> str:
        headers = {
            "Accept": "text/plain",
            "User-Agent": "triton-inference-benchmark/1.0",
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = urllib.request.Request(
            self.endpoint_url,
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read(MAX_TELEMETRY_RESPONSE_BYTES + 1)
        if len(payload) > MAX_TELEMETRY_RESPONSE_BYTES:
            raise ValueError(
                "telemetry response exceeded "
                f"{MAX_TELEMETRY_RESPONSE_BYTES} byte limit"
            )
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("telemetry response must be UTF-8 Prometheus text") from exc


def build_http_telemetry_client(
    endpoint_url: str,
    timeout_seconds: float,
    api_key_env: str = "",
) -> HttpPrometheusTelemetryClient:
    bearer_token: str | None = None
    if api_key_env:
        bearer_token = os.environ.get(api_key_env)
        if not bearer_token:
            raise ValueError(
                f"telemetry API key environment variable is missing or empty: {api_key_env}"
            )
    return HttpPrometheusTelemetryClient(
        endpoint_url,
        timeout_seconds=timeout_seconds,
        bearer_token=bearer_token,
    )


class InferenceClient(Protocol):
    def infer(self) -> StreamingInferenceObservation | None:
        """Execute one inference request or raise an exception."""


class OutputInferenceClient(Protocol):
    def infer_output(self, sample_id: int) -> str | OutputObservation:
        """Execute a deterministic input and return private output evidence."""


def capture_triton_outputs(result: Any) -> OutputObservation:
    """Capture exact and numeric Triton outputs without preparing serializable data."""
    response = result.get_response()

    if isinstance(response, dict):
        output_names = [
            output["name"]
            for output in response.get("outputs", [])
            if isinstance(output, dict) and "name" in output
        ]
    else:
        output_names = [
            output.name
            for output in getattr(response, "outputs", [])
            if getattr(output, "name", None)
        ]

    if not output_names:
        raise RuntimeError("Triton response did not include output metadata")

    hasher = hashlib.sha256()
    numeric_outputs: list[NumericOutput] = []
    all_outputs_numeric = True
    for output_name in sorted(output_names):
        output = result.as_numpy(output_name)
        if output is None:
            raise RuntimeError(f"Triton response did not include output: {output_name}")

        metadata = json.dumps(
            {
                "dtype": str(output.dtype),
                "name": output_name,
                "shape": list(output.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload = (
            json.dumps(output.tolist(), sort_keys=True, default=str).encode()
            if output.dtype.hasobject
            else output.tobytes(order="C")
        )
        hasher.update(len(metadata).to_bytes(8, byteorder="big"))
        hasher.update(metadata)
        hasher.update(len(payload).to_bytes(8, byteorder="big"))
        hasher.update(payload)

        dtype_kind = getattr(output.dtype, "kind", None)
        if dtype_kind not in {"b", "i", "u", "f", "c"}:
            all_outputs_numeric = False
            continue
        try:
            flattened_values = output.ravel(order="C").tolist()
        except (AttributeError, TypeError, ValueError):
            all_outputs_numeric = False
            continue
        if not isinstance(flattened_values, list):
            flattened_values = [flattened_values]
        if not all(
            isinstance(value, (bool, int, float, complex))
            for value in flattened_values
        ):
            all_outputs_numeric = False
            continue
        numeric_outputs.append(
            NumericOutput(
                name=output_name,
                dtype=str(output.dtype),
                shape=tuple(int(dimension) for dimension in output.shape),
                values=tuple(flattened_values),
            )
        )

    return OutputObservation(
        fingerprint=hasher.hexdigest(),
        numeric_outputs=(
            tuple(numeric_outputs)
            if all_outputs_numeric and len(numeric_outputs) == len(output_names)
            else None
        ),
    )


def fingerprint_triton_outputs(result: Any) -> str:
    return capture_triton_outputs(result).fingerprint


class MockInferenceClient:
    """Dependency-free client used for CI, demos, and benchmark harness tests."""

    def __init__(
        self,
        seed: int = 7,
        min_latency_ms: float = 8.0,
        max_latency_ms: float = 35.0,
        failure_rate: float = 0.02,
    ) -> None:
        self.seed = seed
        self.random = random.Random(seed)
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.failure_rate = failure_rate

    def infer(self) -> None:
        latency_ms = self.random.uniform(self.min_latency_ms, self.max_latency_ms)
        time.sleep(latency_ms / 1000)
        if self.random.random() < self.failure_rate:
            raise RuntimeError("synthetic inference failure")

    def infer_output(self, sample_id: int) -> OutputObservation:
        sample_random = random.Random(self.seed + sample_id)
        latency_ms = sample_random.uniform(self.min_latency_ms, self.max_latency_ms)
        time.sleep(latency_ms / 1000)
        payload = f"mock-output:{self.seed}:{sample_id}".encode()
        return OutputObservation(
            fingerprint=hashlib.sha256(payload).hexdigest(),
            numeric_outputs=(
                NumericOutput(
                    name="mock_scores",
                    dtype="float64",
                    shape=(2,),
                    values=(float(sample_id), float((self.seed + sample_id) % 17) / 17),
                ),
            ),
        )


class TritonHttpInferenceClient:
    """HTTP client for a live Triton-compatible inference server endpoint."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        input_name: str,
        input_shape: tuple[int, ...],
        propagate_trace_context: bool = False,
    ) -> None:
        try:
            import numpy as np
            import tritonclient.http as httpclient
            from tritonclient.utils import np_to_triton_dtype
        except ImportError as exc:
            raise RuntimeError(
                "Live Triton mode requires numpy and tritonclient. "
                "Install them with: pip install -r requirements.txt"
            ) from exc

        self.np = np
        self.httpclient = httpclient
        self.np_to_triton_dtype = np_to_triton_dtype
        self.server_url = server_url
        self.thread_local = threading.local()
        self.model_name = model_name
        self.input_name = input_name
        self.input_shape = input_shape
        self.propagate_trace_context = propagate_trace_context

    def infer(self) -> None:
        input_data = self.np.random.rand(*self.input_shape).astype(self.np.float32)
        self._infer(input_data)

    def infer_output(self, sample_id: int) -> OutputObservation:
        random_generator = self.np.random.default_rng(sample_id)
        input_data = random_generator.random(self.input_shape).astype(self.np.float32)
        result = self._infer(input_data)
        return capture_triton_outputs(result)

    def _infer(self, input_data: Any) -> Any:
        request_input = self.httpclient.InferInput(
            self.input_name,
            input_data.shape,
            self.np_to_triton_dtype(input_data.dtype),
        )
        request_input.set_data_from_numpy(input_data)
        client = getattr(self.thread_local, "client", None)
        if client is None:
            client = self.httpclient.InferenceServerClient(url=self.server_url)
            self.thread_local.client = client
        headers = (
            {"traceparent": build_traceparent()}
            if self.propagate_trace_context
            else None
        )
        return client.infer(self.model_name, [request_input], headers=headers)


class OpenAICompatibleStreamingClient:
    """Streaming client for OpenAI-compatible text completion endpoints."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        prompt: str,
        max_tokens: int,
        timeout_seconds: float,
        api_key: str | None = None,
        propagate_trace_context: bool = False,
    ) -> None:
        self.endpoint_url = normalize_openai_completions_url(server_url)
        self.model_name = model_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.propagate_trace_context = propagate_trace_context

    def infer(self) -> StreamingInferenceObservation:
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "triton-inference-benchmark/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_traceparent = (
            build_traceparent() if self.propagate_trace_context else None
        )
        if request_traceparent is not None:
            headers["traceparent"] = request_traceparent

        request = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        first_chunk_at: float | None = None
        previous_chunk_at: float | None = None
        inter_chunk_gaps_ms: list[float] = []
        observed_output_chunks = 0
        reported_output_tokens: int | None = None
        output_bytes = 0

        response_trace_context: str | None = None
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            if request_traceparent is not None:
                response_trace_context = classify_response_traceparent(
                    request_traceparent,
                    response.headers.get("traceparent"),
                )
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                raw_event = line.removeprefix("data:").strip()
                if raw_event == "[DONE]":
                    break
                if not raw_event:
                    continue
                event = json.loads(raw_event)
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(
                    usage.get("completion_tokens"), int
                ):
                    reported_output_tokens = usage["completion_tokens"]

                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                text = choice.get("text") if isinstance(choice, dict) else None
                if not isinstance(text, str) or not text:
                    continue

                now = time.perf_counter()
                if first_chunk_at is None:
                    first_chunk_at = now
                elif previous_chunk_at is not None:
                    inter_chunk_gaps_ms.append((now - previous_chunk_at) * 1000)
                previous_chunk_at = now
                observed_output_chunks += 1
                output_bytes += len(text.encode("utf-8"))

        if first_chunk_at is None:
            raise RuntimeError("stream completed without a non-empty text event")

        return StreamingInferenceObservation(
            time_to_first_token_ms=(first_chunk_at - started) * 1000,
            inter_chunk_latency_ms=(
                statistics.fmean(inter_chunk_gaps_ms) if inter_chunk_gaps_ms else 0.0
            ),
            observed_output_chunks=observed_output_chunks,
            reported_output_tokens=reported_output_tokens,
            output_bytes=output_bytes,
            response_trace_context=response_trace_context,
        )


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


def execute_with_retries(client: InferenceClient, retries: int) -> InferenceResult:
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
            future = executor.submit(execute_with_retries, client, retries)
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


def _summarize_retry_attempts(
    results: list[InferenceResult],
    configured_retries: int,
) -> dict[str, object]:
    """Summarize client calls without claiming that an endpoint received them."""
    logical_requests = len(results)
    attempt_counts = [max(1, int(result.attempt_count)) for result in results]
    client_attempts = sum(attempt_counts)
    retry_attempts = client_attempts - logical_requests
    return {
        "configured_retries_per_request": configured_retries,
        "logical_requests": logical_requests,
        "client_attempts": client_attempts,
        "retry_attempts": retry_attempts,
        "retried_requests": sum(count > 1 for count in attempt_counts),
        "recovered_requests": sum(
            result.ok and count > 1
            for result, count in zip(results, attempt_counts)
        ),
        "exhausted_requests": sum(not result.ok for result in results),
        "client_attempt_amplification": round(
            client_attempts / logical_requests,
            4,
        )
        if logical_requests
        else 0,
        "scope": (
            "Counts calls made by this harness to InferenceClient.infer. A client "
            "attempt does not prove endpoint, router, model-server, or accelerator receipt."
        ),
    }


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
) -> dict[str, object]:
    if telemetry_sample_interval_seconds < 0:
        raise ValueError("telemetry sample interval must be zero or greater")
    if telemetry_sample_interval_seconds and telemetry_client is None:
        raise ValueError("telemetry sampling requires a telemetry client")

    warmup_results: list[InferenceResult] = []
    warmup_duration_seconds = 0.0
    if config.warmup_requests:
        warmup_results, warmup_duration_seconds, _ = _run_request_phase(
            client,
            request_count=config.warmup_requests,
            concurrency=config.concurrency,
            retries=config.retries,
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

    try:
        results, duration_seconds, load_schedule = _run_request_phase(
            client,
            request_count=config.num_requests,
            concurrency=config.concurrency,
            retries=config.retries,
            phase_started=phase_started if sampler is not None else None,
            request_rate_rps=config.request_rate_rps,
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
    if warmup_results:
        summary["warmup"] = _summarize_request_phase(
            warmup_results,
            warmup_duration_seconds,
            config.retries,
        )
    if telemetry_before is not None and telemetry_after is not None:
        summary = attach_telemetry_snapshots(
            summary,
            telemetry_after,
            baseline_prometheus_text=telemetry_before,
            source="http_prometheus_snapshot",
            alignment="harness_bracketed_measured_phase",
            counter_window_source="paired_http_prometheus_scrapes",
            max_server_failure_rate=max_server_failure_rate,
            max_server_queue_fraction=max_server_queue_fraction,
        )
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


def summarize_results(
    results: list[InferenceResult],
    duration_seconds: float,
    config: BenchmarkConfig,
) -> dict[str, object]:
    latencies = [result.latency_ms for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    successes = len(latencies)
    total = len(results)
    persisted_config = asdict(config)
    prompt = str(persisted_config.pop("openai_prompt", ""))
    persisted_config["openai_prompt_bytes"] = len(prompt.encode("utf-8"))
    persisted_config["openai_prompt_sha256"] = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    sanitized_server_url = sanitize_server_url(config.server_url)
    persisted_config["server_url"] = sanitized_server_url

    summary: dict[str, object] = {
        "mode": config.mode,
        "server_url": (
            sanitized_server_url if config.mode in {"triton", "openai"} else None
        ),
        "model_name": config.model_name,
        "num_requests": total,
        "concurrency": config.concurrency,
        "duration_seconds": round(duration_seconds, 4),
        "successful_requests": successes,
        "failed_requests": len(failures),
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
        "retry": _summarize_retry_attempts(results, config.retries),
        "config": persisted_config,
    }

    if config.propagate_trace_context:
        summary["trace_context"] = {
            "propagation": "w3c_traceparent",
            "scope": "fresh context for each physical live HTTP request attempt",
            "identifiers_persisted": False,
            "server_acceptance": "not verified",
            "note": (
                "Header injection does not prove server span creation, export, "
                "sampling behavior, or clock synchronization."
            ),
        }

    streaming = [
        result.streaming
        for result in results
        if result.ok and result.streaming is not None
    ]
    if streaming:
        ttft_values = [item.time_to_first_token_ms for item in streaming]
        inter_chunk_values = [item.inter_chunk_latency_ms for item in streaming]
        reported_tokens = [
            item.reported_output_tokens
            for item in streaming
            if item.reported_output_tokens is not None
        ]

        def distribution(values: list[float]) -> dict[str, float]:
            return {
                "avg": round(statistics.fmean(values), 4),
                "p50": round(percentile(values, 50), 4),
                "p95": round(percentile(values, 95), 4),
                "p99": round(percentile(values, 99), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        complete_token_usage = len(reported_tokens) == len(streaming)
        total_reported_tokens = sum(reported_tokens)
        summary["streaming"] = {
            "request_count": len(streaming),
            "reported_token_request_count": len(reported_tokens),
            "missing_token_usage_requests": len(streaming) - len(reported_tokens),
            "reported_output_tokens": total_reported_tokens,
            "observed_output_chunks": sum(
                item.observed_output_chunks for item in streaming
            ),
            "output_bytes": sum(item.output_bytes for item in streaming),
            "output_tokens_per_second": round(
                total_reported_tokens / duration_seconds,
                4,
            )
            if complete_token_usage and duration_seconds > 0
            else None,
            "time_to_first_token_ms": distribution(ttft_values),
            "inter_chunk_latency_ms": distribution(inter_chunk_values),
            "claim_scope": {
                "ttft": "measured to first non-empty streamed text event",
                "inter_chunk": "mean gap between non-empty streamed text events per request",
                "tokens": "server-reported usage only",
                "chunks": "transport events; not treated as tokens",
            },
        }

        response_contexts = [
            item.response_trace_context
            for item in streaming
            if item.response_trace_context is not None
        ]
        if config.propagate_trace_context and config.mode == "openai":
            continuation_counts = {
                status: response_contexts.count(status)
                for status in ("matched", "missing", "invalid", "mismatched")
            }
            matched = continuation_counts["matched"]
            response_count = len(streaming)
            trace_context = summary.get("trace_context")
            assert isinstance(trace_context, dict)
            trace_context["server_acceptance"] = "response trace context classified"
            trace_context["response_continuation"] = {
                "request_count": response_count,
                "matched_responses": matched,
                "missing_responses": continuation_counts["missing"],
                "invalid_responses": continuation_counts["invalid"],
                "mismatched_responses": continuation_counts["mismatched"],
                "match_coverage": round(matched / response_count, 4)
                if response_count
                else 0.0,
                "complete": response_count > 0 and matched == response_count,
                "identifiers_persisted": False,
                "scope": (
                    "same trace ID observed in a valid response traceparent with a "
                    "different span ID"
                ),
                "note": (
                    "Response context does not prove server span creation/export, "
                    "collector delivery, sampling, clock synchronization, or "
                    "accelerator attribution."
                ),
            }

    return summary


def build_trace_context_gate(metrics: dict[str, object]) -> dict[str, object]:
    """Fail closed unless every measured OpenAI response continues its request trace."""
    reasons: list[str] = []
    failed_requests = metrics.get("failed_requests")
    if not isinstance(failed_requests, int) or failed_requests:
        reasons.append("one or more measured requests failed")

    trace_context = metrics.get("trace_context")
    continuation = (
        trace_context.get("response_continuation")
        if isinstance(trace_context, dict)
        else None
    )
    if not isinstance(continuation, dict):
        reasons.append("response trace-continuation evidence is unavailable")
        return {"passed": False, "failure_reasons": reasons}

    status_labels = (
        ("missing_responses", "missing"),
        ("invalid_responses", "invalid"),
        ("mismatched_responses", "mismatched"),
    )
    for key, label in status_labels:
        count = continuation.get(key)
        if not isinstance(count, int) or count:
            reasons.append(f"one or more response traceparent headers were {label}")
    if continuation.get("complete") is not True:
        reasons.append("response trace-continuation coverage was incomplete")

    return {
        "passed": not reasons,
        "failure_reasons": reasons,
        "scope": "measured successful OpenAI-compatible HTTP responses",
    }


def _complete_streaming_output_tokens(metrics: dict[str, object]) -> int | None:
    streaming = metrics.get("streaming")
    if not isinstance(streaming, dict):
        return None
    request_count = streaming.get("request_count")
    reported_count = streaming.get("reported_token_request_count")
    output_tokens = streaming.get("reported_output_tokens")
    if (
        isinstance(request_count, int)
        and request_count > 0
        and reported_count == request_count
        and isinstance(output_tokens, int)
        and output_tokens >= 0
    ):
        return output_tokens
    return None


def build_cost_model(
    metrics: dict[str, object],
    config: CostModelConfig,
) -> dict[str, object]:
    successful_requests = int(_number(dict(metrics), "successful_requests"))
    duration_seconds = _number(dict(metrics), "duration_seconds")
    duration_hours = duration_seconds / 3600

    input_tokens = successful_requests * config.input_tokens_per_request
    measured_output_tokens = _complete_streaming_output_tokens(metrics)
    output_tokens = (
        measured_output_tokens
        if measured_output_tokens is not None
        else successful_requests * config.output_tokens_per_request
    )
    output_tokens_per_request = (
        round(output_tokens / successful_requests, 4)
        if successful_requests
        else 0.0
    )
    output_tokens_source = (
        "server-reported streaming usage"
        if measured_output_tokens is not None
        else "configured estimate"
    )
    total_tokens = input_tokens + output_tokens

    accelerator_cost_usd = (
        config.gpu_count * config.gpu_hourly_cost_usd * duration_hours
    )
    energy_kwh = (
        config.gpu_count * config.power_watts_per_gpu * duration_hours / 1000
    )
    electricity_cost_usd = energy_kwh * config.electricity_cost_usd_per_kwh
    total_cost_usd = accelerator_cost_usd + electricity_cost_usd

    def per_million(cost: float, units: int) -> float | None:
        if units <= 0:
            return None
        return round(cost * 1_000_000 / units, 6)

    def per_second(units: int) -> float:
        if duration_seconds <= 0:
            return 0.0
        return round(units / duration_seconds, 4)

    return {
        "estimate": True,
        "workload": {
            "input_tokens_per_request": config.input_tokens_per_request,
            "output_tokens_per_request": output_tokens_per_request,
            "output_tokens_source": output_tokens_source,
            "successful_input_tokens": input_tokens,
            "successful_output_tokens": output_tokens,
            "successful_total_tokens": total_tokens,
            "output_tokens_per_second": per_second(output_tokens),
            "total_tokens_per_second": per_second(total_tokens),
        },
        "capacity": {
            "gpu_count": config.gpu_count,
            "successful_requests_per_gpu_hour": round(
                successful_requests / duration_hours / config.gpu_count,
                4,
            )
            if duration_hours > 0
            else 0.0,
            "output_tokens_per_gpu_second": round(
                output_tokens / duration_seconds / config.gpu_count,
                4,
            )
            if duration_seconds > 0
            else 0.0,
        },
        "cost": {
            "gpu_hourly_cost_usd": config.gpu_hourly_cost_usd,
            "power_watts_per_gpu": config.power_watts_per_gpu,
            "electricity_cost_usd_per_kwh": config.electricity_cost_usd_per_kwh,
            "accelerator_cost_usd": round(accelerator_cost_usd, 6),
            "energy_kwh": round(energy_kwh, 6),
            "electricity_cost_usd": round(electricity_cost_usd, 6),
            "total_estimated_cost_usd": round(total_cost_usd, 6),
            "cost_per_million_requests_usd": per_million(
                total_cost_usd,
                successful_requests,
            ),
            "cost_per_million_input_tokens_usd": per_million(
                total_cost_usd,
                input_tokens,
            ),
            "cost_per_million_output_tokens_usd": per_million(
                total_cost_usd,
                output_tokens,
            ),
            "cost_per_million_total_tokens_usd": per_million(
                total_cost_usd,
                total_tokens,
            ),
        },
        "assumptions": [
            "GPU capacity is reserved for the measured request-phase wall-clock duration; configured warmup is excluded.",
            "Token counts describe successful requests only.",
            "GPU hourly price and electricity are additive when both are configured.",
            "Network, storage, CPU, idle fleet, and engineering costs are excluded.",
        ],
    }


def build_llm_metrics(
    metrics: dict[str, object],
    config: LlmMetricsConfig,
    cost_config: CostModelConfig | None = None,
) -> dict[str, object]:
    successful_requests = int(_number(dict(metrics), "successful_requests"))
    duration_seconds = _number(dict(metrics), "duration_seconds")
    measured_output_tokens = _complete_streaming_output_tokens(metrics)
    output_tokens = (
        measured_output_tokens
        if measured_output_tokens is not None
        else successful_requests * cost_config.output_tokens_per_request
        if cost_config
        else 0
    )
    measured_ttft = None
    measured_inter_chunk = None
    streaming = metrics.get("streaming")
    if isinstance(streaming, dict):
        ttft = streaming.get("time_to_first_token_ms")
        inter_chunk = streaming.get("inter_chunk_latency_ms")
        if isinstance(ttft, dict) and isinstance(ttft.get("avg"), (int, float)):
            measured_ttft = float(ttft["avg"])
        if isinstance(inter_chunk, dict) and isinstance(
            inter_chunk.get("avg"), (int, float)
        ):
            measured_inter_chunk = float(inter_chunk["avg"])
    if measured_ttft is not None and measured_inter_chunk is not None:
        latency_source = (
            "measured TTFT and inter-chunk; caller-provided inter-token latency"
        )
    elif measured_ttft is not None:
        latency_source = "measured TTFT; caller-provided inter-token latency"
    elif measured_inter_chunk is not None:
        latency_source = "measured inter-chunk; caller-provided decode latency"
    else:
        latency_source = "caller-provided decode measurements"
    latency_metrics: dict[str, float] = {
        "time_to_first_token": (
            measured_ttft
            if measured_ttft is not None
            else config.time_to_first_token_ms
        ),
        "inter_token": config.inter_token_latency_ms,
    }
    if measured_inter_chunk is not None:
        latency_metrics["inter_chunk"] = measured_inter_chunk
    gpu_count = cost_config.gpu_count if cost_config else 1
    power_watts = cost_config.power_watts_per_gpu if cost_config else 0.0
    board_joules = gpu_count * power_watts * duration_seconds
    quality_delta = None
    quality_degradation_pct = None
    if (
        config.baseline_quality_score is not None
        and config.candidate_quality_score is not None
    ):
        quality_delta = round(
            config.baseline_quality_score - config.candidate_quality_score,
            8,
        )
        if config.baseline_quality_score:
            quality_degradation_pct = round(
                quality_delta / config.baseline_quality_score * 100,
                6,
            )

    return {
        "context_tokens_per_request": config.context_tokens_per_request,
        "batch_size": config.batch_size,
        "latency_ms": latency_metrics,
        "throughput": {
            "output_tokens_per_second": round(output_tokens / duration_seconds, 4)
            if duration_seconds > 0
            else 0.0,
            "requests_per_gpu_hour": round(
                successful_requests / duration_seconds * 3600 / gpu_count,
                4,
            )
            if duration_seconds > 0
            else 0.0,
        },
        "memory": {
            "kv_cache_bytes_per_request": config.kv_cache_bytes_per_request,
            "bytes_read_per_output_token": config.bytes_read_per_output_token,
        },
        "energy": {
            "estimated_board_joules": round(board_joules, 6),
            "estimated_joules_per_output_token": round(
                board_joules / output_tokens,
                6,
            )
            if output_tokens
            else None,
        },
        "quality": {
            "baseline_score": config.baseline_quality_score,
            "candidate_score": config.candidate_quality_score,
            "absolute_degradation": quality_delta,
            "relative_degradation_percent": quality_degradation_pct,
        },
        "claim_scope": {
            "latency_source": latency_source,
            "traffic": "logical bytes supplied by the benchmark operator",
            "energy": "board-power estimate without idle-power subtraction",
            "quality": "metric semantics are defined by the supplied evaluation",
        },
    }


def save_metrics(metrics: dict[str, object], output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_path = output_path / f"benchmark_{timestamp}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


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


TELEMETRY_COUNTER_KEYS = {
    "request_success": "request_success_total",
    "request_failure": "request_failure_total",
    "request_duration_us": "request_duration_us_total",
    "queue_duration_us": "queue_duration_us_total",
    "compute_infer_duration_us": "compute_infer_duration_us_total",
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
) -> dict[str, object]:
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
    elif max_server_failure_rate is not None or max_server_queue_fraction is not None:
        raise ValueError("telemetry thresholds require paired pre-run and post-run snapshots")
    return enriched_metrics


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


def load_metrics(path: str | Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")
    return raw


def _percent_change(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else 100.0
    return round(((candidate - baseline) / baseline) * 100, 4)


def build_regression_report(
    baseline: dict[str, object],
    candidate: dict[str, object],
    max_p95_regression_pct: float = 10.0,
    max_success_rate_drop: float = 0.01,
) -> dict[str, object]:
    baseline_metrics: dict[str, Any] = dict(baseline)
    candidate_metrics: dict[str, Any] = dict(candidate)

    baseline_p95 = _latency(baseline_metrics, "p95")
    candidate_p95 = _latency(candidate_metrics, "p95")
    baseline_success = _number(baseline_metrics, "success_rate")
    candidate_success = _number(candidate_metrics, "success_rate")
    baseline_throughput = _number(baseline_metrics, "throughput_rps")
    candidate_throughput = _number(candidate_metrics, "throughput_rps")

    p95_delta_pct = _percent_change(baseline_p95, candidate_p95)
    success_rate_delta = round(candidate_success - baseline_success, 4)
    throughput_delta_pct = _percent_change(baseline_throughput, candidate_throughput)

    regression_reasons: list[str] = []
    if p95_delta_pct > max_p95_regression_pct:
        regression_reasons.append(
            f"p95 latency increased {p95_delta_pct}% above {max_p95_regression_pct}% threshold"
        )
    if success_rate_delta < -max_success_rate_drop:
        regression_reasons.append(
            f"success rate changed {success_rate_delta} below -{max_success_rate_drop} threshold"
        )

    return {
        "baseline": {
            "p95_latency_ms": baseline_p95,
            "success_rate": baseline_success,
            "throughput_rps": baseline_throughput,
        },
        "candidate": {
            "p95_latency_ms": candidate_p95,
            "success_rate": candidate_success,
            "throughput_rps": candidate_throughput,
        },
        "changes": {
            "p95_latency_delta_pct": p95_delta_pct,
            "success_rate_delta": success_rate_delta,
            "throughput_delta_pct": throughput_delta_pct,
        },
        "thresholds": {
            "max_p95_regression_pct": max_p95_regression_pct,
            "max_success_rate_drop": max_success_rate_drop,
        },
        "regression": bool(regression_reasons),
        "regression_reasons": regression_reasons,
    }


def save_regression_report(report: dict[str, object], metrics_path: Path) -> Path:
    report_path = metrics_path.with_name(f"{metrics_path.stem}_comparison.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


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
    parser = argparse.ArgumentParser(description="Benchmark Triton-style inference workloads.")
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
        help=(
            "Optional HTTP(S) Prometheus endpoint scraped after warmup, immediately "
            "before measured requests, and immediately after they complete."
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
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be zero or greater")
    if args.request_rate_rps < 0:
        parser.error("--request-rate-rps must be zero or greater")
    if args.retries < 0:
        parser.error("--retries must be zero or greater")
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
    if args.telemetry_url and (
        args.telemetry_baseline_prometheus or args.telemetry_prometheus
    ):
        parser.error(
            "--telemetry-url is mutually exclusive with telemetry snapshot files"
        )
    if args.telemetry_api_key_env and not args.telemetry_url:
        parser.error("--telemetry-api-key-env requires --telemetry-url")
    if args.telemetry_sample_interval_seconds < 0:
        parser.error("--telemetry-sample-interval-seconds must be zero or greater")
    if args.telemetry_sample_interval_seconds and not args.telemetry_url:
        parser.error("--telemetry-sample-interval-seconds requires --telemetry-url")
    if args.telemetry_timeout_seconds <= 0:
        parser.error("--telemetry-timeout-seconds must be greater than zero")
    if args.telemetry_url:
        try:
            HttpPrometheusTelemetryClient(
                args.telemetry_url,
                timeout_seconds=args.telemetry_timeout_seconds,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if any(value is not None for value in telemetry_thresholds) and not (
        paired_snapshot_files or args.telemetry_url
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
            request_rate_rps=args.request_rate_rps,
            propagate_trace_context=args.propagate_trace_context,
            output_dir=args.output_dir,
            seed=args.seed,
            openai_prompt=args.openai_prompt,
            openai_max_tokens=openai_max_tokens,
            openai_timeout_seconds=args.openai_timeout_seconds,
            openai_api_key_env=args.openai_api_key_env,
        ),
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
        telemetry_url=args.telemetry_url,
        telemetry_timeout_seconds=args.telemetry_timeout_seconds,
        telemetry_sample_interval_seconds=args.telemetry_sample_interval_seconds,
        telemetry_api_key_env=args.telemetry_api_key_env,
        max_server_failure_rate=args.max_server_failure_rate,
        max_server_queue_fraction=args.max_server_queue_fraction,
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
    )


def main() -> None:
    options = parse_args()
    config = options.config
    telemetry_client = (
        build_http_telemetry_client(
            options.telemetry_url,
            timeout_seconds=options.telemetry_timeout_seconds,
            api_key_env=options.telemetry_api_key_env,
        )
        if options.telemetry_url
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

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
