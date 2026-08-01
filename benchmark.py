from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
    batch_invariance_probes: int = 0
    baseline_path: str | None = None
    max_p95_regression_pct: float = 10.0
    max_success_rate_drop: float = 0.01
    fail_on_regression: bool = False
    fail_on_batch_variance: bool = False


@dataclass(frozen=True)
class StreamingInferenceObservation:
    time_to_first_token_ms: float
    inter_chunk_latency_ms: float
    observed_output_chunks: int
    reported_output_tokens: int | None
    output_bytes: int


@dataclass(frozen=True)
class InferenceResult:
    ok: bool
    latency_ms: float
    error: str | None = None
    streaming: StreamingInferenceObservation | None = None


@dataclass(frozen=True)
class OutputInferenceResult:
    sample_id: int
    ok: bool
    latency_ms: float
    output_fingerprint: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class PrometheusSample:
    metric: str
    labels: dict[str, str]
    value: float


class InferenceClient(Protocol):
    def infer(self) -> StreamingInferenceObservation | None:
        """Execute one inference request or raise an exception."""


class OutputInferenceClient(Protocol):
    def infer_output(self, sample_id: int) -> str:
        """Execute a deterministic input and return an output fingerprint."""


def fingerprint_triton_outputs(result: Any) -> str:
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
    return hasher.hexdigest()


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

    def infer_output(self, sample_id: int) -> str:
        sample_random = random.Random(self.seed + sample_id)
        latency_ms = sample_random.uniform(self.min_latency_ms, self.max_latency_ms)
        time.sleep(latency_ms / 1000)
        payload = f"mock-output:{self.seed}:{sample_id}".encode()
        return hashlib.sha256(payload).hexdigest()


class TritonHttpInferenceClient:
    """HTTP client for a live Triton-compatible inference server endpoint."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        input_name: str,
        input_shape: tuple[int, ...],
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

    def infer(self) -> None:
        input_data = self.np.random.rand(*self.input_shape).astype(self.np.float32)
        self._infer(input_data)

    def infer_output(self, sample_id: int) -> str:
        random_generator = self.np.random.default_rng(sample_id)
        input_data = random_generator.random(self.input_shape).astype(self.np.float32)
        result = self._infer(input_data)
        return fingerprint_triton_outputs(result)

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
        return client.infer(self.model_name, [request_input])


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
    ) -> None:
        self.endpoint_url = normalize_openai_completions_url(server_url)
        self.model_name = model_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

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

        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
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

    for _ in range(retries + 1):
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
            )
        except Exception as exc:  # noqa: BLE001 - benchmark harness records client failures.
            last_error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    return InferenceResult(ok=False, latency_ms=latency_ms, error=last_error)


def execute_output_with_retries(
    client: OutputInferenceClient,
    sample_id: int,
    retries: int,
) -> OutputInferenceResult:
    start = time.perf_counter()
    last_error: str | None = None

    for _ in range(retries + 1):
        try:
            fingerprint = client.infer_output(sample_id)
            latency_ms = (time.perf_counter() - start) * 1000
            return OutputInferenceResult(
                sample_id=sample_id,
                ok=True,
                latency_ms=latency_ms,
                output_fingerprint=fingerprint,
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


def run_batch_invariance_probe(
    client: OutputInferenceClient,
    probe_count: int,
    concurrency: int,
    retries: int = 0,
    seed: int = 7,
) -> dict[str, object]:
    if probe_count <= 0:
        raise ValueError("probe_count must be greater than zero")
    if concurrency <= 1:
        raise ValueError("concurrency must be greater than one")

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
    compared_outputs = 0
    errors: list[dict[str, object]] = []

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
        if baseline.output_fingerprint == candidate.output_fingerprint:
            matched_outputs += 1
        else:
            mismatched_sample_ids.append(sample_id)

    failed_probes = len(errors)
    exact_match = (
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
        "mismatched_outputs": len(mismatched_sample_ids),
        "failed_probes": failed_probes,
        "noise_failures": noise_failures,
        "match_rate": round(matched_outputs / probe_count, 4),
        "exact_match": exact_match,
        "mismatched_sample_ids": mismatched_sample_ids,
        "errors": errors,
    }


def _run_request_phase(
    client: InferenceClient,
    request_count: int,
    concurrency: int,
    retries: int,
) -> tuple[list[InferenceResult], float]:
    start = time.perf_counter()
    results: list[InferenceResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(execute_with_retries, client, retries)
            for _ in range(request_count)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    duration_seconds = time.perf_counter() - start
    return results, duration_seconds


def _summarize_request_phase(
    results: list[InferenceResult],
    duration_seconds: float,
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
    }


def run_benchmark(client: InferenceClient, config: BenchmarkConfig) -> dict[str, object]:
    warmup_results: list[InferenceResult] = []
    warmup_duration_seconds = 0.0
    if config.warmup_requests:
        warmup_results, warmup_duration_seconds = _run_request_phase(
            client,
            request_count=config.warmup_requests,
            concurrency=config.concurrency,
            retries=config.retries,
        )

    results, duration_seconds = _run_request_phase(
        client,
        request_count=config.num_requests,
        concurrency=config.concurrency,
        retries=config.retries,
    )
    summary = summarize_results(results, duration_seconds, config)
    summary["measurement_scope"] = {
        "headline_phase": "measured requests",
        "warmup_excluded": bool(config.warmup_requests),
        "note": (
            "Warmup requests precondition the client and server path; they do not "
            "establish a process, model, or accelerator cold-start measurement."
        ),
    }
    if warmup_results:
        summary["warmup"] = _summarize_request_phase(
            warmup_results,
            warmup_duration_seconds,
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
        "config": persisted_config,
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

    return summary


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


def _stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "avg": round(statistics.fmean(values), 4),
        "max": round(max(values), 4),
    }


def _sum_values(values: list[float]) -> float:
    return round(sum(values), 4)


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
        },
        "notes": notes,
    }


def attach_telemetry_summary(
    metrics: dict[str, object],
    telemetry_prometheus_path: str | Path,
) -> dict[str, object]:
    path = Path(telemetry_prometheus_path)
    enriched_metrics = dict(metrics)
    model_name = str(metrics.get("model_name", "unknown"))
    enriched_metrics["telemetry"] = build_telemetry_summary(
        path.read_text(encoding="utf-8"),
        model_name=model_name,
        source=str(path),
    )
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
                "Outputs that changed between isolated and concurrent execution.",
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
                "Ratio of fixed inputs with identical isolated and concurrent outputs.",
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
            "Optional Prometheus text snapshot from Triton/DCGM scraped near the run; "
            "a correlated telemetry summary is attached to the JSON and .prom outputs."
        ),
    )
    parser.add_argument(
        "--batch-invariance-probes",
        type=int,
        default=0,
        help=(
            "Run this many fixed inputs in isolation and under concurrent noise traffic, "
            "then compare exact output fingerprints."
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
    if args.batch_invariance_probes < 0:
        parser.error("--batch-invariance-probes must be zero or greater")
    if args.batch_invariance_probes and args.concurrency <= 1:
        parser.error("--batch-invariance-probes requires --concurrency greater than one")
    if args.fail_on_batch_variance and not args.batch_invariance_probes:
        parser.error("--fail-on-batch-variance requires --batch-invariance-probes")
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
        batch_invariance_probes=args.batch_invariance_probes,
        baseline_path=args.baseline,
        max_p95_regression_pct=args.max_p95_regression_pct,
        max_success_rate_drop=args.max_success_rate_drop,
        fail_on_regression=args.fail_on_regression,
        fail_on_batch_variance=args.fail_on_batch_variance,
    )


def main() -> None:
    options = parse_args()
    config = options.config
    client = build_client(config)
    metrics = run_benchmark(client, config)
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
        metrics = attach_telemetry_summary(metrics, options.telemetry_prometheus_path)
    if options.batch_invariance_probes:
        metrics["batch_invariance"] = run_batch_invariance_probe(
            client,
            probe_count=options.batch_invariance_probes,
            concurrency=config.concurrency,
            retries=config.retries,
            seed=config.seed,
        )
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
        and not batch_invariance.get("exact_match", False)
        and exit_code == 0
    ):
        exit_code = 3

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
