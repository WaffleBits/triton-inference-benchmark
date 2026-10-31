"""Inference clients: mock, Triton HTTP and OpenAI-compatible streaming."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import threading
import time
import urllib.request
from typing import Any, Protocol

from bench_core.config import (
    NumericOutput,
    OutputObservation,
    StreamingInferenceObservation,
    build_traceparent,
    classify_response_traceparent,
    normalize_openai_completions_url,
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
        invalid_reported_output_tokens = False
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
                if usage is not None:
                    if not isinstance(usage, dict):
                        invalid_reported_output_tokens = True
                        reported_output_tokens = None
                    elif "completion_tokens" in usage:
                        token_count = usage["completion_tokens"]
                        if token_count is None:
                            pass
                        elif type(token_count) is int and token_count >= 0:
                            if not invalid_reported_output_tokens:
                                reported_output_tokens = token_count
                        else:
                            invalid_reported_output_tokens = True
                            reported_output_tokens = None

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
            invalid_reported_output_tokens=invalid_reported_output_tokens,
        )
