"""Latency, cost and LLM metric summaries."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from bench_core.config import (
    BenchmarkConfig,
    CostModelConfig,
    InferenceResult,
    LlmMetricsConfig,
    _number,
    percentile,
    sanitize_server_url,
)


def _summarize_retry_attempts(
    results: list[InferenceResult],
    configured_retries: int,
) -> dict[str, object]:
    """Summarize client calls without claiming that an endpoint received them."""
    logical_requests = len(results)
    attempt_counts = [max(1, int(result.attempt_count)) for result in results]
    client_attempts = sum(attempt_counts)
    retry_attempts = client_attempts - logical_requests
    recovered_latencies = [
        result.latency_ms
        for result, count in zip(results, attempt_counts)
        if result.ok and count > 1
    ]
    recovered_latency_summary = {
        "request_count": len(recovered_latencies),
        "avg": round(statistics.fmean(recovered_latencies), 4)
        if recovered_latencies
        else None,
        "p50": round(percentile(recovered_latencies, 50), 4)
        if recovered_latencies
        else None,
        "p95": round(percentile(recovered_latencies, 95), 4)
        if recovered_latencies
        else None,
        "p99": round(percentile(recovered_latencies, 99), 4)
        if recovered_latencies
        else None,
        "min": round(min(recovered_latencies), 4) if recovered_latencies else None,
        "max": round(max(recovered_latencies), 4) if recovered_latencies else None,
        "scope": (
            "client-observed end-to-end latency for measured logical requests that "
            "succeeded after retry"
        ),
        "note": (
            "Includes failed-attempt and retry time observed by the client; it is not "
            "service MTTR or proof of process recovery."
        ),
    }
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
        "recovered_request_latency_ms": recovered_latency_summary,
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
            and not item.invalid_reported_output_tokens
        ]
        invalid_token_usage_requests = sum(
            item.invalid_reported_output_tokens for item in streaming
        )
        missing_token_usage_requests = sum(
            item.reported_output_tokens is None
            and not item.invalid_reported_output_tokens
            for item in streaming
        )

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
            "missing_token_usage_requests": missing_token_usage_requests,
            "invalid_token_usage_requests": invalid_token_usage_requests,
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
