from __future__ import annotations

import argparse
import json
import random
import statistics
import time
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


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str = "mock"
    server_url: str = "localhost:8000"
    model_name: str = "resnet50_trt_fp16"
    input_name: str = "input"
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    num_requests: int = 200
    concurrency: int = 10
    retries: int = 2
    output_dir: str = "benchmark_results"
    seed: int = 7


@dataclass(frozen=True)
class CliOptions:
    config: BenchmarkConfig
    export_prometheus: bool = False
    telemetry_prometheus_path: str | None = None
    baseline_path: str | None = None
    max_p95_regression_pct: float = 10.0
    max_success_rate_drop: float = 0.01
    fail_on_regression: bool = False


@dataclass(frozen=True)
class InferenceResult:
    ok: bool
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class PrometheusSample:
    metric: str
    labels: dict[str, str]
    value: float


class InferenceClient(Protocol):
    def infer(self) -> None:
        """Execute one inference request or raise an exception."""


class MockInferenceClient:
    """Dependency-free client used for CI, demos, and benchmark harness tests."""

    def __init__(
        self,
        seed: int = 7,
        min_latency_ms: float = 8.0,
        max_latency_ms: float = 35.0,
        failure_rate: float = 0.02,
    ) -> None:
        self.random = random.Random(seed)
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.failure_rate = failure_rate

    def infer(self) -> None:
        latency_ms = self.random.uniform(self.min_latency_ms, self.max_latency_ms)
        time.sleep(latency_ms / 1000)
        if self.random.random() < self.failure_rate:
            raise RuntimeError("synthetic inference failure")


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
        self.client = httpclient.InferenceServerClient(url=server_url)
        self.model_name = model_name
        self.input_name = input_name
        self.input_shape = input_shape

    def infer(self) -> None:
        input_data = self.np.random.rand(*self.input_shape).astype(self.np.float32)
        request_input = self.httpclient.InferInput(
            self.input_name,
            input_data.shape,
            self.np_to_triton_dtype(input_data.dtype),
        )
        request_input.set_data_from_numpy(input_data)
        self.client.infer(self.model_name, [request_input])


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
            client.infer()
            latency_ms = (time.perf_counter() - start) * 1000
            return InferenceResult(ok=True, latency_ms=latency_ms)
        except Exception as exc:  # noqa: BLE001 - benchmark harness records client failures.
            last_error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    return InferenceResult(ok=False, latency_ms=latency_ms, error=last_error)


def run_benchmark(client: InferenceClient, config: BenchmarkConfig) -> dict[str, object]:
    start = time.perf_counter()
    results: list[InferenceResult] = []

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [
            executor.submit(execute_with_retries, client, config.retries)
            for _ in range(config.num_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    duration_seconds = time.perf_counter() - start
    return summarize_results(results, duration_seconds, config)


def summarize_results(
    results: list[InferenceResult],
    duration_seconds: float,
    config: BenchmarkConfig,
) -> dict[str, object]:
    latencies = [result.latency_ms for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    successes = len(latencies)
    total = len(results)

    return {
        "mode": config.mode,
        "server_url": config.server_url if config.mode == "triton" else None,
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
        "config": asdict(config),
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
    raise ValueError(f"Unsupported mode: {config.mode}")


def parse_shape(raw_shape: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw_shape.split(",") if part.strip())


def parse_args() -> CliOptions:
    parser = argparse.ArgumentParser(description="Benchmark Triton-style inference workloads.")
    parser.add_argument("--mode", choices=["mock", "triton"], default="mock")
    parser.add_argument("--server-url", default="localhost:8000")
    parser.add_argument("--model-name", default="resnet50_trt_fp16")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=7)
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
    args = parser.parse_args()

    return CliOptions(
        config=BenchmarkConfig(
            mode=args.mode,
            server_url=args.server_url,
            model_name=args.model_name,
            input_name=args.input_name,
            input_shape=parse_shape(args.input_shape),
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            retries=args.retries,
            output_dir=args.output_dir,
            seed=args.seed,
        ),
        export_prometheus=args.prometheus,
        telemetry_prometheus_path=args.telemetry_prometheus,
        baseline_path=args.baseline,
        max_p95_regression_pct=args.max_p95_regression_pct,
        max_success_rate_drop=args.max_success_rate_drop,
        fail_on_regression=args.fail_on_regression,
    )


def main() -> None:
    options = parse_args()
    config = options.config
    client = build_client(config)
    metrics = run_benchmark(client, config)
    if options.telemetry_prometheus_path:
        metrics = attach_telemetry_summary(metrics, options.telemetry_prometheus_path)
    metrics_path = save_metrics(metrics, config.output_dir)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    if options.export_prometheus:
        prometheus_path = save_prometheus_metrics(metrics, metrics_path)
        print(f"Saved Prometheus metrics to {prometheus_path}")

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
            raise SystemExit(2)


if __name__ == "__main__":
    main()
