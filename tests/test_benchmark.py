import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, local
from unittest.mock import patch

from benchmark import (
    MAX_TELEMETRY_RESPONSE_BYTES,
    BenchmarkConfig,
    CostModelConfig,
    HttpPrometheusTelemetryClient,
    InferenceResult,
    LlmMetricsConfig,
    MockInferenceClient,
    TritonHttpInferenceClient,
    attach_telemetry_summary,
    build_http_telemetry_client,
    build_cost_model,
    build_gpu_gauge_window,
    build_llm_metrics,
    build_regression_report,
    build_telemetry_counter_window,
    build_telemetry_gate,
    build_telemetry_summary,
    fingerprint_triton_outputs,
    format_prometheus_metrics,
    parse_args,
    parse_prometheus_samples,
    percentile,
    run_batch_invariance_probe,
    run_benchmark,
    resolve_workload_profile,
    summarize_results,
)


class BenchmarkHarnessTest(unittest.TestCase):
    def test_workload_profiles_are_explicit_and_distinct(self) -> None:
        interactive = resolve_workload_profile("interactive")
        long_context = resolve_workload_profile("long-context")
        throughput = resolve_workload_profile("throughput")

        self.assertIsNotNone(interactive)
        self.assertIsNotNone(long_context)
        self.assertIsNotNone(throughput)
        self.assertLess(
            interactive.context_tokens_per_request,
            long_context.context_tokens_per_request,
        )
        self.assertLess(interactive.batch_size, throughput.batch_size)
        self.assertGreater(
            long_context.kv_cache_bytes_per_request,
            interactive.kv_cache_bytes_per_request,
        )

    def test_unknown_workload_profile_reports_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "interactive"):
            resolve_workload_profile("unknown")

    def test_triton_output_fingerprint_is_stable_across_metadata_order(self) -> None:
        class FakeDtype:
            hasobject = False

            def __str__(self) -> str:
                return "float32"

        class FakeOutput:
            dtype = FakeDtype()
            shape = (1, 2)

            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            def tobytes(self, order: str) -> bytes:
                self.assert_c_order(order)
                return self.payload

            @staticmethod
            def assert_c_order(order: str) -> None:
                if order != "C":
                    raise AssertionError(f"unexpected byte order: {order}")

        class FakeResult:
            def __init__(self, output_order: list[str]) -> None:
                self.output_order = output_order
                self.outputs = {
                    "scores": FakeOutput(b"\x01\x02"),
                    "tokens": FakeOutput(b"\x03\x04"),
                }

            def get_response(self) -> dict[str, object]:
                return {
                    "outputs": [{"name": name} for name in self.output_order],
                }

            def as_numpy(self, output_name: str) -> FakeOutput:
                return self.outputs[output_name]

        forward = fingerprint_triton_outputs(FakeResult(["scores", "tokens"]))
        reversed_order = fingerprint_triton_outputs(FakeResult(["tokens", "scores"]))
        changed = fingerprint_triton_outputs(FakeResult(["scores"]))

        self.assertEqual(forward, reversed_order)
        self.assertNotEqual(forward, changed)

    def test_triton_http_client_reuses_one_connection_per_worker(self) -> None:
        class FakeInputData:
            shape = (1,)
            dtype = "float32"

        class FakeInferInput:
            def __init__(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
                self.name = name
                self.shape = shape
                self.dtype = dtype

            def set_data_from_numpy(self, input_data: FakeInputData) -> None:
                self.input_data = input_data

        class FakeServerClient:
            def __init__(self, url: str) -> None:
                self.url = url

            def infer(self, model_name: str, inputs: list[FakeInferInput]) -> object:
                return self

        class FakeHttpClient:
            InferInput = FakeInferInput
            InferenceServerClient = FakeServerClient

        client = TritonHttpInferenceClient.__new__(TritonHttpInferenceClient)
        client.httpclient = FakeHttpClient
        client.np_to_triton_dtype = str
        client.server_url = "localhost:8000"
        client.thread_local = local()
        client.model_name = "model"
        client.input_name = "input"
        barrier = Barrier(2)

        def run_worker() -> tuple[int, int]:
            first = client._infer(FakeInputData())
            barrier.wait()
            second = client._infer(FakeInputData())
            return id(first), id(second)

        with ThreadPoolExecutor(max_workers=2) as executor:
            worker_results = list(executor.map(lambda _: run_worker(), range(2)))

        self.assertEqual(worker_results[0][0], worker_results[0][1])
        self.assertEqual(worker_results[1][0], worker_results[1][1])
        self.assertNotEqual(worker_results[0][0], worker_results[1][0])

    def test_percentile_handles_boundaries(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]

        self.assertEqual(percentile(values, 0), 10.0)
        self.assertEqual(percentile(values, 50), 30.0)
        self.assertEqual(percentile(values, 100), 50.0)

    def test_summarize_results_calculates_success_rate_and_latency(self) -> None:
        config = BenchmarkConfig(num_requests=4, concurrency=2)
        results = [
            InferenceResult(ok=True, latency_ms=10.0),
            InferenceResult(ok=True, latency_ms=20.0),
            InferenceResult(ok=True, latency_ms=30.0),
            InferenceResult(ok=False, latency_ms=5.0, error="boom"),
        ]

        metrics = summarize_results(results, duration_seconds=0.5, config=config)

        self.assertEqual(metrics["successful_requests"], 3)
        self.assertEqual(metrics["failed_requests"], 1)
        self.assertEqual(metrics["success_rate"], 0.75)
        self.assertEqual(metrics["throughput_rps"], 6.0)
        self.assertEqual(metrics["latency_ms"]["p50"], 20.0)

    def test_mock_benchmark_runs_without_triton_dependencies(self) -> None:
        config = BenchmarkConfig(
            mode="mock",
            num_requests=12,
            concurrency=3,
            retries=1,
            seed=11,
        )

        metrics = run_benchmark(MockInferenceClient(seed=11, failure_rate=0), config)

        self.assertEqual(metrics["num_requests"], 12)
        self.assertEqual(metrics["failed_requests"], 0)
        self.assertGreater(metrics["throughput_rps"], 0)

    def test_http_telemetry_brackets_only_the_measured_request_phase(self) -> None:
        events: list[str] = []

        class RecordingClient:
            def infer(self) -> None:
                events.append("infer")

        class RecordingTelemetryClient:
            def __init__(self) -> None:
                self.scrape_count = 0

            def scrape(self) -> str:
                events.append("scrape")
                self.scrape_count += 1
                successful = 100 if self.scrape_count == 1 else 103
                request_duration = 1000 if self.scrape_count == 1 else 4000
                queue_duration = 100 if self.scrape_count == 1 else 250
                compute_duration = 800 if self.scrape_count == 1 else 3300
                return (
                    f'nv_inference_request_success{{model="review-model"}} {successful}\n'
                    'nv_inference_request_failure{model="review-model"} 0\n'
                    f'nv_inference_request_duration_us{{model="review-model"}} {request_duration}\n'
                    f'nv_inference_queue_duration_us{{model="review-model"}} {queue_duration}\n'
                    f'nv_inference_compute_infer_duration_us{{model="review-model"}} {compute_duration}\n'
                )

        metrics = run_benchmark(
            RecordingClient(),
            BenchmarkConfig(
                model_name="review-model",
                warmup_requests=2,
                num_requests=3,
                concurrency=1,
                retries=0,
            ),
            telemetry_client=RecordingTelemetryClient(),
            max_server_failure_rate=0.01,
            max_server_queue_fraction=0.10,
        )

        self.assertEqual(
            events,
            ["infer", "infer", "scrape", "infer", "infer", "infer", "scrape"],
        )
        self.assertEqual(
            metrics["telemetry_window"]["alignment"],
            "harness_bracketed_measured_phase",
        )
        self.assertEqual(metrics["telemetry_window"]["derived"]["request_total"], 3)
        self.assertTrue(metrics["telemetry_gate"]["passed"])

    def test_http_telemetry_samples_gpu_gauges_during_measured_phase(self) -> None:
        class SlowClient:
            def infer(self) -> None:
                time.sleep(0.04)

        class SamplingTelemetryClient:
            def __init__(self) -> None:
                self.scrape_count = 0

            def scrape(self) -> str:
                self.scrape_count += 1
                value = self.scrape_count * 10
                return (
                    f'DCGM_FI_DEV_GPU_UTIL{{gpu="0"}} {value}\n'
                    f'DCGM_FI_DEV_MEM_COPY_UTIL{{gpu="0"}} {value / 2}\n'
                    f'DCGM_FI_DEV_FB_USED{{gpu="0"}} {1024 + value}\n'
                    f'nv_inference_request_success{{model="review-model"}} {100 + value}\n'
                    'nv_inference_request_failure{model="review-model"} 0\n'
                    f'nv_inference_request_duration_us{{model="review-model"}} {1000 + value}\n'
                    f'nv_inference_queue_duration_us{{model="review-model"}} {100 + value}\n'
                    "nv_inference_compute_infer_duration_us"
                    f'{{model="review-model"}} {800 + value}\n'
                )

        telemetry_client = SamplingTelemetryClient()
        metrics = run_benchmark(
            SlowClient(),
            BenchmarkConfig(
                model_name="review-model",
                num_requests=4,
                concurrency=2,
                retries=0,
            ),
            telemetry_client=telemetry_client,
            telemetry_sample_interval_seconds=0.005,
        )

        window = metrics["telemetry_gauge_window"]
        self.assertGreaterEqual(window["in_window_scrape_count"], 2)
        self.assertEqual(
            window["scrape_count"],
            window["in_window_scrape_count"] + 2,
        )
        self.assertEqual(window["alignment"], "harness_bracketed_measured_phase")
        self.assertGreater(window["gpu"]["utilization_pct"]["sample_count"], 2)
        self.assertIn("p95", window["gpu"]["utilization_pct"])
        serialized = json.dumps(metrics)
        self.assertNotIn("DCGM_FI_DEV_GPU_UTIL", serialized)
        self.assertNotIn("nv_inference_request_success", serialized)

    def test_in_window_telemetry_scrape_failure_aborts_qualification(self) -> None:
        class SlowClient:
            def infer(self) -> None:
                time.sleep(0.04)

        class FailingTelemetryClient:
            def __init__(self) -> None:
                self.scrape_count = 0

            def scrape(self) -> str:
                self.scrape_count += 1
                if self.scrape_count > 1:
                    raise RuntimeError("fixture scrape failed")
                return 'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 20\n'

        with self.assertRaisesRegex(RuntimeError, "in-window telemetry scrape failed"):
            run_benchmark(
                SlowClient(),
                BenchmarkConfig(num_requests=2, concurrency=1, retries=0),
                telemetry_client=FailingTelemetryClient(),
                telemetry_sample_interval_seconds=0.005,
            )

    def test_http_telemetry_client_sends_only_explicit_authentication(self) -> None:
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                return None

            def read(self, size: int) -> bytes:
                self.read_size = size
                return b'metric_total 1\n'

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        with (
            patch.dict(os.environ, {"AMBIENT_API_KEY": "must-not-be-sent"}, clear=True),
            patch("benchmark.urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            unauthenticated = build_http_telemetry_client(
                "https://metrics.example.test/metrics",
                timeout_seconds=3.5,
                api_key_env="",
            )
            self.assertEqual(unauthenticated.scrape(), "metric_total 1\n")

        self.assertIsNone(requests[0][0].get_header("Authorization"))
        self.assertEqual(requests[0][1], 3.5)

        with (
            patch.dict(os.environ, {"TELEMETRY_TOKEN": "explicit-secret"}, clear=True),
            patch("benchmark.urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            authenticated = build_http_telemetry_client(
                "https://metrics.example.test/metrics",
                timeout_seconds=2,
                api_key_env="TELEMETRY_TOKEN",
            )
            authenticated.scrape()

        self.assertEqual(
            requests[1][0].get_header("Authorization"),
            "Bearer explicit-secret",
        )

    def test_http_telemetry_client_rejects_missing_explicit_token(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TELEMETRY_TOKEN"):
                build_http_telemetry_client(
                    "https://metrics.example.test/metrics",
                    timeout_seconds=2,
                    api_key_env="TELEMETRY_TOKEN",
                )

    def test_http_telemetry_client_rejects_unsafe_urls_and_large_responses(self) -> None:
        for endpoint in (
            "file:///tmp/private.prom",
            "https://operator:secret@metrics.example.test/metrics",
            "metrics.example.test/metrics",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    HttpPrometheusTelemetryClient(endpoint)

        class LargeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                return None

            def read(self, size: int) -> bytes:
                return b"x" * size

        client = HttpPrometheusTelemetryClient("https://metrics.example.test/metrics")
        with patch("benchmark.urllib.request.urlopen", return_value=LargeResponse()):
            with self.assertRaisesRegex(ValueError, "exceeded"):
                client.scrape()

        self.assertGreater(MAX_TELEMETRY_RESPONSE_BYTES, 0)

    def test_bracketed_http_telemetry_artifact_omits_endpoint_token_and_raw_scrape(self) -> None:
        responses = iter(
            [
                b'nv_inference_request_success{model="review-model"} 10\n'
                b'nv_inference_request_failure{model="review-model"} 0\n',
                b'nv_inference_request_success{model="review-model"} 11\n'
                b'nv_inference_request_failure{model="review-model"} 0\n',
            ]
        )

        class FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                return None

            def read(self, size: int) -> bytes:
                return self.payload

        endpoint = "https://private.example.test/metrics?tenant=secret-tenant"
        token = "private-bearer-token"
        telemetry_client = HttpPrometheusTelemetryClient(endpoint, bearer_token=token)
        with patch(
            "benchmark.urllib.request.urlopen",
            side_effect=lambda request, timeout: FakeResponse(next(responses)),
        ):
            metrics = run_benchmark(
                MockInferenceClient(seed=11, failure_rate=0),
                BenchmarkConfig(
                    model_name="review-model",
                    num_requests=1,
                    concurrency=1,
                    retries=0,
                ),
                telemetry_client=telemetry_client,
                max_server_failure_rate=0.01,
            )

        serialized = json.dumps(metrics)
        self.assertNotIn(endpoint, serialized)
        self.assertNotIn("secret-tenant", serialized)
        self.assertNotIn(token, serialized)
        self.assertNotIn("nv_inference_request_success", serialized)
        self.assertEqual(metrics["telemetry"]["source"], "http_prometheus_snapshot")

    def test_warmup_requests_run_before_and_stay_out_of_measured_results(self) -> None:
        class CountingClient:
            def __init__(self) -> None:
                self.calls = 0

            def infer(self) -> None:
                self.calls += 1

        client = CountingClient()
        config = BenchmarkConfig(
            mode="mock",
            warmup_requests=2,
            num_requests=3,
            concurrency=1,
            retries=0,
        )

        metrics = run_benchmark(client, config)

        self.assertEqual(client.calls, 5)
        self.assertEqual(metrics["num_requests"], 3)
        self.assertEqual(metrics["successful_requests"], 3)
        self.assertEqual(metrics["warmup"]["request_count"], 2)
        self.assertEqual(metrics["warmup"]["successful_requests"], 2)
        self.assertTrue(metrics["measurement_scope"]["warmup_excluded"])

    def test_parses_warmup_request_count(self) -> None:
        with patch(
            "sys.argv",
            ["benchmark.py", "--warmup-requests", "7", "--num-requests", "3"],
        ):
            options = parse_args()

        self.assertEqual(options.config.warmup_requests, 7)
        self.assertEqual(options.config.num_requests, 3)

    def test_cost_model_normalizes_gpu_time_and_successful_tokens(self) -> None:
        cost_model = build_cost_model(
            {
                "successful_requests": 100,
                "duration_seconds": 3600.0,
            },
            CostModelConfig(
                input_tokens_per_request=1000,
                output_tokens_per_request=250,
                gpu_count=2,
                gpu_hourly_cost_usd=4.0,
                power_watts_per_gpu=500.0,
                electricity_cost_usd_per_kwh=0.10,
            ),
        )

        self.assertEqual(
            cost_model["workload"]["successful_total_tokens"],
            125000,
        )
        self.assertEqual(
            cost_model["capacity"]["successful_requests_per_gpu_hour"],
            50.0,
        )
        self.assertEqual(cost_model["cost"]["accelerator_cost_usd"], 8.0)
        self.assertEqual(cost_model["cost"]["energy_kwh"], 1.0)
        self.assertEqual(cost_model["cost"]["electricity_cost_usd"], 0.1)
        self.assertEqual(
            cost_model["cost"]["cost_per_million_total_tokens_usd"],
            64.8,
        )

    def test_cost_model_charges_wall_clock_when_requests_fail(self) -> None:
        cost_model = build_cost_model(
            {
                "successful_requests": 0,
                "duration_seconds": 1800.0,
            },
            CostModelConfig(
                input_tokens_per_request=100,
                output_tokens_per_request=25,
                gpu_count=1,
                gpu_hourly_cost_usd=10.0,
            ),
        )

        self.assertEqual(cost_model["cost"]["total_estimated_cost_usd"], 5.0)
        self.assertIsNone(cost_model["cost"]["cost_per_million_requests_usd"])
        self.assertIsNone(
            cost_model["cost"]["cost_per_million_total_tokens_usd"]
        )

    def test_llm_metrics_report_decode_memory_energy_and_quality(self) -> None:
        metrics = build_llm_metrics(
            {
                "successful_requests": 10,
                "duration_seconds": 20.0,
            },
            LlmMetricsConfig(
                context_tokens_per_request=2048,
                batch_size=4,
                time_to_first_token_ms=90.0,
                inter_token_latency_ms=11.5,
                kv_cache_bytes_per_request=117_440_512,
                bytes_read_per_output_token=1_554_916_608,
                baseline_quality_score=0.8,
                candidate_quality_score=0.78,
            ),
            CostModelConfig(
                output_tokens_per_request=128,
                gpu_count=1,
                power_watts_per_gpu=165.0,
            ),
        )

        self.assertEqual(metrics["throughput"]["output_tokens_per_second"], 64.0)
        self.assertEqual(metrics["throughput"]["requests_per_gpu_hour"], 1800.0)
        self.assertEqual(
            metrics["energy"]["estimated_joules_per_output_token"],
            2.578125,
        )
        self.assertEqual(metrics["quality"]["absolute_degradation"], 0.02)
        self.assertEqual(metrics["quality"]["relative_degradation_percent"], 2.5)

    def test_batch_invariance_probe_matches_deterministic_outputs(self) -> None:
        report = run_batch_invariance_probe(
            MockInferenceClient(seed=11, failure_rate=0),
            probe_count=6,
            concurrency=3,
            retries=1,
            seed=11,
        )

        self.assertTrue(report["exact_match"])
        self.assertEqual(report["matched_outputs"], 6)
        self.assertEqual(report["mismatched_outputs"], 0)
        self.assertEqual(report["match_rate"], 1.0)

    def test_batch_invariance_probe_detects_layout_sensitive_outputs(self) -> None:
        class LayoutSensitiveClient:
            def infer_output(self, sample_id: int) -> str:
                import threading

                phase = (
                    "isolated"
                    if threading.current_thread() is threading.main_thread()
                    else "concurrent"
                )
                return f"{sample_id}:{phase}"

        report = run_batch_invariance_probe(
            LayoutSensitiveClient(),
            probe_count=4,
            concurrency=2,
        )

        self.assertFalse(report["exact_match"])
        self.assertEqual(report["mismatched_outputs"], 4)
        self.assertEqual(report["mismatched_sample_ids"], [0, 1, 2, 3])

    def test_batch_invariance_probe_fails_when_noise_workload_fails(self) -> None:
        class NoiseFailingClient:
            def infer_output(self, sample_id: int) -> str:
                if sample_id >= 1_000_000:
                    raise RuntimeError("noise failed")
                return str(sample_id)

        report = run_batch_invariance_probe(
            NoiseFailingClient(),
            probe_count=4,
            concurrency=2,
        )

        self.assertFalse(report["exact_match"])
        self.assertEqual(report["matched_outputs"], 4)
        self.assertEqual(report["noise_failures"], 4)

    def test_prometheus_export_includes_core_metrics(self) -> None:
        config = BenchmarkConfig(num_requests=2, concurrency=2, retries=1)
        metrics = summarize_results(
            [
                InferenceResult(ok=True, latency_ms=10.0),
                InferenceResult(ok=False, latency_ms=20.0, error="boom"),
            ],
            duration_seconds=1.0,
            config=config,
        )

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_requests_total", output)
        self.assertIn('mode="mock"', output)
        self.assertIn('model="resnet50_trt_fp16"', output)
        self.assertIn('outcome="success"} 1', output)
        self.assertIn('outcome="failure"} 1', output)
        self.assertIn('quantile="0.95"} 10', output)

    def test_prometheus_export_keeps_warmup_metrics_separate(self) -> None:
        metrics = {
            "mode": "mock",
            "model_name": "review-model",
            "num_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "success_rate": 1.0,
            "duration_seconds": 0.5,
            "throughput_rps": 2.0,
            "latency_ms": {
                "avg": 10.0,
                "p50": 10.0,
                "p95": 10.0,
                "p99": 10.0,
                "min": 10.0,
                "max": 10.0,
            },
            "config": {"concurrency": 1, "retries": 0, "warmup_requests": 2},
            "warmup": {
                "request_count": 2,
                "successful_requests": 2,
                "failed_requests": 0,
                "success_rate": 1.0,
                "duration_seconds": 0.25,
                "throughput_rps": 8.0,
                "latency_ms": {
                    "avg": 4.0,
                    "p50": 4.0,
                    "p95": 5.0,
                    "p99": 5.0,
                    "min": 3.0,
                    "max": 5.0,
                },
            },
        }

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_warmup_requests_total", output)
        self.assertIn('phase="warmup",outcome="success"} 2', output)
        self.assertIn("triton_benchmark_warmup_duration_seconds", output)
        self.assertIn('phase="warmup",quantile="0.95"} 5', output)

    def test_prometheus_export_includes_batch_invariance_metrics(self) -> None:
        config = BenchmarkConfig(num_requests=1, concurrency=2)
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=1.0,
            config=config,
        )
        metrics["batch_invariance"] = {
            "probe_count": 4,
            "mismatched_outputs": 1,
            "failed_probes": 0,
            "noise_failures": 0,
            "match_rate": 0.75,
            "exact_match": False,
        }

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_batch_invariance_probes_total", output)
        self.assertIn("triton_benchmark_batch_invariance_mismatches_total", output)
        self.assertIn("triton_benchmark_batch_invariance_noise_failures_total", output)
        self.assertIn("triton_benchmark_batch_invariance_match_rate", output)
        self.assertIn(
            'triton_benchmark_batch_invariance_exact_match{mode="mock",'
            'model="resnet50_trt_fp16"} 0\n',
            output,
        )

    def test_prometheus_export_includes_cost_model_metrics(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=2.0,
            config=BenchmarkConfig(num_requests=1),
        )
        metrics["cost_model"] = build_cost_model(
            metrics,
            CostModelConfig(
                input_tokens_per_request=100,
                output_tokens_per_request=25,
                gpu_count=1,
                gpu_hourly_cost_usd=3.60,
            ),
        )

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_workload_tokens_total", output)
        self.assertIn('direction="output"} 25', output)
        self.assertIn("triton_benchmark_token_throughput_per_second", output)
        self.assertIn("triton_benchmark_estimated_cost_usd", output)
        self.assertIn("triton_benchmark_estimated_cost_per_million_usd", output)
        self.assertIn('unit="total_token"} 16', output)

    def test_prometheus_export_includes_llm_metrics(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=1.0,
            config=BenchmarkConfig(num_requests=1),
        )
        metrics["llm_metrics"] = build_llm_metrics(
            metrics,
            LlmMetricsConfig(
                context_tokens_per_request=2048,
                batch_size=4,
                time_to_first_token_ms=90.0,
                inter_token_latency_ms=11.5,
                kv_cache_bytes_per_request=117_440_512,
                bytes_read_per_output_token=1_554_916_608,
                baseline_quality_score=0.8,
                candidate_quality_score=0.78,
            ),
            CostModelConfig(
                output_tokens_per_request=128,
                power_watts_per_gpu=165.0,
            ),
        )

        output = format_prometheus_metrics(metrics)

        self.assertIn('phase="ttft"} 90', output)
        self.assertIn('phase="itl"} 11.5', output)
        self.assertIn("triton_benchmark_llm_kv_cache_bytes", output)
        self.assertIn("triton_benchmark_llm_bytes_read_per_output_token", output)
        self.assertIn("triton_benchmark_llm_joules_per_output_token", output)
        self.assertIn("triton_benchmark_llm_quality_degradation_percent", output)

    def test_batch_invariance_probe_requires_concurrent_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than one"):
            run_batch_invariance_probe(
                MockInferenceClient(seed=11, failure_rate=0),
                probe_count=2,
                concurrency=1,
            )

    def test_regression_report_flags_p95_increase(self) -> None:
        baseline = {
            "success_rate": 1.0,
            "throughput_rps": 100.0,
            "latency_ms": {"p95": 100.0},
        }
        candidate = {
            "success_rate": 0.99,
            "throughput_rps": 95.0,
            "latency_ms": {"p95": 125.0},
        }

        report = build_regression_report(
            baseline,
            candidate,
            max_p95_regression_pct=10.0,
            max_success_rate_drop=0.02,
        )

        self.assertTrue(report["regression"])
        self.assertEqual(report["changes"]["p95_latency_delta_pct"], 25.0)
        self.assertEqual(report["changes"]["success_rate_delta"], -0.01)

    def test_parse_prometheus_samples_handles_labels_and_escapes(self) -> None:
        samples = parse_prometheus_samples(
            'metric_total{model="resnet50_trt_fp16",note="line one\\nquote\\\""} 7\n'
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].metric, "metric_total")
        self.assertEqual(samples[0].labels["model"], "resnet50_trt_fp16")
        self.assertEqual(samples[0].labels["note"], 'line one\nquote"')
        self.assertEqual(samples[0].value, 7.0)

    def test_telemetry_summary_correlates_gpu_and_triton_samples(self) -> None:
        telemetry_text = """
        # HELP synthetic fixture
        DCGM_FI_DEV_GPU_UTIL{gpu="0"} 72
        DCGM_FI_DEV_GPU_UTIL{gpu="1"} 88
        DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 21
        DCGM_FI_DEV_FB_USED{gpu="0"} 9216
        nv_inference_request_success{model="resnet50_trt_fp16",version="1"} 100
        nv_inference_request_failure{model="resnet50_trt_fp16",version="1"} 2
        nv_inference_queue_duration_us{model="resnet50_trt_fp16",version="1"} 4300
        nv_inference_compute_infer_duration_us{model="other",version="1"} 999
        """

        summary = build_telemetry_summary(telemetry_text, model_name="resnet50_trt_fp16")

        self.assertEqual(summary["sample_count"], 8)
        self.assertEqual(summary["gpu"]["utilization_pct"]["avg"], 80.0)
        self.assertEqual(summary["gpu"]["utilization_pct"]["max"], 88.0)
        self.assertEqual(summary["gpu"]["memory_used_mib"]["max"], 9216.0)
        self.assertEqual(summary["triton"]["request_success_total"], 100.0)
        self.assertEqual(summary["triton"]["request_failure_total"], 2.0)
        self.assertEqual(summary["triton"]["queue_duration_us_total"], 4300.0)
        self.assertEqual(summary["triton"]["compute_infer_duration_us_total"], 0)

    def test_gpu_gauge_window_aggregates_repeated_bounded_scrapes(self) -> None:
        window = build_gpu_gauge_window(
            [
                """
                DCGM_FI_DEV_GPU_UTIL{gpu="0"} 20
                DCGM_FI_DEV_GPU_UTIL{gpu="1"} 40
                DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 10
                DCGM_FI_DEV_FB_USED{gpu="0"} 1024
                """,
                """
                DCGM_FI_DEV_GPU_UTIL{gpu="0"} 60
                DCGM_FI_DEV_GPU_UTIL{gpu="1"} 80
                DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 30
                dcgm_gpu_memory_used_bytes{gpu="0"} 2147483648
                """,
                """
                DCGM_FI_DEV_GPU_UTIL{gpu="0"} 100
                DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 50
                DCGM_FI_DEV_FB_USED{gpu="0"} 3072
                """,
            ],
            in_window_scrape_count=1,
            configured_interval_seconds=0.25,
        )

        self.assertEqual(window["scrape_count"], 3)
        self.assertEqual(window["in_window_scrape_count"], 1)
        self.assertEqual(window["gpu"]["utilization_pct"]["sample_count"], 5)
        self.assertEqual(window["gpu"]["utilization_pct"]["avg"], 60.0)
        self.assertEqual(window["gpu"]["utilization_pct"]["p50"], 60.0)
        self.assertEqual(window["gpu"]["utilization_pct"]["p95"], 100.0)
        self.assertEqual(window["gpu"]["memory_used_mib"]["max"], 3072.0)
        self.assertIn("not time-weighted", " ".join(window["notes"]))

    def test_telemetry_counter_window_derives_observed_window_rates(self) -> None:
        before = build_telemetry_summary(
            """
            nv_inference_request_success{model="review-model"} 400
            nv_inference_request_failure{model="review-model"} 0
            nv_inference_request_duration_us{model="review-model"} 9400000
            nv_inference_queue_duration_us{model="review-model"} 210000
            nv_inference_compute_infer_duration_us{model="review-model"} 7600000
            """,
            model_name="review-model",
        )
        after = build_telemetry_summary(
            """
            nv_inference_request_success{model="review-model"} 500
            nv_inference_request_failure{model="review-model"} 1
            nv_inference_request_duration_us{model="review-model"} 18400000
            nv_inference_queue_duration_us{model="review-model"} 710000
            nv_inference_compute_infer_duration_us{model="review-model"} 15100000
            nv_inference_request_success{model="other"} 9999
            """,
            model_name="review-model",
        )

        window = build_telemetry_counter_window(before, after)

        self.assertTrue(window["valid"])
        self.assertEqual(window["alignment"], "operator_supplied_unverified")
        self.assertEqual(window["deltas"]["request_success"], 100.0)
        self.assertEqual(window["deltas"]["request_failure"], 1.0)
        self.assertEqual(window["deltas"]["request_duration_us"], 9000000.0)
        self.assertEqual(window["deltas"]["queue_duration_us"], 500000.0)
        self.assertEqual(window["derived"]["request_total"], 101.0)
        self.assertEqual(window["derived"]["server_failure_rate"], 0.009901)
        self.assertEqual(window["derived"]["server_queue_fraction"], 0.055556)

    def test_telemetry_gate_reports_all_failed_thresholds(self) -> None:
        gate = build_telemetry_gate(
            {
                "deltas": {},
                "derived": {
                    "server_failure_rate": 0.02,
                    "server_queue_fraction": 0.15,
                },
            },
            max_server_failure_rate=0.01,
            max_server_queue_fraction=0.10,
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["server_failure_rate"]["passed"])
        self.assertFalse(gate["checks"]["server_queue_fraction"]["passed"])
        self.assertEqual(len(gate["failure_reasons"]), 2)

    def test_telemetry_gate_fails_closed_on_counter_reset(self) -> None:
        before = build_telemetry_summary(
            """
            nv_inference_request_success{model="review-model"} 100
            nv_inference_request_failure{model="review-model"} 2
            """,
            model_name="review-model",
        )
        after = build_telemetry_summary(
            """
            nv_inference_request_success{model="review-model"} 3
            nv_inference_request_failure{model="review-model"} 0
            """,
            model_name="review-model",
        )

        window = build_telemetry_counter_window(before, after)
        gate = build_telemetry_gate(window, max_server_failure_rate=0.01)

        self.assertFalse(window["valid"])
        self.assertEqual(
            window["counter_resets"],
            ["request_failure", "request_success"],
        )
        self.assertIsNone(window["derived"]["server_failure_rate"])
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["server_failure_rate"]["evaluable"])

    def test_telemetry_gate_fails_closed_when_duration_counter_is_missing(self) -> None:
        before = build_telemetry_summary(
            'nv_inference_queue_duration_us{model="review-model"} 100\n',
            model_name="review-model",
        )
        after = build_telemetry_summary(
            'nv_inference_queue_duration_us{model="review-model"} 200\n',
            model_name="review-model",
        )

        window = build_telemetry_counter_window(before, after)
        gate = build_telemetry_gate(window, max_server_queue_fraction=0.10)

        self.assertIn("request_duration_us", window["unavailable_counters"])
        self.assertIsNone(window["derived"]["server_queue_fraction"])
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["server_queue_fraction"]["evaluable"])

    def test_attached_telemetry_does_not_persist_paths_or_raw_scrapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before_path = Path(directory) / "private-before.prom"
            after_path = Path(directory) / "private-after.prom"
            before_path.write_text(
                'nv_inference_request_success{model="review-model"} 1\n'
                'nv_inference_request_failure{model="review-model"} 0\n',
                encoding="utf-8",
            )
            after_path.write_text(
                'nv_inference_request_success{model="review-model"} 2\n'
                'nv_inference_request_failure{model="review-model"} 0\n',
                encoding="utf-8",
            )

            enriched = attach_telemetry_summary(
                {"model_name": "review-model"},
                after_path,
                telemetry_baseline_prometheus_path=before_path,
                max_server_failure_rate=0.01,
            )
            serialized = json.dumps(enriched)

        self.assertNotIn(directory, serialized)
        self.assertNotIn("private-before.prom", serialized)
        self.assertNotIn("private-after.prom", serialized)
        self.assertNotIn("nv_inference_request_success", serialized)
        self.assertEqual(enriched["telemetry"]["source"], "prometheus_snapshot")
        self.assertTrue(enriched["telemetry_gate"]["passed"])

    def test_prometheus_export_includes_correlated_telemetry(self) -> None:
        config = BenchmarkConfig()
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=1.0,
            config=config,
        )
        metrics["telemetry"] = {
            "gpu": {
                "utilization_pct": {"avg": 80.0, "max": 88.0},
                "memory_copy_utilization_pct": {},
                "memory_used_mib": {"avg": 9216.0, "max": 9216.0},
            },
            "triton": {
                "request_success_total": 100.0,
                "request_failure_total": 2.0,
                "request_duration_us_total": 9000.0,
                "queue_duration_us_total": 4300.0,
                "compute_infer_duration_us_total": 5000.0,
            },
        }

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_gpu_utilization_percent", output)
        self.assertIn('stat="avg"} 80', output)
        self.assertIn("triton_benchmark_gpu_memory_used_mib", output)
        self.assertIn("triton_benchmark_server_request_success_total", output)
        self.assertIn("triton_benchmark_server_queue_duration_us_total", output)

    def test_prometheus_export_includes_telemetry_window_and_gate(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=1.0,
            config=BenchmarkConfig(num_requests=1),
        )
        metrics["telemetry_window"] = {
            "deltas": {
                "request_success": 100.0,
                "request_failure": 1.0,
                "request_duration_us": 9000000.0,
                "queue_duration_us": 500000.0,
                "compute_infer_duration_us": 7500000.0,
            },
            "derived": {
                "request_total": 101.0,
                "server_failure_rate": 0.009901,
                "server_queue_fraction": 0.055556,
            },
        }
        metrics["telemetry_gate"] = {
            "passed": True,
            "checks": {},
            "failure_reasons": [],
        }

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_server_counter_delta", output)
        self.assertIn('counter="request_success"} 100', output)
        self.assertIn("triton_benchmark_server_failure_rate", output)
        self.assertIn("triton_benchmark_server_queue_duration_fraction", output)
        self.assertIn("triton_benchmark_telemetry_gate_passed", output)

    def test_prometheus_export_includes_sampled_gpu_gauge_window(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=1.0,
            config=BenchmarkConfig(num_requests=1),
        )
        metrics["telemetry_gauge_window"] = {
            "scrape_count": 5,
            "in_window_scrape_count": 3,
            "gpu": {
                "utilization_pct": {
                    "sample_count": 10,
                    "avg": 70.0,
                    "min": 40.0,
                    "p50": 72.0,
                    "p95": 91.0,
                    "max": 95.0,
                },
                "memory_copy_utilization_pct": {},
                "memory_used_mib": {},
            },
        }

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_gpu_window_scrapes", output)
        self.assertIn('phase="measured",scope="in_window"} 3', output)
        self.assertIn("triton_benchmark_gpu_window_utilization_percent", output)
        self.assertIn('quantile="0.95"} 91', output)

    def test_parses_telemetry_gate_options(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--telemetry-baseline-prometheus",
                "before.prom",
                "--telemetry-prometheus",
                "after.prom",
                "--max-server-failure-rate",
                "0.01",
                "--max-server-queue-fraction",
                "0.20",
                "--fail-on-telemetry-gate",
            ],
        ):
            options = parse_args()

        self.assertEqual(options.telemetry_baseline_prometheus_path, "before.prom")
        self.assertEqual(options.max_server_failure_rate, 0.01)
        self.assertEqual(options.max_server_queue_fraction, 0.20)
        self.assertTrue(options.fail_on_telemetry_gate)

    def test_parses_bracketed_http_telemetry_options(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--telemetry-url",
                "https://metrics.example.test/metrics",
                "--telemetry-timeout-seconds",
                "3.5",
                "--telemetry-sample-interval-seconds",
                "0.25",
                "--telemetry-api-key-env",
                "TELEMETRY_TOKEN",
                "--max-server-failure-rate",
                "0.01",
                "--fail-on-telemetry-gate",
            ],
        ):
            options = parse_args()

        self.assertEqual(
            options.telemetry_url,
            "https://metrics.example.test/metrics",
        )
        self.assertEqual(options.telemetry_timeout_seconds, 3.5)
        self.assertEqual(options.telemetry_sample_interval_seconds, 0.25)
        self.assertEqual(options.telemetry_api_key_env, "TELEMETRY_TOKEN")
        self.assertEqual(options.max_server_failure_rate, 0.01)
        self.assertTrue(options.fail_on_telemetry_gate)

    def test_telemetry_url_is_mutually_exclusive_with_snapshot_files(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--telemetry-url",
                "https://metrics.example.test/metrics",
                "--telemetry-prometheus",
                "after.prom",
            ],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_telemetry_api_key_env_requires_telemetry_url(self) -> None:
        with patch(
            "sys.argv",
            ["benchmark.py", "--telemetry-api-key-env", "TELEMETRY_TOKEN"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_telemetry_sampling_interval_requires_telemetry_url(self) -> None:
        with patch(
            "sys.argv",
            ["benchmark.py", "--telemetry-sample-interval-seconds", "0.25"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_telemetry_threshold_requires_paired_snapshots_or_url(self) -> None:
        with patch(
            "sys.argv",
            ["benchmark.py", "--max-server-failure-rate", "0.01"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
