import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local
from unittest.mock import patch

from benchmark import (
    BenchmarkConfig,
    CostModelConfig,
    InferenceResult,
    LlmMetricsConfig,
    MockInferenceClient,
    TritonHttpInferenceClient,
    build_cost_model,
    build_llm_metrics,
    build_regression_report,
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


if __name__ == "__main__":
    unittest.main()
