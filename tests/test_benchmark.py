import unittest

from benchmark import (
    BenchmarkConfig,
    InferenceResult,
    MockInferenceClient,
    build_regression_report,
    build_telemetry_summary,
    format_prometheus_metrics,
    parse_prometheus_samples,
    percentile,
    run_benchmark,
    summarize_results,
)


class BenchmarkHarnessTest(unittest.TestCase):
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
