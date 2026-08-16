import json
import os
import re
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
    NumericOutput,
    OutputObservation,
    RequestPathMetricNames,
    StreamingInferenceObservation,
    TritonHttpInferenceClient,
    attach_telemetry_summary,
    build_http_telemetry_client,
    build_retry_gate,
    build_constant_rate_submission_offsets,
    build_cost_model,
    build_trace_context_gate,
    build_traceparent,
    build_gpu_gauge_window,
    build_llm_metrics,
    build_regression_report,
    build_request_path_accounting,
    build_request_path_gate,
    build_telemetry_counter_window,
    build_telemetry_gate,
    build_telemetry_summary,
    capture_triton_outputs,
    classify_response_traceparent,
    execute_with_retries,
    fingerprint_triton_outputs,
    format_prometheus_metrics,
    main,
    parse_args,
    parse_prometheus_samples,
    percentile,
    run_batch_invariance_probe,
    run_benchmark,
    resolve_workload_profile,
    summarize_results,
)


class BenchmarkHarnessTest(unittest.TestCase):
    def test_build_traceparent_returns_unique_w3c_sampled_contexts(self) -> None:
        traceparents = {build_traceparent() for _ in range(32)}

        self.assertEqual(len(traceparents), 32)
        for traceparent in traceparents:
            self.assertRegex(
                traceparent,
                r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$",
            )

    def test_classifies_response_trace_continuation_without_retaining_ids(self) -> None:
        request = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

        self.assertEqual(
            classify_response_traceparent(
                request,
                "00-4bf92f3577b34da6a3ce929d0e0e4736-1111111111111111-01",
            ),
            "matched",
        )
        self.assertEqual(classify_response_traceparent(request, None), "missing")
        self.assertEqual(
            classify_response_traceparent(request, "not-a-traceparent"),
            "invalid",
        )
        self.assertEqual(
            classify_response_traceparent(
                request,
                "00-4BF92F3577B34DA6A3CE929D0E0E4736-1111111111111111-01",
            ),
            "invalid",
        )
        self.assertEqual(
            classify_response_traceparent(
                request,
                "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-1111111111111111-01",
            ),
            "mismatched",
        )
        self.assertEqual(classify_response_traceparent(request, request), "invalid")

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

    def test_triton_output_capture_retains_numeric_values_only_in_observation(self) -> None:
        class FakeDtype:
            hasobject = False
            kind = "f"

            def __str__(self) -> str:
                return "float32"

        class FakeFlattened:
            @staticmethod
            def tolist() -> list[float]:
                return [1.25, 2.5]

        class FakeOutput:
            dtype = FakeDtype()
            shape = (1, 2)

            @staticmethod
            def tobytes(order: str) -> bytes:
                if order != "C":
                    raise AssertionError(order)
                return b"private-tensor-bytes"

            @staticmethod
            def ravel(order: str) -> FakeFlattened:
                if order != "C":
                    raise AssertionError(order)
                return FakeFlattened()

        class FakeResult:
            @staticmethod
            def get_response() -> dict[str, object]:
                return {"outputs": [{"name": "scores"}]}

            @staticmethod
            def as_numpy(output_name: str) -> FakeOutput:
                if output_name != "scores":
                    raise AssertionError(output_name)
                return FakeOutput()

        observation = capture_triton_outputs(FakeResult())

        self.assertEqual(len(observation.fingerprint), 64)
        self.assertEqual(
            observation.numeric_outputs,
            (NumericOutput("scores", "float32", (1, 2), (1.25, 2.5)),),
        )

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
            headers: list[object] = []

            def __init__(self, url: str) -> None:
                self.url = url

            def infer(
                self,
                model_name: str,
                inputs: list[FakeInferInput],
                headers=None,
            ) -> object:
                self.headers.append(headers)
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
        client.propagate_trace_context = True
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
        self.assertEqual(len(FakeServerClient.headers), 4)
        traceparents = {
            headers["traceparent"]
            for headers in FakeServerClient.headers
            if headers is not None
        }
        self.assertEqual(len(traceparents), 4)
        for traceparent in traceparents:
            self.assertRegex(traceparent, r"^00-[0-9a-f]{32}-[0-9a-f]{16}-01$")

    def test_triton_http_client_does_not_add_trace_context_by_default(self) -> None:
        class FakeInputData:
            shape = (1,)
            dtype = "float32"

        class FakeInferInput:
            def __init__(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
                return None

            def set_data_from_numpy(self, input_data: FakeInputData) -> None:
                return None

        class FakeServerClient:
            headers = "not-called"

            def __init__(self, url: str) -> None:
                return None

            def infer(self, model_name, inputs, headers=None):
                FakeServerClient.headers = headers
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
        client.propagate_trace_context = False

        client._infer(FakeInputData())

        self.assertIsNone(FakeServerClient.headers)

    def test_percentile_handles_boundaries(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]

        self.assertEqual(percentile(values, 0), 10.0)
        self.assertEqual(percentile(values, 50), 30.0)
        self.assertEqual(percentile(values, 100), 50.0)

    def test_builds_constant_rate_submission_offsets(self) -> None:
        self.assertEqual(
            build_constant_rate_submission_offsets(4, request_rate_rps=4.0),
            [0.0, 0.25, 0.5, 0.75],
        )
        self.assertEqual(
            build_constant_rate_submission_offsets(1, request_rate_rps=7.5),
            [0.0],
        )
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            build_constant_rate_submission_offsets(2, request_rate_rps=0)

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

    def test_retry_attempts_are_counted_per_logical_request(self) -> None:
        class FlakyClient:
            def __init__(self) -> None:
                self.calls = 0

            def infer(self) -> None:
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("transient fixture failure")

        result = execute_with_retries(FlakyClient(), retries=2)

        self.assertTrue(result.ok)
        self.assertEqual(result.attempt_count, 3)

        metrics = summarize_results(
            [
                result,
                InferenceResult(ok=True, latency_ms=1.0, attempt_count=1),
                InferenceResult(
                    ok=False,
                    latency_ms=2.0,
                    error="private failure detail",
                    attempt_count=3,
                ),
            ],
            duration_seconds=1.0,
            config=BenchmarkConfig(num_requests=3, retries=2),
        )

        retry = metrics["retry"]
        self.assertEqual(retry["logical_requests"], 3)
        self.assertEqual(retry["client_attempts"], 7)
        self.assertEqual(retry["retry_attempts"], 4)
        self.assertEqual(retry["retried_requests"], 2)
        self.assertEqual(retry["recovered_requests"], 1)
        self.assertEqual(retry["exhausted_requests"], 1)
        self.assertEqual(retry["client_attempt_amplification"], 2.3333)
        self.assertNotIn("private failure detail", json.dumps(metrics))

    def test_retry_gate_and_prometheus_use_measured_client_attempts(self) -> None:
        metrics = summarize_results(
            [
                InferenceResult(ok=True, latency_ms=1.0, attempt_count=2),
                InferenceResult(ok=True, latency_ms=1.0, attempt_count=1),
                InferenceResult(ok=True, latency_ms=1.0, attempt_count=1),
                InferenceResult(ok=True, latency_ms=1.0, attempt_count=1),
            ],
            duration_seconds=1.0,
            config=BenchmarkConfig(num_requests=4, retries=1),
        )

        passing = build_retry_gate(metrics["retry"], 1.25)
        failing = build_retry_gate(metrics["retry"], 1.20)
        rounding_boundary = build_retry_gate(
            {
                "logical_requests": 10_000,
                "client_attempts": 10_001,
                "client_attempt_amplification": 1.0,
            },
            1.00005,
        )

        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])
        self.assertFalse(rounding_boundary["passed"])
        self.assertEqual(
            rounding_boundary["checks"]["client_attempt_amplification"]["observed"],
            1.0001,
        )
        self.assertIn("exceeded", " ".join(failing["failure_reasons"]))

        prometheus = format_prometheus_metrics({**metrics, "retry_gate": passing})
        self.assertIn("triton_benchmark_client_attempts_total", prometheus)
        self.assertIn("triton_benchmark_retry_attempts_total", prometheus)
        self.assertIn("triton_benchmark_recovered_requests_total", prometheus)
        self.assertIn("triton_benchmark_client_attempt_amplification", prometheus)
        self.assertIn("triton_benchmark_retry_gate_passed", prometheus)
        self.assertIn("triton_benchmark_retry_gate_max_amplification", prometheus)

    def test_retry_gate_cli_exits_six_when_amplification_exceeds_budget(self) -> None:
        class FlakyClient:
            def __init__(self) -> None:
                self.calls = 0

            def infer(self) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient fixture failure")

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch(
                "sys.argv",
                [
                    "benchmark.py",
                    "--num-requests",
                    "2",
                    "--concurrency",
                    "1",
                    "--retries",
                    "1",
                    "--max-client-attempt-amplification",
                    "1",
                    "--fail-on-retry-gate",
                    "--output-dir",
                    output_dir,
                ],
            ),
            patch("benchmark.build_client", return_value=FlakyClient()),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(SystemExit, "6"):
                main()

    def test_request_path_gate_cli_exits_seven_on_accounting_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before_path = root / "before.prom"
            after_path = root / "after.prom"
            output_dir = root / "results"
            before_path.write_text(
                "router_total 0\nbackend_total 0\nsuccess_total 0\n",
                encoding="utf-8",
            )
            after_path.write_text(
                "router_total 2\nbackend_total 1\nsuccess_total 1\n",
                encoding="utf-8",
            )
            with (
                patch(
                    "sys.argv",
                    [
                        "benchmark.py",
                        "--num-requests",
                        "1",
                        "--concurrency",
                        "1",
                        "--retries",
                        "0",
                        "--telemetry-baseline-prometheus",
                        str(before_path),
                        "--telemetry-prometheus",
                        str(after_path),
                        "--request-path-ingress-metric",
                        "router_total",
                        "--request-path-backend-metric",
                        "backend_total",
                        "--request-path-success-metric",
                        "success_total",
                        "--fail-on-request-path-gap",
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(SystemExit, "7"):
                    main()

    def test_trace_context_summary_and_prometheus_omit_identifiers(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=1.0,
            config=BenchmarkConfig(
                mode="openai",
                num_requests=1,
                propagate_trace_context=True,
            ),
        )

        self.assertEqual(metrics["trace_context"]["propagation"], "w3c_traceparent")
        self.assertFalse(metrics["trace_context"]["identifiers_persisted"])
        self.assertEqual(metrics["trace_context"]["server_acceptance"], "not verified")
        serialized = json.dumps(metrics)
        self.assertIsNone(re.search(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", serialized))

        prometheus = format_prometheus_metrics(metrics)
        self.assertIn("triton_benchmark_trace_context_enabled", prometheus)
        self.assertIn('model="resnet50_trt_fp16"} 1', prometheus)
        self.assertIsNone(
            re.search(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", prometheus)
        )

    def test_summarizes_and_gates_response_trace_continuation(self) -> None:
        metrics = summarize_results(
            [
                InferenceResult(
                    ok=True,
                    latency_ms=10.0,
                    streaming=StreamingInferenceObservation(
                        4.0, 1.0, 2, 3, 8, "matched"
                    ),
                ),
                InferenceResult(
                    ok=True,
                    latency_ms=11.0,
                    streaming=StreamingInferenceObservation(
                        5.0, 1.5, 2, 3, 8, "missing"
                    ),
                ),
            ],
            duration_seconds=1.0,
            config=BenchmarkConfig(
                mode="openai",
                num_requests=2,
                propagate_trace_context=True,
            ),
        )

        continuation = metrics["trace_context"]["response_continuation"]
        self.assertEqual(continuation["request_count"], 2)
        self.assertEqual(continuation["matched_responses"], 1)
        self.assertEqual(continuation["missing_responses"], 1)
        self.assertEqual(continuation["match_coverage"], 0.5)
        self.assertFalse(continuation["complete"])
        self.assertFalse(continuation["identifiers_persisted"])

        gate = build_trace_context_gate(metrics)
        self.assertFalse(gate["passed"])
        self.assertIn("missing", " ".join(gate["failure_reasons"]))
        prometheus = format_prometheus_metrics({**metrics, "trace_context_gate": gate})
        self.assertIn("triton_benchmark_trace_response_total", prometheus)
        self.assertIn('status="matched"} 1', prometheus)
        self.assertIn('status="missing"} 1', prometheus)
        self.assertIn("triton_benchmark_trace_context_gate_passed", prometheus)
        self.assertIsNone(
            re.search(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", prometheus)
        )

    def test_trace_context_gate_passes_complete_measured_responses(self) -> None:
        metrics = summarize_results(
            [
                InferenceResult(
                    ok=True,
                    latency_ms=10.0,
                    streaming=StreamingInferenceObservation(
                        4.0, 1.0, 2, 3, 8, "matched"
                    ),
                )
            ],
            duration_seconds=1.0,
            config=BenchmarkConfig(
                mode="openai",
                num_requests=1,
                propagate_trace_context=True,
            ),
        )

        gate = build_trace_context_gate(metrics)

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failure_reasons"], [])

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

    def test_open_loop_rate_paces_measured_submissions_and_reports_lag(self) -> None:
        class RecordingClient:
            def __init__(self) -> None:
                self.started: list[float] = []

            def infer(self) -> None:
                self.started.append(time.perf_counter())

        client = RecordingClient()
        metrics = run_benchmark(
            client,
            BenchmarkConfig(
                mode="mock",
                num_requests=4,
                concurrency=4,
                retries=0,
                request_rate_rps=20.0,
            ),
        )

        self.assertEqual(len(client.started), 4)
        self.assertGreaterEqual(client.started[-1] - client.started[0], 0.12)
        schedule = metrics["load_schedule"]
        self.assertEqual(schedule["mode"], "open_loop_constant_rate")
        self.assertEqual(schedule["configured_request_rate_rps"], 20.0)
        self.assertEqual(schedule["request_count"], 4)
        self.assertEqual(schedule["scheduled_submission_span_seconds"], 0.15)
        self.assertGreater(schedule["achieved_submission_rate_rps"], 0)
        self.assertGreater(schedule["achieved_request_start_rate_rps"], 0)
        self.assertIn("p95", schedule["submission_lag_ms"])
        self.assertIn("p95", schedule["executor_queue_ms"])
        self.assertIn("p95", schedule["request_start_lag_ms"])

        prometheus = format_prometheus_metrics(metrics)
        self.assertIn("triton_benchmark_configured_request_rate_rps", prometheus)
        self.assertIn("triton_benchmark_submission_lag_ms", prometheus)
        self.assertIn("triton_benchmark_achieved_request_start_rate_rps", prometheus)
        self.assertIn("triton_benchmark_executor_queue_ms", prometheus)
        self.assertIn("triton_benchmark_request_start_lag_ms", prometheus)
        self.assertIn('quantile="0.95"', prometheus)

    def test_open_loop_reports_executor_queue_when_workers_are_saturated(self) -> None:
        class SlowClient:
            def infer(self) -> None:
                time.sleep(0.03)

        metrics = run_benchmark(
            SlowClient(),
            BenchmarkConfig(
                mode="mock",
                num_requests=6,
                concurrency=1,
                retries=0,
                request_rate_rps=100.0,
            ),
        )

        schedule = metrics["load_schedule"]
        self.assertLess(schedule["submission_lag_ms"]["p95"], 20.0)
        self.assertGreater(schedule["executor_queue_ms"]["p95"], 20.0)
        self.assertGreater(schedule["request_start_lag_ms"]["p95"], 20.0)
        self.assertLess(
            schedule["achieved_request_start_rate_rps"],
            schedule["achieved_submission_rate_rps"],
        )

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

    def test_request_path_accounting_correlates_private_counter_deltas(self) -> None:
        metrics = summarize_results(
            [
                InferenceResult(ok=True, latency_ms=10.0, attempt_count=2),
                InferenceResult(ok=True, latency_ms=10.0),
                InferenceResult(ok=True, latency_ms=10.0),
                InferenceResult(ok=True, latency_ms=10.0),
            ],
            duration_seconds=1.0,
            config=BenchmarkConfig(num_requests=4, retries=1),
        )
        metric_names = RequestPathMetricNames(
            ingress="private_router_receipts_total",
            backend="private_backend_receipts_total",
            success="private_backend_success_total",
        )
        before = """
        private_router_receipts_total{tenant="private-a"} 10
        private_backend_receipts_total{tenant="private-a"} 8
        private_backend_success_total{tenant="private-a"} 7
        """
        after = """
        private_router_receipts_total{tenant="private-a"} 15
        private_backend_receipts_total{tenant="private-a"} 12
        private_backend_success_total{tenant="private-a"} 11
        """

        accounting = build_request_path_accounting(
            metrics,
            before,
            after,
            metric_names,
            source="paired_http_prometheus_scrapes",
            alignment="harness_bracketed_measured_phase",
        )
        gate = build_request_path_gate(accounting)

        self.assertTrue(accounting["valid"])
        self.assertEqual(accounting["client"]["logical_requests"], 4)
        self.assertEqual(accounting["client"]["client_attempts"], 5)
        self.assertEqual(accounting["stages"]["ingress"]["delta"], 5)
        self.assertEqual(accounting["stages"]["backend"]["delta"], 4)
        self.assertEqual(accounting["stages"]["success"]["delta"], 4)
        self.assertEqual(accounting["ratios"]["ingress_per_client_attempt"], 1.0)
        self.assertEqual(accounting["ratios"]["backend_per_ingress"], 0.8)
        self.assertEqual(accounting["ratios"]["success_per_backend"], 1.0)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failure_reasons"], [])

        exported = format_prometheus_metrics(
            {**metrics, "request_path": accounting, "request_path_gate": gate}
        )
        self.assertIn("triton_benchmark_request_path_events_total", exported)
        self.assertIn('stage="ingress"} 5', exported)
        self.assertIn("triton_benchmark_request_path_ratio", exported)
        self.assertEqual(
            exported.count("# HELP triton_benchmark_request_path_ratio "),
            1,
        )
        self.assertIn("triton_benchmark_request_path_gate_passed", exported)

        serialized = json.dumps(accounting) + exported
        self.assertNotIn("private_router_receipts_total", serialized)
        self.assertNotIn("private_backend_receipts_total", serialized)
        self.assertNotIn("private_backend_success_total", serialized)
        self.assertNotIn("private-a", serialized)

    def test_request_path_accounting_fails_closed_on_churn_and_reset(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=1.0,
            config=BenchmarkConfig(num_requests=1),
        )
        metric_names = RequestPathMetricNames(
            ingress="router_total",
            backend="backend_total",
            success="success_total",
        )

        accounting = build_request_path_accounting(
            metrics,
            """
            router_total{worker="a"} 4
            backend_total 9
            success_total 7
            """,
            """
            router_total{worker="b"} 5
            backend_total 8
            success_total 8
            """,
            metric_names,
        )
        gate = build_request_path_gate(accounting)

        self.assertFalse(accounting["valid"])
        self.assertIsNone(accounting["stages"]["ingress"]["delta"])
        self.assertIsNone(accounting["stages"]["backend"]["delta"])
        self.assertIn("series_membership_changed", accounting["issues"])
        self.assertIn("counter_reset", accounting["issues"])
        self.assertFalse(gate["passed"])
        self.assertIn("invalid", " ".join(gate["failure_reasons"]))

    def test_request_path_accounting_rejects_malformed_counter_evidence(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=10.0)],
            duration_seconds=1.0,
            config=BenchmarkConfig(num_requests=1),
        )
        metric_names = RequestPathMetricNames(
            ingress="router_total",
            backend="backend_total",
            success="success_total",
        )
        cases = {
            "counter_missing": ("", ""),
            "non_finite_counter": ("router_total NaN", "router_total 1"),
            "negative_counter": ("router_total -1", "router_total 0"),
            "non_integral_counter": ("router_total 0.5", "router_total 1.5"),
            "duplicate_series": (
                "router_total{worker=\"a\"} 0\nrouter_total{worker=\"a\"} 0",
                "router_total{worker=\"a\"} 1\nrouter_total{worker=\"a\"} 1",
            ),
        }

        for expected_issue, (before_ingress, after_ingress) in cases.items():
            with self.subTest(issue=expected_issue):
                accounting = build_request_path_accounting(
                    metrics,
                    f"{before_ingress}\nbackend_total 0\nsuccess_total 0\n",
                    f"{after_ingress}\nbackend_total 1\nsuccess_total 1\n",
                    metric_names,
                )
                self.assertFalse(accounting["valid"])
                self.assertIn(expected_issue, accounting["issues"])
                self.assertFalse(build_request_path_gate(accounting)["passed"])

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

    def test_parses_open_loop_request_rate(self) -> None:
        with patch(
            "sys.argv",
            ["benchmark.py", "--request-rate-rps", "12.5", "--num-requests", "3"],
        ):
            options = parse_args()

        self.assertEqual(options.config.request_rate_rps, 12.5)

        with patch("sys.argv", ["benchmark.py", "--request-rate-rps", "-1"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_parses_and_validates_retry_amplification_gate(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--max-client-attempt-amplification",
                "1.25",
                "--fail-on-retry-gate",
            ],
        ):
            options = parse_args()

        self.assertEqual(options.max_client_attempt_amplification, 1.25)
        self.assertTrue(options.fail_on_retry_gate)

        for argv in (
            ["benchmark.py", "--max-client-attempt-amplification", "0.99"],
            ["benchmark.py", "--max-client-attempt-amplification", "nan"],
            ["benchmark.py", "--fail-on-retry-gate"],
            ["benchmark.py", "--retries", "-1"],
        ):
            with self.subTest(argv=argv), patch("sys.argv", argv):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parses_trace_context_opt_in_for_live_mode(self) -> None:
        with patch(
            "sys.argv",
            ["benchmark.py", "--mode", "openai", "--propagate-trace-context"],
        ):
            options = parse_args()

        self.assertTrue(options.config.propagate_trace_context)

    def test_parses_response_trace_context_gate_for_openai_mode(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--mode",
                "openai",
                "--propagate-trace-context",
                "--fail-on-trace-context-gap",
            ],
        ):
            options = parse_args()

        self.assertTrue(options.fail_on_trace_context_gap)

    def test_response_trace_context_gate_requires_openai_propagation(self) -> None:
        invalid_argv = (
            ["benchmark.py", "--fail-on-trace-context-gap"],
            [
                "benchmark.py",
                "--mode",
                "triton",
                "--propagate-trace-context",
                "--fail-on-trace-context-gap",
            ],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv), patch("sys.argv", argv):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_trace_context_opt_in_rejects_mock_mode(self) -> None:
        with patch("sys.argv", ["benchmark.py", "--propagate-trace-context"]):
            with self.assertRaises(SystemExit):
                parse_args()

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
        self.assertFalse(report["passed"])
        self.assertEqual(report["mismatched_outputs"], 4)
        self.assertEqual(report["mismatched_sample_ids"], [0, 1, 2, 3])

    def test_batch_invariance_probe_accepts_numeric_drift_within_policy(self) -> None:
        class NumericallyStableClient:
            def infer_output(self, sample_id: int) -> OutputObservation:
                import threading

                concurrent = threading.current_thread() is not threading.main_thread()
                values = (1.0005, 2.001) if concurrent else (1.0, 2.0)
                return OutputObservation(
                    fingerprint=f"private-output-canary:{sample_id}:{values}",
                    numeric_outputs=(
                        NumericOutput(
                            name="scores",
                            dtype="float32",
                            shape=(2,),
                            values=values,
                        ),
                    ),
                )

        report = run_batch_invariance_probe(
            NumericallyStableClient(),
            probe_count=4,
            concurrency=2,
            absolute_tolerance=0.0001,
            relative_tolerance=0.001,
        )

        self.assertTrue(report["passed"])
        self.assertFalse(report["exact_match"])
        self.assertEqual(report["exact_matches"], 0)
        self.assertEqual(report["tolerance_matches"], 4)
        self.assertEqual(report["matched_outputs"], 4)
        self.assertEqual(report["mismatch_reasons"], {})
        self.assertAlmostEqual(report["max_observed_absolute_error"], 0.001)
        self.assertAlmostEqual(report["max_observed_relative_error"], 0.0005)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("private-output-canary", serialized)
        self.assertNotIn('"values"', serialized)
        self.assertNotIn('"fingerprint"', serialized)
        self.assertFalse(report["comparison_policy"]["output_values_persisted"])

    def test_batch_invariance_probe_rejects_numeric_drift_outside_policy(self) -> None:
        class NumericallyUnstableClient:
            def infer_output(self, sample_id: int) -> OutputObservation:
                import threading

                value = 1.02 if threading.current_thread() is not threading.main_thread() else 1.0
                return OutputObservation(
                    fingerprint=f"{sample_id}:{value}",
                    numeric_outputs=(
                        NumericOutput("scores", "float32", (1,), (value,)),
                    ),
                )

        report = run_batch_invariance_probe(
            NumericallyUnstableClient(),
            probe_count=3,
            concurrency=2,
            absolute_tolerance=0.001,
            relative_tolerance=0.001,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["tolerance_matches"], 0)
        self.assertEqual(report["mismatched_outputs"], 3)
        self.assertEqual(report["mismatch_reasons"], {"outside_tolerance": 3})

    def test_batch_invariance_probe_fails_closed_on_structural_output_change(self) -> None:
        class ShapeChangingClient:
            def infer_output(self, sample_id: int) -> OutputObservation:
                import threading

                concurrent = threading.current_thread() is not threading.main_thread()
                shape = (1, 1) if concurrent else (1,)
                return OutputObservation(
                    fingerprint=f"{sample_id}:{shape}",
                    numeric_outputs=(
                        NumericOutput("scores", "float32", shape, (1.0,)),
                    ),
                )

        report = run_batch_invariance_probe(
            ShapeChangingClient(),
            probe_count=2,
            concurrency=2,
            absolute_tolerance=0.01,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["incompatible_outputs"], 2)
        self.assertEqual(report["mismatch_reasons"], {"structural_incompatibility": 2})

    def test_batch_invariance_probe_fails_closed_on_non_numeric_output_change(self) -> None:
        class TextOutputClient:
            def infer_output(self, sample_id: int) -> str:
                import threading

                phase = (
                    "concurrent"
                    if threading.current_thread() is not threading.main_thread()
                    else "isolated"
                )
                return f"{sample_id}:{phase}"

        report = run_batch_invariance_probe(
            TextOutputClient(),
            probe_count=2,
            concurrency=2,
            absolute_tolerance=0.01,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["incompatible_outputs"], 2)
        self.assertEqual(report["mismatch_reasons"], {"non_numeric_output": 2})

    def test_batch_invariance_probe_rejects_non_finite_numeric_drift(self) -> None:
        class NonFiniteClient:
            def infer_output(self, sample_id: int) -> OutputObservation:
                import threading

                value = float("nan") if threading.current_thread() is not threading.main_thread() else 1.0
                return OutputObservation(
                    fingerprint=f"{sample_id}:{value}",
                    numeric_outputs=(
                        NumericOutput("scores", "float32", (1,), (value,)),
                    ),
                )

        report = run_batch_invariance_probe(
            NonFiniteClient(),
            probe_count=2,
            concurrency=2,
            absolute_tolerance=1.0,
            relative_tolerance=1.0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["mismatch_reasons"], {"non_finite_values": 2})
        self.assertNotIn("NaN", json.dumps(report, allow_nan=False))

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
            "exact_matches": 2,
            "tolerance_matches": 1,
            "mismatched_outputs": 1,
            "failed_probes": 0,
            "noise_failures": 0,
            "match_rate": 0.75,
            "exact_match": False,
            "passed": False,
            "max_observed_absolute_error": 0.001,
            "max_observed_relative_error": 0.0005,
            "comparison_policy": {
                "absolute_tolerance": 0.0001,
                "relative_tolerance": 0.001,
            },
        }

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_batch_invariance_probes_total", output)
        self.assertIn("triton_benchmark_batch_invariance_mismatches_total", output)
        self.assertIn("triton_benchmark_batch_invariance_noise_failures_total", output)
        self.assertIn("triton_benchmark_batch_invariance_match_rate", output)
        self.assertIn("triton_benchmark_batch_invariance_exact_matches_total", output)
        self.assertIn("triton_benchmark_batch_invariance_tolerance_matches_total", output)
        self.assertIn("triton_benchmark_batch_invariance_max_absolute_error", output)
        self.assertIn("triton_benchmark_batch_invariance_max_relative_error", output)
        self.assertIn("triton_benchmark_batch_invariance_absolute_tolerance", output)
        self.assertIn("triton_benchmark_batch_invariance_relative_tolerance", output)
        self.assertIn("triton_benchmark_batch_invariance_passed", output)
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

    def test_parses_batch_output_tolerance_policy(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--batch-invariance-probes",
                "4",
                "--concurrency",
                "2",
                "--batch-output-atol",
                "0.0001",
                "--batch-output-rtol",
                "0.001",
            ],
        ):
            options = parse_args()

        self.assertEqual(options.batch_output_atol, 0.0001)
        self.assertEqual(options.batch_output_rtol, 0.001)

    def test_batch_output_tolerance_requires_probes(self) -> None:
        with patch("sys.argv", ["benchmark.py", "--batch-output-atol", "0.001"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_batch_output_tolerance_must_be_finite_and_non_negative(self) -> None:
        for flag, value in (
            ("--batch-output-atol", "-0.001"),
            ("--batch-output-rtol", "nan"),
        ):
            with self.subTest(flag=flag, value=value):
                with patch(
                    "sys.argv",
                    [
                        "benchmark.py",
                        "--batch-invariance-probes",
                        "2",
                        "--concurrency",
                        "2",
                        flag,
                        value,
                    ],
                ):
                    with self.assertRaises(SystemExit):
                        parse_args()

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
                dcgm_gpu_utilization{gpu="1"} 120
                DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 50
                DCGM_FI_DEV_FB_USED{gpu="0"} 3072
                """,
            ],
            in_window_scrape_count=1,
            configured_interval_seconds=0.25,
        )

        self.assertEqual(window["scrape_count"], 3)
        self.assertEqual(window["in_window_scrape_count"], 1)
        self.assertEqual(window["gpu"]["utilization_pct"]["sample_count"], 6)
        self.assertEqual(window["gpu"]["utilization_pct"]["avg"], 70.0)
        self.assertEqual(window["gpu"]["utilization_pct"]["p50"], 60.0)
        self.assertEqual(window["gpu"]["utilization_pct"]["p95"], 120.0)
        self.assertEqual(window["gpu"]["memory_used_mib"]["max"], 3072.0)
        self.assertTrue(window["series_membership"]["stable"])
        self.assertEqual(window["series_membership"]["series_count"], 4)
        self.assertEqual(
            len(window["series_membership"]["fingerprint_sha256"]),
            64,
        )
        self.assertIn("not time-weighted", " ".join(window["notes"]))

    def test_gpu_gauge_window_rejects_target_churn_without_exposing_labels(self) -> None:
        private_before = 'DCGM_FI_DEV_GPU_UTIL{gpu="private-gpu-a"} 20\n'
        private_after = 'DCGM_FI_DEV_GPU_UTIL{gpu="private-gpu-b"} 30\n'

        with self.assertRaisesRegex(
            ValueError,
            "GPU telemetry series membership changed",
        ) as context:
            build_gpu_gauge_window(
                [private_before, private_before, private_after],
                in_window_scrape_count=1,
                configured_interval_seconds=0.25,
            )

        self.assertNotIn("private-gpu-a", str(context.exception))
        self.assertNotIn("private-gpu-b", str(context.exception))

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
        self.assertTrue(window["series_membership"]["stable"])
        self.assertEqual(window["series_membership"]["series_count"], 5)
        self.assertEqual(
            len(window["series_membership"]["fingerprint_sha256"]),
            64,
        )

    def test_telemetry_gate_fails_closed_on_series_membership_churn(self) -> None:
        def snapshot(instance: str, increment: int) -> str:
            labels = f'model="review-model",instance="{instance}"'
            return (
                f"nv_inference_request_success{{{labels}}} {400 + increment}\n"
                f"nv_inference_request_failure{{{labels}}} {increment}\n"
                f"nv_inference_request_duration_us{{{labels}}} {9400000 + increment}\n"
                f"nv_inference_queue_duration_us{{{labels}}} {210000 + increment}\n"
                f"nv_inference_compute_infer_duration_us{{{labels}}} {7600000 + increment}\n"
            )

        before = build_telemetry_summary(
            snapshot("private-node-a", 0),
            model_name="review-model",
        )
        after = build_telemetry_summary(
            snapshot("private-node-b", 100),
            model_name="review-model",
        )

        window = build_telemetry_counter_window(before, after)
        gate = build_telemetry_gate(window, max_server_failure_rate=1.0)

        self.assertFalse(window["valid"])
        self.assertFalse(window["series_membership"]["stable"])
        self.assertEqual(window["series_membership"]["before_series_count"], 5)
        self.assertEqual(window["series_membership"]["after_series_count"], 5)
        self.assertFalse(gate["passed"])
        self.assertIn("series membership changed", " ".join(gate["failure_reasons"]))
        serialized = json.dumps({"window": window, "gate": gate})
        self.assertNotIn("private-node-a", serialized)
        self.assertNotIn("private-node-b", serialized)
        self.assertNotIn("nv_inference_request_success", serialized)

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
            "series_membership": {
                "stable": True,
                "before_series_count": 5,
                "after_series_count": 5,
            },
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
        self.assertIn("triton_benchmark_server_series_membership_stable", output)
        self.assertIn('snapshot="before"} 5', output)
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
            "series_membership": {
                "stable": True,
                "series_count": 2,
            },
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
        self.assertIn("triton_benchmark_gpu_window_series_membership_stable", output)
        self.assertIn("triton_benchmark_gpu_window_series_count", output)
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

    def test_parses_request_path_metrics_and_isolated_gate(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--telemetry-url",
                "https://metrics.example.test/metrics",
                "--request-path-ingress-metric",
                "router_requests_total",
                "--request-path-backend-metric",
                "backend_requests_total",
                "--request-path-success-metric",
                "backend_success_total",
                "--fail-on-request-path-gap",
            ],
        ):
            options = parse_args()

        self.assertEqual(options.request_path_metrics.ingress, "router_requests_total")
        self.assertEqual(options.request_path_metrics.backend, "backend_requests_total")
        self.assertEqual(options.request_path_metrics.success, "backend_success_total")
        self.assertTrue(options.fail_on_request_path_gap)

    def test_request_path_metrics_require_complete_names_and_paired_telemetry(self) -> None:
        invalid_argv = (
            ["benchmark.py", "--request-path-ingress-metric", "router_total"],
            [
                "benchmark.py",
                "--request-path-ingress-metric",
                "router_total",
                "--request-path-backend-metric",
                "backend_total",
                "--request-path-success-metric",
                "success_total",
            ],
            ["benchmark.py", "--fail-on-request-path-gap"],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv):
                with patch("sys.argv", argv):
                    with self.assertRaises(SystemExit):
                        parse_args()


if __name__ == "__main__":
    unittest.main()
