import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from benchmark import (
    BenchmarkConfig,
    CostModelConfig,
    InferenceResult,
    LlmMetricsConfig,
    OpenAICompatibleStreamingClient,
    StreamingInferenceObservation,
    build_client,
    build_cost_model,
    build_llm_metrics,
    execute_with_retries,
    format_prometheus_metrics,
    normalize_openai_completions_url,
    parse_args,
    summarize_results,
)


class OpenAIStreamingClientTest(unittest.TestCase):
    def test_normalizes_server_root_and_v1_urls(self) -> None:
        self.assertEqual(
            normalize_openai_completions_url("http://localhost:8000"),
            "http://localhost:8000/v1/completions",
        )
        self.assertEqual(
            normalize_openai_completions_url("http://localhost:8000/v1/"),
            "http://localhost:8000/v1/completions",
        )
        self.assertEqual(
            normalize_openai_completions_url(
                "http://localhost:8000/v1/completions"
            ),
            "http://localhost:8000/v1/completions",
        )

    def test_builds_openai_client_from_environment_key(self) -> None:
        config = BenchmarkConfig(
            mode="openai",
            server_url="http://localhost:8000/v1",
            model_name="review-model",
            openai_prompt="synthetic prompt",
            openai_max_tokens=32,
            openai_timeout_seconds=9.0,
            openai_api_key_env="TEST_LLM_KEY",
        )

        with patch.dict(os.environ, {"TEST_LLM_KEY": "test-secret"}, clear=False):
            client = build_client(config)

        self.assertIsInstance(client, OpenAICompatibleStreamingClient)
        self.assertEqual(client.endpoint_url, "http://localhost:8000/v1/completions")
        self.assertEqual(client.prompt, "synthetic prompt")
        self.assertEqual(client.max_tokens, 32)
        self.assertEqual(client.timeout_seconds, 9.0)
        self.assertEqual(client.api_key, "test-secret")

    def test_does_not_send_ambient_api_key_without_explicit_opt_in(self) -> None:
        config = BenchmarkConfig(
            mode="openai",
            server_url="https://third-party.example/v1",
            model_name="test-model",
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-send"}):
            client = build_client(config)

        self.assertIsNone(client.api_key)

    def test_parses_openai_streaming_cli_configuration(self) -> None:
        argv = [
            "benchmark.py",
            "--mode",
            "openai",
            "--server-url",
            "http://localhost:8000/v1",
            "--model-name",
            "review-model",
            "--openai-prompt",
            "synthetic prompt",
            "--openai-max-tokens",
            "32",
            "--openai-timeout-seconds",
            "9",
            "--openai-api-key-env",
            "TEST_LLM_KEY",
        ]

        with patch("sys.argv", argv):
            options = parse_args()

        self.assertEqual(options.config.mode, "openai")
        self.assertEqual(options.config.openai_prompt, "synthetic prompt")
        self.assertEqual(options.config.openai_max_tokens, 32)
        self.assertEqual(options.config.openai_timeout_seconds, 9.0)
        self.assertEqual(options.config.openai_api_key_env, "TEST_LLM_KEY")

    def test_retry_layer_preserves_streaming_observation(self) -> None:
        expected = StreamingInferenceObservation(
            time_to_first_token_ms=12.0,
            inter_chunk_latency_ms=4.0,
            observed_output_chunks=3,
            reported_output_tokens=5,
            output_bytes=17,
        )

        class Client:
            def infer(self) -> StreamingInferenceObservation:
                return expected

        result = execute_with_retries(Client(), retries=0)

        self.assertTrue(result.ok)
        self.assertEqual(result.streaming, expected)

    def test_summarizes_measured_streaming_metrics(self) -> None:
        results = [
            InferenceResult(
                ok=True,
                latency_ms=30.0,
                streaming=StreamingInferenceObservation(10.0, 2.0, 3, 4, 10),
            ),
            InferenceResult(
                ok=True,
                latency_ms=50.0,
                streaming=StreamingInferenceObservation(20.0, 4.0, 5, 6, 20),
            ),
            InferenceResult(ok=False, latency_ms=7.0, error="boom"),
        ]

        metrics = summarize_results(
            results,
            duration_seconds=2.0,
            config=BenchmarkConfig(
                mode="openai",
                server_url="http://localhost:8000/v1",
                model_name="review-model",
                num_requests=3,
                concurrency=2,
            ),
        )

        streaming = metrics["streaming"]
        self.assertEqual(streaming["request_count"], 2)
        self.assertEqual(streaming["reported_output_tokens"], 10)
        self.assertEqual(streaming["observed_output_chunks"], 8)
        self.assertEqual(streaming["output_bytes"], 30)
        self.assertEqual(streaming["output_tokens_per_second"], 5.0)
        self.assertEqual(streaming["time_to_first_token_ms"]["p95"], 20.0)
        self.assertEqual(streaming["inter_chunk_latency_ms"]["p95"], 4.0)
        self.assertEqual(metrics["server_url"], "http://localhost:8000/v1")

    def test_summary_does_not_store_openai_prompt_content(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=1.0)],
            1.0,
            BenchmarkConfig(
                mode="openai",
                server_url="http://localhost:8000/v1",
                openai_prompt="private benchmark prompt",
            ),
        )

        self.assertNotIn("openai_prompt", metrics["config"])
        self.assertEqual(metrics["config"]["openai_prompt_bytes"], 24)
        self.assertEqual(len(metrics["config"]["openai_prompt_sha256"]), 64)

    def test_summary_redacts_url_credentials_query_and_fragment(self) -> None:
        metrics = summarize_results(
            [InferenceResult(ok=True, latency_ms=1.0)],
            1.0,
            BenchmarkConfig(
                mode="openai",
                server_url=(
                    "https://user:secret@example.test:8443/v1"
                    "?api_key=must-not-persist#private"
                ),
            ),
        )

        serialized = json.dumps(metrics)
        self.assertEqual(metrics["server_url"], "https://example.test:8443/v1")
        self.assertEqual(
            metrics["config"]["server_url"],
            "https://example.test:8443/v1",
        )
        self.assertNotIn("secret", serialized)
        self.assertNotIn("must-not-persist", serialized)

    def test_cost_model_prefers_complete_streaming_usage(self) -> None:
        metrics = {
            "successful_requests": 2,
            "duration_seconds": 1.0,
            "streaming": {
                "request_count": 2,
                "reported_token_request_count": 2,
                "reported_output_tokens": 6,
            },
        }

        cost_model = build_cost_model(
            metrics,
            CostModelConfig(
                input_tokens_per_request=512,
                output_tokens_per_request=128,
            ),
        )

        self.assertEqual(cost_model["workload"]["successful_output_tokens"], 6)
        self.assertEqual(cost_model["workload"]["output_tokens_per_request"], 3.0)
        self.assertEqual(
            cost_model["workload"]["output_tokens_source"],
            "server-reported streaming usage",
        )

    def test_llm_metrics_prefers_measured_streaming_values(self) -> None:
        metrics = {
            "successful_requests": 2,
            "duration_seconds": 1.0,
            "streaming": {
                "request_count": 2,
                "reported_token_request_count": 2,
                "reported_output_tokens": 6,
                "time_to_first_token_ms": {"avg": 12.0},
                "inter_chunk_latency_ms": {"avg": 4.0},
            },
        }

        llm_metrics = build_llm_metrics(
            metrics,
            LlmMetricsConfig(
                context_tokens_per_request=512,
                time_to_first_token_ms=80.0,
                inter_token_latency_ms=25.0,
            ),
            CostModelConfig(output_tokens_per_request=128),
        )

        self.assertEqual(llm_metrics["latency_ms"]["time_to_first_token"], 12.0)
        self.assertEqual(llm_metrics["latency_ms"]["inter_token"], 25.0)
        self.assertEqual(llm_metrics["latency_ms"]["inter_chunk"], 4.0)
        self.assertEqual(llm_metrics["throughput"]["output_tokens_per_second"], 6.0)
        self.assertEqual(
            llm_metrics["claim_scope"]["latency_source"],
            "measured TTFT and inter-chunk; caller-provided inter-token latency",
        )

    def test_prometheus_export_includes_measured_streaming_metrics(self) -> None:
        metrics = summarize_results(
            [
                InferenceResult(
                    ok=True,
                    latency_ms=30.0,
                    streaming=StreamingInferenceObservation(10.0, 2.0, 3, 4, 10),
                )
            ],
            duration_seconds=2.0,
            config=BenchmarkConfig(mode="openai", model_name="review-model"),
        )

        output = format_prometheus_metrics(metrics)

        self.assertIn("triton_benchmark_streaming_ttft_ms", output)
        self.assertIn("triton_benchmark_streaming_inter_chunk_latency_ms", output)
        self.assertIn("triton_benchmark_streaming_reported_output_tokens_total", output)
        self.assertIn("triton_benchmark_streaming_observed_output_chunks_total", output)
        self.assertIn("triton_benchmark_streaming_output_tokens_per_second", output)
        self.assertIn('mode="openai"', output)

    def test_measures_streaming_completion_events(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            payload = None

            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                length = int(self.headers["Content-Length"])
                Handler.payload = json.loads(self.rfile.read(length))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [
                    {"choices": [{"text": "hello"}]},
                    {"choices": [{"text": " world"}]},
                    {"choices": [], "usage": {"completion_tokens": 2}},
                ]
                for event in events:
                    self.wfile.write(
                        f"data: {json.dumps(event)}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                    time.sleep(0.005)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAICompatibleStreamingClient(
                server_url=f"http://127.0.0.1:{server.server_port}",
                model_name="review-model",
                prompt="synthetic benchmark prompt",
                max_tokens=8,
                timeout_seconds=5.0,
            )
            observation = client.infer()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(observation.observed_output_chunks, 2)
        self.assertEqual(observation.reported_output_tokens, 2)
        self.assertEqual(observation.output_bytes, len("hello world".encode("utf-8")))
        self.assertGreaterEqual(observation.time_to_first_token_ms, 0.0)
        self.assertGreaterEqual(observation.inter_chunk_latency_ms, 0.0)
        self.assertEqual(Handler.payload["model"], "review-model")
        self.assertEqual(Handler.payload["max_tokens"], 8)
        self.assertTrue(Handler.payload["stream"])
        self.assertEqual(Handler.payload["stream_options"], {"include_usage": True})
        self.assertEqual(Handler.payload["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
