from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark import (
    BenchmarkConfig,
    CoordinatedClientConfig,
    MockInferenceClient,
    parse_args,
    run_benchmark,
)
from coordinated_benchmark import (
    build_coordinator_configuration_fingerprint,
    build_coordinated_summary,
    build_remote_coordinated_summary,
    create_coordinator_resume_manifest,
    derive_coordinator_run_id,
    format_coordinated_prometheus,
    load_coordinator_resume_manifest,
    _run_remote_clients,
    validate_benchmark_args,
    write_coordinator_resume_manifest,
)


RUN_HASH = "a" * 64
CONFIG_HASH = "b" * 64


def shard(
    index: int,
    *,
    started_ns: int,
    ended_ns: int,
    successful: int = 4,
    failed: int = 0,
    attempts: int = 4,
    run_hash: str = RUN_HASH,
    config_hash: str = CONFIG_HASH,
    planned_start_ns: int = 900_000_000,
) -> dict[str, object]:
    logical_requests = successful + failed
    return {
        "mode": "openai",
        "server_url": "https://private.example.test/v1",
        "num_requests": logical_requests,
        "successful_requests": successful,
        "failed_requests": failed,
        "success_rate": successful / logical_requests,
        "duration_seconds": (ended_ns - started_ns) / 1_000_000_000,
        "throughput_rps": 999,
        "latency_ms": {"p95": 123.0},
        "config": {
            "openai_prompt_sha256": "c" * 64,
            "openai_prompt_bytes": 19,
        },
        "private_fixture_value": "sensitive fixture prompt",
        "retry": {
            "logical_requests": logical_requests,
            "client_attempts": attempts,
            "retry_attempts": attempts - logical_requests,
            "retried_requests": max(0, attempts - logical_requests),
            "recovered_requests": max(0, attempts - logical_requests),
            "exhausted_requests": failed,
        },
        "load_schedule": {
            "configured_request_rate_rps": 10.0,
        },
        "coordination": {
            "run_id_sha256": run_hash,
            "run_id_persisted": False,
            "config_fingerprint_sha256": config_hash,
            "client_index": index,
            "client_count": 2,
            "planned_start_unix_ns": planned_start_ns,
            "measured_start_unix_ns": started_ns,
            "measured_end_unix_ns": ended_ns,
            "clock": "host_wall_clock",
        },
    }


def remote_shard(
    index: int,
    *,
    normalized_start_ns: int,
    normalized_end_ns: int,
    offset_ns: int,
    uncertainty_ns: int,
    agent_hash: str,
    result_source: str = "executed",
    transport_retries: int = 0,
    max_transport_recovery_attempts: int = 1,
) -> dict[str, object]:
    planned_coordinator_ns = 900_000_000
    result = shard(
        index,
        started_ns=normalized_start_ns + offset_ns,
        ended_ns=normalized_end_ns + offset_ns,
        planned_start_ns=planned_coordinator_ns + offset_ns,
    )
    result["_remote_agent"] = {
        "agent_id_sha256": agent_hash,
        "agent_id_persisted": False,
        "clock_offset_agent_minus_coordinator_ns": offset_ns,
        "clock_network_delay_ns": uncertainty_ns * 2,
        "clock_uncertainty_ns": uncertainty_ns,
        "clock_sample_count": 5,
        "clock_selection": "minimum_network_delay",
        "planned_start_coordinator_unix_ns": planned_coordinator_ns,
    }
    result["_remote_result_delivery"] = {
        "result_source": result_source,
        "transport_retries": transport_retries,
        "max_transport_recovery_attempts": max_transport_recovery_attempts,
    }
    return result


class CoordinatedBenchmarkTest(unittest.TestCase):
    def test_resume_refuses_missing_shard_without_launching_work(self) -> None:
        resume_token = "private-coordinator-resume-token-32-characters"
        agent_key = "separate-agent-api-key-material"
        profiles = [
            {
                "agent_id_sha256": "d" * 64,
                "agent_id_persisted": False,
                "clock_offset_agent_minus_coordinator_ns": 10,
                "clock_network_delay_ns": 20,
                "clock_uncertainty_ns": 10,
                "clock_sample_count": 3,
                "clock_selection": "minimum_network_delay",
            },
            {
                "agent_id_sha256": "e" * 64,
                "agent_id_persisted": False,
                "clock_offset_agent_minus_coordinator_ns": -10,
                "clock_network_delay_ns": 20,
                "clock_uncertainty_ns": 10,
                "clock_sample_count": 3,
                "clock_selection": "minimum_network_delay",
            },
        ]
        benchmark_args = ["--mode", "mock", "--num-requests", "2"]
        configuration_hash = build_coordinator_configuration_fingerprint(
            client_count=2,
            benchmark_args=benchmark_args,
            timeout_seconds=30,
            max_start_skew_ms=100,
            max_clock_uncertainty_ms=25,
            agent_run_recovery_attempts=1,
            agent_hashes=["d" * 64, "e" * 64],
        )
        manifest = create_coordinator_resume_manifest(
            resume_token=resume_token,
            configuration_sha256=configuration_hash,
            planned_start_coordinator_unix_ns=1_000_000_000,
            clock_profiles=profiles,
            workflow_nonce="a" * 64,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "coordinator-state.json"
            write_coordinator_resume_manifest(state_path, manifest)
            args = SimpleNamespace(
                agent_api_key=agent_key,
                coordinator_state_file=state_path,
                coordinator_resume_token=resume_token,
                clients=2,
                clock_samples=3,
                timeout_seconds=30.0,
                agent_url=["http://127.0.0.1:8001", "http://127.0.0.1:8002"],
                benchmark_args=benchmark_args,
                max_start_skew_ms=100.0,
                max_clock_uncertainty_ms=25.0,
                agent_run_recovery_attempts=1,
                lead_time_ms=500.0,
            )

            def probe(url: str, **_: object) -> dict[str, object]:
                return profiles[0] if url.endswith("8001") else profiles[1]

            def status(url: str, **_: object) -> dict[str, object]:
                if url.endswith("8001"):
                    return {
                        "state": "completed",
                        "result_available": True,
                        "result_source": "cached",
                    }
                return {"state": "missing", "result_available": False}

            with (
                patch("coordinated_benchmark.probe_agent_clock", side_effect=probe),
                patch("coordinated_benchmark.query_agent_run_status", side_effect=status),
                patch("coordinated_benchmark.run_agent_benchmark") as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "no workload was launched"):
                    _run_remote_clients(args)
                run.assert_not_called()

    def test_resume_manifest_is_owner_only_authenticated_and_private(self) -> None:
        resume_token = "private-coordinator-resume-token-32-characters"
        private_prompt = "private coordinator restart prompt"
        private_url = "https://private-agent.example.test"
        private_env = "PRIVATE_COORDINATOR_RESUME_TOKEN"
        agent_hashes = ["d" * 64, "e" * 64]
        configuration_hash = build_coordinator_configuration_fingerprint(
            client_count=2,
            benchmark_args=[
                "--mode",
                "openai",
                "--server-url",
                private_url,
                "--openai-prompt",
                private_prompt,
            ],
            timeout_seconds=30,
            max_start_skew_ms=100,
            max_clock_uncertainty_ms=25,
            agent_run_recovery_attempts=1,
            agent_hashes=agent_hashes,
        )
        clock_profiles = [
            {
                "agent_id_sha256": agent_hashes[0],
                "agent_id_persisted": False,
                "clock_offset_agent_minus_coordinator_ns": 100,
                "clock_network_delay_ns": 20,
                "clock_uncertainty_ns": 10,
                "clock_sample_count": 5,
                "clock_selection": "minimum_network_delay",
            },
            {
                "agent_id_sha256": agent_hashes[1],
                "agent_id_persisted": False,
                "clock_offset_agent_minus_coordinator_ns": -50,
                "clock_network_delay_ns": 40,
                "clock_uncertainty_ns": 20,
                "clock_sample_count": 5,
                "clock_selection": "minimum_network_delay",
            },
        ]
        manifest = create_coordinator_resume_manifest(
            resume_token=resume_token,
            configuration_sha256=configuration_hash,
            planned_start_coordinator_unix_ns=1_000_000_000,
            clock_profiles=clock_profiles,
            workflow_nonce="a" * 64,
        )

        run_id = derive_coordinator_run_id(resume_token, manifest["workflow_nonce"])
        self.assertEqual(len(run_id), 64)
        self.assertEqual(
            run_id,
            derive_coordinator_run_id(resume_token, manifest["workflow_nonce"]),
        )
        self.assertNotIn(resume_token, run_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "coordinator-state.json"
            write_coordinator_resume_manifest(state_path, manifest)
            self.assertEqual(stat.S_IMODE(state_path.stat().st_mode), 0o600)
            loaded = load_coordinator_resume_manifest(state_path, resume_token)
            self.assertEqual(loaded, manifest)
            original_bytes = state_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_coordinator_resume_manifest(state_path, manifest)
            self.assertEqual(state_path.read_bytes(), original_bytes)

            serialized = state_path.read_text(encoding="utf-8")
            for private_value in (
                resume_token,
                private_prompt,
                private_url,
                private_env,
                run_id,
                str(state_path),
                "raw-agent-a",
                "raw-agent-b",
            ):
                self.assertNotIn(private_value, serialized)
            self.assertNotIn('"benchmark_args"', serialized)
            self.assertFalse(loaded["privacy"]["resume_token_persisted"])
            self.assertFalse(loaded["privacy"]["run_id_persisted"])
            self.assertFalse(loaded["privacy"]["benchmark_arguments_persisted"])
            self.assertFalse(loaded["privacy"]["agent_urls_persisted"])
            self.assertFalse(loaded["privacy"]["result_bodies_persisted"])

            with self.assertRaisesRegex(ValueError, "authentication"):
                load_coordinator_resume_manifest(
                    state_path, "different-private-resume-token-32-chars"
                )

            tampered = json.loads(state_path.read_text(encoding="utf-8"))
            tampered["planned_start_coordinator_unix_ns"] += 1
            state_path.write_text(json.dumps(tampered), encoding="utf-8")
            os.chmod(state_path, 0o600)
            with self.assertRaisesRegex(ValueError, "authentication"):
                load_coordinator_resume_manifest(state_path, resume_token)

    def test_child_coordination_hashes_id_and_records_measured_window(self) -> None:
        raw_run_id = "private-coordination-id"
        coordination = CoordinatedClientConfig(
            run_id=raw_run_id,
            client_index=0,
            client_count=2,
            planned_start_unix_ns=time.time_ns() - 1,
        )

        metrics = run_benchmark(
            MockInferenceClient(seed=11, failure_rate=0),
            BenchmarkConfig(num_requests=2, concurrency=1, retries=0),
            coordination=coordination,
        )

        record = metrics["coordination"]
        self.assertEqual(
            record["run_id_sha256"],
            hashlib.sha256(raw_run_id.encode("utf-8")).hexdigest(),
        )
        self.assertFalse(record["run_id_persisted"])
        self.assertEqual(record["client_index"], 0)
        self.assertEqual(record["client_count"], 2)
        self.assertGreaterEqual(
            record["measured_end_unix_ns"],
            record["measured_start_unix_ns"],
        )
        self.assertNotIn(raw_run_id, json.dumps(metrics))

    def test_coordination_cli_options_are_all_or_nothing(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--coordinated-run-id",
                "run-1",
                "--coordinated-client-index",
                "1",
                "--coordinated-client-count",
                "3",
                "--coordinated-start-unix-ns",
                str(time.time_ns() + 1_000_000_000),
            ],
        ):
            options = parse_args()

        self.assertIsNotNone(options.coordination)
        self.assertEqual(options.coordination.client_index, 1)
        self.assertEqual(options.coordination.client_count, 3)

        with patch(
            "sys.argv",
            ["benchmark.py", "--coordinated-run-id", "incomplete"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_aggregate_validates_clients_and_uses_union_window(self) -> None:
        summary = build_coordinated_summary(
            [
                shard(0, started_ns=1_000_000_000, ended_ns=2_000_000_000),
                shard(
                    1,
                    started_ns=1_100_000_000,
                    ended_ns=2_100_000_000,
                    successful=3,
                    failed=1,
                    attempts=5,
                ),
            ],
            max_start_skew_ms=150,
        )

        self.assertEqual(summary["client_count"], 2)
        self.assertEqual(summary["logical_requests"], 8)
        self.assertEqual(summary["successful_requests"], 7)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["client_attempts"], 9)
        self.assertEqual(summary["retry_attempts"], 1)
        self.assertEqual(summary["window"]["union_duration_seconds"], 1.1)
        self.assertEqual(summary["window"]["overlap_duration_seconds"], 0.9)
        self.assertEqual(summary["window"]["start_skew_ms"], 100.0)
        self.assertEqual(summary["throughput_rps"], 6.3636)
        self.assertEqual(summary["configured_aggregate_request_rate_rps"], 20.0)
        self.assertTrue(summary["coordination_gate"]["passed"])
        self.assertFalse(summary["latency"]["global_percentiles_available"])
        self.assertNotIn("latency_ms", summary)

        serialized = json.dumps(summary)
        self.assertNotIn("private.example.test", serialized)
        self.assertNotIn("sensitive fixture prompt", serialized)
        self.assertNotIn('"server_url":', serialized)

    def test_aggregate_fails_closed_on_start_skew(self) -> None:
        summary = build_coordinated_summary(
            [
                shard(0, started_ns=1_000_000_000, ended_ns=2_000_000_000),
                shard(1, started_ns=1_100_000_000, ended_ns=2_100_000_000),
            ],
            max_start_skew_ms=50,
        )

        self.assertFalse(summary["coordination_gate"]["passed"])
        self.assertIn("start skew", " ".join(summary["coordination_gate"]["failure_reasons"]))

    def test_aggregate_fails_closed_without_overlapping_windows(self) -> None:
        summary = build_coordinated_summary(
            [
                shard(0, started_ns=1_000_000_000, ended_ns=1_100_000_000),
                shard(1, started_ns=1_200_000_000, ended_ns=1_300_000_000),
            ],
            max_start_skew_ms=250,
        )

        self.assertEqual(summary["window"]["overlap_duration_seconds"], 0)
        self.assertFalse(summary["coordination_gate"]["passed"])
        self.assertIn(
            "did not overlap",
            " ".join(summary["coordination_gate"]["failure_reasons"]),
        )

    def test_remote_aggregate_normalizes_clocks_and_uses_conservative_bounds(self) -> None:
        summary = build_remote_coordinated_summary(
            [
                remote_shard(
                    0,
                    normalized_start_ns=1_000_000_000,
                    normalized_end_ns=2_000_000_000,
                    offset_ns=100_000_000,
                    uncertainty_ns=1_000_000,
                    agent_hash="d" * 64,
                    result_source="durable",
                    transport_retries=1,
                ),
                remote_shard(
                    1,
                    normalized_start_ns=1_010_000_000,
                    normalized_end_ns=2_010_000_000,
                    offset_ns=-50_000_000,
                    uncertainty_ns=2_000_000,
                    agent_hash="e" * 64,
                ),
            ],
            max_start_skew_ms=20,
            max_clock_uncertainty_ms=5,
        )

        self.assertEqual(summary["scope"], "authenticated_remote_agents")
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(
            summary["window"]["clock"],
            "coordinator_wall_clock_estimated_from_agent_samples",
        )
        self.assertEqual(summary["window"]["observed_start_skew_ms"], 10)
        self.assertEqual(summary["window"]["start_skew_upper_bound_ms"], 14)
        self.assertEqual(summary["window"]["observed_union_duration_seconds"], 1.01)
        self.assertEqual(summary["window"]["union_duration_upper_bound_seconds"], 1.014)
        self.assertEqual(summary["window"]["observed_overlap_duration_seconds"], 0.99)
        self.assertEqual(summary["window"]["overlap_duration_lower_bound_seconds"], 0.986)
        self.assertEqual(summary["clock_quality"]["max_uncertainty_ms"], 2)
        self.assertEqual(
            summary["result_delivery"],
            {
                "max_transport_recovery_attempts_per_agent": 1,
                "transport_retries": 1,
                "executed_results": 1,
                "cached_results": 0,
                "durable_results": 1,
                "recovered_after_transport_failure": 1,
            },
        )
        self.assertTrue(summary["clock_quality"]["passed"])
        self.assertTrue(summary["coordination_gate"]["passed"])
        self.assertAlmostEqual(
            summary["throughput"]["observed_normalized_rps"], 8 / 1.01, places=4
        )
        self.assertAlmostEqual(
            summary["throughput"]["conservative_lower_bound_rps"],
            8 / 1.014,
            places=4,
        )

        serialized = json.dumps(summary)
        self.assertNotIn('"_remote_agent":', serialized)
        self.assertNotIn('"_remote_result_delivery":', serialized)
        self.assertNotIn("agent.example", serialized)
        self.assertFalse(summary["privacy"]["agent_urls_persisted"])
        self.assertFalse(summary["privacy"]["authorization_persisted"])
        self.assertFalse(summary["privacy"]["clock_challenges_persisted"])

        prometheus = format_coordinated_prometheus(summary)
        self.assertIn("triton_coordinated_clock_uncertainty_max_ms 2", prometheus)
        self.assertIn("triton_coordinated_start_skew_upper_bound_ms 14", prometheus)
        self.assertIn("triton_coordinated_throughput_lower_bound_rps", prometheus)
        self.assertIn("triton_coordinated_agent_transport_retries_total 1", prometheus)
        self.assertIn("triton_coordinated_agent_cached_results_total 0", prometheus)
        self.assertIn("triton_coordinated_agent_durable_results_total 1", prometheus)
        self.assertIn("triton_coordinated_agent_recovered_results_total 1", prometheus)
        self.assertNotIn("d" * 64, prometheus)
        self.assertNotIn("e" * 64, prometheus)

    def test_remote_aggregate_fails_on_clock_quality_and_duplicate_agents(self) -> None:
        first = remote_shard(
            0,
            normalized_start_ns=1_000_000_000,
            normalized_end_ns=2_000_000_000,
            offset_ns=100_000_000,
            uncertainty_ns=6_000_000,
            agent_hash="d" * 64,
        )
        second = remote_shard(
            1,
            normalized_start_ns=1_010_000_000,
            normalized_end_ns=2_010_000_000,
            offset_ns=-50_000_000,
            uncertainty_ns=2_000_000,
            agent_hash="e" * 64,
        )
        summary = build_remote_coordinated_summary(
            [first, second],
            max_start_skew_ms=20,
            max_clock_uncertainty_ms=5,
        )
        self.assertFalse(summary["clock_quality"]["passed"])
        self.assertFalse(summary["coordination_gate"]["passed"])
        self.assertIn(
            "clock uncertainty",
            " ".join(summary["coordination_gate"]["failure_reasons"]),
        )

        second["_remote_agent"]["agent_id_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "unique agent identities"):
            build_remote_coordinated_summary(
                [first, second],
                max_start_skew_ms=20,
                max_clock_uncertainty_ms=10,
            )

    def test_aggregate_rejects_incomplete_or_mismatched_shards(self) -> None:
        first = shard(0, started_ns=1_000_000_000, ended_ns=2_000_000_000)
        with self.assertRaisesRegex(ValueError, "complete client index set"):
            build_coordinated_summary([first], max_start_skew_ms=100)

        second = shard(
            1,
            started_ns=1_000_000_000,
            ended_ns=2_000_000_000,
            config_hash="d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "configuration fingerprint"):
            build_coordinated_summary([first, second], max_start_skew_ms=100)

        duplicate = shard(0, started_ns=1_000_000_000, ended_ns=2_000_000_000)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_coordinated_summary([first, duplicate], max_start_skew_ms=100)

    def test_prometheus_export_contains_aggregate_gate_and_counts(self) -> None:
        summary = build_coordinated_summary(
            [
                shard(0, started_ns=1_000_000_000, ended_ns=2_000_000_000),
                shard(1, started_ns=1_010_000_000, ended_ns=2_010_000_000),
            ],
            max_start_skew_ms=50,
        )

        prometheus = format_coordinated_prometheus(summary)

        self.assertIn("triton_coordinated_clients 2", prometheus)
        self.assertIn('triton_coordinated_requests_total{outcome="success"} 8', prometheus)
        self.assertIn("triton_coordinated_client_attempts_total 8", prometheus)
        self.assertIn("triton_coordinated_start_skew_ms 10", prometheus)
        self.assertIn("triton_coordinated_gate_passed 1", prometheus)
        self.assertNotIn(RUN_HASH, prometheus)

    def test_coordinator_rejects_options_with_conflicting_or_shared_windows(self) -> None:
        for args in (
            ["--output-dir", "private"],
            ["--coordinated-run-id", "nested"],
            ["--telemetry-url", "https://metrics.example.test"],
            ["--telemetry-prometheus", "before.prom"],
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "not supported"):
                    validate_benchmark_args(args)


if __name__ == "__main__":
    unittest.main()
