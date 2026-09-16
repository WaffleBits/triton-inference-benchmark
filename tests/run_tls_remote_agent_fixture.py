"""Exercise authenticated remote-agent coordination over verified HTTPS."""

from __future__ import annotations

import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_KEY_ENV = "BENCHMARK_TLS_AGENT_KEY"
API_KEY = "tls-agent-fixture-key-32-characters-long"
PROMPT = "private TLS agent fixture prompt"
TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-01$"
)
SERIALIZED_TRACEPARENT_PATTERN = re.compile(
    r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}"
)


OPENSSL_CONFIG = """\
[req]
prompt = no
distinguished_name = distinguished_name
x509_extensions = extensions

[distinguished_name]
CN = localhost

[extensions]
subjectAltName = @alt_names
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,digitalSignature,keyEncipherment

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
"""


def wait_for_port_file(path: Path, process: subprocess.Popen[str]) -> int:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            return int(path.read_text(encoding="utf-8"))
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"fixture process exited before readiness with status {return_code}"
            )
        time.sleep(0.05)
    raise RuntimeError("fixture process did not become ready")


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def create_certificate(directory: Path, stem: str) -> tuple[Path, Path]:
    config = directory / f"{stem}.openssl.cnf"
    certificate = directory / f"{stem}.crt"
    key = directory / f"{stem}.key"
    config.write_text(OPENSSL_CONFIG, encoding="utf-8")
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-config",
            str(config),
        ],
        check=True,
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.chmod(key, 0o600)
    return certificate, key


def assert_wrong_key_is_rejected(agent_url: str, ca_file: Path) -> None:
    body = json.dumps(
        {"schema_version": 1, "challenge": "a" * 64}, separators=(",", ":")
    ).encode("utf-8")
    request = urllib.request.Request(
        agent_url + "/v1/clock",
        data=body,
        headers={
            "Authorization": "Bearer definitely-the-wrong-agent-key",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    context = ssl.create_default_context(cafile=str(ca_file))
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    try:
        opener.open(request, timeout=3)
    except urllib.error.HTTPError as exc:
        assert exc.code == 401, exc.code
    else:
        raise AssertionError("TLS agent accepted an invalid bearer key")


def coordinator_command(
    result_dir: Path,
    agent_urls: list[str],
    endpoint: str,
    ca_file: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "coordinated_benchmark.py"),
        "--clients",
        "2",
        "--output-dir",
        str(result_dir),
        "--lead-time-ms",
        "500",
        "--max-start-skew-ms",
        "300",
        "--max-clock-uncertainty-ms",
        "100",
        "--clock-samples",
        "5",
        "--timeout-seconds",
        "30",
        "--agent-run-recovery-attempts",
        "0",
        "--agent-url",
        agent_urls[0],
        "--agent-url",
        agent_urls[1],
        "--agent-api-key-env",
        API_KEY_ENV,
        "--agent-ca-file",
        str(ca_file),
        "--",
        "--mode",
        "openai",
        "--server-url",
        endpoint,
        "--model-name",
        "tls-fixture-model",
        "--openai-prompt",
        PROMPT,
        "--num-requests",
        "4",
        "--concurrency",
        "1",
        "--retries",
        "0",
        "--request-rate-rps",
        "10",
        "--propagate-trace-context",
        "--fail-on-trace-context-gap",
        "--prometheus",
    ]


def main() -> None:
    fixture_process: subprocess.Popen[str] | None = None
    agent_processes: list[subprocess.Popen[str]] = []
    with tempfile.TemporaryDirectory(prefix="tls-remote-agent-fixture-") as temp_dir:
        temp_path = Path(temp_dir)
        fixture_port_path = temp_path / "fixture.port"
        trace_path = temp_path / "fixture.trace"
        result_dir = temp_path / "results"
        wrong_result_dir = temp_path / "wrong-ca-results"
        certificate, key = create_certificate(temp_path, "trusted")
        wrong_certificate, _ = create_certificate(temp_path, "wrong")
        child_env = dict(os.environ)
        child_env[API_KEY_ENV] = API_KEY
        try:
            fixture_process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests" / "openai_fixture_server.py"),
                    "--port-file",
                    str(fixture_port_path),
                    "--trace-file",
                    str(trace_path),
                ],
                cwd=ROOT,
                text=True,
            )
            target_port = wait_for_port_file(fixture_port_path, fixture_process)
            endpoint = f"http://127.0.0.1:{target_port}"

            agent_urls: list[str] = []
            for index in range(2):
                port_path = temp_path / f"agent-{index}.port"
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "remote_agent.py"),
                        "--listen-host",
                        "127.0.0.1",
                        "--port",
                        "0",
                        "--port-file",
                        str(port_path),
                        "--agent-id",
                        f"tls-agent-{index}",
                        "--api-key-env",
                        API_KEY_ENV,
                        "--child-timeout-seconds",
                        "30",
                        "--tls-cert-file",
                        str(certificate),
                        "--tls-key-file",
                        str(key),
                    ],
                    cwd=ROOT,
                    env=child_env,
                    text=True,
                )
                agent_processes.append(process)
                port = wait_for_port_file(port_path, process)
                agent_urls.append(f"https://localhost:{port}")

            assert_wrong_key_is_rejected(agent_urls[0], certificate)
            command = coordinator_command(result_dir, agent_urls, endpoint, certificate)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=child_env,
                text=True,
                capture_output=True,
                timeout=40,
            )
            if completed.returncode:
                raise RuntimeError(
                    "TLS remote coordinator CLI failed\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            aggregate_path = result_dir / "coordinated_benchmark.json"
            prometheus_path = result_dir / "coordinated_benchmark.prom"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            prometheus = prometheus_path.read_text(encoding="utf-8")
            serialized = aggregate_path.read_text(encoding="utf-8") + prometheus
            transport = aggregate["agent_protocol"]["transport_security"]
            assert transport == {
                "https_agent_count": 2,
                "loopback_http_agent_count": 0,
                "certificate_verification": "explicit_ca_file",
                "tls_minimum_version": "TLSv1.2",
                "cleartext_non_loopback_rejected": True,
            }
            assert aggregate["scope"] == "authenticated_remote_agents"
            assert aggregate["client_count"] == 2
            assert aggregate["logical_requests"] == 8
            assert aggregate["successful_requests"] == 8
            assert aggregate["failed_requests"] == 0
            assert aggregate["coordination_gate"]["passed"] is True
            assert aggregate["privacy"]["agent_urls_persisted"] is False
            assert aggregate["privacy"]["authorization_persisted"] is False

            traceparents = trace_path.read_text(encoding="utf-8").splitlines()
            assert len(traceparents) == 8, traceparents
            assert len(set(traceparents)) == 8, traceparents
            assert all(TRACEPARENT_PATTERN.fullmatch(value) for value in traceparents)
            assert SERIALIZED_TRACEPARENT_PATTERN.search(serialized) is None
            for private_value in (
                str(certificate),
                str(key),
                str(endpoint),
                PROMPT,
                API_KEY,
                API_KEY_ENV,
                *agent_urls,
                str(result_dir),
            ):
                assert private_value not in serialized, private_value
            assert "Authorization" not in serialized
            assert "triton_coordinated_https_agents 2" in prometheus
            assert "triton_coordinated_loopback_http_agents 0" in prometheus
            assert (
                "triton_coordinated_agent_certificate_verification"
                "{mode=\"explicit_ca_file\"} 1"
            ) in prometheus
            assert "triton_coordinated_cleartext_non_loopback_rejected 1" in prometheus
            assert "triton_coordinated_gate_passed 1" in prometheus
            assert sorted(path.name for path in result_dir.iterdir()) == [
                "coordinated_benchmark.json",
                "coordinated_benchmark.prom",
            ]

            wrong = subprocess.run(
                coordinator_command(
                    wrong_result_dir, agent_urls, endpoint, wrong_certificate
                ),
                cwd=ROOT,
                env=child_env,
                text=True,
                capture_output=True,
                timeout=15,
            )
            assert wrong.returncode != 0
            assert trace_path.read_text(encoding="utf-8").splitlines() == traceparents

            print(
                json.dumps(
                    {
                        "agent_services": aggregate["client_count"],
                        "https_agent_count": transport["https_agent_count"],
                        "certificate_verification": transport[
                            "certificate_verification"
                        ],
                        "logical_requests": aggregate["logical_requests"],
                        "successful_requests": aggregate["successful_requests"],
                        "unique_traceparents": len(set(traceparents)),
                        "wrong_ca_rejected": True,
                        "wrong_ca_target_requests_added": 0,
                        "wrong_bearer_rejected": True,
                        "coordination_gate": aggregate["coordination_gate"]["passed"],
                    },
                    indent=2,
                )
            )
        finally:
            for process in agent_processes:
                stop_process(process)
            stop_process(fixture_process)


if __name__ == "__main__":
    main()
