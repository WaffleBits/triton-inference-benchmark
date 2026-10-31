"""HTTP clients that scrape Prometheus telemetry endpoints."""

from __future__ import annotations

import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from bench_core.config import (
    MAX_TELEMETRY_ENDPOINTS,
    MAX_TELEMETRY_RESPONSE_BYTES,
)


class TelemetrySnapshotClient(Protocol):
    def scrape(self) -> str:
        """Fetch one Prometheus text snapshot or raise an exception."""


class HttpPrometheusTelemetryClient:
    """Opt-in HTTP client for bounded Prometheus text snapshots."""

    def __init__(
        self,
        endpoint_url: str,
        timeout_seconds: float = 10.0,
        bearer_token: str | None = None,
        max_response_bytes: int = MAX_TELEMETRY_RESPONSE_BYTES,
    ) -> None:
        parsed = urllib.parse.urlsplit(endpoint_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("telemetry endpoint must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("telemetry endpoint URL must not contain credentials")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("telemetry endpoint URL contains an invalid port") from exc
        if timeout_seconds <= 0:
            raise ValueError("telemetry timeout must be greater than zero")
        if max_response_bytes <= 0:
            raise ValueError("telemetry response limit must be greater than zero")

        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds
        self.bearer_token = bearer_token
        self.max_response_bytes = max_response_bytes
        self.source_count = 1

    def scrape(self) -> str:
        headers = {
            "Accept": "text/plain",
            "User-Agent": "triton-inference-benchmark/1.0",
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = urllib.request.Request(
            self.endpoint_url,
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read(self.max_response_bytes + 1)
        if len(payload) > self.max_response_bytes:
            raise ValueError(
                "telemetry response exceeded "
                f"{self.max_response_bytes} byte limit"
            )
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("telemetry response must be UTF-8 Prometheus text") from exc


class MultiEndpointPrometheusTelemetryClient:
    """Concurrently combine bounded snapshots from independent metric sources."""

    def __init__(self, clients: tuple[TelemetrySnapshotClient, ...]) -> None:
        if len(clients) < 2:
            raise ValueError("multi-endpoint telemetry requires at least two clients")
        if len(clients) > MAX_TELEMETRY_ENDPOINTS:
            raise ValueError(
                f"telemetry supports at most {MAX_TELEMETRY_ENDPOINTS} endpoints"
            )
        self.clients = clients
        self.source_count = len(clients)

    def scrape(self) -> str:
        with ThreadPoolExecutor(max_workers=self.source_count) as executor:
            futures = [executor.submit(client.scrape) for client in self.clients]
            snapshots = [future.result() for future in futures]

        combined = "\n".join(snapshot.rstrip("\n") for snapshot in snapshots)
        if combined:
            combined = f"{combined}\n"
        if len(combined.encode("utf-8")) > MAX_TELEMETRY_RESPONSE_BYTES:
            raise ValueError(
                "combined telemetry response exceeded "
                f"{MAX_TELEMETRY_RESPONSE_BYTES} byte limit"
            )
        return combined


def _normalize_telemetry_endpoint_urls(
    endpoint_urls: str | tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    urls = (endpoint_urls,) if isinstance(endpoint_urls, str) else tuple(endpoint_urls)
    if not urls:
        raise ValueError("at least one telemetry endpoint is required")
    if len(urls) > MAX_TELEMETRY_ENDPOINTS:
        raise ValueError(
            f"telemetry supports at most {MAX_TELEMETRY_ENDPOINTS} endpoints"
        )
    if len(set(urls)) != len(urls):
        raise ValueError("duplicate telemetry endpoint URLs are not allowed")
    return urls


def build_http_telemetry_client(
    endpoint_urls: str | tuple[str, ...] | list[str],
    timeout_seconds: float,
    api_key_env: str = "",
) -> TelemetrySnapshotClient:
    urls = _normalize_telemetry_endpoint_urls(endpoint_urls)
    bearer_token: str | None = None
    if api_key_env:
        bearer_token = os.environ.get(api_key_env)
        if not bearer_token:
            raise ValueError(
                f"telemetry API key environment variable is missing or empty: {api_key_env}"
            )
    per_source_response_limit = (
        MAX_TELEMETRY_RESPONSE_BYTES
        if len(urls) == 1
        else (MAX_TELEMETRY_RESPONSE_BYTES - len(urls)) // len(urls)
    )
    clients = tuple(
        HttpPrometheusTelemetryClient(
            endpoint_url,
            timeout_seconds=timeout_seconds,
            bearer_token=bearer_token,
            max_response_bytes=per_source_response_limit,
        )
        for endpoint_url in urls
    )
    return (
        clients[0]
        if len(clients) == 1
        else MultiEndpointPrometheusTelemetryClient(clients)
    )
