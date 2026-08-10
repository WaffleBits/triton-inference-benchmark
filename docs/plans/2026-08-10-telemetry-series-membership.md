# Verify telemetry series membership without publishing labels

## Market signal

Reviewed on 2026-08-10 from active official company job-board pages and feeds:

- Anthropic, [Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008), published 2026-05-20: **$350,000–$850,000 USD annual salary**; hybrid in San Francisco, New York City, or Seattle. The role calls for cross-layer investigations, root-cause analysis, reliability, correctness, and distributed-system telemetry.
- OpenAI, [Software Engineer, Inference - Performance Optimization](https://jobs.ashbyhq.com/openai/85fceac9-fb8a-4d71-a524-a8e5f1e9b01b), published 2026-04-25: **$295K–$555K plus equity**; San Francisco. The role calls for end-to-end analysis across application, model, and fleet layers, plus profiling, benchmarking, performance models, latency, utilization, and cost tradeoffs.
- OpenAI, [ChatGPT Performance Engineer](https://jobs.ashbyhq.com/openai/38ddaa2c-a490-427a-8457-0e92bf00138c), published 2026-04-15: **$325K–$405K plus equity**; San Francisco, New York City, Seattle, or US remote. The role emphasizes instrumentation, tracing, observability, regression investigation, reliability, and performance testing across application and infrastructure layers.
- Etched, [Software Engineer – Performance Profiling](https://jobs.ashbyhq.com/etched/610c3836-9798-46ea-931a-02bb95b29467), published 2026-01-18: **$150K–$275K plus significant equity**; on-site in San Jose. The role asks for correlated performance events across devices and hosts, precise synchronization, counter collection, tracing, and bottleneck analysis.

The recurring requirement is not merely collecting counters. Performance evidence must remain attributable and comparable while requests, replicas, accelerators, and scrape targets change.

## Evidence map

### Already demonstrated

- The harness brackets cumulative Triton counters around the measured phase.
- Optional in-window sampling reports bounded DCGM gauge distributions.
- Counter resets, missing families, failed scrapes, endpoint authentication, and artifact privacy have explicit tests.

### Present but buried

- Prometheus samples already retain labels in memory while parsing, so the harness has enough information to identify a series without publishing the label values.

### Missing

- Paired counter windows aggregate series without proving that the before and after snapshots contain the same series.
- Sampled GPU windows can combine values after a replica or accelerator disappears or appears.
- The public artifact cannot distinguish stable scrape membership from target churn.

## Decision

Extend this repository rather than create another project. It already owns the telemetry parser, request-window alignment, release gate, sampled gauge window, artifact privacy policy, deterministic fixture, and CI path.

Implement one coherent capability: **privacy-preserving telemetry series-membership verification**. Canonicalize logical metric names and sorted labels in memory, persist only a SHA-256 fingerprint plus series count, reject sampled GPU windows when membership changes, and make configured paired-counter gates fail closed on membership churn.

## Implementation

1. Add a deterministic fingerprint over unique logical metric names and sorted label pairs; metric values are excluded.
2. Normalize supported exporter aliases before hashing so equivalent metric names do not create false churn.
3. Add hashed Triton-counter and GPU-gauge membership summaries without serializing label keys, label values, endpoint data, or raw scrapes.
4. Mark paired counter windows invalid when the before and after fingerprints differ, and make configured telemetry gates fail closed with a non-sensitive reason.
5. Reject sampled GPU gauge windows if any boundary or in-window scrape changes the known GPU series set.
6. Exercise the real CLI against the deterministic HTTP fixture and verify the artifact contains a stable fingerprint but no metric names, target labels, endpoint, token, or raw response.

## Verification gates

- Unit tests for stable fingerprints, target churn, exporter alias normalization, fail-closed gating, and label privacy.
- Full `unittest` discovery and Python bytecode compilation.
- Real CLI run against `tests/telemetry_fixture_server.py` with repeated in-window scrapes.
- Docker image build and containerized mock execution.
- Dependency audit, `git diff --check`, pull-request CI, and main-branch CI.

## Claim boundary

A matching fingerprint proves only that the selected Prometheus series identities were the same across the captured scrapes. It does not prove that a target was healthy, that labels truthfully identify a physical device, that clocks are synchronized, or that unrelated traffic was absent. The deterministic fixture verifies wiring and semantics; it is not a GPU, model, multi-host, or production-fleet measurement.
