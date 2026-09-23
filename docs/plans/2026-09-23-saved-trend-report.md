# Saved benchmark trend report — 2026-09-23

## Primary-source role evidence

Current official Greenhouse board records were read on 2026-09-23. `first_published`
and `updated_at` below are feed fields, not inferred publication dates.
Compensation is copied exactly from the postings; equity and total compensation are
not estimated.

| Role | Canonical listing | First published | Feed updated | Published compensation | Location / constraints | Repeated requirements relevant here |
|---|---|---:|---:|---|---|---|
| Anthropic, Software Engineer, Research Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5283063008 | 2026-07-06 | 2026-09-09 | `$405,000-$625,000 USD` annual salary | San Francisco or New York City; at least 25% office time | Distributed research infrastructure, reliability, scalability, cloud infrastructure, quantitative operational evidence |
| Anthropic, Performance Engineer, Inference Systems | https://job-boards.greenhouse.io/anthropic/jobs/5224564008 | 2026-05-20 | 2026-08-21 | `$350,000-$850,000 USD` annual salary | San Francisco, New York City, or Seattle; at least 25% office time | Cross-layer throughput, latency, reliability, correctness, telemetry, regression investigation, Python and data analysis |
| Anthropic, Staff + Senior Software Engineer, Inference Deployment | https://job-boards.greenhouse.io/anthropic/jobs/5285557008 | 2026-06-29 | 2026-08-21 | `$320,000-$485,000 USD` annual salary | San Francisco, New York City, or Seattle; at least 25% office time | Distributed serving, routing, orchestration, deployment safety, observability, Python or Rust |
| Together AI, AI Infrastructure Systems Engineer | https://job-boards.greenhouse.io/togetherai/jobs/5138540007 | 2026-05-14 | 2026-09-09 | `$190,000 - $270,000 + equity + benefits` US base salary | San Francisco; no remote option exposed in the feed | Fleet automation, validation, diagnosis/remediation, GPU availability/reliability, Linux, Kubernetes, Python/Go/Rust |

The Anthropic records state that applications are reviewed on a rolling basis and
expose no deadline. The Together AI record exposes no deadline. These are
production-role signals, not claims that this repository demonstrates their scale
or professional level.

## Evidence map

### Already demonstrated

- Repeatable mock, Triton-style, and OpenAI-compatible benchmark execution.
- Latency percentiles, completion throughput, success/failure accounting, retry
  amplification, request-path and lifecycle gates, telemetry summaries, and
  regression comparison against one saved baseline.
- Authenticated agent coordination, explicit TLS trust, bounded result recovery,
  coordinator restart reconciliation, privacy-safe artifacts, and deterministic
  local fixtures with passing CI.

### Present but buried

- `benchmark.py` already writes timestamped JSON result artifacts that contain the
  headline measured-phase metrics needed for a longitudinal view.
- Existing one-baseline regression logic already defines p95 and success-rate
  comparison semantics, but it is only reachable while running a new candidate.
- `docs/OPERATIONS.md` describes release triage, but there is no committed tool
  that turns several saved runs into one inspectable trend artifact.

### Missing

- A privacy-safe report that compares an ordered series of saved benchmark runs.
- A release gate for throughput drops and retry-attempt amplification increases
  across adjacent runs.
- A real CLI fixture proving that saved JSON artifacts can be collected, compared,
  serialized as JSON and Markdown, and rejected when explicit thresholds fail.

## Exactly one selected gap

**Create a saved, privacy-safe benchmark trend report with adjacent-run release
gates.**

This is one reporting extension to `triton-inference-benchmark`, which already owns
metric semantics and artifact privacy. A new repository would duplicate the runner
and weaken the evidence chain. The report will consume only existing result JSON;
it will not scrape endpoints, merge percentile distributions, or manufacture
production-scale measurements.

## Implementation plan

1. Add `benchmark_report.py` with an explicit repeated `--input` interface and
   bounded input count. Keep source paths out of generated artifacts.
2. Validate and project only the measured headline fields: request counts,
   success rate, completion throughput, p50/p95/p99 latency, and client-attempt
   amplification when present. Exclude server URLs, prompts, configuration,
   credentials, trace identifiers, raw telemetry, and child paths.
3. Compare adjacent runs in caller-supplied order. Gate p95 increases, success-rate
   drops, throughput drops, and retry amplification increases using explicit,
   run-scoped thresholds. Fail closed on malformed or non-finite values.
4. Write deterministic JSON and human-readable Markdown reports with a scope note
   distinguishing measured headline values from client-observed retry accounting.
5. Add unit tests first for projection, privacy, change calculations, zero-baseline
   behavior, validation, and gate failures. Add a real CLI fixture that runs the
   existing mock benchmark twice, feeds the resulting JSON files to the report
   command, and checks both output formats.
6. Document the command and claim boundary in `README.md` and `docs/OPERATIONS.md`,
   and run the full repository checks plus the existing fixtures.

## Verification boundary

The implementation will prove reproducible local report generation from saved
benchmark artifacts and explicit adjacent-run gate behavior. It will not prove
causality between runs, server-side or GPU-wide percentile aggregation, traffic
isolation, a universal SLO, multi-host synchronization, production fleet scale, or
that a client retry reached a model server. Operators must keep workload, model,
serving configuration, and measurement conditions comparable before treating a
trend as an engineering finding.
