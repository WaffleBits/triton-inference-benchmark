# Plan: validate server-reported streaming token usage

Date: 2026-10-09
Repository: `WaffleBits/triton-inference-benchmark`
Branch: `feat/validate-stream-usage-20261009`

## Current role sample (official listings checked 2026-10-09)

| Role | Official listing | Published / updated | Compensation as listed | Location / constraints | Recurring requirements relevant to the evidence map |
|---|---|---|---|---|---|
| OpenAI, Software Engineer, Model Runtime | https://jobs.ashbyhq.com/openai/ec317080-e2d2-4a73-93e6-e0a9ae6fdf96 | Published 2026-08-24 (`publishedAt`); listed in the live feed on 2026-10-09 | `$266K – $445K • Offers Equity` | San Francisco; Hybrid (Ashby `workplaceType`) | Runtime correctness and reliability; profiling, observability, benchmarking, performance modeling; latency, throughput, memory efficiency and utilization; quantitatively trustworthy measurement. |
| Anthropic, Performance Engineer, Inference Systems | https://job-boards.greenhouse.io/anthropic/jobs/5224564008 | Original publication date not exposed in the Greenhouse feed; updated 2026-08-21 (`updated_at`) | `Annual Salary: $350,000 — $850,000 USD` | San Francisco, New York City, or Seattle; at least 25% office attendance under the general hybrid policy | Cross-layer throughput/latency/reliability/correctness analysis, output-correctness evaluation, telemetry and modeling, regression detection, and explicit measurement coverage. |
| Anthropic, Research Engineer / Performance Engineer, RL Distributed Systems | https://job-boards.greenhouse.io/anthropic/jobs/5438030008 | Original publication date not exposed in the Greenhouse feed; updated 2026-09-29 (`updated_at`) | `Annual Salary: $500,000 — $850,000 USD` | San Francisco, New York City, or Seattle; at least 25% office attendance under the general hybrid policy | Python plus a systems language; distributed-systems failure/recovery, scheduling, observability, and quantitative throughput/latency/resource reasoning. |

The OpenAI listing was active (`isListed: true`) in Ashby's live feed. Compensation is quoted from the official listing, with equity not converted to total compensation. The Anthropic dates are feed update dates: its Greenhouse feed exposed `updated_at` but not original publication dates. Compensation and location come from the official Greenhouse content/feed; no application deadline was exposed.

## Public evidence inventory and map

The authenticated GitHub account `WaffleBits` was inspected through the GitHub API: 21 public repositories, including the TypeScript/Astro profile, the Rust `deterministic-inference-scheduler`, `triton-kernel-lab`, `triton-inference-benchmark`, and `secure-gpu-inference-gateway`. The benchmark's public `main` is `ec7707e` (content-addressed qualification manifests); its latest observed `main` CI run, 36414084754, passed. No open PR existed in the benchmark repository at selection time. Unrelated open work in the profile repository (PR #62) and gateway repository (PRs #16 and #17) is outside this change and will not be modified.

**Already demonstrated:** the benchmark reports streaming TTFT, inter-chunk timing (not token timing), server-reported output tokens, and only computes server-reported output-token throughput when all successful streaming requests contain token usage. It has synthetic SSE fixtures, privacy checks, saved trend gates, and content-addressed input provenance.

**Present but incomplete:** `OpenAICompatibleStreamingClient.infer` accepts `completion_tokens` using `isinstance(value, int)`. Python treats booleans as integers, and negative integers also pass. The resulting value can enter the reported-token aggregate, derived token rate, and downstream cost model despite violating the non-negative integer semantics of a token count.

**Missing / not claimed:** no real model fleet, live customer traffic, hardware-scale production operation, or general tokenization ground truth is available in this repository. The work will validate only the type/range contract of server-reported usage and its treatment in this local deterministic client fixture.

## Exactly one selected gap

Fail closed on malformed server-reported streaming completion-token counts. Accept only exact non-negative integers (including zero; reject booleans, negatives, floats, strings, and other types). Preserve valid throughput only when usage is valid and present for every successful streaming request; report invalid and missing request counts separately. Do not infer tokens from streamed text chunks or output bytes.

## Implementation and acceptance plan

1. Add failing unit tests for boolean, negative, and non-integral usage values and mixed valid/invalid coverage.
2. Add a deterministic local SSE fixture mode and exercise the actual `benchmark.py` CLI, verifying that malformed usage produces no server-reported token throughput and is explicitly counted in the JSON artifact.
3. Implement strict usage validation with an additive privacy-safe artifact field; update the README's existing streaming-measurement description and this plan with observed results.
4. Run the complete repository unit suite, every canonical CI fixture, `pip-audit -r requirements.txt` on Python 3.13, compile checks, and `git diff --check`; inspect generated JSON for numeric semantics and absence of prompt, output, endpoint, credential, and trace data.
5. Push a feature branch and open a PR only after local checks pass. Require GitHub Actions to pass. Do not modify the profile or resume before the supporting source change is merged; do not merge without explicit authorization.
