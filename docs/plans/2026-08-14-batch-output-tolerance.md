# Add run-scoped numeric tolerance to batch-invariance gates

## Market signal

Reviewed on 2026-08-14 from Anthropic's active official Greenhouse feed. Dates are the feed's exposed update timestamps; compensation is quoted rather than inferred.

- Anthropic, [Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008), updated 2026-08-03: **$350,000–$850,000 USD annual salary**; on-site in San Francisco, New York City, or Seattle. The role asks for cross-layer throughput, latency, reliability, and correctness work and specifically calls for correctness gates that distinguish real model-output regressions from noise across hardware backends.
- Anthropic, [Performance Engineer, GPU](https://job-boards.greenhouse.io/anthropic/jobs/4926227008), updated 2026-08-03: **$280,000–$850,000 USD annual salary**; on-site in San Francisco, New York City, or Seattle. It names CUDA/Triton kernels, Nsight profiling, memory-bandwidth optimization, NCCL/NVLink, and production inference bottleneck analysis.
- Anthropic, [Staff+ Software Engineer, Inference Runtime](https://job-boards.greenhouse.io/anthropic/jobs/5257650008), updated 2026-08-03: **$405,000–$485,000 USD annual salary**; remote-friendly with required travel or based in San Francisco, Seattle, or New York City. It asks for a performance-sensitive Rust/Python runtime, accelerator-agnostic validation, canary/shadow/rollback mechanisms, and deterministic or simulation-based testing for hardware-dependent systems.
- Anthropic, [Software Engineer, Infrastructure, Interpretability](https://job-boards.greenhouse.io/anthropic/jobs/5388612008), updated 2026-08-13: **$320,000–$485,000 USD annual salary**; on-site in San Francisco. It asks for Python plus secure distributed/cloud infrastructure, Kubernetes, schedulers, accelerator-fleet management, developer tooling, and observability that catches regressions.

The recurring signal is performance evidence that remains correctness-gated across batching, accelerator, and runtime changes without treating harmless floating-point drift as a release failure.

## Evidence map

### Already demonstrated

- Deterministic isolated-versus-concurrent batch-invariance probes with exact output fingerprints and a fail-closed CLI gate.
- Live Triton output capture, deterministic mock execution, regression gates, and privacy-safe JSON/Prometheus artifacts.
- Correctness-gated Triton kernels in `triton-kernel-lab` and model/backend/accelerator-scoped numeric release policy in `deterministic-inference-scheduler`.

### Present but buried

- Triton output arrays are already available in memory before they are reduced to SHA-256 fingerprints.
- The batch-invariance runner already joins each fixed synthetic input across isolated and concurrent phases, so it is the correct boundary for a run-scoped comparison policy.

### Missing before this change

- Any floating-point byte difference is currently a mismatch, even when every element remains within an explicitly reviewed tolerance.
- The artifact cannot distinguish exact matches, tolerance-accepted numeric drift, structurally incompatible output, and out-of-policy numeric drift.
- A reviewer cannot inspect the configured absolute/relative tolerance or aggregate worst observed error without retaining model outputs.

## Decision

Extend `triton-inference-benchmark`; do not create another repository. Output capture, deterministic probes, CLI gates, JSON/Prometheus publication, and privacy boundaries already live here.

Implement exactly one coherent capability: **run-scoped absolute/relative numeric tolerance for batch-invariance output gates**, while preserving exact comparison as the zero-tolerance default.

## Implementation

1. Capture sorted Triton output metadata, an exact SHA-256 fingerprint, and numeric values in memory. Do not serialize values, tensor bytes, fingerprints, prompts, or outputs.
2. Add `--batch-output-atol` and `--batch-output-rtol`. Both default to zero and require `--batch-invariance-probes` when nonzero.
3. Preserve exact success when fingerprints match. With a nonzero policy, accept a numeric element only when `absolute_error <= atol + rtol * abs(isolated_value)`.
4. Fail closed on output-name, dtype, shape, element-count, non-numeric, or non-finite incompatibility when fingerprints differ.
5. Report policy, exact-match count, tolerance-match count, structural/numeric incompatibility count, and finite aggregate maximum absolute/relative error. Keep raw values and fingerprints out of JSON and Prometheus.
6. Make `--fail-on-batch-variance` use the policy result, while retaining a separate strict `exact_match` signal.
7. Add tests for within-policy drift, out-of-policy drift, structural mismatch, non-finite values, zero-tolerance compatibility, CLI validation, Prometheus fields, and artifact privacy.
8. Exercise the real CLI with deterministic mock probes and a nonzero policy, then inspect the generated JSON and Prometheus artifacts.

## Verification gates

- Full `unittest` discovery and Python bytecode compilation.
- Real mock CLI run with batch-invariance probes, nonzero tolerances, Prometheus output, and the fail-closed gate enabled.
- Existing paced traced OpenAI-compatible fixture and bracketed-telemetry fixture.
- Docker image build and containerized mock CLI execution.
- Supported Python container dependency audit.
- Artifact privacy inspection and `git diff --check`.
- Pull-request CI and post-merge main CI.

## Claim boundary

This compares deterministic synthetic inputs in isolated and concurrent client phases. Tolerances are operator-configured for one run; they are not automatically safe for a model, backend, dtype, or quality target. Passing does not prove semantic equivalence, production correctness, deterministic kernels, traffic isolation, or cross-accelerator parity. The harness retains numeric output values only in process memory and publishes aggregate errors, not model outputs.
