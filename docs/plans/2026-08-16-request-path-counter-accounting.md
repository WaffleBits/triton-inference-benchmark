# Request-path counter accounting — 2026-08-16

## Market-backed gap

Current high-compensation inference and infrastructure listings repeatedly ask for cross-layer observability, reliability diagnosis, verification, and failure/retry models. This benchmark now counts harness calls and retry amplification, but it cannot show whether those calls reached an ingress layer, were forwarded to a backend, or completed successfully. Its explicit claim boundary identifies that missing evidence.

## Existing-repository decision

Implement this in `triton-inference-benchmark`. The repository already owns measured request phases, retry accounting, paired Prometheus snapshots, privacy-safe series fingerprints, JSON/Prometheus artifacts, fail gates, and a deterministic transient-failure HTTP fixture. A separate repository would duplicate those boundaries and would not connect layer receipts to the measured run.

## One coherent implementation

Add opt-in request-path accounting over three operator-selected cumulative Prometheus counters:

1. Accept ingress-receipt, backend-receipt, and successful-completion metric names only when all three are configured with a paired telemetry source.
2. Compute integer deltas from the snapshots immediately bracketing the measured phase (or from explicitly supplied before/after files).
3. Reject missing, non-finite, negative, non-integral, reset, duplicate, or membership-changing counter evidence.
4. Correlate aggregate stage deltas with measured client attempts and logical outcomes without serializing metric names or labels.
5. Export stage deltas, path ratios, validity, and a fixed isolated-scope accounting gate to JSON and Prometheus.
6. Exercise one deterministic HTTP 503 absorbed before the backend: five client attempts, five ingress receipts, four backend receipts, and four successful completions for four logical requests.

## Gate and claim boundaries

The opt-in accounting gate checks that ingress receipts equal harness client attempts, stage counts are non-increasing, and successful-completion receipts equal successful logical requests. This exact gate is appropriate only when the selected counter series are isolated to the benchmark run.

Paired counters establish aggregate deltas, not per-request causality. Even a harness-bracketed scrape cannot prove traffic isolation, clock synchronization, process identity, or that a metric's implementation is correct. Operator-supplied snapshot files have unverified timing. The artifact retains only fixed stage names, aggregate values, and SHA-256 membership/name fingerprints; raw metric names, labels, endpoint URLs, prompts, outputs, authorization headers, and trace identifiers remain absent.

## Verification

- Add tests before implementation for valid deltas, gate pass/fail behavior, resets and membership changes, CLI validation, Prometheus output, and artifact privacy.
- Run the complete unit/integration suite on the host and in the repository's Python container.
- Run the actual CLI against the deterministic OpenAI-compatible fixture with one transient failure and inspect JSON/Prometheus output.
- Build and run the packaged Docker image. Pin the image's package installer to
  a patched version if a full installed-environment audit flags the base image's
  bundled installer.
- Run `pip-audit -r requirements.txt`, syntax checks, and `git diff --check`.
- Require pull-request CI and post-merge `main` CI to pass before publishing profile or resume claims.
