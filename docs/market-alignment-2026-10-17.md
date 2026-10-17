# Market alignment and Python dependency-audit gate - 2026-10-17

## Current official role sample

Listings were checked on 2026-10-17 against the companies' official Greenhouse feeds/pages. Each role was present in its active feed on the check date. Greenhouse exposes `updated_at`, not an original publication date, for these records. Compensation below is quoted as published; no equity value or total compensation is inferred.

| Role | Official listing | Feed updated | Published compensation | Location / work mode | Recurring requirements relevant to this change |
|---|---|---|---|---|---|
| Anthropic, Performance Engineer, Inference Engine | https://job-boards.greenhouse.io/anthropic/jobs/5418323008 | 2026-09-09 19:09:53 -04:00 | Annual salary: `$350,000 — $850,000 USD` | San Francisco, CA or New York City, NY; company policy expects at least 25% office time, with some roles requiring more | Systems programming and tests; reliability, latency, and throughput; profile and model before changing code; correctness and reproducibility. |
| Datadog, Senior Software Engineer - Bazel Tools | https://careers.datadoghq.com/detail/8095166/ | 2026-09-29 13:18:57 -04:00 | `$244,000 — $305,000 USD` estimated yearly salary | Atlanta, GA; Boston, MA; or New York, NY. The listing does not expose a remote classification. | Build/test infrastructure, Bazel, large codebases, CI/development velocity, and careful changes to shared tooling. |
| Datadog, Senior Software Engineer - Distributed Systems | https://careers.datadoghq.com/detail/3851927/ | 2026-09-29 13:18:57 -04:00 | `$192,000 — $240,000 USD` estimated yearly salary | Boston, MA or New York, NY. The listing does not expose a remote classification. | Backend systems, Go/Java/Rust/C++, durability, low latency, and performance. |

The common signal in this sample is evidence-backed engineering quality: correctness and reliability in systems work, plus tested delivery tooling. These listings are market signals, not claims of applicant fit or production-scale experience.

## Live public evidence inventory

The authenticated GitHub account `WaffleBits` has 21 public repositories. The public profile repository is https://github.com/WaffleBits/WaffleBits. Strong related projects include:

- [`triton-inference-benchmark`](https://github.com/WaffleBits/triton-inference-benchmark): Python inference/load-testing harness with authenticated agents, telemetry, replay/reconciliation fixtures, privacy checks, and reliability gates.
- [`secure-gpu-inference-gateway`](https://github.com/WaffleBits/secure-gpu-inference-gateway): inference gateway demo with policy controls, audit logging, and observability.
- [`deterministic-inference-scheduler`](https://github.com/WaffleBits/deterministic-inference-scheduler): Rust scheduling model with replayable traces and release gates.
- [`triton-kernel-lab`](https://github.com/WaffleBits/triton-kernel-lab): Triton GPU kernels with correctness oracles and measured hardware artifacts.

The benchmark is the best existing home for a delivery-security improvement. Its CI already runs unit and deterministic CLI fixtures; the current workflow did not invoke a Python dependency vulnerability audit on `requirements.txt`. The requirements file is the project's dependency declaration, including a hash-pinned `urllib3` wheel and version-ranged dependencies.

## Evidence map

**Already demonstrated:** CI runs benchmark unit and protocol fixtures. The public workflow at the inspected `main` revision (`d4bb4f4cdb7db7a06850a336476f28bee129fa83`) passed as run [38243517820](https://github.com/WaffleBits/triton-inference-benchmark/actions/runs/38243517820). A local Python 3.13.16 audit of `requirements.txt` reported no known vulnerabilities.

**Present but not enforced:** the repository declares dependencies in `requirements.txt`, but its CI workflow does not scan that file for known vulnerabilities.

**Missing:** a pull-request dependency-audit gate in the benchmark repository. This does not imply broader supply-chain certification or production security.

## Exactly one selected improvement

Add PyPA's `gh-action-pip-audit` to the existing benchmark CI and explicitly audit `requirements.txt`. This is a bounded dependency-security gate that complements the existing test suite and maps directly to recurring delivery-security and correctness requirements. It changes no runtime behavior and makes no new production-scale claim.

## Implementation and acceptance plan

1. Add the official PyPA audit action to `.github/workflows/ci.yml`, with `inputs: requirements.txt` and the supported repository Python runtime (3.13).
2. Update the benchmark README to state the exact CI dependency-audit behavior.
3. Run `pip-audit --requirement requirements.txt` in a clean Python 3.13 container, the full unit suite, every direct CLI fixture listed in CI, Python compilation, and `git diff --check`.
4. Validate the workflow structure and inspect the final diff. Push a feature branch without force, open a PR, and require its CI workflow to pass before publication.

## Implementation and local verification

The workflow now invokes PyPA's `gh-action-pip-audit` v1.1.0 against `requirements.txt` after Python 3.13 setup and before the unit tests. The README describes that exact gate; no runtime code or portfolio claims changed.

Checks were run in a disposable `python:3.13-slim` container on the repository's ARM64 host:

- `pip-audit --requirement requirements.txt` (pip-audit 2.10.1): passed, `No known vulnerabilities found`.
- `python -m pip install -r requirements.txt`: passed for the current dependency resolution.
- `python -m unittest discover -s tests`: 154 tests passed.
- All 14 direct CLI fixture invocations used by the current CI workflow passed, including cancellation for SIGINT and SIGTERM, TLS remote-agent checks, recovery, and privacy assertions.
- `python -m py_compile` for the six production Python modules: passed.
- `git diff --check`: passed.

The host's system Python 3.9.16 cannot resolve the declared `urllib3` wheel because that release requires Python 3.10 or newer; the repository workflow already selects Python 3.13, and verification used that supported runtime.

## Pull-request and publication results

_Pending GitHub push, pull-request CI, and remote read-back._

## Remaining boundary

The audit checks known vulnerabilities in the declared Python dependency graph at CI time. It does not prove the absence of unknown vulnerabilities, dependency provenance beyond existing hashes, runtime hardening, or production operation.
