# Plan: content-addressed qualification evidence

## Selected gap

The benchmark already produces privacy-safe adjacent-run regression reports, but a
reviewer cannot verify that a saved report still corresponds to the exact ordered
JSON artifacts used to create it. The current report omits paths and content by
design; it does not provide a separate, content-addressed qualification record.

This is a bounded gap against recurring infrastructure requirements for long-lived
qualification workflows: artifact management, provenance, regression testing, and
reproducible results. It is not a claim of multi-host execution or production
operation.

## Implementation

1. Add `qualification_manifest.py` with `create` and `verify` commands.
2. Reconstruct the privacy-safe trend report from the supplied ordered artifacts
   before creating a manifest. Fail closed if the report, run order, thresholds,
   or schema do not match.
3. Record only SHA-256 digests, byte counts, run indexes, report gate status, and
   bounded schema metadata. Do not persist source paths, prompts, endpoints,
   credentials, outputs, raw telemetry, or report contents.
4. Add unit tests for validation, privacy, tamper detection, and a real CLI fixture
   that runs two mock benchmarks, the trend-report CLI, manifest creation, and
   verification.
5. Document the workflow and its evidence boundary in `README.md` and
   `DESIGN.md`, then run the fixture from CI.

## Verification boundary

The manifest proves byte identity and safe re-derivation of the saved trend report
for the supplied local artifacts. It does not prove that the artifacts came from a
particular server, model, accelerator, physical host, or production workload. A
manifest with a failed trend regression is rejected when `--require-pass` is used.
