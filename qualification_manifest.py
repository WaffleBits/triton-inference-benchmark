"""Create and verify privacy-safe, content-addressed benchmark attestations.

A qualification manifest binds a saved trend report to the exact ordered JSON
artifacts used to derive it. It stores hashes and bounded metadata only; source
paths, prompts, endpoints, credentials, outputs, and raw telemetry never enter
the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import benchmark_report


MANIFEST_SCHEMA_VERSION = 2
LEGACY_MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "benchmark_qualification_manifest"


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect artifact: {exc}") from exc
    if size > benchmark_report.MAX_INPUT_BYTES:
        raise ValueError("artifact exceeds the input size limit")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read artifact: {exc}") from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("artifact must be a JSON object")
    return decoded, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256(encoded)


def _load_inputs(
    paths: list[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 2 <= len(paths) <= benchmark_report.MAX_INPUTS:
        raise ValueError(
            f"--input must be supplied 2 to {benchmark_report.MAX_INPUTS} times"
        )

    runs: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for run_index, path in enumerate(paths, 1):
        raw_artifact, raw_bytes = _read_json(path)
        runs.append(benchmark_report.project_run(raw_artifact, run_index))
        artifacts.append(
            {
                "run_index": run_index,
                "sha256": _sha256(raw_bytes),
                "bytes": len(raw_bytes),
            }
        )
    return runs, artifacts


def _report_thresholds(report: dict[str, Any]) -> dict[str, float | int]:
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("trend report thresholds are missing")
    required = (
        "max_p95_regression_pct",
        "max_success_rate_drop",
        "max_throughput_drop_pct",
        "max_attempt_amplification_increase",
    )
    schema_version = report.get("schema_version")
    expected = set(required)
    if schema_version == 2:
        expected.update(("paired_baseline_runs", "paired_significance_level"))
    if set(thresholds) != expected:
        raise ValueError("trend report thresholds do not match the current schema")
    values: dict[str, float | int] = {}
    for key in required:
        value = thresholds.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"trend report threshold is invalid: {key}")
        values[key] = float(value)
    if schema_version == 2:
        pair_count = thresholds.get("paired_baseline_runs")
        significance_level = thresholds.get("paired_significance_level")
        if (
            isinstance(pair_count, bool)
            or not isinstance(pair_count, int)
            or isinstance(significance_level, bool)
            or not isinstance(significance_level, (int, float))
        ):
            raise ValueError("paired trend report thresholds are invalid")
        values["paired_baseline_runs"] = pair_count
        values["paired_significance_level"] = float(significance_level)
    return values


def _rebuild_report(
    report: dict[str, Any],
    runs: list[dict[str, object]],
) -> dict[str, object]:
    schema_version = report.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError("unsupported trend report schema")
    if not isinstance(report.get("run_count"), int):
        raise ValueError("trend report run_count is invalid")
    if report["run_count"] != len(runs):
        raise ValueError("trend report run_count does not match input artifacts")
    if not isinstance(report.get("regression"), bool):
        raise ValueError("trend report regression status is invalid")

    thresholds = _report_thresholds(report)
    paired_baseline_runs = thresholds.get("paired_baseline_runs")
    paired_significance_level = thresholds.get("paired_significance_level", 0.05)
    rebuilt = benchmark_report.build_trend_report(
        runs,
        max_p95_regression_pct=thresholds["max_p95_regression_pct"],
        max_success_rate_drop=thresholds["max_success_rate_drop"],
        max_throughput_drop_pct=thresholds["max_throughput_drop_pct"],
        max_attempt_amplification_increase=thresholds[
            "max_attempt_amplification_increase"
        ],
        paired_baseline_runs=(
            int(paired_baseline_runs) if paired_baseline_runs is not None else None
        ),
        paired_significance_level=float(paired_significance_level),
    )
    if rebuilt != report:
        raise ValueError(
            "trend report does not re-derive from the supplied ordered artifacts"
        )
    return rebuilt


def _validate_report(report_path: Path, input_paths: list[Path]) -> tuple[dict[str, Any], bytes, list[dict[str, object]]]:
    report, report_bytes = _read_json(report_path)
    runs, artifacts = _load_inputs(input_paths)
    _rebuild_report(report, runs)
    return report, report_bytes, artifacts


def create_manifest(
    report_path: Path,
    input_paths: list[Path],
    *,
    require_pass: bool = False,
    require_statistical_pass: bool = False,
) -> dict[str, object]:
    """Create a manifest after re-deriving the report from exact input bytes."""
    report, report_bytes, artifacts = _validate_report(report_path, input_paths)
    regression = report["regression"]
    assert isinstance(regression, bool)
    if require_pass and regression:
        raise ValueError("trend report failed its regression gate")
    statistical_regression = report.get("statistical_regression")
    if require_statistical_pass and statistical_regression is None:
        raise ValueError("statistical pass requires paired statistical analysis")
    if require_statistical_pass and statistical_regression:
        raise ValueError("trend report failed its statistical regression gate")

    thresholds = report["thresholds"]
    assert isinstance(thresholds, dict)
    manifest: dict[str, object] = {
        "schema_version": (
            MANIFEST_SCHEMA_VERSION
            if report["schema_version"] == 2
            else LEGACY_MANIFEST_SCHEMA_VERSION
        ),
        "kind": MANIFEST_KIND,
        "trend_report": {
            "sha256": _sha256(report_bytes),
            "bytes": len(report_bytes),
            "schema_version": report["schema_version"],
            "run_count": report["run_count"],
            "regression": regression,
            "thresholds_sha256": _canonical_sha256(thresholds),
        },
        "input_artifacts": artifacts,
        "gate_policy": {
            "require_trend_pass": require_pass,
            "trend_gate": "passed" if not regression else "failed",
        },
        "scope": {
            "artifact_identity": "SHA-256 of exact input bytes",
            "order": "run_index follows the repeated --input order",
            "paths_persisted": False,
            "content_persisted": False,
            "secrets_persisted": False,
            "server_identity_verified": False,
            "production_measurement_verified": False,
        },
    }
    if report["schema_version"] == 2:
        trend_summary = manifest["trend_report"]
        gate_policy = manifest["gate_policy"]
        assert isinstance(trend_summary, dict)
        assert isinstance(gate_policy, dict)
        trend_summary["statistical_regression"] = bool(statistical_regression)
        gate_policy["require_statistical_pass"] = require_statistical_pass
        gate_policy["statistical_gate"] = (
            "passed" if not statistical_regression else "failed"
        )
    return manifest


def verify_manifest(
    manifest_path: Path,
    report_path: Path,
    input_paths: list[Path],
) -> dict[str, object]:
    """Verify manifest metadata, byte digests, and report re-derivation."""
    manifest, _ = _read_json(manifest_path)
    manifest_schema = manifest.get("schema_version")
    if type(manifest_schema) is not int or manifest_schema not in (
        LEGACY_MANIFEST_SCHEMA_VERSION,
        MANIFEST_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported qualification manifest schema")
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError("unexpected qualification manifest kind")
    policy = manifest.get("gate_policy")
    if not isinstance(policy, dict) or not isinstance(
        policy.get("require_trend_pass"), bool
    ):
        raise ValueError("qualification manifest gate policy is invalid")
    if manifest_schema == MANIFEST_SCHEMA_VERSION and not isinstance(
        policy.get("require_statistical_pass"), bool
    ):
        raise ValueError("qualification manifest statistical gate policy is invalid")

    expected = create_manifest(
        report_path,
        input_paths,
        require_pass=policy["require_trend_pass"],
        require_statistical_pass=policy.get("require_statistical_pass", False)
        if manifest_schema == MANIFEST_SCHEMA_VERSION
        else False,
    )
    if (
        manifest_schema == LEGACY_MANIFEST_SCHEMA_VERSION
        and expected["schema_version"] != LEGACY_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("manifest schema does not match trend report schema")
    if expected != manifest:
        raise ValueError("qualification manifest does not match supplied artifacts")

    report, _ = _read_json(report_path)
    return {
        "verified": True,
        "kind": MANIFEST_KIND,
        "run_count": report["run_count"],
        "trend_gate": policy["trend_gate"],
        **(
            {"statistical_regression": report["statistical_regression"]}
            if report.get("schema_version") == 2
            else {}
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or verify a privacy-safe benchmark qualification manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="derive a manifest from saved artifacts")
    create.add_argument("--report", required=True, help="saved benchmark trend JSON")
    create.add_argument(
        "--input",
        action="append",
        required=True,
        help="saved benchmark JSON; repeat in trend-report order",
    )
    create.add_argument("--output", required=True, help="manifest JSON output path")
    create.add_argument(
        "--require-pass",
        action="store_true",
        help="reject a trend report whose regression gate failed",
    )
    create.add_argument(
        "--require-statistical-pass",
        action="store_true",
        help="reject a paired trend report whose statistical gate failed",
    )

    verify = subparsers.add_parser("verify", help="verify a saved manifest")
    verify.add_argument("--manifest", required=True, help="qualification manifest JSON")
    verify.add_argument("--report", required=True, help="saved benchmark trend JSON")
    verify.add_argument(
        "--input",
        action="append",
        required=True,
        help="saved benchmark JSON; repeat in manifest order",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "create":
            manifest = create_manifest(
                Path(args.report),
                [Path(value) for value in args.input],
                require_pass=args.require_pass,
                require_statistical_pass=args.require_statistical_pass,
            )
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(manifest, indent=2))
            print(f"Saved qualification manifest to {output}")
            return 0

        result = verify_manifest(
            Path(args.manifest),
            Path(args.report),
            [Path(value) for value in args.input],
        )
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"qualification manifest error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
