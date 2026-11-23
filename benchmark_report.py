"""Build privacy-safe trend reports from saved benchmark JSON artifacts.

This module deliberately projects a small, measured subset of benchmark output. It
never copies source paths, endpoints, prompts, configuration, trace identifiers, or
raw telemetry into the generated report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


MAX_INPUTS = 64
MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_MODE_LENGTH = 32
SUPPORTED_MODES = {"mock", "triton", "openai"}
MIN_PAIRED_RUNS = 7
MAX_PAIRED_RUNS = MAX_INPUTS // 2
PAIRED_STATISTICAL_METRICS = (
    "p95_latency",
    "success_rate",
    "throughput",
    "attempt_amplification",
)


def _finite_number(value: object, field: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum:g}")
    return number


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _latency(metrics: dict[str, object]) -> dict[str, float]:
    raw = metrics.get("latency_ms")
    if not isinstance(raw, dict):
        raise ValueError("latency_ms must be an object")
    projected = {
        key: round(_finite_number(raw.get(key), f"latency_ms.{key}", minimum=0.0), 4)
        for key in ("p50", "p95", "p99")
    }
    if not projected["p50"] <= projected["p95"] <= projected["p99"]:
        raise ValueError("latency percentiles must be non-decreasing")
    return projected


def _project_retry(metrics: dict[str, object]) -> dict[str, object] | None:
    raw = metrics.get("retry")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("retry must be an object when present")
    amplification = _finite_number(
        raw.get("client_attempt_amplification"),
        "retry.client_attempt_amplification",
        minimum=1.0,
    )
    projected: dict[str, object] = {
        "client_attempt_amplification": round(amplification, 4),
    }
    for key in (
        "retry_attempts",
        "retried_requests",
        "recovered_requests",
        "exhausted_requests",
    ):
        if key in raw:
            projected[key] = _nonnegative_integer(raw[key], f"retry.{key}")
    return projected


def project_run(metrics: dict[str, object], run_index: int) -> dict[str, object]:
    """Project one benchmark artifact to fields safe for a trend report."""
    if not isinstance(metrics, dict):
        raise ValueError("benchmark artifact must be a JSON object")
    if not isinstance(run_index, int) or run_index < 1:
        raise ValueError("run index must be a positive integer")

    mode = metrics.get("mode")
    if not isinstance(mode, str) or mode not in SUPPORTED_MODES or len(mode) > MAX_MODE_LENGTH:
        raise ValueError("mode must be one of mock, triton, or openai")
    num_requests = _nonnegative_integer(metrics.get("num_requests"), "num_requests")
    successful = _nonnegative_integer(
        metrics.get("successful_requests"), "successful_requests"
    )
    if "failed_requests" in metrics:
        failed = _nonnegative_integer(metrics.get("failed_requests"), "failed_requests")
    else:
        if successful > num_requests:
            raise ValueError("successful_requests cannot exceed num_requests")
        # Older committed examples omitted this redundant field. Derive only the
        # arithmetic complement; no retry or server-side value is inferred.
        failed = num_requests - successful
    if successful + failed != num_requests:
        raise ValueError("successful_requests plus failed_requests must equal num_requests")

    success_rate = _finite_number(
        metrics.get("success_rate"), "success_rate", minimum=0.0, maximum=1.0
    )
    expected_success_rate = successful / num_requests if num_requests else 0.0
    if abs(success_rate - expected_success_rate) > 0.0001:
        raise ValueError("success_rate does not match request outcome counts")

    duration_raw = metrics.get("duration_seconds")
    duration_seconds = (
        None
        if duration_raw is None
        else _finite_number(duration_raw, "duration_seconds", minimum=0.0)
    )
    throughput = _finite_number(metrics.get("throughput_rps"), "throughput_rps", minimum=0.0)

    projected: dict[str, object] = {
        "run_index": run_index,
        "mode": mode,
        "num_requests": num_requests,
        "successful_requests": successful,
        "failed_requests": failed,
        "success_rate": round(success_rate, 4),
        "duration_seconds": (
            round(duration_seconds, 4) if duration_seconds is not None else None
        ),
        "throughput_rps": round(throughput, 4),
        "latency_ms": _latency(metrics),
    }
    retry = _project_retry(metrics)
    if retry is not None:
        projected["retry"] = retry
    return projected


def load_run(path: Path, run_index: int) -> dict[str, object]:
    """Load and validate one bounded benchmark artifact without retaining its path."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect benchmark artifact: {exc}") from exc
    if size > MAX_INPUT_BYTES:
        raise ValueError("benchmark artifact exceeds the input size limit")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read benchmark artifact: {exc}") from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark artifact is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("benchmark artifact must be a JSON object")
    return project_run(decoded, run_index)


def _validate_threshold(value: float, name: str) -> float:
    return _finite_number(value, name, minimum=0.0)


def _percent_change(baseline: float, candidate: float) -> float | None:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else None
    return round(((candidate - baseline) / baseline) * 100.0, 4)


def _binomial_upper_tail(exceedances: int, trials: int) -> float:
    """Return P[X >= exceedances] for X~Binomial(trials, 0.5)."""
    if (
        isinstance(exceedances, bool)
        or not isinstance(exceedances, int)
        or isinstance(trials, bool)
        or not isinstance(trials, int)
        or trials < 1
        or not 0 <= exceedances <= trials
    ):
        raise ValueError("binomial exceedances and trials are invalid")
    numerator = sum(math.comb(trials, count) for count in range(exceedances, trials + 1))
    return numerator / (2**trials)


def _holm_adjusted_p_values(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original input order."""
    if not p_values:
        raise ValueError("Holm adjustment requires at least one p-value")
    validated = [
        _finite_number(value, "p-value", minimum=0.0, maximum=1.0)
        for value in p_values
    ]
    ordered = sorted(enumerate(validated), key=lambda item: (item[1], item[0]))
    adjusted = [1.0] * len(validated)
    running_max = 0.0
    family_size = len(validated)
    for rank, (original_index, raw_value) in enumerate(ordered):
        running_max = max(
            running_max,
            min(1.0, (family_size - rank) * raw_value),
        )
        adjusted[original_index] = running_max
    return adjusted


def _paired_adverse_change(
    metric: str,
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> float | None:
    """Project one paired run change into its positive adverse direction."""
    if metric == "p95_latency":
        baseline_latency = baseline["latency_ms"]
        candidate_latency = candidate["latency_ms"]
        assert isinstance(baseline_latency, dict)
        assert isinstance(candidate_latency, dict)
        delta = _percent_change(
            float(baseline_latency["p95"]), float(candidate_latency["p95"])
        )
        return delta
    if metric == "success_rate":
        return round(
            float(baseline["success_rate"]) - float(candidate["success_rate"]), 4
        )
    if metric == "throughput":
        delta = _percent_change(
            float(baseline["throughput_rps"]), float(candidate["throughput_rps"])
        )
        return None if delta is None else round(-delta, 4)
    if metric == "attempt_amplification":
        baseline_retry = baseline.get("retry")
        candidate_retry = candidate.get("retry")
        if not isinstance(baseline_retry, dict) or not isinstance(candidate_retry, dict):
            return None
        return round(
            float(candidate_retry["client_attempt_amplification"])
            - float(baseline_retry["client_attempt_amplification"]),
            4,
        )
    raise ValueError(f"unsupported paired metric: {metric}")


def _build_paired_statistical_analysis(
    runs: list[dict[str, object]],
    *,
    baseline_count: int,
    significance_level: float,
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    baseline_runs = runs[:baseline_count]
    candidate_runs = runs[baseline_count:]
    threshold_by_metric: dict[str, tuple[float, str]] = {
        "p95_latency": (
            float(thresholds["max_p95_regression_pct"]),
            "percentage increase in p95 latency",
        ),
        "success_rate": (
            float(thresholds["max_success_rate_drop"]),
            "absolute drop in success rate",
        ),
        "throughput": (
            float(thresholds["max_throughput_drop_pct"]),
            "percentage drop in throughput",
        ),
        "attempt_amplification": (
            float(thresholds["max_attempt_amplification_increase"]),
            "increase in client attempt amplification",
        ),
    }

    tests: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for metric in PAIRED_STATISTICAL_METRICS:
        allowed_adverse_change, unit = threshold_by_metric[metric]
        adverse_changes: list[float] = []
        note: str | None = None
        for baseline, candidate in zip(baseline_runs, candidate_runs):
            change = _paired_adverse_change(metric, baseline, candidate)
            if change is None:
                if metric == "attempt_amplification":
                    note = "retry data was not present in every paired run"
                else:
                    note = "a percentage comparison had a zero baseline and nonzero candidate"
                break
            adverse_changes.append(change)

        evaluable = note is None
        exceeding_pairs = (
            sum(change > allowed_adverse_change for change in adverse_changes)
            if evaluable
            else 0
        )
        p_value = (
            _binomial_upper_tail(exceeding_pairs, baseline_count)
            if evaluable
            else 1.0
        )
        raw_p_values.append(p_value)
        tests.append(
            {
                "metric": metric,
                "unit": unit,
                "threshold": allowed_adverse_change,
                "pairs": baseline_count,
                "exceeding_pairs": exceeding_pairs,
                "p_value": p_value,
                "evaluable": evaluable,
                "note": note,
            }
        )

    adjusted_p_values = _holm_adjusted_p_values(raw_p_values)
    for result, adjusted_p_value in zip(tests, adjusted_p_values):
        result["adjusted_p_value"] = adjusted_p_value
        result["significance_level"] = significance_level
        result["regression"] = bool(
            result["evaluable"] and adjusted_p_value <= significance_level
        )

    return {
        "method": "one_sided_exact_binomial_test_of_paired_threshold_exceedances",
        "multiple_comparison_correction": "Holm step-down",
        "family_size": len(PAIRED_STATISTICAL_METRICS),
        "significance_level": significance_level,
        "run_count_per_group": baseline_count,
        "data_unit": "paired per-run aggregate benchmark summaries",
        "pairing": "first N inputs are baseline runs; next N are candidate runs; matching positions form pairs",
        "operator_assumptions": (
            "paired observations are independent across runs; the operator matches model, workload, "
            "configuration, environment, and measurement conditions within each pair"
        ),
        "interpretation": (
            "tests whether more than half of paired runs exceed a configured practical regression "
            "threshold; it is not a request-level confidence interval"
        ),
        "tests": tests,
    }


def _comparison_check(
    *,
    baseline: float,
    candidate: float,
    delta: float | None,
    threshold: float,
    direction: str,
    label: str,
) -> tuple[dict[str, object], str | None]:
    if delta is None:
        return (
            {
                "baseline": baseline,
                "candidate": candidate,
                "delta": None,
                "threshold": threshold,
                "evaluable": False,
                "passed": False,
            },
            f"{label} comparison unavailable because the baseline is zero",
        )
    if direction == "increase":
        passed = delta <= threshold
        reason = (
            f"{label} increased {delta:g}% above {threshold:g}% threshold"
            if not passed
            else None
        )
    else:
        passed = delta >= -threshold
        reason = (
            f"{label} changed {delta:g} below -{threshold:g} threshold"
            if not passed
            else None
        )
    return (
        {
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta,
            "threshold": threshold,
            "evaluable": True,
            "passed": passed,
        },
        reason,
    )


def _retry_check(
    baseline: dict[str, object],
    candidate: dict[str, object],
    threshold: float,
) -> tuple[dict[str, object], float | None, str | None]:
    baseline_retry = baseline.get("retry")
    candidate_retry = candidate.get("retry")
    if not isinstance(baseline_retry, dict) or not isinstance(candidate_retry, dict):
        return (
            {
                "baseline": None,
                "candidate": None,
                "delta": None,
                "threshold": threshold,
                "evaluable": False,
                "passed": True,
                "note": "retry amplification was not present in both artifacts",
            },
            None,
            None,
        )
    baseline_value = baseline_retry["client_attempt_amplification"]
    candidate_value = candidate_retry["client_attempt_amplification"]
    assert isinstance(baseline_value, (int, float))
    assert isinstance(candidate_value, (int, float))
    delta = round(float(candidate_value) - float(baseline_value), 4)
    passed = delta <= threshold
    reason = (
        f"client attempt amplification increased {delta:g} above {threshold:g} threshold"
        if not passed
        else None
    )
    return (
        {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
            "threshold": threshold,
            "evaluable": True,
            "passed": passed,
        },
        delta,
        reason,
    )


def build_trend_report(
    runs: list[dict[str, object]],
    *,
    max_p95_regression_pct: float = 10.0,
    max_success_rate_drop: float = 0.01,
    max_throughput_drop_pct: float = 10.0,
    max_attempt_amplification_increase: float = 0.05,
    paired_baseline_runs: int | None = None,
    paired_significance_level: float = 0.05,
) -> dict[str, object]:
    """Build point comparisons and, when requested, a paired run-level test."""
    if not 2 <= len(runs) <= MAX_INPUTS:
        raise ValueError(f"trend report requires 2 to {MAX_INPUTS} runs")
    thresholds = {
        "max_p95_regression_pct": _validate_threshold(
            max_p95_regression_pct, "max_p95_regression_pct"
        ),
        "max_success_rate_drop": _validate_threshold(
            max_success_rate_drop, "max_success_rate_drop"
        ),
        "max_throughput_drop_pct": _validate_threshold(
            max_throughput_drop_pct, "max_throughput_drop_pct"
        ),
        "max_attempt_amplification_increase": _validate_threshold(
            max_attempt_amplification_increase,
            "max_attempt_amplification_increase",
        ),
    }
    if paired_baseline_runs is not None:
        if (
            isinstance(paired_baseline_runs, bool)
            or not isinstance(paired_baseline_runs, int)
            or not MIN_PAIRED_RUNS <= paired_baseline_runs <= MAX_PAIRED_RUNS
        ):
            raise ValueError(
                f"paired_baseline_runs must be an integer at least {MIN_PAIRED_RUNS} "
                f"and at most {MAX_PAIRED_RUNS}"
            )
        if len(runs) != 2 * paired_baseline_runs:
            raise ValueError(
                f"paired analysis requires exactly 2 times paired_baseline_runs "
                f"({2 * paired_baseline_runs}) input artifacts"
            )
        significance_level = _finite_number(
            paired_significance_level,
            "paired significance level",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 < significance_level < 1.0:
            raise ValueError("paired significance level must be greater than 0 and less than 1")
        thresholds["paired_baseline_runs"] = paired_baseline_runs
        thresholds["paired_significance_level"] = significance_level
    modes = {run.get("mode") for run in runs}
    if len(modes) != 1:
        raise ValueError("all trend runs must use the same benchmark mode")

    comparisons: list[dict[str, object]] = []
    for baseline, candidate in zip(runs, runs[1:]):
        baseline_latency = baseline["latency_ms"]
        candidate_latency = candidate["latency_ms"]
        assert isinstance(baseline_latency, dict)
        assert isinstance(candidate_latency, dict)
        baseline_p95 = baseline_latency["p95"]
        candidate_p95 = candidate_latency["p95"]
        assert isinstance(baseline_p95, (int, float))
        assert isinstance(candidate_p95, (int, float))

        baseline_success = baseline["success_rate"]
        candidate_success = candidate["success_rate"]
        baseline_throughput = baseline["throughput_rps"]
        candidate_throughput = candidate["throughput_rps"]
        assert isinstance(baseline_success, (int, float))
        assert isinstance(candidate_success, (int, float))
        assert isinstance(baseline_throughput, (int, float))
        assert isinstance(candidate_throughput, (int, float))

        p95_delta = _percent_change(float(baseline_p95), float(candidate_p95))
        throughput_delta = _percent_change(
            float(baseline_throughput), float(candidate_throughput)
        )
        success_delta = round(float(candidate_success) - float(baseline_success), 4)

        checks: dict[str, object] = {}
        reasons: list[str] = []
        p95_check, p95_reason = _comparison_check(
            baseline=float(baseline_p95),
            candidate=float(candidate_p95),
            delta=p95_delta,
            threshold=thresholds["max_p95_regression_pct"],
            direction="increase",
            label="p95 latency",
        )
        checks["p95_latency"] = p95_check
        if p95_reason:
            reasons.append(p95_reason)

        success_check, success_reason = _comparison_check(
            baseline=float(baseline_success),
            candidate=float(candidate_success),
            delta=success_delta,
            threshold=thresholds["max_success_rate_drop"],
            direction="drop",
            label="success rate",
        )
        checks["success_rate"] = success_check
        if success_reason:
            reasons.append(success_reason)

        throughput_check, throughput_reason = _comparison_check(
            baseline=float(baseline_throughput),
            candidate=float(candidate_throughput),
            delta=throughput_delta,
            threshold=thresholds["max_throughput_drop_pct"],
            direction="drop",
            label="throughput",
        )
        checks["throughput"] = throughput_check
        if throughput_reason:
            reasons.append(throughput_reason)

        retry_check, retry_delta, retry_reason = _retry_check(
            baseline,
            candidate,
            thresholds["max_attempt_amplification_increase"],
        )
        checks["attempt_amplification"] = retry_check
        if retry_reason:
            reasons.append(retry_reason)

        comparisons.append(
            {
                "from_run": baseline["run_index"],
                "to_run": candidate["run_index"],
                "changes": {
                    "p95_latency_delta_pct": p95_delta,
                    "success_rate_delta": success_delta,
                    "throughput_delta_pct": throughput_delta,
                    "client_attempt_amplification_delta": retry_delta,
                },
                "checks": checks,
                "regression": bool(reasons),
                "regression_reasons": reasons,
            }
        )

    report: dict[str, object] = {
        "schema_version": 2 if paired_baseline_runs is not None else 1,
        "run_count": len(runs),
        "scope": {
            "comparison": "adjacent runs in caller-supplied order; percentile distributions are not merged",
            "latency": "client-observed measured-phase p50, p95, and p99 values from each saved run",
            "throughput": "successful completions per measured benchmark duration from each saved run",
            "retry": "client-observed calls to the benchmark client; not server receipt or service MTTR",
            "privacy": "source paths, endpoints, prompts, outputs, configuration, credentials, raw telemetry, and trace identifiers are omitted",
            "operator_boundary": "comparable workload, model, serving configuration, and measurement conditions remain the operator's responsibility",
        },
        "thresholds": thresholds,
        "runs": runs,
        "comparisons": comparisons,
        "regression": any(comparison["regression"] for comparison in comparisons),
    }
    if paired_baseline_runs is not None:
        analysis = _build_paired_statistical_analysis(
            runs,
            baseline_count=paired_baseline_runs,
            significance_level=float(thresholds["paired_significance_level"]),
            thresholds=thresholds,
        )
        report["paired_statistical_analysis"] = analysis
        report["statistical_regression"] = any(
            result["regression"]
            for result in analysis["tests"]
            if isinstance(result, dict)
        )
    return report


def _format_number(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(report: dict[str, object]) -> str:
    """Render a report without copying any source artifact path."""
    lines = [
        "# Benchmark trend report",
        "",
        "Adjacent saved runs in caller-supplied order. This report compares measured benchmark phase summaries; it does not merge percentile distributions.",
        "",
        f"- Runs: {report['run_count']}",
        f"- Regression gate: {'FAILED' if report['regression'] else 'PASSED'}",
        "- Privacy boundary: source paths, endpoints, prompts, configuration, credentials, raw telemetry, and trace identifiers are omitted.",
        "",
        "## Runs",
        "",
        "| Run | Mode | Requests | Success rate | Throughput (rps) | p50 (ms) | p95 (ms) | p99 (ms) | Attempts/request |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        assert isinstance(run, dict)
        latency = run["latency_ms"]
        assert isinstance(latency, dict)
        retry = run.get("retry")
        amplification = retry.get("client_attempt_amplification") if isinstance(retry, dict) else None
        lines.append(
            "| {run} | {mode} | {requests} | {success} | {throughput} | {p50} | {p95} | {p99} | {attempts} |".format(
                run=run["run_index"],
                mode=run["mode"],
                requests=run["num_requests"],
                success=_format_number(run["success_rate"]),
                throughput=_format_number(run["throughput_rps"]),
                p50=_format_number(latency["p50"]),
                p95=_format_number(latency["p95"]),
                p99=_format_number(latency["p99"]),
                attempts=_format_number(amplification),
            )
        )

    lines.extend(["", "## Adjacent comparisons", ""])
    for comparison in report["comparisons"]:
        assert isinstance(comparison, dict)
        status = "REGRESSION" if comparison["regression"] else "within thresholds"
        changes = comparison["changes"]
        assert isinstance(changes, dict)
        lines.append(
            f"- Run {comparison['from_run']} -> run {comparison['to_run']}: **{status}**; "
            f"p95 Δ { _format_number(changes['p95_latency_delta_pct']) }%, "
            f"success-rate Δ { _format_number(changes['success_rate_delta']) }, "
            f"throughput Δ { _format_number(changes['throughput_delta_pct']) }%, "
            f"attempt-amplification Δ { _format_number(changes['client_attempt_amplification_delta']) }."
        )
        reasons = comparison["regression_reasons"]
        assert isinstance(reasons, list)
        for reason in reasons:
            lines.append(f"  - {reason}")
    analysis = report.get("paired_statistical_analysis")
    if isinstance(analysis, dict):
        lines.extend(
            [
                "",
                "## Paired statistical analysis",
                "",
                f"- Method: {analysis['method']}; correction: {analysis['multiple_comparison_correction']} across {analysis['family_size']} metrics.",
                f"- Paired observations: {analysis['run_count_per_group']} independent run-level pairs; {analysis['data_unit']}.",
                f"- Significance level: {_format_number(analysis['significance_level'])}; statistical regression: {'FAILED' if report['statistical_regression'] else 'not detected'}.",
                f"- Pairing: {analysis['pairing']}.",
                f"- Operator assumptions: {analysis['operator_assumptions']}.",
                "- This is not a request-level confidence interval.",
                "",
                "| Metric | Threshold | Exceeding pairs | Raw p-value | Holm-adjusted p-value | Regression |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for result in analysis["tests"]:
            assert isinstance(result, dict)
            status = "yes" if result["regression"] else "no"
            if not result["evaluable"]:
                status = "unavailable"
            lines.append(
                f"| {result['metric']} | {_format_number(result['threshold'])} {result['unit']} | "
                f"{result['exceeding_pairs']}/{result['pairs']} | "
                f"{_format_number(result['p_value'])} | "
                f"{_format_number(result['adjusted_p_value'])} | {status} |"
            )
            if result.get("note"):
                lines.append(f"  - {result['metric']}: {result['note']}")
    lines.extend(
        [
            "",
            "Retry amplification is client-observed attempt accounting, not a server request count or service recovery-time measurement.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ordered saved benchmark artifacts without publishing private fields."
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        help="Saved benchmark JSON artifact; repeat in chronological/comparison order.",
    )
    parser.add_argument("--output-dir", default="benchmark_trend_reports")
    parser.add_argument("--max-p95-regression-pct", type=float, default=10.0)
    parser.add_argument("--max-success-rate-drop", type=float, default=0.01)
    parser.add_argument("--max-throughput-drop-pct", type=float, default=10.0)
    parser.add_argument("--max-attempt-amplification-increase", type=float, default=0.05)
    parser.add_argument(
        "--paired-baseline-runs",
        type=int,
        help=(
            "Enable a run-level paired test. The first N inputs are baseline runs, "
            "the next N are candidates; N must be at least 7."
        ),
    )
    parser.add_argument(
        "--paired-significance-level",
        type=float,
        help="One-sided significance level for the paired test (default: 0.05).",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 2 when any adjacent comparison fails a configured gate.",
    )
    parser.add_argument(
        "--fail-on-statistical-regression",
        action="store_true",
        help="Exit with status 2 when Holm-adjusted paired analysis detects regression.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 2 <= len(args.inputs) <= MAX_INPUTS:
        raise SystemExit(f"--input must be supplied 2 to {MAX_INPUTS} times")
    if args.fail_on_statistical_regression and args.paired_baseline_runs is None:
        print(
            "benchmark trend report error: --fail-on-statistical-regression requires "
            "--paired-baseline-runs",
            file=sys.stderr,
        )
        return 2
    if args.paired_significance_level is not None and args.paired_baseline_runs is None:
        print(
            "benchmark trend report error: --paired-significance-level requires "
            "--paired-baseline-runs",
            file=sys.stderr,
        )
        return 2
    try:
        runs = [load_run(Path(raw_path), index) for index, raw_path in enumerate(args.inputs, 1)]
        report = build_trend_report(
            runs,
            max_p95_regression_pct=args.max_p95_regression_pct,
            max_success_rate_drop=args.max_success_rate_drop,
            max_throughput_drop_pct=args.max_throughput_drop_pct,
            max_attempt_amplification_increase=args.max_attempt_amplification_increase,
            paired_baseline_runs=args.paired_baseline_runs,
            paired_significance_level=(
                args.paired_significance_level
                if args.paired_significance_level is not None
                else 0.05
            ),
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "benchmark_trend.json"
        markdown_path = output_dir / "benchmark_trend.md"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"benchmark trend report error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2))
    print(f"Saved trend JSON to {json_path}")
    print(f"Saved trend Markdown to {markdown_path}")
    if args.fail_on_regression and report["regression"]:
        return 2
    if args.fail_on_statistical_regression and report["statistical_regression"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
