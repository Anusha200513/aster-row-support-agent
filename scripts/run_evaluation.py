"""CLI script to run the complete Aster & Row AI agent evaluation suite."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 encoding on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.agent import handle_turn
from evaluation.evaluator import calculate_metrics, evaluate_suite, load_cases


def print_suite_results(title: str, results: list, metrics: dict) -> None:
    """Print formatted evaluation suite outcomes."""
    print("=" * 80)
    print(f" {title} ({metrics['passed_cases']}/{metrics['total_cases']} PASSED — {metrics['pass_rate']:.1f}%)")
    print("=" * 80)

    for r in results:
        status_symbol = "✓ PASS" if r.passed else "✗ FAIL"
        latency_str = f"{r.elapsed_ms:.0f}ms"
        print(f"[{status_symbol}] {r.case_id:<36} [{r.category:<20}] ({latency_str})")
        if not r.passed:
            for f in r.failures:
                print(f"       -> Failure: {f}")
            print(f"       -> Actual answer: {r.answer[:120]}...")
            print(f"       -> Actual sources: {r.sources}")
            print(f"       -> Actual tool_calls: {r.tool_calls}")
            print(f"       -> Actual handoff: {r.handoff}")

    print("\n--- Category Breakdown ---")
    print(f"{'Category':<26} {'Passed':<10} {'Total':<10} {'Pass Rate':<12} {'Avg Latency':<12}")
    print("-" * 72)
    for cat, c_metrics in metrics["category_metrics"].items():
        print(
            f"{cat:<26} {c_metrics['passed']:<10} {c_metrics['total']:<10} "
            f"{c_metrics['pass_rate']:>6.1f}%     {c_metrics['avg_latency_ms']:>6.0f}ms"
        )
    print()


def main() -> int:
    """Execute visible and original evaluation suites and report findings."""
    visible_path = PROJECT_ROOT / "evaluation" / "visible-cases.json"
    original_path = PROJECT_ROOT / "evaluation" / "original-cases.json"

    print("\n" + "#" * 80)
    print(" ASTER & ROW AI AGENT EVALUATION RUNNER")
    print("#" * 80 + "\n")

    # Load suites
    print(f"Loading visible cases from: {visible_path.name}")
    visible_cases = load_cases(visible_path, expected_count=15)
    print(f"  -> Successfully loaded {len(visible_cases)} visible cases.\n")

    print(f"Loading original cases from: {original_path.name}")
    original_cases = load_cases(original_path)
    print(f"  -> Successfully loaded {len(original_cases)} original cases.\n")

    # Execute Visible Suite
    print("Evaluating Visible Cases against AI Agent...")
    start_time = time.perf_counter()
    visible_results = evaluate_suite(visible_cases, agent_fn=handle_turn)
    visible_metrics = calculate_metrics(visible_results)

    # Execute Original Suite
    print("Evaluating Original Cases against AI Agent...")
    original_results = evaluate_suite(original_cases, agent_fn=handle_turn)
    original_metrics = calculate_metrics(original_results)

    # Combine for Overall Metrics
    all_results = visible_results + original_results
    overall_metrics = calculate_metrics(all_results)
    total_duration = time.perf_counter() - start_time

    # Print Detailed Reports
    print_suite_results("VISIBLE CASES", visible_results, visible_metrics)
    print_suite_results("ORIGINAL CASES", original_results, original_metrics)

    print("=" * 80)
    print(" OVERALL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Cases:     {overall_metrics['total_cases']}")
    print(f"Passed Cases:    {overall_metrics['passed_cases']}")
    print(f"Failed Cases:    {overall_metrics['failed_cases']}")
    print(f"Pass Rate:       {overall_metrics['pass_rate']:.1f}%")
    print(f"Avg Latency:     {overall_metrics['avg_latency_ms']:.0f}ms per case")
    print(f"Total Run Time:  {total_duration:.2f}s")
    print("=" * 80 + "\n")

    return 0 if overall_metrics["failed_cases"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
