"""CLI script to run the complete Aster & Row AI agent evaluation suite.

Usage:
    python scripts/run_evaluation.py                              # Fast local/mock mode (0 Groq API calls)
    python scripts/run_evaluation.py --live                       # Live end-to-end evaluation using Groq API
    python scripts/run_evaluation.py --live --case CASE_ID        # Target a single case against Groq API
    python scripts/run_evaluation.py --live --cases ID1,ID2       # Target multiple cases against Groq API
"""

from __future__ import annotations

import argparse
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
from evaluation.evaluator import (
    calculate_metrics,
    evaluate_suite,
    load_cases,
    mock_agent_handle_turn,
    reset_mock_sessions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Aster & Row AI Agent Evaluation Runner",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live end-to-end evaluation against Groq API (default is local/mock)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Target a single evaluation case by case_id (e.g. --case final-sale-damaged-exception)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Target multiple comma-separated case IDs (e.g. --cases case1,case2)",
    )
    return parser.parse_args(argv)


def extract_target_case_ids(args: argparse.Namespace) -> list[str] | None:
    """Extract ordered list of target case IDs from --case and --cases arguments."""
    target_ids: list[str] = []
    if args.case:
        cid = args.case.strip()
        if cid and cid not in target_ids:
            target_ids.append(cid)
    if args.cases:
        for cid in args.cases.split(","):
            cid_clean = cid.strip()
            if cid_clean and cid_clean not in target_ids:
                target_ids.append(cid_clean)
    return target_ids if target_ids else None


def print_suite_results(title: str, results: list, metrics: dict, is_live: bool) -> None:
    """Print formatted evaluation suite outcomes and diagnostic instrumentation."""
    mode_tag = "LIVE" if is_live else "LOCAL/MOCK"
    print("=" * 80)
    print(f" {title} [{mode_tag}] ({metrics['passed_cases']}/{metrics['total_cases']} PASSED — {metrics['pass_rate']:.1f}%)")
    print("=" * 80)

    for r in results:
        status_symbol = "✓ PASS" if r.passed else "✗ FAIL"
        latency_str = f"{r.elapsed_ms:.0f}ms"
        tool_str = ", ".join(f"{k}:{v}" for k, v in r.tool_call_breakdown.items()) if r.tool_call_breakdown else "none"
        llm_label = "Groq LLM call(s)" if is_live else "mock LLM call(s)"
        llm_turn_str = f"per turn: {r.llm_calls_per_turn}" if r.user_turns > 1 else f"{r.llm_calls} call(s)"
        print(f"[{status_symbol}] {r.case_id:<36} [{r.category:<20}] ({latency_str})")
        print(f"       -> Diagnostics: {r.user_turns} turn(s) | {r.llm_calls} {llm_label} ({llm_turn_str}) | {r.tool_calls_count} tool call(s) [{tool_str}]")
        if is_live and r.avg_llm_latency_ms > 0:
            print(f"       -> LLM Avg Latency: {r.avg_llm_latency_ms:.0f}ms/call")
        if not r.passed:
            for f in r.failures:
                print(f"       -> Failure: {f}")
            print(f"       -> Actual answer: {r.answer[:120]}...")
            print(f"       -> Actual sources: {r.sources}")
            print(f"       -> Actual tool_calls: {r.tool_calls}")
            print(f"       -> Actual handoff: {r.handoff}")

    llm_col_name = "Groq Calls" if is_live else "Mock Calls"
    print("\n--- Category Breakdown ---")
    print(f"{'Category':<22} {'Passed':<8} {'Total':<8} {'Pass Rate':<11} {'Turns':<8} {llm_col_name:<12} {'Tools':<8} {'Avg Latency':<12}")
    print("-" * 90)
    for cat, c_metrics in metrics["category_metrics"].items():
        llm_count = c_metrics.get("groq_api_calls", 0) if is_live else c_metrics.get("mock_llm_calls", 0)
        print(
            f"{cat:<22} {c_metrics['passed']:<8} {c_metrics['total']:<8} "
            f"{c_metrics['pass_rate']:>6.1f}%    "
            f"{c_metrics.get('user_turns', 0):<8} "
            f"{llm_count:<12} "
            f"{c_metrics.get('tool_calls', 0):<8} "
            f"{c_metrics['avg_latency_ms']:>6.0f}ms"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    """Execute visible and original evaluation suites and report findings."""
    args = parse_args(argv)
    is_live = args.live
    target_ids = extract_target_case_ids(args)
    target_id_set = set(target_ids) if target_ids is not None else None

    visible_path = PROJECT_ROOT / "evaluation" / "visible-cases.json"
    original_path = PROJECT_ROOT / "evaluation" / "original-cases.json"

    print("\n" + "#" * 80)
    print(" ASTER & ROW AI AGENT EVALUATION RUNNER")
    if is_live:
        print(" MODE: LIVE / GROQ (Real API inference via Groq)")
    else:
        print(" MODE: LOCAL / MOCK (Fast offline simulation — 0 Groq API calls)")
    if target_ids:
        print(f" TARGETED CASES ({len(target_ids)}): {', '.join(target_ids)}")
    print("#" * 80 + "\n")

    # Reset mock session state before run
    reset_mock_sessions()

    # Load suites
    visible_cases = load_cases(visible_path, expected_count=15)
    original_cases = load_cases(original_path)

    # Apply case filtering if targeted execution was requested
    if target_id_set is not None:
        visible_cases = [c for c in visible_cases if c.get("id") in target_id_set]
        original_cases = [c for c in original_cases if c.get("id") in target_id_set]
        found_ids = {c.get("id") for c in visible_cases + original_cases if c.get("id")}
        missing_ids = [cid for cid in target_ids if cid not in found_ids]
        if missing_ids:
            print(f"Warning: Specified case ID(s) not found in evaluation suites: {missing_ids}\n")
        if not visible_cases and not original_cases:
            print("Error: No matching evaluation cases found to execute.")
            return 1

    print(f"Loading visible cases from: {visible_path.name}")
    print(f"  -> Successfully loaded {len(visible_cases)} visible cases.\n")

    print(f"Loading original cases from: {original_path.name}")
    print(f"  -> Successfully loaded {len(original_cases)} original cases.\n")

    agent_fn = handle_turn if is_live else mock_agent_handle_turn
    mode_desc = "Live Groq Agent" if is_live else "Local Mock Agent (0 API calls)"
    start_time = time.perf_counter()

    visible_results = []
    visible_metrics = None
    if visible_cases:
        print(f"Evaluating Visible Cases ({len(visible_cases)}) against {mode_desc}...")
        visible_results = evaluate_suite(visible_cases, agent_fn=agent_fn, is_live=is_live)
        visible_metrics = calculate_metrics(visible_results)

    original_results = []
    original_metrics = None
    if original_cases:
        print(f"Evaluating Original Cases ({len(original_cases)}) against {mode_desc}...")
        original_results = evaluate_suite(original_cases, agent_fn=agent_fn, is_live=is_live)
        original_metrics = calculate_metrics(original_results)

    # Combine for Overall Metrics
    all_results = visible_results + original_results
    overall_metrics = calculate_metrics(all_results)
    total_duration = time.perf_counter() - start_time

    # Print Detailed Reports
    if visible_results and visible_metrics:
        print_suite_results("VISIBLE CASES", visible_results, visible_metrics, is_live=is_live)
    if original_results and original_metrics:
        print_suite_results("ORIGINAL CASES", original_results, original_metrics, is_live=is_live)

    print("=" * 80)
    print(" OVERALL EVALUATION SUMMARY & DIAGNOSTICS")
    print("=" * 80)
    print(f"Evaluation Mode:             {'LIVE / GROQ' if is_live else 'LOCAL / MOCK'}")
    print(f"Total Cases:                 {overall_metrics['total_cases']}")
    print(f"Passed Cases:                {overall_metrics['passed_cases']}")
    print(f"Failed Cases:                {overall_metrics['failed_cases']}")
    print(f"Pass Rate:                   {overall_metrics['pass_rate']:.1f}%")
    print(f"Total Evaluation Runtime:    {total_duration:.2f}s")
    print(f"Average Case Latency:        {overall_metrics['avg_latency_ms']:.0f}ms")
    print("-" * 80)
    print(f"Total User Turns:            {overall_metrics['total_user_turns']}")
    if is_live:
        print(f"Total Groq API Calls:        {overall_metrics['total_groq_api_calls']}")
        print(f"Average LLM Calls / Case:    {overall_metrics['avg_llm_calls_per_case']:.2f}")
        print(f"Average LLM Calls / Turn:    {overall_metrics['avg_llm_calls_per_turn']:.2f}")
        if overall_metrics.get('avg_llm_latency_ms', 0) > 0:
            print(f"Average LLM Call Latency:    {overall_metrics['avg_llm_latency_ms']:.0f}ms")
    else:
        print(f"Groq API Calls:              0 (No API calls made in local mode)")
        print(f"Mocked LLM Turns Simulated:  {overall_metrics['total_mock_llm_calls']}")
    print(f"Total Tool Calls:            {overall_metrics['total_tool_calls']}")
    tool_breakdown_str = ", ".join(f"{k}: {v}" for k, v in overall_metrics['tool_call_breakdown'].items())
    print(f"Tool Calls Breakdown:        {tool_breakdown_str if tool_breakdown_str else 'none'}")
    print(f"Average Tool Calls / Case:   {overall_metrics['avg_tool_calls_per_case']:.2f}")
    print("=" * 80 + "\n")

    return 0 if overall_metrics["failed_cases"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
