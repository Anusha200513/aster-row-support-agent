import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import handle_turn

TEST_QUERIES = [
    "How long can I return an unused backpack?",
    "Where is ORD-1007?",
    "What are the warranty periods for bags and drinkware?",
    "Are Breeze Tumbler components dishwasher safe?",
]


def main():
    # Ensure stdout/stderr handles UTF-8 on Windows consoles
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print("ASTER & ROW CUSTOMER SUPPORT AGENT — MANUAL TEST SUITE")
    print("=" * 80)

    for idx, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n[{idx}/{len(TEST_QUERIES)}] USER QUESTION:")
        print(f"  \"{query}\"")
        print("-" * 80)

        result = handle_turn(session_id=f"manual-test-{idx}", user_message=query)

        print("FINAL ANSWER:")
        print(result["answer"])
        print("\nSOURCES CITED:")
        if result["sources"]:
            for src in result["sources"]:
                print(f"  - {src}")
        else:
            print("  (None)")

        print("\nTOOL CALLS:")
        if result["tool_calls"]:
            for tc in result["tool_calls"]:
                print(f"  - Tool: {tc['name']} | Args: {json.dumps(tc['arguments'])}")
        else:
            print("  (None)")

        print(f"\nHANDOFF REQUIRED: {result['handoff']}")
        print("=" * 80)


if __name__ == "__main__":
    main()
