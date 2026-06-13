"""
Phase 4 — Routing accuracy eval. COSTS ZERO TOKENS.

The router is pure Python (regex + keywords), so we test the function
directly instead of invoking the whole graph. Run from project root:

    python evals/eval_routing.py

Output: accuracy score + a list of every misroute. This number is the
BASELINE that the Phase 5 LLM router must beat.
"""

import json
import sys
from pathlib import Path

# make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_graph import router_node  # noqa: E402  (imports after path setup)


def main():
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    # skip entries you haven't filled in yet
    dataset = [d for d in dataset if not d["question"].startswith("REPLACE")]
    if not dataset:
        print("Dataset is empty — fill in golden_dataset.json first.")
        return

    correct, misroutes = 0, []
    for entry in dataset:
        # router only reads state["question"], so a minimal dict suffices
        result = router_node({"question": entry["question"], "messages": []})
        got = result["route"]
        want = entry["expected_route"]
        if got == want:
            correct += 1
        else:
            misroutes.append((entry["question"], want, got))

    total = len(dataset)
    print(f"\n🧭 Routing accuracy: {correct}/{total} ({correct/total:.0%})\n")

    if misroutes:
        print("Misroutes:")
        for q, want, got in misroutes:
            print(f"  ✗ '{q}'")
            print(f"      expected: {want}   got: {got}")
    else:
        print("No misroutes 🎉")


if __name__ == "__main__":
    main()