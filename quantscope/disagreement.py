# quantscope/disagreement.py
# Analysis 1: Disagreement Profiling
# Identifies where model variants disagree and who was right.

import json
import numpy as np
from pathlib import Path


def load_predictions(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


def compute_disagreements(data: dict, variant_a: str, variant_b: str) -> dict:
    """
    Compare predictions between two variants.
    variant_a is treated as the reference (usually 'full').
    """
    labels  = np.array(data["labels"])
    preds_a = np.array(data[variant_a]["predictions"])
    preds_b = np.array(data[variant_b]["predictions"])

    agree    = preds_a == preds_b
    disagree = ~agree

    disagree_idx = np.where(disagree)[0]

    # For each disagreement, classify the outcome
    a_correct_b_wrong = []
    b_correct_a_wrong = []
    both_wrong        = []

    for idx in disagree_idx:
        a_right = preds_a[idx] == labels[idx]
        b_right = preds_b[idx] == labels[idx]

        if a_right and not b_right:
            a_correct_b_wrong.append(int(idx))
        elif b_right and not a_right:
            b_correct_a_wrong.append(int(idx))
        else:
            both_wrong.append(int(idx))

    total = len(labels)

    return {
        "total"               : total,
        "agreement_count"     : int(agree.sum()),
        "disagreement_count"  : int(disagree.sum()),
        "agreement_pct"       : round(float(agree.mean() * 100), 2),
        "disagreement_pct"    : round(float(disagree.mean() * 100), 2),
        "a_correct_b_wrong"   : a_correct_b_wrong,   # indices
        "b_correct_a_wrong"   : b_correct_a_wrong,   # indices
        "both_wrong"          : both_wrong,           # indices
        "a_correct_b_wrong_n" : len(a_correct_b_wrong),
        "b_correct_a_wrong_n" : len(b_correct_a_wrong),
        "both_wrong_n"        : len(both_wrong),
        "variant_a"           : variant_a,
        "variant_b"           : variant_b,
    }


def print_disagreement_report(result: dict):
    a = result["variant_a"]
    b = result["variant_b"]
    n = result["disagreement_count"]

    print(f"Disagreement Analysis: {a} vs {b}")
    print("=" * 50)
    print(f"Total samples         : {result['total']}")
    print(f"Agreement             : {result['agreement_count']} ({result['agreement_pct']}%)")
    print(f"Disagreement          : {result['disagreement_count']} ({result['disagreement_pct']}%)")
    print()
    print(f"Of {n} disagreements:")
    print(f"  {a} correct, {b} wrong : {result['a_correct_b_wrong_n']} ({result['a_correct_b_wrong_n']/max(n,1)*100:.1f}%)  ← optimisation HURT")
    print(f"  {b} correct, {a} wrong : {result['b_correct_a_wrong_n']} ({result['b_correct_a_wrong_n']/max(n,1)*100:.1f}%)  ← optimisation HELPED")
    print(f"  Both wrong            : {result['both_wrong_n']} ({result['both_wrong_n']/max(n,1)*100:.1f}%)  ← both failed differently")
    net = result["a_correct_b_wrong_n"] - result["b_correct_a_wrong_n"]
    print(f"\nNet impact: {'-' if net > 0 else '+'}{abs(net)} correct predictions ({abs(net)/result['total']*100:.1f}%)")