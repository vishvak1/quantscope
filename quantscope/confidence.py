# quantscope/confidence.py
# Analysis 4: Confidence as Failure Predictor
# Tests whether the optimised model's confidence score predicts its failures.

import numpy as np


def compute_failure_detection(
    labels, predictions_ref, predictions_opt, confidences_opt,
    thresholds=None
):
    """
    For each confidence threshold, compute:
    - What % of inputs would be flagged (confidence < threshold)
    - What % of disagreements would be caught by flagging
    - Precision: what % of flagged inputs are actual disagreements

    predictions_ref : reference model (full)
    predictions_opt : optimised model (onnx_int8)
    confidences_opt : optimised model confidence scores
    """
    if thresholds is None:
        thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    labels          = np.array(labels)
    predictions_ref = np.array(predictions_ref)
    predictions_opt = np.array(predictions_opt)
    confidences_opt = np.array(confidences_opt)

    is_disagreement = predictions_ref != predictions_opt
    total           = len(labels)
    total_disagreements = is_disagreement.sum()

    results = []
    for thresh in thresholds:
        flagged       = confidences_opt < thresh
        flagged_count = flagged.sum()
        caught        = (flagged & is_disagreement).sum()

        results.append({
            "threshold"              : thresh,
            "inputs_flagged_n"       : int(flagged_count),
            "inputs_flagged_pct"     : round(float(flagged_count / total * 100), 1),
            "disagreements_caught_n" : int(caught),
            "disagreements_caught_pct": round(float(caught / max(total_disagreements, 1) * 100), 1),
            "precision_pct"          : round(float(caught / max(flagged_count, 1) * 100), 1),
        })

    return {
        "thresholds"          : results,
        "total"               : total,
        "total_disagreements" : int(total_disagreements),
    }


def find_optimal_threshold(failure_detection_result, min_recall=0.70):
    """
    Find the threshold that catches at least min_recall % of disagreements
    while flagging the fewest inputs.
    """
    candidates = [
        t for t in failure_detection_result["thresholds"]
        if t["disagreements_caught_pct"] >= min_recall * 100
    ]
    if not candidates:
        return None
    # Among candidates, pick the one that flags fewest inputs
    return min(candidates, key=lambda t: t["inputs_flagged_pct"])


def print_failure_detection_report(result):
    print("Confidence as Failure Predictor")
    print("=" * 70)
    print(f"Total disagreements: {result['total_disagreements']} / {result['total']}")
    print()
    print(
        f"{'Threshold':<12} {'Flagged':>10} {'Flagged%':>10} "
        f"{'Caught':>8} {'Caught%':>9} {'Precision':>10}"
    )
    print("-" * 70)
    for t in result["thresholds"]:
        print(
            f"{t['threshold']:<12.2f} {t['inputs_flagged_n']:>10} "
            f"{t['inputs_flagged_pct']:>9.1f}% {t['disagreements_caught_n']:>8} "
            f"{t['disagreements_caught_pct']:>8.1f}% {t['precision_pct']:>9.1f}%"
        )

    optimal = find_optimal_threshold(result)
    if optimal:
        print(f"\n✅ Optimal threshold (≥70% recall): {optimal['threshold']}")
        print(f"   Catches {optimal['disagreements_caught_pct']}% of disagreements")
        print(f"   Flags only {optimal['inputs_flagged_pct']}% of inputs")