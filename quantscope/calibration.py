# quantscope/calibration.py
# Analysis 3: Confidence Calibration Comparison
# Measures whether model confidence scores reflect actual accuracy.

import numpy as np


def compute_calibration(labels, predictions, confidences, n_bins=10):
    """
    Computes calibration statistics for a model variant.
    Returns per-bin accuracy vs confidence for plotting calibration curves.
    """
    labels      = np.array(labels)
    predictions = np.array(predictions)
    confidences = np.array(confidences)

    bins       = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    bin_results = []
    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lower) & (confidences <= upper)
        if in_bin.sum() == 0:
            continue

        bin_acc  = float((predictions[in_bin] == labels[in_bin]).mean())
        bin_conf = float(confidences[in_bin].mean())
        bin_n    = int(in_bin.sum())

        bin_results.append({
            "bin_lower"  : round(float(lower), 2),
            "bin_upper"  : round(float(upper), 2),
            "bin_conf"   : round(bin_conf, 4),
            "bin_acc"    : round(bin_acc, 4),
            "bin_n"      : bin_n,
            "gap"        : round(abs(bin_conf - bin_acc), 4),
        })

    # Expected Calibration Error (ECE) — weighted avg of gaps across bins
    total  = len(labels)
    ece    = sum(b["bin_n"] / total * b["gap"] for b in bin_results)

    return {
        "bins": bin_results,
        "ece" : round(ece, 4),
        "n"   : total,
    }


def print_calibration_report(variant_name, cal_result):
    print(f"Calibration Report: {variant_name}")
    print("=" * 50)
    print(f"ECE (Expected Calibration Error): {cal_result['ece']:.4f}")
    print(f"  (0 = perfect, higher = more miscalibrated)")
    print()
    print(f"{'Conf Range':<15} {'Mean Conf':>10} {'Accuracy':>10} {'Gap':>8} {'N':>6}")
    print("-" * 55)
    for b in cal_result["bins"]:
        print(
            f"{b['bin_lower']:.2f} – {b['bin_upper']:.2f}   "
            f"{b['bin_conf']:>10.3f} {b['bin_acc']:>10.3f} "
            f"{b['gap']:>8.3f} {b['bin_n']:>6}"
        )