# quantscope/__init__.py

from .disagreement import load_predictions, compute_disagreements, print_disagreement_report
from .calibration  import compute_calibration, print_calibration_report
from .confidence   import compute_failure_detection, find_optimal_threshold, print_failure_detection_report
from .categorise   import (
    build_training_vocab, categorise_all,
    compare_categories, print_category_report
)

__version__ = "0.1.0"
__all__ = [
    "load_predictions",
    "compute_disagreements",
    "print_disagreement_report",
    "compute_calibration",
    "print_calibration_report",
    "compute_failure_detection",
    "find_optimal_threshold",
    "print_failure_detection_report",
    "build_training_vocab",
    "categorise_all",
    "compare_categories",
    "print_category_report",
]