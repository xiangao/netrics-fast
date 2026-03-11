"""
netrics-fast: Memory-efficient dyadic regression with DR_bc standard errors.

Fast implementation of Graham's (forthcoming, Handbook of Econometrics)
bias-corrected dyadic-robust variance estimator. Uses chunked O(nK)
scatter-add Hajek projection — avoids materializing the full n×K score matrix.
"""

from .dyadic_regression import dyadic_regression
from .print_coef import print_coef

__all__ = ["dyadic_regression", "print_coef"]
__version__ = "0.1.0"
