"""
Coefficient table printer.

Based on Bryan S. Graham's ipt.print_coef (July 2017, revised September 2018).
Adapted with attribution for netrics-fast.
"""

import numpy as np


def print_coef(beta, vcov, var_names=None, alpha=0.05):
    """
    Print a formatted coefficient table with standard errors and confidence intervals.

    Parameters
    ----------
    beta : array-like, shape (K,) or (K, 1)
        Estimated coefficients.
    vcov : array-like, shape (K, K)
        Estimated variance-covariance matrix.
    var_names : list of str, optional
        Variable names. Defaults to X_0, X_1, ...
    alpha : float
        Significance level for confidence intervals.
    """
    try:
        from scipy.stats import norm
        crit = norm.ppf(1 - alpha / 2)
    except ImportError:
        # Fallback: z_{0.025} = 1.96 for default alpha=0.05
        crit = 1.959963984540054 if alpha == 0.05 else None
        if crit is None:
            raise ImportError("scipy required for non-default alpha")

    beta = np.asarray(beta).ravel()
    vcov = np.asarray(vcov)
    K = len(beta)

    if var_names is None:
        var_names = [f"X_{k}" for k in range(K)]

    print()
    print(f"Independent variable       Coef.    ( Std. Err.)     "
          f"({1 - alpha:.2f} Confid. Interval )")
    print("-" * 91)

    for k, name in enumerate(var_names):
        se = np.sqrt(vcov[k, k])
        lo = beta[k] - crit * se
        hi = beta[k] + crit * se
        print(f"{name:<25s}{beta[k]:10.6f} ({se:10.6f})     ({lo:10.6f} ,{hi:10.6f})")

    print()
    print("-" * 91)
