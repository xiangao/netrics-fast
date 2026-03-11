"""
Memory-efficient OLS dyadic regression with DR_bc standard errors.

Implements Graham (forthcoming, Handbook of Econometrics) bias-corrected
dyadic-robust variance estimator using a chunked O(nK) scatter-add Hajek
projection that avoids materializing the full n×K score matrix.

Key memory savings vs. netrics.dyadic_regression:
  - OLS via R'R / R'Y (K×K), not storing n×K score matrix
  - Sigma2 via R' diag(e²) R / n (K×K directly)
  - Sigma1 via chunked np.add.at scatter-add over agent IDs
  - Peak memory: O(NK + chunk_size × K) instead of O(nK)
"""

import numpy as np


def dyadic_regression(Y, R, id_i, id_j, directed=False, cov="DR_bc",
                      nocons=False, var_names=None,
                      chunk_size=10_000_000, silent=False):
    """
    OLS dyadic regression with dyadic-robust standard errors.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Dependent variable.
    R : array-like, shape (n, K) or (n,) for single regressor
        Regressors. A constant column is prepended unless nocons=True.
    id_i : array-like, shape (n,)
        Ego agent IDs.
    id_j : array-like, shape (n,)
        Alter agent IDs.
    directed : bool
        True for N(N-1) directed dyads, False for N(N-1)/2 undirected.
    cov : str
        'DR_bc' (bias-corrected dyadic-robust),
        'DR' (dyadic-robust, no bias correction),
        'ind' (heteroskedasticity-robust / independence).
    nocons : bool
        If True, do not prepend a constant column to R.
    var_names : list of str, optional
        Names for the regressors (excluding constant unless nocons=True).
        If None, defaults to X_0, X_1, ... If R is a DataFrame, column
        names are used automatically.
    chunk_size : int
        Rows per chunk for scatter-add (controls peak memory).
    silent : bool
        If True, suppress progress output.

    Returns
    -------
    dict with keys:
        'beta'      : (K,) coefficient vector
        'vcov'      : (K, K) variance-covariance matrix
        'se'        : (K,) standard errors
        'N'         : number of unique agents
        'n'         : number of dyadic observations
        'var_names' : list of variable names
    """
    # ------------------------------------------------------------------
    # Prepare arrays
    # ------------------------------------------------------------------
    Y = np.asarray(Y, dtype=np.float64).ravel()

    # Extract column names from DataFrame before converting to numpy
    if var_names is None and hasattr(R, "columns"):
        var_names = list(R.columns)

    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 1:
        R = R.reshape(-1, 1)

    id_i = np.asarray(id_i)
    id_j = np.asarray(id_j)
    n = len(Y)

    if not nocons:
        R = np.column_stack([np.ones(n, dtype=np.float64), R])

    K = R.shape[1]

    # Agent set
    agents = np.unique(np.concatenate([id_i, id_j]))
    N = len(agents)

    if not silent:
        print(f"n = {n:,} dyads, N = {N:,} agents, K = {K} regressors")

    # ------------------------------------------------------------------
    # OLS: beta = (R'R)^{-1} R'Y  — pure K×K, no n×K intermediates
    # ------------------------------------------------------------------
    RtR = R.T @ R           # K × K
    RtY = R.T @ Y           # K
    beta = np.linalg.solve(RtR, RtY)
    resid = Y - R @ beta    # n

    if not silent:
        print("OLS coefficients computed.")

    # ------------------------------------------------------------------
    # Variance estimation
    # ------------------------------------------------------------------
    iGamma = np.linalg.inv(RtR / n)  # inv(-H/n) where H = -R'R

    if cov == "ind":
        # HC-robust (independence assumption across dyads)
        Sigma2 = (R.T * (resid ** 2)) @ R / n
        vcov = iGamma @ Sigma2 @ iGamma / n
    else:
        # Dyadic-robust: need Sigma1 (Hajek) and Sigma2
        # Sigma2 = (1/n) R' diag(e²) R  — no score matrix needed
        Sigma2 = (R.T * (resid ** 2)) @ R / n

        # Sigma1: Hajek projection via chunked scatter-add
        # Map agent IDs to contiguous indices [0, N)
        agent_to_idx = {a: i for i, a in enumerate(agents)}
        # Vectorized mapping
        map_func = np.vectorize(agent_to_idx.__getitem__)
        idx_i_mapped = map_func(id_i)
        idx_j_mapped = map_func(id_j)

        agent_sum = np.zeros((N, K), dtype=np.float64)
        agent_count = np.zeros(N, dtype=np.int64)

        # Count dyads per agent
        np.add.at(agent_count, idx_i_mapped, 1)
        np.add.at(agent_count, idx_j_mapped, 1)

        # Scatter-add scores in chunks to limit peak memory
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            S_chunk = R[start:end] * resid[start:end, np.newaxis]

            ci = idx_i_mapped[start:end]
            cj = idx_j_mapped[start:end]
            for k in range(K):
                np.add.at(agent_sum[:, k], ci, S_chunk[:, k])
                np.add.at(agent_sum[:, k], cj, S_chunk[:, k])

            if not silent:
                print(f"  Hajek scatter-add: {end:,}/{n:,} ({100 * end / n:.0f}%)")

        # s_bar_l = agent_sum / agent_count
        s_bar = agent_sum / agent_count[:, np.newaxis]
        Sigma1 = (s_bar.T @ s_bar) / N

        # Multiplier: 4 for undirected, 1 for directed
        mult = 1 if directed else 4

        if cov == "DR_bc":
            middle = Sigma1 - 0.5 * Sigma2 / (N - 1)
        else:  # "DR"
            middle = Sigma1

        vcov = mult * (iGamma @ middle @ iGamma) / N

        # Enforce positive definiteness
        eigvals, eigvecs = np.linalg.eig(vcov)
        if not np.all(eigvals > 0):
            n_neg = int(np.sum(eigvals < 0))
            if not silent:
                print(f"Warning: {n_neg} negative eigenvalue(s) set to zero for PD enforcement.")
            eigvals[eigvals < 0] = 0
            vcov = (eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)).real

    se = np.sqrt(np.diag(vcov).real)

    # Build variable names
    if var_names is None:
        if nocons:
            var_names = [f"X_{k}" for k in range(K)]
        else:
            var_names = ["constant"] + [f"X_{k}" for k in range(K - 1)]
    elif not nocons:
        var_names = ["constant"] + list(var_names)

    return {
        "beta": beta,
        "vcov": vcov,
        "se": se,
        "N": N,
        "n": n,
        "var_names": var_names,
    }
