"""Number-of-factors diagnostics."""
from __future__ import annotations
import numpy as np


def parallel_analysis(
    Y: np.ndarray,
    n_iter: int = 100,
    quantile: float = 0.95,
    seed: int = 0,
) -> dict:
    """Horn's parallel analysis.

    Compares the eigenvalues of the data correlation matrix to those
    of `n_iter` random datasets of the same shape with independent
    normal columns. The suggested number of factors is the count of
    observed eigenvalues that exceed the `quantile`-quantile of the
    simulated null eigenvalues.

    Robust to NaNs: NaN cells are mean-imputed for the eigen step
    only (the actual model uses the proper MAR likelihood).
    """
    Y = np.asarray(Y, dtype=float)
    if np.isnan(Y).any():
        col_mean = np.nanmean(Y, axis=0)
        idx = np.where(np.isnan(Y))
        Y = Y.copy()
        Y[idx] = np.take(col_mean, idx[1])

    N, P = Y.shape
    R = np.corrcoef(Y.T)
    obs_eig = np.sort(np.linalg.eigvalsh(R))[::-1]

    rng = np.random.default_rng(seed)
    null_eigs = np.zeros((n_iter, P))
    for i in range(n_iter):
        Z = rng.standard_normal((N, P))
        null_eigs[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]
    null_q = np.quantile(null_eigs, quantile, axis=0)

    suggested = int(np.sum(obs_eig > null_q))
    suggested = max(1, suggested)

    return {
        "observed_eigenvalues": obs_eig,
        "null_quantile": null_q,
        "suggested_K": suggested,
        "kaiser_K": int(np.sum(obs_eig > 1.0)),
    }


def loading_strength_diagnostic(Lambda_post: np.ndarray) -> dict:
    """Posterior column-wise SSL: a column is 'effective' if its
    posterior-mean sum of squared loadings exceeds 1 (i.e. it
    explains more variance than a single standardized item)."""
    L = Lambda_post.mean(axis=(0, 1))  # (P, K)
    ssl = (L ** 2).sum(axis=0)
    effective = int((ssl > 1.0).sum())
    return {"ssl": ssl, "effective_K": max(1, effective)}


def shrinkage_factor_count(
    Lambda_post: np.ndarray,
    ssl_threshold: float = 1.0,
    activation_prob: float = 0.5,
) -> dict:
    """Shrinkage-based posterior estimate of the number of active factors.

    For each posterior draw, compute the per-column sum of squared
    loadings (SSL). A column is 'active' in that draw if its SSL exceeds
    `ssl_threshold` (1.0 = explains more than one standardized item of
    variance, the Kaiser analog). Returns:

    - posterior distribution over the active factor count
    - per-column posterior probability of being active (Prob[SSL > τ])
    - `suggested_K` = number of columns whose activation probability
      exceeds `activation_prob` (default 0.5)

    This is the principled posterior-distribution version of the
    point-estimate SSL diagnostic, and is the appropriate shrinkage
    statistic for horseshoe-prior factor models (see Piironen & Vehtari
    2017; Bhattacharya & Dunson 2011).
    """
    L = np.asarray(Lambda_post)
    # Lambda_post shape: (chains, draws, P, K)
    ssl_draws = (L ** 2).sum(axis=-2)           # (chains, draws, K)
    active = ssl_draws > ssl_threshold
    counts = active.sum(axis=-1).reshape(-1)    # flat posterior of K_active
    prob_active = active.mean(axis=(0, 1))      # (K,)
    # rank columns by activation probability so "suggested_K" has a
    # well-defined column ordering regardless of label-switching
    order = np.argsort(-prob_active)
    suggested = int((prob_active > activation_prob).sum())
    return {
        "ssl_draws": ssl_draws,
        "prob_active": prob_active,
        "count_posterior": counts,
        "posterior_mean_K": float(counts.mean()),
        "posterior_median_K": int(np.median(counts)),
        "suggested_K": max(1, suggested),
        "order": order,
        "threshold": ssl_threshold,
    }
