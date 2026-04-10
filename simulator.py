"""Flexible factor-structure simulator for BEFA experiments."""
from __future__ import annotations
import numpy as np


def make_simple_structure(
    items_per_factor: list[int],
    main_loading_range: tuple[float, float] = (0.6, 0.85),
    cross_loading_prob: float = 0.0,
    cross_loading_range: tuple[float, float] = (0.15, 0.30),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a simple-structure Lambda matrix.

    Each factor 'owns' a block of items with loadings drawn uniformly
    from main_loading_range. With probability `cross_loading_prob`
    each item additionally cross-loads on a different factor with a
    sign-randomized magnitude from cross_loading_range.
    """
    rng = rng or np.random.default_rng(0)
    K = len(items_per_factor)
    P = sum(items_per_factor)
    Lambda = np.zeros((P, K))
    row = 0
    for k, n_k in enumerate(items_per_factor):
        for _ in range(n_k):
            Lambda[row, k] = rng.uniform(*main_loading_range)
            if K > 1 and rng.random() < cross_loading_prob:
                k2 = rng.choice([j for j in range(K) if j != k])
                sign = rng.choice([-1.0, 1.0])
                Lambda[row, k2] = sign * rng.uniform(*cross_loading_range)
            row += 1
    return Lambda


def make_factor_corr(K: int, rho: float) -> np.ndarray:
    """Equicorrelated factor correlation matrix with off-diag = rho."""
    Om = np.full((K, K), float(rho))
    np.fill_diagonal(Om, 1.0)
    return Om


def simulate(
    Lambda: np.ndarray,
    Omega: np.ndarray,
    N: int,
    psi_range: tuple[float, float] = (0.2, 0.5),
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Sample N rows of Y = F Lambda^T + eps with F ~ N(0, Omega)."""
    rng = np.random.default_rng(seed)
    P, K = Lambda.shape
    L = np.linalg.cholesky(Omega + 1e-10 * np.eye(K))
    F = rng.standard_normal((N, K)) @ L.T
    psi = rng.uniform(*psi_range, size=P)
    Y = F @ Lambda.T + rng.standard_normal((N, P)) * np.sqrt(psi)
    truth = dict(Lambda=Lambda, Omega=Omega, psi=np.sqrt(psi), F=F)
    return Y, truth
