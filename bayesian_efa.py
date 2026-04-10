"""
Bayesian Exploratory Factor Analysis (BEFA) with shrinkage priors.

Two flavors:
  - fit_fast(...) : ADVI (mean-field variational) — quick approximation.
  - fit_full(...) : NUTS — full posterior, the proper Bayesian one.

Model
-----
    x_i  ~ Normal(Lambda @ f_i, diag(psi))         i = 1..N
    f_i  ~ MultivariateNormal(0, Omega)            (correlated factors)
    Omega ~ LKJCorr(eta)                           (factor correlation)
    Lambda has the lower-triangular positive-diagonal identifiability
    constraint of Geweke & Zhou (1996).
    Free off-diagonal entries of Lambda use a regularized horseshoe
    prior (Piironen & Vehtari 2017) — a state-of-the-art shrinkage
    prior that sparsifies cross-loadings while leaving genuine
    loadings essentially un-shrunken.
    psi_j ~ HalfNormal (uniquenesses).

Returns an arviz.InferenceData (NUTS) or a PyMC Approximation (ADVI).
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az


def _resolve_identification(
    P: int,
    K: int,
    identification: str,
    anchors: list[int] | None,
):
    """Resolve identification strategy to (diag_idx, free_idx) lists.

    Returns
    -------
    diag_idx : list of (row, col) — positivity-constrained entries
    free_idx : list of (row, col) — unconstrained (shrinkage-prior) entries
    all other (row, col) positions are structural zeros.

    Strategies
    ----------
    - 'lower_triangular' : Geweke-Zhou (1996). Anchors = items 1..K, strict
      lower-triangular pattern with positive diagonal.
    - 'anchor' : Generalized Geweke-Zhou (Aguilar & West 2000). User picks
      K distinct anchor items. Anchor k has a positive loading on factor k
      and ZERO loading on factors > k; non-anchor items are fully free.
    - 'unconstrained' : No structural constraints. All P*K entries are
      free. Identification is handled by post-hoc varimax + column
      reordering + sign flipping (see post_process_loadings).
    """
    ident = identification.lower()
    if ident == "lower_triangular":
        anchors = list(range(K))  # items 0..K-1
        ident = "anchor"           # fall through to anchor logic
    if ident == "anchor":
        if anchors is None:
            anchors = list(range(K))
        if len(anchors) != K:
            raise ValueError(f"need {K} anchors, got {len(anchors)}")
        if len(set(anchors)) != K:
            raise ValueError("anchors must be distinct")
        if any(a < 0 or a >= P for a in anchors):
            raise ValueError("anchors must be in [0, P)")
        anchor_set = set(anchors)
        anchor_of_col = {k: a for k, a in enumerate(anchors)}
        diag_idx: list[tuple[int, int]] = []
        free_idx: list[tuple[int, int]] = []
        for j in range(P):
            for k in range(K):
                if j == anchor_of_col[k]:
                    diag_idx.append((j, k))  # positive
                elif j in anchor_set:
                    # j is an anchor for some other factor k' -> structural
                    # zero on factors > k' to break rotation; free on factors < k'
                    k_prime = anchors.index(j)
                    if k > k_prime:
                        continue  # structural zero
                    else:
                        free_idx.append((j, k))
                else:
                    free_idx.append((j, k))
        return diag_idx, free_idx
    if ident == "unconstrained":
        diag_idx = []
        free_idx = [(j, k) for j in range(P) for k in range(K)]
        return diag_idx, free_idx
    raise ValueError(f"unknown identification: {identification}")


def build_model(
    Y: np.ndarray,
    n_factors: int,
    *,
    tau0: float | None = None,
    slab_scale: float = 2.5,
    slab_df: float = 4.0,
    lkj_eta: float = 2.0,
    standardize: bool = True,
    identification: str = "lower_triangular",
    anchors: list[int] | None = None,
    orthogonal: bool = False,
) -> pm.Model:
    """Build a Bayesian EFA model with regularized-horseshoe loadings
    and an LKJ-prior factor correlation matrix.

    Parameters
    ----------
    Y : (N, P) data matrix
    n_factors : K
    tau0 : global shrinkage scale. If None we use the Piironen-Vehtari
        default tau0 = (p0 / (m - p0)) * (1/sqrt(N)) with p0 = K
        (expected number of non-zero loadings per row).
    slab_scale, slab_df : regularized-horseshoe slab parameters.
    lkj_eta : LKJ concentration for the factor correlation matrix
        (eta=1 uniform, eta>1 favors near-identity).
    standardize : center+scale columns of Y (recommended for EFA).
    """
    Y = np.asarray(Y, dtype=float)
    N, P = Y.shape
    K = int(n_factors)
    if K < 1 or K >= P:
        raise ValueError("Need 1 <= n_factors < n_variables")

    if standardize:
        # column-wise standardization that ignores NaNs (MAR-friendly)
        col_mean = np.nanmean(Y, axis=0)
        col_sd = np.nanstd(Y, axis=0, ddof=1)
        col_sd = np.where(col_sd > 0, col_sd, 1.0)
        Y = (Y - col_mean) / col_sd

    # ---- Missingness patterns (MAR via marginalization) -------------
    # For each unique pattern of observed columns, build a separate
    # MvNormal observation on the corresponding sub-vector and the
    # corresponding sub-blocks of Sigma. This is the principled
    # Bayesian/FIML treatment of MAR data: integrate over the missing
    # cells analytically.
    obs_mask = ~np.isnan(Y)  # (N, P)
    pattern_keys = [tuple(row.tolist()) for row in obs_mask]
    unique_patterns: dict[tuple, list[int]] = {}
    for i, key in enumerate(pattern_keys):
        unique_patterns.setdefault(key, []).append(i)
    has_missing = not all(all(k) for k in unique_patterns.keys())

    # Identifiability strategy (see _resolve_identification docstring):
    #   - 'lower_triangular': Geweke-Zhou 1996 (anchors = first K items)
    #   - 'anchor': generalized GZ with user-chosen anchor items
    #   - 'unconstrained': no structural constraints; resolve rotation,
    #     sign, and column order via post-hoc varimax alignment
    diag_idx, free_idx = _resolve_identification(P, K, identification, anchors)
    m = len(free_idx)
    n_diag = len(diag_idx)

    if tau0 is None:
        # expected non-zero per *column* heuristic; fall back if degenerate
        p0 = max(1.0, float(K))
        tau0 = (p0 / max(1.0, (m - p0))) * (1.0 / np.sqrt(N))

    coords = {
        "obs": np.arange(N),
        "var": [f"x{j+1}" for j in range(P)],
        "factor": [f"F{k+1}" for k in range(K)],
        "factor_": [f"F{k+1}" for k in range(K)],
        "free": np.arange(m),
        "diag": np.arange(n_diag),
    }

    with pm.Model(coords=coords) as model:
        # don't register Y as pm.Data when missing -- we slice per pattern
        if not has_missing:
            Y_data = pm.Data("Y", Y)

        # ---- Regularized horseshoe on free loadings ------------------
        # HalfStudentT(nu=3) is much friendlier for ADVI than HalfCauchy
        # while keeping the heavy-tailed shrinkage character.
        tau = pm.HalfStudentT("tau", nu=3, sigma=tau0)
        lam = pm.HalfStudentT("lam", nu=3, sigma=1.0, dims="free")
        c2 = pm.InverseGamma(
            "c2",
            alpha=slab_df / 2.0,
            beta=slab_df / 2.0 * slab_scale**2,
        )
        lam_tilde = pm.Deterministic(
            "lam_tilde",
            pt.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2)),
            dims="free",
        )
        z = pm.Normal("z", 0.0, 1.0, dims="free")
        free_vals = pm.Deterministic("free_vals", z * tau * lam_tilde, dims="free")

        # ---- Positive "anchor" loadings (may be empty in unconstrained) -
        if n_diag > 0:
            diag_vals = pm.HalfNormal("diag_vals", sigma=1.0, dims="diag")

        # ---- Assemble Lambda -----------------------------------------
        Lambda = pt.zeros((P, K))
        if n_diag > 0:
            for idx, (j, k) in enumerate(diag_idx):
                Lambda = pt.set_subtensor(Lambda[j, k], diag_vals[idx])
        for idx, (j, k) in enumerate(free_idx):
            Lambda = pt.set_subtensor(Lambda[j, k], free_vals[idx])
        Lambda = pm.Deterministic("Lambda", Lambda, dims=("var", "factor"))

        # ---- Factor correlation (LKJ) + unit variances ---------------
        # Factor variances fixed at 1; scale identifiability is carried
        # by the positive diagonal of Lambda.
        # When orthogonal=True, Omega is fixed to identity — no LKJ
        # parameters sampled. Cheaper and sufficient for factor-count
        # screening; use orthogonal=False for the final correlated fit.
        if K == 1 or orthogonal:
            Omega = pm.Deterministic(
                "Omega", pt.eye(K), dims=("factor", "factor_")
            )
        else:
            Omega_mat = pm.LKJCorr(
                "Omega_raw", n=K, eta=lkj_eta, return_matrix=True
            )
            Omega = pm.Deterministic(
                "Omega", Omega_mat, dims=("factor", "factor_")
            )
        Sigma_F = Omega

        # ---- Uniquenesses --------------------------------------------
        psi = pm.HalfNormal("psi", sigma=1.0, dims="var")

        # Marginalize F analytically: Y_i ~ N(0, Lambda Omega Lambda^T + diag(psi^2)).
        Sigma = Lambda @ Sigma_F @ Lambda.T + pt.diag(psi**2) + 1e-6 * pt.eye(P)

        if not has_missing:
            chol_Y = pt.linalg.cholesky(Sigma)
            pm.MvNormal(
                "Y_obs",
                mu=pt.zeros(P),
                chol=chol_Y,
                observed=Y_data,
                dims=("obs", "var"),
            )
        else:
            # one MvNormal per missingness pattern, on the observed sub-vector
            for pi, (key, rows) in enumerate(unique_patterns.items()):
                obs_cols = [j for j, b in enumerate(key) if b]
                if not obs_cols:
                    continue  # row with nothing observed contributes nothing
                Y_sub = Y[np.ix_(rows, obs_cols)]
                if np.isnan(Y_sub).any():
                    Y_sub = np.nan_to_num(Y_sub)  # paranoia, shouldn't happen
                idx = pt.as_tensor_variable(np.array(obs_cols, dtype="int64"))
                Sigma_sub = Sigma[idx][:, idx]
                d = len(obs_cols)
                Sigma_sub = Sigma_sub + 1e-6 * pt.eye(d)
                chol_sub = pt.linalg.cholesky(Sigma_sub)
                pm.MvNormal(
                    f"Y_obs_p{pi}",
                    mu=pt.zeros(d),
                    chol=chol_sub,
                    observed=Y_sub,
                )

    return model


# ----------------------------------------------------------------------
# Fitting interfaces
# ----------------------------------------------------------------------
def fit_fast(
    Y: np.ndarray,
    n_factors: int,
    *,
    n_iter: int = 30_000,
    n_samples: int = 1_000,
    seed: int = 0,
    **model_kwargs,
):
    """ADVI mean-field approximation. Returns (idata, model, approx)."""
    model = build_model(Y, n_factors, **model_kwargs)
    with model:
        # Mean-field ADVI. Caveat: it under-represents posterior
        # correlations between factor-correlation entries, so the
        # estimated Omega from the fast fit can be biased relative to
        # NUTS. Use fit_full() for the proper posterior.
        approx = pm.fit(
            n=n_iter,
            method="advi",
            random_seed=seed,
            progressbar=False,
        )
        idata = approx.sample(n_samples)
    return idata, model, approx


def fit_full(
    Y: np.ndarray,
    n_factors: int,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    seed: int = 0,
    **model_kwargs,
):
    """NUTS sampler. Returns (idata, model)."""
    model = build_model(Y, n_factors, **model_kwargs)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": False},
        )
    return idata, model


# ----------------------------------------------------------------------
# Post-processing helpers
# ----------------------------------------------------------------------
def loadings_summary(idata) -> "az.data.inference_data.xr.Dataset":
    return az.summary(idata, var_names=["Lambda"])


def factor_corr_summary(idata):
    return az.summary(idata, var_names=["Omega"])


def simulate_data(
    N: int = 200,
    P: int = 12,
    K: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a simple-structure dataset with correlated factors."""
    rng = np.random.default_rng(seed)
    Lambda = np.zeros((P, K))
    per = P // K
    for k in range(K):
        Lambda[k * per : (k + 1) * per, k] = rng.uniform(0.6, 0.9, per)
    # a couple of small cross-loadings
    if K >= 2:
        Lambda[0, 1] = 0.25
    if K >= 3:
        Lambda[per, 2] = -0.2
    full_Om = np.array(
        [[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]]
    )
    if K <= 3:
        Omega = full_Om[:K, :K]
    else:
        Omega = np.eye(K) * 0.7 + 0.3
    L = np.linalg.cholesky(Omega)
    F = rng.standard_normal((N, K)) @ L.T
    psi = rng.uniform(0.2, 0.5, P)
    Y = F @ Lambda.T + rng.standard_normal((N, P)) * np.sqrt(psi)
    return Y, Lambda, Omega
