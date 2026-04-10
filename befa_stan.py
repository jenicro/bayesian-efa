"""CmdStan wrapper for the Bayesian EFA model in befa.stan."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from cmdstanpy import CmdStanModel

from bayesian_efa import _resolve_identification

# make sure rtools is on PATH for any (re)compile triggered from Python
_RTOOLS_BIN = r"C:\rtools45\usr\bin"
_RTOOLS_GCC = r"C:\rtools45\x86_64-w64-mingw32.static.posix\bin"
for p in (_RTOOLS_BIN, _RTOOLS_GCC):
    if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

STAN_FILE            = Path(__file__).parent / "befa.stan"
STAN_FILE_MISSING    = Path(__file__).parent / "befa_missing.stan"
STAN_FILE_WOODBURY   = Path(__file__).parent / "befa_woodbury.stan"
STAN_FILE_AUGMENTED  = Path(__file__).parent / "befa_augmented.stan"

# Missing-data model variants (passed as missing_model kwarg throughout).
#   "woodbury"   — marginalised FIML via Woodbury identity; O(n_obs·K² + N·K³);
#                  mathematically identical to pattern-mixture but much faster.
#   "pattern"    — original pattern-mixture FIML; O(n_patterns·d³);
#                  only practical for < ~30 % missingness with small P.
#   "augmented"  — explicit factor-score sampling; O(n_obs) univariate normals;
#                  avoids all Cholesky but samples N·K extra parameters.
MISSING_MODELS = ("woodbury", "pattern", "augmented")

_model_cache: dict[str, CmdStanModel] = {}


def get_model(variant: str = "dense") -> CmdStanModel:
    """Return (and cache) a compiled CmdStanModel.

    variant : "dense" | "pattern" | "woodbury" | "augmented"
    """
    paths = {
        "dense":     STAN_FILE,
        "pattern":   STAN_FILE_MISSING,
        "woodbury":  STAN_FILE_WOODBURY,
        "augmented": STAN_FILE_AUGMENTED,
    }
    if variant not in paths:
        raise ValueError(f"Unknown Stan model variant: {variant!r}")
    if variant not in _model_cache:
        print(f"[befa_stan] Compiling {paths[variant].name} …", flush=True)
        _model_cache[variant] = CmdStanModel(stan_file=str(paths[variant]))
        print(f"[befa_stan] Compilation done.", flush=True)
    return _model_cache[variant]


def _build_indices(
    P: int,
    K: int,
    identification: str = "lower_triangular",
    anchors: list[int] | None = None,
):
    """Resolve identification into Stan 1-indexed (row, col) arrays for
    the free (shrinkage-prior) entries AND the positivity-constrained
    anchor entries."""
    diag_idx, free_idx = _resolve_identification(P, K, identification, anchors)
    rows = [j + 1 for (j, _) in free_idx]
    cols = [k + 1 for (_, k) in free_idx]
    diag_rows = [j + 1 for (j, _) in diag_idx]
    diag_cols = [k + 1 for (_, k) in diag_idx]
    return rows, cols, diag_rows, diag_cols


def _build_csr_data(Y: np.ndarray) -> dict:
    """Build CSR row-format sparse data for the Woodbury model.

    Observations are sorted row-major.  row_start[n] (1-indexed) points to
    the first entry for row n in obs_col / y_vals; row_len[n] is the count.
    Rows with zero observed values are valid — row_start is set to 1 for
    those (it is never accessed because the Stan loop checks row_len[n]==0).
    """
    N, P = Y.shape
    obs_mask = ~np.isnan(Y)
    row_len  = obs_mask.sum(axis=1).astype(int)        # (N,)

    rows, cols = np.where(obs_mask)                     # C order ⇒ sorted by row
    n_obs = int(len(rows))

    # 1-indexed row_start: cumulative sum of row_len, shifted by 1
    row_start_0 = np.concatenate([[0], np.cumsum(row_len[:-1])])  # 0-indexed
    row_start_1 = row_start_0 + 1                                 # 1-indexed

    return {
        "n_obs":     n_obs,
        "row_len":   row_len.tolist(),
        "row_start": row_start_1.tolist(),
        "obs_col":   (cols + 1).tolist(),   # Stan is 1-indexed
        "y_vals":    Y[rows, cols].tolist(),
    }


def _build_sparse_data(Y: np.ndarray) -> dict:
    """Build flat (row, col, value) arrays for the augmented model."""
    rows, cols = np.where(~np.isnan(Y))
    return {
        "n_obs":   int(len(rows)),
        "obs_row": (rows + 1).tolist(),
        "obs_col": (cols + 1).tolist(),
        "y_vals":  Y[rows, cols].tolist(),
    }


def _build_pattern_data(Y: np.ndarray) -> dict:
    """Group rows by missingness pattern, return Stan-ragged arrays."""
    obs_mask = ~np.isnan(Y)
    keys = [tuple(row.tolist()) for row in obs_mask]
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    # drop empty rows
    groups = {k: v for k, v in groups.items() if any(k)}

    pat_dim = []
    pat_nrows = []
    obs_idx_flat = []
    y_flat = []
    for key, rows in groups.items():
        cols = [j + 1 for j, b in enumerate(key) if b]   # 1-indexed
        d = len(cols)
        pat_dim.append(d)
        pat_nrows.append(len(rows))
        obs_idx_flat.extend(cols)
        sub = Y[np.ix_(rows, [c - 1 for c in cols])]
        y_flat.extend(sub.flatten().tolist())

    return dict(
        n_patterns=len(pat_dim),
        pat_dim=pat_dim,
        pat_nrows=pat_nrows,
        total_obs_idx=int(sum(pat_dim)),
        obs_idx_flat=obs_idx_flat,
        total_y=int(sum(d * n for d, n in zip(pat_dim, pat_nrows))),
        y_flat=y_flat,
    )


def make_stan_data(
    Y: np.ndarray,
    K: int,
    *,
    standardize: bool = True,
    tau0: float | None = None,
    slab_scale: float = 2.5,
    slab_df: float = 4.0,
    lkj_eta: float = 2.0,
    identification: str = "lower_triangular",
    anchors: list[int] | None = None,
    missing_model: str = "woodbury",
) -> dict:
    """Assemble the Stan data dictionary.

    missing_model : "woodbury" | "pattern" | "augmented"
        Which likelihood formulation to use when Y contains NaNs.
        Ignored if Y has no missing values (dense model is used instead).
    """
    if missing_model not in MISSING_MODELS:
        raise ValueError(f"missing_model must be one of {MISSING_MODELS}")

    Y = np.asarray(Y, float)
    N, P = Y.shape
    if standardize:
        col_mean = np.nanmean(Y, axis=0)
        col_sd = np.nanstd(Y, axis=0, ddof=1)
        col_sd = np.where(col_sd > 0, col_sd, 1.0)
        Y = (Y - col_mean) / col_sd

    rows, cols, diag_rows, diag_cols = _build_indices(
        P, K, identification=identification, anchors=anchors
    )
    M = len(rows)
    n_diag = len(diag_rows)
    if M == 0:
        raise ValueError("identification left no free loadings — check config")
    if tau0 is None:
        p0 = max(1.0, float(K))
        tau0 = (p0 / max(1.0, M - p0)) * (1.0 / np.sqrt(N))

    has_missing = bool(np.isnan(Y).any())
    miss_frac   = float(np.isnan(Y).mean()) if has_missing else 0.0

    base = dict(
        N=N, P=P, K=K, M=M,
        row_idx=rows, col_idx=cols,
        n_diag=n_diag,
        diag_row_idx=diag_rows,
        diag_col_idx=diag_cols,
        tau0=float(tau0),
        slab_scale=float(slab_scale),
        slab_df=float(slab_df),
        lkj_eta=float(lkj_eta),
    )

    # Determine actual Stan variant to use
    if not has_missing:
        variant = "dense"
        base["Y"] = Y.tolist()
        print(f"[befa_stan] Dense model: N={N}, P={P}, K={K}, no missing data.", flush=True)
    elif missing_model == "woodbury":
        variant = "woodbury"
        csr = _build_csr_data(Y)
        base.update(csr)
        print(
            f"[befa_stan] Woodbury FIML: N={N}, P={P}, K={K}, "
            f"missingness={miss_frac:.1%}, n_obs={csr['n_obs']}/{N*P} cells. "
            f"Cost: O(n_obs·K² + N·K³) per step — no d×d Cholesky.",
            flush=True,
        )
    elif missing_model == "pattern":
        variant = "pattern"
        pat = _build_pattern_data(Y)
        base.update(pat)
        print(
            f"[befa_stan] Pattern-mixture FIML: {pat['n_patterns']} unique patterns, "
            f"missingness={miss_frac:.1%}. "
            f"Each HMC step does {pat['n_patterns']} sub-block Cholesky decomps.",
            flush=True,
        )
    else:   # "augmented"
        variant = "augmented"
        sparse = _build_sparse_data(Y)
        base.update(sparse)
        print(
            f"[befa_stan] Data-augmentation model: N={N}, P={P}, K={K}, "
            f"missingness={miss_frac:.1%}, n_obs={sparse['n_obs']}/{N*P} cells. "
            f"Samples {N}×{K} factor scores explicitly; "
            f"likelihood = {sparse['n_obs']} univariate normals per step.",
            flush=True,
        )

    base["_variant"] = variant
    return base


def _make_stan_init(data: dict, chain_id: int = 0) -> dict:
    """Build a starting point that gives NUTS real gradient signal on step 1.

    For all marginalized models (dense, woodbury, pattern):
        z≈0 → loadings≈0 → Σ≈diag(ψ²), trivially PD.  Stan never needs to
        retry init regardless of K.

    For the augmented model:
        Must NOT initialise both z≈0 AND F_raw=0 simultaneously — the
        likelihood gradient w.r.t. z is proportional to F[n,k], so F_raw=0
        kills all loading gradients and loadings collapse into the horseshoe
        spike.  F_raw is initialised with O(1) normals to provide live signal.

    Each chain gets a reproducible, distinct jitter so chains start from
    slightly different (but still safe) points.
    """
    variant = data.get("_variant", "dense")
    K      = data["K"]
    M      = data["M"]
    n_diag = data["n_diag"]
    P      = data["P"]
    rng    = np.random.default_rng(chain_id + 9999)  # distinct from fit seed

    z_init    = (0.01 * rng.standard_normal(M)).tolist()
    diag_init = np.clip(0.4 + 0.02 * rng.standard_normal(n_diag), 0.05, None).tolist()
    psi_init  = np.clip(0.7 + 0.02 * rng.standard_normal(P), 0.05, None).tolist()

    init = {
        "tau":          float(data["tau0"]),
        "z":            z_init,
        "lam":          [1.0] * M,
        "caux":         1.0,
        "diag_loadings": diag_init,
        "psi":          psi_init,
        "L_Omega":      np.eye(K).tolist(),   # identity → Ω=I, always valid
    }

    if variant == "augmented":
        # Non-zero F_raw provides live likelihood gradient on loadings from
        # step 1.  Scale 0.5 is within the prior range (F_raw ~ N(0,I)).
        N = data["N"]
        init["F_raw"] = (0.5 * rng.standard_normal((N, K))).tolist()

    return init


def fit_stan(
    Y: np.ndarray,
    K: int,
    *,
    chains: int = 2,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 10,
    show_progress: bool = False,
    show_console: bool = True,
    missing_model: str = "woodbury",
    **prior_kwargs,
):
    data    = make_stan_data(Y, K, missing_model=missing_model, **prior_kwargs)
    variant = data.pop("_variant")

    print(f"[befa_stan] Loading/compiling Stan model variant='{variant}' …", flush=True)
    model   = get_model(variant)

    inits = [_make_stan_init(data, chain_id=i) for i in range(chains)]

    log_dir = Path.home() / "stan_logs"
    log_dir.mkdir(exist_ok=True)
    print(f"[befa_stan] Chain CSV output → {log_dir}", flush=True)
    print(
        f"[befa_stan] Starting {chains} chain(s): "
        f"{iter_warmup} warmup + {iter_sampling} sampling draws each.",
        flush=True,
    )

    fit = model.sample(
        data=data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        inits=inits,
        output_dir=str(log_dir),
        show_progress=show_progress,
        show_console=show_console,
        save_warmup=True,   # write warmup rows to CSV so progress polling works
    )
    print(f"[befa_stan] Sampling complete.", flush=True)
    return fit
