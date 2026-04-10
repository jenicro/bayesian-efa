"""
Streamlit app for Bayesian Exploratory Factor Analysis.

Step 1 — design a factor structure (K, items per factor, loadings, factor
correlations, noise) OR upload your own CSV.
Step 2 — pick a backend: PyMC ADVI (fast), PyMC NUTS (full), or CmdStan NUTS (full).
Step 3 — fit and inspect loadings, factor correlations, uniquenesses,
sampling diagnostics, and recovery vs the truth.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import arviz as az
import matplotlib.pyplot as plt

from simulator import make_simple_structure, make_factor_corr, simulate
from fit_runner import FitProgress, start_fit
from diagnostics import (
    parallel_analysis,
    loading_strength_diagnostic,
    shrinkage_factor_count,
)
import pickle

def _fig_png(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return buf.getvalue()


# CmdStan is optional — only import if the user picks it
def _stan_available() -> bool:
    try:
        import befa_stan  # noqa: F401
        return True
    except Exception:
        return False


st.set_page_config(page_title="Bayesian EFA Lab", layout="wide")
st.title("🧠 Bayesian Exploratory Factor Analysis Lab")
st.caption(
    "Design a factor structure → simulate → fit (PyMC ADVI / PyMC NUTS / "
    "CmdStan NUTS) → diagnose recovery."
)

# --- BLAS sanity check (PyMC backends are dramatically slower without it) ---
@st.cache_resource
def _check_pytensor_blas() -> str:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import pytensor
        ldflags = pytensor.config.blas__ldflags
        for warning in w:
            if "BLAS" in str(warning.message):
                return ""  # empty = no BLAS
    return ldflags or ""

_blas = _check_pytensor_blas()
if not _blas:
    import platform
    _is_win = platform.system() == "Windows"
    _fix = (
        "**Fix on Windows** (easiest — use conda/mamba, pip install of PyMC "
        "does not auto-link BLAS):\n"
        "```\n"
        "conda install -c conda-forge pymc\n"
        "```\n"
        "or if you want to stay on pip, install MKL and set the flag:\n"
        "```\n"
        "pip install mkl\n"
        "$env:PYTENSOR_FLAGS='blas__ldflags=-lmkl_rt'\n"
        "```"
    ) if _is_win else (
        "**Fix on Debian/Ubuntu:**\n"
        "```\n"
        "sudo apt-get install -y libopenblas-dev liblapack-dev gfortran\n"
        "export PYTENSOR_FLAGS='blas__ldflags=-lopenblas'\n"
        "```"
    )
    st.error(
        "⚠️ **PyTensor has no BLAS linked.** PyMC backends (ADVI and NUTS) "
        "will be **dramatically slower** (every matrix op falls back to a "
        "pure-C implementation). For a 150-variable model this can mean "
        "hours instead of minutes.\n\n"
        + _fix +
        "\n\nThen restart the app. CmdStan backend is unaffected and still "
        "usable without this fix."
    )

# ====================================================================
# Sidebar: data design
# ====================================================================
with st.sidebar:
    st.header("1 · Data")
    src = st.radio(
        "Source",
        ["Simulate", "Upload CSV", "Server file", "Load results from disk"],
        index=0,
    )

    truth = None
    if src == "Simulate":
        K_true = st.number_input(
            "Number of factors K", min_value=1, max_value=20, value=3, step=1,
            help="True number of latent factors used to generate data.",
        )
        st.markdown("**Items per factor**")
        items = []
        cols = st.columns(min(K_true, 4))
        for k in range(K_true):
            with cols[k % len(cols)]:
                items.append(int(st.number_input(
                    f"F{k+1}", min_value=2, max_value=50, value=4, step=1,
                    key=f"items{k}",
                )))
        N = st.number_input(
            "N (observations)", min_value=20, max_value=20000, value=300, step=10,
            help="Number of observations to simulate.",
        )

        st.markdown("**Loadings**")
        main_lo, main_hi = st.slider(
            "Main loading range", 0.0, 1.0, (0.6, 0.85), 0.05,
            help="Each item's loading on its 'own' factor is drawn uniformly from this range.",
        )
        cross_p = st.slider(
            "Cross-loading probability", 0.0, 1.0, 0.10, 0.05,
            help="Probability that an item also loads on a second, randomly chosen factor.",
        )
        cross_lo, cross_hi = st.slider(
            "Cross-loading range", 0.0, 0.6, (0.15, 0.30), 0.05,
            help="Magnitude (sign-randomized) of any cross-loadings.",
        )

        st.markdown("**Factor correlations**")
        rho = st.slider(
            "Equi-correlation ρ", -0.5, 0.95, 0.30, 0.05,
            help="Off-diagonal of the true factor correlation matrix Ω.",
        )

        st.markdown("**Noise (uniqueness variances)**")
        psi_lo, psi_hi = st.slider(
            "ψ² range", 0.05, 1.0, (0.2, 0.5), 0.05,
            help="Per-item residual variance is drawn uniformly from this range. "
                 "Larger ψ² ⇒ noisier items, harder recovery.",
        )

        st.markdown("**Missing data (MAR)**")
        miss_prop = st.slider(
            "Proportion missing (MCAR ⊂ MAR)", 0.0, 1.0, 0.0, 0.05,
            help="Each cell of Y is set to NaN independently with this "
                 "probability. The Bayesian model integrates over the "
                 "missing entries (FIML / pattern-mixture marginalization).",
        )

        seed_data = st.number_input("Seed (data)", value=0, step=1)

        rng = np.random.default_rng(int(seed_data))
        Lambda_true = make_simple_structure(
            items_per_factor=items,
            main_loading_range=(main_lo, main_hi),
            cross_loading_prob=cross_p,
            cross_loading_range=(cross_lo, cross_hi),
            rng=rng,
        )
        Omega_true = make_factor_corr(K_true, rho)
        Y, truth = simulate(
            Lambda_true, Omega_true, N=N,
            psi_range=(psi_lo, psi_hi), seed=int(seed_data),
        )
        if miss_prop > 0:
            mrng = np.random.default_rng(int(seed_data) + 999)
            mask = mrng.random(Y.shape) < miss_prop
            Y = Y.copy()
            Y[mask] = np.nan
        df = pd.DataFrame(Y, columns=[f"x{j+1}" for j in range(Y.shape[1])])
    elif src == "Upload CSV":
        up = st.file_uploader("CSV", type=["csv"])
        if up is None:
            st.stop()
        df = pd.read_csv(up).select_dtypes(include=[np.number])
        Y = df.values
    elif src == "Server file":
        import os
        default_path = os.path.expanduser("~/befa-app/data.csv")
        server_path = st.text_input("Path on server", value=default_path)
        if not os.path.exists(server_path):
            st.error(f"File not found: {server_path}")
            st.stop()
        df = pd.read_csv(server_path).select_dtypes(include=[np.number])
        Y = df.values
    else:  # "Load results from disk"
        import os
        _res_dir_default = st.session_state.get("_results_dir", str(Path.home() / "befa-app" / "overnight_results"))
        _res_dir = st.text_input("Results directory", value=_res_dir_default, key="_res_dir_input")
        _res_path = Path(_res_dir)
        _y_path = _res_path / "Y.csv"
        if not _y_path.exists():
            st.error(f"Y.csv not found in: {_res_dir}")
            st.stop()
        df = pd.read_csv(_y_path)
        Y = df.values.astype(float)
        # Load ground truth if available
        _truth_path = _res_path / "truth.npz"
        if _truth_path.exists():
            _t = np.load(_truth_path)
            truth = {k: _t[k] for k in _t.files}
        # Pre-populate K and identification from meta.json
        _meta_path = _res_path / "meta.json"
        if _meta_path.exists():
            _meta = json.loads(_meta_path.read_text())
            if "fit_K" not in st.session_state:
                st.session_state.fit_K = _meta.get("K", min(3, Y.shape[1] - 1))
            _disk_identification = _meta.get("identification", "lower_triangular")
        else:
            _disk_identification = None
        # Button to load posteriors
        _load_btn = st.button("Load posteriors from disk", type="primary")
        if _load_btn:
            _lp = _res_path / "Lambda_post.npy"
            _op = _res_path / "Omega_post.npy"
            _pp = _res_path / "psi_post.npy"
            if not (_lp.exists() and _op.exists() and _pp.exists()):
                st.error("Posterior .npy files not found — fit may still be running.")
                st.stop()
            st.session_state._loaded_Lambda = np.load(_lp)
            st.session_state._loaded_Omega  = np.load(_op)
            st.session_state._loaded_psi    = np.load(_pp)
            st.session_state._results_dir   = _res_dir
            _summ_path = _res_path / "summary.csv"
            if _summ_path.exists():
                _s = pd.read_csv(_summ_path, index_col=0)
                _sub = _s.loc[_s.index.str.startswith(("Lambda", "Omega", "psi"))]
                st.session_state._loaded_summary = _sub.rename(
                    columns={"R_hat": "r_hat", "ESS_bulk": "ess_bulk", "ESS_tail": "ess_tail"}
                )
            st.rerun()
        if "_loaded_Lambda" not in st.session_state:
            st.info("Configure K and identification below, then click **Load posteriors from disk**.")
            st.stop()

    nan_count = int(np.isnan(Y).sum())
    st.write(
        f"Shape: **{Y.shape[0]} × {Y.shape[1]}** · "
        f"missing: **{nan_count}** ({nan_count / Y.size:.1%})"
    )

    st.header("2 · Model")
    if "fit_K" not in st.session_state:
        st.session_state.fit_K = min(3, Y.shape[1] - 1)
    # Apply a pending K update from a "Rerun with K=…" button click
    # *before* the widget is instantiated.
    if "pending_fit_K" in st.session_state:
        st.session_state.fit_K = min(
            max(1, int(st.session_state.pop("pending_fit_K"))),
            max(1, Y.shape[1] - 1),
        )
    K = int(st.number_input(
        "Fit K factors", min_value=1, max_value=max(1, Y.shape[1] - 1),
        step=1, key="fit_K",
        help="Number of factors used in the BEFA model. Try the value "
             "suggested by the parallel-analysis / shrinkage diagnostic "
             "after a fit.",
    ))
    orthogonal = st.checkbox(
        "⚡ Orthogonal factors (fast screening, Ω = I)",
        value=False,
        help=(
            "Fix the factor correlation matrix to the identity (Ω = I). "
            "This removes the LKJ prior entirely — no correlation parameters "
            "are sampled, making the model significantly faster. "
            "Use this for a quick screening run to determine K. "
            "For the final inference, uncheck this to allow correlated factors.\n\n"
            "**Post-hoc rotation:** varimax (orthogonal) instead of Promax.\n\n"
            "**CmdStan note:** Stan always uses correlated factors; "
            "orthogonal mode applies to PyMC backends only."
        ),
    )
    if orthogonal:
        lkj_eta = 2.0  # unused but keep a valid default
        st.caption("LKJ η hidden — Ω fixed to identity in orthogonal mode.")
    else:
        lkj_eta = st.slider(
            "LKJ η (factor corr prior)", 0.5, 10.0, 2.0, 0.5,
            help="Concentration parameter of the LKJ prior on the factor "
                 "correlation matrix Ω. η=1 is uniform over correlation "
                 "matrices; η>1 pulls Ω toward the identity (factors a "
                 "priori closer to uncorrelated). Larger η ⇒ stronger "
                 "regularization toward orthogonality.",
        )
    slab_scale = st.slider(
        "Horseshoe slab scale", 0.5, 5.0, 2.5, 0.25,
        help="Slab width c in the *regularized horseshoe* prior "
             "(Piironen & Vehtari 2017). The horseshoe puts most "
             "loadings near zero (sparsity) but allows large signal "
             "loadings to escape unshrunk; the slab caps how large the "
             "non-zero loadings can get. Bigger slab ⇒ less shrinkage "
             "on large loadings.",
    )

    ident_choice = st.selectbox(
        "Identification strategy",
        [
            "Lower-triangular (Geweke-Zhou, fast)",
            "Anchor items (generalized GZ)",
            "Unconstrained + Promax post-hoc",
        ],
        index=0,
        help=(
            "How to fix the rotation / sign / permutation indeterminacy "
            "of the factor loadings.\n\n"
            "• **Lower-triangular**: Λ is lower-triangular with positive "
            "diagonal (Geweke & Zhou 1996). Forces item 1 → F1, item 2 "
            "→ F2, etc. — fast, but the assignment is arbitrary when "
            "variables aren't pre-sorted.\n\n"
            "• **Anchor items**: you pick K 'anchor' variables, one per "
            "factor. Each anchor has a positive loading on its factor "
            "and zeros on later factors (generalized GZ, Aguilar & "
            "West 2000). Same speed as the default; arbitrariness moves "
            "to your choice of anchors.\n\n"
            "• **Unconstrained + Promax**: no structural constraints "
            "during sampling. Rotation, column order, and signs are "
            "resolved post-hoc via Promax *oblique* rotation "
            "(Hendrickson & White 1964) + SSL reordering + sign flip. "
            "Fastest and most symmetric sampling geometry; Promax is "
            "oblique so the LKJ-correlated factor structure is "
            "preserved (unlike varimax, which forces orthogonal "
            "factors)."
        ),
    )
    _ident_map = {
        "Lower-triangular (Geweke-Zhou, fast)": "lower_triangular",
        "Anchor items (generalized GZ)": "anchor",
        "Unconstrained + Promax post-hoc": "unconstrained",
    }
    identification = _ident_map[ident_choice]
    anchors_idx: list[int] | None = None
    if identification == "anchor":
        st.markdown("**Anchor items** (one per factor, must be distinct)")
        col_names = [f"x{j+1}" for j in range(Y.shape[1])]
        default_anchors = list(range(min(K, Y.shape[1])))
        anchors_idx = []
        acols = st.columns(min(K, 4))
        for k in range(K):
            with acols[k % len(acols)]:
                idx = st.selectbox(
                    f"F{k+1} anchor",
                    options=list(range(Y.shape[1])),
                    format_func=lambda j: col_names[j],
                    index=default_anchors[k] if k < len(default_anchors) else 0,
                    key=f"anchor_{k}",
                )
                anchors_idx.append(int(idx))
        if len(set(anchors_idx)) != K:
            st.error("Anchor items must be distinct — pick a different variable per factor.")
            st.stop()

    st.header("3 · Inference backend")
    backends = ["PyMC ADVI (fast)", "PyMC NUTS (full)"]
    if _stan_available():
        backends.append("CmdStan NUTS (full)")
    backend = st.radio("Backend", backends, index=0)

    if "ADVI" in backend:
        n_iter = st.slider("ADVI iterations", 5_000, 80_000, 25_000, 5_000)
    else:
        draws = st.slider("draws", 200, 3000, 600, 100)
        tune = st.slider("warmup/tune", 200, 3000, 600, 100)
        chains = st.slider("chains", 2, 6, 2)
        st.caption(
            "⚡ Init: near-zero smart init (Λ≈0, Ω=I) — eliminates PD-rejection "
            "retries, so even K=15+ starts in seconds."
        )

    # Missing-data model selector (CmdStan only; PyMC always uses pattern-mixture)
    _has_nan = bool(np.isnan(Y).any())
    _miss_frac = float(np.isnan(Y).mean()) if _has_nan else 0.0
    missing_model = "woodbury"   # default; overridden below for CmdStan
    if _has_nan and "CmdStan" in backend:
        st.markdown("**Missing-data likelihood**")
        _model_labels = {
            "woodbury":  "Woodbury FIML  (recommended — proper, fast)",
            "augmented": "Data augmentation  (explicit factor scores — fast, experimental)",
            "pattern":   "Pattern-mixture FIML  (proper, slow for high missingness)",
        }
        _default_idx = 0   # Woodbury is always the default
        _selected = st.radio(
            "Missing-data model",
            list(_model_labels.keys()),
            format_func=lambda k: _model_labels[k],
            index=_default_idx,
            key="missing_model_radio",
            help=(
                "**Woodbury FIML** — analytically marginalises factor scores using "
                "the Woodbury matrix identity. Same posterior as pattern-mixture FIML "
                "but O(n_obs·K²) instead of O(n_patterns·d³). Recommended for any "
                "missingness level.\n\n"
                "**Data augmentation** — samples factor scores F explicitly. "
                "Likelihood is O(n_obs) univariate normals — no Cholesky at all. "
                "Useful when N×K is small and missingness is extreme (>80%), but "
                "can have mixing issues with the horseshoe prior.\n\n"
                "**Pattern-mixture FIML** — original model. Groups rows by "
                "missingness pattern and does one d×d Cholesky per pattern per step. "
                "Only practical when missingness is low (<30%) or P is small."
            ),
        )
        missing_model = _selected
        if _miss_frac >= 0.30 and _selected == "pattern":
            st.warning(
                f"⚠️ {_miss_frac:.0%} missingness with pattern-mixture FIML may be "
                "very slow (O(n_patterns × d³) per step, n_patterns can approach N). "
                "Consider switching to **Woodbury FIML** above."
            )
        if _selected == "augmented":
            st.info(
                "ℹ️ Data augmentation samples N×K extra parameters (factor scores). "
                "This can cause mixing issues with the horseshoe prior. "
                "Woodbury FIML gives the same posterior with better geometry."
            )
    elif _has_nan and "CmdStan" not in backend:
        st.caption(
            f"ℹ️ {_miss_frac:.0%} missing — PyMC backends use pattern-mixture FIML "
            "(built-in). Switch to CmdStan to choose the Woodbury or augmentation model."
        )

    seed_fit = st.number_input("Seed (fit)", value=1, step=1)

    _is_load_src = (src == "Load results from disk")
    c_go, c_cancel = st.columns(2)
    go = c_go.button("🚀 Fit", type="primary", width='stretch', disabled=_is_load_src)
    cancel_clicked = c_cancel.button("⛔ Abort", width='stretch', disabled=_is_load_src)

    # Overnight button — CmdStan only, not when loading from disk
    _overnight_clicked = False
    if "CmdStan" in backend and not _is_load_src:
        _overnight_default_dir = str(
            Path.home() / "befa-app" / f"overnight_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        _overnight_outdir = st.text_input(
            "Overnight output directory", value=_overnight_default_dir, key="_overnight_outdir"
        )
        _overnight_clicked = st.button(
            "Run overnight (detached)", type="secondary", width='stretch',
            help="Launches the fit as an independent background process. "
                 "Safe to close the browser — the fit keeps running on the server. "
                 "Open 'Load results from disk' when done to view results.",
        )

# ====================================================================
# Top: data preview + truth (if simulated)
# ====================================================================
left, mid, right = st.columns([1.1, 1, 1])
with left:
    st.subheader("Data preview")
    st.dataframe(df.head(), height=220)
with mid:
    st.subheader("Empirical correlation")
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    # pairwise-complete correlation (handles NaNs without dropping rows)
    R_emp = pd.DataFrame(Y).corr(method="pearson", min_periods=2).values
    im = ax.imshow(R_emp, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.image(_fig_png(fig), width='stretch')
with right:
    if truth is not None:
        st.subheader("True Λ")
        fig, ax = plt.subplots(figsize=(0.6 * truth["Lambda"].shape[1] + 1.5,
                                        0.25 * truth["Lambda"].shape[0] + 1))
        im = ax.imshow(truth["Lambda"], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.image(_fig_png(fig), width='stretch')
    else:
        st.info("Upload a CSV or simulate data on the left.")

# ====================================================================
# Fit (threaded with progress + cancel)
# ====================================================================
if "prog" not in st.session_state:
    st.session_state.prog = FitProgress()
prog: FitProgress = st.session_state.prog

# Guard against stale "running" state after a hot-reload: if the background
# thread is gone but prog still says "running", reset so Fit works again.
if (prog.status == "running"
        and prog.thread is not None
        and not prog.thread.is_alive()):
    prog.status = "idle"
    prog.message = "(fit thread died — possibly a hot-reload; click Fit to retry)"

if st.session_state.pop("auto_fit", False) and prog.status != "running":
    go = True

if _overnight_clicked and prog.status != "running":
    _odir = Path(st.session_state.get("_overnight_outdir", _overnight_default_dir))
    _odir.mkdir(parents=True, exist_ok=True)
    # Save Y to the output directory so the subprocess has it
    pd.DataFrame(Y, columns=list(df.columns)).to_csv(_odir / "Y.csv", index=False)
    if truth is not None:
        np.savez(_odir / "truth.npz", **{k: v for k, v in truth.items()})
    _cmd = [
        sys.executable, str(Path(__file__).parent / "run_overnight.py"),
        "csv", str(_odir / "Y.csv"),
        "--K", str(K),
        "--chains", str(chains),
        "--warmup", str(tune),
        "--draws", str(draws),
        "--seed", str(int(seed_fit)),
        "--identification", identification,
        "--lkj-eta", str(lkj_eta),
        "--slab-scale", str(slab_scale),
        "--missing-model", missing_model,
        "--outdir", str(_odir),
    ]
    _proc = subprocess.Popen(
        _cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    st.session_state._results_dir = str(_odir)
    st.success(
        f"Overnight fit started (PID {_proc.pid}). "
        f"Results will appear in: `{_odir}`. "
        f"You can close the browser — the fit runs independently on the server. "
        f"Use **Load results from disk** to view results when done."
    )

if go and prog.status != "running":
    if orthogonal and "CmdStan" in backend:
        st.warning(
            "⚠️ Orthogonal mode is PyMC-only. CmdStan will use correlated factors (Ω ~ LKJ). "
            "For orthogonal screening, use PyMC ADVI instead."
        )
    kwargs = dict(
        Y=Y, K=K, lkj_eta=lkj_eta, slab_scale=slab_scale, seed=int(seed_fit),
        identification=identification, anchors=anchors_idx,
        orthogonal=orthogonal,
    )
    if "ADVI" in backend:
        kwargs["n_iter"] = int(n_iter)
    else:
        kwargs.update(draws=int(draws), tune=int(tune), chains=int(chains))
        if "CmdStan" in backend:
            kwargs["missing_model"] = missing_model
    start_fit(prog, backend, **kwargs)

if cancel_clicked and prog.status == "running":
    prog.request_cancel()
    st.toast("⛔ Abort requested — will take effect at next iteration", icon="⛔")

# Live progress panel
if prog.backend and prog.status in ("running", "done", "cancelled", "error"):
    st.subheader(f"Fit · {prog.backend}")
    cancelling = prog.status == "running" and prog.cancel_event.is_set()
    pcol1, pcol2, pcol3 = st.columns([2, 1, 1])
    status_label = "ABORTING…" if cancelling else prog.status.upper()
    pcol1.write(
        f"**Status:** `{status_label}` · *{prog.phase}* — {prog.message}"
    )
    pcol2.metric("iter", f"{prog.iter}/{prog.total or '?'}")
    pcol3.metric("elapsed", f"{prog.elapsed:.1f}s")
    if prog.total:
        st.progress(min(1.0, prog.iter / max(1, prog.total)))
    if cancelling and prog.phase == "compiling":
        st.info(
            "Abort is pending — model is still in the compile/build phase "
            "where there is no callback to interrupt. It will cancel as "
            "soon as sampling/optimization starts."
        )
    if "Stan" in prog.backend and prog.log_lines:
        console_text = "\n".join(reversed(list(prog.log_lines)[-80:]))
        st.text_area(
            "Stan console (newest first)",
            value=console_text,
            height=300,
            disabled=True,
            key="stan_console",
        )

# Auto-refresh while running
if prog.status == "running":
    time.sleep(0.5)
    st.rerun()

if prog.status == "error":
    st.error(f"❌ Fit failed:\n\n{prog.error}")
    st.stop()
if prog.status == "cancelled":
    st.error(
        f"⛔ **Fit aborted by user.** "
        f"Stopped at iteration {prog.iter}/{prog.total or '?'} "
        f"after {prog.elapsed:.1f}s. Click Fit to retry."
    )
    st.stop()
_has_loaded_results = "_loaded_Lambda" in st.session_state
if prog.status != "done" and not _has_loaded_results:
    st.stop()

# ---- Unpack results --------------------------------------------------
elbo = None
rhat_summary = None
if _has_loaded_results and prog.status != "done":
    Lambda_post = st.session_state._loaded_Lambda
    Omega_post  = st.session_state._loaded_Omega
    psi_post    = st.session_state._loaded_psi
    rhat_summary = st.session_state.get("_loaded_summary")
    res = {"_loaded": True}
    backend_label = f"Loaded from disk ({st.session_state.get('_results_dir', '')})"
    st.info(f"Results loaded from: `{st.session_state.get('_results_dir', '')}`")
else:
    res = prog.result
    if "stan_fit" in res:
        fit = res["stan_fit"]
        Lambda_post = fit.stan_variable("Lambda")[None, ...]
        Omega_post = fit.stan_variable("Omega")[None, ...]
        psi_post = fit.stan_variable("psi")[None, ...]
        s = fit.summary()
        rhat_summary = s.loc[s.index.str.startswith(("Lambda", "Omega", "psi"))]
        rhat_summary = rhat_summary.rename(
            columns={"R_hat": "r_hat", "ESS_bulk": "ess_bulk", "ESS_tail": "ess_tail"}
        )
    else:
        idata = res["idata"]
        Lambda_post = idata.posterior["Lambda"].values
        Omega_post = idata.posterior["Omega"].values
        psi_post = idata.posterior["psi"].values
        if "elbo" in res:
            elbo = res["elbo"]
        else:
            rhat_summary = az.summary(idata, var_names=["Lambda", "Omega", "psi"])
    backend_label = prog.backend
    st.success(f"{backend_label} finished in {prog.elapsed:.1f}s")

# ---- Post-hoc identification for unconstrained runs -----------------
if identification == "unconstrained":
    if orthogonal:
        from post_process import varimax_align_posterior
        Lambda_post, Omega_post = varimax_align_posterior(Lambda_post, Omega_post)
        st.caption(
            "🔄 Unconstrained + orthogonal fit — applied **Varimax** orthogonal "
            "rotation (correct for Ω = I) + SSL reordering + sign flip "
            "to every posterior draw."
        )
    else:
        from post_process import promax_align_posterior
        Lambda_post, Omega_post = promax_align_posterior(
            Lambda_post, Omega_post, kappa=4.0
        )
        st.caption(
            "🔄 Unconstrained fit — applied **Promax** oblique rotation "
            "(correlated factors preserved) + SSL reordering + sign flip "
            "to every posterior draw."
        )

# ---- Dead-column detector (identification pathology warning) -------
_col_ssl = (Lambda_post.mean(axis=(0, 1)) ** 2).sum(axis=0)
_dead_cols = [k for k, s in enumerate(_col_ssl) if s < 0.1]
if _dead_cols:
    _dead_str = ", ".join(f"F{k+1} (SSL={_col_ssl[k]:.3f})" for k in _dead_cols)
    _k_alive = K - len(_dead_cols)
    if identification == "unconstrained":
        st.warning(
            f"ℹ️ **{len(_dead_cols)} factor(s) shrunk to zero:** "
            f"{_dead_str}. In the unconstrained + Promax fit this "
            f"almost always means **K is too high** and the horseshoe "
            f"correctly killed redundant columns. The effective number "
            f"of factors is **{_k_alive}** — consider refitting with "
            f"K = {_k_alive}."
        )
    else:
        st.error(
            f"⚠️ **Dead factor(s) detected:** {_dead_str}. "
            f"These columns of Λ collapsed to ≈ 0. Two likely causes:\n\n"
            f"1. **K is too high** — horseshoe correctly killed a "
            f"redundant factor. Refit with K = {_k_alive}.\n"
            f"2. **Identification local mode** — the Geweke-Zhou "
            f"positive diagonal + horseshoe + ADVI can trap a real "
            f"factor in ψ. Try **Identification = 'Unconstrained + "
            f"Promax post-hoc'** and/or a **NUTS** backend."
        )

# ---- Post-run factor-count recommendation ---------------------------
_pa_reco = parallel_analysis(Y, n_iter=100, seed=0)
_shrink_reco = shrinkage_factor_count(Lambda_post)
K_pa_reco = _pa_reco["suggested_K"]
K_shrink_reco = _shrink_reco["suggested_K"]

rec_col1, rec_col2, rec_col3 = st.columns([2, 1, 1])
with rec_col1:
    if K_pa_reco == K_shrink_reco == K:
        st.info(
            f"✅ **Recommendation:** both parallel analysis and the "
            f"horseshoe shrinkage diagnostic agree on **K = {K}**. "
            f"Your current fit is well-sized."
        )
    else:
        msg = (
            f"📊 **Recommendation:** parallel analysis suggests "
            f"**K = {K_pa_reco}**, shrinkage diagnostic suggests "
            f"**K = {K_shrink_reco}** "
            f"(posterior median = {_shrink_reco['posterior_median_K']}). "
            f"You fit with **K = {K}**."
        )
        st.warning(msg)
with rec_col2:
    if K_shrink_reco != K and st.button(
        f"🔁 Rerun with K={K_shrink_reco}", key="rerun_shrink",
        width='stretch',
    ):
        st.session_state.pending_fit_K = K_shrink_reco
        st.session_state.auto_fit = True
        st.rerun()
with rec_col3:
    if K_pa_reco != K and st.button(
        f"🔁 Rerun with K={K_pa_reco}", key="rerun_pa",
        width='stretch',
    ):
        st.session_state.pending_fit_K = K_pa_reco
        st.session_state.auto_fit = True
        st.rerun()

# Posterior summaries
L_mean   = Lambda_post.mean(axis=(0, 1))
L_sd     = Lambda_post.std(axis=(0, 1))
O_mean   = Omega_post.mean(axis=(0, 1))
psi_mean = psi_post.mean(axis=(0, 1))
psi_sd   = psi_post.std(axis=(0, 1))

# Credible-interval percentiles (computed once; CI level chosen in the UI below)
_n_draws = Lambda_post.shape[0] * Lambda_post.shape[1]
_L_draws   = Lambda_post.reshape(_n_draws, Lambda_post.shape[2], Lambda_post.shape[3])
_O_draws   = Omega_post.reshape(_n_draws, Omega_post.shape[2], Omega_post.shape[3])
_psi_draws = psi_post.reshape(_n_draws, psi_post.shape[2])

# ====================================================================
# Procrustes alignment to truth (column permutation + signs)
# ====================================================================
def align(L_est, L_true):
    K = L_true.shape[1]
    if L_est.shape[1] != K:
        return L_est, np.eye(L_est.shape[1])[: K], None
    best = (list(range(K)), np.ones(K))
    best_err = np.inf
    for perm in permutations(range(K)):
        Lp = L_est[:, list(perm)]
        signs = np.sign((Lp * L_true).sum(0))
        signs[signs == 0] = 1
        err = np.linalg.norm(Lp * signs - L_true)
        if np.isfinite(err) and err < best_err:
            best_err = err
            best = (list(perm), signs)
    perm, signs = best
    return L_est[:, perm] * signs, perm, signs


# Cache align() result: K! is expensive for large K and align() is called on
# every Streamlit poll rerun.  Key on the fit result identity so the cache
# is invalidated automatically when a new fit completes.
_align_cache_key = id(res)
_cached = st.session_state.get("_align_cache")
if _cached is not None and _cached["key"] == _align_cache_key:
    L_aligned, perm, signs, O_aligned = (
        _cached["L_aligned"], _cached["perm"], _cached["signs"], _cached["O_aligned"]
    )
else:
    L_aligned, perm, signs = (L_mean, list(range(K)), np.ones(K))
    O_aligned = O_mean
    if truth is not None and truth["Lambda"].shape[1] == K:
        col_sd = Y.std(0, ddof=1)
        L_true_std = truth["Lambda"] / col_sd[:, None]
        L_aligned, perm, signs = align(L_mean, L_true_std)
        O_aligned = O_mean[np.ix_(perm, perm)] * np.outer(signs, signs)
    st.session_state["_align_cache"] = {
        "key": _align_cache_key,
        "L_aligned": L_aligned, "perm": perm,
        "signs": signs, "O_aligned": O_aligned,
    }

# ====================================================================
# Credible interval level selector
# ====================================================================
_ci_col1, _ci_col2 = st.columns([1, 4])
with _ci_col1:
    _ci_level = st.selectbox(
        "Credible interval",
        [89, 90, 95],
        index=1,
        format_func=lambda x: f"{x}%",
        key="ci_level",
    )
with _ci_col2:
    st.caption(
        f"Applies to all tables below. "
        f"**✓** in *excl 0* column = {_ci_level}% CI entirely above or below zero "
        f"(Bayesian analogue of significance). "
        f"For Promax (unconstrained) the CI is spread of aligned draws, not a fully "
        f"Bayesian interval."
    )
_ci_lo_pct = (100 - _ci_level) / 2       # e.g. 5.0  for 90 %
_ci_hi_pct = 100 - _ci_lo_pct            # e.g. 95.0 for 90 %

# Compute percentiles from stacked draws  ── shape (P,K), (K,K), (P,)
L_lo  = np.percentile(_L_draws,   _ci_lo_pct, axis=0)
L_hi  = np.percentile(_L_draws,   _ci_hi_pct, axis=0)
O_lo  = np.percentile(_O_draws,   _ci_lo_pct, axis=0)
O_hi  = np.percentile(_O_draws,   _ci_hi_pct, axis=0)
psi_lo = np.percentile(_psi_draws, _ci_lo_pct, axis=0)
psi_hi = np.percentile(_psi_draws, _ci_hi_pct, axis=0)

# Apply the same perm + sign flip used for L_aligned so CI columns match the table
def _apply_perm_signs(arr_lo, arr_hi, perm, signs):
    """Permute columns and flip lo/hi correctly when sign = -1."""
    lo_p = arr_lo[:, perm] * signs
    hi_p = arr_hi[:, perm] * signs
    # sign=-1 negates and swaps the bounds
    return np.minimum(lo_p, hi_p), np.maximum(lo_p, hi_p)

if perm is not None and list(perm) != list(range(K)):
    L_lo_aligned, L_hi_aligned = _apply_perm_signs(L_lo, L_hi, perm, signs)
    O_lo_aligned = O_lo[np.ix_(perm, perm)] * np.outer(signs, signs)
    O_hi_aligned = O_hi[np.ix_(perm, perm)] * np.outer(signs, signs)
    # swap lo/hi for cells where outer sign product is -1
    _O_sign = np.outer(signs, signs)
    O_lo_aligned = np.where(_O_sign > 0, O_lo_aligned, O_hi[np.ix_(perm, perm)] * _O_sign)
    O_hi_aligned = np.where(_O_sign > 0, O_hi_aligned, O_lo[np.ix_(perm, perm)] * _O_sign)
    O_lo_aligned, O_hi_aligned = (np.minimum(O_lo_aligned, O_hi_aligned),
                                   np.maximum(O_lo_aligned, O_hi_aligned))
else:
    L_lo_aligned, L_hi_aligned = L_lo, L_hi
    O_lo_aligned, O_hi_aligned = O_lo, O_hi

# ====================================================================
# Tabs
# ====================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Loadings Λ", "Factor corr Ω", "Uniquenesses ψ",
     "Sampling diag", "Recovery", "K diagnostic", "Save"]
)

with tab1:
    P = Y.shape[1]
    fig, ax = plt.subplots(figsize=(1.0 * K + 2, 0.32 * P + 1))
    im = ax.imshow(L_aligned, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(K), [f"F{k+1}" for k in range(K)])
    ax.set_yticks(range(P), df.columns)
    for j in range(P):
        for k in range(K):
            ax.text(
                k, j, f"{L_aligned[j, k]:.2f}",
                ha="center", va="center",
                color="white" if abs(L_aligned[j, k]) > 0.4 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.image(_fig_png(fig), width='stretch')

    # --- Loadings table with credible intervals -----------------------
    _L_sd_aligned = L_sd[:, perm] if perm is not None else L_sd
    _rows = []
    for j in range(P):
        for k in range(K):
            m   = L_aligned[j, k]
            sd  = _L_sd_aligned[j, k]
            lo  = L_lo_aligned[j, k]
            hi  = L_hi_aligned[j, k]
            sig = "✓" if (lo > 0 or hi < 0) else ""
            _rows.append({
                "variable":  df.columns[j],
                "factor":    f"F{k+1}",
                "mean":      round(float(m),  3),
                "sd":        round(float(sd), 3),
                f"lo{_ci_level}": round(float(lo), 3),
                f"hi{_ci_level}": round(float(hi), 3),
                "excl 0":    sig,
            })
    _lambda_df = pd.DataFrame(_rows)
    st.dataframe(_lambda_df, hide_index=True, height=400)

    if identification == "unconstrained":
        st.caption(
            f"⚠️ **Promax identification**: the {_ci_level}% CI is the spread of "
            "Procrustes-aligned draws around the shared reference basis — "
            "not a fully Bayesian credible interval on Λ. "
            "Point estimates (mean) are reliable; treat width with caution."
        )

with tab2:
    if orthogonal and "CmdStan" not in backend_label:
        st.info(
            "⚡ Orthogonal model — Ω was fixed to the identity matrix during sampling. "
            "No factor correlations were estimated. Refit without the orthogonal "
            "checkbox to get the full correlated-factor posterior."
        )
    else:
        fig, ax = plt.subplots(figsize=(3 + 0.3 * K, 3 + 0.3 * K))
        im = ax.imshow(O_aligned, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{O_aligned[i, j]:.2f}", ha="center", va="center")
        ax.set_xticks(range(K), [f"F{k+1}" for k in range(K)])
        ax.set_yticks(range(K), [f"F{k+1}" for k in range(K)])
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.image(_fig_png(fig), width='stretch')

        # Factor correlation table (off-diagonal only — diagonal is always 1)
        _O_rows = []
        for i in range(K):
            for j in range(i + 1, K):
                m  = float(O_aligned[i, j])
                lo = float(O_lo_aligned[i, j])
                hi = float(O_hi_aligned[i, j])
                sd = float(_O_draws[:, i, j].std()) if perm is None else \
                     float(_O_draws[:, perm[i], perm[j]].std())
                sig = "✓" if (lo > 0 or hi < 0) else ""
                _O_rows.append({
                    "pair":           f"F{i+1} ↔ F{j+1}",
                    "mean":           round(m,  3),
                    "sd":             round(sd, 3),
                    f"lo{_ci_level}": round(lo, 3),
                    f"hi{_ci_level}": round(hi, 3),
                    "excl 0":         sig,
                })
        if _O_rows:
            st.dataframe(pd.DataFrame(_O_rows), hide_index=True)

        if truth is not None:
            st.caption("True Ω")
            st.dataframe(pd.DataFrame(truth["Omega"]))

with tab3:
    fig, ax = plt.subplots(figsize=(6, 3))
    # Error bars show the CI, not ±1 SD
    _psi_err_lo = psi_mean - psi_lo
    _psi_err_hi = psi_hi - psi_mean
    ax.bar(range(len(psi_mean)), psi_mean,
           yerr=[_psi_err_lo, _psi_err_hi],
           color="steelblue", capsize=3, error_kw={"linewidth": 1.2})
    ax.set_xticks(range(len(psi_mean)), df.columns, rotation=45, ha="right")
    ax.set_ylabel("ψ (residual sd)")
    ax.set_title(f"Uniquenesses — error bars = {_ci_level}% CI")
    st.image(_fig_png(fig), width='stretch')

    _psi_df = pd.DataFrame({
        "variable":       df.columns,
        "mean":           np.round(psi_mean, 3),
        "sd":             np.round(psi_sd, 3),
        f"lo{_ci_level}": np.round(psi_lo, 3),
        f"hi{_ci_level}": np.round(psi_hi, 3),
    })
    st.dataframe(_psi_df, hide_index=True)

with tab4:
    if elbo is not None:
        st.subheader("ADVI ELBO trace")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(elbo)
        ax.set_xlabel("iteration"); ax.set_ylabel("-ELBO")
        ax.set_yscale("symlog")
        st.image(_fig_png(fig), width='stretch')
    if rhat_summary is not None:
        st.subheader("R̂ / ESS")
        cols_show = [c for c in ["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]
                     if c in rhat_summary.columns]
        st.dataframe(rhat_summary[cols_show])
        st.write(
            f"max R̂ = **{rhat_summary['r_hat'].max():.3f}**, "
            f"min ESS_bulk = **{rhat_summary['ess_bulk'].min():.0f}**"
        )

with tab5:
    if truth is None:
        st.info("Recovery diagnostics require simulated data.")
    else:
        col_sd = Y.std(0, ddof=1)
        L_true_std = truth["Lambda"] / col_sd[:, None]
        if L_true_std.shape[1] == K:
            err_L = np.linalg.norm(L_aligned - L_true_std)
            max_L = np.abs(L_aligned - L_true_std).max()
            err_O = np.linalg.norm(O_aligned - truth["Omega"])
            c1, c2, c3 = st.columns(3)
            c1.metric("‖Λ̂ − Λ‖_F", f"{err_L:.3f}")
            c2.metric("max |Λ̂ − Λ|", f"{max_L:.3f}")
            c3.metric("‖Ω̂ − Ω‖_F", f"{err_O:.3f}")

            st.subheader("Λ: estimated vs true (scatter)")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(L_true_std.flatten(), L_aligned.flatten(), alpha=0.7)
            lim = max(np.abs(L_true_std).max(), np.abs(L_aligned).max()) * 1.1
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
            ax.set_xlabel("true"); ax.set_ylabel("estimated")
            ax.set_aspect("equal")
            st.image(_fig_png(fig), width='stretch')
        else:
            st.warning(
                f"Fit K={K} but truth K={L_true_std.shape[1]} — "
                "alignment skipped. Refit with matching K to see recovery."
            )

with tab6:
    st.subheader("How many factors are in the data?")
    st.caption(
        "**Horn's parallel analysis**: compares the eigenvalues of the "
        "data correlation matrix to those of random datasets of the same "
        "shape. Factors whose eigenvalues exceed the 95th-percentile of "
        "the random eigenvalues are considered real."
    )
    pa = _pa_reco
    K_pa = pa["suggested_K"]
    K_kaiser = pa["kaiser_K"]
    lsd = loading_strength_diagnostic(Lambda_post)
    K_ssl = lsd["effective_K"]
    shrink = _shrink_reco
    K_shrink = shrink["suggested_K"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parallel analysis", K_pa, help="Horn 1965")
    c2.metric("Kaiser (eig > 1)", K_kaiser, help="Crude rule of thumb")
    c3.metric("Posterior SSL > 1", K_ssl,
              help="Columns of Λ̂ whose posterior-mean sum of squared loadings > 1")
    c4.metric(
        "Shrinkage (P[SSL>1] > 0.5)", K_shrink,
        help=(
            f"Horseshoe shrinkage diagnostic: number of columns whose "
            f"per-draw SSL exceeds 1 in > 50% of posterior draws. "
            f"Posterior mean K = {shrink['posterior_mean_K']:.2f}, "
            f"median = {shrink['posterior_median_K']}."
        ),
    )

    st.markdown("**Per-column activation probability** P[SSL > 1]")
    prob_df = pd.DataFrame({
        "factor": [f"F{k+1}" for k in range(len(shrink["prob_active"]))],
        "P[active]": shrink["prob_active"],
        "posterior mean SSL": shrink["ssl_draws"].mean(axis=(0, 1)),
    })
    st.dataframe(prob_df, hide_index=True)

    if K_pa != K:
        st.warning(
            f"⚠️ Parallel analysis suggests **K = {K_pa}** factors, "
            f"but you fit with **K = {K}**. Consider refitting with "
            f"K = {K_pa} for a more parsimonious / better-identified model."
        )
    else:
        st.success(f"✅ Your fit K = {K} matches the parallel-analysis suggestion.")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs = np.arange(1, len(pa["observed_eigenvalues"]) + 1)
    ax.plot(xs, pa["observed_eigenvalues"], "o-", label="data eigenvalues")
    ax.plot(xs, pa["null_quantile"], "s--", label="null 95% quantile",
            color="gray")
    ax.axhline(1.0, color="red", lw=0.8, alpha=0.6, label="Kaiser = 1")
    ax.set_xlabel("component"); ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8)
    st.image(_fig_png(fig), width='stretch')

with tab7:
    st.subheader("Save posterior draws")
    st.caption(
        "Download the full posterior so you can re-analyze it later "
        "(ArviZ NetCDF for PyMC backends, raw stacked CSV for any backend)."
    )

    # NetCDF if we have an InferenceData (PyMC backends)
    if "stan_fit" not in res:
        try:
            buf = io.BytesIO()
            res["idata"].to_netcdf(buf)
            st.download_button(
                "⬇️ ArviZ InferenceData (.nc)",
                buf.getvalue(),
                file_name="befa_posterior.nc",
                mime="application/x-netcdf",
            )
        except Exception as e:
            st.warning(f"NetCDF export unavailable: {e}")
    else:
        # Stan: save raw cmdstanpy outputs
        try:
            stan_buf = io.BytesIO()
            pickle.dump(
                {
                    "Lambda": res["stan_fit"].stan_variable("Lambda"),
                    "Omega": res["stan_fit"].stan_variable("Omega"),
                    "psi": res["stan_fit"].stan_variable("psi"),
                    "summary": res["stan_fit"].summary().to_dict(),
                },
                stan_buf,
            )
            st.download_button(
                "⬇️ Stan posterior (pickle)",
                stan_buf.getvalue(),
                file_name="befa_stan_posterior.pkl",
            )
        except Exception as e:
            st.warning(f"Stan pickle export failed: {e}")

    # Long-format CSV (works for everything)
    P = Y.shape[1]
    n_draws = Lambda_post.shape[0] * Lambda_post.shape[1]
    L_flat = Lambda_post.reshape(n_draws, P, K)
    O_flat = Omega_post.reshape(n_draws, K, K)
    psi_flat = psi_post.reshape(n_draws, P)
    draw_idx = np.arange(n_draws)
    x_labels = [f"x{j+1}" for j in range(P)]
    f_labels  = [f"F{k+1}" for k in range(K)]

    # Lambda: (n_draws, P, K) → n_draws*P*K rows
    d_L = np.repeat(draw_idx, P * K)
    r_L = np.tile(np.repeat(x_labels, K), n_draws)
    c_L = np.tile(f_labels, n_draws * P)
    df_lambda = pd.DataFrame({
        "param": "Lambda", "draw": d_L,
        "row": r_L, "col": c_L,
        "value": L_flat.ravel(),
    })

    # Omega: (n_draws, K, K) → n_draws*K*K rows
    d_O = np.repeat(draw_idx, K * K)
    r_O = np.tile(np.repeat(f_labels, K), n_draws)
    c_O = np.tile(f_labels, n_draws * K)
    df_omega = pd.DataFrame({
        "param": "Omega", "draw": d_O,
        "row": r_O, "col": c_O,
        "value": O_flat.ravel(),
    })

    # psi: (n_draws, P) → n_draws*P rows
    d_P = np.repeat(draw_idx, P)
    r_P = np.tile(x_labels, n_draws)
    df_psi = pd.DataFrame({
        "param": "psi", "draw": d_P,
        "row": r_P, "col": "",
        "value": psi_flat.ravel(),
    })

    csv_df = pd.concat([df_lambda, df_omega, df_psi], ignore_index=True)
    st.download_button(
        "⬇️ All draws (long CSV)",
        csv_df.to_csv(index=False).encode(),
        file_name="befa_draws.csv",
        mime="text/csv",
    )
    st.write(f"{n_draws} posterior draws across all parameters.")
